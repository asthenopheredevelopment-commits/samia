"""samia.core.test_semantic_recall_p2 — tests for FEAT-2026-06-10 P2 (MCP wire + read-conflict + bridge).

What: tmp-store tests for the three P2 additions on the semantic recall arm:
  - P2a (MCP wire): core.mcp_server.memory_chainogram_retrieve flag-off byte-identity
    to the bare chainogram result, and flag-on overlay of composed_* keys WITHOUT
    dropping the existing chainogram contract keys.
  - P2b (read-conflict supersession signal): recall() flag-ON path records exactly one
    read_conflict supersession candidate for a planted >=0.92 served pair; nothing below
    the bar; nothing when the kill-switch is off; recall unaffected when the index is
    missing; the F5a dedup guard suppresses a second record on a repeat scan.
  - P2c (entity-bridge atom retrieval): bridge atoms fill the reserved slots, deduped vs
    the vector top-k; absent entity index -> pure vector; the frac env override widens/
    narrows the reserve.

Why: pins the P2 contract — composed extras are ADDITIVE (callers never break), the
  read-conflict signal is record-only + fail-open + dedup-guarded + kill-switchable, and
  the entity-bridge reserve reaches multihop atoms the single vector query misses while
  failing open to today's pure-vector behavior. NEVER touches the live memory dir.

Depends: samia.core.{semantic_recall, context_extension, vector, entity_index,
  mcp_server}, samia.runtime.contradiction (read-conflict store). Controlled embeddings
  are written directly for the P2b cosine-precision tests; the real embedder is used for
  the P2c bridge tests.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from samia.core import semantic_recall as sr
from samia.core import context_extension as cx
from samia.core import entity_index
from samia.core import mcp_server
from samia.core import vector
from samia.runtime import contradiction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    return md


def _turn(md: Path, stem: str, speaker: str, date: str, text: str,
          dia: str) -> None:
    fm = (f"---\nname: {stem}\ndescription: {speaker} — {date}\n"
          f"type: session_offload\ndia: {dia}\n---\n")
    body = f"[{date}] {speaker}: {text}\n"
    (md / "nodes" / f"{stem}.md").write_text(fm + body, encoding="utf-8")


def _atom(md: Path, stem: str, title: str, body: str,
          source: str = "", valid_from: str = "") -> None:
    lines = [f"name: {title}", "type: semantic"]
    if source:
        lines.append(f"source: {source}")
    if valid_from:
        lines.append(f"valid_from: {valid_from}")
    lines.append("tier: cold")
    fm = "---\n" + "\n".join(lines) + "\n---\n"
    (md / "nodes" / f"{stem}.md").write_text(fm + body + "\n", encoding="utf-8")


def _chain(md: Path, chain_id: str, member_stems: list[str]) -> None:
    members = [{"addr": f"A-{chain_id}-{i:03d}",
                "file": f"nodes/{s}.md", "tier": "warm"}
               for i, s in enumerate(member_stems)]
    chain = {"chain_id": chain_id,
             "head_address": members[0]["addr"] if members else None,
             "tail_address": members[-1]["addr"] if members else None,
             "members": members, "total_relevance": 0.0,
             "last_traversal": None, "compressed": False, "edges": []}
    (md / "chains" / f"{chain_id}.json").write_text(
        json.dumps(chain, indent=1), encoding="utf-8")


def _write_controlled_index(md: Path, vecs: dict[str, list[float]]) -> None:
    """Write a hand-crafted vector_index (embeddings.npy + manifest.json).

    vecs maps node filename (with .md) -> a raw vector; each is L2-normalized here so a
    row dot product equals cosine (the index invariant samia.core.vector guarantees).
    Lets P2b assert exact cosine thresholds without depending on the MiniLM embedder.
    """
    idx = md / "vector_index"
    idx.mkdir(parents=True, exist_ok=True)
    names = list(vecs.keys())
    rows = []
    entries = {}
    for i, n in enumerate(names):
        v = np.asarray(vecs[n], dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        rows.append(v)
        entries[n] = {"sha256": f"sha{i}", "title": n[:-3], "row": i}
    np.save(str(idx / "embeddings.npy"), np.vstack(rows))
    (idx / "manifest.json").write_text(
        json.dumps({"model_id": "test", "dim": len(rows[0]),
                    "entries": entries}), encoding="utf-8")


def _clean_env():
    for k in (sr.SEMANTIC_ARM_ENABLED_ENV, sr.RECALL_FACTS_FRACTION_ENV,
              sr.ATOM_BRIDGE_FRAC_ENV, sr.READ_CONFLICT_ENABLED_ENV):
        os.environ.pop(k, None)
    sr._clear_type_cache()
    cx._clear_atom_chain_cache()


def setUpModule():  # noqa: N802
    vector._ensure_model()


# ===========================================================================
# P2a — MCP wire
# ===========================================================================


def _build_basic_store(md: Path) -> None:
    _turn(md, "s01_t000", "Maria", "3 May 2023",
          "I went hiking in the Alps last summer.", "D1:1")
    _turn(md, "s01_t001", "Sam", "3 May 2023",
          "Maria adopted a cat named Pixel in April.", "D1:2")
    _chain(md, "s01", ["s01_t000", "s01_t001"])
    _atom(md, "sem_cat_adopt", "Cat adoption fact",
          "Maria adopted a cat named Pixel.",
          source="s01", valid_from="2023-04-01")
    _atom(md, "sem_hike", "Hiking fact",
          "Maria went hiking in the Alps.", source="s01")
    _chain(md, "fx_s01", ["sem_cat_adopt", "sem_hike"])
    vector.build(md, rebuild=True)


class TestMcpWireFlagOff(unittest.TestCase):
    """Flag OFF: the MCP result is identical to the bare chainogram result."""

    def test_flag_off_identical_to_bare_chainogram(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_basic_store(md)
            _clean_env()  # flag OFF
            # The MCP function delegates to core context_extension; mock that
            # module's chainogram_retrieve to read OUR store so the test is
            # hermetic, and assert flag-off returns its result verbatim.
            bare = cx.chainogram_retrieve(md, "Maria cat Pixel",
                                          budget_tokens=8000, max_chains=8)
            with mock.patch.object(cx, "chainogram_retrieve",
                                   return_value=dict(bare)):
                res = mcp_server.memory_chainogram_retrieve(
                    md, "Maria cat Pixel", budget_tokens=8000, max_chains=8)
            self.assertEqual(res, bare)
            self.assertNotIn("composed_context", res)
            self.assertNotIn("semantic_arm", res)


class TestMcpWireFlagOn(unittest.TestCase):
    """Flag ON: composed_* extras overlaid, existing chainogram keys preserved."""

    def test_flag_on_overlays_composed_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_basic_store(md)
            _clean_env()
            bare = cx.chainogram_retrieve(md, "Maria cat Pixel",
                                          budget_tokens=8000, max_chains=8)
            with mock.patch.dict(os.environ,
                                 {sr.SEMANTIC_ARM_ENABLED_ENV: "1"}), \
                    mock.patch.object(cx, "chainogram_retrieve",
                                      return_value=dict(bare)):
                res = mcp_server.memory_chainogram_retrieve(
                    md, "Maria cat Pixel", budget_tokens=8000, max_chains=8)
            # Existing contract keys still present and unchanged.
            for key in ("loaded_chains", "loaded_nodes", "spent_tokens",
                        "budget_tokens", "rationale"):
                self.assertIn(key, res)
                self.assertEqual(res[key], bare[key])
            # Composed extras present under NEW keys.
            self.assertIn("composed_context", res)
            self.assertIn("CONVERSATION EVIDENCE:", res["composed_context"])
            self.assertIn("facts_n", res)
            self.assertIn("composed_evidence_nodes", res)
            self.assertIn("composed_dia_ids", res)
            self.assertTrue(res["semantic_arm"])


# ===========================================================================
# P2b — read-conflict supersession signal
# ===========================================================================


class TestReadConflictSignal(unittest.TestCase):
    def _store_with_controlled_index(self, tmp: str, cos_atom_evi: float):
        """Plant a store whose served atom + evidence pair has a chosen cosine.

        Two collinear vectors -> cosine cos_atom_evi; an unrelated third row keeps the
        index honest. The atom (sem_dup) and the turn (s01_t001) are the planted pair.
        """
        md = _mem(tmp)
        _turn(md, "s01_t001", "Sam", "3 May 2023",
              "Maria adopted a cat named Pixel.", "D1:2")
        _chain(md, "s01", ["s01_t001"])
        _atom(md, "sem_dup", "Cat fact", "Maria adopted a cat named Pixel.",
              source="s01", valid_from="2023-04-01")
        _atom(md, "sem_other", "Trip fact", "Maria hiked the Alps.",
              source="s01", valid_from="2023-05-01")
        _chain(md, "fx_s01", ["sem_dup", "sem_other"])
        # Controlled index: atom & turn collinear at cos_atom_evi; sem_other orthogonal.
        a = 1.0
        b = cos_atom_evi
        c = float(np.sqrt(max(0.0, 1.0 - b * b)))
        _write_controlled_index(md, {
            "s01_t001.md": [b, c, 0.0],
            "sem_dup.md": [a, 0.0, 0.0],
            "sem_other.md": [0.0, 0.0, 1.0],
        })
        return md

    def _serve(self, md: Path):
        """Drive recall() flag-ON with the atom/evidence sets forced to the planted
        pair (mock atom_retrieve + the chainogram so the served sets are deterministic;
        the read-conflict scan runs over exactly sem_dup x s01_t001)."""
        atoms = [{"node": "sem_dup.md", "title": "Cat fact",
                  "body": "Maria adopted a cat named Pixel.",
                  "valid_from": "2023-04-01", "source": "s01", "score": 0.9}]
        kept = [{"node": "s01_t001.md", "tokens": 20, "chain": "s01"}]
        chain_out = {"loaded_nodes": kept, "loaded_chains": ["s01"],
                     "spent_tokens": 20, "budget_tokens": 6000,
                     "n_singletons": 0, "rationale": "x", "skipped_nodes": []}
        with mock.patch.dict(os.environ, {sr.SEMANTIC_ARM_ENABLED_ENV: "1"}), \
                mock.patch.object(sr, "atom_retrieve", return_value=atoms), \
                mock.patch.object(cx, "chainogram_retrieve",
                                  return_value=chain_out), \
                mock.patch.object(sr, "_focus_evidence", return_value=kept):
            return sr.recall(md, "Maria cat Pixel", budget_tokens=8000)

    def test_above_bar_records_exactly_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            _clean_env()
            md = self._store_with_controlled_index(tmp, cos_atom_evi=0.97)
            self._serve(md)
            recs = contradiction.list_supersession_candidates(
                md, unresolved_only=True)
            rc = [r for r in recs if r.get("mode") == "read_conflict"]
            self.assertEqual(len(rc), 1, f"expected one read_conflict rec: {recs}")
            r = rc[0]
            self.assertGreaterEqual(r["cosine"], 0.92)
            # newer-dated turn has no valid_from -> atom(old) -> turn(new) ordering
            # falls to second-served = turn as new_id.
            self.assertEqual(r["old_id"], "sem_dup.md")
            self.assertEqual(r["new_id"], "s01_t001.md")

    def test_below_bar_records_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            _clean_env()
            md = self._store_with_controlled_index(tmp, cos_atom_evi=0.80)
            self._serve(md)
            recs = contradiction.list_supersession_candidates(md)
            self.assertEqual(
                [r for r in recs if r.get("mode") == "read_conflict"], [])

    def test_kill_switch_off_records_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            _clean_env()
            md = self._store_with_controlled_index(tmp, cos_atom_evi=0.97)
            with mock.patch.dict(os.environ,
                                 {sr.READ_CONFLICT_ENABLED_ENV: "0"}):
                self._serve(md)
            recs = contradiction.list_supersession_candidates(md)
            self.assertEqual(
                [r for r in recs if r.get("mode") == "read_conflict"], [])

    def test_missing_index_recall_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _clean_env()
            md = self._store_with_controlled_index(tmp, cos_atom_evi=0.97)
            # Remove the index entirely: the scan must skip, recall still composes.
            import shutil
            shutil.rmtree(md / "vector_index")
            res = self._serve(md)
            self.assertIn("CONVERSATION EVIDENCE:", res["context"])
            recs = contradiction.list_supersession_candidates(md)
            self.assertEqual(
                [r for r in recs if r.get("mode") == "read_conflict"], [])

    def test_dedup_guard_no_second_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            _clean_env()
            md = self._store_with_controlled_index(tmp, cos_atom_evi=0.97)
            self._serve(md)   # first scan -> one record
            self._serve(md)   # repeat scan -> F5a dedup suppresses the second write
            recs = contradiction.list_supersession_candidates(
                md, unresolved_only=True)
            rc = [r for r in recs if r.get("mode") == "read_conflict"]
            self.assertEqual(len(rc), 1, f"dedup guard failed: {recs}")


# ===========================================================================
# P2c — entity-bridge atom retrieval
# ===========================================================================


class TestEntityBridgeReserve(unittest.TestCase):
    def _bridge_store(self, md: Path) -> None:
        """Plant atoms whose entity (`Pixel_v2`) bridges to a low-vector-rank atom.

        sem_near is vector-close to the query; sem_bridge shares the technical entity
        `pixel_config_v2` with the query but is vector-far. Padding atoms push the bridge
        atom out of the vector top slots so it can only enter via the reserve.
        """
        # The query will mention `pixel_config_v2`; only sem_bridge contains it.
        _atom(md, "sem_bridge", "Bridge fact",
              "The `pixel_config_v2` setting controls the cat tracker.",
              source="s01", valid_from="2023-01-01")
        _atom(md, "sem_near", "Near fact",
              "Maria adopted a cat named Pixel in spring.",
              source="s01", valid_from="2023-04-01")
        # Padding atoms (vector-relevant filler so the reserve choice is observable).
        for i in range(6):
            _atom(md, f"sem_pad{i}", f"Pad {i}",
                  f"Maria mentioned a cat detail number {i}.", source="s01")
        _chain(md, "fx_s01",
               ["sem_bridge", "sem_near"] + [f"sem_pad{i}" for i in range(6)])
        vector.build(md, rebuild=True)
        entity_index.build_index(md)

    def test_bridge_fills_reserved_slots_deduped(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            self._bridge_store(md)
            _clean_env()
            # k=8, frac=0.25 -> reserve=2; query carries the bridge entity.
            with mock.patch.dict(os.environ,
                                 {sr.ATOM_BRIDGE_FRAC_ENV: "0.25"}):
                atoms = sr.atom_retrieve(
                    md, "what is the `pixel_config_v2` cat setting", k=8)
            nodes = [a["node"] for a in atoms]
            # bridge atom present (entity match pulled it in despite low vector rank).
            self.assertIn("sem_bridge.md", nodes)
            # no duplicates (dedup vs vector picks).
            self.assertEqual(len(nodes), len(set(nodes)))

    def test_absent_entity_index_pure_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            self._bridge_store(md)
            # Remove the entity index -> fail-open to pure vector (no crash, atoms still
            # returned from vector relevance only).
            (md / "vector_index" / "entity_bridges.json").unlink()
            _clean_env()
            with mock.patch.dict(os.environ,
                                 {sr.ATOM_BRIDGE_FRAC_ENV: "0.25"}):
                atoms = sr.atom_retrieve(md, "Maria adopted a cat", k=8)
            self.assertTrue(atoms)  # pure-vector path still serves atoms
            self.assertEqual(len({a["node"] for a in atoms}), len(atoms))

    def test_frac_zero_disables_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            self._bridge_store(md)
            _clean_env()
            with mock.patch.dict(os.environ, {sr.ATOM_BRIDGE_FRAC_ENV: "0.0"}):
                # frac 0 -> reserve 0 -> pure vector; bridge-only atom should NOT be
                # forced in unless it is independently vector-relevant.
                self.assertEqual(sr.atom_bridge_frac(), 0.0)
                atoms = sr.atom_retrieve(
                    md, "what is the `pixel_config_v2` cat setting", k=8)
            self.assertTrue(atoms)
            self.assertEqual(len({a["node"] for a in atoms}), len(atoms))

    def test_frac_env_override(self):
        _clean_env()
        self.assertAlmostEqual(sr.atom_bridge_frac(), 0.25)
        with mock.patch.dict(os.environ, {sr.ATOM_BRIDGE_FRAC_ENV: "0.5"}):
            self.assertAlmostEqual(sr.atom_bridge_frac(), 0.5)
        with mock.patch.dict(os.environ, {sr.ATOM_BRIDGE_FRAC_ENV: "2.0"}):
            self.assertAlmostEqual(sr.atom_bridge_frac(), 0.9)  # clamp high
        with mock.patch.dict(os.environ, {sr.ATOM_BRIDGE_FRAC_ENV: "abc"}):
            self.assertAlmostEqual(sr.atom_bridge_frac(), 0.25)  # default


class TestReadConflictEnabledReader(unittest.TestCase):
    def test_default_on_and_kill(self):
        _clean_env()
        self.assertTrue(sr.read_conflict_enabled())
        with mock.patch.dict(os.environ, {sr.READ_CONFLICT_ENABLED_ENV: "0"}):
            self.assertFalse(sr.read_conflict_enabled())
        with mock.patch.dict(os.environ, {sr.READ_CONFLICT_ENABLED_ENV: "1"}):
            self.assertTrue(sr.read_conflict_enabled())
        # any non-"0" token stays ON (fail-toward-recording).
        with mock.patch.dict(os.environ, {sr.READ_CONFLICT_ENABLED_ENV: "yes"}):
            self.assertTrue(sr.read_conflict_enabled())


if __name__ == "__main__":
    unittest.main()

# [Asthenosphere] samia.core.test_semantic_recall_p2
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-10 P2 — MCP wire + read-conflict supersession signal + entity-bridge atom retrieval
# Layer:      test (pytest)
# Role:       tests for samia.core.semantic_recall + mcp_server + entity_index — additive composed_* MCP overlay, record-only/fail-open/dedup-guarded/kill-switchable read-conflict signal, entity-bridge reserve reaching multihop atoms with fail-open to pure vector
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.semantic_recall, samia.core.context_extension, samia.core.entity_index, samia.core.mcp_server, samia.core.vector, samia.runtime.contradiction
# Exposes:    — (test module)
# Lines:      424
# ------------------------------------------------------------------------------
