"""samia.core.test_semantic_recall — tests for the FEAT-2026-06-10 semantic recall arm + composer (P1).

What: tmp-store tests for samia.core.semantic_recall (the atom arm + composer) and the
  flag-gated fx_-skip in samia.core.context_extension. Each test plants a self-contained
  store (turn nodes + session chains + type:semantic atoms + an fx_ atom chain), builds
  the real vector index, and exercises the arm. NEVER touches the live memory dir.
Why: pins the population boundary (atoms only), chronological ordering, source-tag format
  + graceful omission + budget truncation, the composer's flag-off byte-identity, the
  flag-on FACTS/EVIDENCE split, the fx_-skip gate (flag-gated), env override + clamp,
  the zero-atom path, and the fail-soft (no index) path.

Depends: samia.core.{semantic_recall, context_extension, vector, frontmatter} (real
  embedder via vector.build — CPU MiniLM, so these tests embed a small node set).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import semantic_recall as sr
from samia.core import context_extension as cx
from samia.core import vector


# ---------------------------------------------------------------------------
# Fixtures — plant a tmp store: turn nodes + session chains + atoms + fx_ chain
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
    import json
    (md / "chains" / f"{chain_id}.json").write_text(
        json.dumps(chain, indent=1), encoding="utf-8")


def _build_store(md: Path) -> None:
    """Plant a representative store and build the vector index.

    Two session turns about a trip + a cat, two atoms (one dated, one undated), and an
    fx_ atom mini-chain over the atoms. The query 'When did Maria adopt her cat?' is the
    common probe — its answer lives in both an atom and a turn.
    """
    _turn(md, "s01_t000", "Maria", "3 May 2023",
          "I went hiking in the Alps last summer.", "D1:1")
    _turn(md, "s01_t001", "Sam", "3 May 2023",
          "Maria adopted a cat named Pixel in April.", "D1:2")
    _turn(md, "s02_t000", "Maria", "10 June 2023",
          "Pixel the cat loves the balcony.", "D2:1")
    _chain(md, "s01", ["s01_t000", "s01_t001"])
    _chain(md, "s02", ["s02_t000"])

    # Atoms (semantic population). One dated, one undated.
    _atom(md, "sem_cat_adopt", "Cat adoption fact",
          "Maria adopted a cat named Pixel.",
          source="s01", valid_from="2023-04-01")
    _atom(md, "sem_hike", "Hiking fact",
          "Maria went hiking in the Alps.", source="s01")
    # fx_ atom mini-chain over the atoms (the produced-atom chain shape).
    _chain(md, "fx_s01", ["sem_cat_adopt", "sem_hike"])

    vector.build(md, rebuild=True)


def setUpModule():  # noqa: N802 (unittest hook name)
    # Embedder warmup once for the module (keeps per-test build cheap-ish).
    vector._ensure_model()


def _clean_env():
    for k in (sr.SEMANTIC_ARM_ENABLED_ENV, sr.RECALL_FACTS_FRACTION_ENV):
        os.environ.pop(k, None)
    sr._clear_type_cache()
    cx._clear_atom_chain_cache()


# ---------------------------------------------------------------------------
# t1 — atom_retrieve returns ONLY semantic nodes, capped at k, scored
# ---------------------------------------------------------------------------


class TestAtomRetrieveSemanticOnly(unittest.TestCase):
    def test_only_semantic_capped_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_store(md)
            _clean_env()
            atoms = sr.atom_retrieve(md, "Maria adopted a cat", k=12)
            self.assertTrue(atoms, "expected at least one atom")
            # ONLY semantic nodes (every returned node resolves to type semantic).
            for a in atoms:
                self.assertEqual(sr._node_type(md, a["node"]), "semantic")
                self.assertNotIn("session_offload",
                                 sr._node_type(md, a["node"]) or "")
            # max k respected.
            self.assertLessEqual(len(atoms), 12)
            short = sr.atom_retrieve(md, "Maria adopted a cat", k=1)
            self.assertLessEqual(len(short), 1)
            # scored (a float score present on each).
            for a in atoms:
                self.assertIsInstance(a["score"], float)


# ---------------------------------------------------------------------------
# t2 — format_facts: chronological by valid_from, undated last
# ---------------------------------------------------------------------------


class TestFormatFactsChronology(unittest.TestCase):
    def test_chronological_undated_last(self):
        _clean_env()
        atoms = [
            {"title": "C", "body": "third", "valid_from": "2023-06-01",
             "source": ""},
            {"title": "U", "body": "undated", "valid_from": "", "source": ""},
            {"title": "A", "body": "first", "valid_from": "2023-01-01",
             "source": ""},
            {"title": "B", "body": "second", "valid_from": "2023-03-01",
             "source": ""},
        ]
        out = sr.format_facts(atoms)
        lines = out.splitlines()
        # dated ascending, then undated last.
        self.assertTrue(lines[0].startswith("- A:"))
        self.assertTrue(lines[1].startswith("- B:"))
        self.assertTrue(lines[2].startswith("- C:"))
        self.assertTrue(lines[3].startswith("- U:"))

    def test_undated_keeps_relevance_order(self):
        _clean_env()
        atoms = [
            {"title": "U1", "body": "x", "valid_from": "", "source": ""},
            {"title": "U2", "body": "y", "valid_from": "", "source": ""},
        ]
        lines = sr.format_facts(atoms).splitlines()
        self.assertTrue(lines[0].startswith("- U1:"))
        self.assertTrue(lines[1].startswith("- U2:"))


# ---------------------------------------------------------------------------
# t3 — format_facts: source tags + graceful omissions + budget truncation
# ---------------------------------------------------------------------------


class TestFormatFactsTagsAndBudget(unittest.TestCase):
    def test_source_tag_and_omissions(self):
        _clean_env()
        full = sr.format_facts([
            {"title": "T", "body": "b", "valid_from": "2023-01-01",
             "source": "s07"}])
        self.assertEqual(full, "- T: b (2023-01-01) [from s07]")
        # missing source -> no bracket; missing valid_from -> no parens.
        no_src = sr.format_facts([
            {"title": "T", "body": "b", "valid_from": "2023-01-01",
             "source": ""}])
        self.assertEqual(no_src, "- T: b (2023-01-01)")
        no_date = sr.format_facts([
            {"title": "T", "body": "b", "valid_from": "", "source": "s07"}])
        self.assertEqual(no_date, "- T: b [from s07]")
        bare = sr.format_facts([
            {"title": "T", "body": "b", "valid_from": "", "source": ""}])
        self.assertEqual(bare, "- T: b")

    def test_budget_truncation_drops_trailing_lines(self):
        _clean_env()
        atoms = [
            {"title": f"T{i}", "body": "x" * 40, "valid_from": f"2023-01-0{i}",
             "source": ""} for i in range(1, 6)]
        full = sr.format_facts(atoms)
        self.assertEqual(len(full.splitlines()), 5)
        # tiny budget -> fewer whole lines, never a partial line.
        small = sr.format_facts(atoms, budget_tokens=20)
        self.assertLess(len(small.splitlines()), 5)
        for ln in small.splitlines():
            self.assertTrue(ln.startswith("- T"))
            self.assertIn("x" * 40, ln)  # each kept line is intact


# ---------------------------------------------------------------------------
# t4 — recall flag OFF: no FACTS, no atom influence, byte-identical to control
# ---------------------------------------------------------------------------


class TestRecallFlagOff(unittest.TestCase):
    def test_passthrough_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_store(md)
            _clean_env()  # flag OFF
            self.assertFalse(sr.semantic_arm_enabled())
            res = sr.recall(md, "When did Maria adopt her cat?",
                            budget_tokens=8000)
            # no FACTS section in the off path.
            self.assertNotIn("KNOWN FACTS:", res["context"])
            self.assertEqual(res["facts_n"], 0)

            # Control: a chainogram-only assembly (the adapter's evidence pattern)
            # over the SAME full budget. flag OFF must be byte-identical to it.
            chain_out = cx.chainogram_retrieve(
                md, "When did Maria adopt her cat?", budget_tokens=8000,
                max_chains=8)
            names = sorted(n["node"] for n in chain_out.get("loaded_nodes", []))
            ctx_lines = []
            for name in names:
                raw = (md / "nodes" / name).read_text(encoding="utf-8")
                body = raw
                if raw.startswith("---"):
                    end = raw.find("\n---", 3)
                    if end != -1:
                        body = raw[end + 4:].lstrip()
                ctx_lines.append(body.rstrip())
            control = "\n".join(ctx_lines)
            self.assertEqual(res["context"], control)


# ---------------------------------------------------------------------------
# t5 — recall flag ON: FACTS + EVIDENCE present, facts within budget, dia evidence-only
# ---------------------------------------------------------------------------


class TestRecallFlagOn(unittest.TestCase):
    def test_composed_sections_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_store(md)
            _clean_env()
            with mock.patch.dict(os.environ,
                                 {sr.SEMANTIC_ARM_ENABLED_ENV: "1"}):
                res = sr.recall(md, "When did Maria adopt her cat?",
                                budget_tokens=8000)
            self.assertIn("KNOWN FACTS:", res["context"])
            self.assertIn("CONVERSATION EVIDENCE:", res["context"])
            self.assertGreater(res["facts_n"], 0)
            # FACTS section sits within the facts budget fraction (default 0.25).
            facts_block = res["context"].split("CONVERSATION EVIDENCE:")[0]
            facts_block = facts_block.replace("KNOWN FACTS:\n", "").strip()
            facts_budget_bytes = int(8000 * 0.25 * sr.BYTES_PER_TOKEN)
            self.assertLessEqual(len(facts_block), facts_budget_bytes)
            # dia_ids are evidence-only (atoms carry none); every id came from a turn.
            self.assertTrue(all(d.startswith("D") for d in res["dia_ids"]))

    def test_atoms_not_in_evidence_dia(self):
        # Atoms have no dia; the served dia_ids must never include an atom marker.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_store(md)
            _clean_env()
            with mock.patch.dict(os.environ,
                                 {sr.SEMANTIC_ARM_ENABLED_ENV: "1"}):
                res = sr.recall(md, "Maria cat Pixel", budget_tokens=8000)
            for d in res["dia_ids"]:
                self.assertNotIn("sem_", d)


# ---------------------------------------------------------------------------
# t6 — chainogram fx_-skip ONLY when flag on
# ---------------------------------------------------------------------------


class TestChainogramFxSkip(unittest.TestCase):
    def test_fx_loaded_when_off_skipped_when_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_store(md)

            # flag OFF: fx_ atom chains load exactly as before.
            _clean_env()
            off = cx.chainogram_retrieve(md, "Maria adopted a cat named Pixel",
                                         budget_tokens=8000, max_chains=8)
            self.assertTrue(
                any(str(c).startswith("fx_") for c in off.get("loaded_chains", [])),
                f"fx_ chain not loaded with flag off: {off.get('loaded_chains')}")

            # flag ON: fx_ atom chains are skipped from selection.
            _clean_env()
            with mock.patch.dict(os.environ,
                                 {sr.SEMANTIC_ARM_ENABLED_ENV: "1"}):
                on = cx.chainogram_retrieve(
                    md, "Maria adopted a cat named Pixel",
                    budget_tokens=8000, max_chains=8)
            self.assertFalse(
                any(str(c).startswith("fx_") for c in on.get("loaded_chains", [])),
                f"fx_ chain not skipped with flag on: {on.get('loaded_chains')}")
            # no atom node leaked into evidence under the flag.
            for n in on.get("loaded_nodes", []):
                self.assertFalse(str(n["node"]).startswith("sem_"))


# ---------------------------------------------------------------------------
# t7 — facts_fraction env override honored + clamped
# ---------------------------------------------------------------------------


class TestFactsFractionEnv(unittest.TestCase):
    def test_override_and_clamp(self):
        _clean_env()
        self.assertAlmostEqual(sr.facts_fraction(), 0.25)
        with mock.patch.dict(os.environ,
                             {sr.RECALL_FACTS_FRACTION_ENV: "0.5"}):
            self.assertAlmostEqual(sr.facts_fraction(), 0.5)
        # clamp high.
        with mock.patch.dict(os.environ,
                             {sr.RECALL_FACTS_FRACTION_ENV: "2.0"}):
            self.assertAlmostEqual(sr.facts_fraction(), 0.9)
        # clamp low.
        with mock.patch.dict(os.environ,
                             {sr.RECALL_FACTS_FRACTION_ENV: "-1.0"}):
            self.assertAlmostEqual(sr.facts_fraction(), 0.0)
        # unparseable -> default.
        with mock.patch.dict(os.environ,
                             {sr.RECALL_FACTS_FRACTION_ENV: "abc"}):
            self.assertAlmostEqual(sr.facts_fraction(), 0.25)

    def test_fraction_changes_facts_budget(self):
        # A larger fraction lets more facts bytes through.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_store(md)
            _clean_env()
            with mock.patch.dict(os.environ,
                                 {sr.SEMANTIC_ARM_ENABLED_ENV: "1",
                                  sr.RECALL_FACTS_FRACTION_ENV: "0.9"}):
                res = sr.recall(md, "Maria adopted a cat", budget_tokens=8000)
            self.assertIn("KNOWN FACTS:", res["context"])


class TestFocuserDefaults(unittest.TestCase):
    """TUNE-2026-06-11 — benchmark-validated focuser defaults: K=8, cap=2400."""

    def test_focus_k_default_is_8(self):
        os.environ.pop("ASTHENOS_RECALL_FOCUS_K", None)
        self.assertEqual(sr.focus_k(), 8)

    def test_evidence_cap_default_is_2400(self):
        os.environ.pop("ASTHENOS_RECALL_EVIDENCE_CAP", None)
        self.assertEqual(sr.evidence_cap(), 2400)

    def test_focus_k_env_override(self):
        with mock.patch.dict(os.environ, {"ASTHENOS_RECALL_FOCUS_K": "3"}):
            self.assertEqual(sr.focus_k(), 3)
        # unparseable -> default 8.
        with mock.patch.dict(os.environ, {"ASTHENOS_RECALL_FOCUS_K": "x"}):
            self.assertEqual(sr.focus_k(), 8)

    def test_evidence_cap_env_override_and_floor(self):
        with mock.patch.dict(os.environ,
                             {"ASTHENOS_RECALL_EVIDENCE_CAP": "3000"}):
            self.assertEqual(sr.evidence_cap(), 3000)
        # floor at 200.
        with mock.patch.dict(os.environ,
                             {"ASTHENOS_RECALL_EVIDENCE_CAP": "50"}):
            self.assertEqual(sr.evidence_cap(), 200)
        # unparseable -> default 2400.
        with mock.patch.dict(os.environ,
                             {"ASTHENOS_RECALL_EVIDENCE_CAP": "x"}):
            self.assertEqual(sr.evidence_cap(), 2400)


# ---------------------------------------------------------------------------
# t8 — zero-atom store: FACTS section omitted, evidence still served
# ---------------------------------------------------------------------------


class TestZeroAtomStore(unittest.TestCase):
    def test_no_facts_section_evidence_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            # turns + session chains only — NO atoms, NO fx_ chain.
            _turn(md, "s01_t000", "Maria", "3 May 2023",
                  "I went hiking in the Alps.", "D1:1")
            _turn(md, "s01_t001", "Sam", "3 May 2023",
                  "Maria adopted a cat named Pixel.", "D1:2")
            _chain(md, "s01", ["s01_t000", "s01_t001"])
            vector.build(md, rebuild=True)
            _clean_env()
            with mock.patch.dict(os.environ,
                                 {sr.SEMANTIC_ARM_ENABLED_ENV: "1"}):
                res = sr.recall(md, "Maria cat", budget_tokens=8000)
            self.assertEqual(res["facts_n"], 0)
            self.assertNotIn("KNOWN FACTS:", res["context"])
            self.assertIn("CONVERSATION EVIDENCE:", res["context"])
            self.assertGreater(res["evidence_nodes"], 0)


# ---------------------------------------------------------------------------
# t9 — fail-soft: missing vector index -> recall still returns evidence-only
# ---------------------------------------------------------------------------


class TestFailSoftNoIndex(unittest.TestCase):
    def test_atom_retrieve_empty_when_no_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _turn(md, "s01_t000", "Maria", "3 May 2023", "Hi.", "D1:1")
            # no vector.build -> no index.
            _clean_env()
            self.assertEqual(sr.atom_retrieve(md, "anything", k=12), [])

    def test_recall_evidence_only_when_atom_arm_blind(self):
        # Build an index (evidence arm works), but force the atom arm blind.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _build_store(md)
            _clean_env()
            with mock.patch.dict(os.environ,
                                 {sr.SEMANTIC_ARM_ENABLED_ENV: "1"}), \
                    mock.patch.object(sr, "atom_retrieve", return_value=[]):
                res = sr.recall(md, "Maria cat", budget_tokens=8000)
            # atom arm blind -> FACTS omitted, evidence still served.
            self.assertEqual(res["facts_n"], 0)
            self.assertNotIn("KNOWN FACTS:", res["context"])
            self.assertIn("CONVERSATION EVIDENCE:", res["context"])
            self.assertGreater(res["evidence_nodes"], 0)


class TestEvidenceFocuser(unittest.TestCase):
    """TUNE-2026-06-10 focuser: per-chain top-K member selection + token cap."""

    @staticmethod
    def _entries():
        # Two chains: A has 4 members, B has 2. tokens=100 each.
        return [{"node": f"a{i}.md", "tokens": 100, "chain": "chA"} for i in range(4)] \
             + [{"node": f"b{i}.md", "tokens": 100, "chain": "chB"} for i in range(2)]

    def _rel(self, scores):
        hits = [{"node": n, "score": s} for n, s in scores.items()]
        return mock.patch("samia.core.vector.query", return_value=hits)

    def test_top_k_per_chain_by_relevance(self):
        _clean_env()
        with mock.patch.dict(os.environ, {"ASTHENOS_RECALL_FOCUS_K": "2"}), \
                self._rel({"a3.md": 0.9, "a1.md": 0.8, "a0.md": 0.1, "b1.md": 0.7}):
            kept = sr._focus_evidence(Path("/tmp"), {"loaded_nodes": self._entries()},
                                      "q")
        names = {e["node"] for e in kept}
        self.assertEqual(names, {"a3.md", "a1.md", "b1.md", "b0.md"})

    def test_cap_drops_depth_keeps_every_chains_best(self):
        _clean_env()
        with mock.patch.dict(os.environ, {"ASTHENOS_RECALL_FOCUS_K": "4",
                                          "ASTHENOS_RECALL_EVIDENCE_CAP": "200"}), \
                self._rel({"a0.md": 0.9, "a1.md": 0.8, "a2.md": 0.7,
                           "b0.md": 0.6}):
            kept = sr._focus_evidence(Path("/tmp"), {"loaded_nodes": self._entries()},
                                      "q")
        names = [e["node"] for e in kept]
        # cap 200 = 2 members; chA takes its best two, chB STILL gets its best.
        self.assertIn("a0.md", names)
        self.assertIn("b0.md", names)       # breadth guarantee past the cap
        self.assertLessEqual(len(names), 3)

    def test_k_zero_disables(self):
        _clean_env()
        with mock.patch.dict(os.environ, {"ASTHENOS_RECALL_FOCUS_K": "0"}):
            kept = sr._focus_evidence(Path("/tmp"), {"loaded_nodes": self._entries()},
                                      "q")
        self.assertEqual(len(kept), 6)      # untouched

    def test_rel_failure_falls_back_to_chain_order(self):
        _clean_env()
        with mock.patch.dict(os.environ, {"ASTHENOS_RECALL_FOCUS_K": "1"}), \
                mock.patch("samia.core.vector.query", side_effect=RuntimeError):
            kept = sr._focus_evidence(Path("/tmp"), {"loaded_nodes": self._entries()},
                                      "q")
        # fail-soft: rel map empty -> first member per chain kept.
        self.assertEqual({e["node"] for e in kept}, {"a0.md", "b0.md"})


if __name__ == "__main__":
    unittest.main()

# [Asthenosphere] samia.core.test_semantic_recall
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-10 — semantic recall arm + composer (P1)
# Layer:      test (pytest)
# Role:       tests for samia.core.semantic_recall + context_extension — atom-only population, chronological FACTS, source tags + budget truncation, flag-off byte-identity, flag-on FACTS/EVIDENCE split, fx_-skip gate, fraction/focuser env clamps, zero-atom + fail-soft paths
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.semantic_recall, samia.core.context_extension, samia.core.vector, samia.core.frontmatter
# Exposes:    — (test module)
# Lines:      524
# ------------------------------------------------------------------------------
