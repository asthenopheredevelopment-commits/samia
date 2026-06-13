"""Tests for the Tier-2 merge consumer P3 — salience guard (FEAT-2026-06-07).

Layer 1 (Owns / Depends):
    Owns:    unit tests for the P3 SALIENCE GUARD consumed in TWO places:
             (1) the MERGE consumer (drain's abstraction path + the defensive
                 synthesize_abstraction guard) — a DISTINCT high-salience source
                 is SURFACED (status="guarded"), NOT auto-abstracted-away; a TRUE
                 duplicate is exempt (is_duplicate=True -> guard False) so the P1
                 pick-winner dup merge is UNCHANGED; a LOW-salience distinct pair
                 behaves exactly as before P3 (recorded "pending");
             (2) the CONTRADICTION auto-supersede (contradiction.passive_sweep
                 judge-confirmed + mcp_server._online_supersede exact case) — a
                 high-salience loser is NOT auto-superseded, it is surfaced
                 (status="surfaced-salience") instead, and reversibility is
                 intact (nothing removed).
    Depends: samia.core.{merge_consumer,bio,ia,frontmatter}, samia.runtime
             .contradiction, samia.core.mcp_server, unittest, unittest.mock,
             tempfile, json. EVERY test uses a tempdir memory root + a tempdir
             db_dir — NEVER the live ~/.local/share memory or the global
             edges.db, and the cosine finder / LLM judge / synthesizer are MOCKED
             (no embedder, no model load). High salience is planted directly on
             the node's `salience` frontmatter (the SOURCE that bio defines).

Layer 2 (What / Why):
    What: validates P3's Exit from the approved proposal (Q5a / D6 effect iii) —
          (a) a DISTINCT high-salience pair is NOT auto-abstracted by the merge
              consumer; it is surfaced as a guarded candidate (no node removed);
          (b) a high-salience loser is NOT auto-superseded by the contradiction
              passive sweep nor the online write path — it is surfaced instead;
          (c) a TRUE-DUPLICATE high-salience pair STILL auto-merges (the dup
              pick-winner path is is_duplicate=True -> guard False);
          (d) a LOW-salience distinct pair behaves exactly as before P3;
          (e) reversibility intact — the salience-protected node is never removed,
              so there is nothing to restore (and the dup that DID merge restores).
    Why:  Q5a / D6 — do NOT silently abstract-away or supersede an important
          one-shot memory; surface it for the operator. A regression that lets the
          guard leak would auto-remove a high-salience distinct memory (the exact
          loss the salience axis exists to prevent). The guard is read-only and
          wired behind a hasattr so it activates with no re-sequence; these tests
          plant the salience field directly so the guard is exercised live.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import merge_consumer as mc
from samia.core import ia
from samia.runtime import contradiction as con


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(md: Path, name: str, body: str, **fm) -> None:
    lines = ["---", f"name: {name}"]
    for k, v in fm.items():
        lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append(body)
    (md / "nodes" / f"{name}.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


def _mem() -> Path:
    # What: mkdtemp + atexit-registered rmtree of every dir handed out. Why:
    #   mkdtemp does NOT auto-clean, so each call left a tier2_merge_p3_test_* dir
    #   in /tmp and tripped the cold-metal zero-leftover hygiene gate. One atexit
    #   registration covers every `md = _mem()` call site and any test order.
    md = Path(tempfile.mkdtemp(prefix="tier2_merge_p3_test_"))
    atexit.register(shutil.rmtree, md, ignore_errors=True)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    (md / "archive").mkdir(parents=True, exist_ok=True)
    return md


# Near-identical bodies => jaccard ~1.0 (DUP, >= 0.85 bar).
_DUP_BODY = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
             "lambda mu nu xi omicron pi rho sigma")
_DUP_BODY_RICH = _DUP_BODY + " tau upsilon phi chi psi omega extra detail here"

# Overlapping ~0.33 jaccard (surfaced, but ABSTRACT — below 0.85 bar).
_ABS_A = "alpha beta gamma delta rocket engine thrust nozzle combustion"
_ABS_B = "alpha beta gamma delta ocean tide salinity current wave"


def _write_candidates(md: Path, pairs: list[dict], threshold: float = 0.15) -> None:
    payload = {"generated": "x", "threshold": threshold, "candidates": pairs}
    (md / ".consolidation_candidates.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def _cand(a: str, b: str, sim: float) -> dict:
    return {"chain": "k", "a_addr": a, "a_file": f"nodes/{a}.md",
            "b_addr": b, "b_file": f"nodes/{b}.md", "similarity": sim}


class _MergeEnabled:
    """Context manager: set ASTHENOS_TIER2_MERGE_ENABLED for the body."""

    def __init__(self, on: bool = True):
        self.on = on
        self._prev = None

    def __enter__(self):
        self._prev = os.environ.get("ASTHENOS_TIER2_MERGE_ENABLED")
        os.environ["ASTHENOS_TIER2_MERGE_ENABLED"] = "1" if self.on else "0"
        return self

    def __exit__(self, *a):
        if self._prev is None:
            os.environ.pop("ASTHENOS_TIER2_MERGE_ENABLED", None)
        else:
            os.environ["ASTHENOS_TIER2_MERGE_ENABLED"] = self._prev


def _enable_contradiction():
    return mock.patch.dict(os.environ, {"ASTHENOS_CONTRADICTION_ENABLED": "1"})


# ---------------------------------------------------------------------------
# (a) MERGE consumer: a DISTINCT high-salience pair is SURFACED, not abstracted
# ---------------------------------------------------------------------------


class TestMergeConsumerSalienceGuard(unittest.TestCase):

    def test_distinct_high_salience_surfaced_not_abstracted(self):
        """drain surfaces a distinct high-salience pair as 'guarded' (no merge)."""
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            # X is high-salience (>= 0.8 floor); Y is normal. Distinct (~0.33).
            _node(md, "X", _ABS_A, salience=0.95)
            _node(md, "Y", _ABS_B, salience=0.1)
            _write_candidates(md, [_cand("X", "Y", 0.333)])
            with _MergeEnabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            # NOT merged, recorded (drained from the backlog) but as guarded.
            self.assertEqual(res["merged"], 0)
            self.assertEqual(res["recorded"], 1)
            # BOTH originals still live — nothing auto-removed.
            self.assertTrue((md / "nodes" / "X.md").exists())
            self.assertTrue((md / "nodes" / "Y.md").exists())
            # No abstraction node, no archive (reversibility: nothing removed).
            self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])
            self.assertEqual(
                list((md / "archive").glob("*.superseded.json")), [])
            # Surfaced as a GUARDED candidate naming the protected source.
            cands = mc.list_abstraction_candidates(md)
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0]["status"], "guarded")
            self.assertEqual(cands[0]["guarded_source"], "X")

    def test_guarded_pair_is_not_synthesized(self):
        """synthesize_pending leaves a guarded pair alone (only 'pending' synth)."""
        md = _mem()
        _node(md, "X", _ABS_A, salience=0.95)
        _node(md, "Y", _ABS_B, salience=0.1)
        # Record it as guarded (as drain would).
        mc._record_guarded(md, "X", "Y", 0.333, "X")
        with _MergeEnabled(), mock.patch.object(
                con, "synthesize_node", return_value={"title": "t", "body": "b"}):
            res = mc.synthesize_pending(md, budget=10)
        self.assertTrue(res["fired"])
        # Nothing proposed — the guarded entry was not picked up.
        self.assertEqual(res["proposed"], 0)
        self.assertEqual(res["processed"], 0)
        self.assertEqual(mc.list_abstraction_candidates(md)[0]["status"],
                         "guarded")

    def test_synthesize_abstraction_defensively_guards(self):
        """A direct synthesize call on a distinct high-salience pair surfaces it."""
        md = _mem()
        _node(md, "X", _ABS_A, salience=0.95)
        _node(md, "Y", _ABS_B, salience=0.1)
        mc._record_abstract(md, "X", "Y", 0.333)  # queued pending (pre-guard)
        with mock.patch.object(con, "synthesize_node",
                               return_value={"title": "t", "body": "b"}):
            res = mc.synthesize_abstraction(md, "X", "Y")
        self.assertEqual(res["status"], "guarded")
        self.assertEqual(res["guarded_source"], "X")
        # No node created, both live.
        self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])
        self.assertTrue((md / "nodes" / "X.md").exists())
        self.assertTrue((md / "nodes" / "Y.md").exists())


# ---------------------------------------------------------------------------
# (c) a TRUE-DUPLICATE high-salience pair STILL auto-merges (exempt)
# ---------------------------------------------------------------------------


class TestTrueDuplicateExempt(unittest.TestCase):

    def test_high_salience_true_dup_still_merges(self):
        """is_duplicate=True -> guard False -> the dup pick-winner merge fires."""
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            # BOTH high-salience but a TRUE duplicate (jaccard ~1.0 >= 0.85 bar).
            _node(md, "A", _DUP_BODY, access_count=1, salience=0.95)
            _node(md, "B", _DUP_BODY_RICH, access_count=5, salience=0.95)
            a_before = (md / "nodes" / "A.md").read_text(encoding="utf-8")
            _write_candidates(md, [_cand("A", "B", 0.99)])
            with _MergeEnabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            # The dup STILL merges despite high salience (exempt).
            self.assertEqual(res["merged"], 1)
            self.assertEqual(res["recorded"], 0)
            self.assertFalse((md / "nodes" / "A.md").exists())
            self.assertTrue((md / "nodes" / "B.md").exists())
            # Reversibility intact: restore the merged-away dup byte-exact.
            self.assertTrue(ia.restore_node(md, "A")["restored"])
            self.assertEqual((md / "nodes" / "A.md").read_text(encoding="utf-8"),
                             a_before)

    def test_guard_helper_exempts_duplicate(self):
        """_salience_guards_pair returns None for is_duplicate=True."""
        md = _mem()
        _node(md, "A", _DUP_BODY, salience=0.95)
        _node(md, "B", _DUP_BODY_RICH, salience=0.95)
        self.assertIsNone(
            mc._salience_guards_pair(md, "A", "B", is_duplicate=True))
        # But DISTINCT (is_duplicate=False) it would trip on A.
        self.assertEqual(
            mc._salience_guards_pair(md, "A", "B", is_duplicate=False), "A")


# ---------------------------------------------------------------------------
# (d) a LOW-salience distinct pair behaves exactly as before P3
# ---------------------------------------------------------------------------


class TestLowSalienceUnchanged(unittest.TestCase):

    def test_low_salience_distinct_recorded_pending_as_before(self):
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "X", _ABS_A, salience=0.1)
            _node(md, "Y", _ABS_B, salience=0.2)
            _write_candidates(md, [_cand("X", "Y", 0.333)])
            with _MergeEnabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            self.assertEqual(res["merged"], 0)
            self.assertEqual(res["recorded"], 1)
            # Recorded as a NORMAL pending P2 candidate (NOT guarded).
            cands = mc.list_abstraction_candidates(md)
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0]["status"], "pending")
            self.assertNotIn("guarded_source", cands[0])

    def test_no_salience_field_behaves_as_before(self):
        """A node without a salience field is never guarded (P1/P2 pre-Tier-1)."""
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "X", _ABS_A)  # no salience field at all
            _node(md, "Y", _ABS_B)
            _write_candidates(md, [_cand("X", "Y", 0.333)])
            with _MergeEnabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            self.assertEqual(res["recorded"], 1)
            self.assertEqual(mc.list_abstraction_candidates(md)[0]["status"],
                             "pending")


# ---------------------------------------------------------------------------
# (b) CONTRADICTION auto-supersede: high-salience loser surfaced, not removed
# ---------------------------------------------------------------------------


def _mem_pair(tmp: str) -> Path:
    """Two contradiction nodes; node_0 OLDER -> the loser per _pick_superseded."""
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    return md


class TestPassiveSweepSalienceGuard(unittest.TestCase):

    def _finder(self, text, *a, **k):
        if "node 0" in text:
            return [{"node_id": "node_1.md", "title": "node_1", "score": 0.97}]
        return []

    def _judge(self, text, cands):
        return [{"existing_claim_id": "node_1.md", "confidence": 0.95,
                 "explanation": "node_0 X vs node_1 not-X"}]

    def test_high_salience_loser_surfaced_not_superseded(self):
        with tempfile.TemporaryDirectory() as tmp, _enable_contradiction():
            md = _mem_pair(tmp)
            # node_0 is OLDER (the loser) AND high-salience -> guard fires.
            (md / "nodes" / "node_0.md").write_text(
                "---\nname: node_0\nvalid_from: 2026-01-01\nsalience: 0.95\n"
                "---\nbody of node 0 shared overlapping words here today\n",
                encoding="utf-8")
            (md / "nodes" / "node_1.md").write_text(
                "---\nname: node_1\nvalid_from: 2026-01-02\nsalience: 0.1\n"
                "---\nbody of node 1 shared overlapping words here today\n",
                encoding="utf-8")
            with mock.patch.object(con, "find_supersession_candidates",
                                   side_effect=self._finder), \
                    mock.patch.object(con, "judge_contradictions",
                                      side_effect=self._judge):
                out = con.passive_sweep(md, budget=10)
            # The judge confirmed, but the guard SURFACED instead of superseding.
            self.assertEqual(out["superseded"], 0)
            self.assertEqual(out["guarded"], 1)
            # The high-salience loser is STILL live (nothing removed) — reversible.
            self.assertTrue((md / "nodes" / "node_0.md").exists())
            self.assertFalse((md / "archive" / "node_0.superseded.json").exists())
            # Surfaced for review in the unified store (status surfaced-salience).
            recs = con.list_supersession_candidates(md, unresolved_only=True)
            self.assertTrue(any(r.get("status") == "surfaced-salience"
                                and r.get("old_id") == "node_0.md" for r in recs))

    def test_low_salience_loser_still_superseded(self):
        """A LOW-salience loser is auto-superseded exactly as before P3."""
        with tempfile.TemporaryDirectory() as tmp, _enable_contradiction():
            md = _mem_pair(tmp)
            (md / "nodes" / "node_0.md").write_text(
                "---\nname: node_0\nvalid_from: 2026-01-01\nsalience: 0.1\n"
                "---\nbody of node 0 shared overlapping words here today\n",
                encoding="utf-8")
            (md / "nodes" / "node_1.md").write_text(
                "---\nname: node_1\nvalid_from: 2026-01-02\nsalience: 0.1\n"
                "---\nbody of node 1 shared overlapping words here today\n",
                encoding="utf-8")
            with mock.patch.object(con, "find_supersession_candidates",
                                   side_effect=self._finder), \
                    mock.patch.object(con, "judge_contradictions",
                                      side_effect=self._judge):
                out = con.passive_sweep(md, budget=10)
            self.assertEqual(out["superseded"], 1)
            self.assertEqual(out["guarded"], 0)
            self.assertFalse((md / "nodes" / "node_0.md").exists())
            # Restorable.
            self.assertTrue((md / "archive" / "node_0.superseded.json").exists())
            self.assertTrue(ia.restore_node(md, "node_0")["restored"])


class TestOnlineSupersedeSalienceGuard(unittest.TestCase):

    def test_online_high_salience_old_node_surfaced_not_removed(self):
        from samia.core import mcp_server
        with tempfile.TemporaryDirectory() as tmp, _enable_contradiction():
            md = Path(tmp)
            (md / "nodes").mkdir(parents=True, exist_ok=True)
            (md / "biomimetic").mkdir(parents=True, exist_ok=True)
            # old is high-salience; the write would otherwise auto-supersede it.
            (md / "nodes" / "old.md").write_text(
                "---\nname: old\nsubject: topic-z\nsalience: 0.95\n"
                "---\nthe shared claim text about topic z here\n",
                encoding="utf-8")
            (md / "nodes" / "new.md").write_text(
                "---\nname: new\nsubject: topic-z\nsalience: 0.1\n"
                "---\nthe shared claim text about topic z here\n",
                encoding="utf-8")

            with mock.patch.object(
                    mcp_server, "_node_subject", return_value="topic-z"), \
                 mock.patch("samia.core.bio.active_set",
                            return_value=["old.md"]), \
                 mock.patch.object(
                     con, "find_supersession_candidates",
                     return_value=[{"node_id": "old.md", "score": 0.99}]), \
                 mock.patch.object(con, "auto_cosine_threshold",
                                   return_value=0.9):
                res = mcp_server._online_supersede(
                    md, "new.md", "the shared claim text about topic z here",
                    None)
            # Guarded, not superseded — old node still live, nothing archived.
            self.assertEqual(res["superseded"], [])
            self.assertEqual(len(res.get("guarded", [])), 1)
            self.assertTrue((md / "nodes" / "old.md").exists())
            self.assertFalse((md / "archive" / "old.superseded.json").exists())
            recs = con.list_supersession_candidates(md, unresolved_only=True)
            self.assertTrue(any(r.get("status") == "surfaced-salience"
                                for r in recs))


if __name__ == "__main__":
    unittest.main()
