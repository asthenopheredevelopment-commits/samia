"""samia.runtime.test_passive_sweep — tests for the PASSIVE supersession sweep (FEAT-2026-06-07 P3c).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the REM-subscriber PASSIVE arm of the P3 detector:
             contradiction.passive_sweep (incremental cursor advance over a
             multi-node index + full-pass-completes-across-calls + wrap;
             judge-CONFIRMED auto-supersede via the RESTORABLE path with an
             ia.restore_node round-trip + valid_to set; judge-REJECTED/uncertain
             RECORDED mode="passive" not deleted; gated OFF -> no-op;
             work_remaining reflects cursor/candidate state) AND the REM
             registration (priority 25, only fires in REM, double gate).
    Depends: samia.runtime.{contradiction, rem_cycle, rem_subscribers},
             samia.core.ia, unittest, unittest.mock, tempfile, json, os.

Layer 2 (What / Why):
    What: verifies P3c's Exit from the approved proposal — the passive sweep is
          incremental + cursor-tracked, an LLM-judge-confirmed contradiction
          auto-supersedes the loser restorably, weaker hits are recorded not
          deleted, the whole thing is inert unless enabled, and it is wired as a
          REM subscriber at priority 25 that refuses outside REM.
    Why:  PRODUCE-ONLY — every test uses a tempfile memory_dir (NEVER the live
          ~/.local/share memory or the global edges.db) and MOCKS the cosine
          finder + the LLM judge so NO embedding/LLM model loads. The override is
          acceptable ONLY because every auto-supersede is reversible; a
          regression here either makes a deletion permanent or fires the sweep
          while the operator never enabled it.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.runtime import contradiction as con
from samia.runtime import rem_cycle, rem_subscribers
from samia.core import ia


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mem(tmp: str, n: int = 1, *, with_subject: bool = False) -> Path:
    """Build a temp memory tree with n plain nodes (node_0..node_{n-1})."""
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (md / "nodes" / f"node_{i}.md").write_text(
            f"---\nname: node_{i}\naddress: node_{i}\nvalid_from: 2026-01-0{(i % 9) + 1}\n"
            f"---\nbody of node {i} shared overlapping words here today\n",
            encoding="utf-8")
    return md


def _enable():
    """Context manager turning ASTHENOS_CONTRADICTION_ENABLED on for a test."""
    return mock.patch.dict(os.environ, {"ASTHENOS_CONTRADICTION_ENABLED": "1"})


def _patch_finder_judge(finder_ret, judge_ret):
    """Patch the cosine finder + LLM judge so NO model loads.

    find_supersession_candidates is the public wrapper passive_sweep calls; we
    patch IT directly (the jaccard pre-filter inside is bypassed) plus the judge.
    """
    return (
        mock.patch.object(con, "find_supersession_candidates",
                          side_effect=finder_ret if callable(finder_ret)
                          else (lambda *a, **k: finder_ret)),
        mock.patch.object(con, "judge_contradictions",
                          side_effect=judge_ret if callable(judge_ret)
                          else (lambda *a, **k: judge_ret)),
    )


class _RegistryIsolation(unittest.TestCase):
    """Snapshot + clear the global REM subscriber registry around each test."""

    def setUp(self) -> None:
        with rem_cycle._rem_subscribers_lock:
            self._saved = dict(rem_cycle._rem_subscribers)
            rem_cycle._rem_subscribers.clear()

    def tearDown(self) -> None:
        with rem_cycle._rem_subscribers_lock:
            rem_cycle._rem_subscribers.clear()
            rem_cycle._rem_subscribers.update(self._saved)


# ---------------------------------------------------------------------------
# Gating — double gate, inert by default
# ---------------------------------------------------------------------------


class TestGatedOff(unittest.TestCase):
    def test_disabled_no_ops_nothing_superseded(self):
        # Default: ASTHENOS_CONTRADICTION_ENABLED unset -> the sweep no-ops.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp, n=3)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ASTHENOS_CONTRADICTION_ENABLED", None)
                f_p, j_p = _patch_finder_judge(
                    [{"node_id": "node_1.md", "title": "node_1", "score": 0.99}],
                    [{"existing_claim_id": "node_1.md", "confidence": 0.95}])
                with f_p as fmock, j_p as jmock:
                    out = con.passive_sweep(md)
            self.assertFalse(out["enabled"])
            self.assertEqual(out["superseded"], 0)
            self.assertEqual(out["recorded"], 0)
            # finder + judge never even called (inert).
            fmock.assert_not_called()
            jmock.assert_not_called()
            # nothing deleted, no candidate store created.
            self.assertTrue((md / "nodes" / "node_1.md").exists())
            self.assertFalse(
                (md / "biomimetic" / "supersession_candidates.jsonl").exists())


# ---------------------------------------------------------------------------
# Incremental cursor + full pass + wrap
# ---------------------------------------------------------------------------


class TestCursorIncrementsAndWraps(unittest.TestCase):
    def test_full_pass_completes_across_calls_then_wraps(self):
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=5)
            # No candidates at all -> pure cursor-advance exercise (no deletes).
            f_p, j_p = _patch_finder_judge([], [])
            with f_p, j_p:
                # budget=2 over 5 nodes -> indices 0..2, 2..4, 4..5(wrap).
                o1 = con.passive_sweep(md, budget=2)
                self.assertEqual(o1["cursor"]["index"], 2)
                self.assertEqual(o1["processed"], 2)
                self.assertFalse(o1["cursor"]["wrapped"])
                self.assertTrue(o1["work_remaining"])  # not wrapped yet

                o2 = con.passive_sweep(md, budget=2)
                self.assertEqual(o2["cursor"]["index"], 4)
                self.assertFalse(o2["cursor"]["wrapped"])

                o3 = con.passive_sweep(md, budget=2)
                # processed the final node (index 4) then wrapped to 0.
                self.assertEqual(o3["processed"], 1)
                self.assertEqual(o3["cursor"]["index"], 0)
                self.assertTrue(o3["cursor"]["wrapped"])

                # cursor persisted to rem_cursors.json under the right key.
                cur = rem_cycle.read_cursor(md, "contradiction_passive")
                self.assertEqual(cur["index"], 0)
                self.assertTrue(cur["wrapped"])

    def test_resumes_from_persisted_cursor(self):
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=4)
            rem_cycle.write_cursor(md, "contradiction_passive",
                                   {"index": 2, "total": 4})
            f_p, j_p = _patch_finder_judge([], [])
            with f_p, j_p:
                out = con.passive_sweep(md, budget=10)
            # started at 2, processed 2..4 -> wraps.
            self.assertEqual(out["processed"], 2)
            self.assertTrue(out["cursor"]["wrapped"])


# ---------------------------------------------------------------------------
# Judge-CONFIRMED -> auto-supersede via the RESTORABLE path
# ---------------------------------------------------------------------------


class TestConfirmedAutoSupersedeRestorable(unittest.TestCase):
    def test_confirmed_supersedes_loser_and_restore_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=2)
            # node_0 valid_from 2026-01-01 (OLDER -> loser); node_1 2026-01-02.
            # The sweep processes node_0 first; its candidate is node_1, the
            # judge confirms; _pick_superseded retires the OLDER one (node_0).
            def finder(text, *a, **k):
                if "node 0" in text:
                    return [{"node_id": "node_1.md", "title": "node_1", "score": 0.97}]
                return []
            def judge(text, cands):
                return [{"existing_claim_id": "node_1.md", "confidence": 0.95,
                         "explanation": "node_0 X vs node_1 not-X"}]
            f_p, j_p = _patch_finder_judge(finder, judge)
            with f_p, j_p:
                out = con.passive_sweep(md, budget=10)

            self.assertEqual(out["superseded"], 1)
            self.assertGreaterEqual(out["judged"], 1)
            # the OLDER claim (node_0) is the loser -> archived + file gone.
            self.assertFalse((md / "nodes" / "node_0.md").exists())
            self.assertTrue((md / "nodes" / "node_1.md").exists())
            arc = md / "archive" / "node_0.superseded.json"
            self.assertTrue(arc.exists())
            arc_rec = json.loads(arc.read_text())
            self.assertEqual(arc_rec["reason"], "supersede")
            self.assertEqual(arc_rec["superseded_by"], "node_1.md")
            # valid_to was set on the loser BEFORE the archive (carried in fm).
            self.assertIn("valid_to", arc_rec["frontmatter_at_halt"])
            self.assertIsNotNone(arc_rec["frontmatter_at_halt"]["valid_to"])

            # candidate recorded confirmed in the unified store, mode=passive.
            recs = con.list_supersession_candidates(md, unresolved_only=False)
            self.assertTrue(any(r["mode"] == "passive" and r["confirmed"]
                                and r["old_id"] == "node_0.md" for r in recs))

            # RESTORABLE: restore_node re-creates node_0 byte-exact.
            res = ia.restore_node(md, "node_0")
            self.assertTrue(res["restored"])
            self.assertTrue((md / "nodes" / "node_0.md").exists())
            restored = (md / "nodes" / "node_0.md").read_text(encoding="utf-8")
            self.assertEqual(restored, arc_rec["original_text"])


# ---------------------------------------------------------------------------
# Judge-REJECTED / uncertain -> RECORDED (mode=passive), not deleted
# ---------------------------------------------------------------------------


class TestRejectedRecordedNotDeleted(unittest.TestCase):
    def test_judge_rejects_candidate_recorded_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=2)
            def finder(text, *a, **k):
                if "node 0" in text:
                    return [{"node_id": "node_1.md", "title": "node_1", "score": 0.80}]
                return []
            # judge returns NOTHING -> not a confirmed contradiction.
            f_p, j_p = _patch_finder_judge(finder, [])
            with f_p, j_p:
                out = con.passive_sweep(md, budget=10)

            self.assertEqual(out["superseded"], 0)
            self.assertGreaterEqual(out["recorded"], 1)
            # nothing deleted.
            self.assertTrue((md / "nodes" / "node_0.md").exists())
            self.assertTrue((md / "nodes" / "node_1.md").exists())
            # recorded with mode=passive, un-resolved (no confirm/dismiss).
            cands = con.list_supersession_candidates(md)
            self.assertTrue(any(c["mode"] == "passive" for c in cands))
            for c in cands:
                self.assertFalse(c["confirmed"])
                self.assertFalse(c["dismissed"])

    def test_work_remaining_false_after_wrap_despite_pending_candidates(self):
        # G2-2026-06-11 (MACHINE-DRAINABLE ONLY): a WRAPPED sweep reports
        # work_remaining=False even when operator-gated pending candidates exist —
        # those candidates need OPERATOR confirmation, which no machine cycle can
        # supply, so they must NOT hold REM awake. The pending count is surfaced as
        # operator_gated_pending telemetry instead. (This test previously asserted the
        # OPPOSITE — the exact REM-never-rests bug G2 fixes.)
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=1)
            def finder(text, *a, **k):
                return [{"node_id": "extra.md", "title": "extra", "score": 0.78}]
            # plant the "extra" candidate node so it is not filtered as gone.
            (md / "nodes" / "extra.md").write_text(
                "---\nname: extra\naddress: extra\n---\nother text\n",
                encoding="utf-8")
            f_p, j_p = _patch_finder_judge(finder, [])
            with f_p, j_p:
                out = con.passive_sweep(md, budget=10)
            # the only node was processed -> cursor wraps. A candidate is recorded
            # pending operator review, but that is operator-gated -> work_remaining
            # is now False (machine work drained); the pending count is telemetry.
            self.assertTrue(out["cursor"]["wrapped"])
            self.assertGreaterEqual(out["recorded"], 1)
            self.assertFalse(out["work_remaining"])
            self.assertTrue(out.get("operator_gated_pending"))


# ---------------------------------------------------------------------------
# REM subscriber registration — priority 25, only fires in REM
# ---------------------------------------------------------------------------


class TestRegistration(_RegistryIsolation):
    def test_registered_at_priority_25_between_consolidation_and_replay(self):
        rem_subscribers.register_rem_subscribers()
        with rem_cycle._rem_subscribers_lock:
            sub = rem_cycle._rem_subscribers["contradiction_passive"]
        self.assertEqual(sub.priority, 25)
        self.assertEqual(sub.cursor_key, "contradiction_passive")
        # order: ..., consolidation(20), contradiction_passive(25), replay(30) ...
        names = rem_cycle.registered_offline_ops()
        i_con = names.index("consolidation")
        i_pas = names.index("contradiction_passive")
        i_rep = names.index("replay")
        self.assertLess(i_con, i_pas)
        self.assertLess(i_pas, i_rep)

    def test_due_condition_requires_enabled_and_nodes(self):
        rem_subscribers.register_rem_subscribers()
        with rem_cycle._rem_subscribers_lock:
            sub = rem_cycle._rem_subscribers["contradiction_passive"]
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp, n=1)
            # disabled -> not due even though nodes exist.
            os.environ.pop("ASTHENOS_CONTRADICTION_ENABLED", None)
            self.assertFalse(sub.due_fn(md))
            # enabled + nodes -> due.
            with _enable():
                self.assertTrue(sub.due_fn(md))
            # enabled but no nodes -> not due.
            with _enable(), tempfile.TemporaryDirectory() as empty:
                self.assertFalse(sub.due_fn(Path(empty)))

    def test_subscriber_refuses_outside_rem(self):
        rem_subscribers.register_rem_subscribers()
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=2)
            # system is WAKE (default) -> the subscriber wrapper refuses.
            self.assertFalse(rem_cycle.is_rem(md))
            res = rem_subscribers._sub_contradiction_passive(md)
            self.assertFalse(res["fired"])
            self.assertEqual(res["refused"], "not_in_rem")
            # nothing touched.
            self.assertTrue((md / "nodes" / "node_0.md").exists())

    def test_subscriber_fires_in_rem_and_runs_sweep(self):
        rem_subscribers.register_rem_subscribers()
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=2)
            rem_cycle.enter_rem(md, reason="test")
            self.assertTrue(rem_cycle.is_rem(md))
            f_p, j_p = _patch_finder_judge([], [])
            with f_p, j_p:
                res = rem_subscribers._sub_contradiction_passive(md)
            self.assertTrue(res["fired"])
            self.assertTrue(res["enabled"])
            self.assertIn("work_remaining", res)
            self.assertIn("cursor", res)

    def test_driver_runs_passive_only_in_rem(self):
        rem_subscribers.register_rem_subscribers()
        with tempfile.TemporaryDirectory() as tmp, _enable():
            md = _mem(tmp, n=2)
            # WAKE: the driver itself refuses (no offline work outside REM).
            awake = rem_cycle.run_due_subscribers(md)
            self.assertEqual(awake.get("refused"), "not_in_rem")
            self.assertNotIn("contradiction_passive", awake["results"])
            # REM: the driver runs the due subscribers including passive.
            rem_cycle.enter_rem(md, reason="test")
            f_p, j_p = _patch_finder_judge([], [])
            with f_p, j_p:
                ran = rem_cycle.run_due_subscribers(md)
            self.assertIn("contradiction_passive", ran["results"])


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.test_passive_sweep
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07 P3c (REM-subscriber passive supersession sweep)
# Layer:      test (pytest)
# Role:       tests for samia.runtime.contradiction, samia.runtime.rem_cycle,
#             samia.runtime.rem_subscribers — gated-off no-op, incremental cursor
#             advance/wrap, confirmed restorable auto-supersede, rejected-recorded,
#             REM-only subscriber registration at priority 25
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.runtime.contradiction, samia.runtime.rem_cycle, samia.runtime.rem_subscribers, samia.core.ia
# Exposes:    — (test module)
# Lines:      378
# --------------------------------------------------------------------------
