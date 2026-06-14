"""samia.runtime.test_memory_guard_live_isolation — tests for the memory_guard live-pollution incident fix.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the three-part root-cause fix to the
             2026-06-07 live-memory pollution incident:
               (1) TEST ISOLATION -- a flagged write whose TARGET is NOT under
                   the live nodes dir emits ZERO live bug nodes (pytest tempdir
                   writes stop polluting live memory); a flagged write to a real
                   live-dir target STILL emits.
               (2) SELF-AMPLIFICATION GUARD -- a bug_mem_* / bug_* basename write
                   is treated as templated, so contradiction_smell is SKIPPED for
                   it (bug nodes never flag other bug nodes).
               (3) PASSIVE-SWEEP ROBUSTNESS -- find_contradiction_candidates reads
                   the dict-shaped manifest entries via the "row" field, so a
                   bug_mem-like node processes WITHOUT the "passive finder failed
                   ... : 34" (KeyError(row)) storm.
    Depends: samia.runtime.{memory_guard, contradiction, bug_records},
             unittest, unittest.mock, tempfile, json, numpy (skipped if absent).

Layer 2 (What / Why):
    What: pins the incident fix so the cascade + tempdir flood cannot recur.
    Why:  PRODUCE-ONLY -- every test that could touch live memory POINTS the
          live nodes dir at a tempdir (patches bug_records.NODES_DIR), so the
          suite never writes to the real ~/.claude/.../memory/nodes. No model
          loads (the embedder is mocked); no daemon/thread/timer.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.runtime import memory_guard as mg
from samia.runtime import contradiction as con


# ---------------------------------------------------------------------------
# Fix #1 -- test isolation: non-live target emits ZERO live bug nodes
# ---------------------------------------------------------------------------


class TestLiveTargetIsolation(unittest.TestCase):
    def test_nonlive_target_emits_no_bug_node(self):
        # Point the "live" nodes dir at tmp/live; the flagged write targets
        # tmp/elsewhere (a stand-in for a pytest tempdir). emit must be skipped.
        with tempfile.TemporaryDirectory() as tmp:
            live = Path(tmp) / "live" / "nodes"
            live.mkdir(parents=True, exist_ok=True)
            elsewhere = Path(tmp) / "elsewhere" / "nodes"
            elsewhere.mkdir(parents=True, exist_ok=True)
            target = str(elsewhere / "n.md")

            emit_calls: list = []
            with mock.patch("samia.runtime.bug_records.NODES_DIR", live), \
                 mock.patch("samia.runtime.bug_records.emit_bug_node",
                            side_effect=lambda **k: emit_calls.append(k)):
                mg._emit_bug_node_on_flag(
                    target, ["contradiction_smell:jaccard=1.000:vs=abc"], "wid-1")

            # NOTHING emitted -> NOTHING written to the live nodes dir.
            self.assertEqual(emit_calls, [])
            self.assertEqual(list(live.glob("*.md")), [])

    def test_live_target_still_emits(self):
        # A flagged write whose target IS under the live nodes dir still emits
        # a real bug node (real bugs must still be recorded).
        with tempfile.TemporaryDirectory() as tmp:
            live = Path(tmp) / "nodes"
            live.mkdir(parents=True, exist_ok=True)
            target = str(live / "real_node.md")

            emit_calls: list = []
            with mock.patch("samia.runtime.bug_records.NODES_DIR", live), \
                 mock.patch("samia.runtime.bug_records.emit_bug_node",
                            side_effect=lambda **k: emit_calls.append(k)):
                mg._emit_bug_node_on_flag(
                    target, ["injection_marker:you are now"], "wid-2")

            self.assertEqual(len(emit_calls), 1)
            self.assertEqual(emit_calls[0]["source"], "memory_guard")
            self.assertEqual(emit_calls[0]["surface"], target)

    def test_unresolvable_live_dir_fails_safe_no_emit(self):
        # FAIL-SAFE: if the live dir cannot be resolved, prefer NOT emitting.
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "n.md")
            emit_calls: list = []
            with mock.patch.object(mg, "_live_nodes_dir", return_value=None), \
                 mock.patch("samia.runtime.bug_records.emit_bug_node",
                            side_effect=lambda **k: emit_calls.append(k)):
                mg._emit_bug_node_on_flag(target, ["x"], "wid-3")
            self.assertEqual(emit_calls, [])

    def test_stage_write_to_tempdir_writes_nothing_live(self):
        # End-to-end through stage_write: a flagged write to a tempdir target
        # leaves the (tmp-pointed) live nodes dir empty.
        with tempfile.TemporaryDirectory() as tmp:
            live = Path(tmp) / "live" / "nodes"
            live.mkdir(parents=True, exist_ok=True)
            tempdir_target = str(Path(tmp) / "pytest_tmp" / "nodes" / "n.md")

            # Force a flag deterministically: stub the validator to flag.
            with mock.patch("samia.runtime.bug_records.NODES_DIR", live), \
                 mock.patch.object(
                     mg, "_validate_write",
                     return_value=("flagged",
                                   ["contradiction_smell:jaccard=1.000:vs=z"],
                                   [])), \
                 mock.patch.object(mg, "_write_pending"), \
                 mock.patch.object(mg, "STAGED_LOG", Path(tmp) / "staged.jsonl"):
                res = mg.stage_write(
                    kind="write_node", target=tempdir_target,
                    payload={"x": 1}, caller="test")

            self.assertEqual(res["verdict"], "flagged")  # still flagged in-process
            # but NO live bug node written.
            self.assertEqual(list(live.glob("*.md")), [])


# ---------------------------------------------------------------------------
# Fix #2 -- self-amplification guard: bug_mem_* basename treated as templated
# ---------------------------------------------------------------------------


class TestBugNodeTemplatedGuard(unittest.TestCase):
    def test_bug_mem_basename_is_templated(self):
        # bug_records names memory_guard nodes bug_mem_<hash>.md.
        self.assertTrue(mg._is_templated_write({}, "/x/nodes/bug_mem_0005feac6736.md"))

    def test_bug_generic_basename_is_templated(self):
        self.assertTrue(mg._is_templated_write({}, "/x/nodes/bug_aud_deadbeef99.md"))

    def test_non_bug_basename_is_not_templated(self):
        self.assertFalse(
            mg._is_templated_write({}, "/x/nodes/some_normal_node.md"))

    def test_contradiction_smell_skipped_for_bug_node_write(self):
        # The validator must NOT run contradiction_smell for a bug_mem_* write.
        target = "/x/nodes/bug_mem_cafef00dbabe.md"
        with mock.patch.object(mg, "_check_contradiction") as smell, \
             mock.patch.object(mg, "_check_injection", return_value=[]), \
             mock.patch.object(mg, "_run_llm_judge", return_value=[]):
            verdict, reasons, _meta = mg._validate_write(
                {"name": "Flagged memory write: contradiction_smell..."},
                target=target)
        smell.assert_not_called()
        self.assertEqual(verdict, "passed")
        self.assertEqual(reasons, [])

    def test_contradiction_smell_runs_for_normal_write(self):
        # Sanity: a non-templated write DOES run the smell.
        with mock.patch.object(mg, "_check_contradiction", return_value=[]) as smell, \
             mock.patch.object(mg, "_check_injection", return_value=[]), \
             mock.patch.object(mg, "_run_llm_judge", return_value=[]):
            mg._validate_write({"name": "ordinary"}, target="/x/nodes/normal.md")
        smell.assert_called_once()


# ---------------------------------------------------------------------------
# Fix #3 -- passive-sweep robustness: dict manifest entries, no KeyError(row)
# ---------------------------------------------------------------------------


class TestFinderDictManifestNoKeyError(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")

    def _build_index(self, tmp: str) -> Path:
        """Write a tempdir vector_index with the CANONICAL dict-shaped manifest.

        entries is a DICT {fname -> {sha256, title, row}} -- exactly the schema
        samia.core.vector writes -- which the old finder mis-read as a list and
        crashed on with KeyError(<row>) (str -> "34").
        """
        import numpy as np
        md = Path(tmp)
        idx = md / "vector_index"
        idx.mkdir(parents=True, exist_ok=True)
        (md / "nodes").mkdir(parents=True, exist_ok=True)
        # Two bug_mem-like nodes + one ordinary node.
        names = ["bug_mem_aaaa1111.md", "bug_mem_bbbb2222.md", "ordinary.md"]
        for n in names:
            (md / "nodes" / n).write_text(
                f"---\nname: {n}\n---\nflagged memory write contradiction smell "
                f"shared overlapping words here today\n", encoding="utf-8")
        emb = np.eye(3, 384, dtype=np.float32)
        # Row order intentionally NOT alphabetical: prove row-mapping is used.
        entries = {
            "ordinary.md": {"sha256": "s0", "title": "ordinary", "row": 0},
            "bug_mem_aaaa1111.md": {"sha256": "s1", "title": "bm a", "row": 1},
            "bug_mem_bbbb2222.md": {"sha256": "s2", "title": "bm b", "row": 2},
        }
        np.save(str(idx / "embeddings.npy"), emb)
        (idx / "manifest.json").write_text(
            json.dumps({"model_id": "test", "dim": 384, "built_at": "t",
                        "node_count": 3, "entries": entries}),
            encoding="utf-8")
        return md

    def test_find_candidates_reads_dict_manifest_via_row(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            md = self._build_index(tmp)
            # query embedding == row 1 -> top hit is bug_mem_aaaa1111.md.
            q = np.eye(3, 384, dtype=np.float32)[1]
            with mock.patch.object(con, "_embed_text", return_value=q):
                cands = con.find_contradiction_candidates(
                    "any text", memory_dir=md, threshold=0.5)
            # MUST NOT raise KeyError(row); MUST resolve the filename via "row".
            ids = {c["node_id"] for c in cands}
            self.assertIn("bug_mem_aaaa1111.md", ids)
            self.assertNotIn("ordinary.md", ids)  # cosine 0 < threshold

    def test_passive_sweep_processes_bug_node_without_34_error(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            md = self._build_index(tmp)
            # Embed every node to row-1's vector so each finds a real candidate
            # -> exercises the loop body that previously raised KeyError(row).
            q = np.eye(3, 384, dtype=np.float32)[1]
            failures: list = []

            def _capture(msg, *args, **k):
                failures.append(msg % args if args else msg)

            with mock.patch.dict("os.environ",
                                 {"ASTHENOS_CONTRADICTION_ENABLED": "1"}), \
                 mock.patch.object(con, "_embed_text", return_value=q), \
                 mock.patch.object(con, "judge_contradictions", return_value=[]), \
                 mock.patch.object(con._log, "warning", side_effect=_capture):
                out = con.passive_sweep(md, budget=10)

            # No per-node "passive finder failed ... : 34" warning storm.
            self.assertEqual(out.get("finder_failures", 0), 0)
            self.assertFalse(any("finder failed" in f or ": 34" in f
                                 for f in failures), failures)
            # The sweep made a clean pass over the bug-like nodes.
            self.assertTrue(out["enabled"])
            self.assertGreaterEqual(out["processed"], 1)

    def test_passive_sweep_finder_failures_summarized_once(self):
        # If the finder DOES raise, the sweep summarizes (one warning) instead of
        # one-per-node, and never aborts.
        with tempfile.TemporaryDirectory() as tmp:
            md = self._build_index(tmp)
            warnings_seen: list = []

            with mock.patch.dict("os.environ",
                                 {"ASTHENOS_CONTRADICTION_ENABLED": "1"}), \
                 mock.patch.object(con, "find_supersession_candidates",
                                   side_effect=KeyError(34)), \
                 mock.patch.object(con._log, "warning",
                                   side_effect=lambda m, *a, **k:
                                   warnings_seen.append(m % a if a else m)):
                out = con.passive_sweep(md, budget=10)

            self.assertEqual(out["finder_failures"], out["processed"])
            self.assertGreaterEqual(out["processed"], 1)
            # Exactly ONE summarized finder-failure warning (not one per node).
            finder_warns = [w for w in warnings_seen if "finder skipped" in w]
            self.assertEqual(len(finder_warns), 1, warnings_seen)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.test_memory_guard_live_isolation
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      2026-06-07 live-memory pollution incident fix
# Layer:      test (pytest)
# Role:       tests for samia.runtime.memory_guard, samia.runtime.contradiction —
#             non-live target test isolation, bug-node self-amplification guard,
#             passive-sweep dict-manifest robustness (no KeyError(row) storm)
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.runtime.memory_guard, samia.runtime.contradiction, samia.runtime.bug_records
# Exposes:    — (test module)
# Lines:      290
# --------------------------------------------------------------------------
