"""samia.core.test_supersession_mcp — tests for the supersession MCP surface, override design B (FEAT-2026-06-07 P3b).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the OVERRIDE-design supersession surface:
             mcp_server.memory_write_node ONLINE auto-supersede on the write seam
             (exact case auto-retired RESTORABLY; weaker hit recorded not deleted;
             gated behind ASTHENOS_CONTRADICTION_ENABLED), memory_confirm_supersession
             (RESTORABLE retire, reason="supersede" now archives), memory_restore_node
             (un-forget byte-exact), memory_dismiss_supersession, and the unified store.
    Depends: samia.core.{mcp_server, web_store, vector, bio, temporal, ia},
             samia.runtime.contradiction, unittest, unittest.mock, tempfile, json, os.

Layer 2 (What / Why):
    What: Verifies design (B) — the operator OVERRIDE: online auto-supersede of the
          exact case (cosine >= bar + same subject) is RESTORABLE; a confirmed
          supersession is restorable (it was NOT before R1); weaker hits are recorded,
          not deleted; one canonical store. R8: the online behavior is inert unless
          ASTHENOS_CONTRADICTION_ENABLED=1.
    Why:  the override replaced the surface-only Q4a contract; a regression here either
          re-opens the friction (no auto) or makes an auto-deletion permanent
          (no restore). The cosine finder + active_set are mocked so no model loads;
          edges.db is routed to a temp db_dir so the live ~/.local/share is untouched.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import mcp_server, vector, temporal, ia, web_store, bio
from samia.runtime import contradiction as con


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    # an existing node that a new write about the SAME subject will supersede.
    (md / "nodes" / "old.md").write_text(
        "---\nname: shared_subject\nvalid_from: 2026-01-01\nvalid_to: null\n"
        "---\nthe old version of the shared fact\n", encoding="utf-8")
    (md / "biomimetic" / "edge_weights.json").write_text(json.dumps({
        "old.md::survivor.md": {"w": 0.9, "count": 3, "count_genuine": 3},
        "survivor.md::survivor.md": {"w": 0.4, "count": 1, "count_genuine": 0},
    }), encoding="utf-8")
    return md


class _Enabled:
    """Context manager: enable the produce-only gate for the duration of a test."""

    def __enter__(self):
        self._prev = os.environ.get("ASTHENOS_CONTRADICTION_ENABLED")
        os.environ["ASTHENOS_CONTRADICTION_ENABLED"] = "1"
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("ASTHENOS_CONTRADICTION_ENABLED", None)
        else:
            os.environ["ASTHENOS_CONTRADICTION_ENABLED"] = self._prev


class TestOnlineAutoSupersede(unittest.TestCase):
    def test_gated_off_by_default(self):
        # R8: with the flag unset, the write seam does NOT auto-supersede.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            os.environ.pop("ASTHENOS_CONTRADICTION_ENABLED", None)
            fake = [{"node_id": "old.md", "title": "shared_subject", "score": 0.99}]
            with mock.patch.object(con, "find_contradiction_candidates",
                                   return_value=fake):
                res = mcp_server.memory_write_node(
                    md, "new", "shared_subject", "desc",
                    "the new version of the shared fact")
            self.assertEqual(res["written"], "new.md")
            self.assertNotIn("supersession", res)
            self.assertTrue((md / "nodes" / "old.md").exists())  # untouched

    def test_exact_case_auto_supersedes_restorably(self):
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.TemporaryDirectory() as dbd, _Enabled():
            md = _mem(tmp)
            vector._save_manifest(md, {"entries": {
                "old.md": {"row": 0}, "new.md": {"row": 1}}})
            # detector returns the old node well above the 0.92 auto bar (post-jaccard).
            fake = [{"node_id": "old.md", "title": "shared_subject",
                     "score": 0.97, "jaccard": 0.5}]
            with mock.patch.object(con, "find_supersession_candidates",
                                   return_value=fake), \
                 mock.patch("samia.core.bio.active_set", return_value=["old.md"]), \
                 mock.patch("samia.core.web_store._DEFAULT_DB_PATH",
                            str(Path(dbd) / "edges.db")):
                res = mcp_server.memory_write_node(
                    md, "new", "shared_subject", "desc",
                    "the new version of the shared fact")

            sup = res["supersession"]
            self.assertEqual(len(sup["superseded"]), 1)
            self.assertEqual(sup["superseded"][0]["old_id"], "old.md")
            # old node was retired (file archived, restorable).
            self.assertFalse((md / "nodes" / "old.md").exists())
            self.assertTrue((md / "archive" / "old.superseded.json").exists())
            arc = json.loads((md / "archive" / "old.superseded.json").read_text())
            self.assertEqual(arc["reason"], "supersede")
            self.assertEqual(arc["superseded_by"], "new.md")
            # ghost edge purged (P0 cascade still holds).
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            self.assertEqual(set(w), {"survivor.md::survivor.md"})

            # RESTORABLE: un-forget byte-exact via the MCP wrapper.
            restore = mcp_server.memory_restore_node(md, "old")
            self.assertTrue(restore["restored"])
            self.assertTrue((md / "nodes" / "old.md").exists())

    def test_weaker_hit_recorded_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.TemporaryDirectory() as dbd, _Enabled():
            md = _mem(tmp)
            # cosine in [0.75, 0.92) → weaker hit: record, do NOT delete.
            fake = [{"node_id": "old.md", "title": "shared_subject",
                     "score": 0.80, "jaccard": 0.4}]
            with mock.patch.object(con, "find_supersession_candidates",
                                   return_value=fake), \
                 mock.patch("samia.core.bio.active_set", return_value=["old.md"]), \
                 mock.patch("samia.core.web_store._DEFAULT_DB_PATH",
                            str(Path(dbd) / "edges.db")):
                res = mcp_server.memory_write_node(
                    md, "new", "shared_subject", "desc",
                    "a topically related but not exact fact")

            sup = res["supersession"]
            self.assertEqual(sup["superseded"], [])
            self.assertEqual(len(sup["recorded"]), 1)
            # old node intact.
            self.assertTrue((md / "nodes" / "old.md").exists())
            # recorded in the unified store with mode=online, unresolved.
            cands = con.list_supersession_candidates(md)
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0]["old_id"], "old.md")
            self.assertEqual(cands[0]["mode"], "online")

    def test_different_subject_at_high_cosine_is_only_recorded(self):
        # cosine >= bar but subject DIFFERS → not the exact case → recorded only.
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.TemporaryDirectory() as dbd, _Enabled():
            md = _mem(tmp)
            fake = [{"node_id": "old.md", "title": "shared_subject",
                     "score": 0.97, "jaccard": 0.5}]
            with mock.patch.object(con, "find_supersession_candidates",
                                   return_value=fake), \
                 mock.patch("samia.core.bio.active_set", return_value=["old.md"]), \
                 mock.patch("samia.core.web_store._DEFAULT_DB_PATH",
                            str(Path(dbd) / "edges.db")):
                res = mcp_server.memory_write_node(
                    md, "new", "a_totally_different_subject", "desc",
                    "different subject entirely")
            sup = res["supersession"]
            self.assertEqual(sup["superseded"], [])
            self.assertEqual(len(sup["recorded"]), 1)
            self.assertTrue((md / "nodes" / "old.md").exists())


class TestConfirmSupersessionRestorable(unittest.TestCase):
    def test_confirm_sets_valid_to_and_is_restorable(self):
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.TemporaryDirectory() as dbd:
            md = _mem(tmp)
            vector._save_manifest(md, {"entries": {"old.md": {"row": 0}}})
            # surface a (weaker) candidate first, as the online seam would.
            con.record_supersession_candidate(md, "old", "new", cosine=0.80,
                                              mode="online")
            self.assertEqual(len(con.list_supersession_candidates(md)), 1)

            with mock.patch("samia.core.web_store._DEFAULT_DB_PATH",
                            str(Path(dbd) / "edges.db")):
                res = mcp_server.memory_confirm_supersession(
                    md, "old", valid_to="2026-06-07", new_id="new")

            # valid_to recorded in the archive (closed before the archiving forget).
            self.assertTrue(res["closed"])
            self.assertEqual(res["valid_to"], "2026-06-07")
            # R1/R3: reason="supersede" now ARCHIVES → restorable.
            self.assertFalse((md / "nodes" / "old.md").exists())
            arc_path = md / "archive" / "old.superseded.json"
            self.assertTrue(arc_path.exists())
            arc = json.loads(arc_path.read_text())
            self.assertEqual(arc["reason"], "supersede")
            self.assertEqual(arc["frontmatter_at_halt"]["valid_to"], "2026-06-07")
            # ghost edge purged.
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            self.assertEqual(set(w), {"survivor.md::survivor.md"})
            # candidate marked confirmed in the unified store.
            self.assertEqual(res["candidates_confirmed"], 1)
            self.assertEqual(con.list_supersession_candidates(md), [])

            # confirm → restore round-trips (the load-bearing R1 fix).
            restore = mcp_server.memory_restore_node(md, "old")
            self.assertTrue(restore["restored"])
            self.assertTrue((md / "nodes" / "old.md").exists())
            info = temporal.show(md, "old.md")
            self.assertEqual(info["valid_to"], "2026-06-07")

    def test_confirm_then_restore_byte_exact_round_trip(self):
        # R1 acceptance: confirm (reason="supersede") -> restore is byte-exact.
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.TemporaryDirectory() as dbd:
            md = _mem(tmp)
            # confirm with NO valid_to so the file content is unchanged → byte-exact.
            original = (md / "nodes" / "old.md").read_text(encoding="utf-8")
            with mock.patch("samia.core.web_store._DEFAULT_DB_PATH",
                            str(Path(dbd) / "edges.db")):
                # patch temporal.set_valid to a no-op so the file is not rewritten.
                with mock.patch("samia.core.temporal.set_valid"):
                    mcp_server.memory_confirm_supersession(md, "old", new_id="new")
            self.assertFalse((md / "nodes" / "old.md").exists())
            ia.restore_node(md, "old")
            self.assertEqual((md / "nodes" / "old.md").read_text(encoding="utf-8"),
                             original)


class TestDismiss(unittest.TestCase):
    def test_dismiss_resolves_without_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "new", cosine=0.80,
                                              mode="online")
            res = mcp_server.memory_dismiss_supersession(md, "old", new_id="new")
            self.assertEqual(res["dismissed"], 1)
            self.assertEqual(con.list_supersession_candidates(md), [])
            # node untouched.
            self.assertTrue((md / "nodes" / "old.md").exists())


class TestActiveSetPrimitives(unittest.TestCase):
    def test_coactivation_neighbors_reads_edges_db(self):
        with tempfile.TemporaryDirectory() as edb:
            conn = web_store.connect(db_dir=edb)
            web_store.upsert_edge(conn, "A.md", "B.md", 0.9)
            web_store.upsert_edge(conn, "A.md", "C.md", 0.5)
            conn.commit(); conn.close()
            nbs = web_store.coactivation_neighbors("A.md", db_dir=edb)
            self.assertEqual(set(nbs), {"B.md", "C.md"})
            # strongest first.
            self.assertEqual(nbs[0], "B.md")
            # a node with no edges → empty.
            self.assertEqual(web_store.coactivation_neighbors("Z.md", db_dir=edb), [])

    def test_coactivation_neighbors_no_db_is_empty(self):
        with tempfile.TemporaryDirectory() as edb:
            # no edges.db created → no-op-safe [].
            self.assertEqual(
                web_store.coactivation_neighbors("A.md", db_dir=edb), [])

    def test_active_set_unions_neighbors_and_hot_excludes_write(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = Path(tmp)
            (md / "nodes").mkdir(parents=True)
            (md / "nodes" / "W.md").write_text(
                "---\nname: W\nlast_access: 2026-06-07\n---\nbody\n", encoding="utf-8")
            (md / "nodes" / "hot.md").write_text(
                "---\nname: hot\nlast_access: 2026-06-06\n---\nbody\n", encoding="utf-8")
            conn = web_store.connect(db_dir=edb)
            web_store.upsert_edge(conn, "W.md", "nbr.md", 0.9)
            conn.commit(); conn.close()
            locus = bio.active_set(md, ["W.md"], db_dir=edb)
            # co-activation neighbor + hot node present; the write node excluded.
            self.assertIn("nbr.md", locus)
            self.assertIn("hot.md", locus)
            self.assertNotIn("W.md", locus)

    def test_fast_engram_hook_is_empty_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            self.assertEqual(bio._fast_engram_neighbors(md, ["X.md"]), [])


class TestForgetNodeWrapper(unittest.TestCase):
    def test_wrapper_runs_cascade(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dbd:
            md = _mem(tmp)
            with mock.patch("samia.core.web_store._DEFAULT_DB_PATH",
                            str(Path(dbd) / "edges.db")):
                stats = mcp_server.memory_forget_node(md, "old", reason="manual")
            self.assertEqual(stats["node"], "old.md")
            self.assertEqual(stats["reason"], "manual")
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            self.assertEqual(set(w), {"survivor.md::survivor.md"})
            fl = (md / "biomimetic" / "forgotten_log.jsonl").read_text().strip()
            self.assertEqual(json.loads(fl)["id"], "old.md")


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_supersession_mcp
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07 P3b — supersession MCP surface (operator OVERRIDE design B)
# Layer:      test (pytest)
# Role:       tests for samia.core.mcp_server supersession surface — gated online auto-supersede (restorable exact-case retire, weaker-hit record-only), restorable confirm, byte-exact restore, dismiss, plus active-set primitives and the forget cascade
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.mcp_server, samia.core.web_store, samia.core.vector, samia.core.bio, samia.core.temporal, samia.core.ia, samia.runtime.contradiction
# Exposes:    — (test module)
# Lines:      315
# ------------------------------------------------------------------------------
