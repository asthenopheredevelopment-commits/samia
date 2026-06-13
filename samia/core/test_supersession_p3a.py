"""Tests for the P3a contradiction-supersession detector core — FEAT-2026-06-07 P3a.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the reversible negative-consolidation primitives:
             ia.forget_node(reason="contradiction") full-archive + cascade,
             ia.restore_node (byte-exact un-forget + vector un-tombstone),
             ia.detect_wrong_deletion (self-healing auto-restore),
             vector.untombstone_node (inverse of tombstone_node), and the
             contradiction.find_supersession_candidates scope + jaccard filter
             plus record_supersession_candidate. Includes the P0 no-regression
             check: a contradiction-purge still leaves ZERO dangling edges.
    Depends: samia.core.{ia, vector}, samia.runtime.contradiction, unittest,
             unittest.mock, tempfile, json.

Layer 2 (What / Why):
    What: Verifies the Q4-override reversibility chain end to end — a
          contradiction-purge archives the full node, the cascade still purges
          every store, restore re-creates the node byte-exact + un-tombstones
          its vector entry, and the self-healing watch auto-restores on
          re-assertion. The detector wrapper restricts to the online locus and
          drops low-jaccard hits.
    Why:  the override is acceptable ONLY because every auto-deletion is
          reversible; a regression here either re-opens the P0 ghost-edge
          corruption or makes a deletion permanent. All edges.db work is routed
          to a temp db_dir; the cosine finder is mocked so no model loads and
          the live ~/.local/share memory is never touched.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import ia, vector, web_store
from samia.runtime import contradiction


def _mem(tmp: str) -> Path:
    """Build a temp memory tree: A (survivor), B (to-be-superseded), edges A-B + A-A."""
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    (md / "nodes" / "A.md").write_text(
        "---\nname: A\naddress: A\n---\nbody A\n", encoding="utf-8")
    (md / "nodes" / "B.md").write_text(
        "---\nname: claim_subject\naddress: B\nvalid_to: null\n---\n"
        "B asserts the old fact\n", encoding="utf-8")
    (md / "biomimetic" / "edge_weights.json").write_text(json.dumps({
        "A.md::B.md": {"w": 0.9, "count": 3, "count_genuine": 3},
        "A.md::A.md": {"w": 0.4, "count": 1, "count_genuine": 0},
    }), encoding="utf-8")
    return md


class TestContradictionArchiveAndCascade(unittest.TestCase):
    def test_contradiction_full_archives_then_cascades_zero_dangling(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            b_raw = (md / "nodes" / "B.md").read_text(encoding="utf-8")
            # live edges.db (temp) with a B-touching edge
            conn = web_store.connect(db_dir=edb)
            web_store.upsert_edge(conn, "A.md", "B.md", 0.9)
            web_store.upsert_edge(conn, "A.md", "A.md", 0.4)
            conn.commit(); conn.close()
            # vector manifest with B
            vector._save_manifest(md, {"entries": {"A.md": {"row": 0}, "B.md": {"row": 1}}})

            stats = ia.forget_node(md, "B", reason="contradiction",
                                   db_dir=edb, superseded_by="new.md")

            # P3a: full archive written with the node's full frontmatter + body.
            arc_path = md / "archive" / "B.superseded.json"
            self.assertTrue(arc_path.exists())
            arc = json.loads(arc_path.read_text())
            self.assertEqual(arc["original_name"], "B")
            self.assertEqual(arc["reason"], "contradiction")
            self.assertEqual(arc["superseded_by"], "new.md")
            self.assertEqual(arc["frontmatter_at_halt"]["name"], "claim_subject")
            self.assertIn("B asserts the old fact", arc["body"])
            self.assertEqual(arc["original_text"], b_raw)
            self.assertEqual(stats["superseded_archive"], "archive/B.superseded.json")
            # the live file is gone (archived).
            self.assertFalse((md / "nodes" / "B.md").exists())

            # P0 no-regression: cascade still leaves ZERO dangling references.
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            self.assertEqual(set(w), {"A.md::A.md"})  # B-touching edge dropped
            c2 = sqlite3.connect(web_store._db_path(edb))
            left = c2.execute(
                "SELECT count(*) FROM edges WHERE src_node='B.md' OR dst_node='B.md'"
            ).fetchone()[0]
            c2.close()
            self.assertEqual(left, 0)
            mani = vector._load_manifest(md)
            self.assertTrue(mani["entries"]["B.md"].get("tombstoned"))


class TestRestoreNode(unittest.TestCase):
    def test_restore_round_trips_byte_exact_and_untombstones(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            original = (md / "nodes" / "B.md").read_text(encoding="utf-8")
            vector._save_manifest(md, {"entries": {"A.md": {"row": 0}, "B.md": {"row": 1}}})

            ia.forget_node(md, "B", reason="contradiction",
                           db_dir=edb, superseded_by="new.md")
            self.assertFalse((md / "nodes" / "B.md").exists())
            self.assertTrue(vector._load_manifest(md)["entries"]["B.md"].get("tombstoned"))

            res = ia.restore_node(md, "B")
            self.assertTrue(res["restored"])
            # byte-exact restoration.
            self.assertEqual((md / "nodes" / "B.md").read_text(encoding="utf-8"), original)
            # vector entry un-tombstoned -> recall re-admits it.
            self.assertNotIn(
                "tombstoned", vector._load_manifest(md)["entries"]["B.md"])
            # restore_ts stamped on the archive, restore event + log written.
            arc = json.loads((md / "archive" / "B.superseded.json").read_text())
            self.assertIsNotNone(arc["restore_ts"])
            events = (md / ".ia_events.jsonl").read_text().strip().splitlines()
            self.assertTrue(any(json.loads(e)["event"] == "restore" for e in events))
            fl = (md / "biomimetic" / "forgotten_log.jsonl").read_text().splitlines()
            self.assertTrue(any(json.loads(line)["reason"] == "restore" for line in fl))

    def test_restore_idempotent_after_first(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            ia.forget_node(md, "B", reason="contradiction", db_dir=edb)
            self.assertTrue(ia.restore_node(md, "B")["restored"])
            again = ia.restore_node(md, "B")
            self.assertFalse(again["restored"])  # already restored

    def test_restore_missing_archive_errors_soft(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            res = ia.restore_node(md, "ZZZ")
            self.assertFalse(res["restored"])
            self.assertIn("error", res)


class TestSelfHealing(unittest.TestCase):
    def test_reassertion_auto_restores(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            ia.forget_node(md, "B", reason="contradiction",
                           db_dir=edb, superseded_by="new.md")
            self.assertFalse((md / "nodes" / "B.md").exists())
            # a fresh write re-asserts the same subject ("claim_subject").
            (md / "nodes" / "reassert.md").write_text(
                "---\nname: claim_subject\naddress: R\n---\nclaim re-asserted\n",
                encoding="utf-8")
            rep = ia.detect_wrong_deletion(md, "reassert")
            self.assertIn("B", rep["restored"])
            self.assertTrue((md / "nodes" / "B.md").exists())

    def test_superseder_gone_auto_restores(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            # superseded_by points at a node that does NOT exist -> deletion unjustified.
            ia.forget_node(md, "B", reason="contradiction",
                           db_dir=edb, superseded_by="never_written.md")
            rep = ia.detect_wrong_deletion(md, "A")  # unrelated subject
            self.assertIn("B", rep["restored"])
            self.assertTrue((md / "nodes" / "B.md").exists())

    def test_valid_deletion_not_restored(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            # the superseder exists and the new write is an unrelated subject ->
            # the deletion was correct, nothing should be restored.
            (md / "nodes" / "new.md").write_text(
                "---\nname: new_fact\naddress: N\n---\nnew correct fact\n",
                encoding="utf-8")
            ia.forget_node(md, "B", reason="contradiction",
                           db_dir=edb, superseded_by="new.md")
            rep = ia.detect_wrong_deletion(md, "new")
            self.assertEqual(rep["restored"], [])
            self.assertFalse((md / "nodes" / "B.md").exists())


class TestUntombstoneInverse(unittest.TestCase):
    def test_untombstone_is_exact_inverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            vector._save_manifest(md, {"entries": {"A.md": {"row": 0}, "B.md": {"row": 1}}})
            before = vector._load_manifest(md)
            vector.tombstone_node(md, "B")
            self.assertTrue(vector._load_manifest(md)["entries"]["B.md"].get("tombstoned"))
            r = vector.untombstone_node(md, "B")
            self.assertTrue(r["untombstoned"])
            # exact inverse: manifest matches the pre-tombstone state.
            self.assertEqual(vector._load_manifest(md), before)
            # idempotent / not-tombstoned + not-in-index.
            self.assertFalse(vector.untombstone_node(md, "B")["untombstoned"])
            self.assertFalse(vector.untombstone_node(md, "ZZZ")["untombstoned"])


class TestSupersessionCandidateFinder(unittest.TestCase):
    def test_scope_and_jaccard_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            (md / "nodes").mkdir(parents=True)
            # neighbor with high lexical overlap with the incoming text.
            (md / "nodes" / "neighbor.md").write_text(
                "---\nname: neighbor\n---\nthe quick brown fox jumps over lazy dog\n",
                encoding="utf-8")
            # neighbor with NO lexical overlap (cosine hit but jaccard ~0).
            (md / "nodes" / "lowjac.md").write_text(
                "---\nname: lowjac\n---\ncompletely unrelated vocabulary terms here\n",
                encoding="utf-8")
            # a node OUTSIDE the online scope.
            (md / "nodes" / "outside.md").write_text(
                "---\nname: outside\n---\nthe quick brown fox jumps over lazy dog\n",
                encoding="utf-8")
            incoming = "the quick brown fox jumps over the lazy dog today"

            # Mock the cosine finder so no embedding model loads; it returns all three.
            fake = [
                {"node_id": "neighbor.md", "title": "neighbor", "score": 0.9},
                {"node_id": "lowjac.md", "title": "lowjac", "score": 0.88},
                {"node_id": "outside.md", "title": "outside", "score": 0.95},
            ]
            with mock.patch.object(
                    contradiction, "find_contradiction_candidates", return_value=fake):
                out = contradiction.find_supersession_candidates(
                    incoming, scope_nodes=["neighbor.md", "lowjac.md"], memory_dir=md)

            ids = {c["node_id"] for c in out}
            # outside scope -> excluded; low-jaccard -> excluded; only neighbor survives.
            self.assertEqual(ids, {"neighbor.md"})
            self.assertGreaterEqual(out[0]["jaccard"], contradiction._SUPERSESSION_JACCARD)

    def test_passive_scope_none_spans_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            (md / "nodes").mkdir(parents=True)
            (md / "nodes" / "n1.md").write_text(
                "---\nname: n1\n---\nshared overlapping words here today now\n",
                encoding="utf-8")
            fake = [{"node_id": "n1.md", "title": "n1", "score": 0.9}]
            with mock.patch.object(
                    contradiction, "find_contradiction_candidates", return_value=fake):
                out = contradiction.find_supersession_candidates(
                    "shared overlapping words here today now also",
                    scope_nodes=None, memory_dir=md)
            self.assertEqual([c["node_id"] for c in out], ["n1.md"])

    def test_record_candidate_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            rec = contradiction.record_supersession_candidate(
                md, "old", "new", cosine=0.93, jaccard=0.4, mode="online")
            self.assertEqual(rec["old_id"], "old.md")
            self.assertEqual(rec["new_id"], "new.md")
            self.assertEqual(rec["mode"], "online")
            self.assertEqual(rec["status"], "candidate")
            line = (md / "biomimetic" / "supersession_candidates.jsonl").read_text().strip()
            self.assertEqual(json.loads(line)["cosine"], 0.93)


if __name__ == "__main__":
    unittest.main()
