"""Tests for the forget_node cross-tier invalidation cascade — FEAT-2026-06-07 P0 Phase 1.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for ia.forget_node + its per-store endpoints (web_store.delete_node_edges,
             bio.forget_node_weights, bio.sweep_ghost_edges dry-run, chain.strip_member,
             vector.tombstone_node) and the freeze/merge cascade contract.
    Depends: samia.core.{ia,web_store,bio,chain,vector}, unittest, tempfile, json, sqlite3.

Layer 2 (What / Why):
    What: Verifies node death cascades to EVERY live-graph store so no dangling ghost edge
          survives: edges.db (all ref_kinds) + edge_weights.json deleted, chain membership +
          its edges stripped, vector entry tombstoned, forgotten-log appended. Plus the gated
          sweep dry-run reports ghost counts without writing.
    Why:  The audit found 84-100% ghost edges because freeze/merge unlinked the file but never
          cascaded. A regression here re-opens that corruption. edges.db tests pass a temp
          db_dir so they never touch the global memory_graph/edges.db.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from samia.core import ia, web_store, bio, chain, vector


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    # A and C live; B is the node being forgotten (file already gone, as after freeze/merge)
    (md / "nodes" / "A.md").write_text("---\nname: A\n---\nbody A\n", encoding="utf-8")
    (md / "nodes" / "C.md").write_text("---\nname: C\n---\nbody C\n", encoding="utf-8")
    (md / "biomimetic" / "edge_weights.json").write_text(json.dumps({
        "A.md::B.md": {"w": 0.9, "count": 3, "count_genuine": 3},
        "B.md::C.md": {"w": 0.5, "count": 2, "count_genuine": 0},
        "A.md::C.md": {"w": 0.6, "count": 2, "count_genuine": 2},
    }), encoding="utf-8")
    return md


class TestWebStoreDelete(unittest.TestCase):
    def test_delete_node_edges_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = web_store.connect(db_dir=tmp)
            web_store.upsert_edge(conn, "A.md", "B.md", 0.9)
            web_store.upsert_edge(conn, "A.md", "C.md", 0.6)
            conn.commit()
            r = web_store.delete_node_edges(conn, "B.md")
            self.assertEqual(r["edges_deleted"], 1)
            left = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
            self.assertEqual(left, 1)  # A-C survives
            r2 = web_store.delete_node_edges(conn, "B.md")
            self.assertEqual(r2["edges_deleted"], 0)  # idempotent
            conn.close()


class TestBioWeights(unittest.TestCase):
    def test_forget_node_weights_drops_touching(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            r = bio.forget_node_weights(md, "B")
            self.assertEqual(r["dropped"], 2)
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            self.assertEqual(set(w), {"A.md::C.md"})

    def test_sweep_dry_run_reports_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)  # B.md is NOT a live file
            rep = bio.sweep_ghost_edges(md, apply=False)
            self.assertFalse(rep["apply"])
            self.assertEqual(rep["ghost_edges"], 2)        # both B-touching edges
            self.assertEqual(rep["live_live_edges"], 1)    # A-C
            self.assertEqual(rep["attractors_before"], 1)  # A-B (w0.9,cg3) ghost attractor
            self.assertEqual(rep["attractors_after_clean"], 0)
            # dry-run wrote nothing
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            self.assertEqual(len(w), 3)


class TestChainStrip(unittest.TestCase):
    def test_strip_member_removes_member_and_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            cdir = md / "chains"
            cdir.mkdir(parents=True)
            (cdir / "k.json").write_text(json.dumps({
                "members": [
                    {"addr": "a1", "file": "nodes/A.md"},
                    {"addr": "b1", "file": "nodes/B.md"},
                ],
                "edges": [{"from": "a1", "to": "b1", "label": "hebbian"}],
            }), encoding="utf-8")
            r = chain.strip_member(cdir, "B")
            self.assertEqual(r["members_removed"], 1)
            self.assertEqual(r["edges_removed"], 1)
            data = json.loads((cdir / "k.json").read_text())
            self.assertEqual([m["addr"] for m in data["members"]], ["a1"])
            self.assertEqual(data["edges"], [])


class TestVectorTombstone(unittest.TestCase):
    def test_tombstone_sets_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            vector._save_manifest(md, {"entries": {"A.md": {"row": 0}, "B.md": {"row": 1}}})
            r = vector.tombstone_node(md, "B")
            self.assertTrue(r["tombstoned"])
            m = vector._load_manifest(md)
            self.assertTrue(m["entries"]["B.md"].get("tombstoned"))
            self.assertNotIn("tombstoned", m["entries"]["A.md"])
            # idempotent / not-in-index
            r2 = vector.tombstone_node(md, "ZZZ")
            self.assertFalse(r2["tombstoned"])


class TestForgetNodeCascade(unittest.TestCase):
    def test_cascade_purges_all_stores(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            # edges.db in a temp dir (NEVER the global one)
            conn = web_store.connect(db_dir=edb)
            web_store.upsert_edge(conn, "A.md", "B.md", 0.9)
            web_store.upsert_edge(conn, "A.md", "C.md", 0.6)
            conn.commit(); conn.close()
            # chain with B as a member
            (md / "chains" / "k.json").write_text(json.dumps({
                "members": [{"addr": "a1", "file": "nodes/A.md"},
                            {"addr": "b1", "file": "nodes/B.md"}],
                "edges": [{"from": "a1", "to": "b1", "label": "hebbian"}]}), encoding="utf-8")
            # vector manifest with B (written at the real index path)
            vector._save_manifest(md, {"entries": {"A.md": {"row": 0}, "B.md": {"row": 1}}})

            stats = ia.forget_node(md, "B", reason="test", db_dir=edb)

            # edge_weights: B-touching dropped
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            self.assertEqual(set(w), {"A.md::C.md"})
            # edges.db: B edge gone, A-C survives
            c2 = sqlite3.connect(web_store._db_path(edb))
            self.assertEqual(c2.execute("SELECT count(*) FROM edges").fetchone()[0], 1)
            c2.close()
            # chain: B stripped
            data = json.loads((md / "chains" / "k.json").read_text())
            self.assertEqual([m["addr"] for m in data["members"]], ["a1"])
            self.assertEqual(data["edges"], [])
            # vector: B tombstoned
            mani = vector._load_manifest(md)
            self.assertTrue(mani["entries"]["B.md"].get("tombstoned"))
            # forgotten-log written
            fl = (md / "biomimetic" / "forgotten_log.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(fl), 1)
            self.assertEqual(json.loads(fl[0])["id"], "B.md")
            self.assertEqual(json.loads(fl[0])["reason"], "test")


class TestG3GhostEdgeReUpsert(unittest.TestCase):
    """G3-2026-06-11: the EVERY-CYCLE sync (web_store.sync_from_consolidation) must
    not re-create an edge for a DELETED node, and must evict the dead-endpoint pair
    from edge_weights.json in the same pass — mirroring the P0 forget_node cascade."""

    def test_sync_skips_dead_endpoint_pair_and_upserts_live_live(self):
        # B.md is forgotten (no live file); A.md / C.md live (see _mem). The weights
        # map still carries two B-touching pairs + the live A-C pair.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            weights = json.loads(
                (md / "biomimetic" / "edge_weights.json").read_text())
            stats = web_store.sync_from_consolidation(
                weights, node_appearances=None, db_dir=edb, memory_dir=md)
            # Only the live-live A-C edge was formed; both B-touching pairs SKIPPED.
            self.assertEqual(stats["formed"], 1)
            self.assertEqual(stats["skipped_dead"], 2)
            self.assertEqual(set(stats["dead_keys"]),
                             {"A.md::B.md", "B.md::C.md"})
            # edges.db has exactly the one live-live edge; NO ghost row re-created.
            c = sqlite3.connect(web_store._db_path(edb))
            rows = c.execute(
                "SELECT src_node, dst_node FROM edges").fetchall()
            c.close()
            self.assertEqual(rows, [("A.md", "C.md")])

    def test_no_memory_dir_preserves_legacy_behavior(self):
        # Without memory_dir the guard is OFF — every pair >= WEAK_FORM upserts
        # (the pre-G3 behavior preserved for callers that do not pass it).
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            weights = json.loads(
                (md / "biomimetic" / "edge_weights.json").read_text())
            stats = web_store.sync_from_consolidation(
                weights, node_appearances=None, db_dir=edb)
            self.assertEqual(stats["formed"], 3)       # all three form
            self.assertEqual(stats["skipped_dead"], 0)
            self.assertEqual(stats["dead_keys"], [])

    def test_hebbian_consolidate_evicts_dead_pairs_from_edge_weights(self):
        # End-to-end through bio.hebbian_consolidate: after one cycle the dead-endpoint
        # pairs are GONE from edge_weights.json (evicted in-pass), only A-C survives.
        # The global edges.db is REDIRECTED to a temp dir (NEVER the live graph store).
        import unittest.mock as _mock
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            edb_path = str(Path(edb) / "edges.db")
            # A coactivation event (A,C live) so hebbian_consolidate runs past its
            # empty-log early return and reaches the web sync (where the eviction is).
            (md / "biomimetic" / "coactivation_log.jsonl").write_text(
                json.dumps({"nodes": ["A.md", "C.md"], "source": "genuine"}) + "\n",
                encoding="utf-8")
            # Redirect web_store's default db path to the temp db so the real sync
            # call inside hebbian_consolidate (which passes no db_dir) is isolated.
            with _mock.patch.object(web_store, "_DEFAULT_DB_PATH", edb_path), \
                 _mock.patch.object(web_store, "_DEFAULT_DB_DIR", edb):
                bio.hebbian_consolidate(md)
            w = json.loads(
                (md / "biomimetic" / "edge_weights.json").read_text())
            # Both B-touching (dead-endpoint) pairs evicted; A-C (live-live) kept.
            self.assertNotIn("A.md::B.md", w)
            self.assertNotIn("B.md::C.md", w)
            self.assertIn("A.md::C.md", w)


if __name__ == "__main__":
    unittest.main()
