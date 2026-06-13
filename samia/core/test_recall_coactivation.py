"""Tests for samia.core.mcp_server._coactivation_neighbors — Tier-0 D4 read-back.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the conservative co-activation neighbor boost on recall.
    Depends: samia.core.mcp_server, samia.core.web_store, unittest, tempfile (stdlib).

Layer 2 (What / Why):
    What: Verifies a co-activation neighbor is surfaced but (1) never outranks the hit
          that surfaced it (the clamp), (2) is clamped to just-below a weak parent,
          (3) excludes nodes already in the result, (4) respects the max-neighbors cap,
          and (5) fails open when edges.db is absent.
    Why:  D4 opens the Hebbian web to recall; the "nudge, not hijack" invariant is what
          keeps that safe. A regression here would let a weak/noisy edge displace real
          cosine hits — the exact risk the operator flagged in Q1.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from samia.core import mcp_server as mcp
from samia.core import web_store as ws


def _seed_edges(db_dir: str, edges: list[tuple[str, str, float]]) -> None:
    conn = ws.connect(db_dir=db_dir)
    try:
        for a, b, w in edges:
            ws.upsert_edge(conn, a, b, w)
        conn.commit()
    finally:
        conn.close()


class TestCoactivationNeighbors(unittest.TestCase):
    def test_neighbor_surfaced_and_below_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_edges(tmp, [("A", "B", 0.85)])
            out = mcp._coactivation_neighbors(
                [{"node": "A", "score": 0.8}], {"A"}, db_dir=tmp)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["node"], "B")
            self.assertEqual(out[0]["via"], "coactivation")
            # min(0.8*0.95, 0.5*0.85) = min(0.76, 0.425) = 0.425
            self.assertAlmostEqual(out[0]["score"], 0.425, places=3)
            self.assertLess(out[0]["score"], 0.8)   # never outranks its parent

    def test_clamped_just_below_weak_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_edges(tmp, [("A", "B", 0.85)])
            out = mcp._coactivation_neighbors(
                [{"node": "A", "score": 0.30}], {"A"}, db_dir=tmp)
            # ceiling 0.30*0.95=0.285 < lam*w 0.425 -> clamped to 0.285, below parent
            self.assertAlmostEqual(out[0]["score"], 0.285, places=3)
            self.assertLess(out[0]["score"], 0.30)

    def test_existing_nodes_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_edges(tmp, [("A", "B", 0.85)])
            out = mcp._coactivation_neighbors(
                [{"node": "A", "score": 0.8}], {"A", "B"}, db_dir=tmp)
            self.assertEqual(out, [])

    def test_max_neighbors_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_edges(tmp, [("A", f"N{i}", 0.85) for i in range(8)])
            out = mcp._coactivation_neighbors(
                [{"node": "A", "score": 0.8}], {"A"},
                db_dir=tmp, max_neighbors=3)
            self.assertEqual(len(out), 3)

    def test_fail_open_no_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = mcp._coactivation_neighbors(
                [{"node": "A", "score": 0.8}], {"A"}, db_dir=tmp)  # empty dir, no db
            self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
