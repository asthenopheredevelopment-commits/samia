"""samia.core.test_hippocampus — tests for samia.core.hippocampus (FEAT-2026-06-07 Tier-1 P1 engram + P2 ring + P3 lattice + P4 inject + P5 feed-forward).

P5 additions (this file): the feed-forward Q6a check — ring-RAG and engram-RAG hits
each feed a GENUINE co-activation into Tier-0 at the memory_search seam (captured
episodes, not only live searches, drive cortical learning); and a regression guard
that assemble_inject_block STAYS co-activation-silent (P5 must not make inject feed
Tier-0 — only RAG + engram replay do). The genuine-once engram-replay machinery is
tested in test_bio.py; the salience-aware decay + freeze-exemption in test_tier.py.

P3 additions: kWTA sparse code stamped on materialize (orthogonalizes near-duplicates);
the promotion lattice + AUTO trigger (ring->engram on max(genuine-hits, salience) —
a high-salience one-shot promotes without N hits; engram->inject-eligibility on
max(attractor, salience)); and promote-before-evict (a wanted pointer is materialized
before the LRU can drop it). P3 computes ELIGIBILITY only — it never injects (P4) and
never changes decay (P5).


Layer 1 (Owns / Depends):
    Owns:    Unit tests for the P1 engram held-copy store + engram-RAG fast tier:
             materialize creates a SELF-CONTAINED copy, the copy survives main-node
             change/removal (it is a COPY, not a pointer), engram-RAG returns the copy
             recency-preferentially, and the engram arm integrates into memory_search
             without regressing the existing main recall path. PLUS the P2 ring POINTER
             store: a ring entry is a POINTER (deref-at-read, so ring-RAG reflects the
             CURRENT backing — the OPPOSITE of the engram copy invariant), LRU eviction
             bounds the ring, a dangling pointer resolves safely, and the ring arm folds
             into memory_search fail-open without regressing the main path.
    Depends: samia.core.hippocampus, samia.core.vector, samia.core.mcp_server,
             unittest, unittest.mock, tempfile, json, numpy (stdlib + numpy).

Layer 2 (What / Why):
    What: Verifies (1) materialize writes a self-contained held copy (full title + body
          + embedding row) from a source node; (2) the copy survives deleting/editing its
          source (durability — the defining COPY invariant); (3) engram_rag_query returns
          the held copy and ranks a recent copy above an equal-cosine older one; (4)
          memory_search folds the engram fast tier in (a recent engram copy out-ranks an
          equal-cosine main node) AND the main path is unchanged when the engram store is
          empty (no regression).
    Why:  P1 is the first organ of the Tier-1 fast store; the COPY-not-pointer and
          fail-open-integration invariants are load-bearing for every later phase. All
          tests use tempfile dirs and a stubbed deterministic embedder — they never touch
          the live ~/.local/share memory dir, the real vector index, or the global
          edges.db.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from samia.core import hippocampus as hip
from samia.core import vector as vi


# --------------------------------------------------------------------------
# Deterministic stub embedder: maps a token-set to a fixed unit vector so cosine
# is fully controllable without loading the real MiniLM model. Each test seeds a
# tiny vocabulary; the vector is the L2-normalized term-frequency over that vocab.
# --------------------------------------------------------------------------

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _stub_embed_batch(texts):
    out = []
    for t in texts:
        toks = t.lower().split()
        v = np.zeros(len(_VOCAB), dtype=np.float32)
        for i, w in enumerate(_VOCAB):
            v[i] = float(sum(1 for tok in toks if w in tok))
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        else:
            v = np.ones(len(_VOCAB), dtype=np.float32)
            v = v / np.linalg.norm(v)
        out.append(v)
    return np.vstack(out).astype(np.float32)


def _write_node(memory_dir: Path, name: str, title: str, body: str) -> str:
    nodes = memory_dir / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    fname = name if name.endswith(".md") else f"{name}.md"
    fm = (f"---\nname: {title}\ndescription: {title}\ntype: project\n"
          f"chains: []\nrelevance: 0.5\ntier: warm\n---\n")
    (nodes / fname).write_text(fm + body + "\n", encoding="utf-8")
    return fname


class TestMaterialize(unittest.TestCase):
    def test_materialize_creates_self_contained_copy(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha node", "alpha alpha beta")
            store = hip.EngramStore(md)
            rec = store.materialize("n1")

            # The held copy carries full content + its own embedding row.
            self.assertEqual(rec["source_ptr"], "n1.md")
            self.assertIn("alpha alpha beta", rec["body"])
            self.assertEqual(rec["title"], "Alpha node")
            self.assertEqual(rec["embedding_row"], 0)
            self.assertEqual(rec["ttl_days"], hip.ENGRAM_TTL_DAYS_DEFAULT)

            # It is persisted, addressable, and self-contained on disk.
            on_disk = store.get(rec["engram_id"])
            self.assertIsNotNone(on_disk)
            self.assertEqual(on_disk["body"], rec["body"])

            # The dedicated fast index exists (embeddings.npy + manifest.json).
            self.assertTrue(hip._engram_embed_path(md).exists())
            self.assertTrue(hip._engram_manifest_path(md).exists())
            emb = np.load(hip._engram_embed_path(md))
            self.assertEqual(emb.shape[0], 1)

    def test_materialize_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            store = hip.EngramStore(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                store.materialize("does_not_exist")

    def test_rematerialize_is_idempotent_in_place(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            store = hip.EngramStore(md)
            r1 = store.materialize("n1")
            r2 = store.materialize("n1")
            self.assertEqual(r1["engram_id"], r2["engram_id"])
            self.assertEqual(r1["embedding_row"], r2["embedding_row"])
            # Still exactly one row / one held copy.
            self.assertEqual(len(store.all()), 1)
            self.assertEqual(np.load(hip._engram_embed_path(md)).shape[0], 1)


class TestCopyDurability(unittest.TestCase):
    def test_copy_survives_source_deletion(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            fname = _write_node(md, "n1", "Alpha", "alpha alpha beta")
            store = hip.EngramStore(md)
            rec = store.materialize("n1")

            # Simulate main-node removal (freeze/merge/delete of the source).
            (md / "nodes" / fname).unlink()

            # The held copy is intact and still self-contained — it is a COPY.
            survived = store.get(rec["engram_id"])
            self.assertIsNotNone(survived)
            self.assertIn("alpha alpha beta", survived["body"])
            # And it is still retrievable by engram-RAG (no main source needed).
            hits = hip.engram_rag_query(md, "alpha", top_k=5)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["engram_id"], rec["engram_id"])

    def test_copy_survives_source_change(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            fname = _write_node(md, "n1", "Alpha", "alpha alpha")
            store = hip.EngramStore(md)
            rec = store.materialize("n1")
            original_body = store.get(rec["engram_id"])["body"]

            # Mutate the source after materialization.
            (md / "nodes" / fname).write_text(
                "---\nname: Changed\n---\nbeta beta gamma\n", encoding="utf-8")

            # The held copy still reflects the state AT materialization, unchanged.
            self.assertEqual(store.get(rec["engram_id"])["body"], original_body)
            self.assertIn("alpha alpha", store.get(rec["engram_id"])["body"])


class TestEngramRag(unittest.TestCase):
    def test_query_returns_held_copy(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha alpha")
            _write_node(md, "n2", "Beta", "beta beta beta")
            store = hip.EngramStore(md)
            store.materialize("n1")
            store.materialize("n2")

            hits = hip.engram_rag_query(md, "alpha", top_k=5)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["source_ptr"], "n1.md")
            self.assertEqual(hits[0]["via"], "engram")

    def test_empty_store_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            # No materialize calls -> empty store -> empty result, no error.
            self.assertEqual(hip.engram_rag_query(Path(tmp), "alpha"), [])

    def test_recent_copy_outranks_equal_cosine_older(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "old", "Alpha old", "alpha")
            _write_node(md, "new", "Alpha new", "alpha")
            store = hip.EngramStore(md)
            store.materialize("old")
            store.materialize("new")

            # Same embedding (both "alpha") -> identical base cosine. Force the
            # materialized_at timestamps so "old" is far older than "new".
            m = store._load_manifest()
            old_id = hip._engram_id("old")
            new_id = hip._engram_id("new")
            m["entries"][old_id]["materialized_at"] = "2000-01-01T00:00:00"
            m["entries"][new_id]["materialized_at"] = "2099-01-01T00:00:00"
            store._save_manifest(m)

            hits = hip.engram_rag_query(md, "alpha", top_k=5)
            self.assertEqual(hits[0]["source_ptr"], "new.md")
            self.assertGreaterEqual(hits[0]["score"], hits[1]["score"])
            # Equal base cosine, so recency is the only differentiator.
            self.assertAlmostEqual(hits[0]["base_score"], hits[1]["base_score"],
                                   places=5)


class TestMemorySearchIntegration(unittest.TestCase):
    """engram arm folds into memory_search; main path is unregressed when empty."""

    def _patched_vi_query(self, hits):
        """Return a fake vector.query(memory_dir, query, top_k) (GATE6: core sig)."""
        return lambda memory_dir, q, top_k=8: [dict(h) for h in hits]

    def test_engram_hit_supersedes_equal_cosine_main(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha alpha")
            hip.EngramStore(md).materialize("n1")

            # Main returns n1 at some cosine; engram returns the SAME source node,
            # recency-boosted -> the engram (fast) copy should win the slot.
            main_hits = [{"score": 0.50, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "alpha", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False)
            top = out[0]
            self.assertEqual(top.get("node"), "n1.md")
            self.assertEqual(top.get("via"), "engram")
            self.assertGreater(top["score"], 0.50)  # recency-boosted past main

    def test_no_regression_when_engram_empty(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            # No materialize -> no engram store -> pure main path.
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"},
                         {"score": 0.4, "node": "n2.md", "title": "Beta"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "anything", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False)
            self.assertEqual([h["node"] for h in out], ["n1.md", "n2.md"])
            self.assertTrue(all(h.get("via") != "engram" for h in out))

    def test_include_engram_false_skips_fast_tier(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            hip.EngramStore(md).materialize("n1")
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "alpha", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False,
                                        include_engram=False,
                                        include_ring=False)
            self.assertTrue(all(h.get("via") != "engram" for h in out))


# ════════════════════════════════════════════════════════════════════════════
# P2 — RING pointer store + ring-RAG tests.
# ════════════════════════════════════════════════════════════════════════════


class TestRingPointerInvariant(unittest.TestCase):
    """A ring entry is a POINTER (not a copy) — ring-RAG reflects the CURRENT backing.

    This is the OPPOSITE of the engram copy invariant: where the engram held copy stays
    frozen at materialization, the ring pointer derefs at query time and tracks changes.
    """

    def test_ring_entry_is_pointer_not_copy(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha")
            ring = hip.RingStore(md)
            entry = ring.add("n1")
            # The ring entry holds NO content — only a reference + metadata.
            self.assertEqual(entry["ptr"], "n1.md")
            self.assertNotIn("body", entry)
            self.assertNotIn("content", entry)
            self.assertEqual(entry["target_tier"], "main")

    def test_ring_rag_reflects_changed_backing(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            fname = _write_node(md, "n1", "Alpha", "alpha alpha alpha")
            ring = hip.RingStore(md)
            ring.add("n1")

            # A query close to the ORIGINAL backing hits it.
            hits = hip.ring_rag_query(md, "alpha", top_k=5)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["ptr"], "n1.md")
            self.assertEqual(hits[0]["via"], "ring")
            orig_alpha_score = hits[0]["score"]

            # CHANGE the backing main node to entirely different content.
            (md / "nodes" / fname).write_text(
                "---\nname: Changed\n---\nbeta beta beta\n", encoding="utf-8")

            # ring-RAG now reflects the NEW backing (the pointer invariant): the same
            # pointer scores high on "beta" and low on "alpha" — opposite of a copy.
            beta_hits = hip.ring_rag_query(md, "beta", top_k=5)
            self.assertEqual(beta_hits[0]["ptr"], "n1.md")
            self.assertGreater(beta_hits[0]["score"], 0.9)
            alpha_hits = hip.ring_rag_query(md, "alpha", top_k=5)
            self.assertLess(alpha_hits[0]["score"], orig_alpha_score)


class TestRingLRU(unittest.TestCase):
    def test_capacity_bounds_the_ring_lru(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            ring = hip.RingStore(md, capacity=3)
            for i in range(5):
                _write_node(md, f"n{i}", f"Node {i}", "alpha")
                ring.add(f"n{i}")
            live = {e["ptr"] for e in ring.entries()}
            # Capacity 3 -> only the 3 most-recently-added survive (n2,n3,n4).
            self.assertEqual(len(live), 3)
            self.assertEqual(live, {"n2.md", "n3.md", "n4.md"})
            self.assertNotIn("n0.md", live)
            self.assertNotIn("n1.md", live)

    def test_touch_refreshes_lru_recency(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            ring = hip.RingStore(md, capacity=2)
            for i in range(2):
                _write_node(md, f"n{i}", f"N{i}", "alpha")
                ring.add(f"n{i}")
            # Re-add n0 so it is most-recently-accessed, then push a third pointer.
            ring.add("n0")
            _write_node(md, "n2", "N2", "alpha")
            ring.add("n2")
            live = {e["ptr"] for e in ring.entries()}
            # n1 (least-recently-accessed) is evicted; n0 (re-touched) survives.
            self.assertEqual(live, {"n0.md", "n2.md"})


class TestRingDangling(unittest.TestCase):
    def test_dangling_pointer_resolves_safely(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            fname = _write_node(md, "n1", "Alpha", "alpha")
            ring = hip.RingStore(md)
            ring.add("n1")
            # Delete the backing main node -> the pointer is now dangling.
            (md / "nodes" / fname).unlink()
            # resolve() of the dangling entry returns None (no crash).
            entry = ring.entries()[0]
            self.assertIsNone(ring.resolve(entry))
            # ring-RAG drops the dangling pointer -> empty result, no error.
            self.assertEqual(hip.ring_rag_query(md, "alpha", top_k=5), [])

    def test_empty_ring_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            self.assertEqual(hip.ring_rag_query(Path(tmp), "alpha"), [])


class TestRingMemorySearchIntegration(unittest.TestCase):
    """ring arm folds into memory_search; main path is unregressed when empty."""

    def _patched_vi_query(self, hits):
        return lambda memory_dir, q, top_k=8: [dict(h) for h in hits]

    def test_ring_hit_supersedes_equal_cosine_main(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha alpha")
            hip.RingStore(md).add("n1")
            # Main returns n1 at a modest cosine; the ring derefs n1 to "alpha"-heavy
            # content -> cosine ~1.0 on an "alpha" query, so the ring trace wins.
            main_hits = [{"score": 0.40, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "alpha", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False,
                                        include_engram=False)
            top = out[0]
            self.assertEqual(top.get("node"), "n1.md")
            self.assertEqual(top.get("via"), "ring")

    def test_no_regression_when_ring_empty(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            # No ring.add and no engram -> pure main path, unchanged.
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"},
                         {"score": 0.4, "node": "n2.md", "title": "Beta"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "anything", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False,
                                        include_engram=False)
            self.assertEqual([h["node"] for h in out], ["n1.md", "n2.md"])
            self.assertTrue(all(h.get("via") not in ("ring", "engram") for h in out))

    def test_include_ring_false_skips_fast_tier(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            hip.RingStore(md).add("n1")
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "alpha", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False,
                                        include_engram=False,
                                        include_ring=False)
            self.assertTrue(all(h.get("via") != "ring" for h in out))


# ════════════════════════════════════════════════════════════════════════════
# P3 — kWTA on materialize + the promotion lattice + promote-before-evict tests.
#
# All use tempfile dirs + the stubbed embedder; NEVER the live ~/.local/share memory,
# the real vector index, or the global edges.db. They exercise: (1) materialize stamps
# a kWTA sparse key; (2) ring->engram auto-materializes after N genuine hits; (3) a
# HIGH-SALIENCE one-shot promotes + inject-flags WITHOUT N hits (the max(attractor,
# salience) gate); (4) engram inject_eligible is set when the attractor bar OR salience
# is met, NOT otherwise; (5) promote-before-evict materializes a wanted pointer instead
# of dropping it; (6) DECAY/relevance is UNCHANGED (P5 not done); (7) no actual
# injection happens (P4 not done — P3 only computes ELIGIBILITY).
# ════════════════════════════════════════════════════════════════════════════


def _stamp_salience(memory_dir: Path, node: str, value: float) -> None:
    """Write a `salience` value onto a node's frontmatter (the P2 source field)."""
    from samia.core import frontmatter as _fm
    fname = node if node.endswith(".md") else f"{node}.md"
    p = memory_dir / "nodes" / fname
    f, o, b = _fm.read_node(p)
    if "salience" not in f:
        o.append("salience")
    f["salience"] = float(value)
    _fm.write_node(p, f, o, b)


def _stamp_attractor(memory_dir: Path, a: str, b: str, w: float = 0.9,
                     cg: int = 3) -> None:
    """Write a Tier-0 edge_weights.json pair so attractor_strength reads it.

    A promotable attractor needs w >= HEBB_PROMOTION AND count_genuine >=
    HEBB_PROMOTE_REPEATS (=3) (bio._is_promotable; default cg=3). Used to drive the
    engram->inject gate's attractor arm.
    """
    from samia.core import bio as _bio
    fa = a if a.endswith(".md") else f"{a}.md"
    fb = b if b.endswith(".md") else f"{b}.md"
    key = "::".join(sorted([fa, fb]))
    _bio._save_edge_weights(memory_dir, {
        key: {"w": float(w), "count": cg, "count_genuine": cg,
              "count_replay": 0, "last_seen": "2026-06-07"}})


class TestKwtaOnMaterialize(unittest.TestCase):
    def test_materialize_stamps_kwta_code(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha beta")
            rec = hip.EngramStore(md).materialize("n1")
            self.assertIn("kwta_code", rec)
            self.assertIsInstance(rec["kwta_code"], list)
            self.assertTrue(rec["kwta_code"])  # non-empty for a non-zero embedding
            # Persisted on the held copy and deterministic across re-materialize.
            on_disk = hip.EngramStore(md).get(rec["engram_id"])
            self.assertEqual(on_disk["kwta_code"], rec["kwta_code"])
            rec2 = hip.EngramStore(md).materialize("n1")
            self.assertEqual(rec2["kwta_code"], rec["kwta_code"])

    def test_near_duplicate_nodes_get_distinguishable_kwta(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            # Near-duplicate but NOT identical content -> distinguishable sparse keys.
            _write_node(md, "n1", "A", "alpha alpha alpha beta")
            _write_node(md, "n2", "B", "alpha alpha beta beta")
            store = hip.EngramStore(md)
            c1 = store.materialize("n1")["kwta_code"]
            c2 = store.materialize("n2")["kwta_code"]
            self.assertNotEqual(set(c1), set(c2))  # orthogonalized, not collapsed


class TestRingToEngramFrequency(unittest.TestCase):
    def test_auto_materialize_after_n_genuine_hits(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha")
            ring = hip.RingStore(md)
            ring.add("n1")
            # Below the bar: fewer than RING_PROMOTE_HITS genuine hits -> no copy yet.
            for _ in range(hip.RING_PROMOTE_HITS - 1):
                ring.record_genuine_hit("n1")
            self.assertIsNone(hip.promote_ring_pointer(md, "n1"))
            self.assertEqual(hip.EngramStore(md).all(), [])
            # Cross the bar: the Nth genuine hit makes it promote to a kWTA-coded copy.
            ring.record_genuine_hit("n1")
            res = hip.promote_ring_pointer(md, "n1")
            self.assertIsNotNone(res)
            self.assertEqual(res["reason"], "frequency")
            copies = hip.EngramStore(md).all()
            self.assertEqual(len(copies), 1)
            self.assertEqual(copies[0]["source_ptr"], "n1.md")
            self.assertTrue(copies[0]["kwta_code"])  # kWTA-keyed (the consolidation event)


class TestHighSalienceOneShot(unittest.TestCase):
    def test_one_shot_promotes_without_n_hits(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "crit", "Critical", "alpha")
            _stamp_salience(md, "crit", 0.95)  # high-salience one-shot
            ring = hip.RingStore(md)
            ring.add("crit")
            # ZERO genuine hits -> the frequency bar is NOT met; only salience drives it.
            res = hip.promote_ring_pointer(md, "crit")
            self.assertIsNotNone(res)
            self.assertEqual(res["reason"], "salience")
            self.assertEqual(res["genuine_hits"], 0)
            # It materialized AND is inject-eligible via max(attractor, salience).
            copies = hip.EngramStore(md).all()
            self.assertEqual(len(copies), 1)
            self.assertTrue(copies[0]["inject_eligible"])

    def test_low_salience_low_frequency_does_not_promote(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "triv", "Trivial", "alpha")
            _stamp_salience(md, "triv", 0.1)  # low salience, no hits
            hip.RingStore(md).add("triv")
            self.assertIsNone(hip.promote_ring_pointer(md, "triv"))
            self.assertEqual(hip.EngramStore(md).all(), [])


class TestEngramInjectEligibility(unittest.TestCase):
    def test_inject_eligible_when_attractor_bar_met(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _write_node(md, "n2", "Beta", "beta")
            # n1 sits on a PROMOTABLE Tier-0 attractor (w>=bar, cg>=3) -> attractor arm.
            _stamp_attractor(md, "n1", "n2", w=0.9, cg=3)
            rec = hip.EngramStore(md).materialize("n1")
            self.assertTrue(hip.mark_inject_eligible(md, rec["engram_id"]))
            self.assertTrue(hip.EngramStore(md).get(rec["engram_id"])["inject_eligible"])

    def test_inject_eligible_when_salience_high(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _stamp_salience(md, "n1", 0.9)  # salience arm, no attractor
            rec = hip.EngramStore(md).materialize("n1")
            self.assertTrue(hip.mark_inject_eligible(md, rec["engram_id"]))

    def test_not_inject_eligible_when_neither_met(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _stamp_salience(md, "n1", 0.2)  # low salience, NO attractor edge
            rec = hip.EngramStore(md).materialize("n1")
            self.assertFalse(hip.mark_inject_eligible(md, rec["engram_id"]))
            self.assertFalse(hip.EngramStore(md).get(rec["engram_id"])["inject_eligible"])

    def test_sub_bar_attractor_does_not_confer_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _write_node(md, "n2", "Beta", "beta")
            # Below the bar (w<HEBB_PROMOTION) -> not a promotable attractor -> 0 strength.
            _stamp_attractor(md, "n1", "n2", w=0.5, cg=3)
            self.assertEqual(hip.attractor_strength(md, "n1"), 0.0)
            rec = hip.EngramStore(md).materialize("n1")
            self.assertFalse(hip.mark_inject_eligible(md, rec["engram_id"]))

    def test_replay_only_attractor_excluded(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _write_node(md, "n2", "Beta", "beta")
            from samia.core import bio as _bio
            # w>=bar but count_genuine==0 (replay-only) -> not promotable -> 0 strength.
            key = "::".join(sorted(["n1.md", "n2.md"]))
            _bio._save_edge_weights(md, {key: {"w": 0.9, "count": 5,
                                               "count_genuine": 0,
                                               "count_replay": 5}})
            self.assertEqual(hip.attractor_strength(md, "n1"), 0.0)


class TestPromoteBeforeEvict(unittest.TestCase):
    def test_wanted_pointer_materialized_instead_of_dropped(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            ring = hip.RingStore(md, capacity=2)
            # The first pointer is WANTED (high salience) and is about to be LRU-evicted.
            _write_node(md, "want", "Wanted", "alpha")
            _stamp_salience(md, "want", 0.95)
            ring.add("want")
            # Push two more pointers so "want" (oldest) is over capacity.
            for nm in ("n1", "n2"):
                _write_node(md, nm, nm, "beta")
                ring.add(nm)
            live = {e["ptr"] for e in ring.entries()}
            self.assertNotIn("want.md", live)  # dropped from the volatile ring
            # ...but it was MATERIALIZED to an engram copy first (no dangling loss).
            copies = hip.EngramStore(md).all()
            self.assertEqual(len(copies), 1)
            self.assertEqual(copies[0]["source_ptr"], "want.md")

    def test_genuine_hits_pointer_also_materialized_on_evict(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            ring = hip.RingStore(md, capacity=2)
            _write_node(md, "hot", "Hot", "alpha")
            ring.add("hot")
            ring.record_genuine_hit("hot")  # recent genuine hit -> wanted
            # record_genuine_hit refreshes recency, so add two NEWER pointers to evict hot.
            for nm in ("n1", "n2"):
                _write_node(md, nm, nm, "beta")
                ring.add(nm)
            # hot was the oldest after the two adds -> evicted, but materialized first.
            srcs = {c["source_ptr"] for c in hip.EngramStore(md).all()}
            self.assertIn("hot.md", srcs)

    def test_unwanted_pointer_dropped_cold(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            ring = hip.RingStore(md, capacity=2)
            _write_node(md, "cold", "Cold", "alpha")
            ring.add("cold")  # no salience, no genuine hits -> NOT wanted
            for nm in ("n1", "n2"):
                _write_node(md, nm, nm, "beta")
                ring.add(nm)
            # Dropped from the ring AND not materialized (ordinary cold eviction).
            self.assertNotIn("cold.md", {e["ptr"] for e in ring.entries()})
            self.assertEqual(hip.EngramStore(md).all(), [])


class TestP3DoesNotChangeDecayOrInject(unittest.TestCase):
    def test_decay_relevance_fields_unchanged_by_promotion(self):
        """P3 must NOT touch the decay-driving fields (relevance/tier) — that is P5."""
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            from samia.core import frontmatter as fmod
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _stamp_salience(md, "n1", 0.95)
            before, _o, _b = fmod.read_node(md / "nodes" / "n1.md")
            rel_b, tier_b = before.get("relevance"), before.get("tier")
            hip.RingStore(md).add("n1")
            hip.promote_ring_step(md)  # run the full lattice pass
            after, _o2, _b2 = fmod.read_node(md / "nodes" / "n1.md")
            self.assertEqual(after.get("relevance"), rel_b)
            self.assertEqual(after.get("tier"), tier_b)

    def test_promote_step_marks_eligibility_but_does_not_inject(self):
        """P3 computes ELIGIBILITY only — no inject block is assembled/returned (P4)."""
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _stamp_salience(md, "n1", 0.95)
            hip.RingStore(md).add("n1")
            out = hip.promote_ring_step(md)
            # The return is an ELIGIBILITY summary, not an injected context block.
            self.assertIn("inject_eligible", out)
            self.assertIn("promoted", out)
            self.assertNotIn("inject_block", out)
            self.assertNotIn("block", out)
            # The promotion lattice (P3) computes ELIGIBILITY only and never ASSEMBLES an
            # inject block — that is assemble_inject_block (P4), a SEPARATE call. P3's
            # promote_ring_step output must carry no assembled block regardless.
            self.assertNotIn("items", out)
            self.assertNotIn("tokens_used", out)


class TestMemorySearchPromoteTrigger(unittest.TestCase):
    """The AUTO trigger is INERT by default (produce-only) and additive when enabled."""

    def _patched_vi_query(self, hits):
        return lambda memory_dir, q, top_k=8: [dict(h) for h in hits]

    def test_promote_false_is_inert(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha alpha")
            _stamp_salience(md, "n1", 0.95)
            hip.RingStore(md).add("n1")
            main_hits = [{"score": 0.4, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                mcp.memory_search(md, "alpha", top_k=5,
                                  record_coactivation=False,
                                  include_coactivation_neighbors=False,
                                  include_engram=False)
            # promote defaults False -> NO promotion happened (produce-only default).
            self.assertEqual(hip.EngramStore(md).all(), [])

    def test_promote_true_drives_lattice(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha alpha alpha")
            _stamp_salience(md, "n1", 0.95)  # one-shot promotes via salience
            hip.RingStore(md).add("n1")
            main_hits = [{"score": 0.4, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                mcp.memory_search(md, "alpha", top_k=5,
                                  record_coactivation=False,
                                  include_coactivation_neighbors=False,
                                  include_engram=False, promote=True)
            # With promote=True the salient one-shot materialized to an engram copy.
            srcs = {c["source_ptr"] for c in hip.EngramStore(md).all()}
            self.assertIn("n1.md", srcs)


# ════════════════════════════════════════════════════════════════════════════
# P4 — the two inject layers + token budget + co-activation silence tests.
#
# All use tempfile dirs + the stubbed embedder; NEVER the live ~/.local/share memory,
# the real vector index, or the global edges.db. They exercise: (1) engram-inject
# contains the inject_eligible identity set; (2) ring-inject contains the working-set
# pointers; (3) the block respects the token budget (over-budget overflow dropped,
# engram identity prioritized); (4) relevance-gating orders ring-inject; (5) CO-
# ACTIVATION SILENCE — assembling the block does NOT modify edge_weights.json / record
# any genuine co-activation; (6) DECAY/relevance unchanged (P5 not done); (7) empty
# ring+engram -> empty block (fail-open).
# ════════════════════════════════════════════════════════════════════════════


def _materialize_eligible(memory_dir: Path, node: str, salience: float = 0.95):
    """Materialize a node into an engram copy and flag it inject_eligible (via salience)."""
    _stamp_salience(memory_dir, node, salience)
    rec = hip.EngramStore(memory_dir).materialize(node)
    hip.mark_inject_eligible(memory_dir, rec["engram_id"])
    return rec


class TestInjectEngramLayer(unittest.TestCase):
    def test_engram_inject_contains_inject_eligible_set(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            # One eligible identity copy + one NON-eligible copy (low salience, no attractor).
            _write_node(md, "ident", "Identity", "alpha alpha")
            _materialize_eligible(md, "ident", 0.95)
            _write_node(md, "plain", "Plain", "beta beta")
            _stamp_salience(md, "plain", 0.1)
            rec = hip.EngramStore(md).materialize("plain")
            hip.mark_inject_eligible(md, rec["engram_id"])  # stays False

            block = hip.assemble_inject_block(md, "alpha", token_budget=600)
            engram_srcs = {i["source"] for i in block["items"]
                           if i["layer"] == "engram"}
            self.assertIn("ident.md", engram_srcs)
            self.assertNotIn("plain.md", engram_srcs)  # non-eligible excluded
            self.assertEqual(block["engram_count"], 1)

    def test_engram_inject_carries_content_and_token_accounting(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "ident", "Identity", "alpha alpha gamma")
            _materialize_eligible(md, "ident", 0.95)
            block = hip.assemble_inject_block(md, "alpha", token_budget=600)
            item = next(i for i in block["items"] if i["layer"] == "engram")
            self.assertIn("alpha alpha gamma", item["content"])
            self.assertEqual(item["tokens"], hip.estimate_tokens(item["content"]))
            self.assertTrue(block["co_activation_silent"])


class TestInjectRingLayer(unittest.TestCase):
    def test_ring_inject_contains_working_set_pointers(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "r1", "Ring one", "alpha alpha")
            _write_node(md, "r2", "Ring two", "beta beta")
            hip.RingStore(md).add("r1")
            hip.RingStore(md).add("r2")
            block = hip.assemble_inject_block(md, "alpha", token_budget=600)
            ring_srcs = {i["source"] for i in block["items"] if i["layer"] == "ring"}
            self.assertEqual(ring_srcs, {"r1.md", "r2.md"})
            self.assertEqual(block["ring_count"], 2)

    def test_ring_inject_dangling_pointer_contributes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            fname = _write_node(md, "r1", "Ring one", "alpha")
            hip.RingStore(md).add("r1")
            (md / "nodes" / fname).unlink()  # backing gone -> dangling pointer
            block = hip.assemble_inject_block(md, "alpha", token_budget=600)
            self.assertEqual(block["ring_count"], 0)


class TestInjectRelevanceGate(unittest.TestCase):
    def test_relevance_orders_ring_inject(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            # Two ring pointers: one "alpha"-heavy, one "beta"-heavy. A budget too small
            # for both forces the relevance gate to PICK the relevant one for an "alpha"
            # query.
            _write_node(md, "rel", "Relevant", "alpha alpha alpha")
            _write_node(md, "irrel", "Irrelevant", "beta beta beta")
            hip.RingStore(md).add("rel")
            hip.RingStore(md).add("irrel")
            # Budget for exactly one ring body (~ a single node body's tokens).
            one = hip.estimate_tokens(
                hip.RingStore(md).resolve(hip.RingStore(md).entries()[0])["content"])
            block = hip.assemble_inject_block(md, "alpha alpha alpha",
                                              token_budget=one, engram_budget_frac=0.0)
            ring_items = [i for i in block["items"] if i["layer"] == "ring"]
            self.assertEqual(len(ring_items), 1)
            self.assertEqual(ring_items[0]["source"], "rel.md")  # the relevant one wins
            self.assertGreaterEqual(block["dropped"], 1)


class TestInjectBudgetArbitration(unittest.TestCase):
    def test_block_never_exceeds_budget(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            for i in range(4):
                _write_node(md, f"e{i}", f"E{i}", "alpha alpha alpha alpha alpha")
                _materialize_eligible(md, f"e{i}", 0.95)
            for i in range(4):
                _write_node(md, f"r{i}", f"R{i}", "alpha alpha alpha alpha alpha")
                hip.RingStore(md).add(f"r{i}")
            block = hip.assemble_inject_block(md, "alpha", token_budget=20)
            self.assertLessEqual(block["tokens_used"], 20)
            self.assertGreater(block["dropped"], 0)

    def test_engram_identity_prioritized_under_pressure(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            # One eligible engram identity + one ring pointer; a budget that fits ONE body.
            _write_node(md, "ident", "Identity", "alpha alpha alpha")
            _materialize_eligible(md, "ident", 0.95)
            _write_node(md, "ring1", "Ring", "alpha alpha alpha")
            hip.RingStore(md).add("ring1")
            body_tok = hip.estimate_tokens(
                hip.EngramStore(md).all()[0]["body"])
            block = hip.assemble_inject_block(md, "alpha",
                                              token_budget=body_tok,
                                              engram_budget_frac=1.0)
            layers = {i["layer"] for i in block["items"]}
            self.assertIn("engram", layers)        # identity set wins the slot
            self.assertNotIn("ring", layers)       # ring overflow dropped
            self.assertGreaterEqual(block["dropped"], 1)

    def test_ring_fills_remaining_budget_after_engram(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "ident", "Identity", "alpha")
            _materialize_eligible(md, "ident", 0.95)
            _write_node(md, "ring1", "Ring", "alpha")
            hip.RingStore(md).add("ring1")
            # Ample budget: BOTH layers fit -> both present.
            block = hip.assemble_inject_block(md, "alpha", token_budget=600)
            layers = [i["layer"] for i in block["items"]]
            self.assertIn("engram", layers)
            self.assertIn("ring", layers)


class TestInjectCoActivationSilence(unittest.TestCase):
    """The homeostasis keystone (D5/Q6a): assembling a block writes ZERO Tier-0 edges."""

    def test_assemble_does_not_modify_edge_weights(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            from samia.core import bio as _bio
            md = Path(tmp)
            _write_node(md, "a", "A", "alpha")
            _write_node(md, "b", "B", "beta")
            # Seed a pre-existing Tier-0 edge so the file EXISTS and has content.
            _stamp_attractor(md, "a", "b", w=0.9, cg=3)
            ew_path = _bio._bio_paths(md)["edge_weights"]
            before = ew_path.read_text(encoding="utf-8")
            before_mtime = ew_path.stat().st_mtime_ns

            # Populate both inject layers, then assemble repeatedly.
            _materialize_eligible(md, "a", 0.95)
            hip.RingStore(md).add("b")
            for _ in range(3):
                hip.assemble_inject_block(md, "alpha beta", token_budget=600)

            after = ew_path.read_text(encoding="utf-8")
            self.assertEqual(before, after)              # byte-identical: no edge written
            self.assertEqual(before_mtime, ew_path.stat().st_mtime_ns)  # not even touched

    def test_assemble_never_calls_hebbian_record(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            from samia.core import bio as _bio
            md = Path(tmp)
            _write_node(md, "a", "A", "alpha")
            _materialize_eligible(md, "a", 0.95)
            _write_node(md, "b", "B", "beta")
            hip.RingStore(md).add("b")
            # If ANY inject path tried to record a genuine co-activation, this spy fires.
            with mock.patch.object(_bio, "hebbian_record") as spy:
                hip.assemble_inject_block(md, "alpha beta", token_budget=600)
            spy.assert_not_called()


class TestInjectDoesNotChangeDecay(unittest.TestCase):
    def test_decay_relevance_fields_unchanged_by_assemble(self):
        """P4 must NOT touch the decay-driving fields (relevance/tier) — that is P5."""
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            from samia.core import frontmatter as fmod
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            _materialize_eligible(md, "n1", 0.95)
            before, _o, _b = fmod.read_node(md / "nodes" / "n1.md")
            rel_b, tier_b = before.get("relevance"), before.get("tier")
            hip.RingStore(md).add("n1")
            hip.assemble_inject_block(md, "alpha", token_budget=600)
            after, _o2, _b2 = fmod.read_node(md / "nodes" / "n1.md")
            self.assertEqual(after.get("relevance"), rel_b)
            self.assertEqual(after.get("tier"), tier_b)


class TestInjectFailOpen(unittest.TestCase):
    def test_empty_ring_and_engram_yields_empty_block(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            block = hip.assemble_inject_block(md, "alpha", token_budget=600)
            self.assertEqual(block["items"], [])
            self.assertEqual(block["tokens_used"], 0)
            self.assertEqual(block["engram_count"], 0)
            self.assertEqual(block["ring_count"], 0)
            self.assertTrue(block["co_activation_silent"])

    def test_no_inject_eligible_engram_excluded(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            # An engram copy exists but is NOT inject_eligible -> engram-inject empty.
            _write_node(md, "n1", "Alpha", "alpha")
            _stamp_salience(md, "n1", 0.1)
            rec = hip.EngramStore(md).materialize("n1")
            hip.mark_inject_eligible(md, rec["engram_id"])  # stays False
            block = hip.assemble_inject_block(md, "alpha", token_budget=600)
            self.assertEqual(block["engram_count"], 0)


class TestInjectMcpSurface(unittest.TestCase):
    """memory_inject_block + memory_search(include_inject) are INERT operator surfaces."""

    def _patched_vi_query(self, hits):
        return lambda memory_dir, q, top_k=8: [dict(h) for h in hits]

    def test_memory_inject_block_returns_block(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "ident", "Identity", "alpha alpha")
            _materialize_eligible(md, "ident", 0.95)
            block = mcp.memory_inject_block(md, "alpha", token_budget=600)
            self.assertTrue(block["co_activation_silent"])
            self.assertEqual(block["engram_count"], 1)

    def test_memory_search_include_inject_default_off(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", "alpha")
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "alpha", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False,
                                        include_engram=False, include_ring=False)
            # Default: legacy LIST return shape (no inject_block wrapper).
            self.assertIsInstance(out, list)

    def test_memory_search_include_inject_returns_wrapped(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            _write_node(md, "ident", "Identity", "alpha")
            _materialize_eligible(md, "ident", 0.95)
            main_hits = [{"score": 0.7, "node": "ident.md", "title": "Identity"}]
            with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
                out = mcp.memory_search(md, "alpha", top_k=5,
                                        record_coactivation=False,
                                        include_coactivation_neighbors=False,
                                        include_engram=False, include_ring=False,
                                        include_inject=True)
            self.assertIsInstance(out, dict)
            self.assertIn("hits", out)
            self.assertIn("inject_block", out)
            self.assertTrue(out["inject_block"]["co_activation_silent"])


# ════════════════════════════════════════════════════════════════════════════
# P5 — feed-forward (Q6a all-RAG-feeds) + inject stays co-activation-silent.
#
# Q6a: ALL RAG retrievals (main-RAG + engram-RAG + ring-RAG) feed GENUINE
# co-activations into the Tier-0 web — CAPTURED episodes drive cortical learning,
# not only live searches. Verified at the memory_search seam: a ring-RAG and an
# engram-RAG hit each land in the genuine-coactivation record (count_genuine >= 1
# after consolidation). INJECT stays co-activation-SILENT (unchanged from P4).
# All tests use tempfile dirs + the stub embedder; the genuine record is ROUTED
# into the tempfile dir (never the live ~/.local/share / real edges.db).
# ════════════════════════════════════════════════════════════════════════════


class TestFeedForwardAllRag(unittest.TestCase):
    """Q6a: ring-RAG and engram-RAG hits feed GENUINE co-activations into Tier-0."""

    def _patched_vi_query(self, hits):
        return lambda memory_dir, q, top_k=8: [dict(h) for h in hits]

    def _run_search(self, mcp, md, main_hits):
        # GATE6: memory_search now calls samia.core.vector.query +
        # samia.core.bio.hebbian_record directly. Patch only the vector query
        # (the real bio.hebbian_record must run so the genuine co-activation
        # lands on disk for the assertions below).
        with mock.patch.object(vi, "query", self._patched_vi_query(main_hits)):
            return mcp.memory_search(md, "alpha", top_k=8,
                                     record_coactivation=True,
                                     include_coactivation_neighbors=False,
                                     include_engram=True, include_ring=True)

    def test_ring_rag_hit_records_genuine_coactivation(self):
        from samia.core import mcp_server as mcp
        from samia.core import bio as _bio
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            # A main node + a RING pointer to a distinct node, both matching the query
            # so the ring-RAG hit lands in the genuine result set alongside the main hit.
            _write_node(md, "main1", "Main", "alpha alpha")
            _write_node(md, "ring1", "Ring", "alpha beta")
            hip.RingStore(md).add("ring1")
            main_hits = [{"score": 0.6, "node": "main1.md", "title": "Main"}]
            self._run_search(mcp, md, main_hits)
            # Drain the genuine co-activation log; the ring1 pair must have a genuine event.
            _bio.hebbian_consolidate(md, promote=False)
            weights = _bio._load_edge_weights(md)
            touching_ring = [v for k, v in weights.items()
                             if "ring1.md" in k.split("::")]
            self.assertTrue(touching_ring,
                            "ring-RAG hit did not feed a Tier-0 co-activation")
            self.assertTrue(all(v.get("count_genuine", 0) >= 1
                                for v in touching_ring),
                            "ring-RAG co-activation was not GENUINE (Q6a)")

    def test_engram_rag_hit_records_genuine_coactivation(self):
        from samia.core import mcp_server as mcp
        from samia.core import bio as _bio
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            md = Path(tmp)
            # A main node + an ENGRAM held copy of a distinct node, both matching the
            # query so the engram-RAG hit joins the genuine result set.
            _write_node(md, "main1", "Main", "alpha alpha")
            _write_node(md, "eng1", "Engram", "alpha beta")
            hip.EngramStore(md).materialize("eng1")
            main_hits = [{"score": 0.6, "node": "main1.md", "title": "Main"}]
            self._run_search(mcp, md, main_hits)
            _bio.hebbian_consolidate(md, promote=False)
            weights = _bio._load_edge_weights(md)
            touching_eng = [v for k, v in weights.items()
                            if "eng1.md" in k.split("::")]
            self.assertTrue(touching_eng,
                            "engram-RAG hit did not feed a Tier-0 co-activation")
            self.assertTrue(all(v.get("count_genuine", 0) >= 1
                                for v in touching_eng),
                            "engram-RAG co-activation was not GENUINE (Q6a)")


class TestInjectStaysSilentUnderP5(unittest.TestCase):
    """P5 must NOT make inject feed Tier-0 — only RAG + replay do (regression guard)."""

    def test_assemble_inject_block_records_zero_edges(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(vi, "_embed_batch", _stub_embed_batch):
            from samia.core import bio as _bio
            md = Path(tmp)
            _write_node(md, "a", "A", "alpha")
            _write_node(md, "b", "B", "beta")
            _stamp_attractor(md, "a", "b", w=0.9, cg=3)
            ew_path = _bio._bio_paths(md)["edge_weights"]
            before = ew_path.read_text(encoding="utf-8")
            _materialize_eligible(md, "a", 0.95)
            hip.RingStore(md).add("b")
            # Even with a hebbian_record spy, assembling the block must never fire it.
            with mock.patch.object(_bio, "hebbian_record") as spy:
                for _ in range(3):
                    hip.assemble_inject_block(md, "alpha beta", token_budget=600)
            spy.assert_not_called()
            self.assertEqual(before, ew_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_hippocampus
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07 Tier-1 P1 engram + P2 ring + P3 lattice + P4 inject + P5 feed-forward
# Layer:      test (pytest)
# Role:       tests for samia.core.hippocampus (+ vector, mcp_server) — engram held-copy durability/RAG, ring pointer/LRU/dangling, kWTA + promotion lattice eligibility, inject layers/budget/co-activation-silence, all-RAG genuine feed-forward
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.hippocampus, samia.core.vector, samia.core.mcp_server, samia.core.bio, samia.core.frontmatter
# Exposes:    — (test module)
# Lines:      1183
# ------------------------------------------------------------------------------
