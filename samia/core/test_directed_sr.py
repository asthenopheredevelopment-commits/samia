"""samia.core.test_directed_sr — tests for the directed-SR fold (FEAT-2026-06-11 P6, proposal §5.4-5.5 + §16).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the directed successor representation (phase 2):
               PRODUCER — context_extension._record_directed_transitions increments
                 biomimetic/episode_transitions.json ("A->B" -> count) for each in-window
                 co-activation pair, A->B when episode_seq(A) < episode_seq(B); legacy pairs
                 (either endpoint lacking episode_seq) are skipped; the matrix is INCREMENTED,
                 never rebuilt; written under the existing locked_update_json flock.
               CONSUMER — successor._build_directed_out_edges / _build_forward_kernel read
                 the directed counts into the FORWARD kernel M_fwd (reverse=True is the
                 TRANSPOSE M_rev, computed but not wired into N̂); legacy/no-order nodes fall
                 back PER ROW to the symmetric phase-1 kernel.
               FLAG-OFF BYTE-IDENTITY — the producer runs only when ASTHENOS_TEMPORAL_WEIGHT
                 is on, and with no episode_transitions.json the forward kernel is byte-
                 identical to phase 1, so chainogram_retrieve is unchanged.
    Depends: samia.core.context_extension; samia.core.successor; samia.core.bio; unittest,
             tempfile, json, os (stdlib). All tests use tempfile dirs and a deterministic
             _FakeVI stand-in (no torch/HF), and NEVER touch the live memory tree.

Layer 2 (What / Why):
    What: builds a tempfile corpus with explicit episode_seq frontmatter, feeds synthetic
          replay proposals (the from_node/to_node + hot_node/cold_node shape replay_sweep
          emits) to the producer, and asserts the directed counts; then asserts the consumer
          kernels (forward vs reverse) and the per-row symmetric fallback for legacy nodes.
    Why:  §5.5 — direction is decided by episode_seq order, not lexical order; the fold drops
          the standalone λR; legacy nodes never become un-rankable (symmetric fallback); and
          the whole layer is a byte-identical no-op until the master flag + λN flip it on.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from samia.core import bio as _bio
from samia.core import context_extension as ce
from samia.core import successor as sr


# ── deterministic vector stand-in (mirrors test_successor._FakeVI) ────────────────
class _FakeVI:
    """Returns a fixed hit list so the chain scorer runs its real path with no HF backend."""

    def __init__(self, memory_dir: Path, hits: list[dict]):
        self._md = memory_dir
        self._hits = hits

    def _manifest_path(self, memory_dir: Path) -> Path:
        p = Path(memory_dir) / "index" / "manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("{}", encoding="utf-8")
        return p

    def _embed_path(self, memory_dir: Path) -> Path:
        return Path(memory_dir) / "index" / "embeddings.npy"

    def query(self, memory_dir, text, top_k=8):
        return list(self._hits)


class _EnvGuard:
    """Set temporal env vars, restore the prior environment after (scaffold-parity)."""

    def __init__(self, **kv):
        self._kv = kv
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self._kv.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _write_node(md: Path, name: str, *, episode_seq: int | None,
                chain: str = "c", written_at: float = 1781827200.0) -> None:
    """Write one node with (optionally) an episode_seq line — legacy node omits it."""
    nodes = md / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    lines = [
        "---", f"name: {name}", "description: node", "type: project",
        f"chains: [{chain}]", "valid_from: 2026-06-11", "valid_to: null",
        "last_access: 2026-06-11", "access_count: 0", "relevance: 0.5",
        "tier: warm", f"written_at: {written_at}",
    ]
    if episode_seq is not None:
        lines.append(f"episode_seq: {episode_seq}")
    lines += ["---", f"body of {name}", ""]
    (nodes / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def _replay_res(*pairs: tuple[str, str]) -> dict:
    """A replay_sweep-shaped result: proposals carry from_node/to_node."""
    return {"proposals": [{"from_node": a, "to_node": b} for a, b in pairs]}


def _replay_il(*pairs: tuple[str, str]) -> dict:
    """A replay_sweep_interleaved-shaped result: proposals carry hot_node/cold_node."""
    return {"proposals": [{"hot_node": a, "cold_node": b} for a, b in pairs]}


def _transitions(md: Path) -> dict:
    fp = _bio._bio_paths(md)["episode_transitions"]
    return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}


# ════════════════════════════════════════════════════════════════════════════════
# PRODUCER — directed counting from episode_seq order
# ════════════════════════════════════════════════════════════════════════════════
class TestDirectedCounting(unittest.TestCase):
    """_record_directed_transitions increments T_dir in episode_seq order (§5.5)."""

    def test_counts_in_episode_seq_order_not_lexical(self):
        # zebra (seq 1) co-activated with alpha (seq 2): the EARLIER-encoded node precedes,
        # so the edge is zebra->alpha even though alpha < zebra lexically.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "zebra", episode_seq=1)
            _write_node(md, "alpha", episode_seq=2)
            res = ce._record_directed_transitions(
                md, _replay_res(("alpha", "zebra")), {})
            self.assertEqual(res["directed"], 1)
            self.assertEqual(res["skipped_no_seq"], 0)
            t = _transitions(md)
            self.assertEqual(t, {"zebra.md->alpha.md": 1},
                             "edge runs earlier-seq -> later-seq, NOT lexical order")

    def test_increments_not_rebuilds(self):
        # Two passes over the same pair accumulate the count (read-modify-write, §5.5).
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "a", episode_seq=1)
            _write_node(md, "b", episode_seq=2)
            ce._record_directed_transitions(md, _replay_res(("a", "b")), {})
            ce._record_directed_transitions(md, _replay_res(("a", "b")), {})
            self.assertEqual(_transitions(md), {"a.md->b.md": 2},
                             "second pass increments, does not rebuild from scratch")

    def test_interleaved_pairs_counted_too(self):
        # The interleaved variant (hot_node/cold_node) feeds the same directed pass.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "x", episode_seq=5)
            _write_node(md, "y", episode_seq=9)
            res = ce._record_directed_transitions(
                md, {}, _replay_il(("y", "x")))   # y is later-seq, x earlier
            self.assertEqual(res["directed"], 1)
            self.assertEqual(_transitions(md), {"x.md->y.md": 1})

    def test_both_directions_can_coexist(self):
        # a<b and c<b but b<d → three distinct directed keys; reverse pair never collides.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "a", episode_seq=1)
            _write_node(md, "b", episode_seq=2)
            _write_node(md, "c", episode_seq=3)
            # Pair (a,b): a<b → a->b.  Pair (c,b): b<c → b->c.
            ce._record_directed_transitions(
                md, _replay_res(("a", "b"), ("b", "c")), {})
            t = _transitions(md)
            self.assertEqual(t, {"a.md->b.md": 1, "b.md->c.md": 1})

    def test_equal_seq_skipped(self):
        # Equal episode_seq has no defined order (a dense counter never ties distinct nodes,
        # but a corrupt/legacy duplicate must still be skipped, never counted).
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "a", episode_seq=7)
            _write_node(md, "b", episode_seq=7)
            res = ce._record_directed_transitions(md, _replay_res(("a", "b")), {})
            self.assertEqual(res["directed"], 0)
            self.assertEqual(_transitions(md), {})


# ════════════════════════════════════════════════════════════════════════════════
# LEGACY SYMMETRIC FALLBACK
# ════════════════════════════════════════════════════════════════════════════════
class TestLegacyFallback(unittest.TestCase):
    """Pairs/nodes lacking episode_seq fall back to the symmetric phase-1 kernel (§5.5)."""

    def test_legacy_pair_skipped_by_producer(self):
        # b has no episode_seq → the pair has no defined order → producer skips it.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "a", episode_seq=1)
            _write_node(md, "b", episode_seq=None)   # legacy node, no order
            res = ce._record_directed_transitions(md, _replay_res(("a", "b")), {})
            self.assertEqual(res["directed"], 0)
            self.assertEqual(res["skipped_no_seq"], 1)
            self.assertEqual(_transitions(md), {})

    def test_forward_kernel_is_symmetric_without_transitions(self):
        # No episode_transitions.json → _build_forward_kernel == the phase-1 symmetric kernel
        # byte-for-byte (the strict default; phase 1 behavior preserved).
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            bio = md / "biomimetic"
            bio.mkdir(parents=True, exist_ok=True)
            (bio / "edge_weights.json").write_text(json.dumps({
                "a.md::b.md": {"w": 2.0}, "b.md::c.md": {"w": 1.0},
            }), encoding="utf-8")
            symmetric = sr._build_out_edges(_bio._load_edge_weights(md))
            forward = sr._build_forward_kernel(md)
            self.assertEqual(forward, symmetric,
                             "no directed counts ⇒ forward kernel is exactly phase-1 symmetric")

    def test_forward_kernel_per_row_fallback(self):
        # Directed mass for a->b only; node c (no directed out-mass) falls back to its
        # symmetric row from edge_weights.json. So row a is directed, row c is symmetric.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            bio = md / "biomimetic"
            bio.mkdir(parents=True, exist_ok=True)
            (bio / "edge_weights.json").write_text(json.dumps({
                "a.md::b.md": {"w": 1.0}, "b.md::c.md": {"w": 1.0},
            }), encoding="utf-8")
            (bio / "episode_transitions.json").write_text(json.dumps({
                "a.md->b.md": 4,
            }), encoding="utf-8")
            fwd = sr._build_forward_kernel(md)
            # a's row is the DIRECTED row (a->b only, normalized to 1.0) — NOT the symmetric
            # row which would also give a an edge back from b.
            self.assertEqual(set(fwd["a.md"].keys()), {"b.md"})
            self.assertAlmostEqual(fwd["a.md"]["b.md"], 1.0, places=12)
            # c has no directed out-mass → symmetric fallback row (c sees b on the undirected
            # b::c edge).
            self.assertIn("b.md", fwd["c.md"])


# ════════════════════════════════════════════════════════════════════════════════
# CONSUMER — forward vs reverse kernel
# ════════════════════════════════════════════════════════════════════════════════
class TestDirectedKernels(unittest.TestCase):
    """Forward M_fwd vs reverse M_rev (transpose); reverse is computed but not wired (§5.5)."""

    def test_forward_kernel_row_normalized(self):
        # a->b:3, a->c:1 → a's forward row normalizes to {b:3/4, c:1/4}, row-stochastic.
        t = {"a.md->b.md": 3, "a.md->c.md": 1}
        fwd = sr._build_directed_out_edges(t, reverse=False)
        self.assertAlmostEqual(sum(fwd["a.md"].values()), 1.0, places=12)
        self.assertAlmostEqual(fwd["a.md"]["b.md"], 0.75, places=12)
        self.assertAlmostEqual(fwd["a.md"]["c.md"], 0.25, places=12)
        # b and c are sinks in the forward direction (no out-mass) → no rows.
        self.assertNotIn("b.md", fwd)
        self.assertNotIn("c.md", fwd)

    def test_reverse_kernel_is_transpose(self):
        # Forward a->b means reverse b->a; the transpose flips every edge.
        t = {"a.md->b.md": 1, "a.md->c.md": 1}
        rev = sr._build_directed_out_edges(t, reverse=True)
        # b sees a, c sees a (each a sole out-edge → prob 1.0).
        self.assertEqual(set(rev.keys()), {"b.md", "c.md"})
        self.assertAlmostEqual(rev["b.md"]["a.md"], 1.0, places=12)
        self.assertAlmostEqual(rev["c.md"]["a.md"], 1.0, places=12)
        # a is a sink in the reverse direction.
        self.assertNotIn("a.md", rev)

    def test_forward_and_reverse_differ_on_asymmetric_graph(self):
        # The whole point of direction: forward ≠ reverse when the graph is directed.
        t = {"a.md->b.md": 1, "b.md->c.md": 1}
        fwd = sr._build_directed_out_edges(t, reverse=False)
        rev = sr._build_directed_out_edges(t, reverse=True)
        self.assertNotEqual(fwd, rev)
        self.assertIn("b.md", fwd["a.md"])   # forward: a → b
        self.assertIn("a.md", rev["b.md"])   # reverse: b → a

    def test_malformed_and_nonpositive_dropped(self):
        t = {
            "a.md->a.md": 5,      # self-edge → dropped
            "noarrow": 3,          # no "->" → skipped
            "a.md->b.md": -2,      # non-positive count → skipped
            "a.md->c.md": 2,       # valid
        }
        fwd = sr._build_directed_out_edges(t, reverse=False)
        self.assertEqual(set(fwd.get("a.md", {}).keys()), {"c.md"})

    def test_directed_need_prefers_forward_successor(self):
        # End-to-end consumer: with a directed a->b edge, seeding the walk at a reaches b
        # forward. The forward kernel drives need_vector (M_fwd), not the symmetric diffusion.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            bio = md / "biomimetic"
            bio.mkdir(parents=True, exist_ok=True)
            (bio / "episode_transitions.json").write_text(json.dumps({
                "a.md->b.md": 1,
            }), encoding="utf-8")
            need = sr.need_vector(md, [("a", 1.0)], gsr=0.5, l=2)
            self.assertGreater(need.get("b.md", 0.0), 0.0,
                               "forward successor b is reached from seed a")


# ════════════════════════════════════════════════════════════════════════════════
# FLAG-OFF BYTE-IDENTITY
# ════════════════════════════════════════════════════════════════════════════════
class TestFlagOffByteIdentity(unittest.TestCase):
    """The directed layer is a no-op until the master flag (producer) / λN (consumer) flip."""

    _HITS = [
        {"node": "n_one", "score": 0.40},
        {"node": "n_two", "score": 0.62},
        {"node": "n_three", "score": 0.15},
    ]

    def _build_corpus(self, md: Path) -> None:
        nodes = md / "nodes"
        chains = md / "chains"
        nodes.mkdir(parents=True, exist_ok=True)
        chains.mkdir(parents=True, exist_ok=True)
        spec = [
            ("n_one", "c_alpha", [{"label": "hebbian"}, {"label": "hebbian"}], 1),
            ("n_two", "c_beta", [], 2),
            ("n_three", "c_gamma", [{"label": "hebbian"}], 3),
        ]
        for nm, ch, edges, seq in spec:
            _write_node(md, nm, episode_seq=seq, chain=ch)
            (chains / f"{ch}.json").write_text(json.dumps({
                "name": ch, "members": [{"file": f"{nm}.md", "addr": f"{ch}.0"}],
                "edges": edges,
            }), encoding="utf-8")

    def _ranking(self, result: dict) -> list:
        out = [result.get("loaded_chains"), result.get("n_singletons")]
        for entry in result.get("loaded_nodes", []):
            out.append((entry.get("node"), entry.get("chain"),
                        entry.get("addr"), entry.get("score")))
        return out

    def _run(self, md: Path):
        return ce.chainogram_retrieve(md, "any query", _vi_module=_FakeVI(md, self._HITS))

    def test_producer_skipped_when_flag_off(self):
        # idle_replay_tick wires the producer behind temporal_weight_enabled(); with the flag
        # off the directed pass never runs, so episode_transitions.json never appears.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self._build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                # Call the producer guard the way idle_replay_tick does, directly.
                ran = ce.temporal_weight_enabled()
            self.assertFalse(ran)
            self.assertFalse(_bio._bio_paths(md)["episode_transitions"].exists())

    def test_consumer_byte_identical_with_directed_present_lambda_n_zero(self):
        # Even WITH episode_transitions.json on disk, λN=0 (flag-off) means the need term is
        # never computed → chainogram_retrieve is byte-identical to the baseline.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self._build_corpus(md)
            bio = md / "biomimetic"
            bio.mkdir(parents=True, exist_ok=True)
            (bio / "edge_weights.json").write_text(json.dumps({
                "n_one.md::n_two.md": {"w": 1.0},
                "n_two.md::n_three.md": {"w": 1.0},
            }), encoding="utf-8")
            (bio / "episode_transitions.json").write_text(json.dumps({
                "n_one.md->n_two.md": 5, "n_two.md->n_three.md": 5,
            }), encoding="utf-8")
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_LAMBDA_N=None):
                on = self._run(md)
            self.assertNotIn("error", on)
            self.assertEqual(self._ranking(ref), self._ranking(on),
                             "directed counts present but λN=0 ⇒ byte-identical")

    def test_scores_byte_identical_flag_on_zero_with_transitions(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self._build_corpus(md)
            bio = md / "biomimetic"
            bio.mkdir(parents=True, exist_ok=True)
            (bio / "episode_transitions.json").write_text(json.dumps({
                "n_one.md->n_two.md": 3,
            }), encoding="utf-8")
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"):
                on = self._run(md)
            ref_scores = [e["score"] for e in ref["loaded_nodes"]]
            on_scores = [e["score"] for e in on["loaded_nodes"]]
            self.assertEqual(ref_scores, on_scores)


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_directed_sr
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P6 — directed-SR fold (§5.4-5.5 + §16)
# Layer:      test (pytest)
# Role:       tests for samia.core.context_extension + successor (+ bio) — directed counting in episode_seq order, increment-not-rebuild, symmetric per-row legacy fallback, forward/reverse kernels, flag-off byte-identity
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.context_extension, samia.core.successor, samia.core.bio
# Exposes:    — (test module)
# Lines:      418
# ------------------------------------------------------------------------------
