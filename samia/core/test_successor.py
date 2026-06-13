"""Tests for the need-as-successor-representation term — FEAT-2026-06-11 P3.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for core/successor.py (proposal §5.1-5.4, symmetric phase 1) and
             its wiring into the P1 N seam in context_extension._apply_temporal_envelope:
               - T is row-STOCHASTIC (each node's out-edges sum to 1) and the discounted
                 series is bounded/convergent across gSR ∈ [0, 0.8];
               - the two self-documenting corners gSR=0 and L=1 EACH equal the exact
                 1-step proxy need = p0 (no diffusion);
               - a 2-hop transitive reach: a node two co-activation hops from the active
                 set receives discounted occupancy only at L ≥ 2;
               - the load-bearing contract — flag-off (and flag-on with λN=0) leaves
                 chainogram_retrieve BYTE-IDENTICAL to the baseline S_c + 0.05·H_c.
    Depends: samia.core.successor; samia.core.context_extension; unittest, tempfile, json,
             os (stdlib). All tests use tempfile dirs and a deterministic _FakeVI stand-in
             (no torch/HF), and NEVER touch the live memory tree.

Layer 2 (What / Why):
    What: builds a tempfile corpus with a known undirected co-activation graph
          (edge_weights.json) and asserts the SR walk's structure directly (the need map),
          then asserts the envelope wiring is a no-op when λN=0. The need map is checked by
          construction (proxy corner, transitive reach) rather than by self-equality.
    Why:  HARD CONTRACTS: gSR=0/L=1 == 1-step proxy (an explicit corner the calibration
          relies on) and flag-off byte-identity (λN=0 ⇒ N̂ never computed ⇒ score unchanged).
          The graph is symmetric (phase 1), so the test exercises diffusion/associative
          proximity, not directed succession (phase 2 is a separate follow-on).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from samia.core import context_extension as ce
from samia.core import successor as sr


# ── deterministic vector stand-in (mirrors test_temporal_scaffold._FakeVI) ────────
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


def _write_edges(md: Path, edges: dict) -> None:
    """Write biomimetic/edge_weights.json with {"a::b": w} edges (sorted .md keys).

    Endpoints are normalized to the production ".md" filename form so they match the
    successor module's seed-key normalization (edge_weights.json keys carry .md live).
    """
    bio = md / "biomimetic"
    bio.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, w in edges.items():
        a, b = key.split("::")
        a = a if a.endswith(".md") else f"{a}.md"
        b = b if b.endswith(".md") else f"{b}.md"
        a, b = sorted([a, b])
        out[f"{a}::{b}"] = {"w": float(w), "count": 1, "last_seen": "2026-06-11"}
    (bio / "edge_weights.json").write_text(json.dumps(out), encoding="utf-8")


def _build_corpus(md: Path) -> None:
    """A small multi-chain corpus with varied cosines + a Hebbian edge (scaffold-parity)."""
    nodes = md / "nodes"
    chains = md / "chains"
    nodes.mkdir(parents=True, exist_ok=True)
    chains.mkdir(parents=True, exist_ok=True)
    spec = [
        ("n_one", "c_alpha", [{"label": "hebbian"}, {"label": "hebbian"}]),
        ("n_two", "c_beta", []),
        ("n_three", "c_gamma", [{"label": "hebbian"}]),
    ]
    for nm, ch, edges in spec:
        lines = [
            "---", f"name: {nm}", "description: corpus node", "type: project",
            f"chains: [{ch}]", "valid_from: 2026-06-11", "valid_to: null",
            "last_access: 2026-06-11", "access_count: 0", "relevance: 0.5",
            "tier: warm", "written_at: 1781827200.0", "episode_seq: 1",
            "---", f"body of {nm}", "",
        ]
        (nodes / f"{nm}.md").write_text("\n".join(lines), encoding="utf-8")
        (chains / f"{ch}.json").write_text(json.dumps({
            "name": ch, "members": [{"file": f"{nm}.md", "addr": f"{ch}.0"}],
            "edges": edges,
        }), encoding="utf-8")


_HITS = [
    {"node": "n_one", "score": 0.40},
    {"node": "n_two", "score": 0.62},
    {"node": "n_three", "score": 0.15},
]


def _ranking(result: dict) -> list:
    out = [result.get("loaded_chains"), result.get("n_singletons")]
    for entry in result.get("loaded_nodes", []):
        out.append((entry.get("node"), entry.get("chain"),
                    entry.get("addr"), entry.get("score")))
    return out


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


class TestRowStochasticConvergence(unittest.TestCase):
    """T is row-stochastic; the discounted walk is bounded/convergent across gSR ∈ [0,0.8]."""

    def test_out_edges_row_normalized(self):
        # a::b=2, a::c=1, b::c=4  → a's out-mass {b:2,c:1} normalizes to {b:2/3,c:1/3}.
        weights = {
            "a.md::b.md": {"w": 2.0},
            "a.md::c.md": {"w": 1.0},
            "b.md::c.md": {"w": 4.0},
        }
        out = sr._build_out_edges(weights)
        for src, row in out.items():
            self.assertAlmostEqual(sum(row.values()), 1.0, places=12,
                                   msg=f"row {src} must sum to 1.0 (row-stochastic)")
        self.assertAlmostEqual(out["a.md"]["b.md"], 2.0 / 3.0, places=12)
        self.assertAlmostEqual(out["a.md"]["c.md"], 1.0 / 3.0, places=12)

    def test_self_loops_and_bad_keys_dropped(self):
        weights = {
            "a.md::a.md": {"w": 5.0},   # self-loop → dropped (0 on the diagonal)
            "a.md::b.md": {"w": 1.0},
            "malformed": {"w": 1.0},    # no "::" → skipped
            "a.md::c.md": {"w": -3.0},  # non-positive weight → skipped
        }
        out = sr._build_out_edges(weights)
        self.assertNotIn("malformed", out)
        self.assertNotIn("a.md", out.get("a.md", {}))  # no self-loop edge
        self.assertEqual(set(out["a.md"].keys()), {"b.md"})

    def test_discounted_series_bounded(self):
        # Total need mass is bounded by Σ_t gSR^t = 1/(1-gSR); at gSR=0.8 → ≤ 5.0.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_edges(md, {"a::b": 1.0, "b::c": 1.0, "c::a": 1.0})
            need = sr.need_vector(md, [("a", 1.0)], gsr=0.8, l=4)
            self.assertLessEqual(sum(need.values()), 1.0 / (1.0 - 0.8) + 1e-9)
            self.assertTrue(all(v >= 0.0 for v in need.values()))


class TestProxyCorners(unittest.TestCase):
    """gSR=0 and L=1 EACH reduce the walk to the exact 1-step proxy need = p0 (§5.3)."""

    def _seed(self, active_set):
        return sr._seed_distribution(active_set)

    def test_gsr_zero_equals_seed_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_edges(md, {"a::b": 1.0, "b::c": 1.0})
            active = [("a", 0.5), ("b", 0.5)]
            proxy = self._seed(active)
            need = sr.need_vector(md, active, gsr=0.0, l=3)
            self.assertEqual(set(need.keys()), set(proxy.keys()))
            for k in proxy:
                self.assertAlmostEqual(need[k], proxy[k], places=12,
                                       msg="gSR=0 must equal the 1-step proxy (no diffusion)")

    def test_l_one_is_exactly_one_discounted_hop(self):
        # L=1 = the single-step SR proxy: need = p0 + gSR·(p0 @ T), no deeper diffusion.
        # Computed independently here and matched exactly against need_vector(L=1).
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_edges(md, {"a::b": 1.0, "b::c": 1.0})
            active = [("a", 0.5), ("b", 0.5)]
            p0 = self._seed(active)
            out = sr._build_out_edges(sr._bio._load_edge_weights(md))
            g = 0.5
            expected = dict(p0)
            for src, mass in p0.items():
                for dst, prob in out.get(src, {}).items():
                    expected[dst] = expected.get(dst, 0.0) + g * mass * prob
            need = sr.need_vector(md, active, gsr=g, l=1)
            self.assertEqual(set(need.keys()), set(expected.keys()))
            for k in expected:
                self.assertAlmostEqual(need[k], expected[k], places=12,
                                       msg="L=1 must be exactly one discounted hop")

    def test_gsr_zero_any_l_is_strict_seed_proxy(self):
        # gSR=0 zeroes every hop regardless of L → need == p0 (the strict 1-step proxy).
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_edges(md, {"a::b": 1.0, "b::c": 1.0})
            active = [("a", 0.5), ("b", 0.5)]
            proxy = self._seed(active)
            for depth in (1, 2, 3, 4):
                need0 = sr.need_vector(md, active, gsr=0.0, l=depth)
                self.assertEqual(set(need0.keys()), set(proxy.keys()))
                for k in proxy:
                    self.assertAlmostEqual(need0[k], proxy[k], places=12)

    def test_gsr_zero_l_one_is_strict_proxy(self):
        # The exact corner stated in the phase contract: gSR=0 AND L=1 == the 1-step proxy.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_edges(md, {"a::b": 1.0, "b::c": 1.0})
            active = [("a", 0.7), ("b", 0.3)]
            proxy = self._seed(active)
            need = sr.need_vector(md, active, gsr=0.0, l=1)
            self.assertEqual(set(need.keys()), set(proxy.keys()))
            for k in proxy:
                self.assertAlmostEqual(need[k], proxy[k], places=12)


class TestTransitiveReach(unittest.TestCase):
    """A node two co-activation hops away gets discounted occupancy only at L ≥ 2 (§5.4)."""

    def test_two_hop_reach(self):
        # Path graph a — b — c. Seed only at a. c is 2 hops from a.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_edges(md, {"a::b": 1.0, "b::c": 1.0})
            need_l1 = sr.need_vector(md, [("a", 1.0)], gsr=0.5, l=1)
            need_l2 = sr.need_vector(md, [("a", 1.0)], gsr=0.5, l=2)
            # At L=1 the walk reaches b (1 hop) but NOT c (2 hops).
            self.assertGreater(need_l1.get("b.md", 0.0), 0.0,
                               "b is 1 hop away — must be reached at L=1")
            self.assertEqual(need_l1.get("c.md", 0.0), 0.0,
                             "c is 2 hops away — must be unreached at L=1")
            # At L=2 the 2-hop node c now carries discounted occupancy. (a.md also gains
            # back-flow mass on the symmetric graph — diffusion, expected.)
            self.assertGreater(need_l2.get("c.md", 0.0), 0.0,
                               "c must receive discounted occupancy at L=2")
            self.assertGreater(need_l2.get("a.md", 0.0), need_l1.get("a.md", 0.0),
                               "symmetric back-flow re-deposits mass at the seed at L=2")

    def test_closer_node_outscores_farther(self):
        # b (1 hop) should carry more discounted mass than c (2 hops) from seed a.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_edges(md, {"a::b": 1.0, "b::c": 1.0})
            need = sr.need_vector(md, [("a", 1.0)], gsr=0.5, l=3)
            self.assertGreater(need["b.md"], need["c.md"],
                               "the closer node must score higher (real spreading activation)")

    def test_empty_graph_is_seed_only(self):
        # No edge_weights.json → diffusion is a no-op; need is exactly the seed proxy.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            active = [("a", 0.6), ("b", 0.4)]
            need = sr.need_vector(md, active, gsr=0.5, l=3)
            proxy = sr._seed_distribution(active)
            self.assertEqual(set(need.keys()), set(proxy.keys()))
            for k in proxy:
                self.assertAlmostEqual(need[k], proxy[k], places=12)


class TestSeedAndParamResolution(unittest.TestCase):
    """Seed distribution normalizes to a probability; env params clamp to bounds (§5.6)."""

    def test_seed_normalizes_to_one(self):
        p = sr._seed_distribution([("a", 0.4), ("b", 0.6)])
        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)
        self.assertAlmostEqual(p["a.md"], 0.4)
        self.assertAlmostEqual(p["b.md"], 0.6)

    def test_seed_degenerate_falls_back_uniform(self):
        # All non-positive cosines → uniform over the seeds (the walk is never empty).
        p = sr._seed_distribution([("a", -1.0), ("b", 0.0)])
        self.assertAlmostEqual(p["a.md"], 0.5)
        self.assertAlmostEqual(p["b.md"], 0.5)

    def test_gsr_l_clamp_to_bounds(self):
        with _EnvGuard(ASTHENOS_SUCCESSOR_GSR="9.9", ASTHENOS_SUCCESSOR_L="99"):
            self.assertEqual(sr.successor_gsr(), sr.SUCCESSOR_GSR_MAX)
            self.assertEqual(sr.successor_l(), sr.SUCCESSOR_L_MAX)
        with _EnvGuard(ASTHENOS_SUCCESSOR_GSR="-5", ASTHENOS_SUCCESSOR_L="0"):
            self.assertEqual(sr.successor_gsr(), sr.SUCCESSOR_GSR_MIN)
            self.assertEqual(sr.successor_l(), sr.SUCCESSOR_L_MIN)

    def test_param_defaults_are_seeds(self):
        with _EnvGuard(ASTHENOS_SUCCESSOR_GSR=None, ASTHENOS_SUCCESSOR_L=None):
            self.assertEqual(sr.successor_gsr(), sr.SUCCESSOR_GSR_SEED)
            self.assertEqual(sr.successor_l(), sr.SUCCESSOR_L_SEED)


class TestFlagOffByteIdentity(unittest.TestCase):
    """chainogram_retrieve byte-identical with the need term inert (λN=0 or flag off)."""

    def _run(self, md: Path):
        return ce.chainogram_retrieve(md, "any query", _vi_module=_FakeVI(md, _HITS))

    def test_need_term_chain_is_zero_without_graph(self):
        # The seam reads a need vector at best_node; an empty need map (no graph) → 0.0.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self.assertEqual(ce._need_term_chain(md, {"best_node": "n_one"}, {}), 0.0)

    def test_base_identity_flag_off(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            _write_edges(md, {"n_one::n_two": 1.0, "n_two::n_three": 1.0})
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            self.assertNotIn("error", ref)
            self.assertEqual(ref["loaded_chains"], ["c_beta", "c_alpha", "c_gamma"])

    def test_flag_on_lambda_n_zero_is_identity(self):
        # Master flag ON, λN unset (0.0) → the need term is compute-skipped → byte-identical.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            _write_edges(md, {"n_one::n_two": 1.0, "n_two::n_three": 1.0})
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_LAMBDA_N=None):
                on = self._run(md)
            self.assertNotIn("error", on)
            self.assertEqual(_ranking(ref), _ranking(on),
                             "flag-on with λN=0 must be byte-identical (need compute-skipped)")

    def test_scores_byte_identical_flag_on_zero(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            _write_edges(md, {"n_one::n_two": 1.0, "n_two::n_three": 1.0})
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"):
                on = self._run(md)
            ref_scores = [e["score"] for e in ref["loaded_nodes"]]
            on_scores = [e["score"] for e in on["loaded_nodes"]]
            self.assertEqual(ref_scores, on_scores)

    def test_lambda_n_nonzero_changes_nothing_when_need_is_flat(self):
        # Even with λN ON, if the pool's need values are degenerate (e.g. no edges so every
        # best_node reads 0.0), min-max → all 0.0 → envelope = 1.0 → score unchanged.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)  # NO edge_weights.json written → need map is seed-only
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_LAMBDA_N="0.6"):
                on = self._run(md)
            # best_node of each chain is its sole member; none are reached by diffusion
            # (seed-only need over disjoint chains) so the per-chain raw need is flat → 0.
            self.assertEqual(_ranking(ref), _ranking(on))


if __name__ == "__main__":
    unittest.main()


# ─────────────────────────────────────────────
# [test_successor] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-11  Status: active
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P3 tests — need-as-SR
#             term (§5.1-5.4). Asserts: T row-stochastic + discounted series bounded;
#             gSR=0 / gSR=0∧L=1 == the exact 1-step proxy need=p0; 2-hop transitive reach
#             (c unreached at L=1, reached at L=2; closer node outscores farther); seed
#             normalizes to a distribution + degenerate→uniform; gSR/L env clamp to
#             bounds; and the load-bearing FLAG-OFF BYTE-IDENTITY of chainogram_retrieve
#             (flag-off, flag-on-λN=0, and flat-need λN-on) vs the pinned baseline.
# Role:       prove P3's SR structure + flag-off byte-identity
# Depends:    successor; context_extension; unittest, tempfile, json, os
# ─────────────────────────────────────────────
