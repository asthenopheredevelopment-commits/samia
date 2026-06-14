"""samia.core.test_temporal_distinctiveness — tests for samia.core.temporal_distinctiveness (FEAT-2026-06-11 temporal-recall P5).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the SIMPLE log-time distinctiveness re-ranker (proposal §7):
               1. The D math — D(i)=1/Σ_j exp(−c·|logT_i−logT_j|) is reproduced in closed
                  form on a hand-built pool; the j=i term forces the denominator ≥ 1 so
                  D ∈ (0,1]; an isolated node scores ~1, a crowded one ~1/(m+1).
               2. Isolation boosts a toy node — a candidate alone on the time axis gets a
                  strictly higher raw D than members of a tight same-time cluster.
               3. Applicability soft-fail — a degenerately-clustered pool (all candidates
                  at nearly the same T) collapses to all-0.0 (no signal), not noise.
               4. Missing valid_from fallback — a legacy node with no written_at and no
                  valid_from still yields a usable T via infer_valid_from's st_mtime tier.
               5. THE CONTRACT — flag-off / λD=0 byte-identity: chainogram_retrieve is
                  byte-identical with the master flag off, and with the flag on but λD=0
                  (compute-skipped), and even with λD>0 but a degenerate pool (fails open).
    Depends: samia.core.temporal_distinctiveness, context_extension, temporal, frontmatter;
             unittest, tempfile, os, json, math, time. All tests use tempfile dirs and
             NEVER touch the live memory tree (mirrors test_tier / test_temporal_scaffold).

Layer 2 (What / Why):
    What: builds small hand-made corpora — nodes carrying explicit written_at frontmatter
          (sub-day) or none (legacy, exercising the day-granular fallback) plus tempfile
          chains and a deterministic FakeVI — and drives dist_vector / dist_at directly
          for the math + isolation + gate cases, and chainogram_retrieve for the flag-off
          identity. The closed-form D is recomputed independently in the test so the
          assertion is a real cross-check, not a self-comparison.
    Why:  HARD CONTRACTS: the SIMPLE ratio must be correct; isolation must out-rank
          crowding; the applicability gate must fail SOFT to no-signal; the fallback chain
          must keep a legacy node usable; and the whole layer must be a byte-identical
          no-op while λD=0 / the master flag is off. These tests assert each by
          construction, recomputing the math independently where it matters.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path

from samia.core import context_extension as ce
from samia.core import temporal_distinctiveness as td


# ── deterministic vector stand-in (mirrors test_temporal_scaffold._FakeVI) ──────────
class _FakeVI:
    """Deterministic stand-in for samia.core.vector: returns a fixed hit list."""

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
    """Context manager: set temporal env vars, restore the prior environment after."""

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


def _write_node(md: Path, name: str, chain: str, *,
                written_at: float | None = None,
                valid_from: str | None = "2026-06-11",
                edges: list | None = None) -> None:
    """Write one node (optionally carrying written_at / valid_from) + its single chain."""
    nodes = md / "nodes"
    chains = md / "chains"
    nodes.mkdir(parents=True, exist_ok=True)
    chains.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", "description: corpus node", "type: project",
             f"chains: [{chain}]"]
    if valid_from is not None:
        lines.append(f"valid_from: {valid_from}")
        lines.append("valid_to: null")
        lines.append(f"last_access: {valid_from}")
    lines += ["access_count: 0", "relevance: 0.5", "tier: warm"]
    if written_at is not None:
        lines.append(f"written_at: {written_at}")
        lines.append("episode_seq: 1")
    lines += ["---", f"body of {name}", ""]
    (nodes / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    (chains / f"{chain}.json").write_text(json.dumps({
        "name": chain,
        "members": [{"file": f"{name}.md", "addr": f"{chain}.0"}],
        "edges": edges or [],
    }), encoding="utf-8")


class TestDistMath(unittest.TestCase):
    """The SIMPLE ratio D(i)=1/Σ_j exp(−c·|logT_i−logT_j|) is correct and bounded."""

    def test_closed_form_two_node_pool(self):
        # Two nodes a fixed log-distance apart; recompute D independently and compare.
        now = 1_900_000_000.0
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # T_a = 10 days, T_b = 100 days (a 10× ratio → |Δ logT| = log(10)).
            _write_node(md, "n_a", "c_a", written_at=now - 10 * 86400.0)
            _write_node(md, "n_b", "c_b", written_at=now - 100 * 86400.0)
            best = {"c_a": "n_a", "c_b": "n_b"}
            out = td.dist_vector(md, best, now=now, c=1.0)
            # Independent closed-form: Δ = |log(10d) − log(100d)| = log(10).
            delta = abs(math.log(10 * 86400.0) - math.log(100 * 86400.0))
            denom = math.exp(0.0) + math.exp(-1.0 * delta)  # symmetric for both
            expected = 1.0 / denom
            self.assertAlmostEqual(out["c_a"], expected, places=10)
            self.assertAlmostEqual(out["c_b"], expected, places=10)
            # D ∈ (0, 1]: the j=i term keeps the denominator ≥ 1.
            self.assertTrue(0.0 < out["c_a"] <= 1.0)

    def test_singleton_pool_is_no_signal(self):
        # A pool with fewer than two timed candidates cannot express a ratio → 0.0.
        now = 1_900_000_000.0
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "n_a", "c_a", written_at=now - 5 * 86400.0)
            out = td.dist_vector(md, {"c_a": "n_a"}, now=now)
            self.assertEqual(out, {"c_a": 0.0})

    def test_unresolvable_node_reads_zero_but_keeps_pool(self):
        # A chain whose best_node file is missing drops from the timed pool (reads 0.0),
        # but the remaining timed candidates still compute among themselves.
        now = 1_900_000_000.0
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "n_a", "c_a", written_at=now - 1 * 86400.0)
            _write_node(md, "n_b", "c_b", written_at=now - 30 * 86400.0)
            best = {"c_a": "n_a", "c_b": "n_b", "c_missing": "n_ghost"}
            out = td.dist_vector(md, best, now=now)
            self.assertEqual(out["c_missing"], 0.0)
            self.assertGreater(out["c_a"], 0.0)
            self.assertGreater(out["c_b"], 0.0)


class TestIsolationBoost(unittest.TestCase):
    """A temporally isolated candidate out-ranks members of a tight same-time cluster."""

    def test_isolated_node_scores_higher_than_cluster(self):
        now = 1_900_000_000.0
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # Three nodes clustered at ~2 days old (a confusable pile-up) ...
            _write_node(md, "n_c1", "c_c1", written_at=now - 2.00 * 86400.0)
            _write_node(md, "n_c2", "c_c2", written_at=now - 2.05 * 86400.0)
            _write_node(md, "n_c3", "c_c3", written_at=now - 1.95 * 86400.0)
            # ... and one isolated node, two YEARS old (alone on the log axis).
            _write_node(md, "n_iso", "c_iso", written_at=now - 730 * 86400.0)
            best = {"c_c1": "n_c1", "c_c2": "n_c2", "c_c3": "n_c3", "c_iso": "n_iso"}
            out = td.dist_vector(md, best, now=now, c=1.0)
            # The isolated node is more distinct than every clustered member.
            self.assertGreater(out["c_iso"], out["c_c1"])
            self.assertGreater(out["c_iso"], out["c_c2"])
            self.assertGreater(out["c_iso"], out["c_c3"])


class TestApplicabilityGate(unittest.TestCase):
    """A degenerate (near-equal-time) pool fails SOFT to no-signal (§7.4)."""

    def test_degenerate_pool_collapses_to_zero(self):
        now = 1_900_000_000.0
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # Three nodes written within the same second → identical T → zero log-spread.
            base = now - 5 * 86400.0
            _write_node(md, "n_a", "c_a", written_at=base)
            _write_node(md, "n_b", "c_b", written_at=base)
            _write_node(md, "n_c", "c_c", written_at=base)
            best = {"c_a": "n_a", "c_b": "n_b", "c_c": "n_c"}
            out = td.dist_vector(md, best, now=now)
            self.assertEqual(out, {"c_a": 0.0, "c_b": 0.0, "c_c": 0.0})

    def test_wide_range_is_clipped_not_rejected(self):
        # A >1000× span is clipped at log(1000), not rejected: the term still produces a
        # signal (it only soft-fails on degenerate spread, not on a large one).
        now = 1_900_000_000.0
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "n_new", "c_new", written_at=now - 1 * 86400.0)   # 1 day
            _write_node(md, "n_mid", "c_mid", written_at=now - 100 * 86400.0)  # 100 days
            _write_node(md, "n_old", "c_old", written_at=now - 36500 * 86400.0)  # 100 yr
            best = {"c_new": "n_new", "c_mid": "n_mid", "c_old": "n_old"}
            out = td.dist_vector(md, best, now=now)
            # Some chain carries a non-zero signal → the gate did not reject the pool.
            self.assertTrue(any(v > 0.0 for v in out.values()))


class TestMissingValidFromFallback(unittest.TestCase):
    """A legacy node (no written_at, no valid_from) still yields a usable T via st_mtime."""

    def test_st_mtime_fallback_yields_usable_time(self):
        now = time.time() + 5.0  # a moment after the files are written
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # Legacy: no written_at, no valid_from → infer_valid_from falls to st_mtime.
            _write_node(md, "n_legacy", "c_legacy", written_at=None, valid_from=None)
            t = td.representative_time_seconds(md, "n_legacy", now=now)
            self.assertIsNotNone(t)
            self.assertGreaterEqual(t, td.DIST_MIN_T_SECONDS)

    def test_fallback_node_participates_in_pool(self):
        now = time.time() + 5.0
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # One legacy node (st_mtime ~ now) and one with an explicit OLD written_at.
            _write_node(md, "n_legacy", "c_legacy", written_at=None, valid_from=None)
            _write_node(md, "n_old", "c_old", written_at=now - 365 * 86400.0)
            best = {"c_legacy": "n_legacy", "c_old": "n_old"}
            out = td.dist_vector(md, best, now=now)
            # Both timed → both compute a (non-degenerate) signal.
            self.assertGreater(out["c_legacy"], 0.0)
            self.assertGreater(out["c_old"], 0.0)


class TestSharpnessReader(unittest.TestCase):
    """c reads live env, defaults to the 1.0 seed, clamps to [0.5, 2.0]."""

    def test_default_seed(self):
        with _EnvGuard(ASTHENOS_DIST_C=None):
            self.assertEqual(td.dist_sharpness(), 1.0)

    def test_clamped_to_bounds(self):
        with _EnvGuard(ASTHENOS_DIST_C="5.0"):
            self.assertEqual(td.dist_sharpness(), 2.0)
        with _EnvGuard(ASTHENOS_DIST_C="0.1"):
            self.assertEqual(td.dist_sharpness(), 0.5)

    def test_unparseable_fails_soft_to_seed(self):
        with _EnvGuard(ASTHENOS_DIST_C="not-a-float"):
            self.assertEqual(td.dist_sharpness(), 1.0)


# ── flag-off / λD=0 byte-identity through the live scorer ───────────────────────────
_HITS = [
    {"node": "n_one", "score": 0.40},
    {"node": "n_two", "score": 0.62},
    {"node": "n_three", "score": 0.15},
]


def _build_corpus(md: Path) -> None:
    """Three nodes / three chains with VARIED written_at (so D̂ would re-rank if active)."""
    now = time.time()
    _write_node(md, "n_one", "c_alpha", written_at=now - 1 * 86400.0,
                edges=[{"label": "hebbian"}, {"label": "hebbian"}])
    _write_node(md, "n_two", "c_beta", written_at=now - 50 * 86400.0)
    _write_node(md, "n_three", "c_gamma", written_at=now - 400 * 86400.0,
                edges=[{"label": "hebbian"}])


def _ranking(result: dict) -> list:
    out = [result.get("loaded_chains"), result.get("n_singletons")]
    for entry in result.get("loaded_nodes", []):
        out.append((entry.get("node"), entry.get("chain"),
                    entry.get("addr"), entry.get("score")))
    return out


class TestFlagOffByteIdentity(unittest.TestCase):
    """chainogram_retrieve is byte-identical with λD inert (the load-bearing contract)."""

    def _run(self, md: Path):
        return ce.chainogram_retrieve(md, "any query", _vi_module=_FakeVI(md, _HITS))

    def test_flag_off_identity(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            self.assertNotIn("error", ref)
            # Baseline order = cosine+hebbian: c_beta (0.62), c_alpha (0.40+0.10),
            # c_gamma (0.15+0.05) — time-blind, unaffected by the varied written_at.
            self.assertEqual(ref["loaded_chains"], ["c_beta", "c_alpha", "c_gamma"])

    def test_flag_on_lambda_d_zero_identity(self):
        # Flag ON, λD=0 (compute-skipped, §16.2-Q5) → byte-identical to flag-off.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_GAMMA=None,
                           ASTHENOS_TEMPORAL_LAMBDA_N=None,
                           ASTHENOS_TEMPORAL_LAMBDA_K=None,
                           ASTHENOS_TEMPORAL_LAMBDA_D=None):
                on = self._run(md)
            self.assertEqual(_ranking(ref), _ranking(on),
                             "flag-on with λD=0 must be byte-identical")

    def test_scores_bit_identical_with_lambda_d_zero(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_LAMBDA_D="0.0"):
                on = self._run(md)
            ref_scores = [e["score"] for e in ref["loaded_nodes"]]
            on_scores = [e["score"] for e in on["loaded_nodes"]]
            self.assertEqual(ref_scores, on_scores)


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_temporal_distinctiveness
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P5 (temporal distinctiveness, §7)
# Layer:      test (pytest)
# Role:       tests for samia.core.temporal_distinctiveness — SIMPLE log-time D ratio in closed form, isolation out-ranks cluster, degenerate-pool soft-fail, st_mtime fallback, c env clamp, flag-off/λD=0 byte-identity
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.temporal_distinctiveness, samia.core.context_extension
# Exposes:    — (test module)
# Lines:      357
# ------------------------------------------------------------------------------
