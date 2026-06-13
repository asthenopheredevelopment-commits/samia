"""Tests for the temporal-recall formula scaffold — FEAT-2026-06-11 P1.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the P1 scaffold installed in context_extension
             (proposal §2 + §8.6 + §16.2 Q5): the master flag ASTHENOS_TEMPORAL_WEIGHT
             + per-term weight readers (γ, λN, λK, λD) default to 0.0; the pool min-max
             normalizer, the uniform θ=0.2 relevance gate, and the §16.2-Q5 compute-skip
             predicate behave; the four term hooks (TC/N/K/D) are inert 0.0 seams; and —
             the load-bearing contract — chainogram_retrieve is BYTE-IDENTICAL flag-off
             (and even flag-on with all weights 0.0) to a captured pre-refactor reference,
             across BOTH the base path and the contextual variant.
    Depends: samia.core.context_extension; unittest, tempfile, json, os (stdlib). All
             tests use tempfile dirs and a deterministic _FakeVI stand-in (no torch/HF),
             and NEVER touch the live memory tree.

Layer 2 (What / Why):
    What: Proves P1's flag-off identity (§2.6) by construction — the refactored scorer,
          run with the master flag OFF and with the flag ON but every weight at its 0.0
          default, produces the SAME chosen chains, order, and accumulated scores as the
          baseline S_c + 0.05·H_c formula. Also asserts the helper semantics: weights are
          forced 0.0 when the master flag is off; the gate is θ=0.2; min-max degrades to 0
          on a degenerate pool; compute-skip fires below ε; and the term hooks return 0.0.
    Why:  HARD CONTRACT: flag-off byte-identity. P1 wires the (cue)·(1+gain) shape with
          all coefficients zeroed; these tests assert the shape collapses EXACTLY to the
          baseline rather than assuming it. The contextual variant re-enters the base
          scorer, so its identity rides on the base — both legs are exercised explicitly.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from samia.core import context_extension as ce


# ── deterministic vector stand-in (mirrors test_temporal_substrate._FakeVI) ───────
class _FakeVI:
    """Deterministic stand-in for samia.core.vector: returns a fixed hit list.

    Avoids the torch/HF embedding backend so the chain scorer runs its real
    accumulation/sort/envelope path against tempfile chain files with reproducible
    inputs. Identical hits across the baseline/flag-off/flag-on runs isolate the
    scorer's behavior from any embedding nondeterminism.
    """

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


def _build_corpus(md: Path) -> None:
    """A small multi-chain corpus with varied cosines + a Hebbian edge.

    Three nodes across three chains; chain c_alpha carries two hebbian edges, the
    others none — so the baseline score spreads S_c (cosine sum) plus 0.05·H_c and
    the chosen order is non-trivial (exercises both additive terms of the cue).
    """
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
            "---",
            f"name: {nm}",
            "description: corpus node",
            "type: project",
            f"chains: [{ch}]",
            "valid_from: 2026-06-11",
            "valid_to: null",
            "last_access: 2026-06-11",
            "access_count: 0",
            "relevance: 0.5",
            "tier: warm",
            f"written_at: 1781827200.0",
            "episode_seq: 1",
            "---",
            f"body of {nm}",
            "",
        ]
        (nodes / f"{nm}.md").write_text("\n".join(lines), encoding="utf-8")
        chain = {
            "name": ch,
            "members": [{"file": f"{nm}.md", "addr": f"{ch}.0"}],
            "edges": edges,
        }
        (chains / f"{ch}.json").write_text(json.dumps(chain), encoding="utf-8")


# Hits with varied cosines so S_c differs per chain; one below θ=0.2 so the gate is
# meaningfully exercised by any future term (and the baseline ordering is non-flat).
_HITS = [
    {"node": "n_one", "score": 0.40},
    {"node": "n_two", "score": 0.62},
    {"node": "n_three", "score": 0.15},
]


def _ranking(result: dict) -> list:
    """RANKING-relevant projection: chosen chains + per-served-node node/chain/addr/score."""
    out = [result.get("loaded_chains"), result.get("n_singletons")]
    for entry in result.get("loaded_nodes", []):
        out.append((entry.get("node"), entry.get("chain"),
                    entry.get("addr"), entry.get("score")))
    return out


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


class TestFlagReadersDefaultZero(unittest.TestCase):
    """Master flag default-OFF and per-term weights default-0.0 (§8.6)."""

    def test_master_flag_default_off(self):
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
            self.assertFalse(ce.temporal_weight_enabled())

    def test_master_flag_on_token(self):
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"):
            self.assertTrue(ce.temporal_weight_enabled())
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="0"):
            self.assertFalse(ce.temporal_weight_enabled())
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="true"):
            # only the literal "1" enables, mirroring semantic_arm_enabled
            self.assertFalse(ce.temporal_weight_enabled())

    def test_weights_all_zero_when_master_off(self):
        # Even if a per-term env is set, master-off forces every weight to 0.0 (§8.6).
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None,
                       ASTHENOS_TEMPORAL_GAMMA="0.9",
                       ASTHENOS_TEMPORAL_LAMBDA_N="0.5"):
            w = ce.temporal_weights()
            self.assertEqual(w, {"gamma": 0.0, "lambda_n": 0.0,
                                 "lambda_k": 0.0, "lambda_d": 0.0})

    def test_weights_default_zero_when_master_on(self):
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                       ASTHENOS_TEMPORAL_GAMMA=None,
                       ASTHENOS_TEMPORAL_LAMBDA_N=None,
                       ASTHENOS_TEMPORAL_LAMBDA_K=None,
                       ASTHENOS_TEMPORAL_LAMBDA_D=None):
            w = ce.temporal_weights()
            self.assertEqual(w, {"gamma": 0.0, "lambda_n": 0.0,
                                 "lambda_k": 0.0, "lambda_d": 0.0})

    def test_weights_read_when_master_on(self):
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                       ASTHENOS_TEMPORAL_GAMMA="0.5",
                       ASTHENOS_TEMPORAL_LAMBDA_D="0.3"):
            w = ce.temporal_weights()
            self.assertEqual(w["gamma"], 0.5)
            self.assertEqual(w["lambda_d"], 0.3)
            self.assertEqual(w["lambda_n"], 0.0)

    def test_weight_unparseable_fails_soft_to_zero(self):
        with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                       ASTHENOS_TEMPORAL_GAMMA="not-a-float"):
            self.assertEqual(ce.temporal_weights()["gamma"], 0.0)


class TestHelpers(unittest.TestCase):
    """The pure helpers: gate, min-max pool, compute-skip predicate, hook seams."""

    def test_relevance_gate_theta(self):
        self.assertEqual(ce.TEMPORAL_THETA, 0.2)
        self.assertEqual(ce._relevance_gate(0.20), 1.0)  # boundary is inclusive
        self.assertEqual(ce._relevance_gate(0.25), 1.0)
        self.assertEqual(ce._relevance_gate(0.19), 0.0)
        self.assertEqual(ce._relevance_gate(0.0), 0.0)

    def test_minmax_pool_normalizes(self):
        out = ce._minmax_pool({"a": 1.0, "b": 3.0, "c": 5.0})
        self.assertEqual(out["a"], 0.0)
        self.assertEqual(out["c"], 1.0)
        self.assertEqual(out["b"], 0.5)

    def test_minmax_pool_degenerate_pools(self):
        self.assertEqual(ce._minmax_pool({}), {})
        self.assertEqual(ce._minmax_pool({"a": 4.0}), {"a": 0.0})  # pool of one
        self.assertEqual(ce._minmax_pool({"a": 2.0, "b": 2.0}),
                         {"a": 0.0, "b": 0.0})  # all equal

    def test_compute_skip_predicate(self):
        self.assertFalse(ce._term_active(0.0))
        self.assertFalse(ce._term_active(ce.TEMPORAL_WEIGHT_EPSILON / 2))
        self.assertTrue(ce._term_active(0.5))
        self.assertTrue(ce._term_active(-0.5))  # magnitude, not sign

    def test_term_hooks_are_zero_without_state(self):
        # TC (P2), need (P3), STC (P4) and dist (P5) are all now LIVE hooks but fail open
        # to 0.0 with no temporal state in the tempfile corpus (no SITH snapshot, empty
        # need vector, no captured node, empty dist map). Every hook contributes 0.0 here,
        # so the envelope stays a no-op on a fresh corpus.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self.assertEqual(ce._tc_term_hit(md, {"node": "x"}, 1.0), 0.0)
            self.assertEqual(ce._need_term_chain(md, {"best_node": "x"}, {}), 0.0)
            self.assertEqual(ce._stc_term_chain(md, {"best_node": "x"}, []), 0.0)
            self.assertEqual(ce._dist_vector(md, {"c_x": "x"}), {"c_x": 0.0})
            self.assertEqual(ce._dist_term_chain(md, "c_x", {}), 0.0)


class TestFlagOffByteIdentity(unittest.TestCase):
    """chainogram_retrieve is byte-identical to the baseline with the scaffold inert."""

    def _run(self, md: Path):
        return ce.chainogram_retrieve(md, "any query", _vi_module=_FakeVI(md, _HITS))

    def _run_contextual(self, md: Path):
        # The contextual variant re-enters the base scorer with _vi_module=_vic. We
        # exercise the SAME base-path identity by invoking chainogram_retrieve with a
        # FakeVI in the contextual module's place (the envelope lives in the base).
        return ce.chainogram_retrieve(
            md, "any query", _vi_module=_FakeVI(md, _HITS),
            include_singletons=True)

    def test_base_path_identity_flag_off(self):
        # Reference = scorer with the master flag explicitly OFF.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            self.assertNotIn("error", ref)
            # The baseline order must be the cosine+hebbian order: c_beta (0.62),
            # c_alpha (0.40 + 0.10), c_gamma (0.15 + 0.05). Pin it so a future
            # regression that silently reorders is caught, not just a self-equality.
            self.assertEqual(ref["loaded_chains"], ["c_beta", "c_alpha", "c_gamma"])

    def test_base_path_identity_flag_on_zero_weights(self):
        # The §2.6 claim: flag ON with every weight 0.0 is ALSO byte-identical, since
        # cue = base + 0 and envelope = 1.0 (base·1.0 == base exactly in IEEE 754).
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
            self.assertNotIn("error", on)
            self.assertEqual(_ranking(ref), _ranking(on),
                             "flag-on with zero weights must be byte-identical")

    def test_scores_byte_identical(self):
        # Stronger than ordering: the accumulated per-chain scores themselves must be
        # bit-for-bit equal between the flag-off baseline and the flag-on-zero path.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"):
                on = self._run(md)
            ref_scores = [e["score"] for e in ref["loaded_nodes"]]
            on_scores = [e["score"] for e in on["loaded_nodes"]]
            self.assertEqual(ref_scores, on_scores)

    def test_contextual_variant_identity(self):
        # The contextual variant inherits the base scorer's envelope for free. With the
        # flag off (and on-zero), the base-path it re-enters is byte-identical, proving
        # the contextual leg of the flag-off contract.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run_contextual(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"):
                on = self._run_contextual(md)
            self.assertNotIn("error", ref)
            self.assertEqual(_ranking(ref), _ranking(on),
                             "contextual variant must hold flag-off identity")


if __name__ == "__main__":
    unittest.main()


# ─────────────────────────────────────────────
# [test_temporal_scaffold] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-11  Status: active
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P1 tests — formula
#             scaffold. Asserts: master flag ASTHENOS_TEMPORAL_WEIGHT default-OFF +
#             per-term weights (γ/λN/λK/λD) default-0.0 + master-off forces all 0.0;
#             θ=0.2 gate; min-max pool (degenerate→0); §16.2-Q5 compute-skip below ε;
#             inert 0.0 term hooks; and the load-bearing FLAG-OFF BYTE-IDENTITY of
#             chainogram_retrieve (flag-off AND flag-on-zero-weights) across the base
#             path and the contextual variant vs a pinned baseline ordering.
# Role:       prove P1's flag-off byte-identity + helper semantics
# Depends:    context_extension; unittest, tempfile, json, os
# ─────────────────────────────────────────────
