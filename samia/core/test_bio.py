"""Tests for samia.core.bio — FEAT-2026-06-05 Tier-0 D1/D2 homeostatic Hebbian.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the reachable attractor bar (derived alpha), the one-time
             count->w re-seed, and the homeostatic replay regulators (fractional weight,
             decay-transparency, genuine-count promotion gate). PLUS the P5 engram
             feed-forward with GENUINE-ONCE (replay_engram_traces: first replay of a
             pair genuine, re-replays fractional; one trace replayed many times cannot
             cross the attractor bar on replay alone).
    Depends: samia.core.bio, unittest, datetime, json, tempfile, pathlib (stdlib).

Layer 2 (What / Why):
    What: Verifies (1) K genuine co-activations reach the bar and 1-fewer stays below;
          (2) replay is fractional vs genuine; (3) a replay-ONLY edge is capped below the
          bar, never refreshes last_seen, and is NOT promotable no matter how often it
          fires; (4) genuine+replay is promotable; (5) re-seed caps below the bar, seeds
          count_genuine, and is idempotent. PLUS the P2 salience SOURCE (compute_salience
          + salience_merge_guard) and the P3 kWTA sparse-code pattern separation
          (kwta_sparse_code: deterministic, 2-5% sparse, orthogonalizes near-duplicates).
    Why:  These are the runaway/pruning safeguards the operator flagged as load-bearing —
          a regression would let the Hebbian web saturate (reverberation) or stay frozen
          (no consolidation). Pure-helper tests avoid touching the global edges.db.
"""
from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from samia.core import bio


TODAY = dt.date(2026, 6, 5)


class TestReachableBar(unittest.TestCase):
    def test_k_genuine_reaches_bar(self):
        w: dict = {}
        for _ in range(bio.HEBB_PROMOTE_REPEATS):
            bio._apply_coactivation(w, ["a", "b"], "genuine", TODAY)
        v = w["a::b"]
        self.assertGreaterEqual(v["w"], bio.HEBB_PROMOTION)
        self.assertEqual(v["count_genuine"], bio.HEBB_PROMOTE_REPEATS)
        self.assertTrue(bio._is_promotable(v))

    def test_one_fewer_genuine_below_bar(self):
        w: dict = {}
        for _ in range(bio.HEBB_PROMOTE_REPEATS - 1):
            bio._apply_coactivation(w, ["a", "b"], "genuine", TODAY)
        self.assertLess(w["a::b"]["w"], bio.HEBB_PROMOTION)
        self.assertFalse(bio._is_promotable(w["a::b"]))


class TestReplayHomeostasis(unittest.TestCase):
    def test_replay_is_fractional_vs_genuine(self):
        wg: dict = {}
        wr: dict = {}
        bio._apply_coactivation(wg, ["a", "b"], "genuine", TODAY)
        bio._apply_coactivation(wr, ["a", "b"], "replay", TODAY)
        self.assertLess(wr["a::b"]["w"], wg["a::b"]["w"])

    def test_replay_only_capped_below_bar_and_unpromotable(self):
        """A full day of replay (no genuine recall) must NOT promote — the runaway guard."""
        w: dict = {}
        for _ in range(200):
            bio._apply_coactivation(w, ["a", "b"], "replay", TODAY)
        v = w["a::b"]
        self.assertLessEqual(v["w"], bio.REPLAY_ONLY_W_CEILING + 1e-9)
        self.assertLess(v["w"], bio.HEBB_PROMOTION)
        self.assertEqual(v["count_genuine"], 0)
        self.assertFalse(bio._is_promotable(v))

    def test_replay_does_not_refresh_decay_clock(self):
        w: dict = {}
        bio._apply_coactivation(w, ["a", "b"], "genuine", dt.date(2026, 6, 1))
        w["a::b"]["last_seen"] = "2026-06-01"   # stale the clock
        bio._apply_coactivation(w, ["a", "b"], "replay", TODAY)  # later day
        self.assertEqual(w["a::b"]["last_seen"], "2026-06-01")   # replay left it alone

    def test_one_genuine_then_replay_NOT_promotable(self):
        """Tier-0 multi-day fix (2026-06-07): one genuine SEED + heavy replay (even
        many days' worth of daily fractional events) stays capped below the bar and is
        NOT promotable. Was the bug — the ceiling used to lift at count_genuine>=1, so
        replay could farm a once-recalled pair to ~0.98 over days. Now the ceiling holds
        until HEBB_PROMOTE_REPEATS genuine events."""
        w: dict = {}
        bio._apply_coactivation(w, ["a", "b"], "genuine", TODAY)   # count_genuine=1
        for _ in range(50):                                        # heavy replay
            bio._apply_coactivation(w, ["a", "b"], "replay", TODAY)
        v = w["a::b"]
        self.assertEqual(v["count_genuine"], 1)
        self.assertLessEqual(v["w"], bio.REPLAY_ONLY_W_CEILING + 1e-9)
        self.assertLess(v["w"], bio.HEBB_PROMOTION)
        self.assertFalse(bio._is_promotable(v))

    def test_multiday_replay_after_one_genuine_cannot_promote(self):
        """(a) The exact gap the Tier-1 P5 verifier found: ONE genuine event then MANY
        days of daily fractional replay (one replay per day, decay applied between days)
        must leave w capped < HEBB_PROMOTION and _is_promotable False."""
        w: dict = {}
        day0 = dt.date(2026, 6, 1)
        bio._apply_coactivation(w, ["a", "b"], "genuine", day0)   # the single seed
        # 60 days of one fractional replay per day, with the daily decay/prune in between
        for d in range(1, 61):
            day = day0 + dt.timedelta(days=d)
            bio._decay_and_prune(w, day)        # age once per day
            if "a::b" not in w:                 # pruned away -> certainly not promotable
                break
            bio._apply_coactivation(w, ["a", "b"], "replay", day)
        if "a::b" in w:
            v = w["a::b"]
            self.assertEqual(v["count_genuine"], 1)
            self.assertLess(v["w"], bio.HEBB_PROMOTION)
            self.assertLessEqual(v["w"], bio.REPLAY_ONLY_W_CEILING + 1e-9)
            self.assertFalse(bio._is_promotable(v))

    def test_two_genuine_not_promotable_boundary(self):
        """(c) Boundary: 1 and 2 genuine events are NOT promotable; only at
        HEBB_PROMOTE_REPEATS does the gate (and the ceiling) open."""
        for k in (1, 2):
            w: dict = {}
            for _ in range(k):
                bio._apply_coactivation(w, ["a", "b"], "genuine", TODAY)
            # even saturate with replay — still ceilinged and gated below K=3
            for _ in range(50):
                bio._apply_coactivation(w, ["a", "b"], "replay", TODAY)
            v = w["a::b"]
            self.assertEqual(v["count_genuine"], k)
            self.assertLess(v["w"], bio.HEBB_PROMOTION,
                            f"{k} genuine should stay below the bar")
            self.assertFalse(bio._is_promotable(v),
                             f"{k} genuine must not be promotable")

    def test_genuine_path_still_promotes(self):
        """(b) The natural genuine path MUST still promote: HEBB_PROMOTE_REPEATS genuine
        co-activations lift the ceiling and make the edge promotable. After the bar is
        met, replay may continue to reinforce (ceiling lifted)."""
        w: dict = {}
        for _ in range(bio.HEBB_PROMOTE_REPEATS):
            bio._apply_coactivation(w, ["a", "b"], "genuine", TODAY)
        v = w["a::b"]
        self.assertEqual(v["count_genuine"], bio.HEBB_PROMOTE_REPEATS)
        self.assertGreaterEqual(v["w"], bio.HEBB_PROMOTION)
        self.assertTrue(bio._is_promotable(v))
        # ceiling is now lifted: a later replay no longer clamps w back below the bar
        bio._apply_coactivation(w, ["a", "b"], "replay", TODAY)
        self.assertGreaterEqual(w["a::b"]["w"], bio.HEBB_PROMOTION)
        self.assertTrue(bio._is_promotable(w["a::b"]))

    def test_ceiling_below_bar_invariant(self):
        """The cap must actually block promotion: REPLAY_ONLY_W_CEILING < HEBB_PROMOTION."""
        self.assertLess(bio.REPLAY_ONLY_W_CEILING, bio.HEBB_PROMOTION)


class TestReseed(unittest.TestCase):
    def _mem(self, tmp: str) -> Path:
        md = Path(tmp)
        bd = md / "biomimetic"
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "edge_weights.json").write_text(json.dumps({
            "a::b": {"w": 0.30, "count": 10, "last_seen": "2026-06-03"},
            "c::d": {"w": 0.30, "count": 1, "last_seen": "2026-06-03"},
        }), encoding="utf-8")
        return md

    def test_reseed_from_genuine_history_promotable_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = self._mem(tmp)
            r1 = bio.reseed_edge_weights(md)
            self.assertEqual(r1["reseeded"], 2)
            w = json.loads((md / "biomimetic" / "edge_weights.json").read_text())
            # v3: high-genuine-count edge re-seeds to its EARNED strength and is promotable
            # on its genuine history (no sub-bar cap blocking it).
            self.assertGreaterEqual(w["a::b"]["w"], bio.HEBB_PROMOTION)
            self.assertEqual(w["a::b"]["count_genuine"], 10)
            self.assertTrue(bio._is_promotable(w["a::b"]))
            # a cg=1 edge stays naturally below the bar (no cap needed)
            self.assertLess(w["c::d"]["w"], bio.HEBB_PROMOTION)
            self.assertFalse(bio._is_promotable(w["c::d"]))
            # idempotent
            r2 = bio.reseed_edge_weights(md)
            self.assertEqual(r2.get("skipped"), "already-done")


class TestPerDayDecay(unittest.TestCase):
    def test_decay_once_per_day_not_per_pass(self):
        """The Phase-1-exposed bug: decay must apply once per day, not every pass."""
        w = {"a::b": {"w": 0.83, "count": 5, "count_genuine": 5,
                      "last_seen": "2026-06-03"}}
        today = dt.date(2026, 6, 6)   # 3 days stale
        bio._decay_and_prune(w, today)
        after_first = w["a::b"]["w"]
        self.assertAlmostEqual(after_first, 0.83 * (1 - bio.HEBB_DECAY * 3), places=6)
        self.assertEqual(w["a::b"]["last_decay"], "2026-06-06")
        # 50 more passes the SAME day must NOT decay further (the fix)
        for _ in range(50):
            bio._decay_and_prune(w, today)
        self.assertAlmostEqual(w["a::b"]["w"], after_first, places=9)

    def test_decay_advances_next_day(self):
        w = {"a::b": {"w": 0.80, "count": 5, "count_genuine": 5,
                      "last_seen": "2026-06-06", "last_decay": "2026-06-06"}}
        bio._decay_and_prune(w, dt.date(2026, 6, 7))
        self.assertAlmostEqual(w["a::b"]["w"], 0.80 * (1 - bio.HEBB_DECAY), places=6)
        self.assertEqual(w["a::b"]["last_decay"], "2026-06-07")

    def test_prune_below_threshold(self):
        w = {"a::b": {"w": 0.04, "count": 1, "count_genuine": 0,
                      "last_seen": "2026-06-06"}}
        self.assertEqual(bio._decay_and_prune(w, dt.date(2026, 6, 6)), 1)
        self.assertNotIn("a::b", w)


class TestReseedV2(unittest.TestCase):
    def test_reseed_from_genuine_leaves_replay_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            bd = md / "biomimetic"
            bd.mkdir(parents=True)
            (bd / "edge_weights.json").write_text(json.dumps({
                "g::h": {"w": 0.10, "count": 9, "count_genuine": 9,
                         "count_replay": 0, "last_seen": "2026-06-03"},
                "r::s": {"w": 0.30, "count": 4, "count_genuine": 0,
                         "count_replay": 4, "last_seen": "2026-06-06"},
            }), encoding="utf-8")
            bio.reseed_edge_weights(md)
            w = json.loads((bd / "edge_weights.json").read_text())
            # v3: genuine edge restored from its (intact) count_genuine to its EARNED,
            # promotable strength -- up from the eroded 0.10, no sub-bar cap.
            self.assertGreaterEqual(w["g::h"]["w"], bio.HEBB_PROMOTION)
            self.assertTrue(bio._is_promotable(w["g::h"]))
            # replay-only edge (count_genuine==0) left to its normal lifecycle, untouched
            self.assertAlmostEqual(w["r::s"]["w"], 0.30, places=6)


# ════════════════════════════════════════════════════════════════════════════
# FEAT-2026-06-07 Tier-1 P2 (D6) — the salience SOURCE: compute_salience +
# salience_merge_guard. SOURCE/storage/explicit-tag only; EFFECTS are P3/P5.
# ════════════════════════════════════════════════════════════════════════════

import tempfile as _tf  # noqa: E402  (test-local alias, mirrors module style)
from unittest import mock as _mock  # noqa: E402

import numpy as _np  # noqa: E402

from samia.core import vector as _vi  # noqa: E402


def _stub_embed_unit(texts):
    """Deterministic stub embedder over a tiny vocab (controllable cosine)."""
    vocab = ["alpha", "beta", "gamma", "delta", "novel", "zeta"]
    out = []
    for t in texts:
        toks = t.lower().split()
        v = _np.zeros(len(vocab), dtype=_np.float32)
        for i, w in enumerate(vocab):
            v[i] = float(sum(1 for tok in toks if w in tok))
        n = _np.linalg.norm(v)
        v = v / n if n > 0 else _np.ones(len(vocab), dtype=_np.float32) / (
            len(vocab) ** 0.5)
        out.append(v)
    return _np.vstack(out).astype(_np.float32)


def _write_sal_node(md, name, title, body, access_count=0):
    nodes = md / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    fname = name if name.endswith(".md") else f"{name}.md"
    fm = (f"---\nname: {title}\ndescription: {title}\ntype: project\n"
          f"chains: []\nlast_access: 2026-06-07\naccess_count: {access_count}\n"
          f"relevance: 0.5\ntier: warm\n---\n")
    (nodes / fname).write_text(fm + body + "\n", encoding="utf-8")
    return fname


def _build_main_index(md, vecs):
    """Write a tiny main vector index (embeddings.npy) for the surprise signal."""
    idx = md / "vector_index"
    idx.mkdir(parents=True, exist_ok=True)
    _np.save(idx / "embeddings.npy", _np.vstack(vecs).astype(_np.float32))


class TestComputeSalience(unittest.TestCase):
    def test_normalized_and_each_signal_contributes(self):
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch.object(_vi, "_embed_batch", _stub_embed_unit):
            md = Path(tmp)
            # A novel node (high surprise) with some repetition; index holds an
            # unrelated vector so max_cosine is low -> surprise is high.
            _build_main_index(md, [_stub_embed_unit(["beta beta"])[0]])
            _write_sal_node(md, "n1", "Novel", "novel novel novel", access_count=10)
            sal = bio.compute_salience(md, "n1", write=False)
            self.assertGreaterEqual(sal, 0.0)
            self.assertLessEqual(sal, 1.0)
            # Surprise (high, orthogonal to the index) + repetition (saturated) push
            # salience above the surprise-weight floor alone.
            self.assertGreater(sal, bio.SALIENCE_W_SURPRISE * 0.5)

    def test_missing_signal_contributes_zero_no_crash(self):
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch.object(_vi, "_embed_batch", _stub_embed_unit):
            md = Path(tmp)
            # NO vector index (surprise missing), NO supersession store (contradiction
            # missing), NO access/edges (repetition missing) -> all signals 0, no crash.
            _write_sal_node(md, "n1", "Bare", "alpha", access_count=0)
            sal = bio.compute_salience(md, "n1", write=False)
            self.assertEqual(sal, 0.0)

    def test_explicit_tag_overrides_composite(self):
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch.object(_vi, "_embed_batch", _stub_embed_unit):
            md = Path(tmp)
            # A near-duplicate-of-index node => LOW surprise => LOW composite.
            idx_vec = _stub_embed_unit(["alpha alpha"])[0]
            _build_main_index(md, [idx_vec])
            _write_sal_node(md, "n1", "Dup", "alpha alpha", access_count=0)
            composite = bio.compute_salience(md, "n1", write=False)
            self.assertLess(composite, 0.5)
            # The explicit tag clamps it HIGH regardless of the low composite.
            tagged = bio.compute_salience(md, "n1", explicit_tag=True, write=False)
            self.assertGreaterEqual(tagged, bio.SALIENCE_TAG_VALUE)
            self.assertGreater(tagged, composite)

    def test_salience_written_to_frontmatter(self):
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch.object(_vi, "_embed_batch", _stub_embed_unit):
            md = Path(tmp)
            _build_main_index(md, [_stub_embed_unit(["beta"])[0]])
            _write_sal_node(md, "n1", "Novel", "novel novel")
            bio.compute_salience(md, "n1", explicit_tag=True, write=True)
            from samia.core import frontmatter as fm
            parsed_fm, _order, _body = fm.read_node(md / "nodes" / "n1.md")
            self.assertIn("salience", parsed_fm)
            self.assertGreaterEqual(float(parsed_fm["salience"]),
                                    bio.SALIENCE_TAG_VALUE)
            self.assertTrue(parsed_fm.get("salience_tag"))

    def test_contradiction_involvement_bumps_salience(self):
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch.object(_vi, "_embed_batch", _stub_embed_unit):
            md = Path(tmp)
            _build_main_index(md, [_stub_embed_unit(["alpha alpha"])[0]])
            _write_sal_node(md, "n1", "Dup", "alpha alpha", access_count=0)
            base = bio.compute_salience(md, "n1", write=False)
            # Record a supersession candidate naming n1 -> contradiction signal fires.
            from samia.runtime import contradiction as con
            con.record_supersession_candidate(md, "old_node", "n1",
                                               cosine=0.9, mode="passive")
            bumped = bio.compute_salience(md, "n1", write=False)
            self.assertGreater(bumped, base)

    def test_effects_not_applied_decay_unchanged(self):
        """P2 writes the salience SOURCE only — it must NOT touch relevance/tier (decay
        effect is P5). Writing salience leaves the decay-driving fields intact."""
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch.object(_vi, "_embed_batch", _stub_embed_unit):
            md = Path(tmp)
            _build_main_index(md, [_stub_embed_unit(["beta"])[0]])
            _write_sal_node(md, "n1", "Novel", "novel novel")
            from samia.core import frontmatter as fm
            before, _o, _b = fm.read_node(md / "nodes" / "n1.md")
            rel_before, tier_before = before.get("relevance"), before.get("tier")
            bio.compute_salience(md, "n1", explicit_tag=True, write=True)
            after, _o2, _b2 = fm.read_node(md / "nodes" / "n1.md")
            # Salience added; the decay fields are untouched (no EFFECT applied).
            self.assertIn("salience", after)
            self.assertEqual(after.get("relevance"), rel_before)
            self.assertEqual(after.get("tier"), tier_before)


class TestSalienceMergeGuard(unittest.TestCase):
    def test_guard_true_for_high_salience_distinct(self):
        with _tf.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_sal_node(md, "n1", "Important", "x")
            # Stamp a high salience directly.
            from samia.core import frontmatter as fm
            f, o, b = fm.read_node(md / "nodes" / "n1.md")
            f["salience"] = 0.95
            o.append("salience")
            fm.write_node(md / "nodes" / "n1.md", f, o, b)
            self.assertTrue(bio.salience_merge_guard(md, "n1"))

    def test_guard_false_for_true_duplicate(self):
        with _tf.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_sal_node(md, "n1", "Important", "x")
            from samia.core import frontmatter as fm
            f, o, b = fm.read_node(md / "nodes" / "n1.md")
            f["salience"] = 0.99
            o.append("salience")
            fm.write_node(md / "nodes" / "n1.md", f, o, b)
            # A TRUE duplicate is still deduped regardless of salience.
            self.assertFalse(bio.salience_merge_guard(md, "n1", is_duplicate=True))

    def test_guard_false_for_low_salience(self):
        with _tf.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_sal_node(md, "n1", "Trivial", "x")  # no salience field -> 0.0
            self.assertFalse(bio.salience_merge_guard(md, "n1"))


# ════════════════════════════════════════════════════════════════════════════
# FEAT-2026-06-07 Tier-1 P3 (D2) — kWTA sparse-code pattern separation tests.
# Determinism (same embedding -> same key) + sparsity (2-5% active) + orthogonalization
# (near-duplicate embeddings get DISTINGUISHABLE keys; more-different inputs share fewer
# winners). The cosine dedup gate (pattern_separation_decision) stays a separate job.
# ════════════════════════════════════════════════════════════════════════════


class TestKwtaSparseCode(unittest.TestCase):
    def _unit(self, rng, dim=384):
        v = rng.standard_normal(dim).astype(_np.float32)
        return v / _np.linalg.norm(v)

    def test_deterministic_per_embedding(self):
        rng = _np.random.default_rng(7)
        emb = self._unit(rng)
        # Same input -> identical sparse key (fixed-seed projection).
        self.assertEqual(bio.kwta_sparse_code(emb), bio.kwta_sparse_code(emb))
        self.assertEqual(bio.kwta_sparse_code(emb.copy()),
                         bio.kwta_sparse_code(emb))

    def test_sparsity_in_band(self):
        rng = _np.random.default_rng(1)
        code = bio.kwta_sparse_code(self._unit(rng))
        frac = len(code) / bio.KWTA_PROJ_DIM
        # 2-5% active (D2); default frac 0.03.
        self.assertGreaterEqual(frac, 0.02)
        self.assertLessEqual(frac, 0.05)
        # winners are unique, sorted indices in range.
        self.assertEqual(sorted(set(code)), code)
        self.assertTrue(all(0 <= i < bio.KWTA_PROJ_DIM for i in code))

    def test_near_duplicates_get_distinguishable_keys(self):
        rng = _np.random.default_rng(3)
        a = self._unit(rng)
        noise = rng.standard_normal(a.shape[0]).astype(_np.float32)
        near = a + 0.02 * noise          # cosine ~0.97 — a near-duplicate session
        near = near / _np.linalg.norm(near)
        code_a = set(bio.kwta_sparse_code(a))
        code_near = set(bio.kwta_sparse_code(near))
        # Distinguishable: the two near-duplicate episodes do NOT collapse to the
        # SAME sparse key (the orthogonalization the dedup-cosine gate cannot give).
        self.assertNotEqual(code_a, code_near)

    def test_orthogonalization_separates_more_different_inputs(self):
        rng = _np.random.default_rng(5)
        a = self._unit(rng)
        noise = rng.standard_normal(a.shape[0]).astype(_np.float32)
        near = a + 0.02 * noise
        near = near / _np.linalg.norm(near)
        far = a + 0.5 * noise            # much more different (lower cosine)
        far = far / _np.linalg.norm(far)
        ca, cn, cf = (set(bio.kwta_sparse_code(x)) for x in (a, near, far))
        # The MORE different an episode is, the FEWER winners it shares — the
        # orthogonalizing property (separation scales with distance).
        self.assertLess(len(ca & cf), len(ca & cn))

    def test_empty_embedding_yields_empty_code(self):
        self.assertEqual(bio.kwta_sparse_code(_np.zeros(384, dtype=_np.float32)), [])
        self.assertEqual(bio.kwta_sparse_code(_np.array([], dtype=_np.float32)), [])


# ════════════════════════════════════════════════════════════════════════════
# FEAT-2026-06-07 Tier-1 P5 — engram replay with GENUINE-ONCE feed-forward (D5/Q6a).
#
# Replay the CAPTURED engram held copies into Tier-0 co-activations: a pair's FIRST
# replay is genuine (+count_genuine), every re-replay is fractional then ages. The
# load-bearing invariant: a single captured trace replayed MANY times must NOT push an
# edge over the attractor bar on replay alone (one genuine per pair < HEBB_PROMOTE_REPEATS).
# ════════════════════════════════════════════════════════════════════════════

_REPLAY_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _replay_stub_embed(texts):
    out = []
    for t in texts:
        toks = t.lower().split()
        v = _np.zeros(len(_REPLAY_VOCAB), dtype=_np.float32)
        for i, w in enumerate(_REPLAY_VOCAB):
            v[i] = float(sum(1 for tok in toks if w in tok))
        n = _np.linalg.norm(v)
        v = v / n if n > 0 else _np.ones(len(_REPLAY_VOCAB), dtype=_np.float32)
        out.append(v / _np.linalg.norm(v) if n == 0 else v)
    return _np.vstack(out).astype(_np.float32)


def _replay_write_node(md: Path, name: str, body: str) -> None:
    nodes = md / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    fm = (f"---\nname: {name}\ndescription: {name}\ntype: project\n"
          f"chains: []\nrelevance: 0.5\ntier: warm\n---\n")
    (nodes / f"{name}.md").write_text(fm + body + "\n", encoding="utf-8")


class TestEngramReplayGenuineOnce(unittest.TestCase):
    def _build_two_engrams(self, md: Path):
        """Two near-identical source nodes -> two engram copies (a replayable pair)."""
        from samia.core import hippocampus as hip
        # Same vocab -> high cosine so the pair clears the neighbor threshold.
        _replay_write_node(md, "n1", "alpha alpha beta")
        _replay_write_node(md, "n2", "alpha alpha beta gamma")
        hip.EngramStore(md).materialize("n1")
        hip.EngramStore(md).materialize("n2")

    def test_first_replay_genuine_rest_fractional(self):
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch("samia.core.vector._embed_batch", _replay_stub_embed):
            md = Path(tmp)
            self._build_two_engrams(md)
            # First pass: a previously-unseen pair logs GENUINE.
            r1 = bio.replay_engram_traces(md, sample=10, threshold=0.5)
            self.assertGreaterEqual(r1["raw_pairs"], 1)
            self.assertEqual(r1["genuine"], r1["raw_pairs"])
            self.assertEqual(r1["fractional"], 0)
            # A second pass the SAME day is rate-limited (no double-fire farming).
            r_same = bio.replay_engram_traces(md, sample=10, threshold=0.5)
            self.assertEqual(r_same["genuine"], 0)
            self.assertEqual(r_same["fractional"], 0)
            self.assertEqual(r_same["skipped_same_day"], r_same["raw_pairs"])
            # Roll the per-pair ledger back a day to simulate the NEXT REM cycle:
            # now the SAME pair logs FRACTIONAL only (re-replay, never genuine again).
            st = bio._load_engram_replay_state(md)
            for v in st["genuine_pairs"].values():
                v["last_replay"] = "2020-01-01"
            bio._save_engram_replay_state(md, st)
            r2 = bio.replay_engram_traces(md, sample=10, threshold=0.5)
            self.assertEqual(r2["genuine"], 0)
            self.assertEqual(r2["fractional"], r2["raw_pairs"])

    def test_many_replays_do_not_cross_attractor_bar(self):
        """The homeostasis keystone: replaying ONE trace many times cannot promote it.

        The offline replay path fires REPEATEDLY (every REM cycle / idle pulse). Here
        the SAME single trace pair is replayed 50 times with NO genuine human/RAG
        recall in between. Genuine-once + the per-pair daily rate-limit must keep it
        sub-bar: the pair gets AT MOST ONE genuine event (well below the
        HEBB_PROMOTE_REPEATS bar) and the within-cycle re-replays are rate-limited, so
        replay ALONE never crosses the attractor bar (Q6a/D5 — one captured trace
        cannot be farmed into an attractor by repeated replay).
        """
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch("samia.core.vector._embed_batch", _replay_stub_embed):
            md = Path(tmp)
            self._build_two_engrams(md)
            # Replay the same single trace pair many times within the cycle window,
            # draining the co-activation log into edge_weights each time.
            for _ in range(50):
                bio.replay_engram_traces(md, sample=10, threshold=0.5)
                bio.hebbian_consolidate(md, promote=False)
            weights = bio._load_edge_weights(md)
            # The engram-derived pair must NOT have crossed the bar: AT MOST ONE
            # genuine event ever (the rest rate-limited) — replay alone cannot promote.
            self.assertTrue(weights, "expected the replayed pair to be recorded")
            for key, v in weights.items():
                self.assertLessEqual(int(v.get("count_genuine", 0)), 1,
                                     f"{key} got >1 genuine from replay-only")
                self.assertFalse(bio._is_promotable(v),
                                 f"{key} crossed the attractor bar on replay alone")

    def test_empty_engram_store_is_noop(self):
        with _tf.TemporaryDirectory() as tmp, \
                _mock.patch("samia.core.vector._embed_batch", _replay_stub_embed):
            md = Path(tmp)
            out = bio.replay_engram_traces(md)
            self.assertEqual(out["raw_pairs"], 0)
            self.assertEqual(out["genuine"], 0)
            self.assertEqual(out["fractional"], 0)


if __name__ == "__main__":
    unittest.main()
