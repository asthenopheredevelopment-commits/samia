"""Tests for samia.core.temporal_recall_stc — FEAT-2026-06-11 temporal-recall P4.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the STC tagging-and-capture machinery (proposal §6 + §16.2 Q2):
               1. EPISODE_SEQ window scoping — capture targets the N-before / M-after
                  NEAREST episodes by counter (NOT a wall-clock [t−9h,t+3h] window): two
                  weak nodes equidistant in real time but at different episode_seq are
                  treated differently, and the strong-before-weak asymmetry (N>M) holds.
               2. Guards — the cosine gate (cos<θ → not captured), the strong→weak-only
                  direction (a strong neighbour is never captured), and the
                  1-event/chain/hour rate-limit (a second fire on the same chain is a no-op).
               3. The three effects — recall (stc_chain_score max-over-members + the P1 K
                  seam), promotion (the OR-gate third arm in promote_ring_pointer), and
                  decay (the combined-capped damping in tier.step_relevance).
               4. THE CONTRACT — flag-off byte-identity: capture writes NOTHING with the
                  master flag off, so the decay path and the chainogram_retrieve scorer are
                  byte-identical to today (no node ever carries stc_capture_score).
    Depends: samia.core.temporal_recall_stc, context_extension, tier, hippocampus, bio,
             vector (EMBED_DIM monkeypatched small to avoid torch/HF), frontmatter;
             unittest, tempfile, os, json, numpy. All tests use tempfile dirs and NEVER
             touch the live memory tree (mirrors test_tier / test_temporal_recall_sith).

Layer 2 (What / Why):
    What: builds small hand-made corpora — nodes with explicit episode_seq / written_at /
          salience frontmatter plus a controllable vector index (manifest + embeddings.npy)
          so the cosine guard is checkable in closed form — and drives capture_event +
          the three effect read-outs directly. The flag-off identity is proven through
          chainogram_retrieve with a deterministic FakeVI (no embedding backend) and
          through tier.decay_pass.
    Why:  HARD CONTRACTS: the window MUST be episode_seq-relative (§16.2 Q2, superseding
          §6.3); the guards MUST hold; the three effects must be inert at zero; and the
          whole layer must be a byte-identical no-op while the master flag is off. These
          tests assert each by construction.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from samia.core import context_extension as ce
from samia.core import temporal_recall_stc as stc
from samia.core import tier as _tier
from samia.core import vector as _vec


# ── small-dim patch so cosines are closed-form without the HF backend ───────────────
_DIM = 4


class _DimPatch:
    def __enter__(self):
        self._saved = _vec.EMBED_DIM
        _vec.EMBED_DIM = _DIM
        return self

    def __exit__(self, *exc):
        _vec.EMBED_DIM = self._saved
        return False


class _EnvGuard:
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


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def _write_node(md: Path, name: str, *, chains: list[str], episode_seq=None,
                written_at=None, salience=0.0, relevance=0.5, tier="warm",
                grade=None) -> None:
    """Write one node .md with the explicit frontmatter the STC window/guards read."""
    nodes = md / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    ch = "[" + ", ".join(chains) + "]"
    lines = ["---", f"name: {name}", "description: n", "type: project",
             f"chains: {ch}", "valid_from: 2026-06-11", "valid_to: null",
             "last_access: 2026-06-11", "access_count: 0",
             f"relevance: {relevance}", f"tier: {tier}", f"salience: {salience}"]
    if grade is not None:
        lines.append(f"material_grade: {grade}")
    if episode_seq is not None:
        lines.append(f"episode_seq: {episode_seq}")
    if written_at is not None:
        lines.append(f"written_at: {written_at}")
    lines += ["---", f"body of {name}", ""]
    (nodes / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def _write_chain(md: Path, chain: str, members: list[str]) -> None:
    chains = md / "chains"
    chains.mkdir(parents=True, exist_ok=True)
    data = {"name": chain,
            "members": [{"file": f"{m}.md", "addr": f"{chain}.{i}"}
                        for i, m in enumerate(members)],
            "edges": []}
    (chains / f"{chain}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_index(md: Path, vecs: dict) -> None:
    """Write a vector_index (manifest + embeddings.npy) so bio._node_embedding works.

    vecs: {node_name -> _DIM-vector}. Rows are assigned in dict order.
    """
    idx = md / "vector_index"
    idx.mkdir(parents=True, exist_ok=True)
    rows = list(vecs.items())
    arr = np.array([np.asarray(v, dtype=np.float64) for _n, v in rows])
    np.save(idx / "embeddings.npy", arr)
    manifest = {"entries": {f"{n}.md": {"row": i} for i, (n, _v) in enumerate(rows)}}
    (idx / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _captured_score(md: Path, node: str):
    """Read the RAW stored stc_capture_score off a node, or None if absent."""
    from samia.core import frontmatter as _fm
    p = md / "nodes" / f"{node}.md"
    fm, _o, _b = _fm.read_node(p)
    return fm.get("stc_capture_score")


# ── 1. EPISODE_SEQ window scoping (NOT wall-clock) ──────────────────────────────────
class TestEpisodeSeqWindow(unittest.TestCase):

    def _corpus(self, md: Path) -> None:
        # One chain; an anchor at seq=10, weak neighbours at various seqs. All embeddings
        # are aligned (cos=1) so the cosine gate always passes — isolate the WINDOW.
        e = _unit([1, 1, 1, 1])
        _write_chain(md, "c1",
                     ["wb1", "wb2", "anchor", "wf1", "wf2"])
        # written_at all within the wall-clock cap (same instant) so the cap never fires
        # → only episode_seq ordinal nearness decides membership.
        wat = 1_000_000.0
        _write_node(md, "wb1", chains=["c1"], episode_seq=8, written_at=wat, salience=0.1)
        _write_node(md, "wb2", chains=["c1"], episode_seq=9, written_at=wat, salience=0.1)
        _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=wat,
                    salience=0.95)
        _write_node(md, "wf1", chains=["c1"], episode_seq=11, written_at=wat, salience=0.1)
        _write_node(md, "wf2", chains=["c1"], episode_seq=12, written_at=wat, salience=0.1)
        _write_index(md, {n: e for n in ("wb1", "wb2", "anchor", "wf1", "wf2")})

    def test_window_captures_ordinal_neighbours(self):
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self._corpus(md)
            res = stc.capture_event(md, "anchor", now=1_000_000.0)
            self.assertTrue(res["fired"])
            captured = set(res["captured"])
            # All 4 weak neighbours fall inside N=9-before / M=3-after of seq 10.
            self.assertEqual(captured, {"wb1.md", "wb2.md", "wf1.md", "wf2.md"})
            for n in ("wb1", "wb2", "wf1", "wf2"):
                self.assertIsNotNone(_captured_score(md, n))
            # The anchor itself is never captured.
            self.assertIsNone(_captured_score(md, "anchor"))

    def test_window_is_ordinal_not_wallclock(self):
        # A weak node ordinally FAR (seq 100) but written at the SAME instant as the
        # anchor is OUT of the episode_seq window — proving the unit is episode_seq, not
        # wall-clock. We seed MORE than M after-neighbours nearer in seq than 'far' so
        # 'far' is pushed past the M-nearest cut; a wall-clock window (all same instant)
        # would have captured it, an ordinal one does not.
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            e = _unit([1, 1, 1, 1])
            wat = 1_000_000.0
            nearers = [f"a{i}" for i in range(stc.STC_WINDOW_FWD_M)]  # exactly M nearer
            _write_chain(md, "c1", ["anchor"] + nearers + ["far"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=wat,
                        salience=0.95)
            vecs = {"anchor": e}
            for i, nm in enumerate(nearers):
                _write_node(md, nm, chains=["c1"], episode_seq=11 + i, written_at=wat,
                            salience=0.1)
                vecs[nm] = e
            # 'far' is ordinally distant (seq 100); SAME real time but beyond the M nearer.
            _write_node(md, "far", chains=["c1"], episode_seq=100, written_at=wat,
                        salience=0.1)
            vecs["far"] = e
            _write_index(md, vecs)
            res = stc.capture_event(md, "anchor", now=wat)
            self.assertEqual(set(res["captured"]),
                             {f"{nm}.md" for nm in nearers})
            self.assertIsNone(_captured_score(md, "far"))

    def test_strong_before_weak_asymmetry(self):
        # N(before) > M(after): with more weak neighbours than the window on each side,
        # MORE before-neighbours are captured than after-neighbours (the biology).
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            e = _unit([1, 1, 1, 1])
            wat = 1_000_000.0
            befores = [f"b{i}" for i in range(stc.STC_WINDOW_BACK_N + 3)]  # more than N
            afters = [f"a{i}" for i in range(stc.STC_WINDOW_FWD_M + 3)]    # more than M
            members = befores + ["anchor"] + afters
            _write_chain(md, "c1", members)
            seq = 1
            vecs = {}
            for nm in befores:
                _write_node(md, nm, chains=["c1"], episode_seq=seq, written_at=wat,
                            salience=0.1)
                vecs[nm] = e
                seq += 1
            anchor_seq = seq
            _write_node(md, "anchor", chains=["c1"], episode_seq=anchor_seq,
                        written_at=wat, salience=0.95)
            vecs["anchor"] = e
            seq += 1
            for nm in afters:
                _write_node(md, nm, chains=["c1"], episode_seq=seq, written_at=wat,
                            salience=0.1)
                vecs[nm] = e
                seq += 1
            _write_index(md, vecs)
            res = stc.capture_event(md, "anchor", now=wat)
            cap = set(res["captured"])
            n_before = sum(1 for c in cap if c[:-3] in befores)
            n_after = sum(1 for c in cap if c[:-3] in afters)
            self.assertEqual(n_before, stc.STC_WINDOW_BACK_N)
            self.assertEqual(n_after, stc.STC_WINDOW_FWD_M)
            self.assertGreater(n_before, n_after)


# ── 2. Guards ───────────────────────────────────────────────────────────────────────
class TestGuards(unittest.TestCase):

    def test_cosine_gate_blocks_off_topic(self):
        # A weak node ORTHOGONAL to the anchor (cos=0 < θ=0.2) is not captured even when
        # ordinally adjacent; an aligned one IS.
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            wat = 1_000_000.0
            _write_chain(md, "c1", ["anchor", "ontopic", "offtopic"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=wat,
                        salience=0.95)
            _write_node(md, "ontopic", chains=["c1"], episode_seq=11, written_at=wat,
                        salience=0.1)
            _write_node(md, "offtopic", chains=["c1"], episode_seq=12, written_at=wat,
                        salience=0.1)
            _write_index(md, {"anchor": _unit([1, 0, 0, 0]),
                              "ontopic": _unit([1, 0, 0, 0]),       # cos 1
                              "offtopic": _unit([0, 1, 0, 0])})     # cos 0
            res = stc.capture_event(md, "anchor", now=wat)
            self.assertEqual(set(res["captured"]), {"ontopic.md"})

    def test_strong_to_weak_only(self):
        # A neighbour that is itself STRONG (salience >= threshold) is never captured —
        # capture flows from strong to weak only.
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            e = _unit([1, 1, 1, 1])
            wat = 1_000_000.0
            _write_chain(md, "c1", ["anchor", "weak", "strong"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=wat,
                        salience=0.95)
            _write_node(md, "weak", chains=["c1"], episode_seq=11, written_at=wat,
                        salience=0.1)
            _write_node(md, "strong", chains=["c1"], episode_seq=12, written_at=wat,
                        salience=0.9)   # >= STC_STRONG_THRESHOLD → not capturable
            _write_index(md, {n: e for n in ("anchor", "weak", "strong")})
            res = stc.capture_event(md, "anchor", now=wat)
            self.assertEqual(set(res["captured"]), {"weak.md"})

    def test_wallclock_cap_blocks_distant_realtime(self):
        # Ordinally adjacent but written FAR in real time (beyond STC_WALLCLOCK_CAP_S) →
        # the wall-clock cap (the §16.2 Q2 bound on the human-side span) skips it.
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            e = _unit([1, 1, 1, 1])
            anchor_wat = 1_000_000.0
            far_wat = anchor_wat - (stc.STC_WALLCLOCK_CAP_S + 100.0)
            _write_chain(md, "c1", ["distant", "anchor", "near"])
            _write_node(md, "distant", chains=["c1"], episode_seq=9, written_at=far_wat,
                        salience=0.1)
            _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=anchor_wat,
                        salience=0.95)
            _write_node(md, "near", chains=["c1"], episode_seq=11, written_at=anchor_wat,
                        salience=0.1)
            _write_index(md, {n: e for n in ("distant", "anchor", "near")})
            res = stc.capture_event(md, "anchor", now=anchor_wat)
            self.assertEqual(set(res["captured"]), {"near.md"})

    def test_rate_limit_one_per_chain_per_hour(self):
        # A second strong anchor on the SAME chain within the hour is a no-op (rate-limit).
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            e = _unit([1, 1, 1, 1])
            wat = 1_000_000.0
            _write_chain(md, "c1", ["anchor", "weak"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=wat,
                        salience=0.95)
            _write_node(md, "weak", chains=["c1"], episode_seq=11, written_at=wat,
                        salience=0.1)
            _write_index(md, {"anchor": e, "weak": e})
            now = wat
            first = stc.capture_event(md, "anchor", now=now)
            self.assertTrue(first["fired"])
            # second fire 10 minutes later → blocked (within STC_RATE_LIMIT_S).
            second = stc.capture_event(md, "anchor", now=now + 600.0)
            self.assertFalse(second["fired"])
            self.assertEqual(second["reason"], "rate-limited")
            # an hour+ later → allowed again.
            third = stc.capture_event(md, "anchor", now=now + stc.STC_RATE_LIMIT_S + 1.0)
            self.assertTrue(third["fired"])

    def test_legacy_anchor_without_seq_is_noop(self):
        # An anchor lacking episode_seq (legacy / secondary write seam) cannot place a
        # window → no capture, no crash (additive-optional, §3.2).
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            e = _unit([1, 1, 1, 1])
            _write_chain(md, "c1", ["anchor", "weak"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=None,
                        written_at=1_000_000.0, salience=0.95)
            _write_node(md, "weak", chains=["c1"], episode_seq=11,
                        written_at=1_000_000.0, salience=0.1)
            _write_index(md, {"anchor": e, "weak": e})
            res = stc.capture_event(md, "anchor", now=1_000_000.0)
            self.assertFalse(res["fired"])
            self.assertEqual(res["reason"], "anchor-no-seq")
            self.assertIsNone(_captured_score(md, "weak"))


# ── 3a. Effect (1): recall read-out (max over members + attenuation) ───────────────
class TestRecallReadout(unittest.TestCase):

    def test_max_over_members(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            now = 2_000_000.0
            _write_node(md, "m_lo", chains=["c1"], salience=0.1)
            _write_node(md, "m_hi", chains=["c1"], salience=0.1)
            _write_node(md, "m_none", chains=["c1"], salience=0.1)
            # stamp two members with capture scores at `now` (no attenuation), one untagged.
            stc._stamp_capture(md, "m_lo", 0.3, now)
            stc._stamp_capture(md, "m_hi", 0.9, now)
            score = stc.stc_chain_score(md, ["m_lo.md", "m_hi.md", "m_none.md"], now=now)
            self.assertAlmostEqual(score, 0.9, places=6)   # MAX, not mean

    def test_attenuation_half_life(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            cap_at = 1_000_000.0   # non-zero capture instant (a real written_at)
            _write_node(md, "m", chains=["c1"], salience=0.1)
            stc._stamp_capture(md, "m", 1.0, cap_at)
            # exactly one half-life later → 0.5; two → 0.25.
            one = stc.current_capture_score(
                md, "m", now=cap_at + stc.STC_HALFLIFE_DAYS * 86400.0)
            two = stc.current_capture_score(
                md, "m", now=cap_at + 2 * stc.STC_HALFLIFE_DAYS * 86400.0)
            self.assertAlmostEqual(one, 0.5, places=4)
            self.assertAlmostEqual(two, 0.25, places=4)
            # at the capture instant itself → full score.
            self.assertAlmostEqual(stc.current_capture_score(md, "m", now=cap_at),
                                   1.0, places=6)

    def test_missing_score_reads_zero(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _write_node(md, "m", chains=["c1"], salience=0.1)
            self.assertEqual(stc.current_capture_score(md, "m"), 0.0)
            self.assertEqual(stc.stc_chain_score(md, ["m.md"]), 0.0)


# ── 3b. Effect (2): promotion OR-gate third arm ────────────────────────────────────
class TestPromotionGate(unittest.TestCase):

    def test_stc_arm_promotes_weak_node(self):
        # A weak, low-frequency, low-salience node carrying a fresh high capture score is
        # promotion-eligible via the STC arm. _entry_stc reads the attenuated score with
        # now=real-time, so stamp the capture at real time (a just-captured node).
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            import time as _t
            from samia.core import hippocampus as _hip
            _write_node(md, "weak", chains=["c1"], salience=0.1)
            stc._stamp_capture(md, "weak", 0.9, _t.time())   # fresh → ~0.9
            ring = _hip.RingStore(md)
            entry = {"ptr": "weak", "target_tier": "main", "genuine_hits": 0}
            # _entry_stc reads the (barely-attenuated) score; gate fires on the STC arm.
            self.assertGreaterEqual(ring._entry_stc(entry), _hip.STC_PROMOTE_THRESHOLD)

    def test_below_arm_does_not_fire(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            from samia.core import hippocampus as _hip
            _write_node(md, "weak", chains=["c1"], salience=0.1)
            # no capture score → arm reads 0.0 < threshold.
            ring = _hip.RingStore(md)
            entry = {"ptr": "weak", "target_tier": "main", "genuine_hits": 0}
            self.assertEqual(ring._entry_stc(entry), 0.0)


# ── 3d. Integration: the capture TRIGGER fires through bio.compute_salience ─────────
class TestCaptureTriggerWiring(unittest.TestCase):

    def test_explicit_tag_anchor_captures_via_compute_salience(self):
        # An explicit-tag write clamps salience to 0.95 (>= STC_STRONG_THRESHOLD); running
        # the live bio.compute_salience write path should FIRE capture on the neighbours.
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            from samia.core import bio as _bio
            e = _unit([1, 1, 1, 1])
            wat = 1_000_000.0
            _write_chain(md, "c1", ["anchor", "weak"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=wat,
                        salience=0.0)
            _write_node(md, "weak", chains=["c1"], episode_seq=11, written_at=wat,
                        salience=0.1)
            _write_index(md, {"anchor": e, "weak": e})
            # compute_salience writes salience=0.95 (explicit tag) then fires capture.
            sal = _bio.compute_salience(md, "anchor", content="anchor body",
                                        explicit_tag=True, write=True)
            self.assertGreaterEqual(sal, stc.STC_STRONG_THRESHOLD)
            self.assertIsNotNone(_captured_score(md, "weak"))

    def test_trigger_inert_when_flag_off(self):
        # Same write, flag OFF → compute_salience still writes salience but capture is a
        # no-op (no stc_capture_score on the neighbour).
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            from samia.core import bio as _bio
            e = _unit([1, 1, 1, 1])
            wat = 1_000_000.0
            _write_chain(md, "c1", ["anchor", "weak"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=10, written_at=wat,
                        salience=0.0)
            _write_node(md, "weak", chains=["c1"], episode_seq=11, written_at=wat,
                        salience=0.1)
            _write_index(md, {"anchor": e, "weak": e})
            _bio.compute_salience(md, "anchor", content="anchor body",
                                  explicit_tag=True, write=True)
            self.assertIsNone(_captured_score(md, "weak"))


# ── 3c. Effect (3): decay damping (combined + capped) ──────────────────────────────
class TestDecayDamping(unittest.TestCase):

    def test_stc_slows_decay(self):
        # A captured node decays SLOWER than an identical uncaptured one (stale regime).
        plain, _r = _tier.step_relevance(0.6, False, days_since_access=30,
                                         grade="natural", salience=0.0, stc_capture=0.0)
        tagged, _r2 = _tier.step_relevance(0.6, False, days_since_access=30,
                                           grade="natural", salience=0.0, stc_capture=1.0)
        # both decay toward 0; the tagged one has a SMALLER decrement → higher new_rel.
        self.assertLess(plain, tagged)
        self.assertLess(tagged, 0.6)   # still decays (decay-everywhere)

    def test_combined_capped_never_zeroes(self):
        # Fully salient AND fully captured → damping is CAPPED at DAMP_CAP, so the node
        # still decays at >= (1-DAMP_CAP) of rate (never frozen).
        new_rel, _r = _tier.step_relevance(0.6, False, days_since_access=30,
                                           grade="natural", salience=1.0, stc_capture=1.0)
        self.assertLess(new_rel, 0.6)
        # decrement = 0.6 * rate * (1-DAMP_CAP); confirm it is the floor, not zero.
        rate = _tier.DECAY_RATE_BY_GRADE["natural"]
        expected = 0.6 + (0.0 - 0.6) * rate * (1.0 - _tier.DAMP_CAP)
        self.assertAlmostEqual(new_rel, expected, places=9)

    def test_zero_stc_is_identity(self):
        # stc=0 → the damping reduces to salience-only → byte-identical to the prior fn.
        with_stc, _r = _tier.step_relevance(0.6, False, days_since_access=30,
                                            grade="natural", salience=0.4, stc_capture=0.0)
        # recompute salience-only by hand.
        rate = _tier.DECAY_RATE_BY_GRADE["natural"]
        rate2 = rate * (1.0 - _tier.SALIENCE_DECAY_DAMPING * 0.4)
        expected = 0.6 + (0.0 - 0.6) * rate2
        self.assertAlmostEqual(with_stc, expected, places=12)


# ── 4. THE CONTRACT — flag-off byte-identity ───────────────────────────────────────
class _FakeVI:
    """Deterministic stand-in for samia.core.vector (mirrors test_temporal_recall_sith)."""

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


def _build_chainogram_corpus(md: Path) -> None:
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
        lines = ["---", f"name: {nm}", "description: n", "type: project",
                 f"chains: [{ch}]", "valid_from: 2026-06-11", "valid_to: null",
                 "last_access: 2026-06-11", "access_count: 0", "relevance: 0.5",
                 "tier: warm", "written_at: 1781827200.0", "episode_seq: 1",
                 "---", f"body of {nm}", ""]
        (nodes / f"{nm}.md").write_text("\n".join(lines), encoding="utf-8")
        chain = {"name": ch, "members": [{"file": f"{nm}.md", "addr": f"{ch}.0"}],
                 "edges": edges}
        (chains / f"{ch}.json").write_text(json.dumps(chain), encoding="utf-8")


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


class TestFlagOffByteIdentity(unittest.TestCase):
    """With the P4 K seam wired in, flag-off (and λK=0) is still byte-identical."""

    def _run(self, md: Path):
        return ce.chainogram_retrieve(md, "q", _vi_module=_FakeVI(md, _HITS))

    def test_recall_flag_off_identity(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_chainogram_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            self.assertNotIn("error", ref)
            self.assertEqual(ref["loaded_chains"], ["c_beta", "c_alpha", "c_gamma"])

    def test_recall_flag_on_lambda_k_zero_identity(self):
        # Master ON, λK defaults 0.0 → compute-skip → no STC read → score = base·1.0.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_chainogram_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_LAMBDA_K=None):
                on = self._run(md)
            self.assertEqual(_ranking(ref), _ranking(on))

    def test_recall_lambda_k_nonzero_no_capture_still_identity(self):
        # Even with λK>0, if no node carries a capture score (none ever fired) every
        # K_raw is 0.0 (fails open) → K̂=0 for every chain → score unchanged.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_chainogram_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_LAMBDA_K="0.5"):
                on = self._run(md)
            ref_scores = [e["score"] for e in ref["loaded_nodes"]]
            on_scores = [e["score"] for e in on["loaded_nodes"]]
            self.assertEqual(ref_scores, on_scores)

    def test_capture_writes_nothing_when_flag_off(self):
        # The capture trigger is inert under the master flag off — no frontmatter written.
        with _DimPatch(), _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            e = _unit([1, 1, 1, 1])
            _write_chain(md, "c1", ["anchor", "weak"])
            _write_node(md, "anchor", chains=["c1"], episode_seq=10,
                        written_at=1_000_000.0, salience=0.95)
            _write_node(md, "weak", chains=["c1"], episode_seq=11,
                        written_at=1_000_000.0, salience=0.1)
            _write_index(md, {"anchor": e, "weak": e})
            res = stc.capture_event(md, "anchor", now=1_000_000.0)
            self.assertFalse(res["fired"])
            self.assertEqual(res["reason"], "flag-off")
            self.assertIsNone(_captured_score(md, "weak"))

    def test_decay_pass_byte_identical_without_capture(self):
        # tier.decay_pass over nodes that carry NO stc_capture_score is byte-identical
        # whether or not the temporal flag is set (the field is simply absent).
        from samia.core import tier as _t
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # two stale nodes (no STC field), salience present.
            _write_node(md, "s1", chains=["c1"], salience=0.3, relevance=0.6,
                        tier="warm", grade="natural")
            _write_node(md, "s2", chains=["c1"], salience=0.0, relevance=0.4,
                        tier="warm", grade="natural")
            # make them stale by backdating last_access via a direct rewrite is complex;
            # instead drive decay_pass with an explicit far-future `today`.
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                off = _t.decay_pass(md / "nodes", dry=True, today="2026-12-31",
                                    auto_freeze=False)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_LAMBDA_K="0.5"):
                on = _t.decay_pass(md / "nodes", dry=True, today="2026-12-31",
                                   auto_freeze=False)
            self.assertEqual(off, on)


if __name__ == "__main__":
    unittest.main()
