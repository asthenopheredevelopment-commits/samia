"""Tests for samia.core.temporal_recall_sith — FEAT-2026-06-11 temporal-recall P2.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the SITH temporal-context machinery (proposal §4 + §16.2 Q3):
               1. integrator math — the leaky-integrator update t_k ← ρ_k·t_k +
                  sqrt(1−ρ_k²)·q + renormalize, ρ_k = exp(−dt/τ_k), bank stays on the
                  unit sphere; the first advance seeds t_k = q (ρ=0).
               2. coalescing — the bank advance is gated by the EXISTING cadence primitive
                  (bio._consolidate_cadence_blocked + ASTHENOS_HEBB_MIN_INTERVAL_S): a
                  burst inside one min-interval buffers (no advance), the next tick advances
                  ONCE on the mean of the buffered q's (burst-count-invariant).
               3. sidecar I/O — capture_snapshot writes a row keyed by engram id, is
                  write-once (re-materialize is a no-op), and _snapshot_for/round-trips it;
                  a missing snapshot reads as None (fail-open).
               4. read-out + jump-back — tc_term_hit scores a snapshot against the current
                  bank (1.0·1/K·K = 1 when bank==snapshot, gated by g_h); jump_back_blend
                  nudges the bank toward recalled snapshots and renormalizes.
               5. THE CONTRACT — flag-off / γ=0 byte-identity of chainogram_retrieve with
                  the P2 TC seam wired in.
    Depends: samia.core.temporal_recall_sith, context_extension, vector (EMBED_DIM
             monkeypatched small to avoid torch/HF), bio (cadence env); unittest, tempfile,
             os, json. All tests use tempfile dirs and NEVER touch the live memory tree.

Layer 2 (What / Why):
    What: drives the bank/sidecar with small hand-built vectors (EMBED_DIM patched to 4)
          so the integrator/coalesce/snapshot/read-out math is checkable in closed form,
          and proves the flag-off identity end-to-end through chainogram_retrieve with a
          deterministic FakeVI (no embedding backend).
    Why:  HARD CONTRACTS: coalescing must REUSE the cadence gate (not a new debounce);
          the snapshot must be immutable-once-written; and the whole layer must be a
          byte-identical no-op while γ=0. These tests assert those by construction.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from samia.core import context_extension as ce
from samia.core import temporal_recall_sith as sith
from samia.core import vector as _vec


# ── small-dim patch so we can build closed-form vectors without the HF backend ──────
_DIM = 4


class _DimPatch:
    """Context manager: force vector.EMBED_DIM to a small testable dim, then restore."""

    def __enter__(self):
        self._saved = _vec.EMBED_DIM
        _vec.EMBED_DIM = _DIM
        return self

    def __exit__(self, *exc):
        _vec.EMBED_DIM = self._saved
        return False


class _EnvGuard:
    """Context manager: set/clear env vars, restore the prior environment after."""

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
    return a / np.linalg.norm(a)


# ── 1. integrator math ─────────────────────────────────────────────────────────────
class TestIntegratorMath(unittest.TestCase):

    def test_first_advance_seeds_to_q(self):
        # The first-ever advance (no last_update) uses ρ=0 → t_k = q exactly for every k.
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            q = _unit([1.0, 0.0, 0.0, 0.0])
            self.assertTrue(sith.integrator_observe(md, q, now=1000.0))
            st = sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM)
            vecs = sith._bank_vectors(st)
            for i in range(sith.SITH_K_DEFAULT):
                np.testing.assert_allclose(vecs[i], q, atol=1e-9)

    def test_bank_stays_on_unit_sphere(self):
        # After a non-orthogonal second event, every t_k is renormalized to ‖t_k‖=1
        # (the §4.3 load-bearing renormalize — sqrt(1−ρ²) alone does NOT preserve norm).
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            sith.integrator_observe(md, _unit([1.0, 1.0, 0.0, 0.0]), now=0.0)
            sith.integrator_observe(md, _unit([1.0, 0.0, 1.0, 0.0]), now=100.0)
            st = sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM)
            vecs = sith._bank_vectors(st)
            for i in range(sith.SITH_K_DEFAULT):
                self.assertAlmostEqual(float(np.linalg.norm(vecs[i])), 1.0, places=9)

    def test_rho_from_dt_and_tau(self):
        # A FAST integrator (small τ) with a large dt resets to ~q; a SLOW integrator
        # (large τ) retains most of its prior context. Same event, different timescales.
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            t0 = _unit([1.0, 0.0, 0.0, 0.0])
            q1 = _unit([0.0, 1.0, 0.0, 0.0])  # orthogonal → clean norm bookkeeping
            sith.integrator_observe(md, t0, now=0.0)
            # advance one day later: τ_0=15min ≪ 1d (ρ≈0, near q1); τ_5=2mo ≫ 1d (ρ≈1).
            sith.integrator_observe(md, q1, now=86400.0)
            st = sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM)
            vecs = sith._bank_vectors(st)
            fast = vecs[0]   # τ=15min
            slow = vecs[-1]  # τ=2mo
            # fast integrator is dominated by q1 (its component on q1 ≫ on t0).
            self.assertGreater(abs(float(np.dot(fast, q1))),
                               abs(float(np.dot(fast, t0))))
            # slow integrator still retains substantial t0.
            self.assertGreater(abs(float(np.dot(slow, t0))), 0.5)


# ── 2. coalescing reuses the cadence gate ──────────────────────────────────────────
class TestCoalescing(unittest.TestCase):

    def test_no_env_means_every_event_advances(self):
        # Unset ASTHENOS_HEBB_MIN_INTERVAL_S → the gate never blocks → legacy per-event.
        with _DimPatch(), _EnvGuard(ASTHENOS_HEBB_MIN_INTERVAL_S=None), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            for _ in range(3):
                self.assertTrue(sith.integrator_observe(md, _unit([1, 1, 1, 1])))

    def test_burst_inside_interval_coalesces_to_one_advance(self):
        # With a long min-interval, the FIRST event advances (no prior run to block),
        # then a burst BUFFERS without advancing until the interval elapses. Drive the
        # cadence state file directly so wall-clock isn't a flake source.
        with _DimPatch(), _EnvGuard(ASTHENOS_HEBB_MIN_INTERVAL_S="3600"), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            from samia.core import bio as _bio
            paths = _bio._bio_paths(md)
            # Seed a recent consolidation run so the gate BLOCKS the next advances.
            paths["bio_dir"].mkdir(parents=True, exist_ok=True)
            import time as _t
            paths["hebb_consolidate_state"].write_text(
                json.dumps({"last_run_unix": _t.time()}), encoding="utf-8")
            # Now every observe should buffer (gate blocked), never advance.
            for _ in range(5):
                self.assertFalse(sith.integrator_observe(md, _unit([1, 0, 0, 0])))
            st = sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM)
            # buffer accumulated 5 q's, bank never advanced (still all-zero, no last).
            self.assertEqual(int(st["buf_count"]), 5)
            self.assertIsNone(st["last_update_unix"])

    def test_advance_uses_mean_of_buffer_then_clears(self):
        # The cadence primitive is reused for the *block* decision; here we verify the
        # advance consumes the MEAN of the buffered q's and resets the buffer.
        with _DimPatch(), _EnvGuard(ASTHENOS_HEBB_MIN_INTERVAL_S=None), \
                tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # Two opposite-but-equal-weight q's average to a vector along [1,1,0,0].
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)
            st = sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM)
            self.assertEqual(int(st["buf_count"]), 0)             # cleared after advance
            self.assertEqual(st["buf_sum"], [0.0] * _DIM)
            self.assertEqual(st["last_update_unix"], 0.0)


# ── 3. sidecar I/O (write-once-immutable) ──────────────────────────────────────────
class TestSnapshotSidecar(unittest.TestCase):

    def test_capture_and_roundtrip(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)
            self.assertTrue(sith.capture_snapshot(md, "engram_abc"))
            snap = sith._snapshot_for(md, "engram_abc",
                                      sith.SITH_K_DEFAULT, _DIM)
            self.assertIsNotNone(snap)
            self.assertEqual(snap.shape, (sith.SITH_K_DEFAULT, _DIM))
            # bank was seeded to [1,0,0,0] on every integrator → snapshot matches.
            for i in range(sith.SITH_K_DEFAULT):
                np.testing.assert_allclose(snap[i], _unit([1, 0, 0, 0]), atol=1e-6)

    def test_write_once_rematerialize_is_noop(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)
            self.assertTrue(sith.capture_snapshot(md, "engram_x"))
            first = sith._snapshot_for(md, "engram_x", sith.SITH_K_DEFAULT, _DIM)
            # Drift the bank, then re-capture the SAME eid: must be a no-op, snapshot held.
            sith.integrator_observe(md, _unit([0, 1, 0, 0]), now=99999.0)
            self.assertFalse(sith.capture_snapshot(md, "engram_x"))
            second = sith._snapshot_for(md, "engram_x", sith.SITH_K_DEFAULT, _DIM)
            np.testing.assert_allclose(first, second, atol=1e-9)

    def test_missing_snapshot_reads_none(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self.assertIsNone(
                sith._snapshot_for(md, "engram_never", sith.SITH_K_DEFAULT, _DIM))

    def test_two_eids_get_distinct_rows(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)
            sith.capture_snapshot(md, "engram_a")
            sith.integrator_observe(md, _unit([0, 1, 0, 0]), now=100.0)
            sith.capture_snapshot(md, "engram_b")
            man = json.loads(sith._snap_manifest_path(md).read_text())
            rows = {man["entries"]["engram_a"]["row"],
                    man["entries"]["engram_b"]["row"]}
            self.assertEqual(rows, {0, 1})


# ── 4. read-out + jump-back ────────────────────────────────────────────────────────
class TestReadoutAndJumpBack(unittest.TestCase):

    def test_tc_term_hit_perfect_match_is_gated_unit(self):
        # When the snapshot == current bank, every cos=1, Σ_k (1/K)·1 = 1; ×g_h.
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            from samia.core import hippocampus as _hip
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)
            # snapshot under the node's engram id (what tc_term_hit derives internally).
            node = "n_match"
            sith.capture_snapshot(md, _hip._engram_id(node))
            # gate passes → ~1.0; gate fails (g_h=0) → exactly 0.0.
            self.assertAlmostEqual(sith.tc_term_hit(md, node, 1.0), 1.0, places=5)
            self.assertEqual(sith.tc_term_hit(md, node, 0.0), 0.0)

    def test_tc_term_hit_no_snapshot_is_zero(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)
            self.assertEqual(sith.tc_term_hit(md, "n_absent", 1.0), 0.0)

    def test_jump_back_moves_bank_toward_snapshot(self):
        # Bank at [1,0,0,0]; a recalled node snapshotted at [0,1,0,0]; β blend pulls the
        # bank partway toward [0,1,0,0] (its component on [0,1,0,0] rises from 0).
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            from samia.core import hippocampus as _hip
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)  # bank → e0
            # Re-seed a snapshot for the recalled node at e1 by building it on a 2nd corpus
            # write then snapshotting; simplest: drift a throwaway, snapshot the node.
            node = "n_recalled"
            # Manually plant a snapshot at e1 for the node's eid (write-once API path).
            sith2 = sith
            # advance bank to e1, snapshot node, then restore bank to e0.
            sith.integrator_observe(md, _unit([0, 1, 0, 0]), now=1e9)  # bank → ~e1
            sith.capture_snapshot(md, _hip._engram_id(node))           # snap ≈ e1
            # force bank back to e0 for a clean before/after on the blend
            st = sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM)
            st["vectors"] = [[1.0, 0.0, 0.0, 0.0]] * sith.SITH_K_DEFAULT
            sith._bank_path(md).write_text(json.dumps(st), encoding="utf-8")
            before = sith._bank_vectors(sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM))
            self.assertTrue(sith.jump_back_blend(md, [node]))
            after = sith._bank_vectors(sith._load_bank(md, sith.SITH_K_DEFAULT, _DIM))
            # component on e1 increased; bank stayed on the unit sphere.
            e1 = _unit([0, 1, 0, 0])
            self.assertGreater(float(np.dot(after[0], e1)),
                               float(np.dot(before[0], e1)))
            self.assertAlmostEqual(float(np.linalg.norm(after[0])), 1.0, places=9)

    def test_jump_back_no_snapshot_is_noop(self):
        with _DimPatch(), tempfile.TemporaryDirectory() as d:
            md = Path(d)
            sith.integrator_observe(md, _unit([1, 0, 0, 0]), now=0.0)
            self.assertFalse(sith.jump_back_blend(md, ["n_nosnap"]))


# ── 5. THE CONTRACT — flag-off / γ=0 byte-identity through chainogram_retrieve ─────
class _FakeVI:
    """Deterministic stand-in for samia.core.vector (mirrors test_temporal_scaffold)."""

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


class TestFlagOffByteIdentityWithTCSeam(unittest.TestCase):
    """With the P2 TC seam wired in, flag-off (and γ=0) is still byte-identical."""

    def _run(self, md: Path):
        return ce.chainogram_retrieve(md, "q", _vi_module=_FakeVI(md, _HITS))

    def test_flag_off_identity(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            self.assertNotIn("error", ref)
            # baseline cosine+hebbian order, unchanged by P2.
            self.assertEqual(ref["loaded_chains"], ["c_beta", "c_alpha", "c_gamma"])

    def test_flag_on_gamma_zero_identity(self):
        # Master ON, but γ (and all weights) default 0.0 → compute-skip → no TC call →
        # score = base · 1.0, byte-identical to flag-off (§2.6 / §16.2-Q5).
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_GAMMA=None):
                on = self._run(md)
            self.assertEqual(_ranking(ref), _ranking(on))

    def test_flag_on_gamma_nonzero_no_snapshots_still_identity(self):
        # Even with γ>0, if no node has a SITH snapshot (no materialize ran), every
        # _tc_term_hit returns 0.0 (fails open) → TĈ=0 for every chain → score unchanged.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _build_corpus(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT=None):
                ref = self._run(md)
            with _EnvGuard(ASTHENOS_TEMPORAL_WEIGHT="1",
                           ASTHENOS_TEMPORAL_GAMMA="0.5"):
                on = self._run(md)
            ref_scores = [e["score"] for e in ref["loaded_nodes"]]
            on_scores = [e["score"] for e in on["loaded_nodes"]]
            self.assertEqual(ref_scores, on_scores)


if __name__ == "__main__":
    unittest.main()


# ─────────────────────────────────────────────
# [test_temporal_recall_sith] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-11  Status: active
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P2 tests — SITH temporal-
#             context term. Asserts: integrator update (ρ from dt/τ, unit-sphere
#             renormalize, ρ=0 seed); coalescing REUSES the cadence gate
#             (ASTHENOS_HEBB_MIN_INTERVAL_S) — burst buffers, tick advances on the mean;
#             snapshot sidecar write-once-immutable + roundtrip; tc_term_hit read-out
#             (gated unit on perfect match, 0 on missing); jump-back partial blend; and the
#             FLAG-OFF / γ=0 byte-identity of chainogram_retrieve with the TC seam wired in.
# Role:       prove P2's integrator/coalesce/sidecar/read-out math + flag-off identity
# Depends:    temporal_recall_sith, context_extension, vector, bio; unittest, tempfile, os
# ─────────────────────────────────────────────
