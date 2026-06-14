"""samia.core.temporal_recall_sith -- SITH multi-timescale temporal-context term (P2).

Layer 1 (Owns / Depends):
    Owns:    The SITH (Scale-Invariant Temporal History) temporal-context machinery
             of the temporal-recall layer (FEAT-2026-06-11-memory-temporal-recall-
             formula-v01 §4 + §16.2 Q3) — the inside of the additive cue's
               TĈ_c = Σ_{h∈hits∩c} g_h · Σ_{k=1..K} ω_k·cos(t_k, c_{h,k}).
             Three pieces:
               1. The leaky-integrator BANK: K (default 6, configurable 5-8) context
                  vectors t_k on the unit sphere, time constants τ_k log-spaced across
                  the tier horizons, updated t_k ← ρ_k·t_k + sqrt(1−ρ_k²)·q then
                  renormalized, with ρ_k = exp(−dt/τ_k) and dt from sub-day wall-clock.
                  The update is COALESCED to one tick per ASTHENOS_HEBB_MIN_INTERVAL_S
                  by reusing bio._consolidate_cadence_blocked (§16.2 Q3 — burst-invariant).
               2. The encode-snapshot SIDECAR (keyed by engram id, write-once-immutable):
                  the bank state {c_{h,k}} captured at hippocampus.materialize, stored as
                  a packed K·dim matrix + manifest mirroring the engram-embedding layout
                  (NOT frontmatter — §4.4).
               3. The query-time read-out (tc_term_hit) + the recall jump-back partial
                  blend (β≈0.3) consumed by bio.hebbian_record (§4.5).
    Depends: numpy; samia.core.vector (EMBED_DIM, _embed_batch); samia.core.bio
             (_bio_paths, _consolidate_cadence_blocked, HEBB_MIN_INTERVAL_ENV — REUSED,
             not reinvented); samia.core.hippocampus (_engram_id, _hippocampus_dir);
             samia.core.atomic_state.locked_update_json (the EXISTING flock primitive).

Layer 2 (What / Why):
    What: integrator_observe(memory_dir, q) appends q to a coalesce buffer and, when the
          shared cadence gate says the min-interval has elapsed, advances the bank on the
          MEAN of the buffered q-vectors (one update per tick, not per event). The bank
          + buffer + last-update clock live in biomimetic/sith_bank.json under the same
          locked_update_json discipline as the other per-corpus state. capture_snapshot
          writes the live bank into the sidecar for an eid, once (re-materialize is a
          no-op). tc_term_hit reads a hit's snapshot and scores it against the current
          bank; jump_back_blend nudges the bank toward recalled nodes' snapshots.
    Why:  §4. The corpus clock is day-granular; a single drift rate cannot resolve a node
          co-accessed an hour ago and one two months ago. A log-spaced bank does — some
          τ_k always sits near the relevant lag (the SITH/Laplace generalization of CMR).
          The snapshot lets the contiguity cosine cos(t_k, c_{h,k}) compare NOW-context to
          ENCODE-context (TCM's contiguity effect). Coalescing by reuse makes the new
          per-write hot path O(wall-clock), not O(event-count) — burst-count-invariant,
          the §16.2-Q3 robustness requirement — with NO new debounce primitive.

Flag posture: P2 is read by the formula ONLY through context_extension._tc_term_hit, which
    runs ONLY when ASTHENOS_TEMPORAL_WEIGHT is on AND γ ≥ ε (§16.2-Q5 compute-skip). With
    the master flag off or γ=0, none of this module is on any retrieval path, so the
    chainogram_retrieve flag-off byte-identity holds. The write-side hooks (snapshot at
    materialize, integrator_observe / jump_back at hebbian_record) are fail-soft and never
    break a write/recall; a legacy node with no snapshot yields a 0.0 TĈ contribution
    (fails open), so the term is additive-optional with no migration.
"""
from __future__ import annotations

import json
import math
import time as _time
from pathlib import Path

import numpy as np

from . import vector as _vi

# ── Seed bank parameters (§4.3 / §4.6). All join the joint-calibration vector later;
#    here they are the frozen seeds. K=6 with τ_k log-spaced across the tier horizons
#    (ring/hot minutes-to-hours → cold/frozen months). ω_k uniform (DRIFT Q1). The bank
#    is NOT activated until a calibration flips weights on — these are only consulted
#    when the TC term is computed, which is gated off while γ=0.
SITH_K_DEFAULT = 6                                  # 5-8 allowed; 6 is the seed bank
# τ_k in SECONDS, log-spaced to the tier horizons (§4.3 table):
#   15min, 2h, 12h, 3d, 2wk, 2mo.
SITH_TAU_SECONDS_SEED = (
    15 * 60.0,          # k0 — ring / hot working set
    2 * 3600.0,         # k1 — within-session, hot
    12 * 3600.0,        # k2 — day boundary
    3 * 86400.0,        # k3 — warm-fresh
    14 * 86400.0,       # k4 — warm→cold decay band
    60 * 86400.0,       # k5 — cold / frozen archive
)
# β — the recall context-reinstatement blend (§4.5, DRIFT Q4). 0 < β < 1; full
#   reinstatement (β=1) is forbidden (it would erase ongoing drift and the forward
#   asymmetry). 0.3 is the seed; joins the calibration vector.
SITH_JUMPBACK_BETA = 0.3

# Bank state file: one per corpus, alongside the other biomimetic/ state JSONs. Holds
#   the K unit-vectors, the coalesce buffer (pending q-means + count), and the last
#   bank-advance wall-clock (for the ρ_k = exp(−dt/τ_k) dt).
_BANK_RELNAME = "sith_bank.json"
# Snapshot sidecar: mirrors the engram-embedding model — a packed K·dim matrix +
#   a manifest mapping engram-id → row, under hippocampus/ (NOT frontmatter, §4.4).
_SNAP_DIRNAME = "context_snapshots"


def sith_tau_seconds(k: int = SITH_K_DEFAULT) -> tuple[float, ...]:
    """Return the K log-spaced time constants τ_k (seconds), for a bank of size k.

    What: the K=6 seed bank is returned verbatim; for any other k in 5-8 the τ_k are
      log-interpolated across the same [15min, 2mo] tier-horizon span so the bank stays
      log-spaced (§4.3) regardless of K.
    Why: K joins the calibration vector (5-8); the τ_k are DERIVED from K (not a free
      axis), so a calibration that picks K=7 still gets a log-spaced bank.
    """
    if k == SITH_K_DEFAULT:
        return SITH_TAU_SECONDS_SEED
    lo = math.log(SITH_TAU_SECONDS_SEED[0])
    hi = math.log(SITH_TAU_SECONDS_SEED[-1])
    if k <= 1:
        return (SITH_TAU_SECONDS_SEED[0],)
    return tuple(math.exp(lo + (hi - lo) * i / (k - 1)) for i in range(k))


def _embed_dim() -> int:
    """Embedding dim the bank vectors live in (mirrors the main index, MiniLM=384)."""
    return int(_vi.EMBED_DIM)


# ── paths ────────────────────────────────────────────────────────────────────────

def _bank_path(memory_dir: Path):
    """biomimetic/sith_bank.json — the live integrator bank + coalesce buffer."""
    from . import bio as _bio
    return _bio._bio_paths(memory_dir)["bio_dir"] / _BANK_RELNAME


def _snap_dir(memory_dir: Path) -> Path:
    """hippocampus/context_snapshots/ — the encode-snapshot sidecar root."""
    from . import hippocampus as _hip
    return _hip._hippocampus_dir(memory_dir) / _SNAP_DIRNAME


def _snap_manifest_path(memory_dir: Path) -> Path:
    return _snap_dir(memory_dir) / "manifest.json"


def _snap_matrix_path(memory_dir: Path) -> Path:
    return _snap_dir(memory_dir) / "snapshots.npy"


# ── q normalization ────────────────────────────────────────────────────────────────

def _l2norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector; a zero vector is returned unchanged (no div-by-zero)."""
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v
    return v / n


# ── bank state (load / default / save) ───────────────────────────────────────────

def _fresh_bank(k: int, dim: int) -> dict:
    """A fresh bank: K zero context-vectors, empty coalesce buffer, no last-update.

    The integrators start at the zero vector (no context yet); the first observed q
    seeds them via the standard update. A zero t_k yields cos(t_k, ·)=0, so an
    un-warmed bank injects no temporal signal — fail-open, like a missing snapshot.
    """
    return {
        "k": int(k),
        "dim": int(dim),
        "vectors": [[0.0] * dim for _ in range(k)],
        # coalesce buffer: running SUM of observed q's + count, so the tick can take
        # the MEAN without storing every vector (§16.2-Q3 mean-of-coalesced-q).
        "buf_sum": [0.0] * dim,
        "buf_count": 0,
        "last_update_unix": None,    # wall-clock of the last bank ADVANCE (for dt)
    }


def _load_bank(memory_dir: Path, k: int, dim: int) -> dict:
    """Read the persisted bank, or a fresh one. Heals a corrupt/empty/shape-wrong file."""
    p = _bank_path(memory_dir)
    if not p.exists():
        return _fresh_bank(k, dim)
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _fresh_bank(k, dim)
    # Defensive: a bank built under a different K/dim (a re-seed) is rebuilt fresh
    # rather than crashing the update; the temporal layer is inert until calibration.
    if st.get("k") != k or st.get("dim") != dim or "vectors" not in st:
        return _fresh_bank(k, dim)
    return st


def _bank_vectors(st: dict) -> np.ndarray:
    """The K×dim bank as a float array."""
    return np.asarray(st.get("vectors", []), dtype=np.float64)


# ── the integrator update (coalesced) ────────────────────────────────────────────

def integrator_observe(memory_dir: Path, q, *, k: int = SITH_K_DEFAULT,
                       now: float | None = None) -> bool:
    """Feed one context-bearing event into the bank, COALESCED off the cadence gate.

    What: append the L2-normalized embedding q to the bank's coalesce buffer (a running
      sum + count). Then consult the SHARED cadence gate (bio._consolidate_cadence_blocked
      + ASTHENOS_HEBB_MIN_INTERVAL_S): if the min-interval has NOT elapsed since the last
      bank advance, return without advancing — the q stays buffered. If it HAS (or the env
      is unset → never blocks), advance every integrator ONCE on the MEAN of the buffered
      q-vectors with dt = now − last_update, then clear the buffer and stamp the advance.
      Returns True iff the bank was advanced this call.
    Why:  §16.2-Q3. Reusing the existing min-interval debounce (NOT a new one) makes the
      bank update O(wall-clock), not O(event-count): a burst of N writes inside one tick
      collapses to a single advance on their mean, so the bank is burst-count-invariant.
      The wall-clock-derived ρ_k = exp(−dt/τ_k) of §4.3 is preserved — dt is the true
      elapsed time since the last advance, which is what lifts the bank off the day axis.

    Fail-soft: any internal error leaves the bank untouched and returns False — the SITH
      term simply contributes nothing for that event rather than breaking the write/recall.
    """
    try:
        dim = _embed_dim()
        qv = _l2norm(np.asarray(q, dtype=np.float64).reshape(-1))
        if qv.shape[0] != dim:
            return False
        from . import bio as _bio
        from .atomic_state import locked_update_json
        paths = _bio._bio_paths(memory_dir)
        paths["bio_dir"].mkdir(parents=True, exist_ok=True)
        now_t = float(now) if now is not None else _time.time()
        advanced = False
        with locked_update_json(_bank_path(memory_dir),
                                default=_fresh_bank(k, dim)) as st:
            if st.get("k") != k or st.get("dim") != dim or "vectors" not in st:
                st.clear()
                st.update(_fresh_bank(k, dim))
            # 1. Buffer this q (running sum + count → cheap mean, no vector list).
            buf = np.asarray(st["buf_sum"], dtype=np.float64) + qv
            st["buf_sum"] = buf.tolist()
            st["buf_count"] = int(st.get("buf_count", 0)) + 1
            # 2. Cadence gate (REUSED): block → leave buffered, advance next tick.
            if _bio._consolidate_cadence_blocked(paths):
                return False
            # 3. Advance on the MEAN of the coalesced q's.
            mean_q = buf / max(st["buf_count"], 1)
            mean_q = _l2norm(mean_q)
            last = st.get("last_update_unix")
            taus = sith_tau_seconds(k)
            vecs = _bank_vectors(st)
            if vecs.shape != (k, dim):
                vecs = np.zeros((k, dim), dtype=np.float64)
            for i in range(k):
                # First-ever advance (no last) → ρ=0 → t_k = q exactly (seed the bank).
                if last is None:
                    rho = 0.0
                else:
                    dt = max(0.0, now_t - float(last))
                    rho = math.exp(-dt / float(taus[i]))
                t = rho * vecs[i] + math.sqrt(max(0.0, 1.0 - rho * rho)) * mean_q
                vecs[i] = _l2norm(t)          # renormalize (§4.3 — load-bearing)
            st["vectors"] = vecs.tolist()
            st["buf_sum"] = [0.0] * dim
            st["buf_count"] = 0
            st["last_update_unix"] = now_t
            advanced = True
        return advanced
    except Exception:
        return False


# ── encode-snapshot sidecar (write-once-immutable, keyed by engram id) ────────────

def _load_snap_manifest(memory_dir: Path) -> dict:
    p = _snap_manifest_path(memory_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_snap_matrix(memory_dir: Path):
    p = _snap_matrix_path(memory_dir)
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def capture_snapshot(memory_dir: Path, engram_id: str, *,
                     k: int = SITH_K_DEFAULT) -> bool:
    """Capture the live bank state as the encode-snapshot {c_{h,k}} for engram_id.

    What: flatten the current K×dim bank into one row of the snapshot sidecar matrix and
      record engram_id → row in the sidecar manifest — but ONLY if engram_id is not
      already snapshotted (the write-once guard). Returns True iff a row was written.
    Why:  §4.4 — the snapshot is captured at materialize and is IMMUTABLE once written:
      keyed by the engram id and guarded by `eid ∉ manifest`, so a re-materialize is a
      no-op on the sidecar (it preserves the FIRST encode-time context, not the drifted
      one). Stored in a sidecar (NOT frontmatter) because a K·384 bank is ~9KB and the
      only consumer is the retrieval-time SITH cosine — it has no place on the hot parse
      path. Mirrors the engram-embedding manifest+packed-matrix layout (reuse the model).

    Fail-soft: any error leaves the sidecar untouched and returns False — a missing
      snapshot simply yields a 0.0 TĈ for that hit (fails open).
    """
    try:
        dim = _embed_dim()
        from .atomic_state import locked_update_json
        d = _snap_dir(memory_dir)
        d.mkdir(parents=True, exist_ok=True)
        # Read the live bank (NOT under the manifest lock — bank has its own lock).
        st = _load_bank(memory_dir, k, dim)
        flat = _bank_vectors(st).reshape(-1)
        if flat.shape[0] != k * dim:
            # Un-warmed / shape-wrong bank → store an all-zero snapshot (cos→0, fails open).
            flat = np.zeros(k * dim, dtype=np.float64)
        wrote = False
        with locked_update_json(_snap_manifest_path(memory_dir), default={}) as man:
            entries = man.setdefault("entries", {})
            if engram_id in entries:
                return False                 # write-once: re-materialize is a no-op
            mat = _load_snap_matrix(memory_dir)
            row = flat.astype(np.float32).reshape(1, -1)
            if mat is None or mat.shape[0] == 0:
                out = row
                idx = 0
            elif mat.shape[1] != row.shape[1]:
                # Shape change (re-seed): rebuild from this row (snapshots are inert
                # until calibration, so a fresh matrix is acceptable, no migration).
                out = row
                idx = 0
                entries.clear()
            else:
                idx = int(mat.shape[0])
                out = np.vstack([mat, row])
            np.save(_snap_matrix_path(memory_dir), out.astype(np.float32))
            entries[engram_id] = {"row": idx, "k": int(k), "dim": int(dim)}
            man["k"] = int(k)
            man["dim"] = int(dim)
            wrote = True
        return wrote
    except Exception:
        return False


def _snapshot_for(memory_dir: Path, engram_id: str, k: int, dim: int):
    """Return the K×dim encode-snapshot for engram_id, or None if absent (fail-open)."""
    man = _load_snap_manifest(memory_dir)
    entry = (man.get("entries") or {}).get(engram_id)
    if not entry or entry.get("row") is None:
        return None
    mat = _load_snap_matrix(memory_dir)
    if mat is None:
        return None
    row = int(entry["row"])
    if row < 0 or row >= mat.shape[0]:
        return None
    flat = np.asarray(mat[row], dtype=np.float64)
    if flat.shape[0] != k * dim:
        return None
    return flat.reshape(k, dim)


# ── the formula read-out (consumed by context_extension._tc_term_hit) ─────────────

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a, b); 0.0 when either vector has zero norm (an un-warmed bank/snapshot)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def tc_term_hit(memory_dir: Path, node_name: str, g_h: float, *,
                k: int = SITH_K_DEFAULT) -> float:
    """Per-hit SITH temporal-context contribution g_h · Σ_k ω_k·cos(t_k, c_{h,k}).

    What: resolve the hit node's encode-snapshot {c_{h,k}} from the sidecar (keyed by the
      node's engram id), read the CURRENT bank {t_k}, and return
        g_h · Σ_{k} ω_k · cos(t_k, c_{h,k})    with uniform ω_k = 1/K.
      A node with no snapshot (legacy / never materialized) contributes 0.0 (fails open).
    Why:  §4.1/§4.4 — this is the inside of the additive cue TĈ_c, summed over the chain's
      gated hits and γ-weighted by the caller. cos(t_k, c_{h,k}) compares NOW-context to
      ENCODE-context (TCM's contiguity effect). The g_h gate (θ=0.2) is applied by the
      caller; passing it here keeps the per-hit formula self-contained and lets a gate of
      0.0 short-circuit to 0.0 with no sidecar I/O.

    Fail-soft: any error → 0.0 (the term simply does not fire for that hit).
    """
    try:
        if not g_h:
            return 0.0
        dim = _embed_dim()
        from . import hippocampus as _hip
        eid = _hip._engram_id(node_name)
        snap = _snapshot_for(memory_dir, eid, k, dim)
        if snap is None:
            return 0.0
        st = _load_bank(memory_dir, k, dim)
        vecs = _bank_vectors(st)
        if vecs.shape != (k, dim):
            return 0.0
        omega = 1.0 / float(k)               # uniform read-out weights (DRIFT Q1)
        acc = 0.0
        for i in range(k):
            acc += omega * _cos(vecs[i], snap[i])
        return float(g_h) * acc
    except Exception:
        return 0.0


# ── recall jump-back: partial context reinstatement (consumed by bio.hebbian_record) ──

def jump_back_blend(memory_dir: Path, retrieved_nodes: list[str], *,
                    k: int = SITH_K_DEFAULT,
                    beta: float = SITH_JUMPBACK_BETA) -> bool:
    """Nudge the live bank partway toward the recalled nodes' encode contexts (§4.5).

    What: for each integrator, blend t_k ← (1−β)·t_k + β·mean_n c_{n,k}, where the mean is
      over the recalled nodes that HAVE a snapshot (nodes without one are skipped from the
      mean — fail-open). Renormalize each t_k after the blend. Returns True iff a blend
      was applied (at least one recalled node had a snapshot).
    Why:  §4.5 — context reinstatement is TCM/CMR's signature mechanism: recalling an item
      pulls the present context partway back toward that item's encode context, producing
      the lag-CRP forward asymmetry and the clustering of successive recalls. The blend is
      PARTIAL (β<1, seed 0.3) so the present is nudged, not overwritten — full reinstatement
      would erase ongoing drift. It mutates ONLY the live bank, never any node's stored
      snapshot (snapshots are immutable, §4.4). Mirrors the bank's own lock discipline.

    Fail-soft: any error leaves the bank untouched and returns False — jump-back is a
      soft enhancement of recall, never a write/recall blocker.
    """
    try:
        if not retrieved_nodes:
            return False
        dim = _embed_dim()
        from . import hippocampus as _hip
        # Gather the recalled nodes' encode snapshots (skip those without one).
        snaps = []
        for n in retrieved_nodes:
            eid = _hip._engram_id(n)
            s = _snapshot_for(memory_dir, eid, k, dim)
            if s is not None:
                snaps.append(s)
        if not snaps:
            return False
        target = np.mean(np.stack(snaps, axis=0), axis=0)   # mean_n c_{n,k}, shape K×dim
        from . import bio as _bio
        from .atomic_state import locked_update_json
        _bio._bio_paths(memory_dir)["bio_dir"].mkdir(parents=True, exist_ok=True)
        with locked_update_json(_bank_path(memory_dir),
                                default=_fresh_bank(k, dim)) as st:
            if st.get("k") != k or st.get("dim") != dim or "vectors" not in st:
                st.clear()
                st.update(_fresh_bank(k, dim))
            vecs = _bank_vectors(st)
            if vecs.shape != (k, dim):
                vecs = np.zeros((k, dim), dtype=np.float64)
            for i in range(k):
                blended = (1.0 - beta) * vecs[i] + beta * target[i]
                vecs[i] = _l2norm(blended)
            st["vectors"] = vecs.tolist()
        return True
    except Exception:
        return False


# [Asthenosphere] samia.core.temporal_recall_sith
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P2 — SITH temporal-
#             context term (§4 + §16.2 Q3). K=6 log-spaced leaky integrators; update
#             coalesced to one tick per ASTHENOS_HEBB_MIN_INTERVAL_S by REUSING
#             bio._consolidate_cadence_blocked (no new debounce); encode-snapshot
#             sidecar (write-once, keyed by engram id, NOT frontmatter); jump-back
#             partial blend β≈0.3. Inert at retrieval until ASTHENOS_TEMPORAL_WEIGHT +
#             γ≥ε flip it on; flag-off is a byte-identical no-op.
# Layer:      core (pure library, no daemon dependency)
# Role:       compute the additive-cue SITH temporal-context term TĈ_c
# Stability:  stable -- v1.0.0; additive-optional, inert until the temporal flag + γ flip on.
# ErrorModel: fail-soft throughout — every write-side hook (integrator_observe,
#             capture_snapshot, jump_back_blend) swallows errors and returns False,
#             leaving the bank/sidecar untouched; tc_term_hit returns 0.0 on any error
#             or a missing snapshot, so a legacy hit contributes nothing (fails open).
#             Bank/sidecar state is mutated only under locked_update_json (flock).
# Depends:    numpy; vector (EMBED_DIM/_embed_batch); bio (cadence primitive — reused);
#             hippocampus (_engram_id/_hippocampus_dir); atomic_state (locked_update_json).
#             stdlib (json, math, time, pathlib).
# Exposes:    sith_tau_seconds, integrator_observe, capture_snapshot, tc_term_hit,
#             jump_back_blend. Constants: SITH_K_DEFAULT, SITH_TAU_SECONDS_SEED,
#             SITH_JUMPBACK_BETA.
# Lines:      486
# --------------------------------------------------------------------------
