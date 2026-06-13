"""temporal_distinctiveness.py -- log-time distinctiveness term (P5, SIMPLE ratio rule).

Layer 1 (Owns / Depends):
    Owns:    The distinctiveness re-ranker D̂_c of the temporal-recall layer
             (FEAT-2026-06-11-memory-temporal-recall-formula-v01 §7) — the third
             multiplicative modulator in the gain envelope
             (1 + λN·N̂_c + λK·K̂_c + λD·D̂_c). The SIMPLE temporal-ratio rule
             (Brown, Neath & Chater 2007):

                 D(i) = 1 / Σ_{j∈C} exp( −c · | logT_i − logT_j | )

             over the candidate pool C (the chains assembled in chainogram_retrieve,
             one representative time per chain at best_node(c)). T_i is elapsed seconds
             from the representative event time to "now"; the log axis makes the rule
             scale-free. D(i) ∈ (0, 1] — larger = more temporally ISOLATED = should
             rank higher (an isolated candidate → 1; one crowded by m near-time
             neighbours → 1/(m+1)). The pool-scan is O(|C|²) but |C| ≤ 8 (max_chains),
             so it is a sub-millisecond, stateless, no-new-files compute (§7.5).
    Depends: math (stdlib log/exp); samia.core.temporal (infer_valid_from — REUSED for
             the day-granular T_i fallback chain incl. st_mtime, §7.3); samia.core.
             frontmatter (read_node — the preferred high-resolution written_at source).
             No numpy: the pool is tiny so a plain Python double loop is cheapest.

Layer 2 (What / Why):
    What: dist_vector(memory_dir, best_nodes, now=None) reads each chain's
          representative event time T_i (written_at float first, else the
          infer_valid_from three-tier day-date, else st_mtime — §7.3), applies the
          §7.4 APPLICABILITY GATE (if the pool spans < a meaningful log-time ratio it
          collapses to no-signal — every chain reads 0.0, so the term contributes
          nothing rather than emit noise), and otherwise computes the SIMPLE ratio
          D(i) for every chain. dist_at(dist, cname) reads that {cname -> D_raw} map
          for one chain. The caller (context_extension._apply_temporal_envelope) pool
          min-max normalizes the raw D values into D̂_c ∈ [0,1].
    Why:  §7. The other temporal terms answer "what was active near encoding" (TĈ),
          "what does the need-path reach" (N̂), "what was tagged-and-captured" (K̂).
          None answer the orthogonal ratio-rule question: is this memory temporally
          ISOLATED among the candidates, or buried in a dense cluster of competitors?
          A memory alone on the time axis is a cleaner retrieval target. This is
          distinct from SAM/IA's existing recency factor (a plain exponential of age,
          blind to neighbour spacing) — D̂ is "how alone in time," not "how old."

Honest ceiling (§7.3 caveat, stated in-tree per the proposal's instruction): D̂ ranks
    by distinctiveness of WRITE / MATERIALIZATION time, an approximation of true event
    distinctiveness. written_at (when present) is sub-day; the infer_valid_from fallback
    is day-granular (parse_date truncates to s[:10], DATE_RE matches only YYYY-MM-DD),
    so many same-day candidates collapse to |logT_i − logT_j| = 0 and are mutually
    maximally confusable; the weakest tier is st_mtime (filesystem write time). This is
    the honest ceiling of the available substrate — and why λD is seeded small and
    calibration-deferred. The term is safe by construction: flag-off (λD=0) is byte-
    identical; the applicability gate fails soft to no-signal; the compute is stateless
    and pool-bounded.

Flag posture: P5 is read by the formula ONLY through context_extension._dist_vector /
    _dist_term_chain, which run ONLY when ASTHENOS_TEMPORAL_WEIGHT is on AND λD ≥ ε
    (§16.2-Q5 compute-skip). With the master flag off or λD=0 this module is on no
    retrieval path, so the chainogram_retrieve flag-off byte-identity holds. A pool with
    no usable times, or one whose dynamic range is degenerate, yields an all-0.0 map →
    every D̂_c = 0 → the envelope reduces to 1.0 (fails open) — additive-optional, no
    migration; legacy nodes lacking written_at degrade to the infer_valid_from chain.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import time as _time
from pathlib import Path
from typing import Optional

# ── Seed parameters (§7.2/§7.4/§7.6; c joins the joint-calibration vector later) ─────
# DIST_SHARPNESS_SEED -- What: SIMPLE's c, the distinctiveness sharpness — larger c →
#   sharper confusability falloff → more candidates count as "isolated".
# Why: §7.2. Seed c=1.0, bound [0.5, 2.0]. Read each call (live env, default seed) so a
#   calibration adapter can sweep it without re-import; only consulted when the dist term
#   is computed at all, which is gated off while λD=0. Mirrors successor's gSR/L readers.
DIST_SHARPNESS_SEED = 1.0
DIST_SHARPNESS_MIN = 0.5
DIST_SHARPNESS_MAX = 2.0
DIST_SHARPNESS_ENV = "ASTHENOS_DIST_C"

# DIST_MAX_RATIO -- What: the <1000× upper sanity bound on the pool's dynamic range
#   (§7.4). Beyond ~1000× the oldest candidate's T is dominated by st_mtime/materialized_
#   at artifacts rather than true event time, so we clip |logT_i − logT_j| at log(1000)
#   before the exp (the tails are not trustworthy past that span).
# DIST_MIN_LOG_SPREAD -- What: the lower edge of the applicability gate. If the pool's
#   total log-time spread is effectively zero (all candidates clustered within a hair on
#   the log axis), D(i) is uniform and min-max degenerate — we emit no signal instead of
#   noise (§7.4). A tiny epsilon distinguishes a genuine spread from float wobble.
# Why: §7.4 — both edges fail SOFT to "no signal", consistent with every other temporal
#   term's flag-off discipline: D̂ can never HURT a result, only re-order inside the
#   trustworthy regime.
DIST_MAX_RATIO = 1000.0
DIST_LOG_CLIP = math.log(DIST_MAX_RATIO)
DIST_MIN_LOG_SPREAD = 1e-9

# DIST_MIN_T_SECONDS -- What: floor on T_i so log(T_i) is finite and well-behaved even
#   for a just-written node (T near 0). One second is below the day-granular floor of the
#   fallback path and harmless when written_at gives sub-second resolution.
# Why: log(0) is −inf; a 1s floor keeps the log axis defined without distorting any
#   realistic age (a node is never meaningfully "0 seconds old" at retrieval).
DIST_MIN_T_SECONDS = 1.0


def dist_sharpness() -> float:
    """Resolve c (distinctiveness sharpness), live env, clamped to [0.5, 2.0] (§7.2/§7.6).

    What: reads ASTHENOS_DIST_C each call; missing/unparseable ⇒ the 1.0 seed; always
      clamped to the [0.5, 2.0] bound.
    Why: c seeds 1.0 and joins the joint-calibration vector; clamping keeps a calibration
      sweep inside the SIMPLE-validated range no matter what env value is set. Read-each-
      call mirrors successor.successor_gsr so a sweep needs no re-import.
    """
    raw = os.environ.get(DIST_SHARPNESS_ENV)
    if raw is None:
        val = DIST_SHARPNESS_SEED
    else:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = DIST_SHARPNESS_SEED
    return min(DIST_SHARPNESS_MAX, max(DIST_SHARPNESS_MIN, val))


def _node_path(memory_dir: Path, node: str) -> Path:
    """Resolve a bare node name to its nodes/<name>.md path."""
    fname = node if node.endswith(".md") else f"{node}.md"
    return Path(memory_dir) / "nodes" / fname


def representative_time_seconds(memory_dir: Path, node: Optional[str],
                                now: Optional[float] = None) -> Optional[float]:
    """Elapsed seconds from a node's representative event time to "now" (the T_i, §7.3).

    What: resolve the node's representative event time and return (now − that time) in
      seconds, floored at DIST_MIN_T_SECONDS. Source preference (§7.3): (1) the high-
      resolution written_at Unix float when present (sub-day), else (2) infer_valid_from's
      three-tier day-granular chain (last_access date → earliest body date → st_mtime).
      Returns None when the node file is missing/unreadable so the caller can drop it.
    Why: §7.3 — D̂ uses the time of best_node(c) as the chain's representative event time.
      written_at is the PREFERRED sub-day source; the infer_valid_from fallback (incl. its
      st_mtime bottom tier, temporal.py:103) keeps a legacy node — no written_at, no
      valid_from, no body date — still yielding a usable T (degrading to filesystem write
      time, the weakest event-time proxy). Honest ceiling: this ranks write/materialization
      distinctiveness, an approximation of event distinctiveness (the §7.3 caveat).
    """
    if not node:
        return None
    p = _node_path(memory_dir, node)
    if not p.exists():
        return None
    t_now = now if now is not None else _time.time()

    # (1) Preferred high-resolution source: the written_at Unix float (§3 / §7.3).
    written_at: Optional[float] = None
    fm_lines: list[str] = []
    body = ""
    try:
        from . import frontmatter as _fm
        fm, _order, body = _fm.read_node(p)
        raw = fm.get("written_at")
        if raw is not None:
            written_at = float(raw)
        # Reconstruct the temporal.py fm-lines view ("key: value") so infer_valid_from
        # (which reads last_access off frontmatter lines) can be reused without re-parse.
        fm_lines = [f"{k}: {v}" for k, v in fm.items()]
    except (TypeError, ValueError):
        written_at = None
    except Exception:
        # Unreadable frontmatter: fall through to the day-granular / st_mtime chain below.
        written_at = None

    if written_at is not None and written_at > 0.0:
        return max(DIST_MIN_T_SECONDS, t_now - written_at)

    # (2) Fallback: infer_valid_from's day-granular three-tier chain (incl. st_mtime).
    try:
        from . import temporal as _tq
        vf_date = _tq.infer_valid_from(p, fm_lines, body)
        # Anchor the day-date at its start (midnight) — the day-granular floor (§7.3).
        midnight = _dt.datetime(vf_date.year, vf_date.month, vf_date.day)
        secs = midnight.timestamp()
        return max(DIST_MIN_T_SECONDS, t_now - secs)
    except Exception:
        return None


def dist_vector(memory_dir: Path, best_nodes: dict,
                now: Optional[float] = None, *,
                c: Optional[float] = None) -> dict:
    """Pool-scan the SIMPLE ratio rule over the candidate pool (§7.2/§7.4). Computed ONCE.

    What: best_nodes is {cname -> best_node_name}. Resolve each chain's representative
      T_i (representative_time_seconds), then over the pool of usable times compute the
      §7.4 APPLICABILITY GATE and, if it passes, D(i) = 1/Σ_j exp(−c·|logT_i − logT_j|)
      for every chain, clipping |logT_i − logT_j| at log(1000) (§7.4). Returns the raw
      {cname -> D_raw} map (D_raw ∈ (0, 1]); the caller pool min-max normalizes it into
      D̂_c ∈ [0,1]. A chain whose time could not be resolved, OR the whole pool when the
      gate fails (degenerate spread), reads 0.0 — fail-soft to "no signal".
    Why: §7.2/§7.4/§7.5. The pool scan is O(|C|²) with |C| ≤ 8 (max_chains) — sub-ms,
      stateless, no new files. The applicability gate is the load-bearing safety: SIMPLE's
      ratio rule is only informative when the pool actually spans a range of temporal
      distances; if every candidate has nearly the same T the denominator → |C| for all
      and D is uniform (min-max degenerate), so we emit 0.0 for the whole pool rather than
      noise. Both edges (degenerate spread; >1000× range clipped) fail SOFT, so D̂ can
      never hurt a result, only re-order inside the trustworthy regime. Fails open: an
      unusable pool → all-0.0 → every D̂_c = 0 → envelope reduces to 1.0.

    Two-stage (dist_vector → dist_at) mirrors successor.need_vector/need_at: the pool-wide
    quantity is computed once per query and read per chain, never re-scanned per chain.
    """
    if not best_nodes:
        return {}
    sharp = dist_sharpness() if c is None else min(
        DIST_SHARPNESS_MAX, max(DIST_SHARPNESS_MIN, float(c)))
    t_now = now if now is not None else _time.time()

    # Resolve each chain's representative T_i; a chain with no usable time is dropped from
    # the active distinctiveness pool (it reads 0.0 below — fail-soft, additive-optional).
    log_t: dict[str, float] = {}
    for cname, node in best_nodes.items():
        t = representative_time_seconds(memory_dir, node, now=t_now)
        if t is None or t <= 0.0:
            continue
        log_t[cname] = math.log(t)

    # Default every input chain to 0.0 (no signal); only gate-passing chains overwrite.
    out: dict[str, float] = {cname: 0.0 for cname in best_nodes}
    if len(log_t) < 2:
        # A pool of fewer than two timed candidates cannot express a ratio → no signal.
        return out

    # §7.4 applicability gate (lower edge): if the pool's total log-time spread is
    # effectively zero, D is uniform and min-max degenerate → emit no signal.
    lv = list(log_t.values())
    if (max(lv) - min(lv)) < DIST_MIN_LOG_SPREAD:
        return out

    # SIMPLE ratio over the timed pool; clip the log-distance at log(1000) (upper edge).
    for ci, lti in log_t.items():
        denom = 0.0
        for _cj, ltj in log_t.items():
            d = abs(lti - ltj)
            if d > DIST_LOG_CLIP:
                d = DIST_LOG_CLIP
            denom += math.exp(-sharp * d)
        # denom ≥ 1 (the j=i term is exp(0)=1), so D ∈ (0, 1]; guard anyway.
        out[ci] = (1.0 / denom) if denom > 0.0 else 0.0
    return out


def dist_at(dist: dict, cname: Optional[str]) -> float:
    """Read the per-chain distinctiveness map at one chain (§7.2 best-node modulator).

    What: D_raw(c) = dist[cname]; a chain absent from the map (no usable time, or the
      applicability gate failed) reads 0.0.
    Why: §7.3 — D̂ is a best-node modulator, evaluated at the chain's representative time.
      The caller pool min-max normalizes the per-chain raw values into D̂_c ∈ [0,1], so a
      0.0 chain contributes no lift — a bounded modulator that can never flip sign or
      dominate the additive base. Mirrors successor.need_at.
    """
    if not dist or not cname:
        return 0.0
    return float(dist.get(cname, 0.0))


# ─────────────────────────────────────────────
# [temporal_distinctiveness] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-11  Status: active
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P5 — temporal
#             distinctiveness (§7). SIMPLE log-time ratio D(i)=1/Σ_j exp(−c·|logT_i−
#             logT_j|) over the candidate pool (c=1.0, bound [0.5,2.0]); T_i = seconds
#             since best_node's written_at (sub-day) else infer_valid_from day-date
#             (incl. st_mtime fallback, temporal.py:103). Applicability gate: degenerate
#             log-spread OR >1000× range → soft-fail to no-signal (clip at log(1000)).
#             Two-stage dist_vector→dist_at (pool computed once, read per chain) mirrors
#             successor.need_vector/need_at. Inert at retrieval until ASTHENOS_TEMPORAL_
#             WEIGHT + λD≥ε flip it on; flag-off / λD=0 is a byte-identical no-op.
# Role:       compute the multiplicative distinctiveness modulator D̂_c (SIMPLE ratio)
# Depends:    math, time, datetime, os (stdlib); temporal (infer_valid_from — REUSED),
#             frontmatter (read_node — the written_at high-resolution source)
# Citations:  Brown, Neath & Chater (2007) SIMPLE, Psychol. Rev. 114(3) 539–576;
#             Bjork & Whitten (1974) Cognitive Psychology 6(2) 173–189.
# ─────────────────────────────────────────────
