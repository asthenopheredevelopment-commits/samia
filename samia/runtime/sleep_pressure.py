"""samia.runtime.sleep_pressure — composite sleep-pressure / health metric.

Layer 1 (Owns / Depends):
    Owns:    compute_pressure(mem) — the composite "how much reconciliation is
             owed" score, summing NORMALIZED clutter backlogs into a single
             interpretable [0..N] number PLUS the per-signal breakdown (the
             operator-visible health gauge). Owns each signal's reader and its
             normalization cap.
    Depends: samia.runtime.contradiction (list_supersession_candidates — the
             unresolved contradiction backlog), the on-disk clutter sources
             under <mem> (.session_offload/, .consolidation_candidates.json,
             nodes/, biomimetic/coactivation_log.jsonl, biomimetic/
             edge_weights.json), and samia.core.bio.HEBB_PROMOTION (the
             edges-grown-without-promotion bar). Every reader is fail-soft.

Layer 2 (What / Why):
    What: REM P1's metric. It reads the live SAM/IA backlogs first-hand and
          returns {signals, score, threshold, sleep_needed}. Each signal is a
          distinct kind of accumulated clutter the REM batch later resolves:
          unconsolidated-offload backlog, unresolved contradiction candidates,
          near-duplicate consolidation backlog, hot/warm tier overflow,
          coactivation-log depth, and edges-grown-without-promotion. The P1
          metric is deliberately SIMPLE (Q2 "start simple"): a SUM of each
          signal normalized to ~[0,1] against its own configurable cap, with
          per-signal weights defaulting to 1.0 (env-tunable later in P3).
    Why:  The honest "is rest owed?" measure that gates REM entry (Q1) and
          becomes the operator-visible health gauge (rem_status). A composite
          (not one dominant signal) is the only faithful read of total owed
          reconciliation; normalizing each signal by its own cap keeps any
          raw-count signal from swamping the others. An ABSENT source must
          contribute 0 and be noted — the metric never crashes on a missing
          file (feedback_harness_memory_vs_viability: degrade, never break).

P1 is produce-only: pure reads, no mutation, no thread, no clock. The only
caller in P1 is rem_cycle.should_sleep / the rem_status surface.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

_log = logging.getLogger("samia.runtime.sleep_pressure")

# ---------------------------------------------------------------------------
# Configuration — threshold + per-signal normalization caps + weights.
#
# Each signal is normalized to [0,1] by dividing its raw count by its CAP
# (clamped). The composite is the weighted SUM, so a fully-saturated single
# signal contributes its weight (default 1.0). The threshold is expressed on
# the same composite scale; with six signals at weight 1.0 the score ranges
# [0, 6], and the default threshold of 1.0 means "roughly one fully-saturated
# backlog (or the equivalent spread across several) is owed". All env-tunable
# so P3 can tune weights/caps without a code change (Q2 defers tuning).
# ---------------------------------------------------------------------------

THRESHOLD = float(os.environ.get("REM_SLEEP_PRESSURE_THRESHOLD", "1.0"))

# Per-signal caps (the raw count that normalizes to 1.0). Sane defaults grounded
# in the live tree (e.g. the verified ~600-pair near-dup backlog).
_CAPS: dict[str, float] = {
    "offload_backlog": float(os.environ.get("REM_CAP_OFFLOAD", "20")),
    "contradiction_backlog": float(os.environ.get("REM_CAP_CONTRADICTION", "25")),
    "near_dup_backlog": float(os.environ.get("REM_CAP_NEAR_DUP", "600")),
    "tier_overflow": float(os.environ.get("REM_CAP_TIER_OVERFLOW", "200")),
    "coactivation_depth": float(os.environ.get("REM_CAP_COACTIVATION", "500")),
    "edges_unpromoted": float(os.environ.get("REM_CAP_EDGES_UNPROMOTED", "300")),
}

# Per-signal weights (Q2: default 1.0, env-tunable; P3 tunes empirically).
_WEIGHTS: dict[str, float] = {
    name: float(os.environ.get(f"REM_WEIGHT_{name.upper()}", "1.0"))
    for name in _CAPS
}

# Tier-overflow caps: how many nodes the hot+warm tiers may hold before the
# overflow signal saturates. Counts beyond (HOT_CAP + WARM_CAP) are clutter.
_HOT_CAP = float(os.environ.get("REM_TIER_HOT_CAP", "150"))
_WARM_CAP = float(os.environ.get("REM_TIER_WARM_CAP", "400"))


def _clamp01(x: float) -> float:
    """Clamp to [0,1]. What: bound a normalized signal. Why: a raw count above
    its cap must saturate at 1.0, never exceed it (keeps the sum interpretable)."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ---------------------------------------------------------------------------
# Per-signal readers — each returns (raw_count, present). present=False means
# the source is ABSENT (missing file / unavailable module); such a signal
# contributes 0 and is noted, never raised (Q2 / fail-soft contract).
# ---------------------------------------------------------------------------


def _read_offload_backlog(mem: Path) -> tuple[float, bool]:
    """Unconsolidated session-offload backlog.

    What: counts the per-session offload state files under <mem>/.session_offload/
          (samia.core.session_offload writes <session_id>.json as it offloads
          sliding-window blocks). Each is an offload that has not yet been
          consolidated back into durable memory.
    Why:  a growing offload directory is the "raw episodic awaiting consolidation"
          clutter REM drains. Absent dir → 0 (no sessions offloaded yet).
    """
    d = mem / ".session_offload"
    if not d.is_dir():
        return 0.0, False
    try:
        return float(sum(1 for _ in d.glob("*.json"))), True
    except OSError:
        return 0.0, False


def _read_contradiction_backlog(mem: Path) -> tuple[float, bool]:
    """Unresolved contradiction (supersession) candidates.

    What: counts the UNRESOLVED rows in biomimetic/supersession_candidates.jsonl
          via contradiction.list_supersession_candidates(unresolved_only=True).
    Why:  each unresolved candidate is a pending clash REM's passive sweep will
          reconcile (P2). The store may not exist yet → 0 + absent.
    """
    store = mem / "biomimetic" / "supersession_candidates.jsonl"
    if not store.exists():
        return 0.0, False
    try:
        from samia.runtime import contradiction
        rows = contradiction.list_supersession_candidates(mem, unresolved_only=True)
        return float(len(rows)), True
    except Exception as exc:  # fail-soft: module import / read failure → absent
        _log.debug("sleep_pressure: contradiction backlog read failed: %s", exc)
        return 0.0, False


def _read_near_dup_backlog(mem: Path) -> tuple[float, bool]:
    """Near-duplicate consolidation backlog.

    What: reads <mem>/.consolidation_candidates.json (produced by
          consolidation.surface) and counts its `candidates` list — the
          verified ~600-pair Tier-2 abstractive-consolidation backlog.
    Why:  the largest standing clutter signal; REM's Tier-2 subscriber (P2)
          drains it. Missing/corrupt file → 0 + absent.
    """
    p = mem / ".consolidation_candidates.json"
    if not p.exists():
        return 0.0, False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        cands = payload.get("candidates", []) if isinstance(payload, dict) else []
        return float(len(cands)), True
    except Exception as exc:
        _log.debug("sleep_pressure: near-dup backlog read failed: %s", exc)
        return 0.0, False


def _read_tier_overflow(mem: Path) -> tuple[float, bool]:
    """Hot+warm tier overflow.

    What: walks <mem>/nodes/*.md, reads each node's `tier`, and returns the
          COUNT of hot+warm nodes ABOVE the (_HOT_CAP + _WARM_CAP) budget.
          Below budget → 0 raw (no overflow); above → the excess.
    Why:  too many resident hot/warm nodes is the decay/prune backlog REM works
          down. Missing nodes/ dir → 0 + absent. Reads frontmatter directly so
          there is no dependency on a live tier-count index.
    """
    nodes_dir = mem / "nodes"
    if not nodes_dir.is_dir():
        return 0.0, False
    try:
        from samia.core import frontmatter as _fm
    except Exception:
        _fm = None
    hot_warm = 0
    try:
        for md in nodes_dir.glob("*.md"):
            tier = ""
            try:
                if _fm is not None:
                    parsed, _body = _fm.parse(md.read_text(encoding="utf-8"))
                    if parsed is not None:
                        fm, _order = parsed
                        tier = str(fm.get("tier", "")).lower()
            except Exception:
                tier = ""
            if tier in ("hot", "warm"):
                hot_warm += 1
    except OSError:
        return 0.0, False
    budget = _HOT_CAP + _WARM_CAP
    overflow = max(0.0, float(hot_warm) - budget)
    return overflow, True


def _read_coactivation_depth(mem: Path) -> tuple[float, bool]:
    """Coactivation-log depth.

    What: counts lines in biomimetic/coactivation_log.jsonl — the unprocessed
          Hebbian coactivation events awaiting the next consolidation drain.
    Why:  a deep log means many coactivations have accrued without being folded
          into edge weights; REM's hebbian_consolidate (P2) drains it. Absent
          log → 0 + absent.
    """
    p = mem / "biomimetic" / "coactivation_log.jsonl"
    if not p.exists():
        return 0.0, False
    try:
        with p.open("rb") as f:
            return float(sum(1 for line in f if line.strip())), True
    except OSError as exc:
        _log.debug("sleep_pressure: coactivation depth read failed: %s", exc)
        return 0.0, False


def _read_edges_unpromoted(mem: Path) -> tuple[float, bool]:
    """Edges grown without promotion.

    What: counts entries in biomimetic/edge_weights.json whose effective genuine
          count is BELOW the promotion bar (bio.HEBB_PROMOTION accounting):
          weight `w` < HEBB_PROMOTION. These are edges that have accrued but not
          yet crossed into the durable attractor set.
    Why:  many sub-threshold edges mean replay/dreaming (SWR) has reconciliation
          to do; REM's replay subscriber (P2) revisits them. Absent file → 0.
    """
    p = mem / "biomimetic" / "edge_weights.json"
    if not p.exists():
        return 0.0, False
    try:
        bar = _hebb_promotion_bar()
        weights = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(weights, dict):
            return 0.0, True
        unpromoted = sum(
            1 for v in weights.values()
            if isinstance(v, dict) and float(v.get("w", 0.0)) < bar
        )
        return float(unpromoted), True
    except Exception as exc:
        _log.debug("sleep_pressure: edges-unpromoted read failed: %s", exc)
        return 0.0, False


def _hebb_promotion_bar() -> float:
    """The promotion threshold (bio.HEBB_PROMOTION).

    What: returns the attractor/promotion bar. Why: grounds the
    edges-grown-without-promotion signal in the same constant the consolidation
    path uses; falls back to the documented 0.85 if bio is unimportable."""
    try:
        from samia.core import bio
        return float(bio.HEBB_PROMOTION)
    except Exception:
        return 0.85


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------

# name -> reader. Stable order so the score sum is deterministic.
_SIGNAL_READERS = {
    "offload_backlog": _read_offload_backlog,
    "contradiction_backlog": _read_contradiction_backlog,
    "near_dup_backlog": _read_near_dup_backlog,
    "tier_overflow": _read_tier_overflow,
    "coactivation_depth": _read_coactivation_depth,
    "edges_unpromoted": _read_edges_unpromoted,
}


def compute_pressure(mem: Path, threshold: float | None = None) -> dict[str, Any]:
    """Compute the composite sleep-pressure score + per-signal breakdown.

    What: reads each clutter backlog first-hand, normalizes it to [0,1] against
          its own cap, weights it, and SUMS into the composite `score`. Returns
          {signals, score, threshold, sleep_needed}. `signals[name]` carries
          {raw, cap, normalized, weight, contribution, present} — the
          operator-visible health gauge. An absent source contributes 0 with
          present=False (never raises).
    Why:  REM P1's metric (Q2 "start simple": a sum of normalized backlogs).
          should_sleep() consults `sleep_needed`; rem_status surfaces the whole
          breakdown so a stuck-high gauge is operator-visible.

    Args:
        mem: the memory root directory (the `<mem>` every reader resolves under).
        threshold: override the configured THRESHOLD (mainly for tests).
    """
    mem = Path(mem)
    thr = THRESHOLD if threshold is None else float(threshold)
    signals: dict[str, dict[str, Any]] = {}
    absent: list[str] = []
    score = 0.0

    for name, reader in _SIGNAL_READERS.items():
        try:
            raw, present = reader(mem)
        except Exception as exc:  # last-resort guard — a reader must never crash
            _log.debug("sleep_pressure: signal %s reader crashed: %s", name, exc)
            raw, present = 0.0, False
        cap = _CAPS[name]
        weight = _WEIGHTS[name]
        normalized = _clamp01(raw / cap) if cap > 0 else 0.0
        contribution = normalized * weight
        if not present:
            absent.append(name)
        signals[name] = {
            "raw": raw,
            "cap": cap,
            "normalized": round(normalized, 4),
            "weight": weight,
            "contribution": round(contribution, 4),
            "present": present,
        }
        score += contribution

    return {
        "signals": signals,
        "absent_sources": absent,
        "score": round(score, 4),
        "threshold": thr,
        "sleep_needed": score >= thr,
    }


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.sleep_pressure
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P1)
# Layer:      runtime (library helper, no daemon loop)
# Role:       composite sleep-pressure / health metric (Q2 simple normalized sum)
# Stability:  stable — P1 produce-only metric; pure reads, no mutation/thread/clock.
# ErrorModel: fail-soft throughout — every per-signal reader and the last-resort
#             per-signal guard degrade a missing/corrupt source to (0, present=False)
#             rather than raising; compute_pressure never crashes on an absent file.
# Depends:    json, logging, os, pathlib, typing (stdlib). samia.runtime.contradiction,
#             samia.core.bio, samia.core.frontmatter (all lazy / fail-soft).
# Exposes:    compute_pressure, THRESHOLD.
# Lines:      343
# --------------------------------------------------------------------------
