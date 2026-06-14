"""samia.core.context_extension.scheduling — SM-2 spaced repetition + compaction skip filter.

Layer 1 (Owns / Depends):
    Owns:    the SM-2 spaced-repetition arm — the per-node review update
             (sm2_review_update, the Wozniak ef/interval/count math), the due-list query
             (sm2_due_for_review), the seed/sweep caps (SM2_SEED_* / SM2_SWEEP_*), the
             usage→quality deriver (_sm2_quality_from_usage), and the scheduled sweep
             driver (sm2_sweep_tick, the dormant-loop caller wired by the scheduler);
             plus the compaction-aware skip filter (compaction_skip_filter).
    Depends: the package config leaf (the _dt date stamp, the _bio + _tq aliases, the
             _nodes_dir helper + the _read_full_fm reader).

Layer 2 (What / Why):
    What: the spaced-repetition scheduler that brings the ~2.9k-node corpus under SM-2
          over several daily ticks (incremental seed + capped review), deriving review
          quality from the node's tier/relevance hotness; plus the compaction filter
          that decides which transcript chunks are already-in-memory and can be skipped.
    Why:  these are the WRITE-side lifecycle ticks (review schedule + compaction novelty
          gate) — distinct from the read-path retrieval and the replay host. Grouping
          them keeps the SM-2 constants single-sourced and the scheduler's two entry
          points (sm2_sweep_tick + the per-node update) together.
"""

from __future__ import annotations

from pathlib import Path

# Shared leaf — the _dt date stamp, the _bio + _tq aliases, and the node dir + full-fm
# reader helpers.
from .config import (
    _dt,
    _bio,
    _tq,
    _nodes_dir,
    _read_full_fm,
)


def sm2_review_update(memory_dir: Path, node_name: str,
                      recalled: bool = True, quality: int = 4) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    p = nodes_dir / node_name
    if not p.suffix:
        p = p.with_suffix(".md")
    if not p.exists():
        return {"error": f"node not found: {p.name}"}
    fm_lines, body = _read_full_fm(p)
    today = _dt.date.today()

    try:
        ef = float(_tq.fm_get(fm_lines, "easiness_factor") or "2.5")
    except Exception:
        ef = 2.5
    try:
        interval = int(_tq.fm_get(fm_lines, "review_interval_days") or "1")
    except Exception:
        interval = 1
    try:
        rc = int(_tq.fm_get(fm_lines, "review_count") or "0")
    except Exception:
        rc = 0

    if not recalled or quality < 3:
        rc = 0
        interval = 1
    else:
        rc += 1
        if rc == 1:
            interval = 1
        elif rc == 2:
            interval = 6
        else:
            interval = max(1, int(round(interval * ef)))
        ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        ef = max(1.3, ef)

    next_review = (today + _dt.timedelta(days=interval)).isoformat()
    fm_lines = _tq.fm_set(fm_lines, "easiness_factor", f"{ef:.3f}")
    fm_lines = _tq.fm_set(fm_lines, "review_interval_days", str(interval))
    fm_lines = _tq.fm_set(fm_lines, "review_count", str(rc))
    fm_lines = _tq.fm_set(fm_lines, "next_review", next_review)
    _tq.write_node(p, fm_lines, body)
    return {"node": p.name, "easiness_factor": round(ef, 3),
            "review_interval_days": interval, "review_count": rc,
            "next_review": next_review}


def sm2_due_for_review(memory_dir: Path,
                       today: _dt.date | None = None) -> list[dict]:
    nodes_dir = _nodes_dir(memory_dir)
    today = today or _dt.date.today()
    out: list[dict] = []
    for p in nodes_dir.glob("*.md"):
        fm_lines, _ = _tq.read_node(p)
        nr = _tq.parse_date(_tq.fm_get(fm_lines, "next_review"))
        if not nr or nr > today:
            continue
        out.append({"node": p.name,
                    "next_review": nr.isoformat(),
                    "review_count": _tq.fm_get(fm_lines, "review_count") or "0"})
    return sorted(out, key=lambda r: r["next_review"])


# SM-2 frontmatter seed defaults. These mirror sm2_review_update's own
# fallbacks (ef 2.5 / interval 1 / count 0) so a freshly-seeded node behaves
# identically to one that had implicit defaults. Wozniak canonical constants
# (the rc==1/rc==2/ef-update math) live in sm2_review_update and are NOT
# duplicated here.
SM2_SEED_EASINESS = 2.5
SM2_SEED_INTERVAL_DAYS = 1
SM2_SEED_REVIEW_COUNT = 0
# Per-tick seed cap. The corpus is ~2.9k nodes; seeding all at once would
# rewrite every file in a single tick (mass churn + watcher storm). Cap the
# seed work so the corpus is brought under SM-2 over ~a few daily ticks.
SM2_SWEEP_SEED_CAP = 200
# Per-tick review cap. Once seeded, due nodes accrue gradually; bound the
# review work per tick for the same churn reason.
SM2_SWEEP_REVIEW_CAP = 200


def _sm2_quality_from_usage(fm_lines: list[str]) -> tuple[bool, int]:
    """Derive an SM-2 (recalled, quality) pair from an existing usage signal.

    Why: the spec forbids a hardcoded quality=4. The tier-decay subsystem
    already maintains a per-node ``relevance`` in [0,1] (samia.core.tier) as
    the canonical hotness signal — hot nodes are the ones recently recalled,
    cold/frozen ones have decayed from disuse. We reuse that rather than
    invent a parallel metric. Map relevance onto SM-2 quality via the same
    tier thresholds tier.tier_for uses (0.75 hot / 0.50 warm / 0.25 cold):

        relevance >= 0.75 (hot)   -> quality 5  (perfect recall)
        relevance >= 0.50 (warm)  -> quality 4  (good recall)
        relevance >= 0.25 (cold)  -> quality 3  (passing recall)
        relevance <  0.25 (frozen)-> quality 1, recalled=False (a lapse)

    Falls back to the categorical ``tier`` label when ``relevance`` is absent
    or malformed (the corpus carries one or the other on every node).
    """
    rel_raw = _tq.fm_get(fm_lines, "relevance")
    relevance: float | None
    try:
        relevance = float(rel_raw) if rel_raw is not None else None
    except (TypeError, ValueError):
        relevance = None
    if relevance is None:
        tier_to_rel = {"hot": 0.80, "warm": 0.60, "cold": 0.40, "frozen": 0.10}
        relevance = tier_to_rel.get(
            (_tq.fm_get(fm_lines, "tier") or "warm").strip().lower(), 0.60)
    if relevance >= 0.75:
        return True, 5
    if relevance >= 0.50:
        return True, 4
    if relevance >= 0.25:
        return True, 3
    return False, 1


def sm2_sweep_tick(memory_dir: Path,
                   today: _dt.date | None = None,
                   seed_cap: int = SM2_SWEEP_SEED_CAP,
                   review_cap: int = SM2_SWEEP_REVIEW_CAP) -> dict:
    """Scheduled SM-2 spaced-repetition sweep (the dormant-loop driver).

    What: in one pass it (1) INCREMENTALLY seeds SM-2 frontmatter onto nodes
    that lack ``next_review`` — capped at ``seed_cap`` per tick so the ~2.9k
    corpus is migrated over several daily ticks instead of one mass rewrite —
    then (2) iterates ``sm2_due_for_review`` and applies ``sm2_review_update``
    to each due node (capped at ``review_cap``), deriving the review quality
    from the node's usage signal via ``_sm2_quality_from_usage``.

    Why: ``sm2_review_update``/``sm2_due_for_review`` are correct but had no
    scheduled caller, so 0 of the corpus carried ``next_review`` and the loop
    never advanced. The scheduler now calls this daily (see
    samia.runtime.scheduler job ``sm2_review_sweep``). Seeding and reviewing
    in the same tick is intentional: a node seeded with next_review=today is
    immediately due, so it gets its first usage-derived review in the same
    sweep, bootstrapping the schedule from real hotness rather than a default.
    Canonical Wozniak constants are untouched — the seed values only fill the
    same defaults sm2_review_update already assumes.
    """
    nodes_dir = _nodes_dir(memory_dir)
    today = today or _dt.date.today()
    today_iso = today.isoformat()

    seeded: list[str] = []
    for p in sorted(nodes_dir.glob("*.md")):
        if len(seeded) >= seed_cap:
            break
        fm_lines, body = _tq.read_node(p)
        if _tq.fm_get(fm_lines, "next_review") is not None:
            continue
        fm_lines = _tq.fm_set(fm_lines, "easiness_factor",
                              f"{SM2_SEED_EASINESS:.3f}")
        fm_lines = _tq.fm_set(fm_lines, "review_interval_days",
                              str(SM2_SEED_INTERVAL_DAYS))
        fm_lines = _tq.fm_set(fm_lines, "review_count",
                              str(SM2_SEED_REVIEW_COUNT))
        fm_lines = _tq.fm_set(fm_lines, "next_review", today_iso)
        _tq.write_node(p, fm_lines, body)
        seeded.append(p.name)

    reviewed: list[dict] = []
    for due in sm2_due_for_review(memory_dir, today=today):
        if len(reviewed) >= review_cap:
            break
        p = nodes_dir / due["node"]
        fm_lines, _ = _tq.read_node(p)
        recalled, quality = _sm2_quality_from_usage(fm_lines)
        result = sm2_review_update(memory_dir, due["node"],
                                   recalled=recalled, quality=quality)
        result["quality"] = quality
        result["recalled"] = recalled
        reviewed.append(result)

    return {
        "seeded_count": len(seeded),
        "seeded": seeded[:10],
        "seed_cap_hit": len(seeded) >= seed_cap,
        "reviewed_count": len(reviewed),
        "reviewed": reviewed[:10],
        "review_cap_hit": len(reviewed) >= review_cap,
    }


# ---------------------------------------------------------------------------
# Compaction-aware skip filter
# ---------------------------------------------------------------------------


def compaction_skip_filter(memory_dir: Path, transcript_chunks: list[str],
                           threshold: float = 0.78) -> dict:
    keep: list[dict] = []
    skip: list[dict] = []
    for i, c in enumerate(transcript_chunks):
        d = _bio.pattern_separation_decision(memory_dir, c, threshold=threshold)
        if d["action"] == "merge_into":
            skip.append({"chunk_index": i, "covered_by": d["target"],
                         "score": d["score"]})
        else:
            keep.append({"chunk_index": i, "reason": "novel",
                         "best_score": d["score"]})
    return {"summarize": keep, "skip_already_in_memory": skip,
            "skip_ratio": (len(skip) / max(1, len(transcript_chunks)))}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.context_extension.scheduling
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      SM-2 spaced repetition (sweep driver wired into samia.runtime.scheduler)
#             + the compaction-aware skip filter.
#             + Phase-B modularization (carved from the monolith, ZERO behavior change).
# Layer:      core (pure library, no daemon dependency)
# Role:       the write-side lifecycle ticks — the spaced-repetition review schedule and
#             the compaction novelty gate.
# Stability:  stable — canonical Wozniak constants live in sm2_review_update and are not
#             duplicated; the seed defaults mirror its fallbacks exactly.
# ErrorModel: fail-soft — frontmatter field parses fall back to the SM-2 defaults; a
#             missing node returns {"error": ...}. Seed/review caps bound the per-tick
#             write churn.
# Depends:    .config (_dt, _bio, _tq, _nodes_dir, _read_full_fm).
# Exposes:    sm2_review_update, sm2_due_for_review, sm2_sweep_tick,
#             compaction_skip_filter + the SM2_* caps (public).
# Lines:      261
# --------------------------------------------------------------------------
