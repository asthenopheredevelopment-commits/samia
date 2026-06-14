"""samia.core.merge_consumer.drain — the cursor-tracked batch drain orchestrator.

Layer 1 (Owns / Depends):
    Owns:    drain (the batch entry the REM tier2_merge subscriber calls — walk a
             budget-sized slice of the candidate backlog from a persisted cursor,
             classify each pair, dispatch it, remove the dispatched rows so the
             backlog shrinks) and _read_threshold (preserve the surfacer's recorded
             threshold when rewriting the candidate file).
    Depends: .config (is_enabled + _con + _CANDIDATE_FILE), .candidates
             (load_candidates, _resolve_pair, classify_pair), .winner (merge_dup),
             .abstraction (_salience_guards_pair, _record_guarded, _record_abstract,
             _enqueue_abstract_pair).

Layer 2 (What / Why):
    What: the top of the package's dependency DAG — it wires the READ/classify, the
          dup ACT, the abstract RECORD, the salience GUARD, and the fact-extract
          ENQUEUE into one pass and then shrinks the backlog. Stale pairs (a node
          already gone) are dropped as drained; the cursor does not advance past
          removed rows so a small budget still makes forward progress every cycle.
    Why:  Q3a — draining is what lets REM reach REST (work_remaining is True iff
          candidates remain after this slice). Q5a — the whole entry is a no-op
          unless is_enabled(), inert by default (the contradiction passive-sweep
          posture: produce nothing on import, nothing without the operator flag).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import is_enabled, _con, _CANDIDATE_FILE
from .candidates import load_candidates, _resolve_pair, classify_pair
from .winner import merge_dup
from .abstraction import (
    _salience_guards_pair,
    _record_guarded,
    _record_abstract,
    _enqueue_abstract_pair,
)


def drain(memory_dir: Path, budget: int = 20, cursor: int = 0,
          db_dir: Optional[str] = None) -> dict:
    """Drain a slice of the candidate backlog (the cursor-tracked batch entry).

    What: walk up to ``budget`` candidates from index ``cursor``; classify each;
          AUTO-merge the "dup" ones (pick-winner + restorable supersede +
          provenance edge), record the "abstract" ones for P2; remove every
          DISPATCHED pair (merged or recorded) from .consolidation_candidates.json
          so the backlog SHRINKS (Q3a). Stale pairs (a node already gone) are
          dropped as drained. Returns {merged, recorded, drained, skipped,
          processed, cursor, remaining, work_remaining}.
    Why:  Q3a — draining is what lets REM reach REST: work_remaining is True iff
          candidates still remain after this slice. The cursor (caller-persisted
          under rem_cursors.json key "tier2_merge") guarantees forward progress
          across REM cycles even under a small budget.

    Gated: a no-op (nothing merged) unless is_enabled() — Q5a, inert by default.
    """
    if not is_enabled():
        cands = load_candidates(Path(memory_dir))
        return {
            "fired": False, "refused": "not_enabled",
            "merged": 0, "recorded": 0, "drained": 0, "skipped": 0,
            "processed": 0, "cursor": cursor, "remaining": len(cands),
            "work_remaining": len(cands) > 0,
        }

    cands = load_candidates(Path(memory_dir))
    n = len(cands)
    start = max(0, int(cursor))
    end = min(n, start + max(0, int(budget)))

    merged = recorded = skipped = 0
    dispatched_indices: set[int] = set()

    for i in range(start, end):
        cand = cands[i]
        pair = _resolve_pair(Path(memory_dir), cand)
        if pair is None:
            # Stale (a node already merged/forgotten) — drop it as drained.
            dispatched_indices.add(i)
            skipped += 1
            continue
        a_id, b_id = pair
        kind = classify_pair(
            Path(memory_dir), a_id, b_id,
            candidate_similarity=cand.get("similarity"))
        if kind == "dup":
            # P3 salience guard: a TRUE duplicate is exempt (is_duplicate=True ->
            # guard always False), so the dup pick-winner merge is UNCHANGED — a
            # duplicate carries the same content, merging it loses nothing.
            merge_dup(Path(memory_dir), a_id, b_id, db_dir=db_dir)
            merged += 1
            dispatched_indices.add(i)
        else:
            # P3 salience guard: a DISTINCT pair where either source is a high-
            # salience memory is SURFACED for operator review (status="guarded"),
            # NOT auto-recorded for P2 abstraction — do not abstract away an
            # important one-shot memory. The pair is still drained (removed from
            # the backlog) but recorded as guarded, not pending.
            guarded = _salience_guards_pair(Path(memory_dir), a_id, b_id,
                                            is_duplicate=False)
            if guarded is not None:
                _record_guarded(Path(memory_dir), a_id, b_id,
                                cand.get("similarity"), guarded)
            else:
                _record_abstract(Path(memory_dir), a_id, b_id,
                                 cand.get("similarity"))
                # FEAT-2026-06-10-memory-fact-extract-producer-v01 P1 — the
                # 'abstract' distinct-but-overlapping pair is the SECOND producer
                # feed (Q1d). Enqueue BOTH node texts as ONE extraction record
                # (concatenated with a separator) so the extractor distils atomic
                # facts across the pair.
                # What: append one {text, source:"a.md+b.md",
                #   enqueued_by:"merge_abstract"} record to the fact-extract queue.
                # Why: distinct-but-overlapping pairs are consolidation-shaped and
                #   exactly where cross-source gist lives; extraction is ADDITIVE
                #   (this is NOT the gated abstractive MERGE — P2's separate
                #   machinery — and deletes nothing). Gated on
                #   fact_extract_enabled() so flag-off writes nothing; fail-OPEN so
                #   a queue error never blocks the drain.
                _enqueue_abstract_pair(Path(memory_dir), a_id, b_id)
            recorded += 1
            dispatched_indices.add(i)

    # Remove dispatched pairs from the candidate file so the backlog shrinks.
    drained = len(dispatched_indices)
    if drained:
        kept = [c for j, c in enumerate(cands) if j not in dispatched_indices]
        _con.surface(
            Path(memory_dir), kept,
            _read_threshold(Path(memory_dir)))
        remaining = len(kept)
        # The cursor does not advance past removed entries: indices shift left by
        # the count removed at/below the cursor, so resume from the same logical
        # position (start) which now points at the next un-dispatched pair.
        new_cursor = start
    else:
        remaining = n
        new_cursor = end

    return {
        "fired": True,
        "merged": merged,
        "recorded": recorded,
        "drained": drained,
        "skipped": skipped,
        "processed": end - start,
        "cursor": new_cursor,
        "remaining": remaining,
        "work_remaining": remaining > 0,
    }


def _read_threshold(memory_dir: Path) -> float:
    """Preserve the surfacer's recorded threshold when rewriting the file."""
    p = Path(memory_dir) / _CANDIDATE_FILE
    if not p.exists():
        return _con.DEFAULT_THRESHOLD
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return float(payload.get("threshold", _con.DEFAULT_THRESHOLD))
    except (OSError, ValueError, TypeError):
        return _con.DEFAULT_THRESHOLD


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.merge_consumer.drain
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.merge_consumer monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       the top of the package DAG — the cursor-tracked batch entry the REM
#             tier2_merge subscriber calls. Walks a budget slice from a persisted
#             cursor, classifies + dispatches each pair (dup ACT / abstract RECORD
#             / salience GUARD / fact-extract ENQUEUE), removes the dispatched rows
#             so the backlog shrinks and REM can reach REST.
# Stability:  stable — the carve preserved the gating, the cursor non-advance over
#             removed rows, the dispatch branch order, and the returned dict shape.
# ErrorModel: drain is a no-op (fired=False) unless is_enabled(); a stale pair is
#             dropped as drained; the dup merge / abstract record / enqueue each
#             carry their own fail-soft posture (see winner.py / abstraction.py).
# Depends:    json, pathlib, typing (stdlib). .config (is_enabled, _con,
#             _CANDIDATE_FILE), .candidates (load_candidates, _resolve_pair,
#             classify_pair), .winner (merge_dup), .abstraction (guard/record/
#             enqueue).
# Exposes:    drain, _read_threshold.
# Lines:      189
# --------------------------------------------------------------------------
