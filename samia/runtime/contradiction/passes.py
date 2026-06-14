"""samia.runtime.contradiction.passes — passive REM sweep + memory-guard integration.

Layer 1 (Owns / Depends):
    Owns:    the PASSIVE supersession sweep (FEAT-2026-06-07 P3c) and its helpers
             (_list_node_ids, _node_field, _pick_superseded,
             _salience_guards_supersede, passive_sweep), the REM due-signal
             (passive_has_work), and the AUD60 memory-guard integration point
             (check_contradiction).
    Depends: the package config leaf (constants + state + the scoping helpers +
             _node_type + _list helpers + _log/json), the detection / judge / store
             arms (reached THROUGH the package facade for the patch-seam targets and
             the facade-rebound flags), and — lazily/function-locally —
             samia.runtime.rem_cycle, samia.core.{temporal,ia,integrity,bio}.

Layer 2 (What / Why):
    What: passive_sweep is the offline (REM-idle) exhaustive reconciler: walk a
          cursor-tracked slice of the index, find supersession candidates, judge
          them, AUTO-supersede the loser via the RESTORABLE path (unless the P3
          salience guard fires), and record the weaker/unjudged hits.
          check_contradiction is the online pre-write hook memory_guard calls.
    Why:  the bounded online locus cannot be the global reconciler; the passive arm
          is. It is DOUBLE-gated (a REM subscriber AND is_enabled()), so it is inert
          by default, and every auto-supersede is reversible (archive + restore).

PATCH SEAMS (exemplar rule): passive_sweep + check_contradiction call functions that
    are mock.patch.object(contradiction, ...) targets AND owned by SIBLING submodules
    (is_enabled, find_supersession_candidates, judge_contradictions,
    list_supersession_candidates, find_contradiction_candidates) and read flags tests
    rebind on the facade (_ENABLED, _JUDGE_ENABLED), so every such call/read goes
    THROUGH the package facade so a package-level patch/rebind is honored. The
    record_supersession_candidate / mark_supersession_confirmed writes are reached
    through the facade too for parity. _log is config._log (the shared object whose
    .warning the live-isolation test patches in place).
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

# Shared leaf — the passive budget + cursor key, the node-type scoping helpers, the
# package logger + json, and the datetime/timezone used for the valid_to close stamp.
from . import config as _cfg
from .config import datetime, timezone


def _list_node_ids(memory_dir: Path) -> list[str]:
    """Sorted node ids (file stems) of the whole nodes/ index.

    What: the ordered universe the passive cursor walks. Sorted so the cursor
          index is stable across calls (a node added mid-pass shifts the tail,
          which the wrap-at-end reset tolerates).
    Why:  the passive sweep spans the WHOLE index (scope=None); it needs a
          deterministic order to advance a numeric cursor and detect wrap.
    """
    nodes = memory_dir / "nodes"
    if not nodes.is_dir():
        return []
    try:
        return sorted(p.stem for p in nodes.glob("*.md"))
    except OSError:
        return []


def _node_field(memory_dir: Path, node_id: str, key: str) -> Any:
    """Read one frontmatter field of a node (None if missing/unreadable).

    What: parse nodes/<id>.md and return frontmatter[key].
    Why:  the loser-selection rule needs valid_from (age) and confidence to pick
          which of a contradicting pair is superseded; reading on demand avoids
          loading every node up front.
    """
    fname = node_id if node_id.endswith(".md") else f"{node_id}.md"
    p = memory_dir / "nodes" / fname
    if not p.exists():
        return None
    try:
        from samia.core import frontmatter as _fm
        parsed, _ = _fm.parse(p.read_text(encoding="utf-8"))
        if parsed is not None:
            return parsed[0].get(key)
    except Exception:
        return None
    return None


def _pick_superseded(memory_dir: Path, a_id: str, b_id: str) -> tuple[str, str]:
    """Pick (loser, winner) for a confirmed contradiction by the proposal's rule.

    What: the loser is the OLDER (earlier valid_from) / LOWER-confidence claim;
          the survivor is the newer / higher-confidence one. Ties fall back to a
          stable id order so the choice is deterministic.
    Why:  the proposal directs auto-supersede to retire "the older/lower-
          confidence claim" — a newer, contradictory belief supersedes the stale
          one. Always reversible (restore_node), so the conservative tie-break is
          safe.
    """
    conf_a = _node_field(memory_dir, a_id, "confidence")
    conf_b = _node_field(memory_dir, b_id, "confidence")
    try:
        if conf_a is not None and conf_b is not None and float(conf_a) != float(conf_b):
            return (a_id, b_id) if float(conf_a) < float(conf_b) else (b_id, a_id)
    except (TypeError, ValueError):
        pass
    vf_a = _node_field(memory_dir, a_id, "valid_from")
    vf_b = _node_field(memory_dir, b_id, "valid_from")
    if isinstance(vf_a, str) and isinstance(vf_b, str) and vf_a != vf_b:
        return (a_id, b_id) if vf_a < vf_b else (b_id, a_id)
    # Deterministic fallback: the lexically-greater id is treated as "newer".
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


def _salience_guards_supersede(memory_dir: Path, loser: str) -> bool:
    """True iff the salience guard protects the loser from auto-supersede (P3).

    What: consult bio.salience_merge_guard on the loser with is_duplicate=False.
          Returns True when the loser is a DISTINCT high-salience memory the guard
          protects — the caller then SURFACES the supersession for operator review
          instead of auto-removing it. False when the guard is unavailable (bio
          without salience_merge_guard — the online/passive paths ship before
          Tier-1's salience field lands) or the loser is not high-salience.
    Why:  D6 effect (iii) / Q5a — the salience merge/supersede guard is CONSUMED
          by the contradiction detector (here) AND the merge consumer. A
          contradiction pair is distinct (X vs not-X), so is_duplicate stays
          False; an exact duplicate is not the guard's target. Wired behind a
          hasattr guard so the detector runs fully before the salience field
          exists and activates with no re-sequence once Tier-1 Phase 5 lands.
          Pure read; mutates nothing.
    """
    try:
        from samia.core import bio as _bio
    except Exception:
        return False
    guard = getattr(_bio, "salience_merge_guard", None)
    if guard is None:
        return False
    try:
        return bool(guard(Path(memory_dir), loser, is_duplicate=False))
    except Exception:
        return False


def passive_sweep(memory_dir: Path,
                  budget: Optional[int] = None,
                  cursor: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """PASSIVE supersession sweep over a bounded slice of the WHOLE index (P3c).

    What: the offline (REM-idle) arm of the P3 detector. Walks a cursor-tracked
          slice of nodes/ (<= ``budget`` nodes per call), and for each node in
          the slice:
            1. runs find_supersession_candidates(scope_nodes=None) — cosine over
               the WHOLE index (the passive scope) + the jaccard pre-filter;
            2. runs the LLM judge (judge_contradictions) to CONFIRM a true
               contradiction (A asserts X vs B asserts not-X); the judge is
               affordable here because REM is idle-budgeted;
            3. on a judge-CONFIRMED contradiction, AUTO-supersedes the loser
               (older / lower-confidence, per _pick_superseded) via the
               RESTORABLE path: set valid_to on the loser + ia.forget_node(
               reason="supersede") (full archive -> restore_node un-forgets) —
               UNLESS the P3 SALIENCE GUARD fires (the loser is a DISTINCT high-
               salience memory), in which case the supersession is SURFACED for
               operator review (status="surfaced-salience") instead of auto-
               removed (D6 effect iii / Q5a);
            4. records weaker / unjudged / judge-uncertain hits to the unified
               candidate store with mode="passive" (NOT deleted) for review.
          The cursor (an int index over the sorted node list) advances by the
          slice size and WRAPS/resets at the end so a full pass completes across
          many REM cycles. Checkpointed via the rem_cycle cursor helpers.
    Why:  Q3 + the Q4 override. The passive mode is the exhaustive global
          reconciler the bounded online locus cannot be; it is gated behind REM
          (a subscriber) AND behind ASTHENOS_CONTRADICTION_ENABLED so it is inert
          by default. Every auto-supersede is reversible — the auto action carries
          no irreversible risk (full archive + restore_node + self-healing).

    Gating (DOUBLE, both inert by default):
        (a) it is a REM subscriber, so the driver only calls it inside REM; and
        (b) it no-ops unless is_enabled() (ASTHENOS_CONTRADICTION_ENABLED) — the
            same posture as the P3b online path.

    Args:
        memory_dir: the memory root.
        budget: max nodes to process this call (default _PASSIVE_BUDGET).
        cursor: an explicit cursor override (tests); else read from rem_cursors.

    Returns:
        {work_remaining, made_progress, judged, superseded, recorded, guarded,
         cursor, processed, total} — work_remaining is True while the cursor has
        not yet wrapped a full pass OR candidates are pending; guarded counts the
        judge-confirmed contradictions the P3 salience guard surfaced for review
        instead of auto-superseding.
    """
    # Reach the package facade for the patch-seam targets the sweep drives:
    # is_enabled / find_supersession_candidates / judge_contradictions /
    # record_supersession_candidate / mark_supersession_confirmed /
    # list_supersession_candidates are all mock.patch.object(contradiction, ...)
    # targets owned by sibling submodules; a package-level patch must rebind what
    # this sweep actually calls.
    from samia.runtime import contradiction as _pkg

    out: dict[str, Any] = {
        "work_remaining": False, "made_progress": False,
        "judged": 0, "superseded": 0, "recorded": 0, "guarded": 0,
        "processed": 0, "total": 0,
    }

    # GATE (b): inert unless the operator enabled contradiction detection.
    if not _pkg.is_enabled():
        out["enabled"] = False
        return out
    out["enabled"] = True

    mem = Path(memory_dir)
    node_ids = _list_node_ids(mem)
    total = len(node_ids)
    out["total"] = total
    if total == 0:
        return out

    # Cursor: an int index over the sorted node list. Read from the REM cursor
    # store unless an explicit override is given (tests / direct calls).
    if cursor is None:
        try:
            from samia.runtime import rem_cycle as _rem
            cursor = _rem.read_cursor(mem, _cfg._PASSIVE_CURSOR_KEY)
        except Exception:
            cursor = {}
    start = int((cursor or {}).get("index", 0))
    if start < 0 or start >= total:
        start = 0  # wrap / out-of-range reset

    cap = _cfg._PASSIVE_BUDGET if budget is None else int(budget)
    end = min(start + max(0, cap), total)
    slice_ids = node_ids[start:end]
    out["processed"] = len(slice_ids)

    today = datetime.now(tz=timezone.utc).date().isoformat()
    judged = superseded = recorded = guarded = 0

    # What: count finder/judge failures across the slice instead of logging one
    #   warning per node.
    # Why: a systemic finder fault (e.g. the old entries[i] KeyError that
    #   stringified to "34") otherwise produced one warning PER node every REM
    #   cycle -- a churning log storm across thousands of nodes. We now summarize
    #   once at debug level with a representative example, so a transient/unprocessable
    #   node is skipped cleanly without per-node noise.
    finder_fail = 0
    finder_fail_example: tuple[str, str] | None = None
    judge_fail = 0
    judge_fail_example: tuple[str, str] | None = None

    skipped_excluded = 0
    for node_id in slice_ids:
        # already-purged within this slice (e.g. it lost an earlier pair) -> skip.
        if not (mem / "nodes" / f"{node_id}.md").exists():
            continue
        # TYPE-SCOPING: the node-being-checked is episodic/experiential
        # (session_offload / bug) -> SKIP it entirely. Don't even spend cosine
        # work on an excluded node; it cannot be a contradictable content claim.
        if _cfg.is_excluded_node(mem, node_id):
            skipped_excluded += 1
            continue
        text = _pkg._node_text_for_id(mem, node_id)
        if not text:
            continue
        try:
            # Per-population bar, incoming side (TUNE-2026-06-10 (2)): when the
            # node BEING SWEPT is itself a semantic atom, the whole scan runs at
            # the higher bar (its template kin saturate the recall-first band).
            _thr = (_cfg._SEMANTIC_PAIR_THRESHOLD
                    if _cfg._node_type(mem, str(node_id)) == "semantic" else None)
            cands = _pkg.find_supersession_candidates(text, scope_nodes=None,
                                                      memory_dir=mem, threshold=_thr)
        except Exception as exc:  # fail-soft: a finder error never aborts the sweep.
            finder_fail += 1
            if finder_fail_example is None:
                finder_fail_example = (node_id, repr(exc))
            continue
        # Drop self + already-gone candidates.
        cands = [c for c in cands
                 if str(c["node_id"]) not in (node_id, f"{node_id}.md")
                 and (mem / "nodes" /
                      (str(c["node_id"]) if str(c["node_id"]).endswith(".md")
                       else f"{c['node_id']}.md")).exists()]
        if not cands:
            continue

        # LLM judge: confirm a TRUE contradiction (X vs not-X) before acting.
        try:
            verdicts = _pkg.judge_contradictions(text, cands)
        except Exception as exc:  # fail-soft: judge error -> record, never delete.
            judge_fail += 1
            if judge_fail_example is None:
                judge_fail_example = (node_id, repr(exc))
            verdicts = []
        if verdicts:
            judged += 1

        confirmed_ids = {str(v.get("existing_claim_id", "")).rstrip(".md")
                         for v in verdicts}
        for c in cands:
            cand_id = str(c["node_id"])
            cand_stem = cand_id[:-3] if cand_id.endswith(".md") else cand_id
            if cand_stem in confirmed_ids and cand_stem:
                # Judge-CONFIRMED contradiction -> auto-supersede the LOSER via
                # the RESTORABLE path (set valid_to on loser + forget archive).
                loser, winner = _pick_superseded(mem, node_id, cand_stem)
                if not (mem / "nodes" / f"{loser}.md").exists():
                    continue
                jv = next((v for v in verdicts
                           if str(v.get("existing_claim_id", "")).rstrip(".md")
                           == cand_stem), None)
                # P3 SALIENCE GUARD (D6 effect iii / Q5a): do NOT auto-supersede a
                # DISTINCT high-salience loser — surface it for operator review
                # instead (record a guarded candidate, never auto-remove). The
                # guard is about distinct high-salience claims; the contradiction
                # pair is distinct by construction (X vs not-X), so is_duplicate
                # stays False.
                if _salience_guards_supersede(mem, loser):
                    _pkg.record_supersession_candidate(
                        mem, loser, winner, cosine=float(c.get("score", 0.0)),
                        jaccard=c.get("jaccard"), mode="passive", judge=jv,
                        status="surfaced-salience")
                    guarded += 1
                    continue
                try:
                    from samia.core import temporal as _temporal
                    _temporal.set_valid(mem, f"{loser}.md", None, today)
                except Exception:
                    pass  # best-effort close; the archive preserves the body.
                try:
                    from samia.core import ia as _ia
                    _ia.forget_node(mem, f"{loser}.md", reason="supersede",
                                    superseded_by=f"{winner}.md")
                except Exception as exc:
                    _cfg._log.warning("contradiction: passive supersede failed %s: %s",
                                      loser, exc)
                    continue
                _pkg.record_supersession_candidate(
                    mem, loser, winner, cosine=float(c.get("score", 0.0)),
                    jaccard=c.get("jaccard"), mode="passive", judge=jv,
                    status="confirmed")
                _pkg.mark_supersession_confirmed(mem, loser, winner)
                superseded += 1
                # FEAT-2026-06-07 granular-recall-repaired-decay P2 — RECONCILIATION
                # repair: the surviving WINNER was just READ + reconciled, so PARTIALLY
                # heal its integrity (anchor-first, strength < 1.0). Reconciling a memory
                # heals what it touches (Q3a, partial). Additive + fail-soft — a repair
                # error never aborts the sweep; gated by ASTHENOS_CONTRADICTION_ENABLED
                # (we are already inside the is_enabled() guard, so inert by default).
                try:
                    from samia.core import integrity as _integrity
                    _integrity.partial_repair(mem, f"{winner}.md",
                                              trigger="reconciliation")
                except Exception:
                    pass
            else:
                # Weaker / unjudged / judge-uncertain -> RECORD (not deleted).
                _pkg.record_supersession_candidate(
                    mem, cand_stem, node_id, cosine=float(c.get("score", 0.0)),
                    jaccard=c.get("jaccard"), mode="passive")
                recorded += 1

    # What: emit ONE summarized warning per sweep for skipped (un-processable)
    #   nodes, instead of one per node.
    # Why: avoid the per-node "passive finder failed for %s: 34" log storm; the
    #   sweep already fails soft (skip + continue). A single count + example keeps
    #   forensics without churn. Counts are surfaced in the return dict too.
    out["finder_failures"] = finder_fail
    out["judge_failures"] = judge_fail
    if finder_fail:
        ex_id, ex_msg = finder_fail_example or ("?", "?")
        _cfg._log.warning(
            "contradiction: passive finder skipped %d/%d unprocessable node(s) "
            "(e.g. %s: %s)", finder_fail, len(slice_ids), ex_id, ex_msg)
    if judge_fail:
        ex_id, ex_msg = judge_fail_example or ("?", "?")
        _cfg._log.warning(
            "contradiction: passive judge failed for %d node(s) (e.g. %s: %s)",
            judge_fail, ex_id, ex_msg)

    out["judged"] = judged
    out["superseded"] = superseded
    out["recorded"] = recorded
    out["guarded"] = guarded
    out["skipped_excluded"] = skipped_excluded
    out["made_progress"] = bool(slice_ids)

    # Advance the cursor; WRAP/reset at the end of a full pass.
    new_index = end if end < total else 0
    wrapped = end >= total
    cursor_out = {"index": new_index, "total": total, "wrapped": wrapped,
                  "remaining": (not wrapped)}
    out["cursor"] = cursor_out

    # work_remaining (G2-2026-06-11, MACHINE-DRAINABLE ONLY): True ONLY while THIS
    # subscriber can still drain work in a future REM cycle WITHOUT operator action —
    # i.e. the cursor has not yet wrapped a full pass over the index. Un-resolved
    # supersession candidates pending operator/judge confirmation are OPERATOR-GATED:
    # no machine cycle can clear them, so they MUST NOT keep REM awake (they used to
    # OR into work_remaining here, which made every wake report work_remaining=true and
    # never let evaluate() reach REST). The pending count is still surfaced as telemetry
    # (operator_gated_pending) for observability, but it does not gate the sleep cycle.
    pending = False
    try:
        # list_supersession_candidates is a patch seam (test_rem_subscribers patches
        # it here); reach it through the facade so the patched stub is consulted.
        pending = bool(_pkg.list_supersession_candidates(mem, unresolved_only=True))
    except Exception:
        pending = False
    out["operator_gated_pending"] = pending
    out["work_remaining"] = (not wrapped)

    # Checkpoint the cursor (unless an explicit cursor override was supplied,
    # in which case the caller owns persistence).
    if cursor is not None and cursor.get("__no_persist__"):
        pass
    else:
        try:
            from samia.runtime import rem_cycle as _rem
            _rem.write_cursor(mem, _cfg._PASSIVE_CURSOR_KEY, cursor_out)
        except Exception as exc:
            _cfg._log.debug("contradiction: passive cursor checkpoint failed: %s", exc)
    return out


def passive_has_work(memory_dir: Path) -> bool:
    """due_condition for the REM subscriber: there are nodes to sweep AND enabled.

    What: True iff contradiction detection is enabled AND the nodes/ index is
          non-empty. The wrap-at-end cursor means a finished pass simply restarts
          on the next cycle; "has nodes" is the right standing due-signal.
    Why:  Q3 — the passive sweep is due whenever there is an index to reconcile
          and the operator has enabled the feature; otherwise it never fires
          (double-gate: REM + is_enabled()).
    """
    # is_enabled is a facade-rebound seam; reach it through the package facade.
    from samia.runtime import contradiction as _pkg
    if not _pkg.is_enabled():
        return False
    nodes = Path(memory_dir) / "nodes"
    if not nodes.is_dir():
        return False
    try:
        return any(nodes.glob("*.md"))
    except OSError:
        return False


def check_contradiction(
    payload: dict[str, Any],
    memory_dir: Optional[Path] = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Full contradiction check pipeline for a memory write payload.

    What: orchestrates Phase 1 (embedding candidates) and Phase 2 (LLM judge)
          and returns both the reason strings for memory_guard and the
          detailed contradiction metadata for the pending queue.
    Why:  single integration point that memory_guard calls during
          _validate_write. Returns data in the format memory_guard expects
          (reason strings) plus extended metadata for the MemGuardPanel
          (contradiction_with field).

    Parameters
    ----------
    payload : dict
        The memory write payload.
    memory_dir : Path or None
        Memory directory for vector index access.

    Returns
    -------
    (reasons, contradiction_metadata) tuple.
        reasons: list of strings for memory_guard's reason field.
        contradiction_metadata: list of dicts with node_id, title, score,
            plus optional judge fields (explanation, confidence).
    """
    # Reach the facade for the facade-rebound flags (_ENABLED / _JUDGE_ENABLED, both
    # rebound by test_contradiction) and the patch-seam targets the test patches
    # (find_contradiction_candidates, judge_contradictions) + the single-owned
    # _MEMORY_DIR state.
    from samia.runtime import contradiction as _pkg

    if not _pkg._ENABLED:
        return [], []

    text = _cfg.json.dumps(payload, default=str)
    mem = memory_dir or _pkg._MEMORY_DIR

    # Phase 1: embedding candidate finder.
    candidates = _pkg.find_contradiction_candidates(text, memory_dir=mem)
    if not candidates:
        return [], []

    reasons: list[str] = []
    metadata: list[dict[str, Any]] = []

    # Phase 2: optional LLM judge.
    if _pkg._JUDGE_ENABLED:
        judge_results = _pkg.judge_contradictions(text, candidates)
        if judge_results:
            for jr in judge_results:
                cid = jr.get("existing_claim_id", "?")
                conf = jr.get("confidence", 0)
                reasons.append(
                    f"contradiction_judge:id={cid}:conf={conf:.2f}"
                )
                metadata.append({
                    "node_id": cid,
                    "explanation": jr.get("explanation", ""),
                    "confidence": conf,
                    "source": "llm_judge",
                })
            return reasons, metadata

    # What: if no judge or judge found nothing, report embedding candidates.
    # Why: even without the LLM judge, high-similarity candidates are worth
    #   flagging for operator review.
    for c in candidates:
        reasons.append(
            f"contradiction_embedding:node={c['node_id']}:sim={c['score']:.3f}"
        )
        metadata.append({
            "node_id": c["node_id"],
            "title": c.get("title", ""),
            "score": c["score"],
            "source": "embedding_similarity",
        })

    return reasons, metadata


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.contradiction.passes
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.runtime.contradiction monolith
#             during modularization (the passive REM sweep + memory-guard arm).
# Layer:      runtime (library helper, no daemon loop)
# Role:       the PASSIVE REM-subscriber sweep (FEAT-2026-06-07 P3c: incremental,
#             cursor-tracked, whole-index cosine + LLM judge -> auto-supersede the
#             loser via the RESTORABLE path, double-gated REM + is_enabled()), the
#             REM due-signal passive_has_work, and the AUD60 memory-guard integration
#             check_contradiction.
# Stability:  v0.4 — all phases wired, default-off via env.
#             G2-2026-06-11: passive_sweep.work_remaining reflects ONLY the machine-
#             drainable cursor (not wrapped); operator-gated pending candidates are
#             surfaced as the operator_gated_pending telemetry key, not OR'd into
#             work_remaining (lets REM evaluate() reach REST).
# ErrorModel: fail-soft throughout — a finder/judge/supersede/repair error never
#             aborts the sweep (counted + summarized, never per-node spam). Every
#             auto-supersede is reversible (valid_to close + ia.forget archive +
#             restore_node). check_contradiction is fail-open (no candidates -> []).
# Depends:    .config (constants + state + scoping + _log/json); the detection/judge/
#             store arms + the facade-rebound flags reached THROUGH the package facade
#             (patch seams); samia.runtime.rem_cycle + samia.core.{temporal,ia,
#             integrity,bio,frontmatter} (lazy, function-local).
# Exposes:    passive_sweep, passive_has_work, check_contradiction (public);
#             _list_node_ids, _node_field, _pick_superseded,
#             _salience_guards_supersede (internal).
# Lines:      559
# --------------------------------------------------------------------------
