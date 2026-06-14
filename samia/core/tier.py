"""samia.core.tier — relevance decay + tier classification.

Layer 1 (Owns / Depends):
    Owns:    tier_for, step_relevance, decay_pass, decay_tick (+ the
             TIER_THRESHOLDS / DECAY_RATE / NEUTRAL constants).
    Depends: stdlib (datetime, pathlib). samia.core.frontmatter; samia.core.ia
             (optional, auto_freeze); samia.core.timestamp.
Layer 2 (What / Why):
    What: relevance decay + tier classification — step_relevance is the pure
          math; decay_pass / decay_tick walk nodes/ and apply it on the daemon's
          continuous schedule.
    Why:  memory must forget continuously, so tiering + relevance decay is the
          forgetting axis (distinct from the content-integrity decay axis).

Carved from memory_session_boot.py. Holds the two pieces that the future
runtime daemon's `tier_decay_tick` job (see design doc §1.3) calls on its 6h
schedule, and that hooks/scripts call ad-hoc.

Public API:
  TIER_THRESHOLDS, DECAY_RATE, NEUTRAL — constants
  tier_for(relevance)                  → tier name
  step_relevance(old_rel, touched)     → (new_rel, reason)
  decay_pass(nodes_dir, dry, today)    → list[Transition]

`step_relevance` is the pure math; `decay_pass` walks `nodes/`, applies it,
and writes back unless `dry=True`.

Acceptance: byte-identical to pre-refactor memory_session_boot.decay() output
on the same node corpus.
"""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Optional

from . import frontmatter as _fm

# ── Constants (operator-confirmed 2026-04-30, see docs/memory_tier_decay_design_2026-04-30.md) ──
DECAY_RATE_BY_GRADE = {
    "enriched": 0.001,   # U-235 analog: ~permanent
    "fertile":  0.01,    # U-238: slow but eventual
    "natural":  0.05,    # default
    "depleted": 0.10,    # fast decay
    "waste":    0.20,    # very fast sink
}
DEFAULT_GRADE = "natural"
DECAY_RATE = DECAY_RATE_BY_GRADE[DEFAULT_GRADE]   # legacy alias
NEUTRAL = 0.5
WARM_FRESHNESS_DAYS = 7
TIER_THRESHOLDS = [("hot", 0.75), ("warm", 0.50), ("cold", 0.25), ("frozen", 0.0)]
TIER_DECAY_INTERVAL_S = 6 * 3600   # 6h cooldown for the idle-pulse subscriber

# ── Salience-aware decay (FEAT-2026-06-07 Tier-1 P5, D6 effect ii) ──────────
# SALIENCE_DECAY_DAMPING — What: the maximum fraction by which a node's
#   per-step relevance decrement is REDUCED at full salience (1.0). The
#   effective decay rate becomes rate * (1 - SALIENCE_DECAY_DAMPING * salience),
#   so salience 0 -> rate unchanged (backward-compatible) and salience 1.0 ->
#   rate * (1 - SALIENCE_DECAY_DAMPING) (slowest).
# Why: D6 effect (ii) — a high-salience memory (a rare critical realization)
#   must decay SLOWER than a trivial one of the same age, so the forgetting
#   curve does not reclaim the rare-but-important at the same rate as the
#   trivial. It MODULATES the EXISTING days_since_access × grade decay; it does
#   not add a parallel curve. 0.9 leaves a salience-1.0 node decaying at 10% of
#   its grade rate (an order-of-magnitude slower) while a salience-0 node is
#   untouched. HIGH but <1 so even a max-salience node still eventually decays
#   (decay-everywhere correction: salience dampens, never freezes, the rate).
#   This is DISTINCT from the granular content-integrity decay (a separate
#   approved proposal) — P5 only makes the EXISTING relevance decay salience-aware.
SALIENCE_DECAY_DAMPING = 0.9

# SALIENCE_FREEZE_EXEMPT — What: the salience at/above which a node is EXEMPT
#   from the decay_pass auto-freeze/eviction (it stays resident in nodes/).
# Why: D6 effect (ii) — an emotionally-charged / operator-tagged one-shot must
#   persist through the forgetting curve that would otherwise auto-freeze a
#   low-frequency node. A HIGH named threshold (Risk 8: salience inflation) so
#   ONLY the genuine top tier is protected; below it, auto-freeze is unchanged.
#   The exemption is LOSS-FREE — it keeps the node in the live (un-frozen) tier;
#   it never deletes or moves anything (freezing only archives + unlinks).
SALIENCE_FREEZE_EXEMPT = 0.85

# ── STC tagging-and-capture decay (FEAT-2026-06-11 temporal-recall P4, §6.5 effect 3) ──
# STC_DECAY_DAMPING — What: STC's weight in the COMBINED step_relevance damping term.
#   The salience and STC contributions SUM into one capped damping factor on the
#   decrement (additive-then-capped), NOT two stacked multipliers (which could zero the
#   decrement and immortalize a node — the exact failure §6.5 warns against).
# Why: §6.5 effect 3 — a weak memory written near a strongly-salient one should be
#   slightly HARDER to lose (the rescue), so its (decaying) stc_capture_score slows its
#   relevance decay. 0.5 lets a fully-captured node meaningfully slow without freezing.
#   A node with stc=0 (legacy / never captured / temporal flag off) leaves the damping
#   salience-only -> decay byte-identical to today (the identity-at-zero property).
STC_DECAY_DAMPING = 0.5

# DAMP_CAP — What: the floor-protecting cap on the COMBINED (salience+STC) damping so
#   the decrement is never driven to zero (a node always decays at >= 1-DAMP_CAP of rate).
# Why: §6.5 — even a fully-salient, fully-captured node must still decay (>= 5% of rate
#   at 0.95), never frozen. The cap is what makes the additive composition safe: salience
#   alone could reach 0.9, STC alone 0.5; their sum is capped here, not allowed to run to 1.
DAMP_CAP = 0.95


def tier_for(relevance: float) -> str:
    """Map a [0,1] relevance to a tier label."""
    for name, floor in TIER_THRESHOLDS:
        if relevance >= floor:
            return name
    return "frozen"


def step_relevance(
    old_rel: float,
    touched_today: bool,
    days_since_access: int = 0,
    grade: str = DEFAULT_GRADE,
    salience: float = 0.0,
    stc_capture: float = 0.0,
) -> tuple[float, str]:
    """Two-regime decay with fissile-material grade + salience + STC modulation.

    Within WARM_FRESHNESS_DAYS: pull toward NEUTRAL (warm-anchored — fresh
    nodes mean-revert to 0.5).
    Outside: pull toward 0 (stale-decay sink — eventually crosses cold and
    frozen thresholds, enabling auto-freeze).

    `grade` selects the per-tick rate from DECAY_RATE_BY_GRADE: enriched
    nodes decay astronomically slow; waste nodes decay quickly.

    `salience` (FEAT-2026-06-07 Tier-1 P5, D6 effect ii) and `stc_capture`
    (FEAT-2026-06-11 temporal-recall P4, §6.5 effect 3) jointly DAMPEN the
    per-step decrement. The two contributions SUM into ONE capped damping
    factor — additive-then-capped, not two stacked multipliers — so the
    effective rate becomes
        rate * (1 - min(SALIENCE_DECAY_DAMPING*sal + STC_DECAY_DAMPING*stc,
                         DAMP_CAP))
    A high-salience OR strongly-captured node decays SLOWER than an equal-age
    plain one; the cap (0.95) keeps even a fully-salient, fully-captured node
    decaying at >= 5% of rate (never frozen). Both default 0.0 -> the damping
    term is 0 -> rate unchanged -> behavior is byte-identical to the prior
    function (backward-compatible). `stc_capture` is the ALREADY-ATTENUATED
    capture scalar (temporal_recall_stc.current_capture_score) the caller reads;
    with the temporal flag off no node carries it, so stc=0 -> salience-only
    damping -> decay byte-identical to today. This modulates the EXISTING
    relevance decay only; it is DISTINCT from the separate granular
    content-integrity decay proposal.
    """
    if touched_today:
        return old_rel, "touched-today"
    rate = DECAY_RATE_BY_GRADE.get(grade, DECAY_RATE_BY_GRADE[DEFAULT_GRADE])
    # Salience + STC dampened decay: SUM the two contributions into one damping
    # term, CAP it (never zero the decrement), then scale the rate down by it. A
    # missing/zero sal AND stc leaves rate exactly as before (no behavior change).
    sal = min(1.0, max(0.0, float(salience)))
    stc = min(1.0, max(0.0, float(stc_capture)))
    damping = SALIENCE_DECAY_DAMPING * sal + STC_DECAY_DAMPING * stc
    if damping > 0.0:
        damping = min(damping, DAMP_CAP)
        rate = rate * (1.0 - damping)
    if days_since_access <= WARM_FRESHNESS_DAYS:
        target, regime = NEUTRAL, "fresh"
    else:
        target, regime = 0.0, "stale"
    return old_rel + (target - old_rel) * rate, f"decay-{regime}-{grade}"


def _days_since(last_access_iso: str, today_iso: str) -> int:
    """Days between last_access and today. Missing/malformed → 9999 (very stale)."""
    if not last_access_iso:
        return 9999
    try:
        last = date.fromisoformat(str(last_access_iso))
        cur = date.fromisoformat(today_iso)
        return max(0, (cur - last).days)
    except (ValueError, TypeError):
        return 9999


def decay_pass(
    nodes_dir: Path,
    dry: bool,
    today: Optional[str] = None,
    auto_freeze: bool = True,
) -> list[dict]:
    """Apply one decay tick across nodes_dir/*.md.

    Per-node logic: read relevance + tier + last_access + material_grade
    from frontmatter; compute days_since_access; apply two-regime decay
    via step_relevance; rewrite frontmatter with new relevance + tier
    (and material_grade=natural if missing). Collect transitions.

    When `auto_freeze=True` (default) and dry=False, nodes that
    transition INTO `tier: frozen` are then passed to samia.core.ia.freeze()
    which compresses them into archive/<id>.frozen.json and removes them
    from nodes/. Pass auto_freeze=False to observe transitions without
    archiving.

    Returns a list of transition records (one per changed node).
    """
    today_iso = today or date.today().isoformat()
    transitions: list[dict] = []
    if not nodes_dir.exists():
        print(f"[tier] no nodes/ at {nodes_dir} — migration not applied yet")
        return transitions

    freeze_queue: list[dict] = []

    for md in sorted(nodes_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        parsed, body = _fm.parse(text)
        if parsed is None:
            continue
        fm, order = parsed

        # AUD61 Phase 3: respect target_state lifecycle field.
        # What: skip nodes with target_state=frozen or archived.
        # Why: frozen nodes are operator-protected from automated lifecycle
        #   changes; archived nodes are historical and must not re-promote.
        ts = str(fm.get("target_state", "live")).lower()
        if ts in ("frozen", "archived"):
            continue

        last = fm.get("last_access", "")
        touched = str(last) == today_iso
        days = 0 if touched else _days_since(str(last), today_iso)

        old_rel = float(fm.get("relevance", NEUTRAL))
        old_tier = str(fm.get("tier", "warm"))
        grade = str(fm.get("material_grade", DEFAULT_GRADE))
        # Salience (FEAT-2026-06-07 Tier-1 P5, D6 ii): read the [0,1] field the
        # Tier-1 salience source (bio.compute_salience) wrote. Missing/malformed
        # -> 0.0 -> decay + auto-freeze behave EXACTLY as before (backward-compat).
        try:
            salience = float(fm.get("salience", 0.0) or 0.0)
        except (TypeError, ValueError):
            salience = 0.0
        # STC capture scalar (FEAT-2026-06-11 temporal-recall P4, §6.5 effect 3): read
        # the stc_capture_score + stc_capture_at the STC capture event stamped (beside
        # `salience`, on the SAME frontmatter block we already parsed — no new I/O) and
        # ATTENUATE by the ~3-day half-life via the shared pure helper, so the decay tick
        # sees the same decayed scalar recall/promotion read. Missing/malformed (legacy /
        # never captured / temporal flag off) -> 0.0 -> damping is salience-only -> decay
        # byte-identical to today. Lazy import keeps tier import-time free of the cycle.
        stc_capture = 0.0
        try:
            raw_stc = float(fm.get("stc_capture_score", 0.0) or 0.0)
            if raw_stc > 0.0:
                from . import temporal_recall_stc as _stc
                cap_at = float(fm.get("stc_capture_at", 0.0) or 0.0)
                stc_capture = _stc.attenuate(raw_stc, cap_at)
        except (TypeError, ValueError, ImportError):
            stc_capture = 0.0

        new_rel, reason = step_relevance(old_rel, touched, days, grade, salience,
                                         stc_capture)
        new_tier = tier_for(new_rel)
        changed_tier = new_tier != old_tier
        changed_rel = abs(new_rel - old_rel) > 1e-4

        if changed_tier or changed_rel:
            t = {
                "node": md.stem,
                "address": fm.get("address", ""),
                "old_rel": round(old_rel, 4),
                "new_rel": round(new_rel, 4),
                "old_tier": old_tier,
                "new_tier": new_tier,
                "grade": grade,
                "days_stale": days,
                "reason": reason,
            }
            transitions.append(t)

        if not dry and (changed_tier or changed_rel):
            # Don't resurrect a node a concurrent freeze/archive removed since we
            # read it (fix 2026-06-02): glob() snapshots paths, but freeze() (this
            # pass's deferred phase, or another concurrent process) unlinks node
            # files. Without this guard, write_text would RE-CREATE the archived
            # node. Re-checking immediately before the write narrows the window to
            # near-zero (full OS-level locking deferred to the concurrency work).
            if not md.exists():
                continue
            fm["relevance"] = round(new_rel, 4)
            fm["tier"] = new_tier
            if "material_grade" not in fm:
                fm["material_grade"] = DEFAULT_GRADE
                order.append("material_grade")
            md.write_text(_fm.serialize(fm, order, body), encoding="utf-8")

            if auto_freeze and new_tier == "frozen" and old_tier != "frozen":
                # FEAT-2026-06-07 Tier-1 P5 (D6 ii): FREEZE/eviction EXEMPTION.
                # A node whose salience clears SALIENCE_FREEZE_EXEMPT is NOT
                # auto-frozen by the decay pass — it stays resident (a high-
                # salience one-shot persists through the forgetting curve that
                # would otherwise reclaim a low-frequency node). The relevance
                # decrement above STILL applies (decay-everywhere: salience
                # dampens + exempts the lifecycle action, it does not stop the
                # number moving); only the auto-freeze ACTION is skipped. A
                # salience-0 node is never exempt -> auto-freeze unchanged.
                if salience >= SALIENCE_FREEZE_EXEMPT:
                    t["freeze_exempt"] = True
                    t["salience"] = round(salience, 4)
                else:
                    freeze_queue.append(t)

    # Deferred auto-freeze: ia.freeze() removes node files; doing it after
    # the iteration avoids mutating the directory mid-walk.
    if not dry and auto_freeze and freeze_queue:
        try:
            from . import ia as _ia
        except ImportError as e:
            print(f"[tier] auto-freeze unavailable (ia import failed): {e}")
            return transitions
        for t in freeze_queue:
            try:
                # ia.freeze signature is freeze(memory_dir, node_name) -> None.
                # (Fix 2026-06-02: was node_id=/reason= kwargs -> TypeError on
                # every call, silently swallowed below, so auto-freeze never
                # actually archived a node. The reason string ia.freeze doesn't
                # accept; provenance lives in the node's own tier frontmatter.)
                _ia.freeze(nodes_dir.parent, t["node"])
                t["frozen"] = True
            except Exception as e:
                t["freeze_error"] = str(e)
                print(f"[tier] auto-freeze FAILED for {t['node']}: {e}")

    return transitions


def decay_tick(memory_dir: Path, force: bool = False,
               erode_integrity: bool | None = None,
               terminal_freeze: bool | None = None) -> dict:
    """6h-cadence decay subscriber for the idle pulse — CONTINUOUS (wake+REM).

    Decay is the short-term forgetting curve. Per the CLS model it runs in
    BOTH the waking state and the sleeping (REM) state — it is NOT gated behind
    REM (sleep is for consolidation/replay, not forgetting). It is driven only
    by the idle_pulse "decay" subscriber, never by the REM driver.

    FEAT-2026-06-07 granular-recall-repaired-decay P1/P3: the SECOND, content-fidelity
    decay axis rides this SAME continuous tick (Q6a, layer-don't-replace). After the
    relevance/tier pass this also runs integrity.integrity_decay_pass — the slow
    per-character content erosion (modulated by salience + tier + recency) and the
    terminal freeze-at-floor.

    ACTIVATION WIRING (granular env flags, all default OFF): `erode_integrity` and
    `terminal_freeze` default None — when the caller does NOT pass one, it RESOLVES from
    the live env flag:
      - erode_integrity -> integrity.decay_enabled()  (ASTHENOS_INTEGRITY_DECAY_ENABLED)
      - terminal_freeze -> integrity.freeze_enabled() (ASTHENOS_INTEGRITY_FREEZE_ENABLED)
    An explicit True/False OVERRIDES the flag (tests pass them explicitly). With both flags
    unset + no explicit args, the integrity sweep does NOT run (erode_integrity resolves
    False) — byte-identical to the prior inert behavior. The two flags are INDEPENDENT:
    DECAY without FREEZE erodes but never freezes-at-floor; FREEZE only takes effect once
    DECAY has eroded a node below INTEGRITY_FLOOR. It is ADDITIVE: the relevance/lifecycle
    decay is UNCHANGED regardless of either flag, and the integrity sweep never erodes a
    node without a recoverable anchor (no data loss). It is NOT a separate scheduler — it is
    invoked here, on the existing cadence, so both axes decay together, ungated, in wake+REM.

    State file `<memory_dir>/.tier_decay_state.json` records `last_tick_unix`;
    the function no-ops when fewer than TIER_DECAY_INTERVAL_S seconds have
    elapsed since the last fire (unless force=True). A cross-session flock
    ensures exactly one concurrent session does the work per interval.

    Returns a telemetry dict suitable for the auditor to log.
    """
    import json as _json
    import time as _time
    import os as _os
    import fcntl as _fcntl

    # CLS rationale (operator correction 2026-06-07): DECAY is the short-term
    # forgetting curve — it runs CONTINUOUSLY across BOTH the waking state and
    # the sleeping (REM) state. Sleep is for CONSOLIDATION + REPLAY (strength-
    # ening/abstracting), NOT for forgetting; so decay is NOT REM-gated. (The
    # REM gate that FEAT-2026-06-07 P2 added here was wrong by design and has
    # been removed.) The ONLY throttles are the existing 6h DECAY cadence
    # cooldown (state-file below) and the cross-session flock — both preserved.
    # decay_tick is driven solely by the idle_pulse "decay" subscriber on its
    # 6h cadence, in BOTH WAKE and REM (NOT by the REM driver — not REM-gated,
    # not a REM subscriber, so there is no double-drive).

    state_path = memory_dir / ".tier_decay_state.json"
    lock_path = memory_dir / ".tier_decay_state.lock"

    def _read_state() -> dict:
        if state_path.exists():
            try:
                return _json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _atomic_write_state(st: dict) -> None:
        # temp + os.replace = atomic rename; eliminates the write_text() truncate-then-
        # write window where a concurrent reader could see a half-written/empty file.
        tmp = state_path.with_name(state_path.name + f".tmp.{_os.getpid()}")
        tmp.write_text(_json.dumps(st, indent=2), encoding="utf-8")
        _os.replace(tmp, state_path)

    # Cross-session serialization (TOCTOU fix 2026-06-03): hook_idle_pulse.sh fires
    # decay_tick on EVERY tool call from EVERY concurrent Claude session (HAP, up to 8).
    # Without a lock, two sessions both read "interval elapsed", both run decay_pass
    # (double decay) and both write (lost update / truncated-mid-write corruption). An
    # exclusive NON-BLOCKING flock makes exactly ONE session run the tick per interval;
    # others no-op cleanly. The elapsed-gate is RE-checked INSIDE the lock.
    try:
        _lock_f = open(lock_path, "w")
    except OSError:
        _lock_f = None
    if _lock_f is not None:
        try:
            _fcntl.flock(_lock_f, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            _lock_f.close()
            return {"fired": False, "reason": "locked_by_concurrent_session"}

    try:
        state = _read_state()
        last_unix = float(state.get("last_tick_unix", 0))
        now = _time.time()
        # What: distinguish "no prior tick" from "time since prior tick". On a fresh
        #   store last_unix == 0, so the raw (now - last_unix) is the full UNIX epoch
        #   (~1.78e9) -- which used to (a) make the first tick fire because the huge
        #   value cleared the interval gate, and (b) get reported verbatim as
        #   "elapsed_seconds", producing the nonsense log line
        #   "tier_decay_tick: ok -- elapsed_seconds=1781307888".
        # Why: the FIRING decision still wants "fire on a fresh store" (establish the
        #   baseline + run the initial decay), but the REPORTED elapsed must be honest.
        #   With no previous tick the truthful elapsed is 0 (this run is the baseline),
        #   so we report 0 while a dedicated first_tick flag drives firing. This is the
        #   root fix at the producer; the scheduler's _summarize just prints what we
        #   return.
        first_tick = last_unix <= 0
        elapsed = 0.0 if first_tick else (now - last_unix)

        if not force and not first_tick and elapsed < TIER_DECAY_INTERVAL_S:
            return {
                "fired": False,
                "elapsed_seconds": int(elapsed),
                "interval_seconds": TIER_DECAY_INTERVAL_S,
            }

        nodes_dir = memory_dir / "nodes"
        transitions = decay_pass(nodes_dir, dry=False, auto_freeze=True)

        # FEAT-2026-06-07 granular-recall-repaired-decay P1/P3: the second (content-
        # fidelity) axis rides the SAME tick. Additive + INERT by default so the
        # relevance/lifecycle decay above is unchanged and nothing erodes/freezes until
        # operator-gated activation. NEVER erodes a node without a recoverable anchor (no
        # data loss). Fail-soft — an integrity error never breaks the relevance tick.
        #
        # Resolve the two GRANULAR flags HERE (the call site), keeping
        # integrity_decay_pass's own signature defaults untouched (tests rely on them):
        #   - erode_integrity defaults to integrity.decay_enabled() when not passed.
        #   - terminal_freeze defaults to integrity.freeze_enabled() when not passed.
        # An explicit arg overrides the flag. Both unset => erode resolves False => the
        # whole integrity sweep is skipped (byte-identical to the prior inert behavior).
        n_eroded = 0
        try:
            from . import integrity as _integrity
            _erode = (erode_integrity if erode_integrity is not None
                      else _integrity.decay_enabled())
            _freeze = (terminal_freeze if terminal_freeze is not None
                       else _integrity.freeze_enabled())
            if _erode:
                eroded = _integrity.integrity_decay_pass(
                    memory_dir, dry=False, terminal_freeze=_freeze)
                n_eroded = len(eroded)
        except Exception as e:
            print(f"[tier] integrity erosion pass failed (relevance unaffected): {e}")

        state["last_tick_unix"] = now
        # UTC-aware timestamp (AUD63): naive datetime.now() wrote local-TZ w/o offset.
        from samia.core.timestamp import now_utc_iso as _now_utc_iso
        state["last_tick_iso"] = _now_utc_iso()
        state["last_transition_count"] = len(transitions)
        state["last_freeze_count"] = sum(1 for t in transitions if t.get("frozen"))
        _atomic_write_state(state)

        return {
            "fired": True,
            "elapsed_seconds": int(elapsed),
            "n_transitions": len(transitions),
            "n_frozen": state["last_freeze_count"],
            "n_integrity_eroded": n_eroded,
            "transitions": transitions,
        }
    finally:
        if _lock_f is not None:
            try:
                _fcntl.flock(_lock_f, _fcntl.LOCK_UN)
            finally:
                _lock_f.close()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.tier
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD15 (original refactor) + AUD61 Phase 3 (target_state skip)
#             + AUD63 Phase 3 (UTC timestamp fix in decay_tick)
#             + 2026-06-07 operator correction: decay is CONTINUOUS (wake+REM),
#               NOT REM-gated (removed the FEAT-P2 is_rem gate in decay_tick;
#               CLS — forgetting runs always, only consolidation/replay sleep).
#             + FEAT-2026-06-07 Tier-1 P5 (D6 ii): SALIENCE-AWARE relevance decay.
#               step_relevance gained an optional `salience` param that DAMPENS
#               the per-step decrement (rate * (1 - SALIENCE_DECAY_DAMPING*sal));
#               decay_pass reads the `salience` frontmatter, passes it through,
#               and EXEMPTS a node with salience >= SALIENCE_FREEZE_EXEMPT from
#               auto-freeze (it stays resident). salience 0 -> behavior unchanged
#               (backward-compatible). Modulates the EXISTING relevance decay only;
#               DISTINCT from the separate granular content-integrity decay.
#             + FEAT-2026-06-07 granular-recall-repaired-decay P1: decay_tick gained
#               an optional `erode_integrity` flag that ALSO runs the SECOND content-
#               fidelity axis (integrity.integrity_decay_pass) on the SAME continuous tick
#               (Q6a, layer-don't-replace). This is PURELY ADDITIVE — step_relevance /
#               decay_pass / the relevance lifecycle math are UNCHANGED; the integrity
#               sweep rides alongside and never erodes a node without a recoverable
#               anchor (no data loss).
#             + FEAT-2026-06-11 temporal-recall P4 (§6.5 effect 3) — STC-aware decay:
#               step_relevance gained an optional `stc_capture` arg (the ALREADY-ATTENUATED
#               capture scalar); decay_pass reads stc_capture_score + stc_capture_at off
#               the SAME parsed frontmatter and attenuates via temporal_recall_stc.attenuate
#               (~3-day half-life). The salience + STC contributions SUM into ONE capped
#               damping term (additive-then-capped at DAMP_CAP=0.95, never zeroed). stc=0
#               (legacy / never captured / temporal flag off) -> salience-only damping ->
#               decay byte-identical. PURELY ADDITIVE; the lifecycle math is unchanged.
#             + FEAT-2026-06-07 granular-recall-repaired-decay ACTIVATION WIRING:
#               decay_tick's `erode_integrity` + new `terminal_freeze` params default
#               None and RESOLVE from the live env flags when not passed —
#               erode_integrity<-integrity.decay_enabled() (ASTHENOS_INTEGRITY_DECAY_
#               ENABLED), terminal_freeze<-integrity.freeze_enabled() (ASTHENOS_INTEGRITY_
#               FREEZE_ENABLED); terminal_freeze is forwarded into integrity_decay_pass.
#               An explicit arg overrides the flag; both flags unset => erode resolves
#               False => the integrity sweep is skipped (byte-identical to inert). The two
#               flags are INDEPENDENT (decay-without-freeze erodes but never freezes).
#               The relevance/lifecycle decay is UNAFFECTED by either flag.
# Layer:      core (pure library, no daemon dependency)
# Role:       relevance-decay + tier classification — the salience/STC-damped
#             forgetting curve (step_relevance), the nodes/ decay walk + auto-freeze
#             (decay_pass), and the 6h idle-pulse decay subscriber (decay_tick).
# Stability:  v1.1 -- AUD61 target_state integration, AUD63 UTC fix
# ErrorModel: decay_pass skips malformed nodes (no frontmatter -> continue);
#             auto_freeze is deferred to end of walk to avoid mid-iteration
#             directory mutation.
# Depends:    datetime, pathlib (stdlib). samia.core.frontmatter.
#             samia.core.ia (optional, for auto_freeze).
#             samia.core.timestamp (AUD63, for decay_tick UTC).
# Exposes:    TIER_THRESHOLDS, DECAY_RATE, NEUTRAL, DECAY_RATE_BY_GRADE,
#             tier_for, step_relevance, decay_pass, decay_tick.
# Lines:      537
# --------------------------------------------------------------------------
