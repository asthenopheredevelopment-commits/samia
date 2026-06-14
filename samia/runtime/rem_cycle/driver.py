"""samia.runtime.rem_cycle.driver — the offline-work engine: tick (decision-and-
apply) + run_due_subscribers (run the due REM subscribers, ONLY while in REM).

Layer 1 (Owns / Depends):
    Owns:    tick (one decision-and-apply step the daemon idle_pulse loop calls) and
             run_due_subscribers (the interruptible, budget-bounded driver that runs
             the due offline-op subscribers in priority order — ONLY while is_rem()).
    Depends: .config — CRITICALLY the SAME shared registry objects (_rem_subscribers
             dict + _rem_subscribers_lock) that registry.py / register_rem_subscribers
             populate (it iterates THAT dict so it runs the registered rows) + the
             DEFAULT_RUN_BUDGET + the action vocabulary + _log/_log_event. .state
             (is_rem / current_state / enter_rem / wake / review), .registry
             (_cursor_has_remaining / _any_subscriber_work_remaining), .trigger
             (should_enter_rem / evaluate). time (stdlib, the per-run stopwatch).

Layer 2 (What / Why):
    What: run_due_subscribers iterates the registry singleton under its Lock in
          ascending priority, running each due subscriber in its own try/except,
          honoring the run-only-in-REM gate + interruptibility (stop after the
          current subscriber when activity arrives) + a per-cycle budget. tick reads
          the phase and applies the should_enter_rem / evaluate decision, driving the
          subscribers ONLY on STAY_REM.
    Why:  the offline-work engine — the ONLY place P2 runs offline work, and only
          inside the sleep window. Carved out of the 1074-line monolith; it sits at
          the top of the package DAG (depends on config/state/registry/trigger), so
          the heavy driver is isolated from the leaf primitives.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_RUN_BUDGET,
    ENTER_REVIEWING,
    REST,
    REVIEWING,
    SNOOZE,
    STAY_REM,
    WAKE,
    WAKE_SAFETY,
    WAKE_YIELD,
    _log,
    _log_event,
    _rem_subscribers,
    _rem_subscribers_lock,
)
from .registry import _any_subscriber_work_remaining, _cursor_has_remaining
from .state import current_state, enter_rem, is_rem, review, wake
from .trigger import evaluate, should_enter_rem


def tick(
    mem: Path,
    activity: bool = False,
    idle_seconds: float | None = None,
    now: float | None = None,
    run_work: bool = True,
) -> dict[str, Any]:
    """One decision-and-apply step (the entry the daemon idle_pulse loop calls).

    What: when WAKE, consults should_enter_rem and enters REM if triggered; when
          in a REM phase, consults evaluate() and APPLIES the resulting
          transition (wake / review / snooze-via-enter_rem). On the STAY_REM
          decision (in REM, work owed) it DRIVES the offline-op subscribers
          (P2: run_due_subscribers), honoring the interrupt. Returns the action
          taken + the resulting state. NO loop, NO timer — one step per call.
    Why:  the seam the resident idle_pulse loop drives (cadence-gated). P1 ran
          NO offline work here; P2 runs the due subscribers ONLY on STAY_REM, so
          heavy reconciliation happens only inside the sleep window and never
          on a waking idle pulse. Splitting the pure decision (evaluate) from the
          apply (tick) keeps the decision testable in isolation; ``run_work`` can
          be False to exercise the transition logic without running subscribers.

    Args:
        run_work: when True (default), the STAY_REM branch runs due subscribers;
                  set False to test/observe transitions without running work.
    """
    state = current_state(mem)
    phase = state["phase"]

    if phase == WAKE:
        decision = should_enter_rem(mem, idle_seconds=idle_seconds, now=now)
        if decision["enter"]:
            new = enter_rem(mem, reason=decision["reason"])
            return {"action": "enter_rem", "reason": decision["reason"],
                    "state": new}
        return {"action": "stay_wake", "reason": decision["reason"], "state": state}

    ev = evaluate(mem, activity=activity, idle_seconds=idle_seconds, now=now)
    action = ev["action"]
    if action == WAKE_YIELD:
        return {"action": WAKE_YIELD, "reason": ev["reason"],
                "state": wake(mem, reason="operator_activity")}
    if action == WAKE_SAFETY:
        return {"action": WAKE_SAFETY, "reason": ev["reason"],
                "state": wake(mem, reason="max_duration")}
    if action == ENTER_REVIEWING:
        # Enter REVIEWING only on a genuine first transition into it; while the
        # window is open we leave the phase as REVIEWING (no churn).
        if phase != REVIEWING:
            return {"action": ENTER_REVIEWING, "reason": ev["reason"],
                    "state": review(mem, reason=ev["reason"])}
        return {"action": "reviewing_wait", "reason": ev.get("reason"),
                "state": state}
    if action == SNOOZE:
        return {"action": SNOOZE, "reason": ev["reason"],
                "state": enter_rem(mem, reason="snooze")}
    if action == REST:
        # Settle into idle REM-rest: leave REVIEWING and rest in REM doing
        # nothing (still in REM until operator activity or fresh pressure). Only
        # transition if we were REVIEWING — an already-REM rest needs no churn.
        new = enter_rem(mem, reason="rest") if phase == REVIEWING else state
        return {"action": REST, "reason": ev["reason"], "state": new}
    # STAY_REM: in REM with work owed -> drive the due offline-op subscribers.
    # This is the ONLY place P2 runs offline work, and only while in REM. The
    # interrupt (activity) is False here (path a is handled above as WAKE_YIELD),
    # so a normal STAY_REM tick runs work; an activity-interrupt that arrives
    # mid-call to run_due_subscribers is honored inside the driver.
    out: dict[str, Any] = {"action": STAY_REM, "reason": ev.get("reason"),
                           "state": state}
    if run_work:
        out["work"] = run_due_subscribers(mem, activity=activity)
    return out


# ---------------------------------------------------------------------------
# The driver: run due REM subscribers (P2) — only ever while is_rem()
# ---------------------------------------------------------------------------


def run_due_subscribers(
    mem: Path,
    activity: bool = False,
    budget: int | None = None,
) -> dict[str, Any]:
    """Run the due REM subscribers in priority order — ONLY while in REM.

    What: when is_rem(mem), iterates the registered offline-op subscribers in
          ascending priority (LOW first), running each that is due (its due_fn,
          or cursor-remaining work) inside its own try/except. Honors:
            - the run-only-in-REM gate (returns immediately, logged, when WAKE);
            - interruptibility (Q4): if ``activity`` arrives, STOP after the
              CURRENT subscriber — remaining subscribers are deferred to the next
              cycle (each subscriber checkpoints its own cursor internally);
            - a per-cycle ``budget`` (DEFAULT_RUN_BUDGET) bounding runs so a
              single tick stays responsive; the rest resume next cycle.
          Returns {ran, results, interrupted, work_remaining, deferred}.
    Why:  the offline-work engine. One failing subscriber must NOT abort the
          rest (fail-open per subscriber, logged loudly — never swallowed
          silently). work_remaining feeds the P1 evaluate() snooze/rest decision
          (replacing the pressure>0 proxy): a True means a subsequent tick
          should keep the cycle alive.

    Args:
        activity: True iff operator/agent activity arrived (the interrupt
                  signal; mirrors evaluate()'s path-a). When True the driver
                  stops after the current subscriber so REM yields promptly.
        budget: max subscriber RUNS this call (default DEFAULT_RUN_BUDGET).
    """
    if not is_rem(mem):
        # The gate: never run offline work outside REM. Logged, not silent.
        _log_event(mem, "driver_refused",
                   {"phase": current_state(mem)["phase"]})
        return {"ran": 0, "results": {}, "interrupted": False,
                "work_remaining": False, "deferred": [], "refused": "not_in_rem"}

    cap = DEFAULT_RUN_BUDGET if budget is None else int(budget)
    # Snapshot the SINGLE shared registry under its Lock — this is the SAME dict
    # register_rem_subscribers() populates, so the rows here are the registered ops.
    with _rem_subscribers_lock:
        subs = sorted(_rem_subscribers.values(), key=lambda s: (s.priority, s.name))

    ran = 0
    results: dict[str, Any] = {}
    interrupted = False
    deferred: list[str] = []

    for sub in subs:
        # Interrupt check FIRST each iteration: operator activity yields promptly
        # (Q4 path a). The current subscriber already ran; the rest defer.
        if activity:
            interrupted = True
            deferred.append(sub.name)
            continue
        if ran >= cap:
            deferred.append(sub.name)
            continue
        # Due-condition: skip a subscriber with nothing to do this cycle UNLESS
        # its cursor records remaining (interrupted mid-batch) work.
        is_due = True
        try:
            if sub.due_fn is not None:
                is_due = bool(sub.due_fn(Path(mem)))
        except Exception:
            _log.warning("rem_cycle: due_fn(%s) raised — treating as due",
                         sub.name, exc_info=True)
            is_due = True
        if sub.cursor_key and _cursor_has_remaining(mem, sub.cursor_key):
            is_due = True
        if not is_due:
            continue

        t0 = time.monotonic()
        try:
            res = sub.fn(Path(mem))
            sub.run_count += 1
            sub.last_run_wall = time.time()
            sub.last_result = res if isinstance(res, dict) else {"result": res}
            results[sub.name] = sub.last_result
            ran += 1
        except Exception as e:  # FAIL-OPEN per subscriber — never abort the rest
            sub.error_count += 1
            sub.last_error = f"{type(e).__name__}: {e}"
            results[sub.name] = {"error": sub.last_error}
            # Log LOUDLY (warning) — a failing offline op must be operator-visible.
            _log.warning("rem_cycle: subscriber %s FAILED (non-fatal, continuing): %s",
                         sub.name, sub.last_error, exc_info=True)
            _log_event(mem, "subscriber_error",
                       {"op": sub.name, "error": sub.last_error})
            ran += 1  # count the attempt against the budget (no infinite retry storm)
        else:
            _log_dur_ms = (time.monotonic() - t0) * 1000.0
            _log.debug("rem_cycle: subscriber %s ran in %.1fms",
                       sub.name, _log_dur_ms)

    work_remaining = _any_subscriber_work_remaining(mem)
    _log_event(mem, "driver_run",
               {"ran": ran, "interrupted": interrupted,
                "deferred": deferred, "work_remaining": work_remaining})
    return {"ran": ran, "results": results, "interrupted": interrupted,
            "work_remaining": work_remaining, "deferred": deferred}


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_cycle.driver
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_cycle monolith during modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the offline-work engine — tick (one decision-and-apply step the daemon
#             idle_pulse loop calls) + run_due_subscribers (the interruptible, budget-
#             bounded driver that runs the due offline-op subscribers in priority
#             order, ONLY while is_rem()).
# Stability:  stable — behavior byte-identical to the monolith's tick/driver sections.
# ErrorModel: FAIL-OPEN per subscriber — one failing subscriber is logged loudly +
#             recorded (error_count / last_error / event) and the rest still run; the
#             gate refuses outside REM with a logged no-op (never a silent drop); a
#             broken due_fn is treated as due (logged).
# Depends:    .config (DEFAULT_RUN_BUDGET + action vocabulary + _log/_log_event + the
#             SINGLE shared _rem_subscribers dict + _rem_subscribers_lock), .state
#             (is_rem/current_state/enter_rem/wake/review), .registry
#             (_cursor_has_remaining/_any_subscriber_work_remaining), .trigger
#             (should_enter_rem/evaluate); time (stdlib).
# Exposes:    tick, run_due_subscribers.
# Lines:      261
# Note:       run_due_subscribers iterates the SAME config-owned registry dict that
#             register_rem_subscribers() populates (single-owned singleton) — so it
#             runs the rows the rem_subscribers package registered. STAY_REM is the
#             ONLY place offline work runs, and only inside the sleep window.
# ─────────────────────────────────────────────
