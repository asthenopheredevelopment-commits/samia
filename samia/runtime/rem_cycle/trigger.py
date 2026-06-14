"""samia.runtime.rem_cycle.trigger — idle/activity inputs, the entry trigger
(Q1), and the three-wake-path decision (Q4 evaluate, plus the work_remains read).

Layer 1 (Owns / Depends):
    Owns:    the idle inputs (seconds_since_last_activity / is_idle), the explicit
             force-flag setter (request_sleep_now), the entry decision
             (should_enter_rem), the work-remains read (work_remains), and the pure
             three-wake-path decision (evaluate — it does NOT mutate state).
    Depends: .config (the constants + _state_path/_now/_default_state/_log_event +
             re-exported locked_update_json + sleep_pressure), .state (current_state),
             .registry (_any_subscriber_work_remaining — the subscriber half of
             work_remains). sleep_pressure.compute_pressure is the entry/exit metric.

Layer 2 (What / Why):
    What: should_enter_rem enters REM iff the force flag is set OR (pressure>=thresh
          AND idle). evaluate() returns the next action while in a REM phase (the
          three wake paths a/b/c) WITHOUT mutating state — the caller (tick) applies
          the transition. work_remains ORs the pressure>0 backlog proxy with the
          authoritative subscriber signal.
    Why:  the autonomous health response (pressure) plus the on-demand override
          (force), and the review-and-maybe-snooze wake gate. Carving the pure
          decision out of the monolith keeps evaluate() deterministic + testable in
          isolation; it depends only on config/state/registry (acyclic).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import (
    IDLE_GATE_S,
    MAX_DURATION_S,
    REM,
    REVIEW_WAIT_S,
    REVIEWING,
    ENTER_REVIEWING,
    REST,
    SNOOZE,
    STAY_REM,
    WAKE_SAFETY,
    WAKE_YIELD,
    _default_state,
    _log,
    _log_event,
    _now,
    _state_path,
    locked_update_json,
    sleep_pressure,
)
from .registry import _any_subscriber_work_remaining
from .state import current_state


# ---------------------------------------------------------------------------
# Idle / activity inputs (no clock spun here — read or passed in)
# ---------------------------------------------------------------------------


def seconds_since_last_activity(now: float | None = None) -> float | None:
    """Seconds since the last recorded activity (heartbeat tick / tool / write).

    What: reads the LAST `ts` from the heartbeat activity log
          (~/.local/share/asthenos/heartbeat/ticks.jsonl — appended on every
          PostToolUse) and returns now - last. Returns None when no activity
          log exists (treated by callers as "idle unknown").
    Why:  reuses the existing, well-tested activity signal rather than inventing
          a new clock (feedback_scheduling_minimize_clocks). Tests pass the idle
          seconds in directly via should_enter_rem(..., idle_seconds=...).
    """
    log_path = (Path.home() / ".local" / "share" / "asthenos"
                / "heartbeat" / "ticks.jsonl")
    if not log_path.exists():
        return None
    last_ts: float | None = None
    try:
        with log_path.open("rb") as f:
            try:
                f.seek(-4096, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts")
            if isinstance(ts, str):
                import datetime as _dt
                last_ts = _dt.datetime.fromisoformat(ts).timestamp()
                break
            if isinstance(ts, (int, float)):
                last_ts = float(ts)
                break
    except (OSError, ValueError) as exc:
        _log.debug("rem_cycle: activity log read failed: %s", exc)
        return None
    if last_ts is None:
        return None
    return max(0.0, (now if now is not None else _now()) - last_ts)


def is_idle(idle_seconds: float | None = None, now: float | None = None) -> bool:
    """True iff the system has been idle long enough to sleep without disrupting.

    What: idle iff seconds-since-last-activity >= IDLE_GATE_S. `idle_seconds` may
          be passed in (tests / a caller that already tracks idleness);
          otherwise it is read from the heartbeat activity log.
    Why:  the idle half of the Q1 trigger. Unknown idleness (no activity log) is
          treated as NOT idle — fail closed, never sleep into active work.
    """
    secs = idle_seconds if idle_seconds is not None else \
        seconds_since_last_activity(now)
    if secs is None:
        return False
    return secs >= IDLE_GATE_S


# ---------------------------------------------------------------------------
# Trigger (entry) — Q1
# ---------------------------------------------------------------------------


def request_sleep_now(mem: Path) -> dict[str, Any]:
    """Set the explicit "sleep now" force flag (the operator/agent trigger).

    What: persists force_requested=True so the next should_enter_rem / tick
          enters REM regardless of pressure or idleness.
    Why:  Q1's explicit-trigger path — forces a cycle on demand (rem_sleep_now
          IPC op / an agent calling for a consolidation pass). Mitigates risk-1
          (an over-conservative trigger never firing).
    """
    p = _state_path(mem)
    p.parent.mkdir(parents=True, exist_ok=True)
    with locked_update_json(p, default=None) as state:
        if not state:
            state.update(_default_state())
        state["force_requested"] = True
        result = dict(state)
    _log_event(mem, "force_requested", {})
    return result


def should_enter_rem(
    mem: Path,
    idle_seconds: float | None = None,
    now: float | None = None,
    pressure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether to enter REM (Q1 trigger).

    What: returns {enter, reason, forced, idle, pressure}. enter=True iff EITHER
          the explicit force flag is set (forced path) OR (sleep_pressure >=
          threshold AND the idle gate is satisfied). NEVER a bare timer.
    Why:  the autonomous health response (pressure) plus the on-demand override
          (force). Pressure-without-idle must NOT enter (it would compete with
          active cognition — the exact swarm bug); idle-without-pressure must NOT
          enter (nothing is owed).

    Args:
        idle_seconds: seconds since last activity (else read from the log).
        now: wall time override (tests / deterministic eval).
        pressure: a precomputed compute_pressure() result (else computed here).
    """
    state = current_state(mem)
    forced = bool(state.get("force_requested"))
    if pressure is None:
        pressure = sleep_pressure.compute_pressure(mem)
    idle = is_idle(idle_seconds, now)
    if forced:
        return {"enter": True, "reason": "explicit_trigger", "forced": True,
                "idle": idle, "pressure": pressure}
    if pressure["sleep_needed"] and idle:
        return {"enter": True, "reason": "pressure_and_idle", "forced": False,
                "idle": idle, "pressure": pressure}
    return {"enter": False,
            "reason": ("not_idle" if pressure["sleep_needed"] else
                       ("no_pressure" if idle else "no_pressure_not_idle")),
            "forced": False, "idle": idle, "pressure": pressure}


# ---------------------------------------------------------------------------
# Wake + back-to-sleep decision (the three wake paths) — Q4
# ---------------------------------------------------------------------------


def work_remains(mem: Path, pressure: dict[str, Any] | None = None) -> bool:
    """True iff offline reconciliation is still owed.

    What: work remains iff EITHER the sleep_pressure score > 0 (some backlog is
          non-empty) OR any registered REM subscriber is due / has cursor-
          remaining work (P2 OR-in). The subscriber signal is the authoritative
          one once subscribers exist; the pressure fallback covers the case
          where a backlog exists that no subscriber has yet claimed.
    Why:  drives the snooze decision (path b): REM is sticky while idle + work
          remains; once backlogs drain AND no subscriber owes work, the system
          settles into idle REM-rest rather than re-snoozing forever (risk-3).
          P2 replaces P1's pressure-only proxy with the real subscriber signal.
    """
    if pressure is None:
        pressure = sleep_pressure.compute_pressure(mem)
    if pressure["score"] > 0.0:
        return True
    return _any_subscriber_work_remaining(mem)


def evaluate(
    mem: Path,
    activity: bool = False,
    idle_seconds: float | None = None,
    now: float | None = None,
    pressure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide the next action while in a REM phase (the three wake paths).

    Returns {action, reason, ...}. action is one of:
        WAKE_YIELD   (path a) operator/agent activity arrived -> end the cycle.
        WAKE_SAFETY  (path c) MAX_DURATION_S elapsed since since_ts -> backstop.
        ENTER_REVIEWING (path b, step 1) natural drain / pressure<threshold ->
                     enter the bounded review window.
        SNOOZE       (path b, step 2) review window elapsed, still idle + work
                     remains -> re-enter REM (the snooze).
        REST         (path b, settle) review window elapsed, no work remains ->
                     stay in REM doing nothing (idle REM-rest).
        STAY_REM     none of the above -> keep working this cycle.

    What: the pure decision; it does NOT mutate state — the caller (the daemon
          tick, P2) applies the transition (wake/review/enter_rem). This keeps
          evaluate() deterministic + trivially testable.
    Why:  Q4 encodes wake as a review-and-maybe-snooze gate, not a flat stop.
          Operator activity ALWAYS preempts (path a is checked first and is the
          only TRUE end of a cycle); the cap is the hard backstop; pressure-drop
          / drain is a natural completion that must not strand work.

    Args:
        activity: True iff operator/agent activity arrived this evaluation
                  (the path-a signal; passed in — REM yields to the waking path).
        idle_seconds / now / pressure: as in should_enter_rem.
    """
    state = current_state(mem)
    phase = state["phase"]
    now_v = now if now is not None else _now()

    # Outside any REM phase there is nothing to evaluate.
    if phase not in (REM, REVIEWING):
        return {"action": STAY_REM, "reason": "not_in_rem", "phase": phase}

    # PATH A — operator/agent activity always preempts and ENDS the cycle.
    if activity:
        return {"action": WAKE_YIELD, "reason": "operator_activity",
                "phase": phase}

    # PATH C — max-sleep-duration cap (safety backstop).
    since = float(state.get("since_ts") or now_v)
    # For REVIEWING, measure the cap against when the cycle's REM work began is
    # less meaningful; use since_ts of the current phase as the conservative
    # backstop — a long-stuck REVIEWING also force-wakes.
    if (now_v - since) >= MAX_DURATION_S:
        return {"action": WAKE_SAFETY, "reason": "max_duration",
                "elapsed_s": round(now_v - since, 1), "phase": phase}

    if pressure is None:
        pressure = sleep_pressure.compute_pressure(mem)
    idle = is_idle(idle_seconds, now_v)
    remains = work_remains(mem, pressure)

    if phase == REM:
        # Already settled into idle REM-rest (last transition was "rest") and
        # still nothing owed -> stay at rest, no REVIEWING churn (risk-3). Fresh
        # pressure re-arming the cycle is handled by the natural-completion branch
        # below once work_remains again becomes true.
        if state.get("reason") == "rest" and not remains:
            return {"action": REST, "reason": "resting", "phase": phase,
                    "pressure": pressure}
        # Natural completion: pressure dropped below threshold OR no work remains
        # -> enter REVIEWING (path b step 1). Otherwise keep working.
        if (not pressure["sleep_needed"]) or (not remains):
            return {"action": ENTER_REVIEWING,
                    "reason": ("pressure_below_threshold"
                               if not pressure["sleep_needed"] else "work_drained"),
                    "phase": phase, "pressure": pressure}
        return {"action": STAY_REM, "reason": "pressure_and_work",
                "phase": phase, "pressure": pressure}

    # phase == REVIEWING — has the bounded review window elapsed?
    rstart = float(state.get("review_started_ts") or since)
    window_elapsed = (now_v - rstart) >= REVIEW_WAIT_S
    if not window_elapsed:
        # Still inside the review window: keep waiting for instructions.
        return {"action": ENTER_REVIEWING, "reason": "review_window_open",
                "phase": phase, "remaining_s": round(REVIEW_WAIT_S - (now_v - rstart), 1)}
    # Window elapsed, no activity (checked above). Snooze iff idle + work remains.
    if idle and remains:
        return {"action": SNOOZE, "reason": "idle_and_work_remains",
                "phase": phase, "pressure": pressure}
    # No work remains (or not idle but no activity) -> settle into idle REM-rest.
    return {"action": REST, "reason": ("no_work_remains" if not remains
                                       else "not_idle"),
            "phase": phase, "pressure": pressure}


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_cycle.trigger
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_cycle monolith during
#             modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the idle inputs (seconds_since_last_activity / is_idle), the entry
#             trigger (request_sleep_now / should_enter_rem — Q1), the work-remains
#             read (work_remains), and the pure three-wake-path decision (evaluate —
#             Q4, state-mutation-free).
# Stability:  stable — behavior byte-identical to the monolith's trigger/decision
#             sections; evaluate() remains pure (no state writes).
# ErrorModel: seconds_since_last_activity is fail-soft (missing log / parse error ->
#             None -> treated NOT idle, fail-closed); should_enter_rem / evaluate
#             never raise (work_remains delegates to the fail-soft registry query).
# Depends:    .config (constants + _state_path/_now/_default_state/_log_event +
#             locked_update_json + sleep_pressure), .state (current_state), .registry
#             (_any_subscriber_work_remaining); json/os (stdlib).
# Exposes:    seconds_since_last_activity, is_idle, request_sleep_now,
#             should_enter_rem, work_remains, evaluate.
# Note:       evaluate() never mutates state — the caller (driver.tick) applies the
#             transition. Unknown idleness fails closed (never sleeps into active
#             work).
# Lines:      331
# ─────────────────────────────────────────────
