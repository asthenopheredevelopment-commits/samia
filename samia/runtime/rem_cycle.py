"""samia.runtime.rem_cycle — the persisted WAKE<->REM sleep state machine.

Layer 1 (Owns / Depends):
    Owns:    the persisted REM state (<mem>/biomimetic/rem_state.json:
             {phase, since_ts, reason, cycle_id, ...}), the gate every offline
             op will consult (is_rem / current_state), the WAKE<->REM<->REVIEWING
             transitions (enter_rem / wake / review), the entry trigger
             (should_enter_rem — pressure AND idle, OR an explicit force flag),
             the three-wake-path decision logic (evaluate), and the
             rem_status / rem_sleep_now read/trigger surface that mcp_server +
             the daemon IPC wrap.
    Depends: samia.core.atomic_state (locked_update_json — flock + atomic
             replace, the established race-safe state pattern), and
             samia.runtime.sleep_pressure (compute_pressure — the entry/exit
             metric). The idle / activity / "now" inputs are passed IN (or read
             from the heartbeat activity log) — NO background thread, NO clock.

Layer 2 (What / Why):
    What: REM P1's state machine. A two-state core (WAKE / REM) with a REVIEWING
          sub-state for the operator's "snooze" refinement. Offline ops (wired
          in P2) refuse to run outside REM; P1 only PROVIDES the gate, the entry
          trigger, and the wake/back-to-sleep decision. The three wake paths
          (Q4) are modeled as explicit outcomes of evaluate():
            (a) operator/agent activity      -> wake_yield  (the only true end)
            (b) natural completion / drain    -> reviewing -> snooze | rest
            (c) max-sleep-duration cap        -> wake_safety (backstop)
    Why:  The single shared SLEEP boundary so heavy offline reconciliation runs
          in a contained, idle-gated window and yields instantly to active
          cognition (bug_idle_pulse_hook_python_swarm: never compete with the
          waking machine; feedback_scheduling_minimize_clocks: trigger/event
          entry, never a bare timer — the only time components allowed are the
          idle gate and the max-duration cap).

P1 is PRODUCE-ONLY: pure functions + a small JSON state file. No daemon, no
thread, no timer is started here; `now` / `idle` / `activity` are parameters or
read from the existing heartbeat activity log. Activation (daemon restart +
P2 subscriber wiring) is operator-gated and out of P1's scope.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from samia.core.atomic_state import locked_update_json
from samia.core.paths import resolve_memory_root
from samia.runtime import sleep_pressure

_log = logging.getLogger("samia.runtime.rem_cycle")

# ---------------------------------------------------------------------------
# Phases + tunable time components.
#
# Only TWO time components exist (Q1 / feedback_scheduling_minimize_clocks):
#   - the IDLE GATE: how many seconds of no activity counts as "idle enough".
#   - the MAX-SLEEP-DURATION cap: the safety backstop (path c).
# Both are env-tunable. Nothing here spins a clock; callers pass `now`.
# ---------------------------------------------------------------------------

WAKE = "wake"
REM = "rem"
REVIEWING = "reviewing"

# Idle gate: seconds since the last activity (heartbeat tick / tool / write)
# below which the system is considered ACTIVE (not idle enough to sleep).
IDLE_GATE_S = float(os.environ.get("REM_IDLE_GATE_S", "120"))

# Max-sleep-duration cap (path c): the hard backstop so a misbehaving subscriber
# can never hold REM forever.
MAX_DURATION_S = float(os.environ.get("REM_MAX_DURATION_S", str(30 * 60)))

# Review window (path b): how long REVIEWING waits for instructions before it
# snoozes back into REM (when idle + work remains). Bounded; risk-3 mitigation.
REVIEW_WAIT_S = float(os.environ.get("REM_REVIEW_WAIT_S", "30"))

# evaluate() action vocabulary (the next-action a caller/loop should take).
STAY_REM = "stay_rem"
ENTER_REVIEWING = "reviewing"
SNOOZE = "snooze"
WAKE_YIELD = "wake_yield"
WAKE_SAFETY = "wake_safety"
REST = "rest"  # idle REM-rest: in REM, no work remains, pressure low — do nothing

# Per-cycle subscriber budget (P2): the max number of subscriber RUNS the driver
# performs in one run_due_subscribers() call before yielding back to the loop.
# Bounds a single REM tick's work so the resident loop stays responsive; the
# checkpoint/cursor model resumes the rest next cycle. Env-tunable.
DEFAULT_RUN_BUDGET = int(os.environ.get("REM_RUN_BUDGET", "8"))


# ===========================================================================
# REM OFFLINE-OP SUBSCRIBER REGISTRY (P2)
#
# Mirrors samia.runtime.idle_pulse's _Subscriber/register_subscriber pattern,
# adding what offline reconciliation needs that the idle-pulse loop does not:
#   - priority      : run order (LOW int = runs FIRST). Decay/prune before
#                     replay/consolidation so the heavy graph work runs on a
#                     pruned set.
#   - due_fn(mem)   : a per-subscriber due-condition (e.g. "only if the
#                     near-dup backlog is non-empty"). A subscriber whose
#                     due_fn is False is skipped this cycle (no wasted work).
#   - fn(mem)       : the work. Returns a dict carrying a "work_remaining"
#                     (and/or "made_progress") signal so the driver can OR it
#                     into evaluate()'s work_remains and so an interrupt can
#                     leave a resumable cursor.
#   - cursor_key    : a stable key under <mem>/biomimetic/rem_cursors.json the
#                     subscriber checkpoints into; the registry reads it to
#                     report cursor-remaining work (the work_remains OR-in).
#
# The registry is PRODUCE-ONLY data: registering does not run anything. The
# driver (run_due_subscribers) only runs subscribers WHEN is_rem() is True and
# is only ever called by the daemon tick (operator-gated activation).
# ===========================================================================


@dataclass
class _RemSubscriber:
    """One REM offline-op subscriber (priority + due-condition + cursor)."""

    name: str
    fn: Callable[[Path], Any]
    priority: int = 100            # LOW runs FIRST; default mid-band
    due_fn: Callable[[Path], bool] | None = None  # None => always due in REM
    cursor_key: str | None = None  # key in rem_cursors.json this op checkpoints
    # Stats (observability parity with idle_pulse _Subscriber).
    last_run_wall: float = 0.0
    run_count: int = 0
    error_count: int = 0
    last_error: str = ""
    last_result: dict[str, Any] = field(default_factory=dict)


_rem_subscribers: dict[str, _RemSubscriber] = {}
_rem_subscribers_lock = threading.Lock()


def register_offline_op(
    name: str,
    fn: Callable[[Path], Any],
    priority: int = 100,
    due_condition: Callable[[Path], bool] | None = None,
    cursor_key: str | None = None,
) -> _RemSubscriber:
    """Register (or update) a REM offline-op subscriber.

    What: maps ``name`` to a work callable that the REM driver runs — and ONLY
          runs — while the system is in REM, in ascending ``priority`` order,
          subject to its own ``due_condition``. Re-registering an existing name
          updates its fn/priority/due/cursor but preserves accumulated stats
          (idempotent across daemon re-init), mirroring idle_pulse.
    Why:  Q3 — the extensible home for every offline reconciliation op. Each op
          declares run order (priority) + when it has work (due_condition) +
          where it checkpoints (cursor_key). The whole point of REM: heavy work
          registers HERE and refuses to run outside the sleep window.

    Args:
        name: stable subscriber id (also the rem_status row key).
        fn: fn(mem) -> dict | Any. SHOULD return a dict carrying a
            "work_remaining" bool (and optionally "made_progress") so the
            driver can OR it into work_remains and the cursor reflects progress.
        priority: run order; LOWER runs FIRST (decay before consolidate/replay).
        due_condition: due_fn(mem) -> bool; skipped this cycle when False.
            None => always due once in REM.
        cursor_key: the key under rem_cursors.json this op checkpoints; the
            registry reads it for cursor-remaining work (the work_remains OR-in).
    """
    with _rem_subscribers_lock:
        existing = _rem_subscribers.get(name)
        if existing is not None:
            existing.fn = fn
            existing.priority = int(priority)
            existing.due_fn = due_condition
            existing.cursor_key = cursor_key
            return existing
        sub = _RemSubscriber(
            name=name, fn=fn, priority=int(priority),
            due_fn=due_condition, cursor_key=cursor_key,
        )
        _rem_subscribers[name] = sub
        return sub


def registered_offline_ops() -> list[str]:
    """Names of all registered REM offline ops (priority order). Pure read."""
    with _rem_subscribers_lock:
        subs = sorted(_rem_subscribers.values(), key=lambda s: (s.priority, s.name))
    return [s.name for s in subs]


# ---------------------------------------------------------------------------
# Cursor checkpoint store (per-subscriber resumable progress)
# ---------------------------------------------------------------------------


def _cursors_path(mem: Path) -> Path:
    return Path(mem) / "biomimetic" / "rem_cursors.json"


def read_cursor(mem: Path, key: str) -> dict[str, Any]:
    """Read a subscriber's persisted cursor (the resume point). Empty if none.

    What: returns the dict a subscriber last checkpointed under ``key``.
    Why:  an interrupt (Q4 path a/c) leaves the cursor; the next REM cycle reads
          it to resume mid-batch instead of restarting or skipping work.
    """
    p = _cursors_path(mem)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cur = data.get(key) if isinstance(data, dict) else None
        return dict(cur) if isinstance(cur, dict) else {}
    except (OSError, ValueError):
        return {}


def write_cursor(mem: Path, key: str, cursor: dict[str, Any]) -> None:
    """Checkpoint a subscriber's cursor (race-safe, atomic).

    What: merges ``cursor`` under ``key`` in rem_cursors.json via the established
          flock+atomic-replace pattern.
    Why:  the resume point for interruptible offline work (Q4 / risk-4). A
          subscriber calls this at safe boundaries so an interrupt never strands
          half-applied work — it resumes from the cursor next cycle.
    """
    p = _cursors_path(mem)
    p.parent.mkdir(parents=True, exist_ok=True)
    with locked_update_json(p, default=None) as data:
        if not data:
            data.clear()
        data[key] = dict(cursor)


def _cursor_has_remaining(mem: Path, key: str) -> bool:
    """True iff a subscriber's cursor records remaining work.

    What: a cursor with ``"remaining": <truthy>`` or ``"done": False`` is
          unfinished. Absent/empty cursor => no recorded remaining work.
    Why:  feeds work_remains so evaluate() does not prematurely drain a cycle
          while a subscriber still has cursor-tracked work (replaces P1's
          pressure>0 proxy with a real signal).
    """
    cur = read_cursor(mem, key)
    if not cur:
        return False
    if "remaining" in cur:
        return bool(cur["remaining"])
    if "done" in cur:
        return not bool(cur["done"])
    return False


def _any_subscriber_work_remaining(mem: Path) -> bool:
    """True iff ANY registered subscriber is due OR has cursor-remaining work.

    What: the subscriber half of work_remains — a subscriber whose due_fn is
          True (has work to start) or whose cursor records remaining work (was
          interrupted mid-batch) means REM is not drained.
    Why:  Q4 path b must not snooze→rest while a subscriber still owes work;
          this OR-in replaces P1's pressure-only proxy.
    """
    with _rem_subscribers_lock:
        subs = list(_rem_subscribers.values())
    for sub in subs:
        if sub.cursor_key and _cursor_has_remaining(mem, sub.cursor_key):
            return True
        try:
            if sub.due_fn is None or sub.due_fn(Path(mem)):
                return True
        except Exception:  # a broken due_fn must not crash work_remains
            _log.debug("rem_cycle: due_fn for %s raised (treated not-due)",
                       sub.name, exc_info=True)
    return False


# ---------------------------------------------------------------------------
# The run-only-in-REM gate (P2)
# ---------------------------------------------------------------------------


def gate_offline_op(mem: Path, op_name: str) -> bool:
    """The gate: True iff ``op_name`` may run now (i.e. the system is in REM).

    What: the single guard every migrated offline op consults at entry. Returns
          True only when is_rem(mem); otherwise it LOGS a "refused — not in REM"
          no-op to rem_events.jsonl and returns False (a LOGGED no-op, never a
          silent drop — risk-5).
    Why:  Q5 — offline reconciliation refuses to run outside the sleep window so
          it never competes with active cognition. The refusal is operator-
          visible so a "REM never enters" regression surfaces in the event log.
    """
    if is_rem(mem):
        return True
    _log_event(mem, "offline_refused", {"op": op_name, "phase": current_state(mem)["phase"]})
    return False


def only_in_rem(op_name: str | None = None):
    """Decorator form of the gate for offline-op functions whose first arg is mem.

    What: wraps ``fn(mem, ...)`` so it returns a logged refusal dict
          ({"fired": False, "refused": "not_in_rem", "op": name}) when called
          outside REM, and runs normally inside REM. ``op_name`` defaults to the
          wrapped function's __name__.
    Why:  lets a migrated op gate itself at its own entry — so it refuses
          regardless of WHICH caller (idle_pulse, scheduler.py, a manual call,
          a future caller) invokes it. The gate travels with the op (root-cause
          gating), not with one registration site.
    """
    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = op_name or getattr(fn, "__name__", "offline_op")

        @functools.wraps(fn)
        def _wrapped(mem: Path, *args: Any, **kwargs: Any) -> Any:
            if not gate_offline_op(Path(mem), name):
                return {"fired": False, "refused": "not_in_rem", "op": name}
            return fn(mem, *args, **kwargs)

        return _wrapped

    return _decorate


def _state_path(mem: Path) -> Path:
    return Path(mem) / "biomimetic" / "rem_state.json"


def _events_path(mem: Path) -> Path:
    return Path(mem) / "biomimetic" / "rem_events.jsonl"


def _now() -> float:
    """Wall-clock epoch seconds. What: the single time source. Why: persisted
    state must round-trip across a restart, so since_ts is wall time, not
    monotonic; callers may pass `now` explicitly (tests, deterministic eval)."""
    return time.time()


def _default_state() -> dict[str, Any]:
    """A fresh WAKE state. What: the seed when no state file exists. Why: a
    never-slept system starts awake with no cycle and no forced trigger."""
    return {
        "phase": WAKE,
        "since_ts": _now(),
        "reason": "init",
        "cycle_id": None,
        "last_wake_reason": None,
        "force_requested": False,
        "review_started_ts": None,
    }


def _log_event(mem: Path, event: str, detail: dict[str, Any]) -> None:
    """Append one transition record to rem_events.jsonl (fail-soft).

    What: mirrors the .ia_events.jsonl / heartbeat ticks.jsonl convention — one
          JSONL line per transition for observability.
    Why:  the in_rem refusals + transitions must be operator-visible (risk-5);
          a log-write failure must never break a transition.
    """
    rec = {"ts": _now(), "event": event, **detail}
    try:
        p = _events_path(mem)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        _log.debug("rem_cycle: event log write failed (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# State read / persist
# ---------------------------------------------------------------------------


def current_state(mem: Path) -> dict[str, Any]:
    """Return the persisted REM state (seeding a fresh WAKE if none exists).

    What: the read every offline op + the rem_status surface consults.
    Why:  the gate. Uses locked_update_json so a concurrent writer never yields
          a half-written file; seeds + persists the default on first read so
          the file always exists after one call.
    """
    p = _state_path(mem)
    p.parent.mkdir(parents=True, exist_ok=True)
    with locked_update_json(p, default=None) as state:
        if not state:
            state.update(_default_state())
        # Return a copy so callers can't mutate the persisted dict out-of-band.
        return dict(state)


def is_rem(mem: Path) -> bool:
    """True iff the system is in REM (or REVIEWING — still resting).

    What: the boolean gate P2's offline ops use to refuse-outside-REM.
    Why:  REVIEWING is a REM sub-state (the snooze gate), so work that was mid-
          batch is still legitimately "in REM" until a true wake (path a/c).
    """
    return current_state(mem)["phase"] in (REM, REVIEWING)


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def enter_rem(mem: Path, reason: str) -> dict[str, Any]:
    """Transition WAKE/REVIEWING -> REM (start or resume a cycle).

    What: sets phase=REM, stamps since_ts, and assigns a cycle_id when starting
          a fresh cycle (a snooze from REVIEWING keeps the same cycle_id). Clears
          the explicit force flag (it has been honored).
    Why:  the single entry point for both the trigger path (should_enter_rem)
          and the explicit rem_sleep_now path; logs the transition.
    """
    p = _state_path(mem)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    with locked_update_json(p, default=None) as state:
        if not state:
            state.update(_default_state())
        was = state.get("phase")
        # Fresh cycle vs. a snooze-resume from REVIEWING (keep cycle_id).
        if was != REVIEWING or not state.get("cycle_id"):
            state["cycle_id"] = uuid.uuid4().hex[:12]
        state["phase"] = REM
        state["since_ts"] = now
        state["reason"] = reason
        state["force_requested"] = False
        state["review_started_ts"] = None
        result = dict(state)
    _log_event(mem, "enter_rem", {"reason": reason, "from": was,
                                  "cycle_id": result["cycle_id"]})
    return result


def wake(mem: Path, reason: str) -> dict[str, Any]:
    """Transition any REM phase -> WAKE (truly end the cycle).

    What: sets phase=WAKE, records last_wake_reason, clears the cycle_id and
          review marker. This is the only transition that ENDS a cycle (paths a
          and c); a natural drop is handled by review()/evaluate(), not here.
    Why:  operator activity (path a) and the max-duration cap (path c) both land
          here; subscribers (P2) checkpoint their cursors before this fires.
    """
    p = _state_path(mem)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    with locked_update_json(p, default=None) as state:
        if not state:
            state.update(_default_state())
        was = state.get("phase")
        ended_cycle = state.get("cycle_id")
        state["phase"] = WAKE
        state["since_ts"] = now
        state["reason"] = reason
        state["last_wake_reason"] = reason
        state["cycle_id"] = None
        state["force_requested"] = False
        state["review_started_ts"] = None
        result = dict(state)
    _log_event(mem, "wake", {"reason": reason, "from": was,
                             "ended_cycle": ended_cycle})
    return result


def review(mem: Path, reason: str) -> dict[str, Any]:
    """Transition REM -> REVIEWING (the snooze gate, Q4 path b).

    What: enters the bounded review window — phase=REVIEWING, stamps
          review_started_ts. The cycle is NOT ended (cycle_id preserved); a
          subsequent evaluate() will snooze back to REM (if idle + work remains)
          or settle into idle REM-rest.
    Why:  the operator's refinement: a natural wake (pressure dropped / work
          drained) must not strand leftover work — surface it, wait briefly, and
          re-sleep if nothing else is asked.
    """
    p = _state_path(mem)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    with locked_update_json(p, default=None) as state:
        if not state:
            state.update(_default_state())
        was = state.get("phase")
        state["phase"] = REVIEWING
        state["since_ts"] = now
        state["review_started_ts"] = now
        state["reason"] = reason
        result = dict(state)
    _log_event(mem, "review", {"reason": reason, "from": was,
                               "cycle_id": result.get("cycle_id")})
    return result


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


# ---------------------------------------------------------------------------
# Observability / IPC-MCP surface (thin — Q1 explicit trigger + the gauge read)
# ---------------------------------------------------------------------------


def subscriber_status(mem: Path) -> list[dict[str, Any]]:
    """Per-subscriber registry status (parity with idle_pulse_status).

    What: for each registered REM offline op (priority order) returns name,
          priority, run/error counts, last error, whether it is due now, and its
          cursor (the resume point). Pure read.
    Why:  rem_status surfaces this so an operator can see WHICH offline ops are
          registered, whether one is failing, and whether work remains (cursor).
    """
    with _rem_subscribers_lock:
        subs = sorted(_rem_subscribers.values(), key=lambda s: (s.priority, s.name))
    rows: list[dict[str, Any]] = []
    for s in subs:
        try:
            due = (s.due_fn is None) or bool(s.due_fn(Path(mem)))
        except Exception:
            due = True
        rows.append({
            "name": s.name,
            "priority": s.priority,
            "due": due,
            "run_count": s.run_count,
            "error_count": s.error_count,
            "last_error": s.last_error or None,
            "last_run_wall": s.last_run_wall or None,
            "cursor_key": s.cursor_key,
            "cursor": (read_cursor(mem, s.cursor_key) if s.cursor_key else None),
        })
    return rows


def rem_status(mem: Path) -> dict[str, Any]:
    """Read the current REM state + the live sleep-pressure breakdown.

    What: the operator-visible health gauge — returns the persisted state, the
          full compute_pressure() reading, the idle/activity signal, AND the
          registered offline-op subscribers (P2: priority + last-run + cursor).
          The read mcp_server.memory_rem_status and the rem_status IPC op wrap.
    Why:  surfaces a stuck-high pressure gauge / a wedged phase / a failing
          subscriber to the operator (risk-1 / risk-5 / risk-6). Pure read.
    """
    state = current_state(mem)
    pressure = sleep_pressure.compute_pressure(mem)
    idle_s = seconds_since_last_activity()
    return {
        "state": state,
        "pressure": pressure,
        "idle_seconds": (round(idle_s, 1) if idle_s is not None else None),
        "idle_gate_s": IDLE_GATE_S,
        "max_duration_s": MAX_DURATION_S,
        "review_wait_s": REVIEW_WAIT_S,
        "subscribers": subscriber_status(mem),
    }


def rem_sleep_now(mem: Path) -> dict[str, Any]:
    """Explicit "sleep now" trigger (Q1) — set the force flag.

    What: the function the rem_sleep_now IPC op / an agent calls to force a
          cycle. Returns {ok, state}. The next tick honors the flag.
    Why:  on-demand cycle independent of pressure/idle (risk-1 mitigation). P1
          only sets the flag (produce-only); the daemon tick applies it.
    """
    state = request_sleep_now(mem)
    return {"ok": True, "state": state}


# --- daemon IPC wiring (thin seam — only active after an operator daemon restart) ---

_OPS_REGISTERED = False


def _handle_rem_status(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: return rem_status for the memory_dir in args (or the root)."""
    mem = Path(args.get("memory_dir")) if args.get("memory_dir") else _default_mem()
    return rem_status(mem)


def _handle_rem_sleep_now(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: flip the explicit force flag (the operator "sleep now")."""
    mem = Path(args.get("memory_dir")) if args.get("memory_dir") else _default_mem()
    return rem_sleep_now(mem)


def _default_mem() -> Path:
    """The memory root used when an IPC caller omits memory_dir.

    Mirrors idle_pulse's _MEM_ROOT (== .../memory). Resolved through
    samia.core.paths.resolve_memory_root (env -> verified-legacy file-position
    -> XDG fallback), so the root is correct in dev, staged-release, and
    site-packages layouts rather than only the dev tree.

    BUGFIX 2026-06-08: a prior parents[4] derivation pointed at
    .../Asthenosphere, one too high; the verified-legacy clause now returns the
    same .../memory the idle_pulse loop reads/writes whenever a real memory
    root is present, and an env-named or XDG root otherwise (no more drive-root
    scribbling when staged).
    """
    return resolve_memory_root()


def register_ops() -> None:
    """Register the two REM IPC ops (idempotent).

    What: wires rem_status (read the gauge) + rem_sleep_now (explicit trigger)
          onto the daemon IPC, mirroring idle_pulse.register_ops.
    Why:  the thin observability/trigger surface (Q1). Called by the daemon at
          startup; inert until an operator-gated daemon restart.
    """
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return
    from samia.runtime.ipc import register_op

    register_op("rem_status", _handle_rem_status)
    register_op("rem_sleep_now", _handle_rem_sleep_now)
    _OPS_REGISTERED = True


# ─────────────────────────────────────────────
# [rem_cycle] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.runtime
# Version:    1.1.0  Updated: 2026-06-07  Status: active
# Phase:      FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P1 + P2)
# Role:       persisted WAKE<->REM state machine + Q1 trigger + Q4 three wake
#             paths (P1) + the offline-op SUBSCRIBER REGISTRY, the run-only-in-REM
#             GATE, and the interruptible cursor-checkpointing DRIVER (P2)
# Depends:    functools, json, os, threading, time, uuid, dataclasses, pathlib;
#             samia.core.atomic_state (locked_update_json + cursor store),
#             samia.runtime.sleep_pressure (compute_pressure)
# Note:       PRODUCE-ONLY — no thread/timer started here. The driver
#             (run_due_subscribers) only runs subscribers WHEN is_rem() and is
#             only ever called by the daemon tick (operator-gated activation).
#             register_offline_op is the home for every offline reconciliation op.
# ─────────────────────────────────────────────
