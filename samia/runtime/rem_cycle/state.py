"""samia.runtime.rem_cycle.state — the persisted REM state read + the transitions.

Layer 1 (Owns / Depends):
    Owns:    the read every offline op consults (current_state / is_rem) and the
             three state-mutating transitions (enter_rem / wake / review) that move
             the persisted rem_state.json between WAKE / REM / REVIEWING.
    Depends: .config (the phase constants, the _state_path builder, _now,
             _default_state, _log_event, the re-exported locked_update_json — the
             flock+atomic-replace state pattern). uuid for the cycle_id (re-exported
             via config but used here from stdlib through config).

Layer 2 (What / Why):
    What: a two-state core (WAKE / REM) with a REVIEWING sub-state for the
          operator's "snooze" refinement. current_state seeds + returns a COPY of
          the persisted dict; is_rem is the boolean gate (REVIEWING is still "in
          REM"). enter_rem/wake/review are the only writers of the phase field, each
          under locked_update_json so a concurrent writer never yields a half file.
    Why:  the single shared SLEEP boundary. Splitting the read + transitions out of
          the 1074-line monolith into this submodule keeps the state machine
          legible; it depends only on the config leaf (acyclic).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .config import (
    REM,
    REVIEWING,
    WAKE,
    _default_state,
    _log_event,
    _now,
    _state_path,
    locked_update_json,
)


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


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_cycle.state
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_cycle monolith during
#             modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the persisted REM state read (current_state / is_rem) + the three
#             phase transitions (enter_rem / wake / review). The only writers of the
#             phase field; each runs under locked_update_json (flock+atomic replace).
# Stability:  stable — behavior byte-identical to the monolith's state section.
# ErrorModel: none raised here directly; locked_update_json owns the race-safe write
#             and _log_event is fail-soft (config).
# Depends:    .config (REM/REVIEWING/WAKE, _state_path, _now, _default_state,
#             _log_event, locked_update_json), uuid (stdlib).
# Exposes:    current_state, is_rem, enter_rem, wake, review.
# Note:       REVIEWING is a REM sub-state (is_rem True) — work mid-batch is still
#             "in REM" until a true wake (path a/c). enter_rem keeps the cycle_id on a
#             snooze-resume from REVIEWING; wake is the only cycle-ending transition.
# Lines:      179
# ─────────────────────────────────────────────
