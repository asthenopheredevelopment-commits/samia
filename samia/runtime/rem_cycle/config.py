"""samia.runtime.rem_cycle.config — the package leaf: constants, the SINGLE shared
subscriber registry (dict + Lock), the _RemSubscriber row, and the pure path/time/
event helpers every sibling reads.

Layer 1 (Owns / Depends):
    Owns:    the module-level surface the whole package reads — the logger, the
             phase constants (WAKE/REM/REVIEWING) + the evaluate() action vocabulary
             (STAY_REM/ENTER_REVIEWING/SNOOZE/WAKE_YIELD/WAKE_SAFETY/REST), the two
             tunable time components (IDLE_GATE_S / MAX_DURATION_S) + the review
             window (REVIEW_WAIT_S) + the per-cycle DEFAULT_RUN_BUDGET, the
             _RemSubscriber dataclass, and — THE CRITICAL SINGLETON — the
             process-wide subscriber registry: the _rem_subscribers dict AND the
             _rem_subscribers_lock that guards it. It also owns the pure helpers that
             have no sibling dependency: the rem_state/rem_events/rem_cursors path
             builders (_state_path/_events_path/_cursors_path), the _now wall clock,
             the _default_state seed, and the fail-soft _log_event JSONL appender.
    Depends: samia.core.atomic_state (locked_update_json — used by _log_event's
             siblings, re-exported here for the public surface) — actually only the
             stdlib (functools/json/logging/os/threading/time/uuid/dataclasses/
             pathlib/typing) is imported AT module level for the leaf's OWN code;
             locked_update_json + resolve_memory_root + sleep_pressure are
             re-exported here (the monolith pulled them in at module top and the
             public import surface keeps them reachable as rem_cycle.<name>).

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — it imports NOTHING from its
          siblings, so the constants/registry/primitives live in ONE place and are
          never duplicated. The registry dict + Lock live HERE and ONLY here; every
          accessor in registry.py / driver.py imports THIS dict object and THIS Lock
          and mutates/reads them — the registry is a single shared object so a
          subscriber registered by samia.runtime.rem_subscribers is the SAME row the
          driver runs. Splitting the dict/Lock into two submodules would silently
          fork the registry and break dispatch; co-locating them here prevents that.
    Why:  Phase-B modularization carved the 1074-line monolith by responsibility
          (state / gate / registry / trigger / driver / status). That carve leaves a
          shared base of constants + the registry singleton + the pure path/time/
          event helpers that all of them need; concentrating them here keeps the
          registry single-sourced and the import graph acyclic (config depends on
          nothing inside the package).
"""

from __future__ import annotations

# Re-exported module-top names the monolith pulled in and other code imports THROUGH
# this package (functools/json/logging/os/threading/time/uuid + dataclass/field +
# Path + Any/Callable). `annotations` rides the `from __future__` above. They MUST
# stay importable from the package facade — they live here (the leaf owner).
import functools  # noqa: F401  (re-exported public surface)
import json
import logging
import os
import threading
import time  # noqa: F401  (re-exported public surface)
import uuid  # noqa: F401  (re-exported public surface)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable  # noqa: F401  (re-exported public surface)

# Re-exported dependency callables the monolith imported at module top (kept
# reachable as rem_cycle.locked_update_json / rem_cycle.resolve_memory_root) and the
# aliased dependency module sleep_pressure (reached as rem_cycle.sleep_pressure).
from samia.core.atomic_state import locked_update_json
from samia.core.paths import resolve_memory_root  # noqa: F401  (re-exported)
from samia.runtime import sleep_pressure  # noqa: F401  (re-exported)

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


# THE SINGLETON. What: the process-wide subscriber registry (the dict) + the Lock
#   that guards it. Why: register_rem_subscribers() (in the rem_subscribers package)
#   POPULATES this exact dict, and the driver (driver.py) READS this exact dict;
#   every accessor across the package imports THESE objects (never re-creates a
#   second dict/Lock) so the registry is one shared object. The 4 test files that
#   reach in via `rem_cycle._rem_subscribers` / `rem_cycle._rem_subscribers_lock`
#   patch THIS object (re-exported by __init__).
_rem_subscribers: dict[str, _RemSubscriber] = {}
_rem_subscribers_lock = threading.Lock()


def _cursors_path(mem: Path) -> Path:
    return Path(mem) / "biomimetic" / "rem_cursors.json"


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


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_cycle.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_cycle monolith during modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the package leaf — phase/action constants, the tunable time
#             components + run budget, the _RemSubscriber row, the SINGLE shared
#             subscriber registry (_rem_subscribers dict + _rem_subscribers_lock),
#             and the pure path/time/event helpers every sibling reads. Imports
#             nothing from its siblings (the acyclic base).
# Stability:  stable — pure constants + the registry singleton + side-effect-free
#             helpers; the carve changed no value (every constant byte-identical to
#             the monolith).
# ErrorModel: _log_event is fail-soft (an OSError on the JSONL append is swallowed
#             at debug level, never breaks a transition); the path/time/state-seed
#             helpers are pure and never raise.
# Depends:    functools/json/logging/os/threading/time/uuid/dataclasses/pathlib/
#             typing (stdlib). samia.core.atomic_state (locked_update_json),
#             samia.core.paths (resolve_memory_root), samia.runtime.sleep_pressure —
#             all re-exported for the public surface; the leaf's own code uses only
#             json/os/time/threading.
# Exposes:    WAKE/REM/REVIEWING + the action vocabulary, IDLE_GATE_S/MAX_DURATION_S/
#             REVIEW_WAIT_S/DEFAULT_RUN_BUDGET, _RemSubscriber, _rem_subscribers,
#             _rem_subscribers_lock, _cursors_path/_state_path/_events_path/_now/
#             _default_state/_log_event, and the re-exported deps (locked_update_json,
#             resolve_memory_root, sleep_pressure, the stdlib imports).
# Lines:      240
# Note:       THE REGISTRY IS SINGLE-OWNED HERE. Never re-create _rem_subscribers or
#             _rem_subscribers_lock in another submodule — import THESE objects.
# ─────────────────────────────────────────────
