"""samia.runtime.rem_cycle — the persisted WAKE<->REM sleep state machine.

Layer 1 (Owns / Depends):
    Owns:    the persisted REM state (<mem>/biomimetic/rem_state.json:
             {phase, since_ts, reason, cycle_id, ...}), the gate every offline
             op consults (is_rem / gate_offline_op / only_in_rem), the
             WAKE<->REM<->REVIEWING transitions (enter_rem / wake / review), the
             entry trigger (should_enter_rem), the three-wake-path decision logic
             (evaluate), the offline-op SUBSCRIBER REGISTRY (register_offline_op +
             the SINGLE shared _rem_subscribers dict + Lock), the interruptible
             cursor-checkpointing DRIVER (tick / run_due_subscribers), and the
             rem_status / rem_sleep_now read/trigger surface — split by responsibility
             into six submodules behind this re-export facade (the public import
             surface is byte-for-byte unchanged from the pre-split single module).
    Depends: samia.core.atomic_state (locked_update_json — flock + atomic replace),
             samia.core.paths (resolve_memory_root), and samia.runtime.sleep_pressure
             (compute_pressure — the entry/exit metric). The idle / activity / "now"
             inputs are passed IN (or read from the heartbeat activity log) — NO
             background thread, NO clock.

Layer 2 (What / Why):
    What: a two-state core (WAKE / REM) with a REVIEWING sub-state for the
          operator's "snooze" refinement. Offline ops (P2) refuse to run outside REM;
          the three wake paths (Q4) are modeled as explicit outcomes of evaluate()
          ((a) operator activity -> wake_yield, (b) drain -> reviewing -> snooze|rest,
          (c) max-duration cap -> wake_safety).
    Why:  the single shared SLEEP boundary so heavy offline reconciliation runs in a
          contained, idle-gated window and yields instantly to active cognition. The
          1074-line monolith was split by responsibility (config / state / gate /
          registry / trigger / driver / status) with ZERO behavior change; this facade
          re-exports the FULL public surface so every importer
          (`from samia.runtime.rem_cycle import X`) and every attribute reach-in
          (`rem_cycle._rem_subscribers` / `._rem_subscribers_lock` / `._default_mem`,
          the `mock.patch.object(rem_cycle, "gate_offline_op", ...)` target) is
          unaffected.

THE REGISTRY SINGLETON (critical): the process-wide subscriber registry — the
    _rem_subscribers dict AND the _rem_subscribers_lock that guards it — lives in
    EXACTLY ONE submodule (config). register_rem_subscribers() (in the
    samia.runtime.rem_subscribers package) POPULATES that dict and the driver READS
    it; every accessor (registry.py / driver.py / status.py) imports THAT ONE dict
    object and THAT ONE Lock, so the registry is never forked into two. The four test
    files that reach in via `rem_cycle._rem_subscribers` / `._rem_subscribers_lock`
    (test_merge_consumer, test_merge_consumer_p2, test_passive_sweep,
    test_rem_subscribers) patch THIS one object, re-exported here.

PRODUCE-ONLY: no thread / timer / clock is started here. The driver
(run_due_subscribers) only runs subscribers WHEN is_rem() and is only ever called by
the daemon tick (operator-gated activation).

Public surface re-exported here (byte-for-byte the pre-split module):
    re-exported imports : Any, Callable, Path, annotations, dataclass, field,
                          functools, json, logging, os, threading, time, uuid
    re-exported deps    : locked_update_json, resolve_memory_root, sleep_pressure
    constants           : WAKE, REM, REVIEWING, STAY_REM, ENTER_REVIEWING, SNOOZE,
                          WAKE_YIELD, WAKE_SAFETY, REST, IDLE_GATE_S, MAX_DURATION_S,
                          REVIEW_WAIT_S, DEFAULT_RUN_BUDGET
    functions           : current_state, is_rem, enter_rem, wake, review,
                          gate_offline_op, only_in_rem, register_offline_op,
                          registered_offline_ops, read_cursor, write_cursor,
                          seconds_since_last_activity, is_idle, request_sleep_now,
                          should_enter_rem, work_remains, evaluate, tick,
                          run_due_subscribers, subscriber_status, rem_status,
                          rem_sleep_now, register_ops
Internal names also re-exported for direct test/importer access (NOT in __all__):
    _rem_subscribers, _rem_subscribers_lock (the registry singleton — patched by the
    4 test files), _any_subscriber_work_remaining, _cursor_has_remaining (reached by
    test_rem_subscribers), _default_mem (reached by test_paths).
"""

from __future__ import annotations

# The shared base + the SINGLE registry singleton + the re-exported module-top names
# the monolith pulled in (functools/json/logging/os/threading/time/uuid + dataclass/
# field + Path + Any/Callable) and the re-exported deps (locked_update_json/
# resolve_memory_root/sleep_pressure). `annotations` rides the `from __future__`
# above. _rem_subscribers + _rem_subscribers_lock are re-exported because the 4 test
# files patch them THROUGH this package (`rem_cycle._rem_subscribers`).
from .config import (  # noqa: F401
    Any,
    Callable,
    Path,
    dataclass,
    field,
    functools,
    json,
    logging,
    os,
    threading,
    time,
    uuid,
    locked_update_json,
    resolve_memory_root,
    sleep_pressure,
    WAKE,
    REM,
    REVIEWING,
    STAY_REM,
    ENTER_REVIEWING,
    SNOOZE,
    WAKE_YIELD,
    WAKE_SAFETY,
    REST,
    IDLE_GATE_S,
    MAX_DURATION_S,
    REVIEW_WAIT_S,
    DEFAULT_RUN_BUDGET,
    _rem_subscribers,
    _rem_subscribers_lock,
)

# The persisted state read + the WAKE<->REM<->REVIEWING transitions.
from .state import (  # noqa: F401
    current_state,
    is_rem,
    enter_rem,
    wake,
    review,
)

# The run-only-in-REM gate + its decorator form (gate_offline_op is the
# mock.patch.object target; only_in_rem reaches it through this facade — see gate.py).
from .gate import gate_offline_op, only_in_rem  # noqa: F401

# The subscriber registry API + the cursor store + the work-remaining queries
# (_any_subscriber_work_remaining / _cursor_has_remaining are test-reached:
# rem_cycle._any_subscriber_work_remaining / ._cursor_has_remaining).
from .registry import (  # noqa: F401
    register_offline_op,
    registered_offline_ops,
    read_cursor,
    write_cursor,
    _cursor_has_remaining,
    _any_subscriber_work_remaining,
)

# The idle inputs, the entry trigger (Q1), the work-remains read, and the pure
# three-wake-path decision (Q4 evaluate).
from .trigger import (  # noqa: F401
    seconds_since_last_activity,
    is_idle,
    request_sleep_now,
    should_enter_rem,
    work_remains,
    evaluate,
)

# The offline-work engine: tick + the interruptible driver.
from .driver import tick, run_due_subscribers  # noqa: F401

# The observability / IPC surface (_default_mem is test-reached: rem_cycle._default_mem).
from .status import (  # noqa: F401
    subscriber_status,
    rem_status,
    rem_sleep_now,
    register_ops,
    _default_mem,
)

# __all__ — the LOCALLY-owned PUBLIC names (the 52 the baseline records: the 13
# re-exported imports, the 3 re-exported deps, the 13 constants, and the 23
# functions). The verify script diffs the full public surface (dir() minus underscore
# names) against the baseline; __all__ documents the intended export set and bounds
# `from ... import *` to exactly the pre-split public 52. (The private test/importer-
# reached names above are re-exported but intentionally NOT in __all__, mirroring the
# exemplars.)
__all__ = [
    # re-exported imports
    "Any", "Callable", "Path", "annotations", "dataclass", "field",
    "functools", "json", "logging", "os", "threading", "time", "uuid",
    # re-exported deps
    "locked_update_json", "resolve_memory_root", "sleep_pressure",
    # constants
    "WAKE", "REM", "REVIEWING", "STAY_REM", "ENTER_REVIEWING", "SNOOZE",
    "WAKE_YIELD", "WAKE_SAFETY", "REST", "IDLE_GATE_S", "MAX_DURATION_S",
    "REVIEW_WAIT_S", "DEFAULT_RUN_BUDGET",
    # functions
    "current_state", "is_rem", "enter_rem", "wake", "review",
    "gate_offline_op", "only_in_rem", "register_offline_op",
    "registered_offline_ops", "read_cursor", "write_cursor",
    "seconds_since_last_activity", "is_idle", "request_sleep_now",
    "should_enter_rem", "work_remains", "evaluate", "tick",
    "run_due_subscribers", "subscriber_status", "rem_status",
    "rem_sleep_now", "register_ops",
]


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_cycle
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P1 + P2)
#             + Phase-B modularization: the 1074-line monolith carved into a
#               re-export-preserving package (config/state/gate/registry/trigger/
#               driver/status) with ZERO behavior change; this __init__ re-exports the
#               full public surface so every importer + attribute reach-in is
#               unaffected.
# Layer:      runtime (library helper, no daemon loop)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.runtime.rem_cycle import
#             X` keeps working for all 52 public names; the private helpers the
#             targeted tests reach (_rem_subscribers / _rem_subscribers_lock /
#             _any_subscriber_work_remaining / _cursor_has_remaining / _default_mem)
#             are re-exported too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
# ErrorModel: none here (import-time wiring only); each submodule footer documents its
#             own fail-soft / fail-open / gated posture.
# Depends:    .config, .state, .gate, .registry, .trigger, .driver, .status.
# Exposes:    the public 52 (in __all__) + _rem_subscribers/_rem_subscribers_lock (the
#             registry singleton) + _any_subscriber_work_remaining/_cursor_has_remaining
#             + _default_mem for the tests/importers.
# Lines:      215
# Note:       THE REGISTRY IS SINGLE-OWNED in config (the _rem_subscribers dict +
#             _rem_subscribers_lock); register_rem_subscribers() (rem_subscribers pkg)
#             and the driver mutate/read the SAME object. PRODUCE-ONLY — no thread/
#             timer started; the driver runs only WHEN is_rem() (operator-gated).
# ─────────────────────────────────────────────
