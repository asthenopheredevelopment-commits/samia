"""samia.runtime.idle_pulse.test_idle_pulse — tests for samia.runtime.idle_pulse: registry, cadence gating, coalescing,
fail-open isolation, and IPC op shapes.

These tests never load the embedding model: they exercise the registry +
servicing logic with synthetic subscribers, so they are fast and side-effect
free.  FEAT-2026-06-02-idle-pulse-daemon-resident-tick-loop-v01.
"""

from __future__ import annotations

import time

import samia.runtime.idle_pulse as ip


def _reset_registry() -> None:
    """Clear the module-global subscriber registry between tests."""
    with ip._subscribers_lock:
        ip._subscribers.clear()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_subscriber_adds() -> None:
    _reset_registry()
    calls: list = []
    ip.register_subscriber("a", lambda mem: calls.append(mem), 30.0)
    assert "a" in ip._subscribers
    assert ip._subscribers["a"].cadence_s == 30.0


def test_reregister_preserves_stats() -> None:
    _reset_registry()
    sub = ip.register_subscriber("a", lambda mem: None, 30.0)
    sub.run_count = 7
    sub.last_run = 123.0
    # Re-register with a new cadence/callable — stats must survive.
    again = ip.register_subscriber("a", lambda mem: None, 999.0)
    assert again is sub
    assert again.run_count == 7
    assert again.cadence_s == 999.0


# ---------------------------------------------------------------------------
# Cadence gating
# ---------------------------------------------------------------------------


def test_service_due_runs_never_run_subscriber() -> None:
    _reset_registry()
    ran: list = []
    ip.register_subscriber("a", lambda mem: ran.append(1), 1000.0)
    n = ip._service_due()
    assert n == 1
    assert ran == [1]


def test_service_due_skips_recently_run() -> None:
    _reset_registry()
    ran: list = []
    sub = ip.register_subscriber("a", lambda mem: ran.append(1), 1000.0)
    sub.last_run = time.monotonic()  # just ran
    n = ip._service_due()
    assert n == 0
    assert ran == []


def test_service_due_force_runs_all() -> None:
    _reset_registry()
    ran: list = []
    sub = ip.register_subscriber("a", lambda mem: ran.append(1), 1000.0)
    sub.last_run = time.monotonic()  # would normally be skipped
    n = ip._service_due(force=True)
    assert n == 1
    assert ran == [1]


def test_service_due_runs_due_after_cadence() -> None:
    _reset_registry()
    ran: list = []
    sub = ip.register_subscriber("a", lambda mem: ran.append(1), 0.0)  # always due
    sub.last_run = time.monotonic() - 5.0
    n = ip._service_due()
    assert n == 1


# ---------------------------------------------------------------------------
# Fail-open isolation
# ---------------------------------------------------------------------------


def test_subscriber_failure_is_isolated() -> None:
    _reset_registry()
    ran: list = []

    def boom(mem):
        raise RuntimeError("kaboom")

    ip.register_subscriber("bad", boom, 1000.0)
    ip.register_subscriber("good", lambda mem: ran.append(1), 1000.0)
    n = ip._service_due()
    # good still ran; bad recorded an error; the loop did not raise.
    assert ran == [1]
    assert n == 1  # only the good one counts as a successful run
    assert ip._subscribers["bad"].error_count == 1
    assert "RuntimeError" in ip._subscribers["bad"].last_error


# ---------------------------------------------------------------------------
# IPC op shapes
# ---------------------------------------------------------------------------


def test_nudge_sets_dirty() -> None:
    _reset_registry()
    ip._dirty.clear()
    out = ip._handle_idle_pulse_nudge({})
    assert out == {"nudged": True}
    assert ip._dirty.is_set()
    ip._dirty.clear()


def test_status_shape() -> None:
    _reset_registry()
    ip.register_subscriber("a", lambda mem: None, 30.0)
    out = ip._handle_idle_pulse_status({})
    assert set(out) >= {
        "loop_running",
        "loop_seconds",
        "model_resident",
        "subscriber_count",
        "subscribers",
    }
    assert out["subscriber_count"] == 1
    assert out["subscribers"][0]["name"] == "a"
    assert out["subscribers"][0]["cadence_s"] == 30.0


def test_seed_default_subscribers_registers_defaults() -> None:
    _reset_registry()
    ip._seed_default_subscribers()
    names = set(ip._subscribers)
    # Maintenance ticks plus the REM entry-decision tick
    # (FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 P1): the REM
    # state machine's enter/exit evaluation rides this loop (it runs NO offline
    # work in P1 — that is the whole point of the sleep boundary).
    # RELEASE-2026-06-11: docs_sweep and subagent_cleanup are no longer seeded —
    # their modules (top-level docs_sweep_tick and runtime.orchestrator) are not
    # in the memory-core carve, so seeding them produced permanent error rows.
    assert names == {
        "idle_replay",
        "gate",
        "auditor",
        "decay",
        "rem_cycle",
        "anchor_backfill",
    }
    # idle_replay rides the loop cadence; gate carries the conservative 15min.
    assert ip._subscribers["idle_replay"].cadence_s == ip.LOOP_SECONDS
    assert ip._subscribers["gate"].cadence_s == ip.GATE_CADENCE_S
    assert ip._subscribers["rem_cycle"].cadence_s == ip.REM_CYCLE_CADENCE_S


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)


# [Asthenosphere] samia.runtime.idle_pulse.test_idle_pulse
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-02-idle-pulse-daemon-resident-tick-loop-v01
# Layer:      test (pytest)
# Role:       tests for samia.runtime.idle_pulse — subscriber register/re-register stat preservation, cadence gating (due/skip/force), fail-open isolation of a raising subscriber, IPC nudge/status op shapes, default-subscriber seeding (post-RELEASE carve)
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    pytest + samia.runtime.idle_pulse
# Exposes:    — (test module)
# Lines:      197
# ------------------------------------------------------------------------------
