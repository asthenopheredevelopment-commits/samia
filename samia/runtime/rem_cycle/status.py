"""samia.runtime.rem_cycle.status — the thin observability / IPC-MCP surface
(Q1 explicit trigger + the gauge read) that mcp_server + the daemon IPC wrap.

Layer 1 (Owns / Depends):
    Owns:    the per-subscriber registry status (subscriber_status), the operator
             health gauge (rem_status), the explicit "sleep now" trigger
             (rem_sleep_now), the daemon IPC handlers (_handle_rem_status /
             _handle_rem_sleep_now) + their default-mem resolver (_default_mem), and
             the idempotent IPC registration (register_ops + the _OPS_REGISTERED flag).
    Depends: .config (the SINGLE shared _rem_subscribers dict + _rem_subscribers_lock
             + the time constants + the re-exported sleep_pressure), .state
             (current_state), .registry (read_cursor), .trigger (request_sleep_now /
             seconds_since_last_activity). sleep_pressure.compute_pressure is the live
             gauge read. resolve_memory_root is reached THROUGH the package facade
             (PATCH SEAM — see _default_mem).

Layer 2 (What / Why):
    What: rem_status returns the persisted state + the full compute_pressure() read
          + the idle signal + the registered subscribers (priority + last-run +
          cursor). rem_sleep_now sets the force flag. register_ops wires the two IPC
          ops onto the daemon, inert until an operator-gated daemon restart.
    Why:  surfaces a stuck-high gauge / a wedged phase / a failing subscriber to the
          operator. Carved out of the 1074-line monolith; it sits near the top of the
          DAG (reads the registry + state + trigger) with no sibling cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    IDLE_GATE_S,
    MAX_DURATION_S,
    REVIEW_WAIT_S,
    _rem_subscribers,
    _rem_subscribers_lock,
    sleep_pressure,
)
from .registry import read_cursor
from .state import current_state
from .trigger import request_sleep_now, seconds_since_last_activity


def subscriber_status(mem: Path) -> list[dict[str, Any]]:
    """Per-subscriber registry status (parity with idle_pulse_status).

    What: for each registered REM offline op (priority order) returns name,
          priority, run/error counts, last error, whether it is due now, and its
          cursor (the resume point). Pure read.
    Why:  rem_status surfaces this so an operator can see WHICH offline ops are
          registered, whether one is failing, and whether work remains (cursor).
    """
    # Snapshot the SINGLE shared registry under its Lock (same dict the driver runs).
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
    # PATCH SEAM: resolve_memory_root is a `mock.patch.object(rem_cycle,
    # "resolve_memory_root", ...)` target (test_paths). Reach it through the package
    # facade so the patch on the package object takes effect here (the pre-split
    # monolith resolved it on the same module the test patches).
    from samia.runtime import rem_cycle as _pkg
    return _pkg.resolve_memory_root()


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
# [Asthenosphere] samia.runtime.rem_cycle.status
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_cycle monolith during
#             modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the thin observability / IPC-MCP surface — subscriber_status (the
#             per-op registry read), rem_status (the operator health gauge),
#             rem_sleep_now (the explicit Q1 trigger), the daemon IPC handlers +
#             _default_mem resolver, and register_ops (idempotent IPC wiring).
# Stability:  stable — behavior byte-identical to the monolith's observability/IPC
#             section; register_ops stays idempotent via _OPS_REGISTERED.
# ErrorModel: subscriber_status swallows a broken due_fn (treats it due) so the gauge
#             never crashes; rem_status / rem_sleep_now are pure reads / a single
#             flag write. register_ops is inert until a daemon restart.
# Depends:    .config (the SINGLE shared registry dict + Lock + time constants +
#             sleep_pressure), .state (current_state), .registry (read_cursor),
#             .trigger (request_sleep_now / seconds_since_last_activity); the package
#             facade (samia.runtime.rem_cycle) for the resolve_memory_root patch seam.
# Exposes:    subscriber_status, rem_status, rem_sleep_now, register_ops,
#             _handle_rem_status, _handle_rem_sleep_now, _default_mem, _OPS_REGISTERED.
# Note:       reads the SAME config-owned registry dict the driver runs (single-owned
#             singleton). PATCH SEAM — _default_mem reaches resolve_memory_root through
#             the package facade so a mock.patch.object on the package takes effect.
#             The IPC ops are inert until an operator-gated daemon restart.
# Lines:      195
# ─────────────────────────────────────────────
