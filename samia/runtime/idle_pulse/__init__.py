"""samia.runtime.idle_pulse — daemon-resident idle-pulse tick loop.

Layer 1 (Owns / Depends):
    Owns:    in-daemon idle-pulse subscriber registry (name -> tick callable +
             cadence), the resident embedding model (loaded ONCE), the
             self-scheduled servicing loop thread, the coalescing nudge flag,
             and the ``idle_pulse_nudge`` / ``idle_pulse_status`` IPC ops.
    Depends: samia.runtime.ipc (register_op), samia.core.vector (_ensure_model
             — the resident model singleton), and the six tick callables
             (idle_replay / gate / auditor / docs_sweep / decay /
             subagent_cleanup).

Layer 2 (What / Why):
    What: Moves the six maintenance ticks that ``hook_idle_pulse.sh`` used to
          run on EVERY tool call — each inside a fresh ``python3 -`` that
          re-imported the samia stack and reloaded an ~880 MB embedding model —
          into the long-lived daemon.  The model loads ONCE and stays resident;
          a background thread services *due* subscribers on a self-scheduled
          cadence; a coalescing nudge op lets per-tool-call activity poke the
          loop between wakes without piling up work.
    Why:  bug_idle_pulse_hook_python_swarm — the per-call model reloads piled
          into a 28.7 GB / loadavg-317 worker swarm under a 14-agent workflow
          (swap thrash, desktop-wide stutter).  Interim fix #1 (flock + 30s
          min-interval guard in the hook) capped concurrency but still
          cold-loaded the model each fire.  This is fix #2
          (FEAT-2026-06-02-idle-pulse-daemon-resident-tick-loop-v01): resident
          model, no python-fork-per-call.

Decisions settled in the approved preplanner:
    Q1  All six ticks move into the daemon; the embedding model is resident.
    Q2  HYBRID trigger (operator override): a self-scheduled servicing loop
        (catches stalls/timeouts) AND a coalescing ``idle_pulse_nudge`` op
        (the event-based cadence).  Both are needed.
    Q3  30s self-schedule loop, configurable via IDLE_PULSE_LOOP_SECONDS.
    Q4  gate_tick has no internal cooldown today; it gets a conservative 15min
        REGISTRY cadence here (configurable) so the 30s loop never runs it
        needlessly — this bounds it regardless of measured per-run cost, so no
        edit to samia.core.gates is required (additive).
    Q5  Ticks pause while the daemon is down (idempotent maintenance, tolerated
        degraded state).  A daemon-down liveness alert is a follow-up (Phase 3)
        layered on the existing availability/observer subsystem.
    Q6  Explicit in-daemon ``name -> (tick_fn, cadence)`` registry; each tick
        keeps its own cadence; new consumers register via register_subscriber.
    Q7  The hook becomes a CHEAP nudge sender (no python/model) — staged as a
        separate file and switched only at the operator-gated Phase 5, after a
        daemon restart, so there is no coverage gap (the #1 guard stays live).

Coalescing: a nudge only WAKES the loop early; the per-subscriber cadence check
is the true coalescer — idle_replay at a 30s cadence cannot run more than once
per 30s no matter how many nudges arrive between wakes.  So a nudge storm costs
only cheap monotonic comparisons, never extra tick work or an unbounded queue.

Entry points used by the daemon:
    register_ops()           — wire the two IPC ops.
    start_idle_pulse_loop()  — seed subscribers + start the servicing thread.
    stop_idle_pulse_loop()   — clean shutdown (join the thread).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("samia.runtime.idle_pulse")

# FEAT-2026-06-03 drag-lag follow-up: cap the embedding backend's CPU threads.
# Putting torch (CPU) in the daemon venv made the resident model real, but torch's
# OpenMP/MKL intra-op pool BUSY-SPINS (~half a core steadily) and spawns ~one
# thread per core. That steady CPU draw perturbed the X compositor's pointer-event
# delivery and surfaced as bursty Atlas drag input (camera frozen ~1s then a
# "snap"). MiniLM embeddings are tiny — 1 thread + passive wait is ample. These
# must be set BEFORE torch is imported (torch reads them at import; the model
# loads lazily in _preload_model). setdefault so an explicit override still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Path resolution — this file lives at
#   .../memory/tools/samia/runtime/idle_pulse/__init__.py
# parents: [0]=idle_pulse [1]=runtime [2]=samia [3]=tools [4]=memory
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_TOOLS_DIR = _HERE.parents[3]   # .../memory/tools (dev layout; unused elsewhere)
# _MEM_ROOT — What: the `mem` arg every tick takes.
# Why: RELEASE-2026-06-11 — parents[4] is layout-fragile (drive root in the
#   staged release, site-packages' parent when installed). resolve_memory_root
#   is env-first, verifies the legacy candidate IS a memory root, XDG-falls-back.
from samia.core.paths import resolve_memory_root as _rmr
_MEM_ROOT = _rmr(create=False)

# ---------------------------------------------------------------------------
# Cadence configuration (seconds).  Each value mirrors the documented
# per-subscriber cooldown; the loop only CALLS a subscriber once its cadence
# has elapsed (and the tick then self-gates internally as a second guard).
# ---------------------------------------------------------------------------

LOOP_SECONDS = float(os.environ.get("IDLE_PULSE_LOOP_SECONDS", "30"))  # Q3
GATE_CADENCE_S = float(os.environ.get("IDLE_PULSE_GATE_CADENCE", "900"))        # Q4: 15min
AUDITOR_CADENCE_S = float(os.environ.get("IDLE_PULSE_AUDITOR_CADENCE", "900"))  # 15min
DOCS_SWEEP_CADENCE_S = float(os.environ.get("IDLE_PULSE_DOCS_CADENCE", "1800")) # 30min
DECAY_CADENCE_S = float(os.environ.get("IDLE_PULSE_DECAY_CADENCE", "21600"))    # 6h
SUBAGENT_CLEANUP_CADENCE_S = float(
    os.environ.get("IDLE_PULSE_SUBAGENT_CLEANUP_CADENCE", "21600")             # 6h
)
# FEAT-2026-06-08 anchor-capture backstop sweep (Q4a): a low-cadence safety net that
# anchors any un-anchored node the write-path capture missed (a future writer, a
# restore/thaw, an import). A no-op at full coverage. Write-path capture stays primary.
ANCHOR_BACKFILL_CADENCE_S = float(
    os.environ.get("ASTHENOS_ANCHOR_BACKFILL_CADENCE", "86400")                # daily
)
# REM entry-decision cadence (P1): a modest cadence — the loop only re-evaluates
# the WAKE<->REM state machine this often. Decoupled from the heavy-work cadence
# (there is no heavy work in P1; P2's subscribers carry their own due-conditions).
REM_CYCLE_CADENCE_S = float(os.environ.get("IDLE_PULSE_REM_CADENCE", "60"))    # 1min

# ---------------------------------------------------------------------------
# Subscriber registry
# ---------------------------------------------------------------------------


@dataclass
class _Subscriber:
    """One idle-pulse subscriber: a tick callable plus its cadence + stats."""

    name: str
    fn: Callable[[Path], Any]
    cadence_s: float
    last_run: float = 0.0          # time.monotonic() of last START (0 = never)
    last_wall: float = 0.0         # epoch wall time of last run (for status)
    run_count: int = 0
    error_count: int = 0
    last_error: str = ""
    last_duration_ms: float = 0.0


_subscribers: dict[str, _Subscriber] = {}
_subscribers_lock = threading.Lock()


def register_subscriber(
    name: str, fn: Callable[[Path], Any], cadence_s: float
) -> _Subscriber:
    """Register (or update) an idle-pulse subscriber.

    What: maps ``name`` to a tick callable taking the memory root Path, run at
          most once per ``cadence_s``.  Re-registering an existing name updates
          its callable/cadence but preserves its accumulated stats (idempotent
          across daemon re-init).
    Why:  Q6 — new maintenance pieces ride the resident loop by registering
          here, exactly like the original idle-pulse subscriber pattern.
    """
    with _subscribers_lock:
        existing = _subscribers.get(name)
        if existing is not None:
            existing.fn = fn
            existing.cadence_s = float(cadence_s)
            return existing
        sub = _Subscriber(name=name, fn=fn, cadence_s=float(cadence_s))
        _subscribers[name] = sub
        return sub


def _lazy(import_path: str, attr: str) -> Callable[[Path], Any]:
    """Return a callable that imports ``import_path`` lazily and calls ``attr``.

    Why: a missing/broken tick module must never break loop startup; the import
         happens at call time inside the per-subscriber try/except.
    """

    def _call(mem: Path) -> Any:
        import importlib

        mod = importlib.import_module(import_path)
        return getattr(mod, attr)(mem)

    return _call


def _seed_default_subscribers() -> None:
    """Seed the default idle-pulse ticks (idempotent).

    RELEASE-2026-06-11: docs_sweep (top-level docs_sweep_tick module) and
    subagent_cleanup (samia.runtime.orchestrator.subagent_ledger_cleanup) are
    NOT part of the memory-core carve, so seeding them produced permanent
    ModuleNotFoundError rows in idle_pulse_status that a fresh user reads as a
    broken install. Both seeds are dropped here, along with the _TOOLS_DIR
    sys.path injection whose only consumer was the dropped docs_sweep module.
    """
    if _subscribers:
        return

    register_subscriber(
        "idle_replay",
        _lazy("samia.core.context_extension", "idle_replay_tick"),
        LOOP_SECONDS,
    )
    register_subscriber("gate", _lazy("samia.core.gates", "gate_tick"), GATE_CADENCE_S)
    register_subscriber(
        "auditor", _lazy("samia.core.auditor", "auditor_tick"), AUDITOR_CADENCE_S
    )
    register_subscriber("decay", _lazy("samia.core.tier", "decay_tick"), DECAY_CADENCE_S)
    # FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P1): the REM
    # state machine's ENTRY-DECISION tick rides this resident loop. It only
    # evaluates whether to enter/exit REM (pressure + idle gate, the three wake
    # paths) — it runs NO offline work (no work-subscribers exist in P1), so it
    # is cheap. REM WORK never trickles here; that is the whole point of the
    # sleep boundary (the heavy ops migrate behind it in P2).
    register_subscriber(
        "rem_cycle", _lazy("samia.runtime.rem_cycle", "tick"), REM_CYCLE_CADENCE_S
    )
    # FEAT-2026-06-08 anchor-capture backstop (Q4a): a daily full-corpus sweep that anchors
    # any un-anchored node the write-path capture missed. ensure_anchor is capture-if-missing
    # (never refresh), so it cannot clobber an eroded node; a no-op at full coverage.
    register_subscriber(
        "anchor_backfill",
        _lazy("samia.core.integrity", "anchor_backfill_tick"),
        ANCHOR_BACKFILL_CADENCE_S,
    )


# ---------------------------------------------------------------------------
# Resident model
# ---------------------------------------------------------------------------

_model_resident = False


def _preload_model() -> None:
    """Load the embedding model ONCE into the daemon's address space.

    What: calls samia.core.vector._ensure_model() so the ~880 MB model singleton
          is resident before the first tick runs.
    Why:  this is the bug's cure — every later tick reuses the resident model
          instead of the fresh process reloading it (the per-call swarm cost).
          Failure is non-fatal: ticks fall back to lazy-loading the model.
    """
    global _model_resident
    try:
        from samia.core import vector

        vector._ensure_model()
        # Cap torch's intra-op pool post-import too (authoritative regardless of
        # whether the env vars above landed before torch's import). Stops the
        # ~half-core busy-spin that was perturbing GUI pointer delivery.
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass
        _model_resident = True
        _log.info("idle_pulse: embedding model resident (loaded once, torch threads=1)")
    except Exception:
        _log.warning(
            "idle_pulse: resident model preload failed (ticks lazy-load)",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Servicing loop + coalescing nudge
# ---------------------------------------------------------------------------

_thread: threading.Thread | None = None
_stop = threading.Event()
_dirty = threading.Event()  # set by idle_pulse_nudge; wakes the loop early
_started_wall = 0.0


def _service_due(force: bool = False) -> int:
    """Run every subscriber whose cadence has elapsed.  Returns #ran.

    Cadence is measured run-START to run-START; ``last_run`` is stamped before
    the call so a slow tick cannot be re-entered (and the loop is single-
    threaded anyway).  Every tick is wrapped — one failure never stops the rest
    or the loop (fail-open, like the original hook).
    """
    ran = 0
    now = time.monotonic()
    for sub in list(_subscribers.values()):
        if not force and sub.last_run and (now - sub.last_run) < sub.cadence_s:
            continue
        sub.last_run = time.monotonic()
        t0 = sub.last_run
        try:
            sub.fn(_MEM_ROOT)
            sub.run_count += 1
            sub.last_wall = time.time()
            sub.last_duration_ms = (time.monotonic() - t0) * 1000.0
            ran += 1
        except Exception as e:  # fail-open per subscriber
            sub.error_count += 1
            sub.last_error = f"{type(e).__name__}: {e}"
            _log.debug(
                "idle_pulse subscriber %s failed (non-fatal)", sub.name, exc_info=True
            )
    return ran


def _loop() -> None:
    """Background servicing loop (Q2 self-schedule + nudge drain).

    Wakes every LOOP_SECONDS OR immediately when nudged, then services due
    subscribers.  The per-subscriber cadence check coalesces nudge storms.
    """
    _preload_model()
    # First pass: exercise the resident model + let each tick self-gate.
    if not _stop.is_set():
        _service_due(force=True)
    while not _stop.is_set():
        # Wait for the next scheduled wake OR a nudge, whichever comes first.
        nudged = _dirty.wait(timeout=LOOP_SECONDS)
        if _stop.is_set():
            break
        if nudged:
            _dirty.clear()
        _service_due()


def start_idle_pulse_loop() -> None:
    """Seed subscribers and start the servicing thread (idempotent)."""
    global _thread, _started_wall
    if _thread is not None and _thread.is_alive():
        return
    _seed_default_subscribers()
    _stop.clear()
    _dirty.clear()
    _started_wall = time.time()
    _thread = threading.Thread(target=_loop, name="samia-idle-pulse", daemon=True)
    _thread.start()
    _log.info(
        "idle_pulse loop started (interval=%.0fs, subscribers=%d)",
        LOOP_SECONDS,
        len(_subscribers),
    )


def stop_idle_pulse_loop() -> None:
    """Signal the loop to stop and join the thread (clean shutdown)."""
    global _thread
    _stop.set()
    _dirty.set()  # wake the loop so it observes the stop promptly
    if _thread is not None:
        _thread.join(timeout=5.0)
        _thread = None
    _log.info("idle_pulse loop stopped")


# ---------------------------------------------------------------------------
# IPC ops
# ---------------------------------------------------------------------------


def _handle_idle_pulse_nudge(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: poke the servicing loop to wake and run due subscribers.

    Many nudges between wakes collapse to one servicing pass (the cadence check
    is the coalescer), so this is safe to call on every tool call.
    """
    _dirty.set()
    return {"nudged": True}


def _handle_idle_pulse_status(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: loop health + per-subscriber last-run times (verification)."""
    now = time.monotonic()
    subs = []
    for s in list(_subscribers.values()):
        subs.append(
            {
                "name": s.name,
                "cadence_s": s.cadence_s,
                "run_count": s.run_count,
                "error_count": s.error_count,
                "last_error": s.last_error or None,
                "last_run_age_s": (round(now - s.last_run, 1) if s.last_run else None),
                "last_run_wall": s.last_wall or None,
                "last_duration_ms": round(s.last_duration_ms, 1),
            }
        )
    return {
        "loop_running": _thread is not None and _thread.is_alive(),
        "loop_seconds": LOOP_SECONDS,
        "model_resident": _model_resident,
        "started_wall": _started_wall or None,
        "subscriber_count": len(_subscribers),
        "subscribers": subs,
    }


_OPS_REGISTERED = False


def register_ops() -> None:
    """Register idle_pulse IPC ops (idempotent)."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return
    from samia.runtime.ipc import register_op

    register_op("idle_pulse_nudge", _handle_idle_pulse_nudge)
    register_op("idle_pulse_status", _handle_idle_pulse_status)
    _OPS_REGISTERED = True


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.idle_pulse
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-02-idle-pulse-daemon-resident-tick-loop-v01 (Phases 1-2)
# Layer:      runtime (daemon-resident background subsystem)
# Role:       daemon-resident idle-pulse tick loop — a subscriber registry + one
#             resident embedding model + a self-scheduled servicing thread + a
#             coalescing nudge, replacing the per-tool-call python/model-reload swarm.
# Stability:  stable — HYBRID self-schedule + coalescing nudge; per-subscriber
#             cadence gating; idempotent start/stop/register.
# ErrorModel: fail-open per subscriber — one tick raising never stops the rest or the
#             loop (counted in error_count/last_error); resident-model preload failure
#             is non-fatal (ticks fall back to lazy-loading the model).
# Depends:    samia.runtime.ipc (register_op), samia.core.vector (resident model),
#             samia.core.paths (resolve_memory_root), the lazy tick modules
#             (context_extension/gates/auditor/tier/rem_cycle/integrity), torch
#             (optional, CPU thread cap); logging/os/sys/threading/time/dataclasses/
#             pathlib/typing (stdlib).
# Exposes:    register_subscriber, register_ops, start_idle_pulse_loop,
#             stop_idle_pulse_loop.
# Note:       bug_idle_pulse_hook_python_swarm (fix #2 — resident model, no fork/call).
# Lines:      435
# --------------------------------------------------------------------------
