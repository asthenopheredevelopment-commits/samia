"""samia.runtime.maintenanced — the minimal shippable maintenance daemon.

Layer 1 (Owns / Depends):
    Owns:    the maintenance-process lifecycle for the SAM/IA public release
             (single-instance flock under the memory dir, SIGTERM/SIGINT clean
             stop, a periodic tick that drives REM + sleep-pressure), and the
             ``python -m samia.runtime.maintenanced`` CLI.
    Depends: samia.runtime.scheduler (start/stop time-based jobs),
             samia.runtime.watcher (filesystem watch + debounced regen/index),
             samia.runtime.rem_cycle (the WAKE<->REM state machine + tick),
             samia.runtime.rem_subscribers (the REM-gated offline ops),
             samia.runtime.sleep_pressure (the entry/exit metric),
             samia.core.paths (resolve_memory_root). All of these SHIP in the
             MEMORY-CORE carve; nothing here imports a dev-only module.

Layer 2 (What / Why):
    What: the daemonless release's maintenance spine. It does THREE things and
          nothing else — (1) starts the scheduler (tier decay / idle replay /
          attention gc / sm2 sweep, opencode_drain a no-op stub in the release);
          (2) starts the filesystem watcher (vector index + MEMORY.md regen, the
          latter a no-op skip when compact_index does not ship); (3) drives REM
          on a periodic tick (compute_pressure -> rem_cycle.tick), so heavy
          offline reconciliation runs only inside the idle-gated sleep window.
          ``--oneshot`` runs one pass of all three and exits 0 (the cold-metal
          kit + the test suite use this).
    Why:  daemon.py (the personal-assistant spine: vision/drive/skills/IPC op
          registry) is DEV-ONLY and does not ship. The public release still needs
          its memory plane maintained — decay, consolidation, replay, fact
          extraction — but with NONE of the personal arms. maintenanced is that
          carved-down spine: scheduler + watcher + REM, no IPC, no op registry,
          no model loading at startup. LLM-backed subscribers (e.g. fact_extract)
          load weights LAZILY only if their REM work is due; with
          ASTHENOS_MODEL_AUTOFETCH=0 that load refuses rather than downloading,
          so a bare startup never touches the network.

Entry point:
    python -m samia.runtime.maintenanced [--memory-dir PATH] [--interval S] [--oneshot]

Operator-approved scope (2026-06-12): scheduler + watcher + REM driving ONLY.
NOT a refactor of daemon.py.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from samia.core.paths import resolve_memory_root
from samia.runtime import rem_cycle, scheduler, sleep_pressure, watcher

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# _DEFAULT_INTERVAL_S — What: seconds between maintenance ticks (the REM-drive
#     cadence). Why: REM entry is pressure+idle gated, so this is just how often
#     we re-evaluate; a minute matches the scheduler's own tick granularity and
#     keeps the REM driver responsive without spinning a tight clock.
_DEFAULT_INTERVAL_S = 60.0


# ---------------------------------------------------------------------------
# Logging — stdout, glyph-free, single line
# ---------------------------------------------------------------------------

def _log(msg: str, *args: Any) -> None:
    """Emit one glyph-free line to stdout with a UTC timestamp.

    What: the single logging surface; accepts %-style args so it satisfies the
          ``log_fn(msg, *args)`` contract that scheduler/watcher call with.
    Why:  the release runs under journald/`python -m`; plain stdout lines are
          the lowest-friction sink. No emoji/box-drawing (operator terminal
          corrupts on those) and no file handler to manage.
    """
    text = (msg % args) if args else msg
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stdout.write(f"{stamp} [maintenanced] {text}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# MaintenanceDaemon
# ---------------------------------------------------------------------------

class MaintenanceDaemon:
    """The release maintenance spine: scheduler + watcher + REM driver.

    Single-instance via an exclusive flock on a lockfile UNDER the memory dir
    (so two daemons pointed at the same store cannot both run). Reuses the
    PID-flock APPROACH from daemon.py (os.open + fcntl.flock(LOCK_EX|LOCK_NB))
    without importing daemon.py (it does not ship).
    """

    def __init__(self, memory_dir: Path, interval_s: float = _DEFAULT_INTERVAL_S) -> None:
        self._memory_dir = Path(memory_dir)
        self._interval_s = float(interval_s)
        self._lock_path = self._memory_dir / ".runtime" / "maintenanced.lock"
        self._lock_fd: Optional[int] = None
        self._stop_event = threading.Event()
        self._started_scheduler = False
        self._started_watcher = False

    # -- Single-instance lock -------------------------------------------------

    def acquire_lock(self) -> bool:
        """Take the exclusive flock. Returns False when another instance holds it.

        What: opens (creating) the lockfile under <mem>/.runtime/ and grabs a
              non-blocking exclusive flock, writing our PID for diagnostics.
        Why:  two maintenance daemons on one store would double-drive REM and
              race the scheduler state file. flock(LOCK_NB) is the same
              advisory-lock approach daemon.py's _acquire_pid uses; the lock
              lives under the memory dir so it is per-store, not per-host.
        """
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            _log("another maintenanced holds %s — refusing to start", self._lock_path)
            return False
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        self._lock_fd = fd
        return True

    def release_lock(self) -> None:
        """Unlock + remove the lockfile (best-effort)."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    # -- Signals --------------------------------------------------------------

    def _install_signals(self) -> None:
        """Install SIGTERM/SIGINT -> clean stop (only on the main thread).

        What: maps both terminating signals to set the stop event; skips the
              install when called off the main thread (so tests can run() in a
              worker thread without ValueError).
        Why:  the release runs as a long-lived process under systemd/journald;
              SIGTERM is the normal stop. SIGINT covers a foreground Ctrl-C.
        """
        if threading.current_thread() is not threading.main_thread():
            _log("not on main thread — skipping signal install")
            return
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        _log("received %s — initiating clean shutdown", name)
        self._stop_event.set()

    # -- REM seeding ----------------------------------------------------------

    def _register_rem(self) -> None:
        """Register the REM-gated offline ops as subscribers (idempotent).

        What: mirrors daemon.py's seeding (rem_subscribers.register_rem_subscribers)
              but DELIBERATELY skips rem_cycle.register_ops() — that wires IPC
              ops, and the release has no IPC. We only need the SUBSCRIBERS so the
              REM driver has work to run inside the sleep window.
        Why:  registration is produce-only data (no work runs at register time);
              the driver runs them only when a tick decides we are in REM.
        """
        from samia.runtime.rem_subscribers import register_rem_subscribers
        names = register_rem_subscribers()
        _log("REM subscribers registered: %s", ", ".join(names) if names else "(none)")

    def _rem_tick(self, force: bool = False) -> dict[str, Any]:
        """Drive ONE REM decision-and-apply step using only shipped modules.

        What: computes sleep pressure (so the log carries the gauge), optionally
              sets the explicit force flag (used by --oneshot to guarantee a real
              REM pass on a scratch store), then calls rem_cycle.tick(), which
              consults idleness, decides WAKE/REM, and — only on STAY_REM —
              drives the due offline subscribers under their per-cycle budget.
        Why:  rem_cycle.tick is the single seam the dev daemon's idle_pulse loop
              drives; reusing it keeps the entry/exit/wake-path logic identical to
              the dev path. NO timer is spun inside the tick — the periodic
              cadence is OUR loop (or one call, for --oneshot).
        """
        mem = self._memory_dir
        pressure = sleep_pressure.compute_pressure(mem)
        _log("sleep pressure score=%.4f needed=%s",
             pressure.get("score", 0.0), pressure.get("sleep_needed"))
        if force:
            # request_sleep_now sets the explicit force flag so the next tick
            # enters REM regardless of pressure/idle — guarantees --oneshot runs
            # a real REM pass (and thus drives the due subscribers) on a scratch
            # store, even when the idle gate is unmet.
            rem_cycle.request_sleep_now(mem)
        # activity=False: this is a maintenance tick, not active cognition. The
        # tick's own idle gate (REM_IDLE_GATE_S) still protects against sleeping
        # into a live session; the force flag (oneshot) bypasses it intentionally.
        result = rem_cycle.tick(mem, activity=False)
        _log("REM tick action=%s reason=%s",
             result.get("action"), result.get("reason"))
        return result

    # -- Lifecycle ------------------------------------------------------------

    def start_subsystems(self) -> None:
        """Start scheduler + watcher threads and register REM subscribers."""
        _log("memory dir: %s", self._memory_dir)
        scheduler.start(self._memory_dir, _log)
        self._started_scheduler = True
        watcher.start(self._memory_dir, _log)
        self._started_watcher = True
        self._register_rem()

    def stop_subsystems(self) -> None:
        """Stop scheduler + watcher threads (best-effort, idempotent)."""
        if self._started_watcher:
            try:
                watcher.stop()
            except Exception as exc:
                _log("watcher stop error (non-fatal): %s", exc)
            self._started_watcher = False
        if self._started_scheduler:
            try:
                scheduler.stop()
            except Exception as exc:
                _log("scheduler stop error (non-fatal): %s", exc)
            self._started_scheduler = False

    def run_oneshot(self) -> int:
        """Run ONE pass of scheduler + watcher + REM, then exit 0.

        What: persists a scheduler-state pass (every job fires once on a fresh
              store since last_run_unix=0), dispatches any matured watcher
              actions, and forces+drives one REM tick. Releases the lock and
              returns 0. This is the cold-metal-kit / test entrypoint.
        Why:  a deterministic, thread-free single pass: no long-lived threads to
              join, exercises every seam (scheduler jobs, watcher dispatch, REM
              entry + subscriber drive) against a scratch store in one shot.
        """
        _log("oneshot: memory dir %s", self._memory_dir)
        # 1) scheduler: run one inline scan so scheduler_state.json is written.
        self._scheduler_oneshot()
        # 2) watcher: dispatch any matured actions (none on a fresh store, but
        #    proves the dispatch path imports + runs).
        watcher._dispatch_actions(self._memory_dir, _log)
        # 3) REM: register subscribers, then drive TWO ticks. The first tick
        #    (force=True) transitions WAKE -> REM; the second (now in REM) takes
        #    the STAY_REM branch and DRIVES the due offline subscribers. Running
        #    both proves the full entry + subscriber-drive path end-to-end on the
        #    scratch store (what the cold-metal kit validates).
        self._register_rem()
        first = self._rem_tick(force=True)
        if first.get("action") == "enter_rem":
            self._rem_tick(force=False)
        _log("oneshot complete")
        return 0

    def _scheduler_oneshot(self) -> None:
        """Fire every due scheduler job once inline + persist state.

        What: builds a fresh job table, loads persisted timestamps, fires each
              due job through the scheduler's own _run_job, and saves state —
              all WITHOUT spawning the scheduler thread.
        Why:  --oneshot must write scheduler state (a test assertion) without a
              racy start/sleep/stop on the threaded loop. Reuses the scheduler's
              own table + run + persist functions so behavior matches the loop.
        """
        table = scheduler._make_table()
        scheduler._load_state(self._memory_dir, table)
        now = time.time()
        for job in table:
            elapsed = now - job["last_run_unix"]
            required = max(job["effective_interval_s"], job["throttle_min_s"])
            if elapsed >= required:
                scheduler._run_job(job, self._memory_dir, _log)
        scheduler._save_state(self._memory_dir, table)
        _log("oneshot: scheduler state persisted")

    def run(self) -> int:
        """Start everything and block on the tick loop until a stop signal.

        Returns 0 on clean shutdown, 1 when the single-instance lock is held.
        """
        if not self.acquire_lock():
            return 1
        self._install_signals()
        try:
            self.start_subsystems()
            _log("maintenanced ready — tick interval %.0fs", self._interval_s)
            # The periodic tick loop. wait() returns True on stop, False on
            # timeout; either way we drive a REM tick each cadence (cheap on a
            # quiet store — pressure low, idle gate likely unmet -> stay_wake).
            while not self._stop_event.is_set():
                self._rem_tick(force=False)
                self._stop_event.wait(timeout=self._interval_s)
            _log("stop requested — stopping subsystems")
        finally:
            self.stop_subsystems()
            self.release_lock()
        _log("maintenanced exited cleanly")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m samia.runtime.maintenanced",
        description="SAM/IA release maintenance daemon: scheduler + watcher + REM.",
    )
    p.add_argument(
        "--memory-dir",
        type=str,
        default=None,
        help="memory root (default: resolve_memory_root() — env, legacy, or XDG).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=_DEFAULT_INTERVAL_S,
        help=f"seconds between maintenance ticks (default {_DEFAULT_INTERVAL_S:.0f}).",
    )
    p.add_argument(
        "--oneshot",
        action="store_true",
        help="run one tick of scheduler+watcher+REM and exit 0.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for ``python -m samia.runtime.maintenanced``."""
    args = _build_parser().parse_args(argv)
    if args.memory_dir is not None:
        memory_dir = Path(args.memory_dir).expanduser()
    else:
        memory_dir = resolve_memory_root()
    daemon = MaintenanceDaemon(memory_dir, interval_s=args.interval)
    if args.oneshot:
        if not daemon.acquire_lock():
            return 1
        try:
            return daemon.run_oneshot()
        finally:
            daemon.release_lock()
    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())


# ─────────────────────────────────────────────
# [maintenanced] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.runtime
# Version:    1.0.0  Updated: 2026-06-12  Status: active
# Phase:      SAM/IA public release — minimal maintenance daemon (scheduler +
#             watcher + REM driving ONLY; NOT a refactor of daemon.py).
# Role:       the daemonless release's maintenance spine: single-instance flock
#             under the memory dir, periodic REM tick (compute_pressure ->
#             rem_cycle.tick), SIGTERM/SIGINT clean stop, --oneshot single pass.
# Depends:    argparse, fcntl, os, signal, threading, time, pathlib (stdlib);
#             samia.core.paths (resolve_memory_root); samia.runtime.scheduler,
#             .watcher, .rem_cycle, .rem_subscribers, .sleep_pressure (all SHIP).
# Contract:   NO IPC, NO op registry, NO model loading at startup. LLM arms load
#             lazily only if a REM subscriber's work is due; ASTHENOS_MODEL_
#             AUTOFETCH=0 means zero downloads. NEVER drives a real/live store
#             (callers pass --memory-dir or rely on resolve_memory_root); tests
#             use tempfile scratch stores only.
# ─────────────────────────────────────────────
