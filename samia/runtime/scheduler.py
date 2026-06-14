"""samia.runtime.scheduler — time-based maintenance job runner.

Layer 1 (Owns / Depends):
    Owns:    start(memory_dir, log_fn) -> None  (spawns the scheduler thread;
                 idempotent — a second call while alive is a no-op).
             stop() -> None  (sets the stop event, joins the thread).
    Depends: json, threading, time, pathlib, typing (stdlib only). Job callables
             are resolved at import via fail-soft late imports — samia.core.tier
             (decay_tick), samia.core.context_extension (idle_replay_tick,
             sm2_sweep_tick), samia.core.attention (gc), samia.core.opencode_drain
             (drain_tick); any that fail to import resolve to None (stubbed).

Layer 2 (What / Why):
    What: runs inside the daemon's scheduler thread. _tick_loop walks a static
          job table on a 60s tick and fires any job whose effective interval has
          elapsed; _run_job invokes the callable, records last_run/outcome, and
          applies backoff. last-run timestamps are persisted to / restored from
          <memory_dir>/.runtime/scheduler_state.json so cooldowns survive a
          daemon restart.
    Why:  maintenance work (decay, replay, GC, outcome drain) must run on long
          cadences without a backlog: the tick coalesces overruns (it fires once
          when due, it does not queue missed runs). Backoff (3 consecutive throws
          -> 4x interval, reset on success) keeps a persistently-failing job from
          hammering the daemon. Resolving callables to None when their module is
          absent lets the scheduler start in a partial install (a stubbed job is
          a logged no-op, not a crash).

Design doc: plans/sam_ia_runtime_design.md, sections 1.3 and 6.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Job wiring — import live callables, stub what is missing
# ---------------------------------------------------------------------------

def _resolve_tier_decay() -> Optional[Callable]:
    try:
        from samia.core.tier import decay_tick
        return decay_tick
    except Exception:
        return None


def _resolve_idle_replay() -> Optional[Callable]:
    try:
        from samia.core.context_extension import idle_replay_tick
        return idle_replay_tick
    except Exception:
        return None


def _resolve_attention_gc() -> Optional[Callable]:
    try:
        from samia.core.attention import gc
        return gc
    except Exception:
        return None


def _resolve_sm2_sweep() -> Optional[Callable]:
    try:
        from samia.core.context_extension import sm2_sweep_tick
        return sm2_sweep_tick
    except Exception:
        return None


def _resolve_opencode_drain() -> Optional[Callable]:
    """What: Import the opencode outcome drain tick.
    Why: FEAT-verified-outcome-writeback -- materializes abyss outcome traces
         into spine nodes so failures (target_state=frozen) resist decay."""
    try:
        from samia.core.opencode_drain import drain_tick
        return drain_tick
    except Exception:
        return None


_TIER_DECAY_FN = _resolve_tier_decay()
_IDLE_REPLAY_FN = _resolve_idle_replay()
_ATTENTION_GC_FN = _resolve_attention_gc()
_SM2_SWEEP_FN = _resolve_sm2_sweep()
_OPENCODE_DRAIN_FN = _resolve_opencode_drain()

# ---------------------------------------------------------------------------
# Job table (module-level, per design doc section 6.2)
# ---------------------------------------------------------------------------

# Each entry is mutated in-place by the tick loop (last_run_unix,
# last_outcome, consecutive_failures, effective_interval_s).

def _make_table() -> list[dict[str, Any]]:
    """Build a fresh job table. Called once per start()."""
    return [
        {
            "name": "tier_decay_tick",
            "callable": _TIER_DECAY_FN,
            "interval_s": 21600,           # 6h
            "throttle_min_s": 21600,
            "last_run_unix": 0.0,
            "last_outcome": None,
            "consecutive_failures": 0,
            "effective_interval_s": 21600,
        },
        {
            "name": "idle_replay_tick",
            "callable": _IDLE_REPLAY_FN,
            "interval_s": 600,             # 10min; function has own gate
            "throttle_min_s": 600,
            "last_run_unix": 0.0,
            "last_outcome": None,
            "consecutive_failures": 0,
            "effective_interval_s": 600,
        },
        {
            "name": "attention_hint_gc",
            "callable": _ATTENTION_GC_FN,
            "interval_s": 300,             # 5min
            "throttle_min_s": 300,
            "last_run_unix": 0.0,
            "last_outcome": None,
            "consecutive_failures": 0,
            "effective_interval_s": 300,
        },
        {
            "name": "memory_md_regen",
            "callable": None,              # stub — watcher owns debounced regen
            "interval_s": 3600,            # 1h fallback
            "throttle_min_s": 3600,
            "last_run_unix": 0.0,
            "last_outcome": None,
            "consecutive_failures": 0,
            "effective_interval_s": 3600,
        },
        {
            "name": "sm2_review_sweep",
            "callable": _SM2_SWEEP_FN,
            "interval_s": 86400,           # 24h; SM-2 schedule is day-grained
            "throttle_min_s": 86400,
            "last_run_unix": 0.0,
            "last_outcome": None,
            "consecutive_failures": 0,
            "effective_interval_s": 86400,
        },
        # What: Drain opencode outcome traces from SAM abyss into spine nodes.
        # Why: FEAT-verified-outcome-writeback -- failure outcomes need
        #      target_state=frozen spine nodes to survive decay and remain
        #      queryable. sam_manager writes to abyss at ~2s poll; 1h drain
        #      ensures outcomes are persisted before we attempt materialization.
        {
            "name": "opencode_outcomes_drain",
            "callable": _OPENCODE_DRAIN_FN,
            "interval_s": 3600,            # 1h
            "throttle_min_s": 3600,
            "last_run_unix": 0.0,
            "last_outcome": None,
            "consecutive_failures": 0,
            "effective_interval_s": 3600,
        },
    ]


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

_BACKOFF_THRESHOLD = 3    # consecutive failures before backoff
_BACKOFF_MULTIPLIER = 4   # interval multiplied by this on backoff


def _state_path(memory_dir: Path) -> Path:
    return memory_dir / ".runtime" / "scheduler_state.json"


def _load_state(memory_dir: Path, table: list[dict]) -> None:
    """Restore last_run_unix from persisted state so cooldowns survive restarts."""
    sp = _state_path(memory_dir)
    if not sp.exists():
        return
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return
    saved = {e["name"]: e for e in data.get("jobs", [])}
    for job in table:
        s = saved.get(job["name"])
        if s:
            job["last_run_unix"] = float(s.get("last_run_unix", 0.0))
            job["last_outcome"] = s.get("last_outcome")


def _save_state(memory_dir: Path, table: list[dict]) -> None:
    """Persist job timestamps so cooldowns survive daemon restarts."""
    sp = _state_path(memory_dir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": time.time(),
        "jobs": [
            {
                "name": j["name"],
                "last_run_unix": j["last_run_unix"],
                "last_outcome": j["last_outcome"],
            }
            for j in table
        ],
    }
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(sp)


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------

_TICK_INTERVAL_S = 60  # seconds between table scans

# Module-level state guarded by _lock
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_memory_dir: Optional[Path] = None
_log_fn: Optional[Callable] = None


def _run_job(job: dict, memory_dir: Path, log_fn: Callable) -> None:
    """Invoke a single job, update its metadata, handle backoff."""
    fn = job["callable"]
    name = job["name"]

    if fn is None:
        # Stubbed job — log and mark success so the tick doesn't retry
        log_fn(f"[scheduler] {name}: stubbed (no-op)")
        job["last_run_unix"] = time.time()
        job["last_outcome"] = "stub"
        job["consecutive_failures"] = 0
        job["effective_interval_s"] = job["interval_s"]
        return

    # RunAndBackoff — What: on success reset the failure counter + restore the base
    #     interval; on throw bump the counter and, at/over the threshold, stretch the
    #     effective interval by the backoff multiplier.
    try:
        result = fn(memory_dir)
        job["last_run_unix"] = time.time()
        job["last_outcome"] = "ok"
        job["consecutive_failures"] = 0
        job["effective_interval_s"] = job["interval_s"]
        log_fn(f"[scheduler] {name}: ok — {_summarize(result)}")
    except Exception as exc:
        job["last_run_unix"] = time.time()
        job["last_outcome"] = f"error: {exc}"
        job["consecutive_failures"] += 1
        if job["consecutive_failures"] >= _BACKOFF_THRESHOLD:
            job["effective_interval_s"] = job["interval_s"] * _BACKOFF_MULTIPLIER
            log_fn(
                f"[scheduler] {name}: FAILED {job['consecutive_failures']}x "
                f"consecutively, backing off to {job['effective_interval_s']}s — {exc}"
            )
        else:
            log_fn(f"[scheduler] {name}: error ({job['consecutive_failures']}/"
                   f"{_BACKOFF_THRESHOLD}) — {exc}")
    # RunAndBackoff — Why: last_run_unix is stamped on BOTH paths so a failing job still
    #     respects its cooldown (a throw does not busy-retry every tick); stretching the
    #     interval after _BACKOFF_THRESHOLD consecutive throws keeps a broken job from
    #     hammering the daemon, and any success resets it back to the base cadence.


def _summarize(result: Any) -> str:
    """One-line summary of a job result for the log."""
    if result is None:
        return "(no return value)"
    if isinstance(result, dict):
        # Show a compact key=value for small dicts
        parts = [f"{k}={v}" for k, v in list(result.items())[:5]]
        return ", ".join(parts)
    return repr(result)[:120]


def _tick_loop() -> None:
    """Main loop for the scheduler thread."""
    memory_dir = _memory_dir
    log_fn = _log_fn
    assert memory_dir is not None and log_fn is not None

    table = _make_table()
    _load_state(memory_dir, table)

    log_fn(f"[scheduler] started — {len(table)} jobs, tick every {_TICK_INTERVAL_S}s")
    for job in table:
        status = "live" if job["callable"] is not None else "stubbed"
        log_fn(f"[scheduler]   {job['name']}: {status}, interval={job['interval_s']}s")

    # FireDueJobs — What: each tick, fire every job whose elapsed time has reached its
    #     required interval, then persist state once if anything fired; wait on the stop
    #     event between ticks.
    while not _stop_event.is_set():
        now = time.time()
        fired_any = False
        for job in table:
            elapsed = now - job["last_run_unix"]
            required = max(job["effective_interval_s"], job["throttle_min_s"])
            if elapsed >= required:
                _run_job(job, memory_dir, log_fn)
                fired_any = True

        if fired_any:
            try:
                _save_state(memory_dir, table)
            except Exception as exc:
                log_fn(f"[scheduler] failed to persist state: {exc}")

        # Sleep in short increments so stop() is responsive
        _stop_event.wait(timeout=_TICK_INTERVAL_S)
    # FireDueJobs — Why: required = max(effective, throttle_min) so backoff can only ever
    #     lengthen the cadence, never undercut a job's hard minimum; state is saved only on
    #     a tick that fired (cheap idle ticks), and waiting on _stop_event (not sleep) lets
    #     stop() interrupt the wait so the thread joins promptly.

    log_fn("[scheduler] stopped")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(memory_dir, log_fn: Callable) -> None:
    """Spawn the scheduler thread. Idempotent — second call is a no-op."""
    memory_dir = Path(memory_dir)   # accept str | Path
    global _thread, _memory_dir, _log_fn

    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop_event.clear()
        _memory_dir = memory_dir
        _log_fn = log_fn
        _thread = threading.Thread(
            target=_tick_loop, name="samia-scheduler", daemon=True
        )
        _thread.start()


def stop() -> None:
    """Signal the scheduler to stop and wait for the thread to join."""
    global _thread

    with _lock:
        t = _thread
    if t is None:
        return
    _stop_event.set()
    t.join(timeout=10)
    with _lock:
        _thread = None


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.scheduler
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD26 (runtime daemon) + FEAT-verified-outcome-writeback
#             (opencode_outcomes_drain job added to the table)
# Layer:      runtime (long-lived process; daemon scheduler thread)
# Role:       time-based maintenance job runner — a daemon thread that walks a static
#             job table on a 60s tick, fires each job whose effective interval has
#             elapsed (coalescing overruns), persists last-run cooldowns across
#             restarts, and backs off a job that throws 3x consecutively.
# Stability:  stable -- 60s tick over a static job table; coalesced overruns,
#             persisted cooldowns, 3x-fail -> 4x-interval backoff.
# ErrorModel: job callables resolve to None when their module is absent (a stubbed
#             job is a logged no-op). A job throw is caught per-run and drives
#             backoff; state-persist and individual jobs never crash the tick loop.
# Depends:    json, threading, time, pathlib, typing (stdlib).
#             samia.core.tier / context_extension / attention / opencode_drain
#             (all OPTIONAL, fail-soft late imports at module load).
# Exposes:    start, stop.
# Lines:      385
# --------------------------------------------------------------------------
