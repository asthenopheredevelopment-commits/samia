"""samia.core.atomic_state — crash-safe / race-safe writes for shared JSON state files.

Layer 1 (Owns / Depends):
    Owns:    atomic_write_json — temp-file + os.replace whole-file write.
             locked_update_json — flock-guarded read-modify-write context manager.
    Depends: stdlib only (json, os, fcntl [Linux], contextlib, pathlib).
Layer 2 (What / Why):
    What: atomic_write_json serializes obj and swaps it in atomically; locked_update_json
          takes an exclusive flock on a sidecar `<path>.lock`, yields the parsed state for
          in-place mutation, and writes it back atomically on a clean exit.
    Why:  TOCTOU fix 2026-06-03. hook_idle_pulse.sh and other hooks fire on every tool call
          across up to 8 concurrent Claude Code sessions (HAP). Bare `Path.write_text()` does
          open('w') (TRUNCATE) then write(), so a concurrent reader can see a half-written or
          empty file; and read->mutate->write without a lock loses updates. os.replace() is an
          atomic rename within a filesystem (no truncate window); flock serializes RMW.

Layer 3 (Changelog):
    Added 2026-06-03. tier.decay_tick uses its own inline flock+atomic (the hottest path);
    lower-frequency hook writers should adopt these helpers.
"""
from __future__ import annotations

import json as _json
import os as _os
import fcntl as _fcntl
from contextlib import contextmanager
from pathlib import Path


def atomic_write_json(path, obj, indent=2) -> None:
    """Serialize obj to JSON and write it to `path` ATOMICALLY (temp + os.replace).

    Eliminates the write_text() truncate-then-write window: a concurrent reader sees
    either the old complete file or the new complete file, never a partial one.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{_os.getpid()}")
    tmp.write_text(_json.dumps(obj, indent=indent), encoding="utf-8")
    _os.replace(tmp, path)  # atomic within one filesystem


@contextmanager
def locked_update_json(path, default=None, blocking=True):
    """flock-guarded read-modify-write of a JSON state file.

    Usage:
        with locked_update_json(p, default={}) as state:
            state["k"] = v          # mutate in place
        # written atomically on clean exit; lock released

    With blocking=False, raises BlockingIOError if another holder has the lock (caller
    decides whether to skip). The lock is a sidecar `<path>.lock` file.
    """
    path = Path(path)
    lock_path = path.with_name(f"{path.name}.lock")
    lf = open(lock_path, "w")
    # LockedRMW — What: hold the exclusive flock across read -> yield -> atomic write-back.
    try:
        _fcntl.flock(lf, _fcntl.LOCK_EX | (0 if blocking else _fcntl.LOCK_NB))
        state = default if default is not None else {}
        if path.exists():
            try:
                state = _json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                state = default if default is not None else {}
        yield state
        atomic_write_json(path, state)
    # LockedRMW — Why: the lock must span the WHOLE read-modify-write, not just the write,
    #     or two holders interleave a read each and the later write-back clobbers the other's
    #     update; a corrupt/missing file falls back to `default` so a first run still proceeds.
    finally:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_UN)
        finally:
            lf.close()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.atomic_state
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Added 2026-06-03 (TOCTOU fix for concurrent HAP hook writers).
# Layer:      core (pure library, no daemon dependency)
# Role:       crash/race-safe shared-JSON write primitives — atomic whole-file swap +
#             flock-guarded read-modify-write for concurrent HAP hook writers.
# Stability:  stable -- race/crash-safe shared-state JSON write primitives.
# ErrorModel: a corrupt/missing state file degrades to `default` (no raise);
#             locked_update_json(blocking=False) raises BlockingIOError when the
#             lock is held so the caller can choose to skip; atomic_write_json is
#             whole-file (a partial temp write never replaces the live file).
# Depends:    json, os, fcntl (Linux), contextlib, pathlib (stdlib).
# Exposes:    atomic_write_json, locked_update_json.
# Lines:      92
# --------------------------------------------------------------------------
