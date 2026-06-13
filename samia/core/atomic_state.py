"""atomic_state.py -- crash-safe / race-safe writes for shared JSON state files.

What: atomic_write_json (temp-file + os.replace) and locked_update_json (flock-guarded
      read-modify-write) for the many small JSON state files that hook subscribers touch.
Why:  TOCTOU fix 2026-06-03. hook_idle_pulse.sh and other hooks fire on every tool call
      across up to 8 concurrent Claude Code sessions (HAP). Bare `Path.write_text()` does
      open('w') (TRUNCATE) then write(), so a concurrent reader can see a half-written or
      empty file; and read->mutate->write without a lock loses updates. os.replace() is an
      atomic rename within a filesystem (no truncate window); flock serializes RMW.

Added: 2026-06-03. tier.decay_tick uses its own inline flock+atomic (the hottest path);
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
    finally:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_UN)
        finally:
            lf.close()


# ─────────────────────────────────────────────
# [atomic_state] — File Metadata
# Author:     claude (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-03  Status: active
# Role:       race/crash-safe shared-state JSON writes (TOCTOU fix)
# Depends:    json, os, fcntl (Linux), contextlib, pathlib
# ─────────────────────────────────────────────
