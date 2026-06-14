"""samia.runtime.rem_cycle.registry — the subscriber registry API + the per-
subscriber cursor checkpoint store + the work-remaining queries.

Layer 1 (Owns / Depends):
    Owns:    the registration entry point (register_offline_op) + the ordered read
             (registered_offline_ops), the resumable cursor store (read_cursor /
             write_cursor + the _cursor_has_remaining / _cursors_path readers), and
             the subscriber half of the work-remains signal
             (_any_subscriber_work_remaining).
    Depends: .config — and CRITICALLY the SINGLE shared registry objects living
             there: it imports the SAME _rem_subscribers dict and the SAME
             _rem_subscribers_lock (never re-creates them) so a subscriber it
             registers is the SAME row the driver runs. Also config's _RemSubscriber,
             _cursors_path, _log, and the re-exported locked_update_json.

Layer 2 (What / Why):
    What: register_offline_op maps a name -> a work callable + priority + due-
          condition + cursor_key, idempotent across daemon re-init (re-registering a
          name preserves accumulated stats). The cursor store is the resumable
          progress checkpoint (flock+atomic via locked_update_json).
          _any_subscriber_work_remaining ORs together "any subscriber is due" and
          "any cursor records remaining work" — the subscriber half of work_remains.
    Why:  Q3 — the extensible home for every offline reconciliation op. Carving this
          out of the 1074-line monolith keeps registration + cursors + the work-
          remaining query together (they all touch the registry singleton) and
          acyclic (registry depends only on the config leaf).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import json

from .config import (
    _RemSubscriber,
    _cursors_path,
    _log,
    _rem_subscribers,
    _rem_subscribers_lock,
    locked_update_json,
)


def register_offline_op(
    name: str,
    fn: Callable[[Path], Any],
    priority: int = 100,
    due_condition: Callable[[Path], bool] | None = None,
    cursor_key: str | None = None,
) -> _RemSubscriber:
    """Register (or update) a REM offline-op subscriber.

    What: maps ``name`` to a work callable that the REM driver runs — and ONLY
          runs — while the system is in REM, in ascending ``priority`` order,
          subject to its own ``due_condition``. Re-registering an existing name
          updates its fn/priority/due/cursor but preserves accumulated stats
          (idempotent across daemon re-init), mirroring idle_pulse.
    Why:  Q3 — the extensible home for every offline reconciliation op. Each op
          declares run order (priority) + when it has work (due_condition) +
          where it checkpoints (cursor_key). The whole point of REM: heavy work
          registers HERE and refuses to run outside the sleep window.

    Args:
        name: stable subscriber id (also the rem_status row key).
        fn: fn(mem) -> dict | Any. SHOULD return a dict carrying a
            "work_remaining" bool (and optionally "made_progress") so the
            driver can OR it into work_remains and the cursor reflects progress.
        priority: run order; LOWER runs FIRST (decay before consolidate/replay).
        due_condition: due_fn(mem) -> bool; skipped this cycle when False.
            None => always due once in REM.
        cursor_key: the key under rem_cursors.json this op checkpoints; the
            registry reads it for cursor-remaining work (the work_remains OR-in).
    """
    with _rem_subscribers_lock:
        existing = _rem_subscribers.get(name)
        if existing is not None:
            existing.fn = fn
            existing.priority = int(priority)
            existing.due_fn = due_condition
            existing.cursor_key = cursor_key
            return existing
        sub = _RemSubscriber(
            name=name, fn=fn, priority=int(priority),
            due_fn=due_condition, cursor_key=cursor_key,
        )
        _rem_subscribers[name] = sub
        return sub


def registered_offline_ops() -> list[str]:
    """Names of all registered REM offline ops (priority order). Pure read."""
    with _rem_subscribers_lock:
        subs = sorted(_rem_subscribers.values(), key=lambda s: (s.priority, s.name))
    return [s.name for s in subs]


# ---------------------------------------------------------------------------
# Cursor checkpoint store (per-subscriber resumable progress)
# ---------------------------------------------------------------------------


def read_cursor(mem: Path, key: str) -> dict[str, Any]:
    """Read a subscriber's persisted cursor (the resume point). Empty if none.

    What: returns the dict a subscriber last checkpointed under ``key``.
    Why:  an interrupt (Q4 path a/c) leaves the cursor; the next REM cycle reads
          it to resume mid-batch instead of restarting or skipping work.
    """
    p = _cursors_path(mem)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cur = data.get(key) if isinstance(data, dict) else None
        return dict(cur) if isinstance(cur, dict) else {}
    except (OSError, ValueError):
        return {}


def write_cursor(mem: Path, key: str, cursor: dict[str, Any]) -> None:
    """Checkpoint a subscriber's cursor (race-safe, atomic).

    What: merges ``cursor`` under ``key`` in rem_cursors.json via the established
          flock+atomic-replace pattern.
    Why:  the resume point for interruptible offline work (Q4 / risk-4). A
          subscriber calls this at safe boundaries so an interrupt never strands
          half-applied work — it resumes from the cursor next cycle.
    """
    p = _cursors_path(mem)
    p.parent.mkdir(parents=True, exist_ok=True)
    with locked_update_json(p, default=None) as data:
        if not data:
            data.clear()
        data[key] = dict(cursor)


def _cursor_has_remaining(mem: Path, key: str) -> bool:
    """True iff a subscriber's cursor records remaining work.

    What: a cursor with ``"remaining": <truthy>`` or ``"done": False`` is
          unfinished. Absent/empty cursor => no recorded remaining work.
    Why:  feeds work_remains so evaluate() does not prematurely drain a cycle
          while a subscriber still has cursor-tracked work (replaces P1's
          pressure>0 proxy with a real signal).
    """
    cur = read_cursor(mem, key)
    if not cur:
        return False
    if "remaining" in cur:
        return bool(cur["remaining"])
    if "done" in cur:
        return not bool(cur["done"])
    return False


def _any_subscriber_work_remaining(mem: Path) -> bool:
    """True iff ANY registered subscriber is due OR has cursor-remaining work.

    What: the subscriber half of work_remains — a subscriber whose due_fn is
          True (has work to start) or whose cursor records remaining work (was
          interrupted mid-batch) means REM is not drained.
    Why:  Q4 path b must not snooze→rest while a subscriber still owes work;
          this OR-in replaces P1's pressure-only proxy.
    """
    with _rem_subscribers_lock:
        subs = list(_rem_subscribers.values())
    for sub in subs:
        if sub.cursor_key and _cursor_has_remaining(mem, sub.cursor_key):
            return True
        try:
            if sub.due_fn is None or sub.due_fn(Path(mem)):
                return True
        except Exception:  # a broken due_fn must not crash work_remains
            _log.debug("rem_cycle: due_fn for %s raised (treated not-due)",
                       sub.name, exc_info=True)
    return False


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_cycle.registry
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_cycle monolith during
#             modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the subscriber registry API (register_offline_op / registered_offline_
#             ops), the resumable cursor store (read_cursor / write_cursor /
#             _cursor_has_remaining), and the subscriber half of the work-remains
#             signal (_any_subscriber_work_remaining).
# Stability:  stable — behavior byte-identical to the monolith's registry section.
# ErrorModel: read_cursor is fail-soft (OSError/ValueError -> {});
#             _any_subscriber_work_remaining swallows a broken due_fn (logged, treated
#             not-due) so work_remains never crashes; register/write are race-safe via
#             the lock / locked_update_json.
# Depends:    .config — IMPORTS THE SINGLE SHARED _rem_subscribers dict +
#             _rem_subscribers_lock (never re-creates them) + _RemSubscriber /
#             _cursors_path / _log / locked_update_json; json (stdlib).
# Exposes:    register_offline_op, registered_offline_ops, read_cursor, write_cursor,
#             _cursor_has_remaining, _any_subscriber_work_remaining.
# Note:       SINGLE-OWNED REGISTRY — every mutation/read here hits the one config-
#             owned dict under the one config-owned Lock, so register_rem_subscribers()
#             (rem_subscribers pkg) and the driver share the SAME registry object.
# Lines:      204
# ─────────────────────────────────────────────
