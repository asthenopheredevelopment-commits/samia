"""samia.runtime.watcher -- filesystem watcher for the SAM/IA memory runtime.

Monitors nodes/, chains/, pool/, and MEMORY.md for external writes and
schedules debounced maintenance jobs (vector_index_incremental,
memory_md_regen) via an internal action queue.

Backends (tried in order):
  1. inotify_simple  -- best latency, lowest CPU
  2. pyinotify       -- fallback inotify wrapper
  3. polling (mtime) -- universal fallback, 2s scan interval

Public API (called by samia.runtime.daemon):
  start(memory_dir, log_fn) -> None   -- spawns watcher + worker threads
  stop()                    -> None   -- joins threads, releases resources

Loop prevention:
  with suppress_self(path): ...       -- ignores events on path for 5s

Design doc: plans/sam_ia_runtime_design.md, sections 1.2, 4.3, 6.1.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Action queue -- debounced job scheduler
# ---------------------------------------------------------------------------

# Each entry: {action_name: fire_after_timestamp}
# Newer events overwrite older ones (the debounce).
_action_queue: dict[str, float] = {}
_queue_lock = threading.Lock()


def _schedule(action: str, delay_s: float) -> None:
    """Enqueue an action to fire after *delay_s* seconds from now."""
    fire_at = time.monotonic() + delay_s
    with _queue_lock:
        _action_queue[action] = fire_at


# ---------------------------------------------------------------------------
# Self-write suppression (loop prevention)
# ---------------------------------------------------------------------------

_suppress_set: dict[str, float] = {}
_suppress_lock = threading.Lock()
_SUPPRESS_TTL = 5.0


@contextlib.contextmanager
def suppress_self(path: str | Path):
    """Context manager: events on *path* are ignored for 5 seconds."""
    key = str(Path(path).resolve())
    with _suppress_lock:
        _suppress_set[key] = time.monotonic() + _SUPPRESS_TTL
    try:
        yield
    finally:
        pass  # entry expires naturally via TTL


def _is_suppressed(path: str | Path) -> bool:
    key = str(Path(path).resolve())
    with _suppress_lock:
        expires = _suppress_set.get(key)
        if expires is None:
            return False
        if time.monotonic() < expires:
            return True
        del _suppress_set[key]
        return False


def _gc_suppress() -> None:
    """Remove expired entries from the suppression set."""
    now = time.monotonic()
    with _suppress_lock:
        expired = [k for k, v in _suppress_set.items() if now >= v]
        for k in expired:
            del _suppress_set[k]


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

def _classify(path: Path, memory_dir: Path, log_fn: Callable) -> None:
    """Classify a changed file and schedule the appropriate action(s)."""
    if _is_suppressed(path):
        return

    rel = path.relative_to(memory_dir)
    parts = rel.parts

    # MEMORY.md -- ignore (we generate it; loop prevention)
    if rel.name == "MEMORY.md" and len(parts) == 1:
        return

    # nodes/*.md
    if len(parts) == 2 and parts[0] == "nodes" and rel.suffix == ".md":
        log_fn("watcher: node changed: %s", rel)
        _schedule("vector_index_incremental", 30.0)
        _schedule("memory_md_regen", 60.0)
        return

    # chains/*.json
    if len(parts) == 2 and parts[0] == "chains" and rel.suffix == ".json":
        log_fn("watcher: chain changed: %s", rel)
        _schedule("memory_md_regen", 60.0)
        return

    # pool/*
    if len(parts) >= 2 and parts[0] == "pool":
        log_fn("watcher: pool changed: %s", rel)
        return


# ---------------------------------------------------------------------------
# Optional dispatch backends -- resolved once, may be absent in the release
# ---------------------------------------------------------------------------

# What: resolve compact_index.regenerate (the MEMORY.md regenerator) once at
#       import, mirroring scheduler's _resolve_* Optional-callable pattern.
# Why:  core/compact_index.py is a DEV-ONLY module excluded from the public
#       MEMORY-CORE carve (it is not in port_samia.sh's manifest). Importing it
#       lazily inside the dispatch loop meant every node change fired an
#       ImportError that was caught and logged as a per-action "failed" line --
#       noisy and misleading (a deliberately-absent optional feature is not a
#       failure). Resolving once lets the dispatcher SKIP the action with a
#       single debug log when the module does not ship, and stay fail-soft.

def _resolve_compact_regenerate() -> Callable | None:
    """Import compact_index.regenerate, or None when it does not ship."""
    try:
        from samia.core.compact_index import regenerate
        return regenerate
    except Exception:
        return None


_COMPACT_REGENERATE_FN = _resolve_compact_regenerate()
_compact_skip_logged = False  # one-time debug-log latch for the absent case


# ---------------------------------------------------------------------------
# Action dispatcher -- fires matured actions
# ---------------------------------------------------------------------------

def _dispatch_actions(memory_dir: Path, log_fn: Callable) -> None:
    """Fire all actions whose timestamps have passed."""
    global _compact_skip_logged
    now = time.monotonic()
    ready: list[str] = []
    with _queue_lock:
        for action, fire_at in list(_action_queue.items()):
            if now >= fire_at:
                ready.append(action)
                del _action_queue[action]

    for action in ready:
        try:
            if action == "vector_index_incremental":
                log_fn("watcher: firing vector_index_incremental")
                from samia.core.vector import build
                build(memory_dir, rebuild=False)
            elif action == "memory_md_regen":
                # What: skip (not fail) when compact_index does not ship.
                # Why: in the public release the MEMORY.md regenerator is absent
                #      by design; a debounced regen request for an unavailable
                #      backend is a no-op, logged once at debug, never a per-fire
                #      ERROR line.
                if _COMPACT_REGENERATE_FN is None:
                    if not _compact_skip_logged:
                        log_fn("watcher: memory_md_regen unavailable "
                               "(compact_index not in this build) — skipping")
                        _compact_skip_logged = True
                    continue
                log_fn("watcher: firing memory_md_regen")
                with suppress_self(memory_dir / "MEMORY.md"):
                    _COMPACT_REGENERATE_FN(memory_dir)
            else:
                log_fn("watcher: unknown action %s — skipped", action)
        except Exception as exc:
            log_fn("watcher: action %s failed: %s", action, exc)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_BACKEND: str = "none"


def _detect_backend() -> str:
    """Return the best available backend name."""
    try:
        import inotify_simple  # noqa: F401
        return "inotify_simple"
    except ImportError:
        pass
    try:
        import pyinotify  # noqa: F401
        return "pyinotify"
    except ImportError:
        pass
    return "polling"


_BACKEND = _detect_backend()


# ---------------------------------------------------------------------------
# Backend: inotify_simple
# ---------------------------------------------------------------------------

def _watch_inotify_simple(memory_dir: Path, log_fn: Callable,
                          stop_event: threading.Event) -> None:
    """Watch using inotify_simple (lowest overhead)."""
    import inotify_simple  # type: ignore[import-untyped]

    inotify = inotify_simple.INotify()
    flags = inotify_simple.flags.CLOSE_WRITE | inotify_simple.flags.MOVED_TO

    wd_map: dict[int, Path] = {}
    nodes_dir = memory_dir / "nodes"
    chains_dir = memory_dir / "chains"
    pool_dir = memory_dir / "pool"

    for d in (nodes_dir, chains_dir, pool_dir):
        if d.is_dir():
            wd = inotify.add_watch(str(d), flags)
            wd_map[wd] = d
    # MEMORY.md lives in memory_dir itself
    if memory_dir.is_dir():
        wd = inotify.add_watch(str(memory_dir), flags)
        wd_map[wd] = memory_dir

    while not stop_event.is_set():
        events = inotify.read(timeout=1000)
        for ev in events:
            parent = wd_map.get(ev.wd)
            if parent is None or not ev.name:
                continue
            full = parent / ev.name
            _classify(full, memory_dir, log_fn)

    inotify.close()


# ---------------------------------------------------------------------------
# Backend: pyinotify
# ---------------------------------------------------------------------------

def _watch_pyinotify(memory_dir: Path, log_fn: Callable,
                     stop_event: threading.Event) -> None:
    """Watch using pyinotify."""
    import pyinotify  # type: ignore[import-untyped]

    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO

    class Handler(pyinotify.ProcessEvent):
        def process_default(self, event):
            if event.pathname:
                _classify(Path(event.pathname), memory_dir, log_fn)

    handler = Handler()
    notifier = pyinotify.Notifier(wm, handler, timeout=1000)

    for d in (memory_dir / "nodes", memory_dir / "chains",
              memory_dir / "pool", memory_dir):
        if d.is_dir():
            wm.add_watch(str(d), mask, rec=False)

    while not stop_event.is_set():
        if notifier.check_events(timeout=1000):
            notifier.read_events()
            notifier.process_events()

    notifier.stop()


# ---------------------------------------------------------------------------
# Backend: polling (universal fallback)
# ---------------------------------------------------------------------------

def _watch_polling(memory_dir: Path, log_fn: Callable,
                   stop_event: threading.Event) -> None:
    """Poll mtime every 2 seconds. No external deps required."""
    POLL_INTERVAL = 2.0

    def _scan_targets() -> dict[str, float]:
        """Build {path_str: mtime} for all watched files."""
        state: dict[str, float] = {}
        nodes_dir = memory_dir / "nodes"
        if nodes_dir.is_dir():
            for p in nodes_dir.glob("*.md"):
                try:
                    state[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        chains_dir = memory_dir / "chains"
        if chains_dir.is_dir():
            for p in chains_dir.glob("*.json"):
                try:
                    state[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        pool_dir = memory_dir / "pool"
        if pool_dir.is_dir():
            for p in pool_dir.iterdir():
                try:
                    state[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        mem_md = memory_dir / "MEMORY.md"
        if mem_md.exists():
            try:
                state[str(mem_md)] = mem_md.stat().st_mtime
            except OSError:
                pass
        return state

    prev = _scan_targets()

    while not stop_event.is_set():
        stop_event.wait(POLL_INTERVAL)
        if stop_event.is_set():
            break
        curr = _scan_targets()
        # Detect new or modified files
        for path_str, mtime in curr.items():
            old_mtime = prev.get(path_str)
            if old_mtime is None or mtime > old_mtime:
                _classify(Path(path_str), memory_dir, log_fn)
        prev = curr


# ---------------------------------------------------------------------------
# Worker thread -- scans action queue every 5s
# ---------------------------------------------------------------------------

def _worker_loop(memory_dir: Path, log_fn: Callable,
                 stop_event: threading.Event) -> None:
    """Periodically dispatch matured actions."""
    SCAN_INTERVAL = 5.0
    while not stop_event.is_set():
        stop_event.wait(SCAN_INTERVAL)
        if stop_event.is_set():
            break
        _gc_suppress()
        _dispatch_actions(memory_dir, log_fn)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_stop_event: threading.Event | None = None
_watcher_thread: threading.Thread | None = None
_worker_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(memory_dir: str | Path, log_fn: Callable) -> None:
    """Spawn watcher and worker threads.

    Parameters
    ----------
    memory_dir : path to the memory root directory
    log_fn     : callable(msg, *args) for structured logging
    """
    global _stop_event, _watcher_thread, _worker_thread

    memory_dir = Path(memory_dir)
    _stop_event = threading.Event()

    log_fn("watcher: backend=%s", _BACKEND)

    # Choose backend
    if _BACKEND == "inotify_simple":
        target = _watch_inotify_simple
    elif _BACKEND == "pyinotify":
        target = _watch_pyinotify
    else:
        target = _watch_polling

    _watcher_thread = threading.Thread(
        target=target,
        args=(memory_dir, log_fn, _stop_event),
        name="samia-watcher",
        daemon=True,
    )
    _worker_thread = threading.Thread(
        target=_worker_loop,
        args=(memory_dir, log_fn, _stop_event),
        name="samia-watcher-worker",
        daemon=True,
    )
    _watcher_thread.start()
    _worker_thread.start()


def stop() -> None:
    """Signal threads to stop, join them, clear module state."""
    global _stop_event, _watcher_thread, _worker_thread

    if _stop_event is None:
        return

    _stop_event.set()

    if _watcher_thread is not None:
        _watcher_thread.join(timeout=5.0)
        _watcher_thread = None

    if _worker_thread is not None:
        _worker_thread.join(timeout=5.0)
        _worker_thread = None

    _stop_event = None

    # Drain queues
    with _queue_lock:
        _action_queue.clear()
    with _suppress_lock:
        _suppress_set.clear()
