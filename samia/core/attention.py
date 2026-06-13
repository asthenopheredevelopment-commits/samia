"""samia.core.attention — writer + reader for attention_hints.json.

Carved from memory_attention.py. The library plane carries all logic so
the perception pipeline can call read_active_hints() directly and the
daemon's hint-aging job (per design doc §1.3) can call gc() on a schedule.

A hint is a small dict:
  {
    "kind":   "app" | "scene" | "ref" | "regex",
    "value":  str,
    "weight": float,        # 0.0–1.0
    "ttl":    float,        # seconds after `posted` when the hint expires
    "posted": float,        # unix ts when added
    "origin": str,          # "memory" | "manual" | "agent"
    "note":   str,
  }

File layout:
  <memory_dir>/attention_hints.json
  { "version": 1, "hints": [ ... ] }

Public API (parameterized on memory_dir):
  read_active_hints(memory_dir)               → list[dict]   (already GC-filtered)
  add(memory_dir, kind, value, weight, ttl,
      origin, note)                           → None
  list_hints(memory_dir)                      → None         (prints)
  gc(memory_dir)                              → None         (drops expired, prints)
  clear(memory_dir)                           → None         (drops all, prints)

Acceptance: byte-identical to pre-refactor memory_attention.py CLI output
on the same memory tree (design doc §8.1).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

VALID_KINDS = ("app", "scene", "ref", "regex", "gate")
DEFAULT_WEIGHT = 0.6
DEFAULT_TTL = 1800.0  # 30 minutes


def _hints_file(memory_dir: Path) -> Path:
    return memory_dir / "attention_hints.json"


def _load(memory_dir: Path) -> dict:
    fp = _hints_file(memory_dir)
    if not fp.exists():
        return {"version": 1, "hints": []}
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {"version": 1, "hints": []}


def _save(memory_dir: Path, state: dict) -> None:
    fp = _hints_file(memory_dir)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(state, indent=2) + "\n")


def _active(hints: list[dict], now: Optional[float] = None) -> list[dict]:
    if now is None:
        now = time.time()
    alive: list[dict] = []
    for h in hints:
        posted = h.get("posted", 0.0)
        ttl = h.get("ttl", DEFAULT_TTL)
        if now - posted <= ttl:
            alive.append(h)
    return alive


def read_active_hints(memory_dir: Path) -> list[dict]:
    """Public reader for the perception side. Returns non-expired hints.

    Safe to call every tick — never raises. On parse error or missing file
    returns [].
    """
    try:
        return _active(_load(memory_dir).get("hints", []))
    except Exception:
        return []


def add(memory_dir: Path, kind: str, value: str, weight: float, ttl: float,
        origin: str, note: str) -> None:
    if kind not in VALID_KINDS:
        sys.exit(f"add: kind must be one of {VALID_KINDS}")
    if not (0.0 <= weight <= 1.0):
        sys.exit("add: weight must be in [0.0, 1.0]")
    state = _load(memory_dir)
    state["hints"].append({
        "kind": kind,
        "value": value,
        "weight": float(weight),
        "ttl": float(ttl),
        "posted": time.time(),
        "origin": origin,
        "note": note,
    })
    _save(memory_dir, state)
    print(f"[attention] added {kind}={value!r} w={weight} ttl={ttl}s "
          f"origin={origin}")


def list_hints(memory_dir: Path) -> None:
    state = _load(memory_dir)
    hints = state.get("hints", [])
    now = time.time()
    live = _active(hints, now)
    print(f"[attention] {len(live)} active / {len(hints)} total")
    for h in live:
        age = now - h.get("posted", now)
        rem = h.get("ttl", DEFAULT_TTL) - age
        print(f"  {h['kind']:<6} {h['value']!r:<40} "
              f"w={h.get('weight', DEFAULT_WEIGHT):.2f} "
              f"remaining={int(rem)}s origin={h.get('origin', '')}")


def gc(memory_dir: Path) -> None:
    state = _load(memory_dir)
    before = len(state.get("hints", []))
    state["hints"] = _active(state.get("hints", []))
    after = len(state["hints"])
    _save(memory_dir, state)
    print(f"[attention] gc: {before - after} expired, {after} active")


def clear(memory_dir: Path) -> None:
    _save(memory_dir, {"version": 1, "hints": []})
    print("[attention] cleared")
