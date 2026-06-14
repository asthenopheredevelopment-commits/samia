"""samia.core.attention — writer + reader for attention_hints.json.

Layer 1 (Owns / Depends):
    Owns:    read_active_hints — the perception-side reader (GC-filtered, never
                 raises). add, list_hints, gc, clear — the CLI/daemon write ops
                 (post a hint, print live hints, drop expired, drop all).
             VALID_KINDS, DEFAULT_WEIGHT, DEFAULT_TTL — the hint vocabulary +
                 defaults a hint inherits when a field is absent.
    Depends: stdlib only (json, sys, time, pathlib, typing). No samia.core
             siblings — this is a leaf, safe to import from the perception loop.
Layer 2 (What / Why):
    What: a hint is a small dict {kind, value, weight, ttl, posted, origin, note}
          persisted in <memory_dir>/attention_hints.json as {"version", "hints"}.
          A hint is ACTIVE while now - posted <= ttl; read_active_hints returns
          only the active subset, add appends one, gc rewrites the file with the
          expired ones dropped, clear empties it. kind ∈ VALID_KINDS
          (app/scene/ref/regex/gate); weight is clamped to [0,1] at add().
    Why:  the perception pipeline polls read_active_hints every tick to bias what
          it attends to, so that reader must be cheap and crash-proof — a parse
          error or missing file fails soft to [] rather than stalling the loop.
          TTL-on-read (not a background sweep) means an unran gc() never serves a
          stale hint: expiry is computed at read, so the file is only a cache the
          reader re-filters. The library plane holds all logic so a CLI wrapper is
          argparse + the same functions the daemon's hint-aging job calls.

Layer 3 (Changelog):
    (carved from memory_attention.py — library plane extracted from the CLI;
     acceptance: byte-identical to pre-refactor CLI output, design doc §8.1.)
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


# _active — What: filter `hints` to those still within their per-hint TTL window
#     (now - posted <= ttl), defaulting `now` to wall-clock when not supplied.
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
# _active — Why: expiry is evaluated at read, not by a sweep, so a hint with no
#     `posted` (posted=0.0) is treated as posted-at-epoch and almost always already
#     expired — the conservative default for a malformed hint is "drop it".


def read_active_hints(memory_dir: Path) -> list[dict]:
    """Public reader for the perception side. Returns non-expired hints.

    Safe to call every tick — never raises. On parse error or missing file
    returns [].
    """
    try:
        return _active(_load(memory_dir).get("hints", []))
    except Exception:
        return []


# add — What: validate kind + weight, then append one stamped hint (posted=now) to
#     the persisted hint list.
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
# add — Why: sys.exit on a bad kind/weight is the CLI contract (this surface is
#     argparse-driven); the validation lives here, not at the file edge, so every
#     caller — including the daemon — gets the same guarded write.


# list_hints — What: print a one-line summary plus per-hint line (kind/value/weight/
#     remaining-ttl/origin) for the currently-active hints.
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


# gc — What: rewrite the hints file with expired hints physically dropped; print the
#     count removed vs kept.
def gc(memory_dir: Path) -> None:
    state = _load(memory_dir)
    before = len(state.get("hints", []))
    state["hints"] = _active(state.get("hints", []))
    after = len(state["hints"])
    _save(memory_dir, state)
    print(f"[attention] gc: {before - after} expired, {after} active")
# gc — Why: read_active_hints already hides expired hints, so gc is only file hygiene
#     (keep the JSON from growing unboundedly); it is the daemon's scheduled hint-aging
#     job, not a correctness requirement for the reader.


def clear(memory_dir: Path) -> None:
    _save(memory_dir, {"version": 1, "hints": []})
    print("[attention] cleared")


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.attention
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Carved from memory_attention.py (library plane extraction).
# Layer:      core (leaf library — safe to import from the perception loop).
# Role:       attention_hints.json store — TTL-on-read active-hint reader the perception
#             loop polls + the add/list/gc/clear write ops that bias what it attends to.
# Stability:  stable -- attention-hint store; API parameterized on memory_dir.
# ErrorModel: read_active_hints NEVER raises (parse error / missing file -> []);
#             _load fails soft to an empty store; add() sys.exit()s on an invalid
#             kind or out-of-range weight (CLI contract).
# Depends:    json, sys, time, pathlib, typing (stdlib). No samia.core siblings.
# Exposes:    read_active_hints, add, list_hints, gc, clear,
#             VALID_KINDS, DEFAULT_WEIGHT, DEFAULT_TTL.
# Lines:      168
# --------------------------------------------------------------------------
