"""samia.core.gates — Slice 7 unified stall handler (operator-gate side).

Layer 1 (Owns / Depends):
    Owns:    gate_open(text, original_task, non_blocking_fallback=None,
                 ping_cadence_s=900, default_action=None,
                 stall_class="user_gate", warrior=None) -> str — register a gate.
             gate_close(gate_id, resolution) -> bool — mark one resolved.
             list_open_gates() -> list[dict] — snapshot of open gates.
             gate_tick(memory_dir) -> dict — the idle-pulse subscriber that
                 re-surfaces stale gates as attention hints.
    Depends: samia.core.attention (writes the kind="gate" attention hint via its
             private _load/_save, bypassing attention.add's stdout print). stdlib
             only otherwise (json, secrets, time, pathlib).
Layer 2 (What / Why):
    What: a "gate" is any moment we are blocked on an operator decision. gate_open
          records it to ~/.local/share/asthenos/handoff/pending_gates.json. After
          GATE_QUIET_S (300s) of silence a gate becomes a stall; gate_tick (fired
          every tool call by the idle pulse) then posts an attention hint and re-
          posts it every ping_cadence_s. gate_close clears the gate and appends a
          telemetry line to gate_resolutions.jsonl. Internal warrior stalls
          (Slice 3) share the same file + cadence so the operator sees one queue.
    Why:  without this, a gate stalls all forward motion until the operator
          returns. The state file makes gates durable across sessions, the single
          ping cadence keeps the operator's attention surface to one queue, and the
          stall_class split lets user gates and internal-warrior stalls coexist
          without a second dispatcher.

State file shape — pending_gates.json:
    {"version": 1,
     "gates": [{"id": "gate_<unix>_<rand4>", "asked_at": float, "gate_text": str,
                "original_task": str, "non_blocking_fallback": list[str],
                "ping_cadence_s": float, "default_action": str | None,
                "last_ping_at": float,
                "stall_class": "user_gate" | "internal_warrior",
                "warrior": str | None}]}
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Optional

from . import attention as _att

GATE_QUIET_S = 300.0           # 5min — when a gate becomes a stall
DEFAULT_PING_CADENCE_S = 900.0  # 15min — re-surface cadence
HANDOFF_DIR = Path.home() / ".local/share/asthenos/handoff"
GATES_FILE = HANDOFF_DIR / "pending_gates.json"


def _load() -> dict:
    if not GATES_FILE.exists():
        return {"version": 1, "gates": []}
    try:
        return json.loads(GATES_FILE.read_text())
    except Exception:
        return {"version": 1, "gates": []}


def _save(state: dict) -> None:
    GATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    GATES_FILE.write_text(json.dumps(state, indent=2) + "\n")


def _new_gate_id() -> str:
    return f"gate_{int(time.time())}_{secrets.token_hex(2)}"


def gate_open(
    text: str,
    original_task: str,
    non_blocking_fallback: Optional[list[str]] = None,
    ping_cadence_s: float = DEFAULT_PING_CADENCE_S,
    default_action: Optional[str] = None,
    stall_class: str = "user_gate",
    warrior: Optional[str] = None,
) -> str:
    """Register a new gate. Returns gate_id.

    text: what we asked (or what the warrior is stuck on).
    original_task: the task we were doing when the gate opened.
    non_blocking_fallback: ordered list of fallback tasks to advance while waiting.
    ping_cadence_s: how often to re-surface this gate (after GATE_QUIET_S).
    default_action: if set, used by `default if silent` consumers — leave None
                    for genuinely ambiguous gates that need explicit reply.
    stall_class: "user_gate" (waiting on operator) or "internal_warrior" (Slice 3).
    warrior: filament name for internal warrior stalls; None for user gates.
    """
    if stall_class not in ("user_gate", "internal_warrior"):
        raise ValueError(f"stall_class must be user_gate|internal_warrior, got {stall_class!r}")
    state = _load()
    now = time.time()
    gate = {
        "id": _new_gate_id(),
        "asked_at": now,
        "gate_text": text,
        "original_task": original_task,
        "non_blocking_fallback": non_blocking_fallback or [],
        "ping_cadence_s": float(ping_cadence_s),
        "default_action": default_action,
        "last_ping_at": 0.0,
        "stall_class": stall_class,
        "warrior": warrior,
    }
    state["gates"].append(gate)
    _save(state)
    return gate["id"]


def gate_close(gate_id: str, resolution: str) -> bool:
    """Mark a gate resolved. Returns True if found, False otherwise.

    resolution: short string for telemetry — "answered", "defaulted",
                "superseded", "abandoned".
    """
    state = _load()
    found = False
    keep: list[dict] = []
    for g in state.get("gates", []):
        if g.get("id") == gate_id:
            found = True
            _append_resolution(g, resolution)
            continue
        keep.append(g)
    state["gates"] = keep
    if found:
        _save(state)
    return found


def list_open_gates() -> list[dict]:
    """Return all currently-open gates (deep-copy safe via JSON round-trip)."""
    return list(_load().get("gates", []))


def _append_resolution(gate: dict, resolution: str) -> None:
    """Append a one-line resolution event to a sibling jsonl for telemetry."""
    log_path = HANDOFF_DIR / "gate_resolutions.jsonl"
    payload = {
        "ts": time.time(),
        "gate_id": gate.get("id"),
        "asked_at": gate.get("asked_at"),
        "duration_s": time.time() - gate.get("asked_at", time.time()),
        "stall_class": gate.get("stall_class"),
        "resolution": resolution,
        "gate_text": gate.get("gate_text", "")[:200],
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def gate_tick(memory_dir: Path) -> dict:
    """Subscriber, fires from idle_pulse hook every tool call.

    For each open gate:
      - if stale (now - asked_at > GATE_QUIET_S) and never pinged before, OR
      - if (now - last_ping_at > ping_cadence_s)
    write an attention hint (kind=gate) so the next perception/agent
    cycle surfaces it.

    Returns a small dict for diagnostics:
      {"checked": int, "pinged": int, "stale": int}
    """
    state = _load()
    now = time.time()
    checked = pinged = stale = 0
    # StaleAndDueGate — What: for each gate, fire a hint only once it has gone stale
    #     (silent past GATE_QUIET_S) AND is "due" — never pinged, or last ping older
    #     than this gate's own ping_cadence.
    for g in state.get("gates", []):
        checked += 1
        asked_at = g.get("asked_at", now)
        last_ping = g.get("last_ping_at", 0.0)
        ping_cadence = g.get("ping_cadence_s", DEFAULT_PING_CADENCE_S)
        is_stale = (now - asked_at) > GATE_QUIET_S
        if not is_stale:
            continue
        stale += 1
        due = (last_ping == 0.0) or ((now - last_ping) > ping_cadence)
        if not due:
            continue
        # StaleAndDueGate — Why: the two gates separate "newly asked" (let the operator
        #     answer in peace for GATE_QUIET_S) from "re-surface periodically" — without
        #     the cadence check every idle pulse (one per tool call) would re-post a hint.
        try:
            note = (
                f"[{g.get('stall_class', 'user_gate')}] {g.get('gate_text', '')[:160]} "
                f"(open {int((now - asked_at) / 60)}m; "
                f"original_task={g.get('original_task', '?')[:60]})"
            )
            # Write hint directly to bypass attention.add's stdout print, which
            # would noise the idle pulse hook output.
            hints_state = _att._load(memory_dir)
            hints_state["hints"].append({
                "kind": "gate",
                "value": g.get("id", "?"),
                "weight": 0.85,
                "ttl": float(ping_cadence),
                "posted": now,
                "origin": "memory",
                "note": note,
            })
            _att._save(memory_dir, hints_state)
            g["last_ping_at"] = now
            pinged += 1
        except Exception:
            pass
    _save(state)
    return {"checked": checked, "pinged": pinged, "stale": stale}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.gates
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Slice 7 unified stall handler (operator-gate side); Slice 3 internal
#             warrior stalls share the same state file + ping cadence.
# Layer:      core (pure library; gate_tick is an idle-pulse subscriber)
# Role:       durable operator/warrior gate queue — open/tick/close a blocked-on-
#             decision gate in pending_gates.json; the idle-pulse subscriber re-
#             surfaces stale gates as attention hints on a per-gate ping cadence.
# Stability:  stable -- gate lifecycle (open/tick/close) + durable JSON state.
# ErrorModel: _load fails SOFT to an empty {version,gates} state on a missing or
#             corrupt file; gate_tick swallows per-gate hint-write errors (a bad
#             gate never blocks the others) and the resolution log append is
#             best-effort. gate_open raises ValueError on an invalid stall_class.
# Depends:    json, secrets, time, pathlib (stdlib). samia.core.attention
#             (_load/_save, to bypass attention.add's stdout print in the pulse).
# Exposes:    gate_open, gate_close, list_open_gates, gate_tick;
#             GATE_QUIET_S, DEFAULT_PING_CADENCE_S, GATES_FILE constants.
# Lines:      236
# --------------------------------------------------------------------------
