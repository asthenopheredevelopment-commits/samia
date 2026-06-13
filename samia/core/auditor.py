"""samia.core.auditor — Tier 2a of the permission gating system.

Reads the PreToolUse/PostToolUse decision logs, joins them by
tool_use_id, enriches with isSidechain by reading the parent's
transcript JSONL, and appends to the persistent confidence ledger.

This is the *observation* layer — it doesn't make decisions; it
records what happened so a future scoring pass can compute
per-pattern confidence weighted by tier (user/auditor/claude/local_llm).

Files used (under memory_dir):
  Reads:
    .subagent_payload_probe.jsonl   (PreToolUse)
    .subagent_outcomes.jsonl        (PostToolUse, paired by tool_use_id)
    transcript JSONL referenced from each payload (for isSidechain)
  Writes:
    .confidence_ledger.jsonl   (append-only)
    .audit_state.json          (last-processed cursors per log)

Public API (parameterized on memory_dir):
  run_audit(memory_dir) → dict
  auditor_tick(memory_dir, force=False) → dict   (idle-pulse subscriber)

Cadence: 15 min via idle pulse (the 6h-style state-file gate).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

AUDITOR_INTERVAL_S = 900  # 15 minutes


def _safe_load_jsonl_after(path: Path, byte_offset: int) -> tuple[list[dict], int]:
    """Read JSONL entries past byte_offset. Returns (entries, new_offset).
    Skips malformed lines silently."""
    if not path.exists():
        return [], byte_offset
    try:
        sz = path.stat().st_size
    except OSError:
        return [], byte_offset
    if byte_offset >= sz:
        return [], byte_offset
    entries = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(byte_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        new_offset = f.tell()
    return entries, new_offset


def _lookup_sidechain(transcript_path: Optional[str],
                      tool_use_id: str) -> Optional[bool]:
    """Read the transcript JSONL and find whether the call with
    `tool_use_id` was made on a sidechain (sub-agent) or not.

    Returns True/False if the entry was found, None if undeterminable.
    Reads only the last 256KB of the transcript to keep cost bounded.
    """
    if not transcript_path or not tool_use_id:
        return None
    p = Path(transcript_path)
    if not p.exists():
        return None
    try:
        sz = p.stat().st_size
        with p.open("r", encoding="utf-8", errors="replace") as f:
            if sz > 262144:
                f.seek(sz - 262144)
                f.readline()  # skip partial line
            for line in f:
                if tool_use_id not in line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Match: tool_use entries have tool_use_id; we want the
                # one whose toolUseId/id matches and read its sidechain
                # status. CC wraps tool calls inside assistant message
                # content; the isSidechain field is on the outer envelope.
                content = d.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if (isinstance(c, dict)
                                and c.get("id") == tool_use_id
                                and c.get("type") == "tool_use"):
                            return bool(d.get("isSidechain", False))
        return None
    except OSError:
        return None


_BASH_FIRST_TOKENS = re.compile(r"^\s*([\w./-]+(?:\s+[\w./-]+){0,2})")


def _pattern_signature(tool_name: str, tool_input: dict) -> str:
    """Produce a stable signature for pattern matching across decisions."""
    if not tool_input:
        return tool_name or "?"
    if tool_name in ("Edit", "Write", "Read", "NotebookEdit"):
        path = str(tool_input.get("file_path", ""))
        # Glob common project roots so multiple files under the same root
        # collapse to one pattern. Roots derive from the user's home so the
        # signature logic carries to any box (order matters: most-specific first).
        _home = str(Path.home())
        for root in (f"{_home}/Asthenosphere",
                     f"{_home}/Desktop/DinnerBell-BBQ-Dev",
                     f"{_home}/.claude/projects",
                     f"{_home}/Desktop", "/tmp", _home):
            if path.startswith(root):
                return f"{tool_name}({root}/**)"
        return f"{tool_name}({path[:60]})"
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        m = _BASH_FIRST_TOKENS.match(cmd)
        head = m.group(1) if m else cmd[:30]
        return f"Bash({head})"
    if tool_name == "Agent":
        st = tool_input.get("subagent_type", "?")
        return f"Agent({st})"
    return tool_name or "?"


def _outcome_decision(outcome: dict) -> tuple[str, Optional[str]]:
    """Classify a PostToolUse outcome as allow|deny|error and pull error msg if any."""
    if not outcome:
        return "unknown", None
    resp = outcome.get("tool_response")
    if isinstance(resp, dict):
        if resp.get("isError") or resp.get("is_error"):
            err = resp.get("content") or resp.get("error") or "(error)"
            err_str = str(err) if not isinstance(err, str) else err
            return ("deny", err_str[:200]) if "Permission" in err_str else ("error", err_str[:200])
    if outcome.get("error"):
        return "error", str(outcome["error"])[:200]
    return "allow", None


def run_audit(memory_dir: Path) -> dict:
    """Drain new pre/post entries, pair them, append to ledger.

    Returns telemetry: counts of entries processed, pairs formed,
    ledger rows appended.
    """
    state_path = memory_dir / ".audit_state.json"
    pre_path = memory_dir / ".subagent_payload_probe.jsonl"
    post_path = memory_dir / ".subagent_outcomes.jsonl"
    ledger_path = memory_dir / ".confidence_ledger.jsonl"

    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    pre_off = int(state.get("pre_offset", 0))
    post_off = int(state.get("post_offset", 0))

    pre_entries, new_pre_off = _safe_load_jsonl_after(pre_path, pre_off)
    post_entries, new_post_off = _safe_load_jsonl_after(post_path, post_off)

    # Index post by tool_use_id (and remember unmatched for next pass)
    post_by_tuid: dict[str, dict] = {}
    for p in post_entries:
        o = p.get("outcome") if isinstance(p, dict) else None
        if isinstance(o, dict):
            tuid = o.get("tool_use_id")
            if tuid:
                post_by_tuid[tuid] = p

    # Pair pre→post; carry unpaired pres forward by leaving the cursor
    # behind to re-read on next tick. For v1 we accept loss of unpaired
    # pres after one tick; the post log catches up quickly.
    new_rows: list[dict] = []
    for pre in pre_entries:
        payload = pre.get("payload") if isinstance(pre, dict) else None
        if not isinstance(payload, dict):
            continue
        tuid = payload.get("tool_use_id")
        post = post_by_tuid.get(tuid) if tuid else None
        outcome = post.get("outcome") if post else None

        is_sidechain = _lookup_sidechain(payload.get("transcript_path"), tuid) if tuid else None
        tool_name = payload.get("tool_name", "?")
        tool_input = payload.get("tool_input") or {}
        sig = _pattern_signature(tool_name, tool_input)
        decision, err = _outcome_decision(outcome) if outcome else ("pending", None)

        new_rows.append({
            "ts": pre.get("ts"),
            "tool_use_id": tuid,
            "is_sidechain": is_sidechain,
            "agent": "sub-agent" if is_sidechain else ("parent" if is_sidechain is False else "unknown"),
            "tool": tool_name,
            "pattern": sig,
            "decision": decision,
            "decided_by": "system-observation",
            "error": err,
        })

    if new_rows:
        with ledger_path.open("a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    state["pre_offset"] = new_pre_off
    state["post_offset"] = new_post_off
    state["last_run_iso"] = datetime.now().isoformat(timespec="seconds")
    state["last_pre_count"] = len(pre_entries)
    state["last_post_count"] = len(post_entries)
    state["last_paired_count"] = sum(1 for r in new_rows if r["decision"] != "pending")
    state["last_ledger_appends"] = len(new_rows)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return {
        "n_pre": len(pre_entries),
        "n_post": len(post_entries),
        "n_pairs": state["last_paired_count"],
        "n_ledger_appends": len(new_rows),
        "pre_offset": new_pre_off,
        "post_offset": new_post_off,
    }


JUDGE_MAX_PER_TICK = 5   # rate-limit: don't burn local-LLM time on a backlog

def judge_novel_patterns(memory_dir: Path, max_calls: int = JUDGE_MAX_PER_TICK) -> dict:
    """Tier 2b escalation: for patterns in the ledger with no prior verdict
    by user/auditor/claude/local_llm, ask samia.core.judge for a local-LLM
    verdict and append it as a new ledger row.

    Rate-limited to max_calls per tick so a backlog of N novel patterns
    doesn't burn N * cold-start seconds in one go.
    """
    ledger = memory_dir / ".confidence_ledger.jsonl"
    if not ledger.exists():
        return {"judged": 0, "novel": 0}

    # Read all ledger rows; collect patterns that already have an authoritative
    # decision (decided_by != system-observation), and patterns that don't.
    decided: set[str] = set()
    pending: dict[str, dict] = {}
    try:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict):
                continue
            pat = r.get("pattern")
            if not pat:
                continue
            if r.get("decided_by") in ("user", "auditor", "claude", "local_llm"):
                decided.add(pat)
            elif pat not in pending:
                pending[pat] = r
    except OSError:
        return {"judged": 0, "novel": 0, "error": "ledger read failed"}

    novel_patterns = [(p, r) for p, r in pending.items() if p not in decided]
    if not novel_patterns:
        return {"judged": 0, "novel": 0}

    # Rate-limit
    todo = novel_patterns[:max_calls]

    try:
        from . import judge as _judge
    except ImportError:
        return {"judged": 0, "novel": len(novel_patterns), "error": "judge import failed"}

    new_rows: list[dict] = []
    for pat, sample_row in todo:
        ctx = {
            "tool_name": sample_row.get("tool"),
            "is_sidechain": sample_row.get("is_sidechain"),
            "tool_input": {},
        }
        try:
            v = _judge.judge(pat, ctx, timeout_s=15.0)
        except Exception as e:
            v = {"verdict": "unsure", "confidence": 0.0,
                 "rationale": f"judge call failed: {e}", "backend": "none"}
        new_rows.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "pattern": pat,
            "tool": sample_row.get("tool"),
            "is_sidechain": sample_row.get("is_sidechain"),
            "decision": v["verdict"],
            "decided_by": "local_llm",
            "confidence": v["confidence"],
            "rationale": v["rationale"],
            "judge_backend": v["backend"],
            "judge_model": v.get("model"),
            "wall_ms": v.get("wall_ms"),
        })

    if new_rows:
        with ledger.open("a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    return {"judged": len(new_rows), "novel": len(novel_patterns),
            "remaining": max(0, len(novel_patterns) - len(new_rows))}


def auditor_tick(memory_dir: Path, force: bool = False) -> dict:
    """15-minute-gated subscriber for the idle pulse.

    Wraps run_audit + judge_novel_patterns with a state-file cooldown so
    it fires at most every AUDITOR_INTERVAL_S. Designed to be called from
    hook_idle_pulse.sh.
    """
    import time as _time
    state_path = memory_dir / ".auditor_tick_state.json"
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    last_unix = float(state.get("last_tick_unix", 0))
    now = _time.time()
    elapsed = now - last_unix
    if not force and elapsed < AUDITOR_INTERVAL_S:
        return {"fired": False, "elapsed_seconds": int(elapsed),
                "interval_seconds": AUDITOR_INTERVAL_S}

    audit_out = run_audit(memory_dir)
    judge_out = judge_novel_patterns(memory_dir)

    state["last_tick_unix"] = now
    state["last_tick_iso"] = datetime.now().isoformat(timespec="seconds")
    state["last_audit"] = audit_out
    state["last_judge"] = judge_out
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return {
        "fired": True,
        "elapsed_seconds": int(elapsed),
        "audit": audit_out,
        "judge": judge_out,
    }
