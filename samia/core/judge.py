"""samia.core.judge — Tier 2b local-LLM permission judge.

Layer 1 (Owns / Depends):
    Owns:    judge — ask a local LLM whether a sub-agent tool-call pattern is safe
                 to auto-approve; ALWAYS returns a verdict dict, never raises.
             ensure_ollama_up — start the Ollama daemon if down, wait for reachable.
    Depends: stdlib only (json, os, re, shutil, subprocess, time, urllib, typing).
             samia.runtime.client (SamiaClient — imported LAZILY inside
             _judge_daemon; the daemon path is optional). External processes
             (ollama serve, llama-cli) are probed at runtime, never required.
Layer 2 (What / Why):
    What: judge() builds a prompt from (pattern, context), then tries backends in
          order — (0) the SAM/IA runtime daemon's infer op, (1) Ollama's HTTP
          /api/generate, (2) the llama-cli CLI — and parses the model's three-line
          'VERDICT/CONFIDENCE/RATIONALE' reply into a verdict dict
          {verdict, confidence, rationale, backend, model, wall_ms}. If no backend
          answers it returns verdict='unsure', backend='none'.
    Why:  this is Tier 2b of the permission gate: the rule-based auditor (Tier 2a)
          calls here only for NOVEL patterns it can't decide, and an 'unsure'
          result is the signal to escalate to Claude (Tier 3) or the operator
          (Tier 4). Because it sits on the gating hot path, it must degrade rather
          than throw — every failure mode collapses to a well-formed 'unsure'/'none'
          dict so a down LLM means "escalate", never "crash the gate". Backends are
          ordered cheapest/most-integrated first (daemon reuses a warm model) to
          freshest fallback; the small-model retry (FALLBACK_SMALL) covers an
          OOM/load failure of the larger default.

Layer 3 (Changelog):
    (AUD28.7 V1 added Backend 0 — the SAM/IA runtime daemon's infer op — ahead of
     Ollama, so a judgment reuses the daemon's already-loaded LlamaCppBackend.)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("SAMIA_JUDGE_MODEL", "qwen2.5-coder:14b")
FALLBACK_SMALL = "phi4-mini:latest"   # used when DEFAULT_MODEL fails to load
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")


# ── Backend detection / lifecycle ──────────────────────────────────

def _ollama_reachable(timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def ensure_ollama_up(timeout_s: float = 10.0) -> bool:
    """If the Ollama daemon is down, attempt to start it in background.
    Returns True if reachable within timeout_s, False otherwise."""
    if _ollama_reachable():
        return True
    if not shutil.which("ollama"):
        return False
    # LazyStart — What: spawn `ollama serve` detached, then poll reachability until
    #     timeout_s before giving up.
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _ollama_reachable():
            return True
        time.sleep(0.5)
    return False
# LazyStart — Why: the judge must not require an operator to pre-start Ollama, but it
#     also must not block the gate forever — start_new_session detaches the daemon from
#     this process so it survives, and the bounded poll fails soft to the next backend.


def _llama_cli_available() -> bool:
    return shutil.which("llama-cli") is not None


# ── Prompt construction ────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a security-aware auditor for a multi-agent system. "
    "Each judgment evaluates whether a proposed tool call is safe to "
    "auto-approve. Reply ONLY with three lines: 'VERDICT: allow|deny|unsure', "
    "'CONFIDENCE: <0.0-1.0>', 'RATIONALE: <one short sentence>'. "
    "Heuristics: file edits inside the project tree are usually fine; "
    "edits to /etc, /usr, ~/.ssh, ~/.gnupg are deny; destructive shell "
    "commands (rm -rf, dd, mkfs) on broad paths are deny; read-only ops "
    "are usually allow; novel patterns with no rationale are unsure."
)


# _build_prompt — What: render (pattern, context) into the judge prompt — caller label,
#     tool, pattern, and a trimmed summary of the salient tool_input keys.
def _build_prompt(pattern: str, context: dict) -> str:
    is_sidechain = context.get("is_sidechain")
    agent_label = "sub-agent" if is_sidechain else (
        "parent" if is_sidechain is False else "unknown-agent"
    )
    tool = context.get("tool_name", "?")
    inp = context.get("tool_input", {}) or {}
    # Trim tool_input for prompt efficiency
    inp_summary: list[str] = []
    for k in ("file_path", "command", "subagent_type", "url", "pattern"):
        if k in inp:
            v = str(inp[k])
            if len(v) > 200:
                v = v[:200] + "…"
            inp_summary.append(f"  {k}: {v}")
    if not inp_summary:
        inp_summary.append(f"  (params: {sorted(inp.keys())[:5]})")

    return (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Caller: {agent_label}\n"
        f"Tool: {tool}\n"
        f"Pattern: {pattern}\n"
        f"Inputs:\n" + "\n".join(inp_summary) + "\n\n"
        f"Decide. Reply in the three required lines only."
    )
# _build_prompt — Why: only a small whitelist of input keys is surfaced (file_path,
#     command, ...) and each is truncated to 200 chars — a judgment hinges on the
#     pattern + the salient args, and a huge tool_input would waste the model's context
#     and slow the gate. is_sidechain=None (undeterminable) maps to "unknown-agent" so
#     the prompt never claims a caller class it doesn't know.


# ── Response parsing ───────────────────────────────────────────────

_VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(allow|deny|unsure)\b", re.I | re.M)
_CONF_RE = re.compile(r"^\s*CONFIDENCE:\s*([0-9.]+)", re.I | re.M)
_RAT_RE = re.compile(r"^\s*RATIONALE:\s*(.+)$", re.I | re.M)


# _parse_response — What: pull (verdict, confidence, rationale) out of the model's
#     free text via the three line-anchored regexes, with safe defaults for each field.
def _parse_response(text: str) -> tuple[str, float, str]:
    v_m = _VERDICT_RE.search(text or "")
    verdict = v_m.group(1).lower() if v_m else "unsure"
    c_m = _CONF_RE.search(text or "")
    try:
        confidence = float(c_m.group(1)) if c_m else 0.3
    except ValueError:
        confidence = 0.3
    confidence = max(0.0, min(1.0, confidence))
    r_m = _RAT_RE.search(text or "")
    rationale = (r_m.group(1).strip() if r_m else (text or "")[:200].strip())
    return verdict, confidence, rationale[:200]
# _parse_response — Why: a local model often deviates from the exact format, so every
#     field fails soft — a missing verdict becomes 'unsure' (the escalate signal), a
#     missing/garbled confidence becomes a neutral 0.3, and confidence is clamped to
#     [0,1] so a hallucinated "CONFIDENCE: 5" can't poison downstream weighting.


# ── Backend invocation ─────────────────────────────────────────────

def _judge_daemon(prompt: str, timeout_s: float) -> Optional[str]:
    """AUD28.7 V1: try the asthenos-runtime daemon's infer op (LlamaCppBackend).

    Returns the model's raw text or None on failure (daemon down, op error).
    Falls through to _judge_ollama then _judge_llama_cli at the caller.
    """
    try:
        from samia.runtime.client import SamiaClient, DaemonNotRunning
    except ImportError:
        return None
    try:
        with SamiaClient(timeout=timeout_s) as client:
            result = client.call(
                "infer",
                prompt=prompt,
                max_tokens=200,
                temperature=0.1,
                caller_hint="samia.core.judge",
            )
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                return text
        return None
    except (DaemonNotRunning, Exception):
        return None


def _judge_ollama(prompt: str, model: str, timeout_s: float) -> Optional[str]:
    """Returns the model's raw text or None on failure."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.1, "num_predict": 200},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = r.read().decode("utf-8")
        return json.loads(body).get("response", "")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def _judge_llama_cli(prompt: str, model_path: str, timeout_s: float) -> Optional[str]:
    """Fallback path; only used if llama-cli is on PATH and a GGUF path is provided."""
    try:
        result = subprocess.run(
            ["llama-cli", "-m", model_path, "-p", prompt,
             "--n-predict", "200", "--temp", "0.1", "--no-display-prompt"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


# ── Public API ─────────────────────────────────────────────────────

def judge(
    pattern: str,
    context: Optional[dict] = None,
    *,
    model: Optional[str] = None,
    timeout_s: float = 30.0,
) -> dict:
    """Ask the local LLM to judge a sub-agent tool-call pattern.

    Returns a verdict dict (see module docstring). Always returns —
    never raises. On total failure: verdict='unsure', backend='none',
    so the auditor escalates to Tier 3 (Claude) or 4 (user).
    """
    context = context or {}
    model = model or DEFAULT_MODEL
    started = time.time()
    prompt = _build_prompt(pattern, context)

    # BackendCascade — What: try the three backends in priority order (daemon, then
    #     Ollama with a small-model retry, then llama-cli); the FIRST that returns raw
    #     text wins and its parsed verdict is returned. Fall through to 'none' if all fail.
    # Backend 0: SAM/IA daemon (AUD28.7 V1) — daemon-routed, telemetry-emitting.
    raw = _judge_daemon(prompt, timeout_s)
    if raw is not None:
        verdict, confidence, rationale = _parse_response(raw)
        return {
            "verdict": verdict,
            "confidence": confidence,
            "rationale": rationale,
            "backend": "daemon",
            "model": "samia.runtime.LlamaCppBackend",
            "wall_ms": int((time.time() - started) * 1000),
        }

    # Backend 1: Ollama
    if ensure_ollama_up():
        raw = _judge_ollama(prompt, model, timeout_s)
        if raw is None and model != FALLBACK_SMALL:
            raw = _judge_ollama(prompt, FALLBACK_SMALL, timeout_s)
            if raw is not None:
                model = FALLBACK_SMALL
        if raw is not None:
            verdict, confidence, rationale = _parse_response(raw)
            return {
                "verdict": verdict,
                "confidence": confidence,
                "rationale": rationale,
                "backend": "ollama",
                "model": model,
                "wall_ms": int((time.time() - started) * 1000),
            }

    # Backend 2: llama-cli (only if PATH and a GGUF path env var is set)
    gguf = os.environ.get("SAMIA_LLAMA_CLI_GGUF")
    if _llama_cli_available() and gguf:
        raw = _judge_llama_cli(prompt, gguf, timeout_s)
        if raw is not None:
            verdict, confidence, rationale = _parse_response(raw)
            return {
                "verdict": verdict,
                "confidence": confidence,
                "rationale": rationale,
                "backend": "llama-cli",
                "model": gguf,
                "wall_ms": int((time.time() - started) * 1000),
            }

    # Degraded
    return {
        "verdict": "unsure",
        "confidence": 0.0,
        "rationale": "no local LLM backend reachable — escalate to Tier 3",
        "backend": "none",
        "model": None,
        "wall_ms": int((time.time() - started) * 1000),
    }
    # BackendCascade — Why: the daemon path is first because it reuses an
    #     already-warm model (no cold start); the small-model retry rescues a large-model
    #     OOM/load failure without losing the request. A total failure returns
    #     'unsure'/'none' (NOT an exception) because this sits on the gate hot path — the
    #     auditor reads 'unsure' as "escalate to Tier 3/4", which is the safe default.


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.judge
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Tier 2b local-LLM permission judge + AUD28.7 V1 (daemon backend).
# Layer:      core (library; called by samia.core.auditor's escalation path).
# Role:       local-LLM auto-approval judge — cascades daemon/Ollama/llama-cli backends
#             and parses a verdict dict; always answers, never raises, degrades to 'unsure'.
# Stability:  stable -- verdict contract fixed; backends probed at runtime.
# ErrorModel: judge() NEVER raises — every backend failure (daemon down, HTTP/OS
#             error, JSON decode error, subprocess timeout) collapses to a
#             well-formed verdict dict; total failure -> verdict='unsure',
#             backend='none' (the auditor's signal to escalate to Tier 3/4).
# Depends:    json, os, re, shutil, subprocess, time, urllib, typing (stdlib).
#             samia.runtime.client (LAZY, _judge_daemon only). External: ollama,
#             llama-cli (probed, optional).
# Exposes:    judge, ensure_ollama_up.
# Lines:      336
# --------------------------------------------------------------------------
