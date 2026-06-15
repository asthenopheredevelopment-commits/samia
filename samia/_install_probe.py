"""samia._install_probe — the four-layer SAM/IA capability probe behind `samia status`.

Layer 1 (Owns / Depends):
    Owns:    a cheap, side-effect-free check of the FOUR install layers — package /
             store / daemon / mcp — plus a plain-language verdict of WHICH layer is
             missing and the EXACT next command. run_probe() and the _print_human()
             renderer the `samia status` subcommand drives.
    Depends: stdlib only (os, json, shutil, socket, importlib.util, pathlib). It does
             NOT import the heavy samia internals — `samia status` must answer even on a
             half-installed box without dragging in the runtime.

Layer 2 (What / Why):
    What: probes, in order, (1) is `samia` importable / on PATH, (2) is the store
          initialized (`samia init` run), (3) is the user-managed daemon reachable,
          (4) is the MCP server wired into a client config. Returns a structured report
          and the single concrete NEXT step.
    Why:  "never silently inert" — a missing layer is named, with the one command that
          closes it. The store/daemon/mcp markers track exactly what `samia init`, the
          maintenance daemon, and the .mcp.json create, so the probe checks the SAME
          paths the live server uses.

HARD BOUNDARY: this probe only READS state. It never installs, never starts a daemon,
never edits settings. It tells the user what to run; the user runs it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
from pathlib import Path

# StorePath — What: where the store should live (env override → default).
# StorePath — Why: kept in sync with samia.cli `init` and samia.mcp_server_main so the
#             probe checks the SAME dir the server resolves (ASTHENOS_MEMORY_DIR).
_ENV_MEMORY_DIR = "ASTHENOS_MEMORY_DIR"
_DEFAULT_STORE = Path.home() / ".local" / "share" / "asthenos"
_MCP_NAME = "asthenos-memory"


def _resolve_store() -> Path:
    env = os.environ.get(_ENV_MEMORY_DIR)
    return Path(env).expanduser() if env else _DEFAULT_STORE


# LayerPackage — What: is `samia` importable OR a `samia` CLI on PATH?
# LayerPackage — Why: layer 1 of 4. find_spec is import-free (no heavy load); the PATH
#               check covers a pipx/console_script install where import-from-here may
#               not see it but the CLI exists.
def probe_package() -> dict:
    spec = None
    try:
        spec = importlib.util.find_spec("samia")
    except (ImportError, ValueError):
        spec = None
    on_path = shutil.which("samia") or shutil.which("samia-mcp-server")
    ok = bool(spec) or bool(on_path)
    return {
        "layer": "package",
        "ok": ok,
        "detail": (
            f"importable={bool(spec)} cli_on_path={bool(on_path)}"
            if ok else "the `samia` package is not importable and no `samia` CLI is on PATH"
        ),
        "next": None if ok else {
            "why": "Nothing to wire to yet — the package itself is not installed.",
            "command": "pipx install samia   # or: pip install samia",
        },
    }


# LayerStore — What: has `samia init` run (store skeleton present)?
# LayerStore — Why: layer 2. Checks the dirs/files `samia init` creates. A present
#              package with no store means installed-but-never-initialized.
def probe_store() -> dict:
    store = _resolve_store()
    markers = [store / "nodes", store / "chains", store / "config.json"]
    present = [m for m in markers if m.exists()]
    ok = len(present) == len(markers)
    return {
        "layer": "store",
        "ok": ok,
        "detail": (
            f"store ready at {store}"
            if ok else f"store not initialized at {store} "
                       f"({len(present)}/{len(markers)} markers present)"
        ),
        "next": None if ok else {
            "why": "The package is installed but the memory store was never created.",
            "command": "samia init",
        },
    }


# LayerDaemon — What: is the user-managed daemon reachable on its socket?
# LayerDaemon — Why: layer 3, and OPTIONAL. The daemon is opt-in: the store is the
#               persistence, the MCP server a thin client. A missing daemon is INFO,
#               not a blocker — recall/write work without it.
def probe_daemon() -> dict:
    xdg = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    sock = Path(xdg) / "samia-runtimed.sock"
    reachable = False
    if sock.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(str(sock))
                reachable = True
        except OSError:
            reachable = False
    return {
        "layer": "daemon",
        "ok": reachable,
        "optional": True,
        "detail": (
            f"daemon reachable at {sock}"
            if reachable else f"daemon not running ({sock} absent or not accepting) — OPTIONAL"
        ),
        "next": None if reachable else {
            "why": "Optional: the daemon enables background consolidation. Memory works without it.",
            "command": "samia daemon run         # opt-in; add --sandboxed for bwrap/firejail",
        },
    }


# LayerMcp — What: is the MCP server wired into a client config the user can reach?
# LayerMcp — Why: layer 4. Best-effort, read-only scan of the common Claude config
#            locations for an `asthenos-memory` mcpServers entry. Can't see the live
#            in-session MCP roster from here, so absence = "could not confirm".
def probe_mcp() -> dict:
    candidates = [
        Path.home() / ".claude.json",
        Path.home() / ".claude" / "settings.json",
        Path.cwd() / ".mcp.json",
    ]
    wired = False
    where = None
    for c in candidates:
        if not c.exists():
            continue
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        servers = data.get("mcpServers", {}) if isinstance(data, dict) else {}
        if _MCP_NAME in servers:
            wired = True
            where = str(c)
            break
    return {
        "layer": "mcp",
        "ok": wired,
        "detail": (
            f"`{_MCP_NAME}` MCP server wired in {where}"
            if wired else f"could not confirm an `{_MCP_NAME}` MCP entry in known config files"
        ),
        "next": None if wired else {
            "why": "The memory tools won't appear in Claude until the MCP server is wired.",
            "command": "claude mcp add asthenos-memory -- samia-mcp-server",
        },
    }


# RunProbe — What: run all four layers in dependency order and pick the FIRST blocking gap.
# RunProbe — Why: "one step at a time" — surface only the next actionable command
#            (skipping the optional daemon when it's the only gap) so the user is never
#            handed a wall of steps.
def run_probe() -> dict:
    layers = [probe_package(), probe_store(), probe_daemon(), probe_mcp()]
    blocking = next(
        (l for l in layers if not l["ok"] and not l.get("optional")), None
    )
    fully_wired = blocking is None and all(
        l["ok"] for l in layers if not l.get("optional")
    )
    next_step = None
    if blocking is not None:
        nxt = blocking["next"]
        next_step = {
            "layer": blocking["layer"],
            "why": nxt["why"],
            "command": nxt["command"],
            "then": "After running it, re-run `samia status` (and /reload-plugins in Claude Code).",
        }
    return {
        "fully_wired": fully_wired,
        "layers": layers,
        "next_step": next_step,
    }


def render_human(report: dict) -> None:
    """Print the probe report as a plain-language capability ladder + next step."""
    print("SAM/IA install probe — capability layers:")
    for l in report["layers"]:
        mark = "OK " if l["ok"] else ("-- " if l.get("optional") else "XX ")
        print(f"  [{mark}] {l['layer']:<8} {l['detail']}")
    print("")
    if report["fully_wired"]:
        print("All required layers present. SAM/IA should be live in Claude.")
        print("If tools don't appear yet, run /reload-plugins.")
        return
    ns = report["next_step"]
    if ns:
        print(f"Missing layer: {ns['layer']}")
        print(f"Why it can't act yet: {ns['why']}")
        print("")
        print("NEXT STEP (run this in your own shell):")
        print(f"    {ns['command']}")
        print("")
        print(f"Then: {ns['then']}")


# --------------------------------------------------------------------------
# [Asthenosphere] samia._install_probe
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      P1 — canonical core (install-UX: the `samia status` capability probe)
# Layer:      runtime front-end (read-only capability probe, stdlib-only)
# Role:       the four-layer package/store/daemon/mcp probe behind `samia status` —
#             detects the FIRST blocking gap, names it, and prints the single next
#             command. Never installs, never starts a daemon, never edits settings.
# Stability:  stable — markers track `samia init` + the daemon socket + .mcp.json; the
#             mcp scan is best-effort (can't see the live in-session MCP roster).
# ErrorModel: fail-soft everywhere — find_spec/JSON/socket errors collapse to a
#             "not present / could not confirm" verdict; never raises into the CLI.
# Depends:    stdlib only (importlib.util/json/os/shutil/socket/pathlib).
# Exposes:    probe_package/probe_store/probe_daemon/probe_mcp, run_probe(), render_human().
# --------------------------------------------------------------------------
