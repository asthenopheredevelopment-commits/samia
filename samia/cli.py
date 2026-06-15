#!/usr/bin/env python3
"""samia.cli — the user-facing `samia` command (init / daemon / mcp-server / status).

Layer 1 (Owns / Depends):
    Owns:    the `samia` console_script aggregator. Subcommands:
               init        — idempotently create the store/config + EMIT the MCP wiring
               daemon run  — start the maintenance daemon (opt-in; --sandboxed = stub)
               mcp-server  — exec the stdio MCP server (delegates to mcp_server_main)
               status      — the four-layer capability probe (delegates to _install_probe)
             Plus init_main(), the bare `samia-init` console_script alias.
    Depends: stdlib only for init/status; samia.runtime.maintenanced (lazy, daemon only)
             and samia.mcp_server_main (lazy, mcp-server only) — so `samia init` / `samia
             status` never drag in the runtime.

Layer 2 (What / Why):
    What: one thin front-end fanning out to the canonical core. `init` resolves the store
          dir (ASTHENOS_MEMORY_DIR → ~/.local/share/asthenos), creates nodes/ + chains/ +
          index.json + a conservative config.json (heavy local-model arms OFF), and prints
          the MCP wiring two ways — but never starts a daemon and never edits Claude
          settings. `daemon run` wraps maintenanced:main; --sandboxed is a fail-closed stub.
    Why:  "one canonical core, many thin front-ends" (FEAT-2026-06-14 install UX). Every
          installer path (pipx / plugin / raw) ends at this same idempotent contract, so
          there is one setup story and re-running is always safe.

CONTRACT (must hold): init is idempotent (re-run never destroys store data); init EMITS
the wiring, it does not apply it; init never starts the daemon. --sandboxed fails closed
(it never silently runs the daemon UNsandboxed when sandboxing was explicitly requested).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# StoreDefaults — What: the public store location + MCP server name.
# StoreDefaults — Why: env override first (Docker/sandbox/test redirect), then the
#                 user-data-dir norm. The server name is FIXED at "asthenos-memory" so
#                 existing tool ids (mcp__asthenos-memory__memory_search, ...) resolve.
_ENV_MEMORY_DIR = "ASTHENOS_MEMORY_DIR"
_DEFAULT_STORE = Path.home() / ".local" / "share" / "asthenos"
_MCP_NAME = "asthenos-memory"


# ---------------------------------------------------------------------------
# `samia init` — idempotent store creation + MCP-wiring emission (no daemon, no
# settings edit). Faithful to packaging/samia_init_stub.py's reference contract.
# ---------------------------------------------------------------------------


def resolve_memory_dir(explicit: str | None = None) -> Path:
    """Resolve the store dir: explicit arg → env override → default."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(_ENV_MEMORY_DIR)
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_STORE


# CreateSkeleton — What: ensure store dirs + index + config exist; report new vs present.
# CreateSkeleton — Why: the idempotent heart of the contract. mkdir(exist_ok) + write-
#                  only-if-missing means a re-run touches nothing already there, so any
#                  front-end can call `samia init` safely on an initialized box.
def create_skeleton(memory_dir: Path) -> dict:
    created: list[str] = []
    present: list[str] = []

    for sub in ("nodes", "chains"):
        d = memory_dir / sub
        if d.exists():
            present.append(str(d))
        else:
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))

    index_manifest = memory_dir / "index.json"
    if index_manifest.exists():
        present.append(str(index_manifest))
    else:
        index_manifest.write_text(
            json.dumps({"version": 1, "entries": [], "created_by": "samia init"},
                       indent=2),
            encoding="utf-8",
        )
        created.append(str(index_manifest))

    # Conservative public defaults: the local-inference arms (contradiction / fact-
    # extract) reference a local model the public build must NOT assume is present, so
    # they default OFF (operator greenlit 2026-06-15); the semantic arm is on.
    config_path = memory_dir / "config.json"
    if config_path.exists():
        present.append(str(config_path))
    else:
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mcp_server_name": _MCP_NAME,
                    "features": {
                        "contradiction_enabled": False,
                        "fact_extract_enabled": False,
                        "semantic_arm_enabled": True,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        created.append(str(config_path))

    return {"created": created, "present": present}


# EmitWiring — What: print the exact MCP wiring the user must apply, three ways.
# EmitWiring — Why: init EMITS the wiring and does NOT auto-edit Claude settings. The
#              `claude mcp add` one-liner, the `.mcp.json` block, and the plugin path are
#              all shown so the user can pick one; `samia init` stays honest about it.
def emit_wiring(memory_dir: str) -> None:
    block = {
        "mcpServers": {
            _MCP_NAME: {
                "type": "stdio",
                "command": "samia-mcp-server",
                "args": [],
                "env": {_ENV_MEMORY_DIR: memory_dir},
            }
        }
    }
    print("")
    print("=" * 64)
    print("  NEXT STEP — wire SAM/IA into your MCP client (Claude Code)")
    print("=" * 64)
    print("")
    print("  `samia init` does NOT edit your Claude settings (by design).")
    print("  Apply ONE of the following yourself:")
    print("")
    print("  A) One-line escape hatch (no plugin):")
    print("       claude mcp add asthenos-memory -- samia-mcp-server")
    print("")
    print("  B) Or add this to a .mcp.json the client reads:")
    for line in json.dumps(block, indent=2).splitlines():
        print("       " + line)
    print("")
    print("  C) If you installed via the Claude plugin, its .mcp.json already")
    print("     wires this — just run  /reload-plugins  in Claude Code.")
    print("")
    print("  The daemon is OPTIONAL and user-managed. To start it (opt-in):")
    print("       samia daemon run            # add --sandboxed for bwrap/firejail")
    print("=" * 64)


def _do_init(memory_dir_arg: str | None, quiet: bool) -> int:
    memory_dir = resolve_memory_dir(memory_dir_arg)
    result = create_skeleton(memory_dir)
    print(f"[samia init] store: {memory_dir}")
    for p in result["created"]:
        print(f"  created  {p}")
    for p in result["present"]:
        print(f"  exists   {p}  (left untouched)")
    if not result["created"]:
        print("  already initialized — no changes (idempotent re-run).")
    if not quiet:
        emit_wiring(str(memory_dir))
    return 0


def init_main(argv: list[str] | None = None) -> int:
    """Entry point for the `samia-init` console_script (bare alias for `samia init`)."""
    ap = argparse.ArgumentParser(
        prog="samia-init",
        description="Idempotently create the SAM/IA store/config and emit MCP wiring.",
    )
    ap.add_argument("--memory-dir", default=None,
                    help=f"store location (default: ${_ENV_MEMORY_DIR} or {_DEFAULT_STORE})")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the wiring printout (still creates the store)")
    args = ap.parse_args(argv)
    return _do_init(args.memory_dir, args.quiet)


# ---------------------------------------------------------------------------
# `samia daemon run` — opt-in maintenance daemon. --sandboxed is a fail-closed stub.
# ---------------------------------------------------------------------------


def _do_daemon(sandboxed: bool, passthrough: list[str]) -> int:
    if sandboxed:
        # Documented stub (D7): the bwrap/firejail confinement PROFILE is not yet wired.
        # We refuse to run UNsandboxed when sandboxing was explicitly asked for — fail
        # closed, name what's available, and point at the unsandboxed path as the choice.
        have = [t for t in ("bwrap", "firejail") if shutil.which(t)]
        print("[samia daemon] --sandboxed is not yet wired (documented stub).")
        if have:
            print(f"  Detected on PATH: {', '.join(have)} — a confinement profile is the")
            print("  remaining work (bind-mount the store dir, drop the network, etc.).")
        else:
            print("  Neither bwrap nor firejail is on PATH. Install one first, e.g.:")
            print("    sudo apt install bubblewrap     # or: sudo apt install firejail")
        print("  Refusing to start the daemon UNsandboxed when --sandboxed was requested.")
        print("  To run without confinement, re-run WITHOUT --sandboxed:  samia daemon run")
        return 2
    # Lazy import — keep the runtime out of `samia init` / `samia status`.
    from samia.runtime import maintenanced
    return maintenanced.main(passthrough)


# ---------------------------------------------------------------------------
# Argument parsing + dispatch
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="samia",
        description="SAM/IA — biologically-inspired persistent memory for LLM agents.",
    )
    sub = p.add_subparsers(dest="command", metavar="{init,daemon,mcp-server,status}")

    p_init = sub.add_parser("init", help="create the store/config + emit MCP wiring (idempotent)")
    p_init.add_argument("--memory-dir", default=None,
                        help=f"store location (default: ${_ENV_MEMORY_DIR} or {_DEFAULT_STORE})")
    p_init.add_argument("--quiet", action="store_true",
                        help="suppress the wiring printout (still creates the store)")

    p_daemon = sub.add_parser("daemon", help="run the optional maintenance daemon")
    p_daemon.add_argument("action", choices=["run"], help="daemon action (only 'run')")
    p_daemon.add_argument("--sandboxed", action="store_true",
                          help="confine under bwrap/firejail (documented stub — fails closed)")
    # Extra flags (--memory-dir / --interval / --oneshot) forward to maintenanced.

    sub.add_parser("mcp-server", help="run the stdio MCP server (what .mcp.json launches)")

    p_status = sub.add_parser("status", help="probe which install layer is missing")
    p_status.add_argument("--json", action="store_true", help="emit the report as JSON")

    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `samia` console_script — dispatch to the subcommands."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "init":
        return _do_init(args.memory_dir, args.quiet)

    if args.command == "daemon":
        return _do_daemon(args.sandboxed, extra)

    if args.command == "mcp-server":
        from samia import mcp_server_main
        mcp_server_main.main()
        return 0

    if args.command == "status":
        from samia import _install_probe
        report = _install_probe.run_probe()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _install_probe.render_human(report)
        return 0 if report["fully_wired"] else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

# --------------------------------------------------------------------------
# [Asthenosphere] samia.cli
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      P1 — canonical core (install-UX: the `samia` aggregator CLI)
# Layer:      runtime front-end (thin CLI fanning out to the core + runtime planes)
# Role:       backs the `samia` console_script — init (idempotent store + wiring emit),
#             daemon run (opt-in maintenanced wrapper; --sandboxed fails closed),
#             mcp-server (exec the stdio server), status (capability probe); plus
#             init_main() for the `samia-init` alias.
# Stability:  stable — the init contract (idempotent, emits-not-edits, no auto-daemon)
#             is frozen across every installer front-end.
# ErrorModel: init is idempotent + fail-loud on a bad --memory-dir; --sandboxed fails
#             CLOSED (never runs the daemon unsandboxed when confinement was requested);
#             runtime/server imports are lazy so init/status work on a partial install.
# Depends:    stdlib (argparse/json/os/shutil/pathlib); lazy: samia.runtime.maintenanced,
#             samia.mcp_server_main, samia._install_probe.
# Exposes:    main(), init_main(), resolve_memory_dir(), create_skeleton(), emit_wiring().
# --------------------------------------------------------------------------
