"""samia.runtime.migrations.aud61_target_state -- target_state frontmatter migration.

Layer 1 (Owns / Depends):
    Owns:    One-time migration that adds target_state field to all SAM/IA
             memory nodes. Dry-run by default; --apply for real writes.
    Depends: samia.core.frontmatter (read_node, write_node, parse, serialize)
             pathlib, json, argparse (stdlib)

Layer 2 (What / Why):
    What: Scans every .md node in memory/nodes/ and every .frozen.json in
          memory/archive/, assigns target_state based on heuristics:
            - hot/warm tier  -> live
            - cold tier      -> live (conservative default)
            - frozen tier    -> frozen
            - archive/*.frozen.json -> archived
          Prints a dry-run report by default. With --apply, writes the
          target_state field into each node's frontmatter.
    Why:  AUD61 requires a machine-readable lifecycle state field so that
          tier_flow can skip frozen nodes, archived nodes are excluded from
          active queries, and the operator has declarative control over node
          lifecycle. Without this migration, existing nodes lack the field
          and write-time enforcement would reject all legacy writes.

Design doc: AUD61_sam_frontmatter_target_state.md

Usage:
    # Dry-run (default): prints proposed assignments
    python3 -m samia.runtime.migrations.aud61_target_state /path/to/memory

    # Apply: writes target_state into each node
    python3 -m samia.runtime.migrations.aud61_target_state /path/to/memory --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VALID_TARGET_STATES — What: the three valid target_state values.
# VALID_TARGET_STATES — Why: used by validation in both migration and
#   write-time enforcement. Defined here as the canonical source.
VALID_TARGET_STATES = {"live", "frozen", "archived"}

# DEFAULT_TARGET_STATE — What: the default target_state for new nodes.
# DEFAULT_TARGET_STATE — Why: conservative default -- all nodes are live
#   unless explicitly frozen or archived by the operator.
DEFAULT_TARGET_STATE = "live"


# ---------------------------------------------------------------------------
# Heuristic assignment
# ---------------------------------------------------------------------------


def assign_target_state(fm: dict[str, Any], source: str = "nodes") -> str:
    """Determine target_state for a node based on its frontmatter and location.

    What: applies heuristic rules to assign one of {live, frozen, archived}.
    Why:  the migration needs a deterministic, conservative assignment that
          the operator can review and override. The rules are:
          - archive/*.frozen.json -> archived (already out of active store)
          - tier=frozen -> frozen (tier_flow already considers these inert)
          - everything else -> live (conservative; operator can freeze later)

    Parameters
    ----------
    fm : dict
        Parsed frontmatter dict.
    source : str
        "nodes" for live .md files, "archive" for .frozen.json files.

    Returns
    -------
    str
        One of "live", "frozen", "archived".
    """
    if source == "archive":
        return "archived"

    tier = str(fm.get("tier", "")).lower()
    if tier == "frozen":
        return "frozen"

    # What: all other tiers (hot, warm, cold) default to live.
    # Why: cold nodes are still queryable and may be re-accessed; marking
    #   them frozen would prevent tier_flow from promoting them on re-access.
    return "live"


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------


def migrate_nodes(memory_dir: Path, apply: bool = False) -> list[dict[str, Any]]:
    """Scan all nodes and assign target_state.

    What: reads every .md in nodes/ and every .frozen.json in archive/,
          assigns target_state, and optionally writes back.
    Why:  one-time migration to populate the target_state field on all
          existing nodes before write-time enforcement is enabled.

    Parameters
    ----------
    memory_dir : Path
        Root memory directory (parent of nodes/ and archive/).
    apply : bool
        If True, write the target_state field into each node. If False
        (default), only report proposed assignments.

    Returns
    -------
    list of dicts with keys: path, name, old_state, new_state, source, applied.
    """
    from samia.core import frontmatter as _fm

    results: list[dict[str, Any]] = []
    nodes_dir = memory_dir / "nodes"
    archive_dir = memory_dir / "archive"

    # What: process live .md nodes in nodes/.
    # Why: these are the active memory store; most will be assigned "live".
    if nodes_dir.exists():
        for md_path in sorted(nodes_dir.glob("*.md")):
            try:
                fm, order, body = _fm.read_node(md_path)
            except (ValueError, OSError) as exc:
                results.append({
                    "path": str(md_path),
                    "name": md_path.stem,
                    "error": str(exc),
                    "source": "nodes",
                    "applied": False,
                })
                continue

            old_state = fm.get("target_state", None)
            new_state = assign_target_state(fm, source="nodes")

            record = {
                "path": str(md_path),
                "name": fm.get("name", md_path.stem),
                "old_state": old_state,
                "new_state": new_state,
                "tier": fm.get("tier", "unknown"),
                "source": "nodes",
                "applied": False,
            }

            if apply and old_state != new_state:
                fm["target_state"] = new_state
                if "target_state" not in order:
                    order.append("target_state")
                _fm.write_node(md_path, fm, order, body)
                record["applied"] = True

            results.append(record)

    # What: process archived .frozen.json files in archive/.
    # Why: these are already out of the active store; target_state=archived
    #   is purely declarative metadata for consistency.
    if archive_dir.exists():
        for frozen_path in sorted(archive_dir.glob("*.frozen.json")):
            try:
                data = json.loads(frozen_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                results.append({
                    "path": str(frozen_path),
                    "name": frozen_path.stem,
                    "error": str(exc),
                    "source": "archive",
                    "applied": False,
                })
                continue

            fm = data.get("frontmatter", data.get("fm", {}))
            old_state = fm.get("target_state", None)
            new_state = "archived"

            record = {
                "path": str(frozen_path),
                "name": fm.get("name", frozen_path.stem),
                "old_state": old_state,
                "new_state": new_state,
                "source": "archive",
                "applied": False,
            }

            if apply and old_state != new_state:
                fm["target_state"] = new_state
                if "frontmatter" in data:
                    data["frontmatter"] = fm
                elif "fm" in data:
                    data["fm"] = fm
                else:
                    data["target_state"] = new_state
                frozen_path.write_text(
                    json.dumps(data, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                record["applied"] = True

            results.append(record)

    return results


# ---------------------------------------------------------------------------
# Validation helpers (used by write-time enforcement in frontmatter)
# ---------------------------------------------------------------------------


def validate_target_state(
    target_state: Any,
    current_state: str | None = None,
    override: bool = False,
) -> tuple[bool, str]:
    """Validate a target_state value for a write operation.

    What: checks that target_state is valid and respects lifecycle rules.
    Why:  AUD61 Phase 2 write-time enforcement -- reject invalid states
          and enforce frozen/archived write restrictions.

    Parameters
    ----------
    target_state : Any
        The target_state value from the write payload.
    current_state : str or None
        The current target_state of the node being written (None if new).
    override : bool
        If True, allow writes to frozen nodes (operator action).

    Returns
    -------
    (ok, reason) tuple. ok=True if the write should proceed.
    """
    if target_state is None:
        return True, "target_state absent; will default to 'live'"

    if not isinstance(target_state, str):
        return False, f"target_state must be a string, got {type(target_state).__name__}"

    if target_state not in VALID_TARGET_STATES:
        return False, (
            f"target_state '{target_state}' is invalid; "
            f"must be one of: {', '.join(sorted(VALID_TARGET_STATES))}"
        )

    # What: enforce lifecycle restrictions on frozen and archived nodes.
    # Why: frozen nodes are operator-protected; archived nodes are immutable
    #   historical records. Writes to either require explicit intent.
    if current_state == "archived":
        return False, (
            "cannot write to an archived node; transition to 'live' first"
        )

    if current_state == "frozen" and not override:
        return False, (
            "cannot write to a frozen node without override=True "
            "(operator action required)"
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(results: list[dict[str, Any]], apply: bool) -> None:
    """Print a human-readable migration report.

    What: formats the migration results as a summary table.
    Why:  operator review before --apply; post-apply confirmation.
    """
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n{'='*60}")
    print(f"  AUD61 target_state migration [{mode}]")
    print(f"{'='*60}\n")

    errors = [r for r in results if "error" in r]
    assigned = [r for r in results if "error" not in r]

    by_state: dict[str, int] = {}
    for r in assigned:
        st = r["new_state"]
        by_state[st] = by_state.get(st, 0) + 1

    already_set = sum(1 for r in assigned if r["old_state"] == r["new_state"])
    changed = sum(1 for r in assigned if r["old_state"] != r["new_state"])

    print(f"  Total nodes scanned:  {len(results)}")
    print(f"  Errors (skipped):     {len(errors)}")
    print(f"  Already correct:      {already_set}")
    print(f"  Changes proposed:     {changed}")
    print(f"  By target_state:      {by_state}")
    print()

    if changed > 0 and not apply:
        print("  Run with --apply to write changes.\n")

    if errors:
        print("  Errors:")
        for r in errors:
            print(f"    {r['path']}: {r.get('error', '?')}")
        print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the AUD61 migration.

    What: parses args, runs migration, prints report.
    Why:  standalone script for operator to run manually with review.
    """
    parser = argparse.ArgumentParser(
        description="AUD61: Add target_state field to all SAM/IA memory nodes."
    )
    parser.add_argument(
        "memory_dir",
        type=Path,
        help="Root memory directory (parent of nodes/ and archive/).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes (default: dry-run).",
    )
    args = parser.parse_args(argv)

    if not args.memory_dir.exists():
        print(f"ERROR: memory_dir does not exist: {args.memory_dir}", file=sys.stderr)
        return 1

    results = migrate_nodes(args.memory_dir, apply=args.apply)
    _print_report(results, args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.migrations.aud61_target_state
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD61 -- Phase 1 (migration script)
# Layer:      runtime/migrations
# Role:       one-time migration that stamps a target_state field onto every memory
#             node (tier/path heuristics -> live/frozen/archived); dry-run by default,
#             --apply to write.
# Stability:  v1.0 -- one-time migration, stable after first run
# ErrorModel: fail-open per node (skip + log errors, continue); dry-run
#             by default prevents accidental writes.
# Depends:    samia.core.frontmatter (read_node, write_node).
#             pathlib, json, argparse, sys (stdlib).
# Exposes:    migrate_nodes, assign_target_state, validate_target_state,
#             VALID_TARGET_STATES, DEFAULT_TARGET_STATE, main.
# Lines:      366
# --------------------------------------------------------------------------
