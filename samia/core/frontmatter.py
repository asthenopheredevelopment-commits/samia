"""samia.core.frontmatter — YAML-ish frontmatter parsing and serialization.

Carved from duplicated logic in memory_ia.py, memory_session_boot.py,
memory_temporal_query.py, memory_write_check.py, memory_index_compact.py
(and ~10 other tools).

The format is a minimal YAML subset used by SAM memory nodes:
  - Frontmatter delimited by `---` lines at start of file
  - One `key: value` per line
  - Scalars: bool ('true'/'false'), int, float, list ('[a, b, c]'), or str
  - Body is everything after the closing `---`

Order preservation matters because nodes are read/written by humans. We track
key insertion order so that parse → serialize round-trips are stable.

Public API:
  parse(text)       → ((fm, order), body) | (None, text)
  serialize(fm, order, body) → text
  parse_val(raw)    → typed scalar
  fmt_val(v)        → str
  read_node(path)   → (fm, order, body)
  write_node(path, fm, order, body) → None

Byte-identical output guarantee: parse → serialize on any node currently in
memory/nodes/ produces the same bytes (modulo trailing-newline normalization
that the legacy serializers also performed).
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Optional

_LIST_RE = re.compile(r"^\[(.*)\]$")


def parse_val(raw: str) -> Any:
    """Parse one frontmatter scalar.

    Returns bool, int, float, list[str], or str (raw, after .strip()).
    Empty input returns "".
    """
    raw = raw.strip()
    if not raw:
        return ""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    m = _LIST_RE.match(raw)
    if m:
        inner = m.group(1).strip()
        return [] if not inner else [p.strip() for p in inner.split(",") if p.strip()]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def fmt_val(v: Any) -> str:
    """Render a typed value back to its frontmatter string form."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def parse(text: str) -> tuple[Optional[tuple[dict, list[str]]], str]:
    """Parse frontmatter + body.

    Returns ((fm_dict, key_order), body) on success, (None, text) if no
    frontmatter is present.
    """
    if not text.startswith("---"):
        return None, text
    end = text.find("\n---", 4)
    if end < 0:
        return None, text
    header = text[4:end]
    body = text[end + 4:].lstrip("\n")
    fm: dict = {}
    order: list[str] = []
    for line in header.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        fm[k] = parse_val(v)
        order.append(k)
    return (fm, order), body


def serialize(fm: dict, order: list[str], body: str) -> str:
    """Render frontmatter + body back to text.

    Key emission rules:
      - Keys in `order` come first, in that order, but only if they're still
        present in `fm` (deletions in `fm` skip the line).
      - Keys added to `fm` after parsing (not in `order`) emit at the end in
        insertion order.
      - Trailing body has its trailing whitespace stripped and exactly one
        newline appended.

    Output shape:
      ---
      key1: val1
      key2: val2
      ---
      <body>
    """
    lines = ["---"]
    seen: set[str] = set()
    for k in order:
        if k in fm and k not in seen:
            lines.append(f"{k}: {fmt_val(fm[k])}")
            seen.add(k)
    for k, v in fm.items():
        if k not in seen:
            lines.append(f"{k}: {fmt_val(v)}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + body.rstrip() + "\n"


def read_node(path: str | Path) -> tuple[dict, list[str], str]:
    """Read a node file and return (fm, order, body).

    Raises if the file lacks frontmatter — callers that want to handle nodes
    without frontmatter should call parse() directly.
    """
    text = Path(path).read_text(encoding="utf-8")
    parsed, body = parse(text)
    if parsed is None:
        raise ValueError(f"{path}: no frontmatter found")
    fm, order = parsed
    return fm, order, body


def write_node(path: str | Path, fm: dict, order: list[str], body: str,
               override_frozen: bool = False, integrity_rewrite: bool = False) -> None:
    """Write (fm, order, body) back to a node file.

    Parameters
    ----------
    override_frozen : bool
        If True, allow writing to a frozen node (operator action).
        Default False -- writes to frozen nodes are rejected by AUD61
        target_state enforcement.
    integrity_rewrite : bool
        If True, marks this as a content-integrity erosion / recall-repair rewrite
        of an EXISTING node (eroded chars or anchor-restore) -- NOT new semantic
        content. memory_guard SKIPS contradiction_smell for it: an integrity rewrite
        trivially overlaps the node's prior version + similar nodes, so the check
        would false-positive and cascade (BUG-2026-06-08 decay<->memory_guard).
    """
    # AUD61 Phase 2: validate target_state on writes.
    # What: ensures target_state is valid and respects lifecycle rules.
    # Why: prevents invalid state values and enforces frozen/archived
    #   immutability without requiring the migration to have run first
    #   (missing target_state defaults to "live" with no rejection).
    try:
        from samia.runtime.migrations.aud61_target_state import (
            validate_target_state,
            DEFAULT_TARGET_STATE,
        )
        target_state = fm.get("target_state")
        # What: read current state from existing file if it exists.
        # Why: lifecycle restrictions depend on the node's current state,
        #   not just the incoming value.
        current_state = None
        p = Path(path)
        if p.exists():
            try:
                existing_fm, _, _ = read_node(p)
                current_state = existing_fm.get("target_state")
            except (ValueError, OSError):
                pass
        ok, reason = validate_target_state(
            target_state, current_state=current_state, override=override_frozen,
        )
        if not ok:
            raise ValueError(f"AUD61 target_state validation failed: {reason}")
        # What: default target_state to "live" if absent.
        # Why: post-migration, all nodes should have target_state; defaulting
        #   ensures backward compatibility with code that hasn't been updated.
        if target_state is None:
            fm["target_state"] = DEFAULT_TARGET_STATE
            if "target_state" not in order:
                order.append("target_state")
    except ImportError:
        pass  # fail-open: migration module not yet available

    # FEAT-opencode-atoms-integration Phase 1: validate runtime provenance field.
    # What: ensures runtime is one of the allowed values if present.
    # Why: runtime provenance tags nodes by originating harness (opencode vs main).
    #   Invalid values are rejected fail-fast to prevent data corruption.
    #   Missing runtime is acceptable -- readers default to "main".
    VALID_RUNTIMES = {"opencode", "main"}
    runtime = fm.get("runtime")
    if runtime is not None and runtime not in VALID_RUNTIMES:
        raise ValueError(
            f"runtime must be 'opencode' or 'main', got {runtime!r}"
        )

    # AUD48 Phase 1: stage the write for observation before committing.
    # What: logs the write to the memory_guard staging buffer.
    # Why: observation-only (default-pass); the write always proceeds.
    #      Phase 2 will add consensus validation between stage and commit.
    try:
        from samia.runtime.memory_guard import stage_write
        stage_write(
            kind="write_node",
            target=str(path),
            payload={
                "name": fm.get("name", ""),
                "keys": list(fm.keys()),
                "frontmatter_type": fm.get("type"),  # templated-content exclusion gate
                "integrity_rewrite": integrity_rewrite,  # BUG-2026-06-08 decay exclusion gate
            },
            caller="samia.core.frontmatter.write_node",
        )
    except Exception:
        pass  # fail-open: staging failure must never block the write

    # FEAT-2026-06-08 anchor-capture-on-write-path: capture/refresh the pristine recovery
    # anchor for the integrity decay axis. capture_on_write fired ONLY on the MCP write_node
    # op, so session-offloads + internal writers bypassed it and left nodes anchor-less (and
    # therefore un-erodable). Capture here so EVERY genuine writer anchors automatically. The
    # integrity_rewrite gate excludes erosion/repair rewrites (they must NEVER re-anchor a
    # degraded served body). Mutates fm/order (the integrity reset on a genuine rewrite) BEFORE
    # the serialize below so the reset persists. Lazy import dodges the frontmatter<->integrity
    # circular dep (same pattern as stage_write); fail-open so an anchor failure never blocks.
    if not integrity_rewrite:
        try:
            from samia.core import integrity as _integrity
            _p = Path(path)
            _integrity.capture_on_genuine_write(
                _p.parent.parent, _p.stem, fm, order, body)
        except Exception:
            pass  # fail-open: anchor capture must never block the write

    # Fresh-store bootstrap: create the parent dir (nodes/) so the first write
    # into a brand-new memory store does not FileNotFoundError. Internal writers
    # already do this ad hoc; doing it here makes write_node (and the MCP write
    # path that routes through it) self-bootstrapping.
    _out = Path(path)
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text(serialize(fm, order, body), encoding="utf-8")


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.frontmatter
# Phase:      AUD15 (original) + AUD48 Phase 1 (staging hook)
#             + AUD61 Phase 2 (target_state write-time validation)
#             + FEAT-opencode-atoms-integration Phase 1 (runtime provenance validation)
#             + BUG-2026-05-13 templated-content exclusion (frontmatter_type in payload)
# Layer:      core (pure library, no daemon dependency)
# Stability:  v1.3 -- frontmatter_type added to stage_write payload
# ErrorModel: read_node raises ValueError on missing frontmatter;
#             write_node raises ValueError on AUD61 target_state violation;
#             AUD48 staging is fail-open (never blocks writes).
# Depends:    re, pathlib (stdlib).
#             samia.runtime.memory_guard (optional, fail-open).
#             samia.runtime.migrations.aud61_target_state (optional, fail-open).
# Exposes:    parse, serialize, parse_val, fmt_val, read_node, write_node.
# Lines:      ~200
# --------------------------------------------------------------------------
