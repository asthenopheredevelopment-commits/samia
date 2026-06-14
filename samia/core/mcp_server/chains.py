"""samia.core.mcp_server.chains — the chain listing + edge-temporal query arm.

Layer 1 (Owns / Depends):
    Owns:    the chain tool logic — the plain listing/read (memory_list_chains,
             memory_get_chain) and the edge-temporal queries over a chain's
             time-validated edges (memory_chain_query_at, memory_chain_traverse_at,
             memory_chain_set_edge, memory_chain_invalidate_edge,
             memory_chain_snapshot_at).
    Depends: .config (_nodes_dir, _chains_dir, json, Any/Path). Lazy per-call:
             samia.core.temporal (frontmatter split + date parse) and samia.core.chain
             (the edge-temporal engine) — function-local to keep them off the package
             import path.

Layer 2 (What / Why):
    What: every chain tool's underlying logic, parameterized on memory_dir.
          list/get read the chains/ store and enrich members with their node titles;
          the *_at tools delegate to chain.query_at/traverse_at/snapshot_at after an
          ISO-date parse, and set/invalidate edge mutate the chain's edge set.
    Why:  the chain + edge-temporal tools are a single cohesive seam (the temporal
          knowledge-graph surface) distinct from node recall/write; isolating them
          keeps the date-validated edge engine's dependency off the read/write paths.
"""

from __future__ import annotations

from .config import (
    Any,
    Path,
    json,
    _chains_dir,
    _nodes_dir,
)


def memory_list_chains(memory_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cp in sorted(_chains_dir(memory_dir).glob("*.json")):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except Exception as e:
            out.append({"chain": cp.stem, "error": str(e)})
            continue
        members = data.get("members") or data.get("nodes") or []
        head = None
        if members:
            first = members[0]
            head = first.get("file") if isinstance(first, dict) else first
        out.append({
            "chain": cp.stem,
            "node_count": len(members),
            "tier": data.get("tier", "warm"),
            "head": head,
        })
    return out


def memory_get_chain(memory_dir: Path, name: str) -> dict[str, Any]:
    from .. import temporal as _tq
    p = _chains_dir(memory_dir) / f"{name}.json"
    if not p.exists():
        return {"error": f"chain not found: {name}"}
    data = json.loads(p.read_text(encoding="utf-8"))
    enriched = []
    members = data.get("members") or data.get("nodes") or []
    nodes_dir = _nodes_dir(memory_dir)
    for m in members:
        if isinstance(m, dict):
            f = m.get("file") or m.get("node") or ""
        else:
            f = m
        if not f:
            continue
        np_path = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
        if not np_path.suffix:
            np_path = np_path.with_suffix(".md")
        title = np_path.stem
        if np_path.exists():
            fm_lines, _ = _tq.split_frontmatter(np_path.read_text(encoding="utf-8"))
            for line in fm_lines:
                if line.startswith("name:"):
                    title = line.split(":", 1)[1].strip()
                    break
        enriched.append({"node": np_path.name, "title": title})
    data["enriched_nodes"] = enriched
    return data


# ---------------------------------------------------------------------------
# Edge-temporal chain queries
# ---------------------------------------------------------------------------


def memory_chain_query_at(memory_dir: Path, chain: str, at: str) -> list[dict[str, Any]]:
    from .. import temporal as _tq
    from .. import chain as _ct
    d = _tq.parse_date(at)
    if not d:
        return [{"error": "at must be ISO date"}]
    return _ct.query_at(memory_dir, chain, d)


def memory_chain_traverse_at(memory_dir: Path, chain: str, start: str, at: str,
                             depth: int = 3) -> list[dict[str, Any]]:
    from .. import temporal as _tq
    from .. import chain as _ct
    d = _tq.parse_date(at)
    if not d:
        return [{"error": "at must be ISO date"}]
    return _ct.traverse_at(memory_dir, chain, start, d, depth=depth)


def memory_chain_set_edge(
    memory_dir: Path,
    chain: str,
    from_addr: str,
    to_addr: str,
    valid_from: str | None = None,
    valid_to: str | None = None,
    label: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    from .. import chain as _ct
    out = _ct.set_edge(_chains_dir(memory_dir), chain, from_addr, to_addr,
                       valid_from=valid_from, valid_to=valid_to,
                       label=label, confidence=confidence)
    return out or {"error": "set_edge returned no result"}


def memory_chain_invalidate_edge(memory_dir: Path, chain: str, from_addr: str,
                                  to_addr: str, on: str,
                                  label: str | None = None) -> dict[str, Any]:
    from .. import chain as _ct
    n = _ct.invalidate_edge(_chains_dir(memory_dir), chain, from_addr, to_addr, on, label=label)
    return {"closed": n}


def memory_chain_snapshot_at(memory_dir: Path, at: str) -> dict[str, Any]:
    from .. import temporal as _tq
    from .. import chain as _ct
    d = _tq.parse_date(at)
    if not d:
        return {"error": "at must be ISO date"}
    return _ct.snapshot_at(memory_dir, d)


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.mcp_server.chains
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.mcp_server monolith during
#             modularization (the Chains + Edge-temporal query sections).
# Layer:      core (pure library, no daemon dependency)
# Role:       the chain listing + edge-temporal query arm — memory_list_chains /
#             memory_get_chain plus the *_at edge-temporal tools (query_at / traverse_at /
#             set_edge / invalidate_edge / snapshot_at).
# Stability:  stable — behavior byte-identical to the monolith's Chains + Edge-temporal
#             sections; only the imports moved behind .config.
# ErrorModel: list/get fail-soft per chain file (a bad JSON file yields an {error} entry,
#             never a raise); the *_at tools return an {error} dict on a non-ISO date.
# Depends:    .config (_nodes_dir, _chains_dir, json, Any/Path). Lazy per-call:
#             samia.core.temporal, samia.core.chain.
# Exposes:    memory_list_chains, memory_get_chain, memory_chain_query_at,
#             memory_chain_traverse_at, memory_chain_set_edge,
#             memory_chain_invalidate_edge, memory_chain_snapshot_at.
# Lines:      164
# --------------------------------------------------------------------------
