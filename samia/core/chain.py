"""samia.core.chain — edge-level temporal intervals for SAM chains.

Layer 1 (Owns / Depends):
    Owns:    load_chain, save_chain, list_chains — chain-manifest I/O.
             member_node_path, member_addrs, node_valid_from — member resolution.
             add_edge, invalidate_edge, set_edge, strip_member — edge/membership ops.
             edge_valid_at — the point-in-time edge predicate.
             migrate_linear_edges — backfill linear follows-edges on edgeless chains.
             show_chain, query_at, traverse_at, snapshot_at — time-anchored queries.
             ChainNotFound — the missing-manifest exception (a FileNotFoundError).
    Depends: stdlib only (datetime, json, collections, pathlib, typing).
             samia.core.temporal (read_node/parse_date/fm_get) via the lazy _tq()
             helper. samia.runtime.memory_guard.stage_write (lazy, observation-only,
             fail-open) inside the mutating ops.
Layer 2 (What / Why):
    What: a chain manifest (chains/<chain>.json) holds members[] plus an edges[]
          list; every edge carries its own {from, to, valid_from, valid_to, label,
          confidence} interval. The edge ops add/close/modify those intervals;
          edge_valid_at tests whether an edge's [vf, vt] contains a date; and the
          queries (query_at / traverse_at / snapshot_at) only follow edges valid AT
          a given point in time. migrate_linear_edges seeds follows-edges between
          consecutive members for legacy edgeless chains.
    Why:  this is the bi-temporal EDGE layer (design doc §1.1) — node validity says
          when a FACT held, edge validity says when a RELATIONSHIP held, so a graph
          traversal can reconstruct the chain "as of" any date (e.g. which node
          superseded which, when). null valid_to means "still valid" (an open
          interval), which is why the predicates treat a falsy/`"null"` bound as
          open. The library plane returns plain dicts / prints so a CLI and the MCP
          server share one query semantics. The memory_guard staging is
          observation-only (default-pass) and fail-open: it can log but never block
          or break a chain write.

Layer 3 (Changelog):
    (carved from memory_chain_temporal.py — library plane extraction; acceptance:
     byte-identical to pre-refactor CLI output, design doc §8.1. GATE6 rerouted the
     temporal-helper import off the tools/-dir shim onto samia.core.temporal.
     AUD48 Phase 1 added the observation-only memory_guard staging in the mutators.)
"""

from __future__ import annotations

import datetime as _dt
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional


# GATE6: temporal helpers (read_node/parse_date/fm_get) come from the sibling
# core module directly. Was a legacy reachback into the tools/-dir
# memory_temporal_query shim (which itself only re-exports samia.core.temporal).
# Kept as a function so the call sites stay unchanged; lazy to avoid any
# import-cycle surprise (temporal is a leaf, but this mirrors the old laziness).
def _tq():
    """Return the samia.core.temporal helper module."""
    from . import temporal as _temporal
    return _temporal


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class ChainNotFound(FileNotFoundError):
    """Raised by load_chain when a chain manifest is missing.

    Subclasses FileNotFoundError so callers can catch it with `except Exception`
    (it previously raised SystemExit, which `except Exception` cannot catch and
    which kills the host process — unacceptable for a library / MCP surface).
    """


def load_chain(chains_dir: Path, name: str) -> dict:
    p = chains_dir / f"{name}.json"
    if not p.exists():
        raise ChainNotFound(f"chain not found: {name}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_chain(chains_dir: Path, name: str, data: dict) -> None:
    # AUD48 Phase 1: stage the chain write for observation before committing.
    # What: logs the save_chain call to the memory_guard staging buffer.
    # Why: observation-only (default-pass); the write always proceeds.
    try:
        from samia.runtime.memory_guard import stage_write
        stage_write(
            kind="save_chain",
            target=name,
            payload={"edge_count": len(data.get("edges", []))},
            caller="samia.core.chain.save_chain",
        )
    except Exception:
        pass  # fail-open: staging failure must never block the write
    # Fresh-store bootstrap: create chains/ so the first chain write into a
    # brand-new store does not FileNotFoundError.
    chains_dir.mkdir(parents=True, exist_ok=True)
    p = chains_dir / f"{name}.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_chains(chains_dir: Path) -> list[str]:
    return sorted(p.stem for p in chains_dir.glob("*.json"))


# member_node_path — What: resolve a member's addr to its on-disk node Path, honoring
#     both "nodes/"-prefixed and bare filenames and supplying a ".md" suffix if absent.
def member_node_path(memory_dir: Path, chain: dict, addr: str) -> Optional[Path]:
    nodes_dir = memory_dir / "nodes"
    for m in chain.get("members") or []:
        a = m.get("addr") if isinstance(m, dict) else None
        if a == addr:
            f = m.get("file") if isinstance(m, dict) else None
            if not f:
                return None
            p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
            if not p.suffix:
                p = p.with_suffix(".md")
            return p
    return None
# member_node_path — Why: member `file` fields are stored inconsistently (some carry the
#     "nodes/" prefix, some don't, some omit the extension), so this normalizes all three
#     forms to one Path — returning None (not raising) for an addr that isn't a member.


def member_addrs(chain: dict) -> list[str]:
    out: list[str] = []
    for m in chain.get("members") or []:
        if isinstance(m, dict) and m.get("addr"):
            out.append(m["addr"])
    return out


def strip_member(chains_dir: Path, node: str) -> dict:
    """FEAT-2026-06-07 P0: remove a dead node from EVERY chain — drop its members[] entry
    and any edges whose from/to was that member's addr. Part of the forget_node cascade so a
    frozen/merged/purged node leaves no dangling chain membership or edge. Idempotent; only
    rewrites chains that actually contained the node. Returns counts.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    chains_touched = members_removed = edges_removed = 0
    for name in list_chains(chains_dir):
        try:
            data = load_chain(chains_dir, name)
        except (SystemExit, FileNotFoundError):
            continue
        # DeadAddrSweep — What: in each chain, drop the dead node's members[] entry
        #     (collecting its addr) and any edge touching that addr; only rewrite chains
        #     that actually contained the node.
        members = data.get("members") or []
        dead_addrs: set[str] = set()
        kept_members = []
        for m in members:
            f = m.get("file") if isinstance(m, dict) else None
            base = f.split("/")[-1] if f else None
            if base == fname:
                if m.get("addr"):
                    dead_addrs.add(m["addr"])
                members_removed += 1
            else:
                kept_members.append(m)
        if not dead_addrs and len(kept_members) == len(members):
            continue  # node not in this chain
        edges = data.get("edges") or []
        kept_edges = [e for e in edges
                      if e.get("from") not in dead_addrs and e.get("to") not in dead_addrs]
        edges_removed += len(edges) - len(kept_edges)
        data["members"] = kept_members
        data["edges"] = kept_edges
        save_chain(chains_dir, name, data)
        chains_touched += 1
    return {"chains_touched": chains_touched,
            "members_removed": members_removed,
            "edges_removed": edges_removed}
# DeadAddrSweep — Why: a forgotten/merged/purged node must leave NO dangling membership
#     or edge (a traversal hitting a vanished endpoint would dead-end), so the cascade
#     removes both. Idempotent — a chain without the node is left byte-identical — so the
#     forget cascade is safe to re-run.


def node_valid_from(memory_dir: Path, chain: dict, addr: str) -> Optional[_dt.date]:
    p = member_node_path(memory_dir, chain, addr)
    if not p or not p.exists():
        return None
    tq = _tq()
    fm_lines, _ = tq.read_node(p)
    return tq.parse_date(tq.fm_get(fm_lines, "valid_from"))


# ---------------------------------------------------------------------------
# Edge ops
# ---------------------------------------------------------------------------


def _norm_edge(e: dict) -> dict:
    return {
        "from": e["from"],
        "to": e["to"],
        "valid_from": e.get("valid_from"),
        "valid_to": e.get("valid_to"),
        "label": e.get("label", "follows"),
        "confidence": float(e.get("confidence", 1.0)),
    }


def add_edge(chains_dir: Path, chain_name: str, frm: str, to: str,
             valid_from: Optional[str], valid_to: Optional[str] = None,
             label: str = "follows", confidence: float = 1.0) -> dict:
    # AUD48 Phase 1: stage the edge addition for observation.
    # What: logs the add_edge call to the memory_guard staging buffer.
    # Why: observation-only; the edge write always proceeds. save_chain()
    #      also stages, giving two records per add_edge (semantic + persist).
    try:
        from samia.runtime.memory_guard import stage_write
        stage_write(
            kind="add_edge",
            target=chain_name,
            payload={"from": frm, "to": to, "label": label},
            caller="samia.core.chain.add_edge",
        )
    except Exception:
        pass  # fail-open
    chain = load_chain(chains_dir, chain_name)
    addrs = set(member_addrs(chain))
    if frm not in addrs or to not in addrs:
        raise SystemExit(f"edge endpoints must be members: {frm}->{to} not in {sorted(addrs)}")
    edges = chain.setdefault("edges", [])
    edge = _norm_edge({
        "from": frm, "to": to, "label": label,
        "valid_from": valid_from, "valid_to": valid_to,
        "confidence": confidence,
    })
    edges.append(edge)
    save_chain(chains_dir, chain_name, chain)
    return edge


def invalidate_edge(chains_dir: Path, chain_name: str, frm: str, to: str,
                    on: str, label: Optional[str] = None) -> int:
    # AUD48 Phase 1: stage the edge invalidation for observation.
    # What: logs the invalidate_edge call to the memory_guard staging buffer.
    # Why: observation-only; the invalidation always proceeds.
    try:
        from samia.runtime.memory_guard import stage_write
        stage_write(
            kind="invalidate_edge",
            target=chain_name,
            payload={"from": frm, "to": to, "on": on},
            caller="samia.core.chain.invalidate_edge",
        )
    except Exception:
        pass  # fail-open
    chain = load_chain(chains_dir, chain_name)
    closed = 0
    for e in chain.get("edges") or []:
        if e["from"] == frm and e["to"] == to and e.get("valid_to") in (None, "null"):
            if label is None or e.get("label") == label:
                e["valid_to"] = on
                closed += 1
    save_chain(chains_dir, chain_name, chain)
    return closed


def set_edge(chains_dir: Path, chain_name: str, frm: str, to: str,
             valid_from: Optional[str] = None, valid_to: Optional[str] = None,
             label: Optional[str] = None,
             confidence: Optional[float] = None) -> Optional[dict]:
    """Modify the latest matching edge in place; create one if none exists."""
    chain = load_chain(chains_dir, chain_name)
    edges = chain.setdefault("edges", [])
    target = None
    for e in reversed(edges):
        if e["from"] == frm and e["to"] == to:
            if label is None or e.get("label") == label:
                target = e
                break
    if target is None:
        return add_edge(chains_dir, chain_name, frm, to,
                        valid_from=valid_from, valid_to=valid_to,
                        label=label or "follows",
                        confidence=confidence if confidence is not None else 1.0)
    if valid_from is not None:
        target["valid_from"] = valid_from
    if valid_to is not None:
        target["valid_to"] = valid_to if valid_to.lower() != "null" else None
    if label is not None:
        target["label"] = label
    if confidence is not None:
        target["confidence"] = float(confidence)
    save_chain(chains_dir, chain_name, chain)
    return target


# edge_valid_at — What: is date `at` inside edge e's [valid_from, valid_to] interval?
def edge_valid_at(e: dict, at: _dt.date) -> bool:
    tq = _tq()
    vf = tq.parse_date(e.get("valid_from")) if e.get("valid_from") else None
    vt = tq.parse_date(e.get("valid_to")) if e.get("valid_to") not in (None, "null") else None
    if vf and at < vf:
        return False
    if vt and at > vt:
        return False
    return True
# edge_valid_at — Why: an open end — missing valid_from, or valid_to that is None/"null"
#     (still valid) — is treated as unbounded, so each bound only constrains when present;
#     this is the gate every point-in-time query uses to decide which edges to follow.


# ---------------------------------------------------------------------------
# Migration: linear edges between consecutive members
# ---------------------------------------------------------------------------


# migrate_linear_edges — What: for every chain with NO edges, synthesize a linear
#     "follows" edge between each consecutive member pair (dating it from the later of the
#     two nodes' valid_from); a <2-member chain just gets an empty edges[] list.
def migrate_linear_edges(memory_dir: Path, dry_run: bool = False) -> dict:
    chains_dir = memory_dir / "chains"
    written: list[str] = []
    for cname in list_chains(chains_dir):
        chain = load_chain(chains_dir, cname)
        if chain.get("edges"):
            continue
        addrs = member_addrs(chain)
        if len(addrs) < 2:
            chain["edges"] = []
            if not dry_run:
                save_chain(chains_dir, cname, chain)
            written.append(cname)
            continue
        edges = []
        for a, b in zip(addrs, addrs[1:]):
            vf_a = node_valid_from(memory_dir, chain, a)
            vf_b = node_valid_from(memory_dir, chain, b)
            cands = [d for d in (vf_a, vf_b) if d]
            vf = max(cands).isoformat() if cands else None
            edges.append({
                "from": a, "to": b,
                "valid_from": vf, "valid_to": None,
                "label": "follows", "confidence": 1.0,
            })
        chain["edges"] = edges
        if not dry_run:
            save_chain(chains_dir, cname, chain)
        written.append(cname)
    print(f"[chain_temporal] {'would update' if dry_run else 'updated'} {len(written)} chains with edge intervals")
    return {"chains": written}
# migrate_linear_edges — Why: legacy chains predate the edge layer, so point-in-time
#     traversal needs SOME edge per adjacency to follow; skip-if-edges-present keeps it
#     idempotent (never clobbers hand-authored edges). The edge is dated by max(vf_a, vf_b)
#     because the relationship can't predate its later endpoint coming into existence.


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def show_chain(memory_dir: Path, name: str) -> None:
    chains_dir = memory_dir / "chains"
    tq = _tq()
    chain = load_chain(chains_dir, name)
    print(f"chain: {name}  ({len(chain.get('members') or [])} members)")
    addrs = member_addrs(chain)
    for a in addrs:
        p = member_node_path(memory_dir, chain, a)
        vf = node_valid_from(memory_dir, chain, a)
        title = a
        if p and p.exists():
            fm, _ = tq.read_node(p)
            title = tq.fm_get(fm, "name") or a
        print(f"  {a}  valid_from={vf.isoformat() if vf else '?':12s}  {title}")
    edges = chain.get("edges") or []
    print(f"\n  {len(edges)} edges:")
    for e in edges:
        vt = e.get("valid_to") or "still"
        print(f"    {e['from']}  --{e.get('label','follows')}-->  {e['to']}"
              f"   [{e.get('valid_from') or '?'} … {vt}]"
              f"   conf={e.get('confidence', 1.0)}")


# query_at — What: return the members of a chain that are "live" at date `at` — those
#     touched by an edge valid at `at`; an edgeless chain returns ALL members.
def query_at(memory_dir: Path, chain_name: str, at: _dt.date) -> list[dict]:
    chains_dir = memory_dir / "chains"
    tq = _tq()
    chain = load_chain(chains_dir, chain_name)
    keep_edges = [e for e in (chain.get("edges") or []) if edge_valid_at(e, at)]
    reachable: set[str] = set()
    for e in keep_edges:
        reachable.add(e["from"])
        reachable.add(e["to"])
    out = []
    for a in member_addrs(chain):
        # EdgelessFallthrough — every member is kept when the chain has no edges at all
        #     (nothing to time-filter on), otherwise only edge-reachable members survive.
        if a in reachable or not (chain.get("edges") or []):
            p = member_node_path(memory_dir, chain, a)
            title = a
            if p and p.exists():
                fm, _ = tq.read_node(p)
                title = tq.fm_get(fm, "name") or a
            out.append({"addr": a, "title": title,
                        "node": p.name if p else None})
    return out


# traverse_at — What: breadth-first walk forward from `start` along edges valid at date
#     `at`, up to `depth` hops, returning each reached member once with its hop distance.
def traverse_at(memory_dir: Path, chain_name: str, start: str, at: _dt.date,
                depth: int = 3) -> list[dict]:
    chains_dir = memory_dir / "chains"
    tq = _tq()
    chain = load_chain(chains_dir, chain_name)
    valid = [e for e in (chain.get("edges") or []) if edge_valid_at(e, at)]
    # ForwardAdjacency — build a from->edges map over ONLY the time-valid edges so the
    #     BFS naturally follows the graph as it stood at `at`.
    fwd: dict[str, list[dict]] = defaultdict(list)
    for e in valid:
        fwd[e["from"]].append(e)

    visited: set[str] = set()
    order: list[dict] = []
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        addr, d = queue.popleft()
        if addr in visited:
            continue
        visited.add(addr)
        p = member_node_path(memory_dir, chain, addr)
        title = addr
        if p and p.exists():
            fm, _ = tq.read_node(p)
            title = tq.fm_get(fm, "name") or addr
        order.append({"addr": addr, "depth": d, "title": title,
                      "node": p.name if p else None})
        if d >= depth:
            continue
        for e in fwd.get(addr, []):
            if e["to"] not in visited:
                queue.append((e["to"], d + 1))
    return order


def snapshot_at(memory_dir: Path, at: _dt.date) -> dict:
    chains_dir = memory_dir / "chains"
    out: dict = {"at": at.isoformat(), "chains": {}}
    for cname in list_chains(chains_dir):
        chain = load_chain(chains_dir, cname)
        valid = [e for e in (chain.get("edges") or []) if edge_valid_at(e, at)]
        out["chains"][cname] = {
            "valid_edges": valid,
            "valid_edge_count": len(valid),
            "total_edge_count": len(chain.get("edges") or []),
        }
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.chain
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Carved from memory_chain_temporal.py (bi-temporal edge layer).
#             + GATE6 (temporal-helper import rerouted to samia.core.temporal)
#             + AUD48 Phase 1 (observation-only memory_guard staging in mutators)
#             + FEAT-2026-06-07 P0 (strip_member forget cascade).
# Layer:      core (library; shared by the CLI and the MCP server).
# Role:       the bi-temporal EDGE layer for SAM chains — chain-manifest I/O + member
#             resolution + add/close/modify edge-interval ops, the point-in-time
#             edge_valid_at predicate, linear-edge backfill, and the time-anchored
#             query/traverse/snapshot surface.
# Stability:  stable -- edge-interval query primitives; API on memory_dir/chains_dir.
# ErrorModel: load_chain raises ChainNotFound (a FileNotFoundError, so `except
#             Exception` catches it — it no longer SystemExit-kills the host);
#             add_edge raises SystemExit on a non-member endpoint (CLI contract);
#             edge predicates treat a falsy / "null" bound as an OPEN interval end;
#             strip_member is idempotent and skips chains not containing the node;
#             memory_guard staging is fail-open (never blocks/breaks a write).
# Depends:    datetime, json, collections, pathlib, typing (stdlib).
#             samia.core.temporal (LAZY via _tq()).
#             samia.runtime.memory_guard.stage_write (LAZY, observation-only).
# Exposes:    load_chain, save_chain, list_chains, member_node_path, member_addrs,
#             node_valid_from, strip_member, add_edge, invalidate_edge, set_edge,
#             edge_valid_at, migrate_linear_edges, show_chain, query_at,
#             traverse_at, snapshot_at, ChainNotFound.
# Lines:      485
# --------------------------------------------------------------------------
