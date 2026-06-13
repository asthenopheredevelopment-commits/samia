"""samia.core.chain — edge-level temporal intervals for SAM chains.

Carved from memory_chain_temporal.py. Per design doc §1.1, the chain module
exposes the bi-temporal *edge* layer: every edge in a chain manifest carries
its own [valid_from, valid_to] interval, and traversals at a point in time
only follow edges whose interval contains that timestamp.

Edge schema (lives in chains/<chain>.json):

    {
      "edges": [
        {
          "from":       "<member-addr>",
          "to":         "<member-addr>",
          "valid_from": "YYYY-MM-DD" | null,
          "valid_to":   "YYYY-MM-DD" | null,   # null = still valid
          "label":      "supersedes" | "depends_on" | "follows" | ...,
          "confidence": 0.0–1.0
        }
      ]
    }

Public API:
  load_chain, save_chain, list_chains      — manifest I/O
  member_node_path, member_addrs           — member resolution
  node_valid_from                          — read node's valid_from
  add_edge, invalidate_edge, set_edge      — edge ops
  edge_valid_at                            — point-in-time predicate
  migrate_linear_edges                     — backfill linear edges
  show_chain, query_at, traverse_at,
    snapshot_at                            — queries

Note: chain.py depends on memory_temporal_query for read_node/parse_date/fm_get.
That helper module will eventually carve into `samia.core.temporal`; for now
the import is preserved.

Acceptance: byte-identical to pre-refactor memory_chain_temporal.py CLI output
on the same memory tree (design doc §8.1).
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


def edge_valid_at(e: dict, at: _dt.date) -> bool:
    tq = _tq()
    vf = tq.parse_date(e.get("valid_from")) if e.get("valid_from") else None
    vt = tq.parse_date(e.get("valid_to")) if e.get("valid_to") not in (None, "null") else None
    if vf and at < vf:
        return False
    if vt and at > vt:
        return False
    return True


# ---------------------------------------------------------------------------
# Migration: linear edges between consecutive members
# ---------------------------------------------------------------------------


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
        if a in reachable or not (chain.get("edges") or []):
            p = member_node_path(memory_dir, chain, a)
            title = a
            if p and p.exists():
                fm, _ = tq.read_node(p)
                title = tq.fm_get(fm, "name") or a
            out.append({"addr": a, "title": title,
                        "node": p.name if p else None})
    return out


def traverse_at(memory_dir: Path, chain_name: str, start: str, at: _dt.date,
                depth: int = 3) -> list[dict]:
    chains_dir = memory_dir / "chains"
    tq = _tq()
    chain = load_chain(chains_dir, chain_name)
    valid = [e for e in (chain.get("edges") or []) if edge_valid_at(e, at)]
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
