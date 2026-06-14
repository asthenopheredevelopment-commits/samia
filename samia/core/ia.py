"""samia.core.ia — IA runtime primitives: compress, freeze, thaw, merge, step_up.

Layer 1 (Owns / Depends):
    Owns:    Pool dataclass (load/save/recompute_density/next_addr);
             segment_body, segment_hash, compress_body, decompress_body — the pool
                 segmentation/compression primitives.
             compress, freeze, thaw, inspect, step_up, pool_stats, merge — the
                 lifecycle primitives the daemon's ia_consolidation job calls.
             forget_node (P0 cross-tier invalidation cascade), restore_node,
                 detect_wrong_deletion (P3a restorable supersession).
    Depends: stdlib only directly (hashlib, json, re, sys, dataclasses, datetime,
             pathlib). samia.core.frontmatter (parse/serialize). LAZY in the
             cascade: web_store, bio, chain, vector (forget_node); vector
             (restore_node); fact_extractor (freeze enqueue, fail-open).
Layer 2 (What / Why):
    What: per design doc §1.1 + §1.3, the ia_consolidation job drives these:
          freeze cold nodes (compress body into the pool, archive/<id>.frozen.json,
          drop from nodes/, cascade forget); thaw them back on demand; compress live
          bodies into the pool non-destructively; merge two same-chain nodes into
          one (binary SAM merge). The pool is content-addressed (segment_hash) so
          identical segments are stored once and ref-counted.
    Why:  freeze/merge remove a node FILE, which would otherwise leave dangling
          'ghost' edges across the graph — so every removal cascades through
          forget_node (edges.db + edge_weights + chains + vector). Supersession
          (contradiction / supersede) full-archives the live node FIRST so
          restore_node can un-forget it byte-exact; detect_wrong_deletion auto-
          restores a wrongly-deleted belief. Q4 OPERATOR OVERRIDE: auto-supersede is
          made safe by reversibility, so it deletes-but-restorably, not surface-only.

Acceptance: byte-identical to pre-refactor memory_ia.py CLI output on the same
    memory tree (design doc §8.1).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import frontmatter as _fm


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _log_event(memory_dir: Path, event: dict) -> None:
    log_path = memory_dir / ".ia_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": _now_iso(), **event}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ---- pool management ---------------------------------------------------------


@dataclass
class Pool:
    memory_dir: Path
    version: int = 1
    compression_density: float = 0.0
    entries: dict = None  # pool_addr -> {content_hash, refs, description, content}

    @classmethod
    def load(cls, memory_dir: Path) -> "Pool":
        pool_path = memory_dir / "pool" / "pool.json"
        if not pool_path.exists():
            return cls(memory_dir=memory_dir, entries={})
        raw = json.loads(pool_path.read_text())
        return cls(
            memory_dir=memory_dir,
            version=raw.get("version", 1),
            compression_density=raw.get("compression_density", 0.0),
            entries=raw.get("entries", {}),
        )

    def save(self) -> None:
        import os
        pool_path = self.memory_dir / "pool" / "pool.json"
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(
                {
                    "version": self.version,
                    "compression_density": round(self.compression_density, 4),
                    "entries": self.entries,
                },
                indent=2,
            )
            + "\n"
        )
        # ATOMIC write (BUG-2026-06-08): a bare write_text is non-atomic + unlocked.
        # Concurrent IA.save() calls (daemon + MCP server + sessions each hold a pool)
        # raced and left one writer's valid JSON followed by another's STALE TAIL -- a
        # truncated-overwrite that broke pool.json parsing and silently killed the
        # relevance auto-freeze for ~2 days. Write a pid-unique temp then os.replace
        # (atomic rename): a concurrent reader sees old-or-new but never a partial file,
        # and racing replaces resolve to last-writer-wins with no torn tail.
        tmp = pool_path.with_name(f"{pool_path.name}.tmp.{os.getpid()}")
        tmp.write_text(data)
        os.replace(tmp, pool_path)

    def recompute_density(self) -> None:
        if not self.entries:
            self.compression_density = 0.0
            return
        total_refs = sum(e.get("refs", 0) for e in self.entries.values())
        if total_refs == 0:
            self.compression_density = 0.0
            return
        unique = len(self.entries)
        self.compression_density = max(0.0, 1.0 - (unique / total_refs))

    def next_addr(self) -> str:
        return f"POOL-{len(self.entries) + 1:04d}"


def segment_body(body: str) -> list[tuple[str, str]]:
    """Split a node body into (segment_id, content) pairs.

    Segment heuristic: paragraphs (double-newline boundaries). Headers and
    bullet blocks are individual segments.
    """
    segments: list[tuple[str, str]] = []
    chunks = [c.strip() for c in re.split(r"\n\s*\n", body) if c.strip()]
    for i, chunk in enumerate(chunks):
        segments.append((f"seg-{i:03d}", chunk))
    return segments


def segment_hash(content: str) -> str:
    normalized = re.sub(r"\s+", " ", content.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def compress_body(body: str, pool: Pool, node_id: str) -> tuple[dict, dict]:
    """Return (pool_refs, compressed_state).

    pool_refs: list of {pool_addr, segment_id} pointers.
    compressed_state: fields needed to reconstruct node body from pool + deltas.
    """
    refs: list[dict] = []
    deltas: dict = {}
    for seg_id, content in segment_body(body):
        h = segment_hash(content)
        found = None
        for pool_addr, entry in pool.entries.items():
            if entry.get("content_hash") == h:
                found = pool_addr
                break
        if found:
            pool.entries[found]["refs"] = pool.entries[found].get("refs", 0) + 1
            pool.entries[found].setdefault("ref_nodes", []).append(node_id)
            refs.append({"pool_addr": found, "segment_id": seg_id})
        else:
            addr = pool.next_addr()
            desc = content[:80].replace("\n", " ").strip()
            pool.entries[addr] = {
                "content_hash": h,
                "content": content,
                "refs": 1,
                "description": desc,
                "ref_nodes": [node_id],
            }
            refs.append({"pool_addr": addr, "segment_id": seg_id})
    pool.recompute_density()
    compressed = {"refs": refs, "deltas": deltas}
    return compressed, {"n_segments": len(refs)}


def decompress_body(compressed: dict, pool: Pool) -> str:
    parts: list[str] = []
    for ref in compressed.get("refs", []):
        addr = ref["pool_addr"]
        entry = pool.entries.get(addr)
        if not entry:
            parts.append(f"<MISSING POOL ENTRY {addr}>")
            continue
        parts.append(entry["content"])
    return "\n\n".join(parts) + "\n"


# ---- primitives --------------------------------------------------------------


def compress(memory_dir: Path, node_name: str) -> None:
    """Update the pool with segments from a live node, non-destructively.

    Unlike freeze(), the node stays in nodes/ and is not archived. The pool
    gains refs for any new segments and ref-count bumps for existing ones.
    """
    nodes = memory_dir / "nodes"
    node_path = nodes / f"{node_name}.md"
    if not node_path.exists():
        sys.exit(f"compress: no node {node_path}")
    text = node_path.read_text()
    parsed, body = _fm.parse(text)
    if parsed is None:
        sys.exit(f"compress: no frontmatter in {node_path}")
    fm, _order = parsed

    pool = Pool.load(memory_dir)
    node_id = str(fm.get("address", node_name))
    _compressed, stats = compress_body(body, pool, node_id)
    pool.save()

    _log_event(
        memory_dir,
        {
            "event": "compress",
            "node": node_name,
            "address": node_id,
            "n_segments": stats["n_segments"],
            "pool_density": pool.compression_density,
        },
    )
    print(
        f"[ia] compressed {node_name} into pool "
        f"(segments={stats['n_segments']}, pool_density={pool.compression_density:.3f})"
    )


def freeze(memory_dir: Path, node_name: str) -> None:
    nodes = memory_dir / "nodes"
    archive = memory_dir / "archive"
    node_path = nodes / f"{node_name}.md"
    if not node_path.exists():
        sys.exit(f"freeze: no node {node_path}")
    text = node_path.read_text()
    parsed, body = _fm.parse(text)
    if parsed is None:
        sys.exit(f"freeze: no frontmatter in {node_path}")
    fm, order = parsed

    tier = str(fm.get("tier", "warm"))
    if tier == "hot":
        sys.exit(f"freeze: refusing to freeze hot node {node_name}. Demote first.")

    pool = Pool.load(memory_dir)
    node_id = str(fm.get("address", node_name))
    compressed, stats = compress_body(body, pool, node_id)
    pool.save()

    archive.mkdir(parents=True, exist_ok=True)
    frozen = {
        "node_id": node_id,
        "original_file": f"nodes/{node_name}.md",
        "original_name": node_name,
        "halt_ts": _now_iso(),
        "resume_ts": None,
        "frontmatter_at_halt": fm,
        "frontmatter_order": order,
        "compressed_state": compressed,
        "genealogy": fm.get("chains", []),
        "stats": stats,
    }
    out = archive / f"{node_id}.frozen.json"
    out.write_text(json.dumps(frozen, indent=2) + "\n")

    # FEAT-2026-06-10-memory-fact-extract-producer-v01 P1 — enqueue a
    # session-offload's body for write-time fact extraction (the episodic bulk
    # leaves the hot path exactly when its facts should be distilled). The body is
    # already in hand, so order vs unlink is safe (we use `body`, not the file).
    # What: when this node is a session_offload (type field, OR — conservatively —
    #   its name contains 'offload', mirroring is_excluded_node's filename check),
    #   append a {text,source,enqueued_by} record to .fact_extract_queue.jsonl.
    # Why: Q1d — freeze is one of the two consolidation-shaped producer feeds. The
    #   CALL is gated on fact_extract_enabled() so flag-off = zero new writes, and
    #   the whole thing is try/except fail-OPEN: freeze must NEVER break on an
    #   enqueue failure (mirrors freeze's other side-effects). ADDITIVE/keep+link:
    #   the source is still archived/forgotten exactly as before — extraction never
    #   destroys evidence.
    _node_type = str(fm.get("type", "")).strip().lower()
    _is_offload = (_node_type == "session_offload"
                   or "offload" in node_name.lower())
    if _is_offload:
        try:
            from . import fact_extractor as _fx
            if _fx.fact_extract_enabled():
                _fx.enqueue_for_extraction(memory_dir, body, node_id, "freeze")
        except Exception:
            pass  # fail-open: freeze must never break on an enqueue failure

    node_path.unlink()
    # FEAT-2026-06-07 P0: cascade the node's death across the graph so it leaves no ghost edge.
    forget_node(memory_dir, node_name, reason="freeze")
    _log_event(
        memory_dir,
        {
            "event": "freeze",
            "node": node_name,
            "address": node_id,
            "n_segments": stats["n_segments"],
            "pool_density": pool.compression_density,
        },
    )
    print(
        f"[ia] froze {node_name} → {out.name} "
        f"(segments={stats['n_segments']}, pool_density={pool.compression_density:.3f})"
    )


def thaw(memory_dir: Path, frozen_id: str) -> None:
    nodes = memory_dir / "nodes"
    archive = memory_dir / "archive"
    fp = archive / f"{frozen_id}.frozen.json"
    if not fp.exists():
        candidates = list(archive.glob(f"*{frozen_id}*.frozen.json"))
        if len(candidates) == 1:
            fp = candidates[0]
        else:
            sys.exit(f"thaw: no unique archive match for {frozen_id}: {candidates}")

    frozen = json.loads(fp.read_text())
    if frozen.get("resume_ts"):
        sys.exit(f"thaw: CONFLICT — already resumed at {frozen['resume_ts']}")

    pool = Pool.load(memory_dir)
    body = decompress_body(frozen["compressed_state"], pool)

    fm = frozen["frontmatter_at_halt"]
    order = frozen["frontmatter_order"]
    fm["tier"] = "warm"
    fm["relevance"] = 0.5
    fm["last_access"] = _now_iso().split("T")[0]

    name = frozen["original_name"]
    nodes.mkdir(parents=True, exist_ok=True)
    (nodes / f"{name}.md").write_text(_fm.serialize(fm, order, body))

    frozen["resume_ts"] = _now_iso()
    fp.write_text(json.dumps(frozen, indent=2) + "\n")

    _log_event(memory_dir, {"event": "thaw", "node": name, "address": frozen["node_id"]})
    print(f"[ia] thawed {frozen['node_id']} → nodes/{name}.md")


def inspect(memory_dir: Path, frozen_id: str) -> None:
    archive = memory_dir / "archive"
    fp = archive / f"{frozen_id}.frozen.json"
    if not fp.exists():
        cands = list(archive.glob(f"*{frozen_id}*.frozen.json"))
        if len(cands) == 1:
            fp = cands[0]
        else:
            sys.exit(f"inspect: no unique archive match for {frozen_id}")
    frozen = json.loads(fp.read_text())
    fm = frozen["frontmatter_at_halt"]
    print(f"Node:      {frozen['original_name']}")
    print(f"Address:   {frozen['node_id']}")
    print(f"Halt ts:   {frozen['halt_ts']}")
    print(f"Resume ts: {frozen.get('resume_ts') or '(still frozen)'}")
    print(f"Genealogy: {frozen.get('genealogy', [])}")
    print(f"Chains:    {fm.get('chains', [])}")
    print(f"Segments:  {frozen['stats']['n_segments']}")
    print(f"Description: {fm.get('description', '')}")


def step_up(memory_dir: Path, address: str) -> None:
    """Traversal primitive. Find chain containing this address, return prior sibling."""
    chains = memory_dir / "chains"
    if not chains.exists():
        sys.exit("step_up: no chains/ directory")
    for chain_file in sorted(chains.glob("*.json")):
        manifest = json.loads(chain_file.read_text())
        members = manifest["members"]
        for i, m in enumerate(members):
            if m["addr"] == address:
                if i == 0:
                    print(f"{address}  (chain head, no step_up)")
                    return
                prior = members[i - 1]
                print(f"{prior['addr']}  {prior['file']}")
                return
    sys.exit(f"step_up: {address} not found in any chain")


def pool_stats(memory_dir: Path) -> None:
    pool = Pool.load(memory_dir)
    total_refs = sum(e.get("refs", 0) for e in pool.entries.values())
    print(f"Pool entries:         {len(pool.entries)}")
    print(f"Total refs:           {total_refs}")
    print(f"Compression density:  {pool.compression_density:.3f}")
    if pool.entries:
        ranked = sorted(
            pool.entries.items(),
            key=lambda kv: kv[1].get("refs", 0),
            reverse=True,
        )[:5]
        print("Top entries by refs:")
        for addr, e in ranked:
            print(f"  {addr}  refs={e['refs']:3d}  {e['description'][:60]}")


def merge(memory_dir: Path, node_a: str, node_b: str) -> None:
    """SAM binary merge — iterative, bounded, verifiable."""
    nodes = memory_dir / "nodes"
    pa = nodes / f"{node_a}.md"
    pb = nodes / f"{node_b}.md"
    for p in (pa, pb):
        if not p.exists():
            sys.exit(f"merge: missing {p}")

    parsed_a, body_a = _fm.parse(pa.read_text())
    parsed_b, body_b = _fm.parse(pb.read_text())
    if parsed_a is None or parsed_b is None:
        sys.exit("merge: frontmatter missing")
    fm_a, order_a = parsed_a
    fm_b, _ = parsed_b

    chains_a = fm_a.get("chains", [])
    chains_b = fm_b.get("chains", [])
    if not (set(chains_a) & set(chains_b)):
        sys.exit(
            f"merge: refusing cross-chain merge. "
            f"{node_a}:{chains_a} vs {node_b}:{chains_b}"
        )

    if fm_a.get("tier") == "hot" or fm_b.get("tier") == "hot":
        sys.exit("merge: refusing hot-node merge")

    merged_name = f"{node_a}__merged_{node_b}"
    merged_fm = {**fm_b, **fm_a}
    merged_fm["name"] = f"[MERGED] {fm_a.get('name', node_a)} + {fm_b.get('name', node_b)}"

    # FEAT-opencode-atoms-integration Phase 1: runtime provenance inheritance.
    # What: inherit runtime from source nodes on merge.
    # Why: if both sources share the same runtime, the merged node keeps it.
    #   If they differ, default to "main" (conservative -- cross-runtime
    #   merges are a Phase 3 concern).
    runtime_a = fm_a.get("runtime", "main")
    runtime_b = fm_b.get("runtime", "main")
    merged_fm["runtime"] = runtime_a if runtime_a == runtime_b else "main"
    merged_fm["description"] = (
        f"MERGED {fm_a.get('description', '')} | {fm_b.get('description', '')}"
    )[:200]
    merged_fm["access_count"] = int(fm_a.get("access_count", 0)) + int(
        fm_b.get("access_count", 0)
    )
    merged_fm["relevance"] = max(
        float(fm_a.get("relevance", 0.5)), float(fm_b.get("relevance", 0.5))
    )
    merged_fm["last_access"] = max(
        str(fm_a.get("last_access", "")), str(fm_b.get("last_access", ""))
    )
    merged_body = (
        f"# Merged node\n\n"
        f"## From {node_a}\n\n{body_a.strip()}\n\n"
        f"## From {node_b}\n\n{body_b.strip()}\n"
    )

    out = nodes / f"{merged_name}.md"
    out.write_text(_fm.serialize(merged_fm, order_a, merged_body))
    pa.unlink()
    pb.unlink()
    # FEAT-2026-06-07 P0: cascade both source deaths so their edges don't dangle as ghosts.
    forget_node(memory_dir, node_a, reason="merge")
    forget_node(memory_dir, node_b, reason="merge")

    _log_event(
        memory_dir,
        {
            "event": "merge",
            "a": node_a,
            "b": node_b,
            "result": merged_name,
            "chain": chains_a[0] if chains_a else None,
        },
    )
    print(f"[ia] merged {node_a} + {node_b} → nodes/{merged_name}.md")


def _append_forgotten_log(memory_dir: Path, node_id: str, reason: str) -> None:
    """Audit trail of every forget_node call -> biomimetic/forgotten_log.jsonl.

    Freeze also keeps its archive/*.frozen.json; this is the uniform "what was purged and
    why" record, and the only provenance artifact for contradiction-purges.
    """
    bio_dir = memory_dir / "biomimetic"
    bio_dir.mkdir(parents=True, exist_ok=True)
    rec = {"id": node_id, "reason": reason, "ts": _now_iso()}
    with (bio_dir / "forgotten_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# _ARCHIVE_REASONS — What: the forget reasons that mean "this node is being replaced
#   by a competing claim" and so must be FULL-archived (restorable), not just logged.
# _ARCHIVE_REASONS — Why: Q4 OPERATOR OVERRIDE — auto-supersede made safe by
#   reversibility via restore_node. Both the online auto-supersede path
#   (reason="contradiction") and the operator-confirmed supersession path
#   (reason="supersede", mcp_server.memory_confirm_supersession) retire a node for being
#   superseded; BOTH must archive the full node so restore_node can un-forget it byte-exact.
#   freeze/merge are NOT here — they archive (freeze) or fold (merge) the node themselves.
_ARCHIVE_REASONS: set[str] = {"contradiction", "supersede"}


def _archive_superseded(memory_dir: Path, node: str, reason: str,
                        superseded_by: Optional[str]) -> Optional[dict]:
    """FEAT-2026-06-07 P3a — full-node archive for a superseded node.

    What: read the still-live node file and write archive/<node_id>.superseded.json with the
          node's full frontmatter, key order, RAW body, and original raw text, then unlink the
          file. Returns the archive record, or None if the file is already gone.
    Why:  the Q4 OVERRIDE makes supersession auto + reversible (online auto-supersede or
          operator-confirmed). freeze stores a *compressed* state (needs the Pool to thaw);
          supersession must restore byte-exact without the Pool, so we store the uncompressed
          body + the original raw text. This is the only supersession provenance richer than
          the {id,reason,ts} forgotten-log line.
    """
    nodes = memory_dir / "nodes"
    archive = memory_dir / "archive"
    name = node[:-3] if node.endswith(".md") else node
    node_path = nodes / f"{name}.md"
    if not node_path.exists():
        return None
    raw = node_path.read_text(encoding="utf-8")
    parsed, body = _fm.parse(raw)
    fm, order = (parsed if parsed is not None else ({}, []))
    node_id = str(fm.get("address", name))
    archive.mkdir(parents=True, exist_ok=True)
    rec = {
        "node_id": node_id,
        "original_file": f"nodes/{name}.md",
        "original_name": name,
        "halt_ts": _now_iso(),
        "restore_ts": None,
        "frontmatter_at_halt": fm,
        "frontmatter_order": order,
        "body": body,
        "original_text": raw,
        "reason": reason,
        "superseded_by": superseded_by,
    }
    # Keyed by file stem (the node id callers pass) so restore_node finds it directly.
    (archive / f"{name}.superseded.json").write_text(
        json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    node_path.unlink()
    return rec


def forget_node(memory_dir: Path, node: str, reason: str = "manual",
                db_dir: Optional[str] = None,
                superseded_by: Optional[str] = None) -> dict:
    """FEAT-2026-06-07 P0 — cross-tier invalidation cascade.

    Retire a dead/wrong node from EVERY live-graph store so node death leaves no dangling
    'ghost' edge (the audited corruption): hard-delete its edges in edges.db (all ref_kinds)
    + edge_weights.json, strip its chain membership + edges, and tombstone its vector entry
    (excluded at query, hard-dropped on next rebuild). Append a forgotten-log entry. The node
    FILE is expected already gone (freeze archives it; merge folds it). The EXCEPTION is the
    supersession reasons in _ARCHIVE_REASONS ({"contradiction","supersede"}, P3a/P3b): the
    file is still live, so we full-archive it (archive/<id>.superseded.json) BEFORE the
    cascade so restore_node can un-forget it byte-exact. Q4 OPERATOR OVERRIDE — auto-supersede
    is made safe by reversibility via restore_node, so BOTH the online auto-supersede path
    (reason="contradiction") and the operator-confirmed path (reason="supersede") archive.
    Idempotent and fail-soft per store (one store erroring never blocks the others). Returns a
    per-store count dict.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    stats: dict = {"node": fname, "reason": reason}
    # FEAT-2026-06-07 P3a/P3b: a "replaced by a competing claim" reason = reversible
    # auto-supersede → full archive first so restore_node can un-forget byte-exact.
    if reason in _ARCHIVE_REASONS:
        arc = _archive_superseded(memory_dir, node, reason, superseded_by)
        stats["superseded_archive"] = (
            f"archive/{fname[:-3]}.superseded.json" if arc else None)
    try:
        from . import web_store as _ws
        stats["edges_db"] = _ws.forget_node_edges(fname, db_dir)
    except Exception as e:
        stats["edges_db"] = {"error": str(e)}
    try:
        from . import bio as _bio
        stats["edge_weights"] = _bio.forget_node_weights(memory_dir, fname)
    except Exception as e:
        stats["edge_weights"] = {"error": str(e)}
    try:
        from . import chain as _chain
        stats["chains"] = _chain.strip_member(memory_dir / "chains", fname)
    except Exception as e:
        stats["chains"] = {"error": str(e)}
    try:
        from . import vector as _vec
        stats["vector"] = _vec.tombstone_node(memory_dir, fname)
    except Exception as e:
        stats["vector"] = {"error": str(e)}
    _append_forgotten_log(memory_dir, fname, reason)
    print(f"[ia] NOTE: chain manifest needs regeneration — run memory_migrate.py")
    return stats


def restore_node(memory_dir: Path, node_id: str) -> dict:
    """FEAT-2026-06-07 P3a — un-forget a superseded node from its archive.

    Symmetric to thaw, but for a supersession-purge (online auto or operator-confirmed):
    locate archive/<id>.superseded.json,
    re-create nodes/<name>.md byte-exact from the stored raw text, un-tombstone its vector
    entry (so recall includes it again), stamp restore_ts on the archive record, log a
    'restore' IA event, and append a {id, reason:"restore", ts} forgotten-log line. Edges are
    NOT replayed — they re-accrue naturally from subsequent co-activation on the clean web
    (consistent with the P2 salvage philosophy). Returns a status dict.
    """
    archive = memory_dir / "archive"
    nodes = memory_dir / "nodes"
    fp = archive / f"{node_id}.superseded.json"
    if not fp.exists():
        # Tolerate a passed .md suffix or an address-keyed lookup.
        stem = node_id[:-3] if node_id.endswith(".md") else node_id
        fp = archive / f"{stem}.superseded.json"
        if not fp.exists():
            cands = list(archive.glob(f"*{stem}*.superseded.json"))
            if len(cands) == 1:
                fp = cands[0]
            else:
                return {"restored": False, "node": node_id,
                        "error": f"no unique superseded archive for {node_id}"}

    rec = json.loads(fp.read_text(encoding="utf-8"))
    if rec.get("restore_ts"):
        return {"restored": False, "node": node_id,
                "skipped": f"already restored at {rec['restore_ts']}"}

    name = rec["original_name"]
    nodes.mkdir(parents=True, exist_ok=True)
    # Prefer the byte-exact original text; fall back to re-serializing fm+order+body.
    # Why: original_text guarantees the restored file is identical; the re-serialize path
    #   is a safety net for any future archive written without the raw text.
    if rec.get("original_text") is not None:
        (nodes / f"{name}.md").write_text(rec["original_text"], encoding="utf-8")
    else:
        (nodes / f"{name}.md").write_text(
            _fm.serialize(rec["frontmatter_at_halt"], rec["frontmatter_order"], rec["body"]),
            encoding="utf-8")

    # Un-tombstone the vector entry so recall re-admits the restored node.
    vec_stat: dict = {}
    try:
        from . import vector as _vec
        vec_stat = _vec.untombstone_node(memory_dir, f"{name}.md")
    except Exception as e:
        vec_stat = {"error": str(e)}

    rec["restore_ts"] = _now_iso()
    fp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

    _log_event(memory_dir, {"event": "restore", "node": name,
                            "address": rec.get("node_id")})
    _append_forgotten_log(memory_dir, f"{name}.md", "restore")
    return {"restored": True, "node": f"{name}.md",
            "file": f"nodes/{name}.md", "vector": vec_stat}


def detect_wrong_deletion(memory_dir: Path, new_node_id: str) -> dict:
    """FEAT-2026-06-07 P3a — self-healing watch: auto-restore a wrongly-deleted node.

    What: on a fresh write, if its subject (frontmatter name) matches a superseded node whose
          superseder (superseded_by) is now itself absent — i.e. the claim that justified the
          deletion is gone or the original claim is being re-asserted — call restore_node and
          mark it auto-restored. Returns {restored: [...], checked: int}.
    Why:  Q4 — auto-deletion is acceptable *only because it is reversible*. This closes the
          loop: a deletion that turns out wrong (the belief is re-asserted, or its superseder
          vanished) is undone automatically rather than waiting for an operator.
    """
    archive = memory_dir / "archive"
    nodes = memory_dir / "nodes"
    restored: list[str] = []
    checked = 0
    if not archive.exists():
        return {"restored": restored, "checked": checked}

    new_name = new_node_id[:-3] if new_node_id.endswith(".md") else new_node_id
    new_path = nodes / f"{new_name}.md"
    new_subject = ""
    if new_path.exists():
        parsed, _ = _fm.parse(new_path.read_text(encoding="utf-8"))
        if parsed is not None:
            new_subject = str(parsed[0].get("name", "")).strip().lower()

    for fp in sorted(archive.glob("*.superseded.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        if rec.get("restore_ts"):
            continue  # already restored
        checked += 1
        arc_subject = str(
            rec.get("frontmatter_at_halt", {}).get("name", "")).strip().lower()
        superseded_by = rec.get("superseded_by")
        sup_name = (superseded_by[:-3] if isinstance(superseded_by, str)
                    and superseded_by.endswith(".md") else superseded_by)
        # Wrong deletion if: (a) the original claim is re-asserted by the new write
        # (same subject), or (b) the superseder that justified the death no longer exists.
        reasserted = bool(new_subject) and new_subject == arc_subject
        superseder_gone = (
            sup_name is not None and not (nodes / f"{sup_name}.md").exists())
        if reasserted or superseder_gone:
            res = restore_node(memory_dir, rec["original_name"])
            if res.get("restored"):
                restored.append(rec["original_name"])
    return {"restored": restored, "checked": checked}


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.ia
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      §1.1/§1.3 IA primitives (compress/freeze/thaw/merge/step_up)
#             + FEAT-2026-06-07 P0 (forget_node cascade) + P3a/P3b (restorable
#               supersede/restore + detect_wrong_deletion)
#             + FEAT-2026-06-10-memory-fact-extract-producer-v01 (P1: freeze
#               enqueues a session_offload body for fact extraction — gated on
#               ASTHENOS_FACT_EXTRACT_ENABLED, fail-OPEN, ADDITIVE keep+link; the
#               source is still archived/forgotten exactly as before)
# Layer:      core (pure library, no daemon dependency)
# Role:       IA runtime primitives (pool compress, freeze/thaw, merge, forget/
#             restore cascade) — the daemon's ia_consolidation job calls these
# Stability:  stable — byte-identical to pre-refactor memory_ia.py CLI output
#             (design doc §8.1); freeze's fact-extract enqueue is the only new
#             write and is gated + fail-open
# ErrorModel: CLI primitives sys.exit on a missing node / bad frontmatter; the
#             forget_node cascade is idempotent and fail-soft PER STORE (one store
#             erroring never blocks the others, errors captured into the stat dict);
#             Pool.save is atomic (temp + os.replace); freeze's enqueue is fail-OPEN
# Depends:    hashlib, json, re, sys, datetime, pathlib (stdlib);
#             samia.core.frontmatter; lazy: web_store, bio, chain, vector,
#             fact_extractor (freeze enqueue, fail-open)
# Exposes:    Pool, compress, freeze, thaw, inspect, step_up, pool_stats, merge,
#             forget_node, restore_node, detect_wrong_deletion, segment_body,
#             segment_hash, compress_body, decompress_body
# Lines:      733
# --------------------------------------------------------------------------
