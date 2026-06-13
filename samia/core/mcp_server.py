"""samia.core.mcp_server — data-handling primitives for the MCP server.

Carved from memory_mcp_server.py. Each function corresponds to one MCP
tool's underlying logic, parameterized on memory_dir. The MCP wrapper
itself (with FastMCP decorators and stdio main loop) stays in
memory_mcp_server.py — these functions provide the work behind each
@mcp.tool().

Public API: each tool's logic exposed as a plain function taking
memory_dir as the first argument.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from . import web_store as _ws


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _chains_dir(memory_dir: Path) -> Path:
    return memory_dir / "chains"


# ---------------------------------------------------------------------------
# Search & retrieval
# ---------------------------------------------------------------------------


def _filter_by_runtime(hits: list[dict[str, Any]],
                       runtime_filter: str | None) -> list[dict[str, Any]]:
    """Post-retrieval runtime provenance filter.

    What: filters a list of node-hit dicts by the `runtime` frontmatter field.
    Why: allows Atoms panels to show only nodes from a specific runtime
      (opencode or main). Missing runtime field defaults to "main" per the
      backward-compatibility rule from Phase 1 design doc section 3.1.
    """
    if not runtime_filter or runtime_filter == "all":
        return hits
    filtered = []
    for h in hits:
        # What: check runtime from the hit's frontmatter if available.
        # Why: vector index hits may include frontmatter fields; node reads
        #   may not. Default missing to "main" per backward compat rule.
        node_runtime = "main"
        if isinstance(h, dict):
            fm = h.get("frontmatter", {})
            if isinstance(fm, dict):
                node_runtime = fm.get("runtime", "main")
            elif h.get("runtime"):
                node_runtime = h["runtime"]
        if node_runtime == runtime_filter:
            filtered.append(h)
    return filtered


def _rerank_hits(hits: list[dict[str, Any]], query: str,
                 nodes_dir: Path) -> list[dict[str, Any]]:
    """Post-retrieval re-ranking per FEAT-2026-05-18-memory-bridge-and-search D5.

    Three layers applied in order:
    1. term: exact-match boost — if query is a single acronym-shaped token and
       a node's frontmatter `term:` list contains it, boost score by +0.3.
    2. type: weighting — reference/feedback nodes get 1.2x; project 1.0x;
       bug/session_offload 0.8x. Reflects trust hierarchy.
    3. source: docs bonus — nodes with `source: docs` get +0.05 (canonical
       sources slightly preferred over session offloads at equal similarity).

    Fail-open: any parsing error on a node's frontmatter is silently skipped
    (the node keeps its original score).
    """
    import re
    acronym_pat = re.compile(r"^[A-Z][A-Z0-9]{2,}$")
    query_tokens = query.strip().split()
    is_acronym_query = len(query_tokens) == 1 and acronym_pat.match(query_tokens[0])
    acronym_upper = query_tokens[0].upper() if is_acronym_query else ""

    type_weights = {
        "reference": 1.2, "feedback": 1.2,
        "project": 1.0, "user": 1.0,
        "bug": 0.8, "session_offload": 0.8,
    }

    for h in hits:
        node_name = h.get("node", "")
        node_path = nodes_dir / node_name
        if not node_path.suffix:
            node_path = node_path.with_suffix(".md")
        if not node_path.exists():
            continue
        try:
            raw = node_path.read_text(encoding="utf-8", errors="ignore")
            if not raw.startswith("---"):
                continue
            fm_end = raw.index("---", 3)
            fm_block = raw[3:fm_end]
        except (ValueError, OSError):
            continue

        fm: dict[str, str] = {}
        for line in fm_block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip()

        score = h.get("score", 0.0)

        node_type = fm.get("type", "").strip()
        w = type_weights.get(node_type, 1.0)
        score *= w

        if fm.get("source", "").strip() == "docs":
            score += 0.05

        if is_acronym_query:
            term_raw = fm.get("term", "")
            if term_raw.startswith("[") and term_raw.endswith("]"):
                terms = [t.strip().strip("'\"") for t in term_raw[1:-1].split(",")]
            else:
                terms = [term_raw] if term_raw else []
            if any(t.upper() == acronym_upper for t in terms):
                score += 0.3

        h["score"] = score

    hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return hits


def _term_index_lookup(query: str, nodes_dir: Path) -> list[dict[str, Any]]:
    """Direct frontmatter term: scan for single-acronym queries.

    Bypasses vector similarity entirely — scans node files for exact term:
    match. Returns synthetic hit dicts with score=1.0 (always top-ranked).
    Only fires for single-token uppercase queries (acronym-shaped).
    Capped at 5 matches to bound I/O.
    """
    import re
    if not re.match(r"^[A-Z][A-Z0-9]{2,}$", query.strip()):
        return []
    target = query.strip().upper()
    matches: list[dict[str, Any]] = []
    if not nodes_dir.is_dir():
        return []
    for p in nodes_dir.iterdir():
        if not p.suffix == ".md" or not p.is_file():
            continue
        try:
            head = p.read_text(encoding="utf-8", errors="ignore")[:2000]
        except OSError:
            continue
        if not head.startswith("---"):
            continue
        try:
            fm_end = head.index("---", 3)
            fm = head[3:fm_end]
        except ValueError:
            continue
        for line in fm.splitlines():
            if line.strip().startswith("term:"):
                val = line.split(":", 1)[1].strip()
                if val.startswith("[") and val.endswith("]"):
                    terms = [t.strip().strip("'\"").upper() for t in val[1:-1].split(",")]
                else:
                    terms = [val.upper()]
                if target in terms:
                    name_line = next((l for l in fm.splitlines() if l.startswith("name:")), "")
                    title = name_line.split(":", 1)[1].strip() if ":" in name_line else p.stem
                    name_has_term = target.lower() in p.name.lower()
                    score = 1.0 if name_has_term else 0.85
                    matches.append({"score": score, "node": p.name, "title": title})
                    break
    matches.sort(key=lambda h: h["score"], reverse=True)
    return matches[:5]


# Co-activation read-back (FEAT-2026-06-05 Tier-0 D4) — conservative neighbor boost.
# What: surface edges.db co-activation neighbors of the top hits so learned associations
#   bias recall (the "intuition" effect), scored so a neighbor NEVER outranks the hit that
#   surfaced it and nothing real is displaced.
# Why: the Hebbian web was written but never read on the normal recall path (only a
#   default-off, failure-scoped seam). This opens it generally but conservatively.
COACT_LAMBDA = 0.5        # neighbor pull = lambda * edge_weight (then clamped below parent)
COACT_MAX_NEIGHBORS = 3   # cap neighbors appended per search
COACT_PARENT_HITS = 5     # only expand neighbors of the top-N cosine/term hits
COACT_DELTA = 0.05        # parent-score haircut so a neighbor can't tie its parent


def _coactivation_neighbors(parent_hits: list[dict[str, Any]],
                            existing_nodes: set[str],
                            db_dir: str | None = None,
                            lam: float = COACT_LAMBDA,
                            max_neighbors: int = COACT_MAX_NEIGHBORS,
                            parent_n: int = COACT_PARENT_HITS,
                            delta: float = COACT_DELTA) -> list[dict[str, Any]]:
    """Read-only co-activation neighbor expansion for recall (D4).

    Generalizes context_extension._query_failure_associations' edges.db read WITHOUT the
    failure/diagnosis gate. For each of the top `parent_n` hits, pull its co-activation
    neighbors and score each `min(parent_score*(1-delta), lam*edge_weight)` — the clamp
    guarantees a neighbor never outranks the hit that surfaced it (nudge, not hijack).
    Fail-open: any error returns []. Returns up to `max_neighbors` neighbor hit dicts
    tagged via='coactivation'.
    """
    db_path = _ws._db_path(db_dir)
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return []
    best: dict[str, float] = {}
    try:
        for h in parent_hits[:parent_n]:
            pname = h.get("node")
            if not pname:
                continue
            ceiling = float(h.get("score", 0.0)) * (1.0 - delta)
            if ceiling <= 0:
                continue
            try:
                rows = conn.execute(
                    "SELECT dst_node, weight FROM edges WHERE ref_kind=? AND src_node=?",
                    (_ws.COACTIVATION, pname)).fetchall()
                rows += conn.execute(
                    "SELECT src_node, weight FROM edges WHERE ref_kind=? AND dst_node=?",
                    (_ws.COACTIVATION, pname)).fetchall()
            except sqlite3.Error:
                continue
            for nb, w in rows:
                if nb in existing_nodes:
                    continue
                cand = min(ceiling, lam * float(w))
                if cand > 0 and cand > best.get(nb, 0.0):
                    best[nb] = cand
    finally:
        conn.close()
    out = [{"node": nb, "score": s, "via": "coactivation"} for nb, s in best.items()]
    out.sort(key=lambda h: h["score"], reverse=True)
    return out[:max_neighbors]


def _engram_rag_hits(memory_dir: Path, query: str, top_k: int) -> list[dict[str, Any]]:
    """FEAT-2026-06-07 Tier-1 P1 — the engram-RAG fast-tier arm of recall.

    What: query the Tier-1 engram held-copy store (hippocampus.engram_rag_query) and
      return hits shaped like the main recall hits — {score, node, title, via:"engram",
      engram_id} — so they merge cleanly into the existing pipeline. The held copy is a
      self-contained COPY, so `node` (its source_ptr) may no longer exist in main; the
      hit is still valid because the copy carries its own content.
    Why:  P1 EXTENDS the recall path with a fast tier (D1) — engram copies are searched
      ahead of / alongside main, recency-preferential so a recent copy wins over an
      equal-cosine main node. Reuses the same cosine primitive on a smaller matrix; it
      never fans out over the full corpus. Fail-open: any error yields no engram hits and
      the main path is unaffected.
    """
    try:
        from . import hippocampus as _hip
        out: list[dict[str, Any]] = []
        for h in _hip.engram_rag_query(memory_dir, query, top_k=top_k):
            out.append({
                "score": h["score"],
                "node": h.get("source_ptr"),
                "title": h.get("title"),
                "via": "engram",
                "engram_id": h.get("engram_id"),
            })
        return out
    except Exception:
        return []


def _ring_rag_hits(memory_dir: Path, query: str, top_k: int) -> list[dict[str, Any]]:
    """FEAT-2026-06-07 Tier-1 P2 — the ring-RAG fast-tier arm of recall.

    What: query the Tier-1 ring POINTER store (hippocampus.ring_rag_query) and return
      hits shaped like the main recall hits — {score, node, title, via:"ring",
      target_tier, salience_flag} — so they merge cleanly into the existing pipeline.
      A ring entry is a POINTER, deref'd at query time, so `node` (its backing ptr) is
      the CURRENT backing; a dangling pointer is already dropped by ring_rag_query.
    Why:  P2 EXTENDS the recall path with the volatile working-set tier (D1) — the ring
      is searched ahead of / alongside main, the "stutter-continue" half-recall. Reuses
      the same cosine primitive on the small dereferenced pointer set; it never fans out
      over the full corpus. Fail-open: any error yields no ring hits and the main path
      is unaffected.
    """
    try:
        from . import hippocampus as _hip
        out: list[dict[str, Any]] = []
        for h in _hip.ring_rag_query(memory_dir, query, top_k=top_k):
            out.append({
                "score": h["score"],
                "node": h.get("ptr"),
                "title": h.get("title"),
                "via": "ring",
                "target_tier": h.get("target_tier"),
                "salience_flag": h.get("salience_flag", False),
            })
        return out
    except Exception:
        return []


def memory_inject_block(memory_dir: Path, query: str,
                        token_budget: int | None = None,
                        engram_budget_frac: float | None = None) -> dict[str, Any]:
    """FEAT-2026-06-07 Tier-1 P4 — assemble the standing-availability inject block.

    What: build the two-layer inject block (engram-inject = the always-on inject_eligible
      identity set + ring-inject = the live ring working set) relevance-gated against
      `query` within a fixed token budget, priority-arbitrated (engram-inject favored).
      Returns the assembled block (items + layer + token accounting), ready to be
      PREPENDED to a prompt by an operator-gated caller. Fail-open -> empty block.
    Why:  P4/D4 — exposes hippocampus.assemble_inject_block as an MCP surface. This is the
      ASSEMBLER ONLY; the actual per-turn injection into the live prompt stays operator-
      gated/INERT (this tool returns the block; it does NOT mutate any prompt). The block
      is CO-ACTIVATION-SILENT (D5/Q6a): serving it records ZERO genuine co-activations and
      feeds nothing to the Tier-0 web. It is a standing pointer-deref, NOT a search (RAG).
    """
    from . import hippocampus as _hip
    kwargs: dict[str, Any] = {}
    if token_budget is not None:
        kwargs["token_budget"] = int(token_budget)
    if engram_budget_frac is not None:
        kwargs["engram_budget_frac"] = float(engram_budget_frac)
    try:
        return _hip.assemble_inject_block(memory_dir, query, **kwargs)
    except Exception as exc:
        # Fail-open: an assembler error yields an empty, budget-clean block — never a
        # crash on the recall path, and never a silent Tier-0 mutation.
        return {"items": [], "tokens_used": 0,
                "token_budget": int(token_budget or _hip.INJECT_BUDGET_DEFAULT),
                "engram_count": 0, "ring_count": 0, "dropped": 0,
                "co_activation_silent": True, "error": str(exc)}


def memory_search(memory_dir: Path, query: str, top_k: int = 8,
                  record_coactivation: bool = True,
                  include_coactivation_neighbors: bool = True,
                  include_engram: bool = True,
                  include_ring: bool = True,
                  include_inject: bool = False,
                  promote: bool = False,
                  repair_integrity: bool | None = None,
                  runtime: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    from . import vector as _vi
    from . import bio as _bio
    nodes_dir = _nodes_dir(memory_dir)
    term_hits = _term_index_lookup(query, nodes_dir)
    term_node_names = {h["node"] for h in term_hits}
    fetch_k = max(top_k * 2, 20)
    hits = _vi.query(memory_dir, query, top_k=fetch_k)
    hits = [h for h in hits if h.get("node") not in term_node_names]
    hits = _rerank_hits(hits, query, nodes_dir)
    # FEAT-2026-06-07 Tier-1 P1: fold in the engram-RAG fast tier. Engram held copies
    # compete by recency-boosted cosine; an engram hit that mirrors a main hit (same
    # source node) supersedes the main entry (the fast copy is the preferential trace),
    # while distinct engram copies are added to the candidate pool before the top_k cut.
    engram_hits = _engram_rag_hits(memory_dir, query, fetch_k) if include_engram else []
    if engram_hits:
        engram_by_node = {h["node"]: h for h in engram_hits if h.get("node")}
        hits = [h for h in hits if h.get("node") not in engram_by_node]
        hits = sorted(hits + engram_hits, key=lambda h: h.get("score", 0.0),
                      reverse=True)
    # FEAT-2026-06-07 Tier-1 P2: fold in the ring-RAG fast tier (the volatile working
    # set). A ring hit mirroring an existing hit (same backing node) supersedes the
    # existing entry (the ring is the hotter working-set trace); distinct ring hits join
    # the candidate pool before the top_k cut. Additive + fail-open like the engram arm.
    ring_hits = _ring_rag_hits(memory_dir, query, fetch_k) if include_ring else []
    if ring_hits:
        ring_by_node = {h["node"]: h for h in ring_hits if h.get("node")}
        hits = [h for h in hits if h.get("node") not in ring_by_node]
        hits = sorted(hits + ring_hits, key=lambda h: h.get("score", 0.0),
                      reverse=True)
    combined = term_hits + hits
    combined = combined[:top_k]
    # Record GENUINE co-activation on the real retrieved set BEFORE adding neighbors,
    # so the Hebbian feed is never polluted by read-back-derived entries.
    # FEAT-2026-06-07 Tier-1 P5 / Q6a (ALL-RAG-FEEDS): `combined` here ALREADY
    # includes the engram-RAG and ring-RAG fast-tier hits (folded in above before
    # this seam), so this single genuine record feeds main-RAG + engram-RAG +
    # ring-RAG co-activations into Tier-0 uniformly — every effortful RAG retrieval
    # drives the cortical web. INJECT stays co-activation-SILENT (it never reaches
    # this list — it is a separate assemble_inject_block path that never records).
    if record_coactivation and len(combined) >= 2:
        try:
            _bio.hebbian_record(memory_dir, [h["node"] for h in combined], query=query)
        except Exception:
            pass
    # FEAT-2026-06-07 granular-recall-repaired-decay P1 — the RECALL-REPAIR trigger
    # (the SECOND, content-fidelity decay axis; DISTINCT from relevance/tier decay).
    # A genuine recall surfacing a node reconsolidates it: integrity.recall_repair
    # restores the served body BYTE-EXACT from the pristine anchor (Q4a anchor-first,
    # Q3a recall = strongest/full) and resets integrity toward 1.0 — "a node missing a
    # bit is easily read + restored just from recalling it". It runs only on the GENUINE
    # retrieved set (`combined`, the same set that fed the Hebbian record), BEFORE the
    # read-back co-activation neighbors are appended (those are not genuine recalls).
    # INERT by default so the live recall path is unchanged until operator-gated activation.
    # ACTIVATION WIRING (FEAT-2026-06-07 granular-recall-repaired-decay): the param now
    # defaults None — when the caller does NOT pass it (None), it resolves to
    # integrity.repair_enabled() (the live ASTHENOS_INTEGRITY_REPAIR_ENABLED flag, the SAME
    # flag the P2 consolidation-repair subscriber gates on). An explicit True/False from the
    # caller (e.g. tests) still OVERRIDES the flag. So with the flag unset + no explicit
    # arg, this stays exactly the prior inert no-op (byte-identical). Repair is its own
    # concern (its own function), fail-soft per node — a repair error never breaks recall.
    # The anchor is the faithful source; generative fallback (no anchor) is P3, not here.
    _do_repair = repair_integrity
    if _do_repair is None:
        try:
            from . import integrity as _integrity
            _do_repair = _integrity.repair_enabled()
        except Exception:
            _do_repair = False
    if _do_repair:
        try:
            from . import integrity as _integrity
            for h in combined:
                node = h.get("node")
                if node:
                    try:
                        _integrity.recall_repair(memory_dir, node)
                    except Exception:
                        pass
        except Exception:
            pass
    # FEAT-2026-06-07 Tier-1 P3 — the AUTO promotion trigger (a function, no scheduler).
    # When `promote` is enabled (INERT by default — produce-only), each ring-via hit that
    # survived into the genuine result set counts as a GENUINE ring-RAG hit
    # (RingStore.record_genuine_hit) — the ring->engram frequency signal — and one
    # promotion pass runs (hippocampus.promote_ring_step): ring pointers that cross
    # max(genuine-hits, salience) materialize to kWTA-coded engram copies, and the
    # engram->inject set is refreshed on max(attractor, salience). P3 computes ELIGIBILITY
    # only; it never injects (P4) and never touches decay (P5). Fully fail-open.
    if promote:
        try:
            from . import hippocampus as _hip
            for h in combined:
                if h.get("via") == "ring" and h.get("node"):
                    _hip.RingStore(memory_dir).record_genuine_hit(h["node"])
            _hip.promote_ring_step(memory_dir)
        except Exception:
            pass
    # D4: append co-activation neighbors (clamped below their parent; nothing displaced).
    if include_coactivation_neighbors:
        try:
            existing = {h.get("node") for h in combined}
            neighbors = _coactivation_neighbors(combined, existing)
            if neighbors:
                combined = sorted(combined + neighbors,
                                  key=lambda h: h.get("score", 0.0), reverse=True)
        except Exception:
            pass
    combined = _filter_by_runtime(combined, runtime)
    # FEAT-2026-06-07 Tier-1 P4 — optional standing-availability inject block. OFF by
    # default (include_inject=False) so the legacy list return shape is unchanged. When
    # enabled the call returns {hits, inject_block} where inject_block is the two-layer
    # standing context (engram-inject + ring-inject) assembled CO-ACTIVATION-SILENTLY
    # (no Tier-0 feed). This is the assembler surface only; the live per-turn prompt
    # injection stays operator-gated/INERT. Fail-open: an assembler error -> hits alone.
    if include_inject:
        try:
            block = memory_inject_block(memory_dir, query)
        except Exception:
            block = {"items": [], "tokens_used": 0, "co_activation_silent": True}
        return {"hits": combined, "inject_block": block}
    return combined


def memory_temporal_query(
    memory_dir: Path,
    at: str | None = None,
    since: str | None = None,
    range_from: str | None = None,
    range_to: str | None = None,
    semantic: str | None = None,
    top_k: int = 20,
    runtime: str | None = None,
) -> list[dict[str, Any]]:
    from . import temporal as _tq
    at_d = _tq.parse_date(at) if at else None
    since_d = _tq.parse_date(since) if since else None
    rng = None
    if range_from and range_to:
        rng = (_tq.parse_date(range_from), _tq.parse_date(range_to))
        if rng[0] is None or rng[1] is None:
            return [{"error": "range_from / range_to must be ISO dates"}]
    results = _tq.query(memory_dir, at_d, since_d, rng, semantic, top_k)
    return _filter_by_runtime(results, runtime)


def memory_read_node(memory_dir: Path, name: str) -> dict[str, Any]:
    from . import temporal as _tq
    p = _nodes_dir(memory_dir) / name
    if not p.suffix:
        p = p.with_suffix(".md")
    if not p.exists():
        return {"error": f"node not found: {p.name}"}
    raw = p.read_text(encoding="utf-8")
    fm_lines, body = _tq.split_frontmatter(raw)
    fm = {}
    for line in fm_lines:
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return {"node": p.name, "frontmatter": fm, "body": body}


def _register_ring_and_salience(memory_dir: Path, node: str, content: str,
                                salience_tag: bool) -> dict[str, Any]:
    """FEAT-2026-06-07 Tier-1 P2 — capture hook: ring POINTER + salience field.

    What: on a fresh write, (1) register a ring POINTER into the just-written main node
      (RingStore.add — a cheap reference + a salience flag, NOT a copy), and (2) compute
      and persist the node's [0,1] `salience` frontmatter via bio.compute_salience
      (surprise + contradiction-involvement + repetition; an explicit salience_tag
      clamps it high). Returns a small {ring, salience} summary.
    Why:  D6 / P2 capture path (Q1a) — captures land at ring-RAG as POINTERS carrying a
      salience signal; the held engram copy is EARNED later at materialization (P3, not
      here). Fail-open: any error never blocks or corrupts the write.
    """
    out: dict[str, Any] = {}
    try:
        from . import hippocampus as _hip
        ring = _hip.RingStore(memory_dir).add(node, target_tier="main",
                                              salience_flag=salience_tag)
        out["ring"] = {"ptr": ring.get("ptr"),
                       "salience_flag": ring.get("salience_flag")}
    except Exception as e:  # fail-open: ring registration must never break the write.
        out["ring_error"] = str(e)
    try:
        from . import bio as _bio
        out["salience"] = _bio.compute_salience(
            memory_dir, node, content=content,
            explicit_tag=True if salience_tag else None, write=True)
    except Exception as e:  # fail-open: salience write must never break the write.
        out["salience_error"] = str(e)
    return out


def memory_write_node(
    memory_dir: Path,
    name: str,
    title: str,
    description: str,
    body: str,
    type_: str = "project",
    chains: list[str] | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    extract: bool = False,
    extractor_backend: str = "auto",
    runtime: str = "main",
    salience_tag: bool = False,
) -> dict[str, Any]:
    from . import fact_extractor as _fx
    today = _dt.date.today().isoformat()
    nodes_dir = _nodes_dir(memory_dir)

    if extract:
        atoms = _fx.extract_atoms(body, backend=extractor_backend, chains_hint=chains)
        if not atoms:
            return {"error": "extractor produced no atoms", "extracted": 0}
        # Force user-supplied valid_from/valid_to / chains onto atoms when given.
        for a in atoms:
            if valid_from and not a.get("valid_from"):
                a["valid_from"] = valid_from
            if valid_to is not None:
                a["valid_to"] = valid_to
            if chains and not a.get("chains"):
                a["chains"] = list(chains)
        names = _fx.write_atoms_as_nodes(memory_dir, atoms, prefix=name, runtime=runtime)
        return {"extracted": len(names), "nodes": names, "backend": extractor_backend}

    p = nodes_dir / name
    if not p.suffix:
        p = p.with_suffix(".md")
    chains_str = "[" + ", ".join(chains or []) + "]"
    # FEAT-2026-06-11 temporal-recall P0 — write-time substrate (§3).
    # What: mint written_at (Unix float anchor, time.time() at body commit) + one
    #   corpus-global monotone episode_seq, and append them AFTER last_access.
    # Why: the temporal-recall modulators (SITH/distinctiveness need a sub-day anchor;
    #   directed-SR needs a strict total order) read these. ADDITIVE-OPTIONAL: every
    #   existing field is untouched and nothing reads the new fields yet, so retrieval
    #   is unchanged until a later phase + flag enable it. Fail-soft: a substrate hiccup
    #   must never break the write — fall back to omitting the two lines.
    from . import temporal_substrate as _ts
    try:
        _sub = _ts.write_time_fields(memory_dir)
    except Exception:
        _sub = None
    fm = [
        f"name: {title}",
        f"description: {description}",
        f"type: {type_}",
        f"chains: {chains_str}",
        f"valid_from: {valid_from or today}",
        f"valid_to: {valid_to or 'null'}",
        f"last_access: {today}",
        "access_count: 0",
        "relevance: 0.5",
        "tier: warm",
        f"runtime: {runtime}",
    ]
    if _sub is not None:
        fm.append(f"written_at: {_sub['written_at']!r}")
        fm.append(f"episode_seq: {_sub['episode_seq']}")
    fm_text = "\n".join(fm)
    p.write_text(f"---\n{fm_text}\n---\n{body}\n", encoding="utf-8")
    out: dict[str, Any] = {"written": p.name, "valid_from": valid_from or today,
                           "valid_to": valid_to}
    # FEAT-2026-06-07 granular-recall-repaired-decay P2 — ANCHOR CAPTURE ON WRITE.
    # What: capture/refresh the PRISTINE recovery anchor from the just-written body (the
    #   genuine, pre-erosion content). A fresh node gains its anchor here; a genuine
    #   re-write refreshes it to the new pristine body.
    # Why: P1 noted the second decay axis only engages once a node HAS an anchor and did
    #   NOT auto-capture; this is that capture point. CRITICAL SAFETY: `body` here is the
    #   pristine just-written body — the anchor is NEVER captured from the eroded served
    #   body (erode/integrity_decay_pass leave the anchor alone), so repair stays faithful.
    #   Fail-soft + additive: an anchor failure never breaks the write.
    try:
        from . import integrity as _integrity
        out["anchor"] = _integrity.capture_on_write(memory_dir, p.name,
                                                     {"name": title}, body)
    except Exception as e:
        out["anchor_error"] = str(e)
    # FEAT-2026-06-07 Tier-1 P2 — capture lands at the RING as a POINTER (not a copy)
    # carrying a salience signal. Register the pointer + compute/write the salience
    # frontmatter field (explicit salience_tag clamps it high). Fail-open / additive.
    cap = _register_ring_and_salience(memory_dir, p.name,
                                      f"{title}. {description}\n\n{body}",
                                      salience_tag)
    if cap:
        out["capture"] = cap
    # FEAT-2026-06-07 P3b — ONLINE auto-supersede on the write seam.
    # What: after the write lands, check the bounded active-locus for an exact
    #   supersession of a co-activation neighbor / hot node and auto-retire it
    #   (restorably); record weaker hits for the passive judge.
    # Why: Q4 OVERRIDE — close the negative-consolidation loop at write time on the
    #   obvious case. GATED behind ASTHENOS_CONTRADICTION_ENABLED (default OFF) and
    #   fully fail-soft, so it is inert + harmless until the operator enables it.
    sup = _online_supersede(memory_dir, p.name,
                            f"{title}. {description}\n\n{body}", valid_to)
    if sup.get("superseded") or sup.get("recorded"):
        out["supersession"] = sup
    return out


def memory_tag_salient(memory_dir: Path, node: str,
                       value: bool = True) -> dict[str, Any]:
    """FEAT-2026-06-07 Tier-1 P2 (D6) — the EXPLICIT operator/agent salience override.

    What: set (or clear) the explicit salience tag on a node and recompute its
      `salience` frontmatter. value=True is the deliberate "this matters" override that
      clamps salience HIGH (bio.SALIENCE_TAG_VALUE) regardless of the composite signals;
      value=False clears the override so salience falls back to the composite.
    Why:  D6 Q8a — the explicit-tag path, exposed as an MCP/CLI surface (the operator/
      agent override is the only sticky, operator-visible salience component, Risk 9).
      Returns {node, salience, salience_tag}; fail-soft on a missing node.
    """
    from . import bio as _bio
    fname = node if node.endswith(".md") else f"{node}.md"
    if not (_nodes_dir(memory_dir) / fname).exists():
        return {"error": f"node not found: {fname}"}
    sal = _bio.compute_salience(memory_dir, fname, explicit_tag=bool(value),
                                write=True)
    return {"node": fname, "salience": sal, "salience_tag": bool(value)}


def memory_extract_facts(
    memory_dir: Path,
    text: str,
    backend: str = "auto",
    chains_hint: list[str] | None = None,
) -> list[dict[str, Any]]:
    from . import fact_extractor as _fx
    return _fx.extract_atoms(text, backend=backend, chains_hint=chains_hint)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------


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
    from . import temporal as _tq
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
    from . import temporal as _tq
    from . import chain as _ct
    d = _tq.parse_date(at)
    if not d:
        return [{"error": "at must be ISO date"}]
    return _ct.query_at(memory_dir, chain, d)


def memory_chain_traverse_at(memory_dir: Path, chain: str, start: str, at: str,
                             depth: int = 3) -> list[dict[str, Any]]:
    from . import temporal as _tq
    from . import chain as _ct
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
    from . import chain as _ct
    out = _ct.set_edge(_chains_dir(memory_dir), chain, from_addr, to_addr,
                       valid_from=valid_from, valid_to=valid_to,
                       label=label, confidence=confidence)
    return out or {"error": "set_edge returned no result"}


def memory_chain_invalidate_edge(memory_dir: Path, chain: str, from_addr: str,
                                  to_addr: str, on: str,
                                  label: str | None = None) -> dict[str, Any]:
    from . import chain as _ct
    n = _ct.invalidate_edge(_chains_dir(memory_dir), chain, from_addr, to_addr, on, label=label)
    return {"closed": n}


def memory_chain_snapshot_at(memory_dir: Path, at: str) -> dict[str, Any]:
    from . import temporal as _tq
    from . import chain as _ct
    d = _tq.parse_date(at)
    if not d:
        return {"error": "at must be ISO date"}
    return _ct.snapshot_at(memory_dir, d)


# ---------------------------------------------------------------------------
# Biomimetic primitives
# ---------------------------------------------------------------------------


def memory_pattern_separate(memory_dir: Path, text: str,
                             threshold: float = 0.85) -> dict[str, Any]:
    from . import bio as _bio
    return _bio.pattern_separation_decision(memory_dir, text, threshold=threshold)


def memory_hebbian_consolidate(memory_dir: Path) -> dict[str, Any]:
    from . import bio as _bio
    return _bio.hebbian_consolidate(memory_dir)


def memory_replay_sweep(memory_dir: Path, sample: int = 20,
                         threshold: float = 0.55) -> dict[str, Any]:
    from . import bio as _bio
    return _bio.replay_sweep(memory_dir, sample=sample, threshold=threshold)


def memory_reconsolidate(memory_dir: Path, node: str, new_context: str,
                          backend: str = "auto") -> dict[str, Any]:
    from . import bio as _bio
    return _bio.reconsolidate(memory_dir, node, new_context, backend=backend)


def memory_schema_check(memory_dir: Path, text: str,
                         chains: list[str]) -> dict[str, Any]:
    from . import bio as _bio
    return _bio.schema_accelerate(memory_dir, text, chains)


def memory_chain_maturity(memory_dir: Path, chain: str) -> dict[str, Any]:
    from . import bio as _bio
    return _bio.chain_maturity(memory_dir, chain)


# ---------------------------------------------------------------------------
# Forgetting / negative consolidation -- FEAT-2026-06-07 P0
# ---------------------------------------------------------------------------


def _node_subject(memory_dir: Path, node: str) -> str:
    """The subject key of a node = its frontmatter `name`, lower/stripped.

    What: read nodes/<node>.md and return name; the file stem if no name field.
    Why:  the ONLINE exact-supersession test is "same subject key" — a near-
          identical claim ABOUT THE SAME SUBJECT — so we compare frontmatter
          names, not bodies (two unrelated nodes can be cosine-close).
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = _nodes_dir(memory_dir) / fname
    if not p.exists():
        return ""
    try:
        from . import frontmatter as _fm
        parsed, _ = _fm.parse(p.read_text(encoding="utf-8"))
        if parsed is not None:
            return str(parsed[0].get("name", "")).strip().lower()
    except Exception:
        pass
    return fname[:-3].strip().lower()


def _salience_guards_supersede(memory_dir: Path, old_fname: str) -> bool:
    """True iff the P3 salience guard protects old_fname from online auto-supersede.

    What: consult bio.salience_merge_guard on the old node with is_duplicate=False;
          True when it is a DISTINCT high-salience memory the guard protects (the
          caller then SURFACES the supersession for operator review instead of
          auto-removing it). False when bio lacks salience_merge_guard (the online
          path ships before Tier-1's salience field lands) or the node is not
          high-salience.
    Why:  D6 effect (iii) / Q5a — the salience merge/supersede guard is CONSUMED by
          the contradiction detector's ONLINE auto-supersede (here) AND the merge
          consumer. Wired behind a hasattr guard so the online path runs fully
          before the salience field exists, activating with no re-sequence once
          Tier-1 Phase 5 lands. Pure read; mutates nothing.
    """
    try:
        from . import bio as _bio
    except Exception:
        return False
    guard = getattr(_bio, "salience_merge_guard", None)
    if guard is None:
        return False
    try:
        return bool(guard(memory_dir, old_fname, is_duplicate=False))
    except Exception:
        return False


def _online_supersede(memory_dir: Path, new_node: str, text: str,
                      valid_to: str | None) -> dict[str, Any]:
    """FEAT-2026-06-07 P3b — ONLINE auto-supersede on the write path (active-locus).

    What: after a successful write, run find_supersession_candidates scoped to the
          bounded active-set (co-activation neighbors + hot/recent, via
          bio.active_set). For the EXACT case (cosine >= the auto bar AND same
          subject key as the new write) AUTO-supersede NOW via the RESTORABLE forget
          path (set valid_to on the old node, then forget_node(reason="supersede")
          which full-archives it) — UNLESS the P3 SALIENCE GUARD fires (the old node
          is a DISTINCT high-salience memory), in which case the supersession is
          SURFACED for operator review (status="surfaced-salience") instead of auto-
          removed (D6 effect iii / Q5a). WEAKER hits (0.75 <= cosine < auto bar) are
          recorded to the unified candidate store with mode="online" for the later
          passive LLM judge — never auto-deleted online. No LLM call here.
    Why:  Q4 OPERATOR OVERRIDE + the Q4-granularity decision. Auto-supersede is made
          safe by reversibility (restore_node + self-healing). The active-set keeps
          the write path cheap and bounded; the no-judge online path stays
          conservative (only the obvious exact case acts; the rest waits for P3c).
          R8: GATED behind ASTHENOS_CONTRADICTION_ENABLED (default OFF) → inert
          until the operator enables it + restarts the daemon. Fail-soft: a
          detector error never blocks or corrupts the write.
    """
    result: dict[str, Any] = {"superseded": [], "recorded": [], "checked": 0}
    try:
        from samia.runtime import contradiction as _con
    except ImportError:
        return result
    # R8 produce-only gate: the entire online behavior is inert unless enabled.
    if not _con.is_enabled():
        result["enabled"] = False
        return result
    result["enabled"] = True

    new_fname = new_node if new_node.endswith(".md") else f"{new_node}.md"
    try:
        from . import bio as _bio
        scope = _bio.active_set(memory_dir, [new_fname])
    except Exception as e:  # fail-soft: no locus → nothing to do.
        result["error"] = f"active_set: {e}"
        return result
    if not scope:
        return result

    try:
        cands = _con.find_supersession_candidates(
            text, scope_nodes=scope, memory_dir=memory_dir)
    except Exception as e:  # fail-soft: detector error must not break the write.
        result["error"] = f"detector: {e}"
        return result
    result["checked"] = len(cands)
    if not cands:
        return result

    new_subject = _node_subject(memory_dir, new_fname)
    auto_bar = _con.auto_cosine_threshold()
    today = _dt.date.today().isoformat()
    for c in cands:
        old_id = str(c["node_id"])
        old_fname = old_id if old_id.endswith(".md") else f"{old_id}.md"
        if old_fname == new_fname:
            continue
        cosine = float(c.get("score", 0.0))
        same_subject = bool(new_subject) and (
            _node_subject(memory_dir, old_fname) == new_subject)
        if cosine >= auto_bar and same_subject:
            # P3 SALIENCE GUARD (D6 effect iii / Q5a): do NOT auto-supersede a
            # DISTINCT high-salience old node — surface it for operator review
            # instead (record a guarded candidate, never auto-remove). A
            # contradicting/superseding claim pair is distinct, so is_duplicate
            # stays False; an exact duplicate is not the guard's target.
            if _salience_guards_supersede(memory_dir, old_fname):
                _con.record_supersession_candidate(
                    memory_dir, old_fname, new_fname, cosine=cosine,
                    jaccard=c.get("jaccard"), mode="online",
                    status="surfaced-salience")
                result.setdefault("guarded", []).append(
                    {"old_id": old_fname, "cosine": cosine})
                continue
            # EXACT case → auto-supersede now via the RESTORABLE forget path.
            from . import temporal as _temporal
            vt = valid_to or today
            try:
                if (_nodes_dir(memory_dir) / old_fname).exists():
                    _temporal.set_valid(memory_dir, old_fname, None, vt)
            except Exception:
                pass  # best-effort close; the archive below preserves the body.
            from . import ia as _ia
            cascade = _ia.forget_node(memory_dir, old_fname, reason="supersede",
                                      superseded_by=new_fname)
            _con.record_supersession_candidate(
                memory_dir, old_fname, new_fname, cosine=cosine,
                jaccard=c.get("jaccard"), mode="online", status="confirmed")
            _con.mark_supersession_confirmed(memory_dir, old_fname, new_fname)
            result["superseded"].append(
                {"old_id": old_fname, "cosine": cosine, "valid_to": vt,
                 "cascade": cascade})
        else:
            # WEAKER hit → record for the passive judge; nothing deleted.
            _con.record_supersession_candidate(
                memory_dir, old_fname, new_fname, cosine=cosine,
                jaccard=c.get("jaccard"), mode="online")
            result["recorded"].append({"old_id": old_fname, "cosine": cosine})
    return result


def memory_forget_node(memory_dir: Path, node: str,
                       reason: str = "manual") -> dict[str, Any]:
    """Cross-tier invalidation cascade for a dead/superseded node.

    What: thin wrapper over ia.forget_node -- hard-deletes the node's edges from
          edges.db (all ref_kinds) + edge_weights.json, strips its chain
          membership + hebbian edges, tombstones its vector entry, and appends a
          forgotten-log entry. The node FILE is expected already gone.
    Why:  exposes the FEAT-2026-06-07 P0 cascade primitive (built in ia.py and
          auto-wired into freeze/merge) as an explicit MCP/CLI surface for the
          confirm step of a contradiction supersession and for ad-hoc cleanup.
          Idempotent and fail-soft per store.
    """
    from . import ia as _ia
    return _ia.forget_node(memory_dir, node, reason=reason)


def memory_supersession_candidates(memory_dir: Path) -> dict[str, Any]:
    """List un-resolved supersession candidates from the UNIFIED store (R2).

    What: returns the {old_id, new_id, cosine, jaccard, mode, ts, ...} candidates
          recorded by the online write seam (weaker hits) and the passive judge,
          reading the single canonical store
          (contradiction.list_supersession_candidates).
    Why:  R2 — one owner, one schema. The online exact case auto-supersedes
          (restorably); these remaining candidates are the weaker hits awaiting
          the passive LLM judge / operator review. Nothing is deleted until acted.
    """
    try:
        from samia.runtime import contradiction as _con
        return {"candidates": _con.list_supersession_candidates(memory_dir)}
    except Exception as e:  # fail-open: never raise into the MCP loop.
        return {"candidates": [], "error": str(e)}


def memory_confirm_supersession(memory_dir: Path, old_id: str,
                                 valid_to: str | None = None,
                                 new_id: str | None = None) -> dict[str, Any]:
    """Confirm a supersession → RESTORABLE retire of the old node (R3).

    What: sets valid_to on the OLD node (provenance-preserving close), then fires
          the RESTORABLE forget path forget_node(reason="supersede") — which (R1)
          full-archives the node before the cascade so restore_node can un-forget
          it byte-exact — and marks the matching candidate(s) confirmed in the
          unified store.
    Why:  Q4 OPERATOR OVERRIDE — auto-supersede made safe by reversibility. A
          confirmed supersession is now restorable (it was NOT before R1, because
          reason="supersede" did not archive). The node FILE is closed via valid_to
          first (temporal provenance), then archived + its ghost edges purged.
    """
    from . import temporal as _temporal
    today = _dt.date.today().isoformat()
    vt = valid_to or today
    fname = old_id if old_id.endswith(".md") else f"{old_id}.md"
    p = _nodes_dir(memory_dir) / fname
    result: dict[str, Any] = {"old_id": fname, "valid_to": vt}

    # Step 1: close the old node's validity window (provenance-preserving) BEFORE
    # the archiving forget — so the archived frontmatter carries the valid_to.
    if p.exists():
        try:
            _temporal.set_valid(memory_dir, fname, None, vt)
            result["closed"] = True
        except Exception as e:
            result["closed"] = False
            result["close_error"] = str(e)
    else:
        result["closed"] = False
        result["note"] = "node file absent; cascading edge purge only"

    # Step 2: RESTORABLE retire — reason="supersede" full-archives (R1) then cascades.
    from . import ia as _ia
    result["cascade"] = _ia.forget_node(memory_dir, fname, reason="supersede",
                                        superseded_by=new_id)

    # Step 3: mark the candidate(s) confirmed in the unified store.
    try:
        from samia.runtime import contradiction as _con
        result["candidates_confirmed"] = _con.mark_supersession_confirmed(
            memory_dir, old_id=fname, new_id=new_id)
    except Exception as e:  # fail-open: the cascade already ran.
        result["candidates_confirmed"] = 0
        result["candidate_log_error"] = str(e)

    return result


def memory_dismiss_supersession(memory_dir: Path, old_id: str,
                                new_id: str | None = None) -> dict[str, Any]:
    """Dismiss a supersession candidate (false positive) in the unified store.

    What: marks the matching candidate(s) dismissed; deletes nothing, sets no
          valid_to. If the candidate names an already-auto-superseded node, the
          operator can additionally call memory_restore_node to un-forget it.
    Why:  R2 — the operator's reject path on the single store. A 0.75 cosine smell
          is weak; dismissal records the rejection so it stops surfacing.
    """
    fname = old_id if old_id.endswith(".md") else f"{old_id}.md"
    try:
        from samia.runtime import contradiction as _con
        return {"old_id": fname,
                "dismissed": _con.mark_supersession_dismissed(
                    memory_dir, old_id=fname, new_id=new_id)}
    except Exception as e:
        return {"old_id": fname, "dismissed": 0, "error": str(e)}


def memory_restore_node(memory_dir: Path, node_id: str) -> dict[str, Any]:
    """Un-forget a superseded node from its archive (R4 — over ia.restore_node).

    What: thin wrapper over ia.restore_node — re-creates nodes/<id>.md byte-exact
          from archive/<id>.superseded.json, un-tombstones its vector entry, stamps
          restore_ts, logs a restore event.
    Why:  Q4 OVERRIDE — auto-supersede is acceptable ONLY because it is reversible.
          This is the operator/self-healing un-forget surface for an online
          auto-supersede or a confirmed supersession that turned out wrong.
    """
    from . import ia as _ia
    return _ia.restore_node(memory_dir, node_id)


# ---------------------------------------------------------------------------
# Tier-2 merge consumer P2 — gated LLM-synthesized abstraction surface
# (mirror of the P3 supersession confirm/reject; operator-only confirm).
# ---------------------------------------------------------------------------


def memory_merge_candidates(memory_dir: Path) -> dict[str, Any]:
    """List un-resolved Tier-2 merge/abstraction candidates (P2 surface).

    What: returns the {candidate_id, a, b, status, abstraction?, merged_from?}
          records from biomimetic/merge_candidates.jsonl that the consumer queued
          ('pending' — awaiting synthesis) or PROPOSED ('proposed' — a synthesized
          draft awaiting operator confirm). Reads
          merge_consumer.list_abstraction_candidates.
    Why:  Q2c — abstractions are operator-gated. This is the operator's listing
          surface; a 'proposed' entry carries the synthesized title+body so the
          operator can review before confirming. Nothing is applied until acted.
    """
    try:
        from . import merge_consumer as _mc
        return {"candidates": _mc.list_abstraction_candidates(memory_dir)}
    except Exception as e:  # fail-open: never raise into the MCP loop.
        return {"candidates": [], "error": str(e)}


def memory_confirm_merge(memory_dir: Path, candidate_id: str) -> dict[str, Any]:
    """Confirm a PROPOSED abstraction → create the node + supersede both sources.

    What: materialize the proposed draft as a new nodes/<id>.md (synthesized
          content + merged_from provenance frontmatter), then SUPERSEDE both
          source nodes RESTORABLY (reason="supersede" full-archives each so
          memory_restore_node can un-forget them byte-exact) and lay provenance
          edges abstraction->each source. Delegates to
          merge_consumer.confirm_abstraction.
    Why:  Q2c GATE — abstractions create NEW content + can lose nuance, so they
          are applied ONLY on operator confirm; both originals stay restorable.
          Mirrors memory_confirm_supersession.
    """
    try:
        from . import merge_consumer as _mc
        return _mc.confirm_abstraction(memory_dir, candidate_id)
    except Exception as e:
        return {"confirmed": False, "candidate_id": candidate_id, "error": str(e)}


def memory_reject_merge(memory_dir: Path, candidate_id: str) -> dict[str, Any]:
    """Reject a proposed abstraction (changes NOTHING) — the gate's reject arm.

    What: marks the candidate rejected so it stops surfacing; no node created, no
          source superseded, both originals stay live. Delegates to
          merge_consumer.reject_abstraction.
    Why:  Q2c — the operator's reject path; mirrors memory_dismiss_supersession.
          A synthesized abstraction that loses nuance is discarded with zero
          mutation of live memory.
    """
    try:
        from . import merge_consumer as _mc
        return _mc.reject_abstraction(memory_dir, candidate_id)
    except Exception as e:
        return {"rejected": False, "candidate_id": candidate_id, "error": str(e)}


# ---------------------------------------------------------------------------
# Context-extension primitives (SAM/IA × compaction hybrids)
# ---------------------------------------------------------------------------


def memory_chainogram_retrieve(memory_dir: Path, query: str,
                                budget_tokens: int = 8000,
                                max_chains: int = 8,
                                include_failure_associations: bool = False,
                                failure_top_n: int | None = None) -> dict[str, Any]:
    # FEAT-2026-06-10 P2a — semantic-arm wire. When the arm is enabled, OVERLAY the
    # composed read-side context (KNOWN FACTS + CONVERSATION EVIDENCE) onto the
    # existing chainogram result shape: the standard keys (loaded_chains/loaded_nodes/
    # spent_tokens/rationale/...) stay populated exactly as today so MCP clients that
    # read them never break, and the composed extras land under NEW keys (composed_*,
    # facts_n). Flag OFF -> the original path runs untouched (branch around, do not
    # restructure). The arm flag default-OFF means this branch is byte-identical to the
    # pre-P2 behavior until the operator enables ASTHENOS_SEMANTIC_ARM_ENABLED.
    from . import semantic_recall as _sr
    from . import context_extension as _cx
    base = _cx.chainogram_retrieve(memory_dir, query, budget_tokens=budget_tokens,
                                   max_chains=max_chains,
                                   include_failure_associations=include_failure_associations,
                                   failure_top_n=failure_top_n)
    if not _sr.semantic_arm_enabled():
        return base
    # Arm ON: route to the composer for the composed context (it runs its own focused
    # chainogram + atom arm internally). Surface its outputs under additive keys so the
    # tool contract (name + existing keys) is preserved. Fail-open: a composer error
    # leaves the standard chainogram result intact.
    try:
        composed = _sr.recall(memory_dir, query, budget_tokens=budget_tokens)
    except Exception as exc:  # pragma: no cover - defensive
        base["semantic_arm_error"] = str(exc)
        return base
    base["composed_context"] = composed.get("context", "")
    base["facts_n"] = composed.get("facts_n", 0)
    base["composed_evidence_nodes"] = composed.get("evidence_nodes", 0)
    base["composed_dia_ids"] = composed.get("dia_ids", [])
    base["semantic_arm"] = True
    return base


def memory_frozen_prefix(memory_dir: Path, write: bool = True) -> dict[str, Any]:
    from . import context_extension as _cx
    return _cx.frozen_prefix_block(memory_dir, write=write)


def memory_tier_flow_for_budget(memory_dir: Path, query: str,
                                  budget_tokens: int = 8000,
                                  apply: bool = False) -> dict[str, Any]:
    from . import context_extension as _cx
    return _cx.tier_flow_for_budget(memory_dir, query, budget_tokens=budget_tokens,
                                    dry_run=not apply)


def memory_episodic_candidates(memory_dir: Path) -> dict[str, Any]:
    from . import context_extension as _cx
    return _cx.episodic_to_semantic_candidates(memory_dir)


def memory_idle_tick(memory_dir: Path, force: bool = False) -> dict[str, Any]:
    from . import context_extension as _cx
    return _cx.idle_replay_tick(memory_dir, force=force)


def memory_sm2_update(memory_dir: Path, node: str, missed: bool = False,
                       quality: int = 4) -> dict[str, Any]:
    from . import context_extension as _cx
    return _cx.sm2_review_update(memory_dir, node, recalled=not missed, quality=quality)


def memory_sm2_due(memory_dir: Path) -> list[dict[str, Any]]:
    from . import context_extension as _cx
    return _cx.sm2_due_for_review(memory_dir)


def memory_compaction_skip_filter(memory_dir: Path,
                                    transcript_chunks: list[str],
                                    threshold: float = 0.78) -> dict[str, Any]:
    from . import context_extension as _cx
    return _cx.compaction_skip_filter(memory_dir, transcript_chunks, threshold=threshold)


# ---------------------------------------------------------------------------
# Index status
# ---------------------------------------------------------------------------


def memory_index_status(memory_dir: Path) -> dict[str, Any]:
    from . import vector as _vi
    m = _vi._load_manifest(memory_dir)
    return {
        "model": m.get("model_id"),
        "dim": m.get("dim"),
        "node_count": m.get("node_count"),
        "built_at": m.get("built_at"),
    }


def memory_rem_status(memory_dir: Path) -> dict[str, Any]:
    """REM sleep-cycle state + the sleep-pressure health gauge.

    What: returns the persisted WAKE<->REM state plus the composite
          sleep-pressure breakdown (per-signal + score + threshold +
          sleep_needed) — the operator-visible health gauge — so the Atoms /
          Claude surface can read whether reconciliation is owed and whether the
          system is asleep.
    Why:  the thin read half of the REM P1 observability surface
          (FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01). The
          explicit "sleep now" trigger lives on the daemon IPC op rem_sleep_now;
          this is the plain-function read MCP wraps. Lazy import keeps the runtime
          dependency off the core import path.
    """
    from samia.runtime import rem_cycle
    return rem_cycle.rem_status(Path(memory_dir))


def memory_rem_sleep_now(memory_dir: Path) -> dict[str, Any]:
    """Explicit REM "sleep now" trigger (sets the force flag).

    What: flips the force-requested flag so the next daemon tick enters REM
          regardless of pressure/idle; returns {ok, state}.
    Why:  the on-demand cycle trigger (Q1 explicit path / risk-1 mitigation),
          exposed as a plain function the MCP / IPC surface wraps. Produce-only:
          it only records the request; the daemon tick applies it.
    """
    from samia.runtime import rem_cycle
    return rem_cycle.rem_sleep_now(Path(memory_dir))
