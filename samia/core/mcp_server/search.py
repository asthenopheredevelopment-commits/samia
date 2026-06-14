"""samia.core.mcp_server.search — the recall / retrieval arm of the MCP tool server.

Layer 1 (Owns / Depends):
    Owns:    the read-side tool logic — the post-retrieval filters/re-rankers
             (_filter_by_runtime, _rerank_hits, _term_index_lookup), the Tier-0 D4
             co-activation neighbor read-back (_coactivation_neighbors, the
             COACT_*-bounded edges.db pull), the Tier-1 fast-tier arms
             (_engram_rag_hits / _ring_rag_hits), the standing-availability inject
             assembler (memory_inject_block), the main recall pipeline
             (memory_search), the time-scoped query (memory_temporal_query), and the
             single-node read (memory_read_node).
    Depends: .config (_nodes_dir, _ws, sqlite3/os/Any/Path, the COACT_* constants).
             Lazy per-call: samia.core.{vector,bio,hippocampus,integrity,temporal}
             (kept function-local to avoid an import cycle — bio/hippocampus import
             back through the package surface).

Layer 2 (What / Why):
    What: every read tool's underlying logic, parameterized on memory_dir. The
          pipeline folds term-index + cosine + engram-RAG + ring-RAG hits, records
          the GENUINE co-activation BEFORE read-back neighbors, runs recall-repair
          and the AUTO ring->engram promotion, then appends clamped co-activation
          neighbors. The fast-tier and co-activation reads are all fail-open.
    Why:  the recall responsibility is the largest single cohesive seam of the MCP
          server; isolating it keeps the write/forget mutation path and the chain/
          context tools out of the hot read path. The lazy imports keep the heavy
          embedder/hippocampus deps off the package import path (R: import-cost).

PATCH NOTE: _coactivation_neighbors is reached directly by the tests as
    mcp._coactivation_neighbors; it is re-exported by the package facade. memory_search
    (its only in-package caller) is co-located here, so no facade-reach is needed for it
    — the tests patch nothing through the module on this seam (they call it directly).
"""

from __future__ import annotations

from .config import (
    Any,
    Path,
    os,
    sqlite3,
    _nodes_dir,
    _ws,
    COACT_LAMBDA,
    COACT_MAX_NEIGHBORS,
    COACT_PARENT_HITS,
    COACT_DELTA,
)


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
        from .. import hippocampus as _hip
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
        from .. import hippocampus as _hip
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
    from .. import hippocampus as _hip
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
    from .. import vector as _vi
    from .. import bio as _bio
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
            from .. import integrity as _integrity
            _do_repair = _integrity.repair_enabled()
        except Exception:
            _do_repair = False
    if _do_repair:
        try:
            from .. import integrity as _integrity
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
            from .. import hippocampus as _hip
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
    from .. import temporal as _tq
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
    from .. import temporal as _tq
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


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.mcp_server.search
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.mcp_server monolith during
#             modularization (the search & retrieval section — the largest cohesive arm).
# Layer:      core (pure library, no daemon dependency)
# Role:       the recall / retrieval arm — the post-retrieval filters/re-rankers, the
#             Tier-0 D4 co-activation read-back, the Tier-1 engram/ring fast-tier arms,
#             the inject-block assembler, and the memory_search / temporal_query /
#             read_node tool logic.
# Stability:  stable — behavior byte-identical to the monolith's search & retrieval
#             section; only the module-top imports moved behind .config.
# ErrorModel: fail-open on every learned-layer read (co-activation/engram/ring/repair/
#             promote/inject) — an error there yields no extra hits and the main cosine
#             path is unaffected; the core vector query surfaces its own errors as hits.
# Depends:    .config (_nodes_dir, _ws, sqlite3/os/Any/Path, COACT_*). Lazy per-call:
#             samia.core.{vector,bio,hippocampus,integrity,temporal}.
# Exposes:    memory_inject_block, memory_search, memory_temporal_query, memory_read_node
#             (public); _filter_by_runtime/_rerank_hits/_term_index_lookup/
#             _coactivation_neighbors (test-reached)/_engram_rag_hits/_ring_rag_hits.
# Lines:      540
# Note:       _coactivation_neighbors is re-exported by the facade for direct test access
#             (mcp._coactivation_neighbors); its only in-package caller (memory_search) is
#             co-located, so no patch-seam facade-reach is needed here.
# --------------------------------------------------------------------------
