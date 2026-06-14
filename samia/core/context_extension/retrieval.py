"""samia.core.context_extension.retrieval — the chainogram retrieval family.

Layer 1 (Owns / Depends):
    Owns:    the five chain-level retrieval primitives bounded by a token budget —
             the sparse engram retrieval (chainogram_retrieve, the base; carries the
             Hebbian-density boost, the FEAT-2026-06-10 Q4a semantic-arm chain-SELECTION
             skip, the flagged-on temporal envelope hook, and the read-seam overlay),
             the entity-bridge augmented retrieval (chainogram_retrieve_bridged), the
             hybrid union retrieval (chainogram_retrieve_hybrid), the cross-encoder
             reranked retrieval (chainogram_retrieve_reranked), and the contextual-seed
             retrieval (chainogram_retrieve_contextual).
    Depends: the package config leaf (the budget default, np/json, the aliased deps
             _bio/_ct/_vi/_ei/_vic, the path/IO/vector helpers + the reranker accessor
             + _is_atom_chain), the temporal arm (_apply_temporal_envelope, reached
             through the package facade), the read-seam arm (_resolve_read_seam_top_n +
             _query_failure_associations), and — lazily — samia.core.semantic_recall
             (the fx_-skip flag, function-local to break the cycle).

Layer 2 (What / Why):
    What: the read-path primitives that turn a query into a budget-packed set of nodes.
          The base chainogram scores chains by relevance + Hebbian density (+ the
          flagged-on temporal envelope), then packs members under budget; the variants
          add entity bridges, a hybrid union, a cross-encoder rerank, or a contextual
          seed index.
    Why:  these are the context-budget primitives that work WITH production compaction.
          Grouping the whole family keeps the shared packing/scoring idioms in one place.

PATCH SEAM (exemplar rule, HIGH blast radius): chainogram_retrieve is a
    mock.patch.object(cx, "chainogram_retrieve") target (test_semantic_recall_p2 mocks
    it on the package facade to hermetically drive mcp_server) AND is called by the
    siblings chainogram_retrieve_bridged + chainogram_retrieve_contextual. Those two
    callers reach it THROUGH the package facade (from samia.core import context_extension
    as _pkg; _pkg.chainogram_retrieve) so a package-level patch rebinds the function the
    variants actually run — matching the contradiction/mcp_server exemplar discipline.
"""

from __future__ import annotations

from pathlib import Path

# Shared leaf — the budget default, np/json, the aliased deps, the path/IO/vector helpers
# + the reranker accessor + the atom-chain classifier.
from .config import (
    DEFAULT_BUDGET_TOKENS,
    json,
    np,
    _bio,
    _ct,
    _vi,
    _ei,
    _vic,
    _nodes_dir,
    _chains_dir,
    _node_text,
    _is_atom_chain,
    _vi_manifest,
    _vi_query,
    _get_reranker,
    _RERANKER_NAME,
)
# The temporal envelope + read-seam arms. The envelope is folded in flagged-on only.
from .temporal import _apply_temporal_envelope, temporal_weight_enabled
from .readseam import _resolve_read_seam_top_n, _query_failure_associations


# ---------------------------------------------------------------------------
# Primitive A — Sparse engram retrieval
# ---------------------------------------------------------------------------


def chainogram_retrieve(memory_dir: Path, query: str,
                        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                        max_chains: int = 8,
                        include_singletons: bool = False,
                        _vi_module=None,
                        include_failure_associations: bool = False,
                        failure_top_n: int | None = None,
                        _web_db_dir: str | None = None) -> dict:
    """Sparse chain-level retrieval bounded by a token budget."""
    vi = _vi_module if _vi_module is not None else _vi
    manifest_path = _vi_manifest(vi, memory_dir)
    if not manifest_path.exists():
        return {"error": f"no vector index at {manifest_path}"}

    nodes_dir = _nodes_dir(memory_dir)
    chains_dir = _chains_dir(memory_dir)

    hits = _vi_query(vi, memory_dir, query, top_k=24)
    # FEAT-2026-06-10 Q4a — semantic-arm chain SELECTION skip. When the semantic arm is
    # on, atom mini-chains (fx_-prefixed ids, or chains whose first member resolves to a
    # type:semantic node) are EXCLUDED from candidate selection: the atom population is
    # served by the peer semantic arm and the two populations meet in the composer
    # (semantic_recall.recall), not inside this episodic arm. Resolved ONCE here, gated by
    # the flag so the unflagged path never even resolves a type — flag off => byte-
    # identical to today's selection. Lazy import dodges the context_extension<->
    # semantic_recall cycle.
    _semantic_arm_on = False
    try:
        from .. import semantic_recall as _sr
        _semantic_arm_on = _sr.semantic_arm_enabled()
    except Exception:
        _semantic_arm_on = False
    chain_scores: dict[str, dict] = {}
    for h in hits:
        ca = _bio._addr_for_node(memory_dir, h["node"])
        if not ca:
            continue
        chain_name, addr = ca
        if _semantic_arm_on and _is_atom_chain(memory_dir, chain_name):
            continue
        info = chain_scores.setdefault(chain_name, {
            "score": 0.0, "best_node": h["node"], "best_score": 0.0,
            "addrs": set(),
        })
        info["score"] += float(h["score"])
        info["addrs"].add(addr)
        if h["score"] > info["best_score"]:
            info["best_score"] = float(h["score"])
            info["best_node"] = h["node"]

    # Hebbian density boost. After this loop info["score"] = S_c + 0.05·H_c — the
    # raw additive base cue (S_c summed at :434, H_c counted here). This is the
    # byte-identical baseline; the temporal block below ONLY runs flagged-on.
    for cname in chain_scores:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        edges = chain.get("edges") or []
        hebb = sum(1 for e in edges if e.get("label") == "hebbian")
        chain_scores[cname]["score"] += 0.05 * hebb

    # Temporal-recall envelope (FEAT-2026-06-11 P1, §2). Flagged-off this whole
    # block is skipped, so info["score"] stays the raw S_c + 0.05·H_c above and the
    # sort is byte-identical to today (§2.6). Flagged-on, each term whose weight
    # clears ε (§16.2-Q5 compute-skip) is accumulated, pool min-max normalized, and
    # folded into score(c) = (S + 0.05·H + γ·TĈ)·(1 + λN·N̂ + λK·K̂ + λD·D̂). In P1
    # every weight defaults to 0.0 and every hook returns 0.0, so even flagged-on
    # the score is unchanged — the scaffold is a provable no-op until calibration.
    if temporal_weight_enabled():
        _apply_temporal_envelope(memory_dir, chain_scores, hits)

    ordered = sorted(chain_scores.items(), key=lambda kv: -kv[1]["score"])
    ordered = ordered[:max_chains]

    loaded_nodes: list[dict] = []
    skipped: list[dict] = []
    spent = 0
    seen_files: set[str] = set()
    chosen_chains: list[str] = []

    n_singletons = 0
    if include_singletons:
        for h in sorted(hits, key=lambda x: -float(x["score"])):
            if _bio._addr_for_node(memory_dir, h["node"]):
                continue
            fname = h["node"]
            if fname in seen_files:
                continue
            seen_files.add(fname)
            p = nodes_dir / fname
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, _ = _node_text(p)
            entry = {"node": p.name, "tokens": tok,
                     "chain": "<singleton>", "addr": None,
                     "score": float(h["score"])}
            if spent + tok <= budget_tokens:
                loaded_nodes.append(entry)
                spent += tok
                n_singletons += 1
            else:
                skipped.append(entry)

    for cname, info in ordered:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        chosen_chains.append(cname)
        for m in chain.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f or f in seen_files:
                continue
            seen_files.add(f)
            p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, _ = _node_text(p)
            entry = {"node": p.name, "tokens": tok, "chain": cname,
                     "addr": m.get("addr"), "score": info["best_score"]}
            if spent + tok <= budget_tokens:
                loaded_nodes.append(entry)
                spent += tok
            else:
                skipped.append(entry)

    rationale = (f"top-{len(chosen_chains)} chains by "
                 "(relevance + Hebbian density), packed under budget")
    if include_singletons:
        rationale += f"; +{n_singletons} singleton hit(s)"

    out = {
        "loaded_chains": chosen_chains,
        "loaded_nodes": loaded_nodes,
        "skipped_nodes": skipped,
        "budget_tokens": budget_tokens,
        "spent_tokens": spent,
        "n_singletons": n_singletons,
        "rationale": rationale,
    }

    # What: optionally surface cross-chain failure/diagnosis associations from
    #   the coactivation web (read-seam).
    # Why: closes the read side of the failure-experience storm — during
    #   diagnosis, callers see prior failures that are Hebbian-associated with
    #   the loaded nodes, ranked by weight x recency. Additive-only key;
    #   existing callers ignore it via dict.get().
    if include_failure_associations:
        eff_n = _resolve_read_seam_top_n(failure_top_n)
        all_loaded_names = [n["node"] for n in loaded_nodes]
        assocs = _query_failure_associations(
            memory_dir, all_loaded_names, eff_n, db_dir=_web_db_dir,
        )
        out["failure_associations"] = assocs
        if assocs:
            rationale += f"; +{len(assocs)} failure association(s) from web"
            out["rationale"] = rationale

    return out


# ---------------------------------------------------------------------------
# Primitive A.3 — Entity-bridge augmented retrieval
# ---------------------------------------------------------------------------


def chainogram_retrieve_bridged(memory_dir: Path, query: str,
                                budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                                max_chains: int = 8,
                                max_bridge_nodes: int = 8,
                                include_singletons: bool = True,
                                bridge_reserve_frac: float = 0.25) -> dict:
    # PATCH-SEAM reach: chainogram_retrieve is a mock.patch.object(cx, ...) target, so
    # this variant runs the BASE retrieval through the package facade (not the module-
    # local name) — a package-level patch then rebinds what this caller actually invokes.
    from samia.core import context_extension as _pkg
    bridge_reserve = int(budget_tokens * bridge_reserve_frac)
    chain_budget = budget_tokens - bridge_reserve
    out = _pkg.chainogram_retrieve(memory_dir, query, budget_tokens=chain_budget,
                                   max_chains=max_chains,
                                   include_singletons=include_singletons)
    if "error" in out:
        return out
    if _ei is None:
        out["bridge_nodes_added"] = 0
        out["rationale"] = (out.get("rationale", "") +
                            "; entity index unavailable")
        return out

    bridges = _ei.query_bridges(memory_dir, query,
                                max_bridge_nodes=max_bridge_nodes)
    if "error" in bridges:
        out["bridge_nodes_added"] = 0
        out["rationale"] = out.get("rationale", "") + "; " + bridges["error"]
        return out

    nodes_dir = _nodes_dir(memory_dir)
    loaded_files = {n["node"] for n in out.get("loaded_nodes") or []}
    spent = int(out.get("spent_tokens") or 0)
    n_added = 0
    for b in bridges.get("bridge_nodes") or []:
        fname = b["node"]
        if fname in loaded_files:
            continue
        p = nodes_dir / fname
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        tok, _ = _node_text(p)
        if spent + tok > budget_tokens:
            out.setdefault("skipped_nodes", []).append({
                "node": p.name, "tokens": tok, "chain": "<bridge>",
                "addr": None, "score": float(b["score"]),
            })
            continue
        out.setdefault("loaded_nodes", []).append({
            "node": p.name, "tokens": tok, "chain": "<bridge>",
            "addr": None, "score": float(b["score"]),
            "matched_entities": b["entities"],
        })
        loaded_files.add(p.name)
        spent += tok
        n_added += 1

    out["spent_tokens"] = spent
    out["bridge_nodes_added"] = n_added
    out["matched_entities"] = bridges.get("matched_entities") or []
    out["rationale"] = (out.get("rationale", "") +
                        f"; +{n_added} entity-bridge nodes "
                        f"({len(out['matched_entities'])} entities matched)")
    return out


# ---------------------------------------------------------------------------
# Primitive A.2 — Hybrid union retrieval (no rerank)
# ---------------------------------------------------------------------------


def chainogram_retrieve_hybrid(memory_dir: Path, query: str,
                               budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                               max_chains: int = 12,
                               extra_topk: int = 12) -> dict:
    if not _vi_manifest(_vi, memory_dir).exists():
        return {"error": "no vector index — run memory_vector_index.py build"}

    nodes_dir = _nodes_dir(memory_dir)
    chains_dir = _chains_dir(memory_dir)

    hits = _vi_query(_vi, memory_dir, query, top_k=max(40, extra_topk + 24))
    chain_scores: dict[str, float] = {}
    for h in hits:
        ca = _bio._addr_for_node(memory_dir, h["node"])
        if not ca:
            continue
        chain_name, _ = ca
        chain_scores[chain_name] = chain_scores.get(chain_name, 0.0) \
            + float(h["score"])

    ordered_chains = sorted(chain_scores.items(),
                            key=lambda kv: -kv[1])[:max_chains]

    candidates: dict[str, dict] = {}
    top10_files = {h["node"] for h in hits[:10]}

    n_singletons = 0
    for h in hits:
        if _bio._addr_for_node(memory_dir, h["node"]):
            continue
        fname = h["node"]
        p = nodes_dir / fname
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists() or fname in candidates:
            continue
        tok, _ = _node_text(p)
        candidates[fname] = {"node": p.name, "tokens": tok,
                             "chain": "<singleton>", "addr": None,
                             "vec_score": float(h["score"]),
                             "in_top10": fname in top10_files}
        n_singletons += 1

    chosen_chain_names: list[str] = []
    for cname, cscore in ordered_chains:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        chosen_chain_names.append(cname)
        for m in chain.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f or f in candidates:
                continue
            p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, _ = _node_text(p)
            vec_score = next((float(h["score"]) for h in hits
                              if h["node"] == p.name), 0.0)
            candidates[p.name] = {"node": p.name, "tokens": tok,
                                  "chain": cname, "addr": m.get("addr"),
                                  "vec_score": vec_score,
                                  "chain_score": cscore,
                                  "in_top10": p.name in top10_files}

    n_extra = 0
    for h in hits[:extra_topk]:
        fname = h["node"]
        if fname in candidates:
            continue
        p = nodes_dir / fname
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        tok, _ = _node_text(p)
        candidates[fname] = {"node": p.name, "tokens": tok,
                             "chain": "<topk>", "addr": None,
                             "vec_score": float(h["score"]),
                             "in_top10": fname in top10_files}
        n_extra += 1

    max_chain_score = max((c.get("chain_score", 0.0)
                           for c in candidates.values()), default=1.0) or 1.0
    for c in candidates.values():
        chain_norm = c.get("chain_score", 0.0) / max_chain_score
        c["hybrid_score"] = (0.55 * c["vec_score"]
                             + 0.20 * chain_norm
                             + 0.15 * (1.0 if c["in_top10"] else 0.0))

    ordered = sorted(candidates.values(), key=lambda c: -c["hybrid_score"])

    loaded_nodes: list[dict] = []
    skipped: list[dict] = []
    spent = 0
    for c in ordered:
        entry = {"node": c["node"], "tokens": c["tokens"], "chain": c["chain"],
                 "addr": c["addr"], "score": c["hybrid_score"],
                 "vec_score": c["vec_score"]}
        if spent + c["tokens"] <= budget_tokens:
            loaded_nodes.append(entry)
            spent += c["tokens"]
        else:
            skipped.append(entry)

    return {
        "loaded_chains": chosen_chain_names,
        "loaded_nodes": loaded_nodes,
        "skipped_nodes": skipped,
        "budget_tokens": budget_tokens,
        "spent_tokens": spent,
        "n_singletons": n_singletons,
        "n_extra_topk": n_extra,
        "n_candidates": len(candidates),
        "rationale": (f"hybrid union: {len(chosen_chain_names)} chains + "
                      f"{n_singletons} singletons + {n_extra} extra top-k, "
                      "ranked by 0.55·vec + 0.20·chain + 0.15·top10"),
    }


# ---------------------------------------------------------------------------
# Primitive A.1 — Cross-encoder reranked engram retrieval
# ---------------------------------------------------------------------------


def chainogram_retrieve_reranked(memory_dir: Path, query: str,
                                 budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                                 max_chains: int = 12,
                                 candidate_pool: int = 40,
                                 include_singletons: bool = True) -> dict:
    if not _vi_manifest(_vi, memory_dir).exists():
        return {"error": "no vector index — run memory_vector_index.py build"}

    nodes_dir = _nodes_dir(memory_dir)
    chains_dir = _chains_dir(memory_dir)

    hits = _vi_query(_vi, memory_dir, query, top_k=candidate_pool)
    chain_scores: dict[str, dict] = {}
    for h in hits:
        ca = _bio._addr_for_node(memory_dir, h["node"])
        if not ca:
            continue
        chain_name, addr = ca
        info = chain_scores.setdefault(chain_name, {
            "score": 0.0, "best_score": 0.0, "addrs": set(),
        })
        info["score"] += float(h["score"])
        info["addrs"].add(addr)
        if h["score"] > info["best_score"]:
            info["best_score"] = float(h["score"])

    for cname in chain_scores:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        edges = chain.get("edges") or []
        chain_scores[cname]["score"] += 0.05 * sum(
            1 for e in edges if e.get("label") == "hebbian"
        )

    ordered_chains = sorted(chain_scores.items(),
                            key=lambda kv: -kv[1]["score"])[:max_chains]

    candidates: list[dict] = []
    seen_files: set[str] = set()
    n_singletons = 0
    if include_singletons:
        for h in sorted(hits, key=lambda x: -float(x["score"])):
            if _bio._addr_for_node(memory_dir, h["node"]):
                continue
            fname = h["node"]
            if fname in seen_files:
                continue
            seen_files.add(fname)
            p = nodes_dir / fname
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, body = _node_text(p)
            candidates.append({"node": p.name, "tokens": tok, "body": body,
                               "chain": "<singleton>", "addr": None,
                               "vec_score": float(h["score"])})
            n_singletons += 1

    chosen_chain_names: list[str] = []
    for cname, info in ordered_chains:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        chosen_chain_names.append(cname)
        for m in chain.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f or f in seen_files:
                continue
            seen_files.add(f)
            p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, body = _node_text(p)
            candidates.append({"node": p.name, "tokens": tok, "body": body,
                               "chain": cname, "addr": m.get("addr"),
                               "vec_score": float(info["best_score"])})

    if not candidates:
        return {
            "loaded_chains": [], "loaded_nodes": [], "skipped_nodes": [],
            "budget_tokens": budget_tokens, "spent_tokens": 0,
            "n_singletons": 0, "n_candidates": 0,
            "rationale": "no candidates",
        }

    reranker = _get_reranker()
    pairs = [(query, c["body"][:2000]) for c in candidates]
    ce_scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, ce_scores):
        c["ce_score"] = float(s)

    candidates.sort(key=lambda c: -c["ce_score"])

    loaded_nodes: list[dict] = []
    skipped: list[dict] = []
    spent = 0
    for c in candidates:
        entry = {"node": c["node"], "tokens": c["tokens"], "chain": c["chain"],
                 "addr": c["addr"], "score": c["ce_score"],
                 "vec_score": c["vec_score"]}
        if spent + c["tokens"] <= budget_tokens:
            loaded_nodes.append(entry)
            spent += c["tokens"]
        else:
            skipped.append(entry)

    return {
        "loaded_chains": chosen_chain_names,
        "loaded_nodes": loaded_nodes,
        "skipped_nodes": skipped,
        "budget_tokens": budget_tokens,
        "spent_tokens": spent,
        "n_singletons": n_singletons,
        "n_candidates": len(candidates),
        "reranker": _RERANKER_NAME,
        "rationale": (f"cross-encoder reranked {len(candidates)} candidates "
                      f"({n_singletons} singletons + chain members from "
                      f"{len(chosen_chain_names)} chains)"),
    }


# ---------------------------------------------------------------------------
# Primitive A.4 — Contextual-seed engram retrieval
# ---------------------------------------------------------------------------


def chainogram_retrieve_contextual(memory_dir: Path, query: str,
                                   budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                                   max_chains: int = 8,
                                   include_singletons: bool = True) -> dict:
    if _vic is None:
        return {"error": "memory_vector_index_contextual unavailable"}
    if not _vi_manifest(_vic, memory_dir).exists():
        return {"error": (f"no contextual index — run "
                          f"memory_vector_index_contextual.py build")}
    # PATCH-SEAM reach: run the BASE retrieval through the package facade so a
    # package-level mock.patch.object(cx, "chainogram_retrieve") is honored here too.
    from samia.core import context_extension as _pkg
    out = _pkg.chainogram_retrieve(memory_dir, query, budget_tokens=budget_tokens,
                                   max_chains=max_chains,
                                   include_singletons=include_singletons,
                                   _vi_module=_vic)
    if "rationale" in out:
        out["rationale"] = "[contextual-seed] " + out["rationale"]
    out["seed_index"] = "contextual"
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.context_extension.retrieval
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Primitives A / A.1 / A.2 / A.3 / A.4 — the chainogram retrieval family
#             (+ FEAT-2026-06-10 Q4a semantic-arm SELECTION skip + FEAT-2026-06-11 P1
#             temporal envelope hook + the read-seam overlay).
#             + Phase-B modularization (carved from the monolith, ZERO behavior change).
# Layer:      core (pure library, no daemon dependency)
# Role:       the query→budget-packed-nodes read path; five variants over one scoring +
#             packing idiom.
# Stability:  stable — the public retrieval surface; flag-off the base scorer is
#             byte-identical to the pre-temporal baseline (§2.6).
# ErrorModel: fail-soft — a missing vector index returns {"error": ...}; chain-load
#             SystemExit/FileNotFoundError are skipped; the temporal envelope + read-seam
#             overlays are gated/additive-only.
# Depends:    .config (deps + helpers + reranker), .temporal (_apply_temporal_envelope),
#             .readseam (the failure-association overlay); lazily samia.core.semantic_recall.
# Exposes:    chainogram_retrieve{,_bridged,_hybrid,_reranked,_contextual} (public).
# Note:       PATCH SEAM — chainogram_retrieve is a facade mock.patch.object target;
#             the bridged + contextual variants reach it THROUGH the package facade so
#             a package-level patch rebinds what they run.
# Lines:      616
# --------------------------------------------------------------------------
