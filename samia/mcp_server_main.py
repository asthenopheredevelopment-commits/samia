#!/usr/bin/env python3
"""samia.mcp_server_main — packaged stdio MCP server exposing the SAM/IA store.

Layer 1 (Owns / Depends):
    Owns:    the `samia-mcp-server` console_script entry point — a FastMCP stdio
             server named "asthenos-memory" whose @mcp.tool() functions each delegate
             to their samia.core.mcp_server sibling, plus main() (mcp.run()).
    Depends: mcp (FastMCP) + samia.core.mcp_server. Store dir resolved from the
             environment (ASTHENOS_MEMORY_DIR) → ~/.local/share/asthenos default.

Layer 2 (What / Why):
    What: a thin wrapper — the FastMCP main loop lives here; every tool forwards to
          samia.core.mcp_server with the resolved MEMORY_DIR. The server name and the
          full tool set are byte-for-byte the dev server's, so the tool-id prefix
          (mcp__asthenos-memory__memory_search, ...) keeps resolving for existing
          clients.
    Why:  the packaged public surface. The dev tree launched this by absolute path with
          a __file__-relative store dir; a pip-installed wheel cannot — so the store
          resolves from ASTHENOS_MEMORY_DIR (env) → ~/.local/share/asthenos (the dir
          `samia init` creates). One canonical core, many thin front-ends.

Register with an MCP client via:  claude mcp add asthenos-memory -- samia-mcp-server
(or the plugin's bundled .mcp.json). The store path travels in the launcher's env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# MemoryDir — What: the resolved store root the tools operate on.
# MemoryDir — Why: a public package must NOT infer the store from its own install
#             location (the dev tree's `Path(__file__).parent.parent` trick). Env
#             override first (Docker/sandbox/test redirect), then the user-data-dir
#             default that `samia init` creates. See packaging NOTES.md.
MEMORY_DIR = Path(
    os.environ.get("ASTHENOS_MEMORY_DIR", Path.home() / ".local" / "share" / "asthenos")
).expanduser()

from samia.core import mcp_server as _mcps  # noqa: E402

mcp = FastMCP("asthenos-memory")


# ---------------------------------------------------------------------------
# Search & retrieval
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_search(query: str, top_k: int = 8,
                   record_coactivation: bool = True,
                   include_ring: bool = True,
                   include_inject: bool = False) -> list[dict[str, Any]] | dict[str, Any]:
    """Semantic top-k search over all SAM/IA memory nodes.

    Returns a ranked list of {score, node, title} dicts.

    record_coactivation: when True (default), the set of returned nodes is
    logged for Hebbian "fire together, wire together" reinforcement —
    pairs retrieved together drift toward stronger chain edges over time.

    include_ring: when True (default), folds in the Tier-1 ring-RAG fast-tier
    arm (the volatile working-set pointers, deref'd at query time, tagged
    via="ring"). Additive + fail-open; set False to query main+engram only.

    include_inject: when True (default OFF), ALSO returns the Tier-1 P4
    standing-availability inject block — changing the return shape to
    {hits, inject_block}. The inject block is assembled CO-ACTIVATION-SILENTLY
    (no Tier-0 feed) and is the assembler surface only; the live per-turn prompt
    injection stays operator-gated/INERT.
    """
    return _mcps.memory_search(MEMORY_DIR, query, top_k=top_k,
                                record_coactivation=record_coactivation,
                                include_ring=include_ring,
                                include_inject=include_inject)


@mcp.tool()
def memory_temporal_query(
    at: str | None = None,
    since: str | None = None,
    range_from: str | None = None,
    range_to: str | None = None,
    semantic: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Bi-temporal query — filter nodes by event-time validity.

    at:          ISO date — facts true on this date
    since:       ISO date — facts still valid at or after this date
    range_from:  ISO date — start of overlap range
    range_to:    ISO date — end of overlap range
    semantic:    optional text — pre-filter by top-k semantic match
    top_k:       result limit
    """
    return _mcps.memory_temporal_query(MEMORY_DIR, at=at, since=since,
                                         range_from=range_from,
                                         range_to=range_to,
                                         semantic=semantic, top_k=top_k)


@mcp.tool()
def memory_read_node(name: str) -> dict[str, Any]:
    """Fetch a full memory node by filename (with or without .md)."""
    return _mcps.memory_read_node(MEMORY_DIR, name)


@mcp.tool()
def memory_write_node(
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
    salience_tag: bool = False,
) -> dict[str, Any]:
    """Create or update a memory node, optionally splitting into atomic facts.

    name:         filename stem (no .md suffix needed); used as prefix when
                  extract=True, otherwise the single file written
    title:        human-readable name (frontmatter `name:`)
    description:  one-line description
    body:         markdown body — if extract=True, this is the source blob
                  decomposed into atoms
    type_:        user / feedback / project / reference
    chains:       list of chain names this node belongs to
    valid_from:   ISO event-time start (defaults to today)
    valid_to:     ISO event-time end, or None for "still valid"
    extract:      run LLM/rule extractor on body and write one node per atom
                  (overrides the single-write path; title/description/type
                  on this call are ignored — the extractor produces them)
    extractor_backend: auto | rule | anthropic — see memory_fact_extractor
    salience_tag: when True, the explicit operator/agent "this matters" override
                  — clamps the node's [0,1] salience HIGH regardless of the
                  composite signals (Tier-1 P2 / D6 Q8a). Capture also registers
                  a Tier-1 ring POINTER carrying the salience flag.
    """
    return _mcps.memory_write_node(MEMORY_DIR, name, title, description, body,
                                     type_=type_, chains=chains,
                                     valid_from=valid_from, valid_to=valid_to,
                                     extract=extract,
                                     extractor_backend=extractor_backend,
                                     salience_tag=salience_tag)


@mcp.tool()
def memory_tag_salient(node: str, value: bool = True) -> dict[str, Any]:
    """Set (or clear) the EXPLICIT operator/agent salience override on a node
    (Tier-1 P2 / D6 Q8a). value=True is the deliberate "this matters" override —
    clamps the node's [0,1] `salience` frontmatter HIGH regardless of the composite
    signals (surprise / contradiction-involvement / repetition); value=False clears
    it so salience falls back to the composite. Returns {node, salience, salience_tag}."""
    return _mcps.memory_tag_salient(MEMORY_DIR, node, value=value)


@mcp.tool()
def memory_inject_block(query: str, token_budget: int = 600,
                        engram_budget_frac: float = 0.4) -> dict[str, Any]:
    """Assemble the Tier-1 P4 standing-availability inject block (D4 / Q5a).

    Two layers under a FIXED token budget, relevance-gated against `query`:
    ENGRAM-INJECT — the always-on inject_eligible identity / known-cold set
    (favored under pressure, reserved engram_budget_frac of the budget);
    RING-INJECT — the live ring working-set pointers, deref'd to backing content
    (fills the remainder by relevance). When over budget, engram-inject wins and
    ring overflow is dropped; the block NEVER exceeds the cap. CO-ACTIVATION-
    SILENT (no Tier-0 Hebbian feed). Returns {items, tokens_used, token_budget,
    engram_count, ring_count, dropped, co_activation_silent}. This is the
    ASSEMBLER only — the live per-turn prompt injection stays operator-gated/INERT.
    """
    return _mcps.memory_inject_block(MEMORY_DIR, query,
                                       token_budget=token_budget,
                                       engram_budget_frac=engram_budget_frac)


@mcp.tool()
def memory_extract_facts(
    text: str,
    backend: str = "auto",
    chains_hint: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Preview atomic-fact decomposition without writing anything.

    backend: auto (LLM if ANTHROPIC_API_KEY set, else rule), rule, anthropic
    """
    return _mcps.memory_extract_facts(MEMORY_DIR, text, backend=backend,
                                        chains_hint=chains_hint)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_list_chains() -> list[dict[str, Any]]:
    """List all SAM chains and their head node names."""
    return _mcps.memory_list_chains(MEMORY_DIR)


@mcp.tool()
def memory_get_chain(name: str) -> dict[str, Any]:
    """Fetch one chain manifest with node titles."""
    return _mcps.memory_get_chain(MEMORY_DIR, name)


# ---------------------------------------------------------------------------
# Edge-temporal chain queries
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_chain_query_at(chain: str, at: str) -> list[dict[str, Any]]:
    """Members of a chain reachable via edges valid on date `at` (ISO)."""
    return _mcps.memory_chain_query_at(MEMORY_DIR, chain, at)


@mcp.tool()
def memory_chain_traverse_at(chain: str, start: str, at: str, depth: int = 3) -> list[dict[str, Any]]:
    """BFS along chain edges valid on `at`, starting at member address `start`."""
    return _mcps.memory_chain_traverse_at(MEMORY_DIR, chain, start, at, depth=depth)


@mcp.tool()
def memory_chain_set_edge(
    chain: str,
    from_addr: str,
    to_addr: str,
    valid_from: str | None = None,
    valid_to: str | None = None,
    label: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Modify (or create) the latest matching edge on a chain.

    label: e.g., follows, supersedes, depends_on, contradicts, refines
    valid_to=None keeps the edge open; pass an ISO date to close it.
    """
    return _mcps.memory_chain_set_edge(MEMORY_DIR, chain, from_addr, to_addr,
                                         valid_from=valid_from,
                                         valid_to=valid_to, label=label,
                                         confidence=confidence)


@mcp.tool()
def memory_chain_invalidate_edge(chain: str, from_addr: str, to_addr: str,
                                  on: str, label: str | None = None) -> dict[str, Any]:
    """Close every still-open matching edge by setting valid_to=on."""
    return _mcps.memory_chain_invalidate_edge(MEMORY_DIR, chain, from_addr,
                                                to_addr, on, label=label)


@mcp.tool()
def memory_chain_snapshot_at(at: str) -> dict[str, Any]:
    """Per-chain count of edges valid on date `at`."""
    return _mcps.memory_chain_snapshot_at(MEMORY_DIR, at)


# ---------------------------------------------------------------------------
# Biomimetic primitives
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_pattern_separate(text: str, threshold: float = 0.85) -> dict[str, Any]:
    """Decide whether `text` is novel enough to store as a new node, or
    overlaps an existing node enough to merge in. Returns {action, target,
    score, neighbors}. Maps to dentate gyrus pattern separation /
    CA3 pattern completion."""
    return _mcps.memory_pattern_separate(MEMORY_DIR, text, threshold=threshold)


@mcp.tool()
def memory_hebbian_consolidate() -> dict[str, Any]:
    """Fold the co-activation log into edge weights, decay stale ones,
    and promote pairs that crossed the threshold into chain edges with
    label `hebbian`. Run periodically (e.g., session end)."""
    return _mcps.memory_hebbian_consolidate(MEMORY_DIR)


@mcp.tool()
def memory_replay_sweep(sample: int = 20, threshold: float = 0.55) -> dict[str, Any]:
    """Hippocampal-replay analog: pick recently-accessed nodes, find
    semantic neighbors via the vector index, and propose new cross-chain
    edges. Output is suggestive — written to biomimetic/replay_proposals.json
    for review, not auto-applied."""
    return _mcps.memory_replay_sweep(MEMORY_DIR, sample=sample, threshold=threshold)


@mcp.tool()
def memory_reconsolidate(node: str, new_context: str,
                          backend: str = "auto") -> dict[str, Any]:
    """When a node is retrieved in a new context, re-extract atoms and
    update the node body / spawn refining sibling nodes. Maps to Nader
    et al. 2000 reconsolidation. Use sparingly — every call is a write."""
    return _mcps.memory_reconsolidate(MEMORY_DIR, node, new_context, backend=backend)


@mcp.tool()
def memory_schema_check(text: str, chains: list[str]) -> dict[str, Any]:
    """For a candidate new node, check whether its chains are mature
    enough to fast-track ingestion (skip cold start, start at hot tier).
    Maps to Tse et al. 2007 schema-accelerated learning."""
    return _mcps.memory_schema_check(MEMORY_DIR, text, chains)


@mcp.tool()
def memory_chain_maturity(chain: str) -> dict[str, Any]:
    """Whether a chain is mature enough to be a schema (>=4 nodes,
    oldest valid_from > 7 days)."""
    return _mcps.memory_chain_maturity(MEMORY_DIR, chain)


# ---------------------------------------------------------------------------
# Epiphanies associative-edge veto net — operator correction surface. LEDGER-only,
# fail-open; the lever to veto a false association before/after it promotes to a chain.
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_epiphanies_reject_binding(node_a: str, node_b: str, reason: str = "") -> dict[str, Any]:
    """Mark an Epiphanies associative binding (node_a, node_b) a FALSE association. Suppresses it
    from the next consolidate (demoted, never promoted to a chain) UNLESS it keeps recurring (a
    K-recurrence override un-suppresses it). LEDGER-only — no live edge written, no node forgotten."""
    return _mcps.memory_epiphanies_reject_binding(MEMORY_DIR, node_a, node_b, reason)


@mcp.tool()
def memory_epiphanies_unreject_binding(node_a: str, node_b: str) -> dict[str, Any]:
    """Clear an Epiphanies binding suppression (operator changed their mind). Returns {cleared: bool}."""
    return _mcps.memory_epiphanies_unreject_binding(MEMORY_DIR, node_a, node_b)


@mcp.tool()
def memory_epiphanies_reject_candidate(node_a: str, node_b: str) -> dict[str, Any]:
    """Veto a linker CANDIDATE (proposed-but-unvalidated association) as not-related. A K-remint
    override can still resurface it if it keeps genuinely co-occurring."""
    return _mcps.memory_epiphanies_reject_candidate(MEMORY_DIR, node_a, node_b)


@mcp.tool()
def memory_epiphanies_list_suppressions() -> dict[str, Any]:
    """List the active Epiphanies binding suppressions (operator visibility)."""
    return _mcps.memory_epiphanies_list_suppressions(MEMORY_DIR)


@mcp.tool()
def memory_epiphanies_list_candidates(state: str = "") -> dict[str, Any]:
    """List the linker's association candidates (optional state filter, e.g. 'validated'/'genuine').
    Operator visibility into the hypotheses; mutates nothing."""
    return _mcps.memory_epiphanies_list_candidates(MEMORY_DIR, state)


# ---------------------------------------------------------------------------
# Forgetting / negative consolidation -- FEAT-2026-06-07 P0
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_forget_node(node: str, reason: str = "manual") -> dict[str, Any]:
    """Cross-tier invalidation cascade: purge a dead/superseded node's edges
    from edges.db (all ref_kinds) + edge_weights.json, strip its chain
    membership + hebbian edges, tombstone its vector entry, append a
    forgotten-log entry. The node FILE is expected already gone. Idempotent."""
    return _mcps.memory_forget_node(MEMORY_DIR, node, reason=reason)


@mcp.tool()
def memory_supersession_candidates() -> dict[str, Any]:
    """List un-resolved supersession candidates from the unified store
    (old_id, new_id, cosine, jaccard, mode). The online write seam auto-supersedes
    the obvious exact case (restorably) and records WEAKER hits here for the passive
    LLM judge / operator review. Nothing here is deleted until acted."""
    return _mcps.memory_supersession_candidates(MEMORY_DIR)


@mcp.tool()
def memory_confirm_supersession(old_id: str, valid_to: str | None = None,
                                new_id: str | None = None) -> dict[str, Any]:
    """Confirm a supersession → RESTORABLE retire of the OLD node: set valid_to
    (provenance-preserving), then fire the archiving forget cascade
    (reason="supersede" full-archives the node so memory_restore_node can un-forget
    it byte-exact) and mark the candidate confirmed. Auto-supersede made safe by
    reversibility (Q4 override)."""
    return _mcps.memory_confirm_supersession(MEMORY_DIR, old_id,
                                             valid_to=valid_to, new_id=new_id)


@mcp.tool()
def memory_dismiss_supersession(old_id: str,
                                new_id: str | None = None) -> dict[str, Any]:
    """Dismiss a supersession candidate (false positive) in the unified store:
    marks it dismissed so it stops surfacing. Deletes nothing. If it named an
    already-auto-superseded node, call memory_restore_node to un-forget it."""
    return _mcps.memory_dismiss_supersession(MEMORY_DIR, old_id, new_id=new_id)


@mcp.tool()
def memory_restore_node(node_id: str) -> dict[str, Any]:
    """Un-forget a superseded node from its archive: re-create nodes/<id>.md
    byte-exact from archive/<id>.superseded.json, un-tombstone its vector entry,
    log a restore event. The operator/self-healing reversibility surface for an
    online auto-supersede or a confirmed supersession that turned out wrong."""
    return _mcps.memory_restore_node(MEMORY_DIR, node_id)


# ---------------------------------------------------------------------------
# Tier-2 merge consumer P2 — gated abstraction confirm/reject (operator-only)
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_merge_candidates() -> dict[str, Any]:
    """List un-resolved Tier-2 merge/abstraction candidates: the distinct-but-
    overlapping pairs the REM consumer queued ('pending') or SYNTHESIZED a draft
    abstraction for ('proposed', carrying title+body + merged_from). Abstractions
    are operator-gated — nothing is applied until memory_confirm_merge is called."""
    return _mcps.memory_merge_candidates(MEMORY_DIR)


@mcp.tool()
def memory_confirm_merge(candidate_id: str) -> dict[str, Any]:
    """Confirm a PROPOSED abstraction → create the new semantic node from the
    synthesized content (with merged_from provenance), then SUPERSEDE both source
    nodes RESTORABLY (reason="supersede" full-archives each so memory_restore_node
    can un-forget them byte-exact) and lay provenance edges abstraction->source.
    The operator-only gate (Q2c): abstractions are NEVER auto-applied."""
    return _mcps.memory_confirm_merge(MEMORY_DIR, candidate_id)


@mcp.tool()
def memory_reject_merge(candidate_id: str) -> dict[str, Any]:
    """Reject a proposed abstraction — marks it rejected so it stops surfacing.
    Changes NOTHING: no node created, no source superseded, both originals stay
    live. The gate's reject arm (mirrors memory_dismiss_supersession)."""
    return _mcps.memory_reject_merge(MEMORY_DIR, candidate_id)


# ---------------------------------------------------------------------------
# Context-extension primitives (SAM/IA × compaction hybrids)
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_chainogram_retrieve(query: str, budget_tokens: int = 8000,
                            max_chains: int = 8,
                            include_failure_associations: bool = False,
                            failure_top_n: int | None = None) -> dict[str, Any]:
    """Sparse chainogram-style retrieval bounded by a token budget.
    Returns the top chains (not flat nodes) packed under budget.
    Replaces "load everything" with "load only what's likely relevant."

    include_failure_associations: when True, appends a
      "failure_associations" key with cross-chain Hebbian neighbors that
      are failure-outcome or bug-diagnosis nodes, ranked by weight x
      recency. Surfaces prior failure experience during diagnosis.
    failure_top_n: cap on failure associations returned (default from
      ASTHENOS_READ_SEAM_TOP_N env var, else 5).
    """
    return _mcps.memory_chainogram_retrieve(MEMORY_DIR, query,
                                              budget_tokens=budget_tokens,
                                              max_chains=max_chains,
                                              include_failure_associations=include_failure_associations,
                                              failure_top_n=failure_top_n)


@mcp.tool()
def memory_frozen_prefix(write: bool = True) -> dict[str, Any]:
    """Serialize FROZEN-tier nodes into a deterministic cache-stable
    block for the front of context — exploits Anthropic prefix-cache
    (5-min TTL). Compaction would invalidate this; tier-frozen anchoring
    preserves it."""
    return _mcps.memory_frozen_prefix(MEMORY_DIR, write=write)


@mcp.tool()
def memory_tier_flow_for_budget(query: str, budget_tokens: int = 8000,
                                  apply: bool = False) -> dict[str, Any]:
    """Per-turn HOT recomputation under a budget. dry_run by default;
    set apply=True to actually rewrite tier= frontmatter."""
    return _mcps.memory_tier_flow_for_budget(MEMORY_DIR, query,
                                               budget_tokens=budget_tokens,
                                               apply=apply)


@mcp.tool()
def memory_episodic_candidates() -> dict[str, Any]:
    """Find groups of old episodic nodes ripe to collapse into one
    semantic node (two-resolution memory; episodic kept in COLD)."""
    return _mcps.memory_episodic_candidates(MEMORY_DIR)


@mcp.tool()
def memory_idle_tick(force: bool = False) -> dict[str, Any]:
    """DMN-replay tick: run replay sweep + Hebbian consolidate +
    frozen-prefix refresh if the system has been idle long enough."""
    return _mcps.memory_idle_tick(MEMORY_DIR, force=force)


@mcp.tool()
def memory_sm2_update(node: str, missed: bool = False, quality: int = 4) -> dict[str, Any]:
    """Record an SM-2 review event on a node. Updates next_review,
    review_interval_days, easiness_factor, review_count. Schedule is
    consumed by tier_flow as a promotion-priority signal."""
    return _mcps.memory_sm2_update(MEMORY_DIR, node, missed=missed, quality=quality)


@mcp.tool()
def memory_sm2_due() -> list[dict[str, Any]]:
    """List nodes whose next_review date is on or before today."""
    return _mcps.memory_sm2_due(MEMORY_DIR)


@mcp.tool()
def memory_compaction_skip_filter(transcript_chunks: list[str],
                                     threshold: float = 0.78) -> dict[str, Any]:
    """Given conversation chunks production compaction is about to
    summarize, return which chunks are already covered by existing
    memory nodes (skip — redundant) vs which are novel (summarize).
    Bridges /compact and SAM/IA: compact only what isn't already known."""
    return _mcps.memory_compaction_skip_filter(MEMORY_DIR, transcript_chunks,
                                                 threshold=threshold)


# ---------------------------------------------------------------------------
# Index status
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_index_status() -> dict[str, Any]:
    """Return vector-index manifest summary."""
    return _mcps.memory_index_status(MEMORY_DIR)


def main() -> None:
    """Entry point for the `samia-mcp-server` console_script — run the stdio server."""
    mcp.run()


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# [Asthenosphere] samia.mcp_server_main
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      P1 — canonical core (install-UX: the packaged samia-mcp-server entry)
# Layer:      runtime front-end (stdio MCP server; thin wrapper over the core plane)
# Role:       backs the `samia-mcp-server` console_script — a FastMCP "asthenos-memory"
#             stdio server whose tools delegate to samia.core.mcp_server with the
#             env-resolved store dir; the public, portable replacement for the dev
#             tree's absolute-path memory_mcp_server.py launch.
# Stability:  stable — tool set + server name frozen to keep the mcp__asthenos-memory__*
#             tool ids resolving for existing clients.
# ErrorModel: tool bodies fail-loud through to the MCP client (the core delegates raise);
#             store-dir resolution is total (env → default, never raises).
# Depends:    mcp.server.fastmcp.FastMCP; samia.core.mcp_server (the delegate surface).
# Exposes:    mcp (the FastMCP app), the memory_* @mcp.tool() wrappers, main().
# --------------------------------------------------------------------------
