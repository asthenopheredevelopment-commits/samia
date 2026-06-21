"""samia.core.mcp_server — data-handling primitives for the MCP server.

Carved from memory_mcp_server.py. Each public function corresponds to one MCP
tool's underlying logic, parameterized on memory_dir. The MCP wrapper itself
(with FastMCP decorators and the stdio main loop) stays in memory_mcp_server.py —
these functions provide the work behind each @mcp.tool().

Layer 1 (Owns / Depends):
    Owns:    the work behind every MCP tool, split by responsibility into five
             submodules behind this re-export facade (the public import surface is
             byte-for-byte unchanged from the pre-split single module):
               - config       : the re-exported stdlib (_dt/json/os/sqlite3/Path/Any)
                                + the aliased web_store (_ws) + the four public COACT_*
                                co-activation tunables + the _nodes_dir/_chains_dir path
                                primitives — the package's shared, single-owned leaf.
               - search       : the recall/retrieval arm (filters, re-rankers, the
                                Tier-0 co-activation read-back, the Tier-1 engram/ring
                                fast tiers, the inject assembler, memory_search /
                                temporal_query / read_node).
               - write        : the write/capture/forget/supersession arm (the genuine
                                write seam + its capture hook, the ONLINE auto-supersede
                                seam, and the confirm/dismiss/restore surface).
               - chains       : the chain listing + edge-temporal query arm.
               - bio_tools    : the biomimetic primitives + the gated merge-consumer
                                abstraction surface.
               - context_tools: the context-extension tools + the index/REM status reads.
    Depends: samia.core.{web_store,vector,bio,hippocampus,integrity,temporal,chain,
             frontmatter,fact_extractor,temporal_substrate,merge_consumer,
             context_extension,semantic_recall} and samia.runtime.{contradiction,
             rem_cycle} — almost all reached via lazy, function-local imports in the
             submodules to keep the heavy/runtime deps off the package import path and
             to avoid the bio/hippocampus<->mcp_server import cycle.

Layer 2 (What / Why):
    What: the plain-function backend each FastMCP @mcp.tool() wraps. Reads fold term/
          cosine/engram/ring hits + co-activation neighbors; writes capture an
          integrity anchor + ring pointer + salience and run a gated online auto-
          supersede; the chain/bio/context tools delegate to their owning subsystem.
    Why:  the 1315-line monolith was split by RESPONSIBILITY (search / write / chains /
          bio_tools / context_tools) with ZERO behavior change so each tool cluster is
          independently legible; this facade re-exports the FULL public surface so every
          importer (`from samia.core.mcp_server import X`) and every attribute reach-in
          (`mcp_server._coactivation_neighbors`, the `mock.patch.object(mcp_server,
          "_node_subject"/"_online_supersede"/"_register_ring_and_salience", ...)`
          targets) is unaffected.

TWO PATCH SEAMS (exemplar rule): three private helpers are BOTH mock.patch.object
    (mcp_server, ...) targets AND called by a sibling submodule, so the callers reach
    them THROUGH this package facade so a package-level patch rebinds the attribute the
    caller actually reads:
      - memory_write_node (write.py) reaches _register_ring_and_salience AND
        _online_supersede through the facade (test_integrity_p2 patches both);
      - _online_supersede (write.py) reaches _node_subject through the facade
        (test_merge_consumer_p3 patches it).

Public surface re-exported here (byte-for-byte the pre-split module — 49 names):
    re-exported imports : Any, Path, annotations, json, os, sqlite3
    constants           : COACT_LAMBDA, COACT_MAX_NEIGHBORS, COACT_PARENT_HITS,
                          COACT_DELTA
    functions           : memory_inject_block, memory_search, memory_temporal_query,
                          memory_read_node, memory_write_node, memory_tag_salient,
                          memory_extract_facts, memory_list_chains, memory_get_chain,
                          memory_chain_query_at, memory_chain_traverse_at,
                          memory_chain_set_edge, memory_chain_invalidate_edge,
                          memory_chain_snapshot_at, memory_pattern_separate,
                          memory_hebbian_consolidate, memory_replay_sweep,
                          memory_reconsolidate, memory_schema_check,
                          memory_chain_maturity, memory_forget_node,
                          memory_supersession_candidates, memory_confirm_supersession,
                          memory_dismiss_supersession, memory_restore_node,
                          memory_merge_candidates, memory_confirm_merge,
                          memory_reject_merge, memory_chainogram_retrieve,
                          memory_frozen_prefix, memory_tier_flow_for_budget,
                          memory_episodic_candidates, memory_idle_tick,
                          memory_sm2_update, memory_sm2_due,
                          memory_compaction_skip_filter, memory_index_status,
                          memory_rem_status, memory_rem_sleep_now
Internal names also re-exported for direct test/importer access (NOT in __all__):
    _coactivation_neighbors (mcp._coactivation_neighbors), _node_subject /
    _online_supersede / _register_ring_and_salience (the three patch-seam targets),
    _salience_guards_supersede, plus the remaining module-level privates the monolith
    exposed (_dt, _ws, _nodes_dir, _chains_dir, _filter_by_runtime, _rerank_hits,
    _term_index_lookup, _engram_rag_hits, _ring_rag_hits).
"""

from __future__ import annotations

# The shared leaf — the re-exported stdlib (Any/Path/json/os/sqlite3, _dt), the aliased
# web_store dependency (_ws), the four public COACT_* tunables, and the path primitives.
# `annotations` rides the `from __future__` above. These are part of the public surface
# (json/os/sqlite3/Any/Path/annotations + the COACT_* constants), so they must stay
# importable from the package facade.
from .config import (  # noqa: F401
    Any,
    Path,
    json,
    os,
    sqlite3,
    _dt,
    _ws,
    COACT_LAMBDA,
    COACT_MAX_NEIGHBORS,
    COACT_PARENT_HITS,
    COACT_DELTA,
    _nodes_dir,
    _chains_dir,
)

# The recall / retrieval arm. The leading-underscore names are re-exported because the
# tests reach them through the module namespace (mcp._coactivation_neighbors) and to
# keep every module-level private the monolith exposed importable.
from .search import (  # noqa: F401
    memory_inject_block,
    memory_search,
    memory_temporal_query,
    memory_read_node,
    _filter_by_runtime,
    _rerank_hits,
    _term_index_lookup,
    _coactivation_neighbors,
    _engram_rag_hits,
    _ring_rag_hits,
)

# The write / capture / forget / supersession arm. _register_ring_and_salience /
# _online_supersede / _node_subject are the THREE patch-seam targets (re-exported so the
# package-level mock.patch.object rebinds the attribute the sibling callers read through
# this facade); _salience_guards_supersede is re-exported for parity.
from .write import (  # noqa: F401
    memory_write_node,
    memory_tag_salient,
    memory_extract_facts,
    memory_forget_node,
    memory_supersession_candidates,
    memory_confirm_supersession,
    memory_dismiss_supersession,
    memory_restore_node,
    _register_ring_and_salience,
    _node_subject,
    _salience_guards_supersede,
    _online_supersede,
)

# The chain listing + edge-temporal query arm.
from .chains import (  # noqa: F401
    memory_list_chains,
    memory_get_chain,
    memory_chain_query_at,
    memory_chain_traverse_at,
    memory_chain_set_edge,
    memory_chain_invalidate_edge,
    memory_chain_snapshot_at,
)

# The biomimetic primitives + the gated merge-consumer abstraction surface.
from .bio_tools import (  # noqa: F401
    memory_pattern_separate,
    memory_hebbian_consolidate,
    memory_replay_sweep,
    memory_reconsolidate,
    memory_schema_check,
    memory_chain_maturity,
    memory_merge_candidates,
    memory_confirm_merge,
    memory_reject_merge,
    memory_epiphanies_reject_binding,
    memory_epiphanies_unreject_binding,
    memory_epiphanies_reject_candidate,
    memory_epiphanies_list_suppressions,
    memory_epiphanies_list_candidates,
)

# The context-extension tools + the index/REM status reads.
from .context_tools import (  # noqa: F401
    memory_chainogram_retrieve,
    memory_frozen_prefix,
    memory_tier_flow_for_budget,
    memory_episodic_candidates,
    memory_idle_tick,
    memory_sm2_update,
    memory_sm2_due,
    memory_compaction_skip_filter,
    memory_index_status,
    memory_rem_status,
    memory_rem_sleep_now,
)

# __all__ — the LOCALLY-owned PUBLIC names (the 49 the baseline records: the 6 re-
# exported imports, the 4 COACT_* constants, and the 39 functions). The verify script
# diffs the full public surface (dir() minus underscore names) against the baseline;
# __all__ documents the intended export set and bounds `from ... import *` to exactly the
# pre-split public 49. (The private test/importer-reached names above are re-exported but
# intentionally NOT in __all__, mirroring the exemplars.)
__all__ = [
    # re-exported imports
    "Any", "Path", "annotations", "json", "os", "sqlite3",
    # constants
    "COACT_LAMBDA", "COACT_MAX_NEIGHBORS", "COACT_PARENT_HITS", "COACT_DELTA",
    # functions
    "memory_inject_block", "memory_search", "memory_temporal_query",
    "memory_read_node", "memory_write_node", "memory_tag_salient",
    "memory_extract_facts", "memory_list_chains", "memory_get_chain",
    "memory_chain_query_at", "memory_chain_traverse_at", "memory_chain_set_edge",
    "memory_chain_invalidate_edge", "memory_chain_snapshot_at",
    "memory_pattern_separate", "memory_hebbian_consolidate", "memory_replay_sweep",
    "memory_reconsolidate", "memory_schema_check", "memory_chain_maturity",
    "memory_forget_node", "memory_supersession_candidates",
    "memory_confirm_supersession", "memory_dismiss_supersession",
    "memory_restore_node", "memory_merge_candidates", "memory_confirm_merge",
    "memory_reject_merge",
    "memory_epiphanies_reject_binding", "memory_epiphanies_unreject_binding",
    "memory_epiphanies_reject_candidate", "memory_epiphanies_list_suppressions",
    "memory_epiphanies_list_candidates",
    "memory_chainogram_retrieve", "memory_frozen_prefix",
    "memory_tier_flow_for_budget", "memory_episodic_candidates", "memory_idle_tick",
    "memory_sm2_update", "memory_sm2_due", "memory_compaction_skip_filter",
    "memory_index_status", "memory_rem_status", "memory_rem_sleep_now",
]


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.mcp_server
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      data-handling primitives for the MCP server (carved from
#             memory_mcp_server.py)
#             + Phase-B modularization: the 1315-line monolith carved into a
#               re-export-preserving package (config/search/write/chains/bio_tools/
#               context_tools) with ZERO behavior change; this __init__ re-exports the
#               full public surface so every importer + attribute reach-in is unaffected.
# Layer:      core (pure library, no daemon dependency)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.core.mcp_server import X`
#             keeps working for all 49 public names; the private helpers the targeted
#             tests reach (_coactivation_neighbors + the three patch-seam targets
#             _node_subject / _online_supersede / _register_ring_and_salience) and the
#             remaining module-level privates are re-exported too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
# ErrorModel: none here (import-time wiring only); each submodule footer documents its
#             own fail-open / fail-soft / gated posture.
# Depends:    .config, .search, .write, .chains, .bio_tools, .context_tools.
# Exposes:    the public 49 (in __all__) + _coactivation_neighbors / _node_subject /
#             _online_supersede / _register_ring_and_salience / _salience_guards_supersede
#             + the module-level privates (_dt/_ws/_nodes_dir/_chains_dir/_filter_by_
#             runtime/_rerank_hits/_term_index_lookup/_engram_rag_hits/_ring_rag_hits).
# Note:       TWO PATCH SEAMS — memory_write_node reaches _register_ring_and_salience +
#             _online_supersede through this facade; _online_supersede reaches
#             _node_subject through this facade (see module docstring). The online auto-
#             supersede write seam is GATED OFF by default (ASTHENOS_CONTRADICTION_ENABLED).
# Lines:      240
# --------------------------------------------------------------------------
