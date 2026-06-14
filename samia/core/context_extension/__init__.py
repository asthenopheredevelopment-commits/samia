"""samia.core.context_extension — context-extension primitives.

Carved from memory_context_extension.py. Library plane parameterized on
memory_dir; CLI wrapper does argparse + print only.

Where samia.core.bio implements per-node mechanisms (recall, edge
strengthening, retrieval gates), this module implements *context-budget*
primitives that work with — not against — production compaction.

Primitives (parameterized on memory_dir):
    chainogram_retrieve, chainogram_retrieve_bridged,
    chainogram_retrieve_hybrid, chainogram_retrieve_reranked,
    chainogram_retrieve_contextual,
    frozen_prefix_block, tier_flow_for_budget,
    episodic_to_semantic_candidates, idle_replay_tick,
    sm2_review_update, sm2_due_for_review, compaction_skip_filter

Layer 1 (Owns / Depends):
    Owns:    the context-budget read/lifecycle primitives — split by responsibility
             into seven submodules behind this re-export facade (the public import
             surface is byte-for-byte unchanged from the pre-split single module):
               - config     : the re-exported stdlib (json/os/hashlib/sqlite3/time, _dt,
                              Path) + numpy as np, the aliased dependency modules
                              (_bio/_ct/_tq/_vi/_ws + the optional _ei/_vic), EVERY tuning
                              constant, the SINGLE-OWNED state (_ATOM_CHAIN_CACHE +
                              the lazy _RERANKER), and the shared path/IO/vector + atom-
                              chain + reranker helpers — the package's shared, single-
                              owned leaf.
               - temporal   : the temporal-recall envelope (the flag/weight readers
                              temporal_weight_enabled / temporal_weights, the gates +
                              pool normalizer, the P2-P5 term hooks, and
                              _apply_temporal_envelope).
               - readseam   : the cross-chain failure/diagnosis read-seam
                              (_query_failure_associations + helpers).
               - retrieval  : the chainogram retrieval family (chainogram_retrieve +
                              _bridged / _hybrid / _reranked / _contextual).
               - primitives : Primitive B/C/D — frozen_prefix_block, tier_flow_for_budget,
                              episodic_to_semantic_candidates.
               - replay     : Primitive E — idle_replay_tick + the directed-SR producer
                              (_record_directed_transitions + helpers).
               - scheduling : the SM-2 spaced-repetition arm (sm2_review_update /
                              sm2_due_for_review / sm2_sweep_tick + caps) + the
                              compaction-aware skip filter (compaction_skip_filter).
    Depends: samia.core.{bio,chain,temporal,vector,web_store} (aliased in config),
             optionally entity_index / vector_contextual, and lazily
             samia.core.{semantic_recall,successor,temporal_recall_sith,
             temporal_recall_stc,temporal_distinctiveness,frontmatter,atomic_state} +
             samia.runtime.rem_cycle (function-local imports in the submodules to break
             the import cycles + keep the heavy/runtime deps off the import path).

Layer 2 (What / Why):
    What: the context-budget primitives that turn a query into a budget-packed node set
          (retrieval), maintain the FROZEN-tier prefix + tier-flow plan + abstraction
          proposals (primitives), run the REM-gated offline replay + directed-SR producer
          (replay), and drive the SM-2 schedule + compaction novelty gate (scheduling).
          The temporal-recall envelope folds an optional, default-off scoring lift into
          the retrieval scorer; the read-seam surfaces failure experience from the web.
    Why:  the 1959-line monolith was split by RESPONSIBILITY (config / temporal /
          readseam / retrieval / primitives / replay / scheduling) with ZERO behavior
          change; this facade re-exports the FULL public surface so every importer
          (`from samia.core.context_extension import X` / `from samia.core import
          context_extension as ce`) and every attribute reach-in (public OR private) is
          unaffected.

PATCH SEAM + TEST REACH-INS (exemplar rule):
    chainogram_retrieve is a mock.patch.object(cx, "chainogram_retrieve") target
    (test_semantic_recall_p2 mocks it on this facade to hermetically drive mcp_server's
    memory_chainogram_retrieve). The sibling variants chainogram_retrieve_bridged +
    chainogram_retrieve_contextual call it THROUGH this facade (from samia.core import
    context_extension as _pkg; _pkg.chainogram_retrieve) so a package-level patch rebinds
    what they run. mcp_server.context_tools + semantic_recall reach chainogram_retrieve
    through `from .. import context_extension as _cx` (the package), so the patch lands.
    Tests also reach the private temporal helpers (ce._relevance_gate / ce._minmax_pool /
    ce._term_active / ce._tc_term_hit / ce._need_term_chain / ce._stc_term_chain /
    ce._dist_vector / ce._dist_term_chain), the directed-SR producer
    (ce._record_directed_transitions), and the atom-chain cache clear
    (cx._clear_atom_chain_cache) — all re-exported below. No test rebinds a private name
    on the facade that a sibling then reads, and there is NO importlib.reload of this
    module anywhere, so no facade-rebound-state or reload-cascade wiring is required.

Public surface re-exported here (byte-for-byte the pre-split module — 45 names):
    re-exported imports : Path, annotations, hashlib, json, np, os, sqlite3, time
    constants           : BYTES_PER_TOKEN, DEFAULT_BUDGET_TOKENS, EPISODIC_AGE_DAYS,
                          EPISODIC_MIN_SIBLINGS, EPISODIC_SIM_THRESHOLD,
                          IDLE_THRESHOLD_SECONDS, READ_SEAM_TOP_N_DEFAULT,
                          READ_SEAM_TOP_N_ENV, SM2_SEED_EASINESS, SM2_SEED_INTERVAL_DAYS,
                          SM2_SEED_REVIEW_COUNT, SM2_SWEEP_REVIEW_CAP, SM2_SWEEP_SEED_CAP,
                          TC_COSINE_FLOOR_DEFAULT, TC_COSINE_FLOOR_ENV, TEMPORAL_GAMMA_ENV,
                          TEMPORAL_LAMBDA_D_ENV, TEMPORAL_LAMBDA_K_ENV,
                          TEMPORAL_LAMBDA_N_ENV, TEMPORAL_THETA, TEMPORAL_WEIGHT_ENV,
                          TEMPORAL_WEIGHT_EPSILON
    functions           : chainogram_retrieve, chainogram_retrieve_bridged,
                          chainogram_retrieve_contextual, chainogram_retrieve_hybrid,
                          chainogram_retrieve_reranked, compaction_skip_filter,
                          episodic_to_semantic_candidates, frozen_prefix_block,
                          idle_replay_tick, sm2_due_for_review, sm2_review_update,
                          sm2_sweep_tick, temporal_weight_enabled, temporal_weights,
                          tier_flow_for_budget
"""

from __future__ import annotations

# The shared leaf — the re-exported stdlib (json/os/hashlib/sqlite3/time/Path; `annotations`
# rides the `from __future__` above) + numpy as np, the aliased dependency modules, every
# tuning constant, the single-owned state, and the shared helpers. The public surface
# carries json/os/hashlib/sqlite3/time/Path/np/annotations, so they must stay importable
# from the package facade — they are owned in config.
from .config import (  # noqa: F401
    # re-exported imports (public surface)
    Path,
    hashlib,
    json,
    np,
    os,
    sqlite3,
    time,
    # private re-exported import (the public surface carries `np`, not `datetime`)
    _dt,
    # aliased dependency modules (re-exported for parity + direct test/importer reach)
    _bio,
    _ct,
    _tq,
    _vi,
    _ws,
    _ei,
    _vic,
    # constants
    BYTES_PER_TOKEN,
    DEFAULT_BUDGET_TOKENS,
    EPISODIC_AGE_DAYS,
    EPISODIC_MIN_SIBLINGS,
    EPISODIC_SIM_THRESHOLD,
    IDLE_THRESHOLD_SECONDS,
    READ_SEAM_TOP_N_DEFAULT,
    READ_SEAM_TOP_N_ENV,
    TC_COSINE_FLOOR_DEFAULT,
    TC_COSINE_FLOOR_ENV,
    TEMPORAL_WEIGHT_ENV,
    TEMPORAL_GAMMA_ENV,
    TEMPORAL_LAMBDA_N_ENV,
    TEMPORAL_LAMBDA_K_ENV,
    TEMPORAL_LAMBDA_D_ENV,
    TEMPORAL_THETA,
    TEMPORAL_WEIGHT_EPSILON,
    # single-owned state + helpers
    _ATOM_CHAIN_CACHE,
    _clear_atom_chain_cache,
    _is_atom_chain,
    _RERANKER,
    _RERANKER_NAME,
    _get_reranker,
    _nodes_dir,
    _chains_dir,
    _ctx_dir,
    _frozen_prefix_path,
    _idle_state_path,
    _tok_estimate,
    _node_text,
    _read_fm,
    _read_full_fm,
    _vi_manifest,
    _vi_embed,
    _vi_query,
)

# The temporal-recall envelope — the flag/weight readers (public), plus the gates, pool
# normalizer, term-hook seams, and the in-place fold (private; tests reach the hooks +
# gates through the facade as ce._relevance_gate / ce._minmax_pool / ce._term_active /
# ce._tc_term_hit / ce._need_term_chain / ce._stc_term_chain / ce._dist_vector /
# ce._dist_term_chain, and the retrieval arm reaches _apply_temporal_envelope).
from .temporal import (  # noqa: F401
    temporal_weight_enabled,
    temporal_weights,
    _temporal_weight,
    _term_active,
    _relevance_gate,
    _tc_cosine_floor,
    _minmax_pool,
    _tc_term_hit,
    _need_vector,
    _need_term_chain,
    _stc_term_chain,
    _dist_vector,
    _dist_term_chain,
    _apply_temporal_envelope,
)

# The cross-chain failure/diagnosis read-seam.
from .readseam import (  # noqa: F401
    _resolve_read_seam_top_n,
    _is_failure_or_diagnosis_node,
    _query_failure_associations,
)

# The chainogram retrieval family (chainogram_retrieve is the facade patch seam).
from .retrieval import (  # noqa: F401
    chainogram_retrieve,
    chainogram_retrieve_bridged,
    chainogram_retrieve_hybrid,
    chainogram_retrieve_reranked,
    chainogram_retrieve_contextual,
)

# Primitive B/C/D — stable-prefix anchoring, tier flow, episodic→semantic transition.
from .primitives import (  # noqa: F401
    frozen_prefix_block,
    tier_flow_for_budget,
    episodic_to_semantic_candidates,
)

# Primitive E — the idle DMN replay tick + the directed-SR producer (the directed
# helpers are re-exported for the test reach-in ce._record_directed_transitions).
from .replay import (  # noqa: F401
    idle_replay_tick,
    _record_replay_coactivations,
    _replay_pairs,
    _node_episode_seq,
    _record_directed_transitions,
)

# The SM-2 spaced-repetition arm + the compaction-aware skip filter (+ the SM2_* caps).
from .scheduling import (  # noqa: F401
    sm2_review_update,
    sm2_due_for_review,
    sm2_sweep_tick,
    compaction_skip_filter,
    _sm2_quality_from_usage,
    SM2_SEED_EASINESS,
    SM2_SEED_INTERVAL_DAYS,
    SM2_SEED_REVIEW_COUNT,
    SM2_SWEEP_SEED_CAP,
    SM2_SWEEP_REVIEW_CAP,
)

# __all__ — the LOCALLY-owned PUBLIC names (the 45 the baseline records: 8 re-exported
# imports, 22 constants, 15 functions). The verify script diffs the full public surface
# (dir() minus underscore names) against the baseline; __all__ documents the intended
# export set and bounds `from ... import *` to exactly the pre-split public 45. (The
# private test/importer/patch-seam-reached names above are re-exported but intentionally
# NOT in __all__, mirroring the exemplars.)
__all__ = [
    # re-exported imports
    "Path", "annotations", "hashlib", "json", "np", "os", "sqlite3", "time",
    # constants
    "BYTES_PER_TOKEN", "DEFAULT_BUDGET_TOKENS", "EPISODIC_AGE_DAYS",
    "EPISODIC_MIN_SIBLINGS", "EPISODIC_SIM_THRESHOLD", "IDLE_THRESHOLD_SECONDS",
    "READ_SEAM_TOP_N_DEFAULT", "READ_SEAM_TOP_N_ENV", "SM2_SEED_EASINESS",
    "SM2_SEED_INTERVAL_DAYS", "SM2_SEED_REVIEW_COUNT", "SM2_SWEEP_REVIEW_CAP",
    "SM2_SWEEP_SEED_CAP", "TC_COSINE_FLOOR_DEFAULT", "TC_COSINE_FLOOR_ENV",
    "TEMPORAL_GAMMA_ENV", "TEMPORAL_LAMBDA_D_ENV", "TEMPORAL_LAMBDA_K_ENV",
    "TEMPORAL_LAMBDA_N_ENV", "TEMPORAL_THETA", "TEMPORAL_WEIGHT_ENV",
    "TEMPORAL_WEIGHT_EPSILON",
    # functions
    "chainogram_retrieve", "chainogram_retrieve_bridged",
    "chainogram_retrieve_contextual", "chainogram_retrieve_hybrid",
    "chainogram_retrieve_reranked", "compaction_skip_filter",
    "episodic_to_semantic_candidates", "frozen_prefix_block", "idle_replay_tick",
    "sm2_due_for_review", "sm2_review_update", "sm2_sweep_tick",
    "temporal_weight_enabled", "temporal_weights", "tier_flow_for_budget",
]


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.context_extension
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      context-budget primitives (chainogram retrieval family + frozen-prefix +
#             tier flow + episodic→semantic + idle replay + SM-2 + compaction skip) +
#             FEAT-2026-06-11 temporal-recall P1-P6 envelope + the read-seam.
#             + Phase-B modularization: the 1959-line monolith carved into a
#               re-export-preserving package (config/temporal/readseam/retrieval/
#               primitives/replay/scheduling) with ZERO behavior change; this __init__
#               re-exports the full public surface so every importer + attribute reach-in
#               is unaffected.
# Layer:      core (pure library, no daemon dependency)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.core.context_extension
#             import X` / `from samia.core import context_extension as ce` keeps working
#             for all 45 public names; the private test/importer/patch-seam-reached names
#             (the temporal hooks/gates, _apply_temporal_envelope, the read-seam helpers,
#             _record_directed_transitions + the replay helpers, _clear_atom_chain_cache,
#             the aliased deps, _dt, and the shared path/IO/vector helpers) are re-exported
#             too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
# ErrorModel: none here (import-time wiring only); each submodule footer documents its own
#             fail-soft / fail-open / gated-and-inert posture. The temporal envelope is
#             default-off (flag-off path is byte-identical to the pre-temporal baseline)
#             and the replay heavy body is REM-gated.
# Depends:    .config, .temporal, .readseam, .retrieval, .primitives, .replay, .scheduling.
# Exposes:    the public 45 (in __all__) + the private/patch-seam/state names above.
# Lines:      294
# Note:       PATCH SEAM — chainogram_retrieve is a facade mock.patch.object target reached
#             by the bridged/contextual variants THROUGH this facade. No facade-rebound
#             module state + NO importlib.reload of this module exist, so no reload-cascade
#             or state-reach wiring is needed (unlike the contradiction exemplar).
# --------------------------------------------------------------------------
