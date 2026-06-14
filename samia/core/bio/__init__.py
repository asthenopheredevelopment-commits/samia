"""samia.core.bio — biomimetic memory primitives for SAM/IA.

Layer 1 (Owns / Depends):
    Owns:    the package re-export facade — pattern_separation_decision,
             hebbian_record, hebbian_consolidate, reconsolidate,
             replay_sweep(_interleaved), active_set, compute_salience,
             schema_accelerate, chain_maturity (+ the tuning constants).
    Depends: samia.core.chain; lazily samia.core.{temporal, vector,
             fact_extractor, frontmatter, hippocampus} (imported inside the
             functions that need them, to break the import cycle).
Layer 2 (What / Why):
    What: re-export facade for the biomimetic package — five
          neuroscience-grounded memory mechanisms (pattern separation, Hebbian,
          reconsolidation, replay, schema acceleration).
    Why:  carved from the 1756-line memory_biomimetic.py monolith by
          responsibility with ZERO behavior change; this __init__ preserves the
          entire public + reach-in surface so no importer breaks.

Carved from memory_biomimetic.py. Per design doc §1.1, the daemon's
biomimetic background jobs (Hebbian decay, replay sweep, reconsolidation
on retrieval) call these primitives directly.

Five mechanisms grounded in empirical neuroscience:

  1. pattern_separation     — Marr 1971 / Yassa & Stark 2011
                              cosine-threshold gate at write time.
  2. hebbian                — Hebb 1949 / Bliss & Lomo 1973
                              co-activation log → EMA edge weights;
                              promote strong pairs to chain edges.
  3. reconsolidate          — Nader et al. 2000
                              recall is a write opportunity.
  4. replay_sweep           — Wilson & McNaughton 1994
                              hippocampal replay (and SWR-interleaved variant).
  5. schema_accelerate      — Tse et al. 2007
                              new node entering a mature chain skips cold start.

Public API (parameterized on memory_dir):
  pattern_separation_decision(memory_dir, text, threshold)
  hebbian_record(memory_dir, retrieved_nodes, query)
  hebbian_consolidate(memory_dir, promote)
  reconsolidate(memory_dir, node_name, new_context, backend)
  replay_sweep(memory_dir, sample, threshold)
  replay_sweep_interleaved(memory_dir, sample, cold_per_hot, threshold, seed)
  schema_accelerate(memory_dir, text, chains)
  chain_maturity(memory_dir, chain_name)

Acceptance: byte-identical to pre-refactor memory_biomimetic.py CLI output
on the same memory tree (design doc §8.1).

Note: bio depends on `samia.core.chain`, plus `samia.core.{temporal,
vector,fact_extractor}`. Those are lazy-imported inside the functions that
need them (not at module top) to avoid an import cycle: mcp_server imports
bio, and vector/temporal pull in heavier deps; lazy keeps `import bio`
cheap. (GATE6: replaced the legacy `_tools_module()` tools/-dir reachback —
the staged release does not ship the tools/ shims.)

Phase-B modularization (this package):
    The 1756-line monolith was carved by RESPONSIBILITY into a re-export-preserving
    package, with ZERO behavior change; this __init__ re-exports the FULL public surface
    so every importer (`from samia.core import bio` / `from samia.core.bio import X`) and
    every attribute reach-in (public OR private, including every mock.patch.object(bio, ...)
    / mock.patch("samia.core.bio.<name>") target) is unaffected. Submodules:
      - config        : the re-exported stdlib + numpy (Optional/Path/annotations/hashlib/
                        json/np/os/sys + the _dt/_time aliases), the aliased _chain
                        dependency, EVERY tuning constant, the single-owned projection
                        cache (_KWTA_PROJ_CACHE), and the shared _bio_paths file resolver —
                        the package's shared, single-owned leaf.
      - pattern       : pattern separation — the cosine DEDUP gate (_node_embedding,
                        pattern_separation_decision) + the ORTHOGONALIZING kWTA sparse code
                        (_kwta_projection, kwta_sparse_code).
      - hebbian       : the Hebbian arm — hebbian_record, the edge_weights read/write +
                        genuine-attractor accounting, forget/ghost-edge cleanup, the
                        atomic-drain + cadence machinery, the homeostatic update + daily
                        decay/prune, the count->w re-seed, and hebbian_consolidate.
      - reconsolidate : reconsolidate (recall-is-a-write: merge/spawn + log).
      - replay        : the hot/recent working set, the ONLINE active_set, replay_sweep +
                        the SWR-interleaved variant, and replay_engram_traces (genuine-once).
      - salience      : the salience SOURCE — compute_salience + the composite signals +
                        salience_merge_guard.
      - schema        : chain_maturity + schema_accelerate.

PATCH SEAMS (carried over byte-for-byte):
    - hebbian_record is a mock.patch.object(bio, ...) spy target AND is called by a
      sibling (replay.replay_engram_traces); that caller reaches it THROUGH this package
      facade (samia.core.bio as _pkg) so a facade-level patch is honored.
    - active_set is a mock.patch("samia.core.bio.active_set", ...) target with NO internal
      caller — the re-export here suffices.
    - compute_salience is a mock.patch.object(bio, ...) target with NO internal caller —
      the re-export here suffices.
    The single-owned module state (_KWTA_PROJ_CACHE) + the aliased deps (_chain/_dt/_time)
    live in config and are re-exported here so any importer/test that reaches them through
    the package sees the one copy.

Public surface re-exported here (byte-for-byte the pre-split module — 51 names):
    re-exported imports : Optional, Path, annotations, hashlib, json, np, os, sys
    constants           : PATTERN_THRESHOLD_DEFAULT, HEBB_DECAY, HEBB_EMA_ALPHA,
                          HEBB_MIN_INTERVAL_ENV, HEBB_PROMOTE_REPEATS, HEBB_PROMOTION,
                          HEBB_PRUNE, HEBB_REPLAY_COACT_WEIGHT, HEBB_SEED_MARGIN,
                          HOT_RECENCY_DAYS, INTERLEAVE_DEFAULT_COLD_PER_HOT,
                          INTERLEAVE_THRESHOLD, ACTIVE_SET_HOT_N, REPLAY_DEFAULT_SAMPLE,
                          REPLAY_NEIGHBOR_THRESHOLD, REPLAY_ONLY_W_CEILING,
                          SCHEMA_MIN_AGE_DAYS, SCHEMA_MIN_NODES, SALIENCE_*, KWTA_*
    functions           : active_set, chain_maturity, compute_salience,
                          forget_node_weights, hebbian_consolidate, hebbian_record,
                          kwta_sparse_code, pattern_separation_decision, reconsolidate,
                          replay_engram_traces, replay_sweep, replay_sweep_interleaved,
                          reseed_edge_weights, salience_merge_guard, schema_accelerate,
                          sweep_ghost_edges
Internal names also re-exported for direct test/importer/patch-seam access (NOT in
__all__): _dt, _time, _chain, _KWTA_PROJ_CACHE, _bio_paths, _node_embedding,
    _kwta_projection, _addr_for_node, _load_edge_weights, _save_edge_weights,
    _attractor_count, _consolidate_cadence_blocked, _record_consolidate_run,
    _atomic_drain_log, _apply_coactivation, _decay_and_prune, _is_promotable,
    _recently_accessed_nodes, _fast_engram_neighbors, _all_chain_node_names,
    _cold_chains, _embedding_for_node, _pair_key, _load_engram_replay_state,
    _save_engram_replay_state, _node_frontmatter, _salience_surprise,
    _salience_contradiction, _salience_repetition.
"""

from __future__ import annotations

# The shared leaf — the re-exported stdlib + numpy (the baseline's public imports:
# Optional/Path/hashlib/json/np/os/sys; `annotations` rides the `from __future__` above),
# the private _dt/_time aliases, the aliased _chain dependency, EVERY tuning constant, the
# single-owned _KWTA_PROJ_CACHE projection cache, and the shared _bio_paths file resolver.
from .config import (  # noqa: F401
    # re-exported stdlib (public surface)
    Optional,
    Path,
    hashlib,
    json,
    np,
    os,
    sys,
    # private stdlib aliases + the aliased package dependency
    _dt,
    _time,
    _chain,
    # single-owned mutable state
    _KWTA_PROJ_CACHE,
    # shared path resolver
    _bio_paths,
    # tuning constants
    PATTERN_THRESHOLD_DEFAULT,
    HEBB_PROMOTION,
    HEBB_PROMOTE_REPEATS,
    HEBB_EMA_ALPHA,
    HEBB_DECAY,
    HEBB_PRUNE,
    HEBB_REPLAY_COACT_WEIGHT,
    HEBB_SEED_MARGIN,
    REPLAY_ONLY_W_CEILING,
    REPLAY_NEIGHBOR_THRESHOLD,
    REPLAY_DEFAULT_SAMPLE,
    INTERLEAVE_THRESHOLD,
    INTERLEAVE_DEFAULT_COLD_PER_HOT,
    HOT_RECENCY_DAYS,
    SCHEMA_MIN_NODES,
    SCHEMA_MIN_AGE_DAYS,
    HEBB_MIN_INTERVAL_ENV,
    ACTIVE_SET_HOT_N,
    SALIENCE_W_SURPRISE,
    SALIENCE_W_CONTRADICTION,
    SALIENCE_W_REPETITION,
    SALIENCE_REPETITION_SATURATION,
    SALIENCE_TAG_VALUE,
    SALIENCE_MERGE_GUARD_DEFAULT,
    KWTA_PROJ_DIM,
    KWTA_FRAC_DEFAULT,
    KWTA_SEED,
)

# Pattern separation — the cosine dedup gate + the orthogonalizing kWTA sparse code.
from .pattern import (  # noqa: F401
    pattern_separation_decision,
    kwta_sparse_code,
    _node_embedding,
    _kwta_projection,
)

# The Hebbian arm. hebbian_record is a mock.patch.object(bio, ...) seam (replay's
# replay_engram_traces reaches it through this facade); the privates are re-exported for
# direct test/importer access (test_bio / test_hippocampus / test_forget_node /
# successor / sleep_pressure / temporal_recall_sith all reach into them).
from .hebbian import (  # noqa: F401
    hebbian_record,
    forget_node_weights,
    sweep_ghost_edges,
    reseed_edge_weights,
    hebbian_consolidate,
    _load_edge_weights,
    _save_edge_weights,
    _attractor_count,
    _addr_for_node,
    _consolidate_cadence_blocked,
    _record_consolidate_run,
    _atomic_drain_log,
    _apply_coactivation,
    _decay_and_prune,
    _is_promotable,
)

# Reconsolidation (recall-is-a-write).
from .reconsolidate import (  # noqa: F401
    reconsolidate,
)

# The replay arm. active_set is a mock.patch("samia.core.bio.active_set", ...) seam.
from .replay import (  # noqa: F401
    active_set,
    replay_sweep,
    replay_sweep_interleaved,
    replay_engram_traces,
    _recently_accessed_nodes,
    _fast_engram_neighbors,
    _all_chain_node_names,
    _cold_chains,
    _embedding_for_node,
    _pair_key,
    _load_engram_replay_state,
    _save_engram_replay_state,
)

# The salience SOURCE. compute_salience is a mock.patch.object(bio, ...) seam (integrity
# tests rebind it; no internal caller, so re-export suffices).
from .salience import (  # noqa: F401
    compute_salience,
    salience_merge_guard,
    _node_frontmatter,
    _salience_surprise,
    _salience_contradiction,
    _salience_repetition,
)

# Schema-accelerated ingestion.
from .schema import (  # noqa: F401
    chain_maturity,
    schema_accelerate,
)

# __all__ — the LOCALLY-owned PUBLIC names (the 51 the baseline records: 8 re-exported
# imports + 26 constants + 16 functions, plus `annotations` from __future__). The verify
# script diffs the full public surface (dir() minus underscore names) against the baseline;
# __all__ documents the intended export set and bounds `from ... import *` to the pre-split
# public set. (The private test/importer/patch-seam-reached names above are re-exported but
# intentionally NOT in __all__, mirroring the exemplars.)
__all__ = [
    # re-exported imports
    "Optional", "Path", "annotations", "hashlib", "json", "np", "os", "sys",
    # constants
    "PATTERN_THRESHOLD_DEFAULT", "HEBB_PROMOTION", "HEBB_PROMOTE_REPEATS",
    "HEBB_EMA_ALPHA", "HEBB_DECAY", "HEBB_PRUNE", "HEBB_REPLAY_COACT_WEIGHT",
    "HEBB_SEED_MARGIN", "REPLAY_ONLY_W_CEILING", "REPLAY_NEIGHBOR_THRESHOLD",
    "REPLAY_DEFAULT_SAMPLE", "INTERLEAVE_THRESHOLD", "INTERLEAVE_DEFAULT_COLD_PER_HOT",
    "HOT_RECENCY_DAYS", "SCHEMA_MIN_NODES", "SCHEMA_MIN_AGE_DAYS",
    "HEBB_MIN_INTERVAL_ENV", "ACTIVE_SET_HOT_N", "SALIENCE_W_SURPRISE",
    "SALIENCE_W_CONTRADICTION", "SALIENCE_W_REPETITION", "SALIENCE_REPETITION_SATURATION",
    "SALIENCE_TAG_VALUE", "SALIENCE_MERGE_GUARD_DEFAULT", "KWTA_PROJ_DIM",
    "KWTA_FRAC_DEFAULT", "KWTA_SEED",
    # functions
    "pattern_separation_decision", "kwta_sparse_code", "hebbian_record",
    "forget_node_weights", "sweep_ghost_edges", "reseed_edge_weights",
    "hebbian_consolidate", "reconsolidate", "active_set", "replay_sweep",
    "replay_sweep_interleaved", "replay_engram_traces", "compute_salience",
    "salience_merge_guard", "chain_maturity", "schema_accelerate",
]


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): re-export facade carved from the samia.bio monolith
# Layer:      core (pure library, no daemon dependency)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.core import bio` /
#             `from samia.core.bio import X` keeps working for all 51 public names; the
#             private test/importer/patch-seam-reached names (_bio_paths/_node_embedding/
#             _addr_for_node/_load_edge_weights/_save_edge_weights/_is_promotable/
#             _apply_coactivation/_decay_and_prune/_consolidate_cadence_blocked/
#             _node_frontmatter/_fast_engram_neighbors/_load_engram_replay_state/
#             _save_engram_replay_state + the rest + the aliased _dt/_time/_chain +
#             _KWTA_PROJ_CACHE) are re-exported too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
# ErrorModel: none here (import-time wiring only); each submodule footer documents its
#             own fail-soft posture.
# Depends:    .config, .pattern, .hebbian, .reconsolidate, .replay, .salience, .schema.
# Exposes:    the public 51 (in __all__) + the private/patch-seam/state names above.
# Note:       PATCH SEAMS — hebbian_record (called by replay.replay_engram_traces through
#             this facade) + active_set + compute_salience (mock targets with no internal
#             caller). No importlib.reload(bio) exists in the suite, so NO reload-cascade
#             shim is needed (unlike the contradiction exemplar's env-derived constants).
# restart:    bio changes require restarting samia.runtime.daemon (PID ~3167).
# Lines:      277
# --------------------------------------------------------------------------
