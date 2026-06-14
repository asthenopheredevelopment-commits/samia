"""samia.runtime.rem_subscribers — register the offline ops as REM subscribers.

Layer 1 (Owns / Depends):
    Owns:    the MIGRATION wiring (P2) — the mapping of each existing offline
             memory op to a REM subscriber (name, priority, due_condition,
             cursor_key) via rem_cycle.register_offline_op, the two BATCH
             wrappers that turn per-text / per-call primitives into REM-runnable
             ticks (consolidation surfacer, fact-extract batch), and the gated
             entry of those wrappers — split by responsibility into five submodules
             behind this re-export facade (the public import surface is byte-for-
             byte unchanged from the pre-split single module). Owns NO op internals
             — it wraps/registers the existing functions, never rewrites them.
    Depends: samia.runtime.rem_cycle (register_offline_op / the gate /
             read_cursor / write_cursor), and the existing offline ops:
             samia.core.context_extension.idle_replay_tick (replay/dreaming +
             hebbian_consolidate + reseed — already gated at its own entry),
             samia.core.consolidation.audit_all/surface (the near-dup surfacer),
             samia.core.fact_extractor.extract_atoms (the per-text primitive).
             NOTE: tier.decay_tick is NOT here — decay is the short-term
             forgetting curve and runs CONTINUOUSLY across wake+REM (driven by
             the idle_pulse "decay" subscriber, NOT REM-gated; see below).

Layer 2 (What / Why):
    What: REM P2's migration. It registers the heavy offline reconciliation ops
          as REM subscribers so the driver (rem_cycle.run_due_subscribers) runs
          them — and ONLY runs them — inside the sleep window, in priority order,
          each with a due-condition so a cycle with no real backlog does little.
          The two ops that are NOT already tick-shaped get thin batch wrappers:
            - consolidation surfacer: audit_all -> surface (the 600-pair
              .consolidation_candidates.json producer) had NO live idle/scheduler
              caller; this is its first scheduled home (a REM subscriber).
            - fact-extract batch: extract_atoms is a per-text PRIMITIVE (no mem
              arg, not wired anywhere); this wraps it as a batch offline op that
              drains a pending-extraction queue (cursor-checkpointed). It is a
              NEW build (nothing to migrate-away), wired only when a queue exists.
    Why:  Q3 / the proposal's P2 — the heavy STRENGTHENING/ABSTRACTING offline
          ops stop trickling on every idle pulse and run behind REM instead.
          Migrating PRECISELY (gate at each op's own entry + register here)
          means the op refuses outside REM no matter WHICH caller invokes it
          (idle_pulse, the dormant scheduler.py, a future caller), and the
          registry is the operator-visible inventory. The 1109-line monolith was
          split by responsibility (config/due/subscribers/fact_extract/register)
          with ZERO behavior change; this facade re-exports the FULL public surface
          so every importer (`from samia.runtime.rem_subscribers import X`) and
          every attribute reach-in (`rem_subscribers._sub_*`, the `mock.patch.object`
          targets) is unaffected.

CLS rationale — what IS and IS NOT REM-gated (operator correction 2026-06-07):
    REM-gated (sleep = CONSOLIDATION + REPLAY, strengthening/abstracting):
      consolidation, contradiction_passive, replay/dreaming, fact_extract.
    NOT REM-gated — runs CONTINUOUSLY across BOTH wake and REM:
      decay (tier.decay_tick). Decay is the short-term forgetting curve; sleep
      does NOT do the forgetting. It is driven solely by the idle_pulse "decay"
      subscriber on its 6h cadence (NOT a REM subscriber here, so no double-
      drive). It was wrongly REM-gated by P2; that gate has been removed.

Priorities (LOW runs FIRST):
    20  consolidation            — surface near-dup candidates.
    22  tier2_merge              — FEAT-2026-06-07 P1+P2: pick-winner dup-merge.
    25  contradiction_passive    — FEAT-2026-06-07 P3c: incremental supersession.
    28  integrity_repair         — FEAT-2026-06-07 P2: CONSOLIDATION repair.
    29  vector_maintenance       — G4-2026-06-11: vector-index sync on drift.
    30  replay/dreaming          — SWR cross-chain edges + hebbian consolidate.
    40  fact_extract             — drain any pending fact-extraction queue (new).

PRODUCE-ONLY: importing this package does nothing; registration runs only when
register_rem_subscribers() is called (by the daemon at startup, operator-gated).
No thread, no timer, no live mutation here.

Public surface re-exported here (byte-for-byte the pre-split module):
    re-exported imports : Any, Path, annotations, logging, os
    re-exported modules : rem_cycle, sleep_pressure (aliased dependency modules)
    constants           : PRIO_* (7), INTEGRITY_REPAIR_ENABLED_ENV,
                          VECTOR_FULL_REBUILD_S_ENV
    functions           : register_rem_subscribers
Internal names also re-exported for direct test/importer + mock.patch.object access
(NOT in __all__): _integrity_repair_enabled, _vector_full_rebuild_interval_s, the
_due_* conditions + _merge_drainable + the vector-drift helpers, every _sub_*
callable, the fact-extract subsystem (_fact_extract_backend, _persist_atoms,
_fx_stamp_distilled, _fx_provenance_edge, _fe_slug, _dt_today), the budgets +
_CONSOLIDATION_THRESHOLD, _log, _REGISTERED.
"""

from __future__ import annotations

# Re-exported module-top names the monolith pulled in and other code imports
# THROUGH this module (Any/Path/logging/os). `annotations` rides the
# `from __future__` above. The aliased dependency modules rem_cycle/sleep_pressure
# are re-exported too (other code reaches rem_subscribers.rem_cycle). All must stay
# importable from the package facade — they live in config (the leaf owner).
from .config import (  # noqa: F401
    Any,
    Path,
    logging,
    os,
    rem_cycle,
    sleep_pressure,
    _log,
    INTEGRITY_REPAIR_ENABLED_ENV,
    VECTOR_FULL_REBUILD_S_ENV,
    PRIO_CONSOLIDATION,
    PRIO_TIER2_MERGE,
    PRIO_CONTRADICTION_PASSIVE,
    PRIO_INTEGRITY_REPAIR,
    PRIO_VECTOR_MAINTENANCE,
    PRIO_REPLAY,
    PRIO_FACT_EXTRACT,
    _INTEGRITY_REPAIR_BUDGET,
    _TIER2_MERGE_BUDGET,
    _CONSOLIDATION_THRESHOLD,
    _integrity_repair_enabled,
    _vector_full_rebuild_interval_s,
)

# The due-conditions + the vector-drift helpers (test-reached:
# rem_subscribers._due_tier2_merge / ._due_integrity_repair / ._vector_index_drift).
from .due import (  # noqa: F401
    _due_consolidation,
    _due_tier2_merge,
    _merge_drainable,
    _due_contradiction_passive,
    _due_integrity_repair,
    _due_replay,
    _due_fact_extract,
    _live_node_count,
    _vector_index_node_count,
    _vector_index_drift,
)

# The non-fact-extract subscriber callables (test-reached:
# rem_subscribers._sub_tier2_merge / ._sub_integrity_repair / ._sub_consolidation /
# ._sub_vector_maintenance / ._sub_contradiction_passive).
from .subscribers import (  # noqa: F401
    _sub_consolidation,
    _sub_tier2_merge,
    _sub_contradiction_passive,
    _sub_integrity_repair,
    _sub_vector_maintenance,
    _sub_replay,
)

# The batch fact-extract subsystem. _fact_extract_backend + _fx_provenance_edge are
# re-exported as the LOAD-BEARING mock.patch.object targets (the tests patch them on
# THIS package object; _sub_fact_extract/_persist_atoms reach them through the facade
# so the patch takes effect — see fact_extract.py's PATCH SEAM note).
from .fact_extract import (  # noqa: F401
    _sub_fact_extract,
    _fact_extract_backend,
    _persist_atoms,
    _fx_stamp_distilled,
    _fx_provenance_edge,
    _fe_slug,
    _dt_today,
)

# The registration entry point (top of the DAG) + the idempotent flag.
from .register import register_rem_subscribers, _REGISTERED  # noqa: F401

# __all__ — the LOCALLY-owned PUBLIC names (the 17 the baseline records: the 5
# re-exported imports, the 2 aliased dependency modules, the 7 PRIO_* bands, the 2
# env-flag names, and register_rem_subscribers). The verify script diffs the full
# public surface (dir() minus underscore names) against the baseline; __all__
# documents the intended export set and bounds `from ... import *` to exactly the
# pre-split public 17. (The private test-reached names above are re-exported but
# intentionally NOT in __all__, mirroring the merge_consumer exemplar.)
__all__ = [
    # re-exported imports + aliased dependency modules
    "Any", "Path", "annotations", "logging", "os",
    "rem_cycle", "sleep_pressure",
    # constants
    "PRIO_CONSOLIDATION", "PRIO_TIER2_MERGE", "PRIO_CONTRADICTION_PASSIVE",
    "PRIO_INTEGRITY_REPAIR", "PRIO_VECTOR_MAINTENANCE", "PRIO_REPLAY",
    "PRIO_FACT_EXTRACT",
    "INTEGRITY_REPAIR_ENABLED_ENV", "VECTOR_FULL_REBUILD_S_ENV",
    # functions
    "register_rem_subscribers",
]


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_subscribers
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P2)
#             + FEAT-2026-06-07-memory-p3-contradiction-detector-v01 (P3c) +
#               FEAT-2026-06-07-memory-tier2-merge-consumer-v01 (P1+P2) +
#               FEAT-2026-06-07-memory-granular-recall-repaired-decay (P2) +
#               FEAT-2026-06-10-memory-fact-extract-producer-v01 (P1) +
#               TUNE-2026-06-10 (decision c) + G2/G4-2026-06-11
#             + Phase-B modularization: the 1109-line monolith carved into a
#               re-export-preserving package (config/due/subscribers/fact_extract/
#               register) with ZERO behavior change; this __init__ re-exports the
#               full public surface so every importer + attribute reach-in is
#               unaffected.
# Layer:      runtime (library helper, no daemon loop)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.runtime.rem_subscribers
#             import X` keeps working for all 17 public names; the private helpers
#             the targeted tests reach (the _due_*/_sub_* callables) and the two
#             mock.patch.object targets (_fact_extract_backend, _fx_provenance_edge)
#             are re-exported too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
# ErrorModel: none here (import-time wiring only); each submodule footer documents
#             its own gated / fail-soft posture.
# Depends:    .config, .due, .subscribers, .fact_extract, .register.
# Exposes:    the public 17 (in __all__) + _integrity_repair_enabled/
#             _vector_full_rebuild_interval_s/_log/_REGISTERED + the budgets +
#             every _due_*/_merge_drainable/_vector_* helper + every _sub_* callable
#             + the fact-extract subsystem (_fact_extract_backend/_persist_atoms/
#             _fx_stamp_distilled/_fx_provenance_edge/_fe_slug/_dt_today) for tests.
# Note:       PRODUCE-ONLY — import does nothing; registration runs only when
#             register_rem_subscribers() is called (daemon startup, operator-gated).
#             Wraps existing ops; no op internals rewritten. Decay is driven only by
#             idle_pulse (continuous, NOT REM-gated).
# Lines:      214
# ─────────────────────────────────────────────
