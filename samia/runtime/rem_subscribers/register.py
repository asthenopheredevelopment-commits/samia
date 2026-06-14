"""samia.runtime.rem_subscribers.register — the registration entry point.

Layer 1 (Owns / Depends):
    Owns:    register_rem_subscribers (the idempotent migration entry the daemon
             calls at startup) and the _REGISTERED flag. Wires each subscriber
             callable into the REM registry with its priority + due_condition +
             cursor_key, in priority order (LOW runs FIRST).
    Depends: .config (rem_cycle.register_offline_op + the registered-ops reader,
             _log, the PRIO_* bands), .due (the due_conditions), .subscribers (the
             six non-fact-extract callables), .fact_extract (_sub_fact_extract).

Layer 2 (What / Why):
    What: the top of the package's dependency DAG — it imports the callables +
          their due-conditions from the sibling submodules and registers the
          consolidation -> tier2_merge -> contradiction_passive -> integrity_repair
          -> vector_maintenance -> replay -> fact_extract chain. Returns the
          registered names in priority order.
    Why:  the migration entry point (Q3 / proposal P2). Called by the daemon at
          startup so REM's driver runs these STRENGTHENING/ABSTRACTING ops inside
          the sleep window. Decay is NOT here — it is the continuous forgetting
          curve (wake+REM), driven by the idle_pulse "decay" subscriber, never the
          REM driver (no double-drive).
"""

from __future__ import annotations

from .config import (
    rem_cycle,
    _log,
    PRIO_CONSOLIDATION,
    PRIO_TIER2_MERGE,
    PRIO_CONTRADICTION_PASSIVE,
    PRIO_INTEGRITY_REPAIR,
    PRIO_VECTOR_MAINTENANCE,
    PRIO_REPLAY,
    PRIO_FACT_EXTRACT,
)
from .due import (
    _due_consolidation,
    _due_tier2_merge,
    _due_contradiction_passive,
    _due_integrity_repair,
    _vector_index_drift,
    _due_replay,
    _due_fact_extract,
)
from .subscribers import (
    _sub_consolidation,
    _sub_tier2_merge,
    _sub_contradiction_passive,
    _sub_integrity_repair,
    _sub_vector_maintenance,
    _sub_replay,
)
from .fact_extract import _sub_fact_extract


# ---------------------------------------------------------------------------
# Registration entry point (idempotent) — called by the daemon at startup.
# ---------------------------------------------------------------------------

_REGISTERED = False


def register_rem_subscribers() -> list[str]:
    """Register the REM-gated offline ops as REM subscribers (idempotent).

    What: wires consolidation -> tier2_merge -> contradiction_passive -> replay
          -> fact-extract into the REM registry with their priorities +
          due-conditions + cursor keys. Returns the registered names in priority
          order. Decay is NOT here — it is the continuous forgetting curve
          (wake+REM), driven by the idle_pulse "decay" subscriber, never the REM
          driver (no double-drive).
    Why:  the migration entry point. Called by the daemon at startup so REM's
          driver runs these STRENGTHENING/ABSTRACTING ops inside the sleep
          window. FEAT-2026-06-07 P3c adds the contradiction_passive sweep at
          priority 25 (between consolidation and replay); it is double-gated
          (REM + ASTHENOS_CONTRADICTION_ENABLED), inert by default. FEAT-2026-
          06-07 P1 adds the tier2_merge pick-winner dup-merge DRAIN at priority
          22 (between consolidation and contradiction_passive); double-gated
          (REM + ASTHENOS_TIER2_MERGE_ENABLED), inert by default — it drains the
          surfacer's near-dup backlog so work_remaining can finally go false.
    """
    global _REGISTERED
    rem_cycle.register_offline_op(
        "consolidation", _sub_consolidation, priority=PRIO_CONSOLIDATION,
        due_condition=_due_consolidation, cursor_key="consolidation",
    )
    rem_cycle.register_offline_op(
        "tier2_merge", _sub_tier2_merge, priority=PRIO_TIER2_MERGE,
        due_condition=_due_tier2_merge, cursor_key="tier2_merge",
    )
    rem_cycle.register_offline_op(
        "contradiction_passive", _sub_contradiction_passive,
        priority=PRIO_CONTRADICTION_PASSIVE,
        due_condition=_due_contradiction_passive,
        cursor_key="contradiction_passive",
    )
    rem_cycle.register_offline_op(
        "integrity_repair", _sub_integrity_repair,
        priority=PRIO_INTEGRITY_REPAIR,
        due_condition=_due_integrity_repair,
        cursor_key="integrity_repair",
    )
    rem_cycle.register_offline_op(
        "vector_maintenance", _sub_vector_maintenance,
        priority=PRIO_VECTOR_MAINTENANCE,
        due_condition=_vector_index_drift,
        cursor_key="vector_maintenance",
    )
    rem_cycle.register_offline_op(
        "replay", _sub_replay, priority=PRIO_REPLAY,
        due_condition=_due_replay, cursor_key="replay",
    )
    rem_cycle.register_offline_op(
        "fact_extract", _sub_fact_extract, priority=PRIO_FACT_EXTRACT,
        due_condition=_due_fact_extract, cursor_key="fact_extract",
    )
    _REGISTERED = True
    names = rem_cycle.registered_offline_ops()
    _log.info("rem_subscribers: registered %d REM offline ops: %s",
              len(names), names)
    return names


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_subscribers.register
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_subscribers monolith during modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the migration entry point — wire each subscriber callable into the
#             REM registry with its priority + due_condition + cursor_key, in
#             priority order (consolidation 20 -> tier2_merge 22 ->
#             contradiction_passive 25 -> integrity_repair 28 ->
#             vector_maintenance 29 -> replay 30 -> fact_extract 40). Decay is NOT
#             registered here (continuous, idle_pulse-driven).
# Stability:  stable — the carve preserved every registration (name, priority,
#             due_condition, cursor_key) and the idempotent _REGISTERED flag
#             byte-identical to the monolith.
# ErrorModel: none here — registration is pure wiring; register_offline_op owns
#             its own dedup/replace semantics.
# Depends:    .config (rem_cycle, _log, PRIO_* bands), .due (the due_conditions),
#             .subscribers (the six callables), .fact_extract (_sub_fact_extract).
# Exposes:    register_rem_subscribers, _REGISTERED.
# Lines:      145
# ─────────────────────────────────────────────
