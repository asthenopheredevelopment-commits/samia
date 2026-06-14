"""samia.runtime.rem_subscribers.config — shared constants, env flags, deps leaf.

Layer 1 (Owns / Depends):
    Owns:    the module-level surface the whole package reads — the logger, the
             priority bands (PRIO_*, LOW runs FIRST), the two operator env-flag
             names (INTEGRITY_REPAIR_ENABLED_ENV / VECTOR_FULL_REBUILD_S_ENV), the
             per-cycle budgets + the consolidation threshold, and the two live
             flag readers (_integrity_repair_enabled / _vector_full_rebuild_interval_s).
             Re-exports the two sibling-shared dependency modules (rem_cycle /
             sleep_pressure) so every sibling imports them THROUGH one owner.
    Depends: samia.runtime.rem_cycle (the registry + gate + cursor helpers),
             samia.runtime.sleep_pressure (the due-condition signal readers).
             logging/os/pathlib/typing from stdlib.

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — it imports nothing from its
          siblings, so the priorities/flags/budgets live in one place and are
          never duplicated. rem_cycle + sleep_pressure are re-exported here (not
          re-imported per submodule) so the carve has ONE binding of each aliased
          dependency module and the public import surface keeps them reachable as
          rem_subscribers.rem_cycle / rem_subscribers.sleep_pressure.
    Why:  splitting the 1109-line monolith by responsibility (due-conditions,
          subscriber callables, the fact-extract subsystem, registration) leaves a
          shared base of constants + flag readers that all of them need;
          concentrating them here keeps the bands single-sourced and the import
          graph acyclic (config depends on nothing inside the package).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path  # noqa: F401  (re-exported public surface)
from typing import Any  # noqa: F401  (re-exported public surface)

from samia.runtime import rem_cycle, sleep_pressure  # noqa: F401  (re-exported)

_log = logging.getLogger("samia.runtime.rem_subscribers")

# ASTHENOS_INTEGRITY_REPAIR_ENABLED — What: the enable flag for the P2 integrity
#   consolidation-repair subscriber (the CONSOLIDATION repair trigger). Default OFF.
# Why: FEAT-2026-06-07 granular-recall-repaired-decay P2 / Q3a — sleep PARTIALLY heals
#   the integrity of the nodes it consolidates. Double-gated like the other P2 ops:
#   REM (the subscriber gate) + this enable flag, both inert by default (produce-only).
INTEGRITY_REPAIR_ENABLED_ENV = "ASTHENOS_INTEGRITY_REPAIR_ENABLED"


def _integrity_repair_enabled() -> bool:
    """Live read of the ASTHENOS_INTEGRITY_REPAIR_ENABLED master switch (default OFF).

    Delegates to integrity.repair_enabled() so the recall-repair seam (memory_search) and
    this P2 consolidation-repair subscriber share ONE reader of ONE flag (the activation
    wiring exposed the same flag at core level). Falls back to a direct env read if the
    core import is unavailable, preserving the prior behavior + the same env var name.
    """
    try:
        from samia.core import integrity as _integrity
        return _integrity.repair_enabled()
    except Exception:
        return os.environ.get(INTEGRITY_REPAIR_ENABLED_ENV, "0") == "1"


# Priority bands (LOW runs FIRST). The P3-passive contradiction sweep slots at
# 25, between consolidation (20) and replay (30) per the reserved placeholder.
# (No decay band: decay is NOT a REM subscriber — it runs continuously across
# wake+REM via the idle_pulse "decay" subscriber. See package CLS rationale.)
PRIO_CONSOLIDATION = 20
PRIO_TIER2_MERGE = 22  # FEAT-2026-06-07 P1 (pick-winner dup-merge drain)
PRIO_CONTRADICTION_PASSIVE = 25  # FEAT-2026-06-07 P3c
PRIO_INTEGRITY_REPAIR = 28  # FEAT-2026-06-07 granular-recall-repaired-decay P2
PRIO_VECTOR_MAINTENANCE = 29  # G4-2026-06-11: keep the vector index in sync with nodes/
PRIO_REPLAY = 30
PRIO_FACT_EXTRACT = 40

# ASTHENOS_VECTOR_FULL_REBUILD_S — What: the cadence (seconds) between FULL vector
#   index rebuilds (rebuild=True, re-embeds every node). Default 7 days. The
#   incremental build (manifest-cached, embeds only new/changed nodes) runs every
#   cycle on drift; the full rebuild is the periodic floor-sweep that re-embeds the
#   whole corpus (catches model/content drift the sha256 cache cannot see).
# Why: G4-2026-06-11 (operator choice 4a) — the index drifted because nothing
#   rebuilt it automatically. The incremental path keeps it fresh cheaply; the full
#   rebuild is bounded to a long cadence so it never thrashes a REM cycle.
VECTOR_FULL_REBUILD_S_ENV = "ASTHENOS_VECTOR_FULL_REBUILD_S"
_VECTOR_FULL_REBUILD_S_DEFAULT = 7 * 24 * 3600  # 7 days


def _vector_full_rebuild_interval_s() -> float:
    """The full-rebuild cadence in seconds (env-overridable, 7-day default).

    Fail-soft: a malformed env value falls back to the 7-day default.
    """
    raw = os.environ.get(VECTOR_FULL_REBUILD_S_ENV)
    if raw is None:
        return float(_VECTOR_FULL_REBUILD_S_DEFAULT)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(_VECTOR_FULL_REBUILD_S_DEFAULT)


# Per-cycle budget for the integrity consolidation-repair subscriber (cursor-tracked).
_INTEGRITY_REPAIR_BUDGET = 50

# Per-cycle drain budget for the tier2_merge subscriber (cursor-tracked across
# cycles). Bounded so a large backlog does not stall a single REM cycle.
_TIER2_MERGE_BUDGET = 50

# Near-dup surfacing threshold for the consolidation subscriber (the surfacer's
# own knee; matches consolidation.DEFAULT_THRESHOLD intent — kept explicit so a
# tune does not silently change behavior).
_CONSOLIDATION_THRESHOLD = 0.15


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_subscribers.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_subscribers monolith
#             during modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       shared base of the rem_subscribers package — the logger, the
#             priority bands (LOW runs FIRST), the two operator env-flag names,
#             the per-cycle budgets + consolidation threshold, the two live flag
#             readers, and the re-exported rem_cycle/sleep_pressure dependency
#             modules every sibling imports through.
# Stability:  stable — pure constants + two side-effect-free env readers; the
#             carve changed no value (bands/flags/budgets byte-identical to the
#             monolith).
# ErrorModel: none — _integrity_repair_enabled and _vector_full_rebuild_interval_s
#             are fail-soft env reads (a missing core import / malformed value
#             falls back to the prior default); they never raise.
# Depends:    logging, os, pathlib, typing (stdlib). samia.runtime.rem_cycle,
#             samia.runtime.sleep_pressure (re-exported).
# Exposes:    PRIO_* (7), INTEGRITY_REPAIR_ENABLED_ENV, VECTOR_FULL_REBUILD_S_ENV,
#             _integrity_repair_enabled, _vector_full_rebuild_interval_s, the
#             budgets/threshold, _log, rem_cycle/sleep_pressure, Any/Path.
# Lines:      136
# ─────────────────────────────────────────────
