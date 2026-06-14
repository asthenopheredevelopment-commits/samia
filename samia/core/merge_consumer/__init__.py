"""samia.core.merge_consumer — Tier-2 merge consumer P1+P2+P3 (pick-winner
dup-merge + LLM-synthesized abstraction + salience guard, operator-gated).

Layer 1 (Owns / Depends):
    Owns:    the DRAIN half of the Tier-2 abstractive consolidation backlog,
             split by responsibility into four submodules behind this re-export
             facade (the public import surface is byte-for-byte unchanged from the
             pre-split single module):
               - config    : shared constants, the is_enabled flag, the
                             order-independent _candidate_id, and the re-exported
                             _con/_fm/_ia dependency modules.
               - candidates: load the surfacer's candidates, resolve + classify a
                             pair (dup vs abstract), read a node's frontmatter.
               - winner    : pick the richer survivor, lay the provenance edge,
                             run the AUTO pick-winner dup merge (RESTORABLE).
               - abstraction: the P2 abstraction lifecycle (record/synthesize/
                             confirm/reject), the P3 salience guard, the fact-
                             extract enqueue feed.
               - drain     : the cursor-tracked batch orchestrator the REM
                             tier2_merge subscriber calls.
    Depends: samia.core.{consolidation,ia,frontmatter,web_store,fact_extractor},
             samia.core.{bio,integrity}, samia.runtime.contradiction (the cosine
             finder + the P2 synthesize_node, the SAME judge inference backend).

Layer 2 (What / Why):
    What: P1 pick-winner duplicate merge (AUTO, RESTORABLE) drains the surfacer's
          .consolidation_candidates.json backlog; P2 LLM-synthesized abstraction
          (OPERATOR-GATED) proposes a higher-level node for the distinct minority
          and applies it only on confirm; P3 salience guard surfaces a distinct
          high-salience source instead of abstracting it away.
    Why:  the consolidation surfacer wrote ~600 near-dup pairs but NOTHING drained
          them, so REM's work_remaining stayed true forever. This package is the
          missing DRAIN. The 1175-line monolith was split by responsibility (no
          behavior change) so each subsystem is independently legible; this facade
          re-exports the FULL public surface so every importer
          (`from samia.core.merge_consumer import X`) is unaffected.

PRODUCE-ONLY: importing this package does nothing. drain() is a no-op unless
ASTHENOS_TIER2_MERGE_ENABLED=1 (default OFF), mirroring the contradiction passive
sweep posture — inert until the operator enables it + restarts the daemon. No
thread, no timer, no live mutation on import.

Public surface re-exported here (byte-for-byte the pre-split module):
    re-exported imports : Any, Optional, Path, annotations, hashlib, json, os
    functions           : is_enabled, load_candidates, classify_pair, pick_winner,
                          merge_dup, list_abstraction_candidates,
                          synthesize_abstraction, synthesize_pending,
                          confirm_abstraction, reject_abstraction, drain
Internal names also re-exported for direct test/importer access (NOT in __all__):
    _ia (the ia module — patched as merge_consumer._ia.forget_node), _candidate_id,
    _record_abstract, _record_guarded, _salience_guards_pair, _new_abstraction_id,
    _add_provenance_edge, _enqueue_abstract_pair.
"""

from __future__ import annotations

# Re-exported module-top imports the monolith pulled in and other code imports
# THROUGH this module (Any/Optional/Path/hashlib/json/os). `annotations` rides the
# `from __future__` above. They must stay importable from the package facade.
import hashlib  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Optional  # noqa: F401

# The shared dependency modules + flag + id primitive (config is the package leaf).
# _ia is re-exported so `mock.patch.object(merge_consumer._ia, "forget_node", ...)`
# patches the SAME samia.core.ia singleton every submodule calls through.
from .config import (  # noqa: F401
    is_enabled,
    _candidate_id,
    _con,
    _fm,
    _ia,
)

# Candidate I/O + classification.
from .candidates import (  # noqa: F401
    load_candidates,
    classify_pair,
)

# Winner selection + the AUTO dup merge (+ the provenance edge helper tests use).
from .winner import (  # noqa: F401
    pick_winner,
    merge_dup,
    _add_provenance_edge,
)

# The P2 abstraction lifecycle, the P3 salience guard, the fact-extract enqueue.
# The leading-underscore names are re-exported because the targeted tests reach
# them directly through the module namespace (e.g. mc._record_abstract,
# mc._new_abstraction_id, merge_consumer._enqueue_abstract_pair).
from .abstraction import (  # noqa: F401
    list_abstraction_candidates,
    synthesize_abstraction,
    synthesize_pending,
    confirm_abstraction,
    reject_abstraction,
    _record_abstract,
    _record_guarded,
    _salience_guards_pair,
    _new_abstraction_id,
    _enqueue_abstract_pair,
)

# The cursor-tracked batch drain (top of the DAG).
from .drain import drain  # noqa: F401

# __all__ — the LOCALLY-owned public names (the 11 functions + the 7 re-exported
# imports). The verify script diffs the full public surface (dir() minus
# underscore names) against the baseline; __all__ documents the intended export
# set and bounds `from ... import *` to exactly the pre-split public 18.
__all__ = [
    # re-exported imports (kept importable through the package facade)
    "Any", "Optional", "Path", "annotations", "hashlib", "json", "os",
    # functions
    "is_enabled", "load_candidates", "classify_pair", "pick_winner",
    "merge_dup", "list_abstraction_candidates", "synthesize_abstraction",
    "synthesize_pending", "confirm_abstraction", "reject_abstraction", "drain",
]


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.merge_consumer
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07-memory-tier2-merge-consumer-v01 (P1 pick-winner
#             dup-merge + P2 LLM-synthesized abstraction, operator-gated)
#             + FEAT-2026-06-10-memory-fact-extract-producer-v01 (the drain's
#               'abstract' branch ENQUEUES both node texts as ONE fact-extract
#               record; gated, fail-OPEN, ADDITIVE)
#             + BUG-2026-06-11 runaway-loop (enqueue side: per-pair-once done-set)
#             + Phase-B modularization: the 1175-line monolith carved into a
#               re-export-preserving package (config/candidates/winner/abstraction/
#               drain) with ZERO behavior change; this __init__ re-exports the full
#               public surface so every importer is unaffected.
# Layer:      core (pure library, no daemon dependency)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.core.merge_consumer
#             import X` keeps working for all 18 public names; the private helpers
#             the targeted tests reach (and _ia for monkeypatching forget_node) are
#             re-exported too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
# ErrorModel: none here (import-time wiring only); each submodule footer documents
#             its own fail-soft / gated posture.
# Depends:    .config, .candidates, .winner, .abstraction, .drain.
# Exposes:    the public 18 (in __all__) + _ia/_candidate_id/_record_abstract/
#             _record_guarded/_salience_guards_pair/_new_abstraction_id/
#             _add_provenance_edge/_enqueue_abstract_pair/_con/_fm for tests.
# Lines:      154
# Note:       PRODUCE-ONLY — import does nothing; drain()/synthesize_pending() are
#             no-ops unless ASTHENOS_TIER2_MERGE_ENABLED=1 (default OFF). Every
#             merge (dup or confirmed abstraction) is RESTORABLE. No thread/timer/
#             live mutation on import.
# --------------------------------------------------------------------------
