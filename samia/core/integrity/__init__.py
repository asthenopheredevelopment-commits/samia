"""samia.core.integrity — content-fidelity decay (the SECOND, orthogonal decay axis).

FEAT-2026-06-07 granular-recall-repaired-decay, Phases P1 + P2 + P3.

Layer 1 (Owns / Depends):
    Owns:    The per-node content-INTEGRITY axis — a [0,1] fraction-intact score, a
             retained pristine recovery ANCHOR, a slow per-character body erosion, the
             masked read (what a normal read sees), and the recall-repair trigger that
             restores the body byte-exact from the anchor — split by responsibility into
             five submodules behind this re-export facade (the public import surface is
             byte-for-byte unchanged from the pre-split single module):
               - config : the erosion/repair/floor/modulation constants, the granular
                          activation-flag env names + the generative flag, the live
                          env-flag readers (repair/decay/freeze/generative_enabled), the
                          tier-coupled salience_freeze_exempt reader, and the re-exported
                          _fm dependency — the package's shared, single-owned leaf.
               - anchors: the pristine recovery anchor store + primitives, the [0,1]
                          integrity-field read/write, the distillation gate, and the two
                          GENUINE-WRITE anchor-capture entrypoints.
               - erosion: the slow per-character body erosion (rate math + the anchor-
                          gated erode pass), the masked read seam, the live-salience rate
                          input, and the pure reconsolidation math.
               - repair : the anchor-first repair triggers (recall FULL, consolidation/
                          reconciliation PARTIAL), the P3 no-anchor generative fallback,
                          and the reconsolidation event log.
               - passes : the corpus sweeps — the anchor backstop, the erosion sweep
                          (with the P3 terminal-freeze-at-floor branch), and the
                          consolidation-repair REM pass.
    Depends: samia.core.frontmatter (the canonical node read/serialize seam, re-exported
             through config as _fm), samia.core.timestamp (UTC event stamps), json/os/
             random/pathlib (stdlib). Reads (never writes) salience/tier/last_access
             frontmatter for rate modulation. Reads tier.SALIENCE_FREEZE_EXEMPT +
             reaches bio.compute_salience / ia.freeze / contradiction.synthesize_node
             through FUNCTION-LOCAL imports in the submodules — it RIDES tier, it does
             not couple to it, and the heavy/runtime deps stay off the import path.

Layer 2 (What / Why):
    What: A genuinely SECOND decay axis, distinct from the relevance/tier decay in
          tier.py. Relevance-decay answers WHERE a node lives (lifecycle/tier);
          content-integrity decay answers HOW INTACT its content is (fidelity). A
          node's served/stored body erodes a little at a time (character-by-character),
          slowly, modulated by salience / tier / recency. A RECALL repairs it faithfully
          from the pristine anchor, resetting integrity toward 1.0.
    Why:  The operator's model of forgetting: granular + slow at the character level,
          coupled to reconsolidation on recall. The HYBRID model (Q1c) keeps a pristine
          anchor so early repair is FAITHFUL; the generative fallback (anchor gone) is
          P3. Layer-don't-replace: this composes alongside relevance-decay (Q6a). The
          1342-line monolith was split by RESPONSIBILITY (config / anchors / erosion /
          repair / passes) with ZERO behavior change; this facade re-exports the FULL
          public surface so every importer (`from samia.core.integrity import X`) and
          every attribute reach-in is unaffected.

NO PATCH SEAMS: a grep of the tree confirms no test or sibling reaches an integrity
    PRIVATE through the module namespace and no mock.patch.object(integrity, ...) target
    exists, so the siblings call each other through plain relative imports (no package-
    facade reach is needed — unlike the mcp_server / merge_consumer exemplars).

Public surface re-exported here (byte-for-byte the pre-split module — 52 names):
    re-exported imports : Optional, Path, annotations, json, os, random
    constants           : BASE_EROSION_RATE, CONSOLIDATION_REPAIR_BUDGET, DEFAULT_TIER,
                          EROSION_SENTINEL, GENERATIVE_REPAIR_STRENGTH, INTEGRITY_FLOOR,
                          INTEGRITY_FULL, INTEGRITY_NONE, PARTIAL_REPAIR_STRENGTH,
                          RECALL_REPAIR_STRENGTH, RECENCY_EROSION_CAP,
                          RECENCY_EROSION_PER_DAY, SALIENCE_EROSION_DAMPING,
                          SALIENCE_FREEZE_EXEMPT_DEFAULT, SEMANTIC_EROSION_FACTOR,
                          TIER_EROSION_FACTOR, and the four *_ENABLED_ENV flag names
                          (INTEGRITY_DECAY/FREEZE/GENERATIVE/REPAIR_ENABLED_ENV)
    functions           : repair_enabled, decay_enabled, freeze_enabled,
                          generative_enabled, salience_freeze_exempt, anchor_path,
                          has_anchor, write_anchor, read_anchor, ensure_anchor,
                          is_distilled, get_integrity, set_integrity, capture_on_write,
                          capture_on_genuine_write, erosion_rate, erode, mask_read,
                          reconsolidate_integrity, live_salience, recall_repair,
                          partial_repair, backfill_anchors_pass, anchor_backfill_tick,
                          integrity_decay_pass, consolidation_repair_pass
"""

from __future__ import annotations

# The shared leaf — the re-exported stdlib (json/os/random/Path/Optional; `annotations`
# rides the `from __future__` above), the re-exported _fm dependency, every constant +
# activation-flag env name, the live flag readers, and the tier-coupled exemption reader.
# json/os/random/Optional/Path are part of the public surface, so they must stay
# importable from the package facade.
from .config import (  # noqa: F401
    Optional,
    Path,
    json,
    os,
    random,
    _fm,
    # constants
    BASE_EROSION_RATE,
    DEFAULT_TIER,
    EROSION_SENTINEL,
    GENERATIVE_REPAIR_STRENGTH,
    INTEGRITY_DECAY_ENABLED_ENV,
    INTEGRITY_FLOOR,
    INTEGRITY_FREEZE_ENABLED_ENV,
    INTEGRITY_FULL,
    INTEGRITY_GENERATIVE_ENABLED_ENV,
    INTEGRITY_NONE,
    INTEGRITY_REPAIR_ENABLED_ENV,
    PARTIAL_REPAIR_STRENGTH,
    RECALL_REPAIR_STRENGTH,
    RECENCY_EROSION_CAP,
    RECENCY_EROSION_PER_DAY,
    SALIENCE_EROSION_DAMPING,
    SALIENCE_FREEZE_EXEMPT_DEFAULT,
    SEMANTIC_EROSION_FACTOR,
    TIER_EROSION_FACTOR,
    # flag readers
    repair_enabled,
    decay_enabled,
    freeze_enabled,
    generative_enabled,
    salience_freeze_exempt,
)

# The anchor store + integrity-field accessors + the genuine-write capture seams.
from .anchors import (  # noqa: F401
    anchor_path,
    has_anchor,
    write_anchor,
    read_anchor,
    ensure_anchor,
    is_distilled,
    get_integrity,
    set_integrity,
    capture_on_write,
    capture_on_genuine_write,
)

# The slow per-character body erosion + masked read + live-salience + reconsolidate math.
from .erosion import (  # noqa: F401
    erosion_rate,
    erode,
    mask_read,
    reconsolidate_integrity,
    live_salience,
)

# The anchor-first repair triggers (the P3 generative fallback rides inside these).
from .repair import (  # noqa: F401
    recall_repair,
    partial_repair,
)

# The corpus-walking sweeps + the consolidation-repair budget constant.
from .passes import (  # noqa: F401
    backfill_anchors_pass,
    anchor_backfill_tick,
    integrity_decay_pass,
    consolidation_repair_pass,
    CONSOLIDATION_REPAIR_BUDGET,
)

# __all__ — the LOCALLY-owned PUBLIC names (the 52 the baseline records: 6 re-exported
# imports, 20 constants, 26 functions). The verify script diffs the full public surface
# (dir() minus underscore names) against the baseline; __all__ documents the intended
# export set and bounds `from ... import *` to exactly the pre-split public 52.
__all__ = [
    # re-exported imports
    "Optional", "Path", "annotations", "json", "os", "random",
    # constants
    "BASE_EROSION_RATE", "CONSOLIDATION_REPAIR_BUDGET", "DEFAULT_TIER",
    "EROSION_SENTINEL", "GENERATIVE_REPAIR_STRENGTH", "INTEGRITY_DECAY_ENABLED_ENV",
    "INTEGRITY_FLOOR", "INTEGRITY_FREEZE_ENABLED_ENV", "INTEGRITY_FULL",
    "INTEGRITY_GENERATIVE_ENABLED_ENV", "INTEGRITY_NONE", "INTEGRITY_REPAIR_ENABLED_ENV",
    "PARTIAL_REPAIR_STRENGTH", "RECALL_REPAIR_STRENGTH", "RECENCY_EROSION_CAP",
    "RECENCY_EROSION_PER_DAY", "SALIENCE_EROSION_DAMPING",
    "SALIENCE_FREEZE_EXEMPT_DEFAULT", "SEMANTIC_EROSION_FACTOR", "TIER_EROSION_FACTOR",
    # functions
    "repair_enabled", "decay_enabled", "freeze_enabled", "generative_enabled",
    "salience_freeze_exempt", "anchor_path", "has_anchor", "write_anchor",
    "read_anchor", "ensure_anchor", "is_distilled", "get_integrity", "set_integrity",
    "capture_on_write", "capture_on_genuine_write", "erosion_rate", "erode", "mask_read",
    "reconsolidate_integrity", "live_salience", "recall_repair", "partial_repair",
    "backfill_anchors_pass", "anchor_backfill_tick", "integrity_decay_pass",
    "consolidation_repair_pass",
]


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.integrity
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07 granular-recall-repaired-decay — Phases P1 + P2 + P3
#             (integrity field + anchor + slow per-char erosion + masked read +
#             recall-repair FULL; anchor-capture-on-write + PARTIAL consolidation/
#             reconciliation repair + live-salience erosion modulation; TERMINAL
#             FREEZE-AT-FLOOR + GENERATIVE-RECONSTRUCTION FALLBACK, both gated/inert).
#             TUNE-2026-06-10 (decision c): a DISTILLED tier=="frozen" node erodes at
#             the slowest factor; G1-2026-06-11 (1a): a type:semantic node erodes at
#             SEMANTIC_EROSION_FACTOR regardless of tier.
#             + Phase-B modularization: the 1342-line monolith carved into a
#               re-export-preserving package (config/anchors/erosion/repair/passes)
#               with ZERO behavior change; this __init__ re-exports the full public
#               surface so every importer + attribute reach-in is unaffected.
# Layer:      core (pure library, no daemon dependency)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.core.integrity import X`
#             keeps working for all 52 public names (6 re-exported imports + 20
#             constants + 26 functions).
# Stability:  stable — pure re-export; the implementation lives in the submodules.
#             v1.3.1 second decay axis (content fidelity), composes alongside the
#             relevance/lifecycle decay in tier.py (Q6a, layer-don't-replace);
#             tier.step_relevance / tier.decay_pass are NOT touched.
# ErrorModel: none here (import-time wiring only); each submodule footer documents its
#             own fail-soft / anchor-gated / gated-and-inert posture. PRODUCE-ONLY:
#             import does nothing; the erosion sweep is dry+inert by default, the
#             terminal freeze + generative fallback are OFF by default, and nothing
#             erodes without a recoverable anchor (no data loss).
# Depends:    .config, .anchors, .erosion, .repair, .passes.
# Exposes:    the public 52 (in __all__) + the re-exported _fm. No integrity PRIVATE
#             is reached through the module namespace anywhere in the tree (verified),
#             so no private/patch-seam re-export is required (unlike the exemplars).
# Lines:      217
# --------------------------------------------------------------------------
