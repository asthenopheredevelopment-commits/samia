"""samia.core.hippocampus — the Tier-1 hippocampal fast store (P1 engram + P2 ring + P3 lattice + P4 inject).

FEAT-2026-06-07-memory-tier1-hippocampal-quad-v01, Phases 1 + 2 + 3 + 4.

Layer 1 (Owns / Depends):
    Owns:    the four populations of the Tier-1 hierarchy, split by responsibility
             into cohesive submodules behind this re-export facade (the public import
             surface is byte-for-byte unchanged from the pre-split single module):
               - config    : shared constants (recency bars, ring bounds, P3
                             thresholds, P4 budgets), the hippocampus/ directory-layout
                             helpers, the id/recency/pointer-name primitives, and the
                             re-exported vector backend (_vi).
               - engram    : P1 — EngramStore (self-contained HELD COPIES + the
                             dedicated cosine fast index + the materialize() copy
                             primitive) and engram_rag_query (recency-preferential).
               - ring      : P2 — RingStore (the volatile capacity/LRU-bounded POINTER
                             store + dangling-safe resolve + promote-before-evict) and
                             ring_rag_query (deref the live pointers, cosine them).
               - promotion : P3 — the promotion lattice + its AUTO trigger
                             (promote_ring_pointer / promote_ring_step: ring->engram;
                             mark_inject_eligible / attractor_strength: engram->inject).
               - inject    : P4 — assemble_inject_block (the two inject layers under a
                             FIXED token budget, relevance-gated + priority-arbitrated,
                             CO-ACTIVATION-SILENT) + estimate_tokens.
    Depends: samia.core.vector (the embedding backend + canonical reader + index
             layout). Lazily (function-local, inert when off): samia.core.bio
             (kwta_sparse_code / the Tier-0 attractor signal / the salience field),
             samia.core.frontmatter (the source's written_at/episode_seq),
             samia.core.temporal_recall_sith (the SITH encode event),
             samia.core.temporal_recall_stc (the STC capture score).

Layer 2 (What / Why):
    What: an engram entry is a self-contained COPY (survives source churn); a RING
          entry is a POINTER (deref-at-read -> reflects current backing; dangling ->
          dropped). The promotion lattice consolidates a wanted ring pointer into an
          engram copy (max(genuine-hits, salience, stc)) and flags a held copy
          inject-eligible (max(attractor, salience)); the inject assembler builds the
          two-layer standing context under a fixed budget. Neither store moves/mutates
          the main canonical (loss-free).
    Why:  the 1339-line monolith was split by responsibility (config/engram/ring/
          promotion/inject) with ZERO behavior change; each subsystem is independently
          legible and the import graph is acyclic (config <- engram <- ring <-
          {promotion, inject}). This facade re-exports the FULL public surface so every
          importer (`from samia.core.hippocampus import X`) and every attribute reach-in
          (`hippocampus._hippocampus_dir` / `._engram_id` / `._engram_embed_path` /
          `._engram_manifest_path`, reached by temporal_recall_sith + the targeted
          tests THROUGH the package facade) is unaffected.

NOT here: feed-forward / genuine-once replay, salience-dampened DECAY, freeze-exemption
(P5 — P4 does NOT change decay). The SALIENCE SOURCE itself (bio.compute_salience) is
P2; P3 CONSUMES it; P4 CONSUMES the P3 inject_eligible flag. The per-turn live-prompt
injection stays operator-gated/INERT.

Public surface re-exported here (byte-for-byte the pre-split module):
    re-exported imports : Path, annotations, hashlib, json, np
    constants           : ENGRAM_TTL_DAYS_DEFAULT, INJECT_BUDGET_DEFAULT,
                          INJECT_ENGRAM_BUDGET_FRAC, INJECT_PROMOTE_THRESHOLD,
                          RECENCY_BOOST_DEFAULT, RECENCY_HALFLIFE_DAYS,
                          RING_CAPACITY_DEFAULT, RING_EVICT_WANT_SALIENCE,
                          RING_PROMOTE_HITS, RING_TTL_HOURS_DEFAULT,
                          SALIENCE_PROMOTE_THRESHOLD, STC_PROMOTE_THRESHOLD
    classes             : EngramStore, RingStore
    functions           : assemble_inject_block, attractor_strength, engram_rag_query,
                          estimate_tokens, mark_inject_eligible, promote_ring_pointer,
                          promote_ring_step, ring_rag_query
Internal names also re-exported for direct importer/test access (NOT in __all__):
    _hippocampus_dir, _engram_id, _engram_embed_path, _engram_manifest_path (reached by
    temporal_recall_sith + test_hippocampus + test_temporal_recall_sith THROUGH the
    package facade).
"""

from __future__ import annotations

# The shared constants + path/id primitives + the re-exported module-top names the
# monolith pulled in (Path/hashlib/json/np). `annotations` rides the `from __future__`
# above. _hippocampus_dir/_engram_id/_engram_embed_path/_engram_manifest_path are
# re-exported because temporal_recall_sith + the targeted tests reach them THROUGH this
# package (`from samia.core import hippocampus as _hip; _hip._engram_id(...)`).
from .config import (  # noqa: F401
    Path,
    hashlib,
    json,
    np,
    ENGRAM_TTL_DAYS_DEFAULT,
    RECENCY_BOOST_DEFAULT,
    RECENCY_HALFLIFE_DAYS,
    RING_CAPACITY_DEFAULT,
    RING_TTL_HOURS_DEFAULT,
    RING_PROMOTE_HITS,
    SALIENCE_PROMOTE_THRESHOLD,
    STC_PROMOTE_THRESHOLD,
    INJECT_PROMOTE_THRESHOLD,
    RING_EVICT_WANT_SALIENCE,
    INJECT_BUDGET_DEFAULT,
    INJECT_ENGRAM_BUDGET_FRAC,
    _hippocampus_dir,
    _engram_embed_path,
    _engram_manifest_path,
    _engram_id,
)

# P1 — the engram held-copy store + engram-RAG.
from .engram import EngramStore, engram_rag_query  # noqa: F401

# P2 — the ring POINTER store + ring-RAG.
from .ring import RingStore, ring_rag_query  # noqa: F401

# P3 — the promotion lattice + its AUTO trigger.
from .promotion import (  # noqa: F401
    attractor_strength,
    mark_inject_eligible,
    promote_ring_pointer,
    promote_ring_step,
)

# P4 — the two-layer standing inject block assembler.
from .inject import assemble_inject_block, estimate_tokens  # noqa: F401

# __all__ — the LOCALLY-owned PUBLIC names (the 27 the baseline records: the 5
# re-exported imports, the 12 constants, the 2 classes, the 8 functions). The verify
# script diffs the full public surface (dir() minus underscore names) against the
# baseline; __all__ documents the intended export set and bounds `from ... import *` to
# exactly the pre-split public 27. (The private test/importer-reached names above are
# re-exported but intentionally NOT in __all__, mirroring the exemplars.)
__all__ = [
    # re-exported imports
    "Path", "annotations", "hashlib", "json", "np",
    # constants
    "ENGRAM_TTL_DAYS_DEFAULT", "INJECT_BUDGET_DEFAULT",
    "INJECT_ENGRAM_BUDGET_FRAC", "INJECT_PROMOTE_THRESHOLD",
    "RECENCY_BOOST_DEFAULT", "RECENCY_HALFLIFE_DAYS", "RING_CAPACITY_DEFAULT",
    "RING_EVICT_WANT_SALIENCE", "RING_PROMOTE_HITS", "RING_TTL_HOURS_DEFAULT",
    "SALIENCE_PROMOTE_THRESHOLD", "STC_PROMOTE_THRESHOLD",
    # classes
    "EngramStore", "RingStore",
    # functions
    "assemble_inject_block", "attractor_strength", "engram_rag_query",
    "estimate_tokens", "mark_inject_eligible", "promote_ring_pointer",
    "promote_ring_step", "ring_rag_query",
]


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.hippocampus
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07-memory-tier1-hippocampal-quad-v01 P1 + P2 + P3 + P4
#             + FEAT-2026-06-11-temporal-recall P0/P2/P4 (the substrate lift on
#               materialize + the SITH encode event + the STC promotion OR-gate arm)
#             + Phase-B modularization: the 1339-line monolith carved into a
#               re-export-preserving package (config/engram/ring/promotion/inject) with
#               ZERO behavior change; this __init__ re-exports the full public surface
#               so every importer + attribute reach-in is unaffected.
# Layer:      core (pure library, no daemon dependency)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.core.hippocampus import X`
#             keeps working for all 27 public names; the private helpers the importers/
#             tests reach (_hippocampus_dir / _engram_id / _engram_embed_path /
#             _engram_manifest_path) are re-exported too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
# ErrorModel: none here (import-time wiring only); each submodule footer documents its
#             own fail-open / fail-soft / co-activation-silent posture.
# Depends:    .config, .engram, .ring, .promotion, .inject.
# Exposes:    the public 27 (in __all__) + _hippocampus_dir/_engram_id/
#             _engram_embed_path/_engram_manifest_path for the importers/tests.
# Note:       additive — an engram entry is a COPY (survives source churn), a ring entry
#             is a POINTER (deref-at-read, dangling -> dropped); neither moves/mutates
#             the main canonical (loss-free). assemble_inject_block is co-activation-
#             silent (feeds NOTHING to the Tier-0 web); the per-turn live-prompt
#             injection stays operator-gated. Activation is operator-gated (P5 builds
#             on the engram copies + the inject_eligible set).
# Lines:      171
# ─────────────────────────────────────────────
