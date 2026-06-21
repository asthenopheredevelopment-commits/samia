"""samia.core.bio.config — shared base of the biomimetic package.

Layer 1 (Owns / Depends):
    Owns:    the module-top stdlib + numpy the monolith pulled in and that callers
             + tests reach THROUGH the package facade (json/os/sys/hashlib, the
             `datetime as _dt` + `time as _time` aliases, Path, Optional, numpy as np,
             the `from __future__` annotations); the `from . import chain as _chain`
             aliased dependency (single-owned, imported through config by every arm
             that promotes/loads chains); EVERY tuning constant (the pattern bar, the
             Hebbian attractor/decay/prune bars + the derived alpha + the replay
             regulators, the replay/interleave defaults, the recency/schema windows,
             the min-interval env-var NAME, the active-set hot-N, the salience weights
             + saturation + tag + merge-guard floor, and the kWTA projection dim/frac/
             seed); the SINGLE-OWNED mutable module state (_KWTA_PROJ_CACHE, the per-
             (in_dim, proj_dim, seed) random-projection basis cache); and the shared
             path helper _bio_paths that resolves every biomimetic/* file the arms
             read and write.
    Depends: samia.core.chain (the chain store, aliased _chain). numpy. json/os/sys/
             hashlib/datetime/time/pathlib/typing from stdlib.

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — every sibling submodule imports
          its stdlib aliases, numpy, _chain, the tuning constants, _bio_paths, and
          (for the kWTA arm) the single _KWTA_PROJ_CACHE from here, so there is ONE
          copy of each. The names the baseline records as the public re-exported
          imports (Optional/Path/annotations/hashlib/json/np/os/sys) live here and are
          re-exported by the package facade unchanged.
    Why:  splitting the 1756-line monolith by RESPONSIBILITY (pattern / hebbian /
          reconsolidate / replay / salience / schema) leaves a shared base of imports
          + the dependency aliases + the tuning bars + the path helper + the
          projection cache that all the arms need; concentrating them here keeps the
          import graph acyclic (config depends only on samia.core.chain, never on a
          sibling) and the tuning bars single-sourced. The heavier/cyclic deps
          (vector / temporal / fact_extractor / web_store / hippocampus / contradiction
          / temporal_recall_*) are NOT imported here — each arm imports them lazily
          (function-local) exactly as the monolith did, to keep `import bio` cheap and
          to break the import cycles (mcp_server -> bio -> vector/temporal; bio <->
          temporal_recall_stc/sith; bio <-> contradiction).
"""

from __future__ import annotations

# Re-exported module-top names the monolith pulled in and other code (importers +
# tests) reaches THROUGH the package facade. The baseline records hashlib/json/np/os/
# sys/Path/Optional/annotations as part of the public surface, so they must stay
# importable from the package facade — they are owned here. _dt / _time are the
# private datetime/time aliases the monolith used; single-owned here too.
import datetime as _dt  # noqa: F401
import hashlib  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import sys  # noqa: F401
import time as _time  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Optional  # noqa: F401

import numpy as np  # noqa: F401

# _chain — What: the aliased samia.core.chain dependency (load/save/member_addrs/
#   add_edge). Single-owned here so the hebbian promotion loop, reconsolidate, and
#   chain_maturity all import the SAME alias through config (never a per-arm copy).
# Why: it is the only non-lazy intra-package dependency the monolith imported at top;
#   concentrating it in the leaf keeps the DAG acyclic and the alias single-sourced.
from samia.core import chain as _chain  # noqa: F401


# ---------------------------------------------------------------------------
# 1. Pattern separation
# ---------------------------------------------------------------------------

PATTERN_THRESHOLD_DEFAULT = 0.85

# ---------------------------------------------------------------------------
# 2. Hebbian co-activation — attractor bar + reachability + replay regulators
# ---------------------------------------------------------------------------

# Hebbian attractor bar + reachability (FEAT-2026-06-05 Tier-0 D2).
# What: HEBB_PROMOTION is the attractor/promotion bar; HEBB_PROMOTE_REPEATS is how
#   many GENUINE co-activations of a pair should cross it; HEBB_EMA_ALPHA is DERIVED
#   so exactly that many full-weight events reach the bar (w_K = 1-(1-alpha)^K).
# Why: the audit found alpha=0.3 needed 6 consecutive repeats to reach 0.85 — far
#   more than co-activations ever accrue — so no edge ever promoted (max w 0.832).
#   Deriving alpha from the intended repeat count makes the bar reachable WITHOUT
#   lowering it (keeping 0.85 semantically meaningful).
HEBB_PROMOTION = 0.85
HEBB_PROMOTE_REPEATS = 3
# Derive alpha so HEBB_PROMOTE_REPEATS genuine events land just PAST the bar. The small
# margin (0.005) keeps exactly-K repeats robustly promotable instead of sitting on a
# floating-point knife-edge at the threshold; one fewer repeat stays well below.
HEBB_EMA_ALPHA = 1.0 - (1.0 - min(0.999, HEBB_PROMOTION + 0.005)) ** (
    1.0 / HEBB_PROMOTE_REPEATS)
HEBB_DECAY = 0.005
HEBB_PRUNE = 0.05

# Homeostatic replay regulation (FEAT-2026-06-05 Tier-0 D1).
# What: replay-derived co-activations contribute at HEBB_REPLAY_COACT_WEIGHT of a
#   genuine event, NEVER refresh the decay clock (last_seen), and a replay-ONLY edge
#   (zero genuine co-activations) is both capped below the bar (REPLAY_ONLY_W_CEILING)
#   and barred from promotion (genuine-count gate). HEBB_SEED_MARGIN keeps the
#   one-time count->w re-seed just below the bar so migration never auto-promotes.
# Why: replay deterministically re-discovers the same pairs every pulse; at full
#   weight it would reset the decay clock and saturate the web (runaway recurrent
#   excitation / feedback reverberation, unprunable edges). These regulators let
#   replay ACCELERATE a genuinely-recent pair toward the bar without manufacturing
#   or immortalizing a stale one. (operator-flagged 2026-06-05; see D1.)
HEBB_REPLAY_COACT_WEIGHT = 0.5
HEBB_SEED_MARGIN = 0.02
REPLAY_ONLY_W_CEILING = HEBB_PROMOTION - HEBB_SEED_MARGIN
REPLAY_NEIGHBOR_THRESHOLD = 0.55
REPLAY_DEFAULT_SAMPLE = 20
INTERLEAVE_THRESHOLD = 0.40
INTERLEAVE_DEFAULT_COLD_PER_HOT = 3
HOT_RECENCY_DAYS = 7
SCHEMA_MIN_NODES = 4
SCHEMA_MIN_AGE_DAYS = 7

# HEBB_MIN_INTERVAL_ENV — What: name of an OPTIONAL env var (seconds) that
#   self-gates how often hebbian_consolidate actually drains the log.
# Why: consolidation is wired onto the per-tool PostToolUse idle pulse
#   (IDLE_THRESHOLD_SECONDS=30) AND a 600s scheduler job, so it fires far more
#   often than co-activations accrue. A min-interval gate decouples the
#   consolidation cadence from the hot pulse WITHOUT a workaround in the trigger
#   wiring. Default 0 (unset) preserves legacy every-pulse behavior, so this
#   module is safe to land before the operator sets the env var. Cadence policy
#   is operator-owned in settings.json (see the operator-paste diff).
HEBB_MIN_INTERVAL_ENV = "ASTHENOS_HEBB_MIN_INTERVAL_S"

# ---------------------------------------------------------------------------
# FEAT-2026-06-18 Hebbian edge-quality — stop-node exclusion + significance gate
# ---------------------------------------------------------------------------

# HEBB_STOPNODE_PREFIXES — What: filename-prefix fragments for low-value,
#   high-cardinality nodes that must NEVER form Hebbian co-activation edges.
#   `session_*_offload` = the episodic session-offload dumps; `sem_` = the
#   semantic session-event nodes (sem_youtube_video_accessed…). A node id is a
#   STOP node when its (lowercased, .md-stripped) stem startswith any of these.
# Why: the 2026-06-18 red-team found these saturate as degree-32 hubs whose 32
#   neighbors are all other session-event nodes (the GAAMA mega-hub effect,
#   arXiv:2603.27910) — the "everything-in-a-retrieval-window co-occurs" raw-
#   co-occurrence pathology. They are already filtered from the entity-bridge
#   and the active-set (contradiction.is_excluded_node) but NOT from
#   hebbian_record, so they still pollute edge formation. Filtering them out
#   BEFORE the O(n²) all-pairs is the cheapest, highest-precision win (P2).
# AMENDMENT 2026-06-20 (sparseness heal, explosion-checked): `sem_` REMOVED from the default.
#   The blanket sem_ exclusion FROZE the semantic-concept layer's Hebbian wiring (0 new sem_
#   co-activations after 2026-06-18) — the dominant ongoing SPARSENESS cause (sem_ = ~70% of the
#   store). Live evidence REFUTES the sem_ mega-hub fear: in the pre-filter genuine web, sem_ nodes
#   wired at mean-degree 9.5 / max 39 / p95 20 / ZERO above 50 — LOWER than the NON-excluded `named`
#   concept nodes (mean 17.8, max 62). The true degree-300-518 mega-hubs are `session_*_offload`
#   dumps (KEPT excluded). The `_sel` degree penalty (balancing._sel) STRUCTURALLY self-limits any
#   node: as degree grows, score -> 0 -> below EPI_MAT_FLOOR -> no new edge (this is why historical
#   sem_ capped at 39, not 300). So `_sel` + EPI_MAT_FLOOR + the recall top-k cap (<=12 nodes ->
#   <=66 pairs/recall) + w/S decay are the real, intact explosion guards — no edge/strength
#   explosion from re-including sem_ (empirically bounded web = ~5.7k edges). REVERSIBLE: set
#   ASTHENOS_HEBB_STOPNODE_PREFIXES="session_,sem_" to restore the old behavior instantly.
HEBB_STOPNODE_PREFIXES = ("session_",)

# HEBB_STOPNODE_PREFIXES_ENV — What: env override (comma-separated prefixes) for
#   the stop-node prefix list, so the exclusion is tunable WITHOUT a code edit.
# Why: operator-owned tuning (matches the HEBB_MIN_INTERVAL_ENV / contradiction
#   ASTHENOS_CONTRADICTION_EXCLUDE_TYPES pattern). Unset -> the default tuple
#   above; set -> replaces it wholesale (read live so a daemon that sets the env
#   after import sees the change). `session_` is matched with the extra
#   "offload"-substring guard in hebbian.py so a non-offload session_* node is
#   not over-excluded; bare prefixes here are matched as plain startswith.
HEBB_STOPNODE_PREFIXES_ENV = "ASTHENOS_HEBB_STOPNODE_PREFIXES"

# HEBB_LIFT_GATE_ENABLED_ENV / HEBB_LIFT_MIN_ENV — What: the env flag (default
#   OFF) + threshold for the Phase-2 significance (lift/PMI) gate. lift =
#   C(ij)·N / (C(i)·C(j)); form/weight an edge only when lift > HEBB_LIFT_MIN.
# Why: P1/P4 — raw co-occurrence has NO significance gating, so high-frequency
#   spurious pairs persist while genuine low-frequency pairs decay out (a
#   selection-pressure inversion). lift>2.0 is the standard min-significance cut
#   (Church&Hanks'90, Agrawal&Srikant'94, Levy&Goldberg'14). DEFAULT OFF: the
#   marginal-count store is populated on every consolidation so the gate has data
#   to act on the moment the operator flips the flag, but live edge formation is
#   UNCHANGED until then (ship the substrate, gate the behavior).
HEBB_LIFT_GATE_ENABLED_ENV = "ASTHENOS_HEBB_LIFT_GATE"
HEBB_LIFT_MIN_ENV = "ASTHENOS_HEBB_LIFT_MIN"
HEBB_LIFT_MIN_DEFAULT = 2.0

# HEBB_SATURATE_ENABLED_ENV — What: env flag (default OFF) for the Phase-3
#   saturating weight update (per-node L2 synaptic scaling, Turrigiano'98).
# Why: P3 — the plain EMA toward 1.0 lets busy nodes' edges all saturate near 1;
#   per-node normalization bounds total outgoing weight so weights converge below
#   1 and compete instead of all maxing out. DEFAULT OFF — it touches the live
#   consolidation weight path, so it ships behind a flag for A/B on edge precision.
HEBB_SATURATE_ENABLED_ENV = "ASTHENOS_HEBB_SATURATE"
HEBB_SATURATE_TARGET_DEFAULT = 4.0  # per-node total-outgoing-weight budget (L2-ish cap)

# ---------------------------------------------------------------------------
# FEAT-2026-06-07 P3b — the ONLINE active-set (bounded supersession locus)
# ---------------------------------------------------------------------------

# ACTIVE_SET_HOT_N — What: how many hot/recently-accessed nodes join the locus.
# ACTIVE_SET_HOT_N — Why: the online active-set is "what fires together + what's
#   in working memory"; a small hot/recent top-N keeps it bounded (Q1a + Risk 3).
ACTIVE_SET_HOT_N = 16

# ---------------------------------------------------------------------------
# FEAT-2026-06-07 Tier-1 P2 (D6) — the salience / affective axis constants
# ---------------------------------------------------------------------------

# SALIENCE_W_SURPRISE / _CONTRADICTION / _REPETITION — What: the composite weights
#   for the three signal-derived salience components (D6 Q8a).
# Why: a node's salience is a weighted blend of surprise (novelty vs the index),
#   contradiction-involvement, and repetition. The weights sum to 1.0 so the composite
#   is already in [0,1] before the explicit-tag override; surprise is weighted highest
#   because the one-shot eureka the hippocampus must retain is novelty-driven, not
#   frequency-driven (the whole point of the orthogonal salience axis). Named/tunable.
SALIENCE_W_SURPRISE = 0.5
SALIENCE_W_CONTRADICTION = 0.3
SALIENCE_W_REPETITION = 0.2

# SALIENCE_REPETITION_SATURATION — What: the access/co-activation count at which the
#   repetition component saturates to 1.0.
# Why: repetition is a SMALL, saturating contribution (D6 Q8a: "salience is NOT
#   reducible to frequency"); a handful of accesses is enough to max its 0.2 slice,
#   so frequency never dominates the surprise-led composite. Named/tunable.
SALIENCE_REPETITION_SATURATION = 5.0

# SALIENCE_TAG_VALUE — What: the salience an explicit operator/agent tag clamps to.
# Why: the explicit tag is the deliberate "this matters" HIGH-PRIORITY override (D6
#   Q8a) — it pins salience near 1.0 regardless of the composite. Named/tunable.
SALIENCE_TAG_VALUE = 0.95

# SALIENCE_MERGE_GUARD_DEFAULT — What: the salience floor at/above which the merge/
#   supersede guard fires (a distinct high-salience memory is surfaced, not absorbed).
# Why: D6 effect (iii) — a HIGH named tunable so only the genuine top tier is guarded
#   (Risk 8: salience inflation). The guard is DEFINED here and CONSUMED by the
#   contradiction/merge proposals; the guard itself is a pure read-only predicate.
SALIENCE_MERGE_GUARD_DEFAULT = 0.8

# ---------------------------------------------------------------------------
# FEAT-2026-06-07 Tier-1 P3 (D2) — kWTA pattern separation constants + state
# ---------------------------------------------------------------------------

# KWTA_PROJ_DIM — What: the higher-dim space the embedding is random-projection
#   lifted into before the top-k% winner-take-all.
# Why: a high-dim sparse code is what makes near-duplicate inputs land on largely
#   DISJOINT winner sets (sparse high-dim codes are nearly orthogonal — the
#   expander/pattern-separation property the dentate gyrus exploits). 1024 over the
#   384-dim MiniLM embedding gives ample room for ~2-5% sparse separation. Named.
KWTA_PROJ_DIM = 1024

# KWTA_FRAC_DEFAULT — What: the fraction of projected units kept active (the k of
#   kWTA), default 0.03 (3%, inside the 2-5% band, D2).
# Why: sparse enough that near-duplicates separate, dense enough that the code still
#   carries the episode's identity. Named/tunable; clamped to >=1 winner.
KWTA_FRAC_DEFAULT = 0.03

# KWTA_SEED — What: the FIXED seed for the random-projection matrix.
# Why: determinism — the SAME embedding must always yield the SAME sparse key (so a
#   re-materialize updates rather than forks the code, and tests are reproducible).
#   The projection is a fixed random basis, generated once per (in_dim, proj_dim).
KWTA_SEED = 1729

# _KWTA_PROJ_CACHE — What: the per-(in_dim, proj_dim, seed) random-projection basis
#   cache. SINGLE-OWNED here so the kWTA arm mutates ONE dict in place.
# Why: a fixed random projection makes the sparse code a deterministic function of the
#   input; caching avoids regenerating the basis on every materialize. Module-level
#   mutable state, so it lives in the leaf (one copy) per the package convention.
_KWTA_PROJ_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


# ---------------------------------------------------------------------------
# Shared path helper — every arm resolves the biomimetic/* files through here
# ---------------------------------------------------------------------------


def _bio_paths(memory_dir: Path) -> dict:
    bio_dir = memory_dir / "biomimetic"
    return {
        "bio_dir": bio_dir,
        "hebb_log": bio_dir / "coactivation_log.jsonl",
        # hebb_log_processing — What: exclusive tempfile the consumer drains from.
        # Why: ATOMIC DRAIN. hebbian_consolidate os.replace()s the live log onto
        #   this path up front; concurrent hebbian_record appends then land on a
        #   FRESH live log and survive. Closes the truncate lost-update window.
        "hebb_log_processing": bio_dir / "coactivation_log.jsonl.processing",
        "edge_weights": bio_dir / "edge_weights.json",
        "reconsolidate_log": bio_dir / "reconsolidation_log.jsonl",
        "replay_proposals": bio_dir / "replay_proposals.json",
        "replay_interleaved_proposals": bio_dir / "replay_interleaved_proposals.json",
        # hebb_consolidate_state — What: persists the last consolidation unix ts.
        # Why: backs the optional min-interval cadence gate (HEBB_MIN_INTERVAL_ENV)
        #   so the consolidation cadence is decoupled from the per-tool idle pulse.
        "hebb_consolidate_state": bio_dir / "hebb_consolidate_state.json",
        # engram_replay_state — What: per-PAIR genuine-once ledger for engram
        #   replay (FEAT-2026-06-07 Tier-1 P5). Records which engram-derived pairs
        #   have already had their FIRST (genuine) consolidation replay so re-
        #   replays of the same pair log FRACTIONAL, never genuine.
        # Why: D5/Q6a genuine-once — one captured trace cannot be farmed into an
        #   attractor by repeated replay (first genuine + count_genuine bump; rest
        #   fractional then age). The ledger is the "already genuine-replayed" memory.
        "engram_replay_state": bio_dir / "engram_replay_state.json",
        # episode_transitions — What: the DIRECTED co-activation count matrix T_dir
        #   (FEAT-2026-06-11 temporal-recall P6, §5.5). Sparse map of directed keys
        #   "A->B" -> count, A->B incremented for each in-window co-activation pair
        #   where episode_seq(A) < episode_seq(B). Sibling of edge_weights.json (the
        #   undirected store, §5.2) — DIRECTED, not the sorted([a,b]) undirected key.
        # Why: the substrate for the phase-2 directed/forward SR (succession, not just
        #   diffusion). Produced offline inside idle_replay_tick (REM-gated), read query-
        #   locally by successor.py and row-normalized on the fly into T_dir; written
        #   under locked_update_json (incremented, never rebuilt). Bounded ≤ 2·|edges|.
        "episode_transitions": bio_dir / "episode_transitions.json",
        # marginal_counts — What: per-node co-activation MARGINAL counts C(i) plus
        #   the grand total N of co-activation EVENTS (FEAT-2026-06-18 Phase-2). Sibling
        #   of edge_weights.json. Shape: {"N": int, "counts": {node_id: int}}.
        # Why: the significance (lift/PMI) gate needs C(i)/C(j)/N to compute
        #   lift = C(ij)·N/(C(i)·C(j)). Populated on EVERY consolidation so the gate
        #   has data the moment the operator flips ASTHENOS_HEBB_LIFT_GATE on; the
        #   gate itself is default-OFF until then (substrate shipped, behavior gated).
        "marginal_counts": bio_dir / "coactivation_marginals.json",
    }


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.bio monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       shared base of the biomimetic package — the re-exported stdlib + numpy
#             (json/os/sys/hashlib/_dt/_time/Path/Optional/np), the aliased _chain
#             dependency, EVERY tuning constant (pattern/Hebbian/replay/interleave/
#             recency/schema/active-set/salience/kWTA), the single-owned _KWTA_PROJ_CACHE
#             projection-basis cache, and the shared _bio_paths file resolver.
# Stability:  stable — pure constants + the path helper; the implementation arms
#             (pattern / hebbian / reconsolidate / replay / salience / schema) all
#             import their shared surface from here so it is never duplicated.
# ErrorModel: none here (constants + a pure path-builder). The heavy/cyclic deps
#             (vector / temporal / fact_extractor / web_store / hippocampus /
#             contradiction / temporal_recall_*) are imported lazily inside each arm,
#             never at this leaf, to keep `import bio` cheap and break import cycles.
# Depends:    samia.core.chain (aliased _chain). numpy + json/os/sys/hashlib/datetime/
#             time/pathlib/typing (stdlib).
# Exposes:    PATTERN_THRESHOLD_DEFAULT, HEBB_* (+ derived alpha + replay regulators),
#             REPLAY_*/INTERLEAVE_*, HOT_RECENCY_DAYS, SCHEMA_MIN_*,
#             HEBB_MIN_INTERVAL_ENV, ACTIVE_SET_HOT_N, SALIENCE_*, KWTA_* (public);
#             _dt, _time, _chain, _KWTA_PROJ_CACHE, _bio_paths (single-owned, imported
#             through config by the sibling arms).
# Lines:      276
# Updated:    2026-06-14
# Status:     active
# --------------------------------------------------------------------------
