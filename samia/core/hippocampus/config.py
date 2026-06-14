"""samia.core.hippocampus.config — shared constants, path helpers, id/recency leaf.

Layer 1 (Owns / Depends):
    Owns:    the module-level surface the whole package reads — the engram recency
             bars (RECENCY_BOOST_DEFAULT / RECENCY_HALFLIFE_DAYS / ENGRAM_TTL_DAYS_
             DEFAULT), the ring bounds (RING_CAPACITY_DEFAULT / RING_TTL_HOURS_DEFAULT),
             the P3 promotion-lattice constants (RING_PROMOTE_HITS / SALIENCE_PROMOTE_
             THRESHOLD / STC_PROMOTE_THRESHOLD / INJECT_PROMOTE_THRESHOLD / RING_EVICT_
             WANT_SALIENCE), the P4 inject budgets (INJECT_BUDGET_DEFAULT / INJECT_
             ENGRAM_BUDGET_FRAC), the directory-layout helpers (_hippocampus_dir /
             _engram_dir / _engram_index_dir / _engram_embed_path / _engram_manifest_
             path / _ring_path), and the small side-effect-free primitives every
             sibling shares (_now_iso, _engram_id, _ptr_name, _recency_factor).
             Re-exports the embedding/vector backend module (_vi) so the carve binds
             it ONCE and every sibling imports it THROUGH this owner.
    Depends: samia.core.vector (the embedding backend + canonical reader + index
             layout — re-exported as _vi).  datetime/hashlib/json/pathlib/numpy from
             stdlib + numpy.

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — it imports nothing from its
          siblings, so the bars/paths/primitives live in one place and are never
          duplicated.  _engram_id / _hippocampus_dir / _engram_embed_path /
          _engram_manifest_path are HERE (not in the engram submodule) because the
          temporal-recall sidecar (temporal_recall_sith) and the targeted tests reach
          them THROUGH the package facade; concentrating them in the leaf keeps the
          public reach-in surface single-sourced and the import graph acyclic.
    Why:  splitting the 1339-line monolith by responsibility (engram store, ring
          store, promotion lattice, inject assembler) leaves a shared base of
          constants + path/id primitives that all four need; concentrating them here
          keeps the bars single-sourced and config depends on nothing in the package.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json  # noqa: F401  (re-exported public surface)
from pathlib import Path  # noqa: F401  (re-exported public surface)

import numpy as np  # noqa: F401  (re-exported public surface)

from .. import vector as _vi  # noqa: F401  (re-exported: the shared embedding backend)

# RECENCY_BOOST_DEFAULT — What: per-query additive bias applied to an engram hit's
#   cosine score, scaled by how recently the copy was materialized/accessed.
# Why: the hippocampal fast tier is preferential-by-recency (CLS); P1's exit gate
#   ("an engram copy wins over an equal-cosine main node") needs the fast-tier hit to
#   out-score a tied main hit. A small additive boost (default 0.05, recency-scaled)
#   does that without distorting genuinely-stronger main cosine hits. Named/tunable.
RECENCY_BOOST_DEFAULT = 0.05

# RECENCY_HALFLIFE_DAYS — What: age (days) at which the recency boost halves.
# Why: a copy materialized today gets the full boost; one materialized long ago gets
#   little — recency-preferential, not blanket fast-tier preference.
RECENCY_HALFLIFE_DAYS = 14.0

# ENGRAM_TTL_DAYS_DEFAULT — What: default TTL stamped on a held copy at materialization.
# Why: the engram tier is days-to-months (D1); the TTL is recorded in P1 but the
#   demotion sweep that ACTS on it is P5 (not built here). Stamped so P5 has the field.
ENGRAM_TTL_DAYS_DEFAULT = 90

# RING_CAPACITY_DEFAULT — What: max live pointers the ring holds before LRU drops
#   the least-recently-accessed entry.
# Why: the ring is volatile working memory (D1), bounded so it stays a hot small set
#   rather than an unbounded second copy of the corpus. Named/tunable. promote-before-
#   evict (a salient pointer materialized before it can drop) is P3 — P2 just bounds.
RING_CAPACITY_DEFAULT = 256

# RING_TTL_HOURS_DEFAULT — What: age (hours) past which a ring pointer is considered
#   stale and skipped by resolve/ring-RAG (volatility, D1).
# Why: the ring tier is hours, not days; an old pointer is no longer "working memory".
#   The automatic sweep that prunes stale entries is later — P2 only honors the TTL on
#   read (a stale pointer simply resolves to nothing), so the field is stamped/tunable.
RING_TTL_HOURS_DEFAULT = 6.0


# ── P3 promotion-lattice constants (FEAT-2026-06-07 Tier-1 P3, D3 + D6) ──────
#
# RING_PROMOTE_HITS — What: the count of GENUINE ring-RAG hits at which a ring
#   pointer auto-materializes to an engram copy (the frequency/recency arm of the
#   ring->engram lattice, D3/Q3a).
# Why: a pointer that keeps being genuinely recalled has EARNED a held copy (the
#   consolidation event). Default 3 mirrors HEBB_PROMOTE_REPEATS so the fast-tier
#   promotion cadence matches Tier-0's attractor cadence. Named/tunable.
RING_PROMOTE_HITS = 3

# SALIENCE_PROMOTE_THRESHOLD — What: the salience at/above which a ring pointer
#   materializes to engram WITHOUT the RING_PROMOTE_HITS frequency bar (the D6 one-
#   shot shortcut: promotion gate = max(access_signal, salience)).
# Why: D6 effect (i) — a high-salience one-shot (a rare critical realization) must
#   earn durability without repetition. A pointer whose salience clears this bar
#   promotes immediately. HIGH named tunable so only the genuine top tier shortcuts.
SALIENCE_PROMOTE_THRESHOLD = 0.8

# STC_PROMOTE_THRESHOLD — What: the (attenuated) stc_capture_score at/above which a weak
#   ring pointer is promotion-eligible via the OR-gate's third arm (FEAT-2026-06-11 P4).
# Why: §6.5 effect 2 — a weak node carrying a high capture score is rescued into the
#   engram store by its strong neighbour WITHOUT the frequency or its own salience bar.
#   Mirrors temporal_recall_stc.STC_PROMOTE_THRESHOLD; defined locally so promote_ring_*
#   keeps a default without a top-level cross-import. Inert when the temporal flag is off
#   (no node carries the field -> the arm is False -> the gate is byte-identical).
STC_PROMOTE_THRESHOLD = 0.50

# INJECT_PROMOTE_THRESHOLD — What: the bar the engram->inject gate compares
#   max(attractor_strength, salience) against to mark an engram copy inject_eligible.
# Why: D6 effect (i) — the engram->inject gate is max(attractor, salience) >= this.
#   Set to HEBB_PROMOTION (0.85) so a frequency-earned attractor (w>=0.85) promotes
#   exactly as before AND a salience>=0.85 one-shot earns the standing slot without
#   the frequency bar. Named/tunable. (Inject ITSELF is P4; P3 only marks eligibility.)
INJECT_PROMOTE_THRESHOLD = 0.85

# RING_EVICT_WANT_SALIENCE — What: the salience at/above which a to-be-evicted ring
#   pointer is "still wanted" and is materialized before being dropped (promote-before-
#   evict, D3).
# Why: a salient pointer about to fall off the LRU must not be silently lost; it earns
#   a held copy first. Aligned with SALIENCE_PROMOTE_THRESHOLD (same "wanted" bar).
RING_EVICT_WANT_SALIENCE = SALIENCE_PROMOTE_THRESHOLD

# INJECT_BUDGET_DEFAULT — What: the FIXED total token budget the assembled inject block
#   must never exceed, split across the two layers (D4/Q5a).
# Why: inject is a context-window-EXTENSION into a finite window; a fixed cap keeps it
#   small and predictable (Risk 3: a budget overrun degrades the prompt). Default 600 per
#   the proposal D4. Named/tunable; the block is test-asserted to never exceed it.
INJECT_BUDGET_DEFAULT = 600

# INJECT_ENGRAM_BUDGET_FRAC — What: the fraction of INJECT_BUDGET reserved for the
#   always-on engram-inject identity set; ring-inject fills the remainder.
# Why: D4/Q5a — engram-inject is the durable, earned identity layer and is FAVORED under
#   budget pressure; ring-inject is turn-relevant working set that fills what is left. A
#   small reserved slice (default 0.4) keeps the identity set standing even when the ring
#   is busy, while leaving the majority for turn-relevant working memory. Named/tunable.
INJECT_ENGRAM_BUDGET_FRAC = 0.4


# ── directory layout helpers (single-sourced so every store agrees on paths) ──

def _hippocampus_dir(memory_dir: Path) -> Path:
    return Path(memory_dir) / "hippocampus"


def _engram_dir(memory_dir: Path) -> Path:
    return _hippocampus_dir(memory_dir) / "engram"


def _engram_index_dir(memory_dir: Path) -> Path:
    return _hippocampus_dir(memory_dir) / "engram_index"


def _engram_embed_path(memory_dir: Path) -> Path:
    return _engram_index_dir(memory_dir) / "embeddings.npy"


def _engram_manifest_path(memory_dir: Path) -> Path:
    return _engram_index_dir(memory_dir) / "manifest.json"


def _ring_path(memory_dir: Path) -> Path:
    return _hippocampus_dir(memory_dir) / "ring.jsonl"


# ── small side-effect-free primitives shared across the package ──────────────

def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _engram_id(source_node: str) -> str:
    """Stable, addressable id for the held copy of a source node.

    What: a deterministic id derived from the source node name (so re-materializing
      the same source updates the same held copy rather than spawning duplicates).
    Why: the engram copy must be ADDRESSABLE (D1) and idempotent under re-materialize;
      a hash of the source name gives both without a counter.
    """
    stem = source_node[:-3] if source_node.endswith(".md") else source_node
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]
    return f"engram_{digest}"


def _ptr_name(ptr: str) -> str:
    """Normalize a pointer to its canonical backing key.

    What: a ring pointer references a backing node — either a main node filename
      ('foo.md') or an engram id ('engram_<hash>'). Main pointers are normalized to
      carry the .md suffix; engram ids are left as-is.
    Why:  the pointer must be a stable address into main/engram so touch/resolve key
      the SAME entry regardless of whether the caller passed the .md suffix.
    """
    if ptr.startswith("engram_"):
        return ptr
    return ptr if ptr.endswith(".md") else f"{ptr}.md"


def _recency_factor(materialized_at: str, halflife_days: float) -> float:
    """Return a [0,1] recency factor (1.0 today, halving every halflife_days)."""
    try:
        ts = _dt.datetime.fromisoformat(materialized_at)
    except (ValueError, TypeError):
        return 0.0
    age_days = max(0.0, (_dt.datetime.now() - ts).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / max(halflife_days, 1e-9)))


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.hippocampus.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.hippocampus monolith during
#             modularization (the leaf of the package DAG — the shared base the 1339-
#             line monolith split into config/engram/ring/promotion/inject left behind).
# Layer:      core (pure library, no daemon dependency)
# Role:       shared base of the hippocampus package — the engram recency bars, the
#             ring bounds, the P3 promotion-lattice constants, the P4 inject budgets,
#             the hippocampus/ directory-layout helpers, the id/recency/pointer-name
#             primitives, and the re-exported vector backend (_vi) every sibling
#             imports through.
# Stability:  stable — pure constants + side-effect-free helpers; the carve changed
#             no value (bars/paths/ids byte-identical to the monolith).
# ErrorModel: none — _recency_factor is fail-soft (a malformed timestamp -> 0.0);
#             the path/id helpers and _now_iso never raise.
# Depends:    datetime, hashlib, json, pathlib, numpy (stdlib + numpy).
#             samia.core.vector (re-exported as _vi).
# Exposes:    RECENCY_BOOST_DEFAULT/RECENCY_HALFLIFE_DAYS/ENGRAM_TTL_DAYS_DEFAULT,
#             RING_CAPACITY_DEFAULT/RING_TTL_HOURS_DEFAULT, the P3 thresholds, the P4
#             budgets, the _*_dir/_*_path helpers, _now_iso/_engram_id/_ptr_name/
#             _recency_factor, _vi, np, json, Path, hashlib.
# Lines:      227
# ─────────────────────────────────────────────
