"""samia.core.context_extension.config — shared base of the context-extension package.

Layer 1 (Owns / Depends):
    Owns:    the module-top stdlib the monolith pulled in and that callers + tests
             reach THROUGH the package facade (json/os/hashlib/sqlite3/time, datetime
             as _dt, Path, numpy as np, the `from __future__` annotations); the aliased
             dependency modules the monolith imported once and every arm reuses (_bio,
             _ct, _tq, _vi, _ws, and the optional _ei / _vic); EVERY tuning constant
             (the byte-per-token + budget defaults, the episodic-transition bars, the
             idle threshold, the read-seam top-N default + env, the SM-2 seed/sweep
             caps, the TC cosine floor, and the whole temporal-recall weight block);
             the SINGLE-OWNED mutable module state (_ATOM_CHAIN_CACHE, the lazy reranker
             singleton _RERANKER + its name); and the low-level shared path/IO/vector
             helpers every arm calls (_nodes_dir / _chains_dir / _ctx_dir +
             _frozen_prefix_path / _idle_state_path, _tok_estimate / _node_text /
             _read_fm / _read_full_fm, the _vi_* vector-module shims, the atom-chain
             cache + classifier, and the lazy reranker accessor).
    Depends: samia.core.bio / chain / temporal / vector / web_store (aliased here),
             optionally samia.core.entity_index / vector_contextual (import-guarded),
             and lazily samia.core.semantic_recall (function-local in _is_atom_chain to
             break the context_extension<->semantic_recall cycle). numpy + json/os/
             hashlib/sqlite3/time/datetime/pathlib from stdlib.

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — every sibling submodule imports
          its constants, its aliased dependency modules, its path/IO/vector helpers,
          and the single-owned module state from here, so there is ONE copy of each.
    Why:  splitting the 1959-line monolith by responsibility (temporal / readseam /
          retrieval / primitives / replay / scheduling) leaves a shared base of imports
          + aliased deps + constants + helpers + the atom-chain/reranker state that all
          the arms need; concentrating them here keeps the import graph acyclic (config
          depends on nothing else in the package) and the tuning bars single-sourced.

STATE SINGLE-OWNERSHIP (exemplar rule): _ATOM_CHAIN_CACHE and the lazy _RERANKER
    singleton are owned here. _clear_atom_chain_cache() is reached on the package facade
    by tests (cx._clear_atom_chain_cache()); it mutates this one cache in place, so the
    facade re-exports the SAME object the siblings read/write. No test rebinds these as
    bare names, so a plain re-export (not a facade-reach) suffices.
"""

from __future__ import annotations

# Re-exported module-top names the monolith pulled in and other code (importers +
# tests) reaches THROUGH the package facade. The baseline records json/os/hashlib/
# sqlite3/time/Path/np/annotations as part of the public surface, so they must stay
# importable from the package facade — they are owned here. _dt is private (the public
# surface carries `np`, not `datetime`), but every arm needs it, so it lives here too.
import datetime as _dt  # noqa: F401
import hashlib  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import sqlite3  # noqa: F401
import time  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401

# Aliased dependency modules — imported ONCE here and reused by every arm. The monolith
# pulled these in at module top; concentrating them in the leaf keeps the import graph
# acyclic and means a sibling reads the same module object every other sibling does.
from .. import bio as _bio  # noqa: F401
from .. import chain as _ct  # noqa: F401
from .. import temporal as _tq  # noqa: F401
from .. import vector as _vi  # noqa: F401
from .. import web_store as _ws  # noqa: F401

# Optional deps — import-guarded so a partial install (no entity index / no contextual
# vector index) degrades to None rather than failing the whole package import.
try:
    from .. import entity_index as _ei  # noqa: F401
except ImportError:
    _ei = None
try:
    from .. import vector_contextual as _vic  # noqa: F401
except ImportError:
    _vic = None

# Rough average bytes per token for English markdown — close enough for
# budgeting without invoking a tokenizer in-process.
BYTES_PER_TOKEN = 3.6
DEFAULT_BUDGET_TOKENS = 8000

EPISODIC_AGE_DAYS = 30
EPISODIC_MIN_SIBLINGS = 3
EPISODIC_SIM_THRESHOLD = 0.55

IDLE_THRESHOLD_SECONDS = 30

# READ_SEAM_TOP_N — What: default cap on cross-chain failure/diagnosis
#   associations surfaced by chainogram_retrieve when
#   include_failure_associations=True.
# Why: bounds output size and query work; overridable per-call or via env var
#   ASTHENOS_READ_SEAM_TOP_N so the operator can tune without code changes.
READ_SEAM_TOP_N_DEFAULT = 5
READ_SEAM_TOP_N_ENV = "ASTHENOS_READ_SEAM_TOP_N"


# ===========================================================================
# Temporal-recall scaffold constants (FEAT-2026-06-11 P1, proposal §2 + §8.6)
# ---------------------------------------------------------------------------
# What: the master flag env name + per-term weight env names + the relevance gate
#   theta + the compute-skip epsilon + the TC-specific cosine floor. The weight
#   READERS live in temporal.py; the constants are single-owned here so both
#   temporal.py and any test that reads ce.TEMPORAL_THETA / ce.TEMPORAL_WEIGHT_EPSILON
#   through the facade see the same values.
# Why: every new coefficient is pinned to 0.0 and behind a default-OFF master flag, so
#   the flag-off path is byte-identical to today's S_c + 0.05·H_c accumulation
#   (§2.6 identity proof).
# ===========================================================================

# Master deploy gate. Default OFF — mirrors semantic_recall.semantic_arm_enabled
#   / integrity's os.environ.get(..., "0") == "1" reader idiom (§8.6). Read each
#   call so a test/daemon/adapter that sets the env after import sees it.
TEMPORAL_WEIGHT_ENV = "ASTHENOS_TEMPORAL_WEIGHT"

# Per-term weight env names. γ scales the additive SITH cue; λN/λK/λD scale the
#   multiplicative need/STC/distinctiveness modulators. ALL DEFAULT TO 0.0, so
#   even with the master flag ON the formula collapses to the baseline until a
#   calibration freezes non-zero values (§8.6 per-term ablation).
TEMPORAL_GAMMA_ENV = "ASTHENOS_TEMPORAL_GAMMA"
TEMPORAL_LAMBDA_N_ENV = "ASTHENOS_TEMPORAL_LAMBDA_N"
TEMPORAL_LAMBDA_K_ENV = "ASTHENOS_TEMPORAL_LAMBDA_K"
TEMPORAL_LAMBDA_D_ENV = "ASTHENOS_TEMPORAL_LAMBDA_D"

# Uniform relevance gate θ = 0.2 (§2.5, FORMULA Q4): a hit injects temporal
#   signal only if its own semantic cosine clears θ. Not re-applied to S/H (the
#   identity baseline). Frozen constant, not an env knob in P1.
TEMPORAL_THETA = 0.2

# Compute-skip epsilon (§16.2 Q5): a term whose |weight| < ε is gated off at the
#   per-query COMPUTE level (its hook is never called), not merely multiplied by
#   zero — a term that earns no lift costs nothing. With all weights at the 0.0
#   default this skips every term, so the flag-off path runs zero temporal code.
TEMPORAL_WEIGHT_EPSILON = 1e-9

TC_COSINE_FLOOR_DEFAULT = 0.4
TC_COSINE_FLOOR_ENV = "ASTHENOS_TEMPORAL_TC_COSINE_FLOOR"


# ---------------------------------------------------------------------------
# Path helpers — the memory_dir-relative layout every arm shares.
# ---------------------------------------------------------------------------


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _chains_dir(memory_dir: Path) -> Path:
    return memory_dir / "chains"


def _ctx_dir(memory_dir: Path) -> Path:
    return memory_dir / "context_extension"


def _frozen_prefix_path(memory_dir: Path) -> Path:
    return _ctx_dir(memory_dir) / "frozen_prefix.txt"


def _idle_state_path(memory_dir: Path) -> Path:
    return _ctx_dir(memory_dir) / "idle_state.json"


def _tok_estimate(text: str) -> int:
    return int(len(text) / BYTES_PER_TOKEN)


def _node_text(path: Path) -> tuple[int, str]:
    raw = path.read_text(encoding="utf-8")
    return _tok_estimate(raw), raw


def _read_fm(path: Path) -> tuple[dict, str]:
    fm_lines, body = _tq.read_node(path)
    fm: dict = {}
    for line in fm_lines:
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, body


def _read_full_fm(p: Path) -> tuple[list[str], str]:
    return _tq.read_node(p)


# Helpers to abstract vector-module path/query differences (plain vs
# contextual). Both library-plane modules share the same _manifest_path /
# query(memory_dir, ...) shape.

def _vi_manifest(vi_module, memory_dir: Path) -> Path:
    return vi_module._manifest_path(memory_dir)


def _vi_embed(vi_module, memory_dir: Path) -> Path:
    return vi_module._embed_path(memory_dir)


def _vi_query(vi_module, memory_dir: Path, query: str, top_k: int) -> list[dict]:
    return vi_module.query(memory_dir, query, top_k=top_k)


# _ATOM_CHAIN_CACHE — What: per-(memory_dir, chain_name) cache of "is this an atom
#   (semantic-population) chain?". Why: the fx_-skip gate (FEAT-2026-06-10 Q4a) is
#   consulted per hit during chain selection; resolving the chain's first-member type
#   once and caching keeps the gate O(1) on repeat. Only populated when the semantic arm
#   is ON (the unflagged path never calls _is_atom_chain), so flag-off behavior and the
#   cache are fully decoupled. SINGLE-OWNED here; tests reach _clear_atom_chain_cache
#   through the facade and it clears THIS object in place.
_ATOM_CHAIN_CACHE: dict[tuple[str, str], bool] = {}


def _clear_atom_chain_cache() -> None:
    """Drop the atom-chain classification cache (tests / after a chain's type changes)."""
    _ATOM_CHAIN_CACHE.clear()


def _is_atom_chain(memory_dir: Path, chain_name: str) -> bool:
    """True iff *chain_name* belongs to the SEMANTIC (atom) population.

    What: an "fx_"-prefixed chain id is an atom mini-chain by construction (O(1) prefix
      check). As a fallback for atom chains NOT carrying that prefix, resolve the chain's
      FIRST member file's `type` and treat the chain as atomic when it is "semantic".
      Cached per (memory_dir, chain_name).
    Why: the populations meet in the composer (semantic_recall.recall), so the episodic
      chainogram must not select atom chains when the semantic arm is on (Q4a). The cheap
      id-prefix check covers the produced atom chains; the member-type fallback is the
      robust backstop. Only called when the flag is ON.
    """
    # Cache hit short-circuit — resolved type is stable for a (dir, chain) pair until a
    # rebuild clears the cache, so a repeat lookup never re-touches disk.
    key = (str(memory_dir), chain_name)
    if key in _ATOM_CHAIN_CACHE:
        return _ATOM_CHAIN_CACHE[key]
    is_atom = False
    if chain_name.startswith("fx_"):
        is_atom = True
    else:
        # Member-type fallback — resolve the chain's first member's `type` via a
        # FUNCTION-LOCAL semantic_recall import (breaks the context_extension<->
        # semantic_recall cycle); any read error leaves is_atom False (fail-open).
        try:
            chain = _ct.load_chain(_chains_dir(memory_dir), chain_name)
            members = chain.get("members") or []
            first = members[0] if members else None
            f = first.get("file") if isinstance(first, dict) else None
            if f:
                from .. import semantic_recall as _sr
                if _sr._node_type(memory_dir, Path(f).name) == "semantic":
                    is_atom = True
        except Exception:
            is_atom = False
    _ATOM_CHAIN_CACHE[key] = is_atom
    return is_atom


# _RERANKER — What: lazily-constructed cross-encoder singleton + its model name.
# Why: the cross-encoder is heavy to load; the reranked retrieval arm builds it once on
#   first use and reuses it. Single-owned here so the one instance is shared and the
#   model-name constant is part of the public surface.
_RERANKER = None
_RERANKER_NAME = "BAAI/bge-reranker-base"


def _get_reranker():
    global _RERANKER
    # Build-once gate — the CrossEncoder load is expensive, so the first call
    # constructs it (device from SAM_RERANKER_DEVICE, default cpu) and caches it.
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        device = os.environ.get("SAM_RERANKER_DEVICE", "cpu")
        _RERANKER = CrossEncoder(_RERANKER_NAME, max_length=512, device=device)
    return _RERANKER


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.context_extension.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase-B modularization — the shared leaf of the context_extension
#             package (carved from the 1959-line monolith with ZERO behavior change).
# Layer:      core (pure library, no daemon dependency)
# Role:       dependency-DAG leaf — re-exported stdlib + numpy, the aliased dependency
#             modules (_bio/_ct/_tq/_vi/_ws/_ei/_vic), every tuning constant, the
#             single-owned module state (_ATOM_CHAIN_CACHE / _RERANKER), and the shared
#             path/IO/vector + atom-chain + reranker helpers.
# Stability:  stable — imported THROUGH this leaf by every sibling; depends on nothing
#             else in the package (acyclic).
# ErrorModel: fail-open — _is_atom_chain swallows read errors to False; the optional
#             _ei/_vic deps degrade to None on ImportError. No state is mutated at
#             import time.
# Depends:    samia.core.{bio,chain,temporal,vector,web_store}; optionally
#             entity_index / vector_contextual; lazily samia.core.semantic_recall.
# Exposes:    the re-exported imports + np, the aliased deps, the constants, the state,
#             and the shared helpers — all reached by siblings via `from . import config`.
# Lines:      296
# --------------------------------------------------------------------------
