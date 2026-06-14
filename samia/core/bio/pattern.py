"""samia.core.bio.pattern — pattern separation (cosine dedup gate + kWTA orthogonalize).

Layer 1 (Owns / Depends):
    Owns:    the two senses of pattern separation. (1) the cosine-threshold DEDUP gate
             at write time — _node_embedding (the manifest-keyed embedding lookup) and
             pattern_separation_decision (Marr 1971 / Yassa & Stark 2011: store-new vs
             merge-into by top-neighbor cosine). (2) the ORTHOGONALIZING sense (D2,
             FEAT-2026-06-07 Tier-1 P3) — _kwta_projection (the fixed seeded random
             basis, cached) and kwta_sparse_code (top-k% winner-take-all sparse code on
             the engram held copy, so two near-duplicate episodes stay individually
             addressable). The two jobs stay separate: kWTA tags the COPY, not the
             retrieval embedding.
    Depends: config (the constants PATTERN_THRESHOLD_DEFAULT / KWTA_* + the single-owned
             _KWTA_PROJ_CACHE + numpy as np); samia.core.vector (lazy, function-local —
             the manifest/embeddings + query seam). No sibling-arm dependency.

Layer 2 (What / Why):
    What: the write-time gate that decides whether new text becomes a new node or
          merges into its nearest neighbor, plus the materialization-time sparse code
          that orthogonalizes near-duplicate engram copies.
    Why:  carved out of the monolith as the "pattern separation" responsibility. The
          vector dep is lazy (function-local) exactly as the monolith had it — vector
          pulls heavier deps and mcp_server imports bio, so keeping it off the import
          path keeps `import bio` cheap.
"""

from __future__ import annotations

from typing import Optional

from . import config as _cfg
from .config import (
    np,
    PATTERN_THRESHOLD_DEFAULT,
    KWTA_PROJ_DIM,
    KWTA_FRAC_DEFAULT,
    KWTA_SEED,
)


# ---------------------------------------------------------------------------
# 1. Pattern separation — cosine dedup gate
# ---------------------------------------------------------------------------


def _node_embedding(memory_dir, name: str) -> Optional["np.ndarray"]:
    from samia.core import vector as _vec
    manifest_path = _vec._manifest_path(memory_dir)
    if not manifest_path.exists():
        return None
    m = _cfg.json.loads(manifest_path.read_text(encoding="utf-8"))
    e = m.get("entries", {}).get(name)
    if not e:
        return None
    embeddings = np.load(_vec._embed_path(memory_dir))
    return embeddings[e["row"]]


def pattern_separation_decision(memory_dir, text: str,
                                threshold: float = PATTERN_THRESHOLD_DEFAULT
                                ) -> dict:
    """Decide whether `text` should be stored as a new node or merged.

    Returns {"action": "store_new"|"merge_into", "target": name|None,
             "score": float, "neighbors": [{node, score}...]}.
    """
    from samia.core import vector as _vec
    hits = _vec.query(memory_dir, text, top_k=5)
    top = hits[0] if hits else None
    neighbors = hits
    if top and top["score"] >= threshold:
        return {"action": "merge_into", "target": top["node"],
                "score": float(top["score"]), "neighbors": neighbors,
                "threshold": threshold}
    return {"action": "store_new", "target": None,
            "score": float(top["score"]) if top else 0.0,
            "neighbors": neighbors, "threshold": threshold}


# ---------------------------------------------------------------------------
# FEAT-2026-06-07 Tier-1 P3 (D2) — kWTA pattern separation (orthogonalize-on-copy)
# ---------------------------------------------------------------------------
#
# This is the ORTHOGONALIZING sense of pattern separation (audit Tier-1 item 4):
# a sparse code computed ONCE at materialization, on the engram held copy, so two
# near-duplicate episodes are stored DISTINCTLY (individually addressable) even
# though the cosine dedup gate (pattern_separation_decision, above) still catches
# true duplicates. kWTA tags the COPY, not the retrieval embedding — retrieval
# cosine is unaffected. The two jobs (orthogonalize vs dedup) stay separate (D2).


def _kwta_projection(in_dim: int, proj_dim: int = KWTA_PROJ_DIM,
                     seed: int = KWTA_SEED) -> "np.ndarray":
    """Return the FIXED (deterministic) random-projection matrix [in_dim, proj_dim].

    What: a seeded Gaussian random matrix, cached per (in_dim, proj_dim, seed) so the
      SAME basis is reused for every code of a given embedding shape.
    Why: a fixed random projection makes the sparse code a deterministic function of
      the input (determinism requirement) while the random basis is what spreads
      near-duplicate inputs onto separable winner sets (orthogonalization). Caching
      avoids regenerating the basis on every materialize.
    """
    # Mutate the SINGLE-OWNED cache in config in place (config._KWTA_PROJ_CACHE) so
    # there is exactly one projection-basis cache shared across the process.
    key = (int(in_dim), int(proj_dim), int(seed))
    cached = _cfg._KWTA_PROJ_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(seed)
        cached = rng.standard_normal((in_dim, proj_dim)).astype(np.float32)
        _cfg._KWTA_PROJ_CACHE[key] = cached
    return cached


def kwta_sparse_code(embedding, frac: float = KWTA_FRAC_DEFAULT,
                     proj_dim: int = KWTA_PROJ_DIM,
                     seed: int = KWTA_SEED) -> list[int]:
    """kWTA sparse code of an embedding (D2 — orthogonalize-on-materialize).

    What: random-projection lift `embedding` into a `proj_dim`-dim space via a FIXED
      seeded basis, then keep the indices of the top `frac` (2-5%) activations
      (k-winners-take-all) — return that sorted winner-index set as the sparse code.
      Deterministic: the same embedding always yields the same code.
    Why: D2 — the orthogonalizing pattern-separation primitive. Two near-duplicate
      embeddings produce LARGELY DISJOINT winner sets (so each engram copy stays
      individually addressable), while a true duplicate (handled by the SEPARATE
      cosine dedup gate, pattern_separation_decision) is still deduped. Runs once on
      the held copy at materialization; never on the retrieval embedding.

    Args:
        embedding: a 1-D vector (any dim — 384-dim MiniLM in production, the small
          stub dim in tests).
        frac: the fraction of projected units to keep active (k of kWTA).
        proj_dim / seed: the fixed projection space + seed (determinism).

    Returns a sorted list of winner indices (the sparse key). Empty for an empty/zero
    embedding.
    """
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.size == 0 or not np.any(vec):
        return []
    proj = _kwta_projection(vec.shape[0], proj_dim, seed)
    activations = vec @ proj                       # [proj_dim] projected response
    k = max(1, int(round(proj_dim * float(frac)))) # at least one winner
    k = min(k, proj_dim)
    # top-k winners by activation; argpartition is O(n), then sort the winners.
    winners = np.argpartition(activations, -k)[-k:]
    return sorted(int(i) for i in winners)


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.pattern
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.bio monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       pattern separation — the cosine DEDUP gate (_node_embedding,
#             pattern_separation_decision; store-new vs merge-into at write time) and
#             the ORTHOGONALIZING kWTA sparse code (_kwta_projection, kwta_sparse_code;
#             individually-addressable near-duplicate engram copies).
# Stability:  stable — pure separation primitives.
# ErrorModel: _node_embedding returns None on a missing manifest/entry (fail-soft);
#             kwta_sparse_code returns [] for an empty/zero embedding.
# Depends:    .config (PATTERN_THRESHOLD_DEFAULT / KWTA_* / np / the single-owned
#             _KWTA_PROJ_CACHE); samia.core.vector (lazy, function-local — manifest /
#             embeddings / query).
# Exposes:    pattern_separation_decision, kwta_sparse_code (public); _node_embedding,
#             _kwta_projection (private, re-exported for tests/importers/parity).
# Note:       the kWTA arm reads/writes the SINGLE-OWNED projection cache through
#             config (_cfg._KWTA_PROJ_CACHE), never a local copy, so there is one cache.
# Lines:      169
# --------------------------------------------------------------------------
