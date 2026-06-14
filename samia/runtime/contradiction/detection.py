"""samia.runtime.contradiction.detection — Phase-1 embedding candidate finders.

Layer 1 (Owns / Depends):
    Owns:    the embedding-similarity candidate finder (_embed_text, _load_index,
             find_contradiction_candidates) and the supersession candidate finder
             that wraps it with the scope + jaccard filters
             (_node_text_for_id, find_supersession_candidates).
    Depends: samia.core.vector (the cached embedder + index schema, optional, lazy),
             samia.core.consolidation (shingles/jaccard, optional, lazy),
             samia.core.frontmatter (node text read, lazy), numpy (optional, lazy).
             The package config leaf (constants + the _MEMORY_DIR state + _node_type
             + is_excluded_node) and — through the PACKAGE FACADE — the patch-seam
             targets _embed_text and find_contradiction_candidates.

Layer 2 (What / Why):
    What: Phase 1 of AUD60 + the FEAT-2026-06-07 P3a supersession finder. Cheap
          vector cosine over the prebuilt index gives candidate nodes; the
          supersession finder adds the type-scoping drop, the online-locus scope
          filter, and the jaccard lexical pre-filter on top.
    Why:  embedding similarity catches topical overlap without LLM inference; the
          supersession finder reuses ONE detector for both online (scoped locus) and
          passive (whole index) modes. Single behavior, two entrypoints.

PATCH SEAMS (exemplar rule): _embed_text and find_contradiction_candidates are BOTH
    mock.patch.object(contradiction, ...) targets AND called by siblings (the finder
    reaches _embed_text; find_supersession_candidates + the passes arm reach
    find_contradiction_candidates), so those calls go THROUGH the package facade so a
    package-level patch rebinds the attribute the caller actually reads.
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

# Shared leaf — the live _MEMORY_DIR fallback, the cosine/semantic/max bars, the
# jaccard floor, the node-type cache + scoping predicate, and the package logger.
from . import config as _cfg


def _embed_text(text: str) -> Optional[Any]:
    """Compute embedding vector for a text string.

    What: uses the SAM vector index's embedding model to vectorize text.
    Why:  reuses the already-loaded embedding infrastructure rather than
          loading a second model. Returns None if the model is unavailable.
    """
    try:
        import numpy as np
        from samia.core.vector import MODEL_ID, EMBED_DIM  # noqa: F401
    except ImportError:
        _cfg._log.debug("contradiction: numpy or vector module not available")
        return None

    try:
        # What: delegate to the vector index's cached embedder (_ensure_model
        #   loads once per process; mask-weighted mean pooling + L2 norm —
        #   the SAME space the index rows live in).
        # Why: BUG-2026-06-10 — the old inline path called
        #   AutoTokenizer/AutoModel.from_pretrained on EVERY call, which hits
        #   the HuggingFace hub over the network each time; any hiccup raised,
        #   was swallowed by the except below, and the finder silently
        #   returned [] (probe showed nondeterministic sequential detection
        #   cutoffs). Delegating makes query embedding cached, offline, and
        #   byte-identical to the index side.
        from samia.core.vector import _embed_batch
        emb = _embed_batch([text])[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as exc:
        _cfg._log.debug("contradiction: embedding failed: %s", exc)
        return None


def _load_index(memory_dir: Path) -> Optional[tuple[Any, list[dict]]]:
    """Load the vector index embeddings and manifest.

    What: reads embeddings.npy and manifest.json from the vector_index dir.
    Why:  the candidate finder needs the pre-built index to compute cosine
          similarity against all existing nodes efficiently.

    Returns (embeddings_array, manifest_entries) or None if unavailable.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    index_dir = memory_dir / "vector_index"
    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"

    if not emb_path.exists() or not manifest_path.exists():
        _cfg._log.debug("contradiction: vector index not built at %s", index_dir)
        return None

    try:
        embeddings = np.load(str(emb_path))
        manifest = _cfg.json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", [])
        return embeddings, entries
    except Exception as exc:
        _cfg._log.debug("contradiction: failed to load index: %s", exc)
        return None


def find_contradiction_candidates(
    text: str,
    memory_dir: Optional[Path] = None,
    threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Find existing nodes whose embeddings are similar to the incoming text.

    What: computes cosine similarity between the incoming text embedding
          and all indexed node embeddings. Returns candidates above the
          similarity threshold.
    Why:  Phase 1 of AUD60 -- cheap vector operation that identifies
          potential contradictions without LLM inference.

    Parameters
    ----------
    text : str
        The incoming write payload as text.
    memory_dir : Path or None
        Memory directory. Falls back to _MEMORY_DIR.
    threshold : float or None
        Cosine similarity threshold. Falls back to _COSINE_THRESHOLD.

    Returns
    -------
    list of dicts with keys: node_id, title, score (cosine similarity).
    Empty list if embedding infrastructure is unavailable or no candidates.
    """
    # Reach the package facade for the SINGLE-OWNED _MEMORY_DIR state (configure() +
    # any test rebind land there) and for _embed_text (a mock.patch.object seam).
    from samia.runtime import contradiction as _pkg

    mem = memory_dir or _pkg._MEMORY_DIR
    if mem is None:
        return []

    thr = threshold if threshold is not None else _cfg._COSINE_THRESHOLD
    incoming_emb = _pkg._embed_text(text)
    if incoming_emb is None:
        return []

    index_data = _load_index(mem)
    if index_data is None:
        return []

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return []

    embeddings, entries = index_data
    if len(entries) == 0 or embeddings.shape[0] == 0:
        return []

    # What: map each embeddings.npy row -> (node filename, title) via the
    #   manifest entry's "row" field, skipping tombstoned (forget_node) entries.
    # Why: the canonical vector_index schema (samia.core.vector) stores entries
    #   as a DICT {fname -> {sha256, title, row}}, NOT a list. The old code
    #   indexed entries[i] with the embeddings row number, which raised
    #   KeyError(<row>) on every node (str(KeyError(34)) -> "34" -- the
    #   "passive finder failed for %s: 34" storm). Mirror vector.query's by_row
    #   mapping so the finder reads the index the way it is actually written.
    by_row = {
        e["row"]: (rel, e.get("title", ""))
        for rel, e in entries.items()
        if isinstance(e, dict) and e.get("row") is not None
        and not e.get("tombstoned")
    }

    # What: batch cosine similarity via matrix-vector dot product.
    # Why: embeddings are already L2-normalized in the index, so dot
    #   product equals cosine similarity. O(N*384) is sub-ms for N<10k.
    scores = embeddings @ incoming_emb
    candidates = []

    for i, score in enumerate(scores):
        if score < thr:
            continue
        info = by_row.get(int(i))
        if info is None:
            continue  # tombstoned or unindexed row -> skip cleanly
        rel, title = info
        # Per-population bar (TUNE-2026-06-10 (2)): a candidate that is a
        # machine-templated semantic atom must clear the HIGHER bar — template
        # kinship puts atom baseline similarity inside the hand-written
        # paraphrase band, so the recall-first threshold floods on them.
        if score < _cfg._SEMANTIC_PAIR_THRESHOLD and \
                _cfg._node_type(mem, str(rel)) == "semantic":
            continue
        candidates.append({
            "node_id": rel,
            "title": title,
            "score": float(round(score, 4)),
        })

    # What: sort by descending similarity and cap at _MAX_CANDIDATES.
    # Why: only the most similar nodes are worth sending to the LLM judge.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:_cfg._MAX_CANDIDATES]


def _node_text_for_id(memory_dir: Path, node_id: str) -> str:
    """Read a node's lexical text (title + body) for the jaccard pre-filter.

    What: load nodes/<id>.md and return description + body; empty if missing.
    Why:  the jaccard pre-filter needs the candidate's CONTENT words. BUG-
          2026-06-10: this returned the RAW file, so frontmatter key/value
          tokens (name/type/target_state/anchor fields) inflated the shingle
          union and dragged genuine supersessions below the 0.25 bar (probe:
          7/10 raw-file jaccards < 0.25 vs 1/10 on body text). Strip the
          frontmatter, keep the human text the docstring always promised.
    """
    fname = node_id if node_id.endswith(".md") else f"{node_id}.md"
    p = memory_dir / "nodes" / fname
    if not p.exists():
        return ""
    try:
        from samia.core.frontmatter import read_node
        fm, _order, body = read_node(p)
        return f"{fm.get('description', '')}\n{body}"
    except Exception:
        try:
            return p.read_text(encoding="utf-8")  # fail-soft: raw beats empty
        except Exception:
            return ""


def find_supersession_candidates(
    text: str,
    scope_nodes: Optional[list[str]] = None,
    memory_dir: Optional[Path] = None,
    threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Find existing nodes that the incoming text may supersede.

    What: wraps find_contradiction_candidates (cosine >= 0.75) with two layers
          the supersession detector needs:
            - scope filter: when scope_nodes is given, keep only candidates in
              that locus (the ONLINE active-set = co-activation neighbors +
              hot/recent); when None, span the whole index (PASSIVE).
            - jaccard pre-filter: drop candidates whose lexical overlap with the
              incoming text is below _SUPERSESSION_JACCARD, bounding the set with
              the same cheap smell memory_guard already uses.
    Why:  Q1a/Q2a — both modes share ONE detector; the online mode pays only for
          its locus, the passive mode sweeps everything, and jaccard keeps the
          cosine work bounded. This reuses the cosine finder and jaccard
          primitive; it reinvents neither.

    Parameters
    ----------
    text : str
        The incoming write payload as text.
    scope_nodes : list[str] or None
        Restrict candidates to this set of node ids (online locus). None = whole
        index (passive). Ids may be given with or without the .md suffix.
    memory_dir : Path or None
        Memory directory. Falls back to _MEMORY_DIR.
    threshold : float or None
        Cosine threshold passed through to find_contradiction_candidates.

    Returns
    -------
    list of dicts {node_id, title, score, jaccard} sorted by descending cosine.
    Empty if embedding infra is unavailable or nothing clears both filters.
    """
    # Reach the facade for the _MEMORY_DIR state + find_contradiction_candidates (a
    # patch seam: test_contradiction patches it; passes/passive_sweep reach this fn).
    from samia.runtime import contradiction as _pkg

    mem = memory_dir or _pkg._MEMORY_DIR
    if mem is None:
        return []

    # Phase 1 reuse: cosine candidate finder over the whole index.
    candidates = _pkg.find_contradiction_candidates(text, memory_dir=mem, threshold=threshold)
    if not candidates:
        return []

    # TYPE-SCOPING (the big lever): drop excluded-type (episodic/experiential)
    # nodes as MATCHES. A session_offload/bug node is never a contradictable
    # content claim, so it must not surface as a supersession candidate even when
    # its embedding is cosine-similar to the incoming text. This is what collapses
    # the ~807K self-similar episodic pairs out of the candidate space.
    candidates = [
        c for c in candidates
        if not _cfg.is_excluded_node(mem, str(c["node_id"]))
    ]
    if not candidates:
        return []

    # Scope filter: restrict to the online locus when one is supplied.
    # Why: the active-set is the cheap immediate check; non-locus nodes are the
    #   passive sweep's job, so they must never surface as online candidates.
    if scope_nodes is not None:
        scope = {s if s.endswith(".md") else f"{s}.md" for s in scope_nodes}
        candidates = [
            c for c in candidates
            if (c["node_id"] if str(c["node_id"]).endswith(".md")
                else f"{c['node_id']}.md") in scope
        ]
        if not candidates:
            return []

    # Jaccard pre-filter: keep only candidates with real lexical overlap.
    # Why: reuse the existing consolidation shingle/jaccard primitive (the same
    #   memory_guard pre-filter) to bound the set and annotate each survivor.
    try:
        from samia.core.consolidation import shingles, jaccard
    except ImportError:
        # Fail-soft: without the primitive, skip the lexical gate (cosine stands).
        return [{**c, "jaccard": None} for c in candidates]

    new_shingles = shingles(text)
    out: list[dict[str, Any]] = []
    for c in candidates:
        cand_shingles = shingles(_node_text_for_id(mem, str(c["node_id"])))
        sim = jaccard(new_shingles, cand_shingles)
        if sim >= _cfg._SUPERSESSION_JACCARD:
            out.append({**c, "jaccard": float(round(sim, 4))})
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.contradiction.detection
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.runtime.contradiction monolith
#             during modularization (the Phase-1 embedding finder arm).
# Layer:      runtime (library helper, no daemon loop)
# Role:       Phase-1 embedding-similarity candidate finder (AUD60) + the
#             FEAT-2026-06-07 P3a supersession candidate finder (scope + jaccard).
# Stability:  stable — pure read/compute over the prebuilt vector index.
# ErrorModel: fail-open — numpy/vector/consolidation unavailability degrades to []
#             (or skips the lexical gate). BUG-2026-06-07 by_row mapping reads the
#             canonical dict-shaped manifest (no KeyError(row) "34" storm).
# Depends:    .config (constants + state + scoping); samia.core.{vector,consolidation,
#             frontmatter} + numpy (all lazy/optional). Reaches _embed_text +
#             find_contradiction_candidates + _MEMORY_DIR through the PACKAGE FACADE
#             (patch seams + single-owned state).
# Exposes:    find_contradiction_candidates, find_supersession_candidates (public);
#             _embed_text, _load_index, _node_text_for_id (internal, _embed_text is a
#             patch seam re-exported by the facade).
# Lines:      349
# --------------------------------------------------------------------------
