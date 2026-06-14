"""samia.core.merge_consumer.candidates — load surfacer candidates, resolve +
classify a pair (dup vs abstract), read a node's frontmatter.

Layer 1 (Owns / Depends):
    Owns:    load_candidates (read the surfacer's .consolidation_candidates.json),
             the file->id + live-pair resolution (_node_id_for_file, _resolve_pair),
             the dup-vs-abstract classifier (classify_pair) with its cosine path
             (_cosine_for_pair) + jaccard fallback, and the node frontmatter read
             primitive (_read_fm) the winner-merge and fact-extract paths share.
    Depends: .config (the dup bars + _con/_fm), samia.runtime.contradiction (the
             cosine finder, imported lazily so no hard embedder dependency).

Layer 2 (What / Why):
    What: the READ + CLASSIFY half of the consumer. The surfacer already scored
          ~600 near-dup pairs into .consolidation_candidates.json; this submodule
          turns each candidate row into a live (a_id, b_id) pair and labels it
          "dup" (P1 pick-winner acts) or "abstract" (left for P2). _read_fm is the
          single node-parse used by both the winner ranking and the abstract
          enqueue, so it lives with the other node-I/O here.
    Why:  Q3a CONSUME — the consumer reads the surfacer's output, never re-audits
          the chains. Q1c TIERED + Q2c AUTO — cosine (>= bar) when a vector index
          exists, deterministic jaccard fallback otherwise, so classification is
          stub-free on cold trees (tests) and embedder-backed live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import (
    _DUP_MERGE_COSINE,
    _DUP_MERGE_JACCARD,
    _CANDIDATE_FILE,
    _con,
    _fm,
)


def load_candidates(memory_dir: Path) -> list[dict]:
    """Read the surfacer's .consolidation_candidates.json candidate list.

    What: returns the ``candidates`` list ({chain, a_addr, a_file, b_addr,
          b_file, similarity}) the consolidation surfacer wrote. Empty list if
          the file is missing/unreadable (nothing surfaced => nothing to drain).
    Why:  Q3a — the consumer CONSUMES the existing surfacer output; it never
          re-audits the chains (that is the surfacer's job at priority 20).
    """
    p = Path(memory_dir) / _CANDIDATE_FILE
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    cands = payload.get("candidates") if isinstance(payload, dict) else None
    return list(cands) if isinstance(cands, list) else []


def _node_id_for_file(rel_file: str) -> str:
    """Map a candidate's ``a_file``/``b_file`` (nodes/<id>.md) to a node id.

    What: strip the nodes/ prefix and .md suffix to the bare id forget_node /
          frontmatter callers use.
    Why:  the surfacer records the relative path; ia.forget_node + the node file
          readers want the stem.
    """
    name = rel_file
    if name.startswith("nodes/"):
        name = name[len("nodes/"):]
    if name.endswith(".md"):
        name = name[:-3]
    return name


def _resolve_pair(memory_dir: Path, cand: dict) -> Optional[tuple[str, str]]:
    """Map a candidate to a live (a_id, b_id) pair, or None if either is gone.

    What: resolves a_file/b_file to ids and confirms both nodes still exist in
          nodes/. Returns None when either node is already merged/forgotten —
          the drain advancing (the pair is stale).
    Why:  a previous cycle (or P3 supersede, or the surfacer's own churn) may
          have already removed one side; merging a phantom would error.
    """
    nodes = Path(memory_dir) / "nodes"
    a_id = _node_id_for_file(str(cand.get("a_file", "")))
    b_id = _node_id_for_file(str(cand.get("b_file", "")))
    if not a_id or not b_id or a_id == b_id:
        return None
    if not (nodes / f"{a_id}.md").exists() or not (nodes / f"{b_id}.md").exists():
        return None
    return a_id, b_id


def _cosine_for_pair(memory_dir: Path, a_id: str, b_id: str) -> Optional[float]:
    """Cosine similarity between the two nodes via the existing finder, or None.

    What: read a's text, run contradiction.find_supersession_candidates scoped
          to [b_id], and return b's cosine score. None when the embedding infra
          (vector index) is unavailable — the caller then falls back to jaccard.
    Why:  Q1c/Q3a — the live path classifies on cosine (the proposal's bar); the
          jaccard fallback keeps P1 deterministic + stub-free when no index
          exists (tests, cold trees). Reuses the cosine finder; reinvents none.
    """
    try:
        from samia.runtime import contradiction as _contra
    except Exception:
        return None
    nodes = Path(memory_dir) / "nodes"
    a_path = nodes / f"{a_id}.md"
    if not a_path.exists():
        return None
    try:
        a_text = a_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        hits = _contra.find_supersession_candidates(
            a_text, scope_nodes=[b_id], memory_dir=Path(memory_dir),
        )
    except Exception:
        return None
    if not hits:
        return None
    b_fname = f"{b_id}.md"
    for h in hits:
        nid = str(h.get("node_id", ""))
        nid_md = nid if nid.endswith(".md") else f"{nid}.md"
        if nid_md == b_fname:
            return float(h.get("score", 0.0))
    return None


def classify_pair(memory_dir: Path, a_id: str, b_id: str,
                  candidate_similarity: Optional[float] = None) -> str:
    """Classify a surfaced pair as "dup" (P1 acts) or "abstract" (left for P2).

    What: return "dup" when the pair clears the HIGH duplicate bar, else
          "abstract". Prefers cosine (>= _DUP_MERGE_COSINE) when a vector index
          exists; otherwise falls back to the surfacer's lexical jaccard score
          (>= _DUP_MERGE_JACCARD). P1 acts ONLY on "dup"; "abstract" pairs are
          recorded for P2's gated LLM-abstraction.
    Why:  Q1c TIERED + Q2c AUTO — the near-exact bulk auto-merges (reversible,
          low-risk); distinct-but-overlapping pairs need a real synthesis, which
          is P2's gated job, not P1's.
    """
    cos = _cosine_for_pair(Path(memory_dir), a_id, b_id)
    if cos is not None:
        return "dup" if cos >= _DUP_MERGE_COSINE else "abstract"
    # Fallback: the surfacer's jaccard score (deterministic, no embedder).
    sim = candidate_similarity
    if sim is None:
        a_body = _con.load_node_body(Path(memory_dir), f"nodes/{a_id}.md") or ""
        b_body = _con.load_node_body(Path(memory_dir), f"nodes/{b_id}.md") or ""
        sim = _con.jaccard(_con.shingles(a_body), _con.shingles(b_body))
    return "dup" if float(sim) >= _DUP_MERGE_JACCARD else "abstract"


def _read_fm(memory_dir: Path, node_id: str) -> tuple[dict, list[str], str]:
    """Parse a node's (frontmatter, key_order, body); empty fm if unparsable."""
    p = Path(memory_dir) / "nodes" / f"{node_id}.md"
    raw = p.read_text(encoding="utf-8")
    parsed, body = _fm.parse(raw)
    if parsed is None:
        return {}, [], raw
    fm, order = parsed
    return fm, order, body


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.merge_consumer.candidates
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.merge_consumer monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       the READ + CLASSIFY half — load the surfacer's scored candidate
#             list, resolve each row to a live (a_id, b_id) pair, label it
#             dup/abstract (cosine then jaccard fallback), parse a node's
#             frontmatter for the winner-merge + fact-extract paths.
# Stability:  stable — pure read/classify; the carve preserved the cosine-first /
#             jaccard-fallback order and both bars byte-identical.
# ErrorModel: load_candidates returns [] on a missing/corrupt file; _resolve_pair
#             returns None on a stale/removed side; _cosine_for_pair returns None
#             when the embedder is unavailable (caller falls back); _read_fm
#             returns ({}, [], raw) on unparsable frontmatter (never raises here).
# Depends:    json, pathlib, typing (stdlib). .config (_DUP bars, _con, _fm).
#             samia.runtime.contradiction (lazy cosine finder).
# Exposes:    load_candidates, classify_pair, _resolve_pair, _node_id_for_file,
#             _cosine_for_pair, _read_fm.
# Lines:      190
# --------------------------------------------------------------------------
