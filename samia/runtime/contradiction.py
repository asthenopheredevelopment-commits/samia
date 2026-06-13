"""samia.runtime.contradiction -- AUD60 embedding-similarity contradiction detection.

Layer 1 (Owns / Depends):
    Owns:    Embedding-based contradiction candidate finder and optional LLM
             judge gate for memory writes. Integrates with memory_guard's
             pre-write validation pipeline.
    Depends: samia.core.vector (embedding infrastructure, optional at import).
             samia.core.frontmatter (node reading).
             numpy (optional -- gracefully degrades without it).
             samia.runtime.inference (in-process LLM judge backend, optional).

Layer 2 (What / Why):
    What: Three-phase contradiction detection pipeline:
          Phase 1 -- Embedding similarity: compute cosine similarity between
            incoming write and existing nodes. Candidates with similarity
            >= 0.75 are flagged as potential contradictions.
          Phase 2 -- LLM judge gate (opt-in): if enabled, routes top-N
            candidates through the in-process inference backend
            (samia.runtime.inference.get_backend) to judge contradiction
            confidence. Blocks writes with confidence >= 0.7.
          Phase 3 -- Memory guard integration: flagged contradictions are
            routed to the pending.jsonl queue with contradiction_with metadata,
            extending the AUD48 MemGuard row format.
    Why:  Without semantic contradiction detection, new writes can silently
          overwrite or conflict with established claims. The shingle/jaccard
          heuristic in AUD48 catches topical overlap but not semantic
          contradiction (two claims about the same topic that disagree).
          Embedding similarity + LLM judge provides the precision needed
          to catch real contradictions while keeping false positives low.

Design doc: AUD60_llm_contradiction_detection.md

Configuration (environment variables):
    ASTHENOS_CONTRADICTION_ENABLED=1   Enable embedding contradiction check
    ASTHENOS_CONTRADICTION_JUDGE=1     Enable LLM judge gate (requires CHIRON)
    ASTHENOS_CONTRADICTION_THRESHOLD   Cosine similarity threshold (default 0.75)
    ASTHENOS_CONTRADICTION_JUDGE_CONF  Judge confidence threshold (default 0.7)
    ASTHENOS_CONTRADICTION_EXCLUDE_TYPES  Comma-separated node `type`s the detector
                                       EXCLUDES (default "session_offload,bug" --
                                       episodic/experiential records). The big
                                       type-scoping lever (TUNE-2026-06-08).
    ASTHENOS_CONTRADICTION_JUDGE_MODEL  gguf for the DEDICATED fast judge backend
                                       (default Qwen3-4B-Instruct-2507-Q4_K_M) -- the
                                       judge no longer rides the slow main 14B.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger("samia.runtime.contradiction")


def _now_iso() -> str:
    """UTC ISO-8601 timestamp for candidate provenance records."""
    return datetime.now(tz=timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# _ENABLED -- What: master switch for contradiction detection.
# _ENABLED -- Why: default-off so the feature doesn't impact write latency
#   until the operator explicitly enables it. The embedding model may not
#   be loaded, and numpy may not be available.
_ENABLED: bool = os.environ.get("ASTHENOS_CONTRADICTION_ENABLED", "0") == "1"

# _JUDGE_ENABLED -- What: opt-in flag for the LLM judge gate.
# _JUDGE_ENABLED -- Why: ON by default since TUNE-2026-06-10 — the lowered
#   cosine bar (below) is recall-first and NEEDS the BitNet stage-2 judge for
#   precision. Degrades gracefully (candidates-only) when the inference
#   backend is unavailable; latency rides the passive sweep, not hot writes.
#   Operator-directed 2026-06-10 (benchmark probe: TPR 0.2 at the old bar).
_JUDGE_ENABLED: bool = os.environ.get("ASTHENOS_CONTRADICTION_JUDGE", "1") == "1"

# _COSINE_THRESHOLD -- What: minimum cosine similarity to flag a candidate.
# _COSINE_THRESHOLD -- Why: TUNE-2026-06-10, operator-directed: 0.75 (AUD60)
#   missed 8/10 genuine paraphrased supersessions in the competitive-benchmark
#   probe (MiniLM paraphrase cosines run 0.49-0.95; title-prefix dilutes
#   further). 0.57 is recall-first; precision is recovered by the stage-2
#   BitNet judge (now default-on above). FPR at the candidate layer is bounded
#   by type-scoping (see TUNE-2026-06-08 below).
_COSINE_THRESHOLD: float = float(
    os.environ.get("ASTHENOS_CONTRADICTION_THRESHOLD", "0.57")
)

# _SEMANTIC_PAIR_THRESHOLD -- What: the HIGHER cosine bar applied to any pair
#   involving a machine-generated semantic atom (type: semantic).
# _SEMANTIC_PAIR_THRESHOLD -- Why: TUNE-2026-06-10 (2) — the fact-extract
#   backfill grew the scoped content corpus 129 -> 5,789 nodes; at the 0.57
#   recall-first bar that is 175,017 pairs (noise ocean: atoms share one
#   template, so their BASELINE mutual similarity sits in the 0.57-0.75 band
#   where hand-written paraphrased supersessions live). Atoms are deduped at
#   0.92 at creation, so >=0.92 survivors are genuinely actionable (~3.4k,
#   judge-drainable). Hand-written-only pairs keep the operator's 0.57.
_SEMANTIC_PAIR_THRESHOLD: float = float(
    os.environ.get("ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD", "0.92")
)

# _JUDGE_CONFIDENCE_THRESHOLD -- What: minimum LLM judge confidence to block.
# _JUDGE_CONFIDENCE_THRESHOLD -- Why: 0.7 per AUD60 proposal. Below this,
#   the contradiction is flagged but not blocked.
_JUDGE_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ASTHENOS_CONTRADICTION_JUDGE_CONF", "0.7")
)

# _MAX_CANDIDATES -- What: maximum candidates to send to the LLM judge.
# _MAX_CANDIDATES -- Why: bounds inference cost per write (max 10 per AUD60).
_MAX_CANDIDATES: int = 10

# _ONLINE_AUTO_COSINE -- What: cosine bar at/above which the ONLINE write path
#   AUTO-supersedes (no LLM judge) the exact case (same subject key + this cosine).
# _ONLINE_AUTO_COSINE -- Why: the Q4-granularity decision — online has no judge,
#   so it auto-acts ONLY on the obvious exact-supersession case (default 0.92,
#   env-tunable). Weaker hits (0.75 <= cosine < 0.92) are recorded for the passive
#   judge, not auto-deleted. Made safe by reversibility via restore_node.
_ONLINE_AUTO_COSINE: float = float(
    os.environ.get("ASTHENOS_SUPERSESSION_AUTO_COSINE", "0.92")
)

# _MEMORY_DIR -- What: root memory directory for node access.
# _MEMORY_DIR -- Why: used by the embedding candidate finder to locate
#   the vector index and node files.
_MEMORY_DIR: Optional[Path] = None

# ---------------------------------------------------------------------------
# TUNE-2026-06-08: TYPE-SCOPING (the big lever)
# ---------------------------------------------------------------------------
#
# OPERATOR RULE: the contradiction/supersession detector EXCLUDES episodic /
# experiential records (a transcript of interactions / "this happened at time X"
# about experiences -- type session_offload, and bug = event records). It INCLUDES
# content / factual claims (project/reference/user/feedback). A song/doc transcript
# stored as reference IS content = included; the distinction is experiential-vs-
# content, mapped here onto the `type` frontmatter field.
#
# Empirically: cosine>=0.75 over the whole 2752-node index = ~807K candidate pairs
# (21% flood) because 2597 nodes are session_offload (episodic, self-similar).
# Scoping to contradictable types collapses that to ~152 nodes / ~393 pairs.
_DEFAULT_EXCLUDE_TYPES = "session_offload,bug"


def excluded_types() -> frozenset[str]:
    """The set of node `type` values the detector EXCLUDES (live env read).

    What: parses ASTHENOS_CONTRADICTION_EXCLUDE_TYPES (comma-separated, default
          "session_offload,bug") into a lowercased frozenset. Read each call so a
          test/daemon that sets the env after import sees the change.
    Why:  the type-scoping lever -- episodic/experiential records (session_offload
          transcripts, bug event records) are NOT contradictable content claims, so
          the detector must never enumerate or match them. Env-overridable so the
          operator can widen/narrow the exclusion without a code change.
    """
    raw = os.environ.get("ASTHENOS_CONTRADICTION_EXCLUDE_TYPES", _DEFAULT_EXCLUDE_TYPES)
    return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())


# _TYPE_CACHE -- What: per-(memory_dir, node) cache of the resolved `type` field.
# _TYPE_CACHE -- Why: passive_sweep + active_set + the finder each resolve a node's
#   type repeatedly across a sweep; reading frontmatter once and caching keeps the
#   scope check cheap (a sweep over thousands of nodes must not re-parse each file
#   per candidate). Keyed by (str(memory_dir), node-stem). Bounded by index size.
_TYPE_CACHE: dict[tuple[str, str], Optional[str]] = {}


def _clear_type_cache() -> None:
    """Drop the node-type cache (tests / after a node's type changes)."""
    _TYPE_CACHE.clear()


def _node_type(memory_dir: Path, node_id: str) -> Optional[str]:
    """Resolve a node's `type` frontmatter field (cached, lowercased).

    What: read nodes/<id>.md frontmatter and return its 'type' value lowercased,
          or None when the node is missing / unreadable / has no type. Cached per
          (memory_dir, stem) so a sweep resolves each node's type at most once.
    Why:  type-scoping needs the node's content/experiential class. Reading on
          demand + caching avoids loading every node up front and avoids re-parsing
          the same file across many candidate comparisons in one sweep.
    """
    stem = node_id[:-3] if node_id.endswith(".md") else node_id
    key = (str(memory_dir), stem)
    if key in _TYPE_CACHE:
        return _TYPE_CACHE[key]
    p = memory_dir / "nodes" / f"{stem}.md"
    val: Optional[str] = None
    if p.exists():
        try:
            from samia.core import frontmatter as _fm
            parsed, _ = _fm.parse(p.read_text(encoding="utf-8"))
            if parsed is not None:
                t = parsed[0].get("type")
                if isinstance(t, str) and t.strip():
                    val = t.strip().lower()
        except Exception:
            val = None
    _TYPE_CACHE[key] = val
    return val


def is_excluded_node(memory_dir: Path, node_id: str) -> bool:
    """True iff *node_id* is an excluded (episodic/experiential) node to SKIP.

    What: returns True when the node's resolved `type` is in excluded_types().
          Missing/unreadable type -> treat as INCLUDED (conservative: a real
          content claim is never silently dropped) EXCEPT the obvious
          session_offload case detectable from the FILENAME (the episodic
          transcripts are named session_*_offload_*), which stays excluded even
          when its frontmatter can't be parsed.
    Why:  the single predicate every detector enumeration/match site consults so
          the experiential-vs-content rule is applied identically online, in the
          passive sweep, and in the active-set. Conservative on ambiguity:
          excluding a genuine claim would silently disable the detector for it, so
          unknown-type defaults to INCLUDED.
    """
    excl = excluded_types()
    t = _node_type(memory_dir, node_id)
    if t is not None:
        return t in excl
    # Unreadable/typeless: include conservatively, EXCEPT obvious episodic by name.
    if "session_offload" in excl:
        stem = node_id[:-3] if node_id.endswith(".md") else node_id
        low = stem.lower()
        if low.startswith("session_") and "offload" in low:
            return True
    return False


def is_enabled() -> bool:
    """Live read of the ASTHENOS_CONTRADICTION_ENABLED master switch.

    What: returns True only when the operator has enabled contradiction/
          supersession detection (default OFF). Reads the env each call so a
          test/daemon that sets the flag after import sees the change.
    Why:  R8 produce-only gating — the ONLINE auto-supersede write-path behavior
          must stay inert until the operator enables it + restarts the daemon.
    """
    return os.environ.get("ASTHENOS_CONTRADICTION_ENABLED", "0") == "1"


def auto_cosine_threshold() -> float:
    """The online exact-supersession auto bar (default 0.92, env-tunable)."""
    return _ONLINE_AUTO_COSINE


def configure(memory_dir: Path) -> None:
    """Set the memory directory for contradiction detection.

    What: stores the memory_dir path for vector index lookups.
    Why:  the contradiction module needs to know where nodes live, but
          shouldn't hardcode paths. The daemon calls this during startup.
    """
    global _MEMORY_DIR
    _MEMORY_DIR = memory_dir
    _log.info(
        "contradiction: configured memory_dir=%s enabled=%s judge=%s",
        memory_dir, _ENABLED, _JUDGE_ENABLED,
    )


# ---------------------------------------------------------------------------
# Phase 1: Embedding-similarity candidate finder
# ---------------------------------------------------------------------------


def _embed_text(text: str) -> Optional[Any]:
    """Compute embedding vector for a text string.

    What: uses the SAM vector index's embedding model to vectorize text.
    Why:  reuses the already-loaded embedding infrastructure rather than
          loading a second model. Returns None if the model is unavailable.
    """
    try:
        import numpy as np
        from samia.core.vector import MODEL_ID, EMBED_DIM
    except ImportError:
        _log.debug("contradiction: numpy or vector module not available")
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
        _log.debug("contradiction: embedding failed: %s", exc)
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
        _log.debug("contradiction: vector index not built at %s", index_dir)
        return None

    try:
        embeddings = np.load(str(emb_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", [])
        return embeddings, entries
    except Exception as exc:
        _log.debug("contradiction: failed to load index: %s", exc)
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
    mem = memory_dir or _MEMORY_DIR
    if mem is None:
        return []

    thr = threshold if threshold is not None else _COSINE_THRESHOLD
    incoming_emb = _embed_text(text)
    if incoming_emb is None:
        return []

    index_data = _load_index(mem)
    if index_data is None:
        return []

    try:
        import numpy as np
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
        if score < _SEMANTIC_PAIR_THRESHOLD and \
                _node_type(mem, str(rel)) == "semantic":
            continue
        candidates.append({
            "node_id": rel,
            "title": title,
            "score": float(round(score, 4)),
        })

    # What: sort by descending similarity and cap at _MAX_CANDIDATES.
    # Why: only the most similar nodes are worth sending to the LLM judge.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:_MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# FEAT-2026-06-07 P3a: supersession candidate finder (online/passive shared core)
# ---------------------------------------------------------------------------

# _SUPERSESSION_JACCARD -- What: cheap lexical pre-filter floor before cosine.
# _SUPERSESSION_JACCARD -- Why: reuse memory_guard's existing 0.25 jaccard smell
#   to bound the cosine candidate set (Q2 answered: jaccard stays as a cheap
#   pre-filter). Env-tunable; 0.25 mirrors memory_guard._CONTRADICTION_THRESHOLD.
_SUPERSESSION_JACCARD: float = float(
    os.environ.get("ASTHENOS_SUPERSESSION_JACCARD", "0.25")
)


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
    mem = memory_dir or _MEMORY_DIR
    if mem is None:
        return []

    # Phase 1 reuse: cosine candidate finder over the whole index.
    candidates = find_contradiction_candidates(text, memory_dir=mem, threshold=threshold)
    if not candidates:
        return []

    # TYPE-SCOPING (the big lever): drop excluded-type (episodic/experiential)
    # nodes as MATCHES. A session_offload/bug node is never a contradictable
    # content claim, so it must not surface as a supersession candidate even when
    # its embedding is cosine-similar to the incoming text. This is what collapses
    # the ~807K self-similar episodic pairs out of the candidate space.
    candidates = [
        c for c in candidates
        if not is_excluded_node(mem, str(c["node_id"]))
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
        if sim >= _SUPERSESSION_JACCARD:
            out.append({**c, "jaccard": float(round(sim, 4))})
    return out


# _SUPERSESSION_STORE -- What: the single canonical candidate store filename.
# _SUPERSESSION_STORE -- Why: R2 reconciliation — BOTH the old memory_guard
#   SUPERSESSION_LOG (run-1, ~/.local/share/.../memory_guard/) and this module
#   (run-2, <memory_dir>/biomimetic/) previously wrote supersession_candidates.jsonl
#   with DIFFERENT schemas. This module is now the ONE owner with ONE schema; the
#   surfacer and confirm/dismiss/list paths all route here.
_SUPERSESSION_STORE = "supersession_candidates.jsonl"


def _supersession_path(memory_dir: Path) -> Path:
    """Canonical path of the unified supersession-candidate store."""
    return memory_dir / "biomimetic" / _SUPERSESSION_STORE


def record_supersession_candidate(
    memory_dir: Path,
    old_id: str,
    new_id: str,
    cosine: float,
    jaccard: Optional[float] = None,
    mode: str = "online",
    judge: Optional[dict[str, Any]] = None,
    status: str = "candidate",
) -> dict[str, Any]:
    """Append a supersession candidate to the unified store (R2 canonical owner).

    What: writes one record with the single schema
          {old_id, new_id, cosine, jaccard, mode(online|passive), judge?, ts,
           status, confirmed, dismissed} to <memory_dir>/biomimetic/
          supersession_candidates.jsonl — the durable provenance of a detected
          (but not yet auto-acted) supersession.
    Why:  the Q4-granularity decision records weaker online hits and all passive
          hits as candidates for the LLM judge / operator surface rather than
          auto-deleting. This is the write side of the single store that the
          memory_guard surfacer and the mcp_server confirm/dismiss/list all share.

          DEDUP (BUG-2026-06-11): the SAME (old_id, new_id) pair was re-detected
          and re-appended on every passive sweep / online write, so the store grew
          unbounded with duplicate UNRESOLVED rows. Skip the append when an
          unresolved record for the same (old_id, new_id) already exists; a
          resolved (confirmed/dismissed) prior record does NOT suppress a fresh
          re-detection (the pair came back after being acted on — that is signal).
    """
    bio_dir = memory_dir / "biomimetic"
    bio_dir.mkdir(parents=True, exist_ok=True)
    norm_old = old_id if old_id.endswith(".md") else f"{old_id}.md"
    norm_new = new_id if new_id.endswith(".md") else f"{new_id}.md"
    # Dedup: an unresolved row for this exact pair already stands — re-recording it
    # only bloats the store and re-surfaces the same candidate. Scan the live
    # unresolved rows once (the canonical reader handles id normalization + skips
    # already confirmed/dismissed rows); return the standing record so callers see
    # "already a candidate" without a second write.
    for r in list_supersession_candidates(memory_dir, unresolved_only=True):
        if (r.get("old_id"), r.get("new_id")) == (norm_old, norm_new):
            return r
    rec: dict[str, Any] = {
        "old_id": norm_old,
        "new_id": norm_new,
        "cosine": float(round(cosine, 4)),
        "jaccard": (float(round(jaccard, 4)) if jaccard is not None else None),
        "mode": mode,
        "ts": _now_iso(),
        "status": status,
        "confirmed": False,
        "dismissed": False,
    }
    if judge is not None:
        rec["judge"] = judge
    with _supersession_path(memory_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def list_supersession_candidates(memory_dir: Path,
                                 unresolved_only: bool = True
                                 ) -> list[dict[str, Any]]:
    """Read the unified supersession store (R2 canonical reader).

    What: returns the recorded candidates; when unresolved_only (default) skips
          any already confirmed or dismissed.
    Why:  the single list path the memory_guard surfacer and the mcp_server /
          Atoms surface both consume. Fail-soft on a missing or partly-corrupt
          file (skips unparseable lines).
    """
    out: list[dict[str, Any]] = []
    p = _supersession_path(memory_dir)
    if not p.exists():
        return out
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if unresolved_only and (rec.get("confirmed")
                                        or rec.get("dismissed")):
                    continue
                out.append(rec)
    except Exception as exc:
        _log.warning("contradiction: supersession store read failed: %s", exc)
    return out


def _mark_supersession_candidate(memory_dir: Path, old_id: str,
                                 new_id: Optional[str], field: str) -> int:
    """Atomically set <field>=True (+ status) on matching un-resolved candidate(s).

    What: rewrites the unified store, marking every entry whose old_id matches
          (and new_id, if given) and that is not already resolved. Returns the
          count touched. tmp + replace keeps the rewrite crash-safe.
    Why:  the single mark path for both confirm and dismiss; resolving a
          candidate stops it re-surfacing. This only RECORDS the decision —
          the actual archiving forget cascade is run by the caller.
    """
    p = _supersession_path(memory_dir)
    if not p.exists():
        return 0
    target = old_id if old_id.endswith(".md") else f"{old_id}.md"
    want_new = (new_id if (new_id is None or new_id.endswith(".md"))
                else f"{new_id}.md")
    ts = _now_iso()
    touched = 0
    try:
        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_old = rec.get("old_id", "")
                rec_old = rec_old if rec_old.endswith(".md") else f"{rec_old}.md"
                already = rec.get("confirmed") or rec.get("dismissed")
                if (not already and rec_old == target
                        and (want_new is None or rec.get("new_id") == want_new)):
                    rec[field] = True
                    rec[f"{field}_at"] = ts
                    rec["status"] = field  # "confirmed" | "dismissed"
                    touched += 1
                rows.append(rec)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec) + "\n")
        tmp.replace(p)
    except Exception as exc:
        _log.warning("contradiction: supersession mark failed: %s", exc)
    return touched


def mark_supersession_confirmed(memory_dir: Path, old_id: str,
                                new_id: Optional[str] = None) -> int:
    """Record confirmation of a supersession candidate in the unified store.

    No delete here — the caller (mcp_server.memory_confirm_supersession) runs the
    archiving forget cascade; this only marks the decision so it stops surfacing.
    """
    return _mark_supersession_candidate(memory_dir, old_id, new_id, "confirmed")


def mark_supersession_dismissed(memory_dir: Path, old_id: str,
                                new_id: Optional[str] = None) -> int:
    """Record dismissal (false positive) of a supersession candidate. No delete."""
    return _mark_supersession_candidate(memory_dir, old_id, new_id, "dismissed")


# ---------------------------------------------------------------------------
# FEAT-2026-06-07 P3c: PASSIVE supersession sweep (REM-cycle subscriber)
# ---------------------------------------------------------------------------

# _PASSIVE_CURSOR_KEY -- What: the rem_cursors.json key the passive sweep
#   checkpoints its cursor under.
# _PASSIVE_CURSOR_KEY -- Why: must match the cursor_key the REM registration
#   uses (rem_subscribers.register) so the driver reads the same resume point.
_PASSIVE_CURSOR_KEY = "contradiction_passive"

# _PASSIVE_BUDGET -- What: default nodes processed per passive_sweep call.
# _PASSIVE_BUDGET -- Why: REM is idle-budgeted; a bounded slice per cycle keeps a
#   single REM tick responsive and lets a full pass complete across many cycles.
_PASSIVE_BUDGET: int = int(os.environ.get("ASTHENOS_SUPERSESSION_PASSIVE_BUDGET", "20"))


def _list_node_ids(memory_dir: Path) -> list[str]:
    """Sorted node ids (file stems) of the whole nodes/ index.

    What: the ordered universe the passive cursor walks. Sorted so the cursor
          index is stable across calls (a node added mid-pass shifts the tail,
          which the wrap-at-end reset tolerates).
    Why:  the passive sweep spans the WHOLE index (scope=None); it needs a
          deterministic order to advance a numeric cursor and detect wrap.
    """
    nodes = memory_dir / "nodes"
    if not nodes.is_dir():
        return []
    try:
        return sorted(p.stem for p in nodes.glob("*.md"))
    except OSError:
        return []


def _node_field(memory_dir: Path, node_id: str, key: str) -> Any:
    """Read one frontmatter field of a node (None if missing/unreadable).

    What: parse nodes/<id>.md and return frontmatter[key].
    Why:  the loser-selection rule needs valid_from (age) and confidence to pick
          which of a contradicting pair is superseded; reading on demand avoids
          loading every node up front.
    """
    fname = node_id if node_id.endswith(".md") else f"{node_id}.md"
    p = memory_dir / "nodes" / fname
    if not p.exists():
        return None
    try:
        from samia.core import frontmatter as _fm
        parsed, _ = _fm.parse(p.read_text(encoding="utf-8"))
        if parsed is not None:
            return parsed[0].get(key)
    except Exception:
        return None
    return None


def _pick_superseded(memory_dir: Path, a_id: str, b_id: str) -> tuple[str, str]:
    """Pick (loser, winner) for a confirmed contradiction by the proposal's rule.

    What: the loser is the OLDER (earlier valid_from) / LOWER-confidence claim;
          the survivor is the newer / higher-confidence one. Ties fall back to a
          stable id order so the choice is deterministic.
    Why:  the proposal directs auto-supersede to retire "the older/lower-
          confidence claim" — a newer, contradictory belief supersedes the stale
          one. Always reversible (restore_node), so the conservative tie-break is
          safe.
    """
    conf_a = _node_field(memory_dir, a_id, "confidence")
    conf_b = _node_field(memory_dir, b_id, "confidence")
    try:
        if conf_a is not None and conf_b is not None and float(conf_a) != float(conf_b):
            return (a_id, b_id) if float(conf_a) < float(conf_b) else (b_id, a_id)
    except (TypeError, ValueError):
        pass
    vf_a = _node_field(memory_dir, a_id, "valid_from")
    vf_b = _node_field(memory_dir, b_id, "valid_from")
    if isinstance(vf_a, str) and isinstance(vf_b, str) and vf_a != vf_b:
        return (a_id, b_id) if vf_a < vf_b else (b_id, a_id)
    # Deterministic fallback: the lexically-greater id is treated as "newer".
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


def _salience_guards_supersede(memory_dir: Path, loser: str) -> bool:
    """True iff the salience guard protects the loser from auto-supersede (P3).

    What: consult bio.salience_merge_guard on the loser with is_duplicate=False.
          Returns True when the loser is a DISTINCT high-salience memory the guard
          protects — the caller then SURFACES the supersession for operator review
          instead of auto-removing it. False when the guard is unavailable (bio
          without salience_merge_guard — the online/passive paths ship before
          Tier-1's salience field lands) or the loser is not high-salience.
    Why:  D6 effect (iii) / Q5a — the salience merge/supersede guard is CONSUMED
          by the contradiction detector (here) AND the merge consumer. A
          contradiction pair is distinct (X vs not-X), so is_duplicate stays
          False; an exact duplicate is not the guard's target. Wired behind a
          hasattr guard so the detector runs fully before the salience field
          exists and activates with no re-sequence once Tier-1 Phase 5 lands.
          Pure read; mutates nothing.
    """
    try:
        from samia.core import bio as _bio
    except Exception:
        return False
    guard = getattr(_bio, "salience_merge_guard", None)
    if guard is None:
        return False
    try:
        return bool(guard(Path(memory_dir), loser, is_duplicate=False))
    except Exception:
        return False


def passive_sweep(memory_dir: Path,
                  budget: Optional[int] = None,
                  cursor: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """PASSIVE supersession sweep over a bounded slice of the WHOLE index (P3c).

    What: the offline (REM-idle) arm of the P3 detector. Walks a cursor-tracked
          slice of nodes/ (<= ``budget`` nodes per call), and for each node in
          the slice:
            1. runs find_supersession_candidates(scope_nodes=None) — cosine over
               the WHOLE index (the passive scope) + the jaccard pre-filter;
            2. runs the LLM judge (judge_contradictions) to CONFIRM a true
               contradiction (A asserts X vs B asserts not-X); the judge is
               affordable here because REM is idle-budgeted;
            3. on a judge-CONFIRMED contradiction, AUTO-supersedes the loser
               (older / lower-confidence, per _pick_superseded) via the
               RESTORABLE path: set valid_to on the loser + ia.forget_node(
               reason="supersede") (full archive -> restore_node un-forgets) —
               UNLESS the P3 SALIENCE GUARD fires (the loser is a DISTINCT high-
               salience memory), in which case the supersession is SURFACED for
               operator review (status="surfaced-salience") instead of auto-
               removed (D6 effect iii / Q5a);
            4. records weaker / unjudged / judge-uncertain hits to the unified
               candidate store with mode="passive" (NOT deleted) for review.
          The cursor (an int index over the sorted node list) advances by the
          slice size and WRAPS/resets at the end so a full pass completes across
          many REM cycles. Checkpointed via the rem_cycle cursor helpers.
    Why:  Q3 + the Q4 override. The passive mode is the exhaustive global
          reconciler the bounded online locus cannot be; it is gated behind REM
          (a subscriber) AND behind ASTHENOS_CONTRADICTION_ENABLED so it is inert
          by default. Every auto-supersede is reversible — the auto action carries
          no irreversible risk (full archive + restore_node + self-healing).

    Gating (DOUBLE, both inert by default):
        (a) it is a REM subscriber, so the driver only calls it inside REM; and
        (b) it no-ops unless is_enabled() (ASTHENOS_CONTRADICTION_ENABLED) — the
            same posture as the P3b online path.

    Args:
        memory_dir: the memory root.
        budget: max nodes to process this call (default _PASSIVE_BUDGET).
        cursor: an explicit cursor override (tests); else read from rem_cursors.

    Returns:
        {work_remaining, made_progress, judged, superseded, recorded, guarded,
         cursor, processed, total} — work_remaining is True while the cursor has
        not yet wrapped a full pass OR candidates are pending; guarded counts the
        judge-confirmed contradictions the P3 salience guard surfaced for review
        instead of auto-superseding.
    """
    out: dict[str, Any] = {
        "work_remaining": False, "made_progress": False,
        "judged": 0, "superseded": 0, "recorded": 0, "guarded": 0,
        "processed": 0, "total": 0,
    }

    # GATE (b): inert unless the operator enabled contradiction detection.
    if not is_enabled():
        out["enabled"] = False
        return out
    out["enabled"] = True

    mem = Path(memory_dir)
    node_ids = _list_node_ids(mem)
    total = len(node_ids)
    out["total"] = total
    if total == 0:
        return out

    # Cursor: an int index over the sorted node list. Read from the REM cursor
    # store unless an explicit override is given (tests / direct calls).
    if cursor is None:
        try:
            from samia.runtime import rem_cycle as _rem
            cursor = _rem.read_cursor(mem, _PASSIVE_CURSOR_KEY)
        except Exception:
            cursor = {}
    start = int((cursor or {}).get("index", 0))
    if start < 0 or start >= total:
        start = 0  # wrap / out-of-range reset

    cap = _PASSIVE_BUDGET if budget is None else int(budget)
    end = min(start + max(0, cap), total)
    slice_ids = node_ids[start:end]
    out["processed"] = len(slice_ids)

    today = datetime.now(tz=timezone.utc).date().isoformat()
    judged = superseded = recorded = guarded = 0

    # What: count finder/judge failures across the slice instead of logging one
    #   warning per node.
    # Why: a systemic finder fault (e.g. the old entries[i] KeyError that
    #   stringified to "34") otherwise produced one warning PER node every REM
    #   cycle -- a churning log storm across thousands of nodes. We now summarize
    #   once at debug level with a representative example, so a transient/unprocessable
    #   node is skipped cleanly without per-node noise.
    finder_fail = 0
    finder_fail_example: tuple[str, str] | None = None
    judge_fail = 0
    judge_fail_example: tuple[str, str] | None = None

    skipped_excluded = 0
    for node_id in slice_ids:
        # already-purged within this slice (e.g. it lost an earlier pair) -> skip.
        if not (mem / "nodes" / f"{node_id}.md").exists():
            continue
        # TYPE-SCOPING: the node-being-checked is episodic/experiential
        # (session_offload / bug) -> SKIP it entirely. Don't even spend cosine
        # work on an excluded node; it cannot be a contradictable content claim.
        if is_excluded_node(mem, node_id):
            skipped_excluded += 1
            continue
        text = _node_text_for_id(mem, node_id)
        if not text:
            continue
        try:
            # Per-population bar, incoming side (TUNE-2026-06-10 (2)): when the
            # node BEING SWEPT is itself a semantic atom, the whole scan runs at
            # the higher bar (its template kin saturate the recall-first band).
            _thr = (_SEMANTIC_PAIR_THRESHOLD
                    if _node_type(mem, str(node_id)) == "semantic" else None)
            cands = find_supersession_candidates(text, scope_nodes=None,
                                                 memory_dir=mem, threshold=_thr)
        except Exception as exc:  # fail-soft: a finder error never aborts the sweep.
            finder_fail += 1
            if finder_fail_example is None:
                finder_fail_example = (node_id, repr(exc))
            continue
        # Drop self + already-gone candidates.
        cands = [c for c in cands
                 if str(c["node_id"]) not in (node_id, f"{node_id}.md")
                 and (mem / "nodes" /
                      (str(c["node_id"]) if str(c["node_id"]).endswith(".md")
                       else f"{c['node_id']}.md")).exists()]
        if not cands:
            continue

        # LLM judge: confirm a TRUE contradiction (X vs not-X) before acting.
        try:
            verdicts = judge_contradictions(text, cands)
        except Exception as exc:  # fail-soft: judge error -> record, never delete.
            judge_fail += 1
            if judge_fail_example is None:
                judge_fail_example = (node_id, repr(exc))
            verdicts = []
        if verdicts:
            judged += 1

        confirmed_ids = {str(v.get("existing_claim_id", "")).rstrip(".md")
                         for v in verdicts}
        for c in cands:
            cand_id = str(c["node_id"])
            cand_stem = cand_id[:-3] if cand_id.endswith(".md") else cand_id
            if cand_stem in confirmed_ids and cand_stem:
                # Judge-CONFIRMED contradiction -> auto-supersede the LOSER via
                # the RESTORABLE path (set valid_to on loser + forget archive).
                loser, winner = _pick_superseded(mem, node_id, cand_stem)
                if not (mem / "nodes" / f"{loser}.md").exists():
                    continue
                jv = next((v for v in verdicts
                           if str(v.get("existing_claim_id", "")).rstrip(".md")
                           == cand_stem), None)
                # P3 SALIENCE GUARD (D6 effect iii / Q5a): do NOT auto-supersede a
                # DISTINCT high-salience loser — surface it for operator review
                # instead (record a guarded candidate, never auto-remove). The
                # guard is about distinct high-salience claims; the contradiction
                # pair is distinct by construction (X vs not-X), so is_duplicate
                # stays False.
                if _salience_guards_supersede(mem, loser):
                    record_supersession_candidate(
                        mem, loser, winner, cosine=float(c.get("score", 0.0)),
                        jaccard=c.get("jaccard"), mode="passive", judge=jv,
                        status="surfaced-salience")
                    guarded += 1
                    continue
                try:
                    from samia.core import temporal as _temporal
                    _temporal.set_valid(mem, f"{loser}.md", None, today)
                except Exception:
                    pass  # best-effort close; the archive preserves the body.
                try:
                    from samia.core import ia as _ia
                    _ia.forget_node(mem, f"{loser}.md", reason="supersede",
                                    superseded_by=f"{winner}.md")
                except Exception as exc:
                    _log.warning("contradiction: passive supersede failed %s: %s",
                                 loser, exc)
                    continue
                record_supersession_candidate(
                    mem, loser, winner, cosine=float(c.get("score", 0.0)),
                    jaccard=c.get("jaccard"), mode="passive", judge=jv,
                    status="confirmed")
                mark_supersession_confirmed(mem, loser, winner)
                superseded += 1
                # FEAT-2026-06-07 granular-recall-repaired-decay P2 — RECONCILIATION
                # repair: the surviving WINNER was just READ + reconciled, so PARTIALLY
                # heal its integrity (anchor-first, strength < 1.0). Reconciling a memory
                # heals what it touches (Q3a, partial). Additive + fail-soft — a repair
                # error never aborts the sweep; gated by ASTHENOS_CONTRADICTION_ENABLED
                # (we are already inside the is_enabled() guard, so inert by default).
                try:
                    from samia.core import integrity as _integrity
                    _integrity.partial_repair(mem, f"{winner}.md",
                                              trigger="reconciliation")
                except Exception:
                    pass
            else:
                # Weaker / unjudged / judge-uncertain -> RECORD (not deleted).
                record_supersession_candidate(
                    mem, cand_stem, node_id, cosine=float(c.get("score", 0.0)),
                    jaccard=c.get("jaccard"), mode="passive")
                recorded += 1

    # What: emit ONE summarized warning per sweep for skipped (un-processable)
    #   nodes, instead of one per node.
    # Why: avoid the per-node "passive finder failed for %s: 34" log storm; the
    #   sweep already fails soft (skip + continue). A single count + example keeps
    #   forensics without churn. Counts are surfaced in the return dict too.
    out["finder_failures"] = finder_fail
    out["judge_failures"] = judge_fail
    if finder_fail:
        ex_id, ex_msg = finder_fail_example or ("?", "?")
        _log.warning(
            "contradiction: passive finder skipped %d/%d unprocessable node(s) "
            "(e.g. %s: %s)", finder_fail, len(slice_ids), ex_id, ex_msg)
    if judge_fail:
        ex_id, ex_msg = judge_fail_example or ("?", "?")
        _log.warning(
            "contradiction: passive judge failed for %d node(s) (e.g. %s: %s)",
            judge_fail, ex_id, ex_msg)

    out["judged"] = judged
    out["superseded"] = superseded
    out["recorded"] = recorded
    out["guarded"] = guarded
    out["skipped_excluded"] = skipped_excluded
    out["made_progress"] = bool(slice_ids)

    # Advance the cursor; WRAP/reset at the end of a full pass.
    new_index = end if end < total else 0
    wrapped = end >= total
    cursor_out = {"index": new_index, "total": total, "wrapped": wrapped,
                  "remaining": (not wrapped)}
    out["cursor"] = cursor_out

    # work_remaining (G2-2026-06-11, MACHINE-DRAINABLE ONLY): True ONLY while THIS
    # subscriber can still drain work in a future REM cycle WITHOUT operator action —
    # i.e. the cursor has not yet wrapped a full pass over the index. Un-resolved
    # supersession candidates pending operator/judge confirmation are OPERATOR-GATED:
    # no machine cycle can clear them, so they MUST NOT keep REM awake (they used to
    # OR into work_remaining here, which made every wake report work_remaining=true and
    # never let evaluate() reach REST). The pending count is still surfaced as telemetry
    # (operator_gated_pending) for observability, but it does not gate the sleep cycle.
    pending = False
    try:
        pending = bool(list_supersession_candidates(mem, unresolved_only=True))
    except Exception:
        pending = False
    out["operator_gated_pending"] = pending
    out["work_remaining"] = (not wrapped)

    # Checkpoint the cursor (unless an explicit cursor override was supplied,
    # in which case the caller owns persistence).
    if cursor is not None and cursor.get("__no_persist__"):
        pass
    else:
        try:
            from samia.runtime import rem_cycle as _rem
            _rem.write_cursor(mem, _PASSIVE_CURSOR_KEY, cursor_out)
        except Exception as exc:
            _log.debug("contradiction: passive cursor checkpoint failed: %s", exc)
    return out


def passive_has_work(memory_dir: Path) -> bool:
    """due_condition for the REM subscriber: there are nodes to sweep AND enabled.

    What: True iff contradiction detection is enabled AND the nodes/ index is
          non-empty. The wrap-at-end cursor means a finished pass simply restarts
          on the next cycle; "has nodes" is the right standing due-signal.
    Why:  Q3 — the passive sweep is due whenever there is an index to reconcile
          and the operator has enabled the feature; otherwise it never fires
          (double-gate: REM + is_enabled()).
    """
    if not is_enabled():
        return False
    nodes = Path(memory_dir) / "nodes"
    if not nodes.is_dir():
        return False
    try:
        return any(nodes.glob("*.md"))
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Phase 2: LLM judge gate (opt-in)
# ---------------------------------------------------------------------------

# _JUDGE_INFER_MAX_TOKENS / _SYNTH_INFER_MAX_TOKENS -- What: generation budgets
#   for the two in-process inference calls (judge verdict / abstraction synthesis).
# _JUDGE_INFER_MAX_TOKENS / _SYNTH_INFER_MAX_TOKENS -- Why: mirror the old
#   llama-cli `-n` values (512 / 768) so the rewired in-process call asks for the
#   same amount of structured JSON output.
_JUDGE_INFER_MAX_TOKENS: int = 512
_SYNTH_INFER_MAX_TOKENS: int = 768

# ---------------------------------------------------------------------------
# TUNE-2026-06-08: DEDICATED FAST JUDGE BACKEND (BitNet-2B, not the main Qwen-14B)
# ---------------------------------------------------------------------------
#
# The judge runs once per candidate-bearing node across a whole-index passive
# sweep; on the main Qwen-14B CPU backend that is far too slow. Route the judge
# (and synthesize_node) to a DEDICATED small BitNet-2B backend, loaded ONCE and
# cached, instead of the slow 14B. The dedicated backend rides inference's
# per-model-path cache (get_backend_for_model), so the judge model loads a single
# time and the judge never duplicates the 14B.
# Generic fallback: env supplies the real path on a configured box (cls-flags
# sets ASTHENOS_CONTRADICTION_JUDGE_MODEL); the literal here is only the
# unset-env default.
# DEFAULT SWAP (SLOT-STUDY 2026-06-12, operator-directed): the BitNet i2_s
# default never loads under stock llama-cpp-python (its int2/ternary kernel is
# bitnet.cpp-specific), so the unset-env judge silently fell to MockBackend —
# probe measured TPR 0.0. Qwen3-4B as judge measured TPR 0.9 / FPR 0.0 on the
# same probe corpus. The default is the REGISTRY LOGICAL NAME (not a path):
# get_backend_for_model -> fetch_model resolves it on disk or via the gated
# self-fetch, and it is the same model the fact extractor uses, so the cached
# backend is shared. BitNet remains selectable via env for bitnet.cpp setups.
_JUDGE_MODEL_DEFAULT = "Qwen3-4B-Instruct-2507-Q4_K_M"


def _judge_model_path() -> str:
    """The dedicated judge model (env-overridable; default = Qwen3-4B registry name)."""
    return os.environ.get("ASTHENOS_CONTRADICTION_JUDGE_MODEL", _JUDGE_MODEL_DEFAULT)


def _judge_backend() -> Any:
    """The DEDICATED, CACHED small backend the judge + synth use (fail-soft).

    What: builds (once) a backend for ASTHENOS_CONTRADICTION_JUDGE_MODEL
          (Qwen3-4B registry default) via inference.get_backend_for_model, which
          caches the LlamaCppBackend by model path so the small model loads a
          SINGLE time and is reused on every judge/synth call. When that factory
          is unavailable (older inference module) or the judge model is missing /
          not a .gguf / llama_cpp absent, it returns the MAIN backend
          (inference.get_backend()) so existing behavior + existing tests (which
          mock get_backend) are preserved. Any import error -> None.
    Why:  fix #2 -- the judge must NOT ride the slow 14B. A dedicated cached small
          backend keeps the passive sweep affordable while preserving the
          fail-soft contract (a MockBackend / unavailable backend -> the judge
          no-ops records-only/None exactly as before).
    """
    try:
        from samia.runtime import inference as _inf
    except Exception as exc:
        _log.debug("contradiction: inference module unavailable: %s", exc)
        return None
    factory = getattr(_inf, "get_backend_for_model", None)
    if factory is not None:
        try:
            dedicated = factory(_judge_model_path())
        except Exception as exc:
            _log.debug("contradiction: judge backend build failed: %s", exc)
            dedicated = None
        # Use the dedicated small backend ONLY when it is a REAL (non-mock)
        # backend -- i.e. the BitNet-2B gguf is present and loadable. When the
        # judge model is absent (the factory returns a MockBackend), fall back to
        # the main in-process backend so existing behavior + the existing tests
        # (which mock get_backend) are preserved.
        if dedicated is not None and type(dedicated).__name__ != "MockBackend":
            return dedicated
    # Fallback: the main in-process backend (and the path existing tests mock).
    try:
        return _inf.get_backend()
    except Exception as exc:
        _log.debug("contradiction: inference backend unavailable: %s", exc)
        return None


def _inference_available() -> bool:
    """True iff a REAL (non-mock) dedicated judge backend is loadable.

    What: asks _judge_backend() (the dedicated BitNet-2B small backend, cached)
          and reports whether it is anything OTHER than MockBackend (the fail-soft
          "no model configured" signal). Any import/init error -> False.
    Why:  the availability probe must reflect the DEDICATED judge backend being
          real, not the main 14B. _judge_backend() returns MockBackend when the
          judge model is unset / missing / llama_cpp absent (and the main fallback
          is also Mock), which is exactly the unavailable case the judge gate and
          synthesis must NO-OP on (records-only / None). Pure read.
    """
    backend = _judge_backend()
    if backend is None:
        return False
    return type(backend).__name__ != "MockBackend"


def _infer_text(prompt: str, max_tokens: int) -> Optional[str]:
    """Generate text from the dedicated judge backend (or None, fail-soft).

    What: routes *prompt* through the DEDICATED small judge backend
          (_judge_backend().complete) -- Qwen3-4B by default, cached/loaded-once,
          NOT the slow main Qwen-14B -- and returns the raw completion text.
          Returns None when the backend is a MockBackend / unavailable /
          load-errored, or the call raises.
    Why:  the single inference entrypoint for both the judge and the abstraction
          synthesizer. Routing to a dedicated cached small backend keeps the
          passive sweep affordable AND stops the judge duplicating the 14B. A
          backend error must NEVER block or corrupt a write, so every failure
          collapses to None (the caller's records-only/None no-op).
    """
    backend = _judge_backend()
    if backend is None:
        _log.debug("contradiction: judge backend unavailable")
        return None
    # MockBackend is the "no real model configured" fail-soft signal: treat it
    # exactly like an unavailable backend so nothing is auto-acted on canned text.
    if type(backend).__name__ == "MockBackend":
        _log.debug("contradiction: inference backend is MockBackend; skipping")
        return None
    try:
        return backend.complete(prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception as exc:
        _log.warning("contradiction: in-process inference call failed: %s", exc)
        return None


def _parse_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract the FIRST JSON object from an LLM completion (tolerates trailing text).

    What: find the first '{', then json.JSONDecoder().raw_decode from there —
          raw_decode parses ONE JSON value and stops, ignoring whatever the model
          appended after the closing brace. Returns the parsed object, or None when
          there is no '{' or the candidate region is not valid JSON.
    Why:  BUG-2026-06-11 judge-parse — ~95% of judge (and synth) outputs were
          discarded because the model emits trailing commentary AFTER a valid JSON
          object (e.g. `{...}\n\nThis means...`). The old json.loads(text[start:])
          fed that whole tail to the parser and raised "Extra data", throwing away
          good JSON. raw_decode keeps the genuinely-non-JSON failure path intact
          (returns None -> caller's existing fallback heuristics still run).
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


# _JUDGE_PROMPT_TEMPLATE -- What: structured prompt for the LLM judge.
# _JUDGE_PROMPT_TEMPLATE -- Why: instructs the model to output JSON with
#   contradiction confidence per candidate. Explicitly excludes temporal
#   changes from contradiction classification.
_JUDGE_PROMPT_TEMPLATE = """You are a memory consistency judge. Given a NEW claim and a list of EXISTING claims from a memory store, determine if the new claim contradicts any existing claim.

A contradiction means both claims cannot be true simultaneously. Temporal changes (preferences that evolved over time) are NOT contradictions.

NEW CLAIM:
{new_claim}

EXISTING CLAIMS:
{existing_claims}

Respond ONLY with a JSON object:
{{"contradictions": [{{"existing_claim_id": "...", "explanation": "...", "confidence": 0.0-1.0}}]}}

If there are no contradictions, respond: {{"contradictions": []}}"""


def judge_contradictions(
    new_text: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the LLM judge on contradiction candidates.

    What: constructs a prompt with the new claim and candidate claims,
          routes it through the in-process inference backend
          (samia.runtime.inference.get_backend().complete), parses the
          structured JSON response.
    Why:  Phase 2 precision gate -- embedding similarity catches topical
          overlap, but the LLM judge distinguishes "same topic, different
          claim" from "same topic, compatible claim".

    Parameters
    ----------
    new_text : str
        The incoming write text.
    candidates : list of dicts
        Candidate nodes from find_contradiction_candidates.

    Returns
    -------
    list of dicts with keys: existing_claim_id, explanation, confidence.
    Only contradictions with confidence >= _JUDGE_CONFIDENCE_THRESHOLD.
    Empty list if the judge is disabled or fails.
    """
    if not _JUDGE_ENABLED:
        return []

    if not candidates:
        return []

    # What: format existing claims for the prompt.
    # Why: the judge needs both the claim ID and text to reference.
    claims_text = ""
    for i, c in enumerate(candidates, 1):
        claims_text += f"{i}. [{c['node_id']}] {c.get('title', '(no title)')}\n"

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        new_claim=new_text[:2000],
        existing_claims=claims_text,
    )

    # What: route through the in-process inference backend.
    # Why: the daemon already holds the loaded Qwen backend in-process (the
    #   passive sweep runs INSIDE the daemon), so we call get_backend().complete
    #   directly — no IPC round-trip, no llama-cli subprocess. _infer_text is the
    #   single fail-soft entrypoint: it returns None when the backend is a
    #   MockBackend / unavailable / load-errored or the call raises, which we map
    #   to the SAME empty (records-only) result as before. A judge error never
    #   blocks or corrupts a write.
    response_text = _infer_text(prompt, _JUDGE_INFER_MAX_TOKENS)
    if not response_text:
        return []

    # What: extract the FIRST JSON object from the response (the model may emit
    #   preamble before AND trailing commentary after the JSON block).
    # Why: BUG-2026-06-11 — raw_decode stops at the JSON object's closing brace
    #   and ignores trailing text, instead of json.loads choking on "Extra data"
    #   (which was discarding ~95% of otherwise-valid judge outputs).
    parsed = _parse_first_json_object(response_text)
    if parsed is None:
        _log.warning("contradiction: no parseable JSON in judge response")
        return []

    try:
        contradictions = parsed.get("contradictions", [])

        # What: filter by confidence threshold.
        # Why: low-confidence judgments are noise; only high-confidence
        #   contradictions should block or flag writes.
        return [
            c for c in contradictions
            if float(c.get("confidence", 0)) >= _JUDGE_CONFIDENCE_THRESHOLD
        ]
    except (KeyError, TypeError, ValueError) as exc:
        _log.warning("contradiction: judge response parse error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# FEAT-2026-06-07 Tier-2 merge consumer P2: LLM ABSTRACTION SYNTHESIS
# (reuses the SAME in-process inference backend as the judge — Gemini probe #3,
#  abstractive compression of an episodic pair into one semantic node).
# ---------------------------------------------------------------------------

# _SYNTH_PROMPT_TEMPLATE -- What: the synthesis prompt for the abstraction call.
# _SYNTH_PROMPT_TEMPLATE -- Why: the merge consumer's P2 distinct-but-overlapping
#   pair needs a single higher-level semantic node that SUBSUMES both sources
#   (episodic->semantic). Asks for strict JSON {title, body} so the staged draft
#   has a clean shape; mirrors the judge's "respond ONLY with JSON" contract.
_SYNTH_PROMPT_TEMPLATE = """You are a memory abstraction synthesizer. Given TWO related memory notes, write ONE higher-level note that captures the shared concept both express, without losing the distinct detail each carries.

NOTE A:
{note_a}

NOTE B:
{note_b}

Respond ONLY with a JSON object:
{{"title": "<short title for the unified note>", "body": "<the synthesized higher-level note body>"}}"""


def synthesis_enabled() -> bool:
    """True iff the local LLM synthesis backend is available to call.

    What: reuses the SAME enable flag as the LLM judge
          (ASTHENOS_CONTRADICTION_JUDGE) AND additionally requires a REAL
          in-process inference backend (get_backend() is not a MockBackend) —
          the synthesis call rides the judge's in-process inference plumbing, so
          it is available exactly when the judge can actually run a model.
    Why:  Tier-2 P2 — synthesis must be a SAFE NO-OP when inference is off
          (same conservative posture as the judge being disabled). The merge
          consumer checks this before attempting any abstraction, leaving the
          pair pending rather than crashing. Probing get_backend() (not
          "is llama-cli on PATH") is the rewired availability signal.
    """
    return _JUDGE_ENABLED and _inference_available()


def synthesize_node(text_a: str, text_b: str) -> Optional[dict[str, Any]]:
    """Synthesize one higher-level node from two source bodies (P2 abstraction).

    What: runs the SAME in-process inference backend judge_contradictions uses
          (samia.runtime.inference.get_backend().complete), with a synthesis
          prompt, and parses the structured JSON {title, body}. Returns that
          dict, or None when synthesis is disabled / the backend is unavailable /
          the response is unparseable.
    Why:  Tier-2 merge consumer P2 (Q1c/Q2c) — abstractive compression of a
          distinct-but-overlapping episodic pair into one semantic node. Reuses
          the existing inference entrypoint (no new model loader); the None
          return is the safe no-op the consumer relies on to leave the pair
          pending instead of crashing.

    Returns
    -------
    {"title": str, "body": str} or None.
    """
    if not synthesis_enabled():
        return None

    prompt = _SYNTH_PROMPT_TEMPLATE.format(
        note_a=str(text_a)[:2000],
        note_b=str(text_b)[:2000],
    )

    # In-process inference (same backend as the judge). _infer_text fails soft to
    # None when the backend is a MockBackend / unavailable / load-errored or the
    # call raises — the merge consumer relies on that None to leave the pair
    # pending instead of crashing.
    response_text = _infer_text(prompt, _SYNTH_INFER_MAX_TOKENS)
    if not response_text:
        return None

    # BUG-2026-06-11 — same trailing-text tolerance as the judge: raw_decode the
    # first JSON object so a synth completion with appended commentary still parses.
    parsed = _parse_first_json_object(response_text)
    if parsed is None:
        _log.warning("contradiction: no parseable JSON in synth response")
        return None
    try:
        title = str(parsed.get("title", "")).strip()
        body = str(parsed.get("body", "")).strip()
    except (KeyError, TypeError, ValueError) as exc:
        _log.warning("contradiction: synth response parse error: %s", exc)
        return None
    if not body:
        _log.warning("contradiction: synth produced empty body")
        return None
    return {"title": title, "body": body}


# ---------------------------------------------------------------------------
# Phase 3: Memory guard integration
# ---------------------------------------------------------------------------


def check_contradiction(
    payload: dict[str, Any],
    memory_dir: Optional[Path] = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Full contradiction check pipeline for a memory write payload.

    What: orchestrates Phase 1 (embedding candidates) and Phase 2 (LLM judge)
          and returns both the reason strings for memory_guard and the
          detailed contradiction metadata for the pending queue.
    Why:  single integration point that memory_guard calls during
          _validate_write. Returns data in the format memory_guard expects
          (reason strings) plus extended metadata for the MemGuardPanel
          (contradiction_with field).

    Parameters
    ----------
    payload : dict
        The memory write payload.
    memory_dir : Path or None
        Memory directory for vector index access.

    Returns
    -------
    (reasons, contradiction_metadata) tuple.
        reasons: list of strings for memory_guard's reason field.
        contradiction_metadata: list of dicts with node_id, title, score,
            plus optional judge fields (explanation, confidence).
    """
    if not _ENABLED:
        return [], []

    text = json.dumps(payload, default=str)
    mem = memory_dir or _MEMORY_DIR

    # Phase 1: embedding candidate finder.
    candidates = find_contradiction_candidates(text, memory_dir=mem)
    if not candidates:
        return [], []

    reasons: list[str] = []
    metadata: list[dict[str, Any]] = []

    # Phase 2: optional LLM judge.
    if _JUDGE_ENABLED:
        judge_results = judge_contradictions(text, candidates)
        if judge_results:
            for jr in judge_results:
                cid = jr.get("existing_claim_id", "?")
                conf = jr.get("confidence", 0)
                reasons.append(
                    f"contradiction_judge:id={cid}:conf={conf:.2f}"
                )
                metadata.append({
                    "node_id": cid,
                    "explanation": jr.get("explanation", ""),
                    "confidence": conf,
                    "source": "llm_judge",
                })
            return reasons, metadata

    # What: if no judge or judge found nothing, report embedding candidates.
    # Why: even without the LLM judge, high-similarity candidates are worth
    #   flagging for operator review.
    for c in candidates:
        reasons.append(
            f"contradiction_embedding:node={c['node_id']}:sim={c['score']:.3f}"
        )
        metadata.append({
            "node_id": c["node_id"],
            "title": c.get("title", ""),
            "score": c["score"],
            "source": "embedding_similarity",
        })

    return reasons, metadata


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.contradiction
# Phase:      AUD60 Phases 1-3 (embedding + judge + guard integration)
#             + FEAT-2026-06-07 P3a (supersession candidate finder/recorder)
#             + FEAT-2026-06-07 P3b/R2 (CANONICAL unified supersession store:
#               record + list-unresolved + mark-confirmed + mark-dismissed)
#             + FEAT-2026-06-07 P3c (PASSIVE REM-subscriber sweep: incremental,
#               cursor-tracked, whole-index cosine + LLM judge -> auto-supersede
#               the loser via the RESTORABLE path; double-gated REM + is_enabled())
#             + FEAT-2026-06-07 Tier-2 merge consumer P2 (LLM ABSTRACTION SYNTHESIS:
#               synthesize_node reuses the SAME in-process inference judge backend
#               with a synthesis prompt to fold a distinct-but-overlapping episodic
#               pair into one semantic node; gated by synthesis_enabled()
#               == _JUDGE_ENABLED AND a real (non-mock) backend, safe None no-op
#               when off — the merge consumer leaves the pair pending)
#             + FIX-2026-06-08 (JUDGE/SYNTH REWIRE: judge_contradictions and
#               synthesize_node no longer shell out to a non-existent `llama-cli`
#               binary; they call the in-process backend
#               samia.runtime.inference.get_backend().complete via _infer_text.
#               get_backend() returning a MockBackend / unavailable / load-error /
#               raise -> the SAME records-only empty / None no-op as before. The
#               availability probe is now "real backend configured", not
#               "llama-cli on PATH". No subprocess use remains.)
#             + BUG-2026-06-07 (find_contradiction_candidates now reads the CANONICAL
#               dict-shaped manifest entries {fname -> {sha256,title,row}} via a
#               by_row map mirroring vector.query, instead of indexing entries[i]
#               with the embeddings ROW number -- the old list-access raised
#               KeyError(row) on every node, stringifying to the "passive finder
#               failed for %s: 34" storm. passive_sweep now SUMMARIZES finder/judge
#               failures (one warning + counts) instead of one warning per node.)
# Layer:      runtime (in-daemon; the single owner of the supersession-candidate
#             store that memory_guard's surfacer + mcp_server's confirm/dismiss/
#             list all route through)
# Stability:  v0.4 -- all phases wired, default-off via env vars
#             + BUG-2026-06-11 judge-parse: judge_contradictions + synthesize_node
#               now parse the FIRST JSON object via json.JSONDecoder().raw_decode
#               (shared _parse_first_json_object) so trailing model commentary
#               after a valid JSON object no longer fails the parse (~95% of judge
#               outputs were being discarded as "Extra data"); genuinely-non-JSON
#               output still falls through to the existing empty/None fallback.
#             + BUG-2026-06-11 supersession dedup: record_supersession_candidate
#               skips the append when an UNRESOLVED row for the same (old_id,new_id)
#               already exists (a resolved prior row does NOT suppress a fresh
#               re-detection), stopping unbounded duplicate growth of the store.
# ErrorModel: fail-open for all phases; embedding/judge unavailability
#             degrades gracefully to empty results (write proceeds). Store
#             read/mark are fail-soft (warn + skip; rewrite is tmp+replace).
# Depends:    numpy (optional), samia.core.vector (optional),
#             samia.core.consolidation (shingles/jaccard, optional),
#             transformers/torch (optional),
#             samia.runtime.inference (in-process LLM judge backend, optional).
#             json, logging, os, datetime, pathlib (stdlib).
# Exposes:    configure, find_contradiction_candidates, judge_contradictions,
#             check_contradiction, find_supersession_candidates,
#             record_supersession_candidate, list_supersession_candidates,
#             mark_supersession_confirmed, mark_supersession_dismissed,
#             passive_sweep, passive_has_work (P3c REM-subscriber arm),
#             synthesis_enabled, synthesize_node (Tier-2 P2 abstraction synthesis),
#             excluded_types, is_excluded_node (TUNE-2026-06-08 type-scoping).
#             + TUNE-2026-06-08 (USABILITY TUNE, the detector was OFF/flooded):
#               (1) TYPE-SCOPING -- excluded_types()/is_excluded_node() exclude
#               episodic/experiential nodes (session_offload, bug; env
#               ASTHENOS_CONTRADICTION_EXCLUDE_TYPES) at EVERY enumeration/match
#               site: find_supersession_candidates (drop excluded matches),
#               passive_sweep (skip excluded node-being-checked before cosine),
#               bio.active_set (drop excluded from the online locus). Collapses
#               ~807K self-similar episodic pairs to a few hundred content pairs.
#               (2) DEDICATED FAST JUDGE -- _judge_backend()/_judge_model_path()
#               route the judge + synthesize_node to a CACHED small BitNet-2B
#               backend (env ASTHENOS_CONTRADICTION_JUDGE_MODEL) via
#               inference.get_backend_for_model, NOT the slow main Qwen-14B;
#               fail-soft falls back to get_backend() (preserves existing tests)
#               and to records-only/None when no real model is present.
# G2-2026-06-11 (REM machine-drainable-only): passive_sweep's work_remaining now
#   reflects ONLY the machine-drainable cursor (not wrapped) — un-resolved
#   supersession candidates pending OPERATOR confirmation no longer OR into
#   work_remaining (they are surfaced as the operator_gated_pending telemetry key
#   instead). A wrapped sweep with only operator-gated pending candidates reports
#   work_remaining=False, letting REM evaluate() reach REST (it could never rest
#   before — every wake reported work). No machine work is lost: the cursor still
#   keeps the sweep alive until a full pass wraps.
# Lines:      ~905
# --------------------------------------------------------------------------
