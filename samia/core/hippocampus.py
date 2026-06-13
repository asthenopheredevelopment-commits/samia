"""samia.core.hippocampus — the Tier-1 hippocampal fast store (P1 engram + P2 ring + P3 lattice + P4 inject).

FEAT-2026-06-07-memory-tier1-hippocampal-quad-v01, Phases 1 + 2 + 3 + 4.

This module builds two of the four populations of the Tier-1 hierarchy:

  P1 — the ENGRAM store: persistent, self-contained HELD COPIES of source nodes
       (the fast-episodic tier, days-to-months) + a dedicated RAG fast index.
  P2 — the RING store: a VOLATILE working set of POINTERS (hours) into main/engram
       — NOT held copies — capacity/LRU bounded, with ring-RAG that dereferences the
       pointers to their backing content at query time.

P1 scope (built in P1):
  - EngramStore: a held-copy store under <memory_dir>/hippocampus/engram/<id>.json.
    Each entry is a SELF-CONTAINED copy (frontmatter + body + embedding row), so it
    survives churn (freeze/merge/delete) of its main source — it is a COPY, not a
    pointer (pointers are the RING, P2).
  - materialize(): the copy primitive. Reads the canonical via the source pointer and
    writes an engram held copy + appends its embedding to the fast index.
  - engram_rag_query(): engram-RAG retrieval over the held copies, recency-preferential.

P2 scope (built here, this phase):
  - RingStore: a capacity/LRU-bounded store of POINTERS (the opposite of the engram's
    held copy). A ring entry references a backing main/engram node by id + minimal
    metadata (last_access, access_count, salience_flag); it holds NO content. Backed by
    <memory_dir>/hippocampus/ring.jsonl so it survives a daemon restart but is
    explicitly volatile. add/touch (LRU) + capacity-bounded eviction + a
    dangling-pointer-SAFE resolve: a pointer whose backing is gone resolves to nothing.
  - ring_rag_query(): ring-RAG retrieval — DEREFERENCES the live ring pointers to their
    backing content, embeds + cosines against the query, returns hits tagged via='ring'.
    Because it dereferences at query time, ring-RAG reflects the CURRENT backing (the
    pointer invariant: opposite of the engram copy invariant). A dangling pointer
    contributes nothing.

P3 scope (built here, this phase):
  - kWTA pattern separation: materialize() now computes a kWTA sparse code (bio.
    kwta_sparse_code — random-projection + top-k% winners) on the held copy's
    embedding and stores it as `kwta_code`, so near-duplicate episodes stay
    individually addressable (orthogonalization). The cosine gate stays a dedup filter.
  - The PROMOTION LATTICE + its AUTO trigger (a FUNCTION, not a new scheduler):
      (a) ring -> engram: promote_ring_pointer / promote_ring_step auto-materialize a
          ring pointer to an engram copy when its genuine ring-RAG hits reach
          RING_PROMOTE_HITS OR max(access_signal, salience) crosses the promote bar —
          so a HIGH-SALIENCE one-shot promotes WITHOUT repeated access (the D6 gate).
      (b) engram -> inject-eligible: mark_inject_eligible flags an engram copy
          inject_eligible when max(attractor_strength, salience) clears
          INJECT_PROMOTE_THRESHOLD (attractor = its Tier-0 edge w>=HEBB_PROMOTION &
          count_genuine>=k via bio._load_edge_weights/_is_promotable, OR salience high).
  - promote-before-evict: RingStore eviction now MATERIALIZES a still-wanted pointer
    (recent genuine hits OR salience >= threshold) into an engram copy before dropping
    it from the ring — no dangling loss of a wanted trace.

P4 scope (built here, this phase):
  - assemble_inject_block(): the TWO inject layers under a FIXED token budget (D4/Q5a).
      ENGRAM-INJECT = the inject_eligible held copies (the P3 flag): the SMALL always-on
          identity / known-cold set ("a name you hold"), reserved a budget slice and
          FAVORED under pressure.
      RING-INJECT  = the live ring pointers dereferenced to backing content: the current
          auto-pilot working set ("a habit"), filling the REMAINING budget by relevance.
    Both layers are RELEVANCE-GATED against the turn query via cosine (estimate_tokens +
    _relevance_score), priority-arbitrated when over budget (engram-inject first, then
    ring-inject by relevance; overflow dropped). The block NEVER exceeds the budget; an
    empty ring + empty engram yields an empty block (fail-open).
  - CO-ACTIVATION SILENCE (D5/Q6a, the homeostasis keystone): NO inject path calls
    hebbian_record / feeds the Tier-0 web. The relevance gate is a bare embed + cosine
    SORT for inclusion, NOT a genuine retrieval — assembling a block records ZERO genuine
    co-activations (a standing deref is not "recalled together"). inject = standing
    availability / O(1) pointer-deref, NOT a search (RAG is the search path).
  - The actual per-turn injection into the LIVE prompt stays operator-gated/INERT: P4
    builds the assembler + an inert MCP surface (mcp_server.memory_inject_block /
    memory_search include_inject), NOT an always-on prompt mutation.

NOT here: feed-forward / genuine-once replay, salience-dampened DECAY, freeze-exemption
(P5 — P4 does NOT change decay). The SALIENCE SOURCE itself (bio.compute_salience writing
the frontmatter field) landed in P2; P3 CONSUMES that field via the max(attractor,
salience) gates; P4 CONSUMES the P3 inject_eligible flag as the engram-inject layer.

Reuses, reinvents nothing: the embedding backend (vector._embed_batch), the canonical
node reader (vector._load_node_text), and the embeddings.npy + manifest.json index
layout (vector.py). engram-RAG / ring-RAG are the SAME cosine primitive on a small set.

Public API (parameterized on memory_dir):
  EngramStore(memory_dir)               — held-copy store handle (P1)
    .materialize(source_node) -> dict   — copy a main node into an engram held copy
    .get(engram_id) -> dict | None      — read a held copy (self-contained)
    .all() -> list[dict]                — every held copy
  engram_rag_query(memory_dir, text, top_k, recency_boost) -> list[dict]
  RingStore(memory_dir)                 — volatile pointer LRU handle (P2)
    .add(ptr, target_tier, salience_flag) -> dict   — register/refresh a pointer (LRU)
    .touch(ptr) -> dict | None                       — mark a pointer accessed (LRU)
    .resolve(entry) -> dict | None                   — deref a pointer (None if dangling)
    .entries() -> list[dict]                         — live pointer entries (recent-first)
  ring_rag_query(memory_dir, text, top_k) -> list[dict]
  assemble_inject_block(memory_dir, query, token_budget, engram_budget_frac) -> dict (P4)
    — the two-layer standing-availability inject block (engram-inject + ring-inject)
      under a fixed token budget, relevance-gated + priority-arbitrated, co-activation-SILENT.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from . import vector as _vi

# RECENCY_BOOST_DEFAULT -- What: per-query additive bias applied to an engram hit's
#   cosine score, scaled by how recently the copy was materialized/accessed.
# Why: the hippocampal fast tier is preferential-by-recency (CLS); P1's exit gate
#   ("an engram copy wins over an equal-cosine main node") needs the fast-tier hit to
#   out-score a tied main hit. A small additive boost (default 0.05, recency-scaled)
#   does that without distorting genuinely-stronger main cosine hits. Named/tunable.
RECENCY_BOOST_DEFAULT = 0.05

# RECENCY_HALFLIFE_DAYS -- What: age (days) at which the recency boost halves.
# Why: a copy materialized today gets the full boost; one materialized long ago gets
#   little — recency-preferential, not blanket fast-tier preference.
RECENCY_HALFLIFE_DAYS = 14.0

# ENGRAM_TTL_DAYS_DEFAULT -- What: default TTL stamped on a held copy at materialization.
# Why: the engram tier is days-to-months (D1); the TTL is recorded in P1 but the
#   demotion sweep that ACTS on it is P5 (not built here). Stamped so P5 has the field.
ENGRAM_TTL_DAYS_DEFAULT = 90


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


class EngramStore:
    """The Tier-1 engram held-copy store (persistent, self-contained copies).

    What: a directory of self-contained JSON held copies plus a dedicated cosine fast
      index (embeddings.npy + manifest.json mirroring vector.py's layout). Each held
      copy carries its own title + body + embedding row, so it is durable against any
      change to the main source (the defining property of a COPY vs a pointer).
    Why: this is the durable fast-episodic tier — the buffer the rest of the lattice
      (ring, promotion, inject, replay) is built over in later phases. In P1 it stands
      alone and is exercised directly.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)

    # -- index plumbing -----------------------------------------------------

    def _load_manifest(self) -> dict:
        p = _engram_manifest_path(self.memory_dir)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        _engram_index_dir(self.memory_dir).mkdir(parents=True, exist_ok=True)
        _engram_manifest_path(self.memory_dir).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_embeddings(self) -> np.ndarray | None:
        p = _engram_embed_path(self.memory_dir)
        if not p.exists():
            return None
        try:
            return np.load(p)
        except (OSError, ValueError):
            return None

    def _save_embeddings(self, arr: np.ndarray) -> None:
        _engram_index_dir(self.memory_dir).mkdir(parents=True, exist_ok=True)
        np.save(_engram_embed_path(self.memory_dir), arr.astype(np.float32))

    # -- held-copy read -----------------------------------------------------

    def get(self, engram_id: str) -> dict | None:
        """Return the self-contained held copy for engram_id, or None."""
        p = _engram_dir(self.memory_dir) / f"{engram_id}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def all(self) -> list[dict]:
        """Return every held copy in the store (unordered)."""
        d = _engram_dir(self.memory_dir)
        if not d.is_dir():
            return []
        out: list[dict] = []
        for p in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    # -- the copy primitive (P1's materialize) ------------------------------

    def materialize(self, source_node: str,
                    ttl_days: int = ENGRAM_TTL_DAYS_DEFAULT) -> dict:
        """Create a self-contained engram held COPY from a main/source node.

        What: read the canonical node (frontmatter title + embedding-ready content via
          vector._load_node_text), embed it with the SAME backend the main index uses,
          and write a self-contained copy under hippocampus/engram/<id>.json — full
          title + body + its own embedding-index row. Idempotent per source (re-running
          updates the same id in place).
        Why: this is the P1 copy step — the held copy survives later freeze/merge/delete
          of the source because it owns its content and embedding outright (Risk 5: main
          keeps the canonical; the copy is additive, never moves data). In P1 this is
          invoked directly; the AUTOMATIC trigger (frequency/salience/kWTA) is P3.

        Returns the held-copy record dict (also persisted). Raises if the source node
        does not exist (materialize is a deliberate copy of a real node).
        """
        fname = source_node if source_node.endswith(".md") else f"{source_node}.md"
        src_path = Path(self.memory_dir) / "nodes" / fname
        if not src_path.exists():
            raise FileNotFoundError(f"source node not found: {fname}")

        # Reuse the canonical reader so the copy embeds EXACTLY as main would.
        title, content = _vi._load_node_text(src_path)
        body = src_path.read_text(encoding="utf-8")

        # Embed via the shared backend (reinvent nothing).
        vec = _vi._embed_batch([content])[0].astype(np.float32)

        # kWTA pattern separation (P3/D2): compute a sparse code on the COPY's
        # embedding so near-duplicate episodes stay individually addressable. This
        # is the ORTHOGONALIZING sense of pattern separation — distinct from the
        # cosine dedup gate — and runs ONCE here, on the held copy, never on the
        # retrieval embedding (retrieval cosine unaffected). Lazy import avoids a
        # core import cycle (bio imports nothing from hippocampus, but keep it lazy).
        from . import bio as _bio
        kwta_code = _bio.kwta_sparse_code(vec)

        eid = _engram_id(source_node)
        manifest = self._load_manifest()
        entries = manifest.get("entries", {})
        embeddings = self._load_embeddings()

        existing = entries.get(eid)
        if existing is not None and embeddings is not None and \
                existing.get("row") is not None and \
                existing["row"] < embeddings.shape[0]:
            # Re-materialize: overwrite the existing row in place.
            row = int(existing["row"])
            embeddings[row] = vec
        else:
            # New copy: append a fresh row.
            if embeddings is None:
                embeddings = vec.reshape(1, -1)
                row = 0
            else:
                row = int(embeddings.shape[0])
                embeddings = np.vstack([embeddings, vec.reshape(1, -1)])

        now = _now_iso()
        # Preserve an already-earned inject_eligible flag across a re-materialize
        # (a prior held copy may already have been flagged by the engram->inject gate).
        prior_copy = self.get(eid)
        prior_eligible = bool(prior_copy and prior_copy.get("inject_eligible"))
        # FEAT-2026-06-11 temporal-recall P0 — carry the SOURCE node's write-time
        # substrate onto the held copy (§3.3/§3.5).
        # What: lift written_at + episode_seq from the source node's frontmatter so the
        #   SITH sidecar can resolve a hit's ENCODING TIME whether the hit resolved to a
        #   main node or to this held engram copy. We use the source's anchor (not a
        #   freshly-minted one) — the copy must represent the original event's time, not
        #   the copy time. On re-materialize we keep any prior engram value, then the
        #   source value, so the order is stable across re-runs.
        # Why: ADDITIVE-OPTIONAL. A legacy source node lacking the fields yields None for
        #   both; every downstream consumer fails open on absence (no migration, no
        #   backfill). Fail-soft: a parse hiccup leaves both None and never breaks the copy.
        src_written_at = None
        src_episode_seq = None
        try:
            from . import frontmatter as _fm_mod
            parsed_src, _ = _fm_mod.parse(src_path.read_text(encoding="utf-8"))
            if parsed_src is not None:
                src_fm, _ = parsed_src
                if src_fm.get("written_at") is not None:
                    src_written_at = float(src_fm.get("written_at"))
                if src_fm.get("episode_seq") is not None:
                    src_episode_seq = int(src_fm.get("episode_seq"))
        except Exception:
            src_written_at = None
            src_episode_seq = None
        written_at = (prior_copy or {}).get("written_at") if prior_copy else None
        if written_at is None:
            written_at = src_written_at
        episode_seq = (prior_copy or {}).get("episode_seq") if prior_copy else None
        if episode_seq is None:
            episode_seq = src_episode_seq
        record = {
            "engram_id": eid,
            "source_ptr": fname,
            "title": title,
            "body": body,
            "embedding_row": row,
            "kwta_code": kwta_code,
            "materialized_at": (existing or {}).get("materialized_at", now),
            "last_access": now,
            "ttl_days": int(ttl_days),
            # FEAT-2026-06-11 temporal-recall P0 — additive-optional substrate fields
            # (the source's encoding-time anchor + order; None on a legacy source).
            "written_at": written_at,
            "episode_seq": episode_seq,
            # inject_eligible defaults False on a fresh copy; the engram->inject gate
            # (mark_inject_eligible, P3) sets it when max(attractor, salience) clears the
            # bar. P3 never injects here — it only stamps the eligibility field.
            "inject_eligible": prior_eligible,
        }

        # Persist the self-contained held copy.
        _engram_dir(self.memory_dir).mkdir(parents=True, exist_ok=True)
        (_engram_dir(self.memory_dir) / f"{eid}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")

        # Persist the fast index (embeddings + manifest entry).
        entries[eid] = {
            "row": row,
            "title": title,
            "source_ptr": fname,
            "materialized_at": record["materialized_at"],
            "last_access": now,
        }
        manifest_out = {
            "model_id": _vi.MODEL_ID,
            "dim": _vi.EMBED_DIM,
            "built_at": now,
            "engram_count": embeddings.shape[0],
            "entries": entries,
        }
        self._save_embeddings(embeddings)
        self._save_manifest(manifest_out)

        # FEAT-2026-06-11 temporal-recall P2 — SITH encode event + snapshot (§4.3/§4.4).
        # materialize is a context-bearing ENCODE event: (1) observe this copy's embedding
        # into the integrator bank (coalesced off the shared cadence gate, so a burst of
        # materializes collapses to one bank advance — §16.2-Q3); then (2) capture the
        # bank state as this copy's encode-snapshot {c_{h,k}}, keyed by `eid`, written ONCE
        # (re-materialize is a no-op, preserving the first encode-time context). The
        # snapshot lives in a sidecar (NOT frontmatter): a K·dim bank is ~9KB and only the
        # retrieval-time SITH cosine reads it. Fail-soft + lazy import: any error leaves
        # the bank/sidecar untouched and a hit on this node yields a 0.0 TĈ (fails open).
        # Inert until ASTHENOS_TEMPORAL_WEIGHT + γ flip the SITH term on.
        try:
            from . import temporal_recall_sith as _sith
            _sith.integrator_observe(self.memory_dir, vec, now=written_at)
            _sith.capture_snapshot(self.memory_dir, eid)
        except Exception:
            pass
        return record


def _recency_factor(materialized_at: str, halflife_days: float) -> float:
    """Return a [0,1] recency factor (1.0 today, halving every halflife_days)."""
    try:
        ts = _dt.datetime.fromisoformat(materialized_at)
    except (ValueError, TypeError):
        return 0.0
    age_days = max(0.0, (_dt.datetime.now() - ts).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / max(halflife_days, 1e-9)))


def engram_rag_query(memory_dir: Path, text: str, top_k: int = 8,
                     recency_boost: float = RECENCY_BOOST_DEFAULT,
                     halflife_days: float = RECENCY_HALFLIFE_DAYS) -> list[dict]:
    """Engram-RAG retrieval: cosine over the engram held copies, recency-preferential.

    What: embed the query with the shared backend, cosine it against the dedicated
      engram fast index, and return hits whose score is the cosine PLUS a recency-scaled
      boost — so a recently-materialized held copy out-ranks an equal-cosine older copy
      and (when fed into the main recall path) an equal-cosine main node. Each hit is
      tagged `via: engram` and carries `engram_id` + `source_ptr` so the caller can
      deref the self-contained copy. Fails open (empty list) when no index exists yet.
    Why: this is the fast-tier read arm of P1 — it searches the engram copies (fast
      tier) ahead of / in addition to main (D1, exit gate). It is the SAME cosine
      primitive as vector.query on a much smaller matrix, so a fast-tier query never
      fans out over the full main corpus.

    Returns a list of {score, base_score, recency, engram_id, source_ptr, title, via}.
    """
    store = EngramStore(memory_dir)
    manifest = store._load_manifest()
    embeddings = store._load_embeddings()
    entries = manifest.get("entries", {})
    if embeddings is None or not entries:
        return []  # fast tier empty — fail open, main retrieval is unaffected

    by_row = {e["row"]: (eid, e) for eid, e in entries.items()
              if e.get("row") is not None}

    q = _vi._embed_batch([text])[0]
    sims = embeddings @ q

    scored: list[dict] = []
    for r in range(embeddings.shape[0]):
        info = by_row.get(r)
        if info is None:
            continue
        eid, entry = info
        base = float(sims[r])
        rec = _recency_factor(entry.get("materialized_at", ""), halflife_days)
        scored.append({
            "engram_id": eid,
            "source_ptr": entry.get("source_ptr"),
            "title": entry.get("title"),
            "base_score": base,
            "recency": rec,
            "score": base + recency_boost * rec,
            "via": "engram",
        })
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# P2 — the RING pointer store (volatile working memory, hours) + ring-RAG.
# ════════════════════════════════════════════════════════════════════════════

# RING_CAPACITY_DEFAULT -- What: max live pointers the ring holds before LRU drops
#   the least-recently-accessed entry.
# Why: the ring is volatile working memory (D1), bounded so it stays a hot small set
#   rather than an unbounded second copy of the corpus. Named/tunable. promote-before-
#   evict (a salient pointer materialized before it can drop) is P3 — P2 just bounds.
RING_CAPACITY_DEFAULT = 256

# RING_TTL_HOURS_DEFAULT -- What: age (hours) past which a ring pointer is considered
#   stale and skipped by resolve/ring-RAG (volatility, D1).
# Why: the ring tier is hours, not days; an old pointer is no longer "working memory".
#   The automatic sweep that prunes stale entries is later — P2 only honors the TTL on
#   read (a stale pointer simply resolves to nothing), so the field is stamped/tunable.
RING_TTL_HOURS_DEFAULT = 6.0


# ── P3 promotion-lattice constants (FEAT-2026-06-07 Tier-1 P3, D3 + D6) ──────
#
# RING_PROMOTE_HITS -- What: the count of GENUINE ring-RAG hits at which a ring
#   pointer auto-materializes to an engram copy (the frequency/recency arm of the
#   ring->engram lattice, D3/Q3a).
# Why: a pointer that keeps being genuinely recalled has EARNED a held copy (the
#   consolidation event). Default 3 mirrors HEBB_PROMOTE_REPEATS so the fast-tier
#   promotion cadence matches Tier-0's attractor cadence. Named/tunable.
RING_PROMOTE_HITS = 3

# SALIENCE_PROMOTE_THRESHOLD -- What: the salience at/above which a ring pointer
#   materializes to engram WITHOUT the RING_PROMOTE_HITS frequency bar (the D6 one-
#   shot shortcut: promotion gate = max(access_signal, salience)).
# Why: D6 effect (i) — a high-salience one-shot (a rare critical realization) must
#   earn durability without repetition. A pointer whose salience clears this bar
#   promotes immediately. HIGH named tunable so only the genuine top tier shortcuts.
SALIENCE_PROMOTE_THRESHOLD = 0.8

# STC_PROMOTE_THRESHOLD -- What: the (attenuated) stc_capture_score at/above which a weak
#   ring pointer is promotion-eligible via the OR-gate's third arm (FEAT-2026-06-11 P4).
# Why: §6.5 effect 2 — a weak node carrying a high capture score is rescued into the
#   engram store by its strong neighbour WITHOUT the frequency or its own salience bar.
#   Mirrors temporal_recall_stc.STC_PROMOTE_THRESHOLD; defined locally so promote_ring_*
#   keeps a default without a top-level cross-import. Inert when the temporal flag is off
#   (no node carries the field -> the arm is False -> the gate is byte-identical).
STC_PROMOTE_THRESHOLD = 0.50

# INJECT_PROMOTE_THRESHOLD -- What: the bar the engram->inject gate compares
#   max(attractor_strength, salience) against to mark an engram copy inject_eligible.
# Why: D6 effect (i) — the engram->inject gate is max(attractor, salience) >= this.
#   Set to HEBB_PROMOTION (0.85) so a frequency-earned attractor (w>=0.85) promotes
#   exactly as before AND a salience>=0.85 one-shot earns the standing slot without
#   the frequency bar. Named/tunable. (Inject ITSELF is P4; P3 only marks eligibility.)
INJECT_PROMOTE_THRESHOLD = 0.85

# RING_EVICT_WANT_SALIENCE -- What: the salience at/above which a to-be-evicted ring
#   pointer is "still wanted" and is materialized before being dropped (promote-before-
#   evict, D3).
# Why: a salient pointer about to fall off the LRU must not be silently lost; it earns
#   a held copy first. Aligned with SALIENCE_PROMOTE_THRESHOLD (same "wanted" bar).
RING_EVICT_WANT_SALIENCE = SALIENCE_PROMOTE_THRESHOLD


def _ring_path(memory_dir: Path) -> Path:
    return _hippocampus_dir(memory_dir) / "ring.jsonl"


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


class RingStore:
    """The Tier-1 ring store — a volatile, capacity/LRU-bounded set of POINTERS.

    What: a small append-backed JSONL of POINTER entries (not content). Each entry is
      {ptr, target_tier(main|engram), ts, last_access, access_count, salience_flag} —
      a lightweight reference into main/engram, the OPPOSITE of the engram's held copy.
      add() registers/refreshes a pointer (most-recently-accessed); when the live set
      exceeds the capacity the least-recently-accessed pointer is LRU-evicted. resolve()
      dereferences a pointer to its backing content and is dangling-SAFE: a pointer
      whose backing (main node / engram copy) is gone resolves to None.
    Why:  this is the volatile working set (D1, Q1a) — captures land here cheaply as a
      pointer + a salience flag; the held copy is EARNED later at materialization (P3,
      not here). Being pointers (deref-at-read) is what makes ring-RAG reflect the
      CURRENT backing — the pointer invariant that distinguishes it from the engram.
    """

    def __init__(self, memory_dir: Path,
                 capacity: int = RING_CAPACITY_DEFAULT,
                 ttl_hours: float = RING_TTL_HOURS_DEFAULT):
        self.memory_dir = Path(memory_dir)
        self.capacity = int(capacity)
        self.ttl_hours = float(ttl_hours)

    # -- persistence (load = compact: last-write-wins per ptr) --------------

    def _load(self) -> dict[str, dict]:
        """Return the live pointer map {ptr_name: entry}, last-write-wins.

        What: read the append-only ring.jsonl, keeping the LAST record per pointer
          (so a re-add/touch supersedes the earlier line without an in-place rewrite).
        Why:  append + load-compaction keeps add/touch O(1) writes; the map is the
          authoritative live set the rest of the API operates on.
        """
        p = _ring_path(self.memory_dir)
        if not p.exists():
            return {}
        live: dict[str, dict] = {}
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
                    name = rec.get("ptr")
                    if not name:
                        continue
                    if rec.get("__evicted__"):
                        live.pop(name, None)
                        continue
                    live[name] = rec
        except OSError:
            return {}
        return live

    def _rewrite(self, live: dict[str, dict]) -> None:
        """Compact the ring.jsonl to exactly the live entries (one line each)."""
        _hippocampus_dir(self.memory_dir).mkdir(parents=True, exist_ok=True)
        p = _ring_path(self.memory_dir)
        tmp = p.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in live.values():
                f.write(json.dumps(rec) + "\n")
        tmp.replace(p)

    # -- mutate (LRU) -------------------------------------------------------

    def add(self, ptr: str, target_tier: str = "main",
            salience_flag: bool = False) -> dict:
        """Register (or refresh) a POINTER into main/engram; LRU-bound the ring.

        What: write a pointer entry (NO content) for `ptr`, marking it the most-
          recently-accessed. Re-adding an existing pointer refreshes its last_access
          and bumps access_count (it does not duplicate). After the add, if the live
          set exceeds capacity, the least-recently-accessed pointer is LRU-evicted.
        Why:  capture (Q1a) registers a cheap pointer + a salience flag here, not a
          copy. The held copy is earned at materialization later (P3). promote-before-
          evict (materialize a salient/promotable pointer before LRU drops it) is P3 —
          P2 only enforces the capacity bound.

        Returns the pointer entry dict (also persisted).
        """
        name = _ptr_name(ptr)
        now = _now_iso()
        live = self._load()
        existing = live.get(name)
        entry = {
            "ptr": name,
            "target_tier": target_tier,
            "ts": (existing or {}).get("ts", now),
            "last_access": now,
            "lru_seq": self._next_seq(live),
            "access_count": int((existing or {}).get("access_count", 0)) + 1,
            # genuine_hits — accrued GENUINE ring-RAG hits (record_genuine_hit, P3),
            # the frequency/recency signal that drives ring->engram promotion. Preserved
            # across re-add (it is the consolidation signal, not the LRU clock).
            "genuine_hits": int((existing or {}).get("genuine_hits", 0)),
            "salience_flag": bool(salience_flag or (existing or {}).get(
                "salience_flag", False)),
        }
        live[name] = entry
        # LRU bound with PROMOTE-BEFORE-EVICT (P3): a victim still WANTED (recent
        # genuine hits OR high salience) is materialized to an engram copy before it is
        # dropped, so no wanted trace is lost. _evict_lru does the over-capacity drop.
        self._evict_lru(live)
        self._rewrite(live)
        return entry

    @staticmethod
    def _next_seq(live: dict[str, dict]) -> int:
        """Monotonic LRU sequence — a precise recency order independent of timestamp
        granularity (second-resolution last_access can tie under rapid add/touch)."""
        return 1 + max((int(r.get("lru_seq", 0)) for r in live.values()),
                       default=0)

    @staticmethod
    def _lru_key(rec: dict):
        """LRU sort key: (last_access, lru_seq) so a re-touch within the same second is
        still ordered most-recent-last via the monotonic sequence."""
        return (rec.get("last_access", ""), int(rec.get("lru_seq", 0)))

    def touch(self, ptr: str) -> dict | None:
        """Mark an existing pointer accessed (LRU bump). None if not in the ring."""
        name = _ptr_name(ptr)
        live = self._load()
        if name not in live:
            return None
        entry = live[name]
        entry["last_access"] = _now_iso()
        entry["lru_seq"] = self._next_seq(live)
        entry["access_count"] = int(entry.get("access_count", 0)) + 1
        live[name] = entry
        self._rewrite(live)
        return entry

    def record_genuine_hit(self, ptr: str) -> dict | None:
        """Record one GENUINE ring-RAG hit on a pointer (the ring->engram signal, P3).

        What: bump the pointer's `genuine_hits` counter (and refresh its LRU recency,
          since a genuine recall is also an access). Returns the updated entry, or None
          if the pointer is not in the ring.
        Why:  D3/Q3a — frequency/recency is the ring->engram promotion driver, and the
          proposal's signal is GENUINE ring-RAG hits (effortful recall), distinct from
          a bare add/touch. promote_ring_step reads `genuine_hits` against
          RING_PROMOTE_HITS. Called from the recall hook (and later a REM/idle pass);
          P3 only provides the function, not a scheduler.
        """
        name = _ptr_name(ptr)
        live = self._load()
        if name not in live:
            return None
        entry = live[name]
        entry["genuine_hits"] = int(entry.get("genuine_hits", 0)) + 1
        entry["last_access"] = _now_iso()
        entry["lru_seq"] = self._next_seq(live)
        entry["access_count"] = int(entry.get("access_count", 0)) + 1
        live[name] = entry
        self._rewrite(live)
        return entry

    # -- promote-before-evict (P3) ------------------------------------------

    def _entry_salience(self, entry: dict) -> float:
        """Salience of a ring pointer's backing node (D6 — the affective signal).

        What: read the `salience` frontmatter the P2 source (bio.compute_salience)
          writes on the backing main node; an engram-backed pointer falls back to 0.0
          (the engram copy carries no node frontmatter). The `salience_flag` set at
          capture is an OR fast-path: a flagged pointer is treated as wanted even if
          the numeric field is absent.
        Why:  D3/D6 — promote-before-evict and the ring->engram one-shot shortcut both
          gate on max(access_signal, salience); this reads that salience cheaply and
          fail-soft (a missing field -> 0.0, never a crash).
        """
        name = entry.get("ptr", "")
        tier = entry.get("target_tier", "main")
        if tier == "engram" or name.startswith("engram_"):
            return 1.0 if entry.get("salience_flag") else 0.0
        try:
            from . import bio as _bio
            fm_bundle = _bio._node_frontmatter(self.memory_dir, name)
            if fm_bundle is not None:
                sal = float(fm_bundle[0].get("salience", 0.0) or 0.0)
            else:
                sal = 0.0
        except Exception:
            sal = 0.0
        if entry.get("salience_flag"):
            sal = max(sal, 1.0)
        return sal

    def _entry_stc(self, entry: dict) -> float:
        """Attenuated stc_capture_score of a ring pointer's backing node (P4 §6.5 effect 2).

        What: read the (time-attenuated) stc_capture_score the STC capture event
          (temporal_recall_stc.capture_event) stamped on the backing main node — the
          single decaying scalar beside `salience`. An engram-backed pointer (no node
          frontmatter) and a never-captured / legacy node both fall back to 0.0.
        Why: §6.5 effect 2 — the promotion OR-gate gains a third arm so a weak node
          carrying a high capture score is rescued into the engram store by its strong
          neighbour WITHOUT meeting the frequency or its own salience bar. Reads the
          SAME attenuated scalar recall + decay read (current_capture_score), fail-soft
          (a missing field -> 0.0, never a crash). With the temporal flag off no node
          ever carries the field, so this is 0.0 everywhere → gate byte-identical.
        """
        name = entry.get("ptr", "")
        tier = entry.get("target_tier", "main")
        if tier == "engram" or name.startswith("engram_"):
            return 0.0  # engram copies carry no node frontmatter / capture score
        try:
            from . import temporal_recall_stc as _stc
            return _stc.current_capture_score(self.memory_dir, name)
        except Exception:
            return 0.0

    def _is_wanted(self, entry: dict) -> bool:
        """True when an about-to-be-evicted pointer is still WANTED (promote-before-evict).

        What: a pointer is wanted when it has recent GENUINE ring-RAG hits (>=1) OR its
          backing salience clears RING_EVICT_WANT_SALIENCE.
        Why:  D3 — a wanted trace must be materialized before the LRU can drop it (no
          dangling loss). A pointer with neither signal is an ordinary cold drop.
        """
        if int(entry.get("genuine_hits", 0)) >= 1:
            return True
        return self._entry_salience(entry) >= RING_EVICT_WANT_SALIENCE

    def _evict_lru(self, live: dict[str, dict]) -> list[str]:
        """Drop over-capacity entries, MATERIALIZING any wanted victim first (P3).

        What: when the live set exceeds capacity, select the least-recently-accessed
          victims; for each victim that is still WANTED (_is_wanted), materialize its
          backing to an engram held copy BEFORE removing the pointer; an unwanted victim
          is dropped cold. Mutates `live` in place; returns the list of engram_ids
          materialized by promote-before-evict.
        Why:  D3/Q4a — promote-before-evict is the airtight half of the lattice: an LRU
          drop never loses a salient/recently-recalled trace. Main keeps the canonical;
          materialize is loss-free (copies, never moves). A main pointer whose source is
          already gone (dangling) is simply dropped — there is nothing to promote.
        """
        promoted: list[str] = []
        if len(live) <= self.capacity:
            return promoted
        ordered = sorted(live.values(), key=self._lru_key)
        for victim in ordered[: len(live) - self.capacity]:
            vptr = victim["ptr"]
            if self._is_wanted(victim) and not str(vptr).startswith("engram_"):
                src_path = self.memory_dir / "nodes" / (
                    vptr if vptr.endswith(".md") else f"{vptr}.md")
                if src_path.exists():
                    try:
                        rec = EngramStore(self.memory_dir).materialize(vptr)
                        promoted.append(rec["engram_id"])
                    except Exception:
                        pass  # fail-soft: a copy failure must not block the LRU bound
            live.pop(vptr, None)
        return promoted

    # -- read (dangling-safe deref) -----------------------------------------

    def _is_stale(self, entry: dict) -> bool:
        """True when a pointer is older than the ring TTL (volatile working set)."""
        try:
            ts = _dt.datetime.fromisoformat(entry.get("last_access", ""))
        except (ValueError, TypeError):
            return False  # un-timestamped entries are kept rather than silently dropped
        age_h = (_dt.datetime.now() - ts).total_seconds() / 3600.0
        return age_h > self.ttl_hours

    def resolve(self, entry: dict) -> dict | None:
        """Dereference a pointer to its CURRENT backing content. Dangling-safe.

        What: read the backing node the pointer references — a main node (the live
          nodes/<ptr> file, read fresh) or an engram held copy (EngramStore.get) —
          and return {ptr, target_tier, title, content, body}. Returns None when the
          backing is GONE (the node file was deleted/frozen, or the engram id no longer
          exists) — a dangling pointer resolves to nothing. Also None past the TTL.
        Why:  the pointer invariant: because resolve reads the backing FRESH, ring-RAG
          reflects the current backing (opposite of the engram copy). The dangling
          policy is the clean P2 contract — P3 adds the automatic promote-before-evict;
          P2 just drops a pointer whose backing is gone.
        """
        if self._is_stale(entry):
            return None
        name = entry.get("ptr", "")
        tier = entry.get("target_tier", "main")
        if tier == "engram" or name.startswith("engram_"):
            copy = EngramStore(self.memory_dir).get(name)
            if copy is None:
                return None  # dangling: the engram copy is gone
            return {
                "ptr": name, "target_tier": "engram",
                "title": copy.get("title"),
                "content": copy.get("body", ""),
                "body": copy.get("body", ""),
            }
        # main pointer: read the canonical fresh (so the deref sees current content).
        fname = name if name.endswith(".md") else f"{name}.md"
        src_path = self.memory_dir / "nodes" / fname
        if not src_path.exists():
            return None  # dangling: the main node is gone
        try:
            title, content = _vi._load_node_text(src_path)
            body = src_path.read_text(encoding="utf-8")
        except OSError:
            return None
        return {"ptr": fname, "target_tier": "main", "title": title,
                "content": content, "body": body}

    def entries(self, include_stale: bool = False) -> list[dict]:
        """Live pointer entries, most-recently-accessed first.

        Stale entries (past the TTL) are excluded unless include_stale is True; they
        are NOT auto-evicted here (the volatile sweep is later) — they are simply not
        surfaced, the same dangling-safe posture resolve() takes.
        """
        rows = list(self._load().values())
        if not include_stale:
            rows = [r for r in rows if not self._is_stale(r)]
        rows.sort(key=self._lru_key, reverse=True)
        return rows


def ring_rag_query(memory_dir: Path, text: str, top_k: int = 8) -> list[dict]:
    """Ring-RAG: dereference the live ring pointers, cosine them against the query.

    What: for each live (non-stale) ring pointer, resolve() its CURRENT backing
      content (dangling pointers contribute nothing), embed the surviving backings with
      the shared backend, cosine against the query embedding, and return hits tagged
      `via: ring` carrying {ptr, target_tier, title, score, salience_flag}. Fails open
      (empty list) when the ring is empty or every pointer dangles.
    Why:  this is the volatile fast-tier read arm (the "stutter-continue" half-recall,
      D1). It dereferences at QUERY time, so a hit reflects the current backing — the
      pointer invariant (a ring entry is a reference, not a copy; if the backing changed
      since the pointer was added, the hit reflects the new backing). Reuses the SAME
      cosine primitive on the small dereferenced set; it does not reinvent embeddings.

    Returns a list of {ptr, target_tier, title, score, salience_flag, via}.
    """
    store = RingStore(memory_dir)
    entries = store.entries()
    if not entries:
        return []  # ring empty — fail open, main retrieval is unaffected

    resolved: list[tuple[dict, dict]] = []
    for e in entries:
        backing = store.resolve(e)
        if backing is None:
            continue  # dangling / stale pointer -> contributes nothing
        resolved.append((e, backing))
    if not resolved:
        return []

    q = _vi._embed_batch([text])[0]
    backing_vecs = _vi._embed_batch([b["content"] for _, b in resolved])
    sims = backing_vecs @ q

    scored: list[dict] = []
    for i, (entry, backing) in enumerate(resolved):
        scored.append({
            "ptr": backing["ptr"],
            "target_tier": backing["target_tier"],
            "title": backing.get("title"),
            "score": float(sims[i]),
            "salience_flag": bool(entry.get("salience_flag", False)),
            "via": "ring",
        })
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# P3 — the PROMOTION LATTICE + its AUTO trigger (FEAT-2026-06-07 Tier-1 P3, D3+D6).
#
# Two upward transitions, gated on max(frequency-signal, salience):
#   (a) ring -> engram   : promote_ring_step / promote_ring_pointer materialize a copy.
#   (b) engram -> inject : mark_inject_eligible flags the "known-cold" standing set.
# These are FUNCTIONS callable from the existing capture/recall hook (and later a
# REM/idle pass). P3 adds NO scheduler — the auto-trigger is just the function.
# ════════════════════════════════════════════════════════════════════════════


def _node_salience(memory_dir: Path, node: str) -> float:
    """Read the [0,1] `salience` the P2 source wrote on a node (fail-soft -> 0.0)."""
    try:
        from . import bio as _bio
        fm_bundle = _bio._node_frontmatter(memory_dir, node)
        if fm_bundle is None:
            return 0.0
        return float(fm_bundle[0].get("salience", 0.0) or 0.0)
    except Exception:
        return 0.0


def attractor_strength(memory_dir: Path, source_node: str) -> float:
    """Tier-0 attractor strength of a node = its strongest PROMOTABLE edge weight (D3).

    What: scan the Tier-0 edge_weights.json (bio._load_edge_weights) for every pair
      touching `source_node`; among the pairs that clear Tier-0's promotion gate
      (bio._is_promotable — w >= HEBB_PROMOTION AND count_genuine >= 1), return the max
      `w`. Returns 0.0 when no PROMOTABLE edge touches the node (so a node sitting on a
      sub-bar or replay-only edge contributes no attractor strength).
    Why:  D3/Q3a — the engram->inject gate reads Tier-0's signal DIRECTLY, inheriting
      its genuine-count guarantee (a replay-only edge can never confer inject-
      eligibility). This is the "known-cold by repetition" half of the max(attractor,
      salience) gate. Reuses bio's primitives; reinvents no edge logic.
    """
    fname = source_node if source_node.endswith(".md") else f"{source_node}.md"
    try:
        from . import bio as _bio
        weights = _bio._load_edge_weights(memory_dir)
    except Exception:
        return 0.0
    best = 0.0
    for key, v in weights.items():
        if fname not in key.split("::"):
            continue
        if not _bio._is_promotable(v):
            continue  # only a GENUINE attractor confers strength (replay-only -> skip)
        best = max(best, float(v.get("w", 0.0)))
    return best


def mark_inject_eligible(memory_dir: Path, engram_id: str,
                         threshold: float = INJECT_PROMOTE_THRESHOLD) -> bool:
    """engram -> inject-eligible gate: flag the copy when max(attractor, salience) clears the bar (D3/D6).

    What: for the held copy `engram_id`, compute max(attractor_strength of its source,
      salience of its source) and set the copy's `inject_eligible` flag True when it
      meets `threshold` (default INJECT_PROMOTE_THRESHOLD). Persists the flag on the
      held-copy JSON. Returns the resulting inject_eligible bool (False if the copy is
      missing). It only ever SETS eligibility — it does not inject (P4).
    Why:  D6 effect (i) — the gate is max(attractor_strength, salience), so a frequency-
      earned attractor (w>=bar) AND a high-salience one-shot (salience>=bar, no
      repetition) both earn the standing inject slot, while a low-frequency low-salience
      copy does NOT. This is the eligibility computation P4's injector will consume.
    """
    store = EngramStore(memory_dir)
    copy = store.get(engram_id)
    if copy is None:
        return False
    source = copy.get("source_ptr", "")
    attr = attractor_strength(memory_dir, source)
    sal = _node_salience(memory_dir, source)
    eligible = max(attr, sal) >= float(threshold)
    if bool(copy.get("inject_eligible")) != eligible:
        copy["inject_eligible"] = eligible
        (_engram_dir(memory_dir) / f"{engram_id}.json").write_text(
            json.dumps(copy, indent=2), encoding="utf-8")
    return eligible


def promote_ring_pointer(memory_dir: Path, ptr: str,
                         hit_threshold: int = RING_PROMOTE_HITS,
                         salience_threshold: float = SALIENCE_PROMOTE_THRESHOLD,
                         stc_threshold: float = STC_PROMOTE_THRESHOLD
                         ) -> dict | None:
    """ring -> engram: materialize ONE wanted pointer when the promote gate fires (D3/D6).

    What: read the ring pointer `ptr`; if max(access_signal, salience, stc) crosses the
      bar — i.e. its GENUINE ring-RAG hits reach `hit_threshold`, OR its backing salience
      reaches `salience_threshold` (the D6 one-shot shortcut), OR its backing (attenuated)
      stc_capture_score reaches `stc_threshold` (the P4/§6.5 capture-rescue arm) —
      materialize its backing to a kWTA-coded engram held copy (the consolidation event)
      and immediately run the engram->inject gate (mark_inject_eligible). Returns
      {engram_id, inject_eligible, reason} on promotion (reason "stc" when capture-earned,
      auditable per §6.5), or None when the gate does not fire / the pointer is
      missing/engram-backed/dangling.
    Why:  D3/D6 — this is the ring->engram arm of the lattice with the salience shortcut:
      a frequently genuinely-recalled pointer earns a copy by frequency, AND a high-
      salience one-shot earns it WITHOUT repeated access. materialize() applies kWTA;
      main keeps the canonical (loss-free).
    """
    name = _ptr_name(ptr)
    if name.startswith("engram_"):
        return None  # already an engram copy; nothing to promote
    ring = RingStore(memory_dir)
    live = ring._load()
    entry = live.get(name)
    if entry is None:
        return None
    hits = int(entry.get("genuine_hits", 0))
    sal = ring._entry_salience(entry)
    freq_ready = hits >= int(hit_threshold)
    sal_ready = sal >= float(salience_threshold)
    # FEAT-2026-06-11 temporal-recall P4 (§6.5 effect 2): the STC capture OR-gate arm —
    # a weak node carrying a high (attenuated) stc_capture_score is rescued into the
    # engram store by its strong neighbour WITHOUT meeting the frequency or salience bar.
    # The gate widens from max(freq, salience) to max(freq, salience, stc). The capture
    # score is 0.0 everywhere when the temporal flag is off (no node carries it), so
    # stc_ready is False and this is byte-identical to today.
    stc = ring._entry_stc(entry)
    stc_ready = stc >= float(stc_threshold)
    if not (freq_ready or sal_ready or stc_ready):
        return None  # promote gate = max(access_signal, salience, stc) not met
    src_path = memory_dir / "nodes" / (name if name.endswith(".md")
                                       else f"{name}.md")
    if not src_path.exists():
        return None  # dangling — nothing to copy
    try:
        rec = EngramStore(memory_dir).materialize(name)
    except Exception:
        return None
    eligible = mark_inject_eligible(memory_dir, rec["engram_id"])
    # reason: capture-earned promotions are auditable as "stc" (§6.5 — preserves the
    # provenance that the weak node was RESCUED, not intrinsically frequent/salient).
    if stc_ready and not (freq_ready or sal_ready):
        reason = "stc"
    elif sal_ready and not freq_ready:
        reason = "salience"
    elif freq_ready and not sal_ready:
        reason = "frequency"
    else:
        reason = "both"
    return {"engram_id": rec["engram_id"], "inject_eligible": eligible,
            "reason": reason, "genuine_hits": hits, "salience": sal, "stc": stc}


# ════════════════════════════════════════════════════════════════════════════
# P4 — the TWO inject layers under a fixed token budget (FEAT-2026-06-07 P4, D4/Q5a).
#
# assemble_inject_block pushes a SMALL standing context into the model's finite window
# (the context-window-extension step). It is NOT a search (RAG is the search path):
#   ENGRAM-INJECT — the inject_eligible held copies (P3 flag): the small ALWAYS-ON
#       identity / known-cold set ("a name you hold"). Reserved budget slice; favored
#       under pressure.
#   RING-INJECT  — the live ring pointers dereferenced to backing content: the current
#       auto-pilot working set ("a habit"). Fills the remaining budget by relevance.
# Both layers are RELEVANCE-GATED against the turn query via cosine — but this is a SORT
# for inclusion, NOT a genuine retrieval, so it is CO-ACTIVATION-SILENT (D5/Q6a): no path
# here calls hebbian_record / feeds the Tier-0 web. Standing deref is O(1), not a search.
#
# P4 builds the ASSEMBLER + an INERT operator-gated surface (mcp_server.memory_inject_block
# / memory_search include_inject). The per-turn injection into the live prompt stays
# operator-gated — P4 does NOT mutate the live prompt and does NOT change decay (P5).
# ════════════════════════════════════════════════════════════════════════════

# INJECT_BUDGET_DEFAULT -- What: the FIXED total token budget the assembled inject block
#   must never exceed, split across the two layers (D4/Q5a).
# Why: inject is a context-window-EXTENSION into a finite window; a fixed cap keeps it
#   small and predictable (Risk 3: a budget overrun degrades the prompt). Default 600 per
#   the proposal D4. Named/tunable; the block is test-asserted to never exceed it.
INJECT_BUDGET_DEFAULT = 600

# INJECT_ENGRAM_BUDGET_FRAC -- What: the fraction of INJECT_BUDGET reserved for the
#   always-on engram-inject identity set; ring-inject fills the remainder.
# Why: D4/Q5a — engram-inject is the durable, earned identity layer and is FAVORED under
#   budget pressure; ring-inject is turn-relevant working set that fills what is left. A
#   small reserved slice (default 0.4) keeps the identity set standing even when the ring
#   is busy, while leaving the majority for turn-relevant working memory. Named/tunable.
INJECT_ENGRAM_BUDGET_FRAC = 0.4


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for the inject budget (chars/4 heuristic).

    What: approximate the token count of `text` as ceil(len/4) — the standard cheap
      heuristic when no model tokenizer is on the hot path. Empty/None -> 0.
    Why: D4 — the inject budget is a coarse cap, not an exact accounting; a len-based
      heuristic keeps assemble_inject_block dependency-free (no tokenizer load on the
      per-turn path) while bounding the block. No general token estimator exists in the
      tree (only judge_eval's LLM-side truncator), so this is the local primitive.
    """
    if not text:
        return 0
    return -(-len(text) // 4)  # ceil division: every started 4-char chunk costs a token


def _relevance_score(query_vec, content: str) -> float:
    """Co-activation-SILENT cosine of `content` vs the query embedding (relevance sort).

    What: embed `content` with the shared backend and cosine it against the already-
      embedded query vector; both vectors are L2-normalized by the backend, so the dot
      product is the cosine. Returns 0.0 on any embed failure (fail-soft).
    Why: D4/D5 — inject relevance-gates each layer for INCLUSION ordering, NOT retrieval.
      This is the same cosine PRIMITIVE the RAG arms use, but it deliberately does NOT
      route through ring_rag_query / engram_rag_query / memory_search and NEVER calls
      hebbian_record — assembling an inject block records ZERO genuine co-activations
      (the homeostasis guard: a standing deref is not "recalled together").
    """
    if not content:
        return 0.0
    try:
        vec = _vi._embed_batch([content])[0]
        return float(np.dot(vec, query_vec))
    except Exception:
        return 0.0


def assemble_inject_block(memory_dir: Path, query_or_context: str,
                          token_budget: int = INJECT_BUDGET_DEFAULT,
                          engram_budget_frac: float = INJECT_ENGRAM_BUDGET_FRAC
                          ) -> dict:
    """Assemble the two-layer standing-availability inject block under a fixed budget (P4/D4).

    What: build a SMALL standing context block to prepend to a prompt — the context-
      window-extension step — from two layers, relevance-gated against `query_or_context`
      and arbitrated by priority within `token_budget`:
        ENGRAM-INJECT — the inject_eligible held copies (P3 flag): the always-on
            identity/known-cold set. Reserved an engram_budget_frac slice; FAVORED under
            pressure (filled FIRST, in relevance order, until its reserved slice — OR the
            whole budget if the ring is empty — is exhausted).
        RING-INJECT  — the live ring pointers dereferenced to backing content: the auto-
            pilot working set. Fills the REMAINING budget by relevance.
      When the two layers together exceed `token_budget`, ARBITRATE by priority: engram-
      inject is selected first; ring-inject then fills only what is left; overflow in
      either layer is DROPPED. The returned block NEVER exceeds the budget.
    Why: D4/Q5a — this is the SETTLED two-inject-layer design: a small always-on identity
      set + a turn-relevant working set, priority-arbitrated under a fixed cap. It is
      CO-ACTIVATION-SILENT (D5/Q6a, _relevance_score / direct deref, NEVER hebbian_record)
      — assembling/serving a block manufactures ZERO Tier-0 edges (the homeostasis guard).
      It is a standing pointer-deref (O(1) reads of the engram flag + the ring pointers),
      NOT a search (RAG is the search path). P4 builds the assembler; the actual per-turn
      injection into the live prompt stays operator-gated/INERT.

    Returns:
        {
          "items": [ {layer, source, title, content, tokens, score}, ... ],  # selected,
                                                                             # in serve order
          "tokens_used": int, "token_budget": int,
          "engram_budget": int, "ring_budget": int,
          "engram_count": int, "ring_count": int,
          "dropped": int,                # candidates excluded for budget/relevance
          "co_activation_silent": True,  # contract marker (D5)
        }
    An empty ring + empty engram yields an empty block (fail-open).
    """
    budget = max(0, int(token_budget))
    frac = min(1.0, max(0.0, float(engram_budget_frac)))
    engram_budget = int(budget * frac)

    # Embed the turn query ONCE for the relevance sort (co-activation-silent: a bare
    # embed, never a retrieval — no hebbian_record, no Tier-0 feed).
    try:
        query_vec = _vi._embed_batch([query_or_context])[0]
    except Exception:
        query_vec = None

    # ---- ENGRAM-INJECT candidates: the inject_eligible held copies (always-on) ----
    engram_cands: list[dict] = []
    for copy in EngramStore(memory_dir).all():
        if not copy.get("inject_eligible"):
            continue
        content = copy.get("body", "") or ""
        score = _relevance_score(query_vec, content) if query_vec is not None else 0.0
        engram_cands.append({
            "layer": "engram",
            "source": copy.get("source_ptr") or copy.get("engram_id"),
            "title": copy.get("title"),
            "content": content,
            "tokens": estimate_tokens(content),
            "score": score,
        })
    engram_cands.sort(key=lambda c: c["score"], reverse=True)

    # ---- RING-INJECT candidates: live ring pointers, dereferenced to backing ----
    ring_cands: list[dict] = []
    ring = RingStore(memory_dir)
    for entry in ring.entries():
        backing = ring.resolve(entry)  # dangling/stale -> None, contributes nothing
        if backing is None:
            continue
        content = backing.get("content", "") or backing.get("body", "") or ""
        score = _relevance_score(query_vec, content) if query_vec is not None else 0.0
        ring_cands.append({
            "layer": "ring",
            "source": backing.get("ptr"),
            "title": backing.get("title"),
            "content": content,
            "tokens": estimate_tokens(content),
            "score": score,
        })
    ring_cands.sort(key=lambda c: c["score"], reverse=True)

    # ---- Priority arbitration under the fixed budget ----
    # Engram-inject (identity, FAVORED) is selected first, up to its reserved slice
    # (or the whole budget if no ring candidate competes). Ring-inject then fills only
    # the REMAINING budget by relevance. Overflow in either layer is dropped.
    selected: list[dict] = []
    used = 0
    dropped = 0

    # Engram cap: its reserved slice, but allow it the whole budget when the ring is
    # empty (a small identity set should not be starved by an unused ring reservation).
    engram_cap = budget if not ring_cands else engram_budget
    for cand in engram_cands:
        if used + cand["tokens"] <= engram_cap:
            selected.append(cand)
            used += cand["tokens"]
        else:
            dropped += 1

    # Ring fills the remaining budget (total budget minus what engram actually used).
    for cand in ring_cands:
        if used + cand["tokens"] <= budget:
            selected.append(cand)
            used += cand["tokens"]
        else:
            dropped += 1

    return {
        "items": selected,
        "tokens_used": used,
        "token_budget": budget,
        "engram_budget": engram_budget,
        "ring_budget": budget - engram_budget,
        "engram_count": sum(1 for c in selected if c["layer"] == "engram"),
        "ring_count": sum(1 for c in selected if c["layer"] == "ring"),
        "dropped": dropped,
        # Contract marker: assembling this block recorded NO genuine co-activation and
        # fed NOTHING to the Tier-0 Hebbian web (D5/Q6a homeostasis guard).
        "co_activation_silent": True,
    }


def promote_ring_step(memory_dir: Path,
                      hit_threshold: int = RING_PROMOTE_HITS,
                      salience_threshold: float = SALIENCE_PROMOTE_THRESHOLD,
                      stc_threshold: float = STC_PROMOTE_THRESHOLD
                      ) -> dict:
    """One promotion PASS over the ring (the AUTO trigger — a function, not a scheduler).

    What: sweep every live ring pointer; promote_ring_pointer each (ring->engram on
      max(freq, salience)), then re-run mark_inject_eligible over EVERY held copy so the
      engram->inject set tracks the current Tier-0 attractor / salience state. Returns
      {promoted: [...], inject_eligible: int, scanned: int}. Pure side-effects on the
      hippocampus stores; mutates no main node and writes no Tier-0 edge.
    Why:  D3 — this is the lattice's auto-trigger, invokable from the capture/recall hook
      and later a REM/idle pass. P3 provides ONLY this function + its gate logic — no new
      timer/scheduler (CONSTRAINT). It computes ELIGIBILITY; it never injects (P4) and
      never touches decay (P5).
    """
    ring = RingStore(memory_dir)
    promoted: list[dict] = []
    scanned = 0
    for entry in ring.entries():
        scanned += 1
        res = promote_ring_pointer(memory_dir, entry["ptr"],
                                   hit_threshold, salience_threshold,
                                   stc_threshold)
        if res is not None:
            promoted.append(res)
    # Refresh inject-eligibility across ALL held copies (attractor/salience may have
    # changed since the copy was materialized) — eligibility computation only (P4 injects).
    eligible = 0
    for copy in EngramStore(memory_dir).all():
        if mark_inject_eligible(memory_dir, copy["engram_id"]):
            eligible += 1
    return {"promoted": promoted, "inject_eligible": eligible, "scanned": scanned}


# ── module metadata ────────────────────────────────────────────────────────
# file:        samia/core/hippocampus.py
# role:        Tier-1 hippocampal fast store — P1 (engram held-copy store + engram-RAG)
#              + P2 (ring POINTER store + ring-RAG) + P3 (kWTA + promotion lattice +
#              promote-before-evict) + P4 (the two inject layers + token budget).
# phase:       FEAT-2026-06-07-memory-tier1-hippocampal-quad-v01 P1 + P2 + P3 + P4.
#              P3 adds: kWTA sparse code on materialize (bio.kwta_sparse_code, D2);
#              the promotion lattice + AUTO trigger (promote_ring_step / _pointer:
#              ring->engram on max(genuine-hits, salience); mark_inject_eligible /
#              attractor_strength: engram->inject on max(attractor, salience), D3/D6);
#              promote-before-evict (RingStore._evict_lru materializes a wanted victim).
#              P4 adds: assemble_inject_block — the two inject layers (engram-inject =
#              the inject_eligible always-on identity set; ring-inject = the live ring
#              working set) under a FIXED token budget (estimate_tokens), relevance-gated
#              (_relevance_score) + priority-arbitrated (engram-inject favored, overflow
#              dropped, block never exceeds the cap), and CO-ACTIVATION-SILENT (D5/Q6a —
#              no hebbian_record, no Tier-0 feed). The per-turn live-prompt injection
#              stays operator-gated/INERT (P4 builds the assembler + an inert MCP surface).
#              NOT here: feed-forward/genuine-once + salience-dampened DECAY (P5 — P4
#              does NOT change decay). The salience SOURCE itself is P2 (bio).
# layer:       core (pure library; no daemon dependency, no global edges.db write).
# owns:        <memory_dir>/hippocampus/engram/<id>.json  (self-contained held copies,
#              now carrying kwta_code + inject_eligible)
#              <memory_dir>/hippocampus/engram_index/      (dedicated cosine fast index)
#              <memory_dir>/hippocampus/ring.jsonl         (volatile pointer LRU,
#              now carrying genuine_hits)
# reuses:      vector._embed_batch (embedding backend), vector._load_node_text
#              (canonical reader), vector's embeddings.npy + manifest.json layout;
#              bio.kwta_sparse_code (P3 sparse code), bio._load_edge_weights +
#              bio._is_promotable (Tier-0 attractor signal), bio._node_frontmatter
#              (the P2 salience field) — reads Tier-0, writes none of it.
# consumers:   mcp_server.memory_search (engram-RAG + ring-RAG arms, additive); the
#              recall/capture hook drives promote_ring_step (the AUTO trigger, no
#              scheduler); mcp_server.memory_inject_block + memory_search(include_inject)
#              serve assemble_inject_block (INERT operator-gated surface); P5 builds on
#              the engram copies + the inject_eligible set. assemble_inject_block feeds
#              NOTHING to the Tier-0 web (co-activation-silent).
# invariants:  an engram entry is a COPY (survives source churn); a RING entry is a
#              POINTER (deref-at-read -> reflects current backing; dangling -> dropped).
#              Neither moves/mutates the main canonical (loss-free); promote-before-evict
#              copies a wanted pointer before the LRU can drop it.
# fix:         FEAT-2026-06-11 temporal-recall P4 (§6.5 effect 2) — STC promotion OR-gate
#              arm: promote_ring_pointer's gate widened from max(freq, salience) to
#              max(freq, salience, stc); RingStore._entry_stc reads the backing node's
#              (attenuated) stc_capture_score via temporal_recall_stc.current_capture_score;
#              a capture-earned promotion is reason="stc" (auditable). Inert under the
#              temporal flag off (no node carries the field -> stc_ready False -> gate
#              byte-identical). promote_ring_step forwards the stc_threshold.
# restart:     additive; activation is operator-gated (daemon/MCP restart) later.
# ─────────────────────────────────────────────────────────────────────────────
