"""samia.core.hippocampus.engram — the P1 engram held-copy store + engram-RAG.

Layer 1 (Owns / Depends):
    Owns:    EngramStore (the persistent, self-contained HELD-COPY store under
             <memory_dir>/hippocampus/engram/<id>.json + its dedicated cosine fast
             index) and engram_rag_query (recency-preferential cosine over the held
             copies). materialize() is the copy primitive (reads the canonical via the
             source pointer, embeds it, stamps a kWTA sparse code + the temporal-recall
             substrate, and persists a self-contained copy + its index row).
    Depends: .config (the directory layout, _engram_id, _now_iso, the recency bars,
             the re-exported vector backend _vi).  Lazily: samia.core.bio
             (kwta_sparse_code), samia.core.frontmatter (the source's written_at/
             episode_seq), samia.core.temporal_recall_sith (the SITH encode event) —
             all function-local to avoid a core import cycle and stay inert when off.

Layer 2 (What / Why):
    What: the durable fast-episodic tier. Each engram entry is a SELF-CONTAINED copy
          (frontmatter title + body + its own embedding row), so it survives churn
          (freeze/merge/delete) of its main source — it is a COPY, not a pointer
          (pointers are the RING). engram_rag_query searches those copies with the
          SAME cosine primitive vector.query uses, on a much smaller matrix, with a
          recency-scaled additive boost so a fresh copy out-ranks an equal-cosine
          older copy (the P1 exit gate).
    Why:  carved out of the 1339-line monolith as the engram responsibility. The ring
          store, the promotion lattice, and the inject assembler all build OVER this
          store (RingStore._evict_lru / promote_ring_pointer materialize through
          EngramStore; assemble_inject_block reads EngramStore.all()), so it sits just
          above config in the package DAG and below ring/promotion/inject.
"""

from __future__ import annotations

import json

import numpy as np

from .config import (
    ENGRAM_TTL_DAYS_DEFAULT,
    RECENCY_BOOST_DEFAULT,
    RECENCY_HALFLIFE_DAYS,
    Path,
    _engram_dir,
    _engram_embed_path,
    _engram_id,
    _engram_index_dir,
    _engram_manifest_path,
    _now_iso,
    _recency_factor,
    _vi,
)


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
        from .. import bio as _bio
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
            from .. import frontmatter as _fm_mod
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
            from .. import temporal_recall_sith as _sith
            _sith.integrator_observe(self.memory_dir, vec, now=written_at)
            _sith.capture_snapshot(self.memory_dir, eid)
        except Exception:
            pass
        return record


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


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.hippocampus.engram
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.hippocampus monolith during
#             modularization (the P1 engram responsibility; bodies byte-identical, the
#             shared constants/paths/ids lifted into .config and relative-imported).
# Layer:      core (pure library, no daemon dependency)
# Role:       P1 — the engram held-copy store (EngramStore: self-contained held
#             copies + the dedicated cosine fast index + the materialize() copy
#             primitive) and engram_rag_query (recency-preferential cosine over the
#             held copies). The durable fast-episodic tier the ring/promotion/inject
#             submodules build over.
# Stability:  stable — bodies byte-identical to the monolith; the carve only moved the
#             shared constants/paths/ids into .config and relative-imports them here.
# ErrorModel: fail-open reads (a missing index/copy -> [] / None); materialize raises
#             FileNotFoundError on a missing source (a deliberate copy of a real node)
#             but the SITH encode event + the temporal-substrate lift are fail-soft.
# Depends:    json, numpy. .config (paths/ids/recency bars/_vi). Lazily: samia.core.bio
#             (kwta_sparse_code), samia.core.frontmatter (parse), samia.core.
#             temporal_recall_sith (integrator_observe/capture_snapshot).
# Exposes:    EngramStore, engram_rag_query.
# Lines:      354
# ─────────────────────────────────────────────
