"""samia.core.vector — semantic vector index for SAM/IA memory nodes.

Layer 1 (Owns / Depends):
    Owns:    build(memory_dir, rebuild=False) -> manifest dict — embed nodes/ into
                 embeddings.npy + manifest.json (sha-keyed incremental cache).
             query(memory_dir, text, top_k=8) -> list[{score, node, title}] — cosine
                 recall, tombstone-aware, cross-embedder guarded.
             info(memory_dir) -> None — print index provenance.
             active_model_id, active_model_dim — the live embedder-selection contract.
             tombstone_node, untombstone_node — forget/restore a node's recall row.
             EmbedModelMismatch — the cross-embedder guard's dedicated exception.
    Depends: numpy; transformers + torch (mean-pool embedder, loaded lazily);
             samia.core.netconsent (cache-miss download gate, lazy import).
Layer 2 (What / Why):
    What: each node embeds to an L2-normalized [hidden_size] row; the matrix lives at
          <memory_dir>/vector_index/embeddings.npy and the manifest records
          {model_id, embed_model, dim, built_at, node_count, entries}. build() reuses a
          cached row when a node's content sha is unchanged; query() embeds the text,
          dots it against the matrix, and returns the top_k non-tombstoned rows.
    Why:  the library plane carries all logic so the daemon (design doc §3.4) calls
          build()/query() directly on a schedule. The active embedder is ASTHENOS_EMBED_
          MODEL (unset => the historical MiniLM-L6-v2 default, so every existing index is
          byte-stable); the dim is read from the model's hidden_size, NOT hard-coded, so
          384/768/1024-d families share one mean-pool path. query() REFUSES a cross-
          embedder query (a cosine across two embedding spaces is meaningless and silently
          wrong — we fail LOUD via EmbedModelMismatch). A cache-miss model fetch is gated
          through netconsent (ASTHENOS_MODEL_AUTOFETCH): off refuses, on is standing
          consent, unset asks at a tty — no silent download ever.

Layer 3 (Changelog):
    Carved from memory_vector_index.py — byte-identical CLI output on the same tree.
    SLOT-SCALING 2026-06-12 (v1.1): ASTHENOS_EMBED_MODEL embedder-selection seam +
        manifest model provenance + cross-embedder query guard.
    SEC 2026-06-12: cache-miss model download gated through netconsent.consent.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Historical default — preserved when ASTHENOS_EMBED_MODEL is unset so every existing
# index / caller stays byte-stable. EMBED_DIM is kept as the default-model dim for
# back-compat with importers that reference vector.EMBED_DIM directly (e.g. the SITH
# bank's _embed_dim mirror); the AUTHORITATIVE dim for a built index is its manifest's
# `dim`, which active_model_dim() resolves dynamically from the loaded model.
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_ENV = "ASTHENOS_EMBED_MODEL"
EMBED_DIM = 384
MAX_TOKENS = 256


def active_model_id() -> str:
    """The HF model id of the embedder selected for THIS process (live env read).

    What: returns ASTHENOS_EMBED_MODEL if set and non-empty, else the MiniLM default.
    Why:  one place owns the env contract so build(), query() and the load path all agree
          on which model is active; a live read (not import-time capture) lets a caller
          set the env before the first embed without re-importing the module.
    """
    val = os.environ.get(EMBED_MODEL_ENV, "").strip()
    return val or DEFAULT_MODEL_ID


# Back-compat shim: some callers import the module-level constant MODEL_ID. Keep it
# pointing at the active model so legacy reads stay correct under selection.
MODEL_ID = active_model_id()


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _index_dir(memory_dir: Path) -> Path:
    return memory_dir / "vector_index"


def _embed_path(memory_dir: Path) -> Path:
    return _index_dir(memory_dir) / "embeddings.npy"


def _manifest_path(memory_dir: Path) -> Path:
    return _index_dir(memory_dir) / "manifest.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_node_text(path: Path) -> tuple[str, str]:
    """Return (title, content_for_embedding)."""
    raw = path.read_text(encoding="utf-8")
    title = path.stem
    desc = ""
    body = raw
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            fm = raw[3:end]
            body = raw[end + 4:].lstrip()
            for line in fm.splitlines():
                if line.startswith("name:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
    head = f"{title}. {desc}".strip(". ").strip()
    if head:
        return title, f"{head}\n\n{body}"
    return title, body


def _load_manifest(memory_dir: Path) -> dict:
    p = _manifest_path(memory_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_manifest(memory_dir: Path, m: dict) -> None:
    _index_dir(memory_dir).mkdir(parents=True, exist_ok=True)
    _manifest_path(memory_dir).write_text(
        json.dumps(m, indent=2), encoding="utf-8")


def tombstone_node(memory_dir: Path, node: str) -> dict:
    """FEAT-2026-06-07 P0: mark a node's vector entry tombstoned so query() excludes it
    immediately, without an expensive per-node embedding delete. The row is hard-dropped on
    the next build(rebuild=True) (the node file is already gone, so it won't be re-added).
    Part of the forget_node cascade. Idempotent. Returns {tombstoned: bool, node}.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    m = _load_manifest(memory_dir)
    entries = m.get("entries", {})
    if fname in entries:
        entries[fname]["tombstoned"] = True
        _save_manifest(memory_dir, m)
        return {"tombstoned": True, "node": fname}
    return {"tombstoned": False, "node": fname, "skipped": "not-in-index"}


def untombstone_node(memory_dir: Path, node: str) -> dict:
    """FEAT-2026-06-07 P3a: inverse of tombstone_node — clear the tombstone so query()
    includes the node again. Used by ia.restore_node to un-forget a contradiction-superseded
    node. The next build(rebuild=True) re-embeds the restored file. Idempotent. Returns
    {untombstoned: bool, node}.
    """
    # What: drop the "tombstoned" flag if it is set on the node's entry.
    # Why: query() (vector.py) skips any entry with tombstoned truthy; clearing it
    #   re-admits the restored node to recall without forcing an immediate re-embed.
    fname = node if node.endswith(".md") else f"{node}.md"
    m = _load_manifest(memory_dir)
    entries = m.get("entries", {})
    entry = entries.get(fname)
    if entry is not None and entry.get("tombstoned"):
        entry.pop("tombstoned", None)
        _save_manifest(memory_dir, m)
        return {"untombstoned": True, "node": fname}
    return {"untombstoned": False, "node": fname,
            "skipped": "not-tombstoned" if entry is not None else "not-in-index"}


# ---------------------------------------------------------------------------
# Embedding backend (transformers + torch, mean-pooled, L2-normalized)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_loaded_model_id: str | None = None  # which id _model/_tokenizer currently hold


def _load_local_only(model_id: str):
    """Load `model_id`'s tokenizer+model from the HF cache, no network. Or None.

    What: tries transformers' local_files_only path for the GIVEN model id. On a
          cache HIT returns (tokenizer, model); on a cache MISS (the model was never
          downloaded) transformers raises, and we return None so the caller can gate a
          fetch through the consent protocol.
    Why:  the common path -- a box that has run this embedder before -- must stay
          SILENT and fast: no prompt, no network probe, just load from cache.
          local_files_only=True is exactly "use the cache or fail", which is the
          cache probe we want; we never let it reach out for the consent path.
    """
    from transformers import AutoModel, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        mdl = AutoModel.from_pretrained(model_id, local_files_only=True)
    except Exception:
        # OSError / EnvironmentError on a cache miss (model not downloaded yet).
        return None
    return tok, mdl


def _ensure_model():
    global _model, _tokenizer, _loaded_model_id
    model_id = active_model_id()
    # Reload only when the ACTIVE model changed (or nothing is loaded). Selecting a new
    # embedder mid-process (a sweep, or a future model menu) re-binds the singleton.
    if _model is not None and _loaded_model_id == model_id:
        return
    import torch  # noqa: F401  (used implicitly)

    # 1) Common path: load from the HF cache with NO network and NO prompt.
    local = _load_local_only(model_id)
    if local is not None:
        _tokenizer, _model = local
        _model.eval()
        _loaded_model_id = model_id
        return

    # 2) Cache miss: a download is required. Gate it through the shared consent
    #    protocol (same one the gguf fetcher uses) -- no silent download.
    from samia.core import netconsent

    approved = netconsent.consent(
        what=f"sentence embedder {model_id}",
        size_hint="~90MB-1.3GB (model-dependent)",
        license_str="Apache-2.0 / MIT (model-dependent)",
        source="huggingface.co",
        manual_hint=(
            f"pre-download {model_id} into the HuggingFace cache "
            f"(e.g. `huggingface-cli download {model_id}`), "
            f"or set {netconsent.AUTOFETCH_ENV}=1 for standing consent."
        ),
    )
    if not approved:
        raise RuntimeError(
            f"semantic vector index needs the {model_id} embedder but it is not "
            f"cached and the download was refused. Set "
            f"{netconsent.AUTOFETCH_ENV}=1 to allow the fetch, or pre-download "
            f"{model_id} into the HuggingFace cache."
        )

    # 3) Approved: allow the network fetch (default from_pretrained behavior).
    from transformers import AutoModel, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModel.from_pretrained(model_id)
    _model.eval()
    _loaded_model_id = model_id


def active_model_dim() -> int:
    """The output dim of the ACTIVE embedder, resolved from the loaded model.

    What: loads the active model (cache-only/consent-gated, same path as embedding)
          and reads its hidden_size — the dim of the mean-pooled sentence vector.
    Why:  candidates span 384 (MiniLM, bge-small), 768 (mpnet) and 1024 (bge-large);
          the index/manifest must record the TRUE dim, not the MiniLM constant, so the
          stored matrix shape and the build-vs-query model guard are both honest.
    """
    _ensure_model()
    return int(_model.config.hidden_size)


# _embed_batch — What: tokenize, run the model, mean-pool over real tokens, L2-normalize,
#     and return a float32 [len(texts), hidden_size] matrix.
def _embed_batch(texts: list[str]) -> np.ndarray:
    import torch
    _ensure_model()
    tok = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = _model(**tok)
    last = out.last_hidden_state
    mask = tok["attention_mask"].unsqueeze(-1).float()
    summed = (last * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype(np.float32)
# _embed_batch — Why: the attention mask weights out padding before the mean (so a short
#     text is not diluted by pad tokens), and L2-normalization makes the later dot product
#     a cosine; counts is clamped to 1e-9 to avoid a divide-by-zero on an all-pad row.


# ---------------------------------------------------------------------------
# Build / query
# ---------------------------------------------------------------------------


# build — What: (re)embed nodes/ into embeddings.npy + manifest.json, reusing a cached
#     row whenever a node's content sha is unchanged (rebuild=True drops the whole cache).
def build(memory_dir: Path, rebuild: bool = False) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    index_dir = _index_dir(memory_dir)
    embed_path = _embed_path(memory_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {} if rebuild else _load_manifest(memory_dir)
    cached_entries = manifest.get("entries", {}) if not rebuild else {}

    nodes = sorted(nodes_dir.glob("*.md"))
    if not nodes:
        print(f"[vector_index] no nodes found at {nodes_dir}", file=sys.stderr)
        return {}

    new_entries: dict[str, dict] = {}
    rows: list = []
    paths: list[str] = []
    titles: list[str] = []

    cached_emb = None
    if not rebuild and embed_path.exists():
        try:
            cached_emb = np.load(embed_path)
        except Exception:
            cached_emb = None

    to_embed_idx: list[int] = []
    to_embed_text: list[str] = []
    to_embed_titles: list[str] = []

    # TOCTOU tolerance: nodes can vanish between listing and reading (offload
    # compaction racing a REM vector_maintenance pass — found live 2026-06-12).
    # Read first and skip the missing so row indices stay dense; a vanished
    # node simply isn't indexed this build and the next pass picks up reality.
    loaded: list[tuple[Path, str, str]] = []
    for p in nodes:
        try:
            loaded.append((p, *_load_node_text(p)))
        except FileNotFoundError:
            continue

    for i, (p, title, content) in enumerate(loaded):
        rel = p.name
        sig = _sha256(content)
        cached = cached_entries.get(rel)
        paths.append(rel)
        titles.append(title)
        new_entries[rel] = {"sha256": sig, "title": title, "row": i}

        if (cached and cached.get("sha256") == sig
                and cached_emb is not None
                and cached.get("row") is not None):
            old_row = cached["row"]
            if old_row < cached_emb.shape[0]:
                rows.append(cached_emb[old_row].copy())
                continue
        rows.append(None)
        to_embed_idx.append(i)
        to_embed_text.append(content)
        to_embed_titles.append(title)

    n = len(loaded)
    if to_embed_idx:
        print(f"[vector_index] embedding {len(to_embed_idx)}/{n} nodes "
              f"(cached: {n - len(to_embed_idx)})")
        BATCH = 16
        new_vecs: list[np.ndarray] = []
        for s in range(0, len(to_embed_text), BATCH):
            chunk = to_embed_text[s:s + BATCH]
            t0 = time.time()
            v = _embed_batch(chunk)
            dt = time.time() - t0
            print(f"  batch {s // BATCH + 1}: {len(chunk)} nodes in {dt:.2f}s")
            new_vecs.append(v)
        new_arr = np.vstack(new_vecs)
        for k, idx in enumerate(to_embed_idx):
            rows[idx] = new_arr[k]
    else:
        print(f"[vector_index] all {n} nodes cached, no embedding work")

    embeddings = np.vstack(rows).astype(np.float32)
    np.save(embed_path, embeddings)

    # Record the ACTIVE model + its true dim. dim is taken from the embedded matrix
    # (ground truth: width of what we just wrote) so a manifest can never disagree with
    # its own embeddings.npy. model_id is the env-selected embedder; query() refuses to
    # serve this index under a different active model (cross-embedder cosine guard).
    model_id = active_model_id()
    built_dim = int(embeddings.shape[1]) if embeddings.size else active_model_dim()
    manifest_out = {
        "model_id": model_id,
        "embed_model": model_id,  # explicit alias requested by the model-menu seam
        "dim": built_dim,
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "node_count": n,
        "entries": new_entries,
    }
    _save_manifest(memory_dir, manifest_out)
    print(f"[vector_index] wrote {embed_path} shape={embeddings.shape}")
    print(f"[vector_index] wrote {_manifest_path(memory_dir)}")
    return manifest_out
# build — Why: re-embedding is the expensive step, so the sha-keyed row cache means only
#     changed/new nodes are embedded on an incremental build; the manifest's dim is taken
#     from the matrix actually written (not a constant) so it can never disagree with
#     embeddings.npy, and recording model_id is what arms query()'s cross-embedder guard.


class EmbedModelMismatch(RuntimeError):
    """Raised when an index built with one embedder is queried under another.

    Carrying a dedicated type (vs a bare RuntimeError) lets a future model-menu caller
    catch exactly this condition — "this store needs a rebuild for the active model" —
    and trigger a rebuild instead of treating it as a generic failure.
    """


def _assert_active_model_matches(manifest: dict, memory_dir: Path) -> None:
    """Guard: the index's build-time model MUST equal the active query-time model.

    What: compares manifest['model_id'] (what built the embeddings.npy) against the
          live ASTHENOS_EMBED_MODEL selection. Mismatch => EmbedModelMismatch with the
          two ids + the env remedy. A legacy manifest with no model_id is treated as the
          historical MiniLM default (back-compat: those indexes WERE MiniLM).
    Why:  cosine similarity across two different embedding spaces is meaningless — the
          query vector and the stored matrix would live on different manifolds, returning
          confident GARBAGE with no error. This is the single seam that makes that
          silent-corruption class impossible: we fail loud, naming the fix (rebuild or
          re-point the env), instead of ranking nonsense.
    """
    built = manifest.get("model_id") or DEFAULT_MODEL_ID
    active = active_model_id()
    if built != active:
        raise EmbedModelMismatch(
            f"vector index at {memory_dir} was built with embedder '{built}' but the "
            f"active {EMBED_MODEL_ENV} is '{active}'. A query across two embedding "
            f"spaces is meaningless. Either set {EMBED_MODEL_ENV}='{built}' to query "
            f"this index, or rebuild it with vector.build(..., rebuild=True) under "
            f"'{active}'."
        )


def query(memory_dir: Path, text: str, top_k: int = 8) -> list[dict]:
    embed_path = _embed_path(memory_dir)
    manifest_p = _manifest_path(memory_dir)
    if not embed_path.exists() or not manifest_p.exists():
        raise SystemExit("[vector_index] no index found — run `build` first")
    manifest = _load_manifest(memory_dir)
    # Fail loud on a cross-embedder query BEFORE touching the matrix (cheap, no np.load).
    _assert_active_model_matches(manifest, memory_dir)
    embeddings = np.load(embed_path)
    entries = manifest.get("entries", {})
    # Exclude tombstoned (forget_node) entries from recall; unknown rows are skipped too.
    by_row = {e["row"]: (rel, e["title"])
              for rel, e in entries.items() if not e.get("tombstoned")}

    q = _embed_batch([text])[0]
    sims = embeddings @ q
    out: list[dict] = []
    for r in np.argsort(-sims):
        if len(out) >= top_k:
            break
        info = by_row.get(int(r))
        if info is None:
            continue  # tombstoned or unindexed row
        rel, title = info
        out.append({"score": float(sims[r]), "node": rel, "title": title})
    return out


def info(memory_dir: Path) -> None:
    manifest_p = _manifest_path(memory_dir)
    embed_path = _embed_path(memory_dir)
    if not manifest_p.exists():
        print("[vector_index] no manifest")
        return
    m = _load_manifest(memory_dir)
    print(f"model: {m.get('model_id')}")
    print(f"active_model: {active_model_id()}")
    print(f"dim: {m.get('dim')}")
    print(f"built_at: {m.get('built_at')}")
    print(f"node_count: {m.get('node_count')}")
    if embed_path.exists():
        e = np.load(embed_path)
        print(f"embeddings: shape={e.shape} dtype={e.dtype} bytes={e.nbytes}")


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.vector
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Carved from memory_vector_index.py
#             + SLOT-SCALING 2026-06-12 (v1.1): ASTHENOS_EMBED_MODEL embedder-
#               selection seam. build() records model_id + true dim (from the
#               embedded matrix) in the manifest; query() raises EmbedModelMismatch
#               on a cross-embedder query (silent-cosine-garbage guard).
#             + SEC 2026-06-12: cache-miss model download gated through netconsent.
# Layer:      core (pure library, no daemon dependency)
# Role:       semantic vector index over nodes/ — sha-keyed incremental embed build,
#             cosine top-k query (tombstone- + cross-embedder-aware), and the
#             ASTHENOS_EMBED_MODEL embedder-selection contract.
# Stability:  v1.1 -- semantic vector index: build / query / model-selection guard.
# ErrorModel: SystemExit (no index); EmbedModelMismatch (cross-embedder guard);
#             RuntimeError (refused model download). FAIL-LOUD on a cross-embedder
#             query; FAIL-SOFT TOCTOU on build (a node that vanishes mid-build is
#             skipped, not indexed this pass).
# Depends:    numpy; transformers + torch (mean-pool); samia.core.netconsent (fetch gate).
# Exposes:    build, query, info, active_model_id, active_model_dim,
#             EmbedModelMismatch, tombstone_node, untombstone_node, _embed_batch.
# Lines:      505
# --------------------------------------------------------------------------
