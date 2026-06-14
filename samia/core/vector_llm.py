"""samia.core.vector_llm — LLM-preamble vector index.

Layer 1 (Owns / Depends):
    Owns:    build(memory_dir, rebuild=False, force_llm=False, limit=None) -> manifest
                 dict — embed each node prefixed with an LLM-generated context preamble.
             query(memory_dir, text, top_k=24) -> list[{node, score, title}].
    Depends: numpy; samia.core.vector (shared embedder + node-text + sha helpers);
             samia.core.vector_contextual (node->chain map); a local LLM backend —
             the SAM/IA runtime daemon (samia.runtime.client) first, else Ollama over
             urllib (MEMORY_PREAMBLE_MODEL, default gemma3:4b).
Layer 2 (What / Why):
    What: each node is prepended with a 50-100 token LLM-generated context preamble
          (Anthropic Contextual Retrieval) before being embedded into embeddings_llm.npy
          / manifest_llm.json. The manifest caches (body_sha256, preamble, embed_sha256)
          per node; a body edit invalidates the preamble, a structural-only change
          re-embeds but reuses the cached LLM text, and force_llm=True regenerates all.
    Why:  the preamble injects chain/sibling context the body alone lacks, so semantic
          recall locates a node by what it MEANS in the chain, not just its words. The
          preamble model is intentionally cheap/local (no thinking mode) and the per-node
          cache keeps an incremental build off the LLM for unchanged nodes. A daemon
          backend (Qwen via LlamaCppBackend) is tried first for telemetry; an Ollama
          fallback honors MEMORY_PREAMBLE_MODEL; a structural fallback covers total LLM
          failure so a build never aborts on a generation error.

Layer 3 (Changelog):
    Carved from memory_vector_index_llm.py. Companion to samia.core.vector and
    samia.core.vector_contextual.
    AUD28.7 V1: prefer the runtime daemon's infer op over Ollama for preambles.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import vector as _vec
from . import vector_contextual as _vc

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
PREAMBLE_MODEL = os.environ.get("MEMORY_PREAMBLE_MODEL", "gemma3:4b")
PREAMBLE_TIMEOUT_S = float(os.environ.get("MEMORY_PREAMBLE_TIMEOUT_S", "30"))
PREAMBLE_MAX_TOKENS = 220
PREAMBLE_TEMPERATURE = 0.2


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _index_dir(memory_dir: Path) -> Path:
    return memory_dir / "vector_index"


def _embed_path(memory_dir: Path) -> Path:
    return _index_dir(memory_dir) / "embeddings_llm.npy"


def _manifest_path(memory_dir: Path) -> Path:
    return _index_dir(memory_dir) / "manifest_llm.json"


def _build_prompt(title: str, body: str, chain_id: str | None,
                  neighbors: list[str]) -> str:
    if chain_id and neighbors:
        siblings = ", ".join(neighbors[:5])
        chain_clause = (
            f"This node belongs to the SAM/IA memory chain '{chain_id}'. "
            f"Sibling nodes in the same chain include: {siblings}."
        )
    elif chain_id:
        chain_clause = (
            f"This node is in the SAM/IA memory chain '{chain_id}' "
            f"with no listed siblings."
        )
    else:
        chain_clause = ("This node is a singleton (not yet attached to "
                        "any chain).")

    body_excerpt = body.strip()
    if len(body_excerpt) > 1800:
        body_excerpt = body_excerpt[:1800] + " …"

    return (
        "You are writing a 50-100 token context preamble for a single "
        "memory node so that semantic search can locate it. The preamble "
        "must concisely state WHAT this node records and HOW it relates "
        "to its chain. Do NOT restate the body verbatim — synthesize.\n\n"
        f"Node title: {title}\n"
        f"{chain_clause}\n\n"
        f"Body excerpt:\n---\n{body_excerpt}\n---\n\n"
        "Output ONLY the 50-100 token preamble itself, in plain prose, "
        "starting with 'This node'. No preface, no headers, no lists."
    )


def _generate_preamble_daemon(prompt: str) -> str | None:
    """AUD28.7 V1: try the SAM/IA runtime daemon's infer op first.

    Note: daemon uses LlamaCppBackend's configured GGUF (Qwen2.5-Coder-14B)
    rather than PREAMBLE_MODEL (gemma3:4b). Returns None on failure;
    caller falls through to Ollama path which honors PREAMBLE_MODEL.
    """
    try:
        from samia.runtime.client import SamiaClient, DaemonNotRunning
    except ImportError:
        return None
    try:
        with SamiaClient(timeout=PREAMBLE_TIMEOUT_S) as client:
            result = client.call(
                "infer",
                prompt=prompt,
                max_tokens=PREAMBLE_MAX_TOKENS,
                temperature=PREAMBLE_TEMPERATURE,
                caller_hint="samia.core.vector_llm._generate_preamble",
            )
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                return text
        return None
    except (DaemonNotRunning, Exception):
        return None


# _generate_preamble — What: produce a node's preamble via the backend cascade
#     (daemon -> Ollama), then strip any <think>...</think> reasoning prefix.
def _generate_preamble(title: str, body: str, chain_id: str | None,
                       neighbors: list[str]) -> str | None:
    prompt = _build_prompt(title, body, chain_id, neighbors)

    # Backend 0: daemon (AUD28.7 V1) — Qwen via LlamaCppBackend, telemetry-emitting.
    text = _generate_preamble_daemon(prompt)
    if not text:
        # Fallback: Ollama gemma3:4b (or whatever PREAMBLE_MODEL points to).
        payload = json.dumps({
            "model": PREAMBLE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": PREAMBLE_MAX_TOKENS,
                "temperature": PREAMBLE_TEMPERATURE,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=PREAMBLE_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[llm_preamble] generate failed for '{title[:40]}…': {e}",
                  file=sys.stderr)
            return None
        text = (data.get("response") or "").strip()

    if not text:
        return None
    if text.startswith("<think>"):
        end = text.find("</think>")
        text = text[end + len("</think>"):].strip() if end >= 0 else ""
    return text or None
# _generate_preamble — Why: a reasoning model may emit a <think> block before the answer;
#     keeping only the post-</think> text avoids embedding the model's scratch reasoning. An
#     unterminated <think> (no closing tag) is treated as all-reasoning -> empty -> None, so
#     the caller falls through to the structural fallback rather than embedding noise.


def _structural_fallback(chain_id: str | None,
                         neighbors: list[str]) -> str:
    if chain_id and neighbors:
        return (
            f"This node is in chain '{chain_id}' with sibling nodes: "
            f"{', '.join(neighbors[:6])}. "
        )
    if chain_id:
        return f"This node is in chain '{chain_id}'. "
    return "This node is a singleton (no chain). "


def build(memory_dir: Path, rebuild: bool = False,
          force_llm: bool = False, limit: int | None = None) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    embed_path = _embed_path(memory_dir)
    manifest_path = _manifest_path(memory_dir)
    _index_dir(memory_dir).mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    if not rebuild and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    cached_entries = manifest.get("entries", {}) if not rebuild else {}

    nodes = sorted(nodes_dir.glob("*.md"))
    if limit:
        nodes = nodes[:limit]
    if not nodes:
        print(f"[llm] no nodes at {nodes_dir}", file=sys.stderr)
        return {}

    node_to_chain = _vc._build_node_to_chain_map(memory_dir)

    cached_emb = None
    if not rebuild and embed_path.exists():
        try:
            cached_emb = np.load(embed_path)
        except Exception:
            cached_emb = None

    new_entries: dict[str, dict] = {}
    rows: list = []
    titles: list[str] = []
    to_embed_idx: list[int] = []
    to_embed_text: list[str] = []
    llm_calls = 0
    llm_fallbacks = 0

    for i, p in enumerate(nodes):
        title, body = _vec._load_node_text(p)
        body_sig = _vec._sha256(body)
        info = node_to_chain.get(p.name) or {}
        chain_id = info.get("chain_id")
        neighbors = info.get("neighbors") or []
        cached = cached_entries.get(p.name) or {}
        cached_body_sig = cached.get("body_sha256")
        # PreambleCache — What: reuse the cached preamble unless force_llm is set or the
        #     node's body sha changed; a (re)generation falls back to a structural string
        #     when the LLM yields nothing.
        preamble: str | None = (cached.get("preamble")
                                if not force_llm else None)

        if cached_body_sig != body_sig:
            preamble = None

        if preamble is None:
            preamble = _generate_preamble(title, body, chain_id, neighbors)
            if preamble:
                llm_calls += 1
            else:
                preamble = _structural_fallback(chain_id, neighbors)
                llm_fallbacks += 1
        # PreambleCache — Why: the LLM call is the expensive step, so the preamble is keyed
        #     on the BODY sha (not the embed text): a structural-only change reuses the LLM
        #     text and only re-embeds, while a body edit forces a fresh preamble. The
        #     structural fallback guarantees every node gets SOME context so a generation
        #     outage degrades quality rather than dropping the node from the index.

        text_for_embed = (preamble.strip() + "\n\n" + body).strip()
        text_sig = _vec._sha256(text_for_embed)

        new_entries[p.name] = {
            "body_sha256": body_sig,
            "embed_sha256": text_sig,
            "preamble": preamble,
            "title": title,
            "row": i,
            "in_chain": bool(chain_id),
        }
        titles.append(title)
        cached_embed_sig = cached.get("embed_sha256")
        if (cached_embed_sig == text_sig and cached_emb is not None
                and cached.get("row") is not None):
            old_row = cached["row"]
            if old_row < cached_emb.shape[0]:
                rows.append(cached_emb[old_row].copy())
                continue
        rows.append(None)
        to_embed_idx.append(i)
        to_embed_text.append(text_for_embed)

    n = len(nodes)
    print(f"[llm] preambles: {llm_calls} fresh, "
          f"{llm_fallbacks} structural fallbacks, "
          f"{n - llm_calls - llm_fallbacks} cached")

    if to_embed_idx:
        print(f"[llm] embedding {len(to_embed_idx)}/{n} "
              f"(cached: {n - len(to_embed_idx)})")
        BATCH = 16
        new_vecs: list[np.ndarray] = []
        for s in range(0, len(to_embed_text), BATCH):
            chunk = to_embed_text[s:s + BATCH]
            t0 = time.time()
            v = _vec._embed_batch(chunk)
            dt = time.time() - t0
            print(f"  batch {s // BATCH + 1}: {len(chunk)} in {dt:.2f}s")
            new_vecs.append(v)
        new_arr = np.vstack(new_vecs)
        for k, idx in enumerate(to_embed_idx):
            rows[idx] = new_arr[k]
    else:
        print(f"[llm] all {n} nodes cached")

    embeddings = np.vstack(rows).astype(np.float32)
    np.save(embed_path, embeddings)

    manifest_out = {
        "model_id": _vec.MODEL_ID,
        "dim": _vec.EMBED_DIM,
        "variant": "llm_preamble",
        "preamble_model": PREAMBLE_MODEL,
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "node_count": n,
        "llm_calls_this_run": llm_calls,
        "fallbacks_this_run": llm_fallbacks,
        "entries": new_entries,
    }
    manifest_path.write_text(json.dumps(manifest_out, indent=2),
                             encoding="utf-8")
    print(f"[llm] index built: {n} nodes → {embed_path.name}")
    return manifest_out


def query(memory_dir: Path, text: str, top_k: int = 24) -> list[dict]:
    embed_path = _embed_path(memory_dir)
    manifest_path = _manifest_path(memory_dir)
    if not manifest_path.exists() or not embed_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings = np.load(embed_path)
    qv = _vec._embed_batch([text])[0]
    sims = embeddings @ qv
    top = np.argsort(-sims)[:top_k]
    out: list[dict] = []
    entries = manifest["entries"]
    name_by_row: dict[int, str] = {}
    for rel, e in entries.items():
        name_by_row[int(e["row"])] = rel
    for idx in top:
        idx_i = int(idx)
        rel = name_by_row.get(idx_i)
        if rel is None:
            continue
        out.append({"node": rel, "score": float(sims[idx_i]),
                    "title": entries[rel].get("title", rel)})
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.vector_llm
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Carved from memory_vector_index_llm.py
#             + AUD28.7 V1: prefer the runtime daemon's infer op over Ollama for
#               preamble generation (telemetry-emitting), Ollama as fallback.
# Layer:      core (library; needs a local LLM backend for preamble generation)
# Role:       LLM-preamble vector index — prepends a 50-100 token LLM-generated context
#             preamble (daemon→Ollama cascade, structural fallback) per node before
#             embedding; body-sha-keyed preamble cache + cosine top-k query.
# Stability:  stable -- LLM-preamble (Contextual Retrieval) vector index variant.
# ErrorModel: a preamble LLM failure (daemon down, Ollama error, empty/think-only
#             output) degrades to a structural fallback string — a build never
#             aborts on a generation error; query() returns [] when no index exists.
# Depends:    numpy; samia.core.vector (embedder + sha + node-text helpers);
#             samia.core.vector_contextual (node->chain map); samia.runtime.client
#             (daemon backend, optional) + Ollama over urllib (MEMORY_PREAMBLE_MODEL).
# Exposes:    build, query.
# Lines:      368
# --------------------------------------------------------------------------
