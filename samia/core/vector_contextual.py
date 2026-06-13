"""samia.core.vector_contextual — contextual vector index.

Carved from memory_vector_index_contextual.py. Companion to samia.core.vector;
prepends a structural-context prefix (chain_id + sibling node names) to each
node's text before embedding. Open-source-budget version of Anthropic's
Contextual Retrieval — uses chain-graph metadata instead of LLM-generated
context per chunk.

Index layout (under <memory_dir>/vector_index/):
    embeddings_contextual.npy
    manifest_contextual.json

Public API (parameterized on memory_dir):
    build(memory_dir, rebuild=False) → manifest dict
    query(memory_dir, text, top_k=24) → list[{score, node, title}]
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import vector as _vec

MODEL_ID = _vec.MODEL_ID
EMBED_DIM = _vec.EMBED_DIM
MAX_TOKENS = _vec.MAX_TOKENS


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _chains_dir(memory_dir: Path) -> Path:
    return memory_dir / "chains"


def _index_dir(memory_dir: Path) -> Path:
    return memory_dir / "vector_index"


def _embed_path(memory_dir: Path) -> Path:
    return _index_dir(memory_dir) / "embeddings_contextual.npy"


def _manifest_path(memory_dir: Path) -> Path:
    return _index_dir(memory_dir) / "manifest_contextual.json"


def _build_node_to_chain_map(memory_dir: Path) -> dict[str, dict]:
    m: dict[str, dict] = {}
    for p in sorted(_chains_dir(memory_dir).glob("*.json")):
        try:
            chain = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        chain_id = chain.get("chain_id") or p.stem
        members = chain.get("members") or []
        member_names = [Path(mm.get("file", "")).stem for mm in members
                        if mm.get("file")]
        for mm in members:
            f = mm.get("file") or ""
            if not f:
                continue
            fname = Path(f).name
            neighbors = [n for n in member_names if n != Path(fname).stem]
            m[fname] = {
                "chain_id": chain_id,
                "addr": mm.get("addr"),
                "neighbors": neighbors[:6],
                "tier": mm.get("tier"),
            }
    return m


def _load_node_text_contextual(path: Path,
                               node_to_chain: dict[str, dict]
                               ) -> tuple[str, str]:
    title, body = _vec._load_node_text(path)
    info = node_to_chain.get(path.name)
    if info:
        chain_id = info.get("chain_id") or "?"
        neighbors = info.get("neighbors") or []
        neighbor_str = ", ".join(neighbors) if neighbors else "(no siblings)"
        prefix = (f"This node is in chain '{chain_id}' "
                  f"with sibling nodes: {neighbor_str}. ")
    else:
        prefix = "This node is a singleton (no chain). "
    return title, prefix + body


def build(memory_dir: Path, rebuild: bool = False) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    embed_path = _embed_path(memory_dir)
    manifest_path = _manifest_path(memory_dir)
    _index_dir(memory_dir).mkdir(parents=True, exist_ok=True)

    manifest = {}
    if not rebuild and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    cached_entries = manifest.get("entries", {}) if not rebuild else {}

    nodes = sorted(nodes_dir.glob("*.md"))
    if not nodes:
        print(f"[contextual] no nodes at {nodes_dir}", file=sys.stderr)
        return {}

    node_to_chain = _build_node_to_chain_map(memory_dir)

    cached_emb = None
    if not rebuild and embed_path.exists():
        try:
            cached_emb = np.load(embed_path)
        except Exception:
            cached_emb = None

    new_entries: dict[str, dict] = {}
    rows: list = []
    paths: list[str] = []
    titles: list[str] = []
    to_embed_idx: list[int] = []
    to_embed_text: list[str] = []

    for i, p in enumerate(nodes):
        rel = p.name
        title, content = _load_node_text_contextual(p, node_to_chain)
        sig = _vec._sha256(content)
        cached = cached_entries.get(rel)
        paths.append(rel)
        titles.append(title)
        new_entries[rel] = {"sha256": sig, "title": title, "row": i,
                            "in_chain": rel in node_to_chain}

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

    n = len(nodes)
    if to_embed_idx:
        print(f"[contextual] embedding {len(to_embed_idx)}/{n} "
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
        print(f"[contextual] all {n} nodes cached")

    embeddings = np.vstack(rows).astype(np.float32)
    np.save(embed_path, embeddings)

    manifest_out = {
        "model_id": MODEL_ID,
        "dim": EMBED_DIM,
        "variant": "contextual",
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "node_count": n,
        "chain_node_count": sum(1 for e in new_entries.values()
                                if e.get("in_chain")),
        "entries": new_entries,
    }
    manifest_path.write_text(json.dumps(manifest_out, indent=2),
                             encoding="utf-8")
    print(f"[contextual] index built: {n} nodes "
          f"({manifest_out['chain_node_count']} in chains) → "
          f"{embed_path.name}")
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
