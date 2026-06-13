"""samia.core.entity_index — inter-chain entity-bridge index.

Carved from memory_entity_index.py. The library plane carries all logic
so the daemon can call build_index/query_bridges directly.

Extracts named entities and technical identifiers from each live node,
inverts to build entity → [nodes]. At retrieval time, entities found in
the query expand the candidate set to bridge across chains that the
within-chain Hebbian traversal would miss.

Pattern: Graphiti / Zep entity-bridge edges, but without a graph DB —
the inverted index is a single JSON file regenerated on demand.

Public API (parameterized on memory_dir):
  build_index(memory_dir) → dict
  load_index(memory_dir) → dict | None
  query_bridges(memory_dir, query, max_bridge_nodes=8, min_entity_len=3) → dict
  extract_entities(text) → set[str]

Acceptance: byte-identical to pre-refactor memory_entity_index.py CLI
behavior on the same memory tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ALLCAPS_STOP = {
    "NOT", "AND", "OR", "BUT", "FOR", "THE", "ALL", "NEW", "API", "WHY",
    "HOW", "WHAT", "WHEN", "TODO", "FIXME", "BUG", "NB", "WAS", "ARE",
    "AKA", "TBD", "ETA", "FYI", "URL", "HTTP", "HTTPS", "JSON", "HTML",
    "CSS", "XML", "YAML", "UTF",
}
COMMON_HYPHEN_STOP = {
    "as-is", "at-least", "make-or", "one-off", "up-to", "drop-in",
    "off-the", "out-of", "as-of", "in-flight", "in-progress", "at-spi",
    "one-shot", "high-level", "low-level", "cross-platform", "run-time",
    "real-time", "re-read", "re-run", "re-do", "re-try", "re-use",
    "re-load", "re-init", "lock-in", "check-in", "follow-up", "sign-in",
    "sign-out", "log-in", "log-out", "set-up", "clean-up", "stand-up",
    "write-up", "fall-back", "work-around", "pre-auth", "pre-built",
    "post-hoc", "case-by", "day-to", "face-to", "self-aware", "self-host",
    "self-care", "both", "first", "never", "always", "before", "after",
    "then", "valid_from", "valid_to", "last_access", "access_count",
    "review_count", "next_review", "review_interval_days",
    "easiness_factor", "relevance",
}


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _index_path(memory_dir: Path) -> Path:
    return memory_dir / "vector_index" / "entity_bridges.json"


def extract_entities(text: str) -> set[str]:
    ents: set[str] = set()

    for m in re.findall(r"`([^`\n]{2,40})`", text):
        s = m.strip()
        if s and not s.isdigit():
            ents.add(s.lower())

    for m in re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9]+(?:[_\-][a-zA-Z0-9]+){1,}\b", text):
        if len(m) >= 5 and m.lower() not in COMMON_HYPHEN_STOP:
            ents.add(m.lower())

    for m in re.findall(
            r"[A-Za-z0-9_\-./]+\.(?:py|md|json|sh|cu|c|h|txt|toml|gguf|enc|loer)\b",
            text):
        ents.add(m.lower())

    for m in re.findall(r"\b[A-Z]{3,}[0-9]*\b", text):
        if m not in ALLCAPS_STOP and len(m) <= 12:
            ents.add(m.lower())

    for m in re.findall(r"#\d{3,}", text):
        ents.add(m)
    for m in re.findall(r"\bv\d+(?:\.\d+){1,3}\b", text):
        ents.add(m.lower())

    for m in re.findall(
            r"\b(?:[A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,}){1,2}\b", text):
        s = m.lower()
        if s not in COMMON_HYPHEN_STOP:
            ents.add(s)

    return ents


def build_index(memory_dir: Path) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    index_path = _index_path(memory_dir)

    entity_to_nodes: dict[str, list[str]] = {}
    node_to_entities: dict[str, list[str]] = {}

    md_files = sorted(nodes_dir.glob("*.md"))
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ents = extract_entities(text)
        node_to_entities[p.name] = sorted(ents)
        for e in ents:
            entity_to_nodes.setdefault(e, []).append(p.name)

    bridge_count = sum(1 for nodes in entity_to_nodes.values()
                       if len(nodes) >= 2)
    singleton_count = sum(1 for nodes in entity_to_nodes.values()
                          if len(nodes) == 1)

    out = {
        "n_nodes": len(node_to_entities),
        "n_entities_total": len(entity_to_nodes),
        "n_bridge_entities": bridge_count,
        "n_singleton_entities": singleton_count,
        "entity_to_nodes": entity_to_nodes,
        "node_to_entities": node_to_entities,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(out, indent=2))
    return out


def load_index(memory_dir: Path) -> dict | None:
    index_path = _index_path(memory_dir)
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


def query_bridges(memory_dir: Path, query: str,
                  max_bridge_nodes: int = 8,
                  min_entity_len: int = 3) -> dict:
    idx = load_index(memory_dir)
    if idx is None:
        return {"error": "no entity index — run memory_entity_index.py build"}

    known: set[str] = set(idx["entity_to_nodes"].keys())

    q_ents = extract_entities(query)

    q_lower = query.lower()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-./]{2,}", q_lower):
        if tok in known:
            q_ents.add(tok)

    tokens = re.findall(r"[a-z][a-z0-9]{2,}", q_lower)
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        if bg in known:
            q_ents.add(bg)

    q_ents = {e for e in q_ents if len(e) >= min_entity_len}
    if not q_ents:
        return {"matched_entities": [], "bridge_nodes": [],
                "rationale": "no entities matched index"}

    node_scores: dict[str, dict] = {}
    matched: list[str] = []
    for e in q_ents:
        nodes = idx["entity_to_nodes"].get(e, [])
        if not nodes:
            continue
        matched.append(e)
        weight = 1.0 / max(len(nodes), 1)
        for n in nodes:
            d = node_scores.setdefault(n, {"score": 0.0, "entities": []})
            d["score"] += weight
            d["entities"].append(e)

    ordered = sorted(node_scores.items(), key=lambda kv: -kv[1]["score"])
    bridge_nodes = [
        {"node": n, "score": d["score"], "entities": d["entities"]}
        for n, d in ordered[:max_bridge_nodes]
    ]
    return {
        "matched_entities": sorted(matched),
        "bridge_nodes": bridge_nodes,
        "rationale": (f"{len(matched)} query entities matched, "
                      f"{len(node_scores)} bridge candidates → "
                      f"top {len(bridge_nodes)}"),
    }
