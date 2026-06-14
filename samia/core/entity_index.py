"""samia.core.entity_index — inter-chain entity-bridge index.

Layer 1 (Owns / Depends):
    Owns:    build_index, load_index, query_bridges, extract_entities — build the
             entity → [nodes] inverted index, load it, expand a query's entities
             into bridge-node candidates, and the entity/identifier extractor those
             three share. All parameterized on memory_dir.
    Depends: stdlib only (json, re, pathlib).
Layer 2 (What / Why):
    What: extract_entities pulls named entities and technical identifiers
          (backtick spans, hyphen/underscore identifiers, file names, ALLCAPS acronyms,
          issue/version tokens, Capitalized phrases) from each live node, and the
          index inverts them to entity → [nodes]. At retrieval time, entities found in
          the query (query_bridges) expand the candidate set across chains.
    Why:  the within-chain Hebbian traversal only walks edges inside one chain, so a
          fact that connects two chains by a shared entity is invisible to it. The
          inverted index supplies those cross-chain bridges. Pattern: Graphiti / Zep
          entity-bridge edges, but without a graph DB — a single JSON file regenerated
          on demand. The stoplists prune generic tokens so a bridge means a shared
          SPECIFIC entity, not a shared common word.
Layer 3 (Changelog):
    (carved from memory_entity_index.py — library plane extracted from the original CLI;
     byte-identical to the pre-refactor CLI behavior on the same memory tree.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ALLCAPS_STOP — What: ALLCAPS tokens that look like acronyms but carry no bridging value.
ALLCAPS_STOP = {
    "NOT", "AND", "OR", "BUT", "FOR", "THE", "ALL", "NEW", "API", "WHY",
    "HOW", "WHAT", "WHEN", "TODO", "FIXME", "BUG", "NB", "WAS", "ARE",
    "AKA", "TBD", "ETA", "FYI", "URL", "HTTP", "HTTPS", "JSON", "HTML",
    "CSS", "XML", "YAML", "UTF",
}
# ALLCAPS_STOP — Why: the ALLCAPS regex would otherwise index generic English/protocol
#     words as entities, creating spurious bridges between unrelated chains.

# COMMON_HYPHEN_STOP — What: hyphen/underscore tokens (and a few phrases) that match the
#     identifier regexes but are generic connectors or frontmatter keys, not entities.
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
# COMMON_HYPHEN_STOP — Why: frontmatter keys (valid_from, last_access, …) and generic
#     hyphenated phrases recur in nearly every node, so without this stoplist they would
#     bridge ALL chains and drown the genuine entity signal.


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _index_path(memory_dir: Path) -> Path:
    return memory_dir / "vector_index" / "entity_bridges.json"


# extract_entities — What: collect the lowercased entity/identifier set from one node's
#     text across six pattern families (code spans, hyphen/underscore ids, file names,
#     ALLCAPS acronyms, issue/version tokens, Capitalized multi-word phrases).
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
# extract_entities — Why: a node bridges another only on a SHARED specific entity, so each
#     family targets a class of stable identifier (a path, an acronym, an issue id) and the
#     length/stoplist guards drop generic words; lowercasing makes the match case-insensitive.


# build_index — What: scan every nodes/*.md, extract its entities, and write the inverted
#     entity → [nodes] map (plus per-node entity lists and bridge/singleton counts) to
#     vector_index/entity_bridges.json; return the same dict.
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
# build_index — Why: the index is regenerated on demand (no graph DB, no incremental
#     maintenance), so the whole nodes/ tree is rescanned each build; a read OSError on one
#     node is skipped rather than aborting the build.


def load_index(memory_dir: Path) -> dict | None:
    index_path = _index_path(memory_dir)
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


# query_bridges — What: given a free-text query, derive its entities, look each up in the
#     loaded index, and return the top-N nodes that share those entities (each scored by the
#     rarity-weighted count of matched entities), plus the matched entities and a rationale.
def query_bridges(memory_dir: Path, query: str,
                  max_bridge_nodes: int = 8,
                  min_entity_len: int = 3) -> dict:
    idx = load_index(memory_dir)
    if idx is None:
        return {"error": "no entity index — run memory_entity_index.py build"}

    known: set[str] = set(idx["entity_to_nodes"].keys())

    q_ents = extract_entities(query)

    # KnownTokenAugment — What: also add any single token / adjacent-token bigram of the
    #     query that already EXISTS as an index entity, beyond what extract_entities found.
    q_lower = query.lower()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-./]{2,}", q_lower):
        if tok in known:
            q_ents.add(tok)

    tokens = re.findall(r"[a-z][a-z0-9]{2,}", q_lower)
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        if bg in known:
            q_ents.add(bg)
    # KnownTokenAugment — Why: extract_entities applies the same length/stoplist guards to
    #     the query as to nodes, so a real index entity phrased plainly in the query could be
    #     missed; gating the augmentation on `tok in known` only ever adds entities the index
    #     can actually bridge on, never noise.

    q_ents = {e for e in q_ents if len(e) >= min_entity_len}
    if not q_ents:
        return {"matched_entities": [], "bridge_nodes": [],
                "rationale": "no entities matched index"}

    # RarityWeightedScore — What: accumulate a per-node score across matched entities, where
    #     each entity contributes 1/len(nodes-it-appears-in) to every node it touches.
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
    # RarityWeightedScore — Why: a rare entity (in few nodes) is a stronger bridge signal
    #     than a common one (an inverse-document-frequency weighting), so 1/len(nodes) up-
    #     weights the distinctive shared entities that actually connect two chains.

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


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.entity_index
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Carved from memory_entity_index.py (library plane extraction).
# Layer:      core (pure library, no daemon dependency)
# Role:       the cross-chain entity-bridge index — invert nodes to shared specific
#             entities so a query expands its candidate set across chains.
# Stability:  stable -- inter-chain entity-bridge index; API parameterized on memory_dir.
# ErrorModel: query_bridges returns {"error": ...} when no index has been built, and an
#             empty bridge set when the query yields no indexed entities; build_index skips
#             a node it cannot read (OSError) rather than aborting; load_index returns None
#             when the index file is absent.
# Depends:    json, re, pathlib (stdlib only).
# Exposes:    build_index, load_index, query_bridges, extract_entities.
#             Constants: ALLCAPS_STOP, COMMON_HYPHEN_STOP.
# Lines:      243
# --------------------------------------------------------------------------
