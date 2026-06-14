"""samia.core.bio.schema — schema-accelerated ingestion (Tse et al. 2007).

Layer 1 (Owns / Depends):
    Owns:    chain_maturity (is a chain old + populated enough to be a schema?) and
             schema_accelerate (a new node entering a MATURE chain with good semantic
             fit skips the cold-start relevance/tier and is born hot — the schema
             consolidation acceleration).
    Depends: config (constants SCHEMA_MIN_NODES / SCHEMA_MIN_AGE_DAYS / _dt / _chain);
             samia.core.bio.pattern (pattern_separation_decision — the semantic-fit
             gate); samia.core.temporal (lazy, function-local — node read + date parse).

Layer 2 (What / Why):
    What: the maturity predicate + the initial-relevance/tier decision for a node
          about to be written.
    Why:  carved out of the monolith as the schema responsibility. temporal is lazy
          (function-local) exactly as the monolith had it. The cross-arm call to
          pattern_separation_decision is a plain import (pattern depends only on config).
"""

from __future__ import annotations

from typing import Optional

from . import config as _cfg
from .config import _dt, _chain, SCHEMA_MIN_NODES, SCHEMA_MIN_AGE_DAYS
from .pattern import pattern_separation_decision


def chain_maturity(memory_dir, chain_name: str) -> dict:
    from samia.core import temporal as _tq
    nodes_dir = memory_dir / "nodes"
    chains_dir = memory_dir / "chains"
    try:
        chain = _chain.load_chain(chains_dir, chain_name)
    except (SystemExit, FileNotFoundError):
        return {"chain": chain_name, "mature": False, "reason": "no manifest"}
    members = chain.get("members") or []
    if len(members) < SCHEMA_MIN_NODES:
        return {"chain": chain_name, "mature": False, "node_count": len(members),
                "reason": "too few nodes"}
    earliest: Optional[_dt.date] = None
    for m in members:
        f = m.get("file") if isinstance(m, dict) else None
        if not f:
            continue
        p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        fm, _ = _tq.read_node(p)
        vf = _tq.parse_date(_tq.fm_get(fm, "valid_from"))
        if vf and (earliest is None or vf < earliest):
            earliest = vf
    if earliest is None:
        return {"chain": chain_name, "mature": False, "reason": "no valid_from on members"}
    age = (_dt.date.today() - earliest).days
    if age < SCHEMA_MIN_AGE_DAYS:
        return {"chain": chain_name, "mature": False, "age_days": age,
                "reason": "chain too young"}
    return {"chain": chain_name, "mature": True, "node_count": len(members),
            "age_days": age, "earliest": earliest.isoformat()}


def schema_accelerate(memory_dir, text: str, chains: list[str]) -> dict:
    """Decide initial relevance/tier for a new node about to be written."""
    if not chains:
        return {"relevance": 0.50, "tier": "warm",
                "rationale": "no chain hint — default warm"}
    matures = [c for c in chains if chain_maturity(memory_dir, c).get("mature")]
    if not matures:
        return {"relevance": 0.50, "tier": "warm",
                "rationale": "chains not mature — default warm"}
    decision = pattern_separation_decision(memory_dir, text, threshold=0.65)
    if decision["score"] < 0.65:
        return {"relevance": 0.50, "tier": "warm",
                "rationale": f"low semantic fit ({decision['score']:.2f})"}
    return {"relevance": 0.72, "tier": "hot",
            "rationale": f"schema-accelerated via mature chain(s) {matures}; "
                         f"semantic fit {decision['score']:.2f}",
            "matured_chains": matures}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.schema
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.bio monolith during
#             modularization.
# Layer:      core (pure library, no daemon dependency)
# Role:       the schema-acceleration arm — chain_maturity (old + populated enough to be
#             a schema?) and schema_accelerate (a well-fitting node entering a mature
#             chain is born hot, skipping the cold-start relevance/tier).
# Stability:  stable — pure read-only decisions; writes nothing.
# ErrorModel: chain_maturity returns {"mature": False, "reason": ...} for a missing /
#             too-small / too-young chain; schema_accelerate defaults to warm on any
#             non-acceleration path.
# Depends:    .config (SCHEMA_MIN_NODES / SCHEMA_MIN_AGE_DAYS / _dt / _chain); .pattern
#             (pattern_separation_decision — the semantic-fit gate); samia.core.temporal
#             (lazy, function-local — node read + date parse).
# Exposes:    chain_maturity, schema_accelerate (public).
# Lines:      101
# --------------------------------------------------------------------------
