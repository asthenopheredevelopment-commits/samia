"""samia.core.bio.reconsolidate — recall-is-a-write reconsolidation (Nader et al. 2000).

Layer 1 (Owns / Depends):
    Owns:    reconsolidate — read a node, extract atoms from new context, and for each
             atom either MERGE it into the recalled node (when pattern separation says
             merge-into THIS node) or SPAWN a refining sibling node; append a
             reconsolidation-log record. The recall event is treated as a write
             opportunity (the empirical reconsolidation window).
    Depends: config (json / _dt / Path / _chain); samia.core.bio.pattern
             (pattern_separation_decision — the dedup gate); samia.core.{temporal,
             fact_extractor} (lazy, function-local — node read/write + atom extraction).

Layer 2 (What / Why):
    What: the single reconsolidation driver. It is the only arm that both reads node
          frontmatter/body and may write new sibling nodes, so it is its own submodule.
    Why:  carved out of the monolith as the reconsolidation responsibility. temporal +
          fact_extractor are lazy (function-local) exactly as the monolith had them
          (fact_extractor pulls the inference backend; temporal pulls heavier deps), so
          `import bio` stays cheap. The cross-arm call to pattern_separation_decision is a
          plain submodule import (pattern depends only on config — no cycle).
"""

from __future__ import annotations

from . import config as _cfg
from .config import json, _dt, _chain, Path
from .pattern import pattern_separation_decision


def reconsolidate(memory_dir: Path, node_name: str, new_context: str,
                  backend: str = "auto") -> dict:
    """Recall + LLM-update a node."""
    nodes_dir = memory_dir / "nodes"
    chains_dir = memory_dir / "chains"
    paths = _cfg._bio_paths(memory_dir)
    from samia.core import temporal as _tq
    from samia.core import fact_extractor as _fx

    p = nodes_dir / node_name
    if not p.suffix:
        p = p.with_suffix(".md")
    if not p.exists():
        return {"error": f"node not found: {p.name}"}

    fm_lines, body = _tq.read_node(p)
    chains = []
    raw_chains = _tq.fm_get(fm_lines, "chains") or ""
    if raw_chains.startswith("[") and raw_chains.endswith("]"):
        inner = raw_chains[1:-1].strip()
        chains = [c.strip() for c in inner.split(",") if c.strip()]

    atoms = _fx.extract_atoms(new_context, backend=backend, chains_hint=chains)
    if not atoms:
        return {"node": p.name, "atoms": 0, "merged": 0, "spawned": 0}

    today = _dt.date.today().isoformat()
    merged = 0
    spawned: list[str] = []

    for atom in atoms:
        decision = pattern_separation_decision(memory_dir, atom["body"])
        if decision["action"] == "merge_into" and decision["target"] == p.name:
            body = body.rstrip() + f"\n\n## reconsolidated {today}\n{atom['body']}\n"
            merged += 1
        elif decision["action"] == "merge_into":
            continue
        else:
            sibling_name = _fx._slug(atom.get("title") or atom.get("description") or "refine")
            sibling_path_stem = f"{Path(p.stem).stem}__refine_{sibling_name}_{today}"
            sib_p = nodes_dir / f"{sibling_path_stem}.md"
            counter = 1
            while sib_p.exists():
                sib_p = nodes_dir / f"{sibling_path_stem}_{counter}.md"
                counter += 1
            chain_field = "[" + ", ".join(chains) + "]"
            fm = [
                f"name: {atom.get('title', sib_p.stem)}",
                f"description: {atom.get('description', '')}",
                f"type: {atom.get('type', 'project')}",
                f"chains: {chain_field}",
                f"valid_from: {atom.get('valid_from') or today}",
                f"valid_to: {atom.get('valid_to') if atom.get('valid_to') else 'null'}",
                f"last_access: {today}",
                "access_count: 0",
                "relevance: 0.55",
                "tier: warm",
                "extracted: true",
                f"refines: {p.name}",
            ]
            out = "---\n" + "\n".join(fm) + "\n---\n" + atom["body"].strip() + "\n"
            sib_p.write_text(out, encoding="utf-8")
            spawned.append(sib_p.name)

            for cn in chains:
                try:
                    chain = _chain.load_chain(chains_dir, cn)
                except (SystemExit, FileNotFoundError):
                    continue
                addrs = {m.get("addr") for m in chain.get("members") or []}
                _ = addrs

    if merged:
        fm_lines = _tq.fm_set(fm_lines, "last_access", today)
        ac = int(_tq.fm_get(fm_lines, "access_count") or "0") + 1
        fm_lines = _tq.fm_set(fm_lines, "access_count", str(ac))
        _tq.write_node(p, fm_lines, body)

    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    rec = {"ts": today, "node": p.name, "atoms": len(atoms),
           "merged": merged, "spawned": spawned}
    with paths["reconsolidate_log"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.reconsolidate
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.bio monolith during
#             modularization.
# Layer:      core (pure library, no daemon dependency)
# Role:       the reconsolidation arm — reconsolidate treats a recall as a write
#             opportunity: extract atoms from new context, then per atom either merge
#             into the recalled node or spawn a refining sibling, and append a
#             reconsolidation-log record.
# Stability:  stable — the only arm that writes new sibling nodes.
# ErrorModel: returns {"error": ...} for a missing node; a chain load failure per atom
#             is caught + skipped; the reconsolidation-log append is best-effort.
# Depends:    .config (json / _dt / _chain / _bio_paths); .pattern
#             (pattern_separation_decision — the dedup gate); samia.core.{temporal,
#             fact_extractor} (lazy, function-local — read/write + atom extraction).
# Exposes:    reconsolidate (public).
# Lines:      133
# --------------------------------------------------------------------------
