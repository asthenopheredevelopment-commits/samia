"""samia.core.consolidation — find merge candidates in SAM chains.

Carved from memory_consolidation_detector.py. Per design doc §1.1 + §1.3,
the daemon's consolidation job calls audit_all() on a schedule; the result
feeds .consolidation_candidates.json which MEMORY.md surfaces.

Public API:
  strip_frontmatter(text)             → str
  shingles(text)                      → set[str]   (content-words)
  jaccard(a, b)                       → float
  load_chain(chain_path)              → dict
  load_node_body(memory_dir, rel)     → str | None
  audit_chain(chain_path, threshold,
              memory_dir)             → list[dict]
  audit_all(memory_dir, threshold,
            chain=None)               → list[dict]   (sorted desc by sim)
  surface(memory_dir, findings,
          threshold)                  → Path         (.consolidation_candidates.json)

Acceptance: byte-identical to pre-refactor memory_consolidation_detector.py
CLI output on the same memory tree (design doc §8.1).

Note: similarity model preserved exactly — content-word Jaccard with the
same stopword list and 3-char minimum. Word-shingles produced 0.01 noise
floor on real prose; content-word overlap gives 0.05–0.25 band for topical
adjacency. Threshold default 0.15 = empirical knee.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_THRESHOLD = 0.15

_FM_SPLIT = re.compile(r"^---\s*$", re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "was", "were", "will", "with", "we",
    "you", "your", "i", "our", "so", "if", "when", "not", "no", "do",
    "does", "did", "been", "being", "which", "what", "where", "why",
    "how", "can", "could", "should", "would", "may", "might", "must",
    "any", "all", "some", "than", "then", "there", "these", "those",
    "they", "them", "their",
}

_MIN_WORD_LEN = 3

# _FX_CHAIN_PREFIX -- What: the filename/chain_id prefix of the fact-extract
#   mini-chains (chains/fx_<source-stem>.json, members are sem_* type:semantic
#   atoms). Why: BUG-2026-06-11 runaway loop (surfacer side) — this surfacer was
#   built for EPISODIC chains, but it now also sweeps the fx_* atom mini-chains,
#   closing a self-feeding loop: atoms -> fx mini-chains -> surfacer ->
#   merge_consumer 'abstract' branch -> fact re-extraction -> more atoms. The
#   contradiction detector already covers atom near-dups; the surfacer's job is
#   EPISODIC consolidation, so it must SKIP fx_* chains.
_FX_CHAIN_PREFIX = "fx_"

# _TYPE_RE -- What: matches a `type: <value>` frontmatter line. Why: the secondary
#   (belt) exclusion check — a chain whose members all resolve to type:semantic is
#   an atom mini-chain regardless of its filename, so it is excluded too.
_TYPE_RE = re.compile(r"^type:\s*(\S+)\s*$", re.MULTILINE)


def _is_semantic_node(memory_dir: Path, rel_file: str) -> bool:
    """True iff the node's frontmatter declares type: semantic.

    What: read the node file head and match its `type:` frontmatter line.
    Why:  BUG-2026-06-11 — the belt check behind the cheap fx_ prefix gate. A
          fact-extract atom is type:semantic; an episodic memory is not. Fail-soft
          to False (treat as episodic / surfaceable) on a missing/unreadable node.
    """
    p = memory_dir / rel_file
    try:
        text = p.read_text()
    except OSError:
        return False
    m = _TYPE_RE.search(text)
    return bool(m) and m.group(1).lower() == "semantic"


def _is_atom_minichain(memory_dir: Path, chain_path: Path) -> bool:
    """True iff a chain is a fact-extract atom mini-chain (exclude from surfacer).

    What: cheap id-prefix check first — chain filename starts with fx_; if not,
          a belt check — EVERY resolvable member node is type:semantic. Either
          condition marks the chain as an atom mini-chain, not an episodic chain.
    Why:  BUG-2026-06-11 runaway loop (surfacer side). The prefix check is O(1)
          and covers the live fx_ chains; the type check catches any atom chain
          that does not carry the prefix. Fail-soft: an unloadable chain is treated
          as episodic (surfaceable) so this never silently drops real memory.
    """
    if chain_path.stem.startswith(_FX_CHAIN_PREFIX):
        return True
    try:
        members = load_chain(chain_path).get("members", [])
    except (OSError, ValueError):
        return False
    resolvable = [m for m in members if (memory_dir / m["file"]).exists()]
    if not resolvable:
        return False
    return all(_is_semantic_node(memory_dir, m["file"]) for m in resolvable)


def strip_frontmatter(text: str) -> str:
    parts = _FM_SPLIT.split(text, maxsplit=2)
    if len(parts) >= 3 and parts[0].strip() == "":
        return parts[2]
    return text


def shingles(text: str) -> set[str]:
    return {
        w for w in (t.lower() for t in _WORD_RE.findall(text))
        if len(w) >= _MIN_WORD_LEN and w not in _STOPWORDS
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_chain(chain_path: Path) -> dict:
    return json.loads(chain_path.read_text())


def load_node_body(memory_dir: Path, rel_file: str) -> Optional[str]:
    p = memory_dir / rel_file
    if not p.exists():
        return None
    return strip_frontmatter(p.read_text())


def audit_chain(chain_path: Path, threshold: float,
                memory_dir: Path) -> list[dict]:
    chain = load_chain(chain_path)
    members = chain.get("members", [])
    if len(members) < 2:
        return []

    bodies: list[tuple[str, str, set]] = []
    for m in members:
        body = load_node_body(memory_dir, m["file"])
        if body is None:
            continue
        bodies.append((m["addr"], m["file"], shingles(body)))

    findings: list[dict] = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a_addr, a_file, a_sh = bodies[i]
            b_addr, b_file, b_sh = bodies[j]
            sim = jaccard(a_sh, b_sh)
            if sim >= threshold:
                findings.append({
                    "chain": chain["chain_id"],
                    "a_addr": a_addr, "a_file": a_file,
                    "b_addr": b_addr, "b_file": b_file,
                    "similarity": round(sim, 3),
                })
    return findings


def audit_all(memory_dir: Path, threshold: float = DEFAULT_THRESHOLD,
              chain: Optional[str] = None) -> list[dict]:
    chains_dir = memory_dir / "chains"
    chain_files = [chains_dir / f"{chain}.json"] if chain \
        else sorted(chains_dir.glob("*.json"))
    all_findings: list[dict] = []
    for cf in chain_files:
        if not cf.exists():
            print(f"skipping: no such chain file {cf}")
            continue
        # BUG-2026-06-11 runaway loop (surfacer side): the surfacer is for
        # EPISODIC chains. Skip the fact-extract atom mini-chains (fx_* / all-
        # type:semantic members) — the contradiction detector covers atom
        # near-dups, and surfacing them closes the atoms -> fx-chains -> surfacer
        # -> merge_abstract -> re-extraction loop. Cheap id-prefix check first.
        if _is_atom_minichain(memory_dir, cf):
            continue
        all_findings.extend(audit_chain(cf, threshold, memory_dir))
    all_findings.sort(key=lambda f: -f["similarity"])
    return all_findings


def surface(memory_dir: Path, findings: list[dict], threshold: float) -> Path:
    surface_file = memory_dir / ".consolidation_candidates.json"
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "candidates": findings,
    }
    surface_file.write_text(json.dumps(payload, indent=2))
    return surface_file


# ─────────────────────────────────────────────
# [consolidation] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.1.0  Updated: 2026-06-11  Status: active
# Role:       find merge candidates in SAM chains — content-word Jaccard over
#             chain-member pairs; surface .consolidation_candidates.json for the
#             Tier-2 merge consumer's drain. EPISODIC chains only.
# Change:     BUG-2026-06-11 runaway loop (surfacer side) — audit_all now SKIPS the
#             fact-extract atom mini-chains (fx_* filename / all-type:semantic
#             members) via _is_atom_minichain. The surfacer was built for episodic
#             chains; sweeping the fx_* atom chains closed a self-feeding loop
#             (atoms -> fx mini-chains -> surfacer -> merge_abstract -> fact
#             re-extraction). The contradiction detector covers atom near-dups.
# Depends:    stdlib only (json, re, datetime, pathlib).
# Note:       similarity model preserved exactly (content-word Jaccard, 0.15 knee).
#             The exclusion is fail-soft: an unloadable chain is treated as episodic
#             (surfaceable), never silently dropping real memory.
# ─────────────────────────────────────────────
