"""samia.core.audit — read-only health check for SAM/IA memory.

Carved from memory_audit.py. The library plane carries all logic so the
daemon (per design doc §4.x) can call audit(memory_dir) on a schedule
without spawning a subprocess.

Reports:
  * Node counts by tier and type
  * Chain health (members with missing files, malformed manifests)
  * Infrastructure presence (pool/, archive/, tools/)
  * Runtime tooling inventory with one-line descriptions
  * MEMORY.md index drift (broken links)

No writes. Safe for every-tick invocation.

Public API (parameterized on memory_dir):
  audit(memory_dir) → dict
  format_text(report) → str

Acceptance: byte-identical to pre-refactor memory_audit.py CLI behavior
on the same memory tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import frontmatter as _fm


def _nodes(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _chains(memory_dir: Path) -> Path:
    return memory_dir / "chains"


def _pool(memory_dir: Path) -> Path:
    return memory_dir / "pool"


def _archive(memory_dir: Path) -> Path:
    return memory_dir / "archive"


def _tools(memory_dir: Path) -> Path:
    return memory_dir / "tools"


def _index(memory_dir: Path) -> Path:
    return memory_dir / "MEMORY.md"


def parse_frontmatter(text: str) -> dict:
    parsed, _ = _fm.parse(text)
    if parsed is None:
        return {}
    return parsed[0]


def tool_docstring(py_path: Path) -> str:
    """Extract the first line of a module's docstring."""
    try:
        text = py_path.read_text()
    except Exception:
        return ""
    m = re.search(r'"""(.+?)"""', text, re.DOTALL)
    if not m:
        return ""
    first = m.group(1).strip().split("\n", 1)[0].strip()
    return first


def audit(memory_dir: Path) -> dict:
    nodes_dir = _nodes(memory_dir)
    chains_dir = _chains(memory_dir)
    pool_dir = _pool(memory_dir)
    archive_dir = _archive(memory_dir)
    tools_dir = _tools(memory_dir)
    index_path = _index(memory_dir)

    report: dict = {"ok": True, "issues": []}

    # ── Nodes by tier + type ───────────────────────────────────────────
    tiers = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0, "unknown": 0}
    types: dict[str, int] = {}
    orphan_tier: list[str] = []
    node_count = 0
    if nodes_dir.exists():
        for md in sorted(nodes_dir.glob("*.md")):
            node_count += 1
            fm = parse_frontmatter(md.read_text())
            tier = fm.get("tier", "unknown").strip()
            tiers[tier] = tiers.get(tier, 0) + 1
            if tier == "unknown":
                orphan_tier.append(md.name)
            t = fm.get("type", "untyped").strip()
            types[t] = types.get(t, 0) + 1
    report["nodes"] = {"count": node_count, "by_tier": tiers, "by_type": types}
    if orphan_tier:
        report["issues"].append(
            {"level": "warn", "kind": "orphan_tier", "count": len(orphan_tier),
             "detail": orphan_tier[:5]})

    # ── Chains + members exist ─────────────────────────────────────────
    chain_count = 0
    missing_members: list[str] = []
    if chains_dir.exists():
        for cj in sorted(chains_dir.glob("*.json")):
            chain_count += 1
            try:
                manifest = json.loads(cj.read_text())
            except Exception as e:
                report["issues"].append(
                    {"level": "error", "kind": "bad_chain_json",
                     "file": cj.name, "detail": str(e)})
                continue
            for m in manifest.get("members", []):
                path = memory_dir / m.get("file", "")
                if not path.exists():
                    missing_members.append(f"{cj.stem}:{m.get('file')}")
    report["chains"] = {"count": chain_count,
                        "missing_members": missing_members}
    if missing_members:
        report["issues"].append(
            {"level": "warn", "kind": "missing_chain_member",
             "count": len(missing_members),
             "detail": missing_members[:5]})

    # ── Infrastructure ─────────────────────────────────────────────────
    report["infrastructure"] = {
        "pool_exists": pool_dir.exists(),
        "archive_exists": archive_dir.exists(),
        "tools_exists": tools_dir.exists(),
    }

    # ── Runtime tools inventory ───────────────────────────────────────
    tools: list[dict] = []
    if tools_dir.exists():
        for py in sorted(tools_dir.glob("*.py")):
            tools.append({"name": py.name,
                          "doc": tool_docstring(py) or "(no docstring)"})
    report["tools"] = tools

    # ── MEMORY.md drift ────────────────────────────────────────────────
    md_missing: list[str] = []
    if index_path.exists():
        txt = index_path.read_text()
        for m in re.finditer(r"\]\(([^)]+\.(?:md|json))\)", txt):
            ref = m.group(1)
            candidate = memory_dir / ref
            if not candidate.exists():
                md_missing.append(ref)
    report["index_drift"] = md_missing
    if md_missing:
        report["issues"].append(
            {"level": "warn", "kind": "memory_md_broken_link",
             "count": len(md_missing), "detail": md_missing[:5]})

    if any(i["level"] == "error" for i in report["issues"]):
        report["ok"] = False
    return report


def format_text(r: dict) -> str:
    out = []
    out.append("── memory audit ──")
    n = r["nodes"]
    tiers = ", ".join(f"{k}={v}" for k, v in n["by_tier"].items() if v)
    out.append(f"nodes: {n['count']} ({tiers})")
    types = ", ".join(f"{k}={v}" for k, v in n["by_type"].items())
    out.append(f"types: {types}")
    out.append(f"chains: {r['chains']['count']}")
    infra = r["infrastructure"]
    out.append(f"infra: pool={'✓' if infra['pool_exists'] else '✗'} "
               f"archive={'✓' if infra['archive_exists'] else '✗'} "
               f"tools={'✓' if infra['tools_exists'] else '✗'}")
    if r["tools"]:
        out.append("runtime tools:")
        for t in r["tools"]:
            out.append(f"  {t['name']:<32} {t['doc'][:80]}")
    if r["issues"]:
        out.append("issues:")
        for i in r["issues"]:
            out.append(f"  [{i['level']}] {i['kind']}: {i.get('count', '')} "
                       f"{str(i.get('detail',''))[:100]}")
    else:
        out.append("issues: none")
    out.append(f"status: {'OK' if r['ok'] else 'ERRORS'}")
    return "\n".join(out)
