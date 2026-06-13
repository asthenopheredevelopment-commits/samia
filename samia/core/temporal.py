"""samia.core.temporal — bi-temporal query for SAM/IA memory.

Carved from memory_temporal_query.py. Library plane parameterized on
memory_dir; CLI wrapper does argparse + print only.

Public API:
  migrate(memory_dir, dry_run=False) → dict
  query(memory_dir, at, since, range_pair, semantic, top_k) → list[dict]
  show(memory_dir, node_name) → dict
  set_valid(memory_dir, node_name, vf, vt) → None
  parse_date, iso, split_frontmatter, fm_get, fm_set,
  read_node, write_node, infer_valid_from,
  load_node_event_time, interval_contains, interval_overlaps
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def parse_date(s: str | None) -> _dt.date | None:
    if not s:
        return None
    s = s.strip().strip('"\'')
    if s.lower() in ("null", "none", "~"):
        return None
    try:
        return _dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def iso(d: _dt.date | None) -> str | None:
    return d.isoformat() if d else None


def split_frontmatter(text: str) -> tuple[list[str], str]:
    if not text.startswith("---"):
        return [], text
    end = text.find("\n---", 3)
    if end == -1:
        return [], text
    fm = text[3:end].lstrip("\n").splitlines()
    body = text[end + 4:].lstrip("\n")
    return fm, body


def fm_get(fm_lines: list[str], key: str) -> str | None:
    pref = f"{key}:"
    for ln in fm_lines:
        if ln.startswith(pref):
            return ln[len(pref):].strip()
    return None


def fm_set(fm_lines: list[str], key: str, value: str) -> list[str]:
    pref = f"{key}:"
    out = []
    found = False
    for ln in fm_lines:
        if ln.startswith(pref):
            out.append(f"{key}: {value}")
            found = True
        else:
            out.append(ln)
    if not found:
        out.append(f"{key}: {value}")
    return out


def write_node(path: Path, fm_lines: list[str], body: str) -> None:
    fm_text = "\n".join(fm_lines)
    out = f"---\n{fm_text}\n---\n{body}"
    path.write_text(out, encoding="utf-8")


def read_node(path: Path) -> tuple[list[str], str]:
    return split_frontmatter(path.read_text(encoding="utf-8"))


def infer_valid_from(path: Path, fm_lines: list[str], body: str) -> _dt.date:
    la = parse_date(fm_get(fm_lines, "last_access"))
    if la:
        return la
    earliest: _dt.date | None = None
    for m in DATE_RE.finditer(body):
        try:
            d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if earliest is None or d < earliest:
                earliest = d
        except ValueError:
            pass
    if earliest:
        return earliest
    return _dt.date.fromtimestamp(path.stat().st_mtime)


def migrate(memory_dir: Path, dry_run: bool = False) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    written = 0
    skipped = 0
    examples: list[str] = []
    for p in sorted(nodes_dir.glob("*.md")):
        fm_lines, body = read_node(p)
        if not fm_lines:
            continue
        if fm_get(fm_lines, "valid_from"):
            skipped += 1
            continue
        d = infer_valid_from(p, fm_lines, body)
        new_fm = fm_set(fm_lines, "valid_from", iso(d))
        if fm_get(new_fm, "valid_to") is None:
            new_fm = fm_set(new_fm, "valid_to", "null")
        if not dry_run:
            write_node(p, new_fm, body)
        written += 1
        if len(examples) < 5:
            examples.append(
                f"  {p.name}: valid_from={iso(d)} valid_to=null")
    return {"written": written, "skipped": skipped, "examples": examples}


def load_node_event_time(p: Path):
    fm_lines, body = read_node(p)
    name = fm_get(fm_lines, "name") or p.stem
    vf = parse_date(fm_get(fm_lines, "valid_from"))
    vt = parse_date(fm_get(fm_lines, "valid_to"))
    return vf, vt, name, body


def interval_contains(at: _dt.date, vf, vt) -> bool:
    if vf and at < vf:
        return False
    if vt and at > vt:
        return False
    return True


def interval_overlaps(a: _dt.date, b: _dt.date, vf, vt) -> bool:
    lo = vf or _dt.date.min
    hi = vt or _dt.date.max
    return not (hi < a or lo > b)


def query(memory_dir: Path, at, since, range_pair,
          semantic: str | None, top_k: int) -> list[dict]:
    nodes_dir = _nodes_dir(memory_dir)
    semantic_note: str | None = None
    if semantic:
        # In-package semantic recall via the core vector index. vector.query
        # signature is query(memory_dir, text, top_k=...) and returns dicts with
        # a "node" filename, so candidates map straight onto nodes_dir.
        from .vector import query as vec_query
        try:
            hits = vec_query(memory_dir, semantic, top_k=max(top_k * 4, 16))
            candidates = [nodes_dir / h["node"] for h in hits]
        except SystemExit as exc:
            # vector.query raises SystemExit("no index found") when no index has
            # been built. A public MCP caller must never get a process exit, so
            # fail soft to the non-semantic scan and record why semantic was
            # skipped instead of killing the host process.
            semantic_note = f"semantic recall unavailable: {exc}"
            candidates = sorted(nodes_dir.glob("*.md"))
    else:
        candidates = sorted(nodes_dir.glob("*.md"))

    out: list[dict] = []
    for p in candidates:
        vf, vt, name, _ = load_node_event_time(p)
        keep = True
        if at is not None:
            keep = keep and interval_contains(at, vf, vt)
        if since is not None:
            keep = keep and ((vt is None) or vt >= since) and \
                ((vf is None) or vf >= since or vt is None or vt >= since)
        if range_pair is not None:
            keep = keep and interval_overlaps(
                range_pair[0], range_pair[1], vf, vt)
        if keep:
            out.append({
                "node": p.name,
                "title": name,
                "valid_from": iso(vf),
                "valid_to": iso(vt),
            })
            if len(out) >= top_k:
                break
    if semantic_note is not None:
        # Fail-soft diagnostic: semantic recall was requested but no index
        # exists, so these results are the non-semantic time scan. Surfaced as
        # a trailing diagnostic entry so callers can detect the degradation
        # without the query raising / exiting.
        out.append({"note": semantic_note})
    return out


def show(memory_dir: Path, node_name: str) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    p = nodes_dir / node_name
    if not p.suffix:
        p = p.with_suffix(".md")
    if not p.exists():
        raise FileNotFoundError(f"node not found: {p}")
    fm_lines, _ = read_node(p)
    return {
        "node": p.name,
        "name": fm_get(fm_lines, "name"),
        "valid_from": fm_get(fm_lines, "valid_from"),
        "valid_to": fm_get(fm_lines, "valid_to"),
        "last_access": fm_get(fm_lines, "last_access"),
        "tier": fm_get(fm_lines, "tier"),
    }


def set_valid(memory_dir: Path, node_name: str,
              vf: str | None, vt: str | None) -> None:
    nodes_dir = _nodes_dir(memory_dir)
    p = nodes_dir / node_name
    if not p.suffix:
        p = p.with_suffix(".md")
    if not p.exists():
        raise FileNotFoundError(f"node not found: {p}")
    fm_lines, body = read_node(p)
    if vf is not None:
        fm_lines = fm_set(fm_lines, "valid_from", vf)
    if vt is not None:
        fm_lines = fm_set(fm_lines, "valid_to", vt)
    write_node(p, fm_lines, body)
