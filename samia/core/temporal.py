"""samia.core.temporal — bi-temporal (valid-time) query over SAM/IA memory nodes.

Layer 1 (Owns / Depends):
    Owns:    migrate, query, show, set_valid — the four operations a CLI/MCP
                 wrapper drives (backfill valid_from, time-window recall, inspect,
                 edit a node's validity interval).
             infer_valid_from — the three-tier valid_from derivation
                 (last_access → earliest body date → file mtime).
             interval_contains, interval_overlaps — the point/range membership
                 predicates the query filters on.
             parse_date, iso, split_frontmatter, fm_get, fm_set, read_node,
                 write_node, load_node_event_time — frontmatter/date primitives.
    Depends: stdlib only (datetime, re, pathlib). samia.core.vector.query is
             imported LAZILY inside query() so the time-window scan has no hard
             dependency on a built vector index.
Layer 2 (What / Why):
    What: a node carries a valid_from/valid_to date interval in its frontmatter.
          query() scans nodes/ and keeps those whose interval matches an `at`
          point, a `since` lower bound, and/or a `range_pair` window, optionally
          pre-filtering the candidate set by semantic recall. migrate() backfills
          valid_from on nodes that lack it; show()/set_valid() inspect and edit it.
    Why:  recall must answer "what was true as of date D", not just "what mentions
          D". Validity is an interval (a fact holds from valid_from until valid_to
          / open-ended), so membership is interval logic, not string match. The
          library plane is parameterized on memory_dir and returns plain dicts so
          the CLI wrapper is argparse + print only and the MCP server reuses the
          identical logic — one query semantics, two front ends.

Layer 3 (Changelog):
    (carved from memory_temporal_query.py — library plane extracted from the
     original CLI script; no behavior change in the carve.)
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


# parse_date — What: coerce a frontmatter date string (or sentinel) to a date|None.
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
# parse_date — Why: valid_to is legitimately open-ended, written as "null"/"~", so
#     those map to None (an open interval), not an error; s[:10] tolerates datetime
#     strings, and a malformed date fails soft to None rather than raising mid-scan.


def iso(d: _dt.date | None) -> str | None:
    return d.isoformat() if d else None


# split_frontmatter — What: split "---\n<fm>\n---\n<body>" into (fm_lines, body).
def split_frontmatter(text: str) -> tuple[list[str], str]:
    if not text.startswith("---"):
        return [], text
    end = text.find("\n---", 3)
    if end == -1:
        return [], text
    fm = text[3:end].lstrip("\n").splitlines()
    body = text[end + 4:].lstrip("\n")
    return fm, body
# split_frontmatter — Why: a node with no/unterminated frontmatter is not an error
#     here — it returns ([], whole-text) so callers treat it as a bodied node with no
#     validity fields, rather than dropping it from the scan.


def fm_get(fm_lines: list[str], key: str) -> str | None:
    pref = f"{key}:"
    for ln in fm_lines:
        if ln.startswith(pref):
            return ln[len(pref):].strip()
    return None


# fm_set — What: set key to value in fm_lines (replace in place, or append if absent).
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
# fm_set — Why: replace-in-place preserves the surrounding frontmatter line order
#     (callers rewrite only valid_from/valid_to), and the append-on-absent branch is
#     what lets migrate() add valid_from to a node that never had it.


def write_node(path: Path, fm_lines: list[str], body: str) -> None:
    fm_text = "\n".join(fm_lines)
    out = f"---\n{fm_text}\n---\n{body}"
    path.write_text(out, encoding="utf-8")


def read_node(path: Path) -> tuple[list[str], str]:
    return split_frontmatter(path.read_text(encoding="utf-8"))


# infer_valid_from — What: derive a node's valid_from date via a three-tier fallback
#     — last_access frontmatter, else the earliest YYYY-MM-DD found in the body, else
#     the file's mtime — always returning a concrete date (never None).
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
# infer_valid_from — Why: migration must stamp SOME valid_from on every legacy node, so
#     the tiers descend from most-trustworthy (an explicit access date) to a guaranteed
#     floor (mtime). Earliest body date — not latest — approximates when the fact first
#     became true. mtime is the weakest proxy (filesystem write, not event time) and is
#     the honest ceiling of what's recoverable from a node that records no date.


# migrate — What: backfill valid_from (and an open valid_to="null") on every node in
#     nodes/ that lacks valid_from; return counts + up to 5 examples. dry_run computes
#     the same plan without writing.
def migrate(memory_dir: Path, dry_run: bool = False) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    written = 0
    skipped = 0
    examples: list[str] = []
    for p in sorted(nodes_dir.glob("*.md")):
        fm_lines, body = read_node(p)
        if not fm_lines:
            continue
        # Idempotency: a node that already has valid_from is left byte-identical, so
        # re-running migrate only ever touches the not-yet-stamped remainder.
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
# migrate — Why: bi-temporal recall is only meaningful once every node carries an
#     interval, so this one-time pass seeds the corpus. The skip-if-present guard makes
#     it idempotent (safe to re-run), and dry_run lets an operator preview the inferred
#     dates — which lean on the weakest infer_valid_from tier — before committing them.


def load_node_event_time(p: Path):
    fm_lines, body = read_node(p)
    name = fm_get(fm_lines, "name") or p.stem
    vf = parse_date(fm_get(fm_lines, "valid_from"))
    vt = parse_date(fm_get(fm_lines, "valid_to"))
    return vf, vt, name, body


# interval_contains — What: is point `at` inside the node's [vf, vt] interval?
def interval_contains(at: _dt.date, vf, vt) -> bool:
    if vf and at < vf:
        return False
    if vt and at > vt:
        return False
    return True
# interval_contains — Why: a falsy vf/vt is an OPEN end (unknown start / still valid),
#     so each bound only constrains when present — an all-open node contains every date.


# interval_overlaps — What: does [a, b] intersect the node's [vf, vt] interval?
def interval_overlaps(a: _dt.date, b: _dt.date, vf, vt) -> bool:
    lo = vf or _dt.date.min
    hi = vt or _dt.date.max
    return not (hi < a or lo > b)
# interval_overlaps — Why: open ends are widened to date.min/date.max so the standard
#     "not (disjoint-left or disjoint-right)" test handles open intervals without a
#     separate branch per None case.


# query — What: scan candidate nodes and return up to top_k whose validity interval
#     satisfies the supplied filters (`at` point-contains, `since` lower-bound,
#     `range_pair` overlap), optionally pre-narrowing candidates by semantic recall.
def query(memory_dir: Path, at, since, range_pair,
          semantic: str | None, top_k: int) -> list[dict]:
    nodes_dir = _nodes_dir(memory_dir)
    semantic_note: str | None = None
    # SemanticPrefilter — What: when a semantic query is given, narrow candidates to the
    #     vector index's top hits (over-fetched 4x) instead of the whole nodes/ glob.
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
    # SemanticPrefilter — Why: the lazy import keeps the time-window scan free of any
    #     hard vector-index dependency; over-fetching 4x leaves headroom for the temporal
    #     filters below to discard hits and still reach top_k. SystemExit is caught (not
    #     a plain Exception) because that is vector.query's specific "no index" signal,
    #     and an MCP/library caller must degrade to a non-semantic scan, never exit.

    # IntervalFilter — What: apply each supplied filter to a node's [vf, vt] interval;
    #     keep stops as soon as one filter rejects, and the loop halts at top_k matches.
    out: list[dict] = []
    for p in candidates:
        vf, vt, name, _ = load_node_event_time(p)
        keep = True
        if at is not None:
            keep = keep and interval_contains(at, vf, vt)
        if since is not None:
            # SinceFilter — What: keep a node iff its validity interval reaches `since`
            #     or later — valid_to open (None) or valid_to >= since.
            keep = keep and ((vt is None) or vt >= since)
            # SinceFilter — Why: `since` is the documented "still valid at or after A"
            #     filter (the one-sided sibling of range_pair's overlap), so it constrains
            #     the interval's END, not its start. The prior trailing predicate
            #     `and ((vf is None) or vf >= since or vt is None or vt >= since)` was
            #     vacuously true given this clause (BUG-2026-06-14-05) and is removed —
            #     zero behavior change, it never filtered.
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
    # IntervalFilter — Why: filters compose with `and` so multiple time constraints
    #     intersect; an absent (None) filter is simply skipped, and short-circuiting at
    #     top_k bounds the scan cost on a large corpus.
    if semantic_note is not None:
        # Fail-soft diagnostic: semantic recall was requested but no index
        # exists, so these results are the non-semantic time scan. Surfaced as
        # a trailing diagnostic entry so callers can detect the degradation
        # without the query raising / exiting.
        out.append({"note": semantic_note})
    return out


# show — What: return one node's temporal frontmatter fields (validity + tier) as a dict.
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


# set_valid — What: overwrite a node's valid_from and/or valid_to (each set only when
#     its argument is non-None), preserving the body.
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
# set_valid — Why: None means "leave this bound unchanged" (not "clear it"), so an
#     operator can correct just valid_to without disturbing valid_from; the literal
#     string is written through unparsed so sentinels like "null" round-trip exactly.


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.temporal
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Carved from memory_temporal_query.py (library plane extraction).
# Layer:      core (pure library, no daemon dependency)
# Role:       bi-temporal (valid-time) query over memory nodes -- interval-membership
#             recall by point/since/range over each node's valid_from/valid_to (optional
#             semantic prefilter), plus the migrate/show/set_valid operations and the
#             frontmatter/date primitives, parameterized on memory_dir for shared CLI+MCP
#             use.
# Stability:  stable -- bi-temporal query primitives; API parameterized on memory_dir.
# ErrorModel: query fails SOFT on a missing vector index (catches vector.query's
#             SystemExit, degrades to a non-semantic scan, appends a {"note": ...}
#             diagnostic); parse_date / split_frontmatter degrade malformed input to
#             None / empty-frontmatter rather than raising; show and set_valid raise
#             FileNotFoundError on an absent node.
# Depends:    datetime, re, pathlib (stdlib).
#             samia.core.vector.query (imported LAZILY in query(), semantic path only).
# Exposes:    migrate, query, show, set_valid, infer_valid_from,
#             interval_contains, interval_overlaps, parse_date, iso,
#             split_frontmatter, fm_get, fm_set, read_node, write_node,
#             load_node_event_time.
# Lines:      343
# --------------------------------------------------------------------------
