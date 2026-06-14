"""samia.core.semantic_recall — the SEMANTIC recall arm + read-side composer.

FEAT-2026-06-10-memory-semantic-recall-arm-v01, Phase P1.

Layer 1 (Owns / Depends):
    Owns:    A SECOND, peer recall arm dedicated to the SEMANTIC node population
             (type:semantic atoms — direct fact lookup), and a thin COMPOSER that
             joins this arm's facts with the EPISODIC chainogram arm's evidence into
             one labelled context. Owns the two env flags that gate the whole feature
             (ASTHENOS_SEMANTIC_ARM_ENABLED, ASTHENOS_RECALL_FACTS_FRACTION).
    Depends: samia.core.vector (the shared MiniLM index query seam — overfetch then
             type-filter), samia.core.frontmatter (parse atom fm for type/title/body/
             valid_from/source), samia.core.context_extension (the episodic arm —
             chainogram_retrieve — and its BYTES_PER_TOKEN budget estimate). Lazy
             imports of context_extension/vector keep the import graph acyclic
             (context_extension lazily imports THIS module for its fx_-skip gate).

Layer 2 (What / Why):
    What: Atoms (type:semantic, the fact-extraction population) are a SEPARATE memory
          population from the episodic session turns the chainogram serves. This module
          gives them their OWN retrieval path — pure vector top-k filtered to semantic
          nodes, emitted as chronological "KNOWN FACTS" lines — and a composer that puts
          those facts BESIDE (never inside) the chainogram's "CONVERSATION EVIDENCE".
          The two arms meet only in recall(); neither arm is made aware of the other's
          population.
    Why:  Serving facts through the episodic chainogram is a category error (semantic
          recall is direct lookup; episodic recall is contextual reinstatement). The P8
          benchmark autopsy located the deficit as context ECONOMY, not content — the
          reader drowned in mixed turns+atoms. LAYER, don't replace: the chainogram is
          left untouched except for one flag-gated SELECTION skip of atom chains; with
          the flag OFF this whole module is inert and recall() is a byte-identical
          chainogram passthrough.
"""

from __future__ import annotations

import os
from pathlib import Path

# BYTES_PER_TOKEN — What: reuse the tree's existing ~3.6 bytes/token estimate so the
#   facts-block budget truncation matches every other budget the context-extension
#   primitives apply. Why: a divergent heuristic here would let the composed context
#   over/under-run the budget relative to the evidence arm; one estimate keeps the split
#   honest. Imported lazily-by-value at call time would be overkill — it is a constant.
from .context_extension import BYTES_PER_TOKEN

SEMANTIC_ARM_ENABLED_ENV = "ASTHENOS_SEMANTIC_ARM_ENABLED"
RECALL_FACTS_FRACTION_ENV = "ASTHENOS_RECALL_FACTS_FRACTION"

# P2c — entity-bridge atom retrieval (FEAT-2026-06-10 P2). The fraction of k the
#   atom arm reserves for entity-bridge candidates (env ASTHENOS_ATOM_BRIDGE_FRAC,
#   default 0.25 -> 3 of 12). Mirrors context_extension.chainogram_retrieve_bridged's
#   bridge_reserve_frac, but reserves SLOTS (k) not BYTES (budget) — atom_retrieve
#   ranks by relevance, truncation is downstream in format_facts.
ATOM_BRIDGE_FRAC_ENV = "ASTHENOS_ATOM_BRIDGE_FRAC"
_DEFAULT_ATOM_BRIDGE_FRAC = 0.25

# P2b — read-conflict supersession kill-switch (env ASTHENOS_READ_CONFLICT_ENABLED,
#   default "1" ON). Independent of the arm flag: the scan ALSO only runs on the
#   flag-ON recall path, but this env lets the conflict scan be killed without
#   disabling the whole arm.
READ_CONFLICT_ENABLED_ENV = "ASTHENOS_READ_CONFLICT_ENABLED"

# Default split: 25% of the budget to facts, 75% to evidence (Q2a). Clamped to a sane
# band so a misconfigured env can never starve the evidence arm entirely.
_DEFAULT_FACTS_FRACTION = 0.25
_FACTS_FRACTION_MIN = 0.0
_FACTS_FRACTION_MAX = 0.9


def semantic_arm_enabled() -> bool:
    """True iff the semantic recall arm + composer are enabled (live env read, default OFF).

    What: reads ASTHENOS_SEMANTIC_ARM_ENABLED each call; "1" => ON, anything else => OFF.
    Why: default OFF means recall() is a byte-identical chainogram passthrough and the
      chainogram's fx_-skip gate never even resolves a type — the unflagged path is
      indistinguishable from today's behavior. Read-each-call (not import-time) so a
      test/daemon/adapter that sets the env after import sees the change, mirroring the
      contradiction.is_enabled() / integrity.repair_enabled() reader pattern.
    """
    return os.environ.get(SEMANTIC_ARM_ENABLED_ENV, "0") == "1"


def facts_fraction() -> float:
    """The fraction of the recall budget allotted to FACTS (live env read, default 0.25).

    What: reads ASTHENOS_RECALL_FACTS_FRACTION as a float, clamped to [0.0, 0.9].
      Unparseable/missing => the 0.25 default.
    Why: the facts/evidence split is the one knob this feature exposes (b1500 lesson:
      one variable at a time, env-tunable not hardcoded). The clamp guarantees the
      evidence arm always keeps at least 10% of the budget no matter the env value.
    """
    raw = os.environ.get(RECALL_FACTS_FRACTION_ENV)
    if raw is None:
        return _DEFAULT_FACTS_FRACTION
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_FACTS_FRACTION
    return max(_FACTS_FRACTION_MIN, min(_FACTS_FRACTION_MAX, val))


def atom_bridge_frac() -> float:
    """Fraction of atom_retrieve's k reserved for entity-bridge candidates (P2c).

    What: env ASTHENOS_ATOM_BRIDGE_FRAC as a float, clamped to [0.0, 0.9].
      Unparseable/missing => the 0.25 default (3 of 12). 0.0 disables the bridge
      reserve entirely (pure-vector atom arm, today's behavior).
    Why: multihop entity-B atoms sit at rank 91-187 in the mixed index — a single
      vector query never reaches them. The reserve guarantees a few slots for
      entity-bridge candidates so they enter the FACTS slice, while the rest of k
      stays pure vector relevance. Clamp guards a misconfigured env from starving
      the vector slots. Mirrors facts_fraction's read-each-call + clamp shape.
    """
    raw = os.environ.get(ATOM_BRIDGE_FRAC_ENV)
    if raw is None:
        return _DEFAULT_ATOM_BRIDGE_FRAC
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_ATOM_BRIDGE_FRAC
    return max(0.0, min(0.9, val))


def read_conflict_enabled() -> bool:
    """True iff the read-conflict supersession scan runs (P2b kill-switch, default ON).

    What: env ASTHENOS_READ_CONFLICT_ENABLED; anything other than "0" => ON (default
      "1"). Read each call.
    Why: a narrow kill-switch for the read-path conflict scan that is INDEPENDENT of
      the arm flag — the scan already only runs on the flag-ON recall path, but an
      operator can silence just the conflict recording (e.g. during a store audit)
      without flipping the whole semantic arm off. "0" is the only OFF token so the
      default-and-typo case stays ON (fail-toward-recording, the F5a dedup guard
      bounds the store).
    """
    return os.environ.get(READ_CONFLICT_ENABLED_ENV, "1") != "0"


# _node_type cache — What: per-(memory_dir, stem) cache of a node's resolved `type`,
#   mirroring contradiction._node_type's cache so atom_retrieve's overfetch filter
#   resolves each candidate's type at most once per process.
# Why: vector overfetch surfaces ~8*k candidates; reading each one's frontmatter once
#   and caching keeps the type filter cheap across repeated recall() calls in a run.
_TYPE_CACHE: dict[tuple[str, str], str | None] = {}


def _clear_type_cache() -> None:
    """Drop the node-type cache (tests / after a node's type changes)."""
    _TYPE_CACHE.clear()


def _node_type(memory_dir: Path, node_id: str) -> str | None:
    """Resolve a node's `type` frontmatter field (cached, lowercased), or None.

    What: read nodes/<stem>.md frontmatter, return its `type` lowercased; None when the
      node is missing/unreadable/typeless. Cached per (memory_dir, stem).
    Why: the atom-population filter — atom_retrieve keeps a hit only when its type is
      "semantic". Mirrors contradiction._node_type so the resolution semantics (lazy
      frontmatter.parse, broad except, cache shape) are identical across the tree.
    """
    stem = node_id[:-3] if node_id.endswith(".md") else node_id
    key = (str(memory_dir), stem)
    if key in _TYPE_CACHE:
        return _TYPE_CACHE[key]
    p = memory_dir / "nodes" / f"{stem}.md"
    val: str | None = None
    if p.exists():
        try:
            from . import frontmatter as _fm
            parsed, _ = _fm.parse(p.read_text(encoding="utf-8"))
            if parsed is not None:
                t = parsed[0].get("type")
                if isinstance(t, str) and t.strip():
                    val = t.strip().lower()
        except Exception:
            val = None
    _TYPE_CACHE[key] = val
    return val


def _atom_fields(memory_dir: Path, node_id: str) -> dict | None:
    """Read an atom node's display fields, or None if unreadable.

    What: returns {title, body, valid_from, source} for a semantic node. title falls
      back to the stem; body is the node body stripped; valid_from/source default "".
    Why: the FACTS line needs exactly these four fields; reading once here keeps
      format_facts a pure formatter over already-resolved dicts.
    """
    stem = node_id[:-3] if node_id.endswith(".md") else node_id
    p = memory_dir / "nodes" / f"{stem}.md"
    if not p.exists():
        return None
    try:
        from . import frontmatter as _fm
        parsed, body = _fm.parse(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    fm = parsed[0] if parsed is not None else {}
    title = fm.get("name") or fm.get("title") or stem
    return {
        "title": str(title).strip(),
        "body": (body or "").strip(),
        "valid_from": str(fm.get("valid_from") or "").strip(),
        "source": str(fm.get("source") or "").strip(),
    }


def _atom_entry(memory_dir: Path, node: str, score: float) -> dict | None:
    """Build a served-atom dict for a node, or None if it is not a readable atom.

    What: returns {node, title, body, valid_from, source, score} for a node whose
      resolved type is "semantic" and whose fields are readable; None otherwise.
    Why: both the vector-relevance fill and the entity-bridge fill (P2c) need the
      SAME atom-entry shape with the SAME population gate — factoring it keeps the two
      fill loops honest (one type check, one field read) and prevents a bridge node
      that is not actually a semantic atom from leaking into the FACTS slice.
    """
    if not node:
        return None
    if _node_type(memory_dir, node) != "semantic":
        return None
    fields = _atom_fields(memory_dir, node)
    if fields is None:
        return None
    return {
        "node": node,
        "title": fields["title"],
        "body": fields["body"],
        "valid_from": fields["valid_from"],
        "source": fields["source"],
        "score": float(score),
    }


def atom_retrieve(memory_dir: Path, query: str, k: int = 12,
                  budget_tokens: int | None = None) -> list[dict]:
    """Vector top-k over ONLY type:semantic nodes, with an entity-bridge reserve.

    What: queries the shared vector index with overfetch (~8*k), keeps only hits whose
      resolved node type is "semantic". A fraction of k (atom_bridge_frac(), default
      0.25 -> 3 of 12) is RESERVED for entity-bridge atoms (P2c): the vector relevance
      fill is capped at k - reserve, then the reserved slots are filled from
      entity_index.query_bridges hits (type:semantic only, NOT already vector-picked);
      any unused bridge slots fall back to more vector relevance so k is never
      under-filled when bridges are thin. Each returned dict carries
      {node, title, body, valid_from, source, score}. When the index/embedder is
      unavailable (no index / SystemExit / any backend error) returns []; when the
      entity index is absent/unbuilt the bridge reserve is simply skipped (fail-open
      to a pure-vector top-k, today's behavior).
    Why: atoms share the one MiniLM index with the episodic turns, so a pure top-k would
      mix the populations; the type filter restores the population boundary at read time
      WITHOUT a second index. The entity-bridge reserve attacks multihop from the
      retrieval side — entity-B atoms that sit at rank 91-187 in the mixed index are
      unreachable by a single vector query but are reachable by an entity match against
      the bridge index. budget_tokens is accepted for signature symmetry with the
      composer but truncation is applied in format_facts; it is unused here.
    """
    del budget_tokens  # truncation happens in format_facts over the joined block
    try:
        from . import vector as _vi
        hits = _vi.query(memory_dir, query, top_k=max(k, 8 * k))
    except SystemExit:
        return []
    except Exception:
        return []

    reserve = int(k * atom_bridge_frac())
    vector_cap = max(0, k - reserve)

    out: list[dict] = []
    seen: set[str] = set()
    # Vector relevance fill, capped at k - reserve so bridge slots stay open.
    for h in hits:
        if len(out) >= vector_cap:
            break
        node = h.get("node")
        entry = _atom_entry(memory_dir, node, float(h.get("score", 0.0)))
        if entry is None:
            continue
        out.append(entry)
        seen.add(node)

    # Entity-bridge fill (P2c): pull bridge atoms NOT already vector-picked into the
    # reserved slots. Fail-open — absent/unbuilt entity index or any bridge error ->
    # the reserve is just left to the vector backfill below (today's behavior).
    if reserve > 0:
        try:
            from . import entity_index as _ei
            bridges = _ei.query_bridges(memory_dir, query)
        except Exception:
            bridges = {"error": "entity index unavailable"}
        if "error" not in bridges:
            for b in bridges.get("bridge_nodes") or []:
                if len(out) >= k:
                    break
                node = b.get("node")
                if not node or node in seen:
                    continue
                entry = _atom_entry(memory_dir, node, float(b.get("score", 0.0)))
                if entry is None:
                    continue  # bridge node is not a semantic atom -> skip cleanly
                out.append(entry)
                seen.add(node)

    # Backfill any slots the bridge reserve did not consume with more vector
    # relevance, so a thin bridge set never under-fills k.
    if len(out) < k:
        for h in hits:
            if len(out) >= k:
                break
            node = h.get("node")
            if not node or node in seen:
                continue
            entry = _atom_entry(memory_dir, node, float(h.get("score", 0.0)))
            if entry is None:
                continue
            out.append(entry)
            seen.add(node)

    return out


def _format_fact_line(atom: dict) -> str:
    """Render one atom as a FACTS line, omitting empty parens/brackets gracefully.

    Shape: "- {title}: {body} ({valid_from}) [from {source}]" — the (valid_from) group
    is dropped when valid_from is empty, the [from source] group when source is empty.
    """
    line = f"- {atom['title']}: {atom['body']}"
    vf = atom.get("valid_from") or ""
    if vf:
        line += f" ({vf})"
    src = atom.get("source") or ""
    if src:
        line += f" [from {src}]"
    return line


def format_facts(atoms: list[dict], budget_tokens: int | None = None) -> str:
    """Format atoms as chronological FACTS lines, truncated to a token budget.

    What: orders atoms CHRONOLOGICALLY by valid_from (ascending); undated atoms (empty
      valid_from) sink to the END, preserving their incoming relevance order. Renders
      each via _format_fact_line, joins with newlines, then drops trailing lines until
      the joined block fits budget_tokens (~BYTES_PER_TOKEN bytes/token, the tree's
      estimate). budget_tokens=None => no truncation.
    Why: facts are most useful in time order for temporal questions, but the relevance
      ranking is the right tiebreak for undated atoms (no date to sort on). Truncating by
      DROPPING whole trailing lines (not mid-line) keeps every served fact intact.
    """
    if not atoms:
        return ""

    # Stable chronological sort: dated atoms ascending by valid_from; undated last.
    # enumerate index is the secondary key so undated atoms keep their relevance order
    # and dated ties break by relevance order too (sort is stable on equal vf).
    def _key(item: tuple[int, dict]) -> tuple[int, str, int]:
        idx, a = item
        vf = a.get("valid_from") or ""
        # undated -> bucket 1 (after all dated bucket-0 atoms); vf "" sorts harmlessly.
        return (1 if not vf else 0, vf, idx)

    ordered = [a for _, a in sorted(enumerate(atoms), key=_key)]
    lines = [_format_fact_line(a) for a in ordered]

    if budget_tokens is None:
        return "\n".join(lines)

    budget_bytes = max(0, int(budget_tokens * BYTES_PER_TOKEN))
    kept: list[str] = []
    spent = 0
    for ln in lines:
        # +1 for the joining newline between kept lines (none before the first).
        add = len(ln) + (1 if kept else 0)
        if spent + add > budget_bytes:
            break
        kept.append(ln)
        spent += add
    return "\n".join(kept)


def _assemble_evidence_text(memory_dir: Path, chain_out: dict) -> tuple[str, list[str]]:
    """Assemble evidence TEXT + dia_ids from a chainogram_retrieve result.

    What: mirrors samia_adapter.retrieve's assembly — sorted loaded-node names ->
      read each body -> strip frontmatter -> join with newlines. While scanning the
      frontmatter, collects every `dia:` value into dia_ids.
    Why: the composer needs the SAME evidence text the existing adapter produces (so the
      evidence arm is byte-identical to today's chainogram path) plus the dia_ids the
      benchmark scores recall on. Atoms carry no dia, so dia_ids are evidence-only.
    """
    nodes_dir = memory_dir / "nodes"
    names = sorted(n["node"] for n in chain_out.get("loaded_nodes", []) or [])
    ctx_lines: list[str] = []
    dia_ids: list[str] = []
    for name in names:
        p = nodes_dir / name
        if not p.exists():
            continue
        raw = p.read_text(encoding="utf-8")
        body = raw
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                for line in raw[3:end].splitlines():
                    if line.startswith("dia:"):
                        dia_ids.append(line.split(":", 1)[1].strip())
                body = raw[end + 4:].lstrip()
        ctx_lines.append(body.rstrip())
    return "\n".join(ctx_lines), dia_ids


def focus_k() -> int:
    """Per-chain member cap for the evidence focuser (0 disables).

    What: env ASTHENOS_RECALL_FOCUS_K, default 8.
    Why: TUNE-2026-06-11 — the K=8/cap=2400 pair is the benchmark-validated
      operating point (0.45 overall recall quality at a measured 1,944 ctx
      tokens, still comfortably under the serving target). K was 6; raising it
      to 8 recovers per-chain depth without overrunning the cap. The focuser
      keeps each selected chain's top-K most query-relevant members so breadth
      (chains) is preserved while depth (members) is bounded. 0 = serve whole
      chains (pre-focuser shape).
    """
    try:
        return max(0, int(os.environ.get("ASTHENOS_RECALL_FOCUS_K", "8")))
    except ValueError:
        return 8


def evidence_cap() -> int:
    """Token cap for the FOCUSED evidence slice (env ASTHENOS_RECALL_EVIDENCE_CAP,
    default 2400 — the benchmark-validated cap: paired with focus_k=8 it measured
    0.45 overall recall quality at 1,944 ctx tokens, under the serving target.
    Raised from 1800 (TUNE-2026-06-11)."""
    try:
        return max(200, int(os.environ.get("ASTHENOS_RECALL_EVIDENCE_CAP", "2400")))
    except ValueError:
        return 2400


def _focus_evidence(memory_dir: Path, chain_out: dict, query: str) -> list[dict]:
    """Select the evidence entries to SERVE: per-chain top-K by query relevance.

    What: groups chain_out['loaded_nodes'] by their chain, ranks members within
      each chain by PER-NODE cosine vs the query (vector.query map; nodes
      outside the top hits score 0.0), keeps the top focus_k() per chain
      (always >= 1, so every selected chain keeps its best member —
      neighborhood breadth survives), then trims globally to evidence_cap()
      tokens in chain order. Selection only — assembly order stays the
      caller's concern.
    Why: the 'focuser' layer (option-1 from the 2026-06-10 layering design,
      applied composer-side only): the b1500 run proved BUDGET cuts starve
      (whole-chain skipping); the focuser cuts member depth instead, keeping
      the needle's neighborhood without the haystack row. Chainogram itself
      is untouched — this filters its OUTPUT.
    """
    entries = list(chain_out.get("loaded_nodes", []) or [])
    k = focus_k()
    if k == 0 or not entries:
        return entries
    # Per-node relevance map (fail-soft: empty map -> chain-order fallback).
    rel: dict[str, float] = {}
    try:
        from . import vector as _v
        for h in _v.query(memory_dir, query, top_k=256):
            rel[h["node"]] = float(h["score"])
    except (Exception, SystemExit):
        pass
    by_chain: dict = {}
    order: list = []
    for e in entries:
        cname = e.get("chain") or f"_single_{e['node']}"
        if cname not in by_chain:
            by_chain[cname] = []
            order.append(cname)
        by_chain[cname].append(e)
    kept: list[dict] = []
    spent = 0
    cap = evidence_cap()
    for cname in order:
        members = sorted(by_chain[cname],
                         key=lambda e: rel.get(e["node"], 0.0), reverse=True)[:k]
        for i, e in enumerate(members):
            tok = int(e.get("tokens") or 0)
            # i == 0: each selected chain's BEST member always serves (breadth
            # guarantee, soft-exceeds the cap by at most one member per chain);
            # deeper members only while under the cap.
            if i == 0 or spent + tok <= cap:
                kept.append(e)
                spent += tok
    return kept


def _load_index_rows(memory_dir: Path) -> tuple | None:
    """Load (embeddings, fname->row map) from the shared vector index, or None.

    What: reads <memory_dir>/vector_index/embeddings.npy + manifest.json and builds
      {fname(.md) -> row} over non-tombstoned entries. Returns (embeddings, by_name)
      or None when numpy/the index is unavailable. Lifts the contradiction._load_index
      by_row pattern but keys by FILENAME (the served node-ids are filenames) instead
      of by row, since the read-conflict scan looks rows up by served node.
    Why: P2b needs cosine between ALREADY-served nodes — no new embedding calls. The
      index embeddings are L2-normalized (samia.core.vector), so a row dot product is
      the cosine directly. Keyed by filename so a served node id maps straight to its
      embedding row with no second lookup table.
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return None
    index_dir = memory_dir / "vector_index"
    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"
    if not emb_path.exists() or not manifest_path.exists():
        return None
    try:
        import json as _json
        import numpy as np
        embeddings = np.load(str(emb_path))
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", {})
        by_name = {
            rel: e["row"]
            for rel, e in entries.items()
            if isinstance(e, dict) and e.get("row") is not None
            and not e.get("tombstoned")
        }
        return embeddings, by_name
    except Exception:
        return None


def _read_conflict_scan(memory_dir: Path, atoms: list[dict],
                        evidence_nodes: list[dict]) -> int:
    """Record read-conflict supersession candidates among served nodes (P2b).

    What: a cheap conflict scan over the nodes this recall() is about to serve —
      every served-atom x served-evidence pair AND every served-atom x served-atom
      pair. Cosine is read from the EXISTING index embeddings (lookup by manifest row,
      no new embedding calls); pairs scoring >= contradiction._SEMANTIC_PAIR_THRESHOLD
      (0.92) are recorded via contradiction.record_supersession_candidate(...,
      mode='read_conflict', status='candidate') — RECORD-ONLY. Returns the count
      recorded (for tests/telemetry). The newer-dated node is the new_id when both
      carry valid_from; otherwise the second-served node is treated as new_id.
    Why: when the read path serves a fact and an evidence turn (or two facts) that are
      near-duplicates, that is a supersession signal the write path may have missed.
      Recording it (the REM judge confirms later — NO judge calls, NO auto-supersede in
      the read path) lets the offline pipeline reconcile it. Fail-open and cheap: sets
      are <=20 atoms x <=15 evidence; pure numpy dot on already-loaded rows; skip
      entirely when the index is unavailable. The F5a dedup guard in
      record_supersession_candidate prevents store spam from repeated recalls.
    """
    if not read_conflict_enabled():
        return 0
    # Cap the working sets per spec (cheap-guarantee): <=20 atoms x <=15 evidence.
    a_nodes = [a["node"] for a in atoms[:20] if a.get("node")]
    e_nodes = [e["node"] for e in evidence_nodes[:15] if e.get("node")]
    if not a_nodes:
        return 0
    loaded = _load_index_rows(memory_dir)
    if loaded is None:
        return 0
    embeddings, by_name = loaded

    def _row(node_id: str):
        rel = node_id if node_id.endswith(".md") else f"{node_id}.md"
        r = by_name.get(rel)
        if r is None:
            return None
        try:
            return embeddings[int(r)]
        except Exception:
            return None

    # valid_from lookup (atoms carry it; evidence turns do not -> None ordering).
    vf_by_node = {a["node"]: (a.get("valid_from") or "") for a in atoms[:20]}

    def _order(n1: str, n2: str) -> tuple[str, str]:
        # new_id = the newer-dated node when both dated; else n2 (later-served) is new.
        d1, d2 = vf_by_node.get(n1, ""), vf_by_node.get(n2, "")
        if d1 and d2:
            return (n1, n2) if d1 <= d2 else (n2, n1)
        return (n1, n2)

    try:
        import numpy as np
        from ..runtime import contradiction as _con
    except Exception:
        return 0

    bar = _con._SEMANTIC_PAIR_THRESHOLD
    recorded = 0

    def _maybe_record(n1: str, n2: str) -> None:
        nonlocal recorded
        v1, v2 = _row(n1), _row(n2)
        if v1 is None or v2 is None:
            return
        cos = float(np.dot(v1, v2))
        if cos < bar:
            return
        old_id, new_id = _order(n1, n2)
        try:
            _con.record_supersession_candidate(
                memory_dir, old_id, new_id, cos, mode="read_conflict")
            recorded += 1
        except Exception:
            return

    # atom x evidence pairs.
    for an in a_nodes:
        for en in e_nodes:
            _maybe_record(an, en)
    # atom x atom pairs (upper triangle, no self).
    for i in range(len(a_nodes)):
        for j in range(i + 1, len(a_nodes)):
            _maybe_record(a_nodes[i], a_nodes[j])

    return recorded


def recall(memory_dir: Path, query: str, budget_tokens: int = 8000) -> dict:
    """Compose the read-side context: KNOWN FACTS (atom arm) + CONVERSATION EVIDENCE.

    What: when semantic_arm_enabled() is FALSE -> a pure chainogram passthrough: run
      context_extension.chainogram_retrieve over the FULL budget and assemble the
      evidence text exactly as the adapter does, with NO FACTS section and NO atom calls
      (byte-identical to a chainogram-only assembly). When TRUE -> split the budget
      (facts_budget = int(budget * facts_fraction()); evidence_budget = budget -
      facts_budget), retrieve up-to-12 atoms, retrieve evidence under the evidence
      budget, and join them as:
          "KNOWN FACTS:\\n{facts}\\n\\nCONVERSATION EVIDENCE:\\n{evidence}"
      The FACTS section is omitted ENTIRELY when 0 atoms are served (evidence still
      flows). Returns {context, facts_n, evidence_nodes, dia_ids}; dia_ids are evidence
      dia ids only (atoms carry none).
    Why: this is the composer — the only place the two populations meet. The flag-off
      passthrough is the LAYER-don't-replace guarantee (today's behavior, byte-for-byte);
      the flag-on split is the context-economy fix (bounded facts BESIDE bounded
      evidence, not mixed inside one arm).
    """
    from . import context_extension as _cx

    if not semantic_arm_enabled():
        chain_out = _cx.chainogram_retrieve(
            memory_dir, query, budget_tokens=budget_tokens, max_chains=8)
        if "error" in chain_out:
            return {"context": "", "facts_n": 0, "evidence_nodes": 0,
                    "dia_ids": []}
        evidence_text, dia_ids = _assemble_evidence_text(memory_dir, chain_out)
        return {
            "context": evidence_text,
            "facts_n": 0,
            "evidence_nodes": len(chain_out.get("loaded_nodes", []) or []),
            "dia_ids": dia_ids,
        }

    facts_budget = int(budget_tokens * facts_fraction())
    evidence_budget = budget_tokens - facts_budget

    atoms = atom_retrieve(memory_dir, query, k=12, budget_tokens=facts_budget)
    facts_block = format_facts(atoms, budget_tokens=facts_budget)

    chain_out = _cx.chainogram_retrieve(
        memory_dir, query, budget_tokens=evidence_budget, max_chains=8)
    if "error" in chain_out:
        evidence_text, dia_ids, evidence_nodes, kept = "", [], 0, []
    else:
        # FOCUSER (TUNE-2026-06-10): serve only each chain's top-K most
        # query-relevant members, capped — selection happens here, then the
        # standard assembly runs over the kept subset (same sorted order).
        kept = _focus_evidence(memory_dir, chain_out, query)
        evidence_text, dia_ids = _assemble_evidence_text(
            memory_dir, {"loaded_nodes": kept})
        evidence_nodes = len(kept)

    if facts_block:
        context = ("KNOWN FACTS:\n" + facts_block
                   + "\n\nCONVERSATION EVIDENCE:\n" + evidence_text)
    else:
        # Zero atoms (or all truncated away): omit the FACTS section entirely so the
        # reader sees only labelled evidence, never an empty "KNOWN FACTS:" header.
        context = "CONVERSATION EVIDENCE:\n" + evidence_text

    # P2b — read-conflict supersession signal. AFTER composing, scan the SERVED atoms
    # against the SERVED evidence nodes (`kept`, empty when the chainogram errored) and
    # atoms among themselves for near-dups via the existing index embeddings;
    # record-only. Fail-open: any error leaves the composed result untouched.
    try:
        _read_conflict_scan(memory_dir, atoms, kept)
    except Exception:
        pass  # read-conflict signal is best-effort; never perturbs recall output

    return {
        "context": context,
        "facts_n": len(atoms),
        "evidence_nodes": evidence_nodes,
        "dia_ids": dia_ids,
    }


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.semantic_recall
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-10-memory-semantic-recall-arm-v01 (P1 + P2b/P2c)
# Layer:      core (pure library, no daemon dependency)
# Role:       the SEMANTIC recall arm + read-side composer -- a peer vector top-k over
#             type:semantic atoms (with an entity-bridge reserve) served as KNOWN FACTS
#             BESIDE the episodic chainogram's CONVERSATION EVIDENCE, plus a record-only
#             read-conflict supersession scan; flag-OFF is a byte-identical chainogram
#             passthrough.
# Stability:  v1.2 -- semantic arm + composer, flag-gated (default OFF)
#             + TUNE-2026-06-11 focuser defaults: focus_k 6->8, evidence_cap
#               1800->2400 (benchmark-validated K=8/cap2400 = 0.45 overall @ 1,944
#               ctx tokens). Both still env-overridable.
#             + P2c entity-bridge atom reserve (atom_bridge_frac, default 0.25 ->
#               3 of 12; fail-open to pure vector when the entity index is absent).
#             + P2b read-conflict supersession scan in recall() flag-ON path
#               (record-only via contradiction.record_supersession_candidate,
#               mode='read_conflict'; cosine from existing index rows, no new
#               embedding calls; gated on ASTHENOS_READ_CONFLICT_ENABLED default ON;
#               fail-open).
# ErrorModel: fail-soft throughout — atom_retrieve returns [] when the index/embedder
#             is unavailable and fails open to pure-vector when the entity index is
#             absent; recall() returns evidence-only (or empty context) when the
#             chainogram errors; the read-conflict scan never perturbs the recall
#             result; flag OFF is a byte-identical chainogram passthrough.
# Depends:    samia.core.vector (query, lazy), samia.core.frontmatter (parse, lazy),
#             samia.core.context_extension (chainogram_retrieve lazy; BYTES_PER_TOKEN),
#             samia.core.entity_index (query_bridges, lazy — P2c),
#             samia.runtime.contradiction (record_supersession_candidate +
#             _SEMANTIC_PAIR_THRESHOLD, lazy — P2b).
# Exposes:    semantic_arm_enabled, facts_fraction, atom_bridge_frac,
#             read_conflict_enabled, atom_retrieve, format_facts, recall.
#             Constants: SEMANTIC_ARM_ENABLED_ENV, RECALL_FACTS_FRACTION_ENV,
#             ATOM_BRIDGE_FRAC_ENV, READ_CONFLICT_ENABLED_ENV.
# ACTIVATION: semantic_arm_enabled() (default OFF) gates the WHOLE feature — the composer
#             split, the chainogram fx_-skip, AND the read-conflict scan. OFF =>
#             recall() == chainogram passthrough and context_extension never resolves
#             an atom-chain type. The entity-bridge reserve and read-conflict scan only
#             run on the flag-ON path.
# Lines:      743
# --------------------------------------------------------------------------
