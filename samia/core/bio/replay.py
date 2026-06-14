"""samia.core.bio.replay — hippocampal replay sweeps + the online active-set.

Layer 1 (Owns / Depends):
    Owns:    the replay family (Wilson & McNaughton 1994). _recently_accessed_nodes
             (the hot/recent working set); the ONLINE active-set seam (_fast_engram_
             neighbors stub + active_set, the bounded supersession locus, FEAT-2026-06-07
             P3b); replay_sweep (cross-chain edge proposals from semantic neighbors of
             recent nodes); the SWR-interleaved variant (_all_chain_node_names /
             _cold_chains / _embedding_for_node / replay_sweep_interleaved); and the
             engram feed-forward with GENUINE-ONCE (_pair_key / _load_engram_replay_state /
             _save_engram_replay_state / replay_engram_traces, FEAT-2026-06-07 Tier-1 P5).
    Depends: config (constants REPLAY_* / INTERLEAVE_* / ACTIVE_SET_HOT_N / np / json /
             _dt / os / sys / Path / _bio_paths); samia.core.bio.hebbian (_addr_for_node
             — chain address resolver — via plain import); samia.core.{vector, temporal,
             web_store, hippocampus} + samia.runtime.contradiction (lazy, function-local
             — the index/query, node read, web neighbors, engram store, scope predicate).

Layer 2 (What / Why):
    What: the offline replay engines (REM-gated in production) + the bounded online
          locus the write path consults. They all sample recent nodes and propose or
          feed-forward associations.
    Why:  carved out of the monolith as the replay responsibility. vector / temporal /
          web_store / hippocampus / contradiction are lazy (function-local) exactly as
          the monolith had them — keeps `import bio` cheap and breaks the bio<->hippocampus
          and bio<->contradiction cycles.

PATCH SEAM (the genuine-once feed):
    replay_engram_traces feeds pairs into the Tier-0 log by calling hebbian_record, which
    is a mock.patch.object(bio, ...) spy target. It reaches hebbian_record THROUGH the
    package facade (`from . import __init__ as _pkg` is not used; instead `from samia.core
    import bio as _pkg; _pkg.hebbian_record(...)`) so a facade-level patch rebinds what the
    feed actually calls. The string-patch target bio.active_set has no internal caller, so
    only the re-export is needed for it.
"""

from __future__ import annotations

from typing import Optional

from . import config as _cfg
from .config import (
    np,
    json,
    _dt,
    os,
    sys,
    Path,
    REPLAY_DEFAULT_SAMPLE,
    REPLAY_NEIGHBOR_THRESHOLD,
    INTERLEAVE_THRESHOLD,
    INTERLEAVE_DEFAULT_COLD_PER_HOT,
    ACTIVE_SET_HOT_N,
    _bio_paths,
)
from .hebbian import _addr_for_node


# ---------------------------------------------------------------------------
# 4. Replay sweep — hot/recent working set
# ---------------------------------------------------------------------------


def _recently_accessed_nodes(memory_dir: Path, top_n: int) -> list[str]:
    from samia.core import temporal as _tq
    nodes_dir = memory_dir / "nodes"
    rows: list[tuple[_dt.date, str]] = []
    for p in nodes_dir.glob("*.md"):
        fm_lines, _ = _tq.read_node(p)
        la = _tq.parse_date(_tq.fm_get(fm_lines, "last_access"))
        if not la:
            continue
        rows.append((la, p.name))
    rows.sort(reverse=True)
    return [name for _, name in rows[:top_n]]


# ---------------------------------------------------------------------------
# FEAT-2026-06-07 P3b — the ONLINE active-set (bounded supersession locus)
# ---------------------------------------------------------------------------


def _fast_engram_neighbors(memory_dir: Path, write_nodes: list[str]) -> list[str]:
    """FEAT-2026-06-07 P3b — pluggable Tier-1 fast-engram seam (P3d hook).

    What: returns the recently-encoded fast-engram neighbors of the write nodes.
          Returns [] today (no Tier-1 fast store exists yet).
    Why:  Q1a — the active-set is PLUGGABLE so the ONLINE detector auto-extends to
          Tier-1 fast engrams once they land, with NO re-sequence of P3a-c. The
          contract (a list of neighbor node ids) is fixed up front; P3d fills only
          this body. Exercised as a no-op seam by the P3b tests.
    """
    return []


def active_set(memory_dir: Path, write_nodes: list[str],
               db_dir: Optional[str] = None,
               hot_n: int = ACTIVE_SET_HOT_N) -> list[str]:
    """FEAT-2026-06-07 P3b — the bounded ONLINE supersession locus for a write.

    What: union of (a) co-activation neighbors of each write node (Tier-0
          edges.db, via web_store.coactivation_neighbors — live + clean post-P2),
          (b) the hot/recently-accessed nodes (_recently_accessed_nodes), and
          (c) the pluggable Tier-1 fast-engram neighbors (empty today). The write
          nodes themselves are excluded. Returns a de-duplicated list of node ids
          (with the .md suffix).
    Why:  Q1a — "what fires together with the new write + what's in working
          memory" is the locus where a contradiction matters immediately. Bounding
          the detector to this set (degree-capped neighbors + a small hot top-N)
          keeps the write-path cheap and is the cheap immediate half of the
          locality split (the passive REM sweep is the exhaustive global half).
    """
    # PATCH SEAM — reach the helper sources THROUGH the package facade so a test's
    # mock.patch.object(bio, "_recently_accessed_nodes", ...) / (bio,
    # "_fast_engram_neighbors", ...) rebinds the functions active_set actually unions
    # (test_contradiction_tuning stubs the locus sources this way). A direct intra-module
    # call would bind the unpatched originals.
    from samia.core import bio as _pkg
    wanted = {n if n.endswith(".md") else f"{n}.md" for n in write_nodes}
    locus: set[str] = set()
    # (a) co-activation neighbors per write node (Tier-0 web).
    try:
        from samia.core import web_store as _ws
        for n in write_nodes:
            for nb in _ws.coactivation_neighbors(n, db_dir=db_dir):
                nb_m = nb if nb.endswith(".md") else f"{nb}.md"
                if nb_m not in wanted:
                    locus.add(nb_m)
    except Exception as e:  # fail-soft: no web → no neighbors, locus still useful.
        print(f"[active_set] coactivation lookup failed: {e}", file=sys.stderr)
    # (b) hot / recently-accessed working set.
    try:
        for nm in _pkg._recently_accessed_nodes(memory_dir, hot_n):
            nm_m = nm if nm.endswith(".md") else f"{nm}.md"
            if nm_m not in wanted:
                locus.add(nm_m)
    except Exception as e:
        print(f"[active_set] hot/recent lookup failed: {e}", file=sys.stderr)
    # (c) pluggable Tier-1 fast engrams (empty today; filled in P3d).
    for nb in _pkg._fast_engram_neighbors(memory_dir, write_nodes):
        nb_m = nb if nb.endswith(".md") else f"{nb}.md"
        if nb_m not in wanted:
            locus.add(nb_m)
    # TYPE-SCOPING (TUNE-2026-06-08): drop episodic/experiential nodes
    # (session_offload / bug) from the ONLINE supersession locus. They are not
    # contradictable content claims, so the detector must never consider them as
    # online candidates -- the same experiential-vs-content rule the passive sweep
    # and the finder apply. Lazy import keeps bio import-light and avoids a cycle
    # (contradiction imports bio for the salience guard). Fail-soft: if the
    # predicate is unavailable, the locus is returned unfiltered (no behavior
    # change), since the finder applies the same scope downstream anyway.
    try:
        from samia.runtime import contradiction as _con
        _is_excluded = getattr(_con, "is_excluded_node", None)
        if _is_excluded is not None:
            locus = {n for n in locus if not _is_excluded(memory_dir, n)}
    except Exception:
        pass
    return sorted(locus)


def replay_sweep(memory_dir: Path,
                 sample: int = REPLAY_DEFAULT_SAMPLE,
                 threshold: float = REPLAY_NEIGHBOR_THRESHOLD) -> dict:
    """Pick recently-accessed nodes, find semantic neighbors, propose cross-chain edges."""
    from samia.core import vector as _vec
    paths = _bio_paths(memory_dir)
    nodes_dir = memory_dir / "nodes"
    if not _vec._manifest_path(memory_dir).exists():
        return {"error": "no vector index — run memory_vector_index.py build"}
    recents = _recently_accessed_nodes(memory_dir, sample)
    if not recents:
        return {"events": 0, "proposals": 0}

    proposals: list[dict] = []
    for name in recents:
        path = nodes_dir / name
        if not path.exists():
            continue
        title, content = _vec._load_node_text(path)
        hits = _vec.query(memory_dir, content[:1500], top_k=6)
        hits = [h for h in hits if h["node"] != name]
        own = _addr_for_node(memory_dir, name)
        if not own:
            continue
        own_chain, own_addr = own
        for h in hits:
            if h["score"] < threshold:
                continue
            other = _addr_for_node(memory_dir, h["node"])
            if not other:
                continue
            other_chain, other_addr = other
            if own_chain == other_chain:
                continue
            proposals.append({
                "from_node": name, "to_node": h["node"],
                "from_chain": own_chain, "to_chain": other_chain,
                "score": float(h["score"]),
            })

    by_pair: dict[tuple[str, str], dict] = {}
    for p in proposals:
        key = (p["from_chain"], p["to_chain"])
        if key not in by_pair or p["score"] > by_pair[key]["score"]:
            by_pair[key] = p

    out = {"ts": _dt.datetime.now().isoformat(timespec="seconds"),
           "sample_size": len(recents),
           "raw_pairs": len(proposals),
           "unique_chain_pairs": len(by_pair),
           "proposals": list(by_pair.values())}
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    paths["replay_proposals"].write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 4b. SWR-style interleaved replay
# ---------------------------------------------------------------------------


def _all_chain_node_names(memory_dir: Path) -> dict[str, list[str]]:
    chains_dir = memory_dir / "chains"
    out: dict[str, list[str]] = {}
    for cp in chains_dir.glob("*.json"):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        members = []
        for m in data.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f:
                continue
            members.append(Path(f).name)
        if members:
            out[cp.stem] = members
    return out


def _cold_chains(hot_nodes: set[str], chain_members: dict[str, list[str]]
                 ) -> list[str]:
    out: list[str] = []
    for cname, names in chain_members.items():
        if not any(n in hot_nodes for n in names):
            out.append(cname)
    return out


def _embedding_for_node(name: str, manifest: dict, embeddings) -> Optional["np.ndarray"]:
    e = manifest.get("entries", {}).get(name)
    if not e:
        return None
    row = e.get("row")
    if row is None or row >= embeddings.shape[0]:
        return None
    return embeddings[row]


def replay_sweep_interleaved(
        memory_dir: Path,
        sample: int = REPLAY_DEFAULT_SAMPLE,
        cold_per_hot: int = INTERLEAVE_DEFAULT_COLD_PER_HOT,
        threshold: float = INTERLEAVE_THRESHOLD,
        seed: Optional[int] = None) -> dict:
    """SWR-style interleaved replay."""
    from samia.core import vector as _vec
    paths = _bio_paths(memory_dir)
    if not _vec._manifest_path(memory_dir).exists():
        return {"error": "no vector index — run memory_vector_index.py build"}

    recents = _recently_accessed_nodes(memory_dir, sample)
    if not recents:
        return {"hot_count": 0, "proposals": []}

    chain_members = _all_chain_node_names(memory_dir)
    hot_set = set(recents)
    cold_chain_list = _cold_chains(hot_set, chain_members)
    if not cold_chain_list:
        return {"hot_count": len(recents), "cold_chains": 0, "proposals": []}

    rng = np.random.default_rng(seed)
    manifest = json.loads(_vec._manifest_path(memory_dir).read_text(encoding="utf-8"))
    embeddings = np.load(_vec._embed_path(memory_dir))

    standard_pairs: set[tuple[str, str]] = set()
    if paths["replay_proposals"].exists():
        try:
            std = json.loads(paths["replay_proposals"].read_text(encoding="utf-8"))
            for p in std.get("proposals", []):
                a = p.get("from_chain"); b = p.get("to_chain")
                if a and b:
                    standard_pairs.add((a, b))
                    standard_pairs.add((b, a))
        except Exception:
            pass

    proposals: list[dict] = []
    skipped_unindexed = 0
    for hot_name in recents:
        hot_emb = _embedding_for_node(hot_name, manifest, embeddings)
        if hot_emb is None:
            skipped_unindexed += 1
            continue
        hot_addr = _addr_for_node(memory_dir, hot_name)
        if hot_addr:
            hot_chain, _ = hot_addr
        else:
            hot_chain = f"_singleton:{Path(hot_name).stem}"

        if len(cold_chain_list) <= cold_per_hot:
            picks = cold_chain_list
        else:
            picks = list(rng.choice(cold_chain_list,
                                    size=cold_per_hot, replace=False))

        for cold_chain in picks:
            members = chain_members[cold_chain]
            if not members:
                continue
            cold_name = members[int(rng.integers(0, len(members)))]
            cold_emb = _embedding_for_node(cold_name, manifest, embeddings)
            if cold_emb is None:
                continue
            denom = (np.linalg.norm(hot_emb) * np.linalg.norm(cold_emb))
            if denom <= 0:
                continue
            cos = float(np.dot(hot_emb, cold_emb) / denom)
            if cos < threshold:
                continue
            novel = (hot_chain, cold_chain) not in standard_pairs
            proposals.append({
                "hot_node": hot_name,
                "cold_node": cold_name,
                "hot_chain": hot_chain,
                "cold_chain": cold_chain,
                "score": cos,
                "novel": novel,
            })

    proposals.sort(key=lambda p: (-p["score"], p["hot_node"], p["cold_node"]))

    by_pair: dict[tuple[str, str], dict] = {}
    for p in proposals:
        key = (p["hot_chain"], p["cold_chain"])
        if key not in by_pair or p["score"] > by_pair[key]["score"]:
            by_pair[key] = p

    out = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "hot_count": len(recents),
        "cold_chain_count": len(cold_chain_list),
        "skipped_unindexed": skipped_unindexed,
        "raw_pairs": len(proposals),
        "unique_chain_pairs": len(by_pair),
        "novel_chain_pair_count": sum(1 for p in by_pair.values() if p["novel"]),
        "threshold": threshold,
        "proposals": list(by_pair.values()),
    }
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    paths["replay_interleaved_proposals"].write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 4c. Engram replay with GENUINE-ONCE feed-forward (FEAT-2026-06-07 Tier-1 P5)
# ---------------------------------------------------------------------------
#
# This is the feed-forward amplifier the Tier-0 SOAK note (usage-bounded genuine
# signal) and the Tier-1 audit (raw_pairs:0 — replay had nothing recent to replay)
# both asked for: replay the captured ENGRAM held copies (real recent episodes)
# into the Tier-0 Hebbian web, so CAPTURED episodes — not only live searches —
# drive cortical learning (Q6a all-RAG-feeds + the engram-replay arm).
#
# GENUINE-ONCE (Q6a, the homeostasis keystone): the FIRST consolidation of an
# engram-derived pair is GENUINE (full weight, refreshes the decay clock, bumps
# count_genuine — the real "recently genuine memory" the hippocampus consolidates);
# EVERY subsequent replay of the SAME pair is FRACTIONAL (source="replay" =
# HEBB_REPLAY_COACT_WEIGHT, decay-transparent), and is RATE-LIMITED to at most ONCE
# PER DAY per pair so a single REM cycle's repeated firings cannot farm the edge.
# The per-pair ledger ({first_genuine, last_replay}) is the "already-genuine-
# replayed" memory. Net effect: a single captured trace replayed MANY times gets
# AT MOST ONE genuine event (engram replay grants exactly ONE genuine per pair) —
# and promotion needs HEBB_PROMOTE_REPEATS GENUINE events — so replay ALONE can
# never farm a trace into an attractor. Genuine RAG recall (memory_search) is what
# carries a genuinely-recurring pair the rest of the way to the bar; replay only
# seeds the one genuine + day-over-day reinforces a still-recent pair (then ages).


def _pair_key(a: str, b: str) -> str:
    x, y = sorted([a, b])
    return f"{x}::{y}"


def _load_engram_replay_state(memory_dir: Path) -> dict:
    fp = _bio_paths(memory_dir)["engram_replay_state"]
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_engram_replay_state(memory_dir: Path, d: dict) -> None:
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    fp = paths["engram_replay_state"]
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    os.replace(tmp, fp)


def replay_engram_traces(memory_dir: Path,
                         sample: int = REPLAY_DEFAULT_SAMPLE,
                         threshold: float = REPLAY_NEIGHBOR_THRESHOLD) -> dict:
    """Replay captured ENGRAM held copies into Tier-0 co-activations (genuine-once).

    What: sample the most-recently-accessed engram held copies (the hippocampal
      recent-episode buffer — the real "recently genuine memory", NOT the flat
      nodes/ pool replay_sweep samples), find each copy's semantic neighbors via
      the engram fast index, and feed each discovered PAIR into the Tier-0
      co-activation log with GENUINE-ONCE semantics:
        - the FIRST time a given pair is replayed it is logged GENUINE
          (hebbian_record source="genuine" — full weight, refreshes last_seen,
          bumps count_genuine) and the pair is recorded in the per-pair ledger;
        - a subsequent replay of the SAME pair is logged FRACTIONAL
          (source="replay" = HEBB_REPLAY_COACT_WEIGHT), RATE-LIMITED to at most
          ONCE PER DAY per pair (so a single REM cycle's repeated firings cannot
          farm the edge), and then AGES under the ordinary daily decay/prune.
      Reuses the EXISTING bio genuine-once/fractional machinery
      (hebbian_record + _apply_coactivation + _decay_and_prune + the
      count_genuine promotion gate) — it adds NO new weight path, only the
      per-pair genuine ledger that makes "first genuine, rest fractional" true.
    Why: D5/Q6a — this is the feed-forward that finally drives cortical learning
      from CAPTURED episodes (fixing both the usage-bounded genuine signal and
      raw_pairs:0). Genuine-once is the homeostasis guard: a single captured
      trace replayed MANY times grants AT MOST ONE genuine event per pair, and
      promotion needs HEBB_PROMOTE_REPEATS genuine events — so replay alone can
      neither manufacture nor immortalize an attractor (the same envelope
      replay_sweep's source="replay" feed already lives in, now extended to the
      engram buffer with the one-genuine seed).

    INERT by default: this is wired into the REM-gated offline replay path
    (context_extension.idle_replay_tick), which refuses outside REM. Calling it
    directly (tests) runs it; it never mutates a main node and never writes a
    Tier-0 edge by itself (it appends to the co-activation LOG, which
    hebbian_consolidate later drains — the same path replay_sweep uses).

    Returns {sampled, raw_pairs, genuine, fractional, skipped_same_day, pairs:[...]}.
    """
    from samia.core import hippocampus as _hip
    # PATCH SEAM — reach hebbian_record THROUGH the package facade so a test's
    # mock.patch.object(bio, "hebbian_record", ...) (a documented spy target) rebinds
    # the function this feed actually calls. A direct .hebbian import would bind the
    # unpatched original. (Lazy here also keeps the facade off the submodule import path.)
    from samia.core import bio as _pkg

    store = _hip.EngramStore(memory_dir)
    manifest = store._load_manifest()
    embeddings = store._load_embeddings()
    entries = manifest.get("entries", {})
    if embeddings is None or not entries:
        return {"sampled": 0, "raw_pairs": 0, "genuine": 0, "fractional": 0,
                "pairs": []}

    # Sample the most-recently-accessed engram copies (the recent-episode buffer).
    rows = sorted(
        ((e.get("last_access", ""), eid, e) for eid, e in entries.items()
         if e.get("row") is not None),
        reverse=True)[:max(1, int(sample))]

    by_row = {e["row"]: (eid, e) for eid, e in entries.items()
              if e.get("row") is not None}

    state = _load_engram_replay_state(memory_dir)
    ledger = state.get("genuine_pairs", {})

    pairs: dict[str, dict] = {}
    for _la, eid, entry in rows:
        row = entry.get("row")
        if row is None or row >= embeddings.shape[0]:
            continue
        src = entry.get("source_ptr")
        if not src:
            continue
        q = embeddings[row]
        sims = embeddings @ q
        # Top neighbors of this engram copy (excluding itself), above threshold.
        order = np.argsort(sims)[::-1]
        for r in order:
            if r == row:
                continue
            if float(sims[r]) < threshold:
                break  # sorted desc — nothing further clears the bar
            info = by_row.get(int(r))
            if info is None:
                continue
            nb_src = info[1].get("source_ptr")
            if not nb_src or nb_src == src:
                continue
            key = _pair_key(src, nb_src)
            if key not in pairs:
                pairs[key] = {"a": src, "b": nb_src,
                              "score": float(sims[r])}

    genuine = 0
    fractional = 0
    today_iso = _dt.date.today().isoformat()
    emitted: list[dict] = []
    skipped_same_day = 0
    for key, p in pairs.items():
        first_time = key not in ledger
        if first_time:
            # First replay of this pair, EVER: GENUINE (full weight, +count_genuine).
            try:
                _pkg.hebbian_record(memory_dir, [p["a"], p["b"]],
                                    query="engram_replay", source="genuine")
            except Exception:
                continue
            genuine += 1
            ledger[key] = {"first_genuine": today_iso, "last_replay": today_iso}
            emitted.append({"a": p["a"], "b": p["b"],
                            "score": p["score"], "source": "genuine"})
            continue
        # Re-replay of an already-genuine pair: FRACTIONAL, but rate-limited to AT
        # MOST ONCE PER DAY per pair. The daily limit + the Tier-0 weight ceiling are
        # the two homeostasis backstops. The within-cycle daily limit stops a single
        # REM cycle from firing the same pair N times in one day; the Tier-0
        # REPLAY_ONLY_W_CEILING (in _apply_coactivation) holds w below the bar across
        # ALL days until HEBB_PROMOTE_REPEATS=3 GENUINE events accrue. Engram replay
        # grants AT MOST ONE genuine per pair, so even unbounded day-over-day
        # fractional replay leaves a once-seeded pair capped sub-bar (the multi-day
        # leak the Tier-1 P5 verifier found is closed at Tier-0). One fractional event
        # per day reinforces a genuinely-recent pair while it ages; only 3 genuine RAG
        # recalls lift the ceiling and carry it to the attractor bar (Q6a/D5).
        if ledger[key].get("last_replay") == today_iso:
            skipped_same_day += 1
            continue
        try:
            _pkg.hebbian_record(memory_dir, [p["a"], p["b"]],
                                query="engram_replay", source="replay")
        except Exception:
            continue
        fractional += 1
        ledger[key]["last_replay"] = today_iso
        emitted.append({"a": p["a"], "b": p["b"],
                        "score": p["score"], "source": "replay"})

    state["genuine_pairs"] = ledger
    state["last_run_iso"] = _dt.datetime.now().isoformat(timespec="seconds")
    _save_engram_replay_state(memory_dir, state)

    return {
        "sampled": len(rows),
        "raw_pairs": len(pairs),
        "genuine": genuine,
        "fractional": fractional,
        "skipped_same_day": skipped_same_day,
        "pairs": emitted,
    }


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.replay
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.bio monolith during
#             modularization.
# Layer:      core (pure library, no daemon dependency)
# Role:       the replay arm — the hot/recent working set (_recently_accessed_nodes),
#             the bounded ONLINE active-set (active_set + the _fast_engram_neighbors
#             stub), replay_sweep + the SWR-interleaved variant, and the engram
#             genuine-once feed-forward (replay_engram_traces + per-pair ledger IO).
# Stability:  stable — the offline replay engines (REM-gated in production) + the online
#             locus the write path consults.
# ErrorModel: active_set is fail-soft per source (no web / no hot -> partial locus, never
#             raises); replay_sweep* return {"error": ...} without a vector index;
#             replay_engram_traces no-ops on an empty engram store and swallows per-pair
#             record failures.
# Depends:    .config (REPLAY_*/INTERLEAVE_*/ACTIVE_SET_HOT_N / np / json / _dt / os / sys /
#             _bio_paths); .hebbian (_addr_for_node); samia.core.{vector, temporal, web_store,
#             hippocampus} + samia.runtime.contradiction (lazy, function-local).
# Exposes:    active_set, replay_sweep, replay_sweep_interleaved, replay_engram_traces
#             (public); _recently_accessed_nodes, _fast_engram_neighbors,
#             _all_chain_node_names, _cold_chains, _embedding_for_node, _pair_key,
#             _load_engram_replay_state, _save_engram_replay_state (private, re-exported).
# Lines:      592
# Note:       PATCH SEAM — replay_engram_traces calls hebbian_record THROUGH the package
#             facade (samia.core.bio as _pkg) so a mock.patch.object(bio, "hebbian_record")
#             spy rebinds what the feed runs. bio.active_set is itself a string-patch target
#             with no internal caller (re-export alone suffices).
# --------------------------------------------------------------------------
