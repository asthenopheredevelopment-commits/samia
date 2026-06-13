"""samia.core.bio — biomimetic memory primitives for SAM/IA.

Carved from memory_biomimetic.py. Per design doc §1.1, the daemon's
biomimetic background jobs (Hebbian decay, replay sweep, reconsolidation
on retrieval) call these primitives directly.

Five mechanisms grounded in empirical neuroscience:

  1. pattern_separation     — Marr 1971 / Yassa & Stark 2011
                              cosine-threshold gate at write time.
  2. hebbian                — Hebb 1949 / Bliss & Lomo 1973
                              co-activation log → EMA edge weights;
                              promote strong pairs to chain edges.
  3. reconsolidate          — Nader et al. 2000
                              recall is a write opportunity.
  4. replay_sweep           — Wilson & McNaughton 1994
                              hippocampal replay (and SWR-interleaved variant).
  5. schema_accelerate      — Tse et al. 2007
                              new node entering a mature chain skips cold start.

Public API (parameterized on memory_dir):
  pattern_separation_decision(memory_dir, text, threshold)
  hebbian_record(memory_dir, retrieved_nodes, query)
  hebbian_consolidate(memory_dir, promote)
  reconsolidate(memory_dir, node_name, new_context, backend)
  replay_sweep(memory_dir, sample, threshold)
  replay_sweep_interleaved(memory_dir, sample, cold_per_hot, threshold, seed)
  schema_accelerate(memory_dir, text, chains)
  chain_maturity(memory_dir, chain_name)

Acceptance: byte-identical to pre-refactor memory_biomimetic.py CLI output
on the same memory tree (design doc §8.1).

Note: bio.py depends on `samia.core.chain`, plus `samia.core.{temporal,
vector,fact_extractor}`. Those are lazy-imported inside the functions that
need them (not at module top) to avoid an import cycle: mcp_server imports
bio, and vector/temporal pull in heavier deps; lazy keeps `import bio`
cheap. (GATE6: replaced the legacy `_tools_module()` tools/-dir reachback —
the staged release does not ship the tools/ shims.)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from . import chain as _chain


PATTERN_THRESHOLD_DEFAULT = 0.85

# Hebbian attractor bar + reachability (FEAT-2026-06-05 Tier-0 D2).
# What: HEBB_PROMOTION is the attractor/promotion bar; HEBB_PROMOTE_REPEATS is how
#   many GENUINE co-activations of a pair should cross it; HEBB_EMA_ALPHA is DERIVED
#   so exactly that many full-weight events reach the bar (w_K = 1-(1-alpha)^K).
# Why: the audit found alpha=0.3 needed 6 consecutive repeats to reach 0.85 — far
#   more than co-activations ever accrue — so no edge ever promoted (max w 0.832).
#   Deriving alpha from the intended repeat count makes the bar reachable WITHOUT
#   lowering it (keeping 0.85 semantically meaningful).
HEBB_PROMOTION = 0.85
HEBB_PROMOTE_REPEATS = 3
# Derive alpha so HEBB_PROMOTE_REPEATS genuine events land just PAST the bar. The small
# margin (0.005) keeps exactly-K repeats robustly promotable instead of sitting on a
# floating-point knife-edge at the threshold; one fewer repeat stays well below.
HEBB_EMA_ALPHA = 1.0 - (1.0 - min(0.999, HEBB_PROMOTION + 0.005)) ** (
    1.0 / HEBB_PROMOTE_REPEATS)
HEBB_DECAY = 0.005
HEBB_PRUNE = 0.05

# Homeostatic replay regulation (FEAT-2026-06-05 Tier-0 D1).
# What: replay-derived co-activations contribute at HEBB_REPLAY_COACT_WEIGHT of a
#   genuine event, NEVER refresh the decay clock (last_seen), and a replay-ONLY edge
#   (zero genuine co-activations) is both capped below the bar (REPLAY_ONLY_W_CEILING)
#   and barred from promotion (genuine-count gate). HEBB_SEED_MARGIN keeps the
#   one-time count->w re-seed just below the bar so migration never auto-promotes.
# Why: replay deterministically re-discovers the same pairs every pulse; at full
#   weight it would reset the decay clock and saturate the web (runaway recurrent
#   excitation / feedback reverberation, unprunable edges). These regulators let
#   replay ACCELERATE a genuinely-recent pair toward the bar without manufacturing
#   or immortalizing a stale one. (operator-flagged 2026-06-05; see D1.)
HEBB_REPLAY_COACT_WEIGHT = 0.5
HEBB_SEED_MARGIN = 0.02
REPLAY_ONLY_W_CEILING = HEBB_PROMOTION - HEBB_SEED_MARGIN
REPLAY_NEIGHBOR_THRESHOLD = 0.55
REPLAY_DEFAULT_SAMPLE = 20
INTERLEAVE_THRESHOLD = 0.40
INTERLEAVE_DEFAULT_COLD_PER_HOT = 3
HOT_RECENCY_DAYS = 7
SCHEMA_MIN_NODES = 4
SCHEMA_MIN_AGE_DAYS = 7

# HEBB_MIN_INTERVAL_ENV — What: name of an OPTIONAL env var (seconds) that
#   self-gates how often hebbian_consolidate actually drains the log.
# Why: consolidation is wired onto the per-tool PostToolUse idle pulse
#   (IDLE_THRESHOLD_SECONDS=30) AND a 600s scheduler job, so it fires far more
#   often than co-activations accrue. A min-interval gate decouples the
#   consolidation cadence from the hot pulse WITHOUT a workaround in the trigger
#   wiring. Default 0 (unset) preserves legacy every-pulse behavior, so this
#   module is safe to land before the operator sets the env var. Cadence policy
#   is operator-owned in settings.json (see the operator-paste diff).
HEBB_MIN_INTERVAL_ENV = "ASTHENOS_HEBB_MIN_INTERVAL_S"


def _bio_paths(memory_dir: Path) -> dict:
    bio_dir = memory_dir / "biomimetic"
    return {
        "bio_dir": bio_dir,
        "hebb_log": bio_dir / "coactivation_log.jsonl",
        # hebb_log_processing — What: exclusive tempfile the consumer drains from.
        # Why: ATOMIC DRAIN. hebbian_consolidate os.replace()s the live log onto
        #   this path up front; concurrent hebbian_record appends then land on a
        #   FRESH live log and survive. Closes the truncate lost-update window.
        "hebb_log_processing": bio_dir / "coactivation_log.jsonl.processing",
        "edge_weights": bio_dir / "edge_weights.json",
        "reconsolidate_log": bio_dir / "reconsolidation_log.jsonl",
        "replay_proposals": bio_dir / "replay_proposals.json",
        "replay_interleaved_proposals": bio_dir / "replay_interleaved_proposals.json",
        # hebb_consolidate_state — What: persists the last consolidation unix ts.
        # Why: backs the optional min-interval cadence gate (HEBB_MIN_INTERVAL_ENV)
        #   so the consolidation cadence is decoupled from the per-tool idle pulse.
        "hebb_consolidate_state": bio_dir / "hebb_consolidate_state.json",
        # engram_replay_state — What: per-PAIR genuine-once ledger for engram
        #   replay (FEAT-2026-06-07 Tier-1 P5). Records which engram-derived pairs
        #   have already had their FIRST (genuine) consolidation replay so re-
        #   replays of the same pair log FRACTIONAL, never genuine.
        # Why: D5/Q6a genuine-once — one captured trace cannot be farmed into an
        #   attractor by repeated replay (first genuine + count_genuine bump; rest
        #   fractional then age). The ledger is the "already genuine-replayed" memory.
        "engram_replay_state": bio_dir / "engram_replay_state.json",
        # episode_transitions — What: the DIRECTED co-activation count matrix T_dir
        #   (FEAT-2026-06-11 temporal-recall P6, §5.5). Sparse map of directed keys
        #   "A->B" -> count, A->B incremented for each in-window co-activation pair
        #   where episode_seq(A) < episode_seq(B). Sibling of edge_weights.json (the
        #   undirected store, §5.2) — DIRECTED, not the sorted([a,b]) undirected key.
        # Why: the substrate for the phase-2 directed/forward SR (succession, not just
        #   diffusion). Produced offline inside idle_replay_tick (REM-gated), read query-
        #   locally by successor.py and row-normalized on the fly into T_dir; written
        #   under locked_update_json (incremented, never rebuilt). Bounded ≤ 2·|edges|.
        "episode_transitions": bio_dir / "episode_transitions.json",
    }


# ---------------------------------------------------------------------------
# 1. Pattern separation
# ---------------------------------------------------------------------------


def _node_embedding(memory_dir: Path, name: str) -> Optional[np.ndarray]:
    from . import vector as _vec
    manifest_path = _vec._manifest_path(memory_dir)
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    e = m.get("entries", {}).get(name)
    if not e:
        return None
    embeddings = np.load(_vec._embed_path(memory_dir))
    return embeddings[e["row"]]


def pattern_separation_decision(memory_dir: Path, text: str,
                                threshold: float = PATTERN_THRESHOLD_DEFAULT
                                ) -> dict:
    """Decide whether `text` should be stored as a new node or merged.

    Returns {"action": "store_new"|"merge_into", "target": name|None,
             "score": float, "neighbors": [{node, score}...]}.
    """
    from . import vector as _vec
    hits = _vec.query(memory_dir, text, top_k=5)
    top = hits[0] if hits else None
    neighbors = hits
    if top and top["score"] >= threshold:
        return {"action": "merge_into", "target": top["node"],
                "score": float(top["score"]), "neighbors": neighbors,
                "threshold": threshold}
    return {"action": "store_new", "target": None,
            "score": float(top["score"]) if top else 0.0,
            "neighbors": neighbors, "threshold": threshold}


# ---------------------------------------------------------------------------
# 2. Hebbian co-activation
# ---------------------------------------------------------------------------


def hebbian_record(memory_dir: Path, retrieved_nodes: list[str],
                   query: Optional[str] = None,
                   source: str = "genuine") -> None:
    """Log one co-activation event.

    source: "genuine" (real recalled-together event — full weight, refreshes the
      decay clock) or "replay" (replay-discovered pair — fractional weight,
      decay-transparent; see hebbian_consolidate / D1). Default "genuine" so the
      existing memory_search call site (and its CLI wrapper) is unchanged.
    """
    if len(retrieved_nodes) < 2:
        return
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "query_hash": hashlib.sha256((query or "").encode()).hexdigest()[:8],
        "nodes": list(retrieved_nodes),
        "source": source,
    }
    with paths["hebb_log"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    # FEAT-2026-06-11 temporal-recall P2 — SITH recall jump-back (§4.5). hebbian_record is
    # the recall hook (fires once per genuine recalled-together event); after the
    # co-activation append, nudge the live SITH integrator bank partway (β≈0.3) toward the
    # recalled nodes' encode-time contexts. This is TCM/CMR context reinstatement — it
    # produces the lag-CRP forward asymmetry the temporal layer is calibrated for. PARTIAL
    # (β<1) so the present context is nudged, not overwritten; mutates only the live bank,
    # never any node's immutable snapshot. The `< 2`-node early return above already gates
    # this to genuine multi-node co-activation. Fail-soft + lazy import: any error is
    # swallowed so a hot recall path is never broken. Inert (no bank, no snapshots) until
    # the SITH machinery is exercised; a no-snapshot recall is a no-op.
    try:
        from . import temporal_recall_sith as _sith
        _sith.jump_back_blend(memory_dir, list(retrieved_nodes))
    except Exception:
        pass


def _load_edge_weights(memory_dir: Path) -> dict:
    fp = _bio_paths(memory_dir)["edge_weights"]
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_edge_weights(memory_dir: Path, d: dict) -> None:
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    paths["edge_weights"].write_text(json.dumps(d, indent=2), encoding="utf-8")


def _attractor_count(weights: dict) -> int:
    """Genuine attractors: at/above the bar AND backed by HEBB_PROMOTE_REPEATS genuine
    co-activations (same gate as _is_promotable — replay alone never counts as an attractor)."""
    return sum(1 for v in weights.values() if _is_promotable(v))


def forget_node_weights(memory_dir: Path, node: str) -> dict:
    """FEAT-2026-06-07 P0: drop every edge_weights.json pair touching `node` (the edge_weights
    endpoint of the forget_node cascade). Idempotent. Returns {dropped}."""
    fname = node if node.endswith(".md") else f"{node}.md"
    weights = _load_edge_weights(memory_dir)
    before = len(weights)
    kept = {k: v for k, v in weights.items() if fname not in k.split("::")}
    if len(kept) != before:
        _save_edge_weights(memory_dir, kept)
    return {"dropped": before - len(kept)}


def sweep_ghost_edges(memory_dir: Path, apply: bool = False,
                      db_dir: Optional[str] = None) -> dict:
    """FEAT-2026-06-07 P0 Phase 2 — DESTRUCTIVE when apply=True (operator-gated).

    Remove every edge whose endpoint is no longer a live file in nodes/ (the ghost
    corruption), from edge_weights.json AND edges.db (via forget_node_edges on each dead
    endpoint), then re-derive the genuine-attractor count on the cleaned live-live web.
    apply defaults False = dry-run report with the counts the operator approves BEFORE the
    irreversible run. Live nodes are never touched.
    """
    nodes_dir = memory_dir / "nodes"
    live = {p.name for p in nodes_dir.glob("*.md")}
    weights = _load_edge_weights(memory_dir)
    ghost_keys, dead_endpoints = [], set()
    for k in weights:
        parts = k.split("::")
        if len(parts) != 2 or parts[0] not in live or parts[1] not in live:
            ghost_keys.append(k)
            for n in parts:
                if n not in live:
                    dead_endpoints.add(n)
    kept = {k: v for k, v in weights.items() if k not in set(ghost_keys)}
    report = {
        "apply": apply,
        "live_nodes": len(live),
        "edges_total": len(weights),
        "ghost_edges": len(ghost_keys),
        "live_live_edges": len(kept),
        "dead_endpoints": len(dead_endpoints),
        "attractors_before": _attractor_count(weights),
        "attractors_after_clean": _attractor_count(kept),
    }
    if not apply:
        report["note"] = "DRY-RUN — no writes; approve before apply=True"
        return report
    _save_edge_weights(memory_dir, kept)
    dropped_db = 0
    try:
        from . import web_store as _ws
        for n in dead_endpoints:
            dropped_db += _ws.forget_node_edges(n, db_dir).get("edges_deleted", 0)
    except Exception as e:
        report["edges_db_error"] = str(e)
    report["edges_db_dropped"] = dropped_db
    report["note"] = "APPLIED — destructive sweep complete"
    return report


def _addr_for_node(memory_dir: Path, node_name: str) -> Optional[tuple[str, str]]:
    """Find (chain_name, addr) for a given node filename, if any."""
    chains_dir = memory_dir / "chains"
    for cp in chains_dir.glob("*.json"):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for m in data.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f:
                continue
            stem = Path(f).name
            if stem == node_name or stem == node_name + ".md":
                return cp.stem, m.get("addr")
    return None


def _consolidate_cadence_blocked(paths: dict) -> bool:
    """Return True if the optional min-interval cadence gate says 'skip now'.

    What: read HEBB_MIN_INTERVAL_ENV (seconds); if positive and the persisted
      last-run timestamp is younger than that, block this drain.
    Why: consolidation rides the per-tool idle pulse (~30s) and a 600s job, both
      far faster than co-activations accrue. This gate decouples consolidation
      cadence from the hot pulse with NO workaround in the trigger wiring. Unset
      / non-positive / unparseable env -> 0 -> never blocks (legacy behavior).
    """
    try:
        min_interval = float(os.environ.get(HEBB_MIN_INTERVAL_ENV, "0") or "0")
    except (TypeError, ValueError):
        min_interval = 0.0
    if min_interval <= 0:
        return False
    sp = paths["hebb_consolidate_state"]
    if not sp.exists():
        return False
    try:
        last = float(json.loads(sp.read_text(encoding="utf-8")).get("last_run_unix", 0.0))
    except Exception:
        return False
    return (_time.time() - last) < min_interval


def _record_consolidate_run(paths: dict) -> None:
    """Persist the consolidation run timestamp for the cadence gate."""
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    sp = paths["hebb_consolidate_state"]
    payload = {"last_run_unix": _time.time(),
               "last_run_iso": _dt.datetime.now().isoformat(timespec="seconds")}
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, sp)


def _atomic_drain_log(paths: dict) -> Optional[Path]:
    """Atomically claim the co-activation log for this consolidation pass.

    What: os.replace() the live log onto a private .processing tempfile and
      return that path. If a prior .processing exists (a pass crashed after
      claiming but before deleting), recover THAT first. Returns None when there
      is nothing to drain (no .processing and no non-empty live log).
    Why: closes the truncate lost-update race. Once the live log is renamed, any
      concurrent hebbian_record append re-creates a FRESH live log and survives,
      because the consumer reads ONLY from the claimed tempfile and never blanks
      the live path. The empty-live-log skip is the structural successor to the
      old `events>0` truncate guard: we never CLAIM an empty log, so an empty
      pass leaves the live path untouched and never shadows a future append.
    """
    proc = paths["hebb_log_processing"]
    if proc.exists():
        # Recover a crashed prior claim. (Caller unlinks it once consumed, so a
        # stale empty .processing cannot persist to shadow the live log.)
        return proc
    live = paths["hebb_log"]
    try:
        # GUARD: only claim when the live log actually has bytes. st_size==0 ->
        # nothing to consolidate -> leave it in place (no rename, no truncate).
        if (not live.exists()) or live.stat().st_size == 0:
            return None
        os.replace(live, proc)   # atomic within one filesystem
    except FileNotFoundError:
        return None
    return proc


def _apply_coactivation(weights: dict, nodes: list[str], source: str,
                        today: _dt.date,
                        node_appearances: Optional[dict] = None) -> set:
    """Fold ONE co-activation record into the edge-weight dict in place (pure, no IO).

    Source-aware homeostatic update (D1):
      - EMA move toward 1.0 scaled by src_w (genuine=1.0, replay=HEBB_REPLAY_COACT_WEIGHT);
        the move never decreases w, so replay only ever nudges upward at a fractional rate.
      - GENUINE events refresh last_seen (the decay clock) and increment count_genuine.
        REPLAY events do NOT touch last_seen (decay-transparency); while the pair has fewer
        than HEBB_PROMOTE_REPEATS genuine co-activations its w is capped at
        REPLAY_ONLY_W_CEILING (< HEBB_PROMOTION), so neither a replay-only edge nor a
        once-seeded edge can be farmed past the bar by replay — even over many days. The
        cap lifts only once the genuine bar (HEBB_PROMOTE_REPEATS) is met.
    Returns the set of edge keys touched.
    """
    src_w = HEBB_REPLAY_COACT_WEIGHT if source == "replay" else 1.0
    touched: set = set()
    if node_appearances is not None:
        for n in nodes:
            node_appearances[n] = node_appearances.get(n, 0) + 1
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = sorted([nodes[i], nodes[j]])
            key = f"{a}::{b}"
            cur = weights.get(key, {"w": 0.0, "count": 0, "count_genuine": 0,
                                    "count_replay": 0,
                                    "last_seen": today.isoformat()})
            cur["w"] = cur["w"] + src_w * HEBB_EMA_ALPHA * (1.0 - cur["w"])
            cur["count"] = cur.get("count", 0) + 1
            if source == "replay":
                cur["count_replay"] = cur.get("count_replay", 0) + 1
                # decay-transparency: do NOT refresh last_seen; cap the weight at
                # REPLAY_ONLY_W_CEILING (< HEBB_PROMOTION) until the pair has accrued
                # HEBB_PROMOTE_REPEATS GENUINE events. The ceiling lifts only after the
                # genuine bar is met, so daily fractional replay of a once-recalled pair
                # can never farm it past the attractor bar — replay alone cannot promote.
                if cur.get("count_genuine", 0) < HEBB_PROMOTE_REPEATS:
                    cur["w"] = min(cur["w"], REPLAY_ONLY_W_CEILING)
            else:
                cur["count_genuine"] = cur.get("count_genuine", 0) + 1
                cur["last_seen"] = today.isoformat()
            weights[key] = cur
            touched.add(key)
    return touched


def _decay_and_prune(weights: dict, today: _dt.date) -> int:
    """Per-DAY weight decay + prune (FEAT-2026-06-05 Tier-0 hotfix). Mutates in place.

    Decays each edge by HEBB_DECAY per elapsed day, applied AT MOST ONCE PER DAY via a
    dedicated `last_decay` marker — NOT per consolidation pass. Phase 1 made consolidation
    run every idle pulse (~30s); the old per-pass loop re-applied (1 - HEBB_DECAY*days) every
    pass without advancing a clock, so a stale edge decayed ~80x/hour (compounding) and pruned
    in hours instead of ~0.5%/day. `last_decay` is separate from `last_seen` so the
    genuine-recency clock (and replay decay-transparency, D1) is preserved. Prunes edges whose
    weight falls below HEBB_PRUNE. Returns the number pruned.
    """
    pruned = 0
    for k, v in list(weights.items()):
        ref = v.get("last_decay") or v.get("last_seen")
        try:
            last = _dt.date.fromisoformat(ref)
        except (TypeError, ValueError):
            last = today
        days = (today - last).days
        if days > 0:
            v["w"] *= max(0.0, 1.0 - HEBB_DECAY * days)
            v["last_decay"] = today.isoformat()
        if v["w"] < HEBB_PRUNE:
            del weights[k]
            pruned += 1
    return pruned


def _is_promotable(v: dict) -> bool:
    """Promotion gate (D1/D2): at/above the bar AND backed by HEBB_PROMOTE_REPEATS genuine
    co-activations.

    The genuine-count requirement is the airtight homeostatic guard: decay is
    daily-granularity but replay fires far more often, so fractional weight alone could let an
    edge climb past the bar by replay (within a single day, or — once seeded by ONE genuine
    event — by daily fractional replay over many days). Requiring HEBB_PROMOTE_REPEATS genuine
    events makes "replay alone cannot manufacture an attractor" true regardless of replay
    frequency or duration, matching the REPLAY_ONLY_W_CEILING cap that holds w below the bar
    until the same genuine bar is met. Replay may SEED (one genuine) and gently reinforce a
    still-recent pair, but cannot carry it across the bar.
    Legacy edges (pre-migration, no count_genuine) fall back to count so they aren't barred.
    """
    return (v.get("w", 0.0) >= HEBB_PROMOTION
            and v.get("count_genuine", v.get("count", 0)) >= HEBB_PROMOTE_REPEATS)


def reseed_edge_weights(memory_dir: Path, force: bool = False) -> dict:
    """One-time count_genuine->w re-seed (D2). Idempotent via a marker file.

    Restores each captured edge's deserved strength from its GENUINE co-activation history
    under the NEW alpha: w = 1-(1-alpha)^count_genuine (bounded <1.0). This is faithful to the
    promotion criterion -- an edge with count_genuine >= HEBB_PROMOTE_REPEATS has, by
    construction, w >= HEBB_PROMOTION, so it is promotable on its genuine history.

    v3 (2026-06-06): REMOVED the sub-bar HEBB_SEED_MARGIN cap. v2 capped every edge at 0.83
    "to require one fresh genuine hit", but that overrode the genuine-count promotion criterion
    and pinned genuinely-strong edges (cg 3-5) below the bar forever (the 2.6h soak found 0
    promotions despite 48 edges with earned history). The two concerns the cap guarded are
    already covered without it: (1) replay-only edges must not promote -- the genuine-count gate
    (_is_promotable) handles that, and cg==0 edges are skipped here; (2) STALE strong edges must
    not promote -- _decay_and_prune applies recency decay from last_seen immediately after this,
    so only RECENT high-cg edges clear the bar while old ones decay back under it.

    Seeds from count_genuine (not total count) so replay never inflates the seed. Edges with no
    genuine history (count_genuine==0) are left to their normal fractional/decay lifecycle. A
    fresh store (no count_genuine field yet) falls back to count = original behavior.
    """
    paths = _bio_paths(memory_dir)
    marker = paths["bio_dir"] / ".reseed_v3.done"
    if marker.exists() and not force:
        return {"reseeded": 0, "skipped": "already-done"}
    weights = _load_edge_weights(memory_dir)
    n = 0
    for k, v in weights.items():
        cg = int(v.get("count_genuine", v.get("count", 0)) or 0)
        if cg <= 0:
            continue
        v["w"] = min(0.999, 1.0 - (1.0 - HEBB_EMA_ALPHA) ** cg)
        v.setdefault("count_genuine", cg)
        v.setdefault("count_replay", 0)
        weights[k] = v
        n += 1
    if weights:
        _save_edge_weights(memory_dir, weights)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"reseeded": n, "alpha": round(HEBB_EMA_ALPHA, 4),
                                  "version": 3}), encoding="utf-8")
    return {"reseeded": n}


def hebbian_consolidate(memory_dir: Path, promote: bool = True) -> dict:
    """Read co-activation log, decay weights, and (optionally) promote pairs."""
    paths = _bio_paths(memory_dir)
    chains_dir = memory_dir / "chains"
    reseed_edge_weights(memory_dir)  # one-time count->w migration (D2); no-op after marker
    if not paths["hebb_log"].exists() and not paths["hebb_log_processing"].exists():
        return {"events": 0, "promoted": 0, "pruned": 0}

    # Cadence gate — decouple the consolidation cadence from the per-tool idle
    # pulse / 600s job. Skip BEFORE the atomic drain so pending appends stay on
    # the live log untouched until the gate next opens (no data lost while gated).
    if _consolidate_cadence_blocked(paths):
        return {"events": 0, "promoted": 0, "pruned": 0, "skipped": "cadence_gate"}

    # ATOMIC DRAIN — claim the log up front. Concurrent appends after this land
    # on a fresh live log and are picked up by the NEXT pass; they are never
    # truncated away. drained is None only when there is genuinely nothing.
    drained = _atomic_drain_log(paths)

    weights = _load_edge_weights(memory_dir)
    today = _dt.date.today()

    _decay_and_prune(weights, today)

    events = 0
    new_keys: set[str] = set()
    # node_appearances — What: tally how often each node co-activates this cycle.
    # Why: feeds per-node mass in the unified web store (web_store), computed OFFLINE
    #      here (never on the hot retrieval path) per the event-driven design.
    node_appearances: dict[str, int] = {}
    if drained is not None:
        with drained.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                events += 1
                nodes = rec.get("nodes") or []
                src = rec.get("source", "genuine")
                new_keys |= _apply_coactivation(
                    weights, nodes, src, today, node_appearances)

    # Delete the drained tempfile ONLY after its bytes are fully folded into
    # `weights` above. Unconditional once claimed: a claimed file is always a
    # non-empty live log (the empty-log GUARD in _atomic_drain_log prevents
    # claiming an empty one) or a recovered crashed claim — in both cases its
    # contents are now accounted for. Leaving it would let a stale .processing
    # shadow the live log and silently swallow future appends.
    if drained is not None:
        try:
            drained.unlink()
        except FileNotFoundError:
            pass
    _record_consolidate_run(paths)

    promoted = 0
    for key, v in weights.items():
        if not _is_promotable(v):
            continue
        a, b = key.split("::", 1)
        addr_a = _addr_for_node(memory_dir, a)
        addr_b = _addr_for_node(memory_dir, b)
        if not addr_a or not addr_b:
            continue
        chain_a, A = addr_a
        chain_b, B = addr_b
        if chain_a != chain_b:
            chain_target = sorted([chain_a, chain_b])[0]
        else:
            chain_target = chain_a
        if not promote:
            continue
        try:
            chain = _chain.load_chain(chains_dir, chain_target)
            existing = [e for e in chain.get("edges", []) if e.get("label") == "hebbian"
                        and {e["from"], e["to"]} == {A, B}]
            if existing:
                existing[0]["confidence"] = min(1.0, v["w"])
                _chain.save_chain(chains_dir, chain_target, chain)
                continue
            members = set(_chain.member_addrs(chain))
            if A in members and B in members:
                _chain.add_edge(chains_dir, chain_target, A, B,
                                valid_from=today.isoformat(),
                                valid_to=None,
                                label="hebbian",
                                confidence=min(1.0, v["w"]))
                promoted += 1
        except Exception as e:
            print(f"[hebbian] promote {key} failed: {e}", file=sys.stderr)

    _save_edge_weights(memory_dir, weights)

    # UnifiedWebSync — What: write ALL co-activation edges (cross-chain + orphan, no
    #   membership gate) into the unified associative web store + bump per-node mass,
    #   then decay/prune. Why: the chain-promotion loop above is only the WITHIN-CHAIN
    #   curated OVERLAY; the real webwork (FEAT-2026-05-29-hebbian-cross-chain-web)
    #   lives in web_store, which the Topology Atlas renders. This is the fix for the
    #   measured 100%-orphan starvation (hebbian_health.py).
    web_stats: dict = {}
    try:
        from . import web_store as _ws
        # G3-2026-06-11 (ghost-edge guard): pass memory_dir so the sync SKIPS pairs
        # with a forgotten endpoint (no re-upsert) and reports them as dead_keys.
        web_stats = _ws.sync_from_consolidation(
            weights, node_appearances, memory_dir=memory_dir)
        # Evict the dead-endpoint pairs from edge_weights.json IN THE SAME PASS so a
        # forgotten node's edges do not survive OR re-grow (mirrors the P0 forget_node
        # cascade). The dead keys were just identified above against the SAME live set.
        dead_keys = web_stats.get("dead_keys") or []
        if dead_keys:
            dead_set = set(dead_keys)
            pruned = {k: v for k, v in weights.items() if k not in dead_set}
            if len(pruned) != len(weights):
                _save_edge_weights(memory_dir, pruned)
                weights = pruned
                print(f"[hebbian] ghost-edge evict: dropped {len(dead_set)} "
                      f"dead-endpoint pair(s) from edge_weights.json",
                      file=sys.stderr)
    except Exception as e:  # never let the web sync break consolidation
        web_stats = {"error": str(e)}
        print(f"[hebbian] web_store sync failed: {e}", file=sys.stderr)

    return {"events": events, "weights_total": len(weights),
            "promoted": promoted, "pruned_after_decay": True,
            "web": web_stats}


# ---------------------------------------------------------------------------
# 3. Reconsolidation
# ---------------------------------------------------------------------------


def reconsolidate(memory_dir: Path, node_name: str, new_context: str,
                  backend: str = "auto") -> dict:
    """Recall + LLM-update a node."""
    nodes_dir = memory_dir / "nodes"
    chains_dir = memory_dir / "chains"
    paths = _bio_paths(memory_dir)
    from . import temporal as _tq
    from . import fact_extractor as _fx

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


# ---------------------------------------------------------------------------
# 4. Replay sweep
# ---------------------------------------------------------------------------


def _recently_accessed_nodes(memory_dir: Path, top_n: int) -> list[str]:
    from . import temporal as _tq
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

# ACTIVE_SET_HOT_N -- What: how many hot/recently-accessed nodes join the locus.
# ACTIVE_SET_HOT_N -- Why: the online active-set is "what fires together + what's
#   in working memory"; a small hot/recent top-N keeps it bounded (Q1a + Risk 3).
ACTIVE_SET_HOT_N = 16


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
    wanted = {n if n.endswith(".md") else f"{n}.md" for n in write_nodes}
    locus: set[str] = set()
    # (a) co-activation neighbors per write node (Tier-0 web).
    try:
        from . import web_store as _ws
        for n in write_nodes:
            for nb in _ws.coactivation_neighbors(n, db_dir=db_dir):
                nb_m = nb if nb.endswith(".md") else f"{nb}.md"
                if nb_m not in wanted:
                    locus.add(nb_m)
    except Exception as e:  # fail-soft: no web → no neighbors, locus still useful.
        print(f"[active_set] coactivation lookup failed: {e}", file=sys.stderr)
    # (b) hot / recently-accessed working set.
    try:
        for nm in _recently_accessed_nodes(memory_dir, hot_n):
            nm_m = nm if nm.endswith(".md") else f"{nm}.md"
            if nm_m not in wanted:
                locus.add(nm_m)
    except Exception as e:
        print(f"[active_set] hot/recent lookup failed: {e}", file=sys.stderr)
    # (c) pluggable Tier-1 fast engrams (empty today; filled in P3d).
    for nb in _fast_engram_neighbors(memory_dir, write_nodes):
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


# ---------------------------------------------------------------------------
# FEAT-2026-06-07 Tier-1 P2 (D6) — the salience / affective axis (SOURCE only)
# ---------------------------------------------------------------------------
#
# This is the salience SOURCE: a cheap composite signal-derived score + an explicit
# operator/agent tag override, normalized to [0,1], written to the node frontmatter.
# The salience EFFECTS (the promotion gate max(attractor,salience) = P3, salience-
# dampened decay + freeze-exemption = P5, the merge/supersede consumer = P3-
# contradiction + the merge consumer) are LATER. P2 builds the SOURCE + storage + the
# explicit-tag path + the read-only salience_merge_guard predicate other phases call.

# SALIENCE_W_SURPRISE / _CONTRADICTION / _REPETITION -- What: the composite weights
#   for the three signal-derived salience components (D6 Q8a).
# Why: a node's salience is a weighted blend of surprise (novelty vs the index),
#   contradiction-involvement, and repetition. The weights sum to 1.0 so the composite
#   is already in [0,1] before the explicit-tag override; surprise is weighted highest
#   because the one-shot eureka the hippocampus must retain is novelty-driven, not
#   frequency-driven (the whole point of the orthogonal salience axis). Named/tunable.
SALIENCE_W_SURPRISE = 0.5
SALIENCE_W_CONTRADICTION = 0.3
SALIENCE_W_REPETITION = 0.2

# SALIENCE_REPETITION_SATURATION -- What: the access/co-activation count at which the
#   repetition component saturates to 1.0.
# Why: repetition is a SMALL, saturating contribution (D6 Q8a: "salience is NOT
#   reducible to frequency"); a handful of accesses is enough to max its 0.2 slice,
#   so frequency never dominates the surprise-led composite. Named/tunable.
SALIENCE_REPETITION_SATURATION = 5.0

# SALIENCE_TAG_VALUE -- What: the salience an explicit operator/agent tag clamps to.
# Why: the explicit tag is the deliberate "this matters" HIGH-PRIORITY override (D6
#   Q8a) — it pins salience near 1.0 regardless of the composite. Named/tunable.
SALIENCE_TAG_VALUE = 0.95

# SALIENCE_MERGE_GUARD_DEFAULT -- What: the salience floor at/above which the merge/
#   supersede guard fires (a distinct high-salience memory is surfaced, not absorbed).
# Why: D6 effect (iii) — a HIGH named tunable so only the genuine top tier is guarded
#   (Risk 8: salience inflation). The guard is DEFINED here and CONSUMED by the
#   contradiction/merge proposals; the guard itself is a pure read-only predicate.
SALIENCE_MERGE_GUARD_DEFAULT = 0.8


def _node_frontmatter(memory_dir: Path, node: str) -> Optional[tuple[dict, list, str]]:
    """Read (fm, order, body) of a node; None if missing/unparseable.

    What: a thin wrapper over frontmatter.read_node for the salience helpers.
    Why:  compute_salience and salience_merge_guard read frontmatter fields
      (access_count, salience, salience_tag) and write the salience field back; one
      reader keeps both paths consistent and fail-soft on a node without frontmatter.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = memory_dir / "nodes" / fname
    if not p.exists():
        return None
    try:
        from . import frontmatter as _fm
        fm, order, body = _fm.read_node(p)
        return fm, order, body
    except Exception:
        return None


def _salience_surprise(memory_dir: Path, content: str,
                       exclude_node: Optional[str] = None) -> float:
    """Surprise/novelty signal: 1 - max_cosine of `content` vs the main vector index.

    What: embed `content` with the shared backend, cosine it against the existing main
      embeddings (vector_index/embeddings.npy), and return 1 - max_cosine clamped to
      [0,1]. A node unlike anything stored scores high (surprising); a near-duplicate
      scores low. Returns 0.0 (a MISSING signal contributes nothing) when the index is
      absent/empty or the embedder is unavailable — never crashes.
    Why:  D6 Q8a signal 1 — prediction-error vs the index, RELATIVE (calibrated against
      what is already stored) so a uniformly-novel corpus does not all score high
      (Risk 8). Reuses vector._embed_batch + the main index; reinvents no embedding.

    Args:
        exclude_node: FEAT-2026-06-11 salience-coverage P2 — when given (a node id,
          with or without .md), DROP that node's own embedding row from the index
          before taking max_cosine (a leave-one-out). DEFAULT None keeps the legacy
          byte-identical behavior. Why: an ALREADY-INDEXED node self-matches (cos≈1)
          so its raw surprise is degenerately ~0; the backfill (which scores nodes that
          are already in the index) excludes the node's own row so surprise reflects
          novelty vs the REST of the corpus, not vs itself. A missing manifest/row is
          fail-soft: excluding nothing degrades to the legacy (self-matched) value.
    """
    try:
        from . import vector as _vi
        ip = memory_dir / "vector_index" / "embeddings.npy"
        if not ip.exists():
            return 0.0  # missing signal -> 0
        emb = np.load(ip)
        if emb.shape[0] == 0:
            return 0.0
        # P2 leave-one-out: drop the node's own row so it cannot self-match to cos≈1.
        # Fail-soft — a missing manifest/row/tombstone just leaves emb intact (legacy).
        if exclude_node is not None:
            try:
                fname = exclude_node if exclude_node.endswith(".md") else f"{exclude_node}.md"
                manifest = _vi._load_manifest(memory_dir)
                entry = manifest.get("entries", {}).get(fname)
                row = entry.get("row") if isinstance(entry, dict) else None
                if isinstance(row, int) and 0 <= row < emb.shape[0]:
                    emb = np.delete(emb, row, axis=0)
            except Exception:
                pass  # excluding nothing falls back to the legacy self-matched value
            if emb.shape[0] == 0:
                return 0.0  # the corpus was just this one node -> no comparison signal
        q = _vi._embed_batch([content])[0]
        sims = emb @ q
        max_cos = float(np.max(sims))
        return float(min(1.0, max(0.0, 1.0 - max_cos)))
    except Exception:
        return 0.0  # any failure is a missing signal, not a crash


def _salience_contradiction(memory_dir: Path, node: str) -> float:
    """Contradiction-involvement signal: 1.0 if the node is in a supersession candidate.

    What: scan the unified supersession-candidate store (contradiction.
      list_supersession_candidates) for any UNRESOLVED candidate naming `node` as
      either the old_id or the new_id; return 1.0 if found, else 0.0.
    Why:  D6 Q8a signal 2 — a node that triggered or resolved a supersession is
      important (a belief-overturning contradiction). Reuses the existing canonical
      candidate reader; a missing store -> 0.0 (missing signal contributes nothing).
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    try:
        from samia.runtime import contradiction as _con
        cands = _con.list_supersession_candidates(memory_dir, unresolved_only=False)
    except Exception:
        return 0.0
    for c in cands:
        if c.get("old_id") == fname or c.get("new_id") == fname:
            return 1.0
    return 0.0


def _salience_repetition(memory_dir: Path, node: str, fm: Optional[dict]) -> float:
    """Repetition signal: small saturating contribution from access + co-activation.

    What: combine the node's frontmatter access_count with its Tier-0 co-activation
      degree (edge_weights pairs touching the node) into a saturating [0,1] value
      (count / SALIENCE_REPETITION_SATURATION, clamped to 1.0).
    Why:  D6 Q8a signal 3 — frequency contributes, but it is the SMALLEST, fastest-
      saturating slice so salience is not reducible to frequency. Reuses the existing
      access_count frontmatter + the edge_weights map; a missing signal -> 0.
    """
    count = 0.0
    if fm is not None:
        try:
            count += float(fm.get("access_count", 0) or 0)
        except (TypeError, ValueError):
            pass
    # Tier-0 co-activation degree (pairs touching the node).
    fname = node if node.endswith(".md") else f"{node}.md"
    try:
        weights = _load_edge_weights(memory_dir)
        for key in weights:
            if fname in key.split("::"):
                count += 1.0
    except Exception:
        pass
    sat = max(SALIENCE_REPETITION_SATURATION, 1e-9)
    return float(min(1.0, count / sat))


def compute_salience(memory_dir: Path, node: str,
                     content: Optional[str] = None,
                     explicit_tag: Optional[bool] = None,
                     write: bool = True,
                     exclude_self_from_surprise: bool = False) -> float:
    """Compute (and optionally persist) a node's [0,1] salience score (D6 Q8a SOURCE).

    What: aggregate the three cheap composite signals — surprise (1 - max_cosine vs the
      vector index), contradiction-involvement (in the supersession store), repetition
      (access + co-activation, saturating) — into a weighted [0,1] score, then apply the
      EXPLICIT operator/agent tag override: an explicit tag (passed via `explicit_tag`
      or already on the node's `salience_tag` frontmatter) clamps salience HIGH
      (SALIENCE_TAG_VALUE) regardless of the composite. When `write` (default), the
      result is written to the node's `salience` frontmatter field (and `salience_tag`
      is persisted when set so the override is sticky/operator-visible).
    Why:  D6 — the salience SOURCE. Each signal is grounded in a primitive already in
      the system (Risk: negligible write/touch cost) and a MISSING signal contributes
      0 (no crash). This builds only the SOURCE + storage + explicit-tag path; the
      EFFECTS (promotion / decay / merge) are P3/P5/consumers — NOT applied here.

    Args:
        node: the node id (with or without .md).
        content: the text to score surprise against; defaults to the node's own
          embedding-ready content (title + description + body) when omitted.
        explicit_tag: True to SET the operator/agent override (clamps high), False to
          leave it, None to honor whatever the node already carries.
        write: when True (default) persist `salience` (+ `salience_tag`) to frontmatter.
        exclude_self_from_surprise: FEAT-2026-06-11 salience-coverage P2 — when True,
          the surprise term excludes the node's OWN embedding row from the index
          (leave-one-out) so an already-indexed node does not self-match to ~0. The
          BACKFILL sets this; every other caller leaves it False so the surprise term is
          byte-identical to today (a fresh at-write node is not yet in the index, so it
          can never self-match anyway). DEFAULT False = legacy behavior.

    Returns the normalized [0,1] salience score.
    """
    fm_bundle = _node_frontmatter(memory_dir, node)
    fm = fm_bundle[0] if fm_bundle else None

    # Resolve the content to score surprise against (default: the node's own text).
    if content is None and fm_bundle is not None:
        try:
            from . import vector as _vi
            fname = node if node.endswith(".md") else f"{node}.md"
            _title, content = _vi._load_node_text(memory_dir / "nodes" / fname)
        except Exception:
            content = ""
    if content is None:
        content = ""

    surprise = _salience_surprise(
        memory_dir, content,
        exclude_node=node if exclude_self_from_surprise else None)
    contradiction = _salience_contradiction(memory_dir, node)
    repetition = _salience_repetition(memory_dir, node, fm)

    composite = (SALIENCE_W_SURPRISE * surprise
                 + SALIENCE_W_CONTRADICTION * contradiction
                 + SALIENCE_W_REPETITION * repetition)
    composite = float(min(1.0, max(0.0, composite)))

    # Explicit-tag override: an explicit tag (arg or pre-existing frontmatter) clamps
    # salience HIGH. This is the deliberate "this matters" high-priority override.
    tagged = bool(explicit_tag)
    if explicit_tag is None and fm is not None:
        tagged = bool(fm.get("salience_tag", False))
    salience = max(composite, SALIENCE_TAG_VALUE) if tagged else composite
    salience = float(round(min(1.0, max(0.0, salience)), 4))

    if write and fm_bundle is not None:
        fm, order, body = fm_bundle
        if "salience" not in fm:
            order.append("salience")
        fm["salience"] = salience
        if tagged and not fm.get("salience_tag"):
            if "salience_tag" not in fm:
                order.append("salience_tag")
            fm["salience_tag"] = True
        try:
            from . import frontmatter as _fm
            fname = node if node.endswith(".md") else f"{node}.md"
            _fm.write_node(memory_dir / "nodes" / fname, fm, order, body)
        except Exception:
            pass  # fail-soft: a write failure must not crash the capture/touch path

    # FEAT-2026-06-11 temporal-recall P4 (§6.2 + §16.2 Q2): the STC capture TRIGGER.
    # compute_salience is the write-time salience source; it is the natural place to
    # evaluate the strong-anchor trigger. When the persisted salience clears the strong
    # bar, fire capture_event — it stamps a decaying stc_capture_score onto temporally-
    # adjacent WEAK nodes in the anchor's EPISODE_SEQ-relative window (N before / M after,
    # wall-clock-capped; cosine + 1/chain/hour guards). capture_event is INERT under the
    # master flag off (it checks temporal_weight_enabled and writes NOTHING), so flag-off
    # writes touch no frontmatter and the decay/promotion/recall paths stay byte-identical.
    # Gate the call on the strong bar here too so a sub-bar write pays nothing. Fail-soft +
    # lazy import to dodge the bio<->temporal_recall_stc cycle; any error is swallowed so
    # the salience/capture/touch path is never broken.
    try:
        from . import temporal_recall_stc as _stc
        if salience >= _stc.STC_STRONG_THRESHOLD:
            _stc.capture_event(memory_dir, node)
    except Exception:
        pass
    return salience


def salience_merge_guard(memory_dir: Path, node: str,
                         threshold: float = SALIENCE_MERGE_GUARD_DEFAULT,
                         is_duplicate: bool = False) -> bool:
    """Read-only merge/supersede guard predicate (D6 effect iii — DEFINED, not consumed).

    What: return True when the node's `salience` frontmatter is >= `threshold` AND the
      node is NOT a true duplicate (is_duplicate False). A consult-only predicate the
      contradiction detector + merge consumer call BEFORE acting: when it fires, the
      consumer must NOT auto-supersede/merge the high-salience distinct memory — it
      surfaces it for review instead. Pure read; mutates nothing, applies no effect.
    Why:  D6 — the guard is DEFINED in P2 and CONSUMED downstream (P3-contradiction +
      the merge consumer). It protects a distinct important memory from being absorbed
      by a later, more-frequent, less-important one. It does NOT change the cosine dedup
      gate's role: a TRUE duplicate (is_duplicate True) is still deduped regardless of
      salience, so the guard returns False for it.
    """
    if is_duplicate:
        return False
    fm_bundle = _node_frontmatter(memory_dir, node)
    if fm_bundle is None:
        return False
    try:
        sal = float(fm_bundle[0].get("salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return sal >= float(threshold)


# ---------------------------------------------------------------------------
# FEAT-2026-06-07 Tier-1 P3 (D2) — kWTA pattern separation (orthogonalize-on-copy)
# ---------------------------------------------------------------------------
#
# This is the ORTHOGONALIZING sense of pattern separation (audit Tier-1 item 4):
# a sparse code computed ONCE at materialization, on the engram held copy, so two
# near-duplicate episodes are stored DISTINCTLY (individually addressable) even
# though the cosine dedup gate (pattern_separation_decision, above) still catches
# true duplicates. kWTA tags the COPY, not the retrieval embedding — retrieval
# cosine is unaffected. The two jobs (orthogonalize vs dedup) stay separate (D2).

# KWTA_PROJ_DIM -- What: the higher-dim space the embedding is random-projection
#   lifted into before the top-k% winner-take-all.
# Why: a high-dim sparse code is what makes near-duplicate inputs land on largely
#   DISJOINT winner sets (sparse high-dim codes are nearly orthogonal — the
#   expander/pattern-separation property the dentate gyrus exploits). 1024 over the
#   384-dim MiniLM embedding gives ample room for ~2-5% sparse separation. Named.
KWTA_PROJ_DIM = 1024

# KWTA_FRAC_DEFAULT -- What: the fraction of projected units kept active (the k of
#   kWTA), default 0.03 (3%, inside the 2-5% band, D2).
# Why: sparse enough that near-duplicates separate, dense enough that the code still
#   carries the episode's identity. Named/tunable; clamped to >=1 winner.
KWTA_FRAC_DEFAULT = 0.03

# KWTA_SEED -- What: the FIXED seed for the random-projection matrix.
# Why: determinism — the SAME embedding must always yield the SAME sparse key (so a
#   re-materialize updates rather than forks the code, and tests are reproducible).
#   The projection is a fixed random basis, generated once per (in_dim, proj_dim).
KWTA_SEED = 1729

_KWTA_PROJ_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _kwta_projection(in_dim: int, proj_dim: int = KWTA_PROJ_DIM,
                     seed: int = KWTA_SEED) -> np.ndarray:
    """Return the FIXED (deterministic) random-projection matrix [in_dim, proj_dim].

    What: a seeded Gaussian random matrix, cached per (in_dim, proj_dim, seed) so the
      SAME basis is reused for every code of a given embedding shape.
    Why: a fixed random projection makes the sparse code a deterministic function of
      the input (determinism requirement) while the random basis is what spreads
      near-duplicate inputs onto separable winner sets (orthogonalization). Caching
      avoids regenerating the basis on every materialize.
    """
    key = (int(in_dim), int(proj_dim), int(seed))
    cached = _KWTA_PROJ_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(seed)
        cached = rng.standard_normal((in_dim, proj_dim)).astype(np.float32)
        _KWTA_PROJ_CACHE[key] = cached
    return cached


def kwta_sparse_code(embedding, frac: float = KWTA_FRAC_DEFAULT,
                     proj_dim: int = KWTA_PROJ_DIM,
                     seed: int = KWTA_SEED) -> list[int]:
    """kWTA sparse code of an embedding (D2 — orthogonalize-on-materialize).

    What: random-projection lift `embedding` into a `proj_dim`-dim space via a FIXED
      seeded basis, then keep the indices of the top `frac` (2-5%) activations
      (k-winners-take-all) — return that sorted winner-index set as the sparse code.
      Deterministic: the same embedding always yields the same code.
    Why: D2 — the orthogonalizing pattern-separation primitive. Two near-duplicate
      embeddings produce LARGELY DISJOINT winner sets (so each engram copy stays
      individually addressable), while a true duplicate (handled by the SEPARATE
      cosine dedup gate, pattern_separation_decision) is still deduped. Runs once on
      the held copy at materialization; never on the retrieval embedding.

    Args:
        embedding: a 1-D vector (any dim — 384-dim MiniLM in production, the small
          stub dim in tests).
        frac: the fraction of projected units to keep active (k of kWTA).
        proj_dim / seed: the fixed projection space + seed (determinism).

    Returns a sorted list of winner indices (the sparse key). Empty for an empty/zero
    embedding.
    """
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.size == 0 or not np.any(vec):
        return []
    proj = _kwta_projection(vec.shape[0], proj_dim, seed)
    activations = vec @ proj                       # [proj_dim] projected response
    k = max(1, int(round(proj_dim * float(frac)))) # at least one winner
    k = min(k, proj_dim)
    # top-k winners by activation; argpartition is O(n), then sort the winners.
    winners = np.argpartition(activations, -k)[-k:]
    return sorted(int(i) for i in winners)


def replay_sweep(memory_dir: Path,
                 sample: int = REPLAY_DEFAULT_SAMPLE,
                 threshold: float = REPLAY_NEIGHBOR_THRESHOLD) -> dict:
    """Pick recently-accessed nodes, find semantic neighbors, propose cross-chain edges."""
    from . import vector as _vec
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


def _embedding_for_node(name: str, manifest: dict, embeddings) -> Optional[np.ndarray]:
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
    from . import vector as _vec
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
    from . import hippocampus as _hip

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
                hebbian_record(memory_dir, [p["a"], p["b"]],
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
            hebbian_record(memory_dir, [p["a"], p["b"]],
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


# ---------------------------------------------------------------------------
# 5. Schema-accelerated ingestion
# ---------------------------------------------------------------------------


def chain_maturity(memory_dir: Path, chain_name: str) -> dict:
    from . import temporal as _tq
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


def schema_accelerate(memory_dir: Path, text: str, chains: list[str]) -> dict:
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


# ── module metadata ────────────────────────────────────────────────────────
# file:        samia/core/bio.py
# role:        biomimetic memory primitives (pattern-sep, Hebbian, replay, etc.)
# fix:         FIX-H Hebbian coactivation-log truncate race — atomic drain
#              (os.replace -> .processing tempfile) replaces the unconditional
#              write_text('') truncate; events>0 guard + optional cadence gate.
# fix:         FEAT-2026-06-05 Tier-0 D1/D2 — reachable attractor bar (alpha derived
#              from HEBB_PROMOTE_REPEATS), one-time count->w re-seed, and homeostatic
#              replay co-activations (source-tagged: fractional weight + decay-
#              transparent + genuine-count promotion gate, so replay alone can neither
#              manufacture nor sustain an attractor — no runaway, pruning preserved).
# fix:         FEAT-2026-06-07 Tier-1 P5 (D5/Q6a) — engram feed-forward with
#              GENUINE-ONCE: replay_engram_traces replays the captured engram held
#              copies into the Tier-0 co-activation log; a pair's FIRST replay is
#              genuine (+count_genuine), every re-replay is fractional then ages
#              (per-pair ledger in engram_replay_state.json). Reuses the existing
#              hebbian_record/_apply_coactivation/_decay_and_prune machinery — replay
#              alone still cannot manufacture or immortalize an attractor (one genuine
#              per pair vs HEBB_PROMOTE_REPEATS needed). Fixes raw_pairs:0 + the
#              usage-bounded genuine signal; REM-gated + inert via idle_replay_tick.
# fix:         FEAT-2026-06-07 Tier-1 P2 (D6) — the salience SOURCE: compute_salience
#              (surprise=1-max_cosine vs the index + contradiction-involvement +
#              saturating repetition + explicit-tag override, normalized [0,1], written
#              to the `salience` frontmatter field) and the read-only salience_merge_guard
#              predicate (DEFINED here, consumed by the contradiction/merge proposals).
#              SOURCE + storage + explicit-tag ONLY; salience EFFECTS (promotion gate,
#              dampened decay, merge auto-action) are P3/P5/consumers, NOT applied here.
# fix:         FEAT-2026-06-11 temporal-recall P4 (§6.2 + §16.2 Q2) — STC capture TRIGGER:
#              compute_salience fires temporal_recall_stc.capture_event when the persisted
#              salience clears STC_STRONG_THRESHOLD (0.70). Stamps a decaying stc_capture_
#              score onto weak nodes in the anchor's EPISODE_SEQ-relative window. INERT
#              under ASTHENOS_TEMPORAL_WEIGHT off (capture writes nothing -> decay/promotion/
#              recall byte-identical); fail-soft + lazy import (no bio<->stc cycle at top).
#              (Also: hebbian_record fires the P2 SITH jump-back blend, see above.)
# owns:        biomimetic/coactivation_log.jsonl drain lifecycle + edge_weights.json
# consumers:   context_extension.idle_replay_tick (caller), web_store (handoff)
# cadence:     ASTHENOS_HEBB_MIN_INTERVAL_S env (settings.json) decouples the
#              consolidation cadence from the per-tool idle pulse / 600s job.
# G3-2026-06-11 (ghost-edge guard): hebbian_consolidate passes memory_dir into
#              web_store.sync_from_consolidation so dead-endpoint pairs are SKIPPED
#              (not re-upserted into edges.db) and EVICTED from edge_weights.json in
#              the same pass (logged count). The every-cycle counterpart to the
#              operator-gated sweep_ghost_edges; mirrors the P0 forget_node cascade.
# restart:     bio.py changes require restarting samia.runtime.daemon (PID ~3167).
