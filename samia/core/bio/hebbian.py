"""samia.core.bio.hebbian — Hebbian co-activation (log → EMA edge weights → promotion).

Layer 1 (Owns / Depends):
    Owns:    the Hebbian arm (Hebb 1949 / Bliss & Lomo 1973). The recall hook
             hebbian_record (append one co-activation event + fire the SITH jump-back
             blend); the edge_weights.json read/write (_load_edge_weights /
             _save_edge_weights); the genuine-attractor accounting (_attractor_count /
             _is_promotable, the D1/D2 genuine-count promotion gate); the forget /
             ghost-edge cleanup (forget_node_weights, sweep_ghost_edges); the chain
             address resolver _addr_for_node; the atomic-drain + cadence machinery
             (_consolidate_cadence_blocked / _record_consolidate_run / _atomic_drain_log);
             the homeostatic in-place weight update + daily decay/prune (_apply_coactivation
             / _decay_and_prune); the one-time count->w re-seed (reseed_edge_weights);
             and the consolidation driver hebbian_consolidate (drain → decay → fold →
             promote within-chain → sync the unified web).
    Depends: config (constants HEBB_* / _bio_paths / _dt / _time / json / os / sys /
             _chain); samia.core.{web_store, temporal_recall_sith} (lazy, function-local
             — the unified web sync + the SITH recall jump-back, both kept off the
             import path to break cycles). No sibling-arm dependency.

Layer 2 (What / Why):
    What: the co-activation → EMA edge-weight → chain-promotion pipeline, with the
          homeostatic guards (source-tagged genuine/replay weights, decay-transparency,
          the genuine-count promotion gate, the daily-once decay, the atomic drain).
    Why:  carved out of the monolith as the Hebbian responsibility. hebbian_record is a
          mock.patch.object(bio, ...) seam (tests spy on it); the ONLY internal caller
          (replay_engram_traces, in the replay sibling) reaches it THROUGH the package
          facade so a facade-level patch is honored. web_store + temporal_recall_sith
          are lazy to keep the import cheap and break the bio<->sith / bio->web_store
          cycles.
"""

from __future__ import annotations

from typing import Optional

from . import config as _cfg
from .config import (
    _dt,
    _time,
    _chain,
    json,
    os,
    sys,
    hashlib,
    Path,
    HEBB_PROMOTION,
    HEBB_PROMOTE_REPEATS,
    HEBB_EMA_ALPHA,
    HEBB_DECAY,
    HEBB_PRUNE,
    HEBB_REPLAY_COACT_WEIGHT,
    REPLAY_ONLY_W_CEILING,
    HEBB_MIN_INTERVAL_ENV,
    HEBB_STOPNODE_PREFIXES,
    HEBB_STOPNODE_PREFIXES_ENV,
    HEBB_LIFT_GATE_ENABLED_ENV,
    HEBB_LIFT_MIN_ENV,
    HEBB_LIFT_MIN_DEFAULT,
    HEBB_SATURATE_ENABLED_ENV,
    HEBB_SATURATE_TARGET_DEFAULT,
    _bio_paths,
)


def _stopnode_prefixes() -> tuple[str, ...]:
    """FEAT-2026-06-18 P2 — the live stop-node prefix list (env-overridable).

    What: read HEBB_STOPNODE_PREFIXES_ENV (comma-separated) if set, else the
      HEBB_STOPNODE_PREFIXES default. Lowercased, blanks dropped, read EACH call
      so a daemon that sets the env after import sees the change.
    Why: single source for the stop-node test, tunable WITHOUT a code edit
      (mirrors HEBB_MIN_INTERVAL_ENV / the contradiction exclude-types env).
    """
    raw = os.environ.get(HEBB_STOPNODE_PREFIXES_ENV, "")
    if raw.strip():
        return tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    return tuple(p.lower() for p in HEBB_STOPNODE_PREFIXES)


def _is_stopnode(node_id: str) -> bool:
    """FEAT-2026-06-18 P2 — True iff `node_id` is a low-value high-cardinality
    stop node that must NOT form Hebbian co-activation edges.

    What: lowercase + strip the .md suffix, then match the live prefix list.
      `session_` carries an extra "offload"-substring guard so only the episodic
      session-offload dumps (`session_*_offload`) are excluded — a non-offload
      session_* node is NOT over-excluded. Every other prefix (e.g. `sem_`) is a
      plain startswith.
    Why: mirrors contradiction.is_excluded_node's filename-by-name handling of the
      same session-offload episodic class. Pure (no IO) so it is cheap to call in
      the hot recall path before the all-pairs.
    """
    stem = node_id[:-3] if node_id.endswith(".md") else node_id
    low = stem.lower()
    for pfx in _stopnode_prefixes():
        if not low.startswith(pfx):
            continue
        # session_* is only a stop node when it is an OFFLOAD dump; bare session_
        # ids that are not offloads remain eligible to co-activate.
        if pfx == "session_" and "offload" not in low:
            continue
        return True
    return False


def _filter_stopnodes(nodes: list[str]) -> list[str]:
    """FEAT-2026-06-18 P2 — drop stop nodes (order-preserving) before the all-pairs."""
    return [n for n in nodes if not _is_stopnode(n)]


def hebbian_record(memory_dir: Path, retrieved_nodes: list[str],
                   query: Optional[str] = None,
                   source: str = "genuine",
                   issue_id: Optional[str] = None) -> None:
    """Log one co-activation event.

    source: "genuine" (real recalled-together event — full weight, refreshes the
      decay clock) or "replay" (replay-discovered pair — fractional weight,
      decay-transparent; see hebbian_consolidate / D1). Default "genuine" so the
      existing memory_search call site (and its CLI wrapper) is unchanged.

    FEAT-2026-06-18 P2 (stop-node exclusion): low-value high-cardinality nodes
      (`session_*_offload` episodic dumps + `sem_*` session-event nodes) are
      filtered out BEFORE anything is logged, so they never form co-activation
      edges (the red-team's degree-32 mega-hub pathology). This mirrors the
      entity-bridge + active-set (contradiction.is_excluded_node) filters already
      present elsewhere. If <2 nodes remain after filtering, we record nothing.
    """
    if len(retrieved_nodes) < 2:
        return
    # STOP-NODE EXCLUSION (P2) — filter BEFORE the early-return re-check + the
    # downstream all-pairs fold. Done here (the single recall hook) so neither the
    # SITH jump-back nor the epiphanies archive nor the consolidation all-pairs
    # ever sees a stop node, and the live coactivation_log.jsonl stays clean at the
    # source. If fewer than 2 genuine nodes survive, there is no pair to record.
    retrieved_nodes = _filter_stopnodes(list(retrieved_nodes))
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
        from samia.core import temporal_recall_sith as _sith
        _sith.jump_back_blend(memory_dir, list(retrieved_nodes))
    except Exception:
        pass

    # Epiphanies v3 (FLAG-GATED, default OFF; FEAT-2026-06-16-memory-episodic-associative-
    # binding). Archive this co-activation durably so SITTINGS can be reconstructed offline —
    # the live hebb_log above is DRAINED at consolidation, so it cannot serve as the sitting
    # source. archive_event no-ops fast + fail-soft when ASTHENOS_EPI_ENABLED is unset, so the
    # recall hot path is byte-for-byte unchanged in the default (off) configuration.
    try:
        from samia.core.bio import epiphanies as _epi
        _epi.archive_event(memory_dir, list(retrieved_nodes), query, source, issue_id=issue_id)
    except Exception:
        pass


def _load_marginal_counts(memory_dir: Path) -> dict:
    """FEAT-2026-06-18 Phase-2 — load the per-node co-activation marginal store.

    Shape: {"N": int (grand total co-activation events), "counts": {node_id: C(i)}}.
    Returns the zero store on a missing/corrupt file (fail-soft, like _load_edge_weights).
    """
    fp = _bio_paths(memory_dir)["marginal_counts"]
    if fp.exists():
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                d.setdefault("N", 0)
                d.setdefault("counts", {})
                return d
        except Exception:
            pass
    return {"N": 0, "counts": {}}


def _save_marginal_counts(memory_dir: Path, d: dict) -> None:
    """FEAT-2026-06-18 Phase-2 — persist the marginal-count store (atomic replace)."""
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    fp = paths["marginal_counts"]
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    os.replace(tmp, fp)


def _lift(c_ij: float, c_i: float, c_j: float, n: float) -> float:
    """FEAT-2026-06-18 Phase-2 — association lift = C(ij)·N / (C(i)·C(j)) (= exp PMI).

    What: the marginal-frequency-corrected co-occurrence significance. lift>1 means
      the pair co-activates MORE than chance given each node's own popularity; lift>2
      is the standard min-significance cut (Church&Hanks'90, Agrawal&Srikant'94).
    Why: raw co-occurrence rewards popular nodes (a busy hub co-occurs with everything
      by chance); dividing by the marginals C(i)·C(j) removes that confound so genuine
      low-frequency associations are not out-competed by frequent spurious ones.
    Returns 0.0 on a degenerate denominator (an unseen/zero-marginal node) so the gate
    treats no-evidence as not-significant rather than crashing.
    """
    denom = c_i * c_j
    if denom <= 0 or n <= 0:
        return 0.0
    return (c_ij * n) / denom


def _lift_gate_enabled() -> bool:
    """FEAT-2026-06-18 Phase-2 — True iff the significance gate is flag-ON (default OFF).

    Reads ASTHENOS_HEBB_LIFT_GATE live each call; truthy values: 1/true/yes/on.
    """
    raw = os.environ.get(HEBB_LIFT_GATE_ENABLED_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _lift_min() -> float:
    """FEAT-2026-06-18 Phase-2 — the live lift threshold (env-overridable, default 2.0)."""
    try:
        return float(os.environ.get(HEBB_LIFT_MIN_ENV, "") or HEBB_LIFT_MIN_DEFAULT)
    except (TypeError, ValueError):
        return HEBB_LIFT_MIN_DEFAULT


def _load_edge_weights(memory_dir: Path) -> dict:
    fp = _bio_paths(memory_dir)["edge_weights"]
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_edge_weights(memory_dir: Path, d: dict) -> None:
    """Persist the RECALL-graph co-activation edge weights to edge_weights.json.

    NOTE: these are RECALL-graph co-activation EDGE weights (edges.db / edge_weights.json) — a
    purely retrieval-side association graph. T2/T3 nodes are admitted as recall-able READ-ONLY
    knowledge by design, so they legitimately participate in the co-activation graph; writing a
    T2/T3 edge weight here only affects retrieval ranking and never crosses into any model-training
    path (a training-curriculum boundary, enforced elsewhere, owns that separation).
    """
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    # ATOMIC write (audit 2026-06-20): tmp + os.replace, mirroring the other writers in this module
    # (e.g. _save_marginal_counts:203-205). A direct write_text could truncate the live recall graph on
    # a mid-write crash; this matters once A' (veto-as-recall-filter) APPLY writes here. _load_edge_
    # weights is fail-soft on a corrupt file, but atomicity avoids the corruption in the first place.
    fp = paths["edge_weights"]
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    os.replace(tmp, fp)


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
        from samia.core import web_store as _ws
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
                        today: "_dt.date",
                        node_appearances: Optional[dict] = None,
                        marginals: Optional[dict] = None) -> set:
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

    FEAT-2026-06-18 Phase-2 — marginals + significance (lift) gate:
      - `marginals` (when given): {"N": int, "counts": {node: C(i)}} is updated IN PLACE
        — N += 1 per record, C(i) += 1 per participating node — so the lift gate always
        has up-to-date marginal-frequency evidence. This tracking runs UNCONDITIONALLY
        (data is cheap and the operator needs it the instant the flag flips).
      - The lift GATE (default OFF; ASTHENOS_HEBB_LIFT_GATE) suppresses the EMA strengthen
        for a pair whose lift = C(ij)·N/(C(i)·C(j)) <= HEBB_LIFT_MIN: per-pair counts and
        the touched-set still update (so an edge can EARN significance over time), but its
        weight is held until the pair is statistically meaningful. With the flag OFF (the
        default) behavior is byte-for-byte the pre-Phase-2 EMA update.
    Returns the set of edge keys touched.
    """
    src_w = HEBB_REPLAY_COACT_WEIGHT if source == "replay" else 1.0
    touched: set = set()
    gate_on = marginals is not None and _lift_gate_enabled()
    lift_min = _lift_min() if gate_on else 0.0
    if node_appearances is not None:
        for n in nodes:
            node_appearances[n] = node_appearances.get(n, 0) + 1
    # Marginal tracking (Phase-2): one EVENT (N) + one participation per node C(i).
    if marginals is not None:
        marginals["N"] = int(marginals.get("N", 0)) + 1
        mc = marginals.setdefault("counts", {})
        for n in nodes:
            mc[n] = int(mc.get(n, 0)) + 1
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = sorted([nodes[i], nodes[j]])
            key = f"{a}::{b}"
            cur = weights.get(key, {"w": 0.0, "count": 0, "count_genuine": 0,
                                    "count_replay": 0,
                                    "last_seen": today.isoformat()})
            cur["count"] = cur.get("count", 0) + 1
            # SIGNIFICANCE GATE (Phase-2, default OFF) — compute lift from the
            # post-increment marginals; below threshold, hold the weight (do not
            # strengthen) but keep counting so the pair can earn significance. With
            # the gate off, strengthen is unconditional (legacy behavior).
            strengthen = True
            if gate_on:
                mc = marginals.get("counts", {})
                lift = _lift(cur["count"], mc.get(a, 0), mc.get(b, 0),
                             marginals.get("N", 0))
                cur["lift"] = round(lift, 4)
                strengthen = lift > lift_min
            if strengthen:
                cur["w"] = cur["w"] + src_w * HEBB_EMA_ALPHA * (1.0 - cur["w"])
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


def _saturation_enabled() -> bool:
    """FEAT-2026-06-18 P3 — True iff per-node synaptic scaling is flag-ON (default OFF)."""
    return os.environ.get(HEBB_SATURATE_ENABLED_ENV, "").strip().lower() in (
        "1", "true", "yes", "on")


def _saturation_target() -> float:
    """FEAT-2026-06-18 P3 — the per-node total-outgoing-weight budget (env-overridable)."""
    try:
        raw = os.environ.get(HEBB_SATURATE_ENABLED_ENV + "_TARGET", "")
        return float(raw) if raw.strip() else HEBB_SATURATE_TARGET_DEFAULT
    except (TypeError, ValueError):
        return HEBB_SATURATE_TARGET_DEFAULT


def _apply_synaptic_scaling(weights: dict) -> int:
    """FEAT-2026-06-18 P3 (FLAG-GATED default OFF) — per-node synaptic scaling.

    What: Turrigiano'98 homeostatic synaptic scaling. For each node whose TOTAL
      incident edge weight exceeds the budget T (_saturation_target), multiply all
      its edges by T / total so its outgoing strength is bounded — busy nodes'
      edges then COMPETE (a new strong edge costs the others) instead of every edge
      independently saturating toward 1. Applied multiplicatively in two passes so
      a shared edge sees the min scale of its two endpoints (never scaled above 1).
    Why: the plain EMA toward 1.0 has no per-node budget, so a hub's 32 edges all
      converge near 1 and the weight stops discriminating. A normalized budget keeps
      weights below saturation and below 1, restoring weight as a ranking signal.
    Returns the number of edges scaled. NO-OP (returns 0) unless the flag is on.
    """
    if not _saturation_enabled():
        return 0
    target = _saturation_target()
    if target <= 0:
        return 0
    incident: dict[str, float] = {}
    for k, v in weights.items():
        if "::" not in k:
            continue
        a, b = k.split("::", 1)
        w = float(v.get("w", 0.0))
        incident[a] = incident.get(a, 0.0) + w
        incident[b] = incident.get(b, 0.0) + w
    scale: dict[str, float] = {
        n: (target / tot) for n, tot in incident.items() if tot > target}
    if not scale:
        return 0
    scaled = 0
    for k, v in weights.items():
        if "::" not in k:
            continue
        a, b = k.split("::", 1)
        s = min(scale.get(a, 1.0), scale.get(b, 1.0))
        if s < 1.0:
            v["w"] = float(v.get("w", 0.0)) * s
            scaled += 1
    return scaled


def _decay_and_prune(weights: dict, today: "_dt.date") -> int:
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
    """Read co-activation log, decay weights, and (optionally) promote pairs.

    NOTE: every weight this driver folds, decays, and promotes is a RECALL-graph co-activation
    EDGE weight (edges.db / edge_weights.json) — a retrieval-side association only. T2/T3 nodes
    are admitted to SAM as recall-able READ-ONLY knowledge by design and legitimately co-activate
    in the recall graph; "promote" here means promote a co-activated PAIR into a chain edge — it
    only affects retrieval ranking and never crosses into any model-training path (that separation
    is owned by a training-curriculum boundary enforced elsewhere, not in this recall layer).
    """
    paths = _bio_paths(memory_dir)
    chains_dir = memory_dir / "chains"
    reseed_edge_weights(memory_dir)  # one-time count->w migration (D2); no-op after marker

    # Epiphanies v3 fold (FLAG-GATED, default OFF; FEAT-2026-06-16-memory-episodic-associative-
    # binding). Runs INDEPENDENTLY of the live hebb_log state — it reads its OWN durable archive
    # and is self-throttled — so it must execute even when the live consolidation early-returns
    # (no pending live events) or is cadence-gated. Computed up front so neither early return
    # skips it. Additive: writes only its own epiphanies_* sidecars and touches chains/* ONLY when
    # ASTHENOS_EPI_PROMOTE_TO_LIVE is on (off by default -> no chain writes); fail-soft; no-op disabled.
    epi_summary: dict = {}
    try:
        from samia.core.bio import epiphanies as _epi
        epi_summary = _epi.consolidate(memory_dir)
    except Exception as e:
        epi_summary = {"error": str(e)}

    if not paths["hebb_log"].exists() and not paths["hebb_log_processing"].exists():
        return {"events": 0, "promoted": 0, "pruned": 0, "epiphanies": epi_summary}

    # Cadence gate — decouple the consolidation cadence from the per-tool idle
    # pulse / 600s job. Skip BEFORE the atomic drain so pending appends stay on
    # the live log untouched until the gate next opens (no data lost while gated).
    if _consolidate_cadence_blocked(paths):
        return {"events": 0, "promoted": 0, "pruned": 0,
                "skipped": "cadence_gate", "epiphanies": epi_summary}

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
    # marginals (FEAT-2026-06-18 Phase-2) — load the persistent per-node C(i) + grand
    # total N store, update it as records fold below, and persist after. Tracked on
    # EVERY pass so the significance (lift) gate has data the moment it is flag-enabled.
    marginals = _load_marginal_counts(memory_dir)
    marginals_before = (marginals.get("N", 0), len(marginals.get("counts", {})))
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
                    weights, nodes, src, today, node_appearances, marginals)
    # Persist the marginal store only if it actually advanced this pass (avoid a
    # no-op write when nothing was drained). Fail-soft: a write error never breaks
    # consolidation (the gate simply has slightly stale data next pass).
    if (marginals.get("N", 0), len(marginals.get("counts", {}))) != marginals_before:
        try:
            _save_marginal_counts(memory_dir, marginals)
        except Exception as e:
            print(f"[hebbian] marginal-count save failed: {e}", file=sys.stderr)

    # SYNAPTIC SCALING (FEAT-2026-06-18 P3, FLAG-GATED default OFF) — bound each node's
    # total incident weight to a budget so busy nodes' edges compete instead of all
    # saturating. Runs BEFORE promotion so the gate sees the scaled weights. No-op
    # (returns 0, no mutation) unless ASTHENOS_HEBB_SATURATE is on, so the live weight
    # path is byte-for-byte unchanged by default.
    _apply_synaptic_scaling(weights)

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
        from samia.core import web_store as _ws
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

    # (Epiphanies v3 fold already ran up front — see the top of this function — so it executes
    # on every invocation regardless of the live-log early returns; epi_summary is reused here.)
    return {"events": events, "weights_total": len(weights),
            "promoted": promoted, "pruned_after_decay": True,
            "web": web_stats, "epiphanies": epi_summary}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.hebbian
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): the Hebbian arm carved from the samia.bio monolith
# Layer:      core (pure library, no daemon dependency)
# Role:       the Hebbian co-activation arm — the recall hook hebbian_record, the
#             edge_weights.json read/write + genuine-attractor accounting, the forget /
#             ghost-edge cleanup, the atomic-drain + cadence machinery, the homeostatic
#             in-place update + daily decay/prune, the one-time count->w re-seed, and the
#             hebbian_consolidate driver (drain -> decay -> fold -> within-chain promote
#             -> unified web sync).
# Stability:  stable — the Tier-0 web writer.
# ErrorModel: hebbian_record + the web sync are fail-soft (errors swallowed / logged to
#             stderr, never raised); _load_edge_weights returns {} on a corrupt file;
#             promotion failures print + continue. sweep_ghost_edges is dry-run by
#             default (apply=True is the operator-gated destructive run).
# Depends:    .config (HEBB_* / _bio_paths / _dt / _time / json / os / sys / hashlib /
#             _chain); samia.core.{web_store, temporal_recall_sith} (lazy, function-local).
# Exposes:    hebbian_record, forget_node_weights, sweep_ghost_edges, reseed_edge_weights,
#             hebbian_consolidate (public); _load_edge_weights, _save_edge_weights,
#             _attractor_count, _addr_for_node, _consolidate_cadence_blocked,
#             _record_consolidate_run, _atomic_drain_log, _apply_coactivation,
#             _decay_and_prune, _is_promotable (private, re-exported for tests/importers).
# Note:       PATCH SEAM — hebbian_record is a mock.patch.object(bio, ...) spy target; its
#             ONLY internal caller (replay.replay_engram_traces) reaches it THROUGH the
#             package facade so a facade-level patch is honored. _consolidate_cadence_blocked
#             + _bio_paths + _load_edge_weights are reached by sibling arms (temporal_recall_
#             sith / salience / successor) THROUGH the package facade for the same reason.
# Updated:    2026-06-18 (FEAT-2026-06-18 edge-quality: P2 stop-node exclusion live;
#             Phase-2 marginal-count store + lift/PMI gate behind ASTHENOS_HEBB_LIFT_GATE
#             [default OFF]; P3 per-node synaptic scaling behind ASTHENOS_HEBB_SATURATE
#             [default OFF]).
# Status:     active
# --------------------------------------------------------------------------
