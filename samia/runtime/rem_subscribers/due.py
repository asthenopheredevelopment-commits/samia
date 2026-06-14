"""samia.runtime.rem_subscribers.due — the subscriber due-conditions.

Layer 1 (Owns / Depends):
    Owns:    every REM subscriber's due_condition — the cheap predicates the
             driver calls to decide whether a subscriber genuinely has work this
             cycle (_due_consolidation / _due_tier2_merge / _merge_drainable /
             _due_contradiction_passive / _due_integrity_repair / _due_replay /
             _due_fact_extract) plus the vector-index drift helpers
             (_live_node_count / _vector_index_node_count / _vector_index_drift).
    Depends: .config (_integrity_repair_enabled + the re-exported sleep_pressure
             signal readers), and — lazily, inside each predicate — the existing
             ops whose enable-flag / backlog the due-condition reads
             (merge_consumer.is_enabled/load_candidates, contradiction.passive_has_work,
             fact_extractor.fact_extract_enabled, vector._load_manifest).

Layer 2 (What / Why):
    What: each due_condition reads the SAME backlog signal the op itself drains, so
          a subscriber only runs when it has real work (no wasted REM cycles). The
          gated ops (tier2_merge / contradiction_passive / integrity_repair /
          fact_extract) read their enable flag here too — the DUE half of the
          double gate (the REM gate is the other half, at the subscriber's entry).
    Why:  Q3 / the proposal's P2 — drift-gating keeps every subscriber a no-op when
          it has nothing to do, so a REM cycle with no backlog does little. The
          predicates fail-soft (an import/read error => not due, except replay which
          defaults due since a no-op replay is cheap) so a transient error never
          holds REM awake nor silently starves a due op.
"""

from __future__ import annotations

from pathlib import Path

from .config import _integrity_repair_enabled, sleep_pressure


# ---------------------------------------------------------------------------
# Due-conditions — each reads the SAME backlog sleep_pressure reads, so a
# subscriber only runs when it genuinely has work (no wasted REM cycles).
# ---------------------------------------------------------------------------


def _due_consolidation(mem: Path) -> bool:
    """Consolidation is due iff there are chains to audit (the surfacer's input).

    What: true when a chains/ dir with at least one chain file exists. The
          surfacer (re)produces .consolidation_candidates.json from chains.
    Why:  no chains => nothing to surface; skip the cycle.
    """
    chains = mem / "chains"
    if not chains.is_dir():
        return False
    try:
        return any(chains.glob("*.json"))
    except OSError:
        return False


def _due_tier2_merge(mem: Path) -> bool:
    """P1 dup-merge is due iff there are candidates to drain AND it is ENABLED.

    What: delegates the enable half to merge_consumer.is_enabled
          (ASTHENOS_TIER2_MERGE_ENABLED) and the work half to a non-empty
          .consolidation_candidates.json (the surfacer's output). True only when
          BOTH hold.
    Why:  Q5a — the DOUBLE gate's due half: the merge subscriber never enters the
          run loop unless the operator enabled the feature AND the surfacer has
          surfaced near-dup pairs to drain (the REM gate is the other half).
          Inert by default, mirroring the contradiction passive sweep.
    """
    try:
        from samia.core import merge_consumer as _mc
        if not _mc.is_enabled():
            return False
        return len(_mc.load_candidates(mem)) > 0
    except Exception:
        return False


def _merge_drainable(mem: Path) -> bool:
    """True iff a future machine REM cycle can drain the surfaced near-dup backlog.

    What: True only when the tier2_merge consumer is ENABLED
          (ASTHENOS_TIER2_MERGE_ENABLED via merge_consumer.is_enabled). The
          surfacer produces near-dup candidates; the tier2_merge subscriber drains
          them — but ONLY when merge is enabled. When merge is off the surfaced
          backlog is operator-gated (confirm-via-MCP), not machine-drainable.
    Why:  G2-2026-06-11 — the consolidation surfacer must report cursor-remaining
          work ONLY for the portion a machine cycle can actually drain. Gating the
          surfacer's "remaining" on this lets evaluate() reach REST when the only
          backlog left is operator-gated. Fail-soft to False (treat as NOT
          machine-drainable) so an import/read error never holds REM awake.
    """
    try:
        from samia.core import merge_consumer as _mc
        return bool(_mc.is_enabled())
    except Exception:
        return False


def _due_contradiction_passive(mem: Path) -> bool:
    """P3c is due iff there are nodes to sweep AND contradiction detection is ON.

    What: delegates to contradiction.passive_has_work — True only when
          ASTHENOS_CONTRADICTION_ENABLED is set (gate b) AND nodes/ is non-empty.
    Why:  the DOUBLE gate's due half — the sweep never even enters the run loop
          unless the operator enabled the feature and there is an index to
          reconcile (the REM gate is the other half). Inert by default.
    """
    try:
        from samia.runtime import contradiction as _con
        return _con.passive_has_work(mem)
    except Exception:
        return False


def _due_integrity_repair(mem: Path) -> bool:
    """Integrity consolidation-repair is due iff ENABLED AND there are nodes to repair.

    What: True only when ASTHENOS_INTEGRITY_REPAIR_ENABLED is set (the enable half of
          the double gate) AND nodes/ has at least one .md file (something to consolidate).
    Why:  FEAT-2026-06-07 granular-recall-repaired-decay P2 / Q3a — the subscriber never
          enters the run loop unless the operator enabled integrity repair and there is a
          corpus to heal (the REM gate is the other half). Inert by default, mirroring the
          contradiction passive sweep + tier2 merge.
    """
    if not _integrity_repair_enabled():
        return False
    nodes = mem / "nodes"
    if not nodes.is_dir():
        return False
    try:
        return any(nodes.glob("*.md"))
    except OSError:
        return False


def _due_replay(mem: Path) -> bool:
    """Replay/dreaming is due iff there is graph clutter to reconcile.

    What: true when the coactivation log has depth OR edges-grown-without-
          promotion is non-zero (the two signals replay/hebbian drain), read via
          sleep_pressure's own signal readers (single source of truth).
    Why:  replay is the heaviest op; only run it when there is genuine SWR /
          consolidation work to do.
    """
    try:
        coact, _ = sleep_pressure._read_coactivation_depth(mem)
        edges, _ = sleep_pressure._read_edges_unpromoted(mem)
        return (coact > 0.0) or (edges > 0.0)
    except Exception:
        # If the signal readers are unavailable, default to due (decay/consolidate
        # already ran first; a no-op replay is cheap relative to missing work).
        return True


def _due_fact_extract(mem: Path) -> bool:
    """Fact-extract is due iff the queue has work AND extraction is ENABLED.

    What: True only when ASTHENOS_FACT_EXTRACT_ENABLED is set (the enable half of
          the double gate, via fact_extractor.fact_extract_enabled) AND
          <mem>/.fact_extract_queue.jsonl exists with >= 1 line (the work half).
    Why:  FEAT-2026-06-10 P1 / Q4c — the DOUBLE gate's due half: the subscriber
          never enters the run loop unless the operator enabled extraction AND a
          producer has queued text (the REM gate is the other half; the drain
          re-checks the flag at its own entry — the single-layer gate the tree
          prefers). Cheap: a flag read short-circuits before any I/O when off.
          Inert by default, mirroring tier2_merge / contradiction_passive.
    """
    try:
        from samia.core import fact_extractor
        if not fact_extractor.fact_extract_enabled():
            return False
    except Exception:
        return False
    q = mem / ".fact_extract_queue.jsonl"
    if not q.exists():
        return False
    try:
        with q.open("rb") as f:
            return any(line.strip() for line in f)
    except OSError:
        return False


def _live_node_count(mem: Path) -> int:
    """Count live nodes/*.md files (the index's intended size). 0 on a missing dir."""
    nodes = mem / "nodes"
    if not nodes.is_dir():
        return 0
    try:
        return sum(1 for _ in nodes.glob("*.md"))
    except OSError:
        return 0


def _vector_index_node_count(mem: Path) -> int | None:
    """The vector index's recorded node_count (None when no index exists yet)."""
    try:
        from samia.core import vector as _vec
        m = _vec._load_manifest(mem)
    except Exception:
        return None
    nc = m.get("node_count") if isinstance(m, dict) else None
    return int(nc) if isinstance(nc, int) else None


def _vector_index_drift(mem: Path) -> bool:
    """due_condition: the vector index has drifted from the live node set.

    What: True iff the index's recorded node_count != the live nodes/*.md count
          (a cheap count comparison — NO per-node hashing), OR no index exists yet
          but live nodes do (the index must be built). The incremental build then
          re-syncs (manifest-cached, embeds only new/changed). This is the standing
          due-signal — drift is the only thing that warrants the (bounded) build.
    Why:  G4-2026-06-11 (operator choice 4a) — the index drifted because nothing
          rebuilt it automatically (today's atoms only entered via a MANUAL rebuild).
          Gating on a count-mismatch keeps the subscriber a no-op when the index is
          already in sync, so it never burns a REM cycle needlessly.
    """
    live = _live_node_count(mem)
    if live == 0:
        return False  # nothing to index -> not due (no thrash on an empty corpus)
    indexed = _vector_index_node_count(mem)
    if indexed is None:
        return True  # live nodes exist but no index -> due (first build)
    return indexed != live


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_subscribers.due
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_subscribers monolith
#             during modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the DUE half of every REM subscriber — the cheap drift/backlog
#             predicates the driver calls before running a subscriber, plus the
#             vector-index drift helpers. The gated ops read their enable flag
#             here too (the due half of each double gate).
# Stability:  stable — pure predicates; the carve preserved every gate (the
#             enable-flag checks, the cosine/jaccard-free count comparison, the
#             replay-defaults-due fallback) byte-identical to the monolith.
# ErrorModel: fail-soft — an import/read error => not due (so a transient fault
#             never starves a due op nor holds REM awake), EXCEPT _due_replay which
#             defaults due (a no-op replay is cheap vs missing work).
# Depends:    pathlib (stdlib). .config (_integrity_repair_enabled, sleep_pressure).
#             samia.core.{merge_consumer,fact_extractor,vector} +
#             samia.runtime.contradiction (all lazy, inside the predicate).
# Exposes:    _due_consolidation, _due_tier2_merge, _merge_drainable,
#             _due_contradiction_passive, _due_integrity_repair, _due_replay,
#             _due_fact_extract, _live_node_count, _vector_index_node_count,
#             _vector_index_drift.
# Lines:      252
# ─────────────────────────────────────────────
