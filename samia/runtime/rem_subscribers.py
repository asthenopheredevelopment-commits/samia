"""samia.runtime.rem_subscribers — register the offline ops as REM subscribers.

Layer 1 (Owns / Depends):
    Owns:    the MIGRATION wiring (P2) — the mapping of each existing offline
             memory op to a REM subscriber (name, priority, due_condition,
             cursor_key) via rem_cycle.register_offline_op, the two BATCH
             wrappers that turn per-text / per-call primitives into REM-runnable
             ticks (consolidation surfacer, fact-extract batch), and the gated
             entry of those wrappers. Owns NO op internals — it wraps/registers
             the existing functions, never rewrites them.
    Depends: samia.runtime.rem_cycle (register_offline_op / the gate /
             read_cursor / write_cursor), and the existing offline ops:
             samia.core.context_extension.idle_replay_tick (replay/dreaming +
             hebbian_consolidate + reseed — already gated at its own entry),
             samia.core.consolidation.audit_all/surface (the near-dup surfacer),
             samia.core.fact_extractor.extract_atoms (the per-text primitive).
             NOTE: tier.decay_tick is NOT here — decay is the short-term
             forgetting curve and runs CONTINUOUSLY across wake+REM (driven by
             the idle_pulse "decay" subscriber, NOT REM-gated; see below).

Layer 2 (What / Why):
    What: REM P2's migration. It registers the heavy offline reconciliation ops
          as REM subscribers so the driver (rem_cycle.run_due_subscribers) runs
          them — and ONLY runs them — inside the sleep window, in priority order,
          each with a due-condition so a cycle with no real backlog does little.
          The two ops that are NOT already tick-shaped get thin batch wrappers:
            - consolidation surfacer: audit_all -> surface (the 600-pair
              .consolidation_candidates.json producer) had NO live idle/scheduler
              caller; this is its first scheduled home (a REM subscriber).
            - fact-extract batch: extract_atoms is a per-text PRIMITIVE (no mem
              arg, not wired anywhere); this wraps it as a batch offline op that
              drains a pending-extraction queue (cursor-checkpointed). It is a
              NEW build (nothing to migrate-away), wired only when a queue exists.
    Why:  Q3 / the proposal's P2 — the heavy STRENGTHENING/ABSTRACTING offline
          ops stop trickling on every idle pulse and run behind REM instead.
          Migrating PRECISELY (gate at each op's own entry + register here)
          means the op refuses outside REM no matter WHICH caller invokes it
          (idle_pulse, the dormant scheduler.py, a future caller), and the
          registry is the operator-visible inventory.

CLS rationale — what IS and IS NOT REM-gated (operator correction 2026-06-07):
    REM-gated (sleep = CONSOLIDATION + REPLAY, strengthening/abstracting):
      consolidation, contradiction_passive, replay/dreaming, fact_extract.
    NOT REM-gated — runs CONTINUOUSLY across BOTH wake and REM:
      decay (tier.decay_tick). Decay is the short-term forgetting curve; sleep
      does NOT do the forgetting. It is driven solely by the idle_pulse "decay"
      subscriber on its 6h cadence (NOT a REM subscriber here, so no double-
      drive). It was wrongly REM-gated by P2; that gate has been removed.

Priorities (LOW runs FIRST):
    20  consolidation            — surface near-dup candidates.
    22  tier2_merge              — FEAT-2026-06-07 P1+P2: pick-winner dup-merge —
                                   DRAIN the surfacer's near-dup backlog (AUTO
                                   merge true dups via the RESTORABLE supersede
                                   path + provenance edge), THEN P2 SYNTHESIZE
                                   (propose, NOT apply) abstractions for the
                                   distinct-but-overlapping pairs (operator-gated
                                   confirm via MCP). Runs right after the surfacer
                                   produces the candidates, before the passive
                                   sweep.
    25  contradiction_passive    — FEAT-2026-06-07 P3c: incremental whole-index
                                   supersession sweep (cosine + LLM judge ->
                                   auto-supersede the loser, RESTORABLE). Runs
                                   before the heaviest replay op.
    28  integrity_repair         — FEAT-2026-06-07 granular-recall-repaired-decay
                                   P2: CONSOLIDATION integrity-repair — PARTIALLY
                                   heal (anchor-first, strength<1.0) the integrity
                                   of a budgeted, cursor-tracked slice of eroded
                                   nodes. Double-gated (REM + INTEGRITY_REPAIR_
                                   ENABLED), inert by default. Sleep heals what it
                                   consolidates.
    29  vector_maintenance       — G4-2026-06-11: keep the vector index in sync with
                                   the live node set. Runs vector.build incrementally
                                   (manifest-cached) every cycle the index has drifted
                                   (manifest node_count != live .md count) + a full
                                   rebuild on a long cadence (ASTHENOS_VECTOR_FULL_
                                   REBUILD_S, default 7 days, cursor-tracked).
    30  replay/dreaming          — SWR cross-chain edges + hebbian consolidate.
    40  fact_extract             — drain any pending fact-extraction queue (new).

PRODUCE-ONLY: importing this module does nothing; registration runs only when
register_rem_subscribers() is called (by the daemon at startup, operator-gated).
No thread, no timer, no live mutation here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from samia.runtime import rem_cycle, sleep_pressure

_log = logging.getLogger("samia.runtime.rem_subscribers")

# ASTHENOS_INTEGRITY_REPAIR_ENABLED — What: the enable flag for the P2 integrity
#   consolidation-repair subscriber (the CONSOLIDATION repair trigger). Default OFF.
# Why: FEAT-2026-06-07 granular-recall-repaired-decay P2 / Q3a — sleep PARTIALLY heals
#   the integrity of the nodes it consolidates. Double-gated like the other P2 ops:
#   REM (the subscriber gate) + this enable flag, both inert by default (produce-only).
INTEGRITY_REPAIR_ENABLED_ENV = "ASTHENOS_INTEGRITY_REPAIR_ENABLED"


def _integrity_repair_enabled() -> bool:
    """Live read of the ASTHENOS_INTEGRITY_REPAIR_ENABLED master switch (default OFF).

    Delegates to integrity.repair_enabled() so the recall-repair seam (memory_search) and
    this P2 consolidation-repair subscriber share ONE reader of ONE flag (the activation
    wiring exposed the same flag at core level). Falls back to a direct env read if the
    core import is unavailable, preserving the prior behavior + the same env var name.
    """
    try:
        from samia.core import integrity as _integrity
        return _integrity.repair_enabled()
    except Exception:
        return os.environ.get(INTEGRITY_REPAIR_ENABLED_ENV, "0") == "1"

# Priority bands (LOW runs FIRST). The P3-passive contradiction sweep slots at
# 25, between consolidation (20) and replay (30) per the reserved placeholder.
# (No decay band: decay is NOT a REM subscriber — it runs continuously across
# wake+REM via the idle_pulse "decay" subscriber. See module CLS rationale.)
PRIO_CONSOLIDATION = 20
PRIO_TIER2_MERGE = 22  # FEAT-2026-06-07 P1 (pick-winner dup-merge drain)
PRIO_CONTRADICTION_PASSIVE = 25  # FEAT-2026-06-07 P3c
PRIO_INTEGRITY_REPAIR = 28  # FEAT-2026-06-07 granular-recall-repaired-decay P2
PRIO_VECTOR_MAINTENANCE = 29  # G4-2026-06-11: keep the vector index in sync with nodes/
PRIO_REPLAY = 30
PRIO_FACT_EXTRACT = 40

# ASTHENOS_VECTOR_FULL_REBUILD_S — What: the cadence (seconds) between FULL vector
#   index rebuilds (rebuild=True, re-embeds every node). Default 7 days. The
#   incremental build (manifest-cached, embeds only new/changed nodes) runs every
#   cycle on drift; the full rebuild is the periodic floor-sweep that re-embeds the
#   whole corpus (catches model/content drift the sha256 cache cannot see).
# Why: G4-2026-06-11 (operator choice 4a) — the index drifted because nothing
#   rebuilt it automatically. The incremental path keeps it fresh cheaply; the full
#   rebuild is bounded to a long cadence so it never thrashes a REM cycle.
VECTOR_FULL_REBUILD_S_ENV = "ASTHENOS_VECTOR_FULL_REBUILD_S"
_VECTOR_FULL_REBUILD_S_DEFAULT = 7 * 24 * 3600  # 7 days


def _vector_full_rebuild_interval_s() -> float:
    """The full-rebuild cadence in seconds (env-overridable, 7-day default).

    Fail-soft: a malformed env value falls back to the 7-day default.
    """
    raw = os.environ.get(VECTOR_FULL_REBUILD_S_ENV)
    if raw is None:
        return float(_VECTOR_FULL_REBUILD_S_DEFAULT)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(_VECTOR_FULL_REBUILD_S_DEFAULT)

# Per-cycle budget for the integrity consolidation-repair subscriber (cursor-tracked).
_INTEGRITY_REPAIR_BUDGET = 50

# Per-cycle drain budget for the tier2_merge subscriber (cursor-tracked across
# cycles). Bounded so a large backlog does not stall a single REM cycle.
_TIER2_MERGE_BUDGET = 50

# Near-dup surfacing threshold for the consolidation subscriber (the surfacer's
# own knee; matches consolidation.DEFAULT_THRESHOLD intent — kept explicit so a
# tune does not silently change behavior).
_CONSOLIDATION_THRESHOLD = 0.15


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


# ---------------------------------------------------------------------------
# Subscriber callables — wrap/adapt existing ops to the fn(mem) -> dict shape.
# Each returns a dict carrying "work_remaining" so the driver can OR it into
# evaluate()'s work_remains. None of these rewrites an op's internals.
# ---------------------------------------------------------------------------


def _sub_consolidation(mem: Path) -> dict[str, Any]:
    """REM subscriber: near-dup consolidation surfacer (the 600-pair backlog).

    What: gated batch wrapper around consolidation.audit_all -> surface. Refuses
          outside REM (logged), then re-audits chains for near-dup pairs and
          writes .consolidation_candidates.json. Checkpoints a cursor recording
          the candidate count so an interrupt + the work_remains signal reflect
          the backlog.
    Why:  this surfacer had NO live scheduled caller; REM is its first home
          (Q3 / proposal P2). It is the producer of the Tier-2 abstractive
          backlog; draining/merging the surfaced pairs is a later subscriber.
    """
    if not rem_cycle.gate_offline_op(Path(mem), "consolidation_surface"):
        return {"fired": False, "refused": "not_in_rem"}
    from samia.core import consolidation
    findings = consolidation.audit_all(Path(mem), threshold=_CONSOLIDATION_THRESHOLD)
    consolidation.surface(Path(mem), findings, _CONSOLIDATION_THRESHOLD)
    n = len(findings)
    # Cursor (G2-2026-06-11, MACHINE-DRAINABLE ONLY): the surfacer is a single-pass
    # PRODUCER — it regenerates the whole candidate file and drains nothing itself, so
    # its OWN remaining work is always 0. The surfaced pairs are drainable by the
    # tier2_merge subscriber ONLY when the operator has enabled merge
    # (ASTHENOS_TIER2_MERGE_ENABLED); when merge is OFF the surfaced backlog is
    # OPERATOR-GATED (confirm-via-MCP), so it must NOT count as cursor-remaining work
    # that holds REM awake. We therefore report "remaining" as the surfaced count ONLY
    # when a machine cycle (tier2_merge) can actually drain it; otherwise 0. (The raw
    # surfaced count is still carried as "surfaced" telemetry for observability.)
    machine_drainable = _merge_drainable(Path(mem))
    remaining = n if machine_drainable else 0
    rem_cycle.write_cursor(Path(mem), "consolidation",
                           {"surfaced": n, "remaining": remaining,
                            "done": True})  # surfacing itself is done this pass
    return {"fired": True, "surfaced": n,
            "operator_gated_backlog": (n if not machine_drainable else 0),
            "work_remaining": False}


def _sub_tier2_merge(mem: Path) -> dict[str, Any]:
    """REM subscriber: Tier-2 pick-winner dup-merge — DRAIN the near-dup backlog.

    What: gated batch wrapper around merge_consumer.drain — refuses outside REM
          (logged), then drains ONE budgeted, cursor-tracked slice of
          .consolidation_candidates.json: AUTO pick-winner-merges the true-dup
          (high-similarity) pairs via the RESTORABLE supersede path + provenance
          edge, records below-bar distinct pairs for P2, and removes every
          dispatched pair from the candidate file so the backlog SHRINKS. drain
          itself is a no-op unless ASTHENOS_TIER2_MERGE_ENABLED is set (the
          enable half of the double gate). Checkpoints the cursor (key
          "tier2_merge") with the remaining count so an interrupt resumes mid-
          batch and work_remaining reflects the real backlog. Then (P2)
          SYNTHESIZES — PROPOSES, never applies — abstractions for the queued
          'abstract' pairs via merge_consumer.synthesize_pending; the operator
          CONFIRM is MCP-only (memory_confirm_merge), never automatic here.
    Why:  FEAT-2026-06-07 P1+P2 / Q3a/Q2c — P1 is the missing DRAIN. The surfacer
          (priority 20) re-produces the candidates each cycle; this consumer
          (priority 22) drains them in the SAME cycle so work_remaining can
          finally go false and REM reach REST. P2 then proposes (operator-gated)
          abstractions for the distinct-but-overlapping minority. Wraps the
          consumer; rewrites no op internals (pick-winner, the restorable forget,
          the provenance edge, the synthesis entrypoint, and the cursor helpers
          all already exist).
    """
    if not rem_cycle.gate_offline_op(Path(mem), "tier2_merge"):
        return {"fired": False, "refused": "not_in_rem"}
    from samia.core import merge_consumer as _mc
    cur = rem_cycle.read_cursor(Path(mem), "tier2_merge")
    start = int(cur.get("index", 0)) if isinstance(cur, dict) else 0
    res = _mc.drain(Path(mem), budget=_TIER2_MERGE_BUDGET, cursor=start)
    rem_cycle.write_cursor(Path(mem), "tier2_merge", {
        "index": res.get("cursor", 0),
        "remaining": res.get("remaining", 0),
        "done": not res.get("work_remaining", False),
    })
    # FEAT-2026-06-07 P2 — after the P1 dup drain, SYNTHESIZE (propose, NOT apply)
    # abstractions for the 'abstract' pairs P1 queued. PROPOSE-only: a draft is
    # surfaced for operator confirm via the MCP merge surface; nothing is created
    # or superseded automatically. Double-gated (REM here + enable flag inside
    # synthesize_pending) AND inference-availability gated (a no-op leaving the
    # pair pending when the synthesis backend is off — same posture as the judge).
    synth = _mc.synthesize_pending(Path(mem), budget=_TIER2_MERGE_BUDGET)
    return {"fired": True, **res, "synthesis": synth}


def _sub_contradiction_passive(mem: Path) -> dict[str, Any]:
    """REM subscriber: PASSIVE supersession sweep (FEAT-2026-06-07 P3c).

    What: gated wrapper around contradiction.passive_sweep — refuses outside REM
          (logged), then runs ONE budgeted, cursor-tracked slice of the whole
          nodes/ index: cosine candidate-find (scope=None) + LLM judge ->
          auto-supersede the loser via the RESTORABLE forget path; weaker /
          unjudged hits recorded (mode="passive"). passive_sweep itself reads +
          checkpoints its cursor (key "contradiction_passive") and is a no-op
          unless ASTHENOS_CONTRADICTION_ENABLED is set (gate b).
    Why:  Q3 + the Q4 override. The passive arm is the exhaustive global
          reconciler; it lives behind REM (this subscriber) AND behind the enable
          flag (passive_sweep's own guard) — both inert by default. Wraps the op,
          never rewrites it (the cosine finder, judge, restorable forget, store,
          and cursor helpers all already exist).
    """
    if not rem_cycle.gate_offline_op(Path(mem), "contradiction_passive"):
        return {"fired": False, "refused": "not_in_rem"}
    from samia.runtime import contradiction as _con
    res = _con.passive_sweep(Path(mem))
    return {"fired": True, **res}


def _sub_integrity_repair(mem: Path) -> dict[str, Any]:
    """REM subscriber: CONSOLIDATION integrity-repair — sleep heals what it touches.

    What: gated wrapper around integrity.consolidation_repair_pass — refuses outside
          REM (logged), no-ops unless ASTHENOS_INTEGRITY_REPAIR_ENABLED is set (the
          enable half), then PARTIALLY repairs (strength < 1.0, anchor-first) the
          integrity of a budgeted, cursor-tracked slice of eroded nodes. Checkpoints
          the cursor (key "integrity_repair") so an interrupt resumes mid-pass and
          work_remaining reflects whether a full pass has wrapped.
    Why:  FEAT-2026-06-07 granular-recall-repaired-decay P2 / Q3a — CONSOLIDATION is a
          PARTIAL repair trigger (distinct from RECALL, which is full): a REM pass heals
          a little of the integrity of the nodes it consolidates. Double-gated (REM here
          + the enable flag), inert by default. Wraps the pure pass; rewrites nothing
          (the partial-repair primitive + cursor helpers already exist). Anchor-first
          only — NO generative repair (that is P3).
    """
    if not rem_cycle.gate_offline_op(Path(mem), "integrity_repair"):
        return {"fired": False, "refused": "not_in_rem"}
    if not _integrity_repair_enabled():
        return {"fired": False, "refused": "not_enabled", "work_remaining": False}
    from samia.core import integrity as _integrity
    cur = rem_cycle.read_cursor(Path(mem), "integrity_repair")
    start = int(cur.get("index", 0)) if isinstance(cur, dict) else 0
    res = _integrity.consolidation_repair_pass(
        Path(mem), budget=_INTEGRITY_REPAIR_BUDGET, cursor=start)
    rem_cycle.write_cursor(Path(mem), "integrity_repair", {
        "index": res.get("cursor", 0),
        "remaining": res.get("work_remaining", False),
        "done": not res.get("work_remaining", False),
    })
    return {"fired": True, **res}


def _sub_vector_maintenance(mem: Path) -> dict[str, Any]:
    """REM subscriber: keep the vector index in sync with the live node set (G4).

    What: gated wrapper around samia.core.vector.build. Refuses outside REM (logged).
          Runs an INCREMENTAL build (rebuild=False) each cycle the index has drifted
          (vector.build is manifest-cached: it embeds ONLY new/changed nodes, so the
          common case is cheap). On a long cadence (ASTHENOS_VECTOR_FULL_REBUILD_S,
          default 7 days, cursor-tracked under key "vector_maintenance") it instead
          runs a FULL build (rebuild=True) that re-embeds the whole corpus. The cursor
          records last_full_ts so the cadence survives restarts. Reports
          work_remaining=False (a build either fully syncs the index or is a no-op —
          there is no resumable backlog the next cycle must finish).
    Why:  G4-2026-06-11 (operator choice 4a) — the index drifted because NOTHING
          rebuilt it automatically (today's 5.8k atoms entered only via a manual
          rebuild). This is the automatic maintenance loop the operator said "that's
          where it was intended": incremental on drift every cycle + a periodic full
          rebuild. Wraps the existing build; rewrites no vector internals.
    """
    if not rem_cycle.gate_offline_op(Path(mem), "vector_maintenance"):
        return {"fired": False, "refused": "not_in_rem"}
    import time as _time
    from samia.core import vector as _vec

    cur = rem_cycle.read_cursor(Path(mem), "vector_maintenance")
    last_full = float(cur.get("last_full_ts", 0.0)) if isinstance(cur, dict) else 0.0
    now = _time.time()
    interval = _vector_full_rebuild_interval_s()
    do_full = (now - last_full) >= interval

    if do_full:
        manifest = _vec.build(Path(mem), rebuild=True)
        last_full = now
        mode = "full_rebuild"
    else:
        manifest = _vec.build(Path(mem), rebuild=False)
        mode = "incremental"

    node_count = int(manifest.get("node_count", 0)) if isinstance(manifest, dict) else 0
    rem_cycle.write_cursor(Path(mem), "vector_maintenance", {
        "last_full_ts": last_full,
        "last_run_ts": now,
        "last_mode": mode,
        "node_count": node_count,
        # No resumable backlog: a build syncs the whole index in one call.
        "remaining": False,
        "done": True,
    })
    return {"fired": True, "mode": mode, "node_count": node_count,
            "work_remaining": False}


def _sub_replay(mem: Path) -> dict[str, Any]:
    """REM subscriber: replay/dreaming + hebbian consolidate.

    What: calls the existing context_extension.idle_replay_tick (which is gated
          on is_rem at its own entry and runs replay_sweep +
          replay_sweep_interleaved + replay-coactivation reseed +
          hebbian_consolidate). Returns its telemetry + a work_remaining derived
          from the coactivation/edge signals.
    Why:  migration without rewrite — the compound op keeps its internal cadence
          (idle_replay 30s gate) and hebbian's HEBB_MIN_INTERVAL gate; the REM
          gate is additive. The registry just schedules it inside REM.
    """
    from samia.core.context_extension import idle_replay_tick
    res = idle_replay_tick(mem)
    coact, _ = sleep_pressure._read_coactivation_depth(mem)
    edges, _ = sleep_pressure._read_edges_unpromoted(mem)
    return {**(res if isinstance(res, dict) else {"result": res}),
            "work_remaining": (coact > 0.0) or (edges > 0.0)}


def _sub_fact_extract(mem: Path, batch: int = 20) -> dict[str, Any]:
    """REM subscriber: batch fact extraction (drain → semantic nodes, FLAG-GATED).

    What: gated wrapper that, WHEN ASTHENOS_FACT_EXTRACT_ENABLED=1, drains up to
          ``batch`` records from <mem>/.fact_extract_queue.jsonl (the new producer
          fills it at freeze + merge-abstract), runs fact_extractor.extract_atoms
          on each text via the cached BitNet-2B backend (the judge-rewire seam),
          and PERSISTS each atom as a full-citizen ``type: semantic`` node:
            (a) DEDUP — skip an atom whose cosine vs the existing index is >= 0.92
                (contradiction.find_contradiction_candidates nonempty = dup);
            (b) PERSIST — frontmatter.write_node nodes/sem_<slug>_<hash>.md (auto-
                anchored by capture_on_genuine_write);
            (c) PROVENANCE — a web_store edge atom->source (ref_kind='provenance');
            (d) MINI-CHAIN — upsert chains/fx_<source-stem>.json with the source
                node + its atoms (>= 2 members) so production chainogram (which
                excludes singletons) can load gist alongside episode.
          Drained records are REMOVED (the remaining slice rewritten atomically);
          queue-consumption IS the cursor. FAIL-SOFT: no real backend leaves every
          item in the queue (work_remaining). When the flag is OFF the queue is
          left UNTOUCHED and the body returns {ran:False, reason:'disabled'} —
          byte-identical no-op.
    Why:  FEAT-2026-06-10 P1 / Q2a+Q3a+Q5a. extract_atoms had no caller and the
          queue had no producer, so this arm was perpetually inert. Atoms are
          ADDITIVE date-stamped facts (the lever the benchmark flagged); nothing
          here deletes/archives/supersedes the source (keep+link). Double-gated:
          the REM gate (below) + the fact_extract_enabled() entry gate (the tree's
          preferred single-layer flag-gate, mirroring tier.decay_tick).
    """
    if not rem_cycle.gate_offline_op(Path(mem), "fact_extract"):
        return {"fired": False, "refused": "not_in_rem"}
    import json as _json
    from samia.core import fact_extractor
    # ENTRY GATE (single layer): flag-off leaves the queue UNTOUCHED. No read, no
    # rewrite, no cursor write — a byte-identical no-op (FEAT-2026-06-10 P1, Q4c).
    if not fact_extractor.fact_extract_enabled():
        return {"fired": False, "ran": False, "reason": "disabled",
                "work_remaining": False}

    q = Path(mem) / ".fact_extract_queue.jsonl"
    if not q.exists():
        return {"fired": False, "extracted": 0, "work_remaining": False}
    try:
        lines = [l for l in q.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    except OSError:
        return {"fired": False, "extracted": 0, "work_remaining": False}

    # FAIL-SOFT backend gate: a missing/mock backend leaves the queue intact so a
    # later cycle with a real model still drains it (mirrors the judge posture).
    backend = _fact_extract_backend()
    if backend is None:
        return {"fired": False, "extracted": 0, "reason": "no_backend",
                "remaining": len(lines), "work_remaining": len(lines) > 0}

    # BUDGET: at most ``batch`` (<= 20) records per call; the rest stay queued.
    take, leave = lines[:batch], lines[batch:]
    extracted = persisted = deduped = 0
    for raw in take:
        try:
            rec = _json.loads(raw)
        except _json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        text = rec.get("text")
        if not text:
            continue
        source = rec.get("source")  # may be absent (old {"text"}-only records)
        # Pass the configured local backend OBJECT (not a string): extract_atoms
        # routes a duck-typed .chat/.complete object through the local model.
        # Passing a string here (the old getattr(backend,'name','auto')) yielded
        # 'auto' -> anthropic-if-key-else-rule — the configured model NEVER
        # generated atoms (FIX-2026-06-10, HIGH). llm_only=True (TUNE-2026-06-10):
        # NEVER persist rule-splitter chunks as semantic nodes — LLM atoms or
        # nothing; a no-atom source stays unstamped and retryable.
        atoms = fact_extractor.extract_atoms(
            text, backend=backend, chains_hint=rec.get("chains"),
            llm_only=True)
        if not atoms:
            # Extraction FAILED (no atoms) — the content is NOT yet semantically
            # covered, so do NOT stamp the source distilled (TUNE-2026-06-10 c).
            continue
        extracted += len(atoms)
        res = _persist_atoms(Path(mem), atoms, source)
        persisted += res["persisted"]
        deduped += res["deduped"]
        # TUNE-2026-06-10 operator decision (c): the episodic trace fades only AFTER
        # the semantic representation forms. This record was processed SUCCESSFULLY
        # (extraction ran AND >= 1 atom persisted OR all atoms were dedup-skipped —
        # both mean the source's content is semantically covered), so stamp the live
        # SOURCE node distilled:true to UNLOCK its (frozen) integrity erosion gate.
        if res["persisted"] >= 1 or res["deduped"] >= 1:
            _fx_stamp_distilled(Path(mem), source)

    # Rewrite the queue with the remainder (queue-consumption IS the cursor).
    remaining = len(leave)
    if leave:
        q.write_text("\n".join(leave) + "\n", encoding="utf-8")
    else:
        try:
            q.unlink()
        except OSError:
            q.write_text("", encoding="utf-8")
    rem_cycle.write_cursor(Path(mem), "fact_extract",
                           {"remaining": remaining, "done": remaining == 0})
    return {"fired": True, "ran": True, "extracted": extracted,
            "persisted": persisted, "deduped": deduped, "remaining": remaining,
            "work_remaining": remaining > 0}


def _fact_extract_backend() -> Any:
    """The cached BitNet-2B backend the drain's extract_atoms rides (fail-soft).

    What: build (once, path-cached) a backend for ASTHENOS_FACT_EXTRACT_MODEL via
          inference.get_backend_for_model — the SAME cached seam the contradiction
          judge uses (contradiction._judge_backend). Returns None when the factory
          is unavailable or the result is a MockBackend (no real model), so the
          drain fail-softly leaves the queue intact.
    Why:  FEAT-2026-06-10 P1 / Q4c — the producer/drain must not load a second copy
          of a model nor block when no model is configured. Mirrors the judge's
          dedicated-cached-small-backend pattern exactly.
    """
    try:
        from samia.runtime import inference as _inf
        from samia.core import fact_extractor
    except Exception:
        return None
    factory = getattr(_inf, "get_backend_for_model", None)
    if factory is None:
        return None
    try:
        backend = factory(fact_extractor.fact_extract_model())
    except Exception:
        return None
    if backend is None or type(backend).__name__ == "MockBackend":
        return None
    return backend


def _persist_atoms(mem: Path, atoms: list[dict], source: Any) -> dict[str, int]:
    """Persist extracted atoms as semantic nodes (dedup → write → prov → chain).

    What: for each atom: (a) DEDUP vs the existing index (cosine >= 0.92 via
          contradiction.find_contradiction_candidates) — skip dups; (b) PERSIST a
          new nodes/sem_<slug>_<shorthash>.md with fm {name, description (atom
          text[:60]), type: semantic, source, extracted_by: fact_extract} through
          frontmatter.write_node (auto-anchored); (c) PROVENANCE a web_store edge
          atom->source (ref_kind='provenance', the merge_consumer P1 pattern);
          (d) MINI-CHAIN upsert chains/fx_<source-stem>.json over the source node +
          its atoms (>= 2 members, else skip — singletons are pointless and
          invisible to production chainogram anyway).
    Why:  FEAT-2026-06-10 P1 / Q2a+Q5a — atoms must be full citizens (indexed,
          contradiction-scoped, chain-loadable), deduped so near-dup spam never
          lands, and lineage-linked back to the source (keep+link, never delete).
    """
    import hashlib
    from samia.core import frontmatter as _fm
    try:
        from samia.runtime import contradiction as _con
    except Exception:
        _con = None
    try:
        from samia.core import chain as _chain
    except Exception:
        _chain = None

    nodes_dir = Path(mem) / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    chains_dir = Path(mem) / "chains"

    src_id = str(source) if source else None
    src_fname = None
    if src_id:
        src_fname = src_id if src_id.endswith(".md") else f"{src_id}.md"

    persisted = 0
    deduped = 0
    atom_members: list[dict] = []

    for a in atoms:
        atom_text = (a.get("body") or a.get("description")
                     or a.get("title") or "").strip()
        if not atom_text:
            continue
        # (a) DEDUP — a nonempty candidate set at the 0.92 bar = a near-duplicate
        # already in the index; skip the atom (no low-quality near-dup spam).
        if _con is not None:
            try:
                dup = _con.find_contradiction_candidates(
                    atom_text, memory_dir=Path(mem), threshold=0.92)
            except Exception:
                dup = []
            if dup:
                deduped += 1
                continue

        # (b) PERSIST — a full-citizen semantic node (auto-anchored by write_node).
        slug = _fe_slug(a.get("title") or a.get("description") or atom_text)
        short = hashlib.sha1(atom_text.encode("utf-8")).hexdigest()[:8]
        name = f"sem_{slug}_{short}"
        path = nodes_dir / f"{name}.md"
        if path.exists():
            # Same atom text already persisted under this source slice; treat as
            # a dedup (idempotent re-drain) rather than overwriting.
            deduped += 1
            continue
        # NOTE: do NOT stamp `runtime` here — write_node validates runtime ∈
        # {opencode, main} (the harness-provenance field) and rejects "rem".
        # Readers default a missing runtime to "main", which is correct: these
        # atoms are produced by the main daemon's REM cycle.
        # Lifecycle stamps (BUG-2026-06-11, deep-exam finding #1): every sibling
        # writer stamps last_access at write; omitting it makes tier._days_since
        # read 9999 -> the atom enters the STALE->0 relevance sink and demotes
        # to cold on the FIRST decay tick, where the 2.5x erosion factor +
        # capped 4.0 recency multiplier erode it at 0.20/pass — 10x faster than
        # the protected frozen-distilled nodes. Fresh stamps put new atoms in
        # the warm mean-revert regime like every other genuine write.
        import datetime as _dt
        fm = {
            "name": a.get("title") or atom_text[:60],
            "description": atom_text[:60],
            "type": "semantic",
            "source": src_id or "",
            "extracted_by": "fact_extract",
            "last_access": _dt.date.today().isoformat(),
            "relevance": 0.5,
            "tier": "warm",
            "material_grade": "enriched",
        }
        order = ["name", "description", "type", "source", "extracted_by",
                 "last_access", "relevance", "tier", "material_grade"]
        try:
            _fm.write_node(path, fm, order, atom_text + "\n")
        except Exception:
            continue
        persisted += 1
        atom_members.append({"addr": short, "file": f"nodes/{name}.md",
                             "tier": "warm"})

        # (c) PROVENANCE — edge atom -> source (mirror merge_consumer P1 exactly).
        if src_fname:
            _fx_provenance_edge(f"{name}.md", src_fname)

    # (d) MINI-CHAIN — source + its atoms, >= 2 members else skip.
    if _chain is not None and atom_members:
        members: list[dict] = []
        if src_fname:
            src_stem = Path(src_fname).stem
            # The source node leads the chain when it still exists (frozen sources
            # are gone — then the chain holds atoms only, per the spec).
            if (nodes_dir / src_fname).exists():
                members.append({"addr": f"src-{src_stem}",
                                "file": f"nodes/{src_fname}",
                                "tier": "warm"})
        members.extend(atom_members)
        if len(members) >= 2:
            chains_dir.mkdir(parents=True, exist_ok=True)
            src_stem = Path(src_fname).stem if src_fname else "merge"
            cname = f"fx_{src_stem}"
            try:
                existing = _chain.load_chain(chains_dir, cname)
            except (SystemExit, FileNotFoundError):
                existing = None
            if isinstance(existing, dict):
                # Upsert: extend members (dedup by addr), keep schema fields.
                seen = {m.get("addr") for m in existing.get("members") or []}
                for m in members:
                    if m.get("addr") not in seen:
                        existing.setdefault("members", []).append(m)
                        seen.add(m.get("addr"))
                data = existing
            else:
                data = {
                    "chain_id": cname,
                    "head_address": members[0]["addr"],
                    "tail_address": members[-1]["addr"],
                    "members": members,
                    "total_relevance": 0.5,
                    "last_traversal": _dt_today(),
                    "compressed": False,
                    "edges": [],
                }
            data["tail_address"] = data["members"][-1]["addr"]
            try:
                _chain.save_chain(chains_dir, cname, data)
            except Exception:
                pass

    return {"persisted": persisted, "deduped": deduped}


def _fx_stamp_distilled(mem: Path, source: Any) -> dict[str, Any]:
    """Stamp distilled:true + distilled_at on a live SOURCE node (TUNE-2026-06-10 c).

    What: after a queue item is processed SUCCESSFULLY (extraction ran + the content
      is semantically covered), if the SOURCE resolves to a LIVE node file
      (mem/nodes/<source>.md exists), rewrite its frontmatter adding distilled:true +
      distilled_at:<iso-utc> via frontmatter.read_node + write_node. The BODY is passed
      back UNCHANGED, so the genuine-write anchor hook (integrity.capture_on_genuine_
      write, fired by write_node since integrity_rewrite defaults False) sees an
      UNCHANGED body vs the anchor and SHA-skips: it returns {"skipped":"unchanged"}
      with NO anchor write and (critically) NO integrity reset (the reset only runs on
      the non-skip branch, after a fresh anchor write). So this frontmatter-only stamp
      never clobbers the pristine anchor and never resets the integrity score —
      verified against integrity.capture_on_genuine_write (the unchanged-body early
      return precedes both the write_anchor and the get_integrity<FULL reset).
    Why: TUNE-2026-06-10 operator decision (c), systems-consolidation gating — the
      distilled marker is the gate that UNLOCKS a frozen node's slow integrity erosion
      (integrity.is_distilled / integrity_decay_pass). The episodic trace fades only
      AFTER the semantic representation forms; this is where "forms" is recorded.

    FAIL-OPEN: a missing source, an absent/unreadable node file, a write rejection
      (AUD61 frozen/archived target_state), or ANY exception is swallowed — a stamp
      failure NEVER breaks the drain (the atoms are already persisted; the marker is
      a best-effort consolidation signal, re-tried on the next successful drain).
    """
    if not source:
        return {"stamped": False, "skipped": "no-source"}
    src = str(source)
    stem = src[:-3] if src.endswith(".md") else src
    node_path = Path(mem) / "nodes" / f"{stem}.md"
    if not node_path.exists():
        # The source is not a live node (e.g. a frozen source whose file is already
        # gone, or a merge-pair pseudo-source) — nothing to stamp, fail-open.
        return {"stamped": False, "skipped": "no-live-source"}
    try:
        from samia.core import frontmatter as _fm
        import datetime as _dt
        fm, order, body = _fm.read_node(node_path)
        if fm.get("distilled") is True:
            # Idempotent: already stamped (an earlier successful drain) — no rewrite.
            return {"stamped": False, "skipped": "already-distilled"}
        if "distilled" not in order:
            order.append("distilled")
        fm["distilled"] = True
        if "distilled_at" not in order:
            order.append("distilled_at")
        fm["distilled_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        # Body UNCHANGED -> capture_on_genuine_write SHA-skips (no anchor clobber, no
        # integrity reset). integrity_rewrite left False so the genuine-write anchor
        # hook runs its unchanged-body skip (it does NOT treat this as an erosion).
        _fm.write_node(node_path, fm, order, body)
        return {"stamped": True, "node": stem}
    except Exception as e:
        # FAIL-OPEN: never let a stamp failure break the drain.
        return {"stamped": False, "skipped": "stamp-error", "error": str(e)}


def _fx_provenance_edge(atom_fname: str, source_fname: str,
                        db_dir: str | None = None) -> None:
    """Lay a web_store edge atom -> source (ref_kind='provenance'), fail-soft.

    What/Why: mirrors merge_consumer._add_provenance_edge exactly — a directed
    edges.db row recording the atom's episodic->semantic lineage; a store error
    never blocks the persist (the node itself is the durable artifact).
    """
    try:
        from samia.core import web_store as _ws
    except Exception:
        return
    try:
        conn = _ws.connect(db_dir)
        try:
            now = _ws._utc_now()
            conn.execute(
                """
                INSERT INTO edges (src_node, dst_node, ref_kind, occurrence_count,
                                   first_seen_at, last_seen_at, weight)
                VALUES (?, ?, ?, 1, ?, ?, 1.0)
                ON CONFLICT(src_node, dst_node, ref_kind) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (atom_fname, source_fname, "provenance", now, now),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return


def _fe_slug(s: str, n: int = 40) -> str:
    """Filesystem-safe slug for a semantic-node filename (mirrors fact_extractor._slug)."""
    import re as _re
    s = _re.sub(r"[^A-Za-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:n] or "atom"


def _dt_today() -> str:
    """Today's ISO date (for the mini-chain's last_traversal stamp)."""
    import datetime as _dt
    return _dt.date.today().isoformat()


# ---------------------------------------------------------------------------
# Registration entry point (idempotent) — called by the daemon at startup.
# ---------------------------------------------------------------------------

_REGISTERED = False


def register_rem_subscribers() -> list[str]:
    """Register the REM-gated offline ops as REM subscribers (idempotent).

    What: wires consolidation -> tier2_merge -> contradiction_passive -> replay
          -> fact-extract into the REM registry with their priorities +
          due-conditions + cursor keys. Returns the registered names in priority
          order. Decay is NOT here — it is the continuous forgetting curve
          (wake+REM), driven by the idle_pulse "decay" subscriber, never the REM
          driver (no double-drive).
    Why:  the migration entry point. Called by the daemon at startup so REM's
          driver runs these STRENGTHENING/ABSTRACTING ops inside the sleep
          window. FEAT-2026-06-07 P3c adds the contradiction_passive sweep at
          priority 25 (between consolidation and replay); it is double-gated
          (REM + ASTHENOS_CONTRADICTION_ENABLED), inert by default. FEAT-2026-
          06-07 P1 adds the tier2_merge pick-winner dup-merge DRAIN at priority
          22 (between consolidation and contradiction_passive); double-gated
          (REM + ASTHENOS_TIER2_MERGE_ENABLED), inert by default — it drains the
          surfacer's near-dup backlog so work_remaining can finally go false.
    """
    global _REGISTERED
    rem_cycle.register_offline_op(
        "consolidation", _sub_consolidation, priority=PRIO_CONSOLIDATION,
        due_condition=_due_consolidation, cursor_key="consolidation",
    )
    rem_cycle.register_offline_op(
        "tier2_merge", _sub_tier2_merge, priority=PRIO_TIER2_MERGE,
        due_condition=_due_tier2_merge, cursor_key="tier2_merge",
    )
    rem_cycle.register_offline_op(
        "contradiction_passive", _sub_contradiction_passive,
        priority=PRIO_CONTRADICTION_PASSIVE,
        due_condition=_due_contradiction_passive,
        cursor_key="contradiction_passive",
    )
    rem_cycle.register_offline_op(
        "integrity_repair", _sub_integrity_repair,
        priority=PRIO_INTEGRITY_REPAIR,
        due_condition=_due_integrity_repair,
        cursor_key="integrity_repair",
    )
    rem_cycle.register_offline_op(
        "vector_maintenance", _sub_vector_maintenance,
        priority=PRIO_VECTOR_MAINTENANCE,
        due_condition=_vector_index_drift,
        cursor_key="vector_maintenance",
    )
    rem_cycle.register_offline_op(
        "replay", _sub_replay, priority=PRIO_REPLAY,
        due_condition=_due_replay, cursor_key="replay",
    )
    rem_cycle.register_offline_op(
        "fact_extract", _sub_fact_extract, priority=PRIO_FACT_EXTRACT,
        due_condition=_due_fact_extract, cursor_key="fact_extract",
    )
    _REGISTERED = True
    names = rem_cycle.registered_offline_ops()
    _log.info("rem_subscribers: registered %d REM offline ops: %s",
              len(names), names)
    return names


# ─────────────────────────────────────────────
# [rem_subscribers] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.runtime
# Version:    1.7.0  Updated: 2026-06-11  Status: active
# Phase:      FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P2)
#             + FEAT-2026-06-07-memory-p3-contradiction-detector-v01 (P3c:
#               register the PASSIVE supersession sweep at priority 25)
#             + FEAT-2026-06-07-memory-tier2-merge-consumer-v01 (P1: register the
#               pick-winner dup-merge DRAIN at priority 22 — double-gated, inert;
#               P2: after the P1 drain the tier2_merge subscriber SYNTHESIZES —
#               PROPOSES, never applies — abstractions for the queued 'abstract'
#               pairs via merge_consumer.synthesize_pending; confirm is MCP-only)
#             + FEAT-2026-06-07-memory-granular-recall-repaired-decay (P2:
#               register the CONSOLIDATION integrity-repair pass at priority 28 —
#               PARTIAL anchor-first repair of eroded nodes; double-gated
#               (REM + ASTHENOS_INTEGRITY_REPAIR_ENABLED), inert by default)
#             + 2026-06-07 operator correction: decay UN-registered as a REM
#               subscriber (CLS — forgetting is continuous across wake+REM,
#               only consolidation/replay/contradiction/fact-extract sleep).
#             + FEAT-2026-06-10-memory-fact-extract-producer-v01 (P1: the
#               fact_extract subscriber now PERSISTS atoms as full-citizen
#               semantic nodes — dedup (cosine>=0.92) → write_node (auto-anchored)
#               → web_store provenance edge atom->source → per-source mini-chain
#               (>=2 members). Double-gated: REM + ASTHENOS_FACT_EXTRACT_ENABLED
#               (entry gate inside the drain + the due-condition); flag-off =
#               byte-identical no-op. Backend via the cached BitNet-2B seam,
#               fail-soft (no backend leaves the queue intact). ADDITIVE — no
#               source is deleted/archived/superseded.)
#             + TUNE-2026-06-10 (decision c, distillation-gated frozen erosion):
#               after a queue item is processed SUCCESSFULLY (extraction ran AND
#               >= 1 atom persisted OR all atoms dedup-skipped — content is
#               semantically covered), _fx_stamp_distilled rewrites the LIVE
#               SOURCE node's frontmatter adding distilled:true + distilled_at
#               (body UNCHANGED -> capture_on_genuine_write SHA-skips, no anchor
#               clobber, no integrity reset). This is the gate that unlocks the
#               frozen node's slow integrity erosion (integrity.is_distilled).
#               NOT stamped on extraction failure / empty-atom results; FAIL-OPEN
#               (a stamp failure never breaks the drain).
# Role:       migrate the strengthening/abstracting offline ops onto the REM
#             subscriber registry + gate (decay is NOT among them)
# Depends:    samia.runtime.rem_cycle (registry + gate + cursors),
#             samia.runtime.sleep_pressure (due-condition signal readers),
#             samia.runtime.contradiction (P3c passive_sweep + passive_has_work),
#             samia.core.merge_consumer (P1 tier2_merge drain + is_enabled),
#             samia.core.{context_extension,consolidation,fact_extractor}
# Note:       PRODUCE-ONLY — registration runs only when register_rem_subscribers
#             is called (daemon startup, operator-gated). Wraps existing ops; no
#             op internals rewritten. fact_extract is a NEW batch wrapper around a
#             per-text primitive (no live cadence to migrate-away). P3c
#             (contradiction_passive) is double-gated: REM + is_enabled().
#             Decay is driven only by idle_pulse (continuous, NOT REM-gated).
# G2-2026-06-11 (REM machine-drainable-only): _sub_consolidation's cursor
#             "remaining" is now gated on _merge_drainable(mem) — the surfaced
#             near-dup backlog counts as cursor-remaining work ONLY when the
#             tier2_merge consumer is ENABLED (a machine cycle can actually drain
#             it). When merge is OFF the surfaced backlog is operator-gated
#             (confirm-via-MCP) -> remaining 0 (surfaced as operator_gated_backlog
#             telemetry), so it no longer holds REM permanently awake.
# G4-2026-06-11 (vector rebuild wiring, operator choice 4a): NEW subscriber
#             vector_maintenance at priority 29 (after integrity_repair, before
#             replay). due_condition _vector_index_drift fires when the index's
#             manifest node_count != the live nodes/*.md count (cheap, no hashing).
#             It runs vector.build INCREMENTALLY (manifest-cached, embeds only
#             new/changed) every drift cycle + a FULL rebuild(rebuild=True) on a long
#             cadence (ASTHENOS_VECTOR_FULL_REBUILD_S, default 7 days, cursor-tracked
#             via last_full_ts under key "vector_maintenance"). The index no longer
#             drifts because nothing rebuilt it ("that's where it was intended").
# ─────────────────────────────────────────────
