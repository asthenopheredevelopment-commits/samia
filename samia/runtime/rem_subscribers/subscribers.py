"""samia.runtime.rem_subscribers.subscribers — the non-fact-extract REM callables.

Layer 1 (Owns / Depends):
    Owns:    the REM subscriber callables that WRAP/adapt the existing offline ops
             to the fn(mem) -> dict shape — _sub_consolidation (the near-dup
             surfacer), _sub_tier2_merge (the pick-winner dup-merge DRAIN + P2
             synthesize), _sub_contradiction_passive (the P3c supersession sweep),
             _sub_integrity_repair (the P2 consolidation integrity-repair pass),
             _sub_vector_maintenance (the G4 vector-index sync), and _sub_replay
             (replay/dreaming + hebbian). Each is REM-gated at its own entry; none
             rewrites an op's internals. (The fact-extract subscriber lives in its
             own submodule — it has a larger persistence subsystem.)
    Depends: .config (rem_cycle gate/cursor helpers, sleep_pressure signal readers,
             _integrity_repair_enabled, the budgets + _CONSOLIDATION_THRESHOLD),
             .due (_merge_drainable — the surfacer's machine-drainable gate), and —
             lazily, inside each callable — the wrapped ops themselves
             (consolidation, merge_consumer, contradiction, integrity, vector,
             context_extension).

Layer 2 (What / Why):
    What: each callable refuses outside REM (rem_cycle.gate_offline_op), runs ONE
          budgeted, cursor-tracked slice of its op, and returns a dict carrying
          "work_remaining" so the driver can OR it into evaluate()'s work_remains.
    Why:  Q3 / the proposal's P2 — the heavy STRENGTHENING/ABSTRACTING ops stop
          trickling on every idle pulse and run behind REM. Wrapping (not
          rewriting) each op means the op keeps its own internal cadence + flag
          gate; the REM gate is additive and the registry just schedules it inside
          the sleep window in priority order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    rem_cycle,
    sleep_pressure,
    _integrity_repair_enabled,
    _vector_full_rebuild_interval_s,
    _CONSOLIDATION_THRESHOLD,
    _TIER2_MERGE_BUDGET,
    _INTEGRITY_REPAIR_BUDGET,
)
from .due import _merge_drainable


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


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_subscribers.subscribers
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_subscribers monolith during modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the non-fact-extract REM subscriber callables — wrap/adapt the
#             existing offline ops (consolidation surfacer, tier2_merge drain+P2
#             synthesize, contradiction passive sweep, integrity-repair pass,
#             vector-index maintenance, replay/dreaming) to the fn(mem) -> dict
#             shape, REM-gated at each entry, never rewriting an op's internals.
# Stability:  stable — the carve preserved every gate, cursor key, budget, and the
#             returned dict shape (incl. "work_remaining") byte-identical to the
#             monolith.
# ErrorModel: each callable refuses outside REM with {"fired": False,
#             "refused": "not_in_rem"}; the wrapped ops own their own fail-soft /
#             flag-gate posture (integrity_repair also short-circuits on the
#             not-enabled flag). No exception is swallowed here beyond the gate.
# Depends:    pathlib, typing (stdlib). .config (rem_cycle, sleep_pressure,
#             _integrity_repair_enabled, _vector_full_rebuild_interval_s, budgets,
#             _CONSOLIDATION_THRESHOLD), .due (_merge_drainable). The wrapped ops
#             (consolidation, merge_consumer, contradiction, integrity, vector,
#             context_extension) are all imported lazily inside the callable.
# Exposes:    _sub_consolidation, _sub_tier2_merge, _sub_contradiction_passive,
#             _sub_integrity_repair, _sub_vector_maintenance, _sub_replay.
# Lines:      289
# ─────────────────────────────────────────────
