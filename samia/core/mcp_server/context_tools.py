"""samia.core.mcp_server.context_tools — context-extension + index/REM status arm.

Layer 1 (Owns / Depends):
    Owns:    the context-extension tool logic (memory_chainogram_retrieve with the
             P2a semantic-arm overlay, memory_frozen_prefix, memory_tier_flow_for_budget,
             memory_episodic_candidates, memory_idle_tick, memory_sm2_update,
             memory_sm2_due, memory_compaction_skip_filter) and the status reads
             (memory_index_status, memory_rem_status, memory_rem_sleep_now).
    Depends: .config (Any/Path). Lazy per-call: samia.core.{context_extension,
             semantic_recall,vector} and samia.runtime.rem_cycle — function-local so
             the runtime dependency stays off the core import path.

Layer 2 (What / Why):
    What: each context tool delegates to its context_extension.* primitive
          (chainogram retrieve, frozen-prefix block, tier-flow budgeting, episodic->
          semantic candidates, idle replay tick, SM-2 review update/due, compaction
          skip filter); the index/REM status tools read the vector manifest and the
          persisted REM state + sleep-pressure gauge. chainogram_retrieve OVERLAYS the
          P2a semantic-arm composed context under additive keys when the arm is enabled.
    Why:  the SAM/IA x compaction context-extension surface plus the read-only status
          gauges are a cohesive observability/budgeting cluster, distinct from node
          recall/write and the chain/bio tools. The lazy rem_cycle import keeps the
          runtime dependency off the core import path (the monolith's note).
"""

from __future__ import annotations

from .config import Any, Path


# ---------------------------------------------------------------------------
# Context-extension primitives (SAM/IA × compaction hybrids)
# ---------------------------------------------------------------------------


def memory_chainogram_retrieve(memory_dir: Path, query: str,
                                budget_tokens: int = 8000,
                                max_chains: int = 8,
                                include_failure_associations: bool = False,
                                failure_top_n: int | None = None) -> dict[str, Any]:
    # FEAT-2026-06-10 P2a — semantic-arm wire. When the arm is enabled, OVERLAY the
    # composed read-side context (KNOWN FACTS + CONVERSATION EVIDENCE) onto the
    # existing chainogram result shape: the standard keys (loaded_chains/loaded_nodes/
    # spent_tokens/rationale/...) stay populated exactly as today so MCP clients that
    # read them never break, and the composed extras land under NEW keys (composed_*,
    # facts_n). Flag OFF -> the original path runs untouched (branch around, do not
    # restructure). The arm flag default-OFF means this branch is byte-identical to the
    # pre-P2 behavior until the operator enables ASTHENOS_SEMANTIC_ARM_ENABLED.
    from .. import semantic_recall as _sr
    from .. import context_extension as _cx
    base = _cx.chainogram_retrieve(memory_dir, query, budget_tokens=budget_tokens,
                                   max_chains=max_chains,
                                   include_failure_associations=include_failure_associations,
                                   failure_top_n=failure_top_n)
    if not _sr.semantic_arm_enabled():
        return base
    # Arm ON: route to the composer for the composed context (it runs its own focused
    # chainogram + atom arm internally). Surface its outputs under additive keys so the
    # tool contract (name + existing keys) is preserved. Fail-open: a composer error
    # leaves the standard chainogram result intact.
    try:
        composed = _sr.recall(memory_dir, query, budget_tokens=budget_tokens)
    except Exception as exc:  # pragma: no cover - defensive
        base["semantic_arm_error"] = str(exc)
        return base
    base["composed_context"] = composed.get("context", "")
    base["facts_n"] = composed.get("facts_n", 0)
    base["composed_evidence_nodes"] = composed.get("evidence_nodes", 0)
    base["composed_dia_ids"] = composed.get("dia_ids", [])
    base["semantic_arm"] = True
    return base


def memory_frozen_prefix(memory_dir: Path, write: bool = True) -> dict[str, Any]:
    from .. import context_extension as _cx
    return _cx.frozen_prefix_block(memory_dir, write=write)


def memory_tier_flow_for_budget(memory_dir: Path, query: str,
                                  budget_tokens: int = 8000,
                                  apply: bool = False) -> dict[str, Any]:
    from .. import context_extension as _cx
    return _cx.tier_flow_for_budget(memory_dir, query, budget_tokens=budget_tokens,
                                    dry_run=not apply)


def memory_episodic_candidates(memory_dir: Path) -> dict[str, Any]:
    from .. import context_extension as _cx
    return _cx.episodic_to_semantic_candidates(memory_dir)


def memory_idle_tick(memory_dir: Path, force: bool = False) -> dict[str, Any]:
    from .. import context_extension as _cx
    return _cx.idle_replay_tick(memory_dir, force=force)


def memory_sm2_update(memory_dir: Path, node: str, missed: bool = False,
                       quality: int = 4) -> dict[str, Any]:
    from .. import context_extension as _cx
    return _cx.sm2_review_update(memory_dir, node, recalled=not missed, quality=quality)


def memory_sm2_due(memory_dir: Path) -> list[dict[str, Any]]:
    from .. import context_extension as _cx
    return _cx.sm2_due_for_review(memory_dir)


def memory_compaction_skip_filter(memory_dir: Path,
                                    transcript_chunks: list[str],
                                    threshold: float = 0.78) -> dict[str, Any]:
    from .. import context_extension as _cx
    return _cx.compaction_skip_filter(memory_dir, transcript_chunks, threshold=threshold)


# ---------------------------------------------------------------------------
# Index status
# ---------------------------------------------------------------------------


def memory_index_status(memory_dir: Path) -> dict[str, Any]:
    from .. import vector as _vi
    m = _vi._load_manifest(memory_dir)
    return {
        "model": m.get("model_id"),
        "dim": m.get("dim"),
        "node_count": m.get("node_count"),
        "built_at": m.get("built_at"),
    }


def memory_rem_status(memory_dir: Path) -> dict[str, Any]:
    """REM sleep-cycle state + the sleep-pressure health gauge.

    What: returns the persisted WAKE<->REM state plus the composite
          sleep-pressure breakdown (per-signal + score + threshold +
          sleep_needed) — the operator-visible health gauge — so the Atoms /
          Claude surface can read whether reconciliation is owed and whether the
          system is asleep.
    Why:  the thin read half of the REM P1 observability surface
          (FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01). The
          explicit "sleep now" trigger lives on the daemon IPC op rem_sleep_now;
          this is the plain-function read MCP wraps. Lazy import keeps the runtime
          dependency off the core import path.
    """
    from samia.runtime import rem_cycle
    return rem_cycle.rem_status(Path(memory_dir))


def memory_rem_sleep_now(memory_dir: Path) -> dict[str, Any]:
    """Explicit REM "sleep now" trigger (sets the force flag).

    What: flips the force-requested flag so the next daemon tick enters REM
          regardless of pressure/idle; returns {ok, state}.
    Why:  the on-demand cycle trigger (Q1 explicit path / risk-1 mitigation),
          exposed as a plain function the MCP / IPC surface wraps. Produce-only:
          it only records the request; the daemon tick applies it.
    """
    from samia.runtime import rem_cycle
    return rem_cycle.rem_sleep_now(Path(memory_dir))


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.mcp_server.context_tools
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.mcp_server monolith during
#             modularization (the Context-extension primitives + Index/REM status sections).
# Layer:      core (pure library, no daemon dependency)
# Role:       the context-extension + index/REM status arm — chainogram_retrieve (with
#             the P2a semantic-arm overlay), frozen_prefix, tier_flow_for_budget,
#             episodic_candidates, idle_tick, sm2_update/due, compaction_skip_filter,
#             and the index_status / rem_status / rem_sleep_now reads.
# Stability:  stable — behavior byte-identical to the monolith's Context-extension +
#             Index status sections; only the imports moved behind .config.
# ErrorModel: most tools surface their context_extension.* result directly; chainogram_
#             retrieve is fail-open on the semantic-arm overlay (a composer error leaves
#             the standard chainogram result intact). rem_sleep_now is produce-only.
# Depends:    .config (Any/Path). Lazy per-call: samia.core.{context_extension,
#             semantic_recall,vector}, samia.runtime.rem_cycle.
# Exposes:    memory_chainogram_retrieve, memory_frozen_prefix,
#             memory_tier_flow_for_budget, memory_episodic_candidates, memory_idle_tick,
#             memory_sm2_update, memory_sm2_due, memory_compaction_skip_filter,
#             memory_index_status, memory_rem_status, memory_rem_sleep_now.
# Lines:      183
# --------------------------------------------------------------------------
