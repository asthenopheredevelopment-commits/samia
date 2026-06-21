"""samia.core.mcp_server.bio_tools — the biomimetic primitive + merge-consumer arm.

Layer 1 (Owns / Depends):
    Owns:    the biomimetic tool logic (memory_pattern_separate,
             memory_hebbian_consolidate, memory_replay_sweep, memory_reconsolidate,
             memory_schema_check, memory_chain_maturity) and the gated Tier-2
             merge-consumer surface (memory_merge_candidates, memory_confirm_merge,
             memory_reject_merge — the operator's list/confirm/reject of LLM-
             synthesized abstractions).
    Depends: .config (Any/Path). Lazy per-call: samia.core.bio (pattern-separation,
             Hebbian consolidate, replay sweep, reconsolidate, schema accelerate,
             chain maturity) and samia.core.merge_consumer (the abstraction
             lifecycle) — function-local to keep them off the package import path.

Layer 2 (What / Why):
    What: each biomimetic tool delegates to its bio.* primitive; the merge-consumer
          tools delegate to merge_consumer.list/confirm/reject_abstraction. The merge
          surface mirrors the P3 supersession confirm/reject — abstractions are
          OPERATOR-GATED (a 'proposed' entry carries the synthesized draft for
          review; nothing is applied until confirmed, and both sources stay
          restorable on confirm).
    Why:  the biomimetic primitives and the merge-consumer surface are the
          consolidation/abstraction tool cluster — cohesive and distinct from node
          recall/write and the chain/context tools. The fail-open wrappers keep a
          bio/merge error from raising into the MCP loop.
"""

from __future__ import annotations

from .config import Any, Path  # noqa: F401


# ---------------------------------------------------------------------------
# Biomimetic primitives
# ---------------------------------------------------------------------------


def memory_pattern_separate(memory_dir: Path, text: str,
                             threshold: float = 0.85) -> dict[str, Any]:
    from .. import bio as _bio
    return _bio.pattern_separation_decision(memory_dir, text, threshold=threshold)


def memory_hebbian_consolidate(memory_dir: Path) -> dict[str, Any]:
    from .. import bio as _bio
    return _bio.hebbian_consolidate(memory_dir)


def memory_replay_sweep(memory_dir: Path, sample: int = 20,
                         threshold: float = 0.55) -> dict[str, Any]:
    from .. import bio as _bio
    return _bio.replay_sweep(memory_dir, sample=sample, threshold=threshold)


def memory_reconsolidate(memory_dir: Path, node: str, new_context: str,
                          backend: str = "auto") -> dict[str, Any]:
    from .. import bio as _bio
    return _bio.reconsolidate(memory_dir, node, new_context, backend=backend)


def memory_schema_check(memory_dir: Path, text: str,
                         chains: list[str]) -> dict[str, Any]:
    from .. import bio as _bio
    return _bio.schema_accelerate(memory_dir, text, chains)


def memory_chain_maturity(memory_dir: Path, chain: str) -> dict[str, Any]:
    from .. import bio as _bio
    return _bio.chain_maturity(memory_dir, chain)


# ---------------------------------------------------------------------------
# Tier-2 merge consumer P2 — gated LLM-synthesized abstraction surface
# (mirror of the P3 supersession confirm/reject; operator-only confirm).
# ---------------------------------------------------------------------------


def memory_merge_candidates(memory_dir: Path) -> dict[str, Any]:
    """List un-resolved Tier-2 merge/abstraction candidates (P2 surface).

    What: returns the {candidate_id, a, b, status, abstraction?, merged_from?}
          records from biomimetic/merge_candidates.jsonl that the consumer queued
          ('pending' — awaiting synthesis) or PROPOSED ('proposed' — a synthesized
          draft awaiting operator confirm). Reads
          merge_consumer.list_abstraction_candidates.
    Why:  Q2c — abstractions are operator-gated. This is the operator's listing
          surface; a 'proposed' entry carries the synthesized title+body so the
          operator can review before confirming. Nothing is applied until acted.
    """
    # FailOpenList — What: lazily reach merge_consumer and return its candidate list,
    #     converting any error into an {error} payload instead of propagating it.
    try:
        from .. import merge_consumer as _mc
        return {"candidates": _mc.list_abstraction_candidates(memory_dir)}
    except Exception as e:  # fail-open: never raise into the MCP loop.
        return {"candidates": [], "error": str(e)}
    # FailOpenList — Why: a merge/bio error must never raise into the MCP request loop —
    #     a degraded {candidates:[], error} reply keeps the server responsive.


def memory_confirm_merge(memory_dir: Path, candidate_id: str) -> dict[str, Any]:
    """Confirm a PROPOSED abstraction → create the node + supersede both sources.

    What: materialize the proposed draft as a new nodes/<id>.md (synthesized
          content + merged_from provenance frontmatter), then SUPERSEDE both
          source nodes RESTORABLY (reason="supersede" full-archives each so
          memory_restore_node can un-forget them byte-exact) and lay provenance
          edges abstraction->each source. Delegates to
          merge_consumer.confirm_abstraction.
    Why:  Q2c GATE — abstractions create NEW content + can lose nuance, so they
          are applied ONLY on operator confirm; both originals stay restorable.
          Mirrors memory_confirm_supersession.
    """
    # ConfirmGate — What: the ONLY applying path — delegate to merge_consumer to create the
    #     abstraction node + supersede both sources RESTORABLY, fail-open on any error.
    try:
        from .. import merge_consumer as _mc
        return _mc.confirm_abstraction(memory_dir, candidate_id)
    except Exception as e:
        return {"confirmed": False, "candidate_id": candidate_id, "error": str(e)}
    # ConfirmGate — Why: Q2c — abstractions create NEW content + can lose nuance, so they
    #     apply ONLY on operator confirm; both originals stay restorable, and an error degrades
    #     to {confirmed:False} rather than raising mid-merge.


def memory_reject_merge(memory_dir: Path, candidate_id: str) -> dict[str, Any]:
    """Reject a proposed abstraction (changes NOTHING) — the gate's reject arm.

    What: marks the candidate rejected so it stops surfacing; no node created, no
          source superseded, both originals stay live. Delegates to
          merge_consumer.reject_abstraction.
    Why:  Q2c — the operator's reject path; mirrors memory_dismiss_supersession.
          A synthesized abstraction that loses nuance is discarded with zero
          mutation of live memory.
    """
    try:
        from .. import merge_consumer as _mc
        return _mc.reject_abstraction(memory_dir, candidate_id)
    except Exception as e:
        return {"rejected": False, "candidate_id": candidate_id, "error": str(e)}


# ---------------------------------------------------------------------------
# Epiphanies associative-edge veto net — the operator's correction surface for
# the associative (hebbian-edge) layer: reject a false binding/candidate, undo a
# rejection, and list the active suppressions/candidates. LEDGER-ONLY (writes no
# live chain edge, forgets no node) + fail-open; mirrors the merge-gate reject arm.
# A rejected binding is demoted (not promotable) UNLESS it keeps recurring — a
# K-recurrence override then un-suppresses it (the evidence outvoted the veto).
# ---------------------------------------------------------------------------


def memory_epiphanies_reject_binding(memory_dir: Path, node_a: str, node_b: str,
                                     reason: str = "") -> dict[str, Any]:
    """Mark an Epiphanies associative binding a FALSE association (Decision-A safety net).
    Suppresses the edge from the next consolidate (demoted, never promoted to a chain) UNLESS
    it re-qualifies enough more occasions to trip the K-recurrence override. LEDGER-only —
    writes no live edge and forgets no node; fail-open."""
    try:
        from ..bio import epiphanies as _epi
        return _epi.reject_binding(memory_dir, node_a, node_b, reason=reason)
    except Exception as e:
        return {"rejected": False, "a": node_a, "b": node_b, "error": str(e)}


def memory_epiphanies_unreject_binding(memory_dir: Path, node_a: str, node_b: str) -> dict[str, Any]:
    """Clear an Epiphanies binding suppression (operator changed their mind). Returns
    {cleared: bool} — True if a suppression existed. Mutates nothing else; fail-open."""
    try:
        from ..bio import epiphanies as _epi
        return {"cleared": bool(_epi.unreject_binding(memory_dir, node_a, node_b))}
    except Exception as e:
        return {"cleared": False, "a": node_a, "b": node_b, "error": str(e)}


def memory_epiphanies_reject_candidate(memory_dir: Path, node_a: str, node_b: str) -> dict[str, Any]:
    """Veto a linker CANDIDATE (a proposed-but-unvalidated association) as not-related. A
    K-remint override can still resurface it if it keeps genuinely co-occurring. Returns
    {rejected: bool}; fail-open."""
    try:
        from ..bio import epiphanies as _epi
        return {"rejected": bool(_epi.reject_candidate(memory_dir, node_a, node_b))}
    except Exception as e:
        return {"rejected": False, "a": node_a, "b": node_b, "error": str(e)}


def memory_epiphanies_list_suppressions(memory_dir: Path) -> dict[str, Any]:
    """List the active Epiphanies binding suppressions (operator visibility). Fail-open."""
    try:
        from ..bio import epiphanies as _epi
        return {"suppressions": _epi.list_suppressions(memory_dir)}
    except Exception as e:
        return {"suppressions": {}, "error": str(e)}


def memory_epiphanies_list_candidates(memory_dir: Path, state: str = "") -> dict[str, Any]:
    """List the linker's association candidates (optional state filter, e.g. 'validated' /
    'genuine'); operator visibility into the hypotheses. Mutates nothing; fail-open."""
    try:
        from ..bio import epiphanies as _epi
        return {"candidates": _epi.list_candidates(memory_dir, state=(state or None))}
    except Exception as e:
        return {"candidates": {}, "error": str(e)}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.mcp_server.bio_tools
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase-B modularization (REF-2026-06-14-02) — carved from the 1315-line
#             memory_mcp_server.py monolith (the biomimetic + merge-consumer section)
# Layer:      core (pure library, no daemon dependency)
# Role:       the biomimetic primitive + merge-consumer arm — pattern_separate /
#             hebbian_consolidate / replay_sweep / reconsolidate / schema_check /
#             chain_maturity plus the gated merge_candidates / confirm_merge /
#             reject_merge abstraction surface.
# Stability:  stable — behavior byte-identical to the monolith's Biomimetic primitives
#             + Tier-2 merge consumer sections; only the imports moved behind .config.
# ErrorModel: the biomimetic tools surface their bio.* primitive's result directly; the
#             merge-consumer tools are fail-open (an error yields {error} / {confirmed:
#             False} / {rejected: False}, never a raise into the MCP loop). All merge
#             confirms are RESTORABLE.
# Depends:    .config (Any/Path). Lazy per-call: samia.core.bio, samia.core.merge_consumer.
# Exposes:    memory_pattern_separate, memory_hebbian_consolidate, memory_replay_sweep,
#             memory_reconsolidate, memory_schema_check, memory_chain_maturity,
#             memory_merge_candidates, memory_confirm_merge, memory_reject_merge,
#             memory_epiphanies_reject_binding, memory_epiphanies_unreject_binding,
#             memory_epiphanies_reject_candidate, memory_epiphanies_list_suppressions,
#             memory_epiphanies_list_candidates (the associative-edge veto net).
# Note:       PRODUCE-ONLY merge gate — confirm_merge is the only applying path and it is
#             operator-invoked; reject/list mutate nothing live.
# Lines:      161
# Updated:    2026-06-14
# Status:     active
# --------------------------------------------------------------------------
