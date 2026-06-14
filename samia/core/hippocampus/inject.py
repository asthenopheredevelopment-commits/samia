"""samia.core.hippocampus.inject — the P4 two-layer standing inject block assembler.

Layer 1 (Owns / Depends):
    Owns:    assemble_inject_block (the TWO inject layers under a FIXED token budget),
             the cheap token estimator estimate_tokens, and the co-activation-SILENT
             relevance cosine _relevance_score. Builds a SMALL standing context block
             (engram-inject = the inject_eligible held copies + ring-inject = the live
             ring pointers dereferenced to backing) priority-arbitrated under the cap.
    Depends: .config (INJECT_BUDGET_DEFAULT / INJECT_ENGRAM_BUDGET_FRAC + the
             re-exported vector backend _vi + numpy), .engram (EngramStore — the
             engram-inject candidates), .ring (RingStore — the ring-inject candidates).

Layer 2 (What / Why):
    What: the context-window-extension step. assemble_inject_block pushes a small
          standing context into the model's finite window — NOT a search (RAG is the
          search path). ENGRAM-INJECT (the inject_eligible held copies) is the always-on
          identity / known-cold set, reserved a budget slice + FAVORED under pressure;
          RING-INJECT (the live ring pointers dereferenced) is the auto-pilot working
          set filling the remainder. Both are relevance-gated by cosine (a SORT for
          inclusion, NOT a retrieval) and priority-arbitrated; the block NEVER exceeds
          the budget.
    Why:  carved out of the 1339-line monolith as the inject responsibility — the top
          of the package DAG (it reads EngramStore + RingStore, nothing reads it inside
          the package). The keystone is CO-ACTIVATION SILENCE (D5/Q6a): no path here
          calls hebbian_record or feeds the Tier-0 web, so assembling/serving a block
          records ZERO genuine co-activations (a standing deref is not "recalled
          together"). The per-turn live-prompt injection stays operator-gated/INERT.
"""

from __future__ import annotations

import numpy as np

from .config import (
    INJECT_BUDGET_DEFAULT,
    INJECT_ENGRAM_BUDGET_FRAC,
    Path,
    _vi,
)
from .engram import EngramStore
from .ring import RingStore


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for the inject budget (chars/4 heuristic).

    What: approximate the token count of `text` as ceil(len/4) — the standard cheap
      heuristic when no model tokenizer is on the hot path. Empty/None -> 0.
    Why: D4 — the inject budget is a coarse cap, not an exact accounting; a len-based
      heuristic keeps assemble_inject_block dependency-free (no tokenizer load on the
      per-turn path) while bounding the block. No general token estimator exists in the
      tree (only judge_eval's LLM-side truncator), so this is the local primitive.
    """
    if not text:
        return 0
    return -(-len(text) // 4)  # ceil division: every started 4-char chunk costs a token


def _relevance_score(query_vec, content: str) -> float:
    """Co-activation-SILENT cosine of `content` vs the query embedding (relevance sort).

    What: embed `content` with the shared backend and cosine it against the already-
      embedded query vector; both vectors are L2-normalized by the backend, so the dot
      product is the cosine. Returns 0.0 on any embed failure (fail-soft).
    Why: D4/D5 — inject relevance-gates each layer for INCLUSION ordering, NOT retrieval.
      This is the same cosine PRIMITIVE the RAG arms use, but it deliberately does NOT
      route through ring_rag_query / engram_rag_query / memory_search and NEVER calls
      hebbian_record — assembling an inject block records ZERO genuine co-activations
      (the homeostasis guard: a standing deref is not "recalled together").
    """
    if not content:
        return 0.0
    try:
        vec = _vi._embed_batch([content])[0]
        return float(np.dot(vec, query_vec))
    except Exception:
        return 0.0


def assemble_inject_block(memory_dir: Path, query_or_context: str,
                          token_budget: int = INJECT_BUDGET_DEFAULT,
                          engram_budget_frac: float = INJECT_ENGRAM_BUDGET_FRAC
                          ) -> dict:
    """Assemble the two-layer standing-availability inject block under a fixed budget (P4/D4).

    What: build a SMALL standing context block to prepend to a prompt — the context-
      window-extension step — from two layers, relevance-gated against `query_or_context`
      and arbitrated by priority within `token_budget`:
        ENGRAM-INJECT — the inject_eligible held copies (P3 flag): the always-on
            identity/known-cold set. Reserved an engram_budget_frac slice; FAVORED under
            pressure (filled FIRST, in relevance order, until its reserved slice — OR the
            whole budget if the ring is empty — is exhausted).
        RING-INJECT  — the live ring pointers dereferenced to backing content: the auto-
            pilot working set. Fills the REMAINING budget by relevance.
      When the two layers together exceed `token_budget`, ARBITRATE by priority: engram-
      inject is selected first; ring-inject then fills only what is left; overflow in
      either layer is DROPPED. The returned block NEVER exceeds the budget.
    Why: D4/Q5a — this is the SETTLED two-inject-layer design: a small always-on identity
      set + a turn-relevant working set, priority-arbitrated under a fixed cap. It is
      CO-ACTIVATION-SILENT (D5/Q6a, _relevance_score / direct deref, NEVER hebbian_record)
      — assembling/serving a block manufactures ZERO Tier-0 edges (the homeostasis guard).
      It is a standing pointer-deref (O(1) reads of the engram flag + the ring pointers),
      NOT a search (RAG is the search path). P4 builds the assembler; the actual per-turn
      injection into the live prompt stays operator-gated/INERT.

    Returns:
        {
          "items": [ {layer, source, title, content, tokens, score}, ... ],  # selected,
                                                                             # in serve order
          "tokens_used": int, "token_budget": int,
          "engram_budget": int, "ring_budget": int,
          "engram_count": int, "ring_count": int,
          "dropped": int,                # candidates excluded for budget/relevance
          "co_activation_silent": True,  # contract marker (D5)
        }
    An empty ring + empty engram yields an empty block (fail-open).
    """
    budget = max(0, int(token_budget))
    frac = min(1.0, max(0.0, float(engram_budget_frac)))
    engram_budget = int(budget * frac)

    # Embed the turn query ONCE for the relevance sort (co-activation-silent: a bare
    # embed, never a retrieval — no hebbian_record, no Tier-0 feed).
    try:
        query_vec = _vi._embed_batch([query_or_context])[0]
    except Exception:
        query_vec = None

    # ---- ENGRAM-INJECT candidates: the inject_eligible held copies (always-on) ----
    engram_cands: list[dict] = []
    for copy in EngramStore(memory_dir).all():
        if not copy.get("inject_eligible"):
            continue
        content = copy.get("body", "") or ""
        score = _relevance_score(query_vec, content) if query_vec is not None else 0.0
        engram_cands.append({
            "layer": "engram",
            "source": copy.get("source_ptr") or copy.get("engram_id"),
            "title": copy.get("title"),
            "content": content,
            "tokens": estimate_tokens(content),
            "score": score,
        })
    engram_cands.sort(key=lambda c: c["score"], reverse=True)

    # ---- RING-INJECT candidates: live ring pointers, dereferenced to backing ----
    ring_cands: list[dict] = []
    ring = RingStore(memory_dir)
    for entry in ring.entries():
        backing = ring.resolve(entry)  # dangling/stale -> None, contributes nothing
        if backing is None:
            continue
        content = backing.get("content", "") or backing.get("body", "") or ""
        score = _relevance_score(query_vec, content) if query_vec is not None else 0.0
        ring_cands.append({
            "layer": "ring",
            "source": backing.get("ptr"),
            "title": backing.get("title"),
            "content": content,
            "tokens": estimate_tokens(content),
            "score": score,
        })
    ring_cands.sort(key=lambda c: c["score"], reverse=True)

    # ---- Priority arbitration under the fixed budget ----
    # Engram-inject (identity, FAVORED) is selected first, up to its reserved slice
    # (or the whole budget if no ring candidate competes). Ring-inject then fills only
    # the REMAINING budget by relevance. Overflow in either layer is dropped.
    selected: list[dict] = []
    used = 0
    dropped = 0

    # Engram cap: its reserved slice, but allow it the whole budget when the ring is
    # empty (a small identity set should not be starved by an unused ring reservation).
    engram_cap = budget if not ring_cands else engram_budget
    for cand in engram_cands:
        if used + cand["tokens"] <= engram_cap:
            selected.append(cand)
            used += cand["tokens"]
        else:
            dropped += 1

    # Ring fills the remaining budget (total budget minus what engram actually used).
    for cand in ring_cands:
        if used + cand["tokens"] <= budget:
            selected.append(cand)
            used += cand["tokens"]
        else:
            dropped += 1

    return {
        "items": selected,
        "tokens_used": used,
        "token_budget": budget,
        "engram_budget": engram_budget,
        "ring_budget": budget - engram_budget,
        "engram_count": sum(1 for c in selected if c["layer"] == "engram"),
        "ring_count": sum(1 for c in selected if c["layer"] == "ring"),
        "dropped": dropped,
        # Contract marker: assembling this block recorded NO genuine co-activation and
        # fed NOTHING to the Tier-0 Hebbian web (D5/Q6a homeostasis guard).
        "co_activation_silent": True,
    }


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.hippocampus.inject
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.hippocampus monolith during
#             modularization (the P4 inject responsibility; bodies byte-identical, the
#             inject budgets/_vi lifted into .config and EngramStore/RingStore into
#             their submodules).
# Layer:      core (pure library, no daemon dependency)
# Role:       P4 — the two-layer standing inject block assembler: assemble_inject_block
#             (engram-inject = the inject_eligible held copies FAVORED under pressure +
#             ring-inject = the live ring pointers dereferenced, filling the remainder)
#             under a FIXED token budget, relevance-gated (_relevance_score) +
#             priority-arbitrated (overflow dropped, block never exceeds the cap), and
#             CO-ACTIVATION-SILENT. estimate_tokens is the cheap budget estimator.
# Stability:  stable — bodies byte-identical to the monolith; the carve only moved the
#             inject budgets + _vi into .config and EngramStore/RingStore into their
#             submodules.
# ErrorModel: fail-open — an embed failure yields a 0.0 relevance / a None query vec
#             (the block still assembles); an empty ring + empty engram yields an empty
#             block. The block is test-asserted to NEVER exceed the token budget.
# Depends:    numpy. .config (inject budgets/_vi), .engram (EngramStore), .ring
#             (RingStore).
# Exposes:    estimate_tokens, assemble_inject_block (and _relevance_score).
# Note:       CO-ACTIVATION-SILENT (D5/Q6a) — NO path here calls hebbian_record or feeds
#             the Tier-0 web; the per-turn live-prompt injection stays operator-gated.
# Lines:      231
# ─────────────────────────────────────────────
