"""samia.core.bio.linker — Epiphanies Feature 2: the cross-conversation LINKER (L3, pure).

Layer 1 (Owns / Depends):
    Owns:    the association-DISCOVERY arm — the third relationship class (Linker-Generated,
             beside Direct-Hebbian and Salience-Reinforcement). On recall it PROPOSES a weak
             CANDIDATE edge from shared latent/structural substrate toward a node that has NEVER
             co-occurred (propose); a candidate that LATER earns a real genuine co-activation is
             VALIDATED — a discovered association that came true (validate); unvalidated
             candidates DECAY faster than replay and evaporate (decay); operator/agent can
             REJECT a candidate (suppress) with a K-recurrence override. PURE: no IO, no index —
             propose() takes an injected neighbor function (cosine top-k) exactly as the L1 model
             takes an injected salience function.
    Depends: .config (REPLAY_NEIGHBOR_THRESHOLD / REPLAY_ONLY_W_CEILING). No live-store contact;
             the wiring (the live cosine neighbor adapter + the candidate store IO) lives in
             bio.epiphanies, FLAG-GATED, so this can never disturb the running store.

Layer 2 (What / Why):
    What: candidate generation + the candidate state machine
          (candidate -> {validated | decayed | suppressed(+K-override)}). A candidate contributes
          ZERO to salience and ZERO to promotion — it is a hypothesis in its own sidecar store,
          NEVER surfaced as memory.
    Why:  the operator's insight — Feature 1 (L1) only STRENGTHENS pairs that already co-occurred;
          the linker DISCOVERS that A and B relate when they never co-occurred. In SHADOW mode the
          anti-confabulation guard is automatic: candidates live in epiphanies_candidates.json,
          never in the genuine/promotable store, never returned to recall. The shadow MEASURES the
          validated-candidate rate — the linker's predictive power — before it is ever wired to act
          (that integration, with the visible 'unconfirmed' label, is a later, separately-gated step).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

from .config import REPLAY_NEIGHBOR_THRESHOLD, REPLAY_ONLY_W_CEILING

# --- linker constants ---
LINK_K_PER_NODE = 5            # cosine neighbors considered per recalled node
LINK_SCORE_FLOOR = 0.62       # min cosine to mint a candidate — rare + plausible only (a high bar
                              # above REPLAY_NEIGHBOR_THRESHOLD so candidates are sparse, not noise)
LINK_INIT_WEIGHT = 0.15       # below REPLAY_ONLY_W_CEILING — a candidate is weak by construction
LINK_DECAY = 0.30             # per-occasion decay for an unvalidated candidate (FASTER than replay —
                              # hypotheses evaporate quickly if reality never confirms them)
LINK_PRUNE = 0.05             # weight below this -> the candidate decayed away
LINK_MAX_CANDIDATES = 50000   # DEFAULT memory budget for the candidate store (env-tunable via
                              # ASTHENOS_LINK_MAX_CANDIDATES; 0/negative => UNLIMITED). NOT a discovery
                              # cap: over budget, propose() evicts the weakest DISPOSABLE hypothesis
                              # (never a validated discovery / veto), so minting NEVER wedges. Raised
                              # from the old hard 3000 — once the tombstone leak was fixed, 3000 capped
                              # HEALTHY growth (steady-state live set ~1-5k; validated never decays).
LINK_SUPPRESS_K = 3           # re-minted this many times after a rejection -> override the veto

assert LINK_INIT_WEIGHT < REPLAY_ONLY_W_CEILING  # a candidate can never reach the promotion bar


def _max_candidates() -> int:
    """Live, env-tunable store budget. ASTHENOS_LINK_MAX_CANDIDATES overrides LINK_MAX_CANDIDATES;
    0 or negative => unlimited (no eviction). Read live so a unit/daemon env change takes effect
    without a code edit."""
    import os
    try:
        v = int(os.environ.get("ASTHENOS_LINK_MAX_CANDIDATES", LINK_MAX_CANDIDATES))
    except (TypeError, ValueError):
        v = LINK_MAX_CANDIDATES
    return v if v > 0 else 0


def _evictable_key(candidates: dict):
    """The most disposable entry to free a budget slot — NEVER a validated discovery or a veto.
    Prefers a leftover 'decayed' tombstone (defensive; evict_decayed normally clears them first),
    else the weakest-weight active 'candidate'. Returns None if the store holds ONLY validated /
    suppressed entries (then the budget is respected rather than dropping signal)."""
    weakest = None
    weakest_w = None
    for k, c in candidates.items():
        st = getattr(c, "state", None)
        if st == "decayed":
            return k
        if st == "candidate" and (weakest_w is None or c.weight < weakest_w):
            weakest_w = c.weight
            weakest = k
    return weakest


@dataclass
class Candidate:
    state: str = "candidate"      # candidate | validated | genuine | decayed | suppressed
    weight: float = LINK_INIT_WEIGHT
    score: float = 0.0            # best cosine likelihood seen at mint/re-mint
    minted_occ: int = 0
    last_occ: int = 0
    remints: int = 0              # times independently re-proposed (mounting evidence)
    validated_occ: int = -1
    graduated_occ: int = -1       # occ when the LIVE edge first cleared the promotion bar (genuine)


def candidate_key(a: str, b: str) -> str:
    a = a if a.endswith(".md") else f"{a}.md"
    b = b if b.endswith(".md") else f"{b}.md"
    x, y = sorted([a, b])
    return f"{x}::{y}"


def propose(recalled_nodes: list, neighbor_fn: Callable[[str], list],
            connected: set, candidates: dict, occ: int) -> int:
    """Mint/re-mint weak candidates from each recalled node's cosine neighbors.

    neighbor_fn(node) -> [(neighbor_node, cosine), ...] (the injected index query). A candidate is
    minted ONLY toward a neighbor that (a) clears LINK_SCORE_FLOOR and (b) has NEVER co-occurred
    (`connected` = the union of genuine epiphanies edges + live Hebbian edges). Re-proposing an
    existing candidate refreshes it and, past LINK_SUPPRESS_K re-mints, OVERRIDES a suppression
    (the evidence outvoted the veto). Bounded by a tunable memory budget (_max_candidates): over it,
    the weakest DISPOSABLE candidate is evicted to admit the new one — a validated discovery or a veto
    is never dropped, and minting never wedges. Returns # newly minted.
    """
    minted = 0
    for node in recalled_nodes:
        try:
            nbrs = neighbor_fn(node) or []
        except Exception:
            continue
        for nbr, cos in nbrs:
            if nbr == node or cos < LINK_SCORE_FLOOR:
                continue
            key = candidate_key(node, nbr)
            if key in connected:
                continue                      # already co-occurred -> not a novel association
            c = candidates.get(key)
            if c is None:
                cap = _max_candidates()
                if cap and len(candidates) >= cap:
                    victim = _evictable_key(candidates)   # over budget: drop the weakest DISPOSABLE
                    if victim is None:                     # hypothesis (never a validated / veto), so
                        continue                           # minting never wedges. None => store is all
                    del candidates[victim]                 # validated/suppressed -> respect the budget.
                candidates[key] = Candidate(score=float(cos), minted_occ=occ, last_occ=occ)
                minted += 1
            else:
                c.remints += 1
                c.last_occ = occ
                c.score = max(c.score, float(cos))
                c.weight = min(REPLAY_ONLY_W_CEILING - 1e-6, c.weight + LINK_INIT_WEIGHT * 0.5)
                if c.state == "suppressed" and c.remints >= LINK_SUPPRESS_K:
                    c.state = "candidate"     # K-recurrence override of the veto
                elif c.state == "decayed":
                    c.state = "candidate"     # Phase 0: a re-surfacing pair is a LIVE hypothesis
                                              # again. Required by the decay-before-propose order
                                              # (epiphanies.consolidate): without it, decay could
                                              # mark a pair 'decayed' just before propose refreshes
                                              # it, and next fold's evict_decayed would drop a pair
                                              # that genuinely re-appeared.
    return minted


def validate(candidates: dict, genuine_edge_keys: set, occ: int) -> int:
    """A candidate whose pair has now earned a REAL genuine co-activation is VALIDATED — the
    linker predicted an association and reality confirmed it. Returns # newly validated."""
    n = 0
    for key, c in candidates.items():
        if c.state == "candidate" and key in genuine_edge_keys:
            c.state = "validated"
            c.validated_occ = occ
            n += 1
    return n


def decay(candidates: dict, occ: int) -> int:
    """Unvalidated candidates decay per OCCASION (faster than replay); below LINK_PRUNE they are
    marked decayed (a hypothesis that reality never confirmed). Returns # newly decayed."""
    n = 0
    for key, c in candidates.items():
        if c.state != "candidate":
            continue
        gap = occ - c.last_occ
        if gap > 0:
            c.weight *= (1.0 - LINK_DECAY) ** gap
        if c.weight < LINK_PRUNE:
            c.state = "decayed"
            n += 1
    return n


def evict_decayed(candidates: dict) -> int:
    """Drop terminal 'decayed' tombstones from the store. Returns # evicted.

    WHY (the 2026-06-19 wedge): decay() marks a dead hypothesis state='decayed' but LEAVES it in
    the dict, and nothing ever removed it -> decayed corpses accumulate and count toward
    LINK_MAX_CANDIDATES, so the store saturates (2949/3000 decayed) and propose() can no longer
    mint ANY new candidate (`len(candidates) >= LINK_MAX_CANDIDATES` is permanently true). Evicting
    them bounds the store by LIVE hypotheses only. VALIDATED discoveries and SUPPRESSED vetoes are
    preserved (only state=='decayed' is dropped); a still-plausible pair simply re-mints next run.
    """
    dead = [k for k, c in candidates.items() if getattr(c, "state", None) == "decayed"]
    for k in dead:
        del candidates[k]
    return len(dead)


def reconcile_genuine(candidates: dict, promotable_keys: set, occ: int) -> dict:
    """Re-derive the honest 'genuine' state from the LIVE edge bar each fold — REVERT, not latch.

    promotable_keys = candidate keys whose CURRENT epiphanies EdgeState meets the promotion bar
    (balancing.promotion_tier != NONE, i.e. S>=1.5 ∧ w>=0.85) AND is not suppressed — computed fresh
    by the caller from the live edges this fold. A 'validated' candidate (the linker discovered the
    association and it materialized a real co-activation at least once) whose key is in that set
    GRADUATES to 'genuine'; a 'genuine' candidate whose key is NO LONGER in the set REVERTS to
    'validated' (the one-time discovery record persists, but the 'currently strong enough' label
    drops the instant the live edge decays below the bar).

    This is a PURE state transition: it writes NO edge, mints NO chain, and touches NO recall
    surface. It fixes the latch bug — validate() set state='validated' at first materialization
    (S>=EPI_MAT_FLOOR ~0.099) and NEVER reverted, so the validated set filled with ghost edges whose
    evidence had since evaporated. 'genuine' here is a pure function of the live bar, so downstream
    logic that keys off it can never be lied to about current strength. Returns
    {newly_genuine, reverted, genuine_total, transitions:[(key, prev_state, new_state)]}."""
    newly_genuine = 0
    reverted = 0
    transitions = []
    for key, c in candidates.items():
        at_bar = key in promotable_keys
        if at_bar and c.state == "validated":
            c.state = "genuine"
            c.graduated_occ = occ
            newly_genuine += 1
            transitions.append((key, "validated", "genuine"))
        elif not at_bar and c.state == "genuine":
            c.state = "validated"          # live edge fell below the bar -> honest revert (no latch)
            reverted += 1
            transitions.append((key, "genuine", "validated"))
    genuine_total = sum(1 for c in candidates.values() if c.state == "genuine")
    return {"newly_genuine": newly_genuine, "reverted": reverted,
            "genuine_total": genuine_total, "transitions": transitions}


def is_genuine(c: Candidate) -> bool:
    return c.state == "genuine"


def reject(candidates: dict, key: str, occ: int) -> bool:
    """Operator/agent vetoes a candidate as not-related -> suppressed (re-mint K times to override)."""
    c = candidates.get(key)
    if c is None:
        return False
    c.state = "suppressed"
    c.remints = 0
    return True


def is_validated(c: Candidate) -> bool:
    return c.state == "validated"


def active_candidates(candidates: dict) -> dict:
    """The live hypotheses (not decayed, not suppressed, not yet validated)."""
    return {k: c for k, c in candidates.items() if c.state == "candidate"}


def to_jsonable(candidates: dict) -> dict:
    return {k: asdict(c) for k, c in candidates.items()}


def from_jsonable(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[k] = Candidate(**v)
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.linker
# Author:     code_warrior (Epiphanies v3 — Feature 2 / L3)
# Project:    Asthenosphere — SAM/IA — Epiphanies (cross-conversation linker)
# Version:    0.1.0  (pure candidate logic; SHADOW — never surfaces, never promotes)
# Phase:      build L3 Feature 2 — association discovery; observational shadow first.
# Layer:      core (pure library — no IO, no index; injected neighbor_fn)
# Role:       propose weak candidates from cosine neighbors of NEVER-co-occurred pairs; validate a
#             candidate when a real co-activation confirms it; decay unvalidated hypotheses; the
#             reject/suppress + K-recurrence override. Zero salience / zero promotion contribution.
# Stability:  new — the anti-confabulation invariant is structural in shadow (separate store,
#             never surfaced). Bounded by LINK_MAX_CANDIDATES + fast decay.
# Depends:    .config (REPLAY_NEIGHBOR_THRESHOLD / REPLAY_ONLY_W_CEILING).
# Exposes:    Candidate, candidate_key, propose, validate, decay, reject, is_validated,
#             active_candidates, to_jsonable, from_jsonable + the LINK_* constants.
# --------------------------------------------------------------------------
