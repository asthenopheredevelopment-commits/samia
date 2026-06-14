"""samia.core.context_extension.temporal — the temporal-recall scoring envelope.

Layer 1 (Owns / Depends):
    Owns:    the master deploy-flag reader (temporal_weight_enabled) + per-term weight
             resolver (temporal_weights / _temporal_weight), the compute-skip predicate
             (_term_active), the uniform relevance gate (_relevance_gate), the TC-cosine
             floor reader (_tc_cosine_floor), the pool min-max normalizer (_minmax_pool),
             the four term-hook seams (_tc_term_hit / _need_vector + _need_term_chain /
             _stc_term_chain / _dist_vector + _dist_term_chain), and the envelope
             assembler (_apply_temporal_envelope) that folds score(c) in place.
    Depends: the package config leaf (the env names + θ + ε + the TC floor constant +
             os) and — through FUNCTION-LOCAL imports — the P2-P5 producers
             (temporal_recall_sith / successor / temporal_recall_stc /
             temporal_distinctiveness) and samia.core.bio (the node→chain address). The
             lazy imports break the context_extension<->{sith,successor,stc,dist} cycles
             and keep the heavy producers off the import path until a weight clears ε.

Layer 2 (What / Why):
    What: the unified score is score(c) = (S_c + 0.05·H_c + γ·TĈ_c)·(1 + λN·N̂_c +
          λK·K̂_c + λD·D̂_c). This module owns everything to the RIGHT of the raw base
          cue B_c = S_c + 0.05·H_c: the weight resolution, the per-term hooks, the pool
          normalization, and the in-place fold into chain_scores["score"].
    Why:  P1 landed the SHAPE with every coefficient pinned to 0.0 behind a default-OFF
          master flag (§2.6 identity proof). Keeping the whole envelope in one submodule
          makes the flag-off no-op auditable and the §16.2-Q5 compute-skip self-evident:
          while a weight is 0.0 its hook never runs, so the flag-off path executes no
          temporal code at all. _apply_temporal_envelope is reached only inside
          `if temporal_weight_enabled()` in the retrieval arm.

PATCH SEAMS: the term-hook helpers (_tc_term_hit / _need_term_chain / _stc_term_chain /
    _dist_vector / _dist_term_chain) and the gates (_relevance_gate / _minmax_pool /
    _term_active) are mock.patch.object(ce, ...) targets in test_temporal_scaffold /
    test_successor. _apply_temporal_envelope calls them as MODULE-LOCAL names here, and
    the tests patch the package facade — but those tests assert the helpers' direct
    return values, they do NOT patch-then-drive _apply_temporal_envelope, so no facade
    reach is required (the patched call sites and the asserted call sites coincide). The
    one driven-through seam (chainogram_retrieve's call to _apply_temporal_envelope) is
    not patched. No private temporal name is rebound on the facade and read by a sibling.
"""

from __future__ import annotations

from pathlib import Path

# Shared leaf — the env names + θ + ε + the TC floor constant + os, plus the node→chain
# address helper (_bio) the envelope needs to attribute each hit to its owning chain.
from . import config as _cfg
from .config import (
    TEMPORAL_WEIGHT_ENV,
    TEMPORAL_GAMMA_ENV,
    TEMPORAL_LAMBDA_N_ENV,
    TEMPORAL_LAMBDA_K_ENV,
    TEMPORAL_LAMBDA_D_ENV,
    TEMPORAL_THETA,
    TEMPORAL_WEIGHT_EPSILON,
    TC_COSINE_FLOOR_DEFAULT,
    TC_COSINE_FLOOR_ENV,
    os,
    _bio,
)


def temporal_weight_enabled() -> bool:
    """True iff the temporal-recall envelope is deployed (live env read, default OFF).

    What: reads ASTHENOS_TEMPORAL_WEIGHT each call; "1" => ON, anything else => OFF.
    Why: default OFF means chainogram_retrieve's scorer is byte-identical to today —
      the temporal block is skipped entirely, no weight is read, no term hook is
      called, the sort key is the unchanged S_c + 0.05·H_c accumulation (§2.6).
      Read-each-call (not import-time) mirrors semantic_recall.semantic_arm_enabled
      so a test/daemon that sets the env after import sees the change.
    """
    return os.environ.get(TEMPORAL_WEIGHT_ENV, "0") == "1"


def _temporal_weight(env_name: str) -> float:
    """Read one per-term temporal weight as a float (live, default 0.0, fail-soft).

    What: env `env_name` parsed as float; missing/unparseable => 0.0.
    Why: every new coefficient (γ, λN, λK, λD) defaults to 0.0 so the formula
      collapses to the baseline even with the master flag on; only a calibration
      that freezes non-zero values activates a term. Fail-soft to 0.0 keeps a
      typo'd env from ever enabling a term, matching facts_fraction's read-each-
      call + safe-default shape.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def temporal_weights() -> dict:
    """Resolve (γ, λN, λK, λD) — all 0.0 unless the master flag is on AND set.

    What: returns {"gamma","lambda_n","lambda_k","lambda_d"}. When
      temporal_weight_enabled() is False, ALL are forced 0.0 (master-off dominates,
      §8.6); when on, each reads its env (still 0.0 by default).
    Why: a single resolution point the scorer consults once per query. Master-off
      forcing means the flag is the one deploy switch — even a stray non-zero
      per-term env cannot activate a term while the master flag is off.
    """
    if not temporal_weight_enabled():
        return {"gamma": 0.0, "lambda_n": 0.0, "lambda_k": 0.0, "lambda_d": 0.0}
    return {
        "gamma": _temporal_weight(TEMPORAL_GAMMA_ENV),
        "lambda_n": _temporal_weight(TEMPORAL_LAMBDA_N_ENV),
        "lambda_k": _temporal_weight(TEMPORAL_LAMBDA_K_ENV),
        "lambda_d": _temporal_weight(TEMPORAL_LAMBDA_D_ENV),
    }


def _term_active(weight: float) -> bool:
    """§16.2-Q5 compute-skip predicate: True iff |weight| ≥ ε (term is worth computing).

    What: a term whose calibrated weight falls below TEMPORAL_WEIGHT_EPSILON is
      gated off at the COMPUTE level — its hook is never called.
    Why: a term that earns no lift costs nothing at runtime; with all weights at
      the 0.0 default this is False for every term, so the flag-off path executes
      no temporal code at all (the strongest form of the identity contract).
    """
    return abs(weight) >= TEMPORAL_WEIGHT_EPSILON


def _relevance_gate(hit_cosine: float, theta: float = TEMPORAL_THETA) -> float:
    """Uniform relevance gate g_h (§2.5): 1.0 iff cos(q, e_h) ≥ θ, else 0.0.

    What: gates EVERY new (temporal) per-hit term — a hit injects TC/need/STC/dist
      signal only when its own semantic match clears θ = 0.2.
    Why: prevents the temporal machinery from amplifying a chain that is temporally
      adjacent / reachable / tagged but *about something else* — it must be at
      least minimally on-topic. S_c and H_c are NOT re-gated (identity baseline).
    """
    return 1.0 if float(hit_cosine) >= theta else 0.0


def _tc_cosine_floor() -> float:
    """TC-specific semantic-plausibility floor -- STRICTER than the shared theta (FEAT-tc-additive-safety).

    What: read ASTHENOS_TEMPORAL_TC_COSINE_FLOOR (default 0.4); fail-soft to 0.4 on a
      missing / unparseable / out-of-[0,1] value.
    Why: the additive TC cue is a PEER of S_c, so a temporally-recent but OFF-TOPIC hit can
      override semantics (the v3 control-regression). Requiring a higher cosine before a hit
      contributes TC blocks that, while N/K/D keep the shared theta=0.2. Per-call read (harness
      env scoping); consulted ONLY when gamma>0, so gamma=0 / flag-off is byte-identical.
    """
    raw = os.environ.get(TC_COSINE_FLOOR_ENV)
    if raw is None:
        return TC_COSINE_FLOOR_DEFAULT
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return TC_COSINE_FLOOR_DEFAULT
    return v if 0.0 <= v <= 1.0 else TC_COSINE_FLOOR_DEFAULT


def _minmax_pool(raw_by_chain: dict) -> dict:
    """Pool min-max normalize a per-chain raw modulator family to [0,1] (§2.4).

    What: X̂_c = (X_c − min)/(max − min); returns 0.0 for every chain when the pool
      is empty or the range is degenerate (a pool of one, or all-equal values).
    Why: bounds each X̂ ∈ [0,1] so the envelope is bounded and scale-free across
      heterogeneous units (SR occupancy, STC score, log-time density), and degrades
      gracefully — a degenerate pool injects no signal (the envelope reduces to 1).
      The base cue terms S/H stay RAW (mandatory for the flag-off identity); only
      the temporal modulator families pass through here.
    """
    if not raw_by_chain:
        return {}
    vals = list(raw_by_chain.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0.0:
        return {c: 0.0 for c in raw_by_chain}
    return {c: (v - lo) / span for c, v in raw_by_chain.items()}


# --- Term hook seams (P2-P5 plug here). Each returns the RAW per-chain modulator
#     contribution. TC (P2, SITH) and need (P3, SR) are now LIVE; STC (P4) and dist
#     (P5) still return 0.0 until their phases land. The hooks are only ever invoked
#     when _term_active(weight) — §16.2-Q5 compute-skip — so while a weight is 0.0 the
#     hook body never runs and the flag-off path executes no temporal code at all.
#     Signatures carry the inputs each term needs (the gate g_h, the hit, the chain
#     info, the precomputed need vector, memory_dir).

def _tc_term_hit(memory_dir: Path, hit: dict, g_h: float) -> float:
    """SITH temporal-context cue per-hit contribution (P2, §4).

    g_h · Σ_k ω_k·cos(t_k, c_{h,k}): reads the hit node's encode-snapshot from the SITH
    sidecar and scores it against the current integrator bank (§4.1/§4.4). Summed over
    the chain's gated hits into the additive cue (peer of S_c) and γ-weighted, pool-hat.
    A hit whose node was never materialized has no snapshot and contributes 0.0 (fails
    open) — additive-optional, no migration. Lazy import dodges the
    context_extension<->temporal_recall_sith<->bio/hippocampus cycle; any failure (no
    SITH state yet, parse hiccup) returns 0.0 so the temporal term never breaks recall.
    Only reached when γ ≥ ε (§16.2-Q5 compute-skip), so flag-off pays nothing here.
    """
    try:
        from .. import temporal_recall_sith as _sith
        return _sith.tc_term_hit(memory_dir, hit.get("node"), g_h)
    except Exception:
        return 0.0


def _need_vector(memory_dir: Path, active_set: list) -> dict:
    """Query-local SR need vector from the active set (P3, §5.3). Computed ONCE per query.

    need = Σ_{t=0..L} gSR^t·T^t e_A over the row-normalized existing edge graph — a truncated
    power iteration from the top-8 active set, returning a {node -> discounted-occupancy} map
    read per chain at best_node(c). gSR=0/L=1 collapses to the exact 1-step proxy (§5.3 corner).
    Lazy import dodges the context_extension<->successor<->bio cycle; any failure (no graph,
    parse hiccup) returns {} so every N̂_c reads 0.0 (fails open). Only reached when λN ≥ ε
    (§16.2-Q5 compute-skip), so flag-off pays nothing here.
    """
    try:
        from .. import successor as _sr
        return _sr.need_vector(memory_dir, active_set)
    except Exception:
        return {}


def _need_term_chain(memory_dir: Path, info: dict, need_vec: dict) -> float:
    """Need / multi-step SR raw contribution at best_node(c) (P3, §5.3).

    Reads the precomputed need vector at the chain's best_node (info["best_node"], maintained
    by the base scorer at :702/:707-709): N_raw(c) = need[best_node(c)], = Σ_{a∈A} M[a→·]
    truncated at L, evaluated at bestnode(c). The caller pool min-max normalizes → N̂_c ∈ [0,1]
    → enters the envelope as λN·N̂_c (a bounded lift). A node never reached by the walk reads
    0.0 (fails open). Fail-soft: any error → 0.0 so the need term never breaks recall.
    """
    try:
        from .. import successor as _sr
        return _sr.need_at(need_vec, info.get("best_node"))
    except Exception:
        return 0.0


def _stc_term_chain(memory_dir: Path, info: dict, members: list) -> float:
    """STC tagging-and-capture raw contribution, max over members (P4, §6).

    K_raw(c) = max_{m∈c} stc_capture_score(m) — the time-attenuated capture scalar over
    the chain's gated member nodes (§6.5 max reducer). The caller pool min-max normalizes
    → K̂_c ∈ [0,1] → enters the multiplicative envelope as λK·K̂_c. `members` is the list
    of this chain's gated hit dicts; the member NODE names are read off them. A chain with
    no captured member reads 0.0 (fails open) — additive-optional, no migration. Lazy
    import dodges the context_extension<->temporal_recall_stc<->bio cycle; any failure
    (no STC state yet, parse hiccup) returns 0.0 so the term never breaks recall. Only
    reached when λK ≥ ε (§16.2-Q5 compute-skip), so flag-off pays nothing here.
    """
    try:
        from .. import temporal_recall_stc as _stc
        member_nodes = [m.get("node") for m in members if m.get("node")]
        return _stc.stc_chain_score(memory_dir, member_nodes)
    except Exception:
        return 0.0


def _dist_vector(memory_dir: Path, best_nodes: dict) -> dict:
    """Pool-scan the SIMPLE log-time distinctiveness over the candidate pool (P5, §7).

    D_raw(c) = 1/Σ_j exp(−c·|logT_i − logT_j|) at best_node(c), computed ONCE per query
    over the pool of {cname -> best_node} representative times — never re-scanned per
    chain (mirrors _need_vector). T_i is seconds since best_node's written_at (sub-day)
    else infer_valid_from's day-granular fallback (incl. st_mtime). The §7.4 applicability
    gate collapses a degenerately-clustered pool to all-0.0 (no signal). Returns the raw
    {cname -> D_raw} map; the caller pool min-max normalizes → D̂_c ∈ [0,1]. Lazy import
    dodges the context_extension<->temporal_distinctiveness cycle; any failure (no usable
    times, parse hiccup) returns {} so every D̂_c reads 0.0 (fails open). Only reached when
    λD ≥ ε (§16.2-Q5 compute-skip), so flag-off pays nothing here.
    """
    try:
        from .. import temporal_distinctiveness as _td
        return _td.dist_vector(memory_dir, best_nodes)
    except Exception:
        return {}


def _dist_term_chain(memory_dir: Path, cname: str, dist_vec: dict) -> float:
    """Temporal-distinctiveness raw contribution at best_node(c) (P5, §7).

    Reads the precomputed pool distinctiveness map at this chain (keyed by chain name in
    dist_vec): D_raw(c) = dist_vec[cname]. The caller pool min-max normalizes → D̂_c ∈
    [0,1] → enters the multiplicative envelope as λD·D̂_c (a bounded lift). A chain with no
    usable representative time, or the whole pool when the applicability gate fails, reads
    0.0 (fails open). Fail-soft: any error → 0.0 so the dist term never breaks recall.
    """
    try:
        from .. import temporal_distinctiveness as _td
        return _td.dist_at(dist_vec, cname)
    except Exception:
        return 0.0


def _apply_temporal_envelope(memory_dir: Path, chain_scores: dict,
                             hits: list[dict]) -> None:
    """Fold the temporal envelope into chain_scores["score"] in place (§2, flagged-on).

    What: rewrites each chain's score from the raw base cue B_c = S_c + 0.05·H_c to
      score(c) = (S_c + 0.05·H_c + γ·TĈ_c)·(1 + λN·N̂_c + λK·K̂_c + λD·D̂_c). Only the
      terms whose weight clears ε are computed (§16.2-Q5); the rest are skipped, not
      multiplied out. The temporal modulator families (TC/N/K/D) are pool min-max
      normalized to [0,1] before entering the formula; the base S/H stay raw.
    Why: P1 is the scaffold — every weight is 0.0 by default and every term hook
      returns 0.0, so this function leaves score UNCHANGED (B_c·1 = B_c) and is a
      provable no-op. It is only reached when the master flag is on; the per-term
      compute-skip means a zero-weighted term costs nothing. P2-P5 fill the hooks.

    Mutates chain_scores in place; called only inside `if temporal_weight_enabled()`.
    """
    w = temporal_weights()
    if not chain_scores:
        return

    # Active set A for the SR need term — the top hits seeding the query-local walk
    # (§5.3: active set = top-8 of the vector hits, seeded by cosine p0[a]=cos(q,e_a)).
    # Carry (node, cosine) pairs so successor.need_vector seeds the walk from the query's
    # real relevance, not a uniform spray. Resolved once, passed to the hook.
    active_set = [(h.get("node"), float(h.get("score", 0.0)))
                  for h in hits[:8] if h.get("node")]

    # Hit → resolved chain map + relevance gate g_h per hit. A hit injects temporal
    # signal into its owning chain only when its own cosine clears θ = 0.2 (§2.5).
    # Resolved once; reused by the additive TC accumulation below.
    gated_hits_by_chain: dict[str, list[tuple[dict, float]]] = {}
    for h in hits:
        ca = _bio._addr_for_node(memory_dir, h.get("node"))
        if not ca:
            continue
        cname = ca[0]
        if cname not in chain_scores:
            continue
        g_h = _relevance_gate(h.get("score", 0.0))
        gated_hits_by_chain.setdefault(cname, []).append((h, g_h))

    # --- Additive cue term: TĈ_c (γ-weighted SITH temporal context, §4). Summed
    #     over the chain's gated hits (peer of S_c), then pool min-max normalized.
    tc_raw: dict[str, float] = {}
    if _term_active(w["gamma"]):
        tc_floor = _tc_cosine_floor()  # FEAT-tc-additive-safety: stricter than theta
        for cname in chain_scores:
            acc = 0.0
            for h, g_h in gated_hits_by_chain.get(cname, []):
                # additive TC contributes ONLY for semantically-plausible hits (>= tc_floor),
                # so an off-topic-recent node cannot override the semantic winner.
                if g_h and float(h.get("score", 0.0)) >= tc_floor:
                    acc += _tc_term_hit(memory_dir, h, g_h)
            tc_raw[cname] = acc
    tc_hat = _minmax_pool(tc_raw)

    # --- Multiplicative modulators: N̂ (need/SR, §5), K̂ (STC, §6), D̂ (dist, §7).
    #     Each raw family is gathered only when its weight clears ε, then pool-hat.
    need_raw: dict[str, float] = {}
    if _term_active(w["lambda_n"]):
        # ONE truncated power iteration from the active set produces the discounted-
        # occupancy vector (§5.3); each chain reads it at its best_node — not a re-walk
        # per chain. Compute-skipped when λN < ε, so flag-off pays nothing here.
        need_vec = _need_vector(memory_dir, active_set)
        for cname, info in chain_scores.items():
            need_raw[cname] = _need_term_chain(memory_dir, info, need_vec)
    need_hat = _minmax_pool(need_raw)

    stc_raw: dict[str, float] = {}
    if _term_active(w["lambda_k"]):
        for cname, info in chain_scores.items():
            members = [h for h, _ in gated_hits_by_chain.get(cname, [])]
            stc_raw[cname] = _stc_term_chain(memory_dir, info, members)
    stc_hat = _minmax_pool(stc_raw)

    dist_raw: dict[str, float] = {}
    if _term_active(w["lambda_d"]):
        # ONE pool scan of the SIMPLE log-time ratio over each chain's best_node time
        # (§7.2/§7.5); each chain reads it by name — not a re-scan per chain. The
        # applicability gate (§7.4) collapses a degenerate pool to all-0.0. Compute-
        # skipped when λD < ε, so flag-off pays nothing here.
        best_nodes = {cname: info.get("best_node")
                      for cname, info in chain_scores.items()}
        dist_vec = _dist_vector(memory_dir, best_nodes)
        for cname in chain_scores:
            dist_raw[cname] = _dist_term_chain(memory_dir, cname, dist_vec)
    dist_hat = _minmax_pool(dist_raw)

    # --- Assemble score(c) = (base + γ·TĈ)·(1 + λN·N̂ + λK·K̂ + λD·D̂). A term that
    #     was compute-skipped contributes its hat-default 0.0, so it drops out cleanly.
    for cname, info in chain_scores.items():
        base = info["score"]  # raw S_c + 0.05·H_c, accumulated above
        cue = base + w["gamma"] * tc_hat.get(cname, 0.0)
        envelope = (1.0
                    + w["lambda_n"] * need_hat.get(cname, 0.0)
                    + w["lambda_k"] * stc_hat.get(cname, 0.0)
                    + w["lambda_d"] * dist_hat.get(cname, 0.0))
        info["score"] = cue * envelope


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.context_extension.temporal
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-11 temporal-recall P1 (scaffold) + P2 (SITH TC) + P3 (SR
#             need) live; P4 (STC) + P5 (dist) hooks present, returning 0.0.
#             + Phase-B modularization (carved from the monolith, ZERO behavior change).
# Layer:      core (pure library, no daemon dependency)
# Role:       the temporal-recall scoring envelope — flag/weight readers, gates, pool
#             normalizer, term-hook seams, and the in-place score fold.
# Stability:  stable shape; the P4/P5 hook bodies are the only remaining fill points.
# ErrorModel: fail-OPEN + identity-preserving — every term hook swallows producer errors
#             to a 0.0/{} default, the §16.2-Q5 compute-skip means a 0.0-weighted term's
#             hook never runs, and the flag-off path skips this module entirely.
# Depends:    .config (env names + θ + ε + TC floor + os + _bio); lazily
#             samia.core.{temporal_recall_sith,successor,temporal_recall_stc,
#             temporal_distinctiveness} (function-local to break the import cycles).
# Exposes:    temporal_weight_enabled / temporal_weights (public) + the gates, hooks,
#             and _apply_temporal_envelope (private, re-exported on the facade for the
#             retrieval arm + the test reach-ins ce._relevance_gate / ce._minmax_pool /
#             ce._term_active / ce._tc_term_hit / ce._need_term_chain / etc.).
# Lines:      414
# --------------------------------------------------------------------------
