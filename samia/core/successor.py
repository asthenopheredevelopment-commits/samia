"""samia.core.successor -- need-as-successor-representation term (P3 symmetric + P6 directed).

Layer 1 (Owns / Depends):
    Owns:    The need re-ranker N̂_c of the temporal-recall layer (FEAT-2026-06-11-
             memory-temporal-recall-formula-v01 §5.1-5.4) — the first multiplicative
             modulator in the gain envelope (1 + λN·N̂_c + λK·K̂_c + λD·D̂_c).
             Two pieces:
               1. The one-step stochastic kernel T: the EXISTING undirected co-activation
                  graph biomimetic/edge_weights.json (keys "a::b", value {w}), read once
                  per query and row-normalized into a Markov kernel T[i][j] = W[i][j]/Σ_k
                  W[i][k]. Row-stochasticity is the convergence guarantee — a sub-stochastic
                  matrix's spectral radius ≤ 1, so the discounted series converges for any
                  gSR ∈ [0, 0.8]. NO stored matrix; built query-locally and discarded.
               2. The query-local TRUNCATED power iteration from the active set A (top-8
                  vector hits): need = Σ_{t=0..L} gSR^t · T^t e_A, accumulated as L sparse
                  mat-vecs, then read per chain at best_node(c). This is the L-truncation
                  of the SR Neumann series M = (I − gSR·T)⁻¹ = Σ_t gSR^t·T^t — M is COMPUTED,
                  never learned, and never materialized as a dense inverse.
    Depends: samia.core.bio (_load_edge_weights — REUSED, not reinvented; the existing
             undirected edge store loader; and _bio_paths["episode_transitions"] for the
             P6 directed count store). No numpy needed: the frontier stays tiny
             (~1.27 edges/node) so a sparse dict walk is cheapest and zero-dependency.

Layer 2 (What / Why):
    What: need_vector(memory_dir, active_set) runs the truncated power iteration from the
          active set (a list of (node, cosine) seeds) over the row-normalized edge graph
          and returns a {node -> discounted-occupancy} map. need_at(need, node) reads that
          map at one node (the chain's best_node). gSR seeds 0.5 (bound [0, 0.8]); L seeds
          3 (bound {1,2,3,4}). The two self-documenting corners hold by construction:
          gSR=0 OR L=1 reduce the walk to its t=0 seed term — the EXACT 1-step proxy
          (need = p0, the normalized seed distribution) — and the deeper truncation gives
          multi-hop transitive reach.
    Why:  §5. The semantic base S_c and Hebbian density 0.05·H_c are both ONE-HOP,
          set-membership quantities that never propagate association ACROSS chains. N̂_c is
          exactly that cross-chain propagation: a discounted multi-step walk over the global
          co-activation graph, seeded from the retrieved set, scoring a chain by how
          reachable its best node is from what the query already activated (the EVB
          A+B merge — one operator, two truncation depths). Phase 1 (P3) runs on the SYMMETRIC
          edge graph (diffusion / associative proximity, NOT directed succession). Phase 2
          (P6, §5.5) layers a DIRECTED forward kernel on top: _build_forward_kernel reads the
          directed count matrix T_dir (biomimetic/episode_transitions.json, produced offline
          in idle_replay_tick from episode_seq order) and row-normalizes it into the forward
          SR M_fwd's one-step kernel — "what comes AFTER the active set" — with a PER-ROW
          symmetric fallback for legacy/no-order nodes (no migration). The reverse kernel
          M_rev (T_dir^T) is computed by _build_directed_out_edges(reverse=True) for credit
          assignment but is deliberately NOT wired into N̂ (the 5→4 fold keeps direction
          folded into the forward need, not a standalone term). With no episode_transitions.
          json the forward kernel is byte-identical to the phase-1 symmetric kernel.

Flag posture: P3 is read by the formula ONLY through context_extension._need_term_chain /
    _need_vector_for_envelope, which run ONLY when ASTHENOS_TEMPORAL_WEIGHT is on AND
    λN ≥ ε (§16.2-Q5 compute-skip). With the master flag off or λN=0 this module is on no
    retrieval path, so the chainogram_retrieve flag-off byte-identity holds. A corpus with
    no edge_weights.json (or an empty active set) yields an empty need map → every N̂_c = 0
    → the envelope reduces to 1.0 (fails open) — additive-optional, no migration.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import bio as _bio

# ── Seed parameters (§5.6 table). gSR and L join the joint-calibration vector later;
#    here they are the frozen seeds, read each call (live env, default seed) so a
#    calibration adapter can sweep them without re-import. Read-each-call mirrors the
#    context_extension temporal-weight readers. They are only consulted when the need
#    term is computed at all, which is gated off while λN=0.
SUCCESSOR_GSR_SEED = 0.5            # per-step discount; 0 ⇒ 1-step proxy (bound [0, 0.8])
SUCCESSOR_L_SEED = 3               # truncation depth / # sparse mat-vecs (bound {1,2,3,4})
SUCCESSOR_ACTIVE_SET_SIZE = 8     # SR seed width: top-8 of the top-24 vector hits (§5.3)

# Env names for the two SR hyperparameters (default to the seeds; clamped to bounds).
SUCCESSOR_GSR_ENV = "ASTHENOS_SUCCESSOR_GSR"
SUCCESSOR_L_ENV = "ASTHENOS_SUCCESSOR_L"

# Bounds (§5.6): gSR ∈ [0, 0.8] keeps T's discounted series convergent and bounds the
#   t=4 term at 0.8⁴≈0.41; L ∈ {1,2,3,4} caps the walk at four hops (truncated, never run
#   to convergence) — the structural over-diffusion guard.
SUCCESSOR_GSR_MIN = 0.0
SUCCESSOR_GSR_MAX = 0.8
SUCCESSOR_L_MIN = 1
SUCCESSOR_L_MAX = 4


def _node_key(node: str) -> str:
    """Canonical node filename used as an edge endpoint (edge_weights keys carry .md)."""
    return node if node.endswith(".md") else f"{node}.md"


def successor_gsr() -> float:
    """Resolve gSR (per-step discount), live env, clamped to [0, 0.8] (§5.6).

    What: reads ASTHENOS_SUCCESSOR_GSR each call; missing/unparseable ⇒ the 0.5 seed;
      always clamped to the [0, 0.8] convergence bound.
    Why: gSR seeds 0.5 and joins the joint-calibration vector; gSR=0 is the self-
      documenting 1-step-proxy corner. Clamping keeps a calibration sweep inside the
      row-stochastic convergence guarantee no matter what env value is set.
    """
    raw = os.environ.get(SUCCESSOR_GSR_ENV)
    if raw is None:
        val = SUCCESSOR_GSR_SEED
    else:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = SUCCESSOR_GSR_SEED
    return min(SUCCESSOR_GSR_MAX, max(SUCCESSOR_GSR_MIN, val))


def successor_l() -> int:
    """Resolve L (truncation depth), live env, clamped to {1,2,3,4} (§5.6).

    What: reads ASTHENOS_SUCCESSOR_L each call; missing/unparseable ⇒ the 3 seed;
      always clamped to the {1,2,3,4} bound.
    Why: L seeds 3 and joins the joint-calibration vector; L=1 is the self-documenting
      1-step-proxy corner (only the t=0 seed term survives). Clamping caps the walk at
      four sparse hops — the bounded-depth half of the over-diffusion guard.
    """
    raw = os.environ.get(SUCCESSOR_L_ENV)
    if raw is None:
        val = SUCCESSOR_L_SEED
    else:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = SUCCESSOR_L_SEED
    return min(SUCCESSOR_L_MAX, max(SUCCESSOR_L_MIN, val))


def _build_out_edges(weights: dict) -> dict:
    """Row-normalize the undirected edge store into a row-stochastic kernel T (§5.2).

    What: from edge_weights.json's "a::b" -> {w} undirected map, accumulate the per-node
      out-mass W[i][j] = w (symmetric: both endpoints of an undirected edge see the other),
      then row-normalize T[i][j] = W[i][j] / Σ_k W[i][k]. Returns {src -> {dst -> prob}}.
      A node with no out-mass simply has no row (absorbing — its walk mass stays put).
    Why: T is the one-step Markov kernel of the SR. Row-stochasticity is the convergence
      guarantee (EVB Q3): a sub-stochastic matrix's spectral radius ≤ 1, so the discounted
      Neumann series converges for any gSR ∈ [0, 0.8]. Built query-locally from the existing
      graph and discarded — no stored matrix, no learning, no staleness on write (§5.3).
      Self-loops (a degenerate "a::a" key, should never occur) are dropped (0 on the diag).
    """
    raw_out: dict[str, dict[str, float]] = {}
    for key, val in weights.items():
        parts = key.split("::")
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        if a == b:
            continue  # no self-loops (0 on the diagonal, §5.2)
        try:
            w = float(val.get("w", 0.0)) if isinstance(val, dict) else float(val)
        except (TypeError, ValueError):
            continue
        if w <= 0.0:
            continue
        # Undirected edge → symmetric out-mass: a sees b and b sees a (§5.2/§5.4).
        raw_out.setdefault(a, {})[b] = raw_out.setdefault(a, {}).get(b, 0.0) + w
        raw_out.setdefault(b, {})[a] = raw_out.setdefault(b, {}).get(a, 0.0) + w
    # Row-normalize each node's out-edges to sum to 1.0 (T row-stochastic).
    out: dict[str, dict[str, float]] = {}
    for src, dsts in raw_out.items():
        total = sum(dsts.values())
        if total <= 0.0:
            continue
        out[src] = {dst: m / total for dst, m in dsts.items()}
    return out


# ── Phase 2: directed SR over episode_transitions.json (§5.5) ─────────────────────


def _load_directed_transitions(memory_dir: Path) -> dict:
    """Read biomimetic/episode_transitions.json (the directed count matrix T_dir, §5.5).

    What: load the sparse directed-count store the offline pass (idle_replay_tick) writes —
      {"A->B": count} (DIRECTED keys, not edge_weights.json's sorted "a::b"). Missing /
      unreadable / empty → {} (every directed read fails open to the symmetric phase-1
      kernel). Mirrors bio._load_edge_weights' fail-soft read for the undirected store.
    Why: §5.5 — successor.py is the CONSUMER of the directed substrate. The file is read
      once per query, row-normalized on the fly, and discarded — same query-local discipline
      as phase 1 (§5.3), now over directed counts. No stored M, no full kernel materialization.
    """
    fp = _bio._bio_paths(memory_dir)["episode_transitions"]
    if not fp.exists():
        return {}
    try:
        data = __import__("json").loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _split_directed_key(key: str) -> tuple[str, str] | None:
    """Parse a directed transition key "A->B" into (src, dst) (§5.5).

    What: split on the "->" arrow into the (earlier, later) endpoints; reject malformed
      keys (no arrow, empty endpoint, or a self-edge A->A which should never be written).
    Why: the directed store uses "->" (not the undirected "::") precisely so a forward
      A->B and a reverse B->A can coexist; a robust parse keeps a corrupt key from
      polluting the kernel (fail-soft, mirroring _build_out_edges' malformed-key skip).
    """
    parts = key.split("->")
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    if not a or not b or a == b:
        return None
    return a, b


def _build_directed_out_edges(transitions: dict, *, reverse: bool = False) -> dict:
    """Row-normalize the directed count matrix into a row-stochastic kernel (§5.5).

    What: from episode_transitions.json's "A->B" -> count, accumulate per-source out-mass and
      row-normalize T_dir[A][B] = count(A->B) / Σ_C count(A->C). Returns {src -> {dst -> prob}}.
      reverse=False builds the FORWARD kernel M_fwd's one-step matrix (A->B as A sees B) — the
      planning/need direction ("what comes AFTER the active set"). reverse=True builds the
      TRANSPOSE T_dir^T (B sees A) — the reverse/credit-assignment kernel ("what comes BEFORE").
      Counts ≤ 0 and malformed/self keys are dropped; a node with no out-mass has no row.
    Why: §5.5 — the forward and reverse SR are two readings of the SAME directed operator at
      one truncation. M_fwd is the N̂ the formula wants (it already prefers temporally-after
      nodes — the 5→4 fold that absorbs the standalone ordinal term R̂). M_rev (reverse=True)
      is COMPUTED here for completeness/credit-assignment but is deliberately NOT wired into
      N̂ (no caller passes it through need_at) — available for a future use, inert today.
      Row-stochasticity keeps the discounted series convergent across gSR ∈ [0, 0.8].
    """
    raw_out: dict[str, dict[str, float]] = {}
    for key, val in transitions.items():
        parsed = _split_directed_key(key)
        if parsed is None:
            continue
        a, b = parsed
        try:
            count = float(val) if not isinstance(val, dict) else float(val.get("count", 0.0))
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        # Forward: a (earlier) → b (later). Reverse: transpose, b → a.
        src, dst = (b, a) if reverse else (a, b)
        raw_out.setdefault(src, {})[dst] = raw_out.setdefault(src, {}).get(dst, 0.0) + count
    out: dict[str, dict[str, float]] = {}
    for src, dsts in raw_out.items():
        total = sum(dsts.values())
        if total <= 0.0:
            continue
        out[src] = {dst: m / total for dst, m in dsts.items()}
    return out


def _build_forward_kernel(memory_dir: Path) -> dict:
    """The forward SR one-step kernel with a per-node symmetric fallback (§5.5).

    What: row-normalize the directed counts into the forward kernel M_fwd's T_dir; for any
      source node that has NO directed out-mass (a legacy node never seen as the earlier end
      of an ordered pair, or a corpus with no transitions at all), fall back to that node's
      SYMMETRIC phase-1 row from edge_weights.json. Returns {src -> {dst -> prob}}, each row
      individually row-stochastic. When episode_transitions.json is empty this is exactly the
      phase-1 symmetric kernel (every row falls back), so phase 1 is the strict default.
    Why: §5.5 "Legacy fallback (EVB Q4) — build T_dir where order is known, fall back to
      symmetric T where it isn't. No node is ever un-rankable." The fallback is PER SOURCE
      ROW: directed succession is used wherever episode_seq established it, and undirected
      diffusion fills the rest, so a mixed-population corpus (some episode_seq-bearing, some
      legacy) ranks with no migration and no node left absorbing-by-omission.
    """
    directed = _build_directed_out_edges(
        _load_directed_transitions(memory_dir), reverse=False)
    symmetric = _build_out_edges(_bio._load_edge_weights(memory_dir))
    if not directed:
        return symmetric  # no directed counts → strict phase-1 symmetric kernel
    out: dict[str, dict[str, float]] = {}
    # Every node either source has a row for: directed where known, symmetric otherwise.
    for src in set(directed) | set(symmetric):
        row = directed.get(src)
        if not row:
            row = symmetric.get(src)            # legacy/no-order row → symmetric fallback
        if row:
            out[src] = row
    return out


def _seed_distribution(active_set) -> dict:
    """Build the normalized seed distribution p0 over the active set (§5.3).

    What: active_set is a list of (node, cosine) seeds (or bare node names, treated as
      uniform weight 1.0). p0[a] = max(cos, 0) for a in A, then normalized to sum 1 so the
      walk starts from a probability distribution. A non-positive or all-zero seed mass
      falls back to a uniform distribution over the seeds (so the walk is never empty when
      the active set is non-empty).
    Why: §5.3 seeds p0 from the query's real, relevant hits (cosines), so the walk diffuses
      from what the query already lit up — never from the whole graph. Normalizing to a
      distribution makes the t=0 term the exact 1-step proxy and keeps the discounted
      occupancies on a stable, comparable scale across queries.
    """
    seeds: dict[str, float] = {}
    for item in active_set:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            node, cos = item[0], item[1]
        else:
            node, cos = item, 1.0
        if not node:
            continue
        try:
            c = float(cos)
        except (TypeError, ValueError):
            c = 0.0
        seeds[_node_key(node)] = max(seeds.get(_node_key(node), 0.0), max(c, 0.0))
    if not seeds:
        return {}
    total = sum(seeds.values())
    if total <= 0.0:
        # Degenerate seed mass (all cosines ≤ 0): uniform over the seeds.
        u = 1.0 / float(len(seeds))
        return {k: u for k in seeds}
    return {k: v / total for k, v in seeds.items()}


def need_vector(memory_dir: Path, active_set, *,
                gsr: float | None = None, l: int | None = None) -> dict:
    """Query-local truncated power iteration: need = Σ_{t=0..L} gSR^t·T^t e_A (§5.3).

    What: row-normalize the existing edge graph into T, build the normalized seed
      distribution p0 over the active set, then accumulate
          need = p0;  pk = p0
          for t in 1..L:  pk = gSR · (pk @ T);  need += pk
      as L sparse mat-vecs (each touches only the out-neighbors of currently-nonzero
      entries — the frontier stays tiny at ~1.27 edges/node). Returns the {node ->
      discounted-occupancy} map need; read per chain at best_node(c) via need_at.
      gsr/l default to the live-env seeds (clamped to bounds).
    Why: §5.2/5.3 — this is the L-truncation of the SR Neumann series M = (I − gSR·T)⁻¹,
      computed query-locally and discarded (no stored M, no staleness). The TWO corners
      are self-documenting and load-bearing for the contract: gSR=0 OR L=1 leaves only the
      t=0 term need = p0 — the EXACT 1-step proxy (no diffusion); a deeper truncation gives
      multi-hop transitive reach. Fails open: an empty graph or empty active set → {} →
      every N̂_c reads 0.0 → the envelope reduces to 1.0 (additive-optional, no migration).
    """
    g = successor_gsr() if gsr is None else min(
        SUCCESSOR_GSR_MAX, max(SUCCESSOR_GSR_MIN, float(gsr)))
    depth = successor_l() if l is None else min(
        SUCCESSOR_L_MAX, max(SUCCESSOR_L_MIN, int(l)))

    p0 = _seed_distribution(active_set)
    if not p0:
        return {}
    # t=0 term: the seed distribution itself (the 1-step proxy when gSR=0 or L=1).
    need: dict[str, float] = dict(p0)
    # gSR=0 short-circuits every hop to 0 mass → need stays p0 (the proxy corner).
    if g <= 0.0:
        return need

    # Phase 2 (§5.5): the FORWARD directed kernel where episode_seq established order, with a
    # per-row symmetric phase-1 fallback for legacy/no-order nodes. With no
    # episode_transitions.json this is BYTE-IDENTICAL to the phase-1 symmetric kernel — phase
    # 1 stays the strict default until the directed substrate is produced (P6 producer runs
    # only when the master temporal flag is on). M_rev (T_dir^T) is built by
    # _build_directed_out_edges(reverse=True) but is NOT wired into N̂ (§5.5).
    out_edges = _build_forward_kernel(memory_dir)
    if not out_edges:
        return need  # no graph → diffusion is a no-op; need is the seed proxy

    pk = dict(p0)
    for _t in range(1, depth + 1):
        nxt: dict[str, float] = {}
        for src, mass in pk.items():
            if mass == 0.0:
                continue
            row = out_edges.get(src)
            if not row:
                continue  # absorbing node: its mass leaves the frontier (no out-edges)
            for dst, prob in row.items():
                nxt[dst] = nxt.get(dst, 0.0) + g * mass * prob
        if not nxt:
            break  # frontier exhausted; deeper hops add nothing
        for dst, m in nxt.items():
            need[dst] = need.get(dst, 0.0) + m
        pk = nxt
    return need


def need_at(need: dict, node: str | None) -> float:
    """Read the discounted-occupancy need map at one node (the chain's best_node, §5.3).

    What: N_raw(c) = need[best_node(c)]; a node absent from the map (never reached by the
      walk) reads 0.0.
    Why: §5.3 projects the per-node need vector to chains by reading at best_node(c). The
      caller pool min-max normalizes the resulting per-chain raw values into N̂_c ∈ [0,1],
      so an absent (0.0) chain contributes no lift — a bounded modulator that can never flip
      sign or dominate the additive base.
    """
    if not need or not node:
        return 0.0
    return float(need.get(_node_key(node), 0.0))


# [Asthenosphere] samia.core.successor
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P3+P6 — need-as-successor-
#             representation term. P3 (§5.1-5.4, symmetric phase 1): row-normalize the
#             EXISTING undirected edge_weights.json into a row-stochastic kernel T; query-
#             local TRUNCATED power iteration from the top-8 active set: need = Σ_{t=0..L}
#             gSR^t·T^t e_A (seeds gSR=0.5, L=3; gSR∈[0,0.8], L∈{1,2,3,4}); NO stored matrix.
#             gSR=0 OR L=1 == the exact 1-step proxy. P6 (§5.5, directed): _build_forward_
#             kernel reads the directed count matrix T_dir (episode_transitions.json, produced
#             offline in idle_replay_tick from episode_seq order) and row-normalizes it into
#             the forward SR M_fwd, with a per-row symmetric fallback for legacy/no-order
#             nodes. The reverse kernel M_rev (T_dir^T) is computed but NOT wired into N̂ (the
#             5→4 fold). No standalone λR. With no episode_transitions.json the forward kernel
#             is byte-identical to phase 1. Inert at retrieval until ASTHENOS_TEMPORAL_WEIGHT +
#             λN≥ε flip it on; flag-off / λN=0 is a byte-identical no-op.
# Layer:      core (pure library, no daemon dependency)
# Role:       compute the multiplicative need modulator N̂_c (the EVB A+B merge)
# Stability:  stable -- v1.0.0; additive-optional, inert until the temporal flag + λN flip on.
# ErrorModel: fail-open at every edge — malformed edge/transition keys and unparseable
#             weights are skipped; an empty graph or empty active set yields an empty need
#             map, so every N̂_c reads 0.0 and the envelope reduces to 1.0 (no migration).
# Depends:    bio (_load_edge_weights + _bio_paths[episode_transitions] — REUSED stores).
#             stdlib (os, pathlib); json via __import__ in the directed reader.
# Exposes:    successor_gsr, successor_l, need_vector, need_at. Constants:
#             SUCCESSOR_GSR_SEED, SUCCESSOR_L_SEED, SUCCESSOR_ACTIVE_SET_SIZE,
#             SUCCESSOR_GSR_ENV, SUCCESSOR_L_ENV, and the GSR/L bound constants.
# Lines:      423
# --------------------------------------------------------------------------
