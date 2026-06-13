"""samia.core.context_extension — context-extension primitives.

Carved from memory_context_extension.py. Library plane parameterized on
memory_dir; CLI wrapper does argparse + print only.

Where samia.core.bio implements per-node mechanisms (recall, edge
strengthening, retrieval gates), this module implements *context-budget*
primitives that work with — not against — production compaction.

Primitives (parameterized on memory_dir):
    chainogram_retrieve, chainogram_retrieve_bridged,
    chainogram_retrieve_hybrid, chainogram_retrieve_reranked,
    chainogram_retrieve_contextual,
    frozen_prefix_block, tier_flow_for_budget,
    episodic_to_semantic_candidates, idle_replay_tick,
    sm2_review_update, sm2_due_for_review, compaction_skip_filter
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from . import bio as _bio
from . import chain as _ct
from . import temporal as _tq
from . import vector as _vi
from . import web_store as _ws

try:
    from . import entity_index as _ei
except ImportError:
    _ei = None
try:
    from . import vector_contextual as _vic
except ImportError:
    _vic = None

# Rough average bytes per token for English markdown — close enough for
# budgeting without invoking a tokenizer in-process.
BYTES_PER_TOKEN = 3.6
DEFAULT_BUDGET_TOKENS = 8000

EPISODIC_AGE_DAYS = 30
EPISODIC_MIN_SIBLINGS = 3
EPISODIC_SIM_THRESHOLD = 0.55

IDLE_THRESHOLD_SECONDS = 30

# READ_SEAM_TOP_N — What: default cap on cross-chain failure/diagnosis
#   associations surfaced by chainogram_retrieve when
#   include_failure_associations=True.
# Why: bounds output size and query work; overridable per-call or via env var
#   ASTHENOS_READ_SEAM_TOP_N so the operator can tune without code changes.
READ_SEAM_TOP_N_DEFAULT = 5
READ_SEAM_TOP_N_ENV = "ASTHENOS_READ_SEAM_TOP_N"


# ===========================================================================
# Temporal-recall scaffold (FEAT-2026-06-11 P1, proposal §2 + §8.6 + §16.2 Q5)
# ---------------------------------------------------------------------------
# What: the master flag + per-term weight readers + pure helpers (pool min-max
#   normalizer, relevance gate, compute-skip predicate) and the four 0.0-
#   returning hook seams (TC/N/K/D) that P2-P5 will fill in. The unified score
#   is score(c) = (S_c + 0.05·H_c + γ·TĈ_c) · (1 + λN·N̂_c + λK·K̂_c + λD·D̂_c).
# Why: P1 lands the *shape* with every new coefficient pinned to 0 and behind a
#   default-OFF master flag, so the flag-off path is byte-identical to today's
#   S_c + 0.05·H_c accumulation (§2.6 identity proof). No temporal module is
#   built yet — the term hooks return 0.0, and §16.2-Q5 compute-skip means a
#   weight < ε never even calls its hook. The whole block is inert until a
#   later operator-gated calibration flips ASTHENOS_TEMPORAL_WEIGHT and freezes
#   the weights.
# ===========================================================================

# Master deploy gate. Default OFF — mirrors semantic_recall.semantic_arm_enabled
#   / integrity's os.environ.get(..., "0") == "1" reader idiom (§8.6). Read each
#   call so a test/daemon/adapter that sets the env after import sees it.
TEMPORAL_WEIGHT_ENV = "ASTHENOS_TEMPORAL_WEIGHT"

# Per-term weight env names. γ scales the additive SITH cue; λN/λK/λD scale the
#   multiplicative need/STC/distinctiveness modulators. ALL DEFAULT TO 0.0, so
#   even with the master flag ON the formula collapses to the baseline until a
#   calibration freezes non-zero values (§8.6 per-term ablation).
TEMPORAL_GAMMA_ENV = "ASTHENOS_TEMPORAL_GAMMA"
TEMPORAL_LAMBDA_N_ENV = "ASTHENOS_TEMPORAL_LAMBDA_N"
TEMPORAL_LAMBDA_K_ENV = "ASTHENOS_TEMPORAL_LAMBDA_K"
TEMPORAL_LAMBDA_D_ENV = "ASTHENOS_TEMPORAL_LAMBDA_D"

# Uniform relevance gate θ = 0.2 (§2.5, FORMULA Q4): a hit injects temporal
#   signal only if its own semantic cosine clears θ. Not re-applied to S/H (the
#   identity baseline). Frozen constant, not an env knob in P1.
TEMPORAL_THETA = 0.2

# Compute-skip epsilon (§16.2 Q5): a term whose |weight| < ε is gated off at the
#   per-query COMPUTE level (its hook is never called), not merely multiplied by
#   zero — a term that earns no lift costs nothing. With all weights at the 0.0
#   default this skips every term, so the flag-off path runs zero temporal code.
TEMPORAL_WEIGHT_EPSILON = 1e-9


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


TC_COSINE_FLOOR_DEFAULT = 0.4
TC_COSINE_FLOOR_ENV = "ASTHENOS_TEMPORAL_TC_COSINE_FLOOR"


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
        from . import temporal_recall_sith as _sith
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
        from . import successor as _sr
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
        from . import successor as _sr
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
        from . import temporal_recall_stc as _stc
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
        from . import temporal_distinctiveness as _td
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
        from . import temporal_distinctiveness as _td
        return _td.dist_at(dist_vec, cname)
    except Exception:
        return 0.0


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _chains_dir(memory_dir: Path) -> Path:
    return memory_dir / "chains"


def _ctx_dir(memory_dir: Path) -> Path:
    return memory_dir / "context_extension"


def _frozen_prefix_path(memory_dir: Path) -> Path:
    return _ctx_dir(memory_dir) / "frozen_prefix.txt"


def _idle_state_path(memory_dir: Path) -> Path:
    return _ctx_dir(memory_dir) / "idle_state.json"


def _tok_estimate(text: str) -> int:
    return int(len(text) / BYTES_PER_TOKEN)


def _node_text(path: Path) -> tuple[int, str]:
    raw = path.read_text(encoding="utf-8")
    return _tok_estimate(raw), raw


def _read_fm(path: Path) -> tuple[dict, str]:
    fm_lines, body = _tq.read_node(path)
    fm: dict = {}
    for line in fm_lines:
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, body


# Helpers to abstract vector-module path/query differences (plain vs
# contextual). Both library-plane modules share the same _manifest_path /
# query(memory_dir, ...) shape.

def _vi_manifest(vi_module, memory_dir: Path) -> Path:
    return vi_module._manifest_path(memory_dir)


def _vi_embed(vi_module, memory_dir: Path) -> Path:
    return vi_module._embed_path(memory_dir)


def _vi_query(vi_module, memory_dir: Path, query: str, top_k: int) -> list[dict]:
    return vi_module.query(memory_dir, query, top_k=top_k)


# _ATOM_CHAIN_CACHE — What: per-(memory_dir, chain_name) cache of "is this an atom
#   (semantic-population) chain?". Why: the fx_-skip gate (FEAT-2026-06-10 Q4a) is
#   consulted per hit during chain selection; resolving the chain's first-member type
#   once and caching keeps the gate O(1) on repeat. Only populated when the semantic arm
#   is ON (the unflagged path never calls _is_atom_chain), so flag-off behavior and the
#   cache are fully decoupled.
_ATOM_CHAIN_CACHE: dict[tuple[str, str], bool] = {}


def _clear_atom_chain_cache() -> None:
    """Drop the atom-chain classification cache (tests / after a chain's type changes)."""
    _ATOM_CHAIN_CACHE.clear()


def _is_atom_chain(memory_dir: Path, chain_name: str) -> bool:
    """True iff *chain_name* belongs to the SEMANTIC (atom) population.

    What: an "fx_"-prefixed chain id is an atom mini-chain by construction (O(1) prefix
      check). As a fallback for atom chains NOT carrying that prefix, resolve the chain's
      FIRST member file's `type` and treat the chain as atomic when it is "semantic".
      Cached per (memory_dir, chain_name).
    Why: the populations meet in the composer (semantic_recall.recall), so the episodic
      chainogram must not select atom chains when the semantic arm is on (Q4a). The cheap
      id-prefix check covers the produced atom chains; the member-type fallback is the
      robust backstop. Only called when the flag is ON.
    """
    key = (str(memory_dir), chain_name)
    if key in _ATOM_CHAIN_CACHE:
        return _ATOM_CHAIN_CACHE[key]
    is_atom = False
    if chain_name.startswith("fx_"):
        is_atom = True
    else:
        try:
            chain = _ct.load_chain(_chains_dir(memory_dir), chain_name)
            members = chain.get("members") or []
            first = members[0] if members else None
            f = first.get("file") if isinstance(first, dict) else None
            if f:
                from . import semantic_recall as _sr
                if _sr._node_type(memory_dir, Path(f).name) == "semantic":
                    is_atom = True
        except Exception:
            is_atom = False
    _ATOM_CHAIN_CACHE[key] = is_atom
    return is_atom


# ---------------------------------------------------------------------------
# Read-seam: cross-chain failure/diagnosis association query
# ---------------------------------------------------------------------------


def _resolve_read_seam_top_n(top_n: int | None) -> int:
    """Resolve the effective top-N for failure associations.

    What: returns the caller's explicit value, else the env var, else the default.
    Why: three-tier override (call-site > env > constant) so N is configurable
         without code changes per the read-seam spec.
    """
    if top_n is not None:
        return max(0, top_n)
    env_val = os.environ.get(READ_SEAM_TOP_N_ENV)
    if env_val is not None:
        try:
            return max(0, int(env_val))
        except (ValueError, TypeError):
            pass
    return READ_SEAM_TOP_N_DEFAULT


def _is_failure_or_diagnosis_node(fm: dict) -> bool:
    """Return True if frontmatter marks a failure-outcome or bug-diagnosis node.

    What: checks two disjoint node types produced by the storm's write-side:
      (a) outcome nodes from opencode_drain — type=reference, chains includes
          both 'bounty_outcomes' and 'verified_outcomes', body outcome in
          ('failure', 'partial') signaled by target_state=frozen.
      (b) bug nodes from bug_records — type=bug, any status except 'wont-fix'
          (wont-fix = dismissed, not actionable failure experience).
    Why: these are the exact frontmatter fields the scout verified; matching on
         them avoids body parsing and stays robust to format drift.
    """
    node_type = (fm.get("type") or "").strip().lower()
    # Bug node path: type=bug, status not dismissed
    if node_type == "bug":
        status = (fm.get("status") or "").strip().lower()
        return status != "wont-fix"
    # Outcome node path: type=reference + frozen target_state (failure/partial)
    if node_type == "reference":
        target_state = (fm.get("target_state") or "").strip().lower()
        chains_raw = (fm.get("chains") or "").strip()
        has_bounty = "bounty_outcomes" in chains_raw
        has_verified = "verified_outcomes" in chains_raw
        return target_state == "frozen" and has_bounty and has_verified
    return False


def _query_failure_associations(
    memory_dir: Path,
    loaded_nodes: list[str],
    top_n: int,
    db_dir: str | None = None,
) -> list[dict]:
    """Query failure/diagnosis associations: direct matches + cross-chain neighbors.

    What: (1) identifies loaded_nodes that ARE failure/diagnosis nodes themselves
      (direct matches, weight=1.0), then (2) reads edges.db (read-only) for
      coactivation neighbors of loaded_nodes and filters those to failure/diagnosis
      nodes. Merges both sources, deduplicates by node name (highest weight wins),
      ranks by weight x recency, returns the top-N associations.
    Why: this is the read-seam — surfacing accumulated failure experience from the
      Hebbian web during diagnosis. Direct matches are the most query-relevant
      failures (they ARE the loaded context) and were previously excluded by the
      neighbor-only filter, causing the read-seam to miss a bounty's own prior
      failures even when they were in loaded_nodes.
    """
    if top_n <= 0 or not loaded_nodes:
        return []

    today = _dt.date.today()
    node_set = set(loaded_nodes)
    nodes_dir = _nodes_dir(memory_dir)

    # What: locate edges.db; fall back to web_store default.
    # Why: tests pass db_dir to use a temp store; production uses the default.
    if db_dir:
        db_path = os.path.join(db_dir, "edges.db")
    else:
        db_path = _ws._db_path(None)

    # What: collect all coactivation neighbors of loaded nodes from edges.db.
    # Why: UNION both src/dst directions because edges are stored in
    #   canonical (_order) form — a loaded node could be on either side.
    #   If edges.db is missing or unreadable, skip neighbor collection
    #   (direct matches may still contribute).
    neighbor_rows: list[tuple[str, float, str]] = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError:
            conn = None
        if conn is not None:
            try:
                for node_name in node_set:
                    for row in conn.execute(
                        "SELECT dst_node, weight, last_seen_at FROM edges "
                        "WHERE ref_kind=? AND src_node=?",
                        (_ws.COACTIVATION, node_name),
                    ).fetchall():
                        if row[0] not in node_set:
                            neighbor_rows.append(row)
                    for row in conn.execute(
                        "SELECT src_node, weight, last_seen_at FROM edges "
                        "WHERE ref_kind=? AND dst_node=?",
                        (_ws.COACTIVATION, node_name),
                    ).fetchall():
                        if row[0] not in node_set:
                            neighbor_rows.append(row)
            finally:
                conn.close()

    # What: collect loaded_nodes that ARE failure/diagnosis nodes (direct matches).
    # Why: the neighbor loop above EXCLUDES loaded_nodes by design (line 220/227),
    #   but a directly-retrieved node that IS a failure (e.g. the bounty's own prior
    #   attempts) is the most query-relevant failure association and must be
    #   surfaced. Weight = 1.0 (strongest, since they ARE query context).
    direct_matches: list[tuple[str, float, str]] = []  # (node_name, weight, last_seen)
    for node_name in node_set:
        p = nodes_dir / node_name
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        try:
            fm, _ = _read_fm(p)
        except Exception:
            continue
        if not _is_failure_or_diagnosis_node(fm):
            continue
        # What: derive last_seen from frontmatter last_access, fall back to today.
        # Why: direct matches have no edge row; last_access is the closest analog
        #   to the edge's last_seen_at for recency scoring.
        la = (fm.get("last_access") or "").strip()
        last_seen = la if la else today.isoformat()
        direct_matches.append((node_name, 1.0, last_seen))

    if not neighbor_rows and not direct_matches:
        return []

    # What: merge direct matches + neighbor rows, deduplicate by node name,
    #   keeping the highest weight per node.
    # Why: a node could appear as both a direct match (weight 1.0) and a
    #   cross-chain neighbor; dedup ensures one entry with the best weight.
    #   Direct matches at weight 1.0 will dominate unless a neighbor has
    #   higher weight (shouldn't happen, but the max keeps it correct).
    best_by_node: dict[str, tuple[float, str, str]] = {}
    #   value: (weight, last_seen, provenance)
    for node_name, weight, last_seen in direct_matches:
        best_by_node[node_name] = (weight, last_seen, "direct_match")
    for neighbor, weight, last_seen in neighbor_rows:
        prev = best_by_node.get(neighbor)
        if prev is None or weight > prev[0]:
            best_by_node[neighbor] = (weight, last_seen, "cross_chain")

    # What: compute effective score = weight x recency_decay, filter to failure nodes.
    # Why: recency-adjusted ranking per web_store's decay formula ensures stale
    #   failure associations sink below fresh ones.
    scored: list[tuple[float, str, float, str, str]] = []
    for neighbor, (weight, last_seen, provenance) in best_by_node.items():
        # Read frontmatter to check failure/diagnosis type
        p = nodes_dir / neighbor
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        try:
            fm, _ = _read_fm(p)
        except Exception:
            continue
        if not _is_failure_or_diagnosis_node(fm):
            continue

        # Compute recency-adjusted score
        days = _ws._days_since(last_seen, today)
        recency_decay = max(0.0, 1.0 - _ws.EDGE_DECAY_PER_DAY * days)
        effective_score = weight * recency_decay
        scored.append((effective_score, neighbor, weight, last_seen, provenance))

    # What: sort descending by effective score, take top N.
    scored.sort(key=lambda t: -t[0])
    scored = scored[:top_n]

    results: list[dict] = []
    for eff_score, neighbor, weight, last_seen, provenance in scored:
        p = nodes_dir / neighbor
        if not p.suffix:
            p = p.with_suffix(".md")
        fm, _ = _read_fm(p)
        node_type = (fm.get("type") or "").strip().lower()
        # What: classify the failure kind for the caller.
        # Why: callers need to know if this is a prior bounty failure or a bug
        #   diagnosis without re-reading frontmatter themselves.
        if node_type == "bug":
            failure_kind = "bug_diagnosis"
        else:
            failure_kind = "bounty_failure"
        results.append({
            "node": neighbor,
            "effective_score": round(eff_score, 4),
            "weight": round(weight, 4),
            "last_seen_at": last_seen,
            "failure_kind": failure_kind,
            "provenance": provenance,
            "name": fm.get("name", ""),
            # What: carry the failure REASON (from the node description) so callers
            #   can act on WHAT failed, not just that a failure occurred.
            # Why: recalling a failure without its reason is signal-free — the whole
            #   point of read-seam is to surface the specific prior mistake to avoid.
            "reason": (fm.get("description") or "").strip(),
        })
    return results


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


# ---------------------------------------------------------------------------
# Primitive A — Sparse engram retrieval
# ---------------------------------------------------------------------------


def chainogram_retrieve(memory_dir: Path, query: str,
                        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                        max_chains: int = 8,
                        include_singletons: bool = False,
                        _vi_module=None,
                        include_failure_associations: bool = False,
                        failure_top_n: int | None = None,
                        _web_db_dir: str | None = None) -> dict:
    """Sparse chain-level retrieval bounded by a token budget."""
    vi = _vi_module if _vi_module is not None else _vi
    manifest_path = _vi_manifest(vi, memory_dir)
    if not manifest_path.exists():
        return {"error": f"no vector index at {manifest_path}"}

    nodes_dir = _nodes_dir(memory_dir)
    chains_dir = _chains_dir(memory_dir)

    hits = _vi_query(vi, memory_dir, query, top_k=24)
    # FEAT-2026-06-10 Q4a — semantic-arm chain SELECTION skip. When the semantic arm is
    # on, atom mini-chains (fx_-prefixed ids, or chains whose first member resolves to a
    # type:semantic node) are EXCLUDED from candidate selection: the atom population is
    # served by the peer semantic arm and the two populations meet in the composer
    # (semantic_recall.recall), not inside this episodic arm. Resolved ONCE here, gated by
    # the flag so the unflagged path never even resolves a type — flag off => byte-
    # identical to today's selection. Lazy import dodges the context_extension<->
    # semantic_recall cycle.
    _semantic_arm_on = False
    try:
        from . import semantic_recall as _sr
        _semantic_arm_on = _sr.semantic_arm_enabled()
    except Exception:
        _semantic_arm_on = False
    chain_scores: dict[str, dict] = {}
    for h in hits:
        ca = _bio._addr_for_node(memory_dir, h["node"])
        if not ca:
            continue
        chain_name, addr = ca
        if _semantic_arm_on and _is_atom_chain(memory_dir, chain_name):
            continue
        info = chain_scores.setdefault(chain_name, {
            "score": 0.0, "best_node": h["node"], "best_score": 0.0,
            "addrs": set(),
        })
        info["score"] += float(h["score"])
        info["addrs"].add(addr)
        if h["score"] > info["best_score"]:
            info["best_score"] = float(h["score"])
            info["best_node"] = h["node"]

    # Hebbian density boost. After this loop info["score"] = S_c + 0.05·H_c — the
    # raw additive base cue (S_c summed at :434, H_c counted here). This is the
    # byte-identical baseline; the temporal block below ONLY runs flagged-on.
    for cname in chain_scores:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        edges = chain.get("edges") or []
        hebb = sum(1 for e in edges if e.get("label") == "hebbian")
        chain_scores[cname]["score"] += 0.05 * hebb

    # Temporal-recall envelope (FEAT-2026-06-11 P1, §2). Flagged-off this whole
    # block is skipped, so info["score"] stays the raw S_c + 0.05·H_c above and the
    # sort is byte-identical to today (§2.6). Flagged-on, each term whose weight
    # clears ε (§16.2-Q5 compute-skip) is accumulated, pool min-max normalized, and
    # folded into score(c) = (S + 0.05·H + γ·TĈ)·(1 + λN·N̂ + λK·K̂ + λD·D̂). In P1
    # every weight defaults to 0.0 and every hook returns 0.0, so even flagged-on
    # the score is unchanged — the scaffold is a provable no-op until calibration.
    if temporal_weight_enabled():
        _apply_temporal_envelope(memory_dir, chain_scores, hits)

    ordered = sorted(chain_scores.items(), key=lambda kv: -kv[1]["score"])
    ordered = ordered[:max_chains]

    loaded_nodes: list[dict] = []
    skipped: list[dict] = []
    spent = 0
    seen_files: set[str] = set()
    chosen_chains: list[str] = []

    n_singletons = 0
    if include_singletons:
        for h in sorted(hits, key=lambda x: -float(x["score"])):
            if _bio._addr_for_node(memory_dir, h["node"]):
                continue
            fname = h["node"]
            if fname in seen_files:
                continue
            seen_files.add(fname)
            p = nodes_dir / fname
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, _ = _node_text(p)
            entry = {"node": p.name, "tokens": tok,
                     "chain": "<singleton>", "addr": None,
                     "score": float(h["score"])}
            if spent + tok <= budget_tokens:
                loaded_nodes.append(entry)
                spent += tok
                n_singletons += 1
            else:
                skipped.append(entry)

    for cname, info in ordered:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        chosen_chains.append(cname)
        for m in chain.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f or f in seen_files:
                continue
            seen_files.add(f)
            p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, _ = _node_text(p)
            entry = {"node": p.name, "tokens": tok, "chain": cname,
                     "addr": m.get("addr"), "score": info["best_score"]}
            if spent + tok <= budget_tokens:
                loaded_nodes.append(entry)
                spent += tok
            else:
                skipped.append(entry)

    rationale = (f"top-{len(chosen_chains)} chains by "
                 "(relevance + Hebbian density), packed under budget")
    if include_singletons:
        rationale += f"; +{n_singletons} singleton hit(s)"

    out = {
        "loaded_chains": chosen_chains,
        "loaded_nodes": loaded_nodes,
        "skipped_nodes": skipped,
        "budget_tokens": budget_tokens,
        "spent_tokens": spent,
        "n_singletons": n_singletons,
        "rationale": rationale,
    }

    # What: optionally surface cross-chain failure/diagnosis associations from
    #   the coactivation web (read-seam).
    # Why: closes the read side of the failure-experience storm — during
    #   diagnosis, callers see prior failures that are Hebbian-associated with
    #   the loaded nodes, ranked by weight x recency. Additive-only key;
    #   existing callers ignore it via dict.get().
    if include_failure_associations:
        eff_n = _resolve_read_seam_top_n(failure_top_n)
        all_loaded_names = [n["node"] for n in loaded_nodes]
        assocs = _query_failure_associations(
            memory_dir, all_loaded_names, eff_n, db_dir=_web_db_dir,
        )
        out["failure_associations"] = assocs
        if assocs:
            rationale += f"; +{len(assocs)} failure association(s) from web"
            out["rationale"] = rationale

    return out


# ---------------------------------------------------------------------------
# Primitive A.3 — Entity-bridge augmented retrieval
# ---------------------------------------------------------------------------


def chainogram_retrieve_bridged(memory_dir: Path, query: str,
                                budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                                max_chains: int = 8,
                                max_bridge_nodes: int = 8,
                                include_singletons: bool = True,
                                bridge_reserve_frac: float = 0.25) -> dict:
    bridge_reserve = int(budget_tokens * bridge_reserve_frac)
    chain_budget = budget_tokens - bridge_reserve
    out = chainogram_retrieve(memory_dir, query, budget_tokens=chain_budget,
                              max_chains=max_chains,
                              include_singletons=include_singletons)
    if "error" in out:
        return out
    if _ei is None:
        out["bridge_nodes_added"] = 0
        out["rationale"] = (out.get("rationale", "") +
                            "; entity index unavailable")
        return out

    bridges = _ei.query_bridges(memory_dir, query,
                                max_bridge_nodes=max_bridge_nodes)
    if "error" in bridges:
        out["bridge_nodes_added"] = 0
        out["rationale"] = out.get("rationale", "") + "; " + bridges["error"]
        return out

    nodes_dir = _nodes_dir(memory_dir)
    loaded_files = {n["node"] for n in out.get("loaded_nodes") or []}
    spent = int(out.get("spent_tokens") or 0)
    n_added = 0
    for b in bridges.get("bridge_nodes") or []:
        fname = b["node"]
        if fname in loaded_files:
            continue
        p = nodes_dir / fname
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        tok, _ = _node_text(p)
        if spent + tok > budget_tokens:
            out.setdefault("skipped_nodes", []).append({
                "node": p.name, "tokens": tok, "chain": "<bridge>",
                "addr": None, "score": float(b["score"]),
            })
            continue
        out.setdefault("loaded_nodes", []).append({
            "node": p.name, "tokens": tok, "chain": "<bridge>",
            "addr": None, "score": float(b["score"]),
            "matched_entities": b["entities"],
        })
        loaded_files.add(p.name)
        spent += tok
        n_added += 1

    out["spent_tokens"] = spent
    out["bridge_nodes_added"] = n_added
    out["matched_entities"] = bridges.get("matched_entities") or []
    out["rationale"] = (out.get("rationale", "") +
                        f"; +{n_added} entity-bridge nodes "
                        f"({len(out['matched_entities'])} entities matched)")
    return out


# ---------------------------------------------------------------------------
# Primitive A.2 — Hybrid union retrieval (no rerank)
# ---------------------------------------------------------------------------


def chainogram_retrieve_hybrid(memory_dir: Path, query: str,
                               budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                               max_chains: int = 12,
                               extra_topk: int = 12) -> dict:
    if not _vi_manifest(_vi, memory_dir).exists():
        return {"error": "no vector index — run memory_vector_index.py build"}

    nodes_dir = _nodes_dir(memory_dir)
    chains_dir = _chains_dir(memory_dir)

    hits = _vi_query(_vi, memory_dir, query, top_k=max(40, extra_topk + 24))
    chain_scores: dict[str, float] = {}
    for h in hits:
        ca = _bio._addr_for_node(memory_dir, h["node"])
        if not ca:
            continue
        chain_name, _ = ca
        chain_scores[chain_name] = chain_scores.get(chain_name, 0.0) \
            + float(h["score"])

    ordered_chains = sorted(chain_scores.items(),
                            key=lambda kv: -kv[1])[:max_chains]

    candidates: dict[str, dict] = {}
    top10_files = {h["node"] for h in hits[:10]}

    n_singletons = 0
    for h in hits:
        if _bio._addr_for_node(memory_dir, h["node"]):
            continue
        fname = h["node"]
        p = nodes_dir / fname
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists() or fname in candidates:
            continue
        tok, _ = _node_text(p)
        candidates[fname] = {"node": p.name, "tokens": tok,
                             "chain": "<singleton>", "addr": None,
                             "vec_score": float(h["score"]),
                             "in_top10": fname in top10_files}
        n_singletons += 1

    chosen_chain_names: list[str] = []
    for cname, cscore in ordered_chains:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        chosen_chain_names.append(cname)
        for m in chain.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f or f in candidates:
                continue
            p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, _ = _node_text(p)
            vec_score = next((float(h["score"]) for h in hits
                              if h["node"] == p.name), 0.0)
            candidates[p.name] = {"node": p.name, "tokens": tok,
                                  "chain": cname, "addr": m.get("addr"),
                                  "vec_score": vec_score,
                                  "chain_score": cscore,
                                  "in_top10": p.name in top10_files}

    n_extra = 0
    for h in hits[:extra_topk]:
        fname = h["node"]
        if fname in candidates:
            continue
        p = nodes_dir / fname
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        tok, _ = _node_text(p)
        candidates[fname] = {"node": p.name, "tokens": tok,
                             "chain": "<topk>", "addr": None,
                             "vec_score": float(h["score"]),
                             "in_top10": fname in top10_files}
        n_extra += 1

    max_chain_score = max((c.get("chain_score", 0.0)
                           for c in candidates.values()), default=1.0) or 1.0
    for c in candidates.values():
        chain_norm = c.get("chain_score", 0.0) / max_chain_score
        c["hybrid_score"] = (0.55 * c["vec_score"]
                             + 0.20 * chain_norm
                             + 0.15 * (1.0 if c["in_top10"] else 0.0))

    ordered = sorted(candidates.values(), key=lambda c: -c["hybrid_score"])

    loaded_nodes: list[dict] = []
    skipped: list[dict] = []
    spent = 0
    for c in ordered:
        entry = {"node": c["node"], "tokens": c["tokens"], "chain": c["chain"],
                 "addr": c["addr"], "score": c["hybrid_score"],
                 "vec_score": c["vec_score"]}
        if spent + c["tokens"] <= budget_tokens:
            loaded_nodes.append(entry)
            spent += c["tokens"]
        else:
            skipped.append(entry)

    return {
        "loaded_chains": chosen_chain_names,
        "loaded_nodes": loaded_nodes,
        "skipped_nodes": skipped,
        "budget_tokens": budget_tokens,
        "spent_tokens": spent,
        "n_singletons": n_singletons,
        "n_extra_topk": n_extra,
        "n_candidates": len(candidates),
        "rationale": (f"hybrid union: {len(chosen_chain_names)} chains + "
                      f"{n_singletons} singletons + {n_extra} extra top-k, "
                      "ranked by 0.55·vec + 0.20·chain + 0.15·top10"),
    }


# ---------------------------------------------------------------------------
# Primitive A.1 — Cross-encoder reranked engram retrieval
# ---------------------------------------------------------------------------

_RERANKER = None
_RERANKER_NAME = "BAAI/bge-reranker-base"


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        import os
        from sentence_transformers import CrossEncoder
        device = os.environ.get("SAM_RERANKER_DEVICE", "cpu")
        _RERANKER = CrossEncoder(_RERANKER_NAME, max_length=512, device=device)
    return _RERANKER


def chainogram_retrieve_reranked(memory_dir: Path, query: str,
                                 budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                                 max_chains: int = 12,
                                 candidate_pool: int = 40,
                                 include_singletons: bool = True) -> dict:
    if not _vi_manifest(_vi, memory_dir).exists():
        return {"error": "no vector index — run memory_vector_index.py build"}

    nodes_dir = _nodes_dir(memory_dir)
    chains_dir = _chains_dir(memory_dir)

    hits = _vi_query(_vi, memory_dir, query, top_k=candidate_pool)
    chain_scores: dict[str, dict] = {}
    for h in hits:
        ca = _bio._addr_for_node(memory_dir, h["node"])
        if not ca:
            continue
        chain_name, addr = ca
        info = chain_scores.setdefault(chain_name, {
            "score": 0.0, "best_score": 0.0, "addrs": set(),
        })
        info["score"] += float(h["score"])
        info["addrs"].add(addr)
        if h["score"] > info["best_score"]:
            info["best_score"] = float(h["score"])

    for cname in chain_scores:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        edges = chain.get("edges") or []
        chain_scores[cname]["score"] += 0.05 * sum(
            1 for e in edges if e.get("label") == "hebbian"
        )

    ordered_chains = sorted(chain_scores.items(),
                            key=lambda kv: -kv[1]["score"])[:max_chains]

    candidates: list[dict] = []
    seen_files: set[str] = set()
    n_singletons = 0
    if include_singletons:
        for h in sorted(hits, key=lambda x: -float(x["score"])):
            if _bio._addr_for_node(memory_dir, h["node"]):
                continue
            fname = h["node"]
            if fname in seen_files:
                continue
            seen_files.add(fname)
            p = nodes_dir / fname
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, body = _node_text(p)
            candidates.append({"node": p.name, "tokens": tok, "body": body,
                               "chain": "<singleton>", "addr": None,
                               "vec_score": float(h["score"])})
            n_singletons += 1

    chosen_chain_names: list[str] = []
    for cname, info in ordered_chains:
        try:
            chain = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        chosen_chain_names.append(cname)
        for m in chain.get("members") or []:
            f = m.get("file") if isinstance(m, dict) else None
            if not f or f in seen_files:
                continue
            seen_files.add(f)
            p = memory_dir / f if f.startswith("nodes/") else nodes_dir / f
            if not p.suffix:
                p = p.with_suffix(".md")
            if not p.exists():
                continue
            tok, body = _node_text(p)
            candidates.append({"node": p.name, "tokens": tok, "body": body,
                               "chain": cname, "addr": m.get("addr"),
                               "vec_score": float(info["best_score"])})

    if not candidates:
        return {
            "loaded_chains": [], "loaded_nodes": [], "skipped_nodes": [],
            "budget_tokens": budget_tokens, "spent_tokens": 0,
            "n_singletons": 0, "n_candidates": 0,
            "rationale": "no candidates",
        }

    reranker = _get_reranker()
    pairs = [(query, c["body"][:2000]) for c in candidates]
    ce_scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, ce_scores):
        c["ce_score"] = float(s)

    candidates.sort(key=lambda c: -c["ce_score"])

    loaded_nodes: list[dict] = []
    skipped: list[dict] = []
    spent = 0
    for c in candidates:
        entry = {"node": c["node"], "tokens": c["tokens"], "chain": c["chain"],
                 "addr": c["addr"], "score": c["ce_score"],
                 "vec_score": c["vec_score"]}
        if spent + c["tokens"] <= budget_tokens:
            loaded_nodes.append(entry)
            spent += c["tokens"]
        else:
            skipped.append(entry)

    return {
        "loaded_chains": chosen_chain_names,
        "loaded_nodes": loaded_nodes,
        "skipped_nodes": skipped,
        "budget_tokens": budget_tokens,
        "spent_tokens": spent,
        "n_singletons": n_singletons,
        "n_candidates": len(candidates),
        "reranker": _RERANKER_NAME,
        "rationale": (f"cross-encoder reranked {len(candidates)} candidates "
                      f"({n_singletons} singletons + chain members from "
                      f"{len(chosen_chain_names)} chains)"),
    }


# ---------------------------------------------------------------------------
# Primitive A.4 — Contextual-seed engram retrieval
# ---------------------------------------------------------------------------


def chainogram_retrieve_contextual(memory_dir: Path, query: str,
                                   budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                                   max_chains: int = 8,
                                   include_singletons: bool = True) -> dict:
    if _vic is None:
        return {"error": "memory_vector_index_contextual unavailable"}
    if not _vi_manifest(_vic, memory_dir).exists():
        return {"error": (f"no contextual index — run "
                          f"memory_vector_index_contextual.py build")}
    out = chainogram_retrieve(memory_dir, query, budget_tokens=budget_tokens,
                              max_chains=max_chains,
                              include_singletons=include_singletons,
                              _vi_module=_vic)
    if "rationale" in out:
        out["rationale"] = "[contextual-seed] " + out["rationale"]
    out["seed_index"] = "contextual"
    return out


# ---------------------------------------------------------------------------
# Primitive B — Stable-prefix anchoring (FROZEN tier)
# ---------------------------------------------------------------------------


def frozen_prefix_block(memory_dir: Path, write: bool = True) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    rows: list[tuple[str, str]] = []
    for p in sorted(nodes_dir.glob("*.md")):
        fm, body = _read_fm(p)
        tier = (fm.get("tier") or "").lower()
        if tier != "frozen":
            continue
        rows.append((p.name, f"# {fm.get('name', p.stem)}\n{body.strip()}\n"))

    if not rows:
        block = "<!-- no FROZEN nodes -->\n"
    else:
        sep = "\n\n---\n\n"
        block = (
            "<!-- frozen-prefix v1 :: do not edit, regenerated by "
            "memory_context_extension.py :: cache-stable -->\n"
            + sep.join(body for _, body in rows)
        )

    h = hashlib.sha256(block.encode("utf-8")).hexdigest()[:16]

    fpp = _frozen_prefix_path(memory_dir)
    if write:
        _ctx_dir(memory_dir).mkdir(parents=True, exist_ok=True)
        fpp.write_text(block, encoding="utf-8")

    return {
        "frozen_count": len(rows),
        "block_bytes": len(block.encode("utf-8")),
        "estimated_tokens": _tok_estimate(block),
        "sha256_16": h,
        "path": str(fpp) if write else None,
        "block": block,
    }


# ---------------------------------------------------------------------------
# Primitive C — Budget-aware tier flow
# ---------------------------------------------------------------------------


def tier_flow_for_budget(memory_dir: Path, query: str,
                         budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                         dry_run: bool = True) -> dict:
    if not _vi_manifest(_vi, memory_dir).exists():
        return {"error": "no vector index"}
    nodes_dir = _nodes_dir(memory_dir)
    today = _dt.date.today()
    embeddings = np.load(_vi_embed(_vi, memory_dir))
    manifest = json.loads(_vi_manifest(_vi, memory_dir).read_text(encoding="utf-8"))
    entries = manifest.get("entries", {})

    qv = _vi._embed_batch([query])[0]
    sims = embeddings @ qv
    weights_path = memory_dir / "biomimetic" / "edge_weights.json"
    edge_w = {}
    if weights_path.exists():
        try:
            edge_w = json.loads(weights_path.read_text(encoding="utf-8"))
        except Exception:
            edge_w = {}
    hebb_count: dict[str, int] = {}
    for k in edge_w:
        a, b = k.split("::", 1)
        hebb_count[a] = hebb_count.get(a, 0) + 1
        hebb_count[b] = hebb_count.get(b, 0) + 1

    rows: list[dict] = []
    for rel, e in entries.items():
        p = nodes_dir / rel
        if not p.exists():
            continue
        fm, _ = _read_fm(p)
        tok, _ = _node_text(p)
        la = _tq.parse_date(fm.get("last_access"))
        days_ago = (today - la).days if la else 365
        recency = max(0.0, 1.0 - min(days_ago, 365) / 365.0)
        try:
            ac = int(fm.get("access_count") or "0")
        except Exception:
            ac = 0
        ac_norm = min(1.0, np.log1p(ac) / 5.0)
        hd = min(1.0, hebb_count.get(rel, 0) / 8.0)
        rel_score = float(sims[e["row"]]) if e.get("row") is not None else 0.0
        score = 0.55 * max(0.0, rel_score) + 0.20 * recency + \
                0.15 * hd + 0.10 * ac_norm
        rows.append({"node": rel, "tokens": tok, "score": round(score, 4),
                     "current_tier": fm.get("tier", "warm")})

    rows.sort(key=lambda r: -r["score"])

    plan: list[dict] = []
    spent = 0
    for r in rows:
        new_tier = "hot" if spent + r["tokens"] <= budget_tokens else "warm"
        if new_tier != r["current_tier"]:
            plan.append({**r, "new_tier": new_tier})
        if new_tier == "hot":
            spent += r["tokens"]

    if not dry_run:
        for change in plan:
            p = nodes_dir / change["node"]
            fm_lines, body = _tq.read_node(p)
            fm_lines = _tq.fm_set(fm_lines, "tier", change["new_tier"])
            _tq.write_node(p, fm_lines, body)

    return {"budget_tokens": budget_tokens, "spent_tokens": spent,
            "changes": plan, "dry_run": dry_run, "ranked_count": len(rows)}


# ---------------------------------------------------------------------------
# Primitive D — Episodic→semantic phase transition
# ---------------------------------------------------------------------------


def episodic_to_semantic_candidates(memory_dir: Path,
                                    chain: str | None = None) -> dict:
    if not _vi_manifest(_vi, memory_dir).exists():
        return {"error": "no vector index"}

    nodes_dir = _nodes_dir(memory_dir)
    chains_dir = _chains_dir(memory_dir)
    embeddings = np.load(_vi_embed(_vi, memory_dir))
    manifest = json.loads(_vi_manifest(_vi, memory_dir).read_text(encoding="utf-8"))
    entries = manifest.get("entries", {})

    today = _dt.date.today()
    chains_to_scan = [chain] if chain else _ct.list_chains(chains_dir)

    proposals: list[dict] = []
    for cname in chains_to_scan:
        try:
            cd = _ct.load_chain(chains_dir, cname)
        except (SystemExit, FileNotFoundError):
            continue
        members = cd.get("members") or []
        rows = []
        for m in members:
            f = m.get("file") if isinstance(m, dict) else None
            if not f:
                continue
            stem = Path(f).name
            if stem not in entries:
                continue
            p = nodes_dir / stem
            if not p.exists():
                continue
            fm, _ = _read_fm(p)
            vf = _tq.parse_date(fm.get("valid_from"))
            if not vf or (today - vf).days < EPISODIC_AGE_DAYS:
                continue
            row_idx = entries[stem]["row"]
            rows.append((stem, vf, row_idx, m.get("addr")))
        if len(rows) < EPISODIC_MIN_SIBLINGS:
            continue
        idxs = [r[2] for r in rows]
        emb = embeddings[idxs]
        sims = emb @ emb.T
        n = len(rows)
        adj = [[i for i in range(n) if i != j and sims[i, j] >= EPISODIC_SIM_THRESHOLD]
               for j in range(n)]
        seen: set[int] = set()
        for start in range(n):
            if start in seen:
                continue
            stack = [start]
            comp: list[int] = []
            while stack:
                k = stack.pop()
                if k in seen:
                    continue
                seen.add(k)
                comp.append(k)
                for nb in adj[k]:
                    if nb not in seen:
                        stack.append(nb)
            if len(comp) < EPISODIC_MIN_SIBLINGS:
                continue
            avg_sim = float(np.mean([sims[i, j] for i in comp for j in comp if i != j]))
            proposals.append({
                "chain": cname,
                "nodes": [rows[i][0] for i in comp],
                "addrs": [rows[i][3] for i in comp],
                "earliest": min(rows[i][1] for i in comp).isoformat(),
                "avg_similarity": round(avg_sim, 3),
                "rationale": (f"{len(comp)} episodic nodes >="
                              f"{EPISODIC_AGE_DAYS}d old, "
                              f"avg cosine {avg_sim:.2f} ≥ {EPISODIC_SIM_THRESHOLD}"),
            })
    return {"proposals": proposals, "count": len(proposals)}


# ---------------------------------------------------------------------------
# Primitive E — Idle DMN replay tick
# ---------------------------------------------------------------------------


def _record_replay_coactivations(memory_dir: Path, replay_res: dict,
                                 replay_il: dict) -> int:
    """Log replay-discovered PAIRS as source='replay' co-activations (FEAT-2026-06-05 D1).

    replay_sweep proposals carry from_node/to_node; the interleaved variant carries
    hot_node/cold_node. We bind each PAIR (not the whole recents sample, which was not
    genuinely co-retrieved) and dedup within the pulse. The fractional, decay-transparent
    treatment + genuine-count promotion gate (bio.py) keep this from running away: replay
    accelerates a genuinely-recent pair toward the bar but cannot promote or immortalize a
    pair that genuine recall never touches.
    """
    pairs: set = set()
    for p in (replay_res or {}).get("proposals", []) or []:
        a, b = p.get("from_node"), p.get("to_node")
        if a and b and a != b:
            pairs.add(tuple(sorted((a, b))))
    for p in (replay_il or {}).get("proposals", []) or []:
        a, b = p.get("hot_node"), p.get("cold_node")
        if a and b and a != b:
            pairs.add(tuple(sorted((a, b))))
    for a, b in pairs:
        try:
            _bio.hebbian_record(memory_dir, [a, b], query="replay", source="replay")
        except Exception:
            pass
    return len(pairs)


def _replay_pairs(replay_res: dict, replay_il: dict) -> set:
    """Collect the (a, b) co-activation pairs a replay tick discovered (P6 §5.5).

    What: the same pairs _record_replay_coactivations binds — replay_sweep proposals carry
      from_node/to_node, the interleaved variant hot_node/cold_node — but returned as raw,
      UN-sorted (a, b) tuples so the directed pass can read episode_seq order on each.
    Why: the directed-SR producer (§5.5) needs the in-window co-activation surface, which is
      exactly the offline-replay host's discovered pairs. We do NOT re-sort here (unlike the
      undirected hebbian path) because direction is decided by episode_seq, not lexical order.
    """
    pairs: set = set()
    for p in (replay_res or {}).get("proposals", []) or []:
        a, b = p.get("from_node"), p.get("to_node")
        if a and b and a != b:
            pairs.add((a, b))
    for p in (replay_il or {}).get("proposals", []) or []:
        a, b = p.get("hot_node"), p.get("cold_node")
        if a and b and a != b:
            pairs.add((a, b))
    return pairs


def _node_episode_seq(memory_dir: Path, node: str) -> int | None:
    """Read a node's episode_seq from frontmatter, or None if absent/unreadable (P6 §5.5).

    What: pull the corpus-global monotone episode_seq (§3.3) for one node. A legacy node
      minted before the substrate landed has no episode_seq → None; an unreadable/missing
      file → None. Mirrors temporal_recall_stc._node_fields' fail-soft read.
    Why: §5.5 — directed-SR direction is decided by seq(A) < seq(B). A pair where EITHER
      endpoint lacks episode_seq has no defined order, so the producer skips it (and the
      consumer falls back to the symmetric phase-1 kernel for that pair). No migration.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = _nodes_dir(memory_dir) / fname
    if not p.exists():
        return None
    try:
        from . import frontmatter as _fm
        fm, _order, _body = _fm.read_node(p)   # 3-tuple (dict, order, body)
    except Exception:
        return None
    raw = fm.get("episode_seq")
    if raw is None:
        return None
    try:
        # episode_seq is a dense integer counter; tolerate a float-stringified value.
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _record_directed_transitions(memory_dir: Path, replay_res: dict,
                                 replay_il: dict) -> dict:
    """Increment biomimetic/episode_transitions.json for directed co-activation pairs (P6).

    What: for each in-window co-activation pair (a, b) the replay tick discovered, read both
      endpoints' episode_seq (§3.3). When seq(A) < seq(B) the directed edge runs A->B (the
      earlier-encoded node precedes the later one); increment T_dir["A->B"] by 1 under
      locked_update_json (flock + atomic os.replace — the EXISTING primitive, no new lock).
      The matrix is INCREMENTED, never rebuilt: each pass adds the in-window pairs it swept
      to the running counts. A pair where either endpoint lacks episode_seq has no defined
      order and is skipped (its consumer falls back to the symmetric phase-1 kernel, §5.5).
      Equal seqs (should not occur for distinct nodes — the counter is dense) are also
      skipped (strict <). Keys carry the production ".md" filename form, matching
      edge_weights.json endpoints, so successor.py reads both stores with one node-key form.
    Why: §5.5 — this is the NET-NEW directed accumulation layered onto the reused offline-
      replay host (idle_replay_tick, REM-gated). It is the producer half of the strict
      producer/consumer split: this writes episode_transitions.json; successor.py reads it
      query-locally to build the forward SR M_fwd. Burst-invariant: episode_seq is the
      substrate's dense monotone unit, so direction is well-defined regardless of write
      density (§16). Fail-soft: any error leaves the counts as-is (never breaks the tick).
    """
    pairs = _replay_pairs(replay_res, replay_il)
    if not pairs:
        return {"pairs": 0, "directed": 0, "skipped_no_seq": 0}

    # Resolve each endpoint's episode_seq once (cache: a node may appear in many pairs).
    seq_cache: dict[str, int | None] = {}

    def _seq(node: str) -> int | None:
        if node not in seq_cache:
            seq_cache[node] = _node_episode_seq(memory_dir, node)
        return seq_cache[node]

    directed: list[tuple[str, str]] = []
    skipped_no_seq = 0
    for a, b in pairs:
        sa, sb = _seq(a), _seq(b)
        if sa is None or sb is None:
            skipped_no_seq += 1          # legacy pair: no order → symmetric fallback (§5.5)
            continue
        if sa < sb:
            src, dst = a, b
        elif sb < sa:
            src, dst = b, a
        else:
            continue                     # equal seq (dense counter ⇒ unreachable) → skip
        src = src if src.endswith(".md") else f"{src}.md"
        dst = dst if dst.endswith(".md") else f"{dst}.md"
        directed.append((src, dst))

    if not directed:
        return {"pairs": len(pairs), "directed": 0, "skipped_no_seq": skipped_no_seq}

    try:
        from .atomic_state import locked_update_json
        tpath = _bio._bio_paths(memory_dir)["episode_transitions"]
        tpath.parent.mkdir(parents=True, exist_ok=True)
        with locked_update_json(tpath, default={}) as st:
            for src, dst in directed:
                key = f"{src}->{dst}"            # DIRECTED key (not the undirected sorted ::)
                try:
                    st[key] = int(st.get(key, 0)) + 1
                except (TypeError, ValueError):
                    st[key] = 1                  # heal a corrupt count forward
    except Exception as _e:
        return {"pairs": len(pairs), "directed": 0,
                "skipped_no_seq": skipped_no_seq, "error": str(_e)}

    return {"pairs": len(pairs), "directed": len(directed),
            "skipped_no_seq": skipped_no_seq}


def idle_replay_tick(memory_dir: Path, force: bool = False) -> dict:
    # FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P2): the HEAVY
    # offline body of this tick (replay_sweep + replay_sweep_interleaved +
    # replay-coactivation reseed + hebbian_consolidate — the single biggest
    # offline-on-idle op) now runs ONLY inside REM. Outside REM the gate refuses
    # the heavy work with a LOGGED no-op (never a silent drop — risk-5), but the
    # CHEAP frozen-prefix MEMORY.md refresh stays on the waking path (the survey
    # flags it LIGHT/load-bearing; gating it would silently stop prefix refresh).
    # `force=True` (manual mcp_server call) bypasses the gate for deliberate use.
    _ctx_dir(memory_dir).mkdir(parents=True, exist_ok=True)
    if not force:
        try:
            from samia.runtime import rem_cycle as _rem
            if not _rem.gate_offline_op(Path(memory_dir), "idle_replay_tick"):
                # Not in REM: do only the light prefix refresh, refuse the heavy body.
                return {"fired": False, "refused": "not_in_rem",
                        "frozen_prefix": frozen_prefix_block(memory_dir, write=True)}
        except Exception:
            # rem_cycle unavailable (e.g. partial install): fail-open to legacy
            # behavior rather than silently disabling replay entirely.
            pass
    isp = _idle_state_path(memory_dir)
    state = {}
    if isp.exists():
        try:
            state = json.loads(isp.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    last_tick = state.get("last_tick_unix", 0)
    now = time.time()
    elapsed = now - last_tick
    if not force and elapsed < IDLE_THRESHOLD_SECONDS:
        return {"fired": False, "elapsed_seconds": int(elapsed),
                "threshold": IDLE_THRESHOLD_SECONDS}

    replay_res = _bio.replay_sweep(memory_dir, sample=15, threshold=0.55)
    replay_il = _bio.replay_sweep_interleaved(
        memory_dir, sample=15, cold_per_hot=3, threshold=0.40)
    # D1: feed replay-discovered PAIRS into the co-activation log as source='replay'
    # BEFORE consolidation drains it, so replay drives Hebbian growth of genuinely-recent
    # pairs (fractional + decay-transparent + genuine-count-gated => no runaway).
    replay_coact = _record_replay_coactivations(memory_dir, replay_res, replay_il)
    # FEAT-2026-06-11 temporal-recall P6 — directed-SR counting pass (§5.5). On the SAME
    # in-window co-activation pairs the replay tick just discovered, increment the directed
    # transition matrix T_dir (biomimetic/episode_transitions.json) for each pair ordered by
    # episode_seq: seq(A)<seq(B) ⇒ count A->B. This is the PRODUCER half of the strict
    # producer/consumer split — successor.py reads the file query-locally to build the
    # forward SR. INERT BY DEFAULT: gated behind the master temporal flag, so a corpus with
    # the feature off never grows the file. Pairs whose endpoints lack episode_seq (legacy)
    # are skipped → the consumer falls back to the symmetric phase-1 kernel for them (no
    # migration). Fail-soft: any error is swallowed so consolidation is never broken.
    if temporal_weight_enabled():
        try:
            directed_transitions = _record_directed_transitions(
                memory_dir, replay_res, replay_il)
        except Exception as _e:
            directed_transitions = {"error": str(_e)}
    else:
        directed_transitions = {"skipped": "temporal_weight_off"}
    # FEAT-2026-06-07 Tier-1 P5 — engram feed-forward (genuine-once). Replay the
    # CAPTURED engram held copies (the real recent-episode buffer, fixing
    # raw_pairs:0) into the co-activation log, ALSO before consolidation drains it.
    # GENUINE-ONCE: each engram-derived pair's FIRST replay is genuine (+count_genuine),
    # re-replays fractional then age — so one captured trace cannot be farmed into an
    # attractor by repeated replay (bio.replay_engram_traces). REM-gated (this whole
    # tick refuses outside REM) + inert by default. Fail-soft: an error never breaks
    # the consolidation that follows.
    try:
        engram_replay = _bio.replay_engram_traces(memory_dir, sample=15,
                                                  threshold=0.55)
    except Exception as _e:
        engram_replay = {"error": str(_e)}
    out = {
        "fired": True,
        "elapsed_seconds": int(elapsed),
        "replay": replay_res,
        "replay_interleaved": replay_il,
        "replay_coactivations": replay_coact,
        "directed_transitions": directed_transitions,
        "engram_replay": engram_replay,
        "hebbian": _bio.hebbian_consolidate(memory_dir),
        "frozen_prefix": frozen_prefix_block(memory_dir, write=True),
    }
    state["last_tick_unix"] = now
    state["last_tick_iso"] = _dt.datetime.now().isoformat(timespec="seconds")
    state["last_report"] = {k: out[k] for k in ("elapsed_seconds",)}
    isp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# SM-2 spaced repetition
# ---------------------------------------------------------------------------


def _read_full_fm(p: Path) -> tuple[list[str], str]:
    return _tq.read_node(p)


def sm2_review_update(memory_dir: Path, node_name: str,
                      recalled: bool = True, quality: int = 4) -> dict:
    nodes_dir = _nodes_dir(memory_dir)
    p = nodes_dir / node_name
    if not p.suffix:
        p = p.with_suffix(".md")
    if not p.exists():
        return {"error": f"node not found: {p.name}"}
    fm_lines, body = _read_full_fm(p)
    today = _dt.date.today()

    try:
        ef = float(_tq.fm_get(fm_lines, "easiness_factor") or "2.5")
    except Exception:
        ef = 2.5
    try:
        interval = int(_tq.fm_get(fm_lines, "review_interval_days") or "1")
    except Exception:
        interval = 1
    try:
        rc = int(_tq.fm_get(fm_lines, "review_count") or "0")
    except Exception:
        rc = 0

    if not recalled or quality < 3:
        rc = 0
        interval = 1
    else:
        rc += 1
        if rc == 1:
            interval = 1
        elif rc == 2:
            interval = 6
        else:
            interval = max(1, int(round(interval * ef)))
        ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        ef = max(1.3, ef)

    next_review = (today + _dt.timedelta(days=interval)).isoformat()
    fm_lines = _tq.fm_set(fm_lines, "easiness_factor", f"{ef:.3f}")
    fm_lines = _tq.fm_set(fm_lines, "review_interval_days", str(interval))
    fm_lines = _tq.fm_set(fm_lines, "review_count", str(rc))
    fm_lines = _tq.fm_set(fm_lines, "next_review", next_review)
    _tq.write_node(p, fm_lines, body)
    return {"node": p.name, "easiness_factor": round(ef, 3),
            "review_interval_days": interval, "review_count": rc,
            "next_review": next_review}


def sm2_due_for_review(memory_dir: Path,
                       today: _dt.date | None = None) -> list[dict]:
    nodes_dir = _nodes_dir(memory_dir)
    today = today or _dt.date.today()
    out: list[dict] = []
    for p in nodes_dir.glob("*.md"):
        fm_lines, _ = _tq.read_node(p)
        nr = _tq.parse_date(_tq.fm_get(fm_lines, "next_review"))
        if not nr or nr > today:
            continue
        out.append({"node": p.name,
                    "next_review": nr.isoformat(),
                    "review_count": _tq.fm_get(fm_lines, "review_count") or "0"})
    return sorted(out, key=lambda r: r["next_review"])


# SM-2 frontmatter seed defaults. These mirror sm2_review_update's own
# fallbacks (ef 2.5 / interval 1 / count 0) so a freshly-seeded node behaves
# identically to one that had implicit defaults. Wozniak canonical constants
# (the rc==1/rc==2/ef-update math) live in sm2_review_update and are NOT
# duplicated here.
SM2_SEED_EASINESS = 2.5
SM2_SEED_INTERVAL_DAYS = 1
SM2_SEED_REVIEW_COUNT = 0
# Per-tick seed cap. The corpus is ~2.9k nodes; seeding all at once would
# rewrite every file in a single tick (mass churn + watcher storm). Cap the
# seed work so the corpus is brought under SM-2 over ~a few daily ticks.
SM2_SWEEP_SEED_CAP = 200
# Per-tick review cap. Once seeded, due nodes accrue gradually; bound the
# review work per tick for the same churn reason.
SM2_SWEEP_REVIEW_CAP = 200


def _sm2_quality_from_usage(fm_lines: list[str]) -> tuple[bool, int]:
    """Derive an SM-2 (recalled, quality) pair from an existing usage signal.

    Why: the spec forbids a hardcoded quality=4. The tier-decay subsystem
    already maintains a per-node ``relevance`` in [0,1] (samia.core.tier) as
    the canonical hotness signal — hot nodes are the ones recently recalled,
    cold/frozen ones have decayed from disuse. We reuse that rather than
    invent a parallel metric. Map relevance onto SM-2 quality via the same
    tier thresholds tier.tier_for uses (0.75 hot / 0.50 warm / 0.25 cold):

        relevance >= 0.75 (hot)   -> quality 5  (perfect recall)
        relevance >= 0.50 (warm)  -> quality 4  (good recall)
        relevance >= 0.25 (cold)  -> quality 3  (passing recall)
        relevance <  0.25 (frozen)-> quality 1, recalled=False (a lapse)

    Falls back to the categorical ``tier`` label when ``relevance`` is absent
    or malformed (the corpus carries one or the other on every node).
    """
    rel_raw = _tq.fm_get(fm_lines, "relevance")
    relevance: float | None
    try:
        relevance = float(rel_raw) if rel_raw is not None else None
    except (TypeError, ValueError):
        relevance = None
    if relevance is None:
        tier_to_rel = {"hot": 0.80, "warm": 0.60, "cold": 0.40, "frozen": 0.10}
        relevance = tier_to_rel.get(
            (_tq.fm_get(fm_lines, "tier") or "warm").strip().lower(), 0.60)
    if relevance >= 0.75:
        return True, 5
    if relevance >= 0.50:
        return True, 4
    if relevance >= 0.25:
        return True, 3
    return False, 1


def sm2_sweep_tick(memory_dir: Path,
                   today: _dt.date | None = None,
                   seed_cap: int = SM2_SWEEP_SEED_CAP,
                   review_cap: int = SM2_SWEEP_REVIEW_CAP) -> dict:
    """Scheduled SM-2 spaced-repetition sweep (the dormant-loop driver).

    What: in one pass it (1) INCREMENTALLY seeds SM-2 frontmatter onto nodes
    that lack ``next_review`` — capped at ``seed_cap`` per tick so the ~2.9k
    corpus is migrated over several daily ticks instead of one mass rewrite —
    then (2) iterates ``sm2_due_for_review`` and applies ``sm2_review_update``
    to each due node (capped at ``review_cap``), deriving the review quality
    from the node's usage signal via ``_sm2_quality_from_usage``.

    Why: ``sm2_review_update``/``sm2_due_for_review`` are correct but had no
    scheduled caller, so 0 of the corpus carried ``next_review`` and the loop
    never advanced. The scheduler now calls this daily (see
    samia.runtime.scheduler job ``sm2_review_sweep``). Seeding and reviewing
    in the same tick is intentional: a node seeded with next_review=today is
    immediately due, so it gets its first usage-derived review in the same
    sweep, bootstrapping the schedule from real hotness rather than a default.
    Canonical Wozniak constants are untouched — the seed values only fill the
    same defaults sm2_review_update already assumes.
    """
    nodes_dir = _nodes_dir(memory_dir)
    today = today or _dt.date.today()
    today_iso = today.isoformat()

    seeded: list[str] = []
    for p in sorted(nodes_dir.glob("*.md")):
        if len(seeded) >= seed_cap:
            break
        fm_lines, body = _tq.read_node(p)
        if _tq.fm_get(fm_lines, "next_review") is not None:
            continue
        fm_lines = _tq.fm_set(fm_lines, "easiness_factor",
                              f"{SM2_SEED_EASINESS:.3f}")
        fm_lines = _tq.fm_set(fm_lines, "review_interval_days",
                              str(SM2_SEED_INTERVAL_DAYS))
        fm_lines = _tq.fm_set(fm_lines, "review_count",
                              str(SM2_SEED_REVIEW_COUNT))
        fm_lines = _tq.fm_set(fm_lines, "next_review", today_iso)
        _tq.write_node(p, fm_lines, body)
        seeded.append(p.name)

    reviewed: list[dict] = []
    for due in sm2_due_for_review(memory_dir, today=today):
        if len(reviewed) >= review_cap:
            break
        p = nodes_dir / due["node"]
        fm_lines, _ = _tq.read_node(p)
        recalled, quality = _sm2_quality_from_usage(fm_lines)
        result = sm2_review_update(memory_dir, due["node"],
                                   recalled=recalled, quality=quality)
        result["quality"] = quality
        result["recalled"] = recalled
        reviewed.append(result)

    return {
        "seeded_count": len(seeded),
        "seeded": seeded[:10],
        "seed_cap_hit": len(seeded) >= seed_cap,
        "reviewed_count": len(reviewed),
        "reviewed": reviewed[:10],
        "review_cap_hit": len(reviewed) >= review_cap,
    }


# ---------------------------------------------------------------------------
# Compaction-aware skip filter
# ---------------------------------------------------------------------------


def compaction_skip_filter(memory_dir: Path, transcript_chunks: list[str],
                           threshold: float = 0.78) -> dict:
    keep: list[dict] = []
    skip: list[dict] = []
    for i, c in enumerate(transcript_chunks):
        d = _bio.pattern_separation_decision(memory_dir, c, threshold=threshold)
        if d["action"] == "merge_into":
            skip.append({"chunk_index": i, "covered_by": d["target"],
                         "score": d["score"]})
        else:
            keep.append({"chunk_index": i, "reason": "novel",
                         "best_score": d["score"]})
    return {"summarize": keep, "skip_already_in_memory": skip,
            "skip_ratio": (len(skip) / max(1, len(transcript_chunks)))}
