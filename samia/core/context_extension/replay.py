"""samia.core.context_extension.replay — the idle DMN replay tick + directed-SR producer.

Layer 1 (Owns / Depends):
    Owns:    Primitive E — the idle DMN replay tick (idle_replay_tick, the REM-gated
             offline replay/consolidation host) and its helpers: the replay-coactivation
             reseed (_record_replay_coactivations), the undirected/raw pair collectors
             (_replay_pairs), the episode_seq reader (_node_episode_seq), and the
             FEAT-2026-06-11 P6 directed-SR counting pass (_record_directed_transitions,
             the PRODUCER half of the producer/consumer split that increments
             biomimetic/episode_transitions.json).
    Depends: the package config leaf (json/time, _dt, the _bio alias, the _ctx_dir +
             _idle_state_path helpers + temporal_weight_enabled gate), the primitives arm
             (frozen_prefix_block, the light prefix refresh on the waking path), and —
             lazily — samia.runtime.rem_cycle (the REM gate) + samia.core.atomic_state
             (the locked_update_json primitive) + samia.core.frontmatter (the episode_seq
             read), each function-local to keep the runtime deps off the import path.

Layer 2 (What / Why):
    What: the single biggest offline-on-idle op — replay_sweep + interleaved replay +
          replay-coactivation reseed + engram feed-forward + Hebbian consolidate, all
          REM-gated, plus the cheap frozen-prefix refresh that stays on the waking path.
          The P6 directed pass rides the SAME in-window pairs, ordering each by
          episode_seq to increment the directed transition matrix T_dir.
    Why:  isolating the replay host keeps its REM gating + fail-soft posture auditable.
          The directed-SR producer is gated behind the master temporal flag (inert by
          default) and never breaks the consolidation it layers onto (fail-soft).
"""

from __future__ import annotations

from pathlib import Path

# Shared leaf — json/time, _dt, the _bio alias, the ctx-dir + idle-state helpers, and the
# temporal master-flag gate (the P6 directed pass is inert unless the flag is on).
from .config import (
    json,
    time,
    _dt,
    _bio,
    _ctx_dir,
    _idle_state_path,
    IDLE_THRESHOLD_SECONDS,
)
from .temporal import temporal_weight_enabled
# The light prefix refresh on the waking path — frozen_prefix_block is public and not a
# patch seam, so a plain intra-package import (not a facade reach) is correct here.
from .primitives import frozen_prefix_block


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
    # _nodes_dir is re-exported through config; resolve via the package leaf so the node
    # path layout stays single-sourced.
    from .config import _nodes_dir
    p = _nodes_dir(memory_dir) / fname
    if not p.exists():
        return None
    try:
        from .. import frontmatter as _fm
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
        from ..atomic_state import locked_update_json
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


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.context_extension.replay
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Primitive E — idle DMN replay tick (REM-gated, FEAT-2026-06-07 P2) +
#             FEAT-2026-06-11 P6 directed-SR producer (episode_transitions.json).
#             + Phase-B modularization (carved from the monolith, ZERO behavior change).
# Layer:      core (pure library, no daemon dependency)
# Role:       the single biggest offline-on-idle op + the directed-SR counting producer;
#             the light frozen-prefix refresh stays on the waking path.
# Stability:  stable — REM-gated heavy body, default-off P6 producer.
# ErrorModel: fail-soft / fail-open — rem_cycle unavailable falls back to legacy replay;
#             the directed pass + engram feed-forward swallow errors so consolidation is
#             never broken; the directed pass is inert unless the master temporal flag is on.
# Depends:    .config (json/time/_dt/_bio + ctx-dir/idle-state helpers), .temporal
#             (temporal_weight_enabled), .primitives (frozen_prefix_block); lazily
#             samia.runtime.rem_cycle + samia.core.{atomic_state,frontmatter}.
# Exposes:    idle_replay_tick (public) + the directed-SR helpers
#             (_record_directed_transitions etc., re-exported on the facade for the
#             test reach-in ce._record_directed_transitions).
# Lines:      311
# --------------------------------------------------------------------------
