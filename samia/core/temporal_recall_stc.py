"""temporal_recall_stc.py -- synaptic tagging-and-capture term (P4).

Layer 1 (Owns / Depends):
    Owns:    The STC (Synaptic Tagging-and-Capture) machinery of the temporal-recall
             layer (FEAT-2026-06-11-memory-temporal-recall-formula-v01 §6 + §16.2 Q2)
             — the inside of the multiplicative envelope's
               K̂_c = min-max-norm( max_{m∈c} stc_capture_score(m) ).
             Three pieces:
               1. The CAPTURE event (write-time): a write whose salience clears
                  STC_STRONG_THRESHOLD is a strong anchor; it stamps a decaying
                  stc_capture_score onto temporally-adjacent WEAK nodes inside an
                  EPISODE_SEQ-relative window (N before / M after, N>M — §16.2 Q2
                  SUPERSEDES the §6.3 wall-clock [t−9h,t+3h]) bounded by a wall-clock
                  cap, behind two guards: cos(anchor,weak) ≥ STC_COSINE_GATE and a
                  1-event/chain/hour rate-limit (ledger via locked_update_json).
               2. The READ-OUT (stc_chain_score): max over chain members of each
                  member's time-attenuated stc_capture_score (the §6.5 max reducer),
                  consumed at retrieval through context_extension._stc_term_chain.
               3. The attenuation helper (current_capture_score): 0.5 ** (days /
                  STC_HALFLIFE_DAYS) applied to a node's stored stc_capture_score
                  (shared by recall read-out, promotion, and decay so all three see
                  the SAME decayed scalar).
    Depends: numpy; samia.core.frontmatter (read/write node frontmatter — the single
             decaying scalar lives beside `salience`, §6.5); samia.core.bio
             (_bio_paths, _addr_for_node, _node_embedding — REUSED); samia.core.temporal
             (parse_date / st_mtime fallbacks for the wall-clock cap on a legacy node);
             samia.core.atomic_state.locked_update_json (the EXISTING flock primitive —
             the rate-limit ledger, NOT a new lock); context_extension.temporal_weight_
             enabled (the master flag — capture is inert when off).

Layer 2 (What / Why):
    What: capture_event(memory_dir, anchor_node) is the write-time trigger — it reads the
          anchor's salience, returns inert when it is below the strong bar OR the master
          flag is off, else scopes the EPISODE_SEQ-relative window over the anchor's
          chain neighbourhood (the nodes nearest by episode_seq, N before / M after,
          inside the wall-clock cap), applies the cosine guard + the per-chain/hour rate
          limit, and stamps a fresh stc_capture_score (+ stc_capture_at written_at) onto
          each captured weak node. stc_chain_score(memory_dir, member_nodes) is the read
          side — max over members of current_capture_score (the time-attenuated scalar).
    Why:  §6. A weak memory written near a strongly-salient one should be slightly harder
          to lose and slightly easier to recall together (Frey & Morris 1997; behavioural
          tagging, Moncada 2022). §16.2 Q2 moves the window unit from wall-clock hours to
          episode_seq so capture scopes to the LOCAL EPISODE NEIGHBOURHOOD regardless of
          write density (burst-invariant) — a wall-clock cap still bounds the human-side
          span. The score is a SINGLE decaying scalar in frontmatter (beside salience),
          which the decay tick already parses, so the read costs no new I/O (§6.5); the
          retrieval-only SITH bank, by contrast, goes to a sidecar (§4.4).

Flag posture: P4 is inert under the master flag off. capture_event fires NOTHING when
    ASTHENOS_TEMPORAL_WEIGHT is off (no frontmatter written), so no node ever carries
    stc_capture_score and the decay path (tier.step_relevance) + the promotion OR-gate
    + the recall read-out all see 0.0 → byte-identical to today. With the flag on but
    λK = 0 the recall read-out is compute-skipped (§16.2 Q5). A legacy node lacking
    written_at/episode_seq simply cannot fall in any episode_seq window and is skipped
    (no crash, no migration) — additive-optional. The stored scalar attenuates with a
    ~3-day half-life and a missing/expired score reads 0.0 (fail-open).
"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from .atomic_state import locked_update_json

# ── Capture parameters (§6.6 seeds; all join the joint-calibration vector later) ────
# STC_STRONG_THRESHOLD -- What: the salience floor a write must clear to be a STRONG
#   anchor (and so capture its weak neighbours).
# Why: §6.2 — reuse the existing compute_salience signal; 0.70 sits BELOW
#   SALIENCE_PROMOTE_THRESHOLD (0.8) on purpose: a write can rescue its weak neighbours
#   without being salient enough to force its own promotion. An explicit tag (0.95)
#   always qualifies. Calibrated vs corpus write-rate (over-firing inflates STC).
STC_STRONG_THRESHOLD = 0.70
STC_STRONG_THRESHOLD_ENV = "ASTHENOS_STC_STRONG_THRESHOLD"


def _strong_threshold() -> float:
    """Salience floor to be a STRONG STC anchor -- env-tunable, default STC_STRONG_THRESHOLD.

    What: read ASTHENOS_STC_STRONG_THRESHOLD per call (lazy os import, matching this module's
      lazy-import idiom); fall back to the 0.70 default on a missing / unparseable / out-of-[0,1]
      value.
    Why:  §6.6 names the STC params as joint-calibration knobs, but the bar was hard-coded.
      Per-call env read (so the harness _theta_env per-eval scoping works) lets calibration set
      it per corpus without a code change -- matching every other ASTHENOS_* knob. Default
      unchanged => byte-identical when the env is unset.
    """
    import os
    try:
        v = float(os.environ.get(STC_STRONG_THRESHOLD_ENV, "") or STC_STRONG_THRESHOLD)
    except (TypeError, ValueError):
        return STC_STRONG_THRESHOLD
    return v if 0.0 <= v <= 1.0 else STC_STRONG_THRESHOLD


# STC_WINDOW_BACK_N / _FWD_M -- What: the EPISODE_SEQ-relative capture window — the N
#   nearest episodes BEFORE the anchor and the M nearest AFTER, by counter (§16.2 Q2).
# Why: N > M preserves the strong-before-weak biological asymmetry (a strong event
#   rescues weak memories that PRECEDED it over a longer ordinal span than ones that
#   FOLLOW). episode_seq is the burst-invariant unit (the substrate already mints it),
#   so capture scopes to the local episode neighbourhood regardless of write density —
#   this SUPERSEDES the §6.3 wall-clock [t−9h,t+3h] window. Seeds 9/3 mirror the 9h/3h
#   asymmetry; tunable, join the calibration vector.
STC_WINDOW_BACK_N = 9
STC_WINDOW_FWD_M = 3

# STC_WALLCLOCK_CAP_S -- What: the wall-clock cap (seconds) bounding the human-side span
#   of the episode_seq window (§16.2 Q2 — "with a wall-clock cap").
# Why: at human pacing a few episodes can straddle days; the cap keeps a capture from
#   reaching across an unboundedly long real-time gap even when only a few episodes
#   intervene. Reads the written_at float; a weak node beyond the cap is skipped. Seed
#   = the §6.3 back-arm (9h) so the human-side bound matches the prior wall-clock span.
STC_WALLCLOCK_CAP_S = 9 * 3600.0

# STC_COSINE_GATE -- What: the semantic floor a weak node's embedding must clear vs the
#   anchor to be captured.
# Why: §6.4 guard 1 — temporal adjacency alone is too permissive; capture is
#   dendritic-compartment-LOCAL, not cell-wide. Reuses the uniform θ=0.2 the whole
#   temporal subsystem shares (one cosine floor across TĈ + STC).
STC_COSINE_GATE = 0.20

# STC_RATE_LIMIT_S -- What: the per-chain rolling rate-limit window (seconds): at most
#   ONE capture event per chain per hour.
# Why: §6.4 guard 2 — the homeostatic brake (biological capture is resource-limited),
#   preventing STC from being farmed into a flat high baseline. Enforced with the same
#   flock + atomic-replace discipline (locked_update_json) as the episode_seq counter.
STC_RATE_LIMIT_S = 3600.0

# STC_CAPTURE_SCORE -- What: the [0,1] score a fresh capture stamps (pre-attenuation).
# Why: a full tag at the moment of capture; it then decays with STC_HALFLIFE_DAYS so the
#   rescue is transient (§6.5 effect 3). Seed 1.0 (a fresh tag is maximal); the recall
#   modulator is min-max pooled across the candidate set anyway, so the absolute value
#   matters only relative to the half-life attenuation.
STC_CAPTURE_SCORE = 1.0

# STC_HALFLIFE_DAYS -- What: the half-life (days) of the stored stc_capture_score.
# Why: §6.5 effect 3 — biological tags clear within hours-to-a-day; a PERMANENT rescue
#   would let one salient write immortalize arbitrary neighbours. After ~3 days the
#   rescue has half-faded; after ~2 weeks it is negligible and the node resumes its
#   intrinsic decay. Shared by recall/promotion/decay so all three read the SAME scalar.
STC_HALFLIFE_DAYS = 3.0

# STC_PROMOTE_THRESHOLD -- What: the (attenuated) stc_capture_score at/above which a weak
#   node becomes promotion-eligible via the OR-gate's third arm.
# Why: §6.5 effect 2 — the direct realization of synaptic capture: a weak memory is
#   rescued into the long-term (engram) store by its strong neighbour, WITHOUT meeting
#   the frequency bar or its own salience bar. Joins the joint-calibration vector.
STC_PROMOTE_THRESHOLD = 0.50


def _bio():
    """Lazy bio import (dodges the bio<->temporal_recall_stc cycle: bio calls capture)."""
    from . import bio as _b
    return _b


def _events_ledger_path(memory_dir: Path) -> Path:
    """The per-chain rate-limit ledger, alongside the other biomimetic/ state JSONs.

    What: biomimetic/stc_events.json — {chain -> last_fire_unix}, updated under the same
      flock + atomic-replace discipline (locked_update_json) as episode_seq.json.
    Why: §6.4 guard 2 — the homeostatic brake needs a durable, concurrency-safe record
      of when each chain last fired; reusing locked_update_json adds no new lock.
    """
    return Path(memory_dir) / "biomimetic" / "stc_events.json"


def _node_chains(memory_dir: Path, node: str) -> list[str]:
    """The chain names a node belongs to (read from its `chains` frontmatter).

    What: parse the node's `chains: [a, b]` frontmatter into a list of chain names.
    Why: the capture window scans the anchor's OWN chain(s) (§6.3) and the rate-limit is
      per-chain; both need the anchor's chain membership. Reads the same field the
      reconsolidate / refine paths read. Missing/unparseable -> [] (fail-open).
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = Path(memory_dir) / "nodes" / fname
    if not p.exists():
        return []
    try:
        from . import frontmatter as _fm
        fm, _order, _body = _fm.read_node(p)
    except Exception:
        return []
    raw = fm.get("chains")
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    if isinstance(raw, str):
        inner = raw.strip()
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
        return [c.strip() for c in inner.split(",") if c.strip()]
    return []


def _chain_members(memory_dir: Path, chain_name: str) -> list[str]:
    """The node filenames that are members of `chain_name` (read from chains/<c>.json).

    What: load chains/<chain_name>.json and return each member's bare node filename.
    Why: the episode_seq window scans the anchor's chain neighbourhood, NOT the whole
      corpus (§6.3 — keeps write-time cost bounded). Missing/unparseable -> [].
    """
    cp = Path(memory_dir) / "chains" / f"{chain_name}.json"
    if not cp.exists():
        return []
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for m in data.get("members") or []:
        f = m.get("file") if isinstance(m, dict) else None
        if f:
            out.append(Path(f).name)
    return out


def _node_fields(memory_dir: Path, node: str) -> Optional[dict]:
    """Read (episode_seq, written_at, salience) for a node, or None if unreadable.

    What: pull the three numeric frontmatter fields the window/guards need. A legacy
      node lacking episode_seq/written_at returns None for those keys (not a crash);
      the caller then SKIPS it (can't place a field-less node in a sub-day window).
    Why: §3.3 additive-optional — every consumer fails open on the field's absence.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = Path(memory_dir) / "nodes" / fname
    if not p.exists():
        return None
    try:
        from . import frontmatter as _fm
        fm, _order, _body = _fm.read_node(p)
    except Exception:
        return None

    def _f(key):
        try:
            v = fm.get(key)
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "episode_seq": _f("episode_seq"),
        "written_at": _f("written_at"),
        "salience": _f("salience") or 0.0,
    }


def current_capture_score(memory_dir: Path, node: str,
                          now: Optional[float] = None) -> float:
    """The time-attenuated stc_capture_score of a node (the §6.5 shared scalar).

    What: read the node's stored stc_capture_score + stc_capture_at (written_at of the
      capture), and attenuate by 0.5 ** (days_since_capture / STC_HALFLIFE_DAYS). A node
      with no stored score (legacy / never captured / flag-off) reads 0.0 (fail-open).
    Why: §6.5 effect 3 — the rescue is TRANSIENT. Sharing this one helper across the
      recall read-out, the promotion OR-gate, and the decay damping guarantees all three
      see the SAME decayed value. Identity-at-zero: an absent score -> 0.0 everywhere.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = Path(memory_dir) / "nodes" / fname
    if not p.exists():
        return 0.0
    try:
        from . import frontmatter as _fm
        fm, _order, _body = _fm.read_node(p)
    except Exception:
        return 0.0
    try:
        raw = float(fm.get("stc_capture_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if raw <= 0.0:
        return 0.0
    try:
        captured_at = float(fm.get("stc_capture_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        captured_at = 0.0
    return attenuate(raw, captured_at, now=now)


def attenuate(raw_score: float, captured_at: float,
              now: Optional[float] = None) -> float:
    """Apply the ~3-day half-life attenuation to a stored capture score (pure).

    What: 0.5 ** (days_since_capture / STC_HALFLIFE_DAYS) · raw_score, days from the
      written_at-style float captured_at. captured_at <= 0 (unknown) -> no attenuation
      (treat as fresh — fail toward retaining the explicitly-stored value).
    Why: §6.5 — a single, testable closed-form decay reused by tier.step_relevance and
      the recall/promotion read-outs so they never diverge.
    """
    if raw_score <= 0.0:
        return 0.0
    if captured_at <= 0.0:
        return float(max(0.0, min(1.0, raw_score)))
    t = now if now is not None else _time.time()
    days = max(0.0, (t - captured_at) / 86400.0)
    factor = 0.5 ** (days / max(STC_HALFLIFE_DAYS, 1e-9))
    return float(max(0.0, min(1.0, raw_score * factor)))


def stc_chain_score(memory_dir: Path, member_nodes: list[str],
                    now: Optional[float] = None) -> float:
    """K_c = max_{m∈c} stc_capture_score(m) — the §6.5 recall read-out (max reducer).

    What: max over the chain's (gated) member nodes of each member's time-attenuated
      capture score. An empty member list / all-uncaptured -> 0.0. The caller (the P1 K
      seam) pool min-max normalizes this to K̂_c ∈ [0,1] and enters it as λK·K̂_c.
    Why: §6.5 effect 1 + §2.3 — a single strongly-rescued member should lift the whole
      chain (max, not average: averaging would dilute a real tag). Fail-open at every
      step so the recall path is never broken by STC.
    """
    best = 0.0
    for m in member_nodes or []:
        try:
            s = current_capture_score(memory_dir, m, now=now)
        except Exception:
            s = 0.0
        if s > best:
            best = s
    return best


def _embedding(memory_dir: Path, node: str):
    """Embedding of a node via bio._node_embedding (REUSED), or None.

    bio._node_embedding keys the vector manifest by the `<node>.md` filename, so the
    bare-or-suffixed node id is normalized to the .md form before lookup.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    try:
        return _bio()._node_embedding(memory_dir, fname)
    except Exception:
        return None


def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _stamp_capture(memory_dir: Path, node: str, score: float,
                   captured_at: float) -> bool:
    """Write stc_capture_score + stc_capture_at onto a weak node's frontmatter (fail-soft).

    What: set the two additive-optional scalars beside `salience` (the field the decay
      tick already parses), preserving every other field. A fresh capture OVERWRITES a
      prior (weaker, more-faded) one — the latest strong neighbour's tag wins.
    Why: §6.5 storage — a single decaying scalar in frontmatter rides the existing decay
      read; no sidecar. Fail-soft: a write failure must never crash the capture path.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = Path(memory_dir) / "nodes" / fname
    if not p.exists():
        return False
    try:
        from . import frontmatter as _fm
        fm, order, body = _fm.read_node(p)
        if "stc_capture_score" not in fm:
            order.append("stc_capture_score")
        fm["stc_capture_score"] = round(float(score), 4)
        if "stc_capture_at" not in fm:
            order.append("stc_capture_at")
        fm["stc_capture_at"] = float(captured_at)
        _fm.write_node(p, fm, order, body)
        return True
    except Exception:
        return False


def _rate_limited(memory_dir: Path, chains: list[str], now: float) -> set:
    """Return the subset of `chains` that fired an STC event within STC_RATE_LIMIT_S.

    What: read the biomimetic/stc_events.json ledger under an exclusive flock; a chain
      whose last_fire_unix is younger than the rate-limit window is BLOCKED this fire.
      The chains that are NOT blocked have their last_fire_unix stamped to `now` in the
      same locked transaction (so a concurrent anchor on the same chain is serialized).
    Why: §6.4 guard 2 — 1 STC event per chain per rolling hour, the homeostatic brake,
      enforced with the EXISTING locked_update_json (no new lock). Returns the blocked
      set so the caller can scope the capture to the non-blocked chains only.
    """
    blocked: set = set()
    ledger_path = _events_ledger_path(memory_dir)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with locked_update_json(ledger_path, default={}) as st:
        for ch in chains:
            try:
                last = float(st.get(ch, 0.0) or 0.0)
            except (TypeError, ValueError):
                last = 0.0
            if (now - last) < STC_RATE_LIMIT_S:
                blocked.add(ch)
            else:
                st[ch] = now  # claim the slot for this chain (serialized under flock)
    return blocked


def capture_event(memory_dir: Path, anchor_node: str,
                  now: Optional[float] = None) -> dict:
    """The write-time STC capture trigger (§6.2-6.4 + §16.2 Q2). Inert when flag off.

    What: if the master temporal flag is off -> no-op (nothing written). Else read the
      anchor's salience; if it is below STC_STRONG_THRESHOLD the write is not a strong
      anchor -> no-op. Else scope the EPISODE_SEQ-relative window over the anchor's
      chain neighbourhood (the N nearest episodes BEFORE and M nearest AFTER the anchor
      by episode_seq, bounded by the wall-clock cap), per non-rate-limited chain, and
      stamp a fresh stc_capture_score on each WEAK node that clears the cosine gate.
    Why: §6 — a strong event temporally captures (rescues) its weak neighbours. §16.2 Q2
      makes the window episode_seq-relative (burst-invariant), SUPERSEDING §6.3. The two
      guards (cosine + 1/chain/hour) and the strong->weak-only direction keep it bounded
      and non-farmable. Capture is gated behind the master flag so flag-off writes touch
      NO frontmatter (the decay/promotion/recall paths then see byte-identical behaviour).

    Returns {"fired": bool, "captured": [node...], "reason": str} — diagnostics only.
    Fail-soft: every error is swallowed so a hot write path is never broken.
    """
    try:
        from . import context_extension as _ce
        if not _ce.temporal_weight_enabled():
            return {"fired": False, "captured": [], "reason": "flag-off"}
    except Exception:
        return {"fired": False, "captured": [], "reason": "flag-unreadable"}

    try:
        anchor = _node_fields(memory_dir, anchor_node)
        if anchor is None:
            return {"fired": False, "captured": [], "reason": "anchor-unreadable"}
        if anchor.get("salience", 0.0) < _strong_threshold():
            return {"fired": False, "captured": [], "reason": "not-strong"}
        anchor_seq = anchor.get("episode_seq")
        anchor_wat = anchor.get("written_at")
        if anchor_seq is None:
            # No order field on the anchor -> can't place a window. Legacy/secondary
            # write seam -> degrade gracefully (no capture), per §3.2.
            return {"fired": False, "captured": [], "reason": "anchor-no-seq"}

        t_now = now if now is not None else _time.time()
        chains = _node_chains(memory_dir, anchor_node)
        if not chains:
            return {"fired": False, "captured": [], "reason": "anchor-no-chain"}

        # §6.4 guard 2: drop chains that fired within the last hour (and claim the slot
        # for the chains that DO fire, under the shared flock).
        blocked = _rate_limited(memory_dir, chains, t_now)
        active_chains = [c for c in chains if c not in blocked]
        if not active_chains:
            return {"fired": False, "captured": [], "reason": "rate-limited"}

        anchor_emb = _embedding(memory_dir, anchor_node)
        anchor_fname = (anchor_node if anchor_node.endswith(".md")
                        else f"{anchor_node}.md")

        captured: list[str] = []
        for ch in active_chains:
            members = _chain_members(memory_dir, ch)
            # Build (episode_seq, node) for every member that has a seq and is not the
            # anchor itself, then take the N nearest BEFORE and M nearest AFTER by seq.
            befores: list[tuple[float, str]] = []
            afters: list[tuple[float, str]] = []
            for m in members:
                if m == anchor_fname:
                    continue
                f = _node_fields(memory_dir, m)
                if f is None or f.get("episode_seq") is None:
                    continue  # legacy / field-less -> can't window it (§3.3 fail-open)
                seq = f["episode_seq"]
                if seq < anchor_seq:
                    befores.append((seq, m))
                elif seq > anchor_seq:
                    afters.append((seq, m))
            # nearest-by-counter: BEFORE descending (closest first), AFTER ascending.
            befores.sort(key=lambda t: -t[0])
            afters.sort(key=lambda t: t[0])
            window = ([m for _s, m in befores[:STC_WINDOW_BACK_N]]
                      + [m for _s, m in afters[:STC_WINDOW_FWD_M]])

            for m in window:
                mf = _node_fields(memory_dir, m)
                if mf is None:
                    continue
                # strong->weak only: capture flows FROM strong TO weak (§6.3).
                if mf.get("salience", 0.0) >= _strong_threshold():
                    continue
                # wall-clock cap (§16.2 Q2): a weak node beyond the human-side span is
                # skipped even if it is ordinally near. Missing written_at -> skip the
                # cap check is unsafe (could capture across an unbounded gap), so a
                # field-less node is conservatively skipped.
                mwat = mf.get("written_at")
                if anchor_wat is not None and mwat is not None:
                    if abs(anchor_wat - mwat) > STC_WALLCLOCK_CAP_S:
                        continue
                elif mwat is None:
                    continue  # no anchor for the cap -> skip (no unbounded capture)
                # §6.4 guard 1: semantic cosine gate.
                if anchor_emb is not None:
                    if _cosine(anchor_emb, _embedding(memory_dir, m)) < STC_COSINE_GATE:
                        continue
                cap_at = anchor_wat if anchor_wat is not None else t_now
                if _stamp_capture(memory_dir, m, STC_CAPTURE_SCORE, cap_at):
                    captured.append(m)

        return {"fired": bool(captured), "captured": captured,
                "reason": "captured" if captured else "no-weak-neighbours"}
    except Exception:
        # Fail-soft: a substrate hiccup must never break the write that triggered it.
        return {"fired": False, "captured": [], "reason": "error"}


# ── module metadata ────────────────────────────────────────────────────────
# file:        samia/core/temporal_recall_stc.py
# role:        synaptic tagging-and-capture (STC) term of the temporal-recall layer
# phase:       FEAT-2026-06-11-memory-temporal-recall-formula-v01 P4 (§6 + §16.2 Q2).
#              Builds the write-time capture trigger (strong anchor -> stamp a decaying
#              stc_capture_score on weak neighbours inside an EPISODE_SEQ-relative window
#              — N before / M after, N>M — with a wall-clock cap; cosine + 1/chain/hour
#              guards), the recall read-out (max over chain members), and the shared
#              ~3-day half-life attenuation consumed by recall / promotion / decay.
# supersession: §16.2 Q2 — the window is EPISODE_SEQ-relative (burst-invariant), NOT the
#              §6.3 wall-clock [t−9h,t+3h]; the wall-clock cap (written_at) only bounds
#              the human-side span. A term whose λK < ε compute-skips its read (§16.2 Q5).
# owns:        <memory_dir>/biomimetic/stc_events.json (per-chain rate-limit ledger);
#              the additive-optional `stc_capture_score` + `stc_capture_at` node
#              frontmatter scalars (beside `salience`).
# reuses:      atomic_state.locked_update_json (the rate-limit ledger — no new lock),
#              bio._node_embedding / _addr_for_node, frontmatter.read_node/write_node,
#              context_extension.temporal_weight_enabled (the master flag).
# flag:        INERT under ASTHENOS_TEMPORAL_WEIGHT off — capture writes NO frontmatter,
#              so the decay / promotion / recall paths are byte-identical to today. With
#              the flag on but λK = 0 the recall read-out is compute-skipped. A legacy
#              node lacking episode_seq/written_at is skipped (additive-optional, no
#              migration); a missing/expired stc_capture_score reads 0.0 (fail-open).
# consumers:   context_extension._stc_term_chain (recall λK modulator),
#              hippocampus.promote_ring_pointer (promotion OR-gate third arm),
#              tier.step_relevance (decay damping), bio (capture trigger at write).
# ─────────────────────────────────────────────
