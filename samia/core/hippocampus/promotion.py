"""samia.core.hippocampus.promotion — the P3 promotion lattice + its AUTO trigger.

Layer 1 (Owns / Depends):
    Owns:    the two upward transitions of the Tier-1 lattice, gated on
             max(frequency-signal, salience[, stc]) — (a) ring -> engram
             (promote_ring_pointer / promote_ring_step materialize a kWTA-coded copy)
             and (b) engram -> inject-eligible (mark_inject_eligible flags the
             "known-cold" standing set). Owns the Tier-0 signal reader
             attractor_strength (a node's strongest PROMOTABLE edge weight) and the
             salience reader _node_salience. These are FUNCTIONS callable from the
             capture/recall hook (and later a REM/idle pass) — P3 adds NO scheduler.
    Depends: .config (the P3 thresholds + _engram_dir), .engram (EngramStore — the
             copy target + the eligibility-flag write), .ring (RingStore — the pointer
             source + its backing salience/STC readers).  Lazily: samia.core.bio
             (_load_edge_weights / _is_promotable / _node_frontmatter) — function-local
             so the attractor signal reads Tier-0 without a top-level cross-import.

Layer 2 (What / Why):
    What: the promotion lattice. promote_ring_pointer materializes ONE wanted pointer
          when its gate (max(genuine-hits, salience, stc)) fires, then runs the
          engram->inject gate on the new copy. mark_inject_eligible computes
          max(attractor_strength, salience) >= bar and stamps inject_eligible on the
          held copy (it never injects — P4 does). promote_ring_step is the AUTO
          trigger: one pass over the ring (promote each) + an inject-eligibility
          refresh across all held copies.
    Why:  carved out of the 1339-line monolith as the lattice responsibility. It sits
          ABOVE .engram and .ring in the package DAG (it materializes through
          EngramStore and reads through RingStore) and is reached by the package facade
          + mcp_server.memory_rem_sleep_now (promote_ring_step) + the targeted tests.
"""

from __future__ import annotations

import json

from .config import (
    INJECT_PROMOTE_THRESHOLD,
    RING_PROMOTE_HITS,
    SALIENCE_PROMOTE_THRESHOLD,
    STC_PROMOTE_THRESHOLD,
    Path,
    _engram_dir,
    _ptr_name,
)
from .engram import EngramStore
from .ring import RingStore


def _node_salience(memory_dir: Path, node: str) -> float:
    """Read the [0,1] `salience` the P2 source wrote on a node (fail-soft -> 0.0)."""
    try:
        from .. import bio as _bio
        fm_bundle = _bio._node_frontmatter(memory_dir, node)
        if fm_bundle is None:
            return 0.0
        return float(fm_bundle[0].get("salience", 0.0) or 0.0)
    except Exception:
        return 0.0


def attractor_strength(memory_dir: Path, source_node: str) -> float:
    """Tier-0 attractor strength of a node = its strongest PROMOTABLE edge weight (D3).

    What: scan the Tier-0 edge_weights.json (bio._load_edge_weights) for every pair
      touching `source_node`; among the pairs that clear Tier-0's promotion gate
      (bio._is_promotable — w >= HEBB_PROMOTION AND count_genuine >= 1), return the max
      `w`. Returns 0.0 when no PROMOTABLE edge touches the node (so a node sitting on a
      sub-bar or replay-only edge contributes no attractor strength).
    Why:  D3/Q3a — the engram->inject gate reads Tier-0's signal DIRECTLY, inheriting
      its genuine-count guarantee (a replay-only edge can never confer inject-
      eligibility). This is the "known-cold by repetition" half of the max(attractor,
      salience) gate. Reuses bio's primitives; reinvents no edge logic.
    """
    fname = source_node if source_node.endswith(".md") else f"{source_node}.md"
    try:
        from .. import bio as _bio
        weights = _bio._load_edge_weights(memory_dir)
    except Exception:
        return 0.0
    best = 0.0
    for key, v in weights.items():
        if fname not in key.split("::"):
            continue
        if not _bio._is_promotable(v):
            continue  # only a GENUINE attractor confers strength (replay-only -> skip)
        best = max(best, float(v.get("w", 0.0)))
    return best


def mark_inject_eligible(memory_dir: Path, engram_id: str,
                         threshold: float = INJECT_PROMOTE_THRESHOLD) -> bool:
    """engram -> inject-eligible gate: flag the copy when max(attractor, salience) clears the bar (D3/D6).

    What: for the held copy `engram_id`, compute max(attractor_strength of its source,
      salience of its source) and set the copy's `inject_eligible` flag True when it
      meets `threshold` (default INJECT_PROMOTE_THRESHOLD). Persists the flag on the
      held-copy JSON. Returns the resulting inject_eligible bool (False if the copy is
      missing). It only ever SETS eligibility — it does not inject (P4).
    Why:  D6 effect (i) — the gate is max(attractor_strength, salience), so a frequency-
      earned attractor (w>=bar) AND a high-salience one-shot (salience>=bar, no
      repetition) both earn the standing inject slot, while a low-frequency low-salience
      copy does NOT. This is the eligibility computation P4's injector will consume.
    """
    store = EngramStore(memory_dir)
    copy = store.get(engram_id)
    if copy is None:
        return False
    source = copy.get("source_ptr", "")
    attr = attractor_strength(memory_dir, source)
    sal = _node_salience(memory_dir, source)
    eligible = max(attr, sal) >= float(threshold)
    if bool(copy.get("inject_eligible")) != eligible:
        copy["inject_eligible"] = eligible
        (_engram_dir(memory_dir) / f"{engram_id}.json").write_text(
            json.dumps(copy, indent=2), encoding="utf-8")
    return eligible


def promote_ring_pointer(memory_dir: Path, ptr: str,
                         hit_threshold: int = RING_PROMOTE_HITS,
                         salience_threshold: float = SALIENCE_PROMOTE_THRESHOLD,
                         stc_threshold: float = STC_PROMOTE_THRESHOLD
                         ) -> dict | None:
    """ring -> engram: materialize ONE wanted pointer when the promote gate fires (D3/D6).

    What: read the ring pointer `ptr`; if max(access_signal, salience, stc) crosses the
      bar — i.e. its GENUINE ring-RAG hits reach `hit_threshold`, OR its backing salience
      reaches `salience_threshold` (the D6 one-shot shortcut), OR its backing (attenuated)
      stc_capture_score reaches `stc_threshold` (the P4/§6.5 capture-rescue arm) —
      materialize its backing to a kWTA-coded engram held copy (the consolidation event)
      and immediately run the engram->inject gate (mark_inject_eligible). Returns
      {engram_id, inject_eligible, reason} on promotion (reason "stc" when capture-earned,
      auditable per §6.5), or None when the gate does not fire / the pointer is
      missing/engram-backed/dangling.
    Why:  D3/D6 — this is the ring->engram arm of the lattice with the salience shortcut:
      a frequently genuinely-recalled pointer earns a copy by frequency, AND a high-
      salience one-shot earns it WITHOUT repeated access. materialize() applies kWTA;
      main keeps the canonical (loss-free).
    """
    name = _ptr_name(ptr)
    if name.startswith("engram_"):
        return None  # already an engram copy; nothing to promote
    ring = RingStore(memory_dir)
    live = ring._load()
    entry = live.get(name)
    if entry is None:
        return None
    hits = int(entry.get("genuine_hits", 0))
    sal = ring._entry_salience(entry)
    freq_ready = hits >= int(hit_threshold)
    sal_ready = sal >= float(salience_threshold)
    # FEAT-2026-06-11 temporal-recall P4 (§6.5 effect 2): the STC capture OR-gate arm —
    # a weak node carrying a high (attenuated) stc_capture_score is rescued into the
    # engram store by its strong neighbour WITHOUT meeting the frequency or salience bar.
    # The gate widens from max(freq, salience) to max(freq, salience, stc). The capture
    # score is 0.0 everywhere when the temporal flag is off (no node carries it), so
    # stc_ready is False and this is byte-identical to today.
    stc = ring._entry_stc(entry)
    stc_ready = stc >= float(stc_threshold)
    if not (freq_ready or sal_ready or stc_ready):
        return None  # promote gate = max(access_signal, salience, stc) not met
    src_path = memory_dir / "nodes" / (name if name.endswith(".md")
                                       else f"{name}.md")
    if not src_path.exists():
        return None  # dangling — nothing to copy
    try:
        rec = EngramStore(memory_dir).materialize(name)
    except Exception:
        return None
    eligible = mark_inject_eligible(memory_dir, rec["engram_id"])
    # reason: capture-earned promotions are auditable as "stc" (§6.5 — preserves the
    # provenance that the weak node was RESCUED, not intrinsically frequent/salient).
    if stc_ready and not (freq_ready or sal_ready):
        reason = "stc"
    elif sal_ready and not freq_ready:
        reason = "salience"
    elif freq_ready and not sal_ready:
        reason = "frequency"
    else:
        reason = "both"
    return {"engram_id": rec["engram_id"], "inject_eligible": eligible,
            "reason": reason, "genuine_hits": hits, "salience": sal, "stc": stc}


def promote_ring_step(memory_dir: Path,
                      hit_threshold: int = RING_PROMOTE_HITS,
                      salience_threshold: float = SALIENCE_PROMOTE_THRESHOLD,
                      stc_threshold: float = STC_PROMOTE_THRESHOLD
                      ) -> dict:
    """One promotion PASS over the ring (the AUTO trigger — a function, not a scheduler).

    What: sweep every live ring pointer; promote_ring_pointer each (ring->engram on
      max(freq, salience)), then re-run mark_inject_eligible over EVERY held copy so the
      engram->inject set tracks the current Tier-0 attractor / salience state. Returns
      {promoted: [...], inject_eligible: int, scanned: int}. Pure side-effects on the
      hippocampus stores; mutates no main node and writes no Tier-0 edge.
    Why:  D3 — this is the lattice's auto-trigger, invokable from the capture/recall hook
      and later a REM/idle pass. P3 provides ONLY this function + its gate logic — no new
      timer/scheduler (CONSTRAINT). It computes ELIGIBILITY; it never injects (P4) and
      never touches decay (P5).
    """
    ring = RingStore(memory_dir)
    promoted: list[dict] = []
    scanned = 0
    for entry in ring.entries():
        scanned += 1
        res = promote_ring_pointer(memory_dir, entry["ptr"],
                                   hit_threshold, salience_threshold,
                                   stc_threshold)
        if res is not None:
            promoted.append(res)
    # Refresh inject-eligibility across ALL held copies (attractor/salience may have
    # changed since the copy was materialized) — eligibility computation only (P4 injects).
    eligible = 0
    for copy in EngramStore(memory_dir).all():
        if mark_inject_eligible(memory_dir, copy["engram_id"]):
            eligible += 1
    return {"promoted": promoted, "inject_eligible": eligible, "scanned": scanned}


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.hippocampus.promotion
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.hippocampus monolith during
#             modularization (the P3 lattice responsibility; bodies byte-identical, the
#             P3 thresholds/_engram_dir lifted into .config and EngramStore/RingStore
#             into their submodules).
# Layer:      core (pure library, no daemon dependency)
# Role:       P3 — the promotion lattice + its AUTO trigger (a FUNCTION, no scheduler):
#             ring->engram (promote_ring_pointer / promote_ring_step materialize a
#             kWTA-coded copy on max(genuine-hits, salience, stc)) and engram->inject
#             (mark_inject_eligible flags the known-cold set on max(attractor,
#             salience)); attractor_strength reads Tier-0's strongest PROMOTABLE edge.
# Stability:  stable — bodies byte-identical to the monolith; the carve only moved the
#             P3 thresholds + _engram_dir into .config and EngramStore/RingStore into
#             their submodules.
# ErrorModel: fail-soft — the salience/attractor readers swallow any error to 0.0; a
#             materialize failure inside promote_ring_pointer returns None (no copy, no
#             crash). It writes NO Tier-0 edge and mutates NO main node.
# Depends:    json. .config (P3 thresholds/_engram_dir/_ptr_name), .engram (EngramStore),
#             .ring (RingStore). Lazily: samia.core.bio (_load_edge_weights/
#             _is_promotable/_node_frontmatter).
# Exposes:    attractor_strength, mark_inject_eligible, promote_ring_pointer,
#             promote_ring_step (and _node_salience).
# Lines:      245
# ─────────────────────────────────────────────
