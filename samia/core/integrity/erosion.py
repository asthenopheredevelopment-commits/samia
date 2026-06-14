"""samia.core.integrity.erosion — the slow per-character body erosion + masked read.

Layer 1 (Owns / Depends):
    Owns:    the per-pass erosion RATE math (erosion_rate — pure, no I/O), the
             deterministic character-drop primitive (_erode_body) + its intact-
             fraction read (_intact_fraction), the single-node erosion pass (erode —
             anchor-gated, mutates fm/order in place), the masked READ seam
             (mask_read — what a normal read sees), the LIVE salience resolver for
             the rate (live_salience), and the reconsolidation math
             (reconsolidate_integrity — pure, shared by both repair triggers).
    Depends: samia.core.integrity.config (the rate/modulation constants + random),
             samia.core.integrity.anchors (has_anchor gate, get_integrity/
             set_integrity field accessors). samia.core.bio (compute_salience,
             read-only, function-LOCAL) for the live-salience term. stdlib math
             (function-LOCAL, in erode).

Layer 2 (What / Why):
    What: the FORGETTING half of the second axis. erode() drops a small,
          deterministic number of visible characters per pass (replacing them with
          the sentinel so the body length + structure are preserved + observable),
          lowers the integrity score to the new intact fraction, and is GATED on a
          recoverable anchor being present. erosion_rate() / reconsolidate_integrity()
          are pure math so both axes stay testable + cheap.
    Why:  Q1c HYBRID + Q2a per-char rate — the erosion is REAL + observable (characters
          genuinely missing from the served body) but bounded per pass (slow), modulated
          by salience / tier (or the semantic permanence override) / recency, and any
          recall fully repairs it from the anchor.

SAFETY (produce-only, no data loss): erode() NEVER runs without a recoverable anchor
    present — if no anchor exists it is a NO-OP (returns the body unchanged). We never
    erode content we cannot faithfully restore.
"""

from __future__ import annotations

from .config import (
    BASE_EROSION_RATE,
    DEFAULT_TIER,
    EROSION_SENTINEL,
    INTEGRITY_FULL,
    INTEGRITY_NONE,
    Optional,
    Path,
    RECALL_REPAIR_STRENGTH,
    RECENCY_EROSION_CAP,
    RECENCY_EROSION_PER_DAY,
    SALIENCE_EROSION_DAMPING,
    SEMANTIC_EROSION_FACTOR,
    TIER_EROSION_FACTOR,
    random,
)
from .anchors import get_integrity, has_anchor, set_integrity


def erosion_rate(integrity: float, days_since_recall: int, tier: str,
                 salience: float = 0.0, is_semantic: bool = False) -> float:
    """The per-pass erosion rate: SLOW base * salience * (tier | semantic) * recency (Q2a).

    Returns the fraction of currently-intact characters to erode this pass. A pure
    function (no I/O) mirroring tier.step_relevance's shape so both axes are testable
    and the integrity work stays as cheap as the relevance step.

    Modulation:
      - tier:    hot erodes slowest, cold fastest (TIER_EROSION_FACTOR).
      - semantic: when is_semantic is True the tier factor is OVERRIDDEN by
                  SEMANTIC_EROSION_FACTOR (0.25, the slowest permanence rate) REGARDLESS
                  of tier — CLS: semantic knowledge is the most-permanence class. Default
                  is_semantic=False keeps the pure tier behavior for all other callers.
      - recency: longer since last recall -> more erosion (capped).
      - salience: high salience -> slower (rate *= 1 - DAMPING*salience). Default
                  salience 0.0 -> neutral 1.0 multiplier (no change), so P1 ships with
                  no salience dependency; the term simply does nothing until a salience
                  field exists. (P2 refines this further.)
    """
    rate = BASE_EROSION_RATE
    # CLS — semantic knowledge is the most-permanence class; everything still fades, at
    # the rate the population deserves. A type:semantic node takes the slowest (hot/frozen)
    # factor IN PLACE OF its tier factor, regardless of which tier it has aged into, while
    # the recency/salience modulation below still applies (it is NOT exempt from erosion).
    if is_semantic:
        rate *= SEMANTIC_EROSION_FACTOR
    else:
        rate *= TIER_EROSION_FACTOR.get(str(tier).lower(), TIER_EROSION_FACTOR[DEFAULT_TIER])
    recency_mult = 1.0 + min(
        RECENCY_EROSION_CAP,
        max(0, int(days_since_recall)) * RECENCY_EROSION_PER_DAY,
    )
    rate *= recency_mult
    sal = min(1.0, max(0.0, float(salience)))
    if sal > 0.0:
        rate *= (1.0 - SALIENCE_EROSION_DAMPING * sal)
    return max(0.0, rate)


def _erode_body(body: str, n_chars: int, seed: int) -> tuple[str, int]:
    """Drop a SMALL, deterministic number of NON-already-eroded characters.

    Returns (new_body, n_eroded). Erosion replaces a character with EROSION_SENTINEL
    so the body length is preserved (positional, observable) and an already-eroded
    position is never re-counted. Whitespace/newline positions are preserved (we erode
    visible content, not structure) so the body stays renderable.
    """
    if n_chars <= 0 or not body:
        return body, 0
    # Candidate positions: visible characters not already eroded and not whitespace.
    candidates = [
        i for i, ch in enumerate(body)
        if ch != EROSION_SENTINEL and not ch.isspace()
    ]
    if not candidates:
        return body, 0
    rng = random.Random(seed)
    rng.shuffle(candidates)
    take = candidates[:min(n_chars, len(candidates))]
    chars = list(body)
    for i in take:
        chars[i] = EROSION_SENTINEL
    return "".join(chars), len(take)


def _intact_fraction(body: str) -> float:
    """Fraction of visible characters still intact (not the sentinel)."""
    visible = [ch for ch in body if not ch.isspace()]
    if not visible:
        return INTEGRITY_FULL
    eroded = sum(1 for ch in visible if ch == EROSION_SENTINEL)
    return max(INTEGRITY_NONE, INTEGRITY_FULL - eroded / len(visible))


def erode(memory_dir: Path, node_name: str, fm: dict, order: list[str], body: str,
          days_since_recall: int = 0, tier: str = DEFAULT_TIER,
          salience: float = 0.0, seed: Optional[int] = None) -> tuple[str, float, int]:
    """Erode a SMALL number of characters from the served body (one slow pass).

    What: drops a small number of visible characters (replacing them with the erosion
      sentinel) proportional to a SLOW rate * salience * tier * recency, lowers the
      integrity score to the new intact fraction, and returns (new_body, new_integrity,
      n_eroded). Mutates `fm`/`order` IN PLACE to carry the new integrity field; the
      CALLER persists the (eroded body + lowered integrity). The ANCHOR is untouched —
      it is the faithful repair source.
    Why: Q1c HYBRID + Q2a per-char rate. The erosion is REAL + observable (characters
      genuinely missing from the served body) but bounded per pass (slow).

    SAFETY (produce-only, no data loss): erosion NEVER runs without a recoverable
      anchor present. If no anchor exists, this is a NO-OP (returns the body unchanged)
      — we never erode content we cannot faithfully restore.
    """
    # NEVER erode without a recoverable anchor (no irrecoverable loss).
    if not has_anchor(memory_dir, node_name, fm):
        return body, get_integrity(fm), 0

    cur_integrity = get_integrity(fm)
    # CLS per-type override: a type:semantic node erodes at the slowest (permanence) rate
    # regardless of tier. erode() already holds the node fm, so derive is_semantic here —
    # no signature churn, and every caller (sweep or direct) gets the override for free.
    node_is_semantic = str(fm.get("type", "")).lower() == "semantic"
    rate = erosion_rate(cur_integrity, days_since_recall, tier, salience,
                        is_semantic=node_is_semantic)

    visible = [ch for ch in body if not ch.isspace() and ch != EROSION_SENTINEL]
    n_visible = len(visible)
    if n_visible == 0 or rate <= 0.0:
        set_integrity(fm, order, cur_integrity)
        return body, cur_integrity, 0

    # Erode a fraction of the CURRENTLY-INTACT characters; round up so a slow rate on a
    # small body still erodes at least one character per pass (observable), but bound it
    # to the visible count so we never over-drop.
    import math
    n_chars = min(n_visible, max(1, math.ceil(n_visible * rate)))

    # Deterministic seed: stable per (node, current integrity) so a pass is reproducible
    # for testing while different passes erode different positions.
    if seed is None:
        seed = abs(hash((node_name, round(cur_integrity, 6)))) % (2 ** 31)

    new_body, n_eroded = _erode_body(body, n_chars, seed)
    new_integrity = _intact_fraction(new_body)
    set_integrity(fm, order, new_integrity)
    # Return the PERSISTED (rounded) value so the returned score matches the field.
    return new_body, get_integrity(fm), n_eroded


def mask_read(memory_dir: Path, node_name: str, body: str,
              fm: Optional[dict] = None) -> str:
    """Return the current (eroded) served body for a node — what a normal read sees.

    What: the served body already carries the erosion in P1 (erode() persists the
      sentinel-bearing body to nodes/<n>.md), so the masked read is simply the stored
      body. This helper is the explicit READ seam: it returns the served (eroded) body,
      DISTINCT from the pristine anchor (read_anchor). It never mutates anything.
    Why: masked reads — the system genuinely experiences the partial forgetting (the
      read reflects the current integrity) while the anchor remains the repair source.
    """
    return body


def reconsolidate_integrity(old_integrity: float,
                            strength: float = RECALL_REPAIR_STRENGTH) -> float:
    """Restore integrity toward FULL by `strength` (pure math).

    strength 1.0 -> full restore (recall, Q3a strongest); partial (consolidation/
    reconciliation) is P2. Returns the new [0,1] integrity.
    """
    s = min(1.0, max(0.0, float(strength)))
    new = old_integrity + (INTEGRITY_FULL - old_integrity) * s
    return min(INTEGRITY_FULL, max(INTEGRITY_NONE, new))


def live_salience(memory_dir: Path, node_name: str, fm: Optional[dict] = None) -> float:
    """Resolve the LIVE [0,1] salience signal for a node's erosion rate (P2 / Q2a).

    What: prefer the LIVE composite salience (bio.compute_salience, write=False — the
      same surprise + contradiction-involvement + repetition signal the rest of the
      system maintains), falling back to the maintained `salience` frontmatter field,
      then to the neutral 0.0 when neither is available. Returns a [0,1] float.
    Why:  Q2a — the erosion rate must read the REAL salience signal so a genuinely
      high-salience node erodes slower, not just a node that happens to carry a stale
      static field. P1 only read the static field; P2 couples the rate to the live
      signal while keeping the static field as a graceful fallback (so the sweep never
      crashes if the inference/embedder backend is unavailable — compute_salience is
      itself fail-soft and returns 0.0 for a missing signal).
    """
    # 1) the LIVE composite (read-only — never writes the field from the erosion path).
    # Function-LOCAL import — bio is a heavy dep (embedder/inference); keeping it off the
    # package import path avoids pulling it in on a plain `import samia.core.integrity`.
    try:
        from .. import bio as _bio
        sal = _bio.compute_salience(memory_dir, node_name, write=False)
        if sal is not None:
            return min(1.0, max(0.0, float(sal)))
    except Exception:
        pass  # fail-soft: fall back to the maintained field below.
    # 2) the maintained frontmatter field (P1's static fallback).
    if fm is not None:
        try:
            return min(1.0, max(0.0, float(fm.get("salience", 0.0) or 0.0)))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.integrity.erosion
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.integrity monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       the slow per-character body erosion (the FORGETTING half) — the pure
#             rate math, the deterministic character-drop, the single-node anchor-
#             gated erode pass, the masked read seam, the live-salience rate input,
#             and the pure reconsolidation math both repair triggers share.
# Stability:  stable — carved byte-identically from the monolith. erosion_rate /
#             reconsolidate_integrity are pure (no I/O); erode mutates fm/order in
#             place and the CALLER persists the eroded body (the anchor is untouched).
# ErrorModel: erode NEVER erodes without a recoverable anchor (no data loss — a no-op
#             returns the body unchanged); live_salience is fail-soft (bio read is
#             function-local + try-guarded -> falls back to the static field -> 0.0).
# Depends:    .config (rate/modulation constants + random), .anchors (has_anchor gate,
#             get_integrity/set_integrity). samia.core.bio (compute_salience, read-only,
#             function-LOCAL). stdlib math (function-LOCAL in erode).
# Exposes:    erosion_rate, erode, mask_read, reconsolidate_integrity, live_salience
#             (+ the _erode_body/_intact_fraction privates).
# Lines:      263
# --------------------------------------------------------------------------
