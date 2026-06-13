"""samia.core.integrity — content-fidelity decay (the SECOND, orthogonal decay axis).

FEAT-2026-06-07 granular-recall-repaired-decay, Phase P1.

Layer 1 (Owns / Depends):
    Owns:    The per-node content-INTEGRITY axis — a [0,1] fraction-intact score, a
             retained pristine recovery ANCHOR, a slow per-character body erosion, the
             masked read (what a normal read sees), and the recall-repair trigger that
             restores the body byte-exact from the anchor.
    Depends: samia.core.frontmatter (the canonical node read/serialize seam),
             samia.core.timestamp (UTC event stamps), json/pathlib/hashlib (stdlib).
             Reads (never writes) salience/tier/last_access frontmatter for rate
             modulation. Does NOT import tier — it RIDES tier.decay_pass, it does not
             touch the relevance/lifecycle math.

Layer 2 (What / Why):
    What: A genuinely SECOND decay axis, distinct from the relevance/tier decay in
          tier.py. Relevance-decay answers WHERE a node lives (lifecycle/tier);
          content-integrity decay answers HOW INTACT its content is (fidelity). A
          node's served/stored body erodes a little at a time (character-by-character),
          slowly, modulated by salience (high salience erodes slower), tier (hot < warm
          < cold), and recency (longer since recall -> more erosion). A RECALL repairs
          it faithfully from the pristine anchor, resetting integrity toward 1.0 — "a
          node missing a bit is easily read + restored just from recalling it".
    Why:  The operator's model of forgetting: granular + slow at the character level,
          coupled to reconsolidation on recall. The HYBRID model (Q1c) keeps a pristine
          anchor so early repair is FAITHFUL (not a guess); generative fallback (when
          the anchor itself is gone) is P3, NOT here. Layer-don't-replace: this composes
          alongside relevance-decay, it never modifies it (Q6a).

SCOPE (P1 + P2 + P3 — the full second axis):
    - P1: integrity field + anchor + slow per-char erosion + masked read + recall-repair.
    - P2: anchor-capture-on-write + consolidation/reconciliation PARTIAL repair +
      live-salience erosion modulation.
    - P3: TERMINAL FREEZE-AT-FLOOR (Q5a) — a node eroded below INTEGRITY_FLOOR without
      repair routes into the existing REVERSIBLE ia.freeze (demotion-to-frozen, restorable
      via ia.thaw), honoring the SAME salience exemption the relevance path uses; and
      GENERATIVE-RECONSTRUCTION FALLBACK (Q1c/Q4a) — when NO anchor remains, repair MAY
      reconstruct the body via the local inference backend, GATED behind an enable flag
      (default OFF) + inference availability, marked generative=true/anchor_faithful=false,
      and NEVER used when an anchor exists (anchor-first always wins).
    - PRODUCE-ONLY: the erosion sweep is a FUNCTION (no scheduler/thread/timer), it is
      additive + INERT by default (terminal_freeze OFF, generative OFF), and it NEVER
      erodes a node without a recoverable anchor present (no irrecoverable data loss).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

from . import frontmatter as _fm

# ── Constants (named + env-tunable in spirit; mirrors tier.DECAY_RATE_BY_GRADE) ──

# INTEGRITY_FULL / INTEGRITY_NONE — What: the endpoints of the fraction-intact scalar.
# Why: integrity is the fraction of the body's characters that remain intact, in
#   [0.0, 1.0]; 1.0 is pristine, 0.0 is fully eroded. A node with no `integrity`
#   field defaults to FULL (a pre-existing node is pristine until it first erodes).
INTEGRITY_FULL = 1.0
INTEGRITY_NONE = 0.0

# BASE_EROSION_RATE — What: the SLOW base fraction of currently-intact characters
#   eroded per decay pass at the neutral modulation (salience 0, warm tier, fresh).
# Why: per-char erosion (Q2a) must be GENTLE — the forgetting curve is slow, and any
#   recall fully repairs it, so a touched node never erodes. A small base keeps an
#   untouched node readable for many passes while still being observable per pass.
BASE_EROSION_RATE = 0.02

# TIER_EROSION_FACTOR — What: the per-tier multiplier on the base rate (hot erodes
#   slowest, cold fastest); the active set barely erodes while cold memories fade.
#   frozen=0.25 (slowest, hot-equivalent): a frozen node, once DISTILLED, erodes at
#   the slowest rate — its episodic trace fades gently after the semantic gist forms.
# Why: Q2a tier modulation — tier means the active set barely erodes while cold
#   memories fade. TUNE-2026-06-10 operator decision (c): frozen nodes erode ONLY
#   once their content is semantically covered (distilled). An UNDISTILLED frozen
#   node still NEVER erodes (the caller's distillation gate skips it); a DISTILLED
#   frozen node is eligible to erode at this slow factor (systems-consolidation: the
#   episodic trace fades only AFTER the semantic representation forms). archived
#   nodes never erode (the caller still skips target_state frozen/archived outright).
TIER_EROSION_FACTOR = {
    "hot":  0.25,
    "warm": 1.0,
    "cold": 2.5,
    "frozen": 0.25,
}
DEFAULT_TIER = "warm"

# SEMANTIC_EROSION_FACTOR — What: the per-pass erosion factor a type:semantic node uses
#   IN PLACE OF its tier factor — 0.25, the hot/frozen (slowest) permanence rate —
#   REGARDLESS of which tier the node currently lives in.
# Why: CLS — semantic knowledge is the most-permanence class; everything still fades,
#   at the rate the population deserves. A distilled fact (type:semantic) is the gist
#   that consolidation extracted; it should erode at the slowest rate the system grants
#   any node even when it has aged into a cold tier, while still eroding (it is NOT
#   exempt — the recency/salience modulation and the anchor gate still apply on top).
SEMANTIC_EROSION_FACTOR = 0.25

# RECENCY_EROSION_PER_DAY / RECENCY_EROSION_CAP — What: a recency multiplier that grows
#   with days-since-last-recall, capped so erosion never runs away.
# Why: Q2a recency modulation — longer since the last recall -> more erosion; ties
#   directly to the recall-repair coupling (a recently-recalled node erodes slowly).
#   The cap bounds the per-pass erosion so it stays SLOW even for ancient nodes.
RECENCY_EROSION_PER_DAY = 0.02
RECENCY_EROSION_CAP = 3.0

# SALIENCE_EROSION_DAMPING — What: the max fraction the erosion rate is REDUCED by at
#   full salience (1.0); effective rate *= (1 - SALIENCE_EROSION_DAMPING * salience).
# Why: Q2a salience modulation — high-salience nodes erode SLOWER (an important/one-
#   shot memory is durable). Salience 0 -> multiplier 1.0 -> no change. HIGH but <1 so
#   even a max-salience node still erodes a little (slow, never zero) until recalled.
SALIENCE_EROSION_DAMPING = 0.9

# EROSION_SENTINEL — What: the character a masked/eroded position renders as.
# Why: the erosion is OBSERVABLE on a read (you can see characters missing) while the
#   pristine anchor remains the faithful repair source. A single visible glyph.
EROSION_SENTINEL = "·"  # middle dot ·

# RECALL_REPAIR_STRENGTH — What: how far a RECALL pulls integrity back toward FULL.
# Why: Q3a — RECALL is the STRONGEST trigger: full reconsolidation. P1 wires recall
#   only (consolidation/reconciliation PARTIAL repair is P2). 1.0 = full restore.
RECALL_REPAIR_STRENGTH = 1.0

# PARTIAL_REPAIR_STRENGTH — What: how far CONSOLIDATION (a REM pass) and RECONCILIATION
#   (a contradiction/merge read) pull integrity back toward FULL — a PARTIAL fraction,
#   strictly < RECALL_REPAIR_STRENGTH.
# Why: Q3a — RECALL is the strongest (full); the offline/edit triggers heal PARTIALLY
#   what they touch ("sleep + reconciliation heal a little"). One partial constant for
#   both P2 offline triggers, well below 1.0 so a partial repair never matches a full
#   recall restore. Tunable in the same spirit as the erosion constants.
PARTIAL_REPAIR_STRENGTH = 0.5

# ── P3 constants — terminal freeze-at-floor + generative-reconstruction fallback ──

# INTEGRITY_FLOOR — What: the readable floor; below it a node has eroded past the point
#   of being usefully readable and (if not repaired) terminally FREEZES (Q5a).
# Why: P3 terminal — forgetting = demotion-to-frozen (reversible via ia.thaw + a later
#   recall reconsolidation), NEVER deletion. A LOW named threshold so the floor only
#   trips on a deeply, persistently-eroded node (one that has gone many passes without
#   any recall/consolidation/reconciliation repair). Env-tunable in the same spirit as
#   the erosion constants; the relevance axis's tier floor is a SEPARATE trigger feeding
#   the SAME reversible freeze path — the two axes compose, not collide.
INTEGRITY_FLOOR = 0.15

# SALIENCE_FREEZE_EXEMPT_DEFAULT — What: the fallback salience-exemption threshold used
#   if the canonical tier.SALIENCE_FREEZE_EXEMPT cannot be read.
# Why: P3 keeps the integrity floor's freeze policy CONSISTENT with the relevance path's
#   P5 salience exemption (a salience >= the exempt threshold node is NOT auto-frozen by
#   EITHER axis). We read the live tier constant at call time (so the two stay in lock-
#   step if it is re-tuned) but never import tier at module scope (integrity RIDES tier,
#   it does not couple to it). This default mirrors tier.SALIENCE_FREEZE_EXEMPT (0.85).
SALIENCE_FREEZE_EXEMPT_DEFAULT = 0.85

# INTEGRITY_GENERATIVE_ENABLED_ENV — What: the enable flag for the P3 generative-
#   reconstruction fallback (default OFF / inert).
# Why: Q1c/Q4a — generative reconstruction is the LAST RESORT, used ONLY when NO anchor
#   remains. It is the one path that can introduce drift (confabulation), so it is gated
#   behind BOTH this explicit flag AND inference availability, and is a SAFE NO-OP when
#   off/unavailable (the same posture as the P1/P2 anchor-missing no-op). Anchor-first
#   ALWAYS wins — generative never runs while an anchor exists.
INTEGRITY_GENERATIVE_ENABLED_ENV = "ASTHENOS_INTEGRITY_GENERATIVE_ENABLED"

# INTEGRITY_REPAIR_ENABLED_ENV / INTEGRITY_DECAY_ENABLED_ENV / INTEGRITY_FREEZE_ENABLED_ENV
#   — What: the three GRANULAR activation flags (each default OFF / inert) that wire the
#   already-built P1-P3 mechanism to the daemon call sites independently, so the operator
#   can enable each axis-arm on its own for systematic testing:
#     - REPAIR -> anchor-first RECALL-repair (memory_search) + the P2 consolidation-repair
#       REM subscriber. The REPAIR flag NAME is the SAME one the P2 subscriber
#       (rem_subscribers._integrity_repair_enabled) already reads — one flag, one meaning;
#       this just exposes a core-level reader so the recall-repair seam can share it.
#     - DECAY  -> the slow per-character EROSION sweep (erode_integrity in decay_tick /
#       integrity_decay_pass).
#     - FREEZE -> the terminal FREEZE-at-floor (terminal_freeze in integrity_decay_pass).
# Why: env flags alone could not activate the mechanism because the daemon call sites
#   default these FUNCTION PARAMS off and never passed them. These helpers read the env
#   live (each call) so a daemon/test that sets a flag post-import sees it, and default
#   '0'/OFF so an unset environment is byte-identical to the current inert behavior. They
#   mirror the existing generative_enabled()/contradiction.is_enabled() reader pattern.
INTEGRITY_REPAIR_ENABLED_ENV = "ASTHENOS_INTEGRITY_REPAIR_ENABLED"
INTEGRITY_DECAY_ENABLED_ENV = "ASTHENOS_INTEGRITY_DECAY_ENABLED"
INTEGRITY_FREEZE_ENABLED_ENV = "ASTHENOS_INTEGRITY_FREEZE_ENABLED"


def repair_enabled() -> bool:
    """True iff anchor-first integrity REPAIR is enabled (ASTHENOS_INTEGRITY_REPAIR_ENABLED).

    What: the live (read-each-call, default OFF) reader for the recall-repair seam in
      memory_search to share with the EXISTING P2 consolidation-repair subscriber. It reads
      the SAME env var name (INTEGRITY_REPAIR_ENABLED_ENV) the P2 subscriber already gates on
      — one flag governs both repair surfaces, never two competing switches.
    Why: env flags alone did not activate recall-repair (memory_search's repair_integrity
      param defaulted False + the daemon never passed it). This reader lets memory_search
      resolve that param from the flag when the caller does not pass one explicitly.
    """
    return os.environ.get(INTEGRITY_REPAIR_ENABLED_ENV, "0") == "1"


def decay_enabled() -> bool:
    """True iff the EROSION sweep is enabled (ASTHENOS_INTEGRITY_DECAY_ENABLED).

    What: the live (read-each-call, default OFF) reader the decay_tick call site uses to
      resolve erode_integrity when not explicitly passed. OFF => decay_tick does NOT run the
      content-integrity erosion (the relevance/lifecycle decay is unaffected either way).
    Why: the erosion sweep is INERT by default; this is the GRANULAR switch that turns on the
      slow per-character body erosion independently of repair + freeze.
    """
    return os.environ.get(INTEGRITY_DECAY_ENABLED_ENV, "0") == "1"


def freeze_enabled() -> bool:
    """True iff terminal FREEZE-at-floor is enabled (ASTHENOS_INTEGRITY_FREEZE_ENABLED).

    What: the live (read-each-call, default OFF) reader the decay_tick call site uses to
      resolve terminal_freeze when not explicitly passed. OFF => a below-floor node is NOT
      auto-frozen (it just keeps eroding); the relevance-axis freeze is independent.
    Why: terminal freeze is the most consequential arm (it demotes a node), so it gets its
      OWN flag — decay can run (erode) WITHOUT freeze, and freeze never fires unless both
      decay reaches the floor AND this flag is set.
    """
    return os.environ.get(INTEGRITY_FREEZE_ENABLED_ENV, "0") == "1"

# GENERATIVE_REPAIR_STRENGTH — What: how far a generative reconstruction raises integrity.
# Why: a reconstructed body is NOT byte-faithful (anchor_faithful=false), so it must not
#   claim a full pristine restore. A PARTIAL raise marks it as recovered-but-uncertain —
#   the served content is restored (so the node is readable again) but the integrity score
#   stays below FULL to keep the provenance honest until a faithful repair re-anchors it.
GENERATIVE_REPAIR_STRENGTH = PARTIAL_REPAIR_STRENGTH


def salience_freeze_exempt() -> float:
    """The salience at/above which the integrity floor does NOT auto-freeze a node.

    What: reads the canonical tier.SALIENCE_FREEZE_EXEMPT (the P5 relevance-path exemption)
      so the integrity floor's freeze policy stays CONSISTENT with the relevance path; falls
      back to SALIENCE_FREEZE_EXEMPT_DEFAULT if tier is unavailable. A function (not a module
      import) so integrity never couples to tier at module scope (it RIDES tier).
    Why: Q5a + the P5 exemption — a salience >= this threshold node is NOT auto-frozen by
      EITHER decay axis (a high-salience one-shot persists through the forgetting curve);
      keep the two axes in lock-step if the constant is re-tuned.
    """
    try:
        from . import tier as _tier
        return float(_tier.SALIENCE_FREEZE_EXEMPT)
    except Exception:
        return SALIENCE_FREEZE_EXEMPT_DEFAULT


def generative_enabled() -> bool:
    """True iff the P3 generative-reconstruction fallback is enabled AND available.

    What: requires BOTH the explicit enable flag (INTEGRITY_GENERATIVE_ENABLED_ENV, default
      OFF, read each call so a test/daemon that sets it post-import sees the change) AND the
      local inference backend being available (contradiction.synthesis_enabled — the SAME
      llama-cli/CHIRON gate the judge/synthesis use).
    Why: Q1c/Q4a — generative reconstruction is the only drift path; it stays inert until
      the operator explicitly enables it and the backend is actually present. Either condition
      missing => a SAFE NO-OP (no crash, no fabrication), mirroring the P1/P2 posture.
    """
    if os.environ.get(INTEGRITY_GENERATIVE_ENABLED_ENV, "0") != "1":
        return False
    try:
        from samia.runtime import contradiction as _contra
        return bool(_contra.synthesis_enabled())
    except Exception:
        return False


def _anchors_dir(memory_dir: Path) -> Path:
    """The retained-anchor store (one pristine snapshot per node)."""
    return memory_dir / "biomimetic" / "integrity_anchors"


def _node_id(node_name: str, fm: Optional[dict] = None) -> str:
    """Resolve a stable anchor id for a node.

    Mirrors ia.freeze's choice: prefer the `address` frontmatter, fall back to the
    node's file stem. Keeps anchor file names stable across a rename of the title.
    """
    if fm is not None:
        addr = fm.get("address")
        if addr:
            return str(addr)
    name = node_name
    if name.endswith(".md"):
        name = name[:-3]
    return name


def anchor_path(memory_dir: Path, node_name: str, fm: Optional[dict] = None) -> Path:
    """Absolute path to a node's pristine recovery anchor."""
    return _anchors_dir(memory_dir) / f"{_node_id(node_name, fm)}.txt"


def is_distilled(fm: dict) -> bool:
    """True iff a node's content has been semantically distilled (fm distilled == True).

    What: reads the boolean `distilled` frontmatter marker stamped by the fact-extract
      drain (rem_subscribers._sub_fact_extract) once a frozen source's content is
      semantically covered (>= 1 atom persisted OR all atoms dedup-skipped). Strictly
      `is True` — any other value (absent, False, a stray string) reads as NOT distilled.
    Why: TUNE-2026-06-10 operator decision (c), systems-consolidation gating — the
      episodic trace fades only AFTER the semantic representation forms. A frozen node
      erodes ONLY once this marker is set; an undistilled frozen node still never erodes
      (the integrity_decay_pass walk skips it). The strict `is True` test keeps the gate
      conservative: erosion (which loses served characters) requires an UNAMBIGUOUS
      distilled marker, never a truthy-but-unintended value.
    """
    return fm.get("distilled") is True


def get_integrity(fm: dict) -> float:
    """Read the [0,1] integrity field; a node with no field is pristine (1.0)."""
    try:
        v = float(fm.get("integrity", INTEGRITY_FULL))
    except (TypeError, ValueError):
        return INTEGRITY_FULL
    return min(INTEGRITY_FULL, max(INTEGRITY_NONE, v))


def set_integrity(fm: dict, order: list[str], value: float) -> None:
    """Write the [0,1] integrity field, appending it to `order` if new.

    Clamps to [0,1]. `serialize` emits appended keys at the end, so a new integrity
    field writes cleanly without disturbing existing key order.
    """
    v = min(INTEGRITY_FULL, max(INTEGRITY_NONE, float(value)))
    if "integrity" not in fm and "integrity" not in order:
        order.append("integrity")
    fm["integrity"] = round(v, 6)


def has_anchor(memory_dir: Path, node_name: str, fm: Optional[dict] = None) -> bool:
    """True iff a recoverable pristine anchor exists for this node."""
    return anchor_path(memory_dir, node_name, fm).exists()


def write_anchor(memory_dir: Path, node_name: str, body: str,
                 fm: Optional[dict] = None) -> Path:
    """Capture (or refresh) the pristine body snapshot for a node.

    What: writes the CURRENT (pristine) body to the anchor store. Called when a node
      is first written/seen and on each faithful repair (re-snapshot from the now-
      confirmed-pristine canonical body). The anchor is NEVER eroded in P1 — it is the
      faithful repair source.
    Why: Q1c HYBRID — real erosion of the served body, but a retained anchor so recall
      restores faithfully (byte-exact), not a guess.
    """
    ap = anchor_path(memory_dir, node_name, fm)
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(body, encoding="utf-8")
    return ap


def read_anchor(memory_dir: Path, node_name: str,
                fm: Optional[dict] = None) -> Optional[str]:
    """Return the pristine anchor body, or None if no anchor exists."""
    ap = anchor_path(memory_dir, node_name, fm)
    if not ap.exists():
        return None
    return ap.read_text(encoding="utf-8")


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


def _node_context(memory_dir: Path, node_name: str, fm: dict) -> str:
    """A small context string (title/description/name) for a generative reconstruction.

    What: gathers the surviving non-body signals (the node's name/title/description from
      frontmatter) to give the reconstruction prompt what little context remains when the
      body has eroded and no anchor exists.
    Why: Q4a — the generative fallback reconstructs from the degraded served content + the
      node's CONTEXT; this is that context. Pure read of frontmatter, never mutates.
    """
    bits = []
    for key in ("name", "title", "description"):
        v = str(fm.get(key, "")).strip()
        if v:
            bits.append(f"{key}: {v}")
    return "\n".join(bits)


def _generative_reconstruct(memory_dir: Path, node_name: str, fm: dict,
                            degraded_body: str) -> Optional[str]:
    """Reconstruct a node body from its degraded content + context (P3 last resort).

    What: when NO anchor remains, reuse the SAME local-inference backend the judge uses
      (contradiction.synthesize_node — llama-cli/CHIRON) with a reconstruction prompt over
      the degraded (eroded) body + the node's surviving context. Returns the reconstructed
      body string, or None when generative repair is disabled/unavailable/unparseable.
    Why: Q1c/Q4a — generative reconstruction is the LAST RESORT used ONLY when the anchor
      itself is gone (a pre-anchor node or a deeply-lost anchor). It reuses the existing
      inference entrypoint (no new model loader) and is a SAFE NO-OP (None) when off — the
      caller treats None exactly like the prior anchor-missing no-op.

    SAFETY: gated by generative_enabled() (flag + backend availability). NEVER called while
      an anchor exists (anchor-first always wins — the callers check has_anchor first).
    """
    if not generative_enabled():
        return None
    try:
        from samia.runtime import contradiction as _contra
    except Exception:
        return None
    context = _node_context(memory_dir, node_name, fm)
    # Reuse synthesize_node's two-text contract: (degraded body, surviving context).
    # It returns {"title", "body"} or None (the safe no-op) — we want the body.
    try:
        synth = _contra.synthesize_node(degraded_body or "", context or node_name)
    except Exception:
        return None
    if not synth:
        return None
    body = str(synth.get("body", "")).strip()
    return body or None


def _log_reconsolidation(memory_dir: Path, record: dict) -> None:
    """Append a reconsolidation event to the bio reconsolidation log (jsonl)."""
    log_path = memory_dir / "biomimetic" / "integrity_reconsolidation_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .timestamp import now_utc_iso
        record.setdefault("ts", now_utc_iso())
    except Exception:
        pass
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def recall_repair(memory_dir: Path, node_name: str,
                  strength: float = RECALL_REPAIR_STRENGTH) -> dict:
    """On a GENUINE recall, restore the body byte-exact from the anchor + reset integrity.

    What: the recall-repair trigger (Q3a strongest, Q4a anchor-first). Reads the node,
      restores its served body from the pristine anchor (faithful, byte-exact — NOT a
      guess), resets integrity toward FULL by `strength` (1.0 = full reconsolidation for
      recall), persists the restored node, and logs a reconsolidation event. P1 wires the
      RECALL trigger only; consolidation/reconciliation partial repair is P2.
    Why: "a node missing a bit is easily read + restored just from recalling it." Any
      genuine retrieval that surfaces the node heals it.

    SAFETY: if no anchor exists, this is a no-op (P1 never guesses; generative fallback
      is P3). Fail-soft — never crashes the recall path.

    Returns a small telemetry dict {repaired, node, old_integrity, new_integrity, ...}.
    """
    nodes_dir = memory_dir / "nodes"
    fname = node_name if node_name.endswith(".md") else f"{node_name}.md"
    node_path = nodes_dir / fname
    if not node_path.exists():
        return {"repaired": False, "node": node_name, "skipped": "no-node-file"}

    try:
        fm, order, body = _fm.read_node(node_path)
    except (ValueError, OSError):
        return {"repaired": False, "node": node_name, "skipped": "unreadable"}

    anchor_body = read_anchor(memory_dir, node_name, fm)
    if anchor_body is None:
        # No anchor remains -> anchor-first cannot repair. P3 (Q1c/Q4a): the LAST-RESORT
        # generative reconstruction MAY rebuild the body from the degraded content +
        # context — ONLY when no anchor exists, ONLY when enabled + the backend is
        # available, and marked generative=true / anchor_faithful=false. Off/unavailable
        # -> a SAFE NO-OP, exactly as before.
        return _recall_generative_fallback(memory_dir, node_name, node_path, fm, order,
                                           body, strength)

    old_integrity = get_integrity(fm)
    new_integrity = reconsolidate_integrity(old_integrity, strength)

    # Anchor-first restore: the served body becomes the pristine anchor body (byte-exact).
    restored_body = anchor_body
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, restored_body, integrity_rewrite=True)
    except ValueError:
        # AUD61 frozen/archived protection (or other validation) — do not force-write.
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": "recall",
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(float(strength), 4),
        "anchor_faithful": True,
        "generative": False,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass  # fail-soft: a logging failure must never break the recall path
    return {"repaired": True, **rec}


def _recall_generative_fallback(memory_dir: Path, node_name: str, node_path: Path,
                                fm: dict, order: list[str], body: str,
                                strength: float) -> dict:
    """The P3 no-anchor generative branch of recall_repair (Q1c/Q4a, last resort).

    What: reached ONLY when no anchor exists. If the generative fallback is enabled + the
      backend is available, reconstructs the body from the degraded served content +
      context, raises integrity PARTIALLY (generative is not byte-faithful), persists it,
      and stamps a generative reconsolidation event (generative=true / anchor_faithful=
      false / confabulation_risk=true). Off/unavailable/unparseable -> the same safe
      no-anchor no-op as before (no crash, no fabrication).
    Why: Q4a — generative fallback ONLY when no anchor remains; honest provenance marking.
    """
    reconstructed = _generative_reconstruct(memory_dir, node_name, fm, body)
    if reconstructed is None:
        return {"repaired": False, "node": node_name, "skipped": "no-anchor"}

    old_integrity = get_integrity(fm)
    # Generative is NOT byte-faithful -> a PARTIAL raise, never a full pristine claim.
    new_integrity = reconsolidate_integrity(old_integrity, GENERATIVE_REPAIR_STRENGTH)
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, reconstructed, integrity_rewrite=True)
    except ValueError:
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": "recall",
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(float(GENERATIVE_REPAIR_STRENGTH), 4),
        "anchor_faithful": False,
        "generative": True,
        "confabulation_risk": True,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass
    return {"repaired": True, **rec}


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
    try:
        from . import bio as _bio
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


def partial_repair(memory_dir: Path, node_name: str,
                   strength: float = PARTIAL_REPAIR_STRENGTH,
                   trigger: str = "consolidation") -> dict:
    """PARTIALLY repair a node's integrity (anchor-first), the P2 offline trigger.

    What: the CONSOLIDATION / RECONCILIATION repair (Q3a, PARTIAL). Reads the node,
      restores its served body from the pristine anchor (anchor-first, byte-exact — NOT
      a guess) but raises integrity only PARTIALLY toward FULL (strength < 1.0), persists
      the result, and logs a reconsolidation event tagged with the trigger. Distinct from
      recall_repair (which is FULL, strength 1.0): sleep + reconciliation heal a little of
      what they touch, recall heals fully.
    Why:  Q3a — RECALL strongest (full), CONSOLIDATION + RECONCILIATION partial. Anchor-
      first only — NO generative repair in P2 (that is P3); a node with no anchor is a
      fail-soft no-op here, exactly like recall_repair.

    SAFETY: anchor-first only; no anchor -> no-op (never guesses). Fail-soft — a repair
      error never breaks the consolidation/reconciliation path that called it. The anchor
      is read, NEVER written here (the anchor is the faithful source, untouched).

    Returns a small telemetry dict {repaired, node, old_integrity, new_integrity, ...}.
    """
    s = min(1.0, max(0.0, float(strength)))
    nodes_dir = memory_dir / "nodes"
    fname = node_name if node_name.endswith(".md") else f"{node_name}.md"
    node_path = nodes_dir / fname
    if not node_path.exists():
        return {"repaired": False, "node": node_name, "skipped": "no-node-file"}

    try:
        fm, order, body = _fm.read_node(node_path)
    except (ValueError, OSError):
        return {"repaired": False, "node": node_name, "skipped": "unreadable"}

    anchor_body = read_anchor(memory_dir, node_name, fm)
    if anchor_body is None:
        # No anchor remains -> anchor-first cannot repair. P3 (Q1c/Q4a): the LAST-RESORT
        # generative reconstruction MAY rebuild the body — ONLY when no anchor exists,
        # ONLY when enabled + the backend is available, marked generative=true /
        # anchor_faithful=false. Off/unavailable -> a SAFE NO-OP, exactly as before.
        return _partial_generative_fallback(memory_dir, node_name, node_path, fm, order,
                                            body, trigger)

    old_integrity = get_integrity(fm)
    new_integrity = reconsolidate_integrity(old_integrity, s)

    # Anchor-first restore: the served body becomes the pristine anchor body (byte-exact).
    # The integrity SCORE only rises partially, but the served content is faithful — the
    # HYBRID model stores integrity as a score, so a partial repair restores the body
    # from the anchor and tracks fidelity via the (partially-raised) score.
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, anchor_body, integrity_rewrite=True)
    except ValueError:
        # AUD61 frozen/archived protection (or other validation) — do not force-write.
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": trigger,
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(s, 4),
        "anchor_faithful": True,
        "generative": False,
        "partial": True,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass  # fail-soft: a logging failure must never break the calling path
    return {"repaired": True, **rec}


def _partial_generative_fallback(memory_dir: Path, node_name: str, node_path: Path,
                                 fm: dict, order: list[str], body: str,
                                 trigger: str) -> dict:
    """The P3 no-anchor generative branch of partial_repair (Q1c/Q4a, last resort).

    What: reached ONLY when no anchor exists on a consolidation/reconciliation repair. If
      the generative fallback is enabled + available, reconstructs the body from the
      degraded served content + context, raises integrity PARTIALLY, persists it, and
      stamps a generative reconsolidation event (generative=true / anchor_faithful=false /
      confabulation_risk=true). Off/unavailable/unparseable -> the same safe no-anchor
      no-op (no crash, no fabrication).
    Why: Q4a — generative fallback ONLY when no anchor remains; honest provenance marking.
    """
    reconstructed = _generative_reconstruct(memory_dir, node_name, fm, body)
    if reconstructed is None:
        return {"repaired": False, "node": node_name, "skipped": "no-anchor"}

    old_integrity = get_integrity(fm)
    new_integrity = reconsolidate_integrity(old_integrity, GENERATIVE_REPAIR_STRENGTH)
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, reconstructed, integrity_rewrite=True)
    except ValueError:
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": trigger,
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(float(GENERATIVE_REPAIR_STRENGTH), 4),
        "anchor_faithful": False,
        "generative": True,
        "confabulation_risk": True,
        "partial": True,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass
    return {"repaired": True, **rec}


def ensure_anchor(memory_dir: Path, node_name: str, fm: dict, body: str) -> bool:
    """Capture a pristine anchor for a node IF it does not already have one.

    What: idempotent first-seen anchor capture — snapshots the current body as the
      pristine recovery anchor only when no anchor exists yet (so a later refresh on a
      faithful repair is the only re-snapshot). Returns True iff it wrote a new anchor.
    Why: erosion is gated on an anchor being present; this is the "node is written/first
      seen" capture point so the very first erosion already has a recoverable source.
    """
    if has_anchor(memory_dir, node_name, fm):
        return False
    write_anchor(memory_dir, node_name, body, fm)
    return True


def capture_on_write(memory_dir: Path, node_name: str, fm: dict, body: str) -> dict:
    """Capture/REFRESH the pristine anchor on a GENUINE node write (P2 capture hook).

    What: the anchor-capture-on-write entrypoint, called ONLY from the genuine node-write
      path (memory_write_node / the capture hook / an ia write of REAL operator/agent
      content). The just-written `body` IS the pristine version, so this REFRESHES the
      anchor to it unconditionally — a fresh node gains an anchor and a genuine re-write
      updates the anchor to the new pristine body. Returns a small {captured, refreshed,
      anchor} telemetry dict. Fail-soft — an anchor-write failure never breaks the write.
    Why:  the P1 caveat: nodes only erode once they HAVE an anchor, and P1 did NOT auto-
      capture. This is the engagement gate — wiring it into the genuine write path is what
      makes the whole second-axis mechanism actually fire (a node can thereafter erode and
      be faithfully repaired from this snapshot).

    CRITICAL SAFETY: this is the ONLY anchor-write entrypoint outside a faithful repair —
      it MUST be called only with the PRISTINE just-written body, NEVER from
      integrity_decay_pass / erode / the erosion-persistence path. Capturing an eroded
      served body would clobber the anchor with degraded content and permanently defeat
      repair (data loss). The erosion sweep persists the eroded body via a path that
      leaves the anchor untouched; this lives only at the genuine-write entrypoints.
    """
    existed = has_anchor(memory_dir, node_name, fm)
    try:
        write_anchor(memory_dir, node_name, body, fm)
    except OSError as e:
        return {"captured": False, "refreshed": False, "error": str(e)}
    return {"captured": not existed, "refreshed": existed,
            "anchor": str(anchor_path(memory_dir, node_name, fm))}


def capture_on_genuine_write(memory_dir: Path, node_name: str, fm: dict,
                             order: list[str], body: str) -> dict[str, Any]:
    """Anchor-capture on a GENUINE node write — the universal write_node seam.

    What: FEAT-2026-06-08. If the body is UNCHANGED from the current anchor, NO-OP
      (Q3b skip-unchanged — avoids re-anchoring the offload path's frequent metadata-
      only re-saves). Otherwise REFRESH the anchor to the new pristine body and, if the
      node was eroded (integrity < FULL), RESET integrity to FULL (Q2a — a genuine
      rewrite is the new pristine baseline, so the forgetting curve restarts from full).
      Mutates fm/order in place when it resets; the caller (write_node) serializes after.
      Returns small telemetry; fail-soft is the caller's responsibility.
    Why: capture_on_write fired ONLY on the MCP write_node op; session-offloads + other
      internal writers bypassed it via frontmatter.write_node, leaving nodes anchor-less
      and therefore un-erodable (erode is anchor-gated). Wiring this into write_node makes
      EVERY genuine writer anchor automatically.
    CRITICAL SAFETY (mirrors capture_on_write): a GENUINE-write entrypoint ONLY. NEVER
      call from erode / integrity_decay_pass / the erosion-persistence path — capturing an
      eroded served body would clobber the pristine anchor and permanently defeat repair.
      The write_node seam enforces this via the integrity_rewrite gate (only integrity_
      rewrite==False writes reach here).
    DEFENSE-IN-DEPTH: never anchor from a body that carries the EROSION_SENTINEL. An
      eroded/served body would clobber the pristine recovery source if anchored, and
      resetting integrity from it would hide real erosion. Only sentinel-free (genuinely
      pristine) content may refresh the anchor — this guards against ANY path that re-saves
      an eroded served body via write_node, not just the integrity_rewrite-marked ones. (A
      brand-new node whose genuine content legitimately contains '·' is anchored by the
      backstop sweep instead; a negligible edge.)
    """
    if EROSION_SENTINEL in body:
        return {"captured": False, "skipped": "eroded-body"}
    existing = read_anchor(memory_dir, node_name, fm)
    if existing is not None and existing == body:
        return {"captured": False, "skipped": "unchanged"}
    try:
        write_anchor(memory_dir, node_name, body, fm)
    except OSError as e:
        return {"captured": False, "skipped": "anchor-write-failed", "error": str(e)}
    reset = False
    if get_integrity(fm) < INTEGRITY_FULL:
        set_integrity(fm, order, INTEGRITY_FULL)
        reset = True
    return {"captured": True, "refreshed": existing is not None, "integrity_reset": reset}


def backfill_anchors_pass(memory_dir: Path, cursor: int = 0,
                          budget: int = 200) -> dict[str, Any]:
    """Backstop sweep — anchor any un-anchored node (FEAT-2026-06-08 Q4a).

    What: cursor-walks a budgeted slice of nodes/ and ensure_anchor()s each UN-anchored
      node (capture-IF-MISSING only — NEVER refresh), checkpointing the cursor. Returns
      {captured, processed, cursor, work_remaining, total}. Because it only captures when
      no anchor exists, it can NEVER clobber an eroded node's pristine anchor, so it is safe
      to run continuously over the whole corpus.
    Why: write-path capture (capture_on_genuine_write at the write_node seam) is the primary
      source of truth; this low-cadence sweep is the BACKSTOP that catches any node a future
      write path, a restore/thaw, or a bulk import leaves anchor-less. A no-op at full
      coverage. Productionizes tools/backfill_integrity_anchors_2026_06_08.py.
    """
    mem = Path(memory_dir)
    node_files = sorted((mem / "nodes").glob("*.md"))
    total = len(node_files)
    start = int(cursor) if 0 <= int(cursor) < total else 0
    end = min(start + max(0, int(budget)), total)
    captured = 0
    for p in node_files[start:end]:
        try:
            fm, _order, body = _fm.read_node(p)
        except Exception:
            continue  # fail-soft: a parse error never aborts the sweep
        try:
            if ensure_anchor(mem, p.stem, fm, body):
                captured += 1
        except OSError:
            continue
    work_remaining = end < total
    return {"captured": captured, "processed": end - start,
            "cursor": end if work_remaining else 0,
            "work_remaining": work_remaining, "total": total}


def anchor_backfill_tick(memory_dir: Path) -> dict[str, Any]:
    """idle_pulse subscriber entry (FEAT-2026-06-08 Q4a) — full-corpus backstop sweep.

    What: loops backfill_anchors_pass over the WHOLE node list per (daily) tick so the
      backstop achieves complete coverage rather than a single budgeted slice. Cheap —
      every already-anchored node is just a has_anchor() stat. Returns aggregate telemetry.
      Subscriber signature fn(mem).
    Why: the daily backstop must catch every straggler, not 1/N of them; the per-pass budget
      only bounds memory/loop length, not coverage. A no-op at full coverage.
    """
    total_captured = 0
    cursor = 0
    passes = 0
    while True:
        res = backfill_anchors_pass(memory_dir, cursor=cursor, budget=500)
        total_captured += res["captured"]
        passes += 1
        if not res["work_remaining"] or passes > 1000:  # 1000 = runaway guard
            break
        cursor = res["cursor"]
    return {"captured": total_captured, "passes": passes}


def integrity_decay_pass(memory_dir: Path, dry: bool = True,
                         today: Optional[str] = None,
                         only_with_anchor: bool = True,
                         terminal_freeze: bool = False) -> list[dict]:
    """The content-integrity erosion SWEEP — the second axis's continuous pass.

    What: walks nodes/*.md and applies one slow per-character erosion pass to each
      eligible node (lowering integrity + eroding the served body), reading the SAME
      last_access/tier/salience the relevance decay already reads. Skips target_state
      frozen/archived nodes (exactly as the relevance step does) and ANY node without a
      recoverable anchor. A tier=="frozen" node is skipped UNLESS it is DISTILLED
      (TUNE-2026-06-10 decision c, systems-consolidation gating: the episodic trace
      fades only AFTER the semantic representation forms — is_distilled(fm) is the
      gate; a distilled frozen node erodes at TIER_EROSION_FACTOR["frozen"]=0.25).
      Returns one record per eroded node. This is the entry the existing decay/idle path
      can invoke ALONGSIDE tier.decay_pass (both axes, ungated, wake+REM).

      P3 TERMINAL FREEZE-AT-FLOOR (Q5a): when `terminal_freeze=True` (and not dry), a node
      whose new integrity falls below INTEGRITY_FLOOR and was NOT repaired this tick is
      routed into the existing REVERSIBLE ia.freeze (demotion-to-frozen, restorable via
      ia.thaw + a later recall reconsolidation), NOT deleted — UNLESS its salience clears
      the salience-exemption threshold (salience_freeze_exempt(), consistent with the
      relevance path's P5 freeze exemption), in which case it stays resident. The freeze is
      DEFERRED to after the walk (ia.freeze removes node files; freezing mid-walk would
      mutate the directory we are iterating), mirroring tier.decay_pass's freeze_queue.
    Why: Q6a — a NEW second axis that RIDES the same continuous tick as relevance-decay,
      without modifying tier.step_relevance / tier.decay_pass; Q5a — the integrity floor is
      a SECOND trigger feeding the SAME reversible freeze path the relevance floor uses, and
      it honors the SAME salience exemption so the two axes' freeze policy stays consistent.

    PRODUCE-ONLY / INERT BY DEFAULT: `dry=True` by default — it computes + reports the
      erosion WITHOUT writing. The caller must explicitly pass dry=False to apply it, and
      `terminal_freeze` is OFF by default (the floor never freezes until opted-in). It
      starts NO scheduler/thread/timer (a plain function the existing pass invokes). It
      NEVER erodes a node without a recoverable anchor (`only_with_anchor`).

    NOTE: this does NOT modify the relevance/tier axis. The two compose, not collide.
    """
    from datetime import date as _date

    nodes_dir = memory_dir / "nodes"
    today_iso = today or _date.today().isoformat()
    out: list[dict] = []
    if not nodes_dir.exists():
        return out

    def _days_since(last_iso: str) -> int:
        if not last_iso:
            return 9999
        try:
            last = _date.fromisoformat(str(last_iso))
            cur = _date.fromisoformat(today_iso)
            return max(0, (cur - last).days)
        except (ValueError, TypeError):
            return 9999

    # Deferred terminal-freeze queue — ia.freeze removes node files, so (exactly like
    # tier.decay_pass's freeze_queue) we freeze AFTER the walk to avoid mutating the
    # directory mid-iteration. Each entry is {node, integrity, salience}.
    freeze_queue: list[dict] = []
    exempt_threshold = salience_freeze_exempt() if terminal_freeze else None

    for md in sorted(nodes_dir.glob("*.md")):
        try:
            fm, order, body = _fm.read_node(md)
        except (ValueError, OSError):
            continue

        # Skip target_state frozen/archived nodes — exactly as the relevance step does.
        # (target_state lifecycle freeze/archive is a HARD skip on EITHER axis,
        # independent of the distillation gate below — those node files are immutable.)
        ts = str(fm.get("target_state", "live")).lower()
        if ts in ("frozen", "archived"):
            continue
        node_tier = str(fm.get("tier", DEFAULT_TIER)).lower()
        # TUNE-2026-06-10 operator decision (c), systems-consolidation (distillation)
        # gating: a tier=="frozen" node erodes ONLY once its content is DISTILLED (the
        # semantic representation has formed — the fact-extract drain stamped
        # distilled:true). An UNDISTILLED frozen node still NEVER erodes (unchanged
        # behavior — its episodic trace stays pristine until the gist exists); a
        # DISTILLED frozen node is ELIGIBLE to erode at TIER_EROSION_FACTOR["frozen"]
        # (0.25, slowest), with the normal anchor-gating + salience/recency modulation
        # below still applying. The episodic trace fades only AFTER the semantic
        # representation forms.
        if node_tier == "frozen" and not is_distilled(fm):
            continue

        node_name = md.stem

        # NEVER erode without a recoverable anchor (no irrecoverable loss).
        if only_with_anchor and not has_anchor(memory_dir, node_name, fm):
            continue

        last = str(fm.get("last_access", ""))
        days = 0 if last == today_iso else _days_since(last)
        # Q2a salience modulation — read the LIVE salience signal (bio.compute_salience,
        # read-only) so a genuinely high-salience node erodes slower; fall back to the
        # maintained frontmatter field, then to neutral 0.0 (graceful + fail-soft).
        salience = live_salience(memory_dir, node_name, fm)

        old_integrity = get_integrity(fm)
        new_body, new_integrity, n_eroded = erode(
            memory_dir, node_name, fm, order, body,
            days_since_recall=days, tier=node_tier, salience=salience,
        )
        if n_eroded <= 0:
            continue

        rec = {
            "node": node_name,
            "old_integrity": round(old_integrity, 6),
            "new_integrity": round(new_integrity, 6),
            "n_eroded": n_eroded,
            "tier": node_tier,
            "days_since_recall": days,
        }

        # P3 terminal freeze-at-floor (Q5a): a node eroded below the readable floor this
        # tick (and not repaired) terminally freezes — UNLESS salience-exempt, consistent
        # with the relevance path. The erosion was NOT a repair, so crossing the floor here
        # is the un-repaired terminal. Salience-exempt nodes stay resident (surface/remain).
        if terminal_freeze and new_integrity < INTEGRITY_FLOOR:
            if salience >= float(exempt_threshold):
                rec["freeze_exempt"] = True
                rec["salience"] = round(float(salience), 4)
            else:
                rec["terminal_freeze"] = True
                rec["salience"] = round(float(salience), 4)
                freeze_queue.append({
                    "node": node_name,
                    "integrity": round(new_integrity, 6),
                    "salience": round(float(salience), 4),
                })

        out.append(rec)

        if not dry:
            if not md.exists():
                continue
            _fm.write_node(md, fm, order, new_body, integrity_rewrite=True)

    # Deferred terminal freeze — after the walk + after the eroded bodies are persisted.
    # Reuse the existing REVERSIBLE ia.freeze (restorable via ia.thaw); NEVER deletion.
    if not dry and terminal_freeze and freeze_queue:
        try:
            from . import ia as _ia
        except ImportError as e:
            print(f"[integrity] terminal-freeze unavailable (ia import failed): {e}")
            return out
        for t in freeze_queue:
            md = nodes_dir / f"{t['node']}.md"
            if not md.exists():
                continue
            try:
                _ia.freeze(memory_dir, t["node"])
                t["frozen"] = True
            except SystemExit as e:
                # ia.freeze sys.exits on a hot node ("demote first"); the integrity floor
                # must never crash the sweep — record + skip (the relevance axis will
                # demote it eventually, then a later integrity pass can freeze it).
                t["freeze_error"] = str(e)
            except Exception as e:
                t["freeze_error"] = str(e)
                print(f"[integrity] terminal-freeze FAILED for {t['node']}: {e}")
            try:
                _log_reconsolidation(memory_dir, {
                    "event": "terminal_freeze",
                    "trigger": "integrity_floor",
                    "node": t["node"],
                    "integrity": t.get("integrity"),
                    "salience": t.get("salience"),
                    "floor": INTEGRITY_FLOOR,
                    "frozen": t.get("frozen", False),
                    "freeze_error": t.get("freeze_error"),
                })
            except Exception:
                pass  # fail-soft: a logging failure must never break the sweep

    return out


# CONSOLIDATION_REPAIR_BUDGET — What: max nodes a single consolidation-repair REM slice
#   touches (cursor-tracked across cycles).
# Why: incremental — a large corpus is repaired a budgeted slice at a time so one REM
#   cycle never stalls; the cursor resumes the next cycle (mirrors the other REM ops).
CONSOLIDATION_REPAIR_BUDGET = 50


def consolidation_repair_pass(memory_dir: Path,
                              budget: int = CONSOLIDATION_REPAIR_BUDGET,
                              cursor: Optional[int] = None,
                              strength: float = PARTIAL_REPAIR_STRENGTH,
                              today: Optional[str] = None) -> dict:
    """The CONSOLIDATION repair pass — sleep heals what it consolidates (P2, Q3a partial).

    What: walks a cursor-tracked slice (<= ``budget`` nodes) of nodes/ and PARTIALLY
      repairs the integrity of each ERODED node it touches (partial_repair, anchor-first,
      strength < 1.0). Skips frozen/archived nodes and any node with no anchor (anchor-
      first only). Cursor is an int index over the sorted node list; it advances by the
      slice size and WRAPS at the end (a full pass spans many REM cycles). Returns a dict
      with the cursor, counts, and a work_remaining signal for the REM driver.
    Why:  Q3a — CONSOLIDATION is a PARTIAL repair trigger: a REM consolidation pass heals
      a little of the integrity of the nodes it consolidates (distinct from RECALL, which
      heals fully). Anchor-first only — NO generative repair (P3). Incremental + cursor-
      tracked so it fits the REM offline-op contract.

    PRODUCE-ONLY: a plain function (no scheduler/thread/timer). Its REM gate + enable flag
      live at the REM-subscriber wiring (rem_subscribers); this is the pure work fn. Only
      ERODED nodes (integrity < FULL) are repaired — a pristine node is skipped (nothing
      to heal), so a fresh corpus is a cheap no-op.
    """
    from datetime import date as _date

    nodes_dir = memory_dir / "nodes"
    out: dict = {"repaired": 0, "touched": 0, "processed": 0, "total": 0,
                 "made_progress": False, "work_remaining": False}
    if not nodes_dir.exists():
        return out

    node_ids = sorted(p.stem for p in nodes_dir.glob("*.md"))
    total = len(node_ids)
    out["total"] = total
    if total == 0:
        return out

    start = int(cursor or 0)
    if start < 0 or start >= total:
        start = 0
    cap = max(0, int(budget))
    end = min(start + cap, total)
    slice_ids = node_ids[start:end]
    out["processed"] = len(slice_ids)

    repaired = touched = 0
    for node_name in slice_ids:
        md = nodes_dir / f"{node_name}.md"
        if not md.exists():
            continue
        try:
            fm, _order, _body = _fm.read_node(md)
        except (ValueError, OSError):
            continue
        # Skip frozen/archived — exactly as the erosion sweep + relevance step do.
        ts = str(fm.get("target_state", "live")).lower()
        if ts in ("frozen", "archived"):
            continue
        if str(fm.get("tier", DEFAULT_TIER)).lower() == "frozen":
            continue
        # Anchor-first only; nothing to repair on a node with no anchor.
        if not has_anchor(memory_dir, node_name, fm):
            continue
        # Only heal an ERODED node — a pristine node has nothing to consolidate-repair.
        if get_integrity(fm) >= INTEGRITY_FULL:
            continue
        touched += 1
        res = partial_repair(memory_dir, node_name, strength=strength,
                             trigger="consolidation")
        if res.get("repaired"):
            repaired += 1

    new_index = end if end < total else 0
    wrapped = end >= total
    out.update({
        "repaired": repaired, "touched": touched,
        "made_progress": bool(slice_ids),
        "cursor": new_index, "wrapped": wrapped,
        "work_remaining": not wrapped,
    })
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.integrity
# Phase:      FEAT-2026-06-07 granular-recall-repaired-decay — Phases P1 + P2 + P3.
#             P1: integrity field + anchor + slow per-char erosion + masked read
#                 + recall-repair (FULL).
#             P2: anchor-capture-on-write (capture_on_write — the engagement gate
#                 P1 deferred), CONSOLIDATION/RECONCILIATION PARTIAL repair
#                 (partial_repair, strength < 1.0, anchor-first), consolidation
#                 repair REM pass (consolidation_repair_pass — cursor-tracked,
#                 incremental), and LIVE-salience erosion modulation
#                 (live_salience → erosion_rate).
#             P3: TERMINAL FREEZE-AT-FLOOR (Q5a) — integrity_decay_pass(
#                 terminal_freeze=True) routes a node eroded below INTEGRITY_FLOOR
#                 (without repair) into the existing REVERSIBLE ia.freeze (restorable
#                 via ia.thaw), honoring the SAME salience exemption the relevance
#                 path uses (salience_freeze_exempt() → tier.SALIENCE_FREEZE_EXEMPT);
#                 GENERATIVE-RECONSTRUCTION FALLBACK (Q1c/Q4a) — recall_repair /
#                 partial_repair, when NO anchor remains, MAY reconstruct the body via
#                 contradiction.synthesize_node (the SAME llama-cli/CHIRON backend the
#                 judge uses), GATED behind ASTHENOS_INTEGRITY_GENERATIVE_ENABLED
#                 (default OFF) + synthesis availability, marked generative=true /
#                 anchor_faithful=false / confabulation_risk=true; NEVER runs while an
#                 anchor exists (anchor-first always wins).
# Layer:      core (pure library, no daemon dependency)
# Stability:  v1.3.1 — P1+P2+P3 second decay axis (content fidelity), composes alongside
#             the relevance/lifecycle decay in tier.py (Q6a, layer-don't-replace).
#             tier.step_relevance / tier.decay_pass are NOT touched (P3 reads, never
#             modifies, tier.SALIENCE_FREEZE_EXEMPT for freeze-policy consistency).
#             TUNE-2026-06-10 (decision c, systems-consolidation gating): a tier=="frozen"
#             node now erodes — but ONLY once DISTILLED (is_distilled(fm), the
#             fact-extract drain's distilled:true marker), at TIER_EROSION_FACTOR
#             ["frozen"]=0.25 (slowest). An UNDISTILLED frozen node is byte-identically
#             skipped (unchanged); hot/warm/cold are byte-identical. The episodic trace
#             fades only AFTER the semantic representation forms.
#             G1-2026-06-11 (operator choice 1a, CLS per-type erosion override): a
#             type:semantic node erodes at SEMANTIC_EROSION_FACTOR (0.25, the
#             hot/frozen permanence rate) IN PLACE OF its tier factor, REGARDLESS of
#             tier — it is the most-permanence class, but still fades (recency/salience
#             modulation + anchor gate still apply). Non-semantic nodes are byte-identical
#             (erosion_rate's is_semantic param defaults False; erode() derives it from fm).
# ErrorModel: erode()/integrity_decay_pass() NEVER erode without a recoverable
#             anchor (no data loss); recall_repair / partial_repair fail soft (no-op
#             on missing node/write-reject, never crash the calling path); the erosion
#             sweep is dry (inert) by default and terminal_freeze is OFF by default
#             (the floor never freezes until opted-in); the floor freeze is DEFERRED
#             post-walk + reuses the REVERSIBLE ia.freeze (demotion, NOT deletion) and
#             swallows ia.freeze's hot-node SystemExit; the GENERATIVE fallback is
#             double-gated (flag + backend) + a SAFE NO-OP when off/unavailable (no
#             crash, no fabrication) + marked generative/anchor_faithful=false +
#             NEVER runs while an anchor exists; capture_on_write is the ONLY non-repair
#             anchor-write entrypoint (genuine write path only — no anchor clobber).
# Depends:    json, os, random, math, datetime, pathlib (stdlib).
#             samia.core.frontmatter (read/write/serialize). samia.core.timestamp
#             (optional, event stamps). samia.core.bio (compute_salience, read-only,
#             for the live-salience erosion term — fail-soft to the static field).
#             samia.core.tier (P3, read-only — SALIENCE_FREEZE_EXEMPT, function-local
#             import, never modified). samia.core.ia (P3 terminal freeze, function-local
#             import). samia.runtime.contradiction (P3 generative fallback — reuses
#             synthesis_enabled/synthesize_node, function-local import).
#             Reads salience/tier/last_access frontmatter.
# Exposes:    is_distilled, get_integrity, set_integrity, anchor_path, has_anchor,
#             write_anchor, read_anchor, ensure_anchor, capture_on_write, erosion_rate,
#             live_salience, erode, mask_read, reconsolidate_integrity,
#             salience_freeze_exempt, generative_enabled, repair_enabled,
#             decay_enabled, freeze_enabled, recall_repair, partial_repair,
#             integrity_decay_pass, consolidation_repair_pass.
#             Constants: INTEGRITY_FLOOR, INTEGRITY_GENERATIVE_ENABLED_ENV,
#             INTEGRITY_REPAIR_ENABLED_ENV, INTEGRITY_DECAY_ENABLED_ENV,
#             INTEGRITY_FREEZE_ENABLED_ENV, GENERATIVE_REPAIR_STRENGTH,
#             SALIENCE_FREEZE_EXEMPT_DEFAULT, TIER_EROSION_FACTOR,
#             SEMANTIC_EROSION_FACTOR.
# ACTIVATION: repair_enabled()/decay_enabled()/freeze_enabled() are the GRANULAR live
#             env readers (default OFF) the daemon call sites use to resolve the
#             repair_integrity / erode_integrity / terminal_freeze params when not
#             explicitly passed. repair_enabled() reuses the SAME flag name the P2
#             consolidation-repair REM subscriber reads (one flag, two repair surfaces).
# Lines:      ~620
# --------------------------------------------------------------------------
