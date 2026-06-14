"""samia.core.integrity.config — the package leaf: constants, flag readers, deps.

Layer 1 (Owns / Depends):
    Owns:    every module-level constant the second decay axis reads (the erosion
             rate + tier/semantic/recency/salience modulation knobs, the integrity
             endpoints + floor, the repair strengths, the erosion sentinel, the
             three granular activation-flag env-var names + the generative flag),
             the LIVE env-flag readers (repair_enabled / decay_enabled /
             freeze_enabled / generative_enabled) and the tier-coupled
             salience-exemption reader (salience_freeze_exempt).  Re-exports the
             single sibling-shared dependency module (_fm) so the carve imports it
             THROUGH one owner instead of each submodule re-importing frontmatter.
    Depends: json/os/random/pathlib/typing (stdlib).  samia.core.frontmatter
             (re-exported as _fm — the canonical node read/serialize seam).
             samia.core.tier (P3 salience-exemption read, function-LOCAL so the
             module never couples to tier at import scope — integrity RIDES tier).
             samia.runtime.contradiction (generative-availability gate, function-
             LOCAL — the same llama-cli/CHIRON backend the judge uses).

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — every sibling submodule
          (anchors / erosion / repair / passes) imports its constants + the _fm
          dependency from here, so the tunable surface lives in one place and is
          never duplicated.  The flag readers live here (not in passes) because
          they are pure env reads that gate every arm and several submodules
          consult them; salience_freeze_exempt() / generative_enabled() ride the
          same flag-reader pattern.
    Why:  splitting the 1342-line monolith by responsibility (anchor store /
          erosion math / repair triggers / corpus sweeps) leaves a shared base of
          constants + flag primitives all four need; concentrating them here keeps
          the bars single-sourced and the import graph acyclic (config depends on
          nothing in the package, only stdlib + re-exported frontmatter).
"""

from __future__ import annotations

import json  # noqa: F401  (re-exported — part of the public surface)
import os
import random  # noqa: F401  (re-exported — part of the public surface)
from pathlib import Path
from typing import Optional

from .. import frontmatter as _fm

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
    # Function-LOCAL import — integrity RIDES tier (reads its constant) but must never
    # couple to it at module scope; a lazy read keeps the two in lock-step on a re-tune
    # without forming an import edge.
    try:
        from .. import tier as _tier
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
    # Function-LOCAL import — the runtime contradiction backend is a heavy/runtime dep;
    # keeping it off the package import path avoids pulling the inference stack in on a
    # plain `import samia.core.integrity`.
    try:
        from samia.runtime import contradiction as _contra
        return bool(_contra.synthesis_enabled())
    except Exception:
        return False


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.integrity.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.integrity monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       shared leaf of the integrity package — the erosion/repair/floor/
#             modulation constants, the three granular activation-flag env names +
#             the generative flag, the live env-flag readers (repair_enabled /
#             decay_enabled / freeze_enabled / generative_enabled), the tier-coupled
#             salience_freeze_exempt reader, and the re-exported _fm dependency.
# Stability:  stable — pure constants + side-effect-free env-flag readers; the carve
#             changed no value (every bar/flag/name byte-identical to the monolith).
# ErrorModel: none on the constants; the flag readers are plain env reads;
#             salience_freeze_exempt / generative_enabled fail-soft (the tier /
#             contradiction reads are function-local + try-guarded -> default/False).
# Depends:    json, os, random, pathlib, typing (stdlib). samia.core.frontmatter
#             (re-exported as _fm). samia.core.tier + samia.runtime.contradiction
#             (function-LOCAL reads only — never an import-scope edge).
# Exposes:    every public constant (INTEGRITY_*/BASE_EROSION_RATE/TIER_EROSION_FACTOR/
#             SEMANTIC_EROSION_FACTOR/RECENCY_*/SALIENCE_*/RECALL_/PARTIAL_/GENERATIVE_
#             REPAIR_STRENGTH/EROSION_SENTINEL/DEFAULT_TIER + the *_ENV names), the
#             flag readers, salience_freeze_exempt, generative_enabled, _fm, and the
#             re-exported json/os/random/Path/Optional.
# Lines:      289
# --------------------------------------------------------------------------
