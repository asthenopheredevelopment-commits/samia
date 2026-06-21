"""samia.core.bio.balancing — Epiphanies v3 episodic-binding model (offline, pure).

Layer 1 (Owns / Depends):
    Owns:    the Epiphanies v3 episodic-binding model. The QUALIFICATION half is unchanged:
             the per-occasion predicate score = S_sal*D*sel*P >= BAR, the selectivity hub
             penalty (_sel), the materialization floor, the familiarity-decaying binding
             salience (composite_salience). FEAT-2026-06-18 replaces the discrete accrual with
             a CONTINUOUS spaced-repetition consolidation curve: segment_sittings now groups raw
             co-recalls into 30-min same-OCCASION units (carrying absolute timestamps; the 40-min
             sitting chop retired); accrue() deposits S += s_gap(gap)*c_count(n) per qualifying
             occasion (s_gap, c_count), cycles S's decay exp(-dt/tau_d) to consolidation-now, and
             runs an SM-2-transcribed run/streak (_sm2_step; lapse-reset below EPI_RUN_FLOOR;
             tau_d grows with spaced reinforcement). The promotion gate (is_promotable) keys on
             the continuous S >= EPI_PROMOTE_S (3.0) AND w >= bar. DUAL AXIS: the integer cg
             (qualifying-occasion count) is kept as the VETO axis alongside the float S. PURE: no
             IO, no wiring — accrue() takes occasions + an injected salience function. The curve
             consts are the calibration-harness-SEARCH-proven interior (2026-06-18).
    Depends: .config (HEBB_EMA_ALPHA / HEBB_DECAY / HEBB_PROMOTION / HEBB_PROMOTE_REPEATS /
             the SALIENCE_* weights / _dt). No live-store contact here; the consolidation
             wiring (reading the archive, calling the live salience source, writing the new
             accounting, env-tuning the curve) is a SEPARATE, FLAG-GATED step (epiphanies.py)
             so this module can never disturb the running store. Operator Decision A (2026-06-16):
             a non-hub pair co-recalled with sustained reinforcement promotes; surprise-driven +
             hub-mechanical junk are blocked by familiarity decay + sel. Promotion-provenance +
             correction is the Phase-6 safety net.

Layer 2 (What / Why):
    What: a faithful, offline re-statement of the calibration harness MODEL v3 (continuous)
          against REAL co-activation records + a real salience source. accrue() reproduces the
          harness's simulate() exactly, so the build-time live-parity test (test_balancing_parity)
          runs the harness fixtures through THIS code and reproduces the harness pass/fail.
    Why:  the discrete sitting-count starved promotion (cg <= #distinct sittings). The continuous
          curve lets a sustained consecutive run AND a tau_d-matched spaced run reach S>=3.0 while
          a too-sparse pair / transient decays out. Keeping it PURE + parity-tested means a wiring
          that silently drops the score-gate / mat-floor / sel / familiarity-decay / the curve
          fails the parity test rather than shipping a starved (or inflated) graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import (
    _dt,
    HEBB_EMA_ALPHA,
    HEBB_DECAY,
    HEBB_PROMOTION,
    HEBB_PROMOTE_REPEATS,
    SALIENCE_W_SURPRISE,
    SALIENCE_W_CONTRADICTION,
    SALIENCE_W_REPETITION,
    SALIENCE_REPETITION_SATURATION,
    SALIENCE_TAG_VALUE,
)

# --- Calibration-harness-proven feasible interior tuple (FINDINGS.md, 2026-06-16) ---
# What: the eight-hard-fixture-feasible constant point (median of the 5.22% feasible set),
#   stress-survived. Why: these are the only free constants of the v3 QUALIFICATION model
#   (the score = S*D*sel*P gate); pinning the harness-proven tuple here is what makes the live
#   module inherit the proof. The CONSOLIDATION layer below (S / tau_d / run) is the new
#   FEAT-2026-06-18 continuous spaced-repetition curve — it changes how a QUALIFYING occasion
#   credits strength, NOT which occasions qualify (the gate is untouched).
EPI_BAR = 0.073            # per-OCCASION qualification threshold on score = S*D*sel*P
EPI_D_SAT = 2.99           # dwell saturation (co-activations-in-occasion for full dwell)
EPI_C_SEL = 0.59           # selectivity penalty steepness on log(1+deg_a*deg_b)
EPI_SEL_FLOOR = 0.27       # selectivity floor (a real hub-connected pair is not over-suppressed)
EPI_SURPRISE_FAM = 0.45    # familiarity base: surprise_eff = surprise0 * fam^(prior recalls)
EPI_P_COHORT = 1.19        # provenance boost for a cohort-validated edge
EPI_P_OP = 1.30            # provenance boost for an operator-tagged edge
EPI_MAT_FLOOR = 0.099      # materialization floor: below this FIRST score, never stored

# --- FEAT-2026-06-18 — continuous spaced-repetition CONSOLIDATION curve (the new model) ---
# What: the free constants of the continuous S-deposit + cycling-decay + SM-2 run curve that
#   replaces the discrete `cg += 1` per sitting (proposal §2). Each is env-overridable at the
#   wiring layer (epiphanies.py via segment+accrue plumbing), defaults proven feasible by the
#   re-run of the calibration-harness SEARCH (calibration_harness.search, 2026-06-18).
# Why: the discrete sitting-count starved promotion (cg <= #distinct sittings; a multi-day
#   open session = 1 sitting forever). The continuous curve lets a genuine within-session burst
#   AND a multi-day run both reach the bar, scaling strength by how far reinforcement spreads.

# OCCASION_GAP_S — the ONE remaining time boundary (proposal §4). Raw co-recalls inside this
# window collapse into one same-occasion unit (deduped -> c_count); the gap BETWEEN consecutive
# occasions drives s_gap. 30 min = 2x the ~15-min longest turn (the dedup floor). The old 40-min
# EPI_SITTING_GAP_S coarse chop is RETIRED — s_gap handles every gap continuously.
EPI_OCCASION_GAP_S = 1800

# s_gap(g) = 1 - exp(-g/EPI_TAU_G_S) for g < 24h, LOCKED to exactly 1.0 for g >= 24h.
# tau_g ~ 8h -> the gap-strength approaches ~0.95 by 24h^- then locks (crossing the session
# boundary = full cross-session credit). Tunes how much a within-day spread is worth vs a
# cross-session reinforcement.
EPI_TAU_G_S = 28800.0          # gap saturation time constant (~8h)
EPI_LOCK_GAP_S = 86400.0       # >= this wall-clock gap => s_gap locks to 1.0 (24h, cross-session)

# c_count(n) = 1 - exp(-n/EPI_N_SAT) — the within-occasion recall-count gradient. n is the
# number of co-recalls of the pair inside the occasion (the old dwell m). Saturating so one
# dense burst cannot run away; n_sat sets where the count credit flattens. 1.2 is the
# harness-SEARCH-proven interior (2026-06-18 re-run, 925/12000 = 7.7% feasible over all 11
# fixtures): low enough that a sustained 4-day consecutive run clears S>=3.0 (heal S=3.12),
# high enough that a single occasion cannot (a within-day burst saturates ~2.x).
EPI_N_SAT = 1.2                # recall-count saturation inside an occasion

# Cycling decay: S decays exp(-dt/tau_d) toward consolidation-NOW. tau_d is the per-edge
# stability; it GROWS with sustained spaced reinforcement (SM-2). The fresh-edge base is the
# DECISION D1 deviation: the proposal stated ~24h, but the harness SEARCH (2026-06-18) found
# ~2 days is the robust interior — at 24h base the spaced-success fixture sits on the S=3.0
# knife-edge (3.01); 2 days lifts it to 3.46 while a 14h-shorter one-off still decays out by
# the next day. So a one-off decays away within ~2 days, a daily run sustains.
EPI_TAU_D_BASE_S = 172800.0    # fresh-edge decay time constant (~2 days base, SEARCH interior)

# RUN / STREAK (SM-2 lapse-reset, transcribed Wozniak ladder). When S decays to <= this floor
# at the moment of the next deposit, the run BREAKS: S, tau_d and the streak reset to baseline
# (the edge persists; only its accumulation resets) and the next recall starts a fresh run.
# Above the floor the partial decay resumes the same run.
EPI_RUN_FLOOR = 0.05           # S below which a run breaks (matches HEBB_PRUNE = 0.05 grain)

# SM-2 Wozniak ladder for tau_d growth (transcribed, NOT the node-scoped scheduling module —
# proposal D6). reps 1 -> interval 1 day; rep 2 -> 6 days; rep >=3 -> prev_interval * ease.
# ease starts at EPI_EASE_START, nudged by EPI_EASE_NUDGE per spaced reinforcement, floored at
# EPI_EASE_FLOOR; a lapse resets reps + interval. tau_d = EPI_TAU_D_BASE_S * interval(days). The
# resulting tau_d is CAPPED at EPI_TAU_D_MAX_S so it can never exceed the archive retention
# (EPI_ARCHIVE_MAX_DAYS) that recompute-from-archive depends on (proposal §9 / risk note).
EPI_EASE_START = 2.5           # SM-2 initial ease factor
EPI_EASE_NUDGE = 0.1           # ease increment per successful spaced reinforcement
EPI_EASE_FLOOR = 1.3           # SM-2 minimum ease (Wozniak floor)
EPI_SM2_INTERVAL1_DAYS = 1.0   # first-rep interval (days)
EPI_SM2_INTERVAL2_DAYS = 6.0   # second-rep interval (days)
EPI_TAU_D_MAX_S = 100.0 * 86400.0  # cap tau_d at 100 days (< EPI_ARCHIVE_MAX_DAYS=120 retention)

# Promotion bar on the continuous strength S (the discrete-count bar value 3 is preserved as
# the float threshold — proposal §2d / D3). is_promotable keys on S >= this for the STRONG tier.
EPI_PROMOTE_S = float(HEBB_PROMOTE_REPEATS)   # 3.0  (STRONG bar — full chain-weight)

# OPTION 3 (operator greenlit 2026-06-18) — a SUB-3.0 WEAK promotion bar. The study found a
# genuine full-day burst (B) saturates ~1.9 < the strong 3.0 bar but is still a real binding worth
# surfacing PROVISIONALLY; a weak edge promotes at a REDUCED chain-weight (origin="weak") so
# downstream recall ranks it lower and its lower w decays it out faster if not reinforced (the
# "faster post-promotion decay" the study wanted). A weak edge that LATER reaches S>=EPI_PROMOTE_S
# UPGRADES to strong via the idempotent ledger. 1.5 cleanly promotes the burst (B=1.94 weak) while
# excluding dense junk (D=1.21) and sparse pairs (E=0.81). env-overridable at the wiring layer.
EPI_PROMOTE_S_WEAK = 1.5                       # WEAK bar — provisional, reduced-weight promotion

# OPTION 3 const-retune — the SELECTIVE spaced-reinforcement lever. A genuine cross-session run
# (C) of only ~3 daily deposits caps at SUM(s_gap*c_count) ~= 2.75 even with zero decay, so it can
# never clear the 3.0 STRONG bar from a tau_d/decay-rate lever alone (that only slows decay, it
# cannot raise the deposit-sum ceiling). The deposit-CREDIT lever does: a genuine cross-session
# RE-confirmation (gap >= EPI_LOCK_GAP_S AND not the run's OPENING deposit — i.e. a truly spaced
# rep, reps>=2) earns EPI_SPACED_BOOST x the normal deposit. This is the SM-2 tau_d-growth SPIRIT
# (spaced reinforcement consolidates DEEPER) expressed as per-deposit credit. It is PERFECTLY
# SELECTIVE: a within-session burst (B/D) has only its OPENING deposit treated as locked (first_
# deposit => no boost) and a run-broken sparse pair (E) re-opens each run (no spaced re-confirm),
# so D/E/B are UNTOUCHED (verified: D=1.21, E=0.81, B=1.94 frozen) while C climbs 2.49 -> 3.33.
# 1.6 clears C with margin (3.33 > 3.3) without lifting junk D above the weak bar (the junk
# separation the study insisted on). env-overridable. (Harness curve-search re-confirmed feasible.)
EPI_SPACED_BOOST = 1.6                         # x-credit on a genuine cross-session re-confirmation

# is_promotable tier sentinels (OPTION 3 two-tier gate). NONE = below the weak bar / w-gate fails.
EPI_TIER_NONE = "none"
EPI_TIER_WEAK = "weak"
EPI_TIER_STRONG = "strong"

# --- FEAT-2026-06-20 intra-day dual-surface credit (default-INERT pure math; the WIRING gates it
# shadow/apply, and accrue applies it ONLY when a credit fn is injected — parity preserved). All
# env-overridable via epiphanies._apply_curve_env (EPI_CURVE_ENV). See the SEWE proposal.
EPI_INTRADAY_LIFT_MIN = 2.0        # shared-floor: occasion-lift FLOOR (Church&Hanks; selective)
EPI_INTRADAY_MIN_SUPPORT = 3       # shared-floor: each node in >= this many distinct genuine occasions
EPI_INTRADAY_MIN_PAIR_SUPPORT = 2  # shared-floor: the pair in >= this many distinct genuine occasions
EPI_INTRADAY_A_MIN_SITTINGS = 3    # Surface A (breadth): >= this many distinct sittings before any credit
EPI_INTRADAY_A_SAT = 3.0           # Surface A: distinct-sitting saturation constant
EPI_INTRADAY_B_LIFT_SPAN = 4.0     # Surface B (depth): selectivity span above the lift floor
EPI_INTRADAY_WA = 0.5              # Surface A weight (0 disables A)
EPI_INTRADAY_WB = 0.5              # Surface B weight (0 disables B)
EPI_INTRADAY_CAP = 0.6            # per-occasion credit cap (deliberately < the 1.0 cross-session lock)
# FEAT-2026-06-20 STRENGTH-EXPLOSION GUARD: a hard ceiling on S (= 2x the STRONG bar). Without it,
# sustained cross-day recurrence grows S UNBOUNDEDLY (sim: 63 @120d, 82 @240d) -> a flooded edge
# stays strong-tier AND decay-sticky long after the flood stops (the "silent massive strengthening"
# risk, widened by re-including high-frequency sem_ event nodes). Beyond the STRONG bar, S only sets
# the tier (inert — all 'strong') + decay headroom, so clamping at 2x strong preserves genuine runs
# (which reach ~3.3, well under the cap) while bounding worst-case stickiness (tau_d-cap decay-to-bar
# ~140d). w/confidence were already bounded (w-EMA -> 1.0; chain confidence = min(1,w)); this bounds
# the S axis. env-overridable (ASTHENOS_EPI_S_CAP); 0/negative would disable (NOT recommended).
EPI_S_CAP = 2.0 * EPI_PROMOTE_S   # = 6.0

# FEAT-2026-06-20 a-posteriori OUTCOME-reward (demonstrated-value -> GRADED REVERENCE). Default-INERT:
# applied ONLY when accrue is given an outcome_credit_of fn (the WIRING gates it shadow/apply), so
# accrue stays byte-identical / parity-locked when None. The model contributes ONLY a polarity sign;
# these consts + the current edge strength + the outcome history set the MAGNITUDE (R1). PLACEHOLDER
# values — calibrated FROM the shadow window (per the SEWE proposal), env-overridable via
# epiphanies._apply_curve_env (EPI_CURVE_ENV).
EPI_OUTCOME_BASE_HUMAN = 0.5    # base credit per attested HUMAN keep/confirm (may cross the STRONG bar)
EPI_OUTCOME_BASE_AUTO = 0.15    # base credit per machine-attested outcome (AUTO — capped to WEAK alone)
EPI_OUTCOME_S_FLOOR = 0.25      # R3 floor: a new low-S edge still earns this frac of the modulated delta
EPI_OM_CAP = 4.0               # R4 momentum ceiling
EPI_OM_GAIN = 0.5              # R4 momentum->acceleration gain (accel in [1, 1+GAIN*CAP] = [1, 3.0])
EPI_OM_TAU_S = 14.0 * 86400.0  # R5 momentum recency-decay (~14d) — om bleeds off if confirmations stop
EPI_REV_STEP_MAX = 3          # max SM-2 reverence rungs advanced per fold (HUMAN channel only)

# Sentinel for "no qualifying deposit yet" on EdgeState.last_t. A genuine occasion can legitimately
# land at epoch t==0.0 (fixtures) so 0.0 cannot double as "never deposited" — that ambiguity made
# every t=0-anchored same-day occasion look like a fresh opening deposit. -1.0 is unreachable for a
# real (non-negative) epoch timestamp.
EPI_NO_DEPOSIT_T = -1.0


@dataclass
class Sitting:
    """One same-occasion unit (FEAT-2026-06-18): the 30-min-grouped co-activation events plus
    the ABSOLUTE deposit timestamp the continuous curve needs.

    `t` is the occasion's wall-clock time in epoch seconds — it drives s_gap (the gap to the
    previous occasion) and the cycling decay to consolidation-now. `day` is retained as the
    integer day-stamp (t // 86400) for the cg-comparable regression fixtures and any caller
    that still reasons in days; it is NOT the segmentation boundary any more (that is t-based).
    """
    day: int                              # integer day-stamp (t // 86400), comparability only
    events: list                          # list[list[str]] — co-activation node groups
    provenance: str = "genuine"           # genuine | cohort_validated | operator
    t: float = 0.0                        # absolute occasion timestamp (epoch seconds)
    event_sources: list = field(default_factory=list)  # FEAT-2026-06-20 §5.1: per-event source,
                                          # PARALLEL to `events` ('genuine'|'inject'|'operator'|...).
                                          # ADDITIVE provenance: lets a consumer attribute a pair to
                                          # the source(s) it co-occurred under, so an inject ride-along
                                          # is not laundered as genuine (the breach: segment_sittings
                                          # folds non-replay inject rows into a sitting). accrue()
                                          # ignores it -> S is byte-identical (parity preserved).


@dataclass
class EdgeState:
    """Per-edge v3 accounting (FEAT-2026-06-18 — DUAL AXIS).

    Two SEPARATE quantities, never conflated:
      - `cg`  : the INTEGER count of distinct qualifying occasions (the VETO axis — the
                suppression-override at epiphanies.py counts LOCKED cross-session occasions;
                a dense single session cannot spam past an operator veto).
      - `S`   : the CONTINUOUS strength (the PROMOTION/quality axis) — the spaced-repetition
                consolidation curve. Promotion latches at S >= EPI_PROMOTE_S (3.0).
    Plus the run/SM-2 state that drives S's cycling decay and tau_d growth.
    """
    cg: int = 0                           # integer qualifying-occasion count (veto axis)
    S: float = 0.0                        # continuous consolidation strength (promotion axis)
    tau_d: float = EPI_TAU_D_BASE_S       # current decay time-constant (grows with spacing)
    w: float = 0.0
    last_day: int = 0
    last_q_day: int = -10 ** 9
    last_t: float = EPI_NO_DEPOSIT_T      # absolute time of the last QUALIFYING deposit (run clock;
    #                                       EPI_NO_DEPOSIT_T = none yet)
    reps: int = 0                         # SM-2 repetition count within the current run
    ease: float = EPI_EASE_START          # SM-2 ease factor (>= EPI_EASE_FLOOR)
    interval_days: float = 0.0            # SM-2 current inter-rep interval (days)
    run_id: int = 0                       # increments each time a run breaks + restarts
    om: float = 0.0                       # FEAT-2026-06-20 outcome-momentum (R4): recomputed fresh each
    #                                       fold from the attested-outcome history (no stale cross-fold
    #                                       state); 0.0 when no outcome_credit_of injected -> parity-safe.
    # Phase-6 promotion-provenance seed (Decision A safety net): the salience terms that
    # carried the most recent qualification, so repetition-only bindings are surfaceable.
    last_terms: dict = field(default_factory=dict)


def composite_salience(surprise: float, contradiction: float, access: int,
                       tagged: bool, recalls_before: int,
                       fam: float = EPI_SURPRISE_FAM) -> tuple[float, dict]:
    """Binding salience with FAMILIARITY-DECAYING surprise + access-only repetition (FIX-5').

    surprise_eff = surprise * fam^(recalls_before): a node recalled often is no longer novel
    (the live 1-max_cosine drops as similar nodes accumulate in the index — so calling the
    live compute_salience at consolidation gives this for free; the harness models it).
    Repetition uses access_count ONLY (degree is split into `sel`, breaking the v2 sign-
    inversion feedback). An operator tag clamps HIGH. Returns (salience, term-breakdown) so
    the promotion-provenance layer can flag a binding carried by repetition alone.
    """
    if tagged:
        return SALIENCE_TAG_VALUE, {"tagged": True}
    surprise_eff = surprise * (fam ** max(0, recalls_before))
    rep = min(1.0, access / max(1e-9, SALIENCE_REPETITION_SATURATION))
    s_term = SALIENCE_W_SURPRISE * surprise_eff
    c_term = SALIENCE_W_CONTRADICTION * contradiction
    r_term = SALIENCE_W_REPETITION * rep
    sal = max(0.0, min(1.0, s_term + c_term + r_term))
    terms = {"surprise": round(s_term, 4), "contradiction": round(c_term, 4),
             "repetition": round(r_term, 4)}
    return sal, terms


def _sel(deg_a: int, deg_b: int) -> float:
    """Selectivity hub penalty (FIX-5'): selective pairs ~1, indiscriminate hubs penalized."""
    dp = deg_a * deg_b
    return max(EPI_SEL_FLOOR, 1.0 / (1.0 + EPI_C_SEL * math.log(1.0 + dp)))


def _provenance_mult(provenance: str) -> float:
    return {"operator": EPI_P_OP, "cohort_validated": EPI_P_COHORT}.get(provenance, 1.0)


def promotion_tier(st: EdgeState) -> str:
    """OPTION 3 (operator greenlit 2026-06-18) TWO-TIER promotion gate. Returns the promotion tier:

      EPI_TIER_STRONG  if S >= EPI_PROMOTE_S       (3.0 — full chain-weight)
      EPI_TIER_WEAK    if EPI_PROMOTE_S_WEAK <= S < EPI_PROMOTE_S  (1.5..3.0 — provisional, reduced
                       chain-weight; ranks lower downstream + decays out faster if not reinforced;
                       UPGRADES to strong via the idempotent ledger once it later reaches S>=3.0)
      EPI_TIER_NONE    otherwise (below the weak bar, OR the w>=HEBB_PROMOTION quality gate fails)

    The w>=HEBB_PROMOTION quality gate is preserved for BOTH tiers so a low-salience pair still
    cannot promote on strength alone (both axes do work — proposal §2d / D3). The promotion axis is
    the continuous S (the discrete-count bar value 3 preserved as the strong float threshold), NOT
    the integer occasion count: a single within-session burst is ONE unbroken run that climbs to
    the WEAK tier on the graduated mid-tiers + c_count, while a genuine multi-day cross-session run
    reaches the STRONG tier via the 1.0 locks + the spaced-reinforcement boost (EPI_SPACED_BOOST).
    """
    if st.w < HEBB_PROMOTION:
        return EPI_TIER_NONE
    if st.S >= EPI_PROMOTE_S:
        return EPI_TIER_STRONG
    if st.S >= EPI_PROMOTE_S_WEAK:
        return EPI_TIER_WEAK
    return EPI_TIER_NONE


def is_promotable(st: EdgeState) -> bool:
    """Live promotion gate (OPTION 3): True if the edge promotes at EITHER tier (weak or strong).

    Kept as the boolean any-tier predicate so every existing call site (consolidate / promote /
    the harness parity test) stays correct: a WEAK edge IS promotable now (provisionally). Callers
    that need to distinguish the chain-weight / provenance use promotion_tier(st) for the tier.
    """
    return promotion_tier(st) != EPI_TIER_NONE


def s_gap(gap_s: float) -> float:
    """Gap-strength: 1 - exp(-gap/tau_g) for gap < 24h, LOCKED to exactly 1.0 for gap >= 24h.

    The geometric rise approaches ~0.95 by 24h^- through the within-day tiers; crossing the 24h
    wall-clock boundary (>= per the §2a/risk-note edge case so consecutive-day deposits lock) is
    full cross-session credit. The first deposit of a (fresh) run has no predecessor -> caller
    passes a >= EPI_LOCK_GAP_S so the opening deposit locks to 1.0 (a fresh recall earns full
    rise; the within-day grading is for SUBSEQUENT same-run reinforcement)."""
    if gap_s >= EPI_LOCK_GAP_S:
        return 1.0
    if gap_s <= 0.0:
        return 0.0
    return 1.0 - math.exp(-gap_s / EPI_TAU_G_S)


def c_count(n: int) -> float:
    """Within-occasion recall-count gradient: 1 - exp(-n/n_sat). Saturating so one dense burst
    cannot run away; generalizes the old dwell D = min(1, m/D_SAT) to a smooth curve."""
    if n <= 0:
        return 0.0
    return 1.0 - math.exp(-float(n) / EPI_N_SAT)


# === FEAT-2026-06-20 intra-day dual-surface credit — PURE math (no IO; default-inert) ===========
# A modular LAYER (not a blanket s_gap raise): a shared NOISE FLOOR both surfaces clear, plus two
# orthogonal credit surfaces (A=breadth/encoding-variability, B=depth/reconsolidation). The impure
# caller (epiphanies.consolidate) supplies the occasion-marginal substrate + the genuine-attributed
# inputs; accrue applies the resulting per-pair credit ONLY when a credit fn is injected (else
# byte-identical). Literature + adversary findings: see the SEWE proposal.

def occ_lift(c_pair: float, c_a: float, c_b: float, n_occ: float) -> float:
    """Occasion-level PMI lift = C_pair*N_occ / (C_a*C_b). High = selective (genuine) association;
    ~1 = chance/mechanical-bundle. Matches the established hebbian._lift semantics."""
    if c_a <= 0 or c_b <= 0 or n_occ <= 0:
        return 0.0
    return (float(c_pair) * float(n_occ)) / (float(c_a) * float(c_b))


def intraday_floor_ok(c_a: int, c_b: int, c_pair: int, n_occ: int) -> bool:
    """Shared NOISE FLOOR (support-THEN-lift): absolute min-support FIRST (anti rare-node coincidence
    — lift is unbounded for rare nodes and N-amplified at scale), THEN lift >= floor (selective, not
    a mechanical bundle). The source==genuine ∧ ¬repetition_only ∧ ¬suppressed terms are enforced by
    the IMPURE caller on genuine-attributed occasion data (segment_sittings.genuine_present_pairs)."""
    if c_a < EPI_INTRADAY_MIN_SUPPORT or c_b < EPI_INTRADAY_MIN_SUPPORT:
        return False
    if c_pair < EPI_INTRADAY_MIN_PAIR_SUPPORT:
        return False
    return occ_lift(c_pair, c_a, c_b, n_occ) >= EPI_INTRADAY_LIFT_MIN


def surface_a(njd: float, distinct_sittings: int) -> float:
    """BREADTH (Glenberg encoding-variability / TCM): credit ∝ neighborhood-Jaccard-distance × a
    distinct-sitting gate. 0 for a stable context (njd~0 -> that is Surface B's domain, never
    penalized) or too few distinct sittings (< EPI_INTRADAY_A_MIN_SITTINGS). Returns [0,1]."""
    if distinct_sittings < EPI_INTRADAY_A_MIN_SITTINGS or njd <= 0.0:
        return 0.0
    sat = 1.0 - math.exp(-(distinct_sittings - 2) / max(1e-9, EPI_INTRADAY_A_SAT))
    return max(0.0, min(1.0, njd)) * max(0.0, min(1.0, sat))


def surface_b(occ_lift_val: float, salience_pe: bool, reengagement: float) -> float:
    """DEPTH (reconsolidation / study-phase retrieval): selectivity (graded ABOVE the floor so a
    just-cleared mechanical bundle scores ~0) × a prediction-error gate (surprise/contradiction —
    the reconsolidation labilization trigger) × active-reengagement. Reclaims stable-context (njd~0)
    genuine recurrence WITHOUT re-admitting loops. Returns [0,1]. NOTE: salience_pe + reengagement
    are PROVISIONAL proxies until pair-level surprise tracking lands (Phase 5); the shadow log
    records the components so the gate can be calibrated before any apply."""
    sel = (occ_lift_val - EPI_INTRADAY_LIFT_MIN) / max(1e-9, EPI_INTRADAY_B_LIFT_SPAN)
    sel = max(0.0, min(1.0, sel))
    pe = 1.0 if salience_pe else 0.0
    re = max(0.0, min(1.0, reengagement))
    return sel * pe * re


def intraday_credit(a_val: float, b_val: float) -> float:
    """Combine the two surfaces once per occasion: min(EPI_INTRADAY_CAP, wA*A + wB*B). ADDITIVE
    (parallel dissociable pathways — CLS/FTT), bounded by the per-surface [0,1] ranges, and CAPPED
    PER-OCCASION below the 1.0 cross-session lock. NOTE (INTRADAY-1): this per-occasion cap does NOT
    bound the same-day CUMULATIVE total — that WEAK-cap (STRONG still requires a >=24h reconfirmation,
    never earned intra-day) is enforced in accrue() by the `pre_id_S < EPI_PROMOTE_S` clamp on the
    intra-day deposit, not here. The floor gate is enforced by the caller. Returns the per-occasion S
    credit (>=0)."""
    return min(EPI_INTRADAY_CAP, EPI_INTRADAY_WA * max(0.0, a_val) + EPI_INTRADAY_WB * max(0.0, b_val))


def _sm2_step(st: EdgeState) -> None:
    """Transcribed SM-2 Wozniak ladder (proposal D6) — advance the run's interval/ease and set
    tau_d. Fed by DEPOSIT SPACING, not node relevance (no node files touched). A lapse (run
    break) is handled by the caller resetting reps/interval/ease to baseline BEFORE calling this
    on the fresh deposit.

    reps 1 -> interval 1 day; rep 2 -> 6 days; rep >= 3 -> prev_interval * ease (ease nudged up
    EPI_EASE_NUDGE per spaced reinforcement, floored EPI_EASE_FLOOR). tau_d = base * interval,
    capped at EPI_TAU_D_MAX_S so it never outruns the archive retention recompute depends on."""
    st.reps += 1
    if st.reps <= 1:
        st.interval_days = EPI_SM2_INTERVAL1_DAYS
    elif st.reps == 2:
        st.interval_days = EPI_SM2_INTERVAL2_DAYS
    else:
        st.ease = max(EPI_EASE_FLOOR, st.ease + EPI_EASE_NUDGE)
        st.interval_days = st.interval_days * st.ease
    st.tau_d = min(EPI_TAU_D_MAX_S, EPI_TAU_D_BASE_S * max(1.0, st.interval_days))


def segment_sittings(records: list, gap_s: int = EPI_OCCASION_GAP_S) -> list:
    """Offline timestamp grouping of co-activation-log records into same-OCCASION units
    (FEAT-2026-06-18 — the 40-min sitting chop retired; the 30-min occasion window is the ONE
    remaining time boundary, proposal §4).

    records: hebb_log/archive rows, each a dict with 'ts' (ISO seconds), 'nodes' (list), and
      'source' ('genuine'|'replay'|...). Replay rows are EXCLUDED (replay is not a recall
      occasion — proposal §5). Consecutive genuine records within gap_s (30 min) collapse into
      one occasion (deduped -> c_count); a larger gap starts a NEW occasion (the gap BETWEEN
      occasions drives s_gap; crossing 24h locks). Each occasion now CARRIES its absolute
      timestamp `t` (epoch seconds of its FIRST record) — the per-row timestamps are no longer
      discarded, because the continuous curve needs the inter-occasion gap + the consolidation-
      now decay. `day` (t // 86400) is retained for cg-comparable callers.
    Durable: reconstructed from the log itself, so it survives a crash (no live recall clock
    is trusted — the v1/v2 'no consumer' error).
    """
    rows = []
    for r in records:
        if r.get("source", "genuine") == "replay":
            continue
        ts = r.get("ts")
        nodes = r.get("nodes") or []
        if not ts or len(nodes) < 2:
            continue
        try:
            t = _dt.datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            continue
        rows.append((t, nodes, r.get("source", "genuine")))
    rows.sort(key=lambda x: x[0])
    sittings: list = []
    cur: list = []
    cur_prov = "genuine"
    last_t = None
    for t, nodes, src in rows:
        if last_t is not None and (t - last_t).total_seconds() > gap_s:
            if cur:
                t0 = cur[0][0]
                sittings.append(Sitting(day=int(t0.timestamp()) // 86400,
                                        events=[n for _, n, _ in cur], event_sources=[s for _, _, s in cur], provenance=cur_prov,
                                        t=t0.timestamp()))
            cur = []
            cur_prov = "genuine"
        cur.append((t, nodes, src))           # §5.1: carry per-row source into the occasion
        if src in ("operator", "cohort_validated"):
            cur_prov = src
        last_t = t
    if cur:
        t0 = cur[0][0]
        sittings.append(Sitting(day=int(t0.timestamp()) // 86400,
                                events=[n for _, n, _ in cur], event_sources=[s for _, _, s in cur], provenance=cur_prov,
                                t=t0.timestamp()))
    return sittings


# FEAT-2026-06-20 §5.1: sources that count as a REAL recall co-activation (NOT a passive standing-
# availability inject surface, NOT replay). 'inject' is deliberately EXCLUDED — it is the breach the
# adversary found (segment_sittings folds inject rows into a sitting; without this attribution an
# inject ride-along pair is credited as genuine). 'replay' never reaches a Sitting (segment drops it).
GENUINE_SOURCES = ("genuine", "operator", "cohort_validated")


def genuine_present_pairs(sitting: "Sitting") -> set:
    """FEAT-2026-06-20 §5.1: the pair keys that co-occur in >=1 GENUINE-sourced event of this
    sitting (provenance attributed at the PAIR level via sitting.event_sources). A consumer that
    must require genuine recall co-occurrence (e.g. the future intra-day credit floor) uses this so
    an inject ride-along is NOT laundered as genuine. Pure / read-only; ADDITIVE (accrue does not
    use it, so S accrual is unchanged). Fail-soft fallback: if event_sources is absent/short for an
    event (older sittings), that event is treated as genuine — preserving today's behavior."""
    srcs = getattr(sitting, "event_sources", None) or []
    pairs: set = set()
    for i, ev in enumerate(sitting.events):
        src = srcs[i] if i < len(srcs) else "genuine"
        if src not in GENUINE_SOURCES:
            continue
        u = sorted(set(ev))
        for a in range(len(u)):
            for b in range(a + 1, len(u)):
                pairs.add(f"{u[a]}::{u[b]}")
    return pairs


def accrue(sittings: list,
           salience_of: Callable[[str, int], tuple],
           query_day: Optional[int] = None,
           intraday_credit_of: Optional[Callable[[str], float]] = None,
           outcome_credit_of: Optional[Callable[[str], Optional[tuple]]] = None) -> dict:
    """Fold occasions into per-edge v3 state — the continuous spaced-repetition consolidation
    curve (FEAT-2026-06-18). Mirrors the calibration harness simulate() exactly (parity test).

    salience_of(node, recalls_before) -> (salience, terms): the injected salience source.
      The live adapter calls compute_salience (familiarity intrinsic to the growing index);
      the parity test injects the harness's composite_salience. recalls_before is this node's
      count of PRIOR occasions (familiarity), supplied by accrue.
    query_day: the consolidation-NOW day. Every edge's S/w is decayed forward to this day's
      start (day * 86400) before returning (default: last occasion's day). The cycling decay
      means S only counts reinforcement that is still 'warm' relative to now.
    Returns {edge_key: EdgeState}.

    The QUALIFICATION predicate (score = S_sal*D*sel*P >= EPI_BAR, the sel hub penalty, the
    EPI_MAT_FLOOR) is UNCHANGED — only qualifying occasions earn any deposit. What changed is
    how a qualifying occasion credits strength: instead of `cg += 1` it deposits the continuous
    `s_gap(gap) * c_count(n)` into S, runs the SM-2 run/streak (lapse-reset below EPI_RUN_FLOOR),
    and cycles S's decay to consolidation-now. The integer `cg` (veto axis) still counts
    qualifying occasions alongside S (DUAL AXIS).
    """
    deg: dict = {}
    recalls: dict = {}
    edges: dict = {}

    def _decay_w_to(st: EdgeState, day: int) -> None:
        # the live per-day multiplicative w-EMA decay (the quality axis — _decay_and_prune law).
        gap = day - st.last_day
        if gap > 0:
            st.w *= max(0.0, 1.0 - HEBB_DECAY) ** gap
            st.last_day = day

    def _decay_S_to(st: EdgeState, t_now: float) -> None:
        # the cycling consolidation decay (proposal §2b): S decays exp(-dt/tau_d). Nothing
        # persists — even the cross-session lock decays, so it must be re-earned. Applied
        # in-place; the run/lapse decision reads the decayed S.
        if st.last_t == EPI_NO_DEPOSIT_T or st.S <= 0.0:
            return
        dt = t_now - st.last_t
        if dt > 0:
            st.S *= math.exp(-dt / st.tau_d)

    for sit in sittings:
        P = _provenance_mult(sit.provenance)
        present = set()
        pair_m: dict = {}
        node_sal: dict = {}
        for ev in sit.events:
            u = sorted(set(ev))
            present.update(u)
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    k = f"{u[i]}::{u[j]}"
                    pair_m[k] = pair_m.get(k, 0) + 1
        # salience for each present node at this occasion's familiarity level (cache per node)
        for n in present:
            node_sal[n] = salience_of(n, recalls.get(n, 0))
        for key, m in pair_m.items():
            a, b = key.split("::", 1)
            (sa, ta), (sb, tb) = node_sal[a], node_sal[b]
            S_sal = math.sqrt(max(0.0, sa) * max(0.0, sb))
            D = min(1.0, m / EPI_D_SAT)
            score = S_sal * D * _sel(deg.get(a, 0), deg.get(b, 0)) * P
            st = edges.get(key)
            if st is None:
                if score < EPI_MAT_FLOOR:
                    continue
                st = EdgeState(last_day=sit.day, last_t=EPI_NO_DEPOSIT_T)
                edges[key] = st
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
            _decay_w_to(st, sit.day)
            if score >= EPI_BAR:
                # --- the continuous deposit + cycling decay + SM-2 run (FEAT-2026-06-18) ---
                # 1. decay the running S to THIS occasion's time before deciding lapse/run.
                _decay_S_to(st, sit.t)
                # 2. gap to the previous qualifying deposit drives s_gap; the FIRST deposit of a
                #    run has no predecessor (last_t == EPI_NO_DEPOSIT_T) -> lock to 1.0 (a fresh
                #    recall earns the full rise; the within-day grading is for subsequent same-run
                #    reinforcement).
                first_deposit = (st.last_t == EPI_NO_DEPOSIT_T)
                gap = EPI_LOCK_GAP_S if first_deposit else (sit.t - st.last_t)
                # 3. LAPSE check: if the decayed S has fallen to/below the run-floor, the run
                #    BREAKS — reset S/tau_d/streak to baseline (the edge persists). The next
                #    recall (this one) starts a FRESH run.
                if not first_deposit and st.S <= EPI_RUN_FLOOR:
                    st.S = 0.0
                    st.tau_d = EPI_TAU_D_BASE_S
                    st.reps = 0
                    st.ease = EPI_EASE_START
                    st.interval_days = 0.0
                    st.run_id += 1
                    gap = EPI_LOCK_GAP_S          # a fresh run's opening deposit locks to 1.0
                    first_deposit = True          # a re-opened run earns NO spaced boost on its open
                # 4. DEPOSIT: S += boost * s_gap(gap) * c_count(n). n is the in-occasion recall count
                #    m. OPTION 3 SELECTIVE spaced-reinforcement boost: a genuine cross-session
                #    RE-confirmation (gap >= EPI_LOCK_GAP_S AND not the run's OPENING deposit — a
                #    truly spaced rep, reps>=2) earns EPI_SPACED_BOOST x credit (the SM-2 tau_d-
                #    growth spirit as per-deposit credit, since a decay-rate lever can't raise the
                #    short-run deposit-sum ceiling past 3.0). PERFECTLY SELECTIVE: a within-session
                #    burst's only locked deposit is its OPENING (first_deposit => no boost) and a
                #    run-broken sparse pair re-opens each run (no spaced re-confirm), so within-
                #    session junk / sparse pairs are UNTOUCHED while a genuine multi-day run climbs.
                spaced = (gap >= EPI_LOCK_GAP_S) and not first_deposit
                boost = EPI_SPACED_BOOST if spaced else 1.0
                st.S += boost * s_gap(gap) * c_count(m)
                # FEAT-2026-06-20 intra-day dual-surface credit: a SAME-DAY (gap < lock) deposit may
                # earn an ADDITIVE credit IFF the wiring injected a credit fn (default None -> inert,
                # so accrue is byte-identical / parity-locked). Cross-session deposits (gap >= lock)
                # are untouched — the spaced-boost path owns those, and the credit NEVER earns the
                # boost, so STRONG still requires a real >= 24h reconfirmation.
                if intraday_credit_of is not None and gap < EPI_LOCK_GAP_S:
                    st.S += max(0.0, intraday_credit_of(key))
                # INTRADAY-1 (audit RE-FIX 2026-06-20): a SAME-DAY deposit (the base s_gap*c_count AND
                # any intra-day credit) must NEVER reach the STRONG bar — STRONG requires a real >=24h
                # cross-session reconfirmation (reps>=2, advanced ONLY by the locked-gap _sm2_step
                # below). The PRIOR clamp read pre_S AFTER the base deposit, so once accumulated S
                # crossed 3.0 the guard disabled ITSELF and a multi-occasion same-day burst ran away
                # past STRONG (adversary-reproduced breach at n_occ>=6). Clamp on the RUN STATE instead:
                # while this is a same-day deposit (gap<lock) AND the run has not yet earned a cross-
                # session reconfirm (reps<2), cap S just below STRONG. WEAK (1.5<=S<3.0) is still
                # reachable same-day (the feature); the cap lifts once a genuine >=24h reconfirm makes
                # reps>=2. The opening deposit (gap==lock) is exempt (gap<lock is False). Parity-safe:
                # no harness same-day fixture reaches 3.0 (B burst ~1.94), so the pure path is unchanged.
                if gap < EPI_LOCK_GAP_S and st.reps < 2:
                    st.S = min(st.S, EPI_PROMOTE_S - 1e-6)
                if EPI_S_CAP > 0.0:
                    st.S = min(st.S, EPI_S_CAP)   # FEAT-2026-06-20: hard S ceiling -> no strength
                    #                               explosion from sustained/flooded recurrence.
                # 5. SM-2 ladder advances tau_d (only a cross-session-spaced reinforcement, i.e.
                #    a >= 24h-locked gap, deepens consolidation; a same-day within-run repeat does
                #    NOT advance the interval — it just tops S up). This is the "locks in the lock
                #    and cycles it": spaced reps grow tau_d, dense same-day reps do not.
                if gap >= EPI_LOCK_GAP_S:
                    _sm2_step(st)
                # 6. the integer veto axis: count this qualifying occasion (DUAL AXIS).
                st.cg += 1
                st.w = st.w + HEBB_EMA_ALPHA * (1.0 - st.w)
                st.last_q_day = sit.day
                st.last_t = sit.t
                # promotion-provenance: record the carrying terms (sum the pair's two breakdowns)
                st.last_terms = {k: round(ta.get(k, 0.0) + tb.get(k, 0.0), 4)
                                 for k in ("surprise", "contradiction", "repetition")}
        for n in present:
            recalls[n] = recalls.get(n, 0) + 1
    # FEAT-2026-06-20 a-posteriori OUTCOME reward (default-INERT: outcome_credit_of=None -> whole block
    # skipped -> accrue byte-identical / parity-locked). Per edge with an ATTESTED-outcome confirmation,
    # grow S (+ durability via SM-2 rungs) by the demonstrated-VALUE delta: R1 polarity-only (sign) *
    # SYSTEM-magnitude (edge strength + outcome history), R2 additive, R3 strength-modulated+floored,
    # R4 momentum (om), R5 double-decay (om recency-decay + the EPI_S_CAP clamp; om is recomputed fresh
    # each fold from the history so there is NO stale cross-fold momentum). The model's a-priori veto
    # NEVER reaches here — it only subtracts from the key-SET downstream; ONLY outcome grows S (C1/K5).
    if outcome_credit_of is not None:
        for _key, st in edges.items():
            cr = outcome_credit_of(_key)
            if not cr:
                continue
            sign, n_conf, channel, dt_last = cr
            if sign <= 0 or n_conf <= 0:
                continue
            om = min(EPI_OM_CAP, float(n_conf)) * math.exp(-max(0.0, float(dt_last)) / EPI_OM_TAU_S)
            accel = 1.0 + EPI_OM_GAIN * om                                   # R4
            s_frac = max(EPI_OUTCOME_S_FLOOR, (st.S / EPI_S_CAP) if EPI_S_CAP > 0.0 else 1.0)  # R3
            base = EPI_OUTCOME_BASE_HUMAN if channel == "HUMAN" else EPI_OUTCOME_BASE_AUTO
            pre_S = st.S
            st.S += base * float(sign) * s_frac * accel                      # R2 additive, R1 magnitude
            # DUAL COEFFICIENT: the AUTO (machine-attested) channel ALONE can lift at most to WEAK;
            # only the HUMAN (attested keep/confirm) channel may cross the STRONG bar. A genuine deposit
            # run that ALREADY reached STRONG (pre_S >= bar) is never clawed back by this clamp.
            if channel != "HUMAN" and pre_S < EPI_PROMOTE_S:
                st.S = min(st.S, EPI_PROMOTE_S - 1e-6)
            if EPI_S_CAP > 0.0:
                st.S = min(st.S, EPI_S_CAP)                                  # R5 axis-2 hard ceiling
            st.om = round(om, 4)
            # REVERENCE: HUMAN-confirmed reuse advances the SM-2 ladder -> tau_d climbs (held more
            # durably / "reverently"); AUTO buys no durability rungs.
            if channel == "HUMAN":
                for _ in range(min(EPI_REV_STEP_MAX, int(n_conf))):
                    _sm2_step(st)
    qd = query_day if query_day is not None else (sittings[-1].day if sittings else 0)
    t_now = float(qd) * 86400.0
    for st in edges.values():
        _decay_w_to(st, qd)
        _decay_S_to(st, t_now)                   # cycle S forward to consolidation-now
    return edges


def carried_by_repetition_only(st: EdgeState, eps: float = 1e-6) -> bool:
    """Phase-6 flag (Decision A safety net): did this binding qualify on the repetition term
    ALONE (no surprise, no contradiction)? These are the pure-frequency promotions the
    operator wants surfaced + correctable. A `tagged` qualification is not repetition-only."""
    t = st.last_terms or {}
    if t.get("tagged"):
        return False
    return (t.get("repetition", 0.0) > eps
            and t.get("surprise", 0.0) <= eps
            and t.get("contradiction", 0.0) <= eps)


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.balancing
# Author:     code_warrior (Epiphanies v3)
# Project:    Asthenosphere — SAM/IA — Epiphanies (episodic associative binding)
# Version:    0.3.0  (OPTION 3, 2026-06-18: two-tier promotion — a sub-3.0 WEAK bar
#             (EPI_PROMOTE_S_WEAK=1.5) alongside the STRONG bar, promotion_tier() returns
#             none/weak/strong, + the SELECTIVE spaced-reinforcement deposit boost EPI_SPACED_BOOST
#             so the genuine cross-session run clears the strong bar without lifting within-session
#             junk above the weak bar)
# Phase:      build — pure model, parity-tested, wired through epiphanies.py (gated P5/P7).
# Layer:      core (pure library — no IO, no daemon, no live-store contact)
# Role:       the Epiphanies v3 model. The QUALIFICATION predicate (score = S_sal*D*sel*P >=
#             EPI_BAR, sel hub penalty, EPI_MAT_FLOOR) is unchanged; FEAT-2026-06-18 replaces the
#             discrete `cg += 1` with a CONTINUOUS consolidation curve: 30-min same-occasion
#             grouping (the 40-min sitting chop retired -> one boundary); per qualifying occasion
#             a deposit S += s_gap(gap)*c_count(n) (s_gap locks to 1.0 at >=24h); a cycling decay
#             S*=exp(-dt/tau_d) to consolidation-now (nothing persists); an SM-2-transcribed
#             run/streak (lapse-reset below EPI_RUN_FLOOR, tau_d grows with spaced reinforcement);
#             promotion latches at S >= EPI_PROMOTE_S (3.0). DUAL AXIS: the integer cg (veto axis)
#             is kept alongside the float S (promotion/quality axis).
# Stability:  the calibration harness + live-parity test (test_balancing_parity) are the contract
#             this module must keep; the harness SEARCH proved the curve consts feasible (2026-06-18).
# ErrorModel: pure; raises only on malformed injected callables. segment_sittings skips
#             unparseable / replay / <2-node rows rather than failing.
# Depends:    .config (HEBB_* / SALIENCE_* / _dt).
# Exposes:    Sitting (now + event_sources, §5.1), EdgeState, composite_salience, segment_sittings,
#             accrue, is_promotable, promotion_tier, s_gap, c_count, carried_by_repetition_only,
#             genuine_present_pairs + GENUINE_SOURCES (§5.1 per-pair provenance attribution) + the EPI_* qualification-
#             tuple and continuous-curve constants (EPI_TAU_G_S / EPI_N_SAT / EPI_TAU_D_BASE_S /
#             EPI_RUN_FLOOR / EPI_PROMOTE_S / EPI_PROMOTE_S_WEAK / EPI_SPACED_BOOST /
#             EPI_TIER_* / EPI_OCCASION_GAP_S ...).
# --------------------------------------------------------------------------
