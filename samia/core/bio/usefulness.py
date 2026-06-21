"""samia.core.bio.usefulness — Phase 1: the a-priori model USEFULNESS veto (generator-independent).

Layer 1 (Owns / Depends):
    Owns:    the a-priori relatedness VETO — read the CONTENT of two linked nodes + the co-occurrence
             TOPOLOGY the cosine proposer never saw, and decide whether a statistically-genuine edge
             genuinely relates. SUBTRACT-ONLY by contract (the caller ANDs the pass-set onto
             promotable_keys via the allow_keys firewall; this module never grows S) and LEDGER-ONLY
             (the verdict is NEVER written onto any EdgeState / shadow_edges.json field — K5
             anti-self-fulfillment).
    Depends: nothing at import time. The model call is INJECTED (infer_fn/parse_fn default to the
             cached contradiction-judge facade, lazy-imported) so the prompt+rubric logic is pure +
             testable without a backend.

GENERATOR-INDEPENDENCE (K1): the link is PROPOSED by the MiniLM cosine bi-encoder (a single pooled
cosine); this verifier is an autoregressive decoder LLM (Qwen3-4B — the cached judge backend) reading
the raw node bodies AND the co-occurrence topology/degree — a different modality/objective the cosine
never had. The topology input is the load-bearing fix: it restores true independence on the
mechanical-bundle axis (a low-degree B-hub that fools cosine is exposed by the mediator evidence).

Fail-soft: infer_fn -> None / parse failure => score_pair returns None == ABSTAIN (NOT a veto), so an
inference outage can never silently drop a genuine link. Anti-Goodhart (K3): rubric weights live in
CODE here, never in the prompt; no single optimizable scalar is exposed to the model.
"""

from __future__ import annotations

from typing import Optional, Callable

USEFULNESS_CONF = 0.7          # model confidence — RECORDED in the verdict for calibration; NO LONGER a
                              # hard veto term (see verdict_from_parsed: a low-confidence verdict must
                              # not veto a statistically-genuine edge — fail-open contract).
USEFULNESS_MIN_SCORE = 0.25    # max-axis pass threshold — sits in the empirically-observed gap between
                              # genuine pairs (max-axis ~0.3) and unrelated pairs (~0.0); see the note in
                              # verdict_from_parsed.
USEFULNESS_MAX_TOKENS = 512
USEFULNESS_BODY_CHARS = 2000   # per-node content budget (mirrors the judge/synth [:2000] caps)
USEFULNESS_VETO_STREAK = 2     # STICKY: a veto acts only after this many consecutive SCORED folds of
                              # raw_veto (ABSTAIN preserves the streak; an explicit non-veto resets it).
                              # A single-fold disagreement fails OPEN.
USEFULNESS_MAX_SCORED = 10     # K4 per-fold budget: score at most this many eligible keys (highest-S first)

# rubric weights — in CODE, never shown to the model (anti-Goodhart). expect-co-recall dominates
# because "would a mind usefully recall one when the other is active" is the operative question.
_W_TOPICAL, _W_CAUSAL, _W_CORECALL = 0.30, 0.30, 0.40

USEFULNESS_PROMPT_TEMPLATE = """You are auditing whether two memory notes genuinely RELATE — whether recalling one when the other is active is USEFUL, not coincidental. Judge ONLY from the note contents and the co-occurrence evidence below; do NOT assume a relation just because they were stored or surfaced near each other.

NOTE A ({a_name}):
{a_text}

NOTE B ({b_name}):
{b_text}

CO-OCCURRENCE EVIDENCE (the structural context the embedding never saw):
{topology}

A link is a MECHANICAL BUNDLE ARTIFACT if A and B are only ever brought together by a shared intermediary / workflow / template (the "mediator" nodes above) rather than by a genuine subject-level relation — name the mediator if so.

Respond with ONLY a JSON object, no prose:
{{"topical": 0.0-1.0, "causal": 0.0-1.0, "expect_corecall": 0.0-1.0, "mechanical_bundle_artifact": true/false, "confidence": 0.0-1.0, "explanation": "one sentence citing the concrete shared subject, or the mediator if it is a bundle artifact"}}"""


def build_prompt(a_name: str, a_text: str, b_name: str, b_text: str, topology: str) -> str:
    return USEFULNESS_PROMPT_TEMPLATE.format(
        a_name=a_name, a_text=(a_text or "")[:USEFULNESS_BODY_CHARS],
        b_name=b_name, b_text=(b_text or "")[:USEFULNESS_BODY_CHARS],
        topology=(topology or "(none)"))


def _to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def verdict_from_parsed(parsed) -> Optional[dict]:
    """Turn the model's parsed JSON into a verdict + the raw (pre-sticky) veto decision. Returns
    None (ABSTAIN) on a malformed/empty parse — never a veto."""
    if not isinstance(parsed, dict):
        return None
    topical = _to_float(parsed.get("topical"))
    causal = _to_float(parsed.get("causal"))
    corecall = _to_float(parsed.get("expect_corecall"))
    bundle = bool(parsed.get("mechanical_bundle_artifact"))
    conf = _to_float(parsed.get("confidence"))
    # RECALIBRATED 2026-06-20 (adversarial audit, finding UF-3 — replayed against the full live ledger
    # + ground-truthed node bodies). Two changes from the first max-axis attempt:
    #   (1) MIN_SCORE 0.5 -> 0.25. The 4B verifier scores genuinely-related pairs conservatively
    #       (max-axis ~0.3) while truly-unrelated pairs score ~0.0 — a clean bimodal gap with NOTHING
    #       in (0.0, 0.3). 0.5 false-vetoed the genuine vision_distill::vision_distill pair (max 0.3);
    #       0.25 sits in the gap so genuine PASS, unrelated (max 0.0) still VETO.
    #   (2) DROP the `conf < USEFULNESS_CONF` veto term. It was redundant on the live data (every
    #       low-conf row also had max<0.25 or bundle) AND it violated the FAIL-OPEN contract: a
    #       low-confidence verdict must not veto a statistically-genuine edge. conf is still recorded.
    # rubric = max-axis: a STRONG signal on ANY single relatedness axis clears the bar; only an
    # all-axes-low pair OR a mechanical bundle is vetoed. The bundle flag flickers on some genuine
    # pairs (occ488), but the sticky-2-fold guard (apply_sticky) keeps a flickering pair OPEN. (_W_*
    # kept for the shadow ledger / future re-tuning as more live pairs accrue.)
    rubric = max(topical, causal, corecall)
    # BUNDLE is now ADVISORY-ONLY (recorded in the verdict, NOT an independent veto) — audit
    # re-measure 2026-06-20. Two findings forced this: (1) the co-occurrence topology mediators are
    # DEGENERATE on the live store (every promotable pair shares the same mega-sitting mediator set,
    # so "require a corroborating mediator to act on a bundle" cannot separate genuine from bundle);
    # (2) the 4B verifier FALSE-flags the genuine same-pipeline pair (vision_distill_arc::
    # vision_distill_phase10) mechanical_bundle_artifact=True at conf 0.9 while simultaneously rating
    # it topical 0.3 and affirming the shared subject. The RELIABLE discriminator is the relatedness
    # max-axis itself (genuine ~0.3 vs true-bundle/unrelated ~0.0): a real mechanical bundle has NO
    # subject relation, so rubric < MIN_SCORE already vetoes it; letting the noisy bundle flag override
    # a genuine axis signal IS the false-veto. So a bundle only "counts" when the axes ALSO show no
    # relation — which the threshold already captures. Veto purely on the relatedness floor.
    raw_veto = rubric < USEFULNESS_MIN_SCORE
    return {"topical": topical, "causal": causal, "expect_corecall": corecall,
            "mechanical_bundle_artifact": bundle, "confidence": round(conf, 4),
            "rubric_score": round(rubric, 4), "raw_veto": bool(raw_veto),
            "explanation": str(parsed.get("explanation", ""))[:300]}


def score_pair(a_name: str, a_text: str, b_name: str, b_text: str, topology: str,
               infer_fn: Optional[Callable[[str], Optional[str]]] = None,
               parse_fn: Optional[Callable[[str], Optional[dict]]] = None) -> Optional[dict]:
    """Score one pair. Returns the verdict dict, or None == ABSTAIN (model unavailable / failed /
    unparseable — fail-soft, NEVER a veto). infer_fn/parse_fn default to the cached contradiction
    judge facade (Qwen3-4B); injected in tests so no backend is needed."""
    if infer_fn is None or parse_fn is None:
        from samia.runtime import contradiction as _con
        if infer_fn is None:
            infer_fn = lambda p: _con._infer_text(p, USEFULNESS_MAX_TOKENS)  # noqa: E731
        if parse_fn is None:
            parse_fn = _con._parse_first_json_object
    prompt = build_prompt(a_name, a_text, b_name, b_text, topology)
    try:
        text = infer_fn(prompt)
    except Exception:
        return None
    if not text:
        return None
    try:
        parsed = parse_fn(text)
    except Exception:
        return None
    if parsed is None:
        return None
    return verdict_from_parsed(parsed)


def apply_sticky(streaks: dict, key: str, raw_veto: bool,
                 threshold: int = USEFULNESS_VETO_STREAK) -> tuple:
    """STICKY-FALSE-VETO guard: a veto only ACTS after `threshold` consecutive SCORED folds of
    raw_veto. ABSTAIN (raw_veto is None) PRESERVES the streak (an inference outage must not erase
    accumulated veto evidence — so "consecutive" means consecutive-among-folds-that-produced-a-verdict,
    not temporally-adjacent); an explicit non-veto (raw_veto False) RESETS the streak to 0. A single
    confident disagreement therefore fails OPEN. Mutates `streaks` in place. Returns (acts, streak)."""
    if raw_veto is None:
        return (False, streaks.get(key, 0))
    n = (streaks.get(key, 0) + 1) if raw_veto else 0
    streaks[key] = n
    return (n >= threshold, n)


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.usefulness
# Author:     code_warrior (Epiphanies v3 — Phase 1, a-priori usefulness veto)
# Project:    Asthenosphere — SAM/IA — Epiphanies (associative-edge content veto)
# Version:    0.2.0  (audit 2026-06-20: max-axis >= 0.25, fail-open; conf no longer a hard veto)
# Phase:      build — the a-priori relatedness VETO (generator-independent, subtract-only, ledger-only)
# Layer:      core (pure library — no IO; the model call is injected via infer_fn/parse_fn)
# Role:       read two linked nodes' CONTENT + the co-occurrence TOPOLOGY the cosine never saw and
#             decide whether a statistically-genuine edge genuinely relates; SUBTRACT-ONLY (narrows
#             the promote allow-set, never grows S) + LEDGER-ONLY + STICKY-2-fold + fail-soft ABSTAIN.
# Stability:  new — fail-open (inference outage => ABSTAIN, never a silent veto); anti-Goodhart
#             (rubric weights live in code, never shown to the model).
# Depends:    samia.runtime.contradiction (the cached judge facade — injected/lazy).
# Exposes:    build_prompt, verdict_from_parsed, score_pair, apply_sticky + the USEFULNESS_* consts.
# --------------------------------------------------------------------------
