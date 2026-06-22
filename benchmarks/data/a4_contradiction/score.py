"""Programmatic scorer for the A4 contradiction / belief-update axis.

A4 is scored entirely on IDS — which memory the system demoted, which survived, and what
recall served — so there is NO reader/judge confound (defect D5). The pinned LLM judge is
N/A for this axis: the gold is an exact-id check, not an open-ended generation. This module
turns a list of per-case observations into the two headline metrics the design names plus
their supporting counts.

Metrics (per ``BENCHMARK_DESIGN_v1.md`` A4 row):

* **demote-correct %** — of all cases, the fraction where the belief-update was handled
  correctly end to end: the system identified the OLD claim (not the new one, not a
  distractor) as the demotion target, demoted it, and then recall served the UPDATED belief
  at rank 1 with the old claim absent. This is the positive capability number.
* **shadow-persist %** — of all cases, the fraction where the demoted OLD claim STILL
  surfaces in recall after the demotion. This is the failure number: a "shadow" of the
  superseded belief that leaks back into retrieval. Lower is better; 0% is ideal.

Supporting counts (reported for transparency, not headline):

* **detect %** — the embedding supersession finder paired not_X's claim with the correct
  OLD node (the contradiction signal was found at all).
* **pick-correct %** — the deterministic loser/winner rule chose the OLD claim as the
  loser and the NEW claim as the winner.

Every input is an id comparison computed by the task; this scorer does no recall itself, so
it is pure and trivially re-runnable over saved raw outputs.
"""

from __future__ import annotations

from typing import Any


def _pct(num: int, den: int) -> float:
    """Percentage with a guarded denominator (0 cases -> 0.0)."""
    return round(100.0 * num / den, 2) if den else 0.0


def score_cases(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case observations into the A4 metrics.

    Parameters
    ----------
    observations:
        One dict per case, each with the boolean keys the task records:
          ``detected``       — supersession finder paired not_X with the correct OLD id.
          ``pick_correct``   — loser==OLD and winner==NEW under the deterministic rule.
          ``demote_correct`` — OLD demoted AND recall served NEW at rank 1 AND OLD absent.
          ``shadow_persist`` — the demoted OLD id still appears in post-demotion recall.
        plus ``case_id`` for the audit trail.

    Returns
    -------
    dict with the headline metrics, supporting counts, and per-case pass/fail, suitable for
    direct JSON serialization into the results file.
    """
    n = len(observations)
    detected = sum(1 for o in observations if o.get("detected"))
    pick_correct = sum(1 for o in observations if o.get("pick_correct"))
    demote_correct = sum(1 for o in observations if o.get("demote_correct"))
    shadow_persist = sum(1 for o in observations if o.get("shadow_persist"))

    return {
        "axis": "a4_contradiction",
        "n_cases": n,
        # Headline metrics named by the design.
        "demote_correct_pct": _pct(demote_correct, n),
        "shadow_persist_pct": _pct(shadow_persist, n),
        # Supporting counts.
        "detect_pct": _pct(detected, n),
        "pick_correct_pct": _pct(pick_correct, n),
        "counts": {
            "detected": detected,
            "pick_correct": pick_correct,
            "demote_correct": demote_correct,
            "shadow_persist": shadow_persist,
        },
        # Scoring is programmatic on ids; the pinned judge is not used for A4.
        "judge_applied": False,
        "judge_note": (
            "A4 gold is an exact-id check (which memory is demoted / survives / served); "
            "scored programmatically. The pinned LLM judge is reserved for open-ended "
            "axes and is N/A here."
        ),
        "per_case": [
            {
                "case_id": o.get("case_id"),
                "detected": bool(o.get("detected")),
                "pick_correct": bool(o.get("pick_correct")),
                "demote_correct": bool(o.get("demote_correct")),
                "shadow_persist": bool(o.get("shadow_persist")),
            }
            for o in observations
        ],
    }
