"""samia.core.bio.outcome — Phase 4b + Fix C: the a-posteriori OUTCOME credit ISSUE-ID causal join.

Builds {edge_key: (sign, n_conf, channel, dt_last)} for balancing.accrue(outcome_credit_of=...).

AUTO channel (Fix A + Fix C): credit the genuine pairs that co-activated in archive rows tagged with
an issue_id that LATER produced a TEST-VERIFIED SUCCESS — i.e. the co-activations of the WORK that
caused the win. This is the causal attribution (the issue_id links the bounty-work recalls to its
outcome), NOT the post-hoc "co-present in a later recall of the outcome node" proxy. The model
contributes nothing; the SIGN is the attested success, the MAGNITUDE is system-computed in balancing.

CAPTURE: hebbian_record/archive_event thread issue_id onto the coact archive row (default None ->
parity). The bounty work session passes its issue_id at recall time (operator-confirmed taggable).

HUMAN channel (review-approve / operator_confirm keep-signal) rides the SAME issue-id join once the
proposal/work issue is taggable; AUTO (capped at WEAK by the dual coefficient) ships first.

HONEYPOT (by construction): a pair never co-active in a success-issue work row earns ZERO — the join
only iterates rows whose issue_id produced a verified success. Pure-ish: lazy imports, fail-soft -> {}.
"""

from __future__ import annotations

import re
import datetime as _dt

# sources that count as genuine work co-activation (mirror balancing.GENUINE_SOURCES; None = legacy)
_GENUINE_SOURCES = (None, "genuine", "operator", "cohort_validated")


def _issue_of(node_name: str):
    """Extract the issue id from an outcome node filename: (bounty_)outcome_<issue>_<attempt>.md.
    OUT-3 (audit 2026-06-20): the issue group is non-greedy and the optional legacy `attempt_` token is
    absorbed, so legacy `..._attempt_<n>.md` names parse to the same issue id as the current
    `..._<n>.md` form (the old greedy `(.+)_(\\d+)` captured `_attempt` into the issue -> join miss)."""
    m = re.match(r"(?:bounty_)?outcome_(.+?)_(?:attempt_)?(\d+)\.md$", node_name)
    return m.group(1) if m else None


def success_issues(memory_dir) -> set:
    """Issues that produced a TEST-VERIFIED SUCCESS outcome node (Fix A) — the causal anchor for AUTO
    credit. Only co-activations tagged with one of these issues are creditable. Fail-soft -> set()."""
    out = set()
    try:
        from samia.core.context_extension import readseam as _rs
        from samia.core import frontmatter as _fm
        for p in (memory_dir / "nodes").glob("*outcome_*.md"):
            try:
                # OUT-1 (audit 2026-06-20): frontmatter.parse() returns ((fm_dict, key_order), body),
                # so parse()[0] is the (dict, order) TUPLE — NOT the dict. The old `parse()[0] or {}`
                # handed a tuple to _is_test_verified_success(fm: dict), which raised AttributeError on
                # fm.get(), silently swallowed below -> success_issues() was ALWAYS empty (the AUTO
                # outcome channel was dead, masquerading as honeypot inertness). Extract the dict.
                parsed = _fm.parse(p.read_text(encoding="utf-8"))[0]
                meta = (parsed[0] if parsed else {}) or {}
                if _rs._is_test_verified_success(meta):
                    iss = _issue_of(p.name)
                    if iss:
                        out.add(iss)
            except Exception:
                continue
    except Exception:
        pass
    return out


def _row_t(r) -> float:
    try:
        return _dt.datetime.fromisoformat(str(r.get("ts")).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def build_outcome_credit(memory_dir, rows, now_t, succ=None) -> dict:
    """Fix C issue-id CAUSAL join. For each archive row whose issue_id produced a test-verified
    success AND whose source is genuine, credit every co-active pair. Returns
    {key: (1.0, n_conf, 'AUTO', dt_last)} with n_conf = # distinct DAYS of such work (dedup) and
    dt_last = seconds since the most-recent such co-activation. Empty when there are no verified-
    success issues (-> inert). HONEYPOT holds: only success-issue work rows are ever iterated.
    succ (the success-issue set) injectable for tests."""
    if succ is None:
        succ = success_issues(memory_dir)
    if not succ:
        return {}
    acc = {}  # key -> (set_of_days, last_t)
    for r in rows or []:
        iss = r.get("issue_id")
        if not iss or iss not in succ:
            continue
        if r.get("source") not in _GENUINE_SOURCES:
            continue
        t = _row_t(r)
        day = int(t // 86400)
        nodes = sorted(set(r.get("nodes") or []))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                k = f"{nodes[i]}::{nodes[j]}"
                days, lt = acc.get(k, (None, 0.0))
                if days is None:
                    days = set()
                days.add(day)
                acc[k] = (days, max(lt, t))
    return {k: (1.0, len(days), "AUTO", max(0.0, now_t - lt)) for k, (days, lt) in acc.items()}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.outcome
# Author:     code_warrior (Epiphanies v3 — Phase 4b / Fix C, a-posteriori outcome reward)
# Project:    Asthenosphere — SAM/IA — Epiphanies (demonstrated-value reward join)
# Version:    0.2.0  (audit 2026-06-20: OUT-1 frontmatter-unpack fix; OUT-3 legacy-name regex)
# Phase:      build — the issue-id CAUSAL join: credit genuine pairs co-active in the work that
#             produced a TEST-VERIFIED success; HONEYPOT-by-construction; AUTO capped to WEAK.
# Layer:      core (impure-lite: reads outcome nodes + the coact archive; lazy imports; fail-soft -> {}).
# Role:       build {edge_key: (sign, n_conf, channel, dt_last)} for balancing.accrue(outcome_credit_of=);
#             the model contributes only the polarity sign — magnitude is system-computed in balancing.
# Stability:  new — inert until verified-success issues + issue-tagged coact rows exist; honeypot holds.
# Depends:    samia.core.context_extension.readseam (_is_test_verified_success), samia.core.frontmatter.
# Exposes:    success_issues, build_outcome_credit (+ _issue_of).
# --------------------------------------------------------------------------
