"""samia.core.context_extension.readseam — cross-chain failure/diagnosis read-seam.

Layer 1 (Owns / Depends):
    Owns:    the read side of the failure-experience storm — _resolve_read_seam_top_n
             (the call-site > env > constant top-N override), _is_failure_or_diagnosis_node
             (the frontmatter classifier for failure-outcome + bug-diagnosis nodes), and
             _query_failure_associations (the edges.db coactivation-neighbor + direct-match
             query that surfaces accumulated failure experience during diagnosis).
    Depends: .config (the READ_SEAM_TOP_N_DEFAULT / READ_SEAM_TOP_N_ENV constants, the
             aliased _ws web_store for the edges.db path + decay constants, _dt for
             today's date, and the shared _nodes_dir / _read_fm helpers); stdlib os /
             sqlite3.

Layer 2 (What / Why):
    What: given the nodes a retrieval just loaded, surface the failure/diagnosis nodes
          that ARE among them (direct matches, weight 1.0) PLUS their cross-chain
          coactivation neighbors from the Hebbian web, deduped + recency-decayed +
          ranked, top-N.
    Why:  closes the read side of the failure storm — during diagnosis the caller sees
          prior failures Hebbian-associated with the loaded context. The retrieval arm
          calls _query_failure_associations through the additive
          include_failure_associations key; mcp_server references it; both reach it
          through the package facade.
"""

from __future__ import annotations

import os
import sqlite3

# Single-owned constants + aliased deps + shared helpers, reached THROUGH the config leaf.
from .config import (
    READ_SEAM_TOP_N_DEFAULT,
    READ_SEAM_TOP_N_ENV,
    _dt,
    _nodes_dir,
    _read_fm,
    _ws,
)


def _resolve_read_seam_top_n(top_n: int | None) -> int:
    """Resolve the effective top-N for failure associations.

    What: returns the caller's explicit value, else the env var, else the default.
    Why: three-tier override (call-site > env > constant) so N is configurable
         without code changes per the read-seam spec.
    """
    if top_n is not None:
        return max(0, top_n)
    env_val = os.environ.get(READ_SEAM_TOP_N_ENV)
    if env_val is not None:
        try:
            return max(0, int(env_val))
        except (ValueError, TypeError):
            pass
    return READ_SEAM_TOP_N_DEFAULT


def _is_failure_or_diagnosis_node(fm: dict) -> bool:
    """Return True if frontmatter marks a failure-outcome or bug-diagnosis node.

    What: checks two disjoint node types produced by the storm's write-side:
      (a) outcome nodes from opencode_drain — type=reference, chains includes
          both 'bounty_outcomes' and 'verified_outcomes', body outcome in
          ('failure', 'partial') signaled by target_state=frozen.
      (b) bug nodes from bug_records — type=bug, any status except 'wont-fix'
          (wont-fix = dismissed, not actionable failure experience).
    Why: these are the exact frontmatter fields the scout verified; matching on
         them avoids body parsing and stays robust to format drift.
    """
    node_type = (fm.get("type") or "").strip().lower()
    # Bug node path: type=bug, status not dismissed
    if node_type == "bug":
        status = (fm.get("status") or "").strip().lower()
        return status != "wont-fix"
    # Outcome node path: type=reference + frozen target_state (failure/partial). A test-verified
    # SUCCESS (FEAT-2026-06-20 Fix A) is explicitly EXCLUDED — it is a positive outcome, not a
    # failure — guarding against any chain-name substring overlap with 'verified_outcomes'.
    if node_type == "reference":
        if (fm.get("outcome_polarity") or "").strip().lower() == "success":
            return False
        target_state = (fm.get("target_state") or "").strip().lower()
        chains_raw = (fm.get("chains") or "").strip()
        has_bounty = "bounty_outcomes" in chains_raw
        has_verified = "verified_outcomes" in chains_raw
        return target_state == "frozen" and has_bounty and has_verified
    return False


def _is_test_verified_success(fm: dict) -> bool:
    """FEAT-2026-06-20 Fix A: a TEST-EXECUTION-attested SUCCESS outcome node — the un-gameable
    positive 'helped do work' signal the outcome-reward AUTO channel credits. Distinct from the
    failure path: keyed on verified_by + outcome_polarity (set by opencode_drain.materialize_node),
    frozen so it persists/joins. Chain-name independent (robust to the substring collision)."""
    if (fm.get("type") or "").strip().lower() != "reference":
        return False
    return ((fm.get("verified_by") or "").strip().lower() == "test_execution"
            and (fm.get("outcome_polarity") or "").strip().lower() == "success"
            and (fm.get("target_state") or "").strip().lower() == "frozen")


def _query_failure_associations(
    memory_dir,
    loaded_nodes: list[str],
    top_n: int,
    db_dir: str | None = None,
) -> list[dict]:
    """Query failure/diagnosis associations: direct matches + cross-chain neighbors.

    What: (1) identifies loaded_nodes that ARE failure/diagnosis nodes themselves
      (direct matches, weight=1.0), then (2) reads edges.db (read-only) for
      coactivation neighbors of loaded_nodes and filters those to failure/diagnosis
      nodes. Merges both sources, deduplicates by node name (highest weight wins),
      ranks by weight x recency, returns the top-N associations.
    Why: this is the read-seam — surfacing accumulated failure experience from the
      Hebbian web during diagnosis. Direct matches are the most query-relevant
      failures (they ARE the loaded context) and were previously excluded by the
      neighbor-only filter, causing the read-seam to miss a bounty's own prior
      failures even when they were in loaded_nodes.
    """
    if top_n <= 0 or not loaded_nodes:
        return []

    today = _dt.date.today()
    node_set = set(loaded_nodes)
    nodes_dir = _nodes_dir(memory_dir)

    # What: locate edges.db; fall back to web_store default.
    # Why: tests pass db_dir to use a temp store; production uses the default.
    if db_dir:
        db_path = os.path.join(db_dir, "edges.db")
    else:
        db_path = _ws._db_path(None)

    # What: collect all coactivation neighbors of loaded nodes from edges.db.
    # Why: UNION both src/dst directions because edges are stored in
    #   canonical (_order) form — a loaded node could be on either side.
    #   If edges.db is missing or unreadable, skip neighbor collection
    #   (direct matches may still contribute).
    # NeighborScan — What: open edges.db read-only and, per loaded node, union its
    #     COACTIVATION edges in BOTH directions (src->dst and dst->src), dropping any
    #     neighbor that is itself a loaded node.
    neighbor_rows: list[tuple[str, float, str]] = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError:
            conn = None
        if conn is not None:
            try:
                for node_name in node_set:
                    for row in conn.execute(
                        "SELECT dst_node, weight, last_seen_at FROM edges "
                        "WHERE ref_kind=? AND src_node=?",
                        (_ws.COACTIVATION, node_name),
                    ).fetchall():
                        if row[0] not in node_set:
                            neighbor_rows.append(row)
                    for row in conn.execute(
                        "SELECT src_node, weight, last_seen_at FROM edges "
                        "WHERE ref_kind=? AND dst_node=?",
                        (_ws.COACTIVATION, node_name),
                    ).fetchall():
                        if row[0] not in node_set:
                            neighbor_rows.append(row)
            finally:
                conn.close()
    # NeighborScan — Why: edges are stored in canonical _order form, so a loaded node can
    #     sit on either side; the read-only handle + a missing-db skip keep the seam non-fatal.

    # What: collect loaded_nodes that ARE failure/diagnosis nodes (direct matches).
    # Why: the neighbor loop above EXCLUDES loaded_nodes by design (line 220/227),
    #   but a directly-retrieved node that IS a failure (e.g. the bounty's own prior
    #   attempts) is the most query-relevant failure association and must be
    #   surfaced. Weight = 1.0 (strongest, since they ARE query context).
    direct_matches: list[tuple[str, float, str]] = []  # (node_name, weight, last_seen)
    for node_name in node_set:
        p = nodes_dir / node_name
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        try:
            fm, _ = _read_fm(p)
        except Exception:
            continue
        if not _is_failure_or_diagnosis_node(fm):
            continue
        # What: derive last_seen from frontmatter last_access, fall back to today.
        # Why: direct matches have no edge row; last_access is the closest analog
        #   to the edge's last_seen_at for recency scoring.
        la = (fm.get("last_access") or "").strip()
        last_seen = la if la else today.isoformat()
        direct_matches.append((node_name, 1.0, last_seen))

    if not neighbor_rows and not direct_matches:
        return []

    # What: merge direct matches + neighbor rows, deduplicate by node name,
    #   keeping the highest weight per node.
    # Why: a node could appear as both a direct match (weight 1.0) and a
    #   cross-chain neighbor; dedup ensures one entry with the best weight.
    #   Direct matches at weight 1.0 will dominate unless a neighbor has
    #   higher weight (shouldn't happen, but the max keeps it correct).
    best_by_node: dict[str, tuple[float, str, str]] = {}
    #   value: (weight, last_seen, provenance)
    for node_name, weight, last_seen in direct_matches:
        best_by_node[node_name] = (weight, last_seen, "direct_match")
    for neighbor, weight, last_seen in neighbor_rows:
        prev = best_by_node.get(neighbor)
        if prev is None or weight > prev[0]:
            best_by_node[neighbor] = (weight, last_seen, "cross_chain")

    # What: compute effective score = weight x recency_decay, filter to failure nodes.
    # Why: recency-adjusted ranking per web_store's decay formula ensures stale
    #   failure associations sink below fresh ones.
    scored: list[tuple[float, str, float, str, str]] = []
    for neighbor, (weight, last_seen, provenance) in best_by_node.items():
        # Read frontmatter to check failure/diagnosis type
        p = nodes_dir / neighbor
        if not p.suffix:
            p = p.with_suffix(".md")
        if not p.exists():
            continue
        try:
            fm, _ = _read_fm(p)
        except Exception:
            continue
        if not _is_failure_or_diagnosis_node(fm):
            continue

        # Rank — What: score each surviving failure node as weight × linear recency-decay
        #     (days since last_seen), then sort descending and keep the top-N.
        days = _ws._days_since(last_seen, today)
        recency_decay = max(0.0, 1.0 - _ws.EDGE_DECAY_PER_DAY * days)
        effective_score = weight * recency_decay
        scored.append((effective_score, neighbor, weight, last_seen, provenance))

    # What: sort descending by effective score, take top N.
    scored.sort(key=lambda t: -t[0])
    scored = scored[:top_n]
    # Rank — Why: a stale failure association is less actionable than a fresh one, so recency
    #     sinks it below recent mistakes; the top-N bound caps what the diagnosis caller sees.

    results: list[dict] = []
    for eff_score, neighbor, weight, last_seen, provenance in scored:
        p = nodes_dir / neighbor
        if not p.suffix:
            p = p.with_suffix(".md")
        fm, _ = _read_fm(p)
        node_type = (fm.get("type") or "").strip().lower()
        # What: classify the failure kind for the caller.
        # Why: callers need to know if this is a prior bounty failure or a bug
        #   diagnosis without re-reading frontmatter themselves.
        if node_type == "bug":
            failure_kind = "bug_diagnosis"
        else:
            failure_kind = "bounty_failure"
        results.append({
            "node": neighbor,
            "effective_score": round(eff_score, 4),
            "weight": round(weight, 4),
            "last_seen_at": last_seen,
            "failure_kind": failure_kind,
            "provenance": provenance,
            "name": fm.get("name", ""),
            # What: carry the failure REASON (from the node description) so callers
            #   can act on WHAT failed, not just that a failure occurred.
            # Why: recalling a failure without its reason is signal-free — the whole
            #   point of read-seam is to surface the specific prior mistake to avoid.
            "reason": (fm.get("description") or "").strip(),
        })
    return results


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.context_extension.readseam
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase-B modularization — the cross-chain failure/diagnosis read-seam
#             carved from the monolith with ZERO behavior change.
# Layer:      core (pure library, no daemon dependency)
# Role:       the read side of the failure-experience storm — direct-match + cross-chain
#             coactivation-neighbor failure surfacing during diagnosis.
# Stability:  stable — additive-only; the retrieval arm consults it through the optional
#             include_failure_associations key, so existing callers are unaffected.
# ErrorModel: fail-soft — a missing/unreadable edges.db skips neighbor collection (direct
#             matches still contribute); per-node frontmatter read errors are swallowed.
# Depends:    .config (READ_SEAM_TOP_N_* constants, _dt, _ws, _nodes_dir, _read_fm);
#             stdlib os / sqlite3.
# Exposes:    _resolve_read_seam_top_n / _is_failure_or_diagnosis_node /
#             _query_failure_associations (all private; re-exported on the facade — the
#             read-seam tests reach _is_failure_or_diagnosis_node + _resolve_read_seam_top_n
#             through it, and mcp_server references _query_failure_associations).
# Lines:      276
# Updated:    2026-06-14
# Status:     active
# --------------------------------------------------------------------------
