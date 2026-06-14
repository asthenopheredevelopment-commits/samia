"""samia.core.context_extension.readseam — the failure/diagnosis read-seam.

Layer 1 (Owns / Depends):
    Owns:    the cross-chain failure-experience query the chainogram surfaces when
             include_failure_associations=True — the three-tier top-N resolver
             (_resolve_read_seam_top_n), the failure/diagnosis frontmatter classifier
             (_is_failure_or_diagnosis_node), and the edges.db coactivation + direct-
             match query (_query_failure_associations).
    Depends: the package config leaf (the read-seam top-N default + env, the _dt date
             stamp, os/sqlite3, the _nodes_dir helper + _read_fm reader, and the
             web_store alias _ws for the edge schema constants + decay math).

Layer 2 (What / Why):
    What: during diagnosis, callers can ask the chainogram to surface accumulated
          failure experience from the Hebbian coactivation web — prior bounty failures
          and bug diagnoses that are either directly retrieved (weight 1.0) OR
          cross-chain neighbors of the loaded nodes — ranked by weight × recency.
    Why:  this is the READ side of the failure-experience storm. Isolating it from the
          retrieval arm keeps chainogram_retrieve's core packing loop readable; the
          arm just calls _query_failure_associations and merges the additive-only key.
"""

from __future__ import annotations

from pathlib import Path

# Shared leaf — the read-seam top-N default + env, the _dt date stamp, os/sqlite3, the
# node dir + frontmatter reader, and the web_store alias for the edge schema + decay.
from .config import (
    READ_SEAM_TOP_N_DEFAULT,
    READ_SEAM_TOP_N_ENV,
    _dt,
    os,
    sqlite3,
    _ws,
    _nodes_dir,
    _read_fm,
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
    # Outcome node path: type=reference + frozen target_state (failure/partial)
    if node_type == "reference":
        target_state = (fm.get("target_state") or "").strip().lower()
        chains_raw = (fm.get("chains") or "").strip()
        has_bounty = "bounty_outcomes" in chains_raw
        has_verified = "verified_outcomes" in chains_raw
        return target_state == "frozen" and has_bounty and has_verified
    return False


def _query_failure_associations(
    memory_dir: Path,
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

        # Compute recency-adjusted score
        days = _ws._days_since(last_seen, today)
        recency_decay = max(0.0, 1.0 - _ws.EDGE_DECAY_PER_DAY * days)
        effective_score = weight * recency_decay
        scored.append((effective_score, neighbor, weight, last_seen, provenance))

    # What: sort descending by effective score, take top N.
    scored.sort(key=lambda t: -t[0])
    scored = scored[:top_n]

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
# Phase:      read-seam (cross-chain failure/diagnosis association query) — the READ
#             side of the failure-experience storm.
#             + Phase-B modularization (carved from the monolith, ZERO behavior change).
# Layer:      core (pure library, no daemon dependency)
# Role:       failure/diagnosis association query surfaced by chainogram_retrieve when
#             include_failure_associations=True.
# Stability:  stable — additive-only output key; existing callers ignore it via .get().
# ErrorModel: fail-soft — a missing/unreadable edges.db skips neighbor collection (direct
#             matches still contribute); per-node frontmatter read errors are swallowed.
# Depends:    .config (the top-N default/env, _dt, os/sqlite3, _nodes_dir, _read_fm, _ws).
# Exposes:    _query_failure_associations (the entrypoint) + _resolve_read_seam_top_n +
#             _is_failure_or_diagnosis_node (re-exported on the facade for parity; the
#             read-seam scout generalizes _query_failure_associations in mcp_server).
# Lines:      266
# --------------------------------------------------------------------------
