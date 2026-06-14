"""samia.core.merge_consumer.winner — pick the richer survivor, lay the
provenance edge, run the AUTO pick-winner dup merge (RESTORABLE).

Layer 1 (Owns / Depends):
    Owns:    pick_winner (rank a duplicate pair by richness), _add_provenance_edge
             (survivor->loser edge in edges.db, fail-soft), and merge_dup (the
             single-pair AUTO pick-winner merge: stamp merged_from, supersede the
             loser RESTORABLY, lay the edge, partially repair the survivor).
    Depends: .config (_ia + _PROVENANCE_KIND), .candidates (_read_fm),
             samia.core.web_store (the edge insert, lazy), samia.core.integrity
             (partial_repair, lazy + fail-soft).

Layer 2 (What / Why):
    What: the P1 pick-winner ACT path. For a pair the classifier called "dup",
          merge_dup keeps the single richest/canonical node and supersedes the
          duplicate on the already-built P3 restorable supersede path, recording a
          provenance edge and a reconciliation integrity repair on the survivor.
    Why:  Q1c/Q2c/Q4a — the duplicate bulk drains cheaply and reversibly with no
          new deletion machinery, no LLM, no salience dependency (a true duplicate
          carries the same content, so merging it loses nothing). A wrong merge is
          recoverable via ia.restore_node and self-heals via detect_wrong_deletion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import _ia, _fm, _PROVENANCE_KIND
from .candidates import _read_fm


def pick_winner(memory_dir: Path, a_id: str, b_id: str) -> tuple[str, str]:
    """Choose the richer/canonical survivor of a duplicate pair.

    What: rank by access_count, then body length, then last_access (each higher
          = richer/more-canonical), returning (survivor, loser). Ties broken
          deterministically by id so a re-run is stable.
    Why:  Q1c pick-winner — keep the single most-complete real node and
          supersede the duplicate; distinct from ia.merge's [MERGED] concat,
          which a true dup does not want (it would double the body).
    """
    fm_a, _oa, body_a = _read_fm(Path(memory_dir), a_id)
    fm_b, _ob, body_b = _read_fm(Path(memory_dir), b_id)

    def _key(fm: dict, body: str, nid: str) -> tuple:
        return (
            int(fm.get("access_count", 0) or 0),
            len(body),
            str(fm.get("last_access", "")),
            nid,  # final deterministic tiebreak
        )

    ka = _key(fm_a, body_a, a_id)
    kb = _key(fm_b, body_b, b_id)
    # Higher key = richer = survivor.
    if ka >= kb:
        return a_id, b_id
    return b_id, a_id


def _add_provenance_edge(survivor: str, loser: str,
                         db_dir: Optional[str] = None) -> dict:
    """Record a provenance edge survivor -> loser in edges.db (ref_kind=provenance).

    What: a directed row (src=survivor.md, dst=loser.md, ref_kind="provenance")
          recording the episodic->semantic lineage of the merge. Self-bootstraps
          the schema via web_store.connect; fail-soft (a store error never blocks
          the merge — the supersede archive is the reversibility guarantee, not
          this edge).
    Why:  Q4a — the surviving node links back to its merged-away source so the
          lineage is traceable + Atlas-visible. Distinct ref_kind keeps it off
          the coactivation PK lane.
    """
    try:
        from .. import web_store as _ws
    except Exception as e:
        return {"written": False, "error": str(e)}
    s = survivor if survivor.endswith(".md") else f"{survivor}.md"
    l = loser if loser.endswith(".md") else f"{loser}.md"
    try:
        conn = _ws.connect(db_dir)
        try:
            now = _ws._utc_now()
            conn.execute(
                """
                INSERT INTO edges (src_node, dst_node, ref_kind, occurrence_count,
                                   first_seen_at, last_seen_at, weight)
                VALUES (?, ?, ?, 1, ?, ?, 1.0)
                ON CONFLICT(src_node, dst_node, ref_kind) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (s, l, _PROVENANCE_KIND, now, now),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        return {"written": False, "error": str(e)}
    return {"written": True, "src": s, "dst": l, "ref_kind": _PROVENANCE_KIND}


def merge_dup(memory_dir: Path, a_id: str, b_id: str,
              db_dir: Optional[str] = None) -> dict:
    """AUTO pick-winner merge of one duplicate pair (Q2c, RESTORABLE).

    What: pick_winner -> stamp merged_from on the survivor frontmatter (append
          the loser id) -> forget_node(loser, reason="supersede",
          superseded_by=survivor) (the BUILT RESTORABLE path: full
          archive/<loser>.superseded.json + restore_node un-forgets byte-exact)
          -> lay a provenance edge survivor->loser. Returns the merge record.
    Why:  Q1c/Q2c/Q4a — the duplicate bulk drains cheaply and reversibly on the
          already-built P3 supersede path; no new deletion machinery, no LLM, no
          salience dependency. A wrong merge is recoverable via restore_node and
          self-heals via detect_wrong_deletion.
    """
    survivor, loser = pick_winner(Path(memory_dir), a_id, b_id)

    # Stamp merged_from on the survivor (the episodic->semantic lineage list).
    fm, order, body = _read_fm(Path(memory_dir), survivor)
    merged_from = fm.get("merged_from")
    merged_list = list(merged_from) if isinstance(merged_from, list) else (
        [merged_from] if merged_from else [])
    if loser not in merged_list:
        merged_list.append(loser)
    fm["merged_from"] = merged_list
    if "merged_from" not in order:
        order.append("merged_from")
    (Path(memory_dir) / "nodes" / f"{survivor}.md").write_text(
        _fm.serialize(fm, order, body), encoding="utf-8")

    # Supersede the loser RESTORABLY (P3 path: full archive + restore_node).
    forget_stats = _ia.forget_node(
        Path(memory_dir), loser, reason="supersede",
        db_dir=db_dir, superseded_by=survivor)

    prov = _add_provenance_edge(survivor, loser, db_dir=db_dir)

    # FEAT-2026-06-07 granular-recall-repaired-decay P2 — RECONCILIATION repair: the
    # surviving node was just READ + reconciled (pick-winner merge), so PARTIALLY heal
    # its integrity (anchor-first, strength < 1.0). Reconciling/merging a memory heals
    # what it touches (Q3a, partial). Additive + fail-soft; this path is reached only
    # via drain (gated by is_enabled() / ASTHENOS_TIER2_MERGE_ENABLED), so it stays
    # inert by default. The superseded loser is handled by its own restorable archive.
    try:
        from .. import integrity as _integrity
        _integrity.partial_repair(Path(memory_dir),
                                  survivor if survivor.endswith(".md")
                                  else f"{survivor}.md",
                                  trigger="reconciliation")
    except Exception:
        pass

    rec = {
        "event": "tier2_merge",
        "mode": "dup",
        "survivor": survivor,
        "loser": loser,
        "archive": forget_stats.get("superseded_archive"),
        "provenance_edge": prov.get("written", False),
    }
    _ia._log_event(Path(memory_dir), rec)
    return rec


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.merge_consumer.winner
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.merge_consumer monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       the P1 pick-winner ACT path — rank the duplicate pair, supersede
#             the loser RESTORABLY (ia.forget_node reason="supersede"), lay the
#             survivor->loser provenance edge, partially repair the survivor's
#             integrity as a reconciliation side effect.
# Stability:  stable — the carve preserved the rank keys, the merged_from stamp,
#             the supersede + edge + repair order, and the returned record shape.
# ErrorModel: _add_provenance_edge is fail-soft ({"written": False, "error": ...}
#             on any store error — the supersede archive is the reversibility
#             guarantee); the integrity partial_repair is wrapped fail-soft;
#             merge_dup itself assumes both nodes exist (the drain resolves first).
# Depends:    pathlib, typing (stdlib). .config (_ia, _PROVENANCE_KIND, _fm),
#             .candidates (_read_fm). samia.core.web_store (lazy edge insert),
#             samia.core.integrity (lazy partial_repair).
# Exposes:    pick_winner, merge_dup, _add_provenance_edge.
# Lines:      186
# --------------------------------------------------------------------------
