"""samia.core.merge_consumer — Tier-2 merge consumer P1+P2+P3 (pick-winner
dup-merge + LLM-synthesized abstraction + salience guard, operator-gated).

Layer 1 (Owns / Depends):
    Owns:    the DRAIN half of the Tier-2 abstractive consolidation backlog.
             P1 pick-winner duplicate merge: reads the surfacer's
             .consolidation_candidates.json, classifies each pair, AUTO-merges
             the true-duplicate (high-similarity) pairs by picking the richer
             survivor and superseding the loser via the RESTORABLE P3 path, lays
             a provenance edge survivor->loser, and removes drained pairs from
             the candidate file so the backlog shrinks (and REM can reach REST).
             P2 LLM-synthesized abstraction (OPERATOR-GATED): for the below-bar
             distinct-but-overlapping 'abstract' pairs P1 queued, calls the
             existing local-inference synthesis entrypoint to PROPOSE (never
             apply) a new higher-level semantic node; only confirm_abstraction
             creates the node + supersedes both sources RESTORABLY (+ provenance
             edges); reject_abstraction changes nothing. Owns the
             ASTHENOS_TIER2_MERGE_ENABLED enable flag read, the dup-vs-abstract
             classification bar, and the merge_candidates.jsonl proposed/confirm/
             reject lifecycle. P3 salience guard: CONSUMES
             bio.salience_merge_guard so a DISTINCT high-salience source is
             SURFACED (status="guarded"), never auto-abstracted-away (a true
             duplicate is exempt -> P1 dup merge unchanged). Owns NO deletion
             machinery, NO surfacer, NO model loader (reuses the judge's inference
             path), NO salience DEFINITION (defined in bio; consumed here).
    Depends: samia.core.consolidation (the .consolidation_candidates.json schema
             + jaccard/shingles primitive), samia.core.ia (forget_node
             reason="supersede" RESTORABLE archive + restore_node),
             samia.core.frontmatter (parse/serialize — winner selection by
             frontmatter; abstraction node write),
             samia.core.web_store (the provenance edge insert),
             samia.runtime.contradiction (the cosine finder, used live when a
             vector index exists; jaccard fallback otherwise; AND the P2 LLM
             synthesis entrypoint synthesize_node — the SAME judge backend).

Layer 2 (What / Why):
    What: P1 of FEAT-2026-06-07-memory-tier2-merge-consumer-v01. The
          consolidation SURFACER (already built, REM subscriber priority 20)
          writes ~600 near-dup pairs to .consolidation_candidates.json but
          NOTHING merged them, so REM's work_remaining stayed true forever and
          the surfacer re-audited the same pairs every cycle. This module is the
          missing DRAIN: for each pair above a HIGH similarity bar it picks the
          richer/canonical winner, supersedes the loser RESTORABLY (full
          archive/<id>.superseded.json + restore_node), records a provenance
          edge, and removes the pair from the candidate file. Below-bar
          distinct-but-overlapping pairs are LEFT for P2's gated LLM-abstraction
          (recorded "abstract", never merged here).
    Why:  Q1c TIERED (dups -> pick-winner) + Q2c AUTO (the dup case is
          reversible/low-risk, fires automatically above the bar) + Q3a CONSUME
          (drain the surfacer's scored output so the backlog shrinks) + Q4a
          RESTORABLE (reuse the P3 archive + restore + provenance edge). This is
          POSITIVE consolidation (compress what is right) reusing P3's NEGATIVE-
          consolidation deletion primitive. Pick-winner needs no salience guard
          (duplicates carry the same content); the guard is a P2/P3 concern.

PRODUCE-ONLY: importing this module does nothing. drain() is a no-op unless
ASTHENOS_TIER2_MERGE_ENABLED=1 (default OFF), mirroring the contradiction passive
sweep posture — inert until the operator enables it + restarts the daemon. No
thread, no timer, no live mutation on import.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from . import consolidation as _con
from . import frontmatter as _fm
from . import ia as _ia

# _DUP_MERGE_COSINE -- What: the HIGH cosine bar above which a pair is a TRUE
#   duplicate (Q2c AUTO pick-winner). Why: only the near-exact bulk merges
#   automatically; distinct-but-overlapping pairs (below the bar) are left for
#   P2's gated LLM-abstraction. Env-tunable; 0.92 per the proposal default.
_DUP_MERGE_COSINE: float = float(
    os.environ.get("ASTHENOS_DUP_MERGE_COSINE", "0.92")
)

# _DUP_MERGE_JACCARD -- What: the dup bar on the surfacer's lexical jaccard
#   score, used when no vector index exists so cosine is unavailable. Why: the
#   surfacer scores pairs by jaccard (consolidation.py); when the embedding
#   index is absent the consumer must still classify deterministically. 0.85 is
#   a HIGH lexical-overlap bar (true near-duplicate prose), distinct from the
#   surfacer's 0.15 surfacing knee. Env-tunable.
_DUP_MERGE_JACCARD: float = float(
    os.environ.get("ASTHENOS_DUP_MERGE_JACCARD", "0.85")
)

_CANDIDATE_FILE = ".consolidation_candidates.json"
_ABSTRACT_LOG = "merge_candidates.jsonl"  # under biomimetic/ — P2's pending queue
_PROVENANCE_KIND = "provenance"

# _ENQUEUED_LOG -- What: a tiny append-only done-set of candidate_ids already fed
#   to the fact-extract queue (one {"candidate_id","ts"} JSONL line per pair).
# Why: BUG-2026-06-11 runaway loop (enqueue side) — the surfacer re-presents the
#   SAME ~52 abstract pairs every REM cycle, so _enqueue_abstract_pair re-enqueued
#   them every drain (+~1,500 lines/hour to .fact_extract_queue.jsonl). The
#   candidate_id presence in merge_candidates.jsonl already dedups the recorded
#   pair; this is the belt-and-suspenders ledger so a pair is enqueued AT MOST
#   ONCE EVER even if its P2 record is later resolved/rewritten out of the
#   unresolved view. Under biomimetic/.
_ENQUEUED_LOG = "fact_extract_enqueued.jsonl"

# _GUARDED_STATUS -- What: the abstraction-candidate status set when the P3 salience
#   guard fires on a DISTINCT high-salience source. Why: D6 effect (iii) / Q5a — a
#   distinct high-salience memory must NOT be auto-abstracted-away; it is SURFACED for
#   operator review (a terminal, listable status the operator resolves via confirm/
#   reject), never silently superseded. Distinct from "pending"/"proposed" so it is
#   visible as "needs review, salience-protected" and is not re-synthesized.
_GUARDED_STATUS = "guarded"


def is_enabled() -> bool:
    """Live read of the ASTHENOS_TIER2_MERGE_ENABLED master switch.

    What: True iff the operator has set ASTHENOS_TIER2_MERGE_ENABLED=1.
    Why:  Q5a — P1 is double-gated (REM + this flag), inert by default. A live
          read (not an import-time constant) lets a test or the daemon flip it
          without re-import, mirroring contradiction.is_enabled().
    """
    return os.environ.get("ASTHENOS_TIER2_MERGE_ENABLED", "0") == "1"


# ---------------------------------------------------------------------------
# Candidate loading + pair resolution
# ---------------------------------------------------------------------------


def load_candidates(memory_dir: Path) -> list[dict]:
    """Read the surfacer's .consolidation_candidates.json candidate list.

    What: returns the ``candidates`` list ({chain, a_addr, a_file, b_addr,
          b_file, similarity}) the consolidation surfacer wrote. Empty list if
          the file is missing/unreadable (nothing surfaced => nothing to drain).
    Why:  Q3a — the consumer CONSUMES the existing surfacer output; it never
          re-audits the chains (that is the surfacer's job at priority 20).
    """
    p = Path(memory_dir) / _CANDIDATE_FILE
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    cands = payload.get("candidates") if isinstance(payload, dict) else None
    return list(cands) if isinstance(cands, list) else []


def _node_id_for_file(rel_file: str) -> str:
    """Map a candidate's ``a_file``/``b_file`` (nodes/<id>.md) to a node id.

    What: strip the nodes/ prefix and .md suffix to the bare id forget_node /
          frontmatter callers use.
    Why:  the surfacer records the relative path; ia.forget_node + the node file
          readers want the stem.
    """
    name = rel_file
    if name.startswith("nodes/"):
        name = name[len("nodes/"):]
    if name.endswith(".md"):
        name = name[:-3]
    return name


def _resolve_pair(memory_dir: Path, cand: dict) -> Optional[tuple[str, str]]:
    """Map a candidate to a live (a_id, b_id) pair, or None if either is gone.

    What: resolves a_file/b_file to ids and confirms both nodes still exist in
          nodes/. Returns None when either node is already merged/forgotten —
          the drain advancing (the pair is stale).
    Why:  a previous cycle (or P3 supersede, or the surfacer's own churn) may
          have already removed one side; merging a phantom would error.
    """
    nodes = Path(memory_dir) / "nodes"
    a_id = _node_id_for_file(str(cand.get("a_file", "")))
    b_id = _node_id_for_file(str(cand.get("b_file", "")))
    if not a_id or not b_id or a_id == b_id:
        return None
    if not (nodes / f"{a_id}.md").exists() or not (nodes / f"{b_id}.md").exists():
        return None
    return a_id, b_id


# ---------------------------------------------------------------------------
# Classification (dup vs abstract) — HIGH bar = true duplicate (P1 acts)
# ---------------------------------------------------------------------------


def _cosine_for_pair(memory_dir: Path, a_id: str, b_id: str) -> Optional[float]:
    """Cosine similarity between the two nodes via the existing finder, or None.

    What: read a's text, run contradiction.find_supersession_candidates scoped
          to [b_id], and return b's cosine score. None when the embedding infra
          (vector index) is unavailable — the caller then falls back to jaccard.
    Why:  Q1c/Q3a — the live path classifies on cosine (the proposal's bar); the
          jaccard fallback keeps P1 deterministic + stub-free when no index
          exists (tests, cold trees). Reuses the cosine finder; reinvents none.
    """
    try:
        from samia.runtime import contradiction as _contra
    except Exception:
        return None
    nodes = Path(memory_dir) / "nodes"
    a_path = nodes / f"{a_id}.md"
    if not a_path.exists():
        return None
    try:
        a_text = a_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        hits = _contra.find_supersession_candidates(
            a_text, scope_nodes=[b_id], memory_dir=Path(memory_dir),
        )
    except Exception:
        return None
    if not hits:
        return None
    b_fname = f"{b_id}.md"
    for h in hits:
        nid = str(h.get("node_id", ""))
        nid_md = nid if nid.endswith(".md") else f"{nid}.md"
        if nid_md == b_fname:
            return float(h.get("score", 0.0))
    return None


def classify_pair(memory_dir: Path, a_id: str, b_id: str,
                  candidate_similarity: Optional[float] = None) -> str:
    """Classify a surfaced pair as "dup" (P1 acts) or "abstract" (left for P2).

    What: return "dup" when the pair clears the HIGH duplicate bar, else
          "abstract". Prefers cosine (>= _DUP_MERGE_COSINE) when a vector index
          exists; otherwise falls back to the surfacer's lexical jaccard score
          (>= _DUP_MERGE_JACCARD). P1 acts ONLY on "dup"; "abstract" pairs are
          recorded for P2's gated LLM-abstraction.
    Why:  Q1c TIERED + Q2c AUTO — the near-exact bulk auto-merges (reversible,
          low-risk); distinct-but-overlapping pairs need a real synthesis, which
          is P2's gated job, not P1's.
    """
    cos = _cosine_for_pair(Path(memory_dir), a_id, b_id)
    if cos is not None:
        return "dup" if cos >= _DUP_MERGE_COSINE else "abstract"
    # Fallback: the surfacer's jaccard score (deterministic, no embedder).
    sim = candidate_similarity
    if sim is None:
        a_body = _con.load_node_body(Path(memory_dir), f"nodes/{a_id}.md") or ""
        b_body = _con.load_node_body(Path(memory_dir), f"nodes/{b_id}.md") or ""
        sim = _con.jaccard(_con.shingles(a_body), _con.shingles(b_body))
    return "dup" if float(sim) >= _DUP_MERGE_JACCARD else "abstract"


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


def _read_fm(memory_dir: Path, node_id: str) -> tuple[dict, list[str], str]:
    """Parse a node's (frontmatter, key_order, body); empty fm if unparsable."""
    p = Path(memory_dir) / "nodes" / f"{node_id}.md"
    raw = p.read_text(encoding="utf-8")
    parsed, body = _fm.parse(raw)
    if parsed is None:
        return {}, [], raw
    fm, order = parsed
    return fm, order, body


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


# ---------------------------------------------------------------------------
# Provenance edge (survivor -> loser source)
# ---------------------------------------------------------------------------


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
        from . import web_store as _ws
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


# ---------------------------------------------------------------------------
# The single-pair pick-winner merge (AUTO, RESTORABLE)
# ---------------------------------------------------------------------------


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
        from . import integrity as _integrity
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


def _candidate_id(a_id: str, b_id: str) -> str:
    """Stable id for an abstract candidate pair (order-independent).

    What: "abs-<sha1(sorted(a,b))[:12]>" — deterministic so the same pair maps
          to the same candidate across drains (no duplicate proposals) and so
          confirm/reject can address it without an opaque counter.
    Why:  P2 confirm/reject (and the MCP surface) need a single addressable key
          per pair; deriving it from the sorted ids keeps it stable + dedup-able.
    """
    key = "|".join(sorted((str(a_id), str(b_id))))
    return "abs-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _record_abstract(memory_dir: Path, a_id: str, b_id: str,
                     similarity: Any) -> None:
    """Queue a distinct-but-overlapping pair for P2's gated LLM-abstraction.

    What: append {candidate_id,a,b,similarity,mode:"abstract",status:"pending"}
          to biomimetic/merge_candidates.jsonl, unless the pair is already
          queued (dedup by candidate_id). P1 does NOT merge these.
    Why:  Q2c — abstractions create NEW content + can lose nuance, so they are
          GATED (P2), not auto-merged. P1 only records them so the drain can
          dispatch the whole candidate set in one pass; P2 later synthesizes a
          proposed abstraction over the pending entries.
    """
    bio_dir = Path(memory_dir) / "biomimetic"
    bio_dir.mkdir(parents=True, exist_ok=True)
    cid = _candidate_id(a_id, b_id)
    # Dedup: do not re-queue a pair already recorded (any status).
    for existing in list_abstraction_candidates(Path(memory_dir),
                                                 unresolved_only=False):
        if existing.get("candidate_id") == cid:
            return
    rec = {
        "candidate_id": cid,
        "a": a_id, "b": b_id, "similarity": similarity,
        "mode": "abstract", "status": "pending", "ts": _ia._now_iso(),
    }
    with (bio_dir / _ABSTRACT_LOG).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _enqueued_candidate_ids(memory_dir: Path) -> set[str]:
    """Read the done-set of candidate_ids already fed to the fact-extract queue.

    What: return the set of candidate_ids recorded in
          biomimetic/fact_extract_enqueued.jsonl. Fail-soft on a missing/partly-
          corrupt file (skips unparseable lines, returns what it can).
    Why:  BUG-2026-06-11 — the enqueue-side belt. Read once per drain pass and
          consulted by _enqueue_abstract_pair so a pair is enqueued at most once
          ever, independent of the P2 record's status churn.
    """
    out: set[str] = set()
    p = Path(memory_dir) / "biomimetic" / _ENQUEUED_LOG
    if not p.exists():
        return out
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = rec.get("candidate_id")
                if cid:
                    out.add(cid)
    except OSError:
        return out
    return out


def _mark_enqueued(memory_dir: Path, candidate_id: str) -> None:
    """Append one {candidate_id, ts} line to the enqueue done-set (PRODUCER).

    What: O_APPEND single-write a {"candidate_id","ts"} record to
          biomimetic/fact_extract_enqueued.jsonl (atomic for a short line, same
          posture as fact_extractor.enqueue_for_extraction).
    Why:  BUG-2026-06-11 — record that this pair has been fed to the fact-extract
          queue so it is never re-enqueued. Fail-open: a ledger write error must
          never disturb the drain.
    """
    bio_dir = Path(memory_dir) / "biomimetic"
    bio_dir.mkdir(parents=True, exist_ok=True)
    rec = {"candidate_id": candidate_id, "ts": _ia._now_iso()}
    p = bio_dir / _ENQUEUED_LOG
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(rec) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _enqueue_abstract_pair(memory_dir: Path, a_id: str, b_id: str) -> None:
    """FEAT-2026-06-10 P1 — enqueue a distinct 'abstract' pair for fact extraction.

    What: read both node bodies, concatenate them with a separator, and (when
          ASTHENOS_FACT_EXTRACT_ENABLED=1) append ONE
          {text, source:"<a>.md+<b>.md", enqueued_by:"merge_abstract"} record to
          <mem>/.fact_extract_queue.jsonl via fact_extractor.enqueue_for_extraction.
          DEDUP (BUG-2026-06-11): skip entirely when this pair's candidate_id is
          already present — either as a recorded merge candidate
          (merge_candidates.jsonl, mirroring _record_abstract) OR in the
          fact_extract_enqueued.jsonl done-set — so a pair is enqueued at most
          once ever.
    Why:  Q1d — the merge 'abstract' branch is the second producer feed (the freeze
          path is the first). Distinct-but-overlapping pairs carry cross-source
          gist worth distilling into ADDITIVE atomic facts (NOT the gated
          abstractive MERGE — that is P2's separate machinery and deletes nothing).
          The surfacer re-presents the SAME ~52 pairs every REM cycle; without the
          candidate_id dedup this re-enqueued them every drain (+~1,500 lines/hour
          to the queue). The candidate_id done-set is the per-pair-once gate: in
          the live drain _record_abstract runs immediately BEFORE this call, so the
          pair's candidate_id is already in merge_candidates.jsonl by now — which
          is why the done-set ledger (not a merge_candidates presence check, which
          would always be true and suppress the FIRST enqueue) is the right gate.
          Gated on the flag (flag-off = no write) and fully fail-OPEN: an enqueue
          error must never disturb the drain. The producer's own sentinel guard
          still applies (an eroded body is refused at the helper).
    """
    try:
        from . import fact_extractor as _fx
        if not _fx.fact_extract_enabled():
            return
        cid = _candidate_id(a_id, b_id)
        # Dedup gate — already fed to the queue (this re-presentation, or any
        # prior REM cycle). Mirrors how _record_abstract dedups by candidate_id,
        # but keyed off the dedicated enqueue done-set so the FIRST enqueue still
        # fires even though _record_abstract has just written this candidate_id to
        # merge_candidates.jsonl.
        if cid in _enqueued_candidate_ids(Path(memory_dir)):
            return
        _fa, _oa, body_a = _read_fm(Path(memory_dir), a_id)
        _fb, _ob, body_b = _read_fm(Path(memory_dir), b_id)
        a_fname = a_id if a_id.endswith(".md") else f"{a_id}.md"
        b_fname = b_id if b_id.endswith(".md") else f"{b_id}.md"
        text = f"{body_a}\n\n---\n\n{body_b}"
        result = _fx.enqueue_for_extraction(
            Path(memory_dir), text, f"{a_fname}+{b_fname}", "merge_abstract")
        # Only mark the pair done once it actually reached the queue (the sentinel
        # guard may have refused an eroded body — leave it un-marked so a later
        # un-eroded body can still be enqueued).
        if isinstance(result, dict) and result.get("enqueued"):
            _mark_enqueued(Path(memory_dir), cid)
    except Exception:
        pass  # fail-open: extraction enqueue must never block the merge drain


# ---------------------------------------------------------------------------
# P3 — salience guard (D6 effect iii / Q5a — CONSUMED here, DEFINED in bio)
# ---------------------------------------------------------------------------


def _salience_guards_pair(memory_dir: Path, a_id: str, b_id: str,
                          is_duplicate: bool = False) -> Optional[str]:
    """Return the source id that trips the salience merge guard, or None.

    What: consult bio.salience_merge_guard on EACH source of a DISTINCT pair; if
          either is a high-salience distinct memory the guard protects, return
          that source id (so the caller surfaces the pair instead of acting).
          Returns None when neither trips, when the guard is unavailable
          (bio without salience_merge_guard — P1/P2 ship before Tier-1's salience
          field lands), or when is_duplicate=True (a true duplicate is exempt —
          merging it loses nothing, so the guard never fires for it).
    Why:  Q5a / D6 effect (iii) — do NOT auto-merge/auto-abstract a high-salience
          DISTINCT memory; surface it for operator review instead. Wired behind a
          hasattr guard so the merge consumer runs fully before the salience field
          exists and activates with no re-sequence once Tier-1 Phase 5 lands. Pure
          read; mutates nothing.
    """
    if is_duplicate:
        return None
    try:
        from . import bio as _bio
    except Exception:
        return None
    guard = getattr(_bio, "salience_merge_guard", None)
    if guard is None:
        return None
    for src in (a_id, b_id):
        try:
            if guard(Path(memory_dir), src, is_duplicate=False):
                return src
        except Exception:
            continue
    return None


def _record_guarded(memory_dir: Path, a_id: str, b_id: str,
                    similarity: Any, guarded_source: str) -> None:
    """Surface a salience-guarded distinct pair for operator review (P3).

    What: write/flip the pair's merge_candidates.jsonl entry to
          status=_GUARDED_STATUS, recording which source tripped the guard. The
          pair is NOT merged and NOT synthesized — the operator decides via the
          existing confirm/reject surface (a guarded candidate is listable like a
          pending one but carries the salience-protected flag).
    Why:  Q5a — the surface path for "this distinct memory is too salient to
          auto-remove". Reuses the existing abstraction-candidate store (one
          surfacer, one operator surface) rather than a parallel store.
    """
    bio_dir = Path(memory_dir) / "biomimetic"
    bio_dir.mkdir(parents=True, exist_ok=True)
    cid = _candidate_id(a_id, b_id)
    records = _all_abstraction_records(Path(memory_dir))
    for rec in records:
        if rec.get("candidate_id") == cid:
            if rec.get("status") in ("confirmed", "rejected"):
                return  # already resolved — leave it
            rec["status"] = _GUARDED_STATUS
            rec["guarded_source"] = guarded_source
            rec["guarded_ts"] = _ia._now_iso()
            _rewrite_abstraction_store(Path(memory_dir), records)
            return
    rec = {
        "candidate_id": cid,
        "a": a_id, "b": b_id, "similarity": similarity,
        "mode": "abstract", "status": _GUARDED_STATUS,
        "guarded_source": guarded_source,
        "ts": _ia._now_iso(), "guarded_ts": _ia._now_iso(),
    }
    with (bio_dir / _ABSTRACT_LOG).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# P2 — LLM-synthesize abstraction (episodic->semantic; OPERATOR-GATED)
#
# Q2c: abstractions create NEW content + can lose nuance, so they are NOT
# auto-applied. The synthesizer SURFACES a PROPOSED abstraction (status
# "proposed", carrying the synthesized title+body) into the same
# biomimetic/merge_candidates.jsonl store; only memory_confirm_merge creates the
# abstraction node + supersedes both sources RESTORABLY. memory_reject_merge
# changes nothing. This mirrors the contradiction supersession confirm pattern.
# ---------------------------------------------------------------------------


def list_abstraction_candidates(memory_dir: Path,
                                unresolved_only: bool = True
                                ) -> list[dict]:
    """Read the abstraction-candidate store (P1 'pending' + P2 'proposed').

    What: returns the {candidate_id, a, b, similarity, mode, status, ...} records
          from biomimetic/merge_candidates.jsonl. When unresolved_only (default),
          skips entries already confirmed or rejected; the operator surface +
          synthesizer both read through here. Fail-soft on a missing/partly-
          corrupt file (skips unparseable lines).
    Why:  P2 — the single read path the synthesizer (pending -> proposed) and the
          MCP confirm/reject surface both consume; mirrors
          contradiction.list_supersession_candidates.
    """
    out: list[dict] = []
    p = Path(memory_dir) / "biomimetic" / _ABSTRACT_LOG
    if not p.exists():
        return out
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if unresolved_only and rec.get("status") in (
                        "confirmed", "rejected"):
                    continue
                out.append(rec)
    except OSError:
        return out
    return out


def _rewrite_abstraction_store(memory_dir: Path, records: list[dict]) -> None:
    """Crash-safe rewrite of the abstraction-candidate store (tmp + replace).

    What: serialize ``records`` back to biomimetic/merge_candidates.jsonl via a
          .tmp file then atomic replace.
    Why:  P2's status transitions (pending->proposed, ->confirmed, ->rejected)
          update existing entries in place; a tmp+replace rewrite keeps the
          jsonl consistent under an interrupt (same pattern as the contradiction
          store's _mark path).
    """
    bio_dir = Path(memory_dir) / "biomimetic"
    bio_dir.mkdir(parents=True, exist_ok=True)
    p = bio_dir / _ABSTRACT_LOG
    tmp = p.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(p)


def _all_abstraction_records(memory_dir: Path) -> list[dict]:
    """Every record in the store (resolved + unresolved), preserving order."""
    return list_abstraction_candidates(Path(memory_dir), unresolved_only=False)


def synthesize_abstraction(memory_dir: Path, a_id: str, b_id: str) -> dict:
    """Synthesize a PROPOSED abstraction for one 'abstract' pair (P2, STAGED).

    What: read both node bodies, call the existing local-inference synthesis
          entrypoint (contradiction.synthesize_node — the SAME llama-cli/CHIRON
          backend the judge uses), and on a result STAGE it: update the pair's
          merge_candidates.jsonl entry to status="proposed" carrying
          {abstraction:{title, body}, merged_from:[a,b]}. NOTHING is created or
          superseded here — only a draft is surfaced for operator confirm.
    Why:  Q2c OPERATOR-GATE — abstractions create new content + can lose nuance,
          so they are proposed, never auto-applied. Reuses the existing inference
          path (no new model loader).

    Conservative posture: if inference is unavailable/disabled
    (contradiction.synthesize_node returns None), the pair is LEFT pending — no
    crash, no write — mirroring the judge being off.

    Returns {candidate_id, status, proposed?} — status "proposed" on success,
    "pending" when inference is unavailable, "skipped" when a node is gone.
    """
    cid = _candidate_id(a_id, b_id)
    nodes = Path(memory_dir) / "nodes"
    a_path = nodes / f"{a_id}.md"
    b_path = nodes / f"{b_id}.md"
    if not a_path.exists() or not b_path.exists():
        return {"candidate_id": cid, "status": "skipped",
                "reason": "source_gone"}

    # P3 salience guard: never synthesize an abstraction over a DISTINCT high-
    # salience source — surface it for operator review instead (Q5a / D6 iii).
    # Defensive: drain already guards before recording, but a direct call (or a
    # pre-P3 'pending' entry surviving a salience-field landing) must respect it.
    guarded = _salience_guards_pair(Path(memory_dir), a_id, b_id,
                                    is_duplicate=False)
    if guarded is not None:
        _record_guarded(Path(memory_dir), a_id, b_id, None, guarded)
        return {"candidate_id": cid, "status": _GUARDED_STATUS,
                "guarded_source": guarded}

    try:
        from samia.runtime import contradiction as _contra
    except Exception:
        return {"candidate_id": cid, "status": "pending",
                "reason": "inference_unavailable"}

    try:
        a_text = a_path.read_text(encoding="utf-8")
        b_text = b_path.read_text(encoding="utf-8")
    except OSError:
        return {"candidate_id": cid, "status": "skipped",
                "reason": "read_error"}

    try:
        synth = _contra.synthesize_node(a_text, b_text)
    except Exception:
        synth = None
    if not synth:
        # Inference disabled/unavailable -> SAFE NO-OP, pair stays pending.
        return {"candidate_id": cid, "status": "pending",
                "reason": "inference_unavailable"}

    # STAGE the proposed abstraction onto the pair's store entry.
    records = _all_abstraction_records(Path(memory_dir))
    found = False
    for rec in records:
        if rec.get("candidate_id") == cid:
            if rec.get("status") in ("confirmed", "rejected"):
                return {"candidate_id": cid, "status": rec["status"],
                        "reason": "already_resolved"}
            rec["status"] = "proposed"
            rec["abstraction"] = {"title": synth.get("title", ""),
                                  "body": synth.get("body", "")}
            rec["merged_from"] = [a_id, b_id]
            rec["proposed_ts"] = _ia._now_iso()
            found = True
            break
    if not found:
        records.append({
            "candidate_id": cid, "a": a_id, "b": b_id, "similarity": None,
            "mode": "abstract", "status": "proposed",
            "abstraction": {"title": synth.get("title", ""),
                            "body": synth.get("body", "")},
            "merged_from": [a_id, b_id],
            "ts": _ia._now_iso(), "proposed_ts": _ia._now_iso(),
        })
    _rewrite_abstraction_store(Path(memory_dir), records)
    return {"candidate_id": cid, "status": "proposed",
            "title": synth.get("title", "")}


def synthesize_pending(memory_dir: Path, budget: int = 10) -> dict:
    """Synthesize PROPOSED abstractions for up to ``budget`` pending pairs (P2).

    What: walk the 'pending' (P1-recorded, not-yet-synthesized) abstraction
          candidates and call synthesize_abstraction on each, up to budget. Each
          success flips the entry pending -> proposed; an inference-off pair stays
          pending. Returns {proposed, pending, skipped, processed}.
    Why:  the batch entry the REM tier2_merge subscriber calls after the P1 drain
          — it PROPOSES (never applies) abstractions for the surfaced 'abstract'
          minority. Gated behind is_enabled() AND inference availability (the
          per-pair synthesize_abstraction is a no-op when inference is off).

    Gated: a no-op unless is_enabled() (the same double-gate as drain).
    """
    if not is_enabled():
        return {"fired": False, "refused": "not_enabled",
                "proposed": 0, "pending": 0, "skipped": 0, "processed": 0}

    pending = [r for r in list_abstraction_candidates(Path(memory_dir))
               if r.get("status") == "pending"]
    proposed = pending_left = skipped = processed = 0
    for rec in pending[:max(0, int(budget))]:
        processed += 1
        res = synthesize_abstraction(
            Path(memory_dir), str(rec.get("a", "")), str(rec.get("b", "")))
        st = res.get("status")
        if st == "proposed":
            proposed += 1
        elif st == "pending":
            pending_left += 1
        else:
            skipped += 1
    return {"fired": True, "proposed": proposed, "pending": pending_left,
            "skipped": skipped, "processed": processed}


def confirm_abstraction(memory_dir: Path, candidate_id: str,
                        db_dir: Optional[str] = None) -> dict:
    """OPERATOR confirm of a proposed abstraction (Q2c GATE — creates content).

    What: on confirm, materialize the proposed draft as a NEW nodes/<id>.md (the
          synthesized title+body + merged_from + provenance frontmatter), then
          SUPERSEDE both source nodes RESTORABLY via
          ia.forget_node(reason="supersede", superseded_by=<abstraction>) and lay
          a provenance edge abstraction->each source. Mark the candidate
          confirmed. Returns the merge record.
    Why:  Q2c — abstractions are applied ONLY on operator confirm. Both originals
          are archived restorable (restore_node un-forgets either byte-exact), so
          even a confirmed-then-regretted abstraction is reversible. Mirrors
          memory_confirm_supersession.
    """
    records = _all_abstraction_records(Path(memory_dir))
    target = None
    for rec in records:
        if rec.get("candidate_id") == candidate_id:
            target = rec
            break
    if target is None:
        return {"confirmed": False, "candidate_id": candidate_id,
                "error": "no such candidate"}
    if target.get("status") == "confirmed":
        return {"confirmed": False, "candidate_id": candidate_id,
                "error": "already confirmed",
                "abstraction_id": target.get("abstraction_id")}
    if target.get("status") != "proposed" or not target.get("abstraction"):
        return {"confirmed": False, "candidate_id": candidate_id,
                "error": "not in proposed state (synthesize first)"}

    a_id = str(target.get("a", ""))
    b_id = str(target.get("b", ""))
    abstraction = target.get("abstraction") or {}
    abs_id = _new_abstraction_id(Path(memory_dir), candidate_id)

    # 1. Materialize the new abstraction node (live content created HERE only).
    _write_abstraction_node(
        Path(memory_dir), abs_id,
        title=str(abstraction.get("title", "")) or abs_id,
        body=str(abstraction.get("body", "")),
        merged_from=[a_id, b_id])

    # 2. Supersede BOTH sources RESTORABLY (P3 path) + provenance edges.
    superseded: list[str] = []
    for src in (a_id, b_id):
        if (Path(memory_dir) / "nodes" / f"{src}.md").exists():
            _ia.forget_node(Path(memory_dir), src, reason="supersede",
                            db_dir=db_dir, superseded_by=abs_id)
            _add_provenance_edge(abs_id, src, db_dir=db_dir)
            superseded.append(src)

    # 3. Mark the candidate confirmed in the store.
    target["status"] = "confirmed"
    target["abstraction_id"] = abs_id
    target["confirmed_ts"] = _ia._now_iso()
    _rewrite_abstraction_store(Path(memory_dir), records)

    rec_out = {
        "event": "tier2_merge",
        "mode": "abstract",
        "confirmed": True,
        "candidate_id": candidate_id,
        "abstraction_id": abs_id,
        "superseded": superseded,
    }
    _ia._log_event(Path(memory_dir), rec_out)
    return rec_out


def reject_abstraction(memory_dir: Path, candidate_id: str) -> dict:
    """OPERATOR reject of a proposed abstraction — changes NOTHING (Q2c).

    What: mark the candidate rejected in the store. No node created, no source
          superseded, no edge laid; both originals stay live.
    Why:  the reject arm of the gate. Mirrors memory_dismiss_supersession — a
          recorded decision that stops the candidate re-surfacing, with zero
          mutation of live memory.
    """
    records = _all_abstraction_records(Path(memory_dir))
    found = False
    for rec in records:
        if rec.get("candidate_id") == candidate_id:
            if rec.get("status") == "confirmed":
                return {"rejected": False, "candidate_id": candidate_id,
                        "error": "already confirmed"}
            rec["status"] = "rejected"
            rec["rejected_ts"] = _ia._now_iso()
            found = True
            break
    if not found:
        return {"rejected": False, "candidate_id": candidate_id,
                "error": "no such candidate"}
    _rewrite_abstraction_store(Path(memory_dir), records)
    return {"rejected": True, "candidate_id": candidate_id}


def _new_abstraction_id(memory_dir: Path, candidate_id: str) -> str:
    """A collision-free node id for a confirmed abstraction.

    What: "merge-<candidate_suffix>" (the candidate_id's hash tail) — stable per
          candidate; if a file with that id already exists, append a numeric
          suffix until free.
    Why:  the abstraction node needs a fresh id distinct from either source;
          deriving it from the candidate keeps it traceable and idempotent.
    """
    suffix = candidate_id[4:] if candidate_id.startswith("abs-") else candidate_id
    base = f"merge-{suffix}"
    nodes = Path(memory_dir) / "nodes"
    cand = base
    i = 1
    while (nodes / f"{cand}.md").exists():
        cand = f"{base}-{i}"
        i += 1
    return cand


def _write_abstraction_node(memory_dir: Path, node_id: str, title: str,
                            body: str, merged_from: list[str]) -> Path:
    """Materialize a confirmed abstraction as nodes/<id>.md with provenance fm.

    What: write the new semantic node carrying the synthesized title+body and
          merged_from provenance frontmatter (the episodic->semantic lineage).
          The body is preserved verbatim from the synthesis.
    Why:  P2 confirm — this is the ONLY place an abstraction node is created
          (operator-gated). The merged_from list mirrors merge_dup's stamp so the
          Topology Atlas + provenance edges trace the lineage.
    """
    from . import frontmatter as _fmod
    today = _ia._now_iso()[:10]
    fm = {
        "name": title or node_id,
        "description": "Tier-2 synthesized abstraction",
        "type": "semantic",
        "merged_from": list(merged_from),
        "valid_from": today,
        "valid_to": "null",
        "last_access": today,
        "access_count": 0,
        "relevance": 0.5,
        "tier": "warm",
    }
    order = list(fm.keys())
    nodes = Path(memory_dir) / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    p = nodes / f"{node_id}.md"
    p.write_text(_fmod.serialize(fm, order, body), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# The cursor-tracked batch drain
# ---------------------------------------------------------------------------


def drain(memory_dir: Path, budget: int = 20, cursor: int = 0,
          db_dir: Optional[str] = None) -> dict:
    """Drain a slice of the candidate backlog (the cursor-tracked batch entry).

    What: walk up to ``budget`` candidates from index ``cursor``; classify each;
          AUTO-merge the "dup" ones (pick-winner + restorable supersede +
          provenance edge), record the "abstract" ones for P2; remove every
          DISPATCHED pair (merged or recorded) from .consolidation_candidates.json
          so the backlog SHRINKS (Q3a). Stale pairs (a node already gone) are
          dropped as drained. Returns {merged, recorded, drained, skipped,
          processed, cursor, remaining, work_remaining}.
    Why:  Q3a — draining is what lets REM reach REST: work_remaining is True iff
          candidates still remain after this slice. The cursor (caller-persisted
          under rem_cursors.json key "tier2_merge") guarantees forward progress
          across REM cycles even under a small budget.

    Gated: a no-op (nothing merged) unless is_enabled() — Q5a, inert by default.
    """
    if not is_enabled():
        cands = load_candidates(Path(memory_dir))
        return {
            "fired": False, "refused": "not_enabled",
            "merged": 0, "recorded": 0, "drained": 0, "skipped": 0,
            "processed": 0, "cursor": cursor, "remaining": len(cands),
            "work_remaining": len(cands) > 0,
        }

    cands = load_candidates(Path(memory_dir))
    n = len(cands)
    start = max(0, int(cursor))
    end = min(n, start + max(0, int(budget)))

    merged = recorded = skipped = 0
    dispatched_indices: set[int] = set()

    for i in range(start, end):
        cand = cands[i]
        pair = _resolve_pair(Path(memory_dir), cand)
        if pair is None:
            # Stale (a node already merged/forgotten) — drop it as drained.
            dispatched_indices.add(i)
            skipped += 1
            continue
        a_id, b_id = pair
        kind = classify_pair(
            Path(memory_dir), a_id, b_id,
            candidate_similarity=cand.get("similarity"))
        if kind == "dup":
            # P3 salience guard: a TRUE duplicate is exempt (is_duplicate=True ->
            # guard always False), so the dup pick-winner merge is UNCHANGED — a
            # duplicate carries the same content, merging it loses nothing.
            merge_dup(Path(memory_dir), a_id, b_id, db_dir=db_dir)
            merged += 1
            dispatched_indices.add(i)
        else:
            # P3 salience guard: a DISTINCT pair where either source is a high-
            # salience memory is SURFACED for operator review (status="guarded"),
            # NOT auto-recorded for P2 abstraction — do not abstract away an
            # important one-shot memory. The pair is still drained (removed from
            # the backlog) but recorded as guarded, not pending.
            guarded = _salience_guards_pair(Path(memory_dir), a_id, b_id,
                                            is_duplicate=False)
            if guarded is not None:
                _record_guarded(Path(memory_dir), a_id, b_id,
                                cand.get("similarity"), guarded)
            else:
                _record_abstract(Path(memory_dir), a_id, b_id,
                                 cand.get("similarity"))
                # FEAT-2026-06-10-memory-fact-extract-producer-v01 P1 — the
                # 'abstract' distinct-but-overlapping pair is the SECOND producer
                # feed (Q1d). Enqueue BOTH node texts as ONE extraction record
                # (concatenated with a separator) so the extractor distils atomic
                # facts across the pair.
                # What: append one {text, source:"a.md+b.md",
                #   enqueued_by:"merge_abstract"} record to the fact-extract queue.
                # Why: distinct-but-overlapping pairs are consolidation-shaped and
                #   exactly where cross-source gist lives; extraction is ADDITIVE
                #   (this is NOT the gated abstractive MERGE — P2's separate
                #   machinery — and deletes nothing). Gated on
                #   fact_extract_enabled() so flag-off writes nothing; fail-OPEN so
                #   a queue error never blocks the drain.
                _enqueue_abstract_pair(Path(memory_dir), a_id, b_id)
            recorded += 1
            dispatched_indices.add(i)

    # Remove dispatched pairs from the candidate file so the backlog shrinks.
    drained = len(dispatched_indices)
    if drained:
        kept = [c for j, c in enumerate(cands) if j not in dispatched_indices]
        _con.surface(
            Path(memory_dir), kept,
            _read_threshold(Path(memory_dir)))
        remaining = len(kept)
        # The cursor does not advance past removed entries: indices shift left by
        # the count removed at/below the cursor, so resume from the same logical
        # position (start) which now points at the next un-dispatched pair.
        new_cursor = start
    else:
        remaining = n
        new_cursor = end

    return {
        "fired": True,
        "merged": merged,
        "recorded": recorded,
        "drained": drained,
        "skipped": skipped,
        "processed": end - start,
        "cursor": new_cursor,
        "remaining": remaining,
        "work_remaining": remaining > 0,
    }


def _read_threshold(memory_dir: Path) -> float:
    """Preserve the surfacer's recorded threshold when rewriting the file."""
    p = Path(memory_dir) / _CANDIDATE_FILE
    if not p.exists():
        return _con.DEFAULT_THRESHOLD
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return float(payload.get("threshold", _con.DEFAULT_THRESHOLD))
    except (OSError, ValueError, TypeError):
        return _con.DEFAULT_THRESHOLD


# ─────────────────────────────────────────────
# [merge_consumer] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.3.0  Updated: 2026-06-11  Status: active (INERT by default)
# Phase:      FEAT-2026-06-07-memory-tier2-merge-consumer-v01 (P1 — pick-winner
#             dup-merge; the missing DRAIN of the consolidation surfacer backlog)
#             + FEAT-2026-06-07-memory-granular-recall-repaired-decay (P2: merge_dup
#               PARTIALLY repairs the surviving node's integrity, anchor-first, as a
#               RECONCILIATION-repair side effect — reached only via the gated drain,
#               fail-soft, inert by default)
# Role:       consume .consolidation_candidates.json; AUTO pick-winner-merge the
#             true-duplicate (high-similarity) pairs via the RESTORABLE P3
#             supersede path + provenance edge; record below-bar distinct pairs
#             for P2; remove drained pairs so the backlog shrinks + REM reaches
#             REST. P2 (LLM-synthesize abstraction) PROPOSES a new semantic node
#             for each 'abstract' pair via the existing inference path and
#             materializes it ONLY on operator confirm_abstraction (supersedes
#             both sources RESTORABLY + provenance edges); reject_abstraction is a
#             no-op. P3 (salience guard) CONSUMES bio.salience_merge_guard behind a
#             hasattr guard: a DISTINCT high-salience source is SURFACED
#             (status="guarded") for operator review instead of auto-abstracted; a
#             TRUE duplicate is exempt (is_duplicate=True), so the P1 dup merge is
#             unchanged.
# Phase:      FEAT-2026-06-07-memory-tier2-merge-consumer-v01 (P1 pick-winner
#             dup-merge + P2 LLM-synthesized abstraction, operator-gated)
#             + FEAT-2026-06-10-memory-fact-extract-producer-v01 (P1: the drain's
#             'abstract' branch additionally ENQUEUES both node texts as ONE
#             fact-extract record (source "a.md+b.md", enqueued_by
#             "merge_abstract") — gated on ASTHENOS_FACT_EXTRACT_ENABLED, fail-
#             OPEN, ADDITIVE. This is NOT the gated abstractive MERGE; it distils
#             atomic facts and deletes nothing.)
#             + BUG-2026-06-11 runaway-loop (enqueue side): _enqueue_abstract_pair
#               now dedups by candidate_id against a per-pair-once done-set ledger
#               (biomimetic/fact_extract_enqueued.jsonl) so the surfacer's repeated
#               re-presentation of the SAME ~52 abstract pairs no longer re-enqueues
#               them every REM cycle (+~1,500 lines/hour to .fact_extract_queue.jsonl).
# Depends:    samia.core.{consolidation,ia,frontmatter,web_store,fact_extractor},
#             samia.runtime.contradiction (cosine finder + P2 synthesize_node,
#             the SAME judge inference backend; live path only).
# Note:       PRODUCE-ONLY — import does nothing; drain()/synthesize_pending() are
#             no-ops unless ASTHENOS_TIER2_MERGE_ENABLED=1 (default OFF), and P2
#             synthesis additionally no-ops (pair stays pending) when the
#             inference backend is unavailable/disabled. Abstractions are NEVER
#             auto-applied — confirm_abstraction is the operator-only gate. Every
#             merge (dup or confirmed abstraction) is RESTORABLE (full
#             archive/<id>.superseded.json + ia.restore_node + detect_wrong_
#             deletion self-heal). No thread/timer/live mutation on import.
#             Pick-winner needs no salience guard (dups share content); the P3
#             guard fires only on DISTINCT high-salience sources (abstraction
#             path), surfacing them for review (status="guarded").
# ─────────────────────────────────────────────
