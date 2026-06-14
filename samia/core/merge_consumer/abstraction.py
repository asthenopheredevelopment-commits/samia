"""samia.core.merge_consumer.abstraction — the P2 abstraction lifecycle, the P3
salience guard, and the fact-extract enqueue feed (operator-gated).

Layer 1 (Owns / Depends):
    Owns:    the abstraction-candidate store (biomimetic/merge_candidates.jsonl)
             read/rewrite (list_abstraction_candidates, _all_abstraction_records,
             _rewrite_abstraction_store) and its lifecycle:
               - _record_abstract (P1 queues a distinct pair "pending"),
               - synthesize_abstraction / synthesize_pending (P2 STAGES a
                 "proposed" draft via the existing inference path),
               - confirm_abstraction (operator GATE — materializes the node +
                 supersedes both sources RESTORABLY) / reject_abstraction (no-op),
               - _new_abstraction_id / _write_abstraction_node (the node write).
             The P3 salience guard (_salience_guards_pair, _record_guarded) that
             SURFACES a distinct high-salience source instead of abstracting it.
             The fact-extract enqueue feed (_enqueued_candidate_ids, _mark_enqueued,
             _enqueue_abstract_pair) with its per-pair-once done-set ledger.
    Depends: .config (_candidate_id, _ia, the store/enqueued filenames, the
             guarded status), .candidates (_read_fm), .winner (_add_provenance_edge),
             samia.core.bio (salience_merge_guard, lazy + hasattr-gated),
             samia.core.fact_extractor (enqueue, lazy),
             samia.runtime.contradiction (synthesize_node, lazy).

Layer 2 (What / Why):
    What: the GATED half of the consumer. P1 only RECORDS the distinct-but-
          overlapping "abstract" minority; P2 PROPOSES a synthesized abstraction
          (never applies it); only the operator's confirm creates the node +
          supersedes the sources. P3 protects a distinct high-salience memory from
          being abstracted away (surfaces it "guarded"). The enqueue feed distils
          cross-source gist into ADDITIVE atomic facts (NOT the abstractive merge).
    Why:  Q2c OPERATOR-GATE — abstractions create new content + can lose nuance,
          so they are proposed, never auto-applied; both originals are archived
          restorable so even a confirmed-then-regretted abstraction is reversible.
          Q5a / D6 (iii) — a distinct high-salience memory is surfaced, not
          silently superseded. Q1d — the abstract branch is the second fact-
          extract producer feed, deduped per-pair-once against a done-set ledger
          (BUG-2026-06-11) so the surfacer's repeated re-presentation never
          re-enqueues the same pair every REM cycle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .config import (
    is_enabled,
    _candidate_id,
    _ia,
    _ABSTRACT_LOG,
    _ENQUEUED_LOG,
    _GUARDED_STATUS,
)
from .candidates import _read_fm
from .winner import _add_provenance_edge


# ---------------------------------------------------------------------------
# P1 record + the fact-extract enqueue feed
# ---------------------------------------------------------------------------


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
        from .. import fact_extractor as _fx
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
        from .. import bio as _bio
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
    from .. import frontmatter as _fmod
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


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.merge_consumer.abstraction
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.merge_consumer monolith during
#             modularization (the P2 abstraction lifecycle + P3 salience guard + fact-
#             extract enqueue feed submodule).
# Layer:      core (pure library, no daemon dependency)
# Role:       the GATED half — P1 records the distinct 'abstract' minority + feeds
#             the fact-extract queue (per-pair-once); P2 STAGES a proposed
#             synthesized abstraction (never applies); confirm_abstraction is the
#             operator GATE that materializes the node + supersedes both sources
#             RESTORABLY; reject_abstraction is a no-op; P3 salience guard surfaces
#             a distinct high-salience source ("guarded") instead of abstracting it.
# Stability:  stable — the carve preserved the store schema, every status
#             transition, the confirm node-write + supersede + edge order, the
#             salience hasattr-gate, and the BUG-2026-06-11 done-set dedup.
# ErrorModel: store reads fail-soft (skip unparseable lines / missing file);
#             synthesize_* leaves a pair pending when inference is off (no crash);
#             the salience guard returns None when bio lacks salience_merge_guard;
#             _enqueue_abstract_pair is fully fail-OPEN (never blocks the drain).
# Depends:    json, os, pathlib, typing (stdlib). .config (_candidate_id, _ia,
#             store filenames, _GUARDED_STATUS, is_enabled), .candidates (_read_fm),
#             .winner (_add_provenance_edge). samia.core.{bio,fact_extractor,
#             frontmatter} + samia.runtime.contradiction (all lazy).
# Exposes:    list_abstraction_candidates, synthesize_abstraction,
#             synthesize_pending, confirm_abstraction, reject_abstraction,
#             _record_abstract, _record_guarded, _salience_guards_pair,
#             _enqueue_abstract_pair, _new_abstraction_id (+ store internals).
# Lines:      658
# --------------------------------------------------------------------------
