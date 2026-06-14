"""samia.runtime.contradiction.store — the unified supersession-candidate store.

Layer 1 (Owns / Depends):
    Owns:    the R2 CANONICAL supersession-candidate store on disk
             (<memory_dir>/biomimetic/supersession_candidates.jsonl) — its path
             primitive (_supersession_path), the append-with-dedup writer
             (record_supersession_candidate), the unresolved-aware reader
             (list_supersession_candidates), and the crash-safe confirm/dismiss
             marker (_mark_supersession_candidate + mark_supersession_confirmed /
             mark_supersession_dismissed).
    Depends: the package config leaf (_SUPERSESSION_STORE filename, _now_iso, _log,
             json). Pure filesystem JSONL; no embedding/LLM dependency.

Layer 2 (What / Why):
    What: the single durable provenance of detected-but-not-yet-acted supersessions.
          One schema, one owner: the memory_guard surfacer, the mcp_server confirm/
          dismiss/list surface, and the passive sweep all route through these
          functions.
    Why:  R2 reconciliation collapsed two divergent stores into this one. record
          dedups unresolved pairs (BUG-2026-06-11) so the store doesn't grow
          unbounded; mark rewrites tmp+replace so a confirm/dismiss is crash-safe.

NO CROSS-SUBMODULE PATCH SEAM here: record_supersession_candidate calls
    list_supersession_candidates directly (same module) — no test patches
    list_supersession_candidates while exercising record's dedup. The passes arm,
    which DOES run alongside a patched list_supersession_candidates, reaches it
    through the package facade (see passes.py).
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

# Shared leaf — the canonical store filename, the provenance stamp, the package
# logger, and the re-exported json.
from . import config as _cfg


def _supersession_path(memory_dir: Path) -> Path:
    """Canonical path of the unified supersession-candidate store."""
    return memory_dir / "biomimetic" / _cfg._SUPERSESSION_STORE


def record_supersession_candidate(
    memory_dir: Path,
    old_id: str,
    new_id: str,
    cosine: float,
    jaccard: Optional[float] = None,
    mode: str = "online",
    judge: Optional[dict[str, Any]] = None,
    status: str = "candidate",
) -> dict[str, Any]:
    """Append a supersession candidate to the unified store (R2 canonical owner).

    What: writes one record with the single schema
          {old_id, new_id, cosine, jaccard, mode(online|passive), judge?, ts,
           status, confirmed, dismissed} to <memory_dir>/biomimetic/
          supersession_candidates.jsonl — the durable provenance of a detected
          (but not yet auto-acted) supersession.
    Why:  the Q4-granularity decision records weaker online hits and all passive
          hits as candidates for the LLM judge / operator surface rather than
          auto-deleting. This is the write side of the single store that the
          memory_guard surfacer and the mcp_server confirm/dismiss/list all share.

          DEDUP (BUG-2026-06-11): the SAME (old_id, new_id) pair was re-detected
          and re-appended on every passive sweep / online write, so the store grew
          unbounded with duplicate UNRESOLVED rows. Skip the append when an
          unresolved record for the same (old_id, new_id) already exists; a
          resolved (confirmed/dismissed) prior record does NOT suppress a fresh
          re-detection (the pair came back after being acted on — that is signal).
    """
    bio_dir = memory_dir / "biomimetic"
    bio_dir.mkdir(parents=True, exist_ok=True)
    norm_old = old_id if old_id.endswith(".md") else f"{old_id}.md"
    norm_new = new_id if new_id.endswith(".md") else f"{new_id}.md"
    # Dedup: an unresolved row for this exact pair already stands — re-recording it
    # only bloats the store and re-surfaces the same candidate. Scan the live
    # unresolved rows once (the canonical reader handles id normalization + skips
    # already confirmed/dismissed rows); return the standing record so callers see
    # "already a candidate" without a second write.
    for r in list_supersession_candidates(memory_dir, unresolved_only=True):
        if (r.get("old_id"), r.get("new_id")) == (norm_old, norm_new):
            return r
    rec: dict[str, Any] = {
        "old_id": norm_old,
        "new_id": norm_new,
        "cosine": float(round(cosine, 4)),
        "jaccard": (float(round(jaccard, 4)) if jaccard is not None else None),
        "mode": mode,
        "ts": _cfg._now_iso(),
        "status": status,
        "confirmed": False,
        "dismissed": False,
    }
    if judge is not None:
        rec["judge"] = judge
    with _supersession_path(memory_dir).open("a", encoding="utf-8") as f:
        f.write(_cfg.json.dumps(rec) + "\n")
    return rec


def list_supersession_candidates(memory_dir: Path,
                                 unresolved_only: bool = True
                                 ) -> list[dict[str, Any]]:
    """Read the unified supersession store (R2 canonical reader).

    What: returns the recorded candidates; when unresolved_only (default) skips
          any already confirmed or dismissed.
    Why:  the single list path the memory_guard surfacer and the mcp_server /
          Atoms surface both consume. Fail-soft on a missing or partly-corrupt
          file (skips unparseable lines).
    """
    out: list[dict[str, Any]] = []
    p = _supersession_path(memory_dir)
    if not p.exists():
        return out
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _cfg.json.loads(line)
                except _cfg.json.JSONDecodeError:
                    continue
                if unresolved_only and (rec.get("confirmed")
                                        or rec.get("dismissed")):
                    continue
                out.append(rec)
    except Exception as exc:
        _cfg._log.warning("contradiction: supersession store read failed: %s", exc)
    return out


def _mark_supersession_candidate(memory_dir: Path, old_id: str,
                                 new_id: Optional[str], field: str) -> int:
    """Atomically set <field>=True (+ status) on matching un-resolved candidate(s).

    What: rewrites the unified store, marking every entry whose old_id matches
          (and new_id, if given) and that is not already resolved. Returns the
          count touched. tmp + replace keeps the rewrite crash-safe.
    Why:  the single mark path for both confirm and dismiss; resolving a
          candidate stops it re-surfacing. This only RECORDS the decision —
          the actual archiving forget cascade is run by the caller.
    """
    p = _supersession_path(memory_dir)
    if not p.exists():
        return 0
    target = old_id if old_id.endswith(".md") else f"{old_id}.md"
    want_new = (new_id if (new_id is None or new_id.endswith(".md"))
                else f"{new_id}.md")
    ts = _cfg._now_iso()
    touched = 0
    try:
        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _cfg.json.loads(line)
                except _cfg.json.JSONDecodeError:
                    continue
                rec_old = rec.get("old_id", "")
                rec_old = rec_old if rec_old.endswith(".md") else f"{rec_old}.md"
                already = rec.get("confirmed") or rec.get("dismissed")
                if (not already and rec_old == target
                        and (want_new is None or rec.get("new_id") == want_new)):
                    rec[field] = True
                    rec[f"{field}_at"] = ts
                    rec["status"] = field  # "confirmed" | "dismissed"
                    touched += 1
                rows.append(rec)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(_cfg.json.dumps(rec) + "\n")
        tmp.replace(p)
    except Exception as exc:
        _cfg._log.warning("contradiction: supersession mark failed: %s", exc)
    return touched


def mark_supersession_confirmed(memory_dir: Path, old_id: str,
                                new_id: Optional[str] = None) -> int:
    """Record confirmation of a supersession candidate in the unified store.

    No delete here — the caller (mcp_server.memory_confirm_supersession) runs the
    archiving forget cascade; this only marks the decision so it stops surfacing.
    """
    return _mark_supersession_candidate(memory_dir, old_id, new_id, "confirmed")


def mark_supersession_dismissed(memory_dir: Path, old_id: str,
                                new_id: Optional[str] = None) -> int:
    """Record dismissal (false positive) of a supersession candidate. No delete."""
    return _mark_supersession_candidate(memory_dir, old_id, new_id, "dismissed")


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.contradiction.store
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.runtime.contradiction monolith
#             during modularization (the R2 canonical supersession store arm).
# Layer:      runtime (library helper, no daemon loop)
# Role:       the R2 CANONICAL unified supersession-candidate store (record +
#             list-unresolved + mark-confirmed + mark-dismissed) the memory_guard
#             surfacer, the mcp_server confirm/dismiss/list surface, and the passive
#             sweep all route through.
# Stability:  stable — JSONL append/read/rewrite; one schema, one owner.
# ErrorModel: fail-soft — read/mark warn + skip on corrupt lines; the mark rewrite is
#             tmp+replace (crash-safe). record dedups unresolved pairs (BUG-2026-06-11)
#             so the store does not grow unbounded.
# Depends:    .config (_SUPERSESSION_STORE, _now_iso, _log, json). No embedding/LLM dep.
# Exposes:    record_supersession_candidate, list_supersession_candidates,
#             mark_supersession_confirmed, mark_supersession_dismissed (public);
#             _supersession_path, _mark_supersession_candidate (internal).
# Lines:      222
# --------------------------------------------------------------------------
