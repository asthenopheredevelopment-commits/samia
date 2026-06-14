"""samia.core.integrity.passes — the corpus-walking sweeps (top of the package DAG).

Layer 1 (Owns / Depends):
    Owns:    the corpus-level passes — the anchor backstop sweeps (backfill_anchors_
             pass + its idle_pulse-tick driver anchor_backfill_tick), the content-
             integrity erosion SWEEP (integrity_decay_pass — the second axis's
             continuous pass, with the P3 terminal-freeze-at-floor branch), and the
             consolidation-repair REM pass (consolidation_repair_pass) + its budget
             constant (CONSOLIDATION_REPAIR_BUDGET).
    Depends: every sibling submodule — samia.core.integrity.config (DEFAULT_TIER /
             INTEGRITY_FLOOR / INTEGRITY_FULL / PARTIAL_REPAIR_STRENGTH /
             salience_freeze_exempt / _fm), .anchors (ensure_anchor / has_anchor /
             get_integrity / is_distilled), .erosion (erode / live_salience), .repair
             (partial_repair). samia.core.ia (P3 terminal freeze, function-LOCAL) +
             the reconsolidation log (via .repair._log_reconsolidation).

Layer 2 (What / Why):
    What: the orchestrators the daemon/idle/REM call sites invoke. The backstop sweeps
          anchor any straggler node; the erosion sweep applies one slow per-character
          pass to every eligible node (reading the SAME last_access/tier/salience the
          relevance decay reads); the consolidation-repair pass PARTIALLY heals a
          budgeted slice of eroded nodes per REM cycle. integrity_decay_pass also owns
          the P3 terminal-freeze-at-floor branch (deferred post-walk).
    Why:  Q6a — a NEW second axis RIDES the same continuous tick as relevance-decay
          without modifying tier.step_relevance/tier.decay_pass; Q5a — the integrity
          floor is a SECOND trigger feeding the SAME reversible ia.freeze path the
          relevance floor uses, honoring the SAME salience exemption so the two axes'
          freeze policy stays consistent.

PRODUCE-ONLY / INERT BY DEFAULT: integrity_decay_pass is dry=True by default (computes +
    reports WITHOUT writing) and terminal_freeze is OFF by default; no scheduler/thread/
    timer; it NEVER erodes a node without a recoverable anchor. consolidation_repair_pass
    only heals ERODED nodes (a pristine corpus is a cheap no-op). The freeze reuses the
    REVERSIBLE ia.freeze (demotion, NOT deletion).
"""

from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_TIER,
    INTEGRITY_FLOOR,
    INTEGRITY_FULL,
    Optional,
    PARTIAL_REPAIR_STRENGTH,
    Path,
    _fm,
    salience_freeze_exempt,
)
from .anchors import ensure_anchor, get_integrity, has_anchor, is_distilled
from .erosion import erode, live_salience
from .repair import _log_reconsolidation, partial_repair


def backfill_anchors_pass(memory_dir: Path, cursor: int = 0,
                          budget: int = 200) -> dict[str, Any]:
    """Backstop sweep — anchor any un-anchored node (FEAT-2026-06-08 Q4a).

    What: cursor-walks a budgeted slice of nodes/ and ensure_anchor()s each UN-anchored
      node (capture-IF-MISSING only — NEVER refresh), checkpointing the cursor. Returns
      {captured, processed, cursor, work_remaining, total}. Because it only captures when
      no anchor exists, it can NEVER clobber an eroded node's pristine anchor, so it is safe
      to run continuously over the whole corpus.
    Why: write-path capture (capture_on_genuine_write at the write_node seam) is the primary
      source of truth; this low-cadence sweep is the BACKSTOP that catches any node a future
      write path, a restore/thaw, or a bulk import leaves anchor-less. A no-op at full
      coverage. Productionizes tools/backfill_integrity_anchors_2026_06_08.py.
    """
    mem = Path(memory_dir)
    node_files = sorted((mem / "nodes").glob("*.md"))
    total = len(node_files)
    start = int(cursor) if 0 <= int(cursor) < total else 0
    end = min(start + max(0, int(budget)), total)
    captured = 0
    for p in node_files[start:end]:
        try:
            fm, _order, body = _fm.read_node(p)
        except Exception:
            continue  # fail-soft: a parse error never aborts the sweep
        try:
            if ensure_anchor(mem, p.stem, fm, body):
                captured += 1
        except OSError:
            continue
    work_remaining = end < total
    return {"captured": captured, "processed": end - start,
            "cursor": end if work_remaining else 0,
            "work_remaining": work_remaining, "total": total}


def anchor_backfill_tick(memory_dir: Path) -> dict[str, Any]:
    """idle_pulse subscriber entry (FEAT-2026-06-08 Q4a) — full-corpus backstop sweep.

    What: loops backfill_anchors_pass over the WHOLE node list per (daily) tick so the
      backstop achieves complete coverage rather than a single budgeted slice. Cheap —
      every already-anchored node is just a has_anchor() stat. Returns aggregate telemetry.
      Subscriber signature fn(mem).
    Why: the daily backstop must catch every straggler, not 1/N of them; the per-pass budget
      only bounds memory/loop length, not coverage. A no-op at full coverage.
    """
    total_captured = 0
    cursor = 0
    passes = 0
    while True:
        res = backfill_anchors_pass(memory_dir, cursor=cursor, budget=500)
        total_captured += res["captured"]
        passes += 1
        if not res["work_remaining"] or passes > 1000:  # 1000 = runaway guard
            break
        cursor = res["cursor"]
    return {"captured": total_captured, "passes": passes}


def integrity_decay_pass(memory_dir: Path, dry: bool = True,
                         today: Optional[str] = None,
                         only_with_anchor: bool = True,
                         terminal_freeze: bool = False) -> list[dict]:
    """The content-integrity erosion SWEEP — the second axis's continuous pass.

    What: walks nodes/*.md and applies one slow per-character erosion pass to each
      eligible node (lowering integrity + eroding the served body), reading the SAME
      last_access/tier/salience the relevance decay already reads. Skips target_state
      frozen/archived nodes (exactly as the relevance step does) and ANY node without a
      recoverable anchor. A tier=="frozen" node is skipped UNLESS it is DISTILLED
      (TUNE-2026-06-10 decision c, systems-consolidation gating: the episodic trace
      fades only AFTER the semantic representation forms — is_distilled(fm) is the
      gate; a distilled frozen node erodes at TIER_EROSION_FACTOR["frozen"]=0.25).
      Returns one record per eroded node. This is the entry the existing decay/idle path
      can invoke ALONGSIDE tier.decay_pass (both axes, ungated, wake+REM).

      P3 TERMINAL FREEZE-AT-FLOOR (Q5a): when `terminal_freeze=True` (and not dry), a node
      whose new integrity falls below INTEGRITY_FLOOR and was NOT repaired this tick is
      routed into the existing REVERSIBLE ia.freeze (demotion-to-frozen, restorable via
      ia.thaw + a later recall reconsolidation), NOT deleted — UNLESS its salience clears
      the salience-exemption threshold (salience_freeze_exempt(), consistent with the
      relevance path's P5 freeze exemption), in which case it stays resident. The freeze is
      DEFERRED to after the walk (ia.freeze removes node files; freezing mid-walk would
      mutate the directory we are iterating), mirroring tier.decay_pass's freeze_queue.
    Why: Q6a — a NEW second axis that RIDES the same continuous tick as relevance-decay,
      without modifying tier.step_relevance / tier.decay_pass; Q5a — the integrity floor is
      a SECOND trigger feeding the SAME reversible freeze path the relevance floor uses, and
      it honors the SAME salience exemption so the two axes' freeze policy stays consistent.

    PRODUCE-ONLY / INERT BY DEFAULT: `dry=True` by default — it computes + reports the
      erosion WITHOUT writing. The caller must explicitly pass dry=False to apply it, and
      `terminal_freeze` is OFF by default (the floor never freezes until opted-in). It
      starts NO scheduler/thread/timer (a plain function the existing pass invokes). It
      NEVER erodes a node without a recoverable anchor (`only_with_anchor`).

    NOTE: this does NOT modify the relevance/tier axis. The two compose, not collide.
    """
    from datetime import date as _date

    nodes_dir = memory_dir / "nodes"
    today_iso = today or _date.today().isoformat()
    out: list[dict] = []
    if not nodes_dir.exists():
        return out

    def _days_since(last_iso: str) -> int:
        if not last_iso:
            return 9999
        try:
            last = _date.fromisoformat(str(last_iso))
            cur = _date.fromisoformat(today_iso)
            return max(0, (cur - last).days)
        except (ValueError, TypeError):
            return 9999

    # Deferred terminal-freeze queue — ia.freeze removes node files, so (exactly like
    # tier.decay_pass's freeze_queue) we freeze AFTER the walk to avoid mutating the
    # directory mid-iteration. Each entry is {node, integrity, salience}.
    freeze_queue: list[dict] = []
    exempt_threshold = salience_freeze_exempt() if terminal_freeze else None

    for md in sorted(nodes_dir.glob("*.md")):
        try:
            fm, order, body = _fm.read_node(md)
        except (ValueError, OSError):
            continue

        # Skip target_state frozen/archived nodes — exactly as the relevance step does.
        # (target_state lifecycle freeze/archive is a HARD skip on EITHER axis,
        # independent of the distillation gate below — those node files are immutable.)
        ts = str(fm.get("target_state", "live")).lower()
        if ts in ("frozen", "archived"):
            continue
        node_tier = str(fm.get("tier", DEFAULT_TIER)).lower()
        # TUNE-2026-06-10 operator decision (c), systems-consolidation (distillation)
        # gating: a tier=="frozen" node erodes ONLY once its content is DISTILLED (the
        # semantic representation has formed — the fact-extract drain stamped
        # distilled:true). An UNDISTILLED frozen node still NEVER erodes (unchanged
        # behavior — its episodic trace stays pristine until the gist exists); a
        # DISTILLED frozen node is ELIGIBLE to erode at TIER_EROSION_FACTOR["frozen"]
        # (0.25, slowest), with the normal anchor-gating + salience/recency modulation
        # below still applying. The episodic trace fades only AFTER the semantic
        # representation forms.
        if node_tier == "frozen" and not is_distilled(fm):
            continue

        node_name = md.stem

        # NEVER erode without a recoverable anchor (no irrecoverable loss).
        if only_with_anchor and not has_anchor(memory_dir, node_name, fm):
            continue

        last = str(fm.get("last_access", ""))
        days = 0 if last == today_iso else _days_since(last)
        # Q2a salience modulation — read the LIVE salience signal (bio.compute_salience,
        # read-only) so a genuinely high-salience node erodes slower; fall back to the
        # maintained frontmatter field, then to neutral 0.0 (graceful + fail-soft).
        salience = live_salience(memory_dir, node_name, fm)

        old_integrity = get_integrity(fm)
        new_body, new_integrity, n_eroded = erode(
            memory_dir, node_name, fm, order, body,
            days_since_recall=days, tier=node_tier, salience=salience,
        )
        if n_eroded <= 0:
            continue

        rec = {
            "node": node_name,
            "old_integrity": round(old_integrity, 6),
            "new_integrity": round(new_integrity, 6),
            "n_eroded": n_eroded,
            "tier": node_tier,
            "days_since_recall": days,
        }

        # P3 terminal freeze-at-floor (Q5a): a node eroded below the readable floor this
        # tick (and not repaired) terminally freezes — UNLESS salience-exempt, consistent
        # with the relevance path. The erosion was NOT a repair, so crossing the floor here
        # is the un-repaired terminal. Salience-exempt nodes stay resident (surface/remain).
        if terminal_freeze and new_integrity < INTEGRITY_FLOOR:
            if salience >= float(exempt_threshold):
                rec["freeze_exempt"] = True
                rec["salience"] = round(float(salience), 4)
            else:
                rec["terminal_freeze"] = True
                rec["salience"] = round(float(salience), 4)
                freeze_queue.append({
                    "node": node_name,
                    "integrity": round(new_integrity, 6),
                    "salience": round(float(salience), 4),
                })

        out.append(rec)

        if not dry:
            if not md.exists():
                continue
            _fm.write_node(md, fm, order, new_body, integrity_rewrite=True)

    # Deferred terminal freeze — after the walk + after the eroded bodies are persisted.
    # Reuse the existing REVERSIBLE ia.freeze (restorable via ia.thaw); NEVER deletion.
    if not dry and terminal_freeze and freeze_queue:
        # Function-LOCAL import — ia is reached only on the (gated, opt-in) freeze path;
        # keeping it off the package import path avoids the bio/ia<->integrity import cycle.
        try:
            from .. import ia as _ia
        except ImportError as e:
            print(f"[integrity] terminal-freeze unavailable (ia import failed): {e}")
            return out
        for t in freeze_queue:
            md = nodes_dir / f"{t['node']}.md"
            if not md.exists():
                continue
            try:
                _ia.freeze(memory_dir, t["node"])
                t["frozen"] = True
            except SystemExit as e:
                # ia.freeze sys.exits on a hot node ("demote first"); the integrity floor
                # must never crash the sweep — record + skip (the relevance axis will
                # demote it eventually, then a later integrity pass can freeze it).
                t["freeze_error"] = str(e)
            except Exception as e:
                t["freeze_error"] = str(e)
                print(f"[integrity] terminal-freeze FAILED for {t['node']}: {e}")
            try:
                _log_reconsolidation(memory_dir, {
                    "event": "terminal_freeze",
                    "trigger": "integrity_floor",
                    "node": t["node"],
                    "integrity": t.get("integrity"),
                    "salience": t.get("salience"),
                    "floor": INTEGRITY_FLOOR,
                    "frozen": t.get("frozen", False),
                    "freeze_error": t.get("freeze_error"),
                })
            except Exception:
                pass  # fail-soft: a logging failure must never break the sweep

    return out


# CONSOLIDATION_REPAIR_BUDGET — What: max nodes a single consolidation-repair REM slice
#   touches (cursor-tracked across cycles).
# Why: incremental — a large corpus is repaired a budgeted slice at a time so one REM
#   cycle never stalls; the cursor resumes the next cycle (mirrors the other REM ops).
CONSOLIDATION_REPAIR_BUDGET = 50


def consolidation_repair_pass(memory_dir: Path,
                              budget: int = CONSOLIDATION_REPAIR_BUDGET,
                              cursor: Optional[int] = None,
                              strength: float = PARTIAL_REPAIR_STRENGTH,
                              today: Optional[str] = None) -> dict:
    """The CONSOLIDATION repair pass — sleep heals what it consolidates (P2, Q3a partial).

    What: walks a cursor-tracked slice (<= ``budget`` nodes) of nodes/ and PARTIALLY
      repairs the integrity of each ERODED node it touches (partial_repair, anchor-first,
      strength < 1.0). Skips frozen/archived nodes and any node with no anchor (anchor-
      first only). Cursor is an int index over the sorted node list; it advances by the
      slice size and WRAPS at the end (a full pass spans many REM cycles). Returns a dict
      with the cursor, counts, and a work_remaining signal for the REM driver.
    Why:  Q3a — CONSOLIDATION is a PARTIAL repair trigger: a REM consolidation pass heals
      a little of the integrity of the nodes it consolidates (distinct from RECALL, which
      heals fully). Anchor-first only — NO generative repair (P3). Incremental + cursor-
      tracked so it fits the REM offline-op contract.

    PRODUCE-ONLY: a plain function (no scheduler/thread/timer). Its REM gate + enable flag
      live at the REM-subscriber wiring (rem_subscribers); this is the pure work fn. Only
      ERODED nodes (integrity < FULL) are repaired — a pristine node is skipped (nothing
      to heal), so a fresh corpus is a cheap no-op.
    """
    from datetime import date as _date

    nodes_dir = memory_dir / "nodes"
    out: dict = {"repaired": 0, "touched": 0, "processed": 0, "total": 0,
                 "made_progress": False, "work_remaining": False}
    if not nodes_dir.exists():
        return out

    node_ids = sorted(p.stem for p in nodes_dir.glob("*.md"))
    total = len(node_ids)
    out["total"] = total
    if total == 0:
        return out

    start = int(cursor or 0)
    if start < 0 or start >= total:
        start = 0
    cap = max(0, int(budget))
    end = min(start + cap, total)
    slice_ids = node_ids[start:end]
    out["processed"] = len(slice_ids)

    repaired = touched = 0
    for node_name in slice_ids:
        md = nodes_dir / f"{node_name}.md"
        if not md.exists():
            continue
        try:
            fm, _order, _body = _fm.read_node(md)
        except (ValueError, OSError):
            continue
        # Skip frozen/archived — exactly as the erosion sweep + relevance step do.
        ts = str(fm.get("target_state", "live")).lower()
        if ts in ("frozen", "archived"):
            continue
        if str(fm.get("tier", DEFAULT_TIER)).lower() == "frozen":
            continue
        # Anchor-first only; nothing to repair on a node with no anchor.
        if not has_anchor(memory_dir, node_name, fm):
            continue
        # Only heal an ERODED node — a pristine node has nothing to consolidate-repair.
        if get_integrity(fm) >= INTEGRITY_FULL:
            continue
        touched += 1
        res = partial_repair(memory_dir, node_name, strength=strength,
                             trigger="consolidation")
        if res.get("repaired"):
            repaired += 1

    new_index = end if end < total else 0
    wrapped = end >= total
    out.update({
        "repaired": repaired, "touched": touched,
        "made_progress": bool(slice_ids),
        "cursor": new_index, "wrapped": wrapped,
        "work_remaining": not wrapped,
    })
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.integrity.passes
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.integrity monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       the corpus-walking sweeps (top of the package DAG) — the anchor
#             backstop (backfill_anchors_pass + anchor_backfill_tick), the
#             content-integrity erosion SWEEP (integrity_decay_pass, incl. the P3
#             terminal-freeze-at-floor branch), and the consolidation-repair REM
#             pass (consolidation_repair_pass + CONSOLIDATION_REPAIR_BUDGET).
# Stability:  stable — carved byte-identically from the monolith. integrity_decay_pass
#             does NOT modify the relevance/tier axis (the two compose). The terminal
#             freeze reuses the REVERSIBLE ia.freeze (demotion, NOT deletion) and is
#             DEFERRED post-walk (ia.freeze removes node files mid-iteration otherwise).
# ErrorModel: PRODUCE-ONLY / INERT BY DEFAULT — integrity_decay_pass is dry=True +
#             terminal_freeze=False by default; NEVER erodes without a recoverable
#             anchor; per-node parse errors are skipped (fail-soft); ia.freeze's
#             hot-node SystemExit is swallowed (record + skip); a log failure never
#             breaks the sweep. consolidation_repair_pass only heals ERODED nodes
#             (a pristine/fresh corpus is a cheap no-op).
# Depends:    .config (DEFAULT_TIER/INTEGRITY_FLOOR/INTEGRITY_FULL/PARTIAL_REPAIR_
#             STRENGTH/salience_freeze_exempt/_fm), .anchors (ensure_anchor/has_anchor/
#             get_integrity/is_distilled), .erosion (erode/live_salience), .repair
#             (partial_repair/_log_reconsolidation). samia.core.ia (function-LOCAL,
#             P3 terminal freeze). stdlib datetime (function-LOCAL).
# Exposes:    backfill_anchors_pass, anchor_backfill_tick, integrity_decay_pass,
#             consolidation_repair_pass, CONSOLIDATION_REPAIR_BUDGET.
# Lines:      415
# --------------------------------------------------------------------------
