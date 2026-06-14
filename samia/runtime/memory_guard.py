"""samia.runtime.memory_guard -- AUD48 memory write defense (Phases 1-3).

Layer 1 (Owns / Depends):
    Owns:    Staging buffer surface for SAM/IA memory writes, staged.jsonl
             append-only log, pending.jsonl flagged-write queue,
             memory_guard_status / memory_guard_pending / memory_guard_approve
             / memory_guard_discard / memory_guard_stats IPC ops
    Depends: samia.runtime.ipc.register_op (op registration)
             samia.core.consolidation.shingles, samia.core.consolidation.jaccard

Layer 2 (What / Why):
    What: Staging buffer that logs every memory write, runs heuristic validation
          (prompt-injection markers + contradiction smell via shingles/jaccard),
          and routes flagged writes to a pending.jsonl queue for operator review.
          Phase 2 adds consensus validation + verdict states; Phase 3 adds the
          operator review IPC ops consumed by the Atoms MemoryGuard panel.
    Why:  SAM/IA memory writes across all five tiers are currently unvalidated
          (the soul-edit gate covers only [soul] blocks).  This staging surface
          + validation layer prevents memory poisoning attacks from prompt
          injection or hallucinating warriors.  Flagged writes are held for
          operator judgment rather than silently committed or silently dropped.

Cite-source: arXiv:2510.02373 -- A-MemGuard staging-buffer + consensus pattern.
Design doc: AUD48_amemguard_memory_defense.md
AUD48 Phase 1 -- observation-only staging buffer.
AUD48 Phase 2 -- consensus validation + blocking (heuristic).
AUD48 Phase 3 -- operator review queue + IPC ops for Atoms panel.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = logging.getLogger("samia.runtime.memory_guard")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# _INJECTION_PATTERNS — What: regex patterns that indicate prompt-injection
#   attempts in a memory write payload.
# _INJECTION_PATTERNS — Why: lightweight first-pass filter to catch obvious
#   injection payloads before they enter the memory store.  Patterns are
#   deliberately broad (case-insensitive) to catch common attack templates.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions|context)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
]

# _CONTRADICTION_THRESHOLD — What: jaccard similarity threshold above which
#   a payload is flagged as potentially contradicting a recent memory node.
# _CONTRADICTION_THRESHOLD — Why: 0.25 is above the content-word noise floor
#   (0.01-0.05) but low enough to catch topical adjacency.  A high overlap with
#   different content suggests contradiction rather than duplication.
_CONTRADICTION_THRESHOLD: float = 0.25

# _CONTRADICTION_WINDOW_H — What: hours to look back for recent nodes when
#   checking contradiction smell.
# _CONTRADICTION_WINDOW_H — Why: 24h window per A-MemGuard paper recommendation;
#   keeps comparison set small and relevant to active session context.
_CONTRADICTION_WINDOW_H: int = 24

# _LLM_JUDGE_ENABLED — What: flag to enable/disable local LLM judge for
#   suspicious write detection.
# _LLM_JUDGE_ENABLED — Why: default-off to avoid heartbeat budget impact.
#   Phase 2 wires the path but does not enable it.  Operator can enable via
#   environment variable ASTHENOS_MEMORY_GUARD_LLM_JUDGE=1.
# _MAX_QUEUE_BYTES — What: per-file size cap for JSONL queue files.
# _MAX_QUEUE_BYTES — Why: backstop against runaway growth (cf. 2026-05-13
#   cascade that ballooned pending.jsonl to ~975k entries / 10 GB).  When a
#   queue file exceeds this cap, the current file is rotated to <path>.1
#   before the next append, keeping disk usage bounded.
_MAX_QUEUE_BYTES: int = 50 * 1024 * 1024  # 50 MB cap per memory-guard queue file

_LLM_JUDGE_ENABLED: bool = os.environ.get(
    "ASTHENOS_MEMORY_GUARD_LLM_JUDGE", "0"
) == "1"

# _BLOCK_FLAGGED — What: when True, flagged writes are blocked (not committed).
# _BLOCK_FLAGGED — Why: enforcement mode is opt-in via env var so operator can
#   switch from observation to blocking after tuning false-positive rate.
_BLOCK_FLAGGED: bool = os.environ.get(
    "ASTHENOS_MEMORY_GUARD_BLOCK_FLAGGED", "0"
) == "1"

# ---------------------------------------------------------------------------
# Staged-write log path + pending queue path
# ---------------------------------------------------------------------------

# STAGED_LOG — What: append-only JSONL log of every memory write that passes
#   through the staging buffer.
# STAGED_LOG — Why: full write surface capture for audit and Phase 2 validation.
STAGED_LOG = (
    Path.home() / ".local" / "share" / "asthenos" / "memory_guard" / "staged.jsonl"
)

# PENDING_LOG — What: JSONL file of writes flagged by validation that await
#   operator review (approve or discard).
# PENDING_LOG — Why: the Atoms MemoryGuard panel reads this file to surface
#   flagged writes for operator judgment.  Entries are removed on approve/discard.
PENDING_LOG = (
    Path.home() / ".local" / "share" / "asthenos" / "memory_guard" / "pending.jsonl"
)

# FEAT-2026-06-07 P3b/R2 -- The supersession-candidate store is OWNED by
# samia.runtime.contradiction (one owner, one schema, under <memory_dir>/biomimetic/).
# The run-1 surface-only SUPERSESSION_LOG (+ its _emit/list/confirm/dismiss markers and
# the stage_write wiring) were RECONCILED away: the operator OVERRODE the surface-only
# Q4a design to ONLINE auto-supersede (made safe by reversibility via restore_node), so
# the write seam (mcp_server.memory_write_node → _online_supersede) now records candidates
# directly into the canonical store and auto-retires the exact case. memory_guard no longer
# duplicates that store.

# ---------------------------------------------------------------------------
# In-memory statistics
# ---------------------------------------------------------------------------

# _stats_lock — What: protects all mutable module-level state.
# _stats_lock — Why: stage_write() may be called from multiple threads
#   (daemon IPC handlers, direct imports).
_stats_lock = threading.Lock()
_total: int = 0
_flagged_today: int = 0
_flagged_by_reason: Counter = Counter()
_recent: list[dict[str, Any]] = []  # last 100 entries (ring buffer)
_by_kind: Counter = Counter()
_by_caller: Counter = Counter()

# _RECENT_MAX — What: max entries kept in the in-memory recent ring buffer.
# _RECENT_MAX — Why: 100 is enough for operator inspection via
#   memory_guard_status without unbounded memory growth.
_RECENT_MAX = 100


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _payload_to_text(payload: dict[str, Any]) -> str:
    """Flatten payload dict to a searchable string.

    What: json-dumps the payload for regex and shingle analysis.
    Why:  payloads are dicts with varying structure; flattening ensures we
          catch injection strings regardless of nesting depth.
    """
    return json.dumps(payload, default=str)


def _check_injection(text: str) -> list[str]:
    """Check for prompt-injection markers in text.

    What: runs each pattern in _INJECTION_PATTERNS against the text.
    Why:  first heuristic line of defense per A-MemGuard framework --
          catches obvious injection templates before deeper analysis.

    Returns list of matched pattern descriptions (empty if clean).
    """
    reasons = []
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            reasons.append(f"injection_marker:{pat.pattern[:40]}")
    return reasons


def _check_contradiction(text: str) -> list[str]:
    """Check if text contradicts recent staged writes (shingle overlap).

    What: computes jaccard similarity between new write shingles and each
          recent staged write within the contradiction window.
    Why:  lightweight contradiction smell detector -- high overlap between
          a new write and a recent write suggests potential conflict.  Does
          not require LLM inference; uses same primitives as consolidation.

    Returns list of contradiction reasons (empty if clean).
    """
    try:
        from samia.core.consolidation import shingles, jaccard
    except ImportError:
        _log.debug("memory_guard: consolidation not importable; skip contradiction check")
        return []

    new_shingles = shingles(text)
    if not new_shingles:
        return []

    reasons = []
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        hours=_CONTRADICTION_WINDOW_H
    )
    cutoff_iso = cutoff.isoformat(timespec="seconds")

    with _stats_lock:
        recent_copy = list(_recent)

    for entry in recent_copy:
        if entry.get("staged_at", "") < cutoff_iso:
            continue
        entry_text = json.dumps(entry.get("payload", {}), default=str)
        entry_shingles = shingles(entry_text)
        if not entry_shingles:
            continue
        sim = jaccard(new_shingles, entry_shingles)
        if sim >= _CONTRADICTION_THRESHOLD:
            reasons.append(
                f"contradiction_smell:jaccard={sim:.3f}:vs={entry.get('write_id', '?')[:8]}"
            )
    return reasons


def _run_llm_judge(text: str) -> list[str]:
    """Optional local LLM judge for suspicious write detection.

    What: routes the write text to local inference and asks if it looks
          suspicious.  Returns reasons list if the judge says yes.
    Why:  deeper semantic analysis beyond heuristic patterns.  Default-off
          to avoid heartbeat budget impact.
    """
    if not _LLM_JUDGE_ENABLED:
        return []

    # Wire path present but default-off.  When enabled, would call:
    # from samia.runtime.inference import query_local
    # result = query_local("Is this memory write suspicious? ...")
    _log.debug("memory_guard: LLM judge path wired but not implemented (enable path)")
    return []


# ---------------------------------------------------------------------------
# Templated-content exclusion
# ---------------------------------------------------------------------------

# _TEMPLATED_TYPES — What: frontmatter type values whose content is
#   systematically generated (templates, not semantic user content).
# _TEMPLATED_TYPES — Why: BUG-2026-05-13-memory-guard-templated-content-exclusion.
#   These content classes trivially exceed the Jaccard >= 0.82 contradiction
#   threshold against any corpus of similar-shape templates. Running
#   contradiction_smell on them creates a positive-feedback loop where each
#   flagged write emits a bug record that itself gets flagged. Exclusion
#   applies ONLY to contradiction_smell; injection markers, LLM judge, and
#   AUD60 embedding checks still run.
#   + "semantic" (BUG-2026-06-10): fact-extracted atoms are machine-generated
#   from one template and mutually similar BY CONSTRUCTION — the backfill's
#   5,628 sem writes cascaded ~5,770 bug_mem nodes through contradiction_smell.
#   Real contradictions among atoms are the contradiction DETECTOR's job
#   (cosine+judge, type-scoped, includes semantic); the write-time jaccard
#   smell is the wrong tool for them.
_TEMPLATED_TYPES: set[str] = {"bug", "session_offload", "semantic"}

# _TEMPLATED_PATH_PATTERNS — What: basename regex patterns for content classes
#   that lack a canonical frontmatter type but are systematically templated.
# _TEMPLATED_PATH_PATTERNS — Why: provider benchmarks, sigil warnings, and
#   SEWE format blocks are emitted by automated generators with near-identical
#   structure. Until their emitters add a proper type field (D3 backlog), these
#   path patterns serve as a fallback exclusion gate.
_TEMPLATED_PATH_PATTERNS: list[re.Pattern] = [
    re.compile(r"^provider-.*-benchmarks\.md$"),
    re.compile(r"^loer_sigil_warnings_.*\.md$"),
    re.compile(r"^sewe_format_blocks_.*\.md$"),
    # What: matches bug record files by basename (e.g. bug_mem_guard_cascade.md,
    #   bug_mem_<hash>.md, bug_aud_<hash>.md).
    # Why: defense-in-depth for the 2026-05-13 cascade gap -- bug records whose
    #   payload had frontmatter_type=None slipped past the _TEMPLATED_TYPES check
    #   and triggered contradiction_smell. This path pattern catches them even
    #   when the type field is missing. bug_records.emit_bug_node names every node
    #   bug_<src3>_<hash>.md (e.g. bug_mem_* from source=memory_guard), and those
    #   nodes are SYSTEMATICALLY templated ("Flagged memory write: contradiction_
    #   smell...") so they trivially overlap each other -- running contradiction_
    #   smell on a bug-node write makes bug nodes flag OTHER bug nodes (the
    #   self-amplifying cascade). Treat any bug_*/bug_mem_* write as templated so
    #   contradiction_smell is SKIPPED for it.
    re.compile(r"^bug_.*\.md$"),
]


def _live_nodes_dir() -> Path | None:
    """Resolve the LIVE SAM/IA nodes directory a bug node would be written into.

    What: returns bug_records.NODES_DIR -- the exact directory emit_bug_node
          writes bug_<src>_<hash>.md files to. None if it cannot be resolved.
    Why:  fix #1 (test isolation) needs to know where a live bug node WOULD land
          so it can refuse to emit one when the flagged write's target is a
          pytest tempdir (e.g. /tmp/tmpXXXX/nodes/n.md) instead of live memory.
          Resolving from bug_records.NODES_DIR (rather than re-deriving a path)
          guarantees we compare against the *actual* write destination.
    """
    try:
        from samia.runtime.bug_records import NODES_DIR
        return Path(NODES_DIR).resolve()
    except Exception:
        return None


def _target_is_live(target: str | None) -> bool:
    """True only when the flagged write's target is under the live nodes dir.

    What: resolves the target path and the live nodes dir, returning True iff
          the target sits inside (or is) the live nodes dir.
    Why:  fix #1 -- a flagged write to a tempdir/test path must NOT cause a live
          bug node to be written. FAIL-SAFE: if either path cannot be resolved we
          return False (do NOT emit). A missed bug node is strictly better than
          the live-memory flood the tests were producing (~2556 spurious nodes).
    """
    if not target:
        return False
    live = _live_nodes_dir()
    if live is None:
        return False
    try:
        tgt = Path(target).resolve()
    except Exception:
        return False
    # target may be the node file itself or a dir under live nodes/.
    candidates = [tgt] + list(tgt.parents)
    return any(c == live for c in candidates)


def _is_templated_write(payload: dict[str, Any], target: str | None) -> bool:
    """Skip contradiction_smell for systematically-templated content classes.

    What: checks whether a write targets a templated content class by
          inspecting the payload's frontmatter_type and/or the target path
          basename against known patterns.
    Why:  unified gate that subsumes the former _is_bug_record_write path-based
          check (BUG-2026-05-13 cascade fix) and extends coverage to all
          observed template-shape surfaces (session_offload, provider benchmarks,
          sigil warnings, SEWE format blocks). Returns True to skip
          contradiction_smell; all other heuristics still run.
    """
    # Integrity-axis rewrites (erosion / recall-repair) re-write an EXISTING node's
    # body (eroded chars or anchor-restore) -- NOT new semantic content. They trivially
    # overlap the node's prior version + textually-similar nodes, so contradiction_smell
    # would false-positive and CASCADE (BUG-2026-06-08: decay's content-node rewrites
    # flagged 25 nodes -> bug_mem 1->24). Skip contradiction_smell for them; injection
    # markers / LLM judge / AUD60 embedding checks still run.
    if payload.get("integrity_rewrite"):
        return True

    # Type-driven check: payload may carry frontmatter_type from write_node.
    fm_type = payload.get("frontmatter_type")
    if fm_type and fm_type in _TEMPLATED_TYPES:
        return True

    # Path-pattern fallback for content classes lacking a type field.
    if target:
        name = Path(target).name
        for pat in _TEMPLATED_PATH_PATTERNS:
            if pat.match(name):
                return True

    return False


def _validate_write(payload: dict[str, Any], target: str = "") -> tuple[str, list[str], list[dict[str, Any]]]:
    """Run all validation checks on a payload.

    What: orchestrates injection check, contradiction smell, optional
          LLM judge, and AUD60 embedding-based contradiction detection.
          Returns (verdict, reasons, contradiction_metadata).
    Why:  single entry point for all validation logic; verdict drives
          commit/flag/block decision. contradiction_metadata is passed
          through to the pending queue for MemGuardPanel display.

    Returns:
        ("passed", [], []) if clean
        ("flagged", [reason1, ...], [meta1, ...]) if suspicious
        ("blocked", [reason1, ...], [meta1, ...]) if blocking is enabled and suspicious
    """
    text = _payload_to_text(payload)
    reasons: list[str] = []
    contradiction_meta: list[dict[str, Any]] = []

    # BUG-2026-05-13-templated-content-exclusion: determine if this is a
    # templated-content write (bug records, session offloads, provider
    # benchmarks, sigil warnings, SEWE format blocks).
    # What: unified gate via _is_templated_write replaces the former
    #       _is_bug_record_write + bug_*.md path check.
    # Why:  templated content trivially exceeds contradiction_smell Jaccard
    #       threshold; skipping prevents the cascade feedback loop. Other
    #       heuristics (injection markers, LLM judge, AUD60) still run.
    _skip_contradiction = _is_templated_write(payload, target)

    reasons.extend(_check_injection(text))
    if not _skip_contradiction:
        reasons.extend(_check_contradiction(text))
    reasons.extend(_run_llm_judge(text))

    # AUD60: embedding-based contradiction detection.
    # What: runs the full contradiction pipeline (embedding candidates + optional
    #   LLM judge) and appends any flagged contradictions to reasons.
    # Why: semantic contradiction detection catches conflicts that the
    #   shingle/jaccard heuristic misses (topically similar but contradictory claims).
    #   Skipped for templated content (same _skip_contradiction gate as the
    #   shingle smell): a bug node must never trip the embedding smell against
    #   OTHER bug nodes either -- both halves of contradiction_smell are gated so
    #   the self-amplifying cascade cannot reform via the AUD60 path.
    if not _skip_contradiction:
        try:
            from samia.runtime.contradiction import check_contradiction
            contra_reasons, contra_meta = check_contradiction(payload)
            reasons.extend(contra_reasons)
            contradiction_meta.extend(contra_meta)
        except ImportError:
            pass  # fail-open: contradiction module not available
        except Exception:
            _log.debug("memory_guard: AUD60 contradiction check failed", exc_info=True)

    if not reasons:
        return ("passed", [], [])

    if _BLOCK_FLAGGED:
        return ("blocked", reasons, contradiction_meta)
    return ("flagged", reasons, contradiction_meta)


# ---------------------------------------------------------------------------
# AUD84 Phase 2 -- Bug node emission on flagged write
# ---------------------------------------------------------------------------


def _emit_bug_node_on_flag(target: str, reasons: list[str], write_id: str) -> None:
    """Emit a bug node when a memory write is flagged.

    What: calls bug_records.emit_bug_node with source='memory_guard',
          surface set to the write target, severity='medium' default.
    Why:  AUD84 source hook (c) -- flagged writes enter the bug-discovery
          pipeline for operator triage in the Bugs tab.

    Fix #1 (test isolation): a LIVE bug node is written ONLY when the flagged
    write's TARGET is under the live nodes dir. A flagged write whose target is a
    pytest tempdir (/tmp/tmpXXXX/nodes/n.md) is in-process flagged + queued in
    pending.jsonl, but NOTHING is written to live memory. This stops every test
    run from polluting the live store (the ~2556-node flood incident). FAIL-SAFE:
    if the live dir can't be resolved, _target_is_live returns False -> we skip
    (the flood is worse than a missed bug node).
    """
    if not _target_is_live(target):
        _log.debug(
            "memory_guard: skip live bug node for non-live target %s "
            "(write_id=%s) -- flagged in-process only", target, write_id)
        return
    try:
        from samia.runtime.bug_records import emit_bug_node
        reason_summary = "; ".join(reasons[:3])
        emit_bug_node(
            source="memory_guard",
            surface=target,
            severity="medium",
            title=f"Flagged memory write: {reason_summary[:80]}",
            description=(
                f"Memory write to {target} was flagged by validation.\n"
                f"Write ID: {write_id}\n"
                f"Reasons: {', '.join(reasons)}\n"
                f"The write was held in pending.jsonl for operator review."
            ),
            evidence_path=str(PENDING_LOG),
        )
    except Exception:
        _log.debug("memory_guard: bug node emission failed", exc_info=True)


# ---------------------------------------------------------------------------
# Rotating-append + cheap line-count helpers
# ---------------------------------------------------------------------------


def _append_jsonl_rotating(path: Path, line: str) -> None:
    """Append one JSONL record; rotate to <path>.1 if file exceeds _MAX_QUEUE_BYTES.

    What: size-aware JSONL append with single-generation rotation.
    Why:  backstop against runaway growth (cf. 2026-05-13 cascade).  Rotating
          instead of truncating preserves one generation of history for forensics.
          The prior .1 file is overwritten on each rotation -- no unbounded fan-out.
    """
    try:
        if path.exists() and path.stat().st_size > _MAX_QUEUE_BYTES:
            os.replace(path, path.with_suffix(path.suffix + ".1"))  # overwrite prior .1
    except OSError:
        pass  # What: swallow stat/replace errors.  Why: append is more important than rotation.
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _count_lines(path: Path) -> int:
    """Count lines in a file without parsing JSON.

    What: binary-mode line count.
    Why:  guard_stats() previously called len(_read_pending()), which loaded +
          JSON-parsed the entire multi-GB pending.jsonl just to get a count.
          This is O(n) in bytes but avoids JSON parse overhead and object
          allocation.
    """
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Pending queue operations
# ---------------------------------------------------------------------------


def _write_pending(record: dict[str, Any]) -> None:
    """Append a flagged record to pending.jsonl.

    What: writes the flagged record to the pending queue file.
    Why:  the Atoms MemoryGuard panel reads this file to surface flagged
          writes for operator review.
    """
    try:
        PENDING_LOG.parent.mkdir(parents=True, exist_ok=True)
        # What: route through _append_jsonl_rotating instead of raw open().
        # Why: backstop against runaway pending.jsonl growth (2026-05-13 cascade).
        _append_jsonl_rotating(PENDING_LOG, json.dumps(record, default=str))
    except Exception:
        _log.warning("memory_guard: pending.jsonl write failed", exc_info=True)


def _read_pending() -> list[dict[str, Any]]:
    """Read all entries from pending.jsonl.

    What: parses the entire pending queue file.
    Why:  used by memory_guard_pending IPC op and approve/discard operations.
    """
    entries = []
    if not PENDING_LOG.exists():
        return entries
    try:
        with PENDING_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        _log.warning("memory_guard: pending.jsonl read failed", exc_info=True)
    return entries


def _rewrite_pending(entries: list[dict[str, Any]]) -> None:
    """Rewrite pending.jsonl with the given entries (removes deleted ones).

    What: atomically replaces pending.jsonl content.
    Why:  approve and discard need to remove entries from the queue.
    """
    try:
        PENDING_LOG.parent.mkdir(parents=True, exist_ok=True)
        tmp = PENDING_LOG.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
        tmp.replace(PENDING_LOG)
    except Exception:
        _log.warning("memory_guard: pending.jsonl rewrite failed", exc_info=True)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def stage_write(kind: str, target: str, payload: dict[str, Any],
                caller: str) -> dict[str, Any]:
    """Stage a memory write: validate, log, and commit or flag.

    What: records the write to staged.jsonl, runs heuristic validation,
          and either commits immediately (passed) or routes to pending.jsonl
          (flagged/blocked) for operator review.
    Why:  Phase 2 consensus validation prevents memory poisoning by catching
          injection attempts and contradictions before they enter the live
          memory store.  Flagged writes are held for operator judgment.

    Parameters
    ----------
    kind : str
        Write operation type (e.g. "add_edge", "write_node", "save_chain").
    target : str
        What is being written to (e.g. chain name, node path).
    payload : dict
        The write payload (caller-defined; logged verbatim).
    caller : str
        Identifier for the code path that triggered the write.

    Returns
    -------
    dict with keys:
        committed: bool -- True if write was committed, False if flagged/blocked
        write_id: str -- unique write identifier
        staged_at: str -- ISO timestamp
        verdict: str -- "passed" | "flagged" | "blocked"
        reasons: list[str] -- validation failure reasons (empty if passed)
    """
    write_id = str(uuid.uuid4())
    staged_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )

    # Run validation
    # What: _validate_write returns (verdict, reasons, contradiction_meta).
    # Why: AUD60 extends the return value with contradiction metadata for
    #   the MemGuardPanel to display which existing nodes conflict.
    #   target is passed so _validate_write can skip contradiction_smell
    #   for bug-record writes (BUG-2026-05-13 cascade fix).
    verdict, reasons, contradiction_meta = _validate_write(payload, target=target)

    committed = verdict == "passed"

    record: dict[str, Any] = {
        "write_id": write_id,
        "kind": kind,
        "target": target,
        "payload": payload,
        "caller": caller,
        "staged_at": staged_at,
        "committed": committed,
        "verdict": verdict,
        "reasons": reasons,
    }

    # AUD60 Phase 3: extend pending queue record with contradiction metadata.
    # What: adds contradiction_with field listing referenced node IDs and
    #   their similarity scores / judge explanations.
    # Why: the Atoms MemoryGuardPanel needs this to display contradiction
    #   details alongside the flagged write.
    if contradiction_meta:
        record["contradiction_with"] = contradiction_meta

    # Append to disk log (best-effort; failure must not block the write).
    # What: JSONL append to staged.jsonl via rotating helper.
    # Why: full audit trail regardless of verdict; rotation caps file size
    #      (backstop against runaway growth, cf. 2026-05-13 cascade).
    try:
        STAGED_LOG.parent.mkdir(parents=True, exist_ok=True)
        _append_jsonl_rotating(STAGED_LOG, json.dumps(record, default=str))
    except Exception:
        _log.warning("memory_guard: staged.jsonl write failed (non-fatal)",
                     exc_info=True)

    # If flagged or blocked, write to pending queue for operator review.
    if verdict in ("flagged", "blocked"):
        _write_pending(record)
        # AUD84-Phase2-HookC — What: emit a bug node for flagged writes.
        # AUD84-Phase2-HookC — Why: AUD84 D4 source (e) -- memory-guard
        #     flagged writes flow into the bug-discovery pipeline. Dedup in
        #     emit_bug_node prevents repeat flags from flooding (D7).
        _emit_bug_node_on_flag(target, reasons, write_id)
        # FEAT-2026-06-07 P3b/R2 -- supersession candidates are no longer emitted
        # here. The operator OVERRODE the surface-only Q4a design to ONLINE
        # auto-supersede (made safe by reversibility via restore_node); the write
        # seam (mcp_server.memory_write_node → _online_supersede) now owns
        # detection + recording into the canonical contradiction store.

    # Update in-memory statistics.
    with _stats_lock:
        global _total, _flagged_today
        _total += 1
        _by_kind[kind] += 1
        _by_caller[caller] += 1
        _recent.append(record)
        if len(_recent) > _RECENT_MAX:
            _recent.pop(0)
        if verdict in ("flagged", "blocked"):
            _flagged_today += 1
            for r in reasons:
                category = r.split(":")[0] if ":" in r else r
                _flagged_by_reason[category] += 1

    _log.debug(
        "memory_guard: staged write %s kind=%s target=%s caller=%s verdict=%s",
        write_id, kind, target, caller, verdict,
    )
    if verdict != "passed":
        _log.info(
            "memory_guard: FLAGGED write %s verdict=%s reasons=%s",
            write_id, verdict, ";".join(reasons),
        )

    return {
        "committed": committed,
        "write_id": write_id,
        "staged_at": staged_at,
        "verdict": verdict,
        "reasons": reasons,
    }


def statistics() -> dict[str, Any]:
    """Return staging buffer statistics.

    What: returns total count, last-hour count, breakdowns by kind and caller,
          and the most recent staged writes.
    Why:  operator visibility into the write surface.  Exposed via the
          memory_guard_status IPC op.
    """
    now = time.time()
    one_hour_ago_iso = datetime.datetime.fromtimestamp(
        now - 3600, tz=datetime.timezone.utc
    ).isoformat(timespec="seconds")

    with _stats_lock:
        last_hour_count = sum(
            1 for r in _recent if r.get("staged_at", "") >= one_hour_ago_iso
        )
        return {
            "total": _total,
            "last_hour_count": last_hour_count,
            "by_kind": dict(_by_kind),
            "by_caller": dict(_by_caller),
            "recent_count": len(_recent),
            "recent": list(_recent[-10:]),  # last 10 for quick view
        }


def pending_list() -> list[dict[str, Any]]:
    """Return the current pending.jsonl entries.

    What: reads and returns all flagged writes awaiting operator review.
    Why:  consumed by the Atoms MemoryGuard panel to display the review queue.
    """
    return _read_pending()


def approve_write(write_id: str) -> dict[str, Any]:
    """Approve a flagged write: commit it and remove from pending.

    What: finds the write in pending.jsonl by write_id, marks it committed,
          removes it from the pending queue.
    Why:  operator judgment overrides heuristic flags -- false positives
          need a path to promotion.

    Returns dict with "ok": True on success, "error": str on failure.
    """
    entries = _read_pending()
    found = None
    remaining = []
    for entry in entries:
        if entry.get("write_id") == write_id:
            found = entry
        else:
            remaining.append(entry)

    if found is None:
        return {"ok": False, "error": f"write_id {write_id} not found in pending"}

    # Mark as committed in staged log (append an approval record).
    approval_record = {
        "write_id": write_id,
        "action": "approved",
        "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "original_verdict": found.get("verdict", "unknown"),
    }
    try:
        STAGED_LOG.parent.mkdir(parents=True, exist_ok=True)
        # What: route through _append_jsonl_rotating.
        # Why: approval records also contribute to staged.jsonl growth.
        _append_jsonl_rotating(STAGED_LOG, json.dumps(approval_record, default=str))
    except Exception:
        _log.warning("memory_guard: staged.jsonl approval write failed", exc_info=True)

    _rewrite_pending(remaining)
    _log.info("memory_guard: approved write %s", write_id)
    return {"ok": True, "write_id": write_id, "action": "approved"}


def discard_write(write_id: str) -> dict[str, Any]:
    """Discard a flagged write: remove from pending without committing.

    What: finds the write in pending.jsonl by write_id, removes it without
          promoting to the live memory store.
    Why:  operator confirms the heuristic flag was correct -- the write was
          indeed suspicious and should not be committed.

    Returns dict with "ok": True on success, "error": str on failure.
    """
    entries = _read_pending()
    found = None
    remaining = []
    for entry in entries:
        if entry.get("write_id") == write_id:
            found = entry
        else:
            remaining.append(entry)

    if found is None:
        return {"ok": False, "error": f"write_id {write_id} not found in pending"}

    # Log the discard action.
    discard_record = {
        "write_id": write_id,
        "action": "discarded",
        "discarded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "original_verdict": found.get("verdict", "unknown"),
    }
    try:
        STAGED_LOG.parent.mkdir(parents=True, exist_ok=True)
        # What: route through _append_jsonl_rotating.
        # Why: discard records also contribute to staged.jsonl growth.
        _append_jsonl_rotating(STAGED_LOG, json.dumps(discard_record, default=str))
    except Exception:
        _log.warning("memory_guard: staged.jsonl discard write failed", exc_info=True)

    _rewrite_pending(remaining)
    _log.info("memory_guard: discarded write %s", write_id)
    return {"ok": True, "write_id": write_id, "action": "discarded"}


def guard_stats() -> dict[str, Any]:
    """Return aggregate guard statistics for the Atoms panel.

    What: returns flagged_today count, breakdown by reason category, and
          pending queue length.
    Why:  the Atoms MemoryGuard panel stats header shows aggregate metrics
          so the operator can assess the threat surface at a glance.
    """
    # What: cheap line count instead of full JSON parse.
    # Why: _read_pending() loaded + parsed the entire multi-GB file just to
    #   count entries.  _count_lines is O(n) in bytes with zero allocation.
    pending_count = _count_lines(PENDING_LOG)
    with _stats_lock:
        return {
            "total_writes": _total,
            "flagged_today": _flagged_today,
            "by_reason": dict(_flagged_by_reason),
            "pending_count": pending_count,
        }


# ---------------------------------------------------------------------------
# IPC handlers
# ---------------------------------------------------------------------------


def _handle_memory_guard_status(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for the memory_guard_status op.

    What: returns statistics() result.
    Why:  lets the operator (or Atoms panel) query the staging surface health
          over the daemon socket without importing this module directly.
    """
    return statistics()


def _handle_memory_guard_pending(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for the memory_guard_pending op.

    What: returns current pending.jsonl entries.
    Why:  the Atoms MemoryGuard panel polls this to display the review queue.
    """
    return {"entries": pending_list()}


def _handle_memory_guard_approve(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for the memory_guard_approve op.

    What: approves a flagged write by write_id.
    Why:  called by the Atoms panel when operator clicks Approve.
    """
    write_id = args.get("write_id", "")
    if not write_id:
        raise ValueError("write_id required")
    result = approve_write(write_id)
    # What: approve_write returns {"ok": False, "error": ...} on not-found.
    # Why: IPC dispatcher wraps in {"ok": True, "result": ...}; returning
    #      an inner {"ok": False} would be swallowed. Raise instead.
    if not result.get("ok"):
        raise ValueError(result.get("error", "approve failed"))
    return {"write_id": result["write_id"], "action": result["action"]}


def _handle_memory_guard_discard(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for the memory_guard_discard op.

    What: discards a flagged write by write_id.
    Why:  called by the Atoms panel when operator clicks Discard.
    """
    write_id = args.get("write_id", "")
    if not write_id:
        raise ValueError("write_id required")
    result = discard_write(write_id)
    # What: discard_write returns {"ok": False, "error": ...} on not-found.
    # Why: IPC dispatcher wraps in {"ok": True, "result": ...}; returning
    #      an inner {"ok": False} would be swallowed. Raise instead.
    if not result.get("ok"):
        raise ValueError(result.get("error", "discard failed"))
    return {"write_id": result["write_id"], "action": result["action"]}


def _handle_memory_guard_stats(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for the memory_guard_stats op.

    What: returns aggregate guard statistics.
    Why:  consumed by the Atoms panel stats header for at-a-glance metrics.
    """
    return guard_stats()


# ---------------------------------------------------------------------------
# Op registration
# ---------------------------------------------------------------------------


def register_ops() -> None:
    """Register all memory_guard IPC ops with the IPC server.

    What: calls samia.runtime.ipc.register_op for each memory_guard operation.
    Why:  the daemon calls this during startup, after IPC server creation,
          matching the pattern established by samia.runtime.heartbeat.
          Phase 2/3 adds pending, approve, discard, and stats ops.
    """
    from samia.runtime.ipc import register_op
    register_op("memory_guard_status", _handle_memory_guard_status)
    register_op("memory_guard_pending", _handle_memory_guard_pending)
    register_op("memory_guard_approve", _handle_memory_guard_approve)
    register_op("memory_guard_discard", _handle_memory_guard_discard)
    register_op("memory_guard_stats", _handle_memory_guard_stats)
    _log.info(
        "memory_guard ops registered (phase 2+3: validation + review queue, "
        "block_flagged=%s, llm_judge=%s)",
        _BLOCK_FLAGGED, _LLM_JUDGE_ENABLED,
    )


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.memory_guard
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD48 -- Phases 1-3 (staging + validation + operator review)
#             AUD60 -- embedding-based contradiction detection integrated
#             AUD84 -- Phase 2 hook (c): emit bug node on flagged write
#             BUG-2026-05-13 -- templated-content exclusion (subsumes bug-only)
#             2026-05-30 -- producer-side hardening (size-cap rotation, cheap
#                           line count for guard_stats, bug_*.md path pattern)
#             FEAT-2026-06-07 P3b/R2 -- supersession candidate store RECONCILED out
#                           of memory_guard (operator OVERRODE surface-only Q4a to
#                           ONLINE auto-supersede); the canonical store + record/
#                           list/confirm/dismiss now live in runtime.contradiction
#                           and the write seam (mcp_server._online_supersede).
#             BUG-2026-06-07 -- live-memory pollution fix. (1) _emit_bug_node_on_flag
#                           now emits a LIVE bug node ONLY when the flagged write's
#                           target is under the live nodes dir (_target_is_live /
#                           _live_nodes_dir resolve bug_records.NODES_DIR); pytest
#                           tempdir writes no longer flood live memory (~2556-node
#                           incident). (2) AUD60 embedding contradiction check is now
#                           gated by the same _skip_contradiction flag so bug nodes
#                           can't flag other bug nodes via either smell path.
# Layer:      runtime (in-daemon, JSONL logs + IPC ops)
# Role:       memory-write defense — a staging buffer that logs every write, runs
#             heuristic + embedding contradiction validation, and routes flagged
#             writes to a pending.jsonl queue for operator approve/discard.
# Stability:  v0.3.4 -- supersession surfacer removed (single owner reconciliation)
# ErrorModel: fail-open for staging log; flagged writes held in pending.jsonl;
#             IPC handlers never raise; approve/discard are atomic rewrites.
#             AUD60 contradiction check is fail-open. AUD84 bug emission is
#             fail-open (never blocks the write).
# Depends:    threading, json, uuid, re, os (stdlib).
#             samia.runtime.ipc (register_op).
#             samia.core.consolidation (shingles, jaccard) -- optional.
#             samia.runtime.contradiction (AUD60) -- optional.
#             samia.runtime.bug_records (emit_bug_node) -- optional, fail-open.
# Exposes:    stage_write, statistics, pending_list, approve_write,
#             discard_write, guard_stats, register_ops, STAGED_LOG, PENDING_LOG.
#             append paths, cheap _count_lines for guard_stats, bug_*.md
#             templated-path pattern for defense-in-depth).
# Lines:      989
# --------------------------------------------------------------------------
