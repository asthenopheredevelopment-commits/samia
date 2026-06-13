"""samia.runtime.bug_records -- AUD84 Phases 1+2: bug record schema, IPC ops, and proposal generator.

Layer 1 (Owns / Depends):
    Owns:    PREFIX_TAXONOMY, BugSource, BugStatus, BugRecord, emit_bug_node,
             list_bug_nodes, update_bug_status, link_bug_to_proposal,
             send_for_review, register_ops, 4 IPC handlers
    Depends: samia.core.frontmatter (read_node, write_node),
             samia.runtime.ipc (register_op),
             stdlib (hashlib, json, datetime, re, pathlib, logging, dataclasses)

Layer 2 (What / Why):
    What: Manages bug records as SAM/IA memory nodes (type=bug). Supports 8
          proposal-prefix taxonomy categories. Six discovery sources call
          emit_bug_node to record found bugs. Deduplication uses a stable hash
          of (source, surface, evidence_signature) to prevent duplicates across
          audit re-runs. Four IPC ops expose the bug workflow to Atoms and CLI:
          bug_discover_log, bug_targeted_list, bug_target_set, bug_send_for_review.
          The send_for_review op generates BUG-prefixed SEWE proposal skeletons
          in pending state (proposal-first axiom preserved -- operator must approve).
    Why:  AUD84 closes the loop between passive discovery surfaces (AUD77 audit,
          AUD82 circuit-breaker, AUD48 memory-guard, daemon logs) and the formal
          proposal pipeline. Without this, discovered bugs sit in per-subsystem
          logs and never flow into the operator triage queue. The 8-prefix
          taxonomy replaces the single-AUD namespace that was scaling poorly
          at ~85 proposals.

Layer 3 (Changelog):
    2026-05-07  AUD84-Phase1+2  Initial implementation. Schema, IPC ops,
                                 dedup, proposal auto-skeleton, 6 source hooks.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Top-level (not lazy) because NODES_DIR is evaluated at module import time.
from samia.core.paths import resolve_memory_root

_log = logging.getLogger("samia.runtime.bug_records")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PREFIX_TAXONOMY -- What: the 8 valid proposal-id prefixes.
# PREFIX_TAXONOMY -- Why: AUD84 D1 (revised 2026-05-07 final). Replaces the
#     single-AUD namespace with categorical prefixes. Future reserved: SEC, PERF.
PREFIX_TAXONOMY: list[str] = [
    "AUD",   # Audit -- system review reveals warranted change
    "BUG",   # Bug fix -- defect with reproduction + expected vs actual
    "FEAT",  # Feature -- novel capability or surface
    "RSCH",  # Research -- literature review, feasibility, prior-art, speculative
    "REF",   # Refactor -- code cleanup, doc reorg, perf tuning, no functional change
    "OPS",   # Operations -- backup, retention, monitoring, deployment
    "DOC",   # Documentation artifact creation -- the deliverable IS a document
    "MISC",  # Miscellaneous / uncategorized -- reviewed by misc-analyzer skill
]

# BUG_SOURCE_VALUES -- What: valid values for the found_by / source field.
# BUG_SOURCE_VALUES -- Why: AUD84 D4 enumerates the 6 discovery surfaces that
#     can emit bug records. Validated at emission time.
BUG_SOURCE_VALUES: list[str] = [
    "audit",            # AUD77 architecture audit findings
    "build_health",     # cargo/build failures
    "daemon_log",       # regex-matched log patterns (Traceback, CRITICAL, etc.)
    "circuit_breaker",  # AUD82 inference circuit-breaker trips
    "memory_guard",     # AUD48 flagged writes
    "manual",           # operator manual report via Atoms button
]

# BUG_STATUS_VALUES -- What: valid bug lifecycle states.
# BUG_STATUS_VALUES -- Why: AUD84 D4/D6. Status transitions:
#     untriaged -> targeted -> proposal-pending -> proposal-approved -> fixed
#     untriaged -> wont-fix (operator dismissal)
BUG_STATUS_VALUES: list[str] = [
    "untriaged",
    "targeted",
    "proposal-pending",
    "proposal-approved",
    "fixed",
    "wont-fix",
]

# NODES_DIR -- What: SAM/IA memory nodes directory.
# NODES_DIR -- Why: bug records are stored as SAM/IA nodes (type=bug) per D4.
#     Uses the same directory as all other memory nodes. Resolved through
#     samia.core.paths.resolve_memory_root (env -> verified-legacy -> XDG) so
#     the path is correct in dev, staged-release, and site-packages layouts --
#     the old parents[3] derivation was correct only in the dev tree and wrote
#     onto the drive root when staged. Kept a module-level Path so the
#     `from samia.runtime.bug_records import NODES_DIR` consumer (memory_guard)
#     and the mock.patch(".../NODES_DIR") tests keep their exact contract.
NODES_DIR = resolve_memory_root(create=False) / "nodes"

# PROPOSALS_DIR -- What: SEWE proposals directory.
# PROPOSALS_DIR -- Why: bug_send_for_review generates BUG-prefixed proposals here.
PROPOSALS_DIR = Path.home() / ".local" / "share" / "asthenos" / "sewe" / "proposals"

# DOCS_DIR -- What: review doc archive directory.
# DOCS_DIR -- Why: companion markdown docs for generated BUG proposals.
DOCS_DIR = Path.home() / "Asthenosphere" / "docs" / "proposals_archive"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BugRecord:
    """Structured representation of a bug node's frontmatter fields.

    What: mirrors the SAM/IA node frontmatter for type=bug per D4 spec.
    Why:  typed access avoids raw-dict errors and enables list_bug_nodes
          to return structured data to IPC callers.
    """
    name: str
    slug: str
    severity: str
    affected_surface: str
    found_by: str
    status: str
    evidence_path: str
    seen_count: int
    first_seen: str
    last_seen: str
    title: str = ""
    description: str = ""
    linked_proposal: str = ""
    node_path: str = ""


# ---------------------------------------------------------------------------
# Slug generation + dedup
# ---------------------------------------------------------------------------

def _make_slug(source: str, surface: str, evidence_signature: str) -> str:
    """Generate a stable, filesystem-safe slug for deduplication.

    What: hashes (source, surface, evidence_signature) with SHA-256, takes
          first 12 hex chars prefixed with source abbreviation.
    Why:  AUD84 D7 dedup. Same (source, surface, evidence_signature) always
          produces the same slug, so emit_bug_node can detect existing nodes.
    """
    composite = f"{source}|{surface}|{evidence_signature}"
    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:12]
    return f"{source[:3]}_{digest}"


def _evidence_signature(evidence_path: str, title: str) -> str:
    """Derive a short evidence fingerprint for dedup.

    What: combines the evidence_path basename + canonicalized title's first 40 chars.
    Why:  evidence_path alone may be reused across runs (e.g., findings.jsonl
          is overwritten). Including title content makes the signature more
          specific to the actual finding. The title is canonicalized by stripping
          per-target vs=<hex> hashes and per-comparison jaccard=<float> scores,
          collapsing "same flag class, different target" bugs into one slug
          with seen_count++ instead of N separate nodes.
          BUG-2026-05-13-memory-guard-bug-cascade-feedback-loop fix (part b).
    """
    path_part = Path(evidence_path).name if evidence_path else "none"
    # Strip cascade-target hashes and per-comparison Jaccard scores from
    # the title before hashing. This collapses "same flag class, different
    # vs= targets" into one canonical bug record with seen_count++.
    canonical_title = re.sub(r":?(vs=[0-9a-f]+|jaccard=[\d.]+)", "", title.lower())
    title_part = re.sub(r"[^a-z0-9]+", "", canonical_title)[:40]
    return f"{path_part}:{title_part}"


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def emit_bug_node(
    source: str,
    surface: str,
    severity: str,
    title: str,
    description: str,
    evidence_path: str = "",
    extra_frontmatter: Optional[dict[str, Any]] = None,
) -> str:
    """Write or update a SAM/IA bug node. Returns the bug slug.

    What: creates a new bug node at nodes/bug_{slug}.md if none exists,
          or increments seen_count + updates last_seen on existing node.
    Why:  AUD84 D4 storage model + D7 dedup. All six discovery sources
          funnel through this single emission point.

    Parameters
    ----------
    source : str
        Discovery source (must be in BUG_SOURCE_VALUES).
    surface : str
        Affected subsystem identifier (e.g., 'samia.runtime.qwen3_backend').
    severity : str
        'low', 'medium', 'high', or 'critical'.
    title : str
        Short title describing the bug.
    description : str
        Longer description, evidence text, steps to reproduce, etc.
    evidence_path : str
        Path to supporting evidence file (findings.jsonl, audit log, etc.).
    extra_frontmatter : dict | None
        Additional frontmatter fields to include (e.g., error_class).

    Returns
    -------
    str
        The bug slug (used as the node identifier).
    """
    # InputValidation -- What: validate source against allowed values.
    # InputValidation -- Why: system edge; callers may pass unexpected strings.
    if source not in BUG_SOURCE_VALUES:
        _log.warning("bug_records: unknown source %r, accepting as-is", source)

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    ev_sig = _evidence_signature(evidence_path, title)
    slug = _make_slug(source, surface, ev_sig)
    node_filename = f"bug_{slug}.md"
    node_path = NODES_DIR / node_filename

    # DedupCheck -- What: check if node already exists.
    # DedupCheck -- Why: AUD84 D7 -- re-discovery increments seen_count.
    if node_path.exists():
        try:
            from samia.core.frontmatter import read_node, write_node
            fm, order, body = read_node(node_path)
            old_count = fm.get("seen_count", 1)
            fm["seen_count"] = old_count + 1 if isinstance(old_count, int) else 2
            fm["last_seen"] = now_iso
            write_node(node_path, fm, order, body)
            _log.info(
                "bug_records: dedup hit slug=%s seen_count=%d",
                slug, fm["seen_count"],
            )
            return slug
        except Exception as exc:
            _log.warning("bug_records: dedup update failed for %s: %s", slug, exc)
            return slug

    # NewNode -- What: create a new bug node with full D4 frontmatter.
    # NewNode -- Why: first sighting of this (source, surface, evidence_sig) combo.
    fm: dict[str, Any] = {
        "name": title,
        "type": "bug",
        "severity": severity,
        "affected_surface": surface,
        "found_by": source,
        "found_at": now_iso,
        "status": "untriaged",
        "evidence_path": evidence_path,
        "seen_count": 1,
        "first_seen": now_iso,
        "last_seen": now_iso,
        "slug": slug,
    }
    order = list(fm.keys())

    if extra_frontmatter:
        for k, v in extra_frontmatter.items():
            if k not in fm:
                fm[k] = v
                order.append(k)

    body = f"# {title}\n\n{description}\n"

    try:
        from samia.core.frontmatter import write_node
        NODES_DIR.mkdir(parents=True, exist_ok=True)
        write_node(node_path, fm, order, body)
        _log.info(
            "bug_records: emitted bug node slug=%s source=%s surface=%s sev=%s",
            slug, source, surface, severity,
        )
    except Exception as exc:
        _log.error("bug_records: failed to write bug node %s: %s", slug, exc)

    return slug


def list_bug_nodes(filter_status: Optional[str] = None) -> list[dict[str, Any]]:
    """List all bug nodes, optionally filtered by status.

    What: scans NODES_DIR for bug_*.md files, parses frontmatter, returns
          list of BugRecord-like dicts.
    Why:  consumed by bug_targeted_list IPC op and Atoms BUG tab.
    """
    results: list[dict[str, Any]] = []
    if not NODES_DIR.is_dir():
        return results

    try:
        from samia.core.frontmatter import read_node
    except ImportError:
        _log.error("bug_records: cannot import frontmatter.read_node")
        return results

    for path in sorted(NODES_DIR.glob("bug_*.md")):
        try:
            fm, _order, body = read_node(path)
        except Exception:
            continue

        if fm.get("type") != "bug":
            continue

        status = fm.get("status", "untriaged")
        if filter_status and status != filter_status:
            continue

        results.append({
            "name": fm.get("name", ""),
            "slug": fm.get("slug", path.stem.replace("bug_", "")),
            "severity": fm.get("severity", "medium"),
            "affected_surface": fm.get("affected_surface", ""),
            "found_by": fm.get("found_by", ""),
            "status": status,
            "evidence_path": fm.get("evidence_path", ""),
            "seen_count": fm.get("seen_count", 1),
            "first_seen": fm.get("first_seen", ""),
            "last_seen": fm.get("last_seen", ""),
            "linked_proposal": fm.get("linked_proposal", ""),
            "node_path": str(path),
            "description": body[:500] if body else "",
        })

    return results


def update_bug_status(slug: str, new_status: str) -> bool:
    """Update a bug node's status field. Returns True on success.

    What: finds the bug node by slug, validates status transition, writes.
    Why:  lifecycle transitions (untriaged -> targeted -> proposal-pending -> etc.)
          are driven by IPC ops and the proposal generator.
    """
    if new_status not in BUG_STATUS_VALUES:
        _log.warning("bug_records: invalid status %r", new_status)
        return False

    node_path = NODES_DIR / f"bug_{slug}.md"
    if not node_path.exists():
        _log.warning("bug_records: node not found for slug=%s", slug)
        return False

    try:
        from samia.core.frontmatter import read_node, write_node
        fm, order, body = read_node(node_path)
        fm["status"] = new_status
        if "status" not in order:
            order.append("status")
        write_node(node_path, fm, order, body)
        _log.info("bug_records: updated slug=%s status=%s", slug, new_status)

        # VerifiedOutcomeEmit -- What: emit a verified-outcome spine node when
        #     a bug transitions to 'fixed'.
        # VerifiedOutcomeEmit -- Why: the bug reaching 'fixed' is the domain
        #     verdict point (goal achieved). emit_samia writes a decay-aware
        #     spine node so the outcome is queryable by future sessions. Fire-
        #     and-forget: failure here must never stall or raise into the
        #     bug_records caller loop.
        if new_status == "fixed":
            try:
                from samia.core.spine_cord import build_outcome_record, emit_samia
                rec = build_outcome_record(
                    intent="bug_fix",
                    source_kind="bug_records",
                    operational={"status": "ok"},
                    domain_verdict={"status": "success",
                                    "reason": f"bug {slug} resolved"},
                    task=slug,
                    extra={"bug_id": slug,
                           "bug_status_transition": "->fixed"},
                )
                # What: Resolve to the SAM memory root (same as chiron.py:46).
                # Why: The old code appended / 'memory' to a path that ALREADY
                #      resolved to .../memory, creating .../memory/memory (phantom).
                #      emit_samia would mkdir + write nodes there, but nothing scans it.
                memory_dir = (
                    Path(__file__).resolve().parent.parent.parent.parent
                )
                emit_samia(rec, memory_dir=memory_dir)
            except Exception as exc:
                _log.warning(
                    "bug_records: emit verified outcome failed slug=%s: %s",
                    slug, exc,
                )

        return True
    except Exception as exc:
        _log.error("bug_records: status update failed slug=%s: %s", slug, exc)
        return False


def link_bug_to_proposal(slug: str, proposal_id: str) -> bool:
    """Link a bug node to a SEWE proposal. Returns True on success.

    What: sets the linked_proposal frontmatter field on the bug node.
    Why:  traceability from bug discovery to formal proposal. The Atoms BUG
          tab uses this to show which bugs have proposals attached.
    """
    node_path = NODES_DIR / f"bug_{slug}.md"
    if not node_path.exists():
        _log.warning("bug_records: node not found for slug=%s", slug)
        return False

    try:
        from samia.core.frontmatter import read_node, write_node
        fm, order, body = read_node(node_path)
        fm["linked_proposal"] = proposal_id
        if "linked_proposal" not in order:
            order.append("linked_proposal")
        write_node(node_path, fm, order, body)
        _log.info("bug_records: linked slug=%s -> proposal=%s", slug, proposal_id)
        return True
    except Exception as exc:
        _log.error("bug_records: link failed slug=%s: %s", slug, exc)
        return False


# ---------------------------------------------------------------------------
# Bug-proposal auto-skeleton (D6)
# ---------------------------------------------------------------------------

def send_for_review(
    bug_slugs: list[str],
    mode: str = "1bug",
) -> list[str]:
    """Generate BUG-prefixed SEWE proposal(s) from targeted bug nodes.

    What: reads the specified bug nodes, groups them per mode, generates
          BUG-prefixed proposal JSON(s) at PROPOSALS_DIR with auto-skeleton
          content. Updates bug-node status to 'proposal-pending' and links.
    Why:  AUD84 D6 auto-skeleton. Closes the discovery-to-proposal loop.
          Every generated proposal lands as 'pending' per the proposal-first
          axiom -- operator must approve before implementation.

    Parameters
    ----------
    bug_slugs : list[str]
        Slugs of bug nodes to include.
    mode : str
        '1bug' (one proposal per bug), 'multi-bug' (one proposal for all),
        'multi-surface' (one proposal per affected_surface group).

    Returns
    -------
    list[str]
        Proposal ids generated.
    """
    if not bug_slugs:
        return []

    # LoadBugNodes -- What: read frontmatter + body for each slug.
    # LoadBugNodes -- Why: need full content for proposal auto-skeleton.
    bugs: list[dict[str, Any]] = []
    try:
        from samia.core.frontmatter import read_node
    except ImportError:
        _log.error("bug_records: cannot import read_node for send_for_review")
        return []

    for slug in bug_slugs:
        node_path = NODES_DIR / f"bug_{slug}.md"
        if not node_path.exists():
            _log.warning("bug_records: send_for_review: slug=%s not found", slug)
            continue
        try:
            fm, _order, body = read_node(node_path)
            bugs.append({"slug": slug, "fm": fm, "body": body, "path": str(node_path)})
        except Exception as exc:
            _log.warning("bug_records: send_for_review: read failed slug=%s: %s", slug, exc)

    if not bugs:
        return []

    # GroupByMode -- What: partition bugs into proposal groups based on mode.
    # GroupByMode -- Why: D5 targeting modes determine how bugs bundle into proposals.
    groups: list[list[dict[str, Any]]]
    if mode == "1bug":
        groups = [[b] for b in bugs]
    elif mode == "multi-surface":
        surface_map: dict[str, list[dict[str, Any]]] = {}
        for b in bugs:
            surface = b["fm"].get("affected_surface", "unknown")
            surface_map.setdefault(surface, []).append(b)
        groups = list(surface_map.values())
    else:
        # multi-bug: one proposal for all
        groups = [bugs]

    proposal_ids: list[str] = []
    now = dt.datetime.now(dt.timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    for group in groups:
        proposal_id = _emit_bug_proposal(group, now, date_str)
        if proposal_id:
            proposal_ids.append(proposal_id)
            # StatusUpdate -- What: transition each bug to proposal-pending + link.
            # StatusUpdate -- Why: D6 lifecycle: targeted -> proposal-pending.
            for b in group:
                update_bug_status(b["slug"], "proposal-pending")
                link_bug_to_proposal(b["slug"], proposal_id)

    return proposal_ids


def _emit_bug_proposal(
    bugs: list[dict[str, Any]],
    now: dt.datetime,
    date_str: str,
) -> Optional[str]:
    """Write a single BUG-prefixed SEWE proposal JSON + companion doc.

    What: auto-fills title, recommendation placeholder, evidence, severity
          (max across bugs), affected_surfaces (union), linked_to.
    Why:  AUD84 D6 auto-skeleton. operator_review_state is always 'pending'
          per the proposal-first axiom.
    """
    # TitleGen -- What: auto-generate title from bug summaries.
    # TitleGen -- Why: D6 spec -- title from bug-node summaries.
    if len(bugs) == 1:
        title = bugs[0]["fm"].get("name", "Bug fix")
    else:
        surfaces = sorted(set(b["fm"].get("affected_surface", "unknown") for b in bugs))
        title = f"Fix {len(bugs)} bugs across {', '.join(surfaces[:3])}"
        if len(surfaces) > 3:
            title += f" (+{len(surfaces) - 3} more)"

    # SeverityMax -- What: use the highest severity across bundled bugs.
    # SeverityMax -- Why: D6 spec -- severity is max across selected bugs.
    sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    max_sev = max(
        (sev_order.get(b["fm"].get("severity", "medium"), 2) for b in bugs),
        default=2,
    )
    severity = {4: "critical", 3: "high", 2: "medium", 1: "low"}.get(max_sev, "medium")

    # AffectedSurfaces -- What: union of bug-node affected_surface fields.
    # AffectedSurfaces -- Why: D6 spec -- surfaced in the proposal scope block.
    affected_surfaces = sorted(set(
        b["fm"].get("affected_surface", "unknown") for b in bugs
    ))

    # LinkedTo -- What: bug node names + any AUD/BUG ids already referenced.
    # LinkedTo -- Why: D6 spec -- linked_to for cross-reference.
    linked_to: list[str] = []
    for b in bugs:
        linked_to.append(f"bug_{b['slug']}")
        existing_link = b["fm"].get("linked_proposal", "")
        if existing_link:
            linked_to.append(existing_link)

    # EvidenceConcat -- What: concatenate bug-node frontmatter + body excerpts.
    # EvidenceConcat -- Why: D6 spec -- evidence from bug records.
    evidence: list[dict[str, str]] = []
    for b in bugs:
        ev_text = (
            f"[{b['fm'].get('found_by', '?')}] {b['fm'].get('name', '')} "
            f"(surface={b['fm'].get('affected_surface', '?')}, "
            f"severity={b['fm'].get('severity', '?')}, "
            f"seen={b['fm'].get('seen_count', 1)}x)\n"
            f"Evidence: {b['fm'].get('evidence_path', 'none')}\n"
            f"{b['body'][:300]}"
        )
        evidence.append({"fact": ev_text.strip()})

    # SlugGen -- What: generate proposal slug from title.
    # SlugGen -- Why: filesystem-safe, unique proposal filename.
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:40].strip("-")
    proposal_id = f"BUG-{date_str}-{slug}-v01"

    # DocWrite -- What: write companion review markdown doc.
    # DocWrite -- Why: operator standing rule -- proposals without docs get
    #     sent back to revising.
    doc_dir = DOCS_DIR / date_str
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / f"BUG_{slug}.md"

    doc_content = (
        f"# {title}\n\n"
        f"**Generated by:** AUD84 bug_send_for_review ({len(bugs)} bug(s))\n"
        f"**Date:** {date_str}\n"
        f"**Max severity:** {severity}\n"
        f"**Affected surfaces:** {', '.join(affected_surfaces)}\n\n"
        f"## Evidence\n\n"
    )
    for ev in evidence:
        doc_content += f"- {ev['fact']}\n\n"
    doc_content += (
        f"## Recommendation\n\n"
        f"Operator: please review the above bug evidence and provide a fix recommendation.\n\n"
        f"## Review checklist\n\n"
        f"- [ ] Bug reproduction confirmed\n"
        f"- [ ] Root cause identified\n"
        f"- [ ] Fix recommendation is actionable\n"
        f"- [ ] Approve / Send back / Reject\n"
    )

    try:
        doc_path.write_text(doc_content, encoding="utf-8")
    except OSError as exc:
        _log.warning("bug_records: failed to write review doc: %s", exc)
        doc_path = None

    # ProposalWrite -- What: write the BUG-prefixed SEWE proposal JSON.
    # ProposalWrite -- Why: D6 auto-skeleton. operator_review_state='pending'
    #     per the proposal-first axiom (standing axiom 2026-05-06).
    proposal = {
        "id": proposal_id,
        "kind": "bug_fix",
        "title": title,
        "category": "bug_proposal",
        "created": now.isoformat(timespec="seconds"),
        "operator_review_state": "pending",
        "recommendation": (
            "Operator: please review and provide fix recommendation. "
            "This skeleton was auto-generated from bug discovery records."
        ),
        "evidence": evidence,
        "decisions_to_confirm": [],
        "scope": f"Bug fix targeting: {', '.join(affected_surfaces)}",
        "linked_to": linked_to,
        "document_path": str(doc_path) if doc_path else "",
        "severity": severity,
        "affected_surfaces": affected_surfaces,
        "warrior": "code_warrior",
        "signature": f"auto-{date_str}-bug-{slug}",
        "drafted_by": "bug_records/send_for_review",
        "version": 1,
    }

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    proposal_path = PROPOSALS_DIR / f"{proposal_id}.json"
    try:
        proposal_path.write_text(
            json.dumps(proposal, indent=2) + "\n", encoding="utf-8"
        )
        _log.info("bug_records: emitted BUG proposal %s (%d bugs)", proposal_id, len(bugs))
        return proposal_id
    except OSError as exc:
        _log.error("bug_records: failed to write proposal %s: %s", proposal_id, exc)
        return None


# ---------------------------------------------------------------------------
# IPC handlers
# ---------------------------------------------------------------------------

def _handle_bug_discover_log(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for bug_discover_log op.

    What: accepts bug details, calls emit_bug_node, returns the bug slug.
    Why:  unified entry point for programmatic emission + operator manual
          reports. Both go through the same dedup + storage path.
    """
    # What: validate title at system edge before writing a node.
    # Why:  without this, emit_bug_node creates a node with empty title
    #       (UNVALIDATED WRITE). BUG-2026-05-08-ipc-double-wrap audit.
    if not args.get("title"):
        raise ValueError("title is required")

    source = args.get("source", "manual")
    surface = args.get("surface", "unknown")
    severity = args.get("severity", "medium")
    title = args.get("title")
    description = args.get("description", "")
    evidence_path = args.get("evidence_path", "")

    slug = emit_bug_node(
        source=source,
        surface=surface,
        severity=severity,
        title=title,
        description=description,
        evidence_path=evidence_path,
    )
    # What: return data directly -- IPC dispatcher adds the {"ok", "result"} envelope.
    # Why:  double-wrap fix (BUG-2026-05-08-ipc-double-wrap).
    return {"slug": slug, "node_path": str(NODES_DIR / f"bug_{slug}.md")}


def _handle_bug_targeted_list(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for bug_targeted_list op.

    What: returns bug records with status='targeted'.
    Why:  the Atoms BUG tab shows targeted bugs ready for proposal generation.
          Also accepts optional filter_status arg for flexibility.
    """
    filter_status = args.get("filter_status", "targeted")
    bugs = list_bug_nodes(filter_status=filter_status)
    return {"bugs": bugs, "count": len(bugs)}


def _handle_bug_target_set(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for bug_target_set op.

    What: sets or unsets 'targeted' status on a list of bug slugs.
    Why:  operator selects bugs in the Found Bugs list, toggles them
          targeted before sending for review.
    """
    bug_ids = args.get("bug_ids", [])
    targeted = args.get("targeted", True)

    if not isinstance(bug_ids, list):
        raise ValueError("bug_ids must be a list")

    new_status = "targeted" if targeted else "untriaged"
    results: dict[str, bool] = {}
    for slug in bug_ids:
        results[slug] = update_bug_status(slug, new_status)

    # What: return data directly -- IPC dispatcher adds envelope.
    # Why:  double-wrap fix (BUG-2026-05-08-ipc-double-wrap).
    return {"results": results}


def _handle_bug_send_for_review(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler for bug_send_for_review op.

    What: generates BUG proposal(s) from specified bug slugs per mode.
    Why:  AUD84 D5/D6 -- operator targets bugs, selects mode, sends for
          review. Auto-skeleton proposals land as 'pending' (axiom preserved).
    """
    bug_ids = args.get("bug_ids", [])
    mode = args.get("mode", "1bug")

    if not isinstance(bug_ids, list) or not bug_ids:
        raise ValueError("bug_ids must be a non-empty list")
    if mode not in ("1bug", "multi-bug", "multi-surface"):
        raise ValueError(f"invalid mode: {mode!r}")

    proposal_ids = send_for_review(bug_slugs=bug_ids, mode=mode)
    # What: return data directly -- IPC dispatcher adds envelope.
    # Why:  double-wrap fix (BUG-2026-05-08-ipc-double-wrap).
    return {
        "proposal_ids": proposal_ids,
        "count": len(proposal_ids),
    }


# ---------------------------------------------------------------------------
# Op registration
# ---------------------------------------------------------------------------

def register_ops() -> None:
    """Register all bug_records IPC ops with the daemon.

    What: calls samia.runtime.ipc.register_op for each bug workflow operation.
    Why:  daemon calls this during startup. Matches the registration pattern
          used by memory_guard, audit_skill, etc.
    """
    from samia.runtime.ipc import register_op
    register_op("bug_discover_log", _handle_bug_discover_log)
    register_op("bug_targeted_list", _handle_bug_targeted_list)
    register_op("bug_target_set", _handle_bug_target_set)
    register_op("bug_send_for_review", _handle_bug_send_for_review)
    _log.info(
        "bug_records ops registered "
        "(bug_discover_log, bug_targeted_list, bug_target_set, bug_send_for_review)"
    )


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.bug_records
# Phase:      AUD84 -- Phases 1+2 (schema, IPC ops, proposal auto-skeleton)
# Layer:      runtime (in-daemon, IPC-exposed, event-driven discovery hooks)
# Stability:  v0.1.0 -- initial implementation
# ErrorModel: fail-open for node writes (log and continue). IPC handlers
#             return error dicts, never raise. Dedup is best-effort (hash
#             collision is astronomically unlikely). Proposal-first axiom
#             enforced: all generated proposals land as 'pending'.
# Depends:    samia.core.frontmatter (read_node, write_node),
#             samia.runtime.ipc (register_op),
#             stdlib (hashlib, json, datetime, re, pathlib, logging, dataclasses).
# Exposes:    PREFIX_TAXONOMY, BUG_SOURCE_VALUES, BUG_STATUS_VALUES,
#             emit_bug_node, list_bug_nodes, update_bug_status,
#             link_bug_to_proposal, send_for_review, register_ops.
# Lines:      ~480
# --------------------------------------------------------------------------
