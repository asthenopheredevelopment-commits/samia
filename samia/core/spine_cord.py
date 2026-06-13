"""samia.core.spine_cord -- Verified-outcome capture: cord schema + per-zone adapters.

What: Defines the OutcomeRecord (the cord) -- the canonical schema for all
      verified outcome records across the Asthenosphere. Provides a hierarchical
      intent taxonomy with agnostic parents and specialized children, per-zone
      emit adapters (sovereign, samia, harness), and a flat 'outcome' field
      derivation that preserves backward compatibility with opencode_drain.

Why:  Before this module, each emitter (bounty_workflow, future daemon sources,
      future harness sources) would independently construct ad-hoc record dicts
      with inconsistent field sets. The cord schema normalizes all outcomes into
      a single structure that the reasoning spine, phrasing transducer, and storm
      learner can consume uniformly. The flat 'outcome' field is CRITICAL: the
      drain (opencode_drain.py:136) identifies outcome records by checking for
      this key's presence and derives target_state/material_grade from its value.

Architecture:
  - OutcomeRecord: dataclass holding intent, source_kind, operational,
    domain_verdict, vertebra_id (future), specialized_payload (future),
    and carried context fields.
  - Intent taxonomy: agnostic parents (fix, feature, refactor, test,
    documentation, optimization, general) + specialized children that
    roll up via agnostic_parent().
  - Per-zone adapters: emit_sovereign (3-sink: outbox + JSONL + rw log),
    emit_samia (spine node + rw log), emit_harness (outbox only).
  - Flat 'outcome' derivation: domain_verdict.status when present,
    operational.status mapped to failure otherwise.

Public API:
  build_outcome_record(...)     -> OutcomeRecord
  serialize_record(rec)         -> dict   (JSON-serializable, flat 'outcome')
  parse_record(data)            -> OutcomeRecord
  agnostic_parent(intent)       -> str
  emit_outcome(rec, zone, ...) -> None   (dispatches to per-zone adapter)
  emit_sovereign(rec, ...)      -> None
  emit_samia(rec, ...)          -> None   [Phase 2+]
  emit_harness(rec, ...)        -> None   [Phase 2+]
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger('samia.core.spine_cord')


# ---------------------------------------------------------------------------
# Intent Taxonomy
# ---------------------------------------------------------------------------

# What: Agnostic (top-level) intent categories -- the parents
# Why: Every outcome maps to one of these; specialized children roll up
AGNOSTIC_INTENTS = frozenset({
    'fix', 'feature', 'refactor', 'test',
    'documentation', 'optimization', 'general',
})

# What: Specialized intent -> agnostic parent mapping
# Why: Domain-specific intents (e.g. compile.rust_borrowck) roll up to their
#      agnostic parent for cross-domain aggregation and transducer keying.
#      Seed with representative examples; grows as new sources are wired.
_SPECIALIZED_CHILDREN: dict[str, str] = {
    # fix children
    'compile': 'fix',
    'compile.rust_borrowck': 'fix',
    'compile.type_mismatch': 'fix',
    'compile.ubeloer': 'fix',
    'compile.linker': 'fix',
    'compile.codegen': 'fix',
    'runtime_crash': 'fix',
    'regression': 'fix',
    'security_patch': 'fix',
    'bug_fix': 'fix',
    'bug_fix.miscompile': 'fix',
    'bug_fix.regalloc': 'fix',
    'bug_fix.segfault': 'fix',
    # feature children
    'api_endpoint': 'feature',
    'ui_component': 'feature',
    'integration': 'feature',
    'boilerplate': 'feature',
    'boilerplate.generation': 'feature',
    'dispatch': 'feature',
    'dispatch.source_wiring': 'feature',
    'dispatch.adapter': 'feature',
    # refactor children
    'extract_module': 'refactor',
    'rename': 'refactor',
    'dead_code_removal': 'refactor',
    # test children
    'unit_test': 'test',
    'integration_test': 'test',
    'benchmark': 'test',
    'experiment': 'test',
    'experiment.ablation': 'test',
    'experiment.parameter_sweep': 'test',
    'experiment.hypothesis': 'test',
    # optimization children
    'perf_hotpath': 'optimization',
    'memory_reduction': 'optimization',
    # documentation children
    'api_docs': 'documentation',
    'inline_comments': 'documentation',
    # general children -- cross-cutting intents that don't fit a single parent
    'consensus': 'general',
    'consensus.mesh': 'general',
    'consensus.tscp': 'general',
    'consensus.dispute': 'general',
    'reflection': 'general',
    'reflection.self_review': 'general',
    'reflection.precompact': 'general',
    'route': 'general',
    'route.chiron': 'general',
    'route.warrior_handoff': 'general',
}


def agnostic_parent(intent: str) -> str:
    """Resolve any intent (agnostic or specialized) to its agnostic parent.

    What: Returns the agnostic category that this intent belongs to.
    Why: Transducer and storm learner aggregate at the agnostic level;
         specialized intents must roll up for cross-domain comparisons.

    Rules:
      1. If intent is itself agnostic, return it unchanged.
      2. If intent is a known specialized child, return its parent.
      3. If intent contains a dot, try the prefix before the first dot.
      4. Otherwise, return 'general' (safe fallback).
    """
    if intent in AGNOSTIC_INTENTS:
        return intent
    if intent in _SPECIALIZED_CHILDREN:
        return _SPECIALIZED_CHILDREN[intent]
    # What: Try dotted prefix for hierarchical intents (e.g. compile.rust_borrowck)
    # Why: New specialized children may not be registered yet; dotted convention
    #      lets them roll up via their prefix
    if '.' in intent:
        prefix = intent.split('.', 1)[0]
        if prefix in _SPECIALIZED_CHILDREN:
            return _SPECIALIZED_CHILDREN[prefix]
        if prefix in AGNOSTIC_INTENTS:
            return prefix
    return 'general'


# ---------------------------------------------------------------------------
# OutcomeRecord Schema (the cord)
# ---------------------------------------------------------------------------

@dataclass
class OutcomeRecord:
    """The cord: canonical schema for a verified outcome record.

    What: Holds all fields required to represent a single verified outcome
          from any zone (sovereign, samia-runtime, harness).
    Why: Normalizes the ad-hoc record dicts that each emitter previously
         built independently. A single schema ensures the drain, transducer,
         and storm learner see consistent field sets.

    Fields:
      intent          -- task intent (ALWAYS present; maps to taxonomy)
      source_kind     -- emitter identity (e.g. 'bounty_verification_gate')
      operational     -- {status: 'ok'|'failed', reason?: str}
      domain_verdict  -- {status: 'success'|'partial'|'failure', score?: num,
                          reason?: list} or None (when intent has no
                          success metric, e.g. a pure action)
      vertebra_id     -- future: position in the reasoning spine (None in P1)
      specialized_payload -- future: per-source extra data (None in P1)
      ts              -- ISO timestamp
      task            -- task identifier (e.g. 'bounty:owner/repo#123')
      attempt         -- attempt number (optional)
      max_attempts    -- max iterations allowed (optional)
      artifact_ref    -- reference back to source artifact (optional)
      primary_model   -- model that produced the solution (optional)
      variant_id      -- phrasing variant used (optional)
      extra           -- overflow dict for source-specific carried context
    """
    intent: str
    source_kind: str
    operational: dict[str, Any]
    domain_verdict: Optional[dict[str, Any]] = None
    vertebra_id: Optional[str] = None
    specialized_payload: Optional[dict[str, Any]] = None
    ts: str = ''
    task: str = ''
    attempt: Optional[int] = None
    max_attempts: Optional[int] = None
    artifact_ref: Optional[str] = None
    primary_model: Optional[str] = None
    variant_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


def _derive_flat_outcome(rec: OutcomeRecord) -> str:
    """Derive the flat 'outcome' value from domain_verdict or operational.

    What: Returns 'success', 'partial', or 'failure' -- the drain-compatible
          flat outcome string.
    Why: opencode_drain.py:136 identifies outcome records by the presence of
         a flat 'outcome' key. opencode_drain.py:167-176 derives target_state
         and material_grade from this value. If this field is missing or nested,
         the drain rejects ALL records and the writeback pipeline breaks.

    Rules:
      - If domain_verdict exists and has a 'status' field, use that.
      - If domain_verdict is None (no success metric for this intent),
        map operational.status: 'ok' -> 'success', anything else -> 'failure'.
      - Fallback: 'failure' (fail-safe for unknown values).
    """
    if rec.domain_verdict is not None:
        status = rec.domain_verdict.get('status', 'failure')
        if status in ('success', 'partial', 'failure'):
            return status
        return 'failure'
    # What: No domain verdict -- derive from operational status
    # Why: Intents without a success metric (pure actions) still need a flat
    #      outcome for the drain; operational.status is the only signal
    op_status = rec.operational.get('status', 'failed')
    return 'success' if op_status == 'ok' else 'failure'


def build_outcome_record(
    intent: str,
    source_kind: str,
    operational: dict[str, Any],
    domain_verdict: Optional[dict[str, Any]] = None,
    *,
    ts: Optional[str] = None,
    task: str = '',
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    artifact_ref: Optional[str] = None,
    primary_model: Optional[str] = None,
    variant_id: Optional[str] = None,
    vertebra_id: Optional[str] = None,
    specialized_payload: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> OutcomeRecord:
    """Construct an OutcomeRecord with defaults and validation.

    What: Factory function that builds a validated OutcomeRecord.
    Why: Centralizes validation (operational.status required, intent must
         resolve to an agnostic parent) so individual emitters cannot produce
         malformed records.
    """
    if 'status' not in operational:
        raise ValueError("operational dict must contain 'status' key")
    if operational['status'] not in ('ok', 'failed'):
        raise ValueError(
            f"operational.status must be 'ok' or 'failed', "
            f"got {operational['status']!r}"
        )
    if domain_verdict is not None:
        if 'status' not in domain_verdict:
            raise ValueError("domain_verdict dict must contain 'status' key")
        if domain_verdict['status'] not in ('success', 'partial', 'failure'):
            raise ValueError(
                f"domain_verdict.status must be success|partial|failure, "
                f"got {domain_verdict['status']!r}"
            )

    return OutcomeRecord(
        intent=intent,
        source_kind=source_kind,
        operational=operational,
        domain_verdict=domain_verdict,
        vertebra_id=vertebra_id,
        specialized_payload=specialized_payload,
        ts=ts or datetime.now().isoformat(),
        task=task,
        attempt=attempt,
        max_attempts=max_attempts,
        artifact_ref=artifact_ref,
        primary_model=primary_model,
        variant_id=variant_id,
        extra=extra or {},
    )


def serialize_record(rec: OutcomeRecord) -> dict:
    """Serialize an OutcomeRecord to a flat JSON-compatible dict.

    What: Converts the dataclass to a dict suitable for JSON serialization
          and emission to all three sinks.
    Why: The critical invariant is that the returned dict contains a TOP-LEVEL
         'outcome' key (string: success|partial|failure). The drain reads this
         key at opencode_drain.py:136 to identify outcome records and at
         lines 167-176 to derive target_state/material_grade. Nesting outcome
         under domain_verdict would break the drain.

    Layout:
      - All OutcomeRecord fields appear at the top level.
      - 'outcome' is added as a flat derived field (NOT nested).
      - 'operational' and 'domain_verdict' appear as nested dicts (additive).
      - None-valued optional fields are omitted for cleanliness.
      - 'extra' dict is merged into the top level (carried context).
    """
    d: dict[str, Any] = {}

    # What: Flat 'outcome' FIRST -- the drain-critical field
    # Why: Must be at top level; opencode_drain.py:136 checks 'outcome' in content
    d['outcome'] = _derive_flat_outcome(rec)

    # What: Core identity fields
    d['intent'] = rec.intent
    d['source_kind'] = rec.source_kind

    # What: Operational and domain verdict as nested dicts (additive metadata)
    # Why: New consumers can read structured data; old consumers ignore these
    d['operational'] = rec.operational
    if rec.domain_verdict is not None:
        d['domain_verdict'] = rec.domain_verdict

    # What: Carried context fields
    d['ts'] = rec.ts
    if rec.task:
        d['task'] = rec.task
    if rec.attempt is not None:
        d['attempt'] = rec.attempt
    if rec.max_attempts is not None:
        d['max_attempts'] = rec.max_attempts
    if rec.artifact_ref is not None:
        d['artifact_ref'] = rec.artifact_ref
    if rec.primary_model is not None:
        d['primary_model'] = rec.primary_model
    if rec.variant_id is not None:
        d['variant_id'] = rec.variant_id

    # What: Future spine fields (seams, present but unused in Phase 1)
    if rec.vertebra_id is not None:
        d['vertebra_id'] = rec.vertebra_id
    if rec.specialized_payload is not None:
        d['specialized_payload'] = rec.specialized_payload

    # What: Merge extra (carried context) into top level
    # Why: Source-specific fields (issue, repo, score, reason, etc.) that the
    #      drain and runtime_warrior read directly must be flat
    if rec.extra:
        d.update(rec.extra)

    return d


def parse_record(data: dict) -> OutcomeRecord:
    """Deserialize a dict (from JSON) back into an OutcomeRecord.

    What: Inverse of serialize_record; reconstructs an OutcomeRecord from a
          serialized dict.
    Why: Consumers that read outcome records from logs or the abyss can
         reconstruct the typed schema for structured access.
    """
    # What: Extract known fields; remainder goes to extra
    known_keys = {
        'outcome', 'intent', 'source_kind', 'operational', 'domain_verdict',
        'ts', 'task', 'attempt', 'max_attempts', 'artifact_ref',
        'primary_model', 'variant_id', 'vertebra_id', 'specialized_payload',
    }
    extra = {k: v for k, v in data.items() if k not in known_keys}

    return OutcomeRecord(
        intent=data.get('intent', 'general'),
        source_kind=data.get('source_kind', 'unknown'),
        operational=data.get('operational', {'status': 'failed'}),
        domain_verdict=data.get('domain_verdict'),
        vertebra_id=data.get('vertebra_id'),
        specialized_payload=data.get('specialized_payload'),
        ts=data.get('ts', ''),
        task=data.get('task', ''),
        attempt=data.get('attempt'),
        max_attempts=data.get('max_attempts'),
        artifact_ref=data.get('artifact_ref'),
        primary_model=data.get('primary_model'),
        variant_id=data.get('variant_id'),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Per-Zone Emit Adapters
# ---------------------------------------------------------------------------

def emit_sovereign(
    rec: OutcomeRecord,
    *,
    outcome_log: Path,
    sam_outbox: Path,
    samia_tools_path: Optional[str] = None,
) -> None:
    """Emit an OutcomeRecord through the sovereign zone's 3 sinks.

    What: Extracted from bounty_workflow._emit_outcome_record(). Writes the
          serialized record to:
          1. Sovereign JSONL log (outcome_log)
          2. SAM outbox trace envelope (sam_outbox/*.json)
          3. Runtime warrior log (via samia.core.runtime_warrior)
    Why: Each sink is independent; failure in one does not block the others.
         The sovereign zone NEVER writes directly to the spine -- it writes
         to the outbox, and the drain materializes spine nodes.

    Parameters:
      rec              -- the OutcomeRecord to emit
      outcome_log      -- path to sovereign JSONL log (e.g. MONITOR/bounty_outcomes.jsonl)
      sam_outbox       -- path to SAM outbox directory (e.g. SAM_OUTBOX)
      samia_tools_path -- path to samia tools dir for runtime_warrior import;
                          defaults to this module's grandparent (memory/tools)
    """
    record = serialize_record(rec)

    # -- Sink 1: sovereign-internal JSONL log --------------------------------
    try:
        with open(str(outcome_log), 'a') as f:
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
    except Exception as exc:
        _log.warning("spine_cord.emit_sovereign: sovereign log failed: %s", exc)

    # -- Sink 2: SAM outbox trace envelope -----------------------------------
    try:
        is_failure = record.get('outcome') in ('failure', 'partial')
        trace_envelope = {
            'content': json.dumps(record),
            'author_token': 'bounty_outcome_writeback',
            'provenance_type': 'verified_outcome',
            'persona_key': 'opencode',
            'tier': 'long_term' if is_failure else 'short_term',
            '_node_meta': {
                'name': (f"bounty_outcome_"
                         f"{record.get('issue', 'unknown')}_"
                         f"{record.get('attempt', 0)}"),
                'type': 'reference',
                'chains': ['bounty_outcomes', 'verified_outcomes'],
                'target_state': 'frozen' if is_failure else 'live',
                'material_grade': 'enriched' if is_failure else 'natural',
                'runtime': 'opencode',
            },
        }
        ts_safe = record.get('ts', datetime.now().isoformat()).replace(':', '-')
        trace_filename = (f"outcome_{record.get('issue', 'x')}_"
                          f"{record.get('attempt', 0)}_{ts_safe}.json")
        trace_path = sam_outbox / trace_filename
        with open(str(trace_path), 'w') as f:
            json.dump(trace_envelope, f, separators=(',', ':'))
    except Exception as exc:
        _log.warning("spine_cord.emit_sovereign: SAM outbox failed: %s", exc)

    # -- Sink 3: runtime warrior log -----------------------------------------
    try:
        import sys
        _tools = samia_tools_path or str(
            Path(__file__).resolve().parent.parent.parent
        )
        if _tools not in sys.path:
            sys.path.insert(0, _tools)
        from samia.core.runtime_warrior import (
            log_runtime_outcome, compute_pattern_signature,
        )

        rw_outcome = record.get('outcome', 'failure')
        if rw_outcome not in ('success', 'partial', 'failure'):
            rw_outcome = 'failure'

        pattern_sig = compute_pattern_signature(
            task_description=(f"{rec.source_kind}:"
                              f"{record.get('repo', '')}:"
                              f"{record.get('issue', '')}"),
            capabilities=['code_generation', 'quality_gate'],
        )
        invocation_id = (f"{rec.source_kind}_"
                         f"{record.get('repo', '').replace('/', '_')}_"
                         f"{record.get('issue', '')}_"
                         f"{record.get('attempt', 0)}")

        log_runtime_outcome(
            invocation_id=invocation_id,
            outcome=rw_outcome,
            pattern_signature=pattern_sig,
            details={
                'score': record.get('score', 0),
                'issue': record.get('issue', ''),
                'issue_title': record.get('issue_title', ''),
                'repo': record.get('repo', ''),
                'attempt': record.get('attempt', 0),
                'max_attempts': record.get('max_attempts', 0),
                'issues_count': len(record.get('issues', [])),
            },
        )
    except Exception as exc:
        _log.warning("spine_cord.emit_sovereign: runtime warrior failed: %s", exc)


_SAFE_FN_RE = re.compile(r'[^a-zA-Z0-9_-]')


def _safe_filename(intent: str, source_kind: str,
                   disambiguator: Optional[str] = None) -> str:
    """Build a stable, filesystem-safe node filename.

    What: Returns a deterministic .md filename for spine node dedup. When
          `disambiguator` is given (e.g. a task / issue / invocation id), it is
          folded in via a short stable hash so DISTINCT outcomes that share
          intent + source_kind get DISTINCT files, while the SAME outcome
          re-emitted stays idempotent (the key is stable, not time-based).
    Why: Without the disambiguator, two different outcomes of the same
         (intent, source_kind) — e.g. bug_fix from progress_ledger for q92 vs
         q93 — silently overwrite each other, losing all but the latest. The
         hash keeps collisions away even when readable slugs are truncated.
    """
    safe_intent = _SAFE_FN_RE.sub('_', str(intent))[:40]
    safe_source = _SAFE_FN_RE.sub('_', str(source_kind))[:40]
    if disambiguator:
        h = hashlib.sha256(str(disambiguator).encode('utf-8')).hexdigest()[:10]
        safe_dis = _SAFE_FN_RE.sub('_', str(disambiguator))[:24]
        return f'outcome_{safe_intent}_{safe_source}_{safe_dis}_{h}.md'
    return f'outcome_{safe_intent}_{safe_source}.md'


def emit_samia(
    rec: OutcomeRecord,
    *,
    memory_dir: Path,
) -> None:
    """Emit an OutcomeRecord via the samia-runtime zone (direct spine write).

    What: Builds a decay-aware frontmatter spine node from the OutcomeRecord
          and writes it directly to the spine via frontmatter.write_node, then
          logs the outcome to runtime_warrior. This is the in-daemon direct
          path -- allowed because callers are inside the samia-runtime zone.
    Why: In-daemon sources (chiron, progress_ledger, bug_records) operate
         inside the samia write-zone and do not need outbox indirection.
         Direct spine writes give them immediate queryability via memory_search
         and decay governance via tier.py. Failures get target_state=frozen +
         material_grade=enriched so they resist decay; successes get live/natural.

    Parameters:
      rec        -- the OutcomeRecord to materialize as a spine node
      memory_dir -- root memory directory (nodes/ subdirectory receives the node)
    """
    from samia.core import frontmatter as fmlib

    content = serialize_record(rec)
    outcome = content.get('outcome', 'unknown')
    is_failure = outcome in ('failure', 'partial')

    # What: Derive lifecycle fields from outcome
    # Why: Mirrors opencode_drain.py:175-176 -- failures/partials get frozen +
    #      enriched to survive decay; successes get live + natural
    target_state = 'frozen' if is_failure else 'live'
    material_grade = 'enriched' if is_failure else 'natural'

    # What: Build the frontmatter dict for the spine node
    # Why: Mirrors opencode_drain.materialize_node (lines 178-189) adapted for
    #      samia-runtime provenance: name includes intent + source_kind instead
    #      of issue + attempt, chains include intent for cross-domain aggregation
    # What: Derive a queryable 'name' from the first available identifying field.
    # Why: The old code used content.get('issue', 'unknown'), which always
    #      returned 'unknown' for non-issue sources (chiron, progress_ledger).
    #      Trying issue -> task -> invocation_id gives a meaningful name.
    _name_id = (content.get('issue')
                or content.get('task')
                or content.get('invocation_id')
                or 'unknown')
    node_fm = {
        'name': (f'outcome_{rec.intent}_{rec.source_kind}_'
                 f'{_name_id}'),
        'description': f'Verified outcome -- {rec.intent} from {rec.source_kind}',
        'type': 'reference',
        'chains': ['verified_outcomes', rec.intent],
        'target_state': target_state,
        'material_grade': material_grade,
        'runtime': 'opencode',
        'tier': 'warm',
        'last_access': date.today().isoformat(),
        'relevance': 0.75 if is_failure else 0.55,
    }

    order = [
        'name', 'description', 'type', 'chains', 'target_state',
        'material_grade', 'runtime', 'tier', 'last_access', 'relevance',
    ]

    # What: Build the node body from outcome fields
    # Why: Body text is indexed by memory_search; reason/diagnosis must be
    #      in the body so they are queryable by future sessions
    body_lines = [
        f'Outcome: {outcome}',
        f'Intent: {rec.intent}',
        f'Source: {rec.source_kind}',
        '',
    ]
    if content.get('reason'):
        body_lines.append(f'**Reason:** {content["reason"]}')
        body_lines.append('')
    if content.get('diagnosis'):
        body_lines.append(f'**Diagnosis:** {content["diagnosis"]}')
        body_lines.append('')
    if content.get('ts'):
        body_lines.append(f'Timestamp: {content["ts"]}')
        body_lines.append('')
    body = '\n'.join(body_lines)

    # What: Write spine node via frontmatter (direct write, samia zone)
    # Why: override_frozen=True allows idempotent re-writes of failure nodes
    #      (same pattern as opencode_drain.materialize_node line 245)
    nodes_dir = memory_dir / 'nodes'
    nodes_dir.mkdir(parents=True, exist_ok=True)
    # What: Build a COMBINED disambiguator from ALL identifying fields.
    # Why: The previous or-chain (task or issue or invocation_id) short-circuited
    #      on the first truthy field. Chiron passes task=task_category (COARSE,
    #      e.g. 'code_generation'), so 'task' won and the unique invocation_id
    #      was NEVER used -- all same-category routes collapsed into ONE node.
    #      Joining ALL fields means any single field differing produces a distinct
    #      key, while identical records stay idempotent (same joined string).
    _disambig = '|'.join(
        str(content.get(k, ''))
        for k in ('invocation_id', 'task', 'issue', 'attempt')
    ) or 'unknown'
    node_path = nodes_dir / _safe_filename(rec.intent, rec.source_kind, _disambig)
    fmlib.write_node(
        path=node_path, fm=node_fm, order=order, body=body,
        override_frozen=True,
    )

    # What: Log to runtime warrior for pattern tracking
    # Why: Same sink 3 pattern as emit_sovereign (lines 424-466); runtime warrior
    #      aggregates outcomes for promotion proposals
    try:
        import sys
        _tools = str(Path(__file__).resolve().parent.parent.parent)
        if _tools not in sys.path:
            sys.path.insert(0, _tools)
        from samia.core.runtime_warrior import (
            log_runtime_outcome, compute_pattern_signature,
        )

        rw_outcome = outcome if outcome in ('success', 'partial', 'failure') else 'failure'
        pattern_sig = compute_pattern_signature(
            task_description=f'{rec.source_kind}:{rec.intent}',
            capabilities=['verified_outcome_capture'],
        )
        ts_hash = str(hash(rec.ts))[:8] if rec.ts else 'nots'
        invocation_id = f'{rec.source_kind}_{rec.intent}_{ts_hash}'

        log_runtime_outcome(
            invocation_id=invocation_id,
            outcome=rw_outcome,
            pattern_signature=pattern_sig,
            details={**content},
        )
    except Exception as exc:
        _log.warning("spine_cord.emit_samia: runtime warrior failed: %s", exc)


def emit_harness(
    rec: OutcomeRecord,
    *,
    outbox_dir: Path,
) -> None:
    """Emit an OutcomeRecord via the harness zone (outbox trace envelope only).

    What: Writes a trace envelope to outbox_dir for sam_manager to relay through
          the external abyss, where the drain materializes it as a spine node.
          Does NOT call frontmatter.write_node or write the spine directly.
    Why: The harness zone (opencode_harness/loer_compiler) is OUTSIDE the samia
         write-zone. Containment boundary: harness writes ONLY to its outbox;
         sam_manager (sovereign) reads both SAM_OUTBOX and harness_outbox, relays
         to the external abyss; the drain reads ONLY from the abyss and
         materializes spine nodes. This keeps the single-writer invariant.

    Parameters:
      rec        -- the OutcomeRecord to emit
      outbox_dir -- path to the harness outbox directory
                    (e.g. opencode_harness/tmp/harness_outbox)
    """
    record = serialize_record(rec)
    outcome = record.get('outcome', 'unknown')
    is_failure = outcome in ('failure', 'partial')

    # What: Build a trace envelope matching emit_sovereign's shape (lines 398-413)
    # Why: sam_manager.process_outbox reads these envelopes and writes them to the
    #      external abyss via sam.write(). The drain then reads the abyss and
    #      materializes spine nodes. The envelope shape must match what sam_manager
    #      expects: content (JSON string), author_token, provenance_type, persona_key,
    #      tier, _node_meta.
    trace_envelope = {
        'content': json.dumps(record),
        'author_token': 'harness_outcome_writeback',
        'provenance_type': 'verified_outcome',
        'persona_key': 'opencode',
        'tier': 'long_term' if is_failure else 'short_term',
        '_node_meta': {
            'name': (f'outcome_{rec.intent}_{rec.source_kind}_'
                     f'{record.get("issue", "unknown")}'),
            'type': 'reference',
            'chains': ['harness_outcomes', 'verified_outcomes'],
            'target_state': 'frozen' if is_failure else 'live',
            'material_grade': 'enriched' if is_failure else 'natural',
            'runtime': 'opencode',
        },
    }

    # What: Write the trace envelope to the outbox directory
    # Why: Deterministic filename allows idempotent re-writes; sam_manager
    #      scans *.json in the outbox directory
    outbox_dir.mkdir(parents=True, exist_ok=True)
    ts_safe = record.get('ts', datetime.now().isoformat()).replace(':', '-')
    safe_intent = _SAFE_FN_RE.sub('_', str(rec.intent))[:30]
    safe_source = _SAFE_FN_RE.sub('_', str(rec.source_kind))[:30]
    trace_filename = f'outcome_{safe_intent}_{safe_source}_{ts_safe}.json'
    trace_path = outbox_dir / trace_filename

    try:
        with open(str(trace_path), 'w') as f:
            json.dump(trace_envelope, f, separators=(',', ':'))
    except Exception as exc:
        _log.warning("spine_cord.emit_harness: outbox write failed: %s", exc)


def emit_outcome(
    rec: OutcomeRecord,
    zone: str,
    **kwargs: Any,
) -> None:
    """Dispatch an outcome record to the appropriate per-zone adapter.

    What: Single entry point that routes to emit_sovereign, emit_samia, or
          emit_harness based on the zone parameter.
    Why: Callers that do not need to know which zone they are in can use this
         dispatcher; zone is determined by the calling context.
    """
    if zone == 'sovereign':
        emit_sovereign(rec, **kwargs)
    elif zone == 'samia':
        emit_samia(rec, **kwargs)
    elif zone == 'harness':
        emit_harness(rec, **kwargs)
    else:
        raise ValueError(f"Unknown zone: {zone!r}. Must be sovereign|samia|harness.")


# ---------------------------------------------------------------------------
# [Asthenosphere] samia.core.spine_cord
# Phase:      FEAT-verified-outcome-capture-architecture-wide (Phase 2 adapters)
# Layer:      core (pure library, no daemon dependency)
# Stability:  v2.0.0
# ErrorModel: build_outcome_record raises ValueError on malformed input;
#             emit_sovereign is fire-and-forget per sink (failure in one does
#             not block others); emit_samia writes spine directly + logs to
#             runtime warrior (fire-and-forget on rw); emit_harness writes a
#             trace envelope to the harness outbox (fire-and-forget).
# Depends:    json, logging, re, dataclasses, datetime, pathlib, typing (stdlib).
#             samia.core.frontmatter (late-import in emit_samia).
#             samia.core.runtime_warrior (late-import in emit_sovereign, emit_samia).
# Exposes:    OutcomeRecord, build_outcome_record, serialize_record,
#             parse_record, agnostic_parent, AGNOSTIC_INTENTS,
#             emit_outcome, emit_sovereign, emit_samia, emit_harness.
# ---------------------------------------------------------------------------
