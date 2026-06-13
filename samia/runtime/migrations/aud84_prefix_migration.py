"""aud84_prefix_migration.py -- AUD84 Phase 3 proposal prefix migration.

One-time retag migration for ~85 existing SEWE proposals. Reclassifies
each proposal from the legacy single-namespace PROP-* id scheme into
one of eight prefix-coded categories (AUD, BUG, FEAT, RSCH, REF, OPS,
DOC, MISC) per the taxonomy defined in AUD84 D1.

Dry-run by default: produces a markdown plan table at
docs/audits/aud84_migration_plan_2026-05-07.md. Only the --apply flag
triggers actual file renames and JSON edits.

Owns: load_proposal_corpus, infer_prefix, generate_new_id, dry_run, apply
Depends on: json (stdlib), pathlib (stdlib), shutil (stdlib), re (stdlib),
            datetime (stdlib), os (stdlib)
"""

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ============================================================================
# Constants -- What: taxonomy prefixes and the safe migration cutoff
# Constants -- Why: AUD84 D1 defines 8 prefixes; D2 cutoff prevents
#   collisions with concurrently-drafted proposals
# ============================================================================

VALID_PREFIXES = ("AUD", "BUG", "FEAT", "RSCH", "REF", "OPS", "DOC", "MISC")

# -- What: proposals created at or after this timestamp are excluded from migration
# -- Why: AUD84 was created at 2026-05-07T03:25; anything at 03:00+ may be
#   concurrently drafted by other HAP agents
MIGRATION_CUTOFF = datetime(2026, 5, 7, 3, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

PROP_DIR_DEFAULT = Path(os.path.expanduser(
    "~/.local/share/asthenos/sewe/proposals"
))

PLAN_OUTPUT_DEFAULT = Path(os.path.expanduser(
    "~/Asthenosphere/docs/audits/aud84_migration_plan_2026-05-07.md"
))

AUDIT_LOG_DEFAULT = Path(os.path.expanduser(
    "~/.local/share/asthenos/audit/aud84_migration.jsonl"
))

# ============================================================================
# Slug extraction helpers
# ============================================================================

# -- What: map AUD numbers to human-readable slugs for proposals with empty titles
# -- Why: early proposals (AUD01-AUD17, VG01, 7c8343, cf0b84) lack title fields;
#   the slug comes from either the filename or the document_path basename
_LEGACY_SLUG_MAP = {
    "PROP-2026-05-01-7c8343": "unroutable-task-sig-birth",
    "PROP-2026-05-01-cf0b84": "kernel-module-warrior-birth-reject",
    "PROP-2026-05-01-VG01": "viberank-graphify-integration",
    "PROP-2026-05-02-AUD01": "containment-systemd-service",
    "PROP-2026-05-02-AUD02": "timer-audit-orphan-services",
    "PROP-2026-05-02-AUD03": "loer-lint-tools",
    "PROP-2026-05-02-AUD04": "ecosystem-completeness-sync",
    "PROP-2026-05-02-AUD05": "backup-includes-retention",
    "PROP-2026-05-02-AUD06": "schema-reconciliation",
    "PROP-2026-05-02-AUD07": "local-inference-dedup-bug",
    "PROP-2026-05-02-AUD08": "atoms-unit-tests",
    "PROP-2026-05-02-AUD09": "changelog-format-sweep",
    "PROP-2026-05-02-AUD10": "proposal-gc-archive-rotation",
    "PROP-2026-05-02-AUD11": "hap-concurrency-experiment",
    "PROP-2026-05-02-AUD12": "cooperative-policy-gating-charter",
    "PROP-2026-05-02-AUD13": "looda-master-refactor-program",
    "PROP-2026-05-02-AUD14-atoms-freetext-submit":
        "atoms-freetext-submit",
    "PROP-2026-05-02-AUD15-bitemporal-fact-validity":
        "bitemporal-fact-validity",
    "PROP-2026-05-02-AUD16-llm-contradiction-detection":
        "llm-contradiction-detection",
    "PROP-2026-05-02-AUD17-sam-frontmatter-target-state":
        "sam-frontmatter-target-state",
    "PROP-2026-05-02-AUD18-atoms-operator-ux":
        "atoms-operator-ux",
    "PROP-2026-05-02-AUD19-jepa-auxiliary-loss-rem":
        "jepa-auxiliary-loss-rem",
    "PROP-2026-05-02-AUD20-soft-chain-retrieval-predictor":
        "soft-chain-retrieval-predictor",
    "PROP-2026-05-02-AUD21-first-bbq-topic-short":
        "first-bbq-topic-short",
    "PROP-2026-05-02-AUD22-sam-ia-vector-layer":
        "sam-ia-vector-layer",
    "PROP-2026-05-02-AUD23-bm25-lexical-index":
        "bm25-lexical-index",
    "PROP-2026-05-02-AUD24-citation-backlink-graph":
        "citation-backlink-graph",
    "PROP-2026-05-02-AUD25-provenance-index":
        "provenance-index",
    "PROP-2026-05-02-AUD26-sam-ia-runtime-daemon":
        "sam-ia-runtime-daemon",
    "PROP-2026-05-02-AUD27-rename-rem-training":
        "rename-rem-training",
    # -- What: disambiguate AUD33 vs AUD37 slugs (both "skills-catalog" on same date)
    "PROP-2026-05-06-AUD33-skills-catalog":
        "skills-catalog-launcher",
    "PROP-2026-05-06-AUD37-skills-catalog-v01":
        "skills-catalog-research-informed",
}


def _slug_from_id(proposal_id: str) -> str:
    """Extract a slug from proposal id or legacy slug map.

    -- What: produces a kebab-case slug suitable for the new filename
    -- Why: new id format is PREFIX-YYYY-MM-DD-slug-vNN; the slug must be
       derived from either the title, the filename stem, or the hardcoded map
    """
    if proposal_id in _LEGACY_SLUG_MAP:
        return _LEGACY_SLUG_MAP[proposal_id]

    # -- What: strip the PROP-YYYY-MM-DD- prefix and -vNN suffix
    # -- Why: the remaining text is already a usable slug
    stripped = re.sub(r"^PROP-\d{4}-\d{2}-\d{2}-", "", proposal_id)
    stripped = re.sub(r"-v\d+$", "", stripped)

    # -- What: remove AUD prefix from slug if present (e.g. AUD28.7-offload-pivot)
    # -- Why: the new prefix replaces AUD; keeping it in slug would be redundant
    stripped = re.sub(r"^AUD\d+\.?\d*-?", "", stripped)

    # -- What: strip trailing version suffix that was already in the original slug
    # -- Why: generate_new_id appends -v01, so an existing -v01 in slug
    #   would cause duplication (e.g. skills-catalog-v01-v01)
    stripped = re.sub(r"-v\d+$", "", stripped)

    return stripped.lower() if stripped else "unknown"


def _extract_date_from_id(proposal_id: str) -> str:
    """Extract the YYYY-MM-DD date portion from a PROP-* id.

    -- What: pulls the original date from the legacy id
    -- Why: migration preserves original dates in filenames per D2
    """
    m = re.match(r"PROP-(\d{4}-\d{2}-\d{2})-", proposal_id)
    if m:
        return m.group(1)
    return "2026-05-01"


# ============================================================================
# Corpus loader
# ============================================================================

def load_proposal_corpus(prop_dir: Path = PROP_DIR_DEFAULT) -> list[dict]:
    """Load all PROP-*.json files from the proposals directory.

    -- What: reads and parses every proposal JSON, filtering to PROP-* prefix
       and pre-cutoff created timestamps
    -- Why: the migration must only touch proposals created before the safe
       cutoff (2026-05-07T03:00 CDT) to avoid colliding with concurrent drafts
    """
    proposals = []

    for fpath in sorted(prop_dir.glob("PROP-*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] skipping unreadable file {fpath}: {exc}",
                  file=sys.stderr)
            continue

        # -- What: parse created timestamp and check against cutoff
        # -- Why: proposals at or after cutoff may be concurrently modified
        created_str = data.get("created", "")
        if created_str:
            try:
                created = datetime.fromisoformat(created_str)
                if created >= MIGRATION_CUTOFF:
                    continue
            except ValueError:
                pass

        data["_source_path"] = str(fpath)
        proposals.append(data)

    return proposals


# ============================================================================
# Prefix inference engine
# ============================================================================

# -- What: compiled regex patterns for TITLE-ONLY keyword matching
# -- Why: D2 inference rules are keyword-driven; scanning title is higher
#   signal than recommendation text which often contains incidental keywords
_BUG_TITLE = re.compile(
    r"\b(fix|bug|fragility|crash|broken|error|dedup)\b",
    re.IGNORECASE,
)
_FEAT_TITLE = re.compile(
    r"\b(novel|add\b|create|panel|tab|skill|feature|pipeline|engine|"
    r"button|toggle|toast|dashboard|cli\s+tab|zoom|pan|highlight|persist|"
    r"detection|vector|lexical|backlink|provenance|daemon|runner|orchestrat|"
    r"vision|interaction|evolution|cascade|HiCache|RadixAttention|catalog|"
    r"soul|defense|guard|foundry|overhaul|parity|expansion|system|"
    r"auxiliary|predictor|integration)\b",
    re.IGNORECASE,
)
_RSCH_TITLE = re.compile(
    r"\b(research|investigat|study|feasibility|spike|experiment|cliff-finding)\b",
    re.IGNORECASE,
)
_REF_TITLE = re.compile(
    r"\b(refactor|cleanup|reorganize|migrat|removal|retire|reconcil|"
    r"normali[sz]|tighten|incremental|rename|disposition|sweep|gc|rotation)\b",
    re.IGNORECASE,
)
_OPS_TITLE = re.compile(
    r"\b(backup|retention|monitor|deploy|perm|systemd|service|"
    r"heartbeat|budget|watchdog|absorb|timer.audit)\b",
    re.IGNORECASE,
)
_DOC_TITLE = re.compile(
    r"\b(script|paper|whitepaper|README|youtube|charter)\b",
    re.IGNORECASE,
)
_AUD_TITLE = re.compile(
    r"\b(audit|efficiency|finding|parity|completeness|lensing)\b",
    re.IGNORECASE,
)

# -- What: manual override map for proposals where automated rules mis-classify
# -- Why: ~15 proposals have kind/title combinations that defeat keyword
#   heuristics; these were identified by reviewing the dry-run output against
#   the operator's mental model (e.g. AUD03 "loer lint tools" is infrastructure
#   tooling = FEAT, not OPS despite "systemd" in recommendation text)
_MANUAL_OVERRIDES: dict[str, tuple[str, float, str]] = {
    "PROP-2026-05-01-VG01": (
        "FEAT", 0.90, "manual: VG01 integrates viberank+graphify into Atoms = new capability"),
    "PROP-2026-05-02-AUD01": (
        "OPS", 0.90, "manual: AUD01 containment systemd service = operations/deployment"),
    "PROP-2026-05-02-AUD02": (
        "OPS", 0.90, "manual: AUD02 timer audit + orphan services = operations"),
    "PROP-2026-05-02-AUD03": (
        "FEAT", 0.85, "manual: AUD03 creates loer_lint/check/audit tools = new tooling capability"),
    "PROP-2026-05-02-AUD04": (
        "AUD", 0.85, "manual: AUD04 ecosystem completeness audit finding -> sync"),
    "PROP-2026-05-02-AUD05": (
        "OPS", 0.90, "manual: AUD05 backup includes + retention tiers = operations"),
    "PROP-2026-05-02-AUD06": (
        "REF", 0.85, "manual: AUD06 schema reconciliation = refactor to match spec"),
    "PROP-2026-05-02-AUD08": (
        "FEAT", 0.80, "manual: AUD08 adds unit test suites = new testing capability"),
    "PROP-2026-05-02-AUD09": (
        "REF", 0.85, "manual: AUD09 changelog format sweep = cleanup refactor"),
    "PROP-2026-05-02-AUD10": (
        "OPS", 0.85, "manual: AUD10 proposal GC + archive rotation = operations"),
    "PROP-2026-05-03-AUD28-legacy-caller-migration": (
        "REF", 0.85, "manual: AUD28 legacy caller migration = refactor/migration"),
    "PROP-2026-05-06-AUD28.7-offload-pivot": (
        "REF", 0.85, "manual: AUD28.7 offload pivot = infrastructure refactor"),
    "PROP-2026-05-06-AUD32-local-orchestrator": (
        "FEAT", 0.90, "manual: AUD32 local orchestrator = novel capability"),
    "PROP-2026-05-06-AUD37-skills-catalog-v01": (
        "FEAT", 0.85, "manual: AUD37 skills catalog v0.1 = feature (research-informed but deliverable is catalog)"),
    "PROP-2026-05-06-AUD39-soul-system-v01": (
        "FEAT", 0.90, "manual: AUD39 soul system = novel capability"),
    "PROP-2026-05-06-AUD50-sglang-hicache-cascade-v01": (
        "FEAT", 0.85, "manual: AUD50 SGLang HiCache = novel caching feature"),
    "PROP-2026-05-06-AUD52-skill-routing-caller-context-v01": (
        "FEAT", 0.85, "manual: AUD52 skill routing + caller context = new capability"),
    "PROP-2026-05-06-AUD55-ask-to-drive-session-v01": (
        "FEAT", 0.90, "manual: AUD55 ask-to-drive session = novel operator-gated capability"),
    "PROP-2026-05-06-AUD57-onetime-script-lifecycle-v01": (
        "OPS", 0.85, "manual: AUD57 one-time script lifecycle management = operations"),
    "PROP-2026-05-06-AUD70-evolution-engine-atoms-panel-v01": (
        "FEAT", 0.85, "manual: AUD70 evolution engine Atoms panel = new panel feature"),
    "PROP-2026-05-06-AUD71-evolution-research-artifact-cleanup-v01": (
        "OPS", 0.85, "manual: AUD71 auto-cleanup for artifacts = operations"),
    "PROP-2026-05-06-AUD76-cf0b84-disposition-v01": (
        "REF", 0.80, "manual: AUD76 cf0b84 disposition = closure/cleanup, not audit"),
}


def infer_prefix(proposal: dict) -> tuple[str, float, str]:
    """Infer the appropriate prefix for a proposal per AUD84 D2 rules.

    Returns (prefix, confidence, rationale).

    -- What: applies the D2 inference cascade to proposal kind, title,
       recommendation, and document_path to determine the best prefix
    -- Why: the cascade is ordered: manual overrides first, then kind-based
       rules, then title keyword matching, with fallback to MISC
    """
    prop_id = proposal.get("id", "")
    kind = (proposal.get("kind") or "").lower().strip()
    title = (proposal.get("title") or "").lower().strip()
    rec = (proposal.get("recommendation") or "").lower().strip()
    doc_path = (proposal.get("document_path") or "").lower().strip()

    # -- What: check manual override map first
    # -- Why: ~15 proposals have edge-case kind/title combos that defeat
    #   keyword heuristics; operator-reviewed overrides take priority
    if prop_id in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[prop_id]

    # -- What: build title-enriched text for keyword scanning
    # -- Why: for proposals with empty titles, the slug from id + doc_path
    #   provides some signal; recommendation text is deprioritized to avoid
    #   incidental keyword matches
    title_text = title or f"{prop_id} {doc_path}"

    # ---- Kind-based rules (highest confidence) ----

    # Rule 1: efficiency_proposal + audit source -> AUD
    if kind == "efficiency_proposal":
        return ("AUD", 0.95, "kind=efficiency_proposal -> AUD per D2")

    # Rule 2: bug_fix_proposal -> BUG
    if kind == "bug_fix_proposal":
        return ("BUG", 0.95, "kind=bug_fix_proposal -> BUG per D2")

    # Rule 3: investigation -> RSCH
    if kind == "investigation":
        return ("RSCH", 0.90, "kind=investigation -> RSCH per D2")

    # Rule 4: ux_proposal -> FEAT
    if kind in ("atoms_ux_proposal", "ux_proposal"):
        return ("FEAT", 0.85, "kind=atoms_ux_proposal -> FEAT per D2")

    # Rule 5: experiment_proposal -> RSCH
    if kind == "experiment_proposal":
        return ("RSCH", 0.90, "kind=experiment_proposal -> RSCH per D2")

    # Rule 6: training_architecture_proposal -> FEAT (novel training capability)
    if kind == "training_architecture_proposal":
        return ("FEAT", 0.80, "kind=training_architecture -> FEAT (novel capability)")

    # Rule 7: studio_delivery_proposal -> DOC (deliverable IS a document/video)
    if kind == "studio_delivery_proposal":
        return ("DOC", 0.75, "kind=studio_delivery -> DOC (deliverable artifact)")

    # Rule 8: warrior_proposal -> FEAT (new warrior = new capability)
    if kind == "warrior_proposal":
        return ("FEAT", 0.85, "kind=warrior_proposal -> FEAT (new capability)")

    # Rule 9: charter_proposal -> DOC (deliverable IS a document)
    if kind == "charter_proposal":
        return ("DOC", 0.80, "kind=charter_proposal -> DOC (document deliverable)")

    # Rule 10: naming_proposal -> REF (rename, no functional change)
    if kind == "naming_proposal":
        return ("REF", 0.85, "kind=naming_proposal -> REF (rename)")

    # Rule 11: documentation_cleanup -> REF
    if kind == "documentation_cleanup":
        return ("REF", 0.85, "kind=documentation_cleanup -> REF")

    # Rule 12: winding_proposal -> AUD (ecosystem lensing audit)
    if kind == "winding_proposal":
        return ("AUD", 0.80, "kind=winding_proposal -> AUD (ecosystem audit)")

    # Rule 13: evolution_proposal -> FEAT
    if kind == "evolution_proposal":
        return ("FEAT", 0.80, "kind=evolution_proposal -> FEAT (new capability)")

    # Rule 14: docs -> REF or AUD per content
    if kind == "docs":
        if _AUD_TITLE.search(title_text):
            return ("AUD", 0.70, "kind=docs + audit-related title -> AUD")
        return ("REF", 0.70, "kind=docs -> REF (disposition/doc cleanup)")

    # Rule 15: birth -> MISC (auto-generated birth proposals)
    if kind == "birth":
        return ("MISC", 0.75, "kind=birth -> MISC (auto-generated, needs manual review)")

    # ---- Kind + title keyword rules for generic kinds ----

    # -- What: check for BUG indicators in refactor-kind proposals
    if kind == "refactor" and _BUG_TITLE.search(title_text):
        bug_score = len(_BUG_TITLE.findall(title_text))
        ref_score = len(_REF_TITLE.findall(title_text))
        if bug_score > ref_score:
            return ("BUG", 0.80,
                    f"kind=refactor + bug title keywords ({bug_score} hits) -> BUG per D2")

    # -- What: ai_interaction_proposal with parity keywords -> AUD
    if kind in ("ai_interaction_proposal",):
        if _AUD_TITLE.search(title_text):
            return ("AUD", 0.80,
                    "kind=ai_interaction + audit/parity title keywords -> AUD")
        return ("FEAT", 0.70,
                "kind=ai_interaction + no audit keywords -> FEAT")

    # -- What: feature/design with new capability indicators -> FEAT
    if kind in ("feature", "feature_proposal", "design"):
        if _BUG_TITLE.search(title_text) and not _FEAT_TITLE.search(title_text):
            return ("BUG", 0.75,
                    "kind=feature/design but title has bug keywords -> BUG")
        return ("FEAT", 0.85, f"kind={kind} -> FEAT per D2")

    # -- What: refactor without bug keywords -> REF
    if kind in ("refactor", "refactor_program_proposal"):
        return ("REF", 0.85, f"kind={kind} -> REF per D2")

    # -- What: infrastructure kinds need title-keyword disambiguation
    if kind in ("infrastructure_proposal", "infrastructure_pivot_proposal",
                "infrastructure_migration_proposal", "memory_architecture_proposal",
                "integration_proposal"):
        # -- What: score title keywords per prefix category
        # -- Why: title is the strongest signal for intent; recommendation
        #   text may mention keywords from adjacent concerns
        feat_hits = len(_FEAT_TITLE.findall(title_text))
        ref_hits = len(_REF_TITLE.findall(title_text))
        aud_hits = len(_AUD_TITLE.findall(title_text))
        ops_hits = len(_OPS_TITLE.findall(title_text))

        scores = {
            "FEAT": feat_hits,
            "REF": ref_hits,
            "AUD": aud_hits,
            "OPS": ops_hits,
        }
        best = max(scores, key=scores.get)

        if scores[best] == 0:
            # No title keyword hits -- default based on kind
            if "migration" in kind:
                return ("REF", 0.65,
                        f"kind={kind}, no title keyword hits -> REF (migration)")
            # memory_architecture_proposal is typically novel capability
            if "memory" in kind:
                return ("FEAT", 0.70,
                        f"kind={kind} -> FEAT (memory architecture = novel capability)")
            return ("FEAT", 0.60,
                    f"kind={kind}, no title keyword hits -> FEAT (default for infra)")

        confidence = min(0.90, 0.65 + 0.05 * scores[best])
        return (best, confidence,
                f"kind={kind}, title keyword winner={best} ({scores[best]} hits)")

    # ---- Directive proposal fallback ----
    if kind == "directive_proposal":
        if _AUD_TITLE.search(title_text):
            return ("AUD", 0.75, "kind=directive + audit title keywords -> AUD")
        if _REF_TITLE.search(title_text):
            return ("REF", 0.75, "kind=directive + refactor title keywords -> REF")
        return ("MISC", 0.50, f"kind={kind}, no matching rule -> MISC")

    # ---- Fallback ----
    return ("MISC", 0.50, f"kind={kind}, no matching rule -> MISC")


# ============================================================================
# New id generation
# ============================================================================

def generate_new_id(old_id: str, prefix: str) -> str:
    """Generate the new prefix-coded id from the old PROP-* id.

    -- What: produces PREFIX-YYYY-MM-DD-slug-v01 format
    -- Why: D2 specifies new ids preserve the original date and use a
       human-readable slug; version always starts at v01 for the migration
    """
    date_str = _extract_date_from_id(old_id)
    slug = _slug_from_id(old_id)

    # -- What: ensure slug is non-empty and sanitized
    if not slug or slug == "unknown":
        slug = old_id.replace("PROP-", "").replace("/", "-").lower()

    return f"{prefix}-{date_str}-{slug}-v01"


# ============================================================================
# Plan generation
# ============================================================================

def _build_plan(proposals: list[dict]) -> list[dict]:
    """Build the migration plan from the loaded corpus.

    -- What: iterates all proposals, infers prefix, generates new id, and
       collects the plan rows
    -- Why: the plan is written as a markdown table for operator review
       before any apply() call
    """
    plan = []
    for prop in proposals:
        old_id = prop.get("id", "")
        current_kind = prop.get("kind", "")
        prefix, confidence, rationale = infer_prefix(prop)
        new_id = generate_new_id(old_id, prefix)

        plan.append({
            "legacy_id": old_id,
            "current_kind": current_kind,
            "inferred_prefix": prefix,
            "new_id": new_id,
            "confidence": confidence,
            "rationale": rationale,
            "source_path": prop.get("_source_path", ""),
            "title": prop.get("title", "") or "(empty)",
        })

    return plan


# -- What: mapping from prefix to the canonical kind string
# -- Why: the kind field in proposal JSON must align with the new prefix
PREFIX_TO_KIND = {
    "AUD": "audit",
    "BUG": "bug_fix",
    "FEAT": "feature",
    "RSCH": "research",
    "REF": "refactor",
    "OPS": "operations",
    "DOC": "documentation",
    "MISC": "miscellaneous",
}


def dry_run(
    prop_dir: Path = PROP_DIR_DEFAULT,
    output_path: Path = PLAN_OUTPUT_DEFAULT,
) -> list[dict]:
    """Generate the migration plan and write it as a markdown table.

    -- What: loads corpus, builds plan, writes markdown, returns plan rows
    -- Why: operator must review the plan before any file modifications;
       this is the safety gate per D2
    """
    proposals = load_proposal_corpus(prop_dir)
    plan = _build_plan(proposals)

    # -- What: sort plan by date then legacy_id for readability
    plan.sort(key=lambda r: r["legacy_id"])

    # -- What: count prefixes and low-confidence entries for summary
    prefix_counts = {}
    low_confidence = []
    for row in plan:
        p = row["inferred_prefix"]
        prefix_counts[p] = prefix_counts.get(p, 0) + 1
        if row["confidence"] < 0.75:
            low_confidence.append(row)

    # -- What: write the markdown plan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("# AUD84 Migration Plan -- Dry Run\n\n")
        out.write(f"Generated: {datetime.now().isoformat()}\n\n")
        out.write(f"Corpus size: {len(plan)} proposals\n\n")

        out.write("## Prefix distribution\n\n")
        out.write("| Prefix | Count |\n")
        out.write("|--------|-------|\n")
        for pfx in VALID_PREFIXES:
            out.write(f"| {pfx} | {prefix_counts.get(pfx, 0)} |\n")
        out.write("\n")

        if low_confidence:
            out.write(f"## Low-confidence entries ({len(low_confidence)} rows, "
                      f"confidence < 0.75) -- OPERATOR REVIEW REQUIRED\n\n")
            out.write("| legacy_id | current_kind | inferred_prefix | "
                      "confidence | rationale |\n")
            out.write("|-----------|-------------|-----------------|"
                      "------------|----------|\n")
            for row in low_confidence:
                out.write(f"| {row['legacy_id']} | {row['current_kind']} | "
                          f"{row['inferred_prefix']} | {row['confidence']:.2f} | "
                          f"{row['rationale']} |\n")
            out.write("\n")

        out.write("## Full migration plan\n\n")
        out.write("| legacy_id | current_kind | inferred_prefix | new_id | "
                  "confidence | rationale |\n")
        out.write("|-----------|-------------|-----------------|--------|"
                  "------------|----------|\n")
        for row in plan:
            out.write(
                f"| {row['legacy_id']} | {row['current_kind']} | "
                f"{row['inferred_prefix']} | {row['new_id']} | "
                f"{row['confidence']:.2f} | {row['rationale']} |\n"
            )
        out.write("\n")

    print(f"[DRY-RUN] Plan written to {output_path}")
    print(f"[DRY-RUN] {len(plan)} proposals categorized:")
    for pfx in VALID_PREFIXES:
        c = prefix_counts.get(pfx, 0)
        if c:
            print(f"  {pfx}: {c}")
    if low_confidence:
        print(f"[DRY-RUN] {len(low_confidence)} low-confidence entries "
              f"flagged for manual review")

    return plan


# ============================================================================
# Apply (operator-gated)
# ============================================================================

def apply(
    prop_dir: Path = PROP_DIR_DEFAULT,
    output_path: Path = PLAN_OUTPUT_DEFAULT,
    audit_log_path: Path = AUDIT_LOG_DEFAULT,
) -> int:
    """Apply the migration: rename files, update ids, set legacy_id.

    -- What: for each proposal in the plan, atomically renames the file from
       PROP-*.json to PREFIX-*.json, updates the id and kind fields, and
       preserves the original id as legacy_id
    -- Why: this is the destructive step; only called with --apply flag after
       operator reviews the dry-run plan

    Returns the count of migrated proposals.
    """
    proposals = load_proposal_corpus(prop_dir)
    plan = _build_plan(proposals)

    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    migrated = 0

    for row in plan:
        source_path = Path(row["source_path"])

        # -- What: idempotency check -- skip if file was already migrated
        # -- Why: re-running on already-migrated corpus should be a no-op
        if not source_path.exists():
            continue

        new_filename = f"{row['new_id']}.json"
        dest_path = source_path.parent / new_filename

        # -- What: skip if destination already exists (already migrated)
        if dest_path.exists():
            _log_action(audit_log_path, row, "skipped", "dest already exists")
            continue

        # -- What: read the original JSON, update fields, write back, then rename
        # -- Why: writing JSON first ensures the file content is correct before
        #   the atomic move; shutil.move handles cross-device renames
        try:
            with open(source_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # Preserve mtime
            orig_stat = os.stat(source_path)

            data["legacy_id"] = data["id"]
            data["id"] = row["new_id"]
            data["kind"] = PREFIX_TO_KIND.get(row["inferred_prefix"],
                                              data.get("kind", ""))

            with open(source_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")

            # -- What: atomic rename via shutil.move
            # -- Why: shutil.move uses os.rename when possible (same filesystem),
            #   which is atomic; falls back to copy+delete for cross-device
            shutil.move(str(source_path), str(dest_path))

            # -- What: restore original mtime on the renamed file
            os.utime(dest_path, (orig_stat.st_atime, orig_stat.st_mtime))

            _log_action(audit_log_path, row, "migrated", "success")
            migrated += 1

        except Exception as exc:
            _log_action(audit_log_path, row, "error", str(exc))
            print(f"[ERROR] {row['legacy_id']}: {exc}", file=sys.stderr)

    print(f"[APPLY] {migrated}/{len(plan)} proposals migrated")
    return migrated


def _log_action(
    log_path: Path,
    row: dict,
    action: str,
    detail: str,
) -> None:
    """Append a JSONL audit log entry.

    -- What: writes one line per migration action to the audit log
    -- Why: provides a reversible audit trail per D2 requirements
    """
    entry = {
        "ts": datetime.now().isoformat(),
        "action": action,
        "legacy_id": row["legacy_id"],
        "new_id": row["new_id"],
        "prefix": row["inferred_prefix"],
        "confidence": row["confidence"],
        "detail": detail,
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    """CLI entry point for the migration script.

    -- What: parses --apply flag and dispatches to dry_run or apply
    -- Why: dry-run is default; --apply is the operator's explicit go-ahead
    """
    do_apply = "--apply" in sys.argv

    if do_apply:
        count = apply()
        print(f"Migration complete. {count} proposals renamed.")
    else:
        plan = dry_run()
        print(f"\nDry-run complete. Review the plan, then re-run with --apply.")


if __name__ == "__main__":
    main()


# [Asthenosphere] -- File Metadata
# Author:     HAP-S (Code Warrior)
# Project:    Asthenosphere / SAM-IA / SEWE
# Version:    1.0.0
# Updated:    2026-05-07
# Status:     active
# Role:       One-time migration script for AUD84 proposal prefix taxonomy
# Stability:  experimental (first run, operator-gated)
# ErrorModel: exceptions logged to audit JSONL; skips on error, continues
# Depends:    json, pathlib, shutil, re, datetime, os, sys (all stdlib)
# Exposes:    load_proposal_corpus, infer_prefix, generate_new_id, dry_run, apply
# Lines:      ~370
