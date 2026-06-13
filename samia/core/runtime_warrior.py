"""samia.core.runtime_warrior -- Ephemeral runtime warrior lifecycle.

Spawns capability-constrained ephemeral warriors from a template .loer,
logs outcomes, computes stable pattern signatures, and surfaces promotion
proposals when recurring patterns succeed.  Slice 4 (2026-05-01).

Public API:
  spawn_runtime_warrior(task, caps, constraints) -> dict
  log_runtime_outcome(id, outcome, sig, details)  -> None
  compute_pattern_signature(task, caps)            -> str
  propose_promotion(min_reuses=3)                  -> list[dict]
  write_promotion_proposal(proposal)               -> Path
"""
from __future__ import annotations

import hashlib, json, re, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -- Paths ------------------------------------------------------------------

RUNTIME_DIR = Path("/tmp/asthenos_runtime_warriors")
LOG_DIR = Path.home() / ".local" / "share" / "asthenos" / "handoff"
LOG_FILE = LOG_DIR / "runtime_warrior_log.jsonl"
TEMPLATE_PATH = (
    Path.home() / "Asthenosphere" / "src" / "directives"
    / "code_warrior" / "runtime_warrior_template.loer"
)
PROMOTION_INBOX = (
    Path.home() / "Asthenosphere" / "src" / "directives"
    / "Inbox" / "runtime_warrior_promotions"
)

FORBIDDEN_CAPABILITIES = frozenset({
    "git_commit", "git_push", "destructive_fs", "oauth",
    "network_unrestricted", "modify_permanent_loer",
    "spawn_runtime_warrior", "financial_access",
})

# -- Pattern signature ------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", text.lower().strip())


def compute_pattern_signature(
    task_description: str, capabilities: list[str],
) -> str:
    """Stable SHA-256 hex digest of normalized task + sorted capabilities.

    Conservative: same words in different order = different sig.
    Capability order is irrelevant (sorted). Duplicates collapsed.
    """
    norm_task = _normalize_text(task_description)
    norm_caps = sorted(set(c.lower().strip() for c in capabilities))
    payload = norm_task + "\x00" + ",".join(norm_caps)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# -- Spawn ------------------------------------------------------------------

def spawn_runtime_warrior(
    task_description: str,
    capabilities: list[str],
    constraints: dict[str, Any] | None = None,
    *, runtime_dir: Path | None = None,
    template_path: Path | None = None,
) -> dict[str, Any]:
    """Stamp an ephemeral .loer from the template; return invocation spec.

    Returns dict: invocation_id, path, pattern_signature, capabilities,
    constraints, recommended_prompt, timestamp.
    """
    constraints = constraints or {}

    requested = {c.lower().strip() for c in capabilities}
    violations = requested & FORBIDDEN_CAPABILITIES
    if violations:
        raise ValueError(f"Forbidden capabilities requested: {sorted(violations)}")

    rd = runtime_dir or RUNTIME_DIR
    tp = template_path or TEMPLATE_PATH

    invocation_id = f"rw_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    pattern_sig = compute_pattern_signature(task_description, capabilities)

    task_name = re.sub(r"[^a-z0-9]+", "_", task_description.lower().strip())[:48]
    task_name = task_name.strip("_") or "unnamed_task"

    template_text = tp.read_text(encoding="utf-8")

    caps_sorted = sorted(set(c.strip() for c in capabilities))
    caps_str = ", ".join(caps_sorted)
    caps_list_str = "\n  ".join(f"- {c}" for c in caps_sorted)
    needs_str = " | ".join(sorted(set(c.strip().title() for c in capabilities)))
    constraints_str = "; ".join(
        f"{k}: {v}" for k, v in sorted(constraints.items())
    ) or "default"

    replacements = {
        "{{TASK_NAME}}": task_name,
        "{{TASK_DESCRIPTION}}": task_description.strip(),
        "{{TIMESTAMP}}": timestamp,
        "{{INVOCATION_ID}}": invocation_id,
        "{{PATTERN_SIGNATURE}}": pattern_sig,
        "{{CAPABILITIES}}": caps_str,
        "{{CAPABILITIES_LIST}}": caps_list_str,
        "{{NEEDS}}": needs_str or "Read",
        "{{CONSTRAINTS}}": constraints_str,
        "{{EXIT_CONDITION}}": constraints.get(
            "exit_condition", "Task deliverables completed and self-assessed."
        ),
    }
    content = template_text
    for ph, val in replacements.items():
        content = content.replace(ph, val)

    rd.mkdir(parents=True, exist_ok=True)
    out_path = rd / f"{invocation_id}.loer"
    out_path.write_text(content, encoding="utf-8")

    recommended_prompt = (
        f"You are an ephemeral runtime warrior. Directive: {out_path}\n"
        f"Task: {task_description.strip()}\n"
        f"Capabilities: {caps_str}\n"
        f"Constraints: NO commits, NO destructive ops, NO OAuth, "
        f"NO network outside whitelist, NO spawning children.\n"
        f"When done, self-assess outcome as success/partial/failure."
    )

    return {
        "invocation_id": invocation_id,
        "path": str(out_path),
        "pattern_signature": pattern_sig,
        "capabilities": caps_sorted,
        "constraints": constraints,
        "recommended_prompt": recommended_prompt,
        "timestamp": timestamp,
    }

# -- Log --------------------------------------------------------------------

def log_runtime_outcome(
    invocation_id: str, outcome: str, pattern_signature: str,
    details: dict[str, Any] | None = None,
    *, log_file: Path | None = None,
) -> None:
    """Append one outcome record to the runtime warrior JSONL log."""
    if outcome not in ("success", "partial", "failure"):
        raise ValueError(f"outcome must be success|partial|failure, got: {outcome}")

    lf = log_file or LOG_FILE
    lf.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "invocation_id": invocation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "pattern_signature": pattern_signature,
        "details": details or {},
    }
    with lf.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

# -- Promotion --------------------------------------------------------------

def _load_log(log_file: Path | None = None) -> list[dict]:
    """Load all records from the runtime warrior log."""
    lf = log_file or LOG_FILE
    if not lf.exists():
        return []
    records: list[dict] = []
    for line in lf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def propose_promotion(
    min_reuses: int = 3, *, log_file: Path | None = None,
) -> list[dict]:
    """Scan log for patterns with N+ successes; return promotion proposals.

    Qualifies when: successes >= min_reuses AND success_rate > 80%.
    """
    records = _load_log(log_file)
    by_sig: dict[str, list[dict]] = {}
    for r in records:
        sig = r.get("pattern_signature", "")
        if sig:
            by_sig.setdefault(sig, []).append(r)

    proposals = []
    for sig, entries in by_sig.items():
        oc = Counter(e.get("outcome", "") for e in entries)
        s, f, p = oc.get("success", 0), oc.get("failure", 0), oc.get("partial", 0)
        total = s + f + p
        if s < min_reuses:
            continue
        rate = s / total if total else 0.0
        if rate <= 0.80:
            continue
        sample = {}
        for e in reversed(entries):
            if e.get("outcome") == "success":
                sample = e.get("details", {})
                break
        proposals.append({
            "pattern_signature": sig, "total": total,
            "successes": s, "failures": f, "partials": p,
            "success_rate": round(rate, 3), "sample_details": sample,
        })
    return proposals


def write_promotion_proposal(
    proposal: dict[str, Any], *, inbox_dir: Path | None = None,
) -> Path:
    """Write a draft .loer to Inbox/runtime_warrior_promotions/ for operator review."""
    inbox = inbox_dir or PROMOTION_INBOX
    inbox.mkdir(parents=True, exist_ok=True)

    sig = proposal["pattern_signature"]
    short = sig[:16]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sample_json = json.dumps(proposal.get("sample_details", {}), indent=2)

    draft = (
        f"# draft_{short}.loer -- Promotion Proposal (auto-generated)\n"
        f"#\n"
        f"# Requires OPERATOR REVIEW before becoming a permanent filament.\n"
        f"# DO NOT deploy without review.\n\n"
        f"!name:        Promoted Runtime Warrior -- {short}\n"
        f"!type:        filament\n"
        f"!compatible:  >=2.8.0\n"
        f"@version:     0.1.0-draft\n"
        f"*author:      system (promotion proposal)\n"
        f"*license:     MIT\n"
        f"*promoted_from_pattern: {sig}\n"
        f"~winding_accuracy  :: enum : MEDIUM\n"
        f"?tags :: list : promoted | runtime | auto-proposed\n"
        f"!category:    promoted\n\n"
        f"[description]\n"
        f"Auto-proposed from recurring runtime warrior pattern {short}...\n"
        f"Used {proposal.get('total','?')} times, "
        f"{proposal.get('successes','?')} successes "
        f"(rate: {proposal.get('success_rate','?')}).\n\n"
        f"OPERATOR: Review, edit, then promote or delete to reject.\n"
        f"[/description]\n\n"
        f"{{requirements}}\n"
        f"  !needs: Read\n"
        f"  !compatible: >=2.8.0\n"
        f"{{/requirements}}\n\n"
        f"[directive]\n\n"
        f"# TODO(operator): Fill directive from recurring task pattern.\n"
        f"# Sample details from last success:\n"
        f"# {sample_json}\n\n"
        f"[/directive]\n\n"
        f"{{changelog}}\n"
        f"  @0.1.0-draft -- {ts} -- system\n"
        f"    + Auto-proposed from pattern {short}\n"
        f"    + {proposal.get('successes','?')} successes "
        f"out of {proposal.get('total','?')} runs\n"
        f"{{/changelog}}\n"
    )

    out_path = inbox / f"draft_{short}.loer"
    out_path.write_text(draft, encoding="utf-8")
    return out_path
