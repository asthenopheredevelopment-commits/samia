"""samia.core.integrity.repair — the recall/consolidation repair triggers + P3 fallback.

Layer 1 (Owns / Depends):
    Owns:    the anchor-first repair TRIGGERS — recall_repair (FULL, Q3a strongest) +
             partial_repair (CONSOLIDATION/RECONCILIATION, PARTIAL) — the P3 no-anchor
             GENERATIVE-RECONSTRUCTION fallback (_generative_reconstruct + its context
             gatherer _node_context + the two per-trigger fallback branches
             _recall_generative_fallback / _partial_generative_fallback) and the
             reconsolidation event log (_log_reconsolidation).
    Depends: samia.core.integrity.config (the repair-strength constants + generative_
             enabled gate + GENERATIVE_REPAIR_STRENGTH + _fm), samia.core.integrity.
             anchors (read_anchor, get_integrity, set_integrity),
             samia.core.integrity.erosion (reconsolidate_integrity math).
             samia.core.timestamp (event stamps, function-LOCAL) +
             samia.runtime.contradiction (P3 synthesize_node, function-LOCAL).

Layer 2 (What / Why):
    What: the HEALING half of the second axis. A repair reads the node, restores its
          served body byte-exact from the pristine anchor (anchor-first — NOT a guess),
          raises integrity toward FULL (fully on recall, partially on consolidation/
          reconciliation), persists it, and logs a reconsolidation event. When NO anchor
          remains, the P3 generative fallback MAY reconstruct the body via the local
          inference backend — gated, marked anchor_faithful=false, never while an anchor
          exists.
    Why:  "a node missing a bit is easily read + restored just from recalling it" — any
          genuine retrieval heals it (Q3a strongest = recall; sleep + reconciliation heal
          partially). Q1c/Q4a — generative is the LAST RESORT (the only drift path), so it
          is double-gated + honestly marked + anchor-first always wins.

SAFETY: anchor-first only in P1/P2 — a node with no anchor is a fail-soft no-op unless the
    gated P3 fallback is enabled. Every trigger fail-softs (no-op on missing node / write-
    reject, a logging failure never breaks the calling path). The anchor is READ, never
    written here.
"""

from __future__ import annotations

import json

from .config import (
    GENERATIVE_REPAIR_STRENGTH,
    Optional,
    PARTIAL_REPAIR_STRENGTH,
    Path,
    RECALL_REPAIR_STRENGTH,
    _fm,
    generative_enabled,
)
from .anchors import get_integrity, read_anchor, set_integrity
from .erosion import reconsolidate_integrity


def _node_context(memory_dir: Path, node_name: str, fm: dict) -> str:
    """A small context string (title/description/name) for a generative reconstruction.

    What: gathers the surviving non-body signals (the node's name/title/description from
      frontmatter) to give the reconstruction prompt what little context remains when the
      body has eroded and no anchor exists.
    Why: Q4a — the generative fallback reconstructs from the degraded served content + the
      node's CONTEXT; this is that context. Pure read of frontmatter, never mutates.
    """
    bits = []
    for key in ("name", "title", "description"):
        v = str(fm.get(key, "")).strip()
        if v:
            bits.append(f"{key}: {v}")
    return "\n".join(bits)


def _generative_reconstruct(memory_dir: Path, node_name: str, fm: dict,
                            degraded_body: str) -> Optional[str]:
    """Reconstruct a node body from its degraded content + context (P3 last resort).

    What: when NO anchor remains, reuse the SAME local-inference backend the judge uses
      (contradiction.synthesize_node — llama-cli/CHIRON) with a reconstruction prompt over
      the degraded (eroded) body + the node's surviving context. Returns the reconstructed
      body string, or None when generative repair is disabled/unavailable/unparseable.
    Why: Q1c/Q4a — generative reconstruction is the LAST RESORT used ONLY when the anchor
      itself is gone (a pre-anchor node or a deeply-lost anchor). It reuses the existing
      inference entrypoint (no new model loader) and is a SAFE NO-OP (None) when off — the
      caller treats None exactly like the prior anchor-missing no-op.

    SAFETY: gated by generative_enabled() (flag + backend availability). NEVER called while
      an anchor exists (anchor-first always wins — the callers check has_anchor first).
    """
    if not generative_enabled():
        return None
    # Function-LOCAL import — the runtime contradiction backend is a heavy/runtime dep kept
    # off the package import path (mirrors config.generative_enabled's lazy reach).
    try:
        from samia.runtime import contradiction as _contra
    except Exception:
        return None
    context = _node_context(memory_dir, node_name, fm)
    # Reuse synthesize_node's two-text contract: (degraded body, surviving context).
    # It returns {"title", "body"} or None (the safe no-op) — we want the body.
    try:
        synth = _contra.synthesize_node(degraded_body or "", context or node_name)
    except Exception:
        return None
    if not synth:
        return None
    body = str(synth.get("body", "")).strip()
    return body or None


def _log_reconsolidation(memory_dir: Path, record: dict) -> None:
    """Append a reconsolidation event to the bio reconsolidation log (jsonl)."""
    log_path = memory_dir / "biomimetic" / "integrity_reconsolidation_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Function-LOCAL import — timestamp is optional; a stamp failure must never break the
    # repair path (the log line is still written without a ts).
    try:
        from ..timestamp import now_utc_iso
        record.setdefault("ts", now_utc_iso())
    except Exception:
        pass
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def recall_repair(memory_dir: Path, node_name: str,
                  strength: float = RECALL_REPAIR_STRENGTH) -> dict:
    """On a GENUINE recall, restore the body byte-exact from the anchor + reset integrity.

    What: the recall-repair trigger (Q3a strongest, Q4a anchor-first). Reads the node,
      restores its served body from the pristine anchor (faithful, byte-exact — NOT a
      guess), resets integrity toward FULL by `strength` (1.0 = full reconsolidation for
      recall), persists the restored node, and logs a reconsolidation event. P1 wires the
      RECALL trigger only; consolidation/reconciliation partial repair is P2.
    Why: "a node missing a bit is easily read + restored just from recalling it." Any
      genuine retrieval that surfaces the node heals it.

    SAFETY: if no anchor exists, this is a no-op (P1 never guesses; generative fallback
      is P3). Fail-soft — never crashes the recall path.

    Returns a small telemetry dict {repaired, node, old_integrity, new_integrity, ...}.
    """
    nodes_dir = memory_dir / "nodes"
    fname = node_name if node_name.endswith(".md") else f"{node_name}.md"
    node_path = nodes_dir / fname
    if not node_path.exists():
        return {"repaired": False, "node": node_name, "skipped": "no-node-file"}

    try:
        fm, order, body = _fm.read_node(node_path)
    except (ValueError, OSError):
        return {"repaired": False, "node": node_name, "skipped": "unreadable"}

    anchor_body = read_anchor(memory_dir, node_name, fm)
    if anchor_body is None:
        # No anchor remains -> anchor-first cannot repair. P3 (Q1c/Q4a): the LAST-RESORT
        # generative reconstruction MAY rebuild the body from the degraded content +
        # context — ONLY when no anchor exists, ONLY when enabled + the backend is
        # available, and marked generative=true / anchor_faithful=false. Off/unavailable
        # -> a SAFE NO-OP, exactly as before.
        return _recall_generative_fallback(memory_dir, node_name, node_path, fm, order,
                                           body, strength)

    old_integrity = get_integrity(fm)
    new_integrity = reconsolidate_integrity(old_integrity, strength)

    # Anchor-first restore: the served body becomes the pristine anchor body (byte-exact).
    restored_body = anchor_body
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, restored_body, integrity_rewrite=True)
    except ValueError:
        # AUD61 frozen/archived protection (or other validation) — do not force-write.
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": "recall",
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(float(strength), 4),
        "anchor_faithful": True,
        "generative": False,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass  # fail-soft: a logging failure must never break the recall path
    return {"repaired": True, **rec}


def _recall_generative_fallback(memory_dir: Path, node_name: str, node_path: Path,
                                fm: dict, order: list[str], body: str,
                                strength: float) -> dict:
    """The P3 no-anchor generative branch of recall_repair (Q1c/Q4a, last resort).

    What: reached ONLY when no anchor exists. If the generative fallback is enabled + the
      backend is available, reconstructs the body from the degraded served content +
      context, raises integrity PARTIALLY (generative is not byte-faithful), persists it,
      and stamps a generative reconsolidation event (generative=true / anchor_faithful=
      false / confabulation_risk=true). Off/unavailable/unparseable -> the same safe
      no-anchor no-op as before (no crash, no fabrication).
    Why: Q4a — generative fallback ONLY when no anchor remains; honest provenance marking.
    """
    reconstructed = _generative_reconstruct(memory_dir, node_name, fm, body)
    if reconstructed is None:
        return {"repaired": False, "node": node_name, "skipped": "no-anchor"}

    old_integrity = get_integrity(fm)
    # Generative is NOT byte-faithful -> a PARTIAL raise, never a full pristine claim.
    new_integrity = reconsolidate_integrity(old_integrity, GENERATIVE_REPAIR_STRENGTH)
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, reconstructed, integrity_rewrite=True)
    except ValueError:
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": "recall",
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(float(GENERATIVE_REPAIR_STRENGTH), 4),
        "anchor_faithful": False,
        "generative": True,
        "confabulation_risk": True,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass
    return {"repaired": True, **rec}


def partial_repair(memory_dir: Path, node_name: str,
                   strength: float = PARTIAL_REPAIR_STRENGTH,
                   trigger: str = "consolidation") -> dict:
    """PARTIALLY repair a node's integrity (anchor-first), the P2 offline trigger.

    What: the CONSOLIDATION / RECONCILIATION repair (Q3a, PARTIAL). Reads the node,
      restores its served body from the pristine anchor (anchor-first, byte-exact — NOT
      a guess) but raises integrity only PARTIALLY toward FULL (strength < 1.0), persists
      the result, and logs a reconsolidation event tagged with the trigger. Distinct from
      recall_repair (which is FULL, strength 1.0): sleep + reconciliation heal a little of
      what they touch, recall heals fully.
    Why:  Q3a — RECALL strongest (full), CONSOLIDATION + RECONCILIATION partial. Anchor-
      first only — NO generative repair in P2 (that is P3); a node with no anchor is a
      fail-soft no-op here, exactly like recall_repair.

    SAFETY: anchor-first only; no anchor -> no-op (never guesses). Fail-soft — a repair
      error never breaks the consolidation/reconciliation path that called it. The anchor
      is read, NEVER written here (the anchor is the faithful source, untouched).

    Returns a small telemetry dict {repaired, node, old_integrity, new_integrity, ...}.
    """
    s = min(1.0, max(0.0, float(strength)))
    nodes_dir = memory_dir / "nodes"
    fname = node_name if node_name.endswith(".md") else f"{node_name}.md"
    node_path = nodes_dir / fname
    if not node_path.exists():
        return {"repaired": False, "node": node_name, "skipped": "no-node-file"}

    try:
        fm, order, body = _fm.read_node(node_path)
    except (ValueError, OSError):
        return {"repaired": False, "node": node_name, "skipped": "unreadable"}

    anchor_body = read_anchor(memory_dir, node_name, fm)
    if anchor_body is None:
        # No anchor remains -> anchor-first cannot repair. P3 (Q1c/Q4a): the LAST-RESORT
        # generative reconstruction MAY rebuild the body — ONLY when no anchor exists,
        # ONLY when enabled + the backend is available, marked generative=true /
        # anchor_faithful=false. Off/unavailable -> a SAFE NO-OP, exactly as before.
        return _partial_generative_fallback(memory_dir, node_name, node_path, fm, order,
                                            body, trigger)

    old_integrity = get_integrity(fm)
    new_integrity = reconsolidate_integrity(old_integrity, s)

    # Anchor-first restore: the served body becomes the pristine anchor body (byte-exact).
    # The integrity SCORE only rises partially, but the served content is faithful — the
    # HYBRID model stores integrity as a score, so a partial repair restores the body
    # from the anchor and tracks fidelity via the (partially-raised) score.
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, anchor_body, integrity_rewrite=True)
    except ValueError:
        # AUD61 frozen/archived protection (or other validation) — do not force-write.
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": trigger,
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(s, 4),
        "anchor_faithful": True,
        "generative": False,
        "partial": True,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass  # fail-soft: a logging failure must never break the calling path
    return {"repaired": True, **rec}


def _partial_generative_fallback(memory_dir: Path, node_name: str, node_path: Path,
                                 fm: dict, order: list[str], body: str,
                                 trigger: str) -> dict:
    """The P3 no-anchor generative branch of partial_repair (Q1c/Q4a, last resort).

    What: reached ONLY when no anchor exists on a consolidation/reconciliation repair. If
      the generative fallback is enabled + available, reconstructs the body from the
      degraded served content + context, raises integrity PARTIALLY, persists it, and
      stamps a generative reconsolidation event (generative=true / anchor_faithful=false /
      confabulation_risk=true). Off/unavailable/unparseable -> the same safe no-anchor
      no-op (no crash, no fabrication).
    Why: Q4a — generative fallback ONLY when no anchor remains; honest provenance marking.
    """
    reconstructed = _generative_reconstruct(memory_dir, node_name, fm, body)
    if reconstructed is None:
        return {"repaired": False, "node": node_name, "skipped": "no-anchor"}

    old_integrity = get_integrity(fm)
    new_integrity = reconsolidate_integrity(old_integrity, GENERATIVE_REPAIR_STRENGTH)
    set_integrity(fm, order, new_integrity)
    try:
        _fm.write_node(node_path, fm, order, reconstructed, integrity_rewrite=True)
    except ValueError:
        return {"repaired": False, "node": node_name, "skipped": "write-rejected"}

    rec = {
        "event": "reconsolidation",
        "trigger": trigger,
        "node": node_name,
        "old_integrity": round(old_integrity, 6),
        "new_integrity": round(new_integrity, 6),
        "strength": round(float(GENERATIVE_REPAIR_STRENGTH), 4),
        "anchor_faithful": False,
        "generative": True,
        "confabulation_risk": True,
        "partial": True,
    }
    try:
        _log_reconsolidation(memory_dir, rec)
    except Exception:
        pass
    return {"repaired": True, **rec}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.integrity.repair
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.integrity monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       the HEALING half — the anchor-first repair triggers (recall_repair FULL,
#             partial_repair PARTIAL), the P3 no-anchor generative fallback (the
#             reconstruct primitive + context gatherer + the two per-trigger branches),
#             and the reconsolidation event log.
# Stability:  stable — carved byte-identically from the monolith. Anchor-first always
#             wins; the anchor is READ here, never written (the faithful source is
#             untouched). recall = full restore (strength 1.0), consolidation/
#             reconciliation = partial (< 1.0).
# ErrorModel: fail-soft throughout — a missing node / unreadable / write-rejected node
#             returns a no-op telemetry dict, never raises into the calling path; a log
#             failure is swallowed. The P3 generative branch is double-gated (config.
#             generative_enabled — flag + backend) + a SAFE NO-OP (None) when off,
#             marked anchor_faithful=false / generative=true / confabulation_risk=true,
#             and NEVER runs while an anchor exists.
# Depends:    json (stdlib). .config (strength constants + generative_enabled + _fm),
#             .anchors (read_anchor/get_integrity/set_integrity), .erosion
#             (reconsolidate_integrity). samia.core.timestamp +
#             samia.runtime.contradiction (function-LOCAL only).
# Exposes:    recall_repair, partial_repair (+ the _node_context/_generative_reconstruct/
#             _log_reconsolidation/_recall_generative_fallback/_partial_generative_
#             fallback privates).
# Lines:      377
# --------------------------------------------------------------------------
