"""samia.core.integrity.anchors — the pristine recovery anchor store + integrity field.

Layer 1 (Owns / Depends):
    Owns:    the retained pristine recovery ANCHOR store (one byte-exact snapshot per
             node) and its primitives (_anchors_dir/_node_id/anchor_path/has_anchor/
             write_anchor/read_anchor/ensure_anchor), the [0,1] integrity-field
             read/write on a node's frontmatter (get_integrity/set_integrity), the
             distillation gate (is_distilled), and the two GENUINE-WRITE anchor-capture
             entrypoints (capture_on_write / capture_on_genuine_write).
    Depends: samia.core.integrity.config (the INTEGRITY_* endpoints, EROSION_SENTINEL,
             _fm). stdlib via config (Path/Optional). typing.Any (annotation only).

Layer 2 (What / Why):
    What: the faithful repair SOURCE half of the second axis. The anchor is the
          pristine body snapshot; the integrity field is the [0,1] fraction-intact
          score on the node's frontmatter. The capture entrypoints snapshot the
          PRISTINE just-written body on a genuine write so the node can thereafter
          erode and be faithfully repaired.
    Why:  Q1c HYBRID — real erosion of the served body, but a retained anchor so a
          recall restores byte-exact, not a guess. Erosion is anchor-GATED (a node
          never erodes without a recoverable anchor), so these capture seams are the
          engagement gate that makes the whole mechanism actually fire.

CRITICAL SAFETY: capture_on_write / capture_on_genuine_write are the ONLY anchor-write
    entrypoints outside a faithful repair. They MUST be called only with the PRISTINE
    just-written body — NEVER from the erosion-persistence path. Capturing an eroded
    served body would clobber the anchor with degraded content and permanently defeat
    repair (data loss). capture_on_genuine_write has a defense-in-depth EROSION_SENTINEL
    guard; the erosion sweep persists via a path that leaves the anchor untouched.
"""

from __future__ import annotations

from typing import Any

from .config import (
    EROSION_SENTINEL,
    INTEGRITY_FULL,
    INTEGRITY_NONE,
    Optional,
    Path,
    _fm,
)


def _anchors_dir(memory_dir: Path) -> Path:
    """The retained-anchor store (one pristine snapshot per node)."""
    return memory_dir / "biomimetic" / "integrity_anchors"


def _node_id(node_name: str, fm: Optional[dict] = None) -> str:
    """Resolve a stable anchor id for a node.

    Mirrors ia.freeze's choice: prefer the `address` frontmatter, fall back to the
    node's file stem. Keeps anchor file names stable across a rename of the title.
    """
    if fm is not None:
        addr = fm.get("address")
        if addr:
            return str(addr)
    name = node_name
    if name.endswith(".md"):
        name = name[:-3]
    return name


def anchor_path(memory_dir: Path, node_name: str, fm: Optional[dict] = None) -> Path:
    """Absolute path to a node's pristine recovery anchor."""
    return _anchors_dir(memory_dir) / f"{_node_id(node_name, fm)}.txt"


def is_distilled(fm: dict) -> bool:
    """True iff a node's content has been semantically distilled (fm distilled == True).

    What: reads the boolean `distilled` frontmatter marker stamped by the fact-extract
      drain (rem_subscribers._sub_fact_extract) once a frozen source's content is
      semantically covered (>= 1 atom persisted OR all atoms dedup-skipped). Strictly
      `is True` — any other value (absent, False, a stray string) reads as NOT distilled.
    Why: TUNE-2026-06-10 operator decision (c), systems-consolidation gating — the
      episodic trace fades only AFTER the semantic representation forms. A frozen node
      erodes ONLY once this marker is set; an undistilled frozen node still never erodes
      (the integrity_decay_pass walk skips it). The strict `is True` test keeps the gate
      conservative: erosion (which loses served characters) requires an UNAMBIGUOUS
      distilled marker, never a truthy-but-unintended value.
    """
    return fm.get("distilled") is True


def get_integrity(fm: dict) -> float:
    """Read the [0,1] integrity field; a node with no field is pristine (1.0)."""
    try:
        v = float(fm.get("integrity", INTEGRITY_FULL))
    except (TypeError, ValueError):
        return INTEGRITY_FULL
    return min(INTEGRITY_FULL, max(INTEGRITY_NONE, v))


def set_integrity(fm: dict, order: list[str], value: float) -> None:
    """Write the [0,1] integrity field, appending it to `order` if new.

    Clamps to [0,1]. `serialize` emits appended keys at the end, so a new integrity
    field writes cleanly without disturbing existing key order.
    """
    v = min(INTEGRITY_FULL, max(INTEGRITY_NONE, float(value)))
    if "integrity" not in fm and "integrity" not in order:
        order.append("integrity")
    fm["integrity"] = round(v, 6)


def has_anchor(memory_dir: Path, node_name: str, fm: Optional[dict] = None) -> bool:
    """True iff a recoverable pristine anchor exists for this node."""
    return anchor_path(memory_dir, node_name, fm).exists()


def write_anchor(memory_dir: Path, node_name: str, body: str,
                 fm: Optional[dict] = None) -> Path:
    """Capture (or refresh) the pristine body snapshot for a node.

    What: writes the CURRENT (pristine) body to the anchor store. Called when a node
      is first written/seen and on each faithful repair (re-snapshot from the now-
      confirmed-pristine canonical body). The anchor is NEVER eroded in P1 — it is the
      faithful repair source.
    Why: Q1c HYBRID — real erosion of the served body, but a retained anchor so recall
      restores faithfully (byte-exact), not a guess.
    """
    ap = anchor_path(memory_dir, node_name, fm)
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(body, encoding="utf-8")
    return ap


def read_anchor(memory_dir: Path, node_name: str,
                fm: Optional[dict] = None) -> Optional[str]:
    """Return the pristine anchor body, or None if no anchor exists."""
    ap = anchor_path(memory_dir, node_name, fm)
    if not ap.exists():
        return None
    return ap.read_text(encoding="utf-8")


def ensure_anchor(memory_dir: Path, node_name: str, fm: dict, body: str) -> bool:
    """Capture a pristine anchor for a node IF it does not already have one.

    What: idempotent first-seen anchor capture — snapshots the current body as the
      pristine recovery anchor only when no anchor exists yet (so a later refresh on a
      faithful repair is the only re-snapshot). Returns True iff it wrote a new anchor.
    Why: erosion is gated on an anchor being present; this is the "node is written/first
      seen" capture point so the very first erosion already has a recoverable source.
    """
    if has_anchor(memory_dir, node_name, fm):
        return False
    write_anchor(memory_dir, node_name, body, fm)
    return True


def capture_on_write(memory_dir: Path, node_name: str, fm: dict, body: str) -> dict:
    """Capture/REFRESH the pristine anchor on a GENUINE node write (P2 capture hook).

    What: the anchor-capture-on-write entrypoint, called ONLY from the genuine node-write
      path (memory_write_node / the capture hook / an ia write of REAL operator/agent
      content). The just-written `body` IS the pristine version, so this REFRESHES the
      anchor to it unconditionally — a fresh node gains an anchor and a genuine re-write
      updates the anchor to the new pristine body. Returns a small {captured, refreshed,
      anchor} telemetry dict. Fail-soft — an anchor-write failure never breaks the write.
    Why:  the P1 caveat: nodes only erode once they HAVE an anchor, and P1 did NOT auto-
      capture. This is the engagement gate — wiring it into the genuine write path is what
      makes the whole second-axis mechanism actually fire (a node can thereafter erode and
      be faithfully repaired from this snapshot).

    CRITICAL SAFETY: this is the ONLY anchor-write entrypoint outside a faithful repair —
      it MUST be called only with the PRISTINE just-written body, NEVER from
      integrity_decay_pass / erode / the erosion-persistence path. Capturing an eroded
      served body would clobber the anchor with degraded content and permanently defeat
      repair (data loss). The erosion sweep persists the eroded body via a path that
      leaves the anchor untouched; this lives only at the genuine-write entrypoints.
    """
    existed = has_anchor(memory_dir, node_name, fm)
    try:
        write_anchor(memory_dir, node_name, body, fm)
    except OSError as e:
        return {"captured": False, "refreshed": False, "error": str(e)}
    return {"captured": not existed, "refreshed": existed,
            "anchor": str(anchor_path(memory_dir, node_name, fm))}


def capture_on_genuine_write(memory_dir: Path, node_name: str, fm: dict,
                             order: list[str], body: str) -> dict[str, Any]:
    """Anchor-capture on a GENUINE node write — the universal write_node seam.

    What: FEAT-2026-06-08. If the body is UNCHANGED from the current anchor, NO-OP
      (Q3b skip-unchanged — avoids re-anchoring the offload path's frequent metadata-
      only re-saves). Otherwise REFRESH the anchor to the new pristine body and, if the
      node was eroded (integrity < FULL), RESET integrity to FULL (Q2a — a genuine
      rewrite is the new pristine baseline, so the forgetting curve restarts from full).
      Mutates fm/order in place when it resets; the caller (write_node) serializes after.
      Returns small telemetry; fail-soft is the caller's responsibility.
    Why: capture_on_write fired ONLY on the MCP write_node op; session-offloads + other
      internal writers bypassed it via frontmatter.write_node, leaving nodes anchor-less
      and therefore un-erodable (erode is anchor-gated). Wiring this into write_node makes
      EVERY genuine writer anchor automatically.
    CRITICAL SAFETY (mirrors capture_on_write): a GENUINE-write entrypoint ONLY. NEVER
      call from erode / integrity_decay_pass / the erosion-persistence path — capturing an
      eroded served body would clobber the pristine anchor and permanently defeat repair.
      The write_node seam enforces this via the integrity_rewrite gate (only integrity_
      rewrite==False writes reach here).
    DEFENSE-IN-DEPTH: never anchor from a body that carries the EROSION_SENTINEL. An
      eroded/served body would clobber the pristine recovery source if anchored, and
      resetting integrity from it would hide real erosion. Only sentinel-free (genuinely
      pristine) content may refresh the anchor — this guards against ANY path that re-saves
      an eroded served body via write_node, not just the integrity_rewrite-marked ones. (A
      brand-new node whose genuine content legitimately contains '·' is anchored by the
      backstop sweep instead; a negligible edge.)
    """
    if EROSION_SENTINEL in body:
        return {"captured": False, "skipped": "eroded-body"}
    existing = read_anchor(memory_dir, node_name, fm)
    if existing is not None and existing == body:
        return {"captured": False, "skipped": "unchanged"}
    try:
        write_anchor(memory_dir, node_name, body, fm)
    except OSError as e:
        return {"captured": False, "skipped": "anchor-write-failed", "error": str(e)}
    reset = False
    if get_integrity(fm) < INTEGRITY_FULL:
        set_integrity(fm, order, INTEGRITY_FULL)
        reset = True
    return {"captured": True, "refreshed": existing is not None, "integrity_reset": reset}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.integrity.anchors
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.integrity monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       the pristine recovery anchor store + the [0,1] integrity field
#             accessors + the distillation gate + the two GENUINE-WRITE anchor-
#             capture entrypoints (the engagement gate the erosion mechanism rides).
# Stability:  stable — primitives carved byte-identically from the monolith; the
#             anchor is NEVER eroded (faithful repair source) and is written ONLY
#             from a genuine-write entrypoint or a faithful repair.
# ErrorModel: write_anchor mkdir+writes (the capture seams catch OSError -> telemetry);
#             read_anchor returns None on a missing anchor; capture_on_genuine_write
#             guards the EROSION_SENTINEL (never anchors a degraded body) + skips an
#             unchanged body. No path here ever erodes or deletes.
# Depends:    .config (INTEGRITY_FULL/INTEGRITY_NONE/EROSION_SENTINEL/_fm/Path/Optional),
#             typing.Any (annotation only).
# Exposes:    anchor_path, has_anchor, write_anchor, read_anchor, ensure_anchor,
#             is_distilled, get_integrity, set_integrity, capture_on_write,
#             capture_on_genuine_write (+ the _anchors_dir/_node_id privates).
# Lines:      250
# --------------------------------------------------------------------------
