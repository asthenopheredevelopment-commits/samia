"""samia.core.bio.salience — the salience / affective axis SOURCE (FEAT-2026-06-07 P2).

Layer 1 (Owns / Depends):
    Owns:    the salience SOURCE (D6 Q8a). _node_frontmatter (the thin frontmatter
             reader the salience helpers share); the three cheap composite signals
             (_salience_surprise = 1 - max-cosine vs the index, _salience_contradiction =
             in the supersession store, _salience_repetition = access + co-activation
             degree, saturating); compute_salience (aggregate + the explicit operator/
             agent tag override, normalized [0,1], optionally persisted to the `salience`
             frontmatter + the STC capture trigger); and salience_merge_guard (the
             read-only merge/supersede guard predicate other phases consult).
    Depends: config (constants SALIENCE_* / np); samia.core.bio.hebbian
             (_load_edge_weights — the Tier-0 degree read — via plain import);
             samia.core.{vector, frontmatter, temporal_recall_stc} + samia.runtime.
             contradiction (lazy, function-local — the embedder/index, frontmatter
             read/write, the supersession reader, the STC capture hook).

Layer 2 (What / Why):
    What: the cheap composite salience score + the explicit-tag override + the
          read-only merge guard. A MISSING signal contributes 0 (never crashes).
    Why:  carved out of the monolith as the salience responsibility. This is the
          SOURCE + storage + explicit-tag + guard ONLY; the salience EFFECTS (promotion
          gate, dampened decay, merge auto-action) live in hippocampus / tier / the
          merge consumer. vector / frontmatter / contradiction / temporal_recall_stc are
          lazy (function-local) exactly as the monolith had them — keeps the import
          cheap and breaks the bio<->contradiction / bio<->stc cycles.
"""

from __future__ import annotations

from typing import Optional

from . import config as _cfg
from .config import (
    np,
    SALIENCE_W_SURPRISE,
    SALIENCE_W_CONTRADICTION,
    SALIENCE_W_REPETITION,
    SALIENCE_REPETITION_SATURATION,
    SALIENCE_TAG_VALUE,
    SALIENCE_MERGE_GUARD_DEFAULT,
)
from .hebbian import _load_edge_weights


def _node_frontmatter(memory_dir, node: str) -> Optional[tuple[dict, list, str]]:
    """Read (fm, order, body) of a node; None if missing/unparseable.

    What: a thin wrapper over frontmatter.read_node for the salience helpers.
    Why:  compute_salience and salience_merge_guard read frontmatter fields
      (access_count, salience, salience_tag) and write the salience field back; one
      reader keeps both paths consistent and fail-soft on a node without frontmatter.
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = memory_dir / "nodes" / fname
    if not p.exists():
        return None
    try:
        from samia.core import frontmatter as _fm
        fm, order, body = _fm.read_node(p)
        return fm, order, body
    except Exception:
        return None


def _salience_surprise(memory_dir, content: str,
                       exclude_node: Optional[str] = None) -> float:
    """Surprise/novelty signal: 1 - max_cosine of `content` vs the main vector index.

    What: embed `content` with the shared backend, cosine it against the existing main
      embeddings (vector_index/embeddings.npy), and return 1 - max_cosine clamped to
      [0,1]. A node unlike anything stored scores high (surprising); a near-duplicate
      scores low. Returns 0.0 (a MISSING signal contributes nothing) when the index is
      absent/empty or the embedder is unavailable — never crashes.
    Why:  D6 Q8a signal 1 — prediction-error vs the index, RELATIVE (calibrated against
      what is already stored) so a uniformly-novel corpus does not all score high
      (Risk 8). Reuses vector._embed_batch + the main index; reinvents no embedding.

    Args:
        exclude_node: FEAT-2026-06-11 salience-coverage P2 — when given (a node id,
          with or without .md), DROP that node's own embedding row from the index
          before taking max_cosine (a leave-one-out). DEFAULT None keeps the legacy
          byte-identical behavior. Why: an ALREADY-INDEXED node self-matches (cos≈1)
          so its raw surprise is degenerately ~0; the backfill (which scores nodes that
          are already in the index) excludes the node's own row so surprise reflects
          novelty vs the REST of the corpus, not vs itself. A missing manifest/row is
          fail-soft: excluding nothing degrades to the legacy (self-matched) value.
    """
    try:
        from samia.core import vector as _vi
        ip = memory_dir / "vector_index" / "embeddings.npy"
        if not ip.exists():
            return 0.0  # missing signal -> 0
        emb = np.load(ip)
        if emb.shape[0] == 0:
            return 0.0
        # LeaveOneOut — What: when exclude_node is set, delete its embedding row from the
        #     index matrix so the node cannot self-match to cos≈1 before max_cosine.
        # P2 leave-one-out: drop the node's own row so it cannot self-match to cos≈1.
        # Fail-soft — a missing manifest/row/tombstone just leaves emb intact (legacy).
        if exclude_node is not None:
            try:
                fname = exclude_node if exclude_node.endswith(".md") else f"{exclude_node}.md"
                manifest = _vi._load_manifest(memory_dir)
                entry = manifest.get("entries", {}).get(fname)
                row = entry.get("row") if isinstance(entry, dict) else None
                if isinstance(row, int) and 0 <= row < emb.shape[0]:
                    emb = np.delete(emb, row, axis=0)
            except Exception:
                pass  # excluding nothing falls back to the legacy self-matched value
            if emb.shape[0] == 0:
                return 0.0  # the corpus was just this one node -> no comparison signal
        # LeaveOneOut — Why: surprise is RELATIVE to the rest of the corpus; an already-
        #     indexed node would otherwise self-match (cos≈1) and score ~0, hiding real novelty.
        q = _vi._embed_batch([content])[0]
        sims = emb @ q
        max_cos = float(np.max(sims))
        return float(min(1.0, max(0.0, 1.0 - max_cos)))
    except Exception:
        return 0.0  # any failure is a missing signal, not a crash


def _salience_contradiction(memory_dir, node: str) -> float:
    """Contradiction-involvement signal: 1.0 if the node is in a supersession candidate.

    What: scan the unified supersession-candidate store (contradiction.
      list_supersession_candidates) for any UNRESOLVED candidate naming `node` as
      either the old_id or the new_id; return 1.0 if found, else 0.0.
    Why:  D6 Q8a signal 2 — a node that triggered or resolved a supersession is
      important (a belief-overturning contradiction). Reuses the existing canonical
      candidate reader; a missing store -> 0.0 (missing signal contributes nothing).
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    try:
        from samia.runtime import contradiction as _con
        cands = _con.list_supersession_candidates(memory_dir, unresolved_only=False)
    except Exception:
        return 0.0
    for c in cands:
        if c.get("old_id") == fname or c.get("new_id") == fname:
            return 1.0
    return 0.0


def _salience_repetition(memory_dir, node: str, fm: Optional[dict]) -> float:
    """Repetition signal: small saturating contribution from access + co-activation.

    What: combine the node's frontmatter access_count with its Tier-0 co-activation
      degree (edge_weights pairs touching the node) into a saturating [0,1] value
      (count / SALIENCE_REPETITION_SATURATION, clamped to 1.0).
    Why:  D6 Q8a signal 3 — frequency contributes, but it is the SMALLEST, fastest-
      saturating slice so salience is not reducible to frequency. Reuses the existing
      access_count frontmatter + the edge_weights map; a missing signal -> 0.
    """
    count = 0.0
    if fm is not None:
        try:
            count += float(fm.get("access_count", 0) or 0)
        except (TypeError, ValueError):
            pass
    # Tier-0 co-activation degree (pairs touching the node).
    fname = node if node.endswith(".md") else f"{node}.md"
    try:
        weights = _load_edge_weights(memory_dir)
        for key in weights:
            if fname in key.split("::"):
                count += 1.0
    except Exception:
        pass
    sat = max(SALIENCE_REPETITION_SATURATION, 1e-9)
    return float(min(1.0, count / sat))


def compute_salience(memory_dir, node: str,
                     content: Optional[str] = None,
                     explicit_tag: Optional[bool] = None,
                     write: bool = True,
                     exclude_self_from_surprise: bool = False) -> float:
    """Compute (and optionally persist) a node's [0,1] salience score (D6 Q8a SOURCE).

    What: aggregate the three cheap composite signals — surprise (1 - max_cosine vs the
      vector index), contradiction-involvement (in the supersession store), repetition
      (access + co-activation, saturating) — into a weighted [0,1] score, then apply the
      EXPLICIT operator/agent tag override: an explicit tag (passed via `explicit_tag`
      or already on the node's `salience_tag` frontmatter) clamps salience HIGH
      (SALIENCE_TAG_VALUE) regardless of the composite. When `write` (default), the
      result is written to the node's `salience` frontmatter field (and `salience_tag`
      is persisted when set so the override is sticky/operator-visible).
    Why:  D6 — the salience SOURCE. Each signal is grounded in a primitive already in
      the system (Risk: negligible write/touch cost) and a MISSING signal contributes
      0 (no crash). This builds only the SOURCE + storage + explicit-tag path; the
      EFFECTS (promotion / decay / merge) are P3/P5/consumers — NOT applied here.

    Args:
        node: the node id (with or without .md).
        content: the text to score surprise against; defaults to the node's own
          embedding-ready content (title + description + body) when omitted.
        explicit_tag: True to SET the operator/agent override (clamps high), False to
          leave it, None to honor whatever the node already carries.
        write: when True (default) persist `salience` (+ `salience_tag`) to frontmatter.
        exclude_self_from_surprise: FEAT-2026-06-11 salience-coverage P2 — when True,
          the surprise term excludes the node's OWN embedding row from the index
          (leave-one-out) so an already-indexed node does not self-match to ~0. The
          BACKFILL sets this; every other caller leaves it False so the surprise term is
          byte-identical to today (a fresh at-write node is not yet in the index, so it
          can never self-match anyway). DEFAULT False = legacy behavior.

    Returns the normalized [0,1] salience score.
    """
    fm_bundle = _node_frontmatter(memory_dir, node)
    fm = fm_bundle[0] if fm_bundle else None

    # Resolve the content to score surprise against (default: the node's own text).
    if content is None and fm_bundle is not None:
        try:
            from samia.core import vector as _vi
            fname = node if node.endswith(".md") else f"{node}.md"
            _title, content = _vi._load_node_text(memory_dir / "nodes" / fname)
        except Exception:
            content = ""
    if content is None:
        content = ""

    surprise = _salience_surprise(
        memory_dir, content,
        exclude_node=node if exclude_self_from_surprise else None)
    contradiction = _salience_contradiction(memory_dir, node)
    repetition = _salience_repetition(memory_dir, node, fm)

    # Aggregate — What: weight the three composite signals into a clamped [0,1] score,
    #     then let an explicit operator/agent tag clamp it HIGH regardless of the composite.
    composite = (SALIENCE_W_SURPRISE * surprise
                 + SALIENCE_W_CONTRADICTION * contradiction
                 + SALIENCE_W_REPETITION * repetition)
    composite = float(min(1.0, max(0.0, composite)))

    # Explicit-tag override: an explicit tag (arg or pre-existing frontmatter) clamps
    # salience HIGH. This is the deliberate "this matters" high-priority override.
    tagged = bool(explicit_tag)
    if explicit_tag is None and fm is not None:
        tagged = bool(fm.get("salience_tag", False))
    salience = max(composite, SALIENCE_TAG_VALUE) if tagged else composite
    salience = float(round(min(1.0, max(0.0, salience)), 4))
    # Aggregate — Why: salience must not be reducible to any single signal, and the human
    #     "this matters" tag has to win over a low composite — so the tag is a floor, not a term.

    if write and fm_bundle is not None:
        fm, order, body = fm_bundle
        if "salience" not in fm:
            order.append("salience")
        fm["salience"] = salience
        if tagged and not fm.get("salience_tag"):
            if "salience_tag" not in fm:
                order.append("salience_tag")
            fm["salience_tag"] = True
        try:
            from samia.core import frontmatter as _fm
            fname = node if node.endswith(".md") else f"{node}.md"
            _fm.write_node(memory_dir / "nodes" / fname, fm, order, body)
        except Exception:
            pass  # fail-soft: a write failure must not crash the capture/touch path

    # FEAT-2026-06-11 temporal-recall P4 (§6.2 + §16.2 Q2): the STC capture TRIGGER.
    # compute_salience is the write-time salience source; it is the natural place to
    # evaluate the strong-anchor trigger. When the persisted salience clears the strong
    # bar, fire capture_event — it stamps a decaying stc_capture_score onto temporally-
    # adjacent WEAK nodes in the anchor's EPISODE_SEQ-relative window (N before / M after,
    # wall-clock-capped; cosine + 1/chain/hour guards). capture_event is INERT under the
    # master flag off (it checks temporal_weight_enabled and writes NOTHING), so flag-off
    # writes touch no frontmatter and the decay/promotion/recall paths stay byte-identical.
    # Gate the call on the strong bar here too so a sub-bar write pays nothing. Fail-soft +
    # lazy import to dodge the bio<->temporal_recall_stc cycle; any error is swallowed so
    # the salience/capture/touch path is never broken.
    try:
        from samia.core import temporal_recall_stc as _stc
        if salience >= _stc.STC_STRONG_THRESHOLD:
            _stc.capture_event(memory_dir, node)
    except Exception:
        pass
    return salience


def salience_merge_guard(memory_dir, node: str,
                         threshold: float = SALIENCE_MERGE_GUARD_DEFAULT,
                         is_duplicate: bool = False) -> bool:
    """Read-only merge/supersede guard predicate (D6 effect iii — DEFINED, not consumed).

    What: return True when the node's `salience` frontmatter is >= `threshold` AND the
      node is NOT a true duplicate (is_duplicate False). A consult-only predicate the
      contradiction detector + merge consumer call BEFORE acting: when it fires, the
      consumer must NOT auto-supersede/merge the high-salience distinct memory — it
      surfaces it for review instead. Pure read; mutates nothing, applies no effect.
    Why:  D6 — the guard is DEFINED in P2 and CONSUMED downstream (P3-contradiction +
      the merge consumer). It protects a distinct important memory from being absorbed
      by a later, more-frequent, less-important one. It does NOT change the cosine dedup
      gate's role: a TRUE duplicate (is_duplicate True) is still deduped regardless of
      salience, so the guard returns False for it.
    """
    if is_duplicate:
        return False
    fm_bundle = _node_frontmatter(memory_dir, node)
    if fm_bundle is None:
        return False
    try:
        sal = float(fm_bundle[0].get("salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return sal >= float(threshold)


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.salience
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): the salience SOURCE arm carved from the samia.bio monolith
# Layer:      core (pure library, no daemon dependency)
# Role:       the salience SOURCE arm — the shared frontmatter reader, the three
#             composite signals (surprise / contradiction-involvement / saturating
#             repetition), compute_salience (aggregate + explicit-tag override +
#             optional persist + the STC capture trigger), and the read-only
#             salience_merge_guard predicate.
# Stability:  stable — the SOURCE + storage + explicit-tag + guard ONLY; the salience
#             EFFECTS live in hippocampus / tier / the merge consumer.
# ErrorModel: every signal is fail-soft (a missing index / store / embedder contributes
#             0, never crashes); a frontmatter write failure is swallowed; the STC
#             capture trigger is fail-soft + INERT under the master flag off.
# Depends:    .config (SALIENCE_* / np); .hebbian (_load_edge_weights — Tier-0 degree);
#             samia.core.{vector, frontmatter, temporal_recall_stc} + samia.runtime.
#             contradiction (lazy, function-local).
# Exposes:    compute_salience, salience_merge_guard (public); _node_frontmatter,
#             _salience_surprise, _salience_contradiction, _salience_repetition
#             (private, re-exported for tests/importers/parity).
# Note:       compute_salience is itself a mock.patch.object(bio, ...) target (integrity
#             tests rebind it) but has NO internal caller in bio — re-export alone
#             suffices; no facade-reach seam is needed for it.
# Lines:      332
# Updated:    2026-06-14
# Status:     active
# --------------------------------------------------------------------------
