"""samia.core.mcp_server.write — the write / capture / forget / supersession arm.

Layer 1 (Owns / Depends):
    Owns:    the mutation-side tool logic — the genuine write seam
             (memory_write_node) and its capture hook (_register_ring_and_salience:
             ring pointer + salience field), the explicit salience override
             (memory_tag_salient), the fact-extract surface (memory_extract_facts),
             the ONLINE auto-supersede write seam (_online_supersede + its helpers
             _node_subject / _salience_guards_supersede), and the
             forget/supersession-resolution surface (memory_forget_node,
             memory_supersession_candidates, memory_confirm_supersession,
             memory_dismiss_supersession, memory_restore_node).
    Depends: .config (_nodes_dir, _dt, Any/Path). Lazy per-call: samia.core.
             {fact_extractor,temporal_substrate,integrity,bio,hippocampus,frontmatter,
             temporal,ia} and samia.runtime.contradiction — all function-local to
             avoid the import cycle (ia/bio import back through the package surface).

Layer 2 (What / Why):
    What: every write/mutation tool's underlying logic. memory_write_node writes the
          node + frontmatter (incl. the optional temporal substrate fields), captures
          the pristine integrity anchor, registers the ring pointer + salience, and
          runs the online auto-supersede seam. The supersession-resolution tools are
          the operator's confirm/dismiss/restore surface over the RESTORABLE forget
          path (every retire archives byte-exact first).
    Why:  the write/forget mutation path is a single cohesive seam, kept out of the
          hot read path (search.py). Every auto-mutation (capture/online-supersede)
          is GATED + fail-soft so a fresh write never breaks, and every retire is
          reversible (restore_node + self-healing).

PATCH SEAMS (exemplar rule — TWO seams here):
  (1) memory_write_node calls _register_ring_and_salience AND _online_supersede, both of
      which are mock.patch.object(mcp_server, ...) targets (test_integrity_p2). So
      memory_write_node reaches BOTH through the package facade
      (`from samia.core import mcp_server as _pkg; _pkg._register_ring_and_salience(...)`),
      so a package-level patch rebinds the attribute the caller actually reads.
  (2) _online_supersede calls _node_subject, which is a mock.patch.object(mcp_server,
      "_node_subject", ...) target (test_merge_consumer_p3) while _online_supersede is
      itself called directly. So _online_supersede reaches _node_subject through the
      package facade for the same reason.
"""

from __future__ import annotations

from .config import (
    Any,
    Path,
    _dt,
    _nodes_dir,
)


def _register_ring_and_salience(memory_dir: Path, node: str, content: str,
                                salience_tag: bool) -> dict[str, Any]:
    """FEAT-2026-06-07 Tier-1 P2 — capture hook: ring POINTER + salience field.

    What: on a fresh write, (1) register a ring POINTER into the just-written main node
      (RingStore.add — a cheap reference + a salience flag, NOT a copy), and (2) compute
      and persist the node's [0,1] `salience` frontmatter via bio.compute_salience
      (surprise + contradiction-involvement + repetition; an explicit salience_tag
      clamps it high). Returns a small {ring, salience} summary.
    Why:  D6 / P2 capture path (Q1a) — captures land at ring-RAG as POINTERS carrying a
      salience signal; the held engram copy is EARNED later at materialization (P3, not
      here). Fail-open: any error never blocks or corrupts the write.
    """
    out: dict[str, Any] = {}
    try:
        from .. import hippocampus as _hip
        ring = _hip.RingStore(memory_dir).add(node, target_tier="main",
                                              salience_flag=salience_tag)
        out["ring"] = {"ptr": ring.get("ptr"),
                       "salience_flag": ring.get("salience_flag")}
    except Exception as e:  # fail-open: ring registration must never break the write.
        out["ring_error"] = str(e)
    try:
        from .. import bio as _bio
        out["salience"] = _bio.compute_salience(
            memory_dir, node, content=content,
            explicit_tag=True if salience_tag else None, write=True)
    except Exception as e:  # fail-open: salience write must never break the write.
        out["salience_error"] = str(e)
    return out


def memory_write_node(
    memory_dir: Path,
    name: str,
    title: str,
    description: str,
    body: str,
    type_: str = "project",
    chains: list[str] | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    extract: bool = False,
    extractor_backend: str = "auto",
    runtime: str = "main",
    salience_tag: bool = False,
) -> dict[str, Any]:
    from .. import fact_extractor as _fx
    today = _dt.date.today().isoformat()
    nodes_dir = _nodes_dir(memory_dir)

    if extract:
        atoms = _fx.extract_atoms(body, backend=extractor_backend, chains_hint=chains)
        if not atoms:
            return {"error": "extractor produced no atoms", "extracted": 0}
        # Force user-supplied valid_from/valid_to / chains onto atoms when given.
        for a in atoms:
            if valid_from and not a.get("valid_from"):
                a["valid_from"] = valid_from
            if valid_to is not None:
                a["valid_to"] = valid_to
            if chains and not a.get("chains"):
                a["chains"] = list(chains)
        names = _fx.write_atoms_as_nodes(memory_dir, atoms, prefix=name, runtime=runtime)
        return {"extracted": len(names), "nodes": names, "backend": extractor_backend}

    p = nodes_dir / name
    if not p.suffix:
        p = p.with_suffix(".md")
    chains_str = "[" + ", ".join(chains or []) + "]"
    # FEAT-2026-06-11 temporal-recall P0 — write-time substrate (§3).
    # What: mint written_at (Unix float anchor, time.time() at body commit) + one
    #   corpus-global monotone episode_seq, and append them AFTER last_access.
    # Why: the temporal-recall modulators (SITH/distinctiveness need a sub-day anchor;
    #   directed-SR needs a strict total order) read these. ADDITIVE-OPTIONAL: every
    #   existing field is untouched and nothing reads the new fields yet, so retrieval
    #   is unchanged until a later phase + flag enable it. Fail-soft: a substrate hiccup
    #   must never break the write — fall back to omitting the two lines.
    from .. import temporal_substrate as _ts
    try:
        _sub = _ts.write_time_fields(memory_dir)
    except Exception:
        _sub = None
    fm = [
        f"name: {title}",
        f"description: {description}",
        f"type: {type_}",
        f"chains: {chains_str}",
        f"valid_from: {valid_from or today}",
        f"valid_to: {valid_to or 'null'}",
        f"last_access: {today}",
        "access_count: 0",
        "relevance: 0.5",
        "tier: warm",
        f"runtime: {runtime}",
    ]
    if _sub is not None:
        fm.append(f"written_at: {_sub['written_at']!r}")
        fm.append(f"episode_seq: {_sub['episode_seq']}")
    fm_text = "\n".join(fm)
    p.write_text(f"---\n{fm_text}\n---\n{body}\n", encoding="utf-8")
    out: dict[str, Any] = {"written": p.name, "valid_from": valid_from or today,
                           "valid_to": valid_to}
    # FEAT-2026-06-07 granular-recall-repaired-decay P2 — ANCHOR CAPTURE ON WRITE.
    # What: capture/refresh the PRISTINE recovery anchor from the just-written body (the
    #   genuine, pre-erosion content). A fresh node gains its anchor here; a genuine
    #   re-write refreshes it to the new pristine body.
    # Why: P1 noted the second decay axis only engages once a node HAS an anchor and did
    #   NOT auto-capture; this is that capture point. CRITICAL SAFETY: `body` here is the
    #   pristine just-written body — the anchor is NEVER captured from the eroded served
    #   body (erode/integrity_decay_pass leave the anchor alone), so repair stays faithful.
    #   Fail-soft + additive: an anchor failure never breaks the write.
    try:
        from .. import integrity as _integrity
        out["anchor"] = _integrity.capture_on_write(memory_dir, p.name,
                                                     {"name": title}, body)
    except Exception as e:
        out["anchor_error"] = str(e)
    # FEAT-2026-06-07 Tier-1 P2 — capture lands at the RING as a POINTER (not a copy)
    # carrying a salience signal. Register the pointer + compute/write the salience
    # frontmatter field (explicit salience_tag clamps it high). Fail-open / additive.
    # PATCH SEAM (1): reach _register_ring_and_salience through the package facade so a
    # `mock.patch.object(mcp_server, "_register_ring_and_salience", ...)` patch (test_
    # integrity_p2) applies to this call too (never the pre-patch module-local function).
    from samia.core import mcp_server as _pkg
    cap = _pkg._register_ring_and_salience(memory_dir, p.name,
                                           f"{title}. {description}\n\n{body}",
                                           salience_tag)
    if cap:
        out["capture"] = cap
    # FEAT-2026-06-07 P3b — ONLINE auto-supersede on the write seam.
    # What: after the write lands, check the bounded active-locus for an exact
    #   supersession of a co-activation neighbor / hot node and auto-retire it
    #   (restorably); record weaker hits for the passive judge.
    # Why: Q4 OVERRIDE — close the negative-consolidation loop at write time on the
    #   obvious case. GATED behind ASTHENOS_CONTRADICTION_ENABLED (default OFF) and
    #   fully fail-soft, so it is inert + harmless until the operator enables it.
    # PATCH SEAM (1): reach _online_supersede through the package facade so a
    # `mock.patch.object(mcp_server, "_online_supersede", ...)` patch (test_integrity_p2)
    # applies to this call too (never the pre-patch module-local function).
    sup = _pkg._online_supersede(memory_dir, p.name,
                                 f"{title}. {description}\n\n{body}", valid_to)
    if sup.get("superseded") or sup.get("recorded"):
        out["supersession"] = sup
    return out


def memory_tag_salient(memory_dir: Path, node: str,
                       value: bool = True) -> dict[str, Any]:
    """FEAT-2026-06-07 Tier-1 P2 (D6) — the EXPLICIT operator/agent salience override.

    What: set (or clear) the explicit salience tag on a node and recompute its
      `salience` frontmatter. value=True is the deliberate "this matters" override that
      clamps salience HIGH (bio.SALIENCE_TAG_VALUE) regardless of the composite signals;
      value=False clears the override so salience falls back to the composite.
    Why:  D6 Q8a — the explicit-tag path, exposed as an MCP/CLI surface (the operator/
      agent override is the only sticky, operator-visible salience component, Risk 9).
      Returns {node, salience, salience_tag}; fail-soft on a missing node.
    """
    from .. import bio as _bio
    fname = node if node.endswith(".md") else f"{node}.md"
    if not (_nodes_dir(memory_dir) / fname).exists():
        return {"error": f"node not found: {fname}"}
    sal = _bio.compute_salience(memory_dir, fname, explicit_tag=bool(value),
                                write=True)
    return {"node": fname, "salience": sal, "salience_tag": bool(value)}


def memory_extract_facts(
    memory_dir: Path,
    text: str,
    backend: str = "auto",
    chains_hint: list[str] | None = None,
) -> list[dict[str, Any]]:
    from .. import fact_extractor as _fx
    return _fx.extract_atoms(text, backend=backend, chains_hint=chains_hint)


# ---------------------------------------------------------------------------
# Forgetting / negative consolidation -- FEAT-2026-06-07 P0
# ---------------------------------------------------------------------------


def _node_subject(memory_dir: Path, node: str) -> str:
    """The subject key of a node = its frontmatter `name`, lower/stripped.

    What: read nodes/<node>.md and return name; the file stem if no name field.
    Why:  the ONLINE exact-supersession test is "same subject key" — a near-
          identical claim ABOUT THE SAME SUBJECT — so we compare frontmatter
          names, not bodies (two unrelated nodes can be cosine-close).
    """
    fname = node if node.endswith(".md") else f"{node}.md"
    p = _nodes_dir(memory_dir) / fname
    if not p.exists():
        return ""
    try:
        from .. import frontmatter as _fm
        parsed, _ = _fm.parse(p.read_text(encoding="utf-8"))
        if parsed is not None:
            return str(parsed[0].get("name", "")).strip().lower()
    except Exception:
        pass
    return fname[:-3].strip().lower()


def _salience_guards_supersede(memory_dir: Path, old_fname: str) -> bool:
    """True iff the P3 salience guard protects old_fname from online auto-supersede.

    What: consult bio.salience_merge_guard on the old node with is_duplicate=False;
          True when it is a DISTINCT high-salience memory the guard protects (the
          caller then SURFACES the supersession for operator review instead of
          auto-removing it). False when bio lacks salience_merge_guard (the online
          path ships before Tier-1's salience field lands) or the node is not
          high-salience.
    Why:  D6 effect (iii) / Q5a — the salience merge/supersede guard is CONSUMED by
          the contradiction detector's ONLINE auto-supersede (here) AND the merge
          consumer. Wired behind a hasattr guard so the online path runs fully
          before the salience field exists, activating with no re-sequence once
          Tier-1 Phase 5 lands. Pure read; mutates nothing.
    """
    try:
        from .. import bio as _bio
    except Exception:
        return False
    guard = getattr(_bio, "salience_merge_guard", None)
    if guard is None:
        return False
    try:
        return bool(guard(memory_dir, old_fname, is_duplicate=False))
    except Exception:
        return False


def _online_supersede(memory_dir: Path, new_node: str, text: str,
                      valid_to: str | None) -> dict[str, Any]:
    """FEAT-2026-06-07 P3b — ONLINE auto-supersede on the write path (active-locus).

    What: after a successful write, run find_supersession_candidates scoped to the
          bounded active-set (co-activation neighbors + hot/recent, via
          bio.active_set). For the EXACT case (cosine >= the auto bar AND same
          subject key as the new write) AUTO-supersede NOW via the RESTORABLE forget
          path (set valid_to on the old node, then forget_node(reason="supersede")
          which full-archives it) — UNLESS the P3 SALIENCE GUARD fires (the old node
          is a DISTINCT high-salience memory), in which case the supersession is
          SURFACED for operator review (status="surfaced-salience") instead of auto-
          removed (D6 effect iii / Q5a). WEAKER hits (0.75 <= cosine < auto bar) are
          recorded to the unified candidate store with mode="online" for the later
          passive LLM judge — never auto-deleted online. No LLM call here.
    Why:  Q4 OPERATOR OVERRIDE + the Q4-granularity decision. Auto-supersede is made
          safe by reversibility (restore_node + self-healing). The active-set keeps
          the write path cheap and bounded; the no-judge online path stays
          conservative (only the obvious exact case acts; the rest waits for P3c).
          R8: GATED behind ASTHENOS_CONTRADICTION_ENABLED (default OFF) → inert
          until the operator enables it + restarts the daemon. Fail-soft: a
          detector error never blocks or corrupts the write.
    """
    result: dict[str, Any] = {"superseded": [], "recorded": [], "checked": 0}
    try:
        from samia.runtime import contradiction as _con
    except ImportError:
        return result
    # R8 produce-only gate: the entire online behavior is inert unless enabled.
    if not _con.is_enabled():
        result["enabled"] = False
        return result
    result["enabled"] = True

    new_fname = new_node if new_node.endswith(".md") else f"{new_node}.md"
    try:
        from .. import bio as _bio
        scope = _bio.active_set(memory_dir, [new_fname])
    except Exception as e:  # fail-soft: no locus → nothing to do.
        result["error"] = f"active_set: {e}"
        return result
    if not scope:
        return result

    try:
        cands = _con.find_supersession_candidates(
            text, scope_nodes=scope, memory_dir=memory_dir)
    except Exception as e:  # fail-soft: detector error must not break the write.
        result["error"] = f"detector: {e}"
        return result
    result["checked"] = len(cands)
    if not cands:
        return result

    # PATCH SEAM (2): reach _node_subject through the package facade so a
    # `mock.patch.object(mcp_server, "_node_subject", ...)` patch (test_merge_consumer_p3)
    # applies here too (never the pre-patch module-local function).
    from samia.core import mcp_server as _pkg
    new_subject = _pkg._node_subject(memory_dir, new_fname)
    auto_bar = _con.auto_cosine_threshold()
    today = _dt.date.today().isoformat()
    for c in cands:
        old_id = str(c["node_id"])
        old_fname = old_id if old_id.endswith(".md") else f"{old_id}.md"
        if old_fname == new_fname:
            continue
        cosine = float(c.get("score", 0.0))
        same_subject = bool(new_subject) and (
            _pkg._node_subject(memory_dir, old_fname) == new_subject)
        if cosine >= auto_bar and same_subject:
            # P3 SALIENCE GUARD (D6 effect iii / Q5a): do NOT auto-supersede a
            # DISTINCT high-salience old node — surface it for operator review
            # instead (record a guarded candidate, never auto-remove). A
            # contradicting/superseding claim pair is distinct, so is_duplicate
            # stays False; an exact duplicate is not the guard's target.
            if _salience_guards_supersede(memory_dir, old_fname):
                _con.record_supersession_candidate(
                    memory_dir, old_fname, new_fname, cosine=cosine,
                    jaccard=c.get("jaccard"), mode="online",
                    status="surfaced-salience")
                result.setdefault("guarded", []).append(
                    {"old_id": old_fname, "cosine": cosine})
                continue
            # EXACT case → auto-supersede now via the RESTORABLE forget path.
            from .. import temporal as _temporal
            vt = valid_to or today
            try:
                if (_nodes_dir(memory_dir) / old_fname).exists():
                    _temporal.set_valid(memory_dir, old_fname, None, vt)
            except Exception:
                pass  # best-effort close; the archive below preserves the body.
            from .. import ia as _ia
            cascade = _ia.forget_node(memory_dir, old_fname, reason="supersede",
                                      superseded_by=new_fname)
            _con.record_supersession_candidate(
                memory_dir, old_fname, new_fname, cosine=cosine,
                jaccard=c.get("jaccard"), mode="online", status="confirmed")
            _con.mark_supersession_confirmed(memory_dir, old_fname, new_fname)
            result["superseded"].append(
                {"old_id": old_fname, "cosine": cosine, "valid_to": vt,
                 "cascade": cascade})
        else:
            # WEAKER hit → record for the passive judge; nothing deleted.
            _con.record_supersession_candidate(
                memory_dir, old_fname, new_fname, cosine=cosine,
                jaccard=c.get("jaccard"), mode="online")
            result["recorded"].append({"old_id": old_fname, "cosine": cosine})
    return result


def memory_forget_node(memory_dir: Path, node: str,
                       reason: str = "manual") -> dict[str, Any]:
    """Cross-tier invalidation cascade for a dead/superseded node.

    What: thin wrapper over ia.forget_node -- hard-deletes the node's edges from
          edges.db (all ref_kinds) + edge_weights.json, strips its chain
          membership + hebbian edges, tombstones its vector entry, and appends a
          forgotten-log entry. The node FILE is expected already gone.
    Why:  exposes the FEAT-2026-06-07 P0 cascade primitive (built in ia.py and
          auto-wired into freeze/merge) as an explicit MCP/CLI surface for the
          confirm step of a contradiction supersession and for ad-hoc cleanup.
          Idempotent and fail-soft per store.
    """
    from .. import ia as _ia
    return _ia.forget_node(memory_dir, node, reason=reason)


def memory_supersession_candidates(memory_dir: Path) -> dict[str, Any]:
    """List un-resolved supersession candidates from the UNIFIED store (R2).

    What: returns the {old_id, new_id, cosine, jaccard, mode, ts, ...} candidates
          recorded by the online write seam (weaker hits) and the passive judge,
          reading the single canonical store
          (contradiction.list_supersession_candidates).
    Why:  R2 — one owner, one schema. The online exact case auto-supersedes
          (restorably); these remaining candidates are the weaker hits awaiting
          the passive LLM judge / operator review. Nothing is deleted until acted.
    """
    try:
        from samia.runtime import contradiction as _con
        return {"candidates": _con.list_supersession_candidates(memory_dir)}
    except Exception as e:  # fail-open: never raise into the MCP loop.
        return {"candidates": [], "error": str(e)}


def memory_confirm_supersession(memory_dir: Path, old_id: str,
                                 valid_to: str | None = None,
                                 new_id: str | None = None) -> dict[str, Any]:
    """Confirm a supersession → RESTORABLE retire of the old node (R3).

    What: sets valid_to on the OLD node (provenance-preserving close), then fires
          the RESTORABLE forget path forget_node(reason="supersede") — which (R1)
          full-archives the node before the cascade so restore_node can un-forget
          it byte-exact — and marks the matching candidate(s) confirmed in the
          unified store.
    Why:  Q4 OPERATOR OVERRIDE — auto-supersede made safe by reversibility. A
          confirmed supersession is now restorable (it was NOT before R1, because
          reason="supersede" did not archive). The node FILE is closed via valid_to
          first (temporal provenance), then archived + its ghost edges purged.
    """
    from .. import temporal as _temporal
    today = _dt.date.today().isoformat()
    vt = valid_to or today
    fname = old_id if old_id.endswith(".md") else f"{old_id}.md"
    p = _nodes_dir(memory_dir) / fname
    result: dict[str, Any] = {"old_id": fname, "valid_to": vt}

    # Step 1: close the old node's validity window (provenance-preserving) BEFORE
    # the archiving forget — so the archived frontmatter carries the valid_to.
    if p.exists():
        try:
            _temporal.set_valid(memory_dir, fname, None, vt)
            result["closed"] = True
        except Exception as e:
            result["closed"] = False
            result["close_error"] = str(e)
    else:
        result["closed"] = False
        result["note"] = "node file absent; cascading edge purge only"

    # Step 2: RESTORABLE retire — reason="supersede" full-archives (R1) then cascades.
    from .. import ia as _ia
    result["cascade"] = _ia.forget_node(memory_dir, fname, reason="supersede",
                                        superseded_by=new_id)

    # Step 3: mark the candidate(s) confirmed in the unified store.
    try:
        from samia.runtime import contradiction as _con
        result["candidates_confirmed"] = _con.mark_supersession_confirmed(
            memory_dir, old_id=fname, new_id=new_id)
    except Exception as e:  # fail-open: the cascade already ran.
        result["candidates_confirmed"] = 0
        result["candidate_log_error"] = str(e)

    return result


def memory_dismiss_supersession(memory_dir: Path, old_id: str,
                                new_id: str | None = None) -> dict[str, Any]:
    """Dismiss a supersession candidate (false positive) in the unified store.

    What: marks the matching candidate(s) dismissed; deletes nothing, sets no
          valid_to. If the candidate names an already-auto-superseded node, the
          operator can additionally call memory_restore_node to un-forget it.
    Why:  R2 — the operator's reject path on the single store. A 0.75 cosine smell
          is weak; dismissal records the rejection so it stops surfacing.
    """
    fname = old_id if old_id.endswith(".md") else f"{old_id}.md"
    try:
        from samia.runtime import contradiction as _con
        return {"old_id": fname,
                "dismissed": _con.mark_supersession_dismissed(
                    memory_dir, old_id=fname, new_id=new_id)}
    except Exception as e:
        return {"old_id": fname, "dismissed": 0, "error": str(e)}


def memory_restore_node(memory_dir: Path, node_id: str) -> dict[str, Any]:
    """Un-forget a superseded node from its archive (R4 — over ia.restore_node).

    What: thin wrapper over ia.restore_node — re-creates nodes/<id>.md byte-exact
          from archive/<id>.superseded.json, un-tombstones its vector entry, stamps
          restore_ts, logs a restore event.
    Why:  Q4 OVERRIDE — auto-supersede is acceptable ONLY because it is reversible.
          This is the operator/self-healing un-forget surface for an online
          auto-supersede or a confirmed supersession that turned out wrong.
    """
    from .. import ia as _ia
    return _ia.restore_node(memory_dir, node_id)


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.mcp_server.write
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.mcp_server monolith during
#             modularization (the write & forgetting / supersession sections).
# Layer:      core (pure library, no daemon dependency)
# Role:       the write / capture / forget / supersession arm — memory_write_node + its
#             capture hook, memory_tag_salient, memory_extract_facts, the ONLINE auto-
#             supersede write seam (_online_supersede + _node_subject/_salience_guards_
#             supersede), and the forget/confirm/dismiss/restore supersession surface.
# Stability:  stable — behavior byte-identical to the monolith's write & forgetting
#             sections; only the imports moved behind .config and the two patch-seam
#             helper calls now reach through the package facade.
# ErrorModel: fail-soft on every auto-mutation — anchor capture, ring/salience capture,
#             and the online auto-supersede each swallow their own errors so a fresh write
#             never breaks; the online seam is GATED OFF by default. Every retire archives
#             byte-exact first (RESTORABLE via memory_restore_node).
# Depends:    .config (_nodes_dir, _dt, Any/Path); the package facade (samia.core.
#             mcp_server) for the two patch seams. Lazy per-call: samia.core.{fact_
#             extractor,temporal_substrate,integrity,bio,hippocampus,frontmatter,temporal,
#             ia}, samia.runtime.contradiction.
# Exposes:    memory_write_node, memory_tag_salient, memory_extract_facts,
#             memory_forget_node, memory_supersession_candidates,
#             memory_confirm_supersession, memory_dismiss_supersession,
#             memory_restore_node (public); _register_ring_and_salience/_node_subject/
#             _salience_guards_supersede/_online_supersede (test-reached + patch seams).
# Lines:      545
# Note:       TWO PATCH SEAMS — memory_write_node reaches _register_ring_and_salience +
#             _online_supersede through the facade; _online_supersede reaches _node_subject
#             through the facade (see module docstring).
# --------------------------------------------------------------------------
