"""samia.runtime.rem_subscribers.fact_extract — the batch fact-extract subsystem.

Layer 1 (Owns / Depends):
    Owns:    the fact-extract REM subscriber + its persistence subsystem —
             _sub_fact_extract (the FLAG-GATED batch drain that turns queued text
             into full-citizen semantic nodes), _fact_extract_backend (the cached
             BitNet-2B backend the drain rides), _persist_atoms (dedup -> write ->
             provenance -> mini-chain), _fx_stamp_distilled (the TUNE-2026-06-10
             distillation marker on the live source), _fx_provenance_edge (the
             web_store lineage edge), and the _fe_slug / _dt_today helpers.
    Depends: .config (rem_cycle gate/cursor + Any), and — lazily, inside each
             function — the existing ops it rides (fact_extractor.extract_atoms,
             inference.get_backend_for_model, frontmatter.write_node/read_node,
             contradiction.find_contradiction_candidates, web_store, chain).

Layer 2 (What / Why):
    What: extract_atoms is a per-text PRIMITIVE; this wraps it as a batch offline
          op that drains a pending-extraction queue (queue-consumption IS the
          cursor) and PERSISTS each atom as a type:semantic node — deduped
          (cosine >= 0.92), auto-anchored, provenance-edged, and mini-chained to
          its source. Double-gated: the REM gate + the fact_extract_enabled()
          entry gate; flag-off is a byte-identical no-op.
    Why:  FEAT-2026-06-10 P1 — extract_atoms had no caller and the queue had no
          producer, so this arm was perpetually inert. Atoms are ADDITIVE
          date-stamped facts; nothing here deletes/archives/supersedes the source
          (keep+link). It lives in its OWN submodule because the persistence
          subsystem (dedup/write/provenance/mini-chain/distill-stamp) is far larger
          than the other thin op-wrappers.

PATCH SEAM (load-bearing): the targeted tests do
``mock.patch.object(rem_subscribers, "_fact_extract_backend", ...)`` and
``mock.patch.object(rem_subscribers, "_fx_provenance_edge", ...)`` — i.e. they
patch the name on the PACKAGE facade. So _sub_fact_extract / _persist_atoms reach
those two helpers THROUGH the package (a function-local ``from samia.runtime import
rem_subscribers as _pkg``; the package is fully imported by call time), NOT via the
submodule-local binding. That makes the package-level patch take effect, exactly as
in the pre-split monolith where the helpers and their callers shared one namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import rem_cycle


def _sub_fact_extract(mem: Path, batch: int = 20) -> dict[str, Any]:
    """REM subscriber: batch fact extraction (drain → semantic nodes, FLAG-GATED).

    What: gated wrapper that, WHEN ASTHENOS_FACT_EXTRACT_ENABLED=1, drains up to
          ``batch`` records from <mem>/.fact_extract_queue.jsonl (the new producer
          fills it at freeze + merge-abstract), runs fact_extractor.extract_atoms
          on each text via the cached BitNet-2B backend (the judge-rewire seam),
          and PERSISTS each atom as a full-citizen ``type: semantic`` node:
            (a) DEDUP — skip an atom whose cosine vs the existing index is >= 0.92
                (contradiction.find_contradiction_candidates nonempty = dup);
            (b) PERSIST — frontmatter.write_node nodes/sem_<slug>_<hash>.md (auto-
                anchored by capture_on_genuine_write);
            (c) PROVENANCE — a web_store edge atom->source (ref_kind='provenance');
            (d) MINI-CHAIN — upsert chains/fx_<source-stem>.json with the source
                node + its atoms (>= 2 members) so production chainogram (which
                excludes singletons) can load gist alongside episode.
          Drained records are REMOVED (the remaining slice rewritten atomically);
          queue-consumption IS the cursor. FAIL-SOFT: no real backend leaves every
          item in the queue (work_remaining). When the flag is OFF the queue is
          left UNTOUCHED and the body returns {ran:False, reason:'disabled'} —
          byte-identical no-op.
    Why:  FEAT-2026-06-10 P1 / Q2a+Q3a+Q5a. extract_atoms had no caller and the
          queue had no producer, so this arm was perpetually inert. Atoms are
          ADDITIVE date-stamped facts (the lever the benchmark flagged); nothing
          here deletes/archives/supersedes the source (keep+link). Double-gated:
          the REM gate (below) + the fact_extract_enabled() entry gate (the tree's
          preferred single-layer flag-gate, mirroring tier.decay_tick).
    """
    if not rem_cycle.gate_offline_op(Path(mem), "fact_extract"):
        return {"fired": False, "refused": "not_in_rem"}
    import json as _json
    from samia.core import fact_extractor
    # PATCH SEAM: reach the backend helper THROUGH the package facade so the
    # tests' mock.patch.object(rem_subscribers, "_fact_extract_backend", ...) takes
    # effect (the package is fully imported by the time this runs). See module head.
    from samia.runtime import rem_subscribers as _pkg
    # ENTRY GATE (single layer): flag-off leaves the queue UNTOUCHED. No read, no
    # rewrite, no cursor write — a byte-identical no-op (FEAT-2026-06-10 P1, Q4c).
    if not fact_extractor.fact_extract_enabled():
        return {"fired": False, "ran": False, "reason": "disabled",
                "work_remaining": False}

    q = Path(mem) / ".fact_extract_queue.jsonl"
    if not q.exists():
        return {"fired": False, "extracted": 0, "work_remaining": False}
    try:
        lines = [l for l in q.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    except OSError:
        return {"fired": False, "extracted": 0, "work_remaining": False}

    # FAIL-SOFT backend gate: a missing/mock backend leaves the queue intact so a
    # later cycle with a real model still drains it (mirrors the judge posture).
    backend = _pkg._fact_extract_backend()
    if backend is None:
        return {"fired": False, "extracted": 0, "reason": "no_backend",
                "remaining": len(lines), "work_remaining": len(lines) > 0}

    # BUDGET: at most ``batch`` (<= 20) records per call; the rest stay queued.
    take, leave = lines[:batch], lines[batch:]
    extracted = persisted = deduped = 0
    for raw in take:
        try:
            rec = _json.loads(raw)
        except _json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        text = rec.get("text")
        if not text:
            continue
        source = rec.get("source")  # may be absent (old {"text"}-only records)
        # Pass the configured local backend OBJECT (not a string): extract_atoms
        # routes a duck-typed .chat/.complete object through the local model.
        # Passing a string here (the old getattr(backend,'name','auto')) yielded
        # 'auto' -> anthropic-if-key-else-rule — the configured model NEVER
        # generated atoms (FIX-2026-06-10, HIGH). llm_only=True (TUNE-2026-06-10):
        # NEVER persist rule-splitter chunks as semantic nodes — LLM atoms or
        # nothing; a no-atom source stays unstamped and retryable.
        atoms = fact_extractor.extract_atoms(
            text, backend=backend, chains_hint=rec.get("chains"),
            llm_only=True)
        if not atoms:
            # Extraction FAILED (no atoms) — the content is NOT yet semantically
            # covered, so do NOT stamp the source distilled (TUNE-2026-06-10 c).
            continue
        extracted += len(atoms)
        res = _persist_atoms(Path(mem), atoms, source)
        persisted += res["persisted"]
        deduped += res["deduped"]
        # TUNE-2026-06-10 operator decision (c): the episodic trace fades only AFTER
        # the semantic representation forms. This record was processed SUCCESSFULLY
        # (extraction ran AND >= 1 atom persisted OR all atoms were dedup-skipped —
        # both mean the source's content is semantically covered), so stamp the live
        # SOURCE node distilled:true to UNLOCK its (frozen) integrity erosion gate.
        if res["persisted"] >= 1 or res["deduped"] >= 1:
            _fx_stamp_distilled(Path(mem), source)

    # Rewrite the queue with the remainder (queue-consumption IS the cursor).
    remaining = len(leave)
    if leave:
        q.write_text("\n".join(leave) + "\n", encoding="utf-8")
    else:
        try:
            q.unlink()
        except OSError:
            q.write_text("", encoding="utf-8")
    rem_cycle.write_cursor(Path(mem), "fact_extract",
                           {"remaining": remaining, "done": remaining == 0})
    return {"fired": True, "ran": True, "extracted": extracted,
            "persisted": persisted, "deduped": deduped, "remaining": remaining,
            "work_remaining": remaining > 0}


def _fact_extract_backend() -> Any:
    """The cached BitNet-2B backend the drain's extract_atoms rides (fail-soft).

    What: build (once, path-cached) a backend for ASTHENOS_FACT_EXTRACT_MODEL via
          inference.get_backend_for_model — the SAME cached seam the contradiction
          judge uses (contradiction._judge_backend). Returns None when the factory
          is unavailable or the result is a MockBackend (no real model), so the
          drain fail-softly leaves the queue intact.
    Why:  FEAT-2026-06-10 P1 / Q4c — the producer/drain must not load a second copy
          of a model nor block when no model is configured. Mirrors the judge's
          dedicated-cached-small-backend pattern exactly.
    """
    try:
        from samia.runtime import inference as _inf
        from samia.core import fact_extractor
    except Exception:
        return None
    factory = getattr(_inf, "get_backend_for_model", None)
    if factory is None:
        return None
    try:
        backend = factory(fact_extractor.fact_extract_model())
    except Exception:
        return None
    if backend is None or type(backend).__name__ == "MockBackend":
        return None
    return backend


def _persist_atoms(mem: Path, atoms: list[dict], source: Any) -> dict[str, int]:
    """Persist extracted atoms as semantic nodes (dedup → write → prov → chain).

    What: for each atom: (a) DEDUP vs the existing index (cosine >= 0.92 via
          contradiction.find_contradiction_candidates) — skip dups; (b) PERSIST a
          new nodes/sem_<slug>_<shorthash>.md with fm {name, description (atom
          text[:60]), type: semantic, source, extracted_by: fact_extract} through
          frontmatter.write_node (auto-anchored); (c) PROVENANCE a web_store edge
          atom->source (ref_kind='provenance', the merge_consumer P1 pattern);
          (d) MINI-CHAIN upsert chains/fx_<source-stem>.json over the source node +
          its atoms (>= 2 members, else skip — singletons are pointless and
          invisible to production chainogram anyway).
    Why:  FEAT-2026-06-10 P1 / Q2a+Q5a — atoms must be full citizens (indexed,
          contradiction-scoped, chain-loadable), deduped so near-dup spam never
          lands, and lineage-linked back to the source (keep+link, never delete).
    """
    import hashlib
    from samia.core import frontmatter as _fm
    # PATCH SEAM: the provenance edge is reached THROUGH the package facade so the
    # tests' mock.patch.object(rem_subscribers, "_fx_provenance_edge", ...) takes
    # effect (mirrors the pre-split single-namespace call). See module head.
    from samia.runtime import rem_subscribers as _pkg
    try:
        from samia.runtime import contradiction as _con
    except Exception:
        _con = None
    try:
        from samia.core import chain as _chain
    except Exception:
        _chain = None

    nodes_dir = Path(mem) / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    chains_dir = Path(mem) / "chains"

    src_id = str(source) if source else None
    src_fname = None
    if src_id:
        src_fname = src_id if src_id.endswith(".md") else f"{src_id}.md"

    persisted = 0
    deduped = 0
    atom_members: list[dict] = []

    for a in atoms:
        atom_text = (a.get("body") or a.get("description")
                     or a.get("title") or "").strip()
        if not atom_text:
            continue
        # (a) DEDUP — a nonempty candidate set at the 0.92 bar = a near-duplicate
        # already in the index; skip the atom (no low-quality near-dup spam).
        if _con is not None:
            try:
                dup = _con.find_contradiction_candidates(
                    atom_text, memory_dir=Path(mem), threshold=0.92)
            except Exception:
                dup = []
            if dup:
                deduped += 1
                continue

        # (b) PERSIST — a full-citizen semantic node (auto-anchored by write_node).
        slug = _fe_slug(a.get("title") or a.get("description") or atom_text)
        short = hashlib.sha1(atom_text.encode("utf-8")).hexdigest()[:8]
        name = f"sem_{slug}_{short}"
        path = nodes_dir / f"{name}.md"
        if path.exists():
            # Same atom text already persisted under this source slice; treat as
            # a dedup (idempotent re-drain) rather than overwriting.
            deduped += 1
            continue
        # NOTE: do NOT stamp `runtime` here — write_node validates runtime ∈
        # {opencode, main} (the harness-provenance field) and rejects "rem".
        # Readers default a missing runtime to "main", which is correct: these
        # atoms are produced by the main daemon's REM cycle.
        # Lifecycle stamps (BUG-2026-06-11, deep-exam finding #1): every sibling
        # writer stamps last_access at write; omitting it makes tier._days_since
        # read 9999 -> the atom enters the STALE->0 relevance sink and demotes
        # to cold on the FIRST decay tick, where the 2.5x erosion factor +
        # capped 4.0 recency multiplier erode it at 0.20/pass — 10x faster than
        # the protected frozen-distilled nodes. Fresh stamps put new atoms in
        # the warm mean-revert regime like every other genuine write.
        import datetime as _dt
        fm = {
            "name": a.get("title") or atom_text[:60],
            "description": atom_text[:60],
            "type": "semantic",
            "source": src_id or "",
            "extracted_by": "fact_extract",
            "last_access": _dt.date.today().isoformat(),
            "relevance": 0.5,
            "tier": "warm",
            "material_grade": "enriched",
        }
        order = ["name", "description", "type", "source", "extracted_by",
                 "last_access", "relevance", "tier", "material_grade"]
        try:
            _fm.write_node(path, fm, order, atom_text + "\n")
        except Exception:
            continue
        persisted += 1
        atom_members.append({"addr": short, "file": f"nodes/{name}.md",
                             "tier": "warm"})

        # (c) PROVENANCE — edge atom -> source (mirror merge_consumer P1 exactly).
        if src_fname:
            _pkg._fx_provenance_edge(f"{name}.md", src_fname)

    # (d) MINI-CHAIN — source + its atoms, >= 2 members else skip.
    if _chain is not None and atom_members:
        members: list[dict] = []
        if src_fname:
            src_stem = Path(src_fname).stem
            # The source node leads the chain when it still exists (frozen sources
            # are gone — then the chain holds atoms only, per the spec).
            if (nodes_dir / src_fname).exists():
                members.append({"addr": f"src-{src_stem}",
                                "file": f"nodes/{src_fname}",
                                "tier": "warm"})
        members.extend(atom_members)
        if len(members) >= 2:
            chains_dir.mkdir(parents=True, exist_ok=True)
            src_stem = Path(src_fname).stem if src_fname else "merge"
            cname = f"fx_{src_stem}"
            try:
                existing = _chain.load_chain(chains_dir, cname)
            except (SystemExit, FileNotFoundError):
                existing = None
            if isinstance(existing, dict):
                # Upsert: extend members (dedup by addr), keep schema fields.
                seen = {m.get("addr") for m in existing.get("members") or []}
                for m in members:
                    if m.get("addr") not in seen:
                        existing.setdefault("members", []).append(m)
                        seen.add(m.get("addr"))
                data = existing
            else:
                data = {
                    "chain_id": cname,
                    "head_address": members[0]["addr"],
                    "tail_address": members[-1]["addr"],
                    "members": members,
                    "total_relevance": 0.5,
                    "last_traversal": _dt_today(),
                    "compressed": False,
                    "edges": [],
                }
            data["tail_address"] = data["members"][-1]["addr"]
            try:
                _chain.save_chain(chains_dir, cname, data)
            except Exception:
                pass

    return {"persisted": persisted, "deduped": deduped}


def _fx_stamp_distilled(mem: Path, source: Any) -> dict[str, Any]:
    """Stamp distilled:true + distilled_at on a live SOURCE node (TUNE-2026-06-10 c).

    What: after a queue item is processed SUCCESSFULLY (extraction ran + the content
      is semantically covered), if the SOURCE resolves to a LIVE node file
      (mem/nodes/<source>.md exists), rewrite its frontmatter adding distilled:true +
      distilled_at:<iso-utc> via frontmatter.read_node + write_node. The BODY is passed
      back UNCHANGED, so the genuine-write anchor hook (integrity.capture_on_genuine_
      write, fired by write_node since integrity_rewrite defaults False) sees an
      UNCHANGED body vs the anchor and SHA-skips: it returns {"skipped":"unchanged"}
      with NO anchor write and (critically) NO integrity reset (the reset only runs on
      the non-skip branch, after a fresh anchor write). So this frontmatter-only stamp
      never clobbers the pristine anchor and never resets the integrity score —
      verified against integrity.capture_on_genuine_write (the unchanged-body early
      return precedes both the write_anchor and the get_integrity<FULL reset).
    Why: TUNE-2026-06-10 operator decision (c), systems-consolidation gating — the
      distilled marker is the gate that UNLOCKS a frozen node's slow integrity erosion
      (integrity.is_distilled / integrity_decay_pass). The episodic trace fades only
      AFTER the semantic representation forms; this is where "forms" is recorded.

    FAIL-OPEN: a missing source, an absent/unreadable node file, a write rejection
      (AUD61 frozen/archived target_state), or ANY exception is swallowed — a stamp
      failure NEVER breaks the drain (the atoms are already persisted; the marker is
      a best-effort consolidation signal, re-tried on the next successful drain).
    """
    if not source:
        return {"stamped": False, "skipped": "no-source"}
    src = str(source)
    stem = src[:-3] if src.endswith(".md") else src
    node_path = Path(mem) / "nodes" / f"{stem}.md"
    if not node_path.exists():
        # The source is not a live node (e.g. a frozen source whose file is already
        # gone, or a merge-pair pseudo-source) — nothing to stamp, fail-open.
        return {"stamped": False, "skipped": "no-live-source"}
    try:
        from samia.core import frontmatter as _fm
        import datetime as _dt
        fm, order, body = _fm.read_node(node_path)
        if fm.get("distilled") is True:
            # Idempotent: already stamped (an earlier successful drain) — no rewrite.
            return {"stamped": False, "skipped": "already-distilled"}
        if "distilled" not in order:
            order.append("distilled")
        fm["distilled"] = True
        if "distilled_at" not in order:
            order.append("distilled_at")
        fm["distilled_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        # Body UNCHANGED -> capture_on_genuine_write SHA-skips (no anchor clobber, no
        # integrity reset). integrity_rewrite left False so the genuine-write anchor
        # hook runs its unchanged-body skip (it does NOT treat this as an erosion).
        _fm.write_node(node_path, fm, order, body)
        return {"stamped": True, "node": stem}
    except Exception as e:
        # FAIL-OPEN: never let a stamp failure break the drain.
        return {"stamped": False, "skipped": "stamp-error", "error": str(e)}


def _fx_provenance_edge(atom_fname: str, source_fname: str,
                        db_dir: str | None = None) -> None:
    """Lay a web_store edge atom -> source (ref_kind='provenance'), fail-soft.

    What/Why: mirrors merge_consumer._add_provenance_edge exactly — a directed
    edges.db row recording the atom's episodic->semantic lineage; a store error
    never blocks the persist (the node itself is the durable artifact).
    """
    try:
        from samia.core import web_store as _ws
    except Exception:
        return
    try:
        conn = _ws.connect(db_dir)
        try:
            now = _ws._utc_now()
            conn.execute(
                """
                INSERT INTO edges (src_node, dst_node, ref_kind, occurrence_count,
                                   first_seen_at, last_seen_at, weight)
                VALUES (?, ?, ?, 1, ?, ?, 1.0)
                ON CONFLICT(src_node, dst_node, ref_kind) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (atom_fname, source_fname, "provenance", now, now),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return


def _fe_slug(s: str, n: int = 40) -> str:
    """Filesystem-safe slug for a semantic-node filename (mirrors fact_extractor._slug)."""
    import re as _re
    s = _re.sub(r"[^A-Za-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:n] or "atom"


def _dt_today() -> str:
    """Today's ISO date (for the mini-chain's last_traversal stamp)."""
    import datetime as _dt
    return _dt.date.today().isoformat()


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_subscribers.fact_extract
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-10-memory-fact-extract-producer-v01 (P1) +
#             TUNE-2026-06-10 (decision c, distillation-gated frozen erosion).
# Layer:      runtime (library helper, no daemon loop)
# Role:       the batch fact-extract subsystem — the FLAG-GATED REM subscriber that
#             drains <mem>/.fact_extract_queue.jsonl, extracts atoms via the cached
#             BitNet-2B backend, and PERSISTS each as a full-citizen type:semantic
#             node (dedup -> write_node auto-anchored -> web_store provenance edge ->
#             per-source mini-chain), then stamps the live source distilled:true.
# Stability:  stable — the carve preserved the double gate, queue-consumption-as-
#             cursor, the 0.92 dedup bar, the frontmatter field/order, the
#             provenance/mini-chain shape, and the SHA-skip distill stamp
#             byte-identical to the monolith. The ONLY change is the patch seam
#             (below): two helper calls now resolve through the package facade.
# ErrorModel: double-gated (REM + fact_extract_enabled); flag-off / missing queue /
#             no real backend are byte-identical no-ops that leave the queue intact
#             (work_remaining reflects the un-drained backlog). _persist_atoms
#             skips on any per-atom error; _fx_stamp_distilled is FAIL-OPEN (a stamp
#             failure never breaks the drain); _fx_provenance_edge is fail-soft.
# Depends:    pathlib, typing (stdlib). .config (rem_cycle). Lazily: fact_extractor,
#             inference, frontmatter, contradiction, web_store, chain. AND the
#             package facade (samia.runtime.rem_subscribers) — the PATCH SEAM:
#             _fact_extract_backend / _fx_provenance_edge are reached through it so
#             mock.patch.object(rem_subscribers, ...) takes effect (see module head).
# Exposes:    _sub_fact_extract, _fact_extract_backend, _persist_atoms,
#             _fx_stamp_distilled, _fx_provenance_edge, _fe_slug, _dt_today.
# Lines:      479
# ─────────────────────────────────────────────
