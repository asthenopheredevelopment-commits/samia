"""samia.core.hippocampus.ring — the P2 ring POINTER store + ring-RAG + promote-before-evict.

Layer 1 (Owns / Depends):
    Owns:    RingStore (the volatile, capacity/LRU-bounded set of POINTERS into
             main/engram backed by <memory_dir>/hippocampus/ring.jsonl) and
             ring_rag_query (deref the live pointers, cosine them against the query).
             Owns the P3 promote-before-evict path (_is_wanted / _evict_lru materialize
             a still-wanted victim before the LRU drops it) and the salience/STC
             backing-signal readers (_entry_salience / _entry_stc).
    Depends: .config (the ring bounds, _ring_path, _ptr_name, _now_iso, the
             RING_EVICT_WANT_SALIENCE bar, the re-exported vector backend _vi);
             .engram (EngramStore — resolve() derefs an engram-backed pointer and
             _evict_lru materializes a wanted victim through it).  Lazily:
             samia.core.bio (_node_frontmatter for the backing salience),
             samia.core.temporal_recall_stc (current_capture_score) — function-local.

Layer 2 (What / Why):
    What: the volatile working set (hours). A ring entry is a POINTER + minimal
          metadata (last_access, access_count, salience_flag, genuine_hits), NOT a
          held copy — the OPPOSITE of the engram invariant. resolve() derefs FRESH so
          ring-RAG always reflects the CURRENT backing; a dangling pointer (backing
          gone) resolves to nothing. promote-before-evict materializes a wanted victim
          (recent genuine hits OR salience >= bar) to an engram copy before the LRU
          can drop it, so no wanted trace is lost.
    Why:  carved out of the 1339-line monolith as the ring responsibility. It sits
          above .engram in the package DAG (resolve/_evict_lru reach EngramStore) and
          below the promotion lattice + the inject assembler (both read RingStore).
"""

from __future__ import annotations

import datetime as _dt
import json

from .config import (
    RING_CAPACITY_DEFAULT,
    RING_EVICT_WANT_SALIENCE,
    RING_TTL_HOURS_DEFAULT,
    Path,
    _hippocampus_dir,
    _now_iso,
    _ptr_name,
    _ring_path,
    _vi,
)
from .engram import EngramStore


class RingStore:
    """The Tier-1 ring store — a volatile, capacity/LRU-bounded set of POINTERS.

    What: a small append-backed JSONL of POINTER entries (not content). Each entry is
      {ptr, target_tier(main|engram), ts, last_access, access_count, salience_flag} —
      a lightweight reference into main/engram, the OPPOSITE of the engram's held copy.
      add() registers/refreshes a pointer (most-recently-accessed); when the live set
      exceeds the capacity the least-recently-accessed pointer is LRU-evicted. resolve()
      dereferences a pointer to its backing content and is dangling-SAFE: a pointer
      whose backing (main node / engram copy) is gone resolves to None.
    Why:  this is the volatile working set (D1, Q1a) — captures land here cheaply as a
      pointer + a salience flag; the held copy is EARNED later at materialization (P3,
      not here). Being pointers (deref-at-read) is what makes ring-RAG reflect the
      CURRENT backing — the pointer invariant that distinguishes it from the engram.
    """

    def __init__(self, memory_dir: Path,
                 capacity: int = RING_CAPACITY_DEFAULT,
                 ttl_hours: float = RING_TTL_HOURS_DEFAULT):
        self.memory_dir = Path(memory_dir)
        self.capacity = int(capacity)
        self.ttl_hours = float(ttl_hours)

    # -- persistence (load = compact: last-write-wins per ptr) --------------

    def _load(self) -> dict[str, dict]:
        """Return the live pointer map {ptr_name: entry}, last-write-wins.

        What: read the append-only ring.jsonl, keeping the LAST record per pointer
          (so a re-add/touch supersedes the earlier line without an in-place rewrite).
        Why:  append + load-compaction keeps add/touch O(1) writes; the map is the
          authoritative live set the rest of the API operates on.
        """
        p = _ring_path(self.memory_dir)
        if not p.exists():
            return {}
        live: dict[str, dict] = {}
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = rec.get("ptr")
                    if not name:
                        continue
                    if rec.get("__evicted__"):
                        live.pop(name, None)
                        continue
                    live[name] = rec
        except OSError:
            return {}
        return live

    def _rewrite(self, live: dict[str, dict]) -> None:
        """Compact the ring.jsonl to exactly the live entries (one line each)."""
        _hippocampus_dir(self.memory_dir).mkdir(parents=True, exist_ok=True)
        p = _ring_path(self.memory_dir)
        tmp = p.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in live.values():
                f.write(json.dumps(rec) + "\n")
        tmp.replace(p)

    # -- mutate (LRU) -------------------------------------------------------

    def add(self, ptr: str, target_tier: str = "main",
            salience_flag: bool = False) -> dict:
        """Register (or refresh) a POINTER into main/engram; LRU-bound the ring.

        What: write a pointer entry (NO content) for `ptr`, marking it the most-
          recently-accessed. Re-adding an existing pointer refreshes its last_access
          and bumps access_count (it does not duplicate). After the add, if the live
          set exceeds capacity, the least-recently-accessed pointer is LRU-evicted.
        Why:  capture (Q1a) registers a cheap pointer + a salience flag here, not a
          copy. The held copy is earned at materialization later (P3). promote-before-
          evict (materialize a salient/promotable pointer before LRU drops it) is P3 —
          P2 only enforces the capacity bound.

        Returns the pointer entry dict (also persisted).
        """
        name = _ptr_name(ptr)
        now = _now_iso()
        live = self._load()
        existing = live.get(name)
        entry = {
            "ptr": name,
            "target_tier": target_tier,
            "ts": (existing or {}).get("ts", now),
            "last_access": now,
            "lru_seq": self._next_seq(live),
            "access_count": int((existing or {}).get("access_count", 0)) + 1,
            # genuine_hits — accrued GENUINE ring-RAG hits (record_genuine_hit, P3),
            # the frequency/recency signal that drives ring->engram promotion. Preserved
            # across re-add (it is the consolidation signal, not the LRU clock).
            "genuine_hits": int((existing or {}).get("genuine_hits", 0)),
            "salience_flag": bool(salience_flag or (existing or {}).get(
                "salience_flag", False)),
        }
        live[name] = entry
        # LRU bound with PROMOTE-BEFORE-EVICT (P3): a victim still WANTED (recent
        # genuine hits OR high salience) is materialized to an engram copy before it is
        # dropped, so no wanted trace is lost. _evict_lru does the over-capacity drop.
        self._evict_lru(live)
        self._rewrite(live)
        return entry

    @staticmethod
    def _next_seq(live: dict[str, dict]) -> int:
        """Monotonic LRU sequence — a precise recency order independent of timestamp
        granularity (second-resolution last_access can tie under rapid add/touch)."""
        return 1 + max((int(r.get("lru_seq", 0)) for r in live.values()),
                       default=0)

    @staticmethod
    def _lru_key(rec: dict):
        """LRU sort key: (last_access, lru_seq) so a re-touch within the same second is
        still ordered most-recent-last via the monotonic sequence."""
        return (rec.get("last_access", ""), int(rec.get("lru_seq", 0)))

    def touch(self, ptr: str) -> dict | None:
        """Mark an existing pointer accessed (LRU bump). None if not in the ring."""
        name = _ptr_name(ptr)
        live = self._load()
        if name not in live:
            return None
        entry = live[name]
        entry["last_access"] = _now_iso()
        entry["lru_seq"] = self._next_seq(live)
        entry["access_count"] = int(entry.get("access_count", 0)) + 1
        live[name] = entry
        self._rewrite(live)
        return entry

    def record_genuine_hit(self, ptr: str) -> dict | None:
        """Record one GENUINE ring-RAG hit on a pointer (the ring->engram signal, P3).

        What: bump the pointer's `genuine_hits` counter (and refresh its LRU recency,
          since a genuine recall is also an access). Returns the updated entry, or None
          if the pointer is not in the ring.
        Why:  D3/Q3a — frequency/recency is the ring->engram promotion driver, and the
          proposal's signal is GENUINE ring-RAG hits (effortful recall), distinct from
          a bare add/touch. promote_ring_step reads `genuine_hits` against
          RING_PROMOTE_HITS. Called from the recall hook (and later a REM/idle pass);
          P3 only provides the function, not a scheduler.
        """
        name = _ptr_name(ptr)
        live = self._load()
        if name not in live:
            return None
        entry = live[name]
        entry["genuine_hits"] = int(entry.get("genuine_hits", 0)) + 1
        entry["last_access"] = _now_iso()
        entry["lru_seq"] = self._next_seq(live)
        entry["access_count"] = int(entry.get("access_count", 0)) + 1
        live[name] = entry
        self._rewrite(live)
        return entry

    # -- promote-before-evict (P3) ------------------------------------------

    def _entry_salience(self, entry: dict) -> float:
        """Salience of a ring pointer's backing node (D6 — the affective signal).

        What: read the `salience` frontmatter the P2 source (bio.compute_salience)
          writes on the backing main node; an engram-backed pointer falls back to 0.0
          (the engram copy carries no node frontmatter). The `salience_flag` set at
          capture is an OR fast-path: a flagged pointer is treated as wanted even if
          the numeric field is absent.
        Why:  D3/D6 — promote-before-evict and the ring->engram one-shot shortcut both
          gate on max(access_signal, salience); this reads that salience cheaply and
          fail-soft (a missing field -> 0.0, never a crash).
        """
        name = entry.get("ptr", "")
        tier = entry.get("target_tier", "main")
        if tier == "engram" or name.startswith("engram_"):
            return 1.0 if entry.get("salience_flag") else 0.0
        try:
            from .. import bio as _bio
            fm_bundle = _bio._node_frontmatter(self.memory_dir, name)
            if fm_bundle is not None:
                sal = float(fm_bundle[0].get("salience", 0.0) or 0.0)
            else:
                sal = 0.0
        except Exception:
            sal = 0.0
        if entry.get("salience_flag"):
            sal = max(sal, 1.0)
        return sal

    def _entry_stc(self, entry: dict) -> float:
        """Attenuated stc_capture_score of a ring pointer's backing node (P4 §6.5 effect 2).

        What: read the (time-attenuated) stc_capture_score the STC capture event
          (temporal_recall_stc.capture_event) stamped on the backing main node — the
          single decaying scalar beside `salience`. An engram-backed pointer (no node
          frontmatter) and a never-captured / legacy node both fall back to 0.0.
        Why: §6.5 effect 2 — the promotion OR-gate gains a third arm so a weak node
          carrying a high capture score is rescued into the engram store by its strong
          neighbour WITHOUT meeting the frequency or its own salience bar. Reads the
          SAME attenuated scalar recall + decay read (current_capture_score), fail-soft
          (a missing field -> 0.0, never a crash). With the temporal flag off no node
          ever carries the field, so this is 0.0 everywhere → gate byte-identical.
        """
        name = entry.get("ptr", "")
        tier = entry.get("target_tier", "main")
        if tier == "engram" or name.startswith("engram_"):
            return 0.0  # engram copies carry no node frontmatter / capture score
        try:
            from .. import temporal_recall_stc as _stc
            return _stc.current_capture_score(self.memory_dir, name)
        except Exception:
            return 0.0

    def _is_wanted(self, entry: dict) -> bool:
        """True when an about-to-be-evicted pointer is still WANTED (promote-before-evict).

        What: a pointer is wanted when it has recent GENUINE ring-RAG hits (>=1) OR its
          backing salience clears RING_EVICT_WANT_SALIENCE.
        Why:  D3 — a wanted trace must be materialized before the LRU can drop it (no
          dangling loss). A pointer with neither signal is an ordinary cold drop.
        """
        if int(entry.get("genuine_hits", 0)) >= 1:
            return True
        return self._entry_salience(entry) >= RING_EVICT_WANT_SALIENCE

    def _evict_lru(self, live: dict[str, dict]) -> list[str]:
        """Drop over-capacity entries, MATERIALIZING any wanted victim first (P3).

        What: when the live set exceeds capacity, select the least-recently-accessed
          victims; for each victim that is still WANTED (_is_wanted), materialize its
          backing to an engram held copy BEFORE removing the pointer; an unwanted victim
          is dropped cold. Mutates `live` in place; returns the list of engram_ids
          materialized by promote-before-evict.
        Why:  D3/Q4a — promote-before-evict is the airtight half of the lattice: an LRU
          drop never loses a salient/recently-recalled trace. Main keeps the canonical;
          materialize is loss-free (copies, never moves). A main pointer whose source is
          already gone (dangling) is simply dropped — there is nothing to promote.
        """
        promoted: list[str] = []
        if len(live) <= self.capacity:
            return promoted
        ordered = sorted(live.values(), key=self._lru_key)
        for victim in ordered[: len(live) - self.capacity]:
            vptr = victim["ptr"]
            if self._is_wanted(victim) and not str(vptr).startswith("engram_"):
                src_path = self.memory_dir / "nodes" / (
                    vptr if vptr.endswith(".md") else f"{vptr}.md")
                if src_path.exists():
                    try:
                        rec = EngramStore(self.memory_dir).materialize(vptr)
                        promoted.append(rec["engram_id"])
                    except Exception:
                        pass  # fail-soft: a copy failure must not block the LRU bound
            live.pop(vptr, None)
        return promoted

    # -- read (dangling-safe deref) -----------------------------------------

    def _is_stale(self, entry: dict) -> bool:
        """True when a pointer is older than the ring TTL (volatile working set)."""
        try:
            ts = _dt.datetime.fromisoformat(entry.get("last_access", ""))
        except (ValueError, TypeError):
            return False  # un-timestamped entries are kept rather than silently dropped
        age_h = (_dt.datetime.now() - ts).total_seconds() / 3600.0
        return age_h > self.ttl_hours

    def resolve(self, entry: dict) -> dict | None:
        """Dereference a pointer to its CURRENT backing content. Dangling-safe.

        What: read the backing node the pointer references — a main node (the live
          nodes/<ptr> file, read fresh) or an engram held copy (EngramStore.get) —
          and return {ptr, target_tier, title, content, body}. Returns None when the
          backing is GONE (the node file was deleted/frozen, or the engram id no longer
          exists) — a dangling pointer resolves to nothing. Also None past the TTL.
        Why:  the pointer invariant: because resolve reads the backing FRESH, ring-RAG
          reflects the current backing (opposite of the engram copy). The dangling
          policy is the clean P2 contract — P3 adds the automatic promote-before-evict;
          P2 just drops a pointer whose backing is gone.
        """
        if self._is_stale(entry):
            return None
        name = entry.get("ptr", "")
        tier = entry.get("target_tier", "main")
        if tier == "engram" or name.startswith("engram_"):
            copy = EngramStore(self.memory_dir).get(name)
            if copy is None:
                return None  # dangling: the engram copy is gone
            return {
                "ptr": name, "target_tier": "engram",
                "title": copy.get("title"),
                "content": copy.get("body", ""),
                "body": copy.get("body", ""),
            }
        # main pointer: read the canonical fresh (so the deref sees current content).
        fname = name if name.endswith(".md") else f"{name}.md"
        src_path = self.memory_dir / "nodes" / fname
        if not src_path.exists():
            return None  # dangling: the main node is gone
        try:
            title, content = _vi._load_node_text(src_path)
            body = src_path.read_text(encoding="utf-8")
        except OSError:
            return None
        return {"ptr": fname, "target_tier": "main", "title": title,
                "content": content, "body": body}

    def entries(self, include_stale: bool = False) -> list[dict]:
        """Live pointer entries, most-recently-accessed first.

        Stale entries (past the TTL) are excluded unless include_stale is True; they
        are NOT auto-evicted here (the volatile sweep is later) — they are simply not
        surfaced, the same dangling-safe posture resolve() takes.
        """
        rows = list(self._load().values())
        if not include_stale:
            rows = [r for r in rows if not self._is_stale(r)]
        rows.sort(key=self._lru_key, reverse=True)
        return rows


def ring_rag_query(memory_dir: Path, text: str, top_k: int = 8) -> list[dict]:
    """Ring-RAG: dereference the live ring pointers, cosine them against the query.

    What: for each live (non-stale) ring pointer, resolve() its CURRENT backing
      content (dangling pointers contribute nothing), embed the surviving backings with
      the shared backend, cosine against the query embedding, and return hits tagged
      `via: ring` carrying {ptr, target_tier, title, score, salience_flag}. Fails open
      (empty list) when the ring is empty or every pointer dangles.
    Why:  this is the volatile fast-tier read arm (the "stutter-continue" half-recall,
      D1). It dereferences at QUERY time, so a hit reflects the current backing — the
      pointer invariant (a ring entry is a reference, not a copy; if the backing changed
      since the pointer was added, the hit reflects the new backing). Reuses the SAME
      cosine primitive on the small dereferenced set; it does not reinvent embeddings.

    Returns a list of {ptr, target_tier, title, score, salience_flag, via}.
    """
    store = RingStore(memory_dir)
    entries = store.entries()
    if not entries:
        return []  # ring empty — fail open, main retrieval is unaffected

    resolved: list[tuple[dict, dict]] = []
    for e in entries:
        backing = store.resolve(e)
        if backing is None:
            continue  # dangling / stale pointer -> contributes nothing
        resolved.append((e, backing))
    if not resolved:
        return []

    q = _vi._embed_batch([text])[0]
    backing_vecs = _vi._embed_batch([b["content"] for _, b in resolved])
    sims = backing_vecs @ q

    scored: list[dict] = []
    for i, (entry, backing) in enumerate(resolved):
        scored.append({
            "ptr": backing["ptr"],
            "target_tier": backing["target_tier"],
            "title": backing.get("title"),
            "score": float(sims[i]),
            "salience_flag": bool(entry.get("salience_flag", False)),
            "via": "ring",
        })
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.hippocampus.ring
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.hippocampus monolith during
#             modularization (the P2 ring responsibility; bodies byte-identical, the
#             shared bounds/paths/ptr-name lifted into .config and EngramStore into
#             .engram).
# Layer:      core (pure library, no daemon dependency)
# Role:       P2 — the ring POINTER store (RingStore: a volatile capacity/LRU-bounded
#             set of pointers into main/engram + dangling-safe resolve + the P3
#             promote-before-evict path) and ring_rag_query (deref the live pointers
#             and cosine them against the query). The volatile working-memory tier.
# Stability:  stable — bodies byte-identical to the monolith; the carve only moved the
#             shared bounds/paths/ptr-name into .config and EngramStore into .engram.
# ErrorModel: fail-open / dangling-safe — a stale or backing-gone pointer resolves to
#             None and contributes nothing; the salience/STC backing readers are
#             fail-soft (a missing field -> 0.0); promote-before-evict swallows a copy
#             failure so a materialize error never blocks the LRU bound.
# Depends:    datetime, json. .config (ring bounds/paths/_ptr_name/_now_iso/_vi),
#             .engram (EngramStore). Lazily: samia.core.bio (_node_frontmatter),
#             samia.core.temporal_recall_stc (current_capture_score).
# Exposes:    RingStore, ring_rag_query.
# Lines:      445
# ─────────────────────────────────────────────
