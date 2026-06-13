"""temporal_substrate.py -- write-time substrate for the temporal-recall layer (P0).

Layer 1 (Owns / Depends):
    Owns:    The two ADDITIVE-OPTIONAL write-time fields the temporal-recall formula
             (FEAT-2026-06-11-memory-temporal-recall-formula-v01) stands on:
               - written_at  : Unix float (time.time()) captured at body commit; the
                               sub-day encoding-time anchor t_h (SITH §4) and T_i for
                               the log-time distinctiveness term (§7).
               - episode_seq : ONE corpus-global, monotone, dense integer; the strict
                               total order seq(A)<seq(B) the directed-SR (§5) and the
                               within-day distinctiveness tie-break read.
             Two helpers: now_written_at() (the float clock) and next_episode_seq()
             (the locked, restart-survivable counter).
    Depends: time (stdlib), pathlib (stdlib), samia.core.atomic_state.locked_update_json
             (the EXISTING flock + atomic-replace primitive — reused, not reinvented).

Layer 2 (What / Why):
    What: next_episode_seq bumps one global counter under an exclusive flock and
          atomic-replaces it, returning the new value. now_written_at returns the
          current Unix float. Neither reads or writes any node frontmatter itself;
          the two primary write sites (mcp_server.memory_write_node,
          fact_extractor.write_atoms_as_nodes) and the engram materialize record
          (hippocampus.EngramStore.materialize) call these and stamp the values.
    Why:  §3 of the proposal. The corpus clock is day-granular end to end
          (valid_from/last_access are calendar dates), which collapses every event
          of one day onto one anchor and leaves the directed-SR window with no
          within-day order. written_at restores sub-second resolution; episode_seq
          restores a strict, gap-free, clock-anomaly-proof order. The counter lives
          in biomimetic/ alongside the other per-corpus state JSONs, survives restart
          by continuing from the last committed value, and is safe under the up-to-8
          concurrent sessions/hooks this codebase already hardens against.

Flag posture: P0 is purely write-time and additive. NOTHING reads these fields yet
    (the master flag ASTHENOS_TEMPORAL_WEIGHT with all weights 0 keeps the formula at
    S_c + 0.05*H_c), so stamping them is a retrieval no-op until a later phase reads
    them. Legacy nodes lack the fields and every future consumer fails open on absence.
"""
from __future__ import annotations

import time
from pathlib import Path

from .atomic_state import locked_update_json

# The single corpus-global counter file, alongside the other biomimetic/ state JSONs
# (edge_weights.json, hebb_consolidate_state.json, ...). One integer for the corpus's
# whole life: it never resets, has no per-session prefix, and survives restart.
_SEQ_RELPATH = ("biomimetic", "episode_seq.json")


def _seq_path(memory_dir: Path) -> Path:
    """Resolve the corpus-global episode_seq.json under memory_dir/biomimetic/."""
    p = Path(memory_dir)
    for part in _SEQ_RELPATH:
        p = p / part
    return p


def now_written_at() -> float:
    """Return the current Unix time as a float (the written_at anchor t_h).

    What: a thin wrapper over time.time() captured at body commit.
    Why:  a continuous elapsed-seconds delta (t_now - written_at) for the SITH drift
          kernels and the log-time distinctiveness term, with sub-second resolution so
          two atoms of one fast burst still receive DISTINCT anchors.
    """
    return time.time()


def next_episode_seq(memory_dir: Path) -> int:
    """Atomically bump and return the next corpus-global episode_seq.

    What: take an exclusive flock on biomimetic/episode_seq.json (via the existing
          atomic_state.locked_update_json — flock + temp-file + os.replace), read the
          last committed value (default 0 on first ever / missing / corrupt file),
          increment by one, commit atomically, and return the NEW value.
    Why:  §3.4. The counter must be (1) strictly increasing and dense, (2) safe under
          up to 8 concurrent sessions/hooks (the flock serializes the read-modify-write),
          and (3) restart-survivable — locked_update_json re-reads the committed JSON on
          every bump and continues from N, so the order is monotone across the corpus's
          whole life regardless of daemon restarts. No new concurrency machinery; one
          small JSON file under the existing biomimetic/ root.

    The biomimetic/ directory is created on demand so the very first write on a fresh
    corpus does not fail.
    """
    seq_path = _seq_path(memory_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    with locked_update_json(seq_path, default={"seq": 0}) as st:
        # Defensive: a hand-edited / partially-written file may carry a non-int. Treat
        # any unusable value as 0 so the counter heals forward (never crashes a write).
        try:
            current = int(st.get("seq", 0) or 0)
        except (TypeError, ValueError):
            current = 0
        st["seq"] = current + 1
        return st["seq"]


def write_time_fields(memory_dir: Path) -> dict:
    """Mint both write-time substrate values for one node write.

    What: returns {"written_at": <float>, "episode_seq": <int>} — the float anchor and
          ONE freshly-bumped counter value — for a single node creation.
    Why:  a single call site per write keeps the two fields minted together (one body
          commit -> one written_at + one episode_seq). The two primary write sites and
          the engram materialize record all route through here so the substrate is
          stamped consistently and the counter is bumped exactly once per node.
    """
    return {
        "written_at": now_written_at(),
        "episode_seq": next_episode_seq(memory_dir),
    }


# ─────────────────────────────────────────────
# [temporal_substrate] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-11  Status: active
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P0 — write-time
#             substrate (§3 + §16.1). Two additive-optional fields (written_at float
#             anchor + one corpus-global monotone episode_seq) stamped at the two
#             primary write sites + the engram materialize record. Inert at retrieval
#             until a later phase reads them; flag-off is a byte-identical no-op.
# Role:       mint + persist the write-time temporal substrate (no read/ranking path)
# Depends:    time, pathlib (stdlib); samia.core.atomic_state (locked_update_json)
# ─────────────────────────────────────────────
