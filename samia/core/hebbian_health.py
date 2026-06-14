#!/usr/bin/env python3
"""samia.core.hebbian_health — read-only health gauge for the Hebbian engram web.

Layer 1 (Owns / Depends):
    Owns:    main — the CLI entry; prints the chain-promotion verdict (within /
                 cross / orphan edge buckets) then the unified-web ("Piece A")
                 success gauge. This module is a diagnostic; it has no library
                 callers — bio.py / web_store.py do the actual writes.
    Depends: samia.core.paths.resolve_memory_root (the real memory root across dev,
             staged-release, and site-packages layouts). stdlib only otherwise
             (json, sys, itertools, pathlib, sqlite3 — lazy in the web report).
Layer 2 (What / Why):
    What: classify every accumulated co-activation edge in edge_weights.json by
          whether its two endpoints can EVER chain-promote (share a chain =
          within_chain; different chains = cross_chain; one node chainless =
          orphan), then read edges.db's 'coactivation' rows to show the unified web
          actually forming. Reports counts, weight maxima, and the promotable split.
    Why:  SAM/IA was designed as a multi-in/multi-out webwork where cross-chains
          form and prune via Hebbian consolidation. The legacy promotion path in
          bio.py can only add an edge when BOTH nodes already co-reside in one
          chain, so cross-chain pairs accumulate weight forever but never promote.
          This tool measures that silently-dropped set: high cross-chain weight +
          zero cross-chain promotions == the membership gate is the bottleneck (see
          project_hebbian_cross_chain memory node). Read-only; it changes nothing.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

from samia.core.paths import resolve_memory_root

# MEMORY_DIR — Why: resolved through samia.core.paths (env -> verified-legacy
#     -> XDG) so biomimetic/ and chains/ are read from the real memory root in
#     dev, staged-release, and site-packages layouts. The old parents[3]
#     derivation was correct only in the dev tree.
MEMORY_DIR = resolve_memory_root(create=False)
BIO_DIR = MEMORY_DIR / "biomimetic"
CHAINS_DIR = MEMORY_DIR / "chains"
EDGE_WEIGHTS = BIO_DIR / "edge_weights.json"
COACT_LOG = BIO_DIR / "coactivation_log.jsonl"

HEBB_PROMOTION = 0.85  # mirror bio.py


def _node_to_chains() -> dict[str, set[str]]:
    """Map node filename -> set of chains it is a member of."""
    m: dict[str, set[str]] = {}
    for cp in CHAINS_DIR.glob("*.json"):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        chain = cp.stem
        for mem in data.get("members") or []:
            f = mem.get("file") if isinstance(mem, dict) else None
            if not f:
                continue
            stem = Path(f).name
            m.setdefault(stem, set()).add(chain)
            m.setdefault(stem.removesuffix(".md"), set()).add(chain)
    return m


# _classify — What: bucket an edge (a, b) by promotability — orphan (a node in no
#     chain), within_chain (share a chain), or cross_chain (different chains).
def _classify(a: str, b: str, n2c: dict[str, set[str]]) -> str:
    ca, cb = n2c.get(a) or n2c.get(a + ".md") or set(), n2c.get(b) or n2c.get(b + ".md") or set()
    if not ca or not cb:
        return "orphan"          # at least one node in no chain -> can never promote
    if ca & cb:
        return "within_chain"    # share a chain -> CAN promote today
    return "cross_chain"         # different chains -> accumulates weight, never promotes
# _classify — Why: only within_chain edges can promote under the legacy chain path;
#     the orphan/cross_chain split is exactly the set the unified web exists to rescue.


def main() -> None:
    n2c = _node_to_chains()
    weights = {}
    if EDGE_WEIGHTS.exists():
        weights = json.loads(EDGE_WEIGHTS.read_text(encoding="utf-8"))

    # PromotabilityBuckets — What: sort each "a::b" edge weight into its promotability
    #     class; malformed keys (no "::") and non-dict values are skipped.
    buckets = {"within_chain": [], "cross_chain": [], "orphan": []}
    for key, v in weights.items():
        if "::" not in key or not isinstance(v, dict):
            continue
        a, b = key.split("::", 1)
        buckets[_classify(a, b, n2c)].append(v.get("w", 0.0))
    # PromotabilityBuckets — Why: the cross_chain + orphan totals vs within_chain are the
    #     headline metric — the share of accumulated weight the chain path can never use.

    # clustering: count chains and lateral (hebbian) edges
    n_chains = len(list(CHAINS_DIR.glob("*.json")))
    hebbian_edges = 0
    cross_chain_edges_in_graph = 0
    for cp in CHAINS_DIR.glob("*.json"):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in data.get("edges", []) or []:
            if e.get("label") == "hebbian":
                hebbian_edges += 1

    coact_events = 0
    if COACT_LOG.exists() and COACT_LOG.stat().st_size:
        coact_events = sum(1 for _ in COACT_LOG.open())

    def stat(ws):
        if not ws:
            return "n=0"
        above = sum(1 for w in ws if w >= HEBB_PROMOTION)
        return f"n={len(ws)} max={max(ws):.3f} >=0.85(promotable)={above}"

    print("=== Hebbian web health ===")
    print(f"chains: {n_chains}  |  hebbian lateral edges in chain graph: {hebbian_edges}")
    print(f"pending co-activation events (unconsolidated): {coact_events}")
    print(f"edge_weights entries: {sum(len(v) for v in buckets.values())}")
    print("--- accumulated co-activation edges by promotability class ---")
    print(f"  within_chain (CAN promote):     {stat(buckets['within_chain'])}")
    print(f"  cross_chain  (NEVER promotes):  {stat(buckets['cross_chain'])}")
    print(f"  orphan       (NEVER promotes):  {stat(buckets['orphan'])}")
    blocked = len(buckets["cross_chain"]) + len(buckets["orphan"])
    total = sum(len(v) for v in buckets.values()) or 1
    blocked_high = sum(1 for w in buckets["cross_chain"] + buckets["orphan"] if w >= HEBB_PROMOTION)
    print("--- verdict (legacy chain-promotion path; UNCHANGED by Piece A by design) ---")
    print(f"  {blocked}/{total} ({100*blocked/total:.0f}%) of accumulated edges are "
          f"barred from CHAIN promotion (cross-chain or orphan).")
    print(f"  {blocked_high} exceed 0.85 but cannot become CHAIN edges — these are "
          f"exactly the associations the unified web now captures instead.")

    _unified_web_report()


def _unified_web_report() -> None:
    """Piece A success gauge: the unified associative web (edges.db coactivation rows).

    After migration, the cross-chain/orphan associations that the chain path barred
    should appear HERE as real edges. Pre-migration this prints 0 — that's the
    'before'; post-migration it should show ~the formerly-blocked count.
    """
    import os
    import sqlite3
    db = os.path.expanduser("~/.local/share/asthenos/memory_graph/edges.db")
    print("=== unified associative web (edges.db, ref_kind=coactivation) — Piece A ===")
    if not os.path.exists(db):
        print("  edges.db absent — web not yet initialized.")
        return
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except Exception as e:
        print(f"  could not open edges.db read-only: {e}")
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)")}
    if "weight" not in cols:
        print("  weight column ABSENT — migration not yet run (web is pre-Piece-A).")
        conn.close()
        return
    rows = conn.execute(
        "SELECT weight FROM edges WHERE ref_kind='coactivation'").fetchall()
    ws = [r[0] for r in rows]
    has_nodes = bool(conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'").fetchone())
    n_mass = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] if has_nodes else 0
    conn.close()
    if not ws:
        print("  coactivation edges: 0  (run migrate_web_store_2026_05_29.py or wait "
              "for the next consolidation event to populate)")
    else:
        strong = sum(1 for w in ws if w >= 0.85)
        print(f"  coactivation edges: {len(ws)}  (max={max(ws):.3f}, "
              f"strong>=0.85={strong})  — the web is FORMING (cross-chain allowed).")
    print(f"  per-node mass entries: {n_mass}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.hebbian_health
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-05-29-hebbian-cross-chain-web-v01 (Piece A) success gauge.
# Layer:      core (read-only diagnostic CLI; no library callers)
# Role:       read-only Hebbian-web health gauge — classify every co-activation edge
#             by chain-promotability (within/cross/orphan) then read edges.db to gauge
#             the unified web forming; the 'before/after' measure for Piece A.
# Stability:  stable -- reporting only; never mutates memory or the web.
# ErrorModel: every file/db read is fail-soft -- unreadable chain JSON is skipped,
#             an absent edge_weights.json yields empty buckets, an absent or
#             pre-migration edges.db prints a "not yet initialized" line; the tool
#             never raises on missing substrate (it is a 'before' gauge by design).
# Depends:    json, sys, itertools, pathlib (stdlib); sqlite3, os (stdlib, lazy in
#             the unified-web report). samia.core.paths.resolve_memory_root.
# Exposes:    main (CLI entry).
# Lines:      201
# --------------------------------------------------------------------------
