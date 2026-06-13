"""samia.core.web_store — the unified associative web (Hebbian → edges.db).

What: The single first-class associative graph over ALL memory nodes. bio.py's
      Hebbian consolidation writes weighted, CROSS-CHAIN co-activation edges here;
      the Topology Atlas (graphify.rs) reads them and renders the physics. This
      collapses the two previously-disconnected edge systems (bio.py edge_weights
      + the Atlas's edges.db) into one store that the real co-activation dynamics
      govern.

Why: Measured 2026-05-29 (hebbian_health.py), 100% of co-activations were
     structurally barred from becoming navigable edges — promotion required both
     nodes to already share a chain, so the webwork SAM/IA was designed around
     never formed. Per the approved FEAT-2026-05-29-hebbian-cross-chain-web-v01
     (Piece A), this module removes that gate: any co-activated pair forms a web
     edge regardless of chain membership, tiered by weight, decayed and pruned
     over time. Chains remain a curated OVERLAY on top of this web.

Scope (Piece A only): raw substrate — weighted cross-chain edges + per-node
     decaying mass + timestamps. The 3D force law (anti-gravity anchors, inertia,
     collision) is Piece B and DERIVES from these fields; nothing here computes
     forces or positions.

Storage: extends the existing edges.db (abyss_graph.py owns markdown_link/
     session_uuid/warrior_name/aud_id ref_kinds; this module owns the
     'coactivation' ref_kind, plus a weight column and a nodes table). All DDL is
     additive + idempotent so both writers coexist on one PK-disjoint table.
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path
from typing import Optional

# RefKind — What: the ref_kind this module owns in edges.db.
# Why: PK is (src,dst,ref_kind); a distinct kind keeps bio.py-written Hebbian edges
#      from colliding with abyss_graph's markdown/session/warrior edges.
COACTIVATION = "coactivation"

# Tiered-formation + pruning constants (mirror bio.py's HEBB_* where shared).
WEAK_FORM = 0.20          # weight at/above which a weak web edge is materialized
STRUCTURAL = 0.85         # weight at/above which an edge is "strong/structural"
PRUNE_BELOW = 0.05        # drop edges whose weight decays below this
EDGE_DECAY_PER_DAY = 0.005
DEGREE_CAP = 32           # max coactivation edges kept per node (synaptic pruning)
MASS_BUMP = 1.0           # mass added per node appearance in a co-activation event
MASS_DECAY_PER_DAY = 0.01 # mass slowly lightens when a node goes unused

_DEFAULT_DB_DIR = os.path.expanduser("~/.local/share/asthenos/memory_graph")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "edges.db")


def _utc_now() -> str:
    # What: UTC ISO8601 with Z, matching abyss_graph's timestamp format.
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _db_path(db_dir: Optional[str] = None) -> str:
    return os.path.join(db_dir, "edges.db") if db_dir else _DEFAULT_DB_PATH


def connect(db_dir: Optional[str] = None) -> sqlite3.Connection:
    """Open edges.db and ensure the additive schema (weight column + nodes table).

    Why: self-bootstrapping like abyss_graph._connect — safe to call before the
    one-shot migration script has run, so consolidation never crashes on a cold db.
    """
    path = _db_path(db_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    # Base edges table (created by abyss_graph; recreate IF NOT EXISTS for safety).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            src_node         TEXT NOT NULL,
            dst_node         TEXT NOT NULL,
            ref_kind         TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL DEFAULT 1,
            first_seen_at    TEXT NOT NULL,
            last_seen_at     TEXT NOT NULL,
            PRIMARY KEY (src_node, dst_node, ref_kind)
        )
    """)
    # AdditiveMigration — What: add the float weight column if absent.
    # Why: edges.db only had integer occurrence_count; Hebbian weight is a float EMA.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)")}
    if "weight" not in cols:
        conn.execute("ALTER TABLE edges ADD COLUMN weight REAL NOT NULL DEFAULT 0.0")
    # Per-node substrate (mass = decaying usage; Piece B derives gravity/inertia/size).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id     TEXT PRIMARY KEY,
            mass        REAL NOT NULL DEFAULT 0.0,
            last_access TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _order(a: str, b: str) -> tuple[str, str]:
    # What: canonical undirected ordering so (a,b) and (b,a) map to one row.
    return (a, b) if a <= b else (b, a)


def upsert_edge(conn: sqlite3.Connection, a: str, b: str, weight: float,
                now: Optional[str] = None) -> bool:
    """Materialize/refresh one coactivation web edge. Cross-chain + orphan allowed.

    Returns True if the edge was written (weight >= WEAK_FORM), False if too weak.
    Tier is conveyed by the stored weight (>= STRUCTURAL == strong/structural).
    """
    if weight < WEAK_FORM or a == b:
        return False
    now = now or _utc_now()
    src, dst = _order(a, b)
    conn.execute("""
        INSERT INTO edges (src_node, dst_node, ref_kind, occurrence_count,
                           first_seen_at, last_seen_at, weight)
        VALUES (?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(src_node, dst_node, ref_kind) DO UPDATE SET
            occurrence_count = occurrence_count + 1,
            last_seen_at = excluded.last_seen_at,
            weight = excluded.weight
    """, (src, dst, COACTIVATION, now, now, float(weight)))
    return True


def bump_mass(conn: sqlite3.Connection, node: str, amount: float = MASS_BUMP,
              now: Optional[str] = None) -> None:
    """Increase a node's mass (routine-use makes nodes heavier). Offline-only."""
    now = now or _utc_now()
    conn.execute("""
        INSERT INTO nodes (node_id, mass, last_access, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            mass = mass + excluded.mass,
            last_access = excluded.last_access
    """, (node, float(amount), now, now))


def decay_prune(conn: sqlite3.Connection, now: Optional[str] = None) -> dict:
    """Decay edge weights + node mass by age; prune weak edges + cap node degree.

    Synaptic pruning (Piece A's 'prune' half of 'form and prune'):
      - edge weight decays by age toward PRUNE_BELOW, then is deleted
      - node mass slowly lightens when unused
      - per-node degree cap keeps only the DEGREE_CAP strongest coactivation edges
    """
    now = now or _utc_now()
    today = _dt.date.today()
    pruned = capped = 0

    # Decay edge weights by days since last_seen, delete those below PRUNE_BELOW.
    for src, dst, w, last in conn.execute(
            "SELECT src_node, dst_node, weight, last_seen_at FROM edges WHERE ref_kind=?",
            (COACTIVATION,)).fetchall():
        days = _days_since(last, today)
        nw = w * max(0.0, 1.0 - EDGE_DECAY_PER_DAY * days) if days > 0 else w
        if nw < PRUNE_BELOW:
            conn.execute("DELETE FROM edges WHERE src_node=? AND dst_node=? AND ref_kind=?",
                         (src, dst, COACTIVATION))
            pruned += 1
        elif nw != w:
            conn.execute("UPDATE edges SET weight=? WHERE src_node=? AND dst_node=? AND ref_kind=?",
                         (nw, src, dst, COACTIVATION))

    # Per-node degree cap: for any node with > DEGREE_CAP coactivation edges, drop the weakest.
    capped += _enforce_degree_cap(conn)

    # Decay node mass by days since last access.
    for node, mass, last in conn.execute(
            "SELECT node_id, mass, last_access FROM nodes").fetchall():
        days = _days_since(last, today)
        if days > 0:
            nm = mass * max(0.0, 1.0 - MASS_DECAY_PER_DAY * days)
            conn.execute("UPDATE nodes SET mass=? WHERE node_id=?", (nm, node))

    conn.commit()
    return {"pruned": pruned, "degree_capped": capped}


def delete_node_edges(conn: sqlite3.Connection, node_id: str) -> dict:
    """Hard-delete every edge (ALL ref_kinds) touching node_id + its nodes(mass) row.

    FEAT-2026-06-07 P0: the edges.db endpoint of the forget_node cascade. A node's death
    must not leave dangling 'ghost' edges (the audited corruption). Deletes WHERE
    src_node OR dst_node = node_id across every ref_kind, then drops the per-node mass row.
    Idempotent (a second call deletes 0). Caller commits via this function.
    """
    cur = conn.execute("DELETE FROM edges WHERE src_node=? OR dst_node=?", (node_id, node_id))
    edges_deleted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    cur2 = conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
    nodes_deleted = cur2.rowcount if cur2.rowcount and cur2.rowcount > 0 else 0
    conn.commit()
    return {"edges_deleted": edges_deleted, "node_rows_deleted": nodes_deleted}


def forget_node_edges(node_id: str, db_dir: Optional[str] = None) -> dict:
    """Module-level wrapper: open edges.db, delete_node_edges, close. No-op-safe if the
    db file does not exist yet."""
    if not os.path.exists(_db_path(db_dir)):
        return {"edges_deleted": 0, "node_rows_deleted": 0, "skipped": "no-db"}
    conn = connect(db_dir)
    try:
        return delete_node_edges(conn, node_id)
    finally:
        conn.close()


def coactivation_neighbors(node: str, db_dir: Optional[str] = None,
                           limit: int = DEGREE_CAP) -> list[str]:
    """FEAT-2026-06-07 P3b — the live + clean Tier-0 co-activation locus of a node.

    What: return the node ids sharing a coactivation edge with `node` (either
          endpoint), strongest-weight first, capped at `limit`. No-op-safe if
          edges.db does not exist yet (returns []).
    Why:  the ONLINE active-set is "what fires together with the new write".
          edges.db is the live + clean (post-P2 ghost-free) Tier-0 web; this is
          the trivial neighbor query over the existing 'coactivation' rows that
          web_store owns. bio.active_set unions this with hot/recent + the
          pluggable Tier-1 hook (P3d).
    """
    if not os.path.exists(_db_path(db_dir)):
        return []
    fname = node if node.endswith(".md") else f"{node}.md"
    conn = connect(db_dir)
    try:
        rows = conn.execute(
            "SELECT dst_node, weight FROM edges WHERE ref_kind=? AND src_node=? "
            "UNION ALL "
            "SELECT src_node, weight FROM edges WHERE ref_kind=? AND dst_node=?",
            (COACTIVATION, fname, COACTIVATION, fname),
        ).fetchall()
    finally:
        conn.close()
    # Strongest first; dedup keeping the max weight seen for each neighbor.
    best: dict[str, float] = {}
    for nb, w in rows:
        if nb == fname:
            continue
        fw = float(w)
        if fw > best.get(nb, float("-inf")):
            best[nb] = fw
    ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [nb for nb, _ in ordered[:limit]]


def _enforce_degree_cap(conn: sqlite3.Connection) -> int:
    # What: for each node whose coactivation degree exceeds DEGREE_CAP, delete the
    #       weakest edges beyond the cap. Why: bounds the web (synaptic pruning).
    capped = 0
    # Build per-node edge lists (undirected: count both endpoints).
    deg: dict[str, list[tuple]] = {}
    for src, dst, w in conn.execute(
            "SELECT src_node, dst_node, weight FROM edges WHERE ref_kind=?",
            (COACTIVATION,)).fetchall():
        deg.setdefault(src, []).append((w, src, dst))
        deg.setdefault(dst, []).append((w, src, dst))
    to_drop: set[tuple] = set()
    for node, edges in deg.items():
        if len(edges) <= DEGREE_CAP:
            continue
        edges.sort()  # weakest first
        for _w, src, dst in edges[:len(edges) - DEGREE_CAP]:
            to_drop.add((src, dst))
    for src, dst in to_drop:
        conn.execute("DELETE FROM edges WHERE src_node=? AND dst_node=? AND ref_kind=?",
                     (src, dst, COACTIVATION))
        capped += 1
    return capped


def _days_since(iso: str, today: _dt.date) -> int:
    try:
        d = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).date()
        return max(0, (today - d).days)
    except Exception:
        return 0


def sync_from_consolidation(weights: dict, node_appearances: Optional[dict] = None,
                            db_dir: Optional[str] = None,
                            memory_dir: Optional[Path] = None) -> dict:
    """High-level entry bio.hebbian_consolidate calls after computing EMA weights.

    weights: {"a::b": {"w": float, ...}} — the bio.py edge_weights map.
    node_appearances: {node_id: count} co-activation appearances this cycle (mass).
    memory_dir: the memory root. When given, edges whose endpoint no longer exists
        under nodes/ are SKIPPED (not re-upserted) and reported in stats["dead_keys"]
        so the caller can evict them from edge_weights.json this same pass — mirroring
        the P0 forget_node cascade (node death must not leave OR re-grow edges).
    Writes all live-endpoint edges >= WEAK_FORM (cross-chain + orphan), bumps mass,
    then prunes.

    GHOST-EDGE GUARD (G3-2026-06-11): without memory_dir this re-created an edges.db
    row for a DELETED node every cycle (the weights map can still carry a pair whose
    endpoint was forgotten). With memory_dir we build the live-node set ONCE per pass
    (cheap, cached) and skip+report dead-endpoint pairs so the death is honored
    automatically, every cycle, not only via the operator-gated sweep_ghost_edges.
    """
    # Build the live-node set ONCE per sync pass (cheap glob, cached for the whole
    # upsert loop) — only when a memory_dir is supplied. Keys are "<name>.md::<name>.md"
    # (full filenames), matching nodes_dir.glob("*.md") names exactly.
    live: Optional[set[str]] = None
    if memory_dir is not None:
        nodes_dir = Path(memory_dir) / "nodes"
        try:
            live = {p.name for p in nodes_dir.glob("*.md")}
        except OSError:
            live = None  # fail-soft: an unreadable nodes/ dir disables the guard

    conn = connect(db_dir)
    try:
        now = _utc_now()
        formed = 0
        skipped_dead = 0
        dead_keys: list[str] = []
        for key, v in weights.items():
            if "::" not in key:
                continue
            a, b = key.split("::", 1)
            # GHOST-EDGE GUARD: a pair with a forgotten endpoint is NOT re-upserted —
            # node death must not re-grow its edges. Record the key so the caller
            # evicts it from edge_weights.json in this same pass.
            if live is not None and (a not in live or b not in live):
                skipped_dead += 1
                dead_keys.append(key)
                continue
            if upsert_edge(conn, a, b, float(v.get("w", 0.0)), now):
                formed += 1
        if node_appearances:
            for node, cnt in node_appearances.items():
                bump_mass(conn, node, MASS_BUMP * float(cnt), now)
        conn.commit()
        stats = decay_prune(conn, now)
        stats["formed"] = formed
        stats["skipped_dead"] = skipped_dead
        stats["dead_keys"] = dead_keys
        return stats
    finally:
        conn.close()


# ── module metadata ────────────────────────────────────────────────────────
# file:        samia/core/web_store.py
# role:        Piece A unified associative web writer (Hebbian → edges.db)
# proposal:    FEAT-2026-05-29-hebbian-cross-chain-web-v01 (approved 2026-05-29)
# owns:        ref_kind='coactivation' rows + weight column + nodes(mass) table
# consumers:   bio.hebbian_consolidation (writer), graphify.rs (reader), Piece B (forces),
#              bio.active_set (P3b reader via coactivation_neighbors)
# gauge:       samia/core/hebbian_health.py
# G3-2026-06-11 (ghost-edge re-upsert fix): sync_from_consolidation now takes an
#              optional memory_dir. When given, it builds the live-node set ONCE per
#              pass (cheap glob, cached) and SKIPS pairs whose endpoint no longer
#              exists under nodes/ (a forgotten node's pair was being re-upserted into
#              edges.db every cycle) — reporting them in stats["dead_keys"] so the
#              caller (bio.hebbian_consolidate) evicts them from edge_weights.json in
#              the SAME pass. Mirrors the P0 forget_node cascade: node death must not
#              leave OR re-grow edges. Without memory_dir, legacy behavior is preserved.
