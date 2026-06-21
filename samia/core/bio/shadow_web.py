"""samia.core.bio.shadow_web — Unit A: the edge-based association shadow web (pure).

Layer 1 (Owns / Depends):
    Owns:    the EDGE-BASED overlay built from the linker's GENUINE candidates that also exist in
             the live epiphanies edge accounting — grouped into connected COMPONENTS (union-find,
             EDGE-LIST only, never an all-pairs member list) — plus STRICT-INCIDENCE surfacing and
             a transition diff for the append-only ledger. PURE: no IO, no live-store contact; the
             caller (epiphanies.consolidate) supplies the candidate dict + the freshly-folded edge
             accounting (`out`) and persists the result.
    Depends: nothing (stdlib only). Candidate.state / edge-rec field names are passed in as data.

Layer 2 (What / Why):
    What: build_assoc_components -> {edges, components, observed_validated, stats};
          surface_for_recall (R2: strictly-incident validated EDGES, never all-pairs of a
          component); diff_transitions (for the ledger).
    Why:  promote_to_live cannot reach NAMED curated nodes (they live outside chains). The linker
          discovers those associations; this overlay is the OBSERVATIONAL INSTRUMENT that groups
          the genuine ones into an edge-based web so it can be measured (Unit A) before any recall
          surfacing (Unit B) or live apply.

ANTI-CONFABULATION INVARIANTS (load-bearing):
    R1 (no transitive assertion): an edge is admitted ONLY at candidate.state == 'genuine'
        (re-derived from the live promotion bar by reconcile_genuine each fold; REVERT-not-latch).
        A multi-hop A-C is NEVER asserted from a path here — components GROUP already-genuine edges;
        they never synthesize a new endpoint edge. (The frontier minter + FULL-STRICT gate are the
        deferred Multi-hop package.)
    R2 (topology-preserving): a component is stored as an EDGE LIST (+ a node count), never an
        all-pairs member list. surface_for_recall returns only edges strictly incident to the
        focus nodes — never "all members of a component are co-retrieved."
    R3 (no live mutation): this module writes NOTHING; the caller persists a sidecar JSON that is
        NOT chains.json members[] and NOT edge_weights.json.
"""

from __future__ import annotations

# EdgeState fields copied VERBATIM from the live epiphanies edge rec (`out[key]`) so the shadow
# never disagrees with the live bar (it mirrors, never recomputes).
EDGE_FIELDS = ("S", "w", "cg", "tau_d", "reps", "run_id", "genuine_tier")

GENUINE = "genuine"
VALIDATED = "validated"


def endpoints(key: str) -> tuple:
    """Split a candidate/edge key 'a.md::b.md' into (a, b). Returns (key, '') if malformed."""
    if "::" in key:
        a, b = key.split("::", 1)
        return a, b
    return key, ""


def _state_of(c) -> str:
    """Read a Candidate's state whether it is a dataclass or a plain dict (fail-soft)."""
    if isinstance(c, dict):
        return c.get("state", "")
    return getattr(c, "state", "")


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path-compress
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_assoc_components(candidates: dict, edges_out: dict) -> dict:
    """Build the edge-based shadow web from the GENUINE linker candidates that also exist in the
    live edge accounting `edges_out` (key -> rec dict with EDGE_FIELDS).

    GENUINE edges form the promotable structure (union-find -> components). VALIDATED-but-not-genuine
    keys are recorded for observability only (they reached the bar at least once but are not at-bar
    now) and are NOT grouped/surfaced. Returns:
        {
          "edges":      {key: {"state":"genuine", "component": cid, <EDGE_FIELDS...>}},
          "components": {cid: {"edges": [key,...], "n_nodes": int}},   # EDGE LIST, never all-pairs
          "observed_validated": [key,...],
          "stats": {genuine_edges, components, observed_validated, max_component_nodes},
        }
    """
    uf = _UnionFind()
    genuine_keys = []
    observed_validated = []
    for key, c in candidates.items():
        st = _state_of(c)
        if key not in edges_out:
            continue
        if st == GENUINE:
            genuine_keys.append(key)
            a, b = endpoints(key)
            uf.union(a, b)            # only genuine edges define component connectivity (R1)
        elif st == VALIDATED:
            observed_validated.append(key)

    edges_map = {}
    comp_edges = {}
    comp_nodes = {}
    for key in genuine_keys:
        a, b = endpoints(key)
        cid = uf.find(a)              # canonical root = component id
        rec = edges_out.get(key, {})
        entry = {"state": GENUINE, "component": cid}
        for f in EDGE_FIELDS:
            entry[f] = rec.get(f)
        edges_map[key] = entry
        comp_edges.setdefault(cid, []).append(key)
        comp_nodes.setdefault(cid, set()).update((a, b))

    components = {
        cid: {"edges": sorted(keys), "n_nodes": len(comp_nodes[cid])}
        for cid, keys in comp_edges.items()
    }
    max_nodes = max((c["n_nodes"] for c in components.values()), default=0)
    return {
        "edges": edges_map,
        "components": components,
        "observed_validated": sorted(observed_validated),
        "stats": {
            "genuine_edges": len(edges_map),
            "components": len(components),
            "observed_validated": len(observed_validated),
            "max_component_nodes": max_nodes,
        },
    }


def surface_for_recall(sidecar: dict, focus_nodes) -> list:
    """R2 STRICT INCIDENCE: return ONLY the genuine edges with at least one endpoint in
    `focus_nodes`, each as its own pairwise record — NEVER the all-pairs closure of a component.
    e.g. focus={A} with genuine A-B and B-C returns EXACTLY [A-B], never [A-B, B-C] and never A-C.
    Returns [{"a","b","key","component", <EDGE_FIELDS...>}].
    """
    focus = set(focus_nodes or [])
    out = []
    for key, entry in (sidecar.get("edges") or {}).items():
        a, b = endpoints(key)
        if a in focus or b in focus:
            rec = {"a": a, "b": b, "key": key, "component": entry.get("component")}
            for f in EDGE_FIELDS:
                rec[f] = entry.get(f)
            out.append(rec)
    return out


def diff_transitions(prev_sidecar: dict, new_sidecar: dict) -> list:
    """Compute genuine-edge transitions between two sidecars for the append-only ledger.
    Returns [{"key","change"}] where change in {"genuine_added","genuine_dropped"}. REVERT-not-latch
    is captured: an edge that left the genuine set (decayed below bar / reverted) is 'genuine_dropped'.
    """
    prev = set((prev_sidecar or {}).get("edges", {}).keys())
    new = set((new_sidecar or {}).get("edges", {}).keys())
    out = [{"key": k, "change": "genuine_added"} for k in sorted(new - prev)]
    out += [{"key": k, "change": "genuine_dropped"} for k in sorted(prev - new)]
    return out


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.shadow_web
# Author:     code_warrior (Epiphanies v3 — Unit A, edge-association overlay)
# Project:    Asthenosphere — SAM/IA — Epiphanies (associative shadow web)
# Version:    0.1.0  (pure instrument — own sidecars only; never recall/chains/edge_weights)
# Phase:      build — the edge-based association overlay: union-find connected COMPONENTS over the
#             linker's GENUINE edges; revert-not-latch transitions captured for the shadow ledger.
# Layer:      core (pure library — no IO; the caller persists the returned sidecars).
# Role:       build_assoc_components / surface_for_recall (strict incidence) / diff_transitions; a
#             pure observational instrument — no recall surfacing, no live apply (those are Unit B).
# Stability:  new — anti-confabulation is structural (separate sidecar store; never surfaced live).
# Depends:    nothing at import (operates on the injected candidate dict + folded edge rec).
# Exposes:    build_assoc_components, surface_for_recall, diff_transitions (+ EDGE_FIELDS).
# --------------------------------------------------------------------------
