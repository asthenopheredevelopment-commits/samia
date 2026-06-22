"""A6 — associative / multi-hop recall (runnable, deterministic, seeded).

What this axis measures
-----------------------
Multi-hop *associative* recall: a linked chain ``a -> b -> c`` is wired into SAM/IA's
Hebbian co-activation graph by co-activating ONLY adjacent atoms (head+middle, then
middle+tail — never head+tail), then the chain head is the query and the chain tail is the
gold answer. The tail is reachable only *transitively*, through the middle node(s), so a
correct answer can only come from association that propagates across atoms — not from
single-vector similarity (the chain atoms are authored on distinct sub-topics precisely so
head<->tail vector similarity is low; see the dataset generator).

This is the design's A6 axis ("seed linked chain a->b->c; query a, expect c; multi-hop
recall@k", surface ``core/chain`` Hebbian + ``successor``). It is the one axis that exercises
cross-atom association rather than retrieval similarity, so it has its OWN data, separate from
A1/A2 (defect D6).

How it is grounded in the REAL SAM/IA API (no invented functions)
-----------------------------------------------------------------
Every call routes through the installed ``samia`` package, exercised exactly the way the
package's own Hebbian pipeline runs:

1. **Store** the chain + distractor + noise atoms as ``type: semantic`` nodes via the
   benchmark's ``SamiaAdapter`` (the same write path A1 uses) into an isolated temp root.
2. **Seed** the chain: for each adjacent ``(u, v)`` edge, call
   ``samia.core.bio.hebbian.hebbian_record(root, [u, v], source="genuine")`` once per
   co-activation (``coactivations_per_edge`` from the dataset), then run
   ``samia.core.bio.hebbian.hebbian_consolidate(root)`` to drain the co-activation log into
   ``biomimetic/edge_weights.json`` (the EMA edge-weight graph). This is the package's real
   Hebbian write pipeline — the same one its recall hook drives in production.
3. **Walk** from the head: ``samia.core.successor.need_vector(root, [(head, 1.0)])`` runs the
   package's query-local truncated power iteration over that edge graph (the multi-hop
   associative reach), and ``need_at`` reads the resulting discounted-occupancy at each
   candidate node. Candidates are ranked by that occupancy (best first) — the multi-hop
   recall ranking.

Why this exercises the multi-hop surface and not ``adapter.recall``
-------------------------------------------------------------------
The adapter's ``recall`` is pure semantic-vector top-k (``semantic_recall.atom_retrieve``);
it does not walk the Hebbian graph (that walk is the ``successor`` term, which the live
retrieval path only mixes in behind the temporal-weight flag). Since the chain atoms are
deliberately NOT vector-similar head-to-tail, scoring A6 through ``adapter.recall`` would
measure the wrong thing. So A6 drives the actual multi-hop surface (``successor.need_vector``
over the real Hebbian ``edge_weights.json``) directly, which is the capability the axis names.
The ``edge_weights`` graph is built only by the real ``hebbian_consolidate`` pipeline above —
nothing here reimplements an association.

Scoring (fully programmatic — no judge; defect D5)
--------------------------------------------------
Each probe has exactly one gold tail id, so scoring reads the id ranking directly:

* **multi-hop recall@k** for k in the dataset's ``k_values``: is the gold tail in the top-k
  of the need-ranked candidate pool?
* **MRR**: mean reciprocal rank of the gold tail (0 if not reached at all).
* **reached rate**: fraction of probes where the gold tail got any positive walk occupancy
  (it was reachable at all), reported alongside recall so "ranked but barely reached" is
  distinguishable from "never reached".
* **noise leakage**: fraction of stored noise atoms that received any positive occupancy from
  any walk — a sanity check that the walk does not diffuse indiscriminately (it should be 0).

All scores are broken down by hop count (2 vs 3) so depth-2 and depth-3 reachability are
reported separately, never hidden behind one aggregate (honesty rail).

Determinism: the dataset is checksum-pinned, the seeding order is fixed, the embedder is the
pinned MiniLM (only needed for the store/index round-trip, not for the walk), and the walk is
a pure function of the edge graph. No network at score time (autofetch off). Two runs produce
identical rankings.

Run::

    python benchmarks/tasks/a6_associative.py            # prints the score summary
    python benchmarks/tasks/a6_associative.py --raw out  # also writes raw per-probe jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Make the benchmark package importable when run as a bare script (python tasks/a6_*.py),
# mirroring how the harness lays out adapters/ next to tasks/.
_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from adapters import MemoryItem, SamiaAdapter  # noqa: E402

# Real SAM/IA surfaces (installed package). Imported at module load so a missing/renamed
# function fails loudly rather than being silently worked around.
from samia.core.bio import hebbian as _hebbian  # noqa: E402
from samia.core.bio import config as _hebb_config  # noqa: E402
from samia.core import successor as _successor  # noqa: E402

_DATA_DIR = _BENCH_ROOT / "data" / "a6_associative"
_DATASET_PATH = _DATA_DIR / "dataset.json"
_SUMS_PATH = _DATA_DIR / "SHA256SUMS"

AXIS = "a6_associative"


# --------------------------------------------------------------------------------------
# Dataset loading (checksum-pinned)
# --------------------------------------------------------------------------------------

def _load_dataset() -> dict:
    """Load the A6 dataset, refusing to run on a checksum mismatch (no-network determinism).

    The committed ``SHA256SUMS`` pins ``dataset.json``'s bytes; if the file on disk does not
    match, the experiment is not the one that was published, so we abort rather than score a
    silently-changed corpus.
    """
    if not _DATASET_PATH.exists():
        raise SystemExit(
            f"A6 dataset missing: {_DATASET_PATH}\n"
            f"Generate it with: python {_DATA_DIR / 'generate.py'}")
    raw = _DATASET_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = None
    if _SUMS_PATH.exists():
        for line in _SUMS_PATH.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == "dataset.json":
                expected = parts[0]
                break
    if expected is None:
        raise SystemExit(f"A6 SHA256SUMS missing/!covers dataset.json: {_SUMS_PATH}")
    if digest != expected:
        raise SystemExit(
            "A6 dataset checksum mismatch (corpus changed since it was pinned):\n"
            f"  expected {expected}\n  got      {digest}\n"
            f"Regenerate + re-pin with: python {_DATA_DIR / 'generate.py'}")
    return json.loads(raw.decode("utf-8"))


def _node_stem(node: str) -> str:
    """Strip a trailing ``.md`` so a need-map key maps back to a dataset item id."""
    return node[:-3] if node.endswith(".md") else node


# --------------------------------------------------------------------------------------
# Seeding the Hebbian chain through the real package pipeline
# --------------------------------------------------------------------------------------

def _seed_chains(root: Path, edges: list[dict], coactivations_per_edge: int) -> dict:
    """Wire every chain edge into the Hebbian graph via the real package pipeline.

    For each ``(from, to)`` adjacency, record ``coactivations_per_edge`` genuine
    co-activation events (``hebbian_record``), then drain the log into ``edge_weights.json``
    (``hebbian_consolidate``). Only adjacent atoms are ever co-activated, so head->tail is
    purely transitive. Returns the consolidation summary + a snapshot of the built edge set
    for the audit trail.
    """
    # The dataset's seeding strength must meet the package's promotion bar so each adjacency
    # is a genuine, promotable association (not a weak transient). Assert the contract so the
    # dataset and the installed package cannot silently drift apart.
    bar = int(_hebb_config.HEBB_PROMOTE_REPEATS)
    if coactivations_per_edge < bar:
        raise SystemExit(
            f"A6 dataset coactivations_per_edge={coactivations_per_edge} is below the "
            f"package's promotion bar HEBB_PROMOTE_REPEATS={bar}; the chain would be wired "
            f"too weakly to be a genuine association. Regenerate the dataset with a value "
            f">= {bar}.")

    for e in edges:
        u = e["from"] if e["from"].endswith(".md") else f"{e['from']}.md"
        v = e["to"] if e["to"].endswith(".md") else f"{e['to']}.md"
        for _ in range(coactivations_per_edge):
            # source="genuine" => full-weight, decay-clock-refreshing co-activation (the real
            # recall-hook event). _hebbian.hebbian_record needs >= 2 nodes; we pass the pair.
            _hebbian.hebbian_record(root, [u, v], query=f"a6-seed-{e['from']}-{e['to']}",
                                    source="genuine")

    # Drain the co-activation log into the EMA edge-weight graph. promote=True lets the
    # package also try to promote pairs into chain edges where members exist; A6 reads the
    # raw edge_weights graph for the walk, so chain promotion is incidental here.
    summary = _hebbian.hebbian_consolidate(root, promote=True)

    weights = _hebbian._load_edge_weights(root)
    built_edges = {k: round(float(v.get("w", 0.0)), 4) for k, v in weights.items()}
    return {
        "consolidate_events": summary.get("events", 0),
        "edge_weights_total": summary.get("weights_total", len(weights)),
        "built_edges": built_edges,
    }


def _rank_candidates(root: Path, head_id: str, candidate_ids: list[str]) -> list[tuple[str, float]]:
    """Rank ``candidate_ids`` by the real multi-hop walk seeded from ``head_id`` (best first).

    Runs ``successor.need_vector`` from the single head seed (the package's query-local
    truncated power iteration over ``edge_weights.json``), then reads ``need_at`` at each
    candidate. Sorts by occupancy descending; the id (ascending) is the deterministic
    tiebreak so equal-occupancy candidates have a stable, reproducible order.
    """
    need = _successor.need_vector(root, [(f"{head_id}.md", 1.0)])
    scored = [(cid, float(_successor.need_at(need, f"{cid}.md"))) for cid in candidate_ids]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


# --------------------------------------------------------------------------------------
# Metrics (programmatic)
# --------------------------------------------------------------------------------------

def _recall_at_k(ranking: list[str], gold_id: str, k: int) -> int:
    """1 if ``gold_id`` is in the top-``k`` of ``ranking`` else 0."""
    return 1 if gold_id in ranking[:k] else 0


def _reciprocal_rank(ranking: list[str], gold_id: str) -> float:
    """1/(rank) of ``gold_id`` (1-based) in ``ranking``; 0.0 if absent."""
    for i, cid in enumerate(ranking, start=1):
        if cid == gold_id:
            return 1.0 / i
    return 0.0


# --------------------------------------------------------------------------------------
# Task runner
# --------------------------------------------------------------------------------------

def run(adapter: SamiaAdapter | None = None) -> dict:
    """Run A6 end to end against SAM/IA and return a JSON-able result dict.

    Stores the corpus, seeds the chains via the real Hebbian pipeline, walks from each head
    via ``successor.need_vector``, and scores multi-hop recall@k + MRR programmatically. If
    no adapter is passed, a fresh isolated ``SamiaAdapter`` is created and cleaned up.
    """
    dataset = _load_dataset()
    items = dataset["items"]
    probes = dataset["probes"]
    edges = dataset["edges"]
    k_values = list(dataset["k_values"])
    coact = int(dataset["coactivations_per_edge"])

    # Candidate pool for ranking = every stored atom except the probe's own head. Pinned once
    # (all item ids) so every probe is ranked over the SAME pool (chains + distractors +
    # noise), which is what makes reaching the gold tail a discriminating result.
    all_ids = [it["id"] for it in items]
    noise_ids = [it["id"] for it in items if it.get("role") == "noise"]

    owns_adapter = adapter is None
    if owns_adapter:
        adapter = SamiaAdapter()
    try:
        adapter.reset()
        # Store every atom (chain + noise) as a type:semantic node via the real write path.
        adapter.store([
            MemoryItem(id=it["id"], text=it["text"],
                       valid_from=it.get("valid_from", ""), source=it.get("source", ""),
                       trusted=it.get("trusted", True))
            for it in items
        ])
        # Build the index once (round-trips the store through the real MiniLM index; the walk
        # itself does not need it, but building it keeps A6's store path identical to A1's and
        # proves the corpus is a valid, indexable atom population).
        adapter.build_index()

        root = adapter._root  # the adapter's isolated temp memory root (its own contract)

        # Seed the chains into the Hebbian edge graph via the real package pipeline. Apply
        # the same no-network env discipline the adapter uses, restoring it afterward so the
        # seeding never leaks config onto an outer process.
        prior_autofetch = os.environ.get("ASTHENOS_MODEL_AUTOFETCH")
        os.environ["ASTHENOS_MODEL_AUTOFETCH"] = "0"
        try:
            seed_info = _seed_chains(root, edges, coact)
        finally:
            if prior_autofetch is None:
                os.environ.pop("ASTHENOS_MODEL_AUTOFETCH", None)
            else:
                os.environ["ASTHENOS_MODEL_AUTOFETCH"] = prior_autofetch

        # Walk + score every probe.
        per_probe: list[dict] = []
        noise_reached: set[str] = set()
        for pr in probes:
            head = pr["probe_node"]
            gold = pr["gold_id"]
            hops = int(pr["hops"])
            pool = [cid for cid in all_ids if cid != head]
            scored = _rank_candidates(root, head, pool)
            ranking = [cid for cid, _ in scored]
            occ = {cid: s for cid, s in scored}

            # Track any noise atom that got positive occupancy from this walk (leakage check).
            for nid in noise_ids:
                if occ.get(nid, 0.0) > 0.0:
                    noise_reached.add(nid)

            gold_occ = occ.get(gold, 0.0)
            rr = _reciprocal_rank(ranking, gold)
            row = {
                "chain": pr["chain"],
                "probe_node": head,
                "gold_id": gold,
                "hops": hops,
                "gold_occupancy": round(gold_occ, 6),
                "gold_reached": gold_occ > 0.0,
                "gold_rank": (ranking.index(gold) + 1) if gold in ranking else None,
                "reciprocal_rank": round(rr, 6),
                "top5": [{"id": cid, "occ": round(s, 6)} for cid, s in scored[:5]],
                "recall_at_k": {str(k): _recall_at_k(ranking, gold, k) for k in k_values},
            }
            per_probe.append(row)
    finally:
        if owns_adapter:
            adapter.close()

    return _aggregate(dataset, per_probe, seed_info, noise_ids, noise_reached)


def _aggregate(dataset: dict, per_probe: list[dict], seed_info: dict,
               noise_ids: list[str], noise_reached: set[str]) -> dict:
    """Roll per-probe rows into the A6 score summary (overall + per-hop-count breakdown)."""
    k_values = list(dataset["k_values"])
    n = len(per_probe)

    def _block(rows: list[dict]) -> dict:
        m = len(rows)
        if m == 0:
            return {"n": 0}
        recall = {str(k): round(sum(r["recall_at_k"][str(k)] for r in rows) / m, 4)
                  for k in k_values}
        mrr = round(sum(r["reciprocal_rank"] for r in rows) / m, 4)
        reached = round(sum(1 for r in rows if r["gold_reached"]) / m, 4)
        return {"n": m, "recall_at_k": recall, "mrr": mrr, "reached_rate": reached}

    by_hops: dict[str, dict] = {}
    for h in sorted({r["hops"] for r in per_probe}):
        by_hops[str(h)] = _block([r for r in per_probe if r["hops"] == h])

    noise_leak = round(len(noise_reached) / len(noise_ids), 4) if noise_ids else 0.0

    return {
        "axis": AXIS,
        "adapter": "samia",
        "schema_version": dataset.get("schema_version"),
        "seed": dataset.get("seed"),
        "probe_count": n,
        "k_values": k_values,
        "scoring": "programmatic (single gold tail per probe; no judge)",
        "overall": _block(per_probe),
        "by_hops": by_hops,
        "noise_leakage": {
            "noise_total": len(noise_ids),
            "noise_reached": sorted(noise_reached),
            "leak_rate": noise_leak,
        },
        "seed_info": seed_info,
        "per_probe": per_probe,
    }


def _print_summary(result: dict) -> None:
    """Human-readable summary to stdout (the report-order honesty view)."""
    ov = result["overall"]
    print(f"[A6 associative / multi-hop]  probes={result['probe_count']}  "
          f"scoring={result['scoring']}")
    print(f"  overall  MRR={ov['mrr']}  reached={ov['reached_rate']}  "
          f"recall@k={ov['recall_at_k']}")
    for h, blk in sorted(result["by_hops"].items()):
        print(f"  {h}-hop   n={blk['n']}  MRR={blk['mrr']}  reached={blk['reached_rate']}  "
              f"recall@k={blk['recall_at_k']}")
    nl = result["noise_leakage"]
    print(f"  noise leakage: {nl['leak_rate']} ({len(nl['noise_reached'])}/{nl['noise_total']} "
          f"noise atoms reached — should be 0)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the A6 associative/multi-hop axis.")
    ap.add_argument("--raw", metavar="PATH",
                    help="write raw per-probe rows as JSONL to PATH")
    ap.add_argument("--json", metavar="PATH",
                    help="write the full result dict as JSON to PATH")
    args = ap.parse_args(argv)

    result = run()
    _print_summary(result)

    if args.raw:
        with open(args.raw, "w", encoding="utf-8") as f:
            for row in result["per_probe"]:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote raw per-probe rows -> {args.raw}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote full result -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
