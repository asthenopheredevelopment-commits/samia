"""A9 — Latency / scale axis: store -> recall wall-clock at N = 100 / 1000 / 10000.

Per ``BENCHMARK_DESIGN_v1.md`` (axis A9): "store -> recall at store sizes 10^2/10^3/10^4;
wall-clock" -> metric "p50/p95 recall ms vs N, ingest items/s", SAM/IA surface "end-to-end".
This is the only end-to-end axis: it does not probe a single module, it times the whole
store -> build-index -> recall path the adapter exposes, so the numbers reflect what a
caller actually pays.

What it measures (all programmatic, no judge — A9 is not open-ended, defect D5 N/A here)
---------------------------------------------------------------------------------------
For each store size N:

* **Ingest throughput** — wall-clock to ``store`` N facts plus build the real MiniLM
  vector index over them, reported as ``ingest_items_per_s = N / ingest_seconds`` (and the
  index-build sub-time on its own, since the embed pass dominates ingest at scale).
* **Recall latency** — per-probe wall-clock (milliseconds) for a deterministic sample of
  queries against the N-fact store, reduced to p50 / p95 / mean / min / max by the scorer.
  A first untimed warm-up recall absorbs one-time import/model-load cost so it does not
  pollute the steady-state distribution (the warm-up time is recorded separately, honestly).
* **Gold-hit rate (correctness sanity)** — fraction of sampled probes whose gold id was in
  the top-k. Latency is reported *beside* this so a fast-but-empty recall can never be
  mistaken for a latency win (honesty rail). It is not the A9 headline metric — recall
  accuracy is axis A1 — but a near-zero hit rate at scale would invalidate the latency
  numbers, so it is surfaced.

Data
----
Reuses the **A1 fact shape** from the committed, checksummed A9 dataset
(``benchmarks/data/a9_latency/``): the N=100 / 1000 store is the first 100 / 1000 facts of
the 10000-fact corpus; probes are aligned one-per-fact with an explicit gold id. The probe
SAMPLE per size is chosen by a fixed seed so the timed set is identical on every run.

Determinism
-----------
Fixed seed for the probe sample; the adapter pins the embedder (MiniLM, cache-only, no
network) and the index build is a pure function of the node set. Wall-clock VALUES vary by
machine (that is the nature of a latency benchmark), but WHICH operations are timed, the
probe sample, and the gold-hit result are fully reproducible. The scorer's percentile math
is dependency-free so the reduction reproduces exactly from the committed raw timings.

Run
---
    python benchmarks/tasks/a9_latency.py                       # all three sizes
    python benchmarks/tasks/a9_latency.py --sizes 100,1000      # a subset (smoke)
    python benchmarks/tasks/a9_latency.py --sample 50           # fewer timed probes

Writes ``results/raw_a9.jsonl`` (one record per size) + ``results/scores_a9.json`` and
prints the per-size table. Exercise it with the test venv's interpreter so the INSTALLED
``samia`` package is timed, never the dev tree.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Make the benchmark package importable whether run as a script or a module.
_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from adapters import MemoryItem, SamiaAdapter  # noqa: E402
import score as _score  # noqa: E402

_DATA_DIR = _BENCH_ROOT / "data" / "a9_latency"
_RESULTS_DIR = _BENCH_ROOT / "results"

#: Store sizes the axis sweeps (mirrors the design table 10^2 / 10^3 / 10^4).
DEFAULT_SIZES = (100, 1000, 10000)

#: How many probes to time per size. Bounded so the 10000-store case does not run 10000
#: recalls; large enough that p50/p95 are stable. Capped at the store size for small N.
DEFAULT_SAMPLE = 200

#: top-k for each recall (the design's recall ranking width; also the gold-hit cutoff).
RECALL_K = 10

#: Seed for the probe-sample selection. Fixed so the timed probe set is identical run to
#: run (the dataset itself is seeded separately by its generator).
SAMPLE_SEED = 1337


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts (one per line)."""
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _require_dataset() -> tuple[list[dict], list[dict]]:
    """Load and verify the committed A9 dataset; fail loudly if it is missing/altered.

    Verifies the on-disk SHA256SUMS so a run can never silently time a tampered or
    regenerated-with-a-different-seed corpus (non-negotiable #3: checksummed datasets).
    """
    facts_p = _DATA_DIR / "facts.jsonl"
    probes_p = _DATA_DIR / "probes.jsonl"
    if not facts_p.exists() or not probes_p.exists():
        raise SystemExit(
            f"A9 dataset missing under {_DATA_DIR}. Generate it first:\n"
            f"    python {_DATA_DIR / 'generate.py'}")
    # Import the generator's checksum verifier (keeps one source of truth for the sums).
    sys.path.insert(0, str(_DATA_DIR))
    import generate as _gen  # type: ignore  # noqa: E402
    if not _gen.check(_DATA_DIR):
        raise SystemExit(
            "A9 dataset checksum mismatch — the committed dataset has been altered. "
            "Re-generate or restore it before measuring latency.")
    facts = _load_jsonl(facts_p)
    probes = _load_jsonl(probes_p)
    if len(facts) != len(probes):
        raise SystemExit("A9 dataset corrupt: facts/probes line counts differ.")
    return facts, probes


def _facts_to_items(facts: list[dict]) -> list[MemoryItem]:
    """Map fact records onto the adapter's MemoryItem (the A1 fact shape)."""
    return [
        MemoryItem(id=f["id"], text=f["text"],
                   valid_from=f.get("valid_from", ""), source=f.get("source", ""))
        for f in facts
    ]


def _sample_indices(n: int, sample: int, seed: int) -> list[int]:
    """A deterministic, sorted sample of ``sample`` indices in [0, n) (or all if n<=sample).

    Sorted so timed probes touch the store in id order (no adversarial access pattern); the
    seed fixes WHICH probes are timed across runs.
    """
    if sample >= n:
        return list(range(n))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), sample))


def measure_size(adapter: SamiaAdapter, items: list[MemoryItem], probes: list[dict],
                 size: int, sample: int, k: int = RECALL_K) -> dict:
    """Store ``size`` facts, time ingest + a probe sample's recall, return a raw record.

    Steps (each wall-clock'd with ``time.perf_counter``):
      1. ``reset`` the adapter to an empty isolated store (not timed).
      2. ``store`` the first ``size`` facts, then ``build_index`` — the ingest cost. The
         build is forced here (not lazily on first recall) so its time is attributed to
         ingest, not to the first recall, and the recall distribution is steady-state.
      3. One untimed warm-up recall (absorbs one-time import/model-load cost), recorded
         separately as ``warmup_ms`` for honesty.
      4. Each sampled probe recalled individually and timed in milliseconds; a gold-in-top-k
         check accumulates ``gold_hits`` (correctness sanity, not the headline metric).

    The returned record is exactly the shape ``score.score_a9`` consumes.
    """
    store_items = items[:size]

    adapter.reset()

    # The largest build embeds 10k nodes over many seconds; during that window a /tmp
    # reaper or a stray maintenance pass can race the index dir away under a long run
    # (the vector module itself notes this TOCTOU class for live stores). Ensure the
    # index dir exists immediately before the build so the embed write lands; this is a
    # task-local robustness guard and does not touch the Foundation adapter.
    (adapter._root / "vector_index").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    adapter.store(store_items)
    build_t0 = time.perf_counter()
    adapter.build_index()
    build_t1 = time.perf_counter()
    ingest_seconds = build_t1 - t0
    index_build_seconds = build_t1 - build_t0

    sample_idx = _sample_indices(size, sample, SAMPLE_SEED)

    # Warm-up: first recall pays one-time costs (model load, type-cache warm). Time it but
    # keep it OUT of the steady-state distribution; record it so the cost is visible.
    warm_probe = probes[sample_idx[0]]
    w0 = time.perf_counter()
    adapter.recall(warm_probe["query"], k=k)
    warmup_ms = (time.perf_counter() - w0) * 1000.0

    recall_ms: list[float] = []
    gold_hits = 0
    for idx in sample_idx:
        probe = probes[idx]
        r0 = time.perf_counter()
        ranking = adapter.recall(probe["query"], k=k)
        recall_ms.append((time.perf_counter() - r0) * 1000.0)
        if probe["gold"] in ranking[:k]:
            gold_hits += 1

    return {
        "size": size,
        "ingest_items": len(store_items),
        "ingest_seconds": ingest_seconds,
        "index_build_seconds": index_build_seconds,
        "recall_ms": recall_ms,
        "warmup_ms": round(warmup_ms, 3),
        "gold_hits": gold_hits,
        "probes": len(sample_idx),
        "k": k,
    }


def run(sizes: tuple[int, ...] = DEFAULT_SIZES, sample: int = DEFAULT_SAMPLE,
        results_dir: Path = _RESULTS_DIR) -> dict:
    """Run A9 across ``sizes``: write raw_a9.jsonl + scores_a9.json, return the scored dict.

    One adapter instance (one isolated temp root) is reused across sizes; ``reset`` between
    sizes guarantees each measurement starts from an empty store. The largest size's facts
    are loaded once and sliced per size (the corpus is a prefix family). The adapter is
    closed (temp root removed) on exit so a run leaves no residue.
    """
    facts, probes = _require_dataset()
    items = _facts_to_items(facts)

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_records: list[dict] = []

    with SamiaAdapter() as adapter:
        for size in sizes:
            if size > len(items):
                print(f"  skip N={size}: dataset has only {len(items)} facts",
                      file=sys.stderr)
                continue
            print(f"  measuring N={size} (sample={min(sample, size)} probes) ...",
                  flush=True)
            rec = measure_size(adapter, items, probes, size, sample)
            raw_records.append(rec)
            print(
                f"    ingest {rec['ingest_seconds']:.2f}s "
                f"({rec['ingest_items'] / rec['ingest_seconds']:.1f} items/s), "
                f"warmup {rec['warmup_ms']:.1f}ms, "
                f"gold {rec['gold_hits']}/{rec['probes']}",
                flush=True)

    raw_path = results_dir / "raw_a9.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for rec in raw_records:
            fh.write(json.dumps(rec, sort_keys=True))
            fh.write("\n")

    scored = _score.score_a9(raw_records)
    scores_path = results_dir / "scores_a9.json"
    scores_path.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return scored


def _print_table(scored: dict) -> None:
    """Print the per-size latency table to stdout (the A9 headline numbers)."""
    print("\nA9 latency / scale — SAM/IA (per store size)")
    print(f"{'N':>7}  {'ingest/s':>9}  {'build_s':>8}  "
          f"{'p50_ms':>8}  {'p95_ms':>8}  {'mean_ms':>8}  {'gold_hit':>8}")
    by_size = scored["by_size"]
    for size in scored["sizes"]:
        r = by_size[str(size)]
        print(f"{r['size']:>7}  {r['ingest_items_per_s']:>9.1f}  "
              f"{r['index_build_seconds']:>8.2f}  {r['recall_p50_ms']:>8.2f}  "
              f"{r['recall_p95_ms']:>8.2f}  {r['recall_mean_ms']:>8.2f}  "
              f"{r['gold_hit_rate']:>8.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A9 latency/scale axis")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                        help="comma-separated store sizes (default 100,1000,10000)")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="probes to time per size (default 200; capped at N)")
    args = parser.parse_args(argv)
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())

    scored = run(sizes=sizes, sample=args.sample)
    _print_table(scored)
    print(f"\nwrote: {_RESULTS_DIR / 'raw_a9.jsonl'} + "
          f"{_RESULTS_DIR / 'scores_a9.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
