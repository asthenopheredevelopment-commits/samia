"""A1 — Retrieval-accuracy axis (deterministic, programmatic).

Task (from ``BENCHMARK_DESIGN_v1.md``): seed ``N`` distinct-topic facts; issue one
paraphrased query per fact; score whether the gold fact is in the recall top-k.

* **Metrics:** recall@1, recall@5, recall@10, and MRR — all computed programmatically from
  the ranked id list the adapter returns. There is **no LLM judge here**: retrieval is
  closed-form (the gold id is either present in the ranking or not), so the design's pinned
  judge — reserved for the *open-ended* axes A3/A4/A7 — is deliberately not invoked. This is
  the reader/judge-confound fix (defect D5): the primary number never depends on generated
  prose or a model's reading of it.
* **Retrieval only (defect D6):** this axis uses its OWN dataset
  (``data/a1_retrieval/dataset.json``) and stores every fact once with no interleaving,
  no delay, no decay. Retention-after-delay is a *separate* axis (A2) with separate data;
  conflating them is the field's most common defect and is structurally avoided here.
* **Clean gold (defects D1/D2/D4):** the dataset's generator guarantees one unambiguous
  gold per query (distinct subject+kind topic, distinct detail), and every item carries an
  explicit gold id + rationale. The checksum is verified before scoring so a drifted dataset
  cannot silently change a number.
* **Determinism / no network at score time:** the dataset is fixed + checksummed; the
  adapter pins the MiniLM embedder cache-only (autofetch off). Same dataset + same installed
  package → same ranking → same score.

Run standalone (exercises the INSTALLED ``samia`` via the SamiaAdapter):

    python benchmarks/tasks/a1_retrieval.py            # default adapter = samia

or import :func:`run` from the harness with any ``MemoryAdapter``. Writes per-item raw
results and the score summary under ``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Make ``benchmarks/`` importable whether run as a module or a script.
_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from adapters import MemoryAdapter, MemoryItem  # noqa: E402

AXIS = "a1_retrieval"
DATA_DIR = _BENCH_ROOT / "data" / AXIS
RESULTS_DIR = _BENCH_ROOT / "results"
# The k values scored. recall@k needs the adapter to return at least k ids, so we recall at
# the largest k once and slice the prefix for the smaller ones (one recall per query).
K_VALUES = (1, 5, 10)
RECALL_K = max(K_VALUES)


# -- dataset loading + integrity ------------------------------------------------------------

def load_dataset(data_dir: Path = DATA_DIR) -> dict:
    """Load the A1 dataset and verify its SHA256 against the committed manifest.

    A drifted/edited dataset must not silently change a published number, so the checksum is
    enforced here (not advisory). Raises ``RuntimeError`` on mismatch or a missing manifest.
    """
    ds_path = data_dir / "dataset.json"
    sums_path = data_dir / "SHA256SUMS"
    if not ds_path.exists():
        raise RuntimeError(
            f"dataset missing: {ds_path} — run `python {data_dir}/generate.py` first")
    raw = ds_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if not sums_path.exists():
        raise RuntimeError(f"checksum manifest missing: {sums_path}")
    expected = sums_path.read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise RuntimeError(
            f"dataset checksum mismatch for {ds_path}\n  expected {expected}\n  got      "
            f"{digest}\nthe committed dataset was modified — regenerate or restore it")
    return json.loads(raw)


# -- programmatic scorer --------------------------------------------------------------------

def score_item(ranked: list[str], gold: str, k_values=K_VALUES) -> dict:
    """Score one query's ranked id list against its gold id (pure, programmatic).

    Returns ``{rank, reciprocal_rank, hit@k...}``. ``rank`` is the 1-based position of the
    gold in ``ranked`` (``None`` if absent); ``reciprocal_rank`` is ``1/rank`` (``0.0`` on a
    miss). ``hit@k`` is True iff the gold appears within the first ``k`` ids. This is the
    whole scorer — no model, no text parsing, just set/position membership.
    """
    rank = None
    for i, rid in enumerate(ranked, start=1):
        if rid == gold:
            rank = i
            break
    out: dict = {
        "rank": rank,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
    }
    for k in k_values:
        out[f"hit@{k}"] = bool(rank is not None and rank <= k)
    return out


def aggregate(per_item: list[dict], k_values=K_VALUES) -> dict:
    """Aggregate per-item scores into recall@k + MRR over the whole dataset."""
    n = len(per_item)
    if n == 0:
        return {"n": 0, "mrr": 0.0, **{f"recall@{k}": 0.0 for k in k_values}}
    agg: dict = {"n": n}
    for k in k_values:
        hits = sum(1 for r in per_item if r["score"][f"hit@{k}"])
        agg[f"recall@{k}"] = round(hits / n, 6)
    agg["mrr"] = round(sum(r["score"]["reciprocal_rank"] for r in per_item) / n, 6)
    return agg


# -- task runner ----------------------------------------------------------------------------

def run(adapter: MemoryAdapter, data_dir: Path = DATA_DIR,
        results_dir: Path = RESULTS_DIR) -> dict:
    """Execute the A1 retrieval axis against ``adapter`` and write raw + summary results.

    Stores every fact once (additive, single index build), recalls each query at
    ``RECALL_K``, scores programmatically, aggregates, and writes
    ``results/raw_a1_retrieval.jsonl`` (one line per query) + ``results/a1_retrieval.json``
    (the summary). Returns the summary dict.
    """
    ds = load_dataset(data_dir)
    items = ds["items"]

    # Reset to an empty, isolated store, then seed all facts in one additive batch.
    adapter.reset()
    mem_items = [MemoryItem(id=it["id"], text=it["text"]) for it in items]
    stored_ids = adapter.store(mem_items)
    assert stored_ids == [it["id"] for it in items], (
        "adapter.store must echo ids in order (round-trip contract)")

    per_item: list[dict] = []
    for it in items:
        ranked = adapter.recall(it["query"], k=RECALL_K)
        sc = score_item(ranked, it["gold"])
        per_item.append({
            "id": it["id"],
            "query": it["query"],
            "gold": it["gold"],
            "ranked": ranked,
            "score": sc,
        })

    summary = {
        "axis": AXIS,
        "adapter": adapter.name,
        "dataset_sha256": hashlib.sha256(
            (data_dir / "dataset.json").read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest(),
        "seed": ds.get("seed"),
        "n": len(items),
        "recall_k": RECALL_K,
        "scoring": "programmatic (gold id position in ranked recall list)",
        "judge_applied": False,
        "metrics": aggregate(per_item),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"raw_{AXIS}.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for row in per_item:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    summary_path = results_dir / f"{AXIS}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")

    summary["_raw_path"] = str(raw_path)
    summary["_summary_path"] = str(summary_path)
    return summary


def _build_adapter(name: str) -> MemoryAdapter:
    """Construct the named adapter. v1 ships only ``samia``."""
    if name == "samia":
        from adapters import SamiaAdapter
        return SamiaAdapter()
    raise SystemExit(f"unknown adapter {name!r} (v1 ships: samia)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the A1 retrieval-accuracy axis.")
    ap.add_argument("--adapter", default="samia")
    args = ap.parse_args(argv)

    adapter = _build_adapter(args.adapter)
    try:
        summary = run(adapter)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    m = summary["metrics"]
    print(f"A1 retrieval — adapter={summary['adapter']} n={summary['n']} "
          f"(dataset {summary['dataset_sha256'][:12]})")
    print(f"  recall@1  = {m['recall@1']:.4f}")
    print(f"  recall@5  = {m['recall@5']:.4f}")
    print(f"  recall@10 = {m['recall@10']:.4f}")
    print(f"  MRR       = {m['mrr']:.4f}")
    print(f"  raw     -> {summary['_raw_path']}")
    print(f"  summary -> {summary['_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
