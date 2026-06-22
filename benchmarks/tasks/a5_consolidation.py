"""A5 — Consolidation gain: recall@k BEFORE vs AFTER a consolidation cycle (Delta recall).

Axis (design doc §A5): score ``recall@k`` *before* vs *after* a REM/merge cycle and report
the lift (Delta recall). The cycle is driven through the **real** adapter API
(``MemoryAdapter.consolidate()``), which on the SAM/IA adapter runs the package's
programmatic consolidation audit (``samia.core.consolidation.audit_all``) plus a vector
index rebuild. The data is A5's OWN fixed, checksummed corpus — separate from A1/A2 so
retrieval, retention and consolidation are never conflated (defect D6).

How A5 is measured here (and the honest caveat)
-----------------------------------------------
1. ``reset`` -> ``store`` the whole A5 corpus (singletons + near-duplicate clusters).
2. ``recall`` every probe, score recall@{1,5,10} + MRR over the gold ids  -> **before**.
3. ``consolidate()`` once, through the real API. Its summary is captured verbatim.
4. ``recall`` every probe again under identical conditions -> **after**.
5. ``delta_recall = after - before`` per k -> the A5 lift.

Scoring is fully programmatic — the gold is a single id per probe (clean labels: defects
D1/D2/D4), so no judge runs on this axis (no reader/judge confound, defect D5).

**The caveat the spec demands, reported, not hidden:** SAM/IA's actual atom *merge* is
gated by an offline REM judge (an LLM), which a deterministic, network-free benchmark must
not invoke. ``consolidate()`` therefore reports ``judge_applied = False``: it surfaces merge
*candidates* and rebuilds the index, but does NOT collapse the near-duplicate clusters. So
A5 here measures the **consolidation pass that is available without the judge** — and the
expected, honest result is that it is recall-**neutral** (the rebuild preserves the
ranking; no lift is fabricated). The end-to-end judge-gated consolidation gain is reported
as NOT MEASURABLE under the determinism rules, with the reason, rather than invented.

Determinism / isolation
------------------------
Fixed dataset (SHA256-pinned; the task refuses a checksum mismatch), pinned MiniLM embedder
(cache-only, autofetch off in the adapter), per-instance temp memory root. Same inputs ->
same rankings -> same scores. Run with the test venv so the INSTALLED package is exercised::

    python benchmarks/tasks/a5_consolidation.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Make the benchmarks package importable whether this is run as a module or a script.
_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from adapters import MemoryItem, SamiaAdapter  # noqa: E402
import score as _score  # noqa: E402

AXIS = "a5_consolidation"
_DATA_DIR = _BENCH_ROOT / "data" / AXIS
_DATASET_PATH = _DATA_DIR / "dataset.json"
_SUMS_PATH = _DATA_DIR / "SHA256SUMS"


def _verify_and_load_dataset() -> dict:
    """Load the fixed A5 dataset, refusing to run on a SHA256 mismatch.

    The committed ``SHA256SUMS`` pins ``dataset.json``'s bytes; a mismatch means the data
    was edited out from under the checksum and any number produced would be unattributable.
    We fail loud (the design's "versioned, checksummed datasets, no surprises" rule) rather
    than silently scoring against altered data.
    """
    if not _DATASET_PATH.exists():
        raise SystemExit(
            f"A5 dataset missing: {_DATASET_PATH}. Generate it with "
            f"`python {_DATA_DIR / 'generate.py'}`.")
    raw = _DATASET_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = _read_expected_sum()
    if expected is not None and digest != expected:
        raise SystemExit(
            f"A5 dataset checksum mismatch for {_DATASET_PATH.name}: "
            f"expected {expected}, got {digest}. Regenerate or restore the pinned data.")
    return json.loads(raw)


def _read_expected_sum() -> str | None:
    """Return the pinned sha256 for ``dataset.json`` from SHA256SUMS, or None if absent."""
    if not _SUMS_PATH.exists():
        return None
    for line in _SUMS_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == "dataset.json":
            return parts[0]
    return None


def _items_from_dataset(dataset: dict) -> list[MemoryItem]:
    """Build the MemoryItem store batch from the dataset's items (store order preserved)."""
    out: list[MemoryItem] = []
    for it in dataset["items"]:
        out.append(MemoryItem(
            id=it["id"],
            text=it["text"],
            valid_from=it.get("valid_from", ""),
            source=it.get("source", ""),
            trusted=bool(it.get("trusted", True)),
            meta={"cluster": it.get("cluster")},
        ))
    return out


def _recall_phase(adapter: SamiaAdapter, probes: list[dict], k: int) -> list[dict]:
    """Recall every probe at depth ``k`` and pair each ranking with its gold id.

    Returns a list of per-probe records (probe text, gold id, kind, cluster, ranking) — the
    raw material both the scorer and the audit trail read. No scoring here; this is the
    measurement, kept separate from the metric so raw outputs are always inspectable.
    """
    records: list[dict] = []
    for p in probes:
        ranking = adapter.recall(p["probe"], k=k)
        records.append({
            "probe": p["probe"],
            "gold_id": p["gold_id"],
            "kind": p["kind"],
            "cluster": p.get("cluster"),
            "ranking": ranking,
        })
    return records


def _score_phase(records: list[dict], k_values: list[int]) -> dict:
    """Score one recall phase: recall@k for each k + MRR over the probe set."""
    rankings = [r["ranking"] for r in records]
    golds = [r["gold_id"] for r in records]
    recall = _score.recall_at_k_set(rankings, golds, k_values)
    return {
        "recall_at_k": {str(k): recall[k] for k in k_values},
        "mrr": _score.mrr(rankings, golds),
    }


def run(adapter: SamiaAdapter | None = None) -> dict:
    """Run A5 end to end and return the result dict (scores + raw + honest caveat).

    Drives the consolidation cycle through the real ``consolidate()`` API and measures
    recall@k before and after it. The returned dict carries: the per-phase scores, the
    Delta-recall metric, the verbatim ``consolidate`` summary (so ``judge_applied`` is on
    the record), the raw per-probe rankings for both phases, and an explicit
    ``judge_gated_merge_measured`` honesty flag.
    """
    dataset = _verify_and_load_dataset()
    k_values: list[int] = list(dataset["k_values"])
    k_max = max(k_values)
    probes: list[dict] = dataset["probes"]
    items = _items_from_dataset(dataset)

    own_adapter = adapter is None
    adapter = adapter or SamiaAdapter()
    try:
        adapter.reset()
        stored = adapter.store(items)
        assert stored == [it.id for it in items], "store did not echo ids in order"

        # Phase 1: recall BEFORE the consolidation cycle.
        before_records = _recall_phase(adapter, probes, k_max)
        before_scores = _score_phase(before_records, k_values)

        # Drive the REAL consolidation API. On the SAM/IA adapter this runs the package's
        # programmatic audit + index rebuild and reports judge_applied=False (the actual
        # merge is REM-judge gated and out of scope for a network-free deterministic run).
        consolidate_summary = adapter.consolidate()

        # Phase 2: recall AFTER the consolidation cycle, identical conditions.
        after_records = _recall_phase(adapter, probes, k_max)
        after_scores = _score_phase(after_records, k_values)

        before_recall = {k: before_scores["recall_at_k"][str(k)] for k in k_values}
        after_recall = {k: after_scores["recall_at_k"][str(k)] for k in k_values}
        delta = _score.delta_recall(before_recall, after_recall)

        judge_applied = bool(consolidate_summary.get("judge_applied", False))

        return {
            "axis": AXIS,
            "adapter": adapter.name,
            "schema_version": dataset["schema_version"],
            "seed": dataset["seed"],
            "k_values": k_values,
            "item_count": len(items),
            "probe_count": len(probes),
            "cluster_count": dataset["cluster_count"],
            "consolidate_summary": consolidate_summary,
            # The headline A5 metric.
            "delta_recall_at_k": {str(k): delta[k] for k in k_values},
            "before": before_scores,
            "after": after_scores,
            # Honesty rail (design doc): the end-to-end judge-gated merge was NOT exercised,
            # so the consolidation GAIN reported is "without judge". judge_applied comes
            # straight from the adapter's own summary — never asserted by the task.
            "judge_applied": judge_applied,
            "judge_gated_merge_measured": judge_applied,
            "consolidation_gain_caveat": (
                "Delta recall measures the consolidation pass available WITHOUT the REM "
                "judge (programmatic audit + index rebuild). The judge-gated atom merge is "
                "an offline LLM step out of scope for a deterministic, network-free run, so "
                "the end-to-end consolidation gain is NOT measured here. A recall-neutral "
                "Delta (~0) is the honest expected result; no lift is fabricated."
            ),
            "raw": {
                "before": before_records,
                "after": after_records,
            },
        }
    finally:
        if own_adapter:
            adapter.close()


def _print_report(result: dict) -> None:
    """Print a short human-readable A5 summary (the raw dict is the machine record)."""
    print(f"axis:    {result['axis']}  adapter: {result['adapter']}")
    print(f"corpus:  {result['item_count']} items / {result['probe_count']} probes / "
          f"{result['cluster_count']} duplicate clusters")
    print(f"consolidate summary: {json.dumps(result['consolidate_summary'])}")
    print("recall@k  before -> after  (delta):")
    for k in result["k_values"]:
        ks = str(k)
        b = result["before"]["recall_at_k"][ks]
        a = result["after"]["recall_at_k"][ks]
        d = result["delta_recall_at_k"][ks]
        print(f"  recall@{k:<2} {b:.3f} -> {a:.3f}  (delta {d:+.3f})")
    print(f"MRR       before {result['before']['mrr']:.3f} -> "
          f"after {result['after']['mrr']:.3f}")
    print(f"judge_applied: {result['judge_applied']}  "
          f"(judge-gated merge measured: {result['judge_gated_merge_measured']})")
    print("caveat: " + result["consolidation_gain_caveat"])


def main() -> int:
    result = run()
    _print_report(result)

    # Write the raw per-probe outputs + scores so a third party can re-score (design rule:
    # everything published). Results land under benchmarks/results/ (created on demand).
    results_dir = _BENCH_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"raw_{AXIS}.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for phase in ("before", "after"):
            for rec in result["raw"][phase]:
                fh.write(json.dumps({"phase": phase, **rec}) + "\n")

    scores_path = results_dir / f"scores_{AXIS}.json"
    scores_view = {k: v for k, v in result.items() if k != "raw"}
    scores_path.write_text(json.dumps(scores_view, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {raw_path}")
    print(f"wrote {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
