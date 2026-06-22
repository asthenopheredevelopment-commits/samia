"""run_benchmark — the SAM/IA capability benchmark orchestrator (all 9 axes, one CLI).

Per ``BENCHMARK_DESIGN_v1.md`` §"Harness architecture": this is the single entry point that
runs every axis module ``tasks/a*.py`` against one ``MemoryAdapter`` (v1: the SAM/IA adapter
over the INSTALLED package), normalizes each axis's heterogeneous result into a uniform
per-axis raw file (``results/raw_<axis>.jsonl``) plus a single ``results/scores.json``, and
emits an auditable run manifest. The numbers it writes are the inputs to ``REPORT.md``.

Design rails enforced here (not just documented):

* **Runs against the installed package** — every axis builds a ``SamiaAdapter`` which imports
  ``samia`` from the active interpreter's site-packages (run this with the test venv's python
  so the installed build is exercised, never a dev tree).
* **Determinism** — a single ``--seed`` is recorded and passed to the axes that draw on it;
  the datasets are fixed + checksum-verified inside each axis; the embedder + judge are pinned
  by the package. Same inputs + same environment → identical ``scores.json`` (the harness
  diffs the scoring-relevant subtree on ``--check-determinism``).
* **No hidden aggregate** — ``scores.json`` carries each axis's own headline metrics verbatim;
  there is no single rolled-up "benchmark score" that could hide a weak axis (honesty rail).
* **Programmatic first, judge auditable** — each axis returns its programmatic metrics as the
  trustworthy numbers; the open-ended judge subset (A3/A4/A7) is carried through as a saved,
  re-scoreable transcript via ``score.py`` and never gates the programmatic figure.

Each axis module exposes a ``run(...)`` whose signature differs (some own an adapter, some
take one; A9 takes store sizes; A4 takes a seed; A7 returns a dataclass). The harness owns a
thin per-axis adapter shim (``_AXES``) that calls the real ``run`` and returns ``(headline,
raw_rows, full_result)`` — it never reimplements a metric, it only maps the axis's own output
into the uniform shape. Nothing about a number is computed here; the axes compute, the harness
collects.

Usage::

    python run_benchmark.py --seed 1337 --sizes 100,1000,10000
    python run_benchmark.py --axes a1,a8            # a subset (smoke)
    python run_benchmark.py --check-determinism     # run twice, assert identical scores
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from adapters import SamiaAdapter  # noqa: E402

RESULTS_DIR = _BENCH_ROOT / "results"

# Axis order is the design's A1..A9. Each entry names the module under tasks/ and the
# adapter-shim that calls its real run() and normalizes the output. The ALL list is the
# default run set.
ALL_AXES = [
    "a1_retrieval", "a2_retention", "a3_temporal", "a4_contradiction",
    "a5_consolidation", "a6_associative", "a7_distill", "a8_provenance", "a9_latency",
]

# Short aliases so ``--axes a1,a8`` works as well as the full module names.
_ALIAS = {f"a{i}": ALL_AXES[i - 1] for i in range(1, 10)}


def _resolve_axes(spec: str | None) -> list[str]:
    """Map a comma ``--axes`` spec (aliases or full names) to ordered module names."""
    if not spec:
        return list(ALL_AXES)
    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        name = _ALIAS.get(tok, tok)
        if name not in ALL_AXES:
            raise SystemExit(f"unknown axis {tok!r}; choose from {', '.join(ALL_AXES)}")
        if name not in out:
            out.append(name)
    return out


# --------------------------------------------------------------------------------------
# Per-axis shims: call the axis's REAL run(), return (headline_metrics, raw_rows, full).
# Each shim only maps the axis's own output into the uniform shape — no metric is computed
# here. headline = the small dict that goes into scores.json; raw_rows = the per-item list
# written to raw_<axis>.jsonl; full = the complete axis result (kept for the report/audit).
# --------------------------------------------------------------------------------------

def _shim_a1(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a1_retrieval")
    with SamiaAdapter() as adapter:
        summary = mod.run(adapter)
    raw_path = _BENCH_ROOT / "results" / "raw_a1_retrieval.jsonl"
    rows = _read_jsonl(raw_path)
    headline = {"n": summary["n"], "judge_applied": summary["judge_applied"],
                **summary["metrics"]}
    return headline, rows, summary


def _shim_a2(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a2_retention")
    result = mod.run()
    rows = result["by_delay"]
    # Headline = retention@k at the longest delay + the matching noise-drop, plus the full
    # delay schedule (each row carries retention + noise-drop) so the report has the curve.
    final = rows[-1] if rows else {}
    headline = {
        "salient_count": result["salient_count"],
        "noise_count": result["noise_count"],
        "k_values": result["k_values"],
        "delay_schedule": result["delay_schedule"],
        "judge_used": result["judge_used"],
        "final_delay": final.get("delay_ticks"),
        "retention_at_k_final": final.get("retention_at_k", {}),
        "noise_drop_rate_final": final.get("noise_drop_rate"),
        "by_delay": [
            {"delay_ticks": r["delay_ticks"],
             "retention_at_k": r["retention_at_k"],
             "noise_drop_rate": r["noise_drop_rate"],
             "salient_surviving": r["salient_surviving"],
             "noise_surviving": r["noise_surviving"]}
            for r in rows
        ],
    }
    return headline, rows, result


def _shim_a3(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a3_temporal")
    result = mod.run(write_results=True)
    rows = result["detail"]["recall_per_query"]
    headline = {
        "n_items": result["n_items"],
        "n_queries": result["n_queries"],
        "temporal_recall": result["metrics"]["temporal_recall"],
        "temporal_recall_by_kind": result["metrics"]["temporal_recall_by_kind"],
        "ordering_acc": result["metrics"]["ordering_acc"],
        "open_ended_judge_status": result["open_ended_judge"]["status"],
        "open_ended_judge_subset_n": len(result["open_ended_judge"]["subset"]),
    }
    return headline, rows, result


def _shim_a4(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a4_contradiction")
    result = mod.run(seed=seed)
    rows = result.get("raw", [])
    headline = {k: v for k, v in result.items() if k not in ("raw", "per_case")}
    return headline, rows, result


def _shim_a5(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a5_consolidation")
    with SamiaAdapter() as adapter:
        result = mod.run(adapter)
    rows: list[dict] = []
    for phase in ("before", "after"):
        for rec in result["raw"][phase]:
            rows.append({"phase": phase, **rec})
    headline = {k: v for k, v in result.items() if k != "raw"}
    return headline, rows, result


def _shim_a6(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a6_associative")
    result = mod.run()
    rows = result["per_probe"]
    headline = {k: v for k, v in result.items() if k != "per_probe"}
    return headline, rows, result


def _shim_a7(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a7_distill")
    result = mod.run().to_json()
    rows = result["per_claim"]
    headline = {k: v for k, v in result.items() if k != "per_claim"}
    # The judge transcripts are saved under the judge subtree; keep them out of the headline
    # body but record the keep_rate + availability (auditable, re-scoreable).
    return headline, rows, result


def _shim_a8(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a8_provenance")
    result = mod.run()
    rows = result["rows"]
    headline = result["scores"]
    return headline, rows, result


def _shim_a9(seed: int, sizes: tuple[int, ...]) -> tuple[dict, list[dict], dict]:
    mod = importlib.import_module("tasks.a9_latency")
    scored = mod.run(sizes=sizes)
    # A9 already wrote results/raw_a9.jsonl (one record per size); read it back as the rows.
    rows = _read_jsonl(_BENCH_ROOT / "results" / "raw_a9.jsonl")
    headline = {
        "metric": scored["metric"],
        "sizes": scored["sizes"],
        "by_size": scored["by_size"],
    }
    return headline, rows, scored


_SHIMS = {
    "a1_retrieval": _shim_a1, "a2_retention": _shim_a2, "a3_temporal": _shim_a3,
    "a4_contradiction": _shim_a4, "a5_consolidation": _shim_a5, "a6_associative": _shim_a6,
    "a7_distill": _shim_a7, "a8_provenance": _shim_a8, "a9_latency": _shim_a9,
}


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, or [] if it does not exist."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as deterministic, sorted-key JSONL (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def run_axis(axis: str, seed: int, sizes: tuple[int, ...]) -> dict:
    """Run one axis via its shim, persist ``raw_<axis>.jsonl``, return its scores entry.

    Returns the per-axis scores-block: the headline metrics, the wall-clock the axis took,
    and the path to its raw rows. Errors are NOT swallowed — a broken axis must fail the run
    loudly (a fabricated/missing number is worse than a crash).
    """
    shim = _SHIMS[axis]
    t0 = time.perf_counter()
    headline, rows, _full = shim(seed, sizes)
    elapsed = time.perf_counter() - t0

    raw_path = RESULTS_DIR / f"raw_{axis}.jsonl"
    _write_jsonl(raw_path, rows)

    return {
        "axis": axis,
        "wall_seconds": round(elapsed, 3),
        "raw_rows": len(rows),
        "raw_path": raw_path.name,
        "metrics": headline,
    }


def run(axes: list[str], seed: int, sizes: tuple[int, ...]) -> dict:
    """Run the requested axes and assemble the full ``scores.json`` document.

    The document carries a run manifest (seed, sizes, interpreter, package version, platform)
    + one block per axis with that axis's own headline metrics verbatim. There is deliberately
    NO single aggregate score (honesty rail): the report reads each axis block directly.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    axis_blocks: dict[str, dict] = {}
    for axis in axes:
        print(f"[run] {axis} ...", flush=True)
        block = run_axis(axis, seed, sizes)
        axis_blocks[axis] = block
        print(f"[run] {axis} done in {block['wall_seconds']}s "
              f"({block['raw_rows']} raw rows)", flush=True)

    document = {
        "benchmark": "samia_capability_v1",
        "design": "BENCHMARK_DESIGN_v1.md",
        "manifest": _run_manifest(seed, sizes, axes),
        "axes": axis_blocks,
        "note": (
            "No aggregate score is reported by design: each axis block above carries its own "
            "headline metrics verbatim so a weak axis is never hidden behind a rolled-up "
            "number. Programmatic metrics are the trustworthy figures; any LLM-judge subset "
            "(A3/A4/A7) is an auditable, re-scoreable side-channel that never gates them."),
    }
    return document


def _samia_version() -> str:
    """Return the installed samia package version (or 'unknown' if it cannot be read)."""
    try:
        import importlib.metadata as _md
        return _md.version("samia")
    except Exception:
        try:
            import samia
            return getattr(samia, "__version__", "unknown")
        except Exception:
            return "unknown"


def _run_manifest(seed: int, sizes: tuple[int, ...], axes: list[str]) -> dict:
    """Provenance for a run: seed, sizes, axes, interpreter + package + platform digests.

    Everything that determines a number's reproducibility EXCEPT the live wall-clock (which a
    latency axis must vary by machine) is captured so a third party knows exactly what
    produced the scores. No host-specific absolute path is recorded (only the python version
    and the package version), keeping the manifest portable + leak-free.
    """
    return {
        "seed": seed,
        "sizes": list(sizes),
        "axes": list(axes),
        "adapter": "samia",
        "samia_version": _samia_version(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "network_at_score_time": False,
    }


def _scoring_view(document: dict) -> dict:
    """Extract the determinism-relevant subtree: per-axis metrics with wall-clock dropped.

    Wall-clock (the ``wall_seconds`` field and A9's millisecond timings) legitimately varies
    run to run, so it is excluded from the determinism check; everything else — every
    programmatic metric, every gold-id outcome — must be byte-identical across runs.
    """
    # Per-axis wall-clock fields that legitimately vary run to run and must be excluded from
    # the determinism check (the scoring numbers around them must still match exactly).
    _WALLCLOCK_KEYS = ("timing_s", "timing", "wall_seconds", "elapsed_s", "elapsed")
    out: dict = {}
    for axis, block in document["axes"].items():
        metrics = json.loads(json.dumps(block["metrics"], sort_keys=True))
        for wk in _WALLCLOCK_KEYS:
            metrics.pop(wk, None)
        if axis == "a9_latency":
            # A9's headline is wall-clock; keep only the determinism-stable correctness fields
            # (sizes + per-size gold-hit rate + probe counts), drop the millisecond timings.
            by_size = {}
            for size, rec in metrics.get("by_size", {}).items():
                by_size[size] = {
                    "size": rec.get("size"),
                    "ingest_items": rec.get("ingest_items"),
                    "k": rec.get("k"),
                    "probes": rec.get("probes"),
                    "gold_hits": rec.get("gold_hits"),
                    "gold_hit_rate": rec.get("gold_hit_rate"),
                }
            metrics = {"sizes": metrics.get("sizes"), "by_size": by_size}
        if axis == "a7_distill":
            # The judge keep_rate is deterministic at temp 0, but transcripts carry raw model
            # prose; keep the keep_rate + programmatic F1, drop the transcript bodies.
            judge = metrics.get("judge", {})
            metrics = {k: v for k, v in metrics.items() if k != "judge"}
            metrics["judge_keep_rate"] = judge.get("keep_rate")
            metrics["judge_available"] = judge.get("available")
        out[axis] = metrics
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the SAM/IA capability benchmark (all 9 axes).")
    ap.add_argument("--seed", type=int, default=1337,
                    help="run seed (recorded in the manifest; passed to seeded axes)")
    ap.add_argument("--sizes", default="100,1000,10000",
                    help="A9 store sizes, comma-separated (default 100,1000,10000)")
    ap.add_argument("--axes", default=None,
                    help="comma-separated axis subset (a1..a9 or full names); default all")
    ap.add_argument("--out", default=str(RESULTS_DIR / "scores.json"),
                    help="path to write the scores document (default results/scores.json)")
    ap.add_argument("--check-determinism", action="store_true",
                    help="run the selected axes TWICE and assert the scoring subtree matches")
    args = ap.parse_args(argv)

    axes = _resolve_axes(args.axes)
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())

    document = run(axes, args.seed, sizes)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print(f"[run] wrote {out_path}")

    if args.check_determinism:
        print("[determinism] second pass ...", flush=True)
        document2 = run(axes, args.seed, sizes)
        view1 = _scoring_view(document)
        view2 = _scoring_view(document2)
        s1 = json.dumps(view1, sort_keys=True)
        s2 = json.dumps(view2, sort_keys=True)
        if s1 == s2:
            print("[determinism] OK — scoring subtree byte-identical across two runs.")
        else:
            (RESULTS_DIR / "determinism_run1.json").write_text(
                json.dumps(view1, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            (RESULTS_DIR / "determinism_run2.json").write_text(
                json.dumps(view2, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            print("[determinism] FAIL — scoring subtree differs across runs; "
                  "wrote determinism_run1.json + determinism_run2.json for diffing.",
                  file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
