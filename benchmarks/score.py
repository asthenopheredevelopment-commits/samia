"""Programmatic scorers for the SAM/IA capability benchmark.

The benchmark's first scoring rule (design doc §4) is *programmatic scoring first*:
exact-id match over the recall ranking, never a reader/judge over generated prose. That is
what keeps the numbers free of the reader/judge confound (defect D5). A pinned local judge
is reserved ONLY for the genuinely open-ended axes (A3/A4/A7); the retrieval-style axes —
including A5 (consolidation gain) — score entirely on the returned id ranking and need no
model at score time.

This module owns those programmatic primitives:

* ``hit_at_k`` / ``recall_at_k`` — is the gold id within the top-k of a ranking.
* ``reciprocal_rank`` / ``mrr`` — mean reciprocal rank over a probe set.
* ``recall_at_k_set`` — recall@k averaged over many (ranking, gold) probes, for each k.
* ``delta_recall`` — the A5 metric: per-k recall AFTER minus recall BEFORE a cycle.
* ``percentile`` / ``score_a9`` — the A9 metric: wall-clock latency distribution
  (p50/p95 recall ms) + ingest throughput (items/s) per store size. Purely numeric over
  committed raw timings; no ranking, no judge.

The programmatic primitives above are pure (stdlib only) and deterministic: same rankings +
same gold → same score, no network, no model.

This module ALSO owns the single **pinned-judge wrapper** (``PINNED_JUDGE_MODEL`` /
``run_pinned_judge`` / ``score_judge_subset``) used ONLY for the open-ended axes A3/A4/A7. It
pins one local model by name, calls it at temperature 0 with a fixed seed (deterministic),
parses the GRADE:/VERDICT: contract, and **saves every transcript** so any reader effect is
auditable + re-scoreable — and it never gates a programmatic number. It is import-lazy (the
transport is only loaded when a judge call is actually made, so a purely-programmatic axis
like A5 never loads it) and it never reaches the network (it talks only to an already-running
local daemon, reporting the subset as pending if that daemon is down — never fabricated).
"""

from __future__ import annotations

from typing import Iterable, Sequence


def hit_at_k(ranking: Sequence[str], gold_id: str, k: int) -> bool:
    """True iff ``gold_id`` appears within the first ``k`` entries of ``ranking``.

    A single-gold retrieval hit: the recall ranking is a best-first list of memory ids
    and the probe has exactly one correct id (clean gold labels, defect D1/D2/D4). An
    empty ranking is a clean miss (recall returned nothing / no index) — never an error.
    """
    return gold_id in ranking[:k]


def recall_at_k(ranking: Sequence[str], gold_id: str, k: int) -> float:
    """recall@k for one single-gold probe: 1.0 if the gold is in the top-k else 0.0.

    With exactly one relevant id per probe, recall@k is the hit indicator; we return it as
    a float so it averages cleanly across a probe set.
    """
    return 1.0 if hit_at_k(ranking, gold_id, k) else 0.0


def reciprocal_rank(ranking: Sequence[str], gold_id: str) -> float:
    """1/(rank of gold) with rank counted from 1; 0.0 when the gold is absent.

    The reciprocal-rank contribution of one probe to MRR. Rewards ranking the single gold
    id higher, not merely including it — the complement to recall@k's binary in-or-out.
    """
    for i, node in enumerate(ranking):
        if node == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def mrr(rankings: Iterable[Sequence[str]], golds: Iterable[str]) -> float:
    """Mean reciprocal rank over paired (ranking, gold) probes; 0.0 over an empty set."""
    rr = [reciprocal_rank(r, g) for r, g in zip(rankings, golds)]
    return sum(rr) / len(rr) if rr else 0.0


def recall_at_k_set(rankings: Sequence[Sequence[str]], golds: Sequence[str],
                    k_values: Iterable[int]) -> dict[int, float]:
    """recall@k averaged over a probe set, for each k in ``k_values``.

    Parameters
    ----------
    rankings, golds:
        Parallel sequences: ``rankings[i]`` is the best-first id list for the i-th probe
        and ``golds[i]`` is that probe's single gold id.
    k_values:
        The cutoffs to report (e.g. ``(1, 5, 10)``).

    Returns ``{k: mean recall@k}``. An empty probe set yields 0.0 at every k.
    """
    n = len(rankings)
    out: dict[int, float] = {}
    for k in k_values:
        if n == 0:
            out[k] = 0.0
            continue
        hits = sum(recall_at_k(rankings[i], golds[i], k) for i in range(n))
        out[k] = hits / n
    return out


def delta_recall(before: dict[int, float], after: dict[int, float]) -> dict[int, float]:
    """The A5 consolidation-gain metric: per-k recall AFTER minus recall BEFORE.

    ``before``/``after`` are ``recall_at_k_set`` maps measured on the SAME probe set, the
    only difference being a ``consolidate()`` call between them. A positive value is a
    consolidation lift; ~0.0 means the pass was recall-neutral; negative means it hurt
    recall. The keys are the shared k cutoffs.
    """
    return {k: round(after[k] - before.get(k, 0.0), 6) for k in after}


# --- A9: latency / scale (wall-clock, no ranking, no judge) -------------------

def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of ``values`` (``pct`` in [0, 100]).

    Deterministic and dependency-free (no numpy) so the latency stats reproduce exactly on
    any machine from the committed raw timings. ``values`` need not be pre-sorted; an empty
    input yields ``0.0``. Uses the standard "linear interpolation between closest ranks"
    method (the one numpy's default ``percentile`` uses), so p50 of an even-length list is
    the mean of the two middle values, not an arbitrary pick.
    """
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _latency_stats(recall_ms: Sequence[float]) -> dict:
    """p50 / p95 / mean / min / max over per-recall millisecond timings (0.0s if empty)."""
    if not recall_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0,
                "min_ms": 0.0, "max_ms": 0.0}
    vals = [float(x) for x in recall_ms]
    return {
        "p50_ms": round(percentile(vals, 50), 3),
        "p95_ms": round(percentile(vals, 95), 3),
        "mean_ms": round(sum(vals) / len(vals), 3),
        "min_ms": round(min(vals), 3),
        "max_ms": round(max(vals), 3),
    }


def score_a9(raw: Iterable[dict]) -> dict:
    """Score the A9 (latency / scale) axis from one raw record per store size.

    Each input record (the A9 task module emits exactly this shape) carries::

        {
          "size": int,                  # store size N (100 / 1000 / 10000)
          "ingest_items": int,          # facts stored at this size
          "ingest_seconds": float,      # wall-clock to store + build the index
          "index_build_seconds": float, # wall-clock of the index build alone (a subset)
          "recall_ms": [float, ...],    # per-probe recall wall-clock, milliseconds
          "gold_hits": int,             # probes whose gold id was in top-k (correctness)
          "probes": int,                # number of probes timed
          "k": int                      # recall k used
        }

    Returns per-size ingest throughput (items/s) and the recall latency distribution
    (p50/p95/mean/min/max ms). It also reports ``gold_hit_rate`` so the latency is always
    presented next to proof the recall path actually returned the right memory at that
    scale — a fast recall that finds nothing is not a latency success (honesty rail). A9's
    headline metric per the design table is ``recall_p50_ms`` / ``recall_p95_ms`` and
    ``ingest_items_per_s``. Purely numeric: no id ranking, no model, fully reproducible
    from the committed raw timings.
    """
    by_size: dict[str, dict] = {}
    for rec in raw:
        size = int(rec["size"])
        recall_ms = list(rec.get("recall_ms", []))
        ingest_s = float(rec.get("ingest_seconds", 0.0))
        items = int(rec.get("ingest_items", 0))
        probes = int(rec.get("probes", len(recall_ms)))
        gold_hits = int(rec.get("gold_hits", 0))
        stats = _latency_stats(recall_ms)
        by_size[str(size)] = {
            "size": size,
            "ingest_items": items,
            "ingest_seconds": round(ingest_s, 3),
            "index_build_seconds": round(float(rec.get("index_build_seconds", 0.0)), 3),
            "ingest_items_per_s": round(items / ingest_s, 2) if ingest_s > 0 else 0.0,
            "k": int(rec.get("k", 0)),
            "probes": probes,
            "gold_hits": gold_hits,
            "gold_hit_rate": round(gold_hits / probes, 4) if probes else 0.0,
            "recall_p50_ms": stats["p50_ms"],
            "recall_p95_ms": stats["p95_ms"],
            "recall_mean_ms": stats["mean_ms"],
            "recall_min_ms": stats["min_ms"],
            "recall_max_ms": stats["max_ms"],
        }
    return {
        "axis": "a9_latency",
        "metric": "p50/p95 recall ms + ingest items/s, per store size",
        "by_size": by_size,
        "sizes": sorted(int(s) for s in by_size),
    }


# --- pinned-judge wrapper (open-ended subset ONLY: A3 / A4 / A7) ---------------
#
# Design rule 4: the LLM judge is used ONLY for the genuinely open-ended axes, with a PINNED
# local model + a FIXED prompt at temperature 0, and EVERY transcript is saved so any reader
# effect is auditable + re-scoreable. The programmatic metric is always the trustworthy
# number; the judge subset is a side-channel that never gates it. This wrapper centralizes the
# pinned model, the deterministic call (temp 0 + fixed seed), the verdict parse, and the
# transcript record so an axis does not reinvent any of it. It is imported lazily (an axis
# that uses no judge never loads the transport), and it NEVER reaches the network: it talks
# only to an already-running local Ollama daemon, and reports the subset as ``pending`` /
# ``unavailable`` (never fabricated) if that daemon is not up.

#: The single pinned local judge model, by name. One named model scores every open-ended
#: subset so a judge number is always attributable. Temperature 0 + fixed seed make it
#: deterministic; it is a LOCAL model (no internet).
PINNED_JUDGE_MODEL = "phi4-mini:latest"
_JUDGE_SEED = 1337


def judge_backend_reachable(timeout_s: float = 2.0) -> bool:
    """True iff the pinned local judge daemon is already running (no network, no daemon start).

    Probes only the local Ollama socket via the package's own reachability check — it does NOT
    download a model or start a daemon, so a deterministic offline run stays offline and the
    caller reports the judge subset as pending instead of fabricating a grade.
    """
    try:
        from samia.core import judge as _judge
        return bool(_judge._ollama_reachable(timeout_s=timeout_s))
    except Exception:
        return False


def run_pinned_judge(prompt: str, *, num_predict: int = 120) -> tuple[str, str]:
    """Send ``prompt`` to the pinned judge at temperature 0 / fixed seed; return (verdict, raw).

    ``verdict`` is one of ``correct`` / ``incorrect`` / ``keep`` / ``drop`` / ``unsure`` parsed
    from the model's reply (the open-ended axes use a GRADE: or VERDICT: contract). ``raw`` is
    the full model response, returned so the transcript is complete and re-scoreable. A
    transport error returns ``("unsure", "")`` — the caller records it as an unresolved judge
    item rather than a fabricated grade. Deterministic: temp 0 + a fixed seed + a pinned model.
    """
    import json as _json
    import re as _re
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    from samia.core import judge as _judge

    payload = {
        "model": PINNED_JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": _JUDGE_SEED, "num_predict": num_predict},
    }
    req = _urlreq.Request(
        f"{_judge.OLLAMA_URL}/api/generate",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=30.0) as r:
            raw = _json.loads(r.read().decode("utf-8")).get("response", "")
    except (_urlerr.URLError, OSError, ValueError) as exc:
        return "unsure", f"<judge transport error: {type(exc).__name__}>"
    up = raw.upper()
    if "GRADE: CORRECT" in up:
        return "correct", raw
    if "GRADE: INCORRECT" in up:
        return "incorrect", raw
    if "VERDICT: KEEP" in up or _re.search(r"\bKEEP\b", up):
        return "keep", raw
    if "VERDICT: DROP" in up or _re.search(r"\bDROP\b", up):
        return "drop", raw
    if _re.search(r"\bCORRECT\b", up) and not _re.search(r"\bINCORRECT\b", up):
        return "correct", raw
    if _re.search(r"\bINCORRECT\b", up):
        return "incorrect", raw
    return "unsure", raw


def score_judge_subset(items: Iterable[dict], *, prompt_key: str = "prompt",
                       transcript_path=None) -> dict:
    """Score an open-ended judge subset with the pinned judge; save every transcript.

    ``items`` are dicts each carrying at least a ``prompt`` (the fully-rendered, fixed judge
    prompt) and whatever id/gold fields the axis wants preserved in the transcript. Returns a
    summary ``{available, model, n, verdict_counts, transcripts, note}``. When the judge
    backend is not reachable the subset is reported ``available=False`` with a reason and NO
    fabricated grades (the programmatic metric stands alone). If ``transcript_path`` is given,
    the transcripts are also written there as pretty JSON so they are independently auditable.
    """
    items = list(items)
    if not items:
        return {"available": False, "model": PINNED_JUDGE_MODEL, "n": 0,
                "verdict_counts": {}, "transcripts": [],
                "note": "no open-ended items in this subset"}
    if not judge_backend_reachable():
        return {"available": False, "model": PINNED_JUDGE_MODEL, "n": len(items),
                "verdict_counts": {}, "transcripts": [],
                "note": ("pinned judge daemon not reachable; subset reported pending "
                         "(not fabricated). Programmatic metrics are the trustworthy numbers.")}
    transcripts: list[dict] = []
    counts: dict[str, int] = {}
    for it in items:
        verdict, raw = run_pinned_judge(it[prompt_key])
        counts[verdict] = counts.get(verdict, 0) + 1
        rec = {k: v for k, v in it.items() if k != prompt_key}
        rec["verdict"] = verdict
        rec["raw"] = raw
        transcripts.append(rec)
    summary = {"available": True, "model": PINNED_JUDGE_MODEL, "n": len(items),
               "verdict_counts": counts, "transcripts": transcripts,
               "note": "pinned judge (temp 0, fixed seed); programmatic metric is primary"}
    if transcript_path is not None:
        import json as _json
        from pathlib import Path as _Path
        p = _Path(transcript_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
                     encoding="utf-8")
    return summary
