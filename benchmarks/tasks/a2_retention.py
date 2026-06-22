"""A2 — Retention / forgetting axis (deterministic, seeded, network-free).

What the axis measures
----------------------
Retention is NOT retrieval (defect D6). A1 asks "with everything present, does the right
memory rank top-k". A2 asks the orthogonal question: after a delay during which the
system's forgetting curve prunes the store, does the **salient** memory survive while the
**noise** is forgotten? The two reported numbers are:

* ``retention@delay`` — fraction of salient golds still recallable in top-k after a delay.
* ``noise-drop rate``  — fraction of noise atoms the forgetting curve evicted (no longer
  in the live store / no longer recallable) after the same delay.

A good memory keeps retention high and noise-drop high at the same delay: it holds the
important and sheds the trivial.

How it exercises the REAL SAM/IA surface
----------------------------------------
The "delay" runs against SAM/IA's actual relevance-decay surface, ``samia.core.tier``:

* Each stored atom is a ``type: semantic`` node (the shape the adapter and the system's own
  ``test_semantic_recall`` plant) carrying the decay frontmatter the pass reads:
  ``relevance``, ``tier``, ``last_access``, ``material_grade``, and the ``salience`` scalar
  the Tier-1 salience source writes. Salient atoms get ``salience=1.0``; noise gets ``0.0``.
* A delay of *t* ticks ages every node's ``last_access`` past the warm-freshness window and
  runs ``tier.decay_pass(nodes_dir, dry=False, auto_freeze=True)`` *t* times. Per the
  package's own constants, a zero-salience node decays toward 0 and, once it crosses the
  freeze threshold, is archived out of ``nodes/`` by ``ia.freeze`` (the eviction). A
  salience-1.0 node decays an order of magnitude slower AND is freeze-exempt
  (``salience >= SALIENCE_FREEZE_EXEMPT``), so it stays resident.
* Recall then goes through the adapter — the real MiniLM vector index over the surviving
  nodes + the ``semantic_recall.atom_retrieve`` semantic arm — so a salient atom only
  "retains" if it both survived the decay AND is still recallable for its probe.

Every delay point is measured on a FRESH store (reset → write the aged corpus → run that
delay's ticks → recall), so each retention@delay is an independent, deterministic function
of the dataset — no cumulative cross-delay coupling.

Scoring
-------
Fully programmatic on the returned id ranking (defect D5: no reader/judge — A2 is not an
open-ended axis). ``retention@delay`` and ``noise-drop`` come straight from the recall
rankings and the surviving-node set; the scorers live in ``benchmarks.score`` (shared) plus
the two A2-specific helpers below. No model, no network at score time.

Determinism
-----------
The dataset is fixed + checksum-verified before the run; the embedder is the pinned MiniLM
(cache-only, autofetch off in the adapter); the decay math is a pure function of the node
frontmatter; the store order is a fixed-seed interleave baked into the dataset. Same inputs
→ same numbers.

Run standalone (smoke):

    python benchmarks/tasks/a2_retention.py            # human-readable summary
    python benchmarks/tasks/a2_retention.py --json     # machine-readable result

Exit code 0 iff the axis produced a result; non-zero on a load/checksum/IO failure.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

# Make the harness package importable whether this module is run as a file
# (``python benchmarks/tasks/a2_retention.py``) or imported as
# ``benchmarks.tasks.a2_retention``.
_BENCH_DIR = Path(__file__).resolve().parents[1]
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from adapters import MemoryAdapter, MemoryItem, SamiaAdapter  # noqa: E402
import score as _score  # noqa: E402

# The system's real relevance-decay surface — the forgetting curve A2 exercises. Imported
# from the INSTALLED package (the adapter already proved the package is importable); these
# are the exact functions the runtime daemon's tier-decay job calls.
from samia.core import tier as _tier  # noqa: E402

AXIS = "a2_retention"
_DATASET_PATH = _BENCH_DIR / "data" / "a2_retention" / "dataset.json"
_SUMS_PATH = _BENCH_DIR / "data" / "a2_retention" / "SHA256SUMS"


# --------------------------------------------------------------------------- dataset load


def load_dataset(verify_checksum: bool = True) -> dict:
    """Load the fixed A2 dataset, verifying its SHA256 against the committed manifest.

    The committed ``SHA256SUMS`` pins the exact dataset bytes; a mismatch means the data
    drifted from what was measured and the axis must NOT run on it (the determinism rule).
    Raises ``RuntimeError`` on a mismatch or missing manifest, ``FileNotFoundError`` if the
    dataset is absent. Pass ``verify_checksum=False`` only for local authoring.
    """
    data = _DATASET_PATH.read_bytes()
    if verify_checksum:
        digest = hashlib.sha256(data).hexdigest()
        expected = _read_expected_digest(_SUMS_PATH, "dataset.json")
        if expected is None:
            raise RuntimeError(f"no checksum for dataset.json in {_SUMS_PATH}")
        if digest != expected:
            raise RuntimeError(
                f"A2 dataset checksum mismatch: got {digest}, expected {expected}. "
                "Regenerate with data/a2_retention/generate.py or restore the committed file."
            )
    return json.loads(data.decode("utf-8"))


def _read_expected_digest(sums_path: Path, name: str) -> str | None:
    """Return the expected SHA256 for ``name`` from a ``sha256sum``-format manifest."""
    if not sums_path.exists():
        return None
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            return parts[0]
    return None


# ----------------------------------------------------------------- decay-aware node writer


def _decay_node_text(item: dict, last_access: str, relevance: float) -> str:
    """Render a dataset item as a ``type: semantic`` atom node with decay frontmatter.

    This is the same atom shape the adapter and ``test_semantic_recall`` plant (frontmatter
    ``name`` + ``type: semantic`` over a body, read by ``semantic_recall._node_type`` /
    ``_atom_fields``), EXTENDED with the fields ``tier.decay_pass`` reads so the real
    forgetting curve operates on it: ``salience`` (the [0,1] importance scalar), ``relevance``
    + ``tier`` (the decaying state), ``last_access`` (age input), and ``material_grade`` (the
    per-grade decay rate; ``natural`` is the package default). A2 stamps these because they
    are decay-specific — the neutral adapter does not carry them; the recall path stays the
    adapter's real semantic arm.
    """
    lines = [
        f"name: {item['id']}",
        "type: semantic",
        f"salience: {float(item.get('salience', 0.0))}",
        f"relevance: {float(relevance)}",
        f"tier: {_tier.tier_for(float(relevance))}",
        f"last_access: {last_access}",
        "material_grade: natural",
    ]
    if item.get("source"):
        lines.append(f"source: {item['source']}")
    if item.get("valid_from"):
        lines.append(f"valid_from: {item['valid_from']}")
    fm = "---\n" + "\n".join(lines) + "\n---\n"
    return fm + str(item["text"]).strip() + "\n"


# An access date placed well before the run date so every node is OUTSIDE the
# warm-freshness window (WARM_FRESHNESS_DAYS) on the very first tick — i.e. every node is
# already "aged" and in the stale-decay regime when the delay begins. Fixed + in the past so
# the delay is deterministic and independent of the wall clock at run time.
_AGED_LAST_ACCESS = "2020-01-01"
# The decay date the ticks are evaluated against — far enough after _AGED_LAST_ACCESS that
# days_since_access >> WARM_FRESHNESS_DAYS. Fixed so the run is wall-clock independent.
_DECAY_TODAY = "2024-01-01"


def _populate_aged(root: Path, items: list[dict]) -> None:
    """Write every item as an aged decay node into ``root/nodes`` (caller resets first)."""
    nodes_dir = root / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        # All nodes start at the same fresh relevance (NEUTRAL); the decay pass is what
        # differentiates salient (dampened, exempt) from noise (decays + freezes).
        text = _decay_node_text(it, _AGED_LAST_ACCESS, _tier.NEUTRAL)
        (nodes_dir / f"{it['id']}.md").write_text(text, encoding="utf-8")


def _run_decay_ticks(root: Path, ticks: int) -> None:
    """Apply the real relevance-decay pass ``ticks`` times over ``root/nodes``.

    Each call is ``tier.decay_pass(nodes_dir, dry=False, today=_DECAY_TODAY,
    auto_freeze=True)`` — the exact forgetting-curve step the runtime daemon runs: it
    rewrites each node's relevance/tier and archives (evicts) nodes that cross the freeze
    threshold and are not salience-exempt. ``ticks=0`` is a no-op (the baseline).
    """
    nodes_dir = root / "nodes"
    for _ in range(ticks):
        _tier.decay_pass(nodes_dir, dry=False, today=_DECAY_TODAY, auto_freeze=True)


# --------------------------------------------------------------- A2-specific programmatic scorers


def retention_at_delay(rankings: list[list[str]], golds: list[str],
                       k_values: tuple[int, ...]) -> dict[int, float]:
    """Fraction of salient golds still recallable in top-k AFTER a delay, per k.

    A salient atom is "retained" at this delay iff its gold id is still within the post-delay
    recall top-k. This is exactly ``recall@k`` over the salient probe set measured on the
    decayed store, so it reuses the shared programmatic primitive — the A2 distinction is
    *when* it is measured (after the forgetting curve ran), not *how* it is scored.
    """
    return _score.recall_at_k_set(rankings, golds, k_values)


def noise_drop_rate(noise_ids: list[str], surviving_ids: set[str]) -> float:
    """Fraction of noise atoms the forgetting curve evicted from the live store.

    ``surviving_ids`` is the set of node ids still present in ``nodes/`` after the delay.
    A noise atom counts as "dropped" iff it is no longer present (the decay pass archived +
    unlinked it). Returns dropped/total over the noise population; 0.0 for an empty noise set
    (no atoms to forget). This reads the live-store membership directly — a dropped noise
    atom is gone from disk, so it can never be recalled either; membership is the ground
    truth and is judge-free.
    """
    if not noise_ids:
        return 0.0
    dropped = sum(1 for nid in noise_ids if nid not in surviving_ids)
    return dropped / len(noise_ids)


# --------------------------------------------------------------------------------- the run


def run(adapter: MemoryAdapter | None = None, verify_checksum: bool = True) -> dict:
    """Run the A2 retention axis end-to-end and return a JSON-able result dict.

    For the baseline (delay 0) and each delay in the dataset's ``delay_ticks`` schedule:
    reset to an empty store, write the aged corpus, run that many decay ticks, rebuild the
    index over the survivors, recall every salient probe, and score retention@k + noise-drop.
    Each delay is an independent fresh store so the numbers do not couple across delays.

    ``adapter`` defaults to a fresh ``SamiaAdapter`` over an isolated temp root (owned + auto
    cleaned). Pass one to target a different memory system (comparison phase) — the task is
    adapter-agnostic; only ``store``/``recall``/``reset`` and an isolated root are used. When
    the adapter is the SamiaAdapter, its ``_root`` is the decay surface's ``nodes/`` dir.
    """
    ds = load_dataset(verify_checksum=verify_checksum)
    items = ds["items"]
    probes = ds["probes"]
    k_values = tuple(ds["k_values"])
    delay_ticks = list(ds["delay_ticks"])
    salient_ids = list(ds["salient_ids"])
    noise_ids = list(ds["noise_ids"])
    max_k = max(k_values)

    probe_queries = [p["probe"] for p in probes]
    probe_golds = [p["gold_id"] for p in probes]

    owns_adapter = adapter is None
    if adapter is None:
        adapter = SamiaAdapter()

    # The installed package logs index/freeze progress to stdout (``[vector_index] ...``,
    # ``[ia] froze ...``, embed batch timings). That chatter is non-deterministic (wall-clock
    # batch times) and would corrupt a ``--json`` stdout, so route it to stderr for the whole
    # run; the clean result goes to stdout only at the end.
    _stdout_to_stderr = contextlib.redirect_stdout(sys.stderr)

    # The decay surface operates on the adapter's memory root nodes/. The SamiaAdapter owns
    # an isolated temp root; A2 reaches it to drive the real decay pass (the adapter does not
    # carry decay frontmatter — that is A2-specific). Guard so a non-samia adapter degrades
    # to a recall-only retention measure rather than erroring.
    root = getattr(adapter, "_root", None)
    decay_capable = isinstance(adapter, SamiaAdapter) and root is not None

    by_delay: list[dict] = []
    try:
        # delay 0 (baseline) then each scheduled delay, each on a fresh store. The
        # redirect keeps the package's index/freeze log chatter off stdout (so --json is
        # clean) for the whole measurement loop.
        with _stdout_to_stderr:
            for ticks in [0] + delay_ticks:
                adapter.reset()
                if decay_capable:
                    _populate_aged(root, items)
                    _run_decay_ticks(root, ticks)
                    # The decay pass mutated nodes/ behind the adapter's back (rewrote
                    # relevance, evicted frozen nodes). Force a fresh index over the
                    # survivors so recall reflects exactly the post-delay live store.
                    adapter._dirty = True  # noqa: SLF001 — task drives the index rebuild
                    surviving = {p.stem for p in (root / "nodes").glob("*.md")}
                else:
                    # Adapter without a decay surface: store via the contract and treat
                    # all items as surviving (no forgetting capability to exercise).
                    adapter.store([
                        MemoryItem(id=it["id"], text=it["text"],
                                   valid_from=it.get("valid_from", ""),
                                   source=it.get("source", ""),
                                   trusted=bool(it.get("trusted", True)),
                                   meta={"kind": it.get("kind", "")})
                        for it in items
                    ])
                    surviving = {it["id"] for it in items}

                rankings = [adapter.recall(q, k=max_k) for q in probe_queries]
                retention = retention_at_delay(rankings, probe_golds, k_values)
                drop = noise_drop_rate(noise_ids, surviving)
                salient_surviving = sum(1 for sid in salient_ids if sid in surviving)

                by_delay.append({
                    "delay_ticks": ticks,
                    "retention_at_k": {str(k): round(v, 6) for k, v in retention.items()},
                    "noise_drop_rate": round(drop, 6),
                    "salient_surviving": salient_surviving,
                    "salient_total": len(salient_ids),
                    "noise_surviving": sum(1 for nid in noise_ids if nid in surviving),
                    "noise_total": len(noise_ids),
                    "rankings": rankings,
                    "golds": probe_golds,
                })
    finally:
        if owns_adapter:
            adapter.reset()
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    return {
        "axis": AXIS,
        "adapter": getattr(adapter, "name", "unknown"),
        "decay_capable": decay_capable,
        "schema_version": ds.get("schema_version"),
        "seed": ds.get("seed"),
        "k_values": list(k_values),
        "delay_schedule": [0] + delay_ticks,
        "salient_count": len(salient_ids),
        "noise_count": len(noise_ids),
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "judge_used": False,
        "by_delay": by_delay,
    }


# ----------------------------------------------------------------------------- CLI / report


def _summary_lines(result: dict) -> list[str]:
    """Render a compact human-readable retention/forgetting report from a result dict."""
    lines: list[str] = []
    lines.append(f"axis: {result['axis']}  adapter: {result['adapter']}  "
                 f"decay_capable: {result['decay_capable']}")
    lines.append(f"salient={result['salient_count']} noise={result['noise_count']} "
                 f"k={result['k_values']} judge_used={result['judge_used']}")
    ks = result["k_values"]
    header = "  delay |  " + "  ".join(f"ret@{k}" for k in ks) + " | noise-drop | salient-live"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row in result["by_delay"]:
        ret = "  ".join(f"{row['retention_at_k'][str(k)]:.3f}" for k in ks)
        lines.append(
            f"  {row['delay_ticks']:>5} |  {ret} |   {row['noise_drop_rate']:.3f}    | "
            f"{row['salient_surviving']}/{row['salient_total']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the A2 retention/forgetting axis.")
    parser.add_argument("--json", action="store_true",
                        help="emit the full machine-readable result as JSON")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the dataset checksum check (local authoring only)")
    args = parser.parse_args(argv)

    result = run(verify_checksum=not args.no_verify)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=True))
    else:
        for line in _summary_lines(result):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
