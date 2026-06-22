"""A4 — Contradiction / belief-update axis (deterministic, network-free).

Task (per ``BENCHMARK_DESIGN_v1.md`` A4): assert a fact ``X``; later assert ``¬X`` carrying
*more evidence*; the system should demote the stale claim and serve the updated belief.
Metrics: **demote-correct %** and **shadow-persist %** (both exact-id, programmatic).

This module drives SAM/IA's REAL contradiction/supersession surfaces through the
``SamiaAdapter`` (the installed ``samia`` package), with NO LLM in the loop, so the run is
deterministic and offline. For each case it:

1. **Asserts X** — stores the original claim plus unrelated distractors and builds the
   index (the prior belief state).
2. **Detects the contradiction** — calls ``samia.runtime.contradiction``'s embedding
   supersession finder (``find_supersession_candidates``) with not_X's parallel claim text
   over the whole index; a hit pairs not_X with the correct OLD node.
3. **Picks the loser deterministically** — calls the package's own
   ``_pick_superseded(memory_dir, old_id, new_id)`` rule (older ``valid_from`` /
   lower confidence loses), which is exactly how the passive sweep chooses what to retire.
4. **Asserts ¬X and demotes the loser** — stores the updated claim, then demotes the loser
   via ``samia.core.vector.tombstone_node`` (the network-free half of SAM/IA's RESTORABLE
   supersede cascade: ``query()`` excludes a tombstoned entry immediately, and
   ``restore_node`` can un-forget it).
5. **Queries** — recalls the probe through the semantic atom arm and measures whether the
   UPDATED belief is served at rank 1 with the OLD claim gone (demote-correct), and whether
   the demoted OLD claim still leaks back (shadow-persist).

Why the embedding finder, not the full passive sweep: SAM/IA's *auto*-supersede
(``passive_sweep``) is gated behind an offline LLM REM judge (``judge_contradictions``),
which a deterministic, network-free benchmark must not invoke (same constraint A5 documents
for the consolidation judge). So A4 exercises every DETERMINISTIC arm of the pipeline — the
embedding detector, the loser-pick rule, and the restorable tombstone demote — and reports
the judge-gated auto path separately, as measured-without-judge (no fabricated number).

Determinism + defect fixes:

* **D6 (retrieval != retention):** A4 is belief-update over its OWN ``cases.jsonl``; it does
  not reuse the A1 retrieval data or the A2 retention data, and nothing is interleaved over
  time. Each case is independent (``reset`` between cases).
* **D5 (no reader/judge confound):** every metric is an id comparison; the scorer reads ids,
  never prose, and no LLM judge runs. The pinned judge is N/A for A4 (recorded explicitly).
* **D1/D2/D4 (clean gold):** each case is a single-attribute flip with one unambiguous OLD
  and NEW id; distractors are on distinct subjects so they can never be the demotion target.
* **No network at score time:** the embedder is the cache-only MiniLM (the adapter sets
  ``ASTHENOS_MODEL_AUTOFETCH=0``); the supersession finder reuses that same cached embedder.

Env knob (documented, restored after the run): SAM/IA's supersession finder applies a higher
per-population cosine bar (``ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD``, default 0.92) to
``type: semantic`` atoms, because at the recall-first bar machine-templated atoms flood. The
A4 cases are HAND-WRITTEN contradiction claims (not template-backfilled), so this task sets
that bar to the operator's hand-written cosine floor (0.57) for the detection step, exactly
the bar the finder uses for hand-written pairs. It captures and restores the prior env value
so it never leaks onto an outer process. This is a configuration choice, not a code change to
the package, and it is reported in the result metadata.

Run standalone::

    python benchmarks/tasks/a4_contradiction.py

Prints the scores JSON. Re-run -> identical numbers (deterministic).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the benchmark package importable when run as a standalone script.
_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from adapters import MemoryItem, SamiaAdapter  # noqa: E402

_DATA_DIR = _BENCH_ROOT / "data" / "a4_contradiction"
_CASES_PATH = _DATA_DIR / "cases.jsonl"
_SUMS_PATH = _DATA_DIR / "SHA256SUMS"

# The env var that sets SAM/IA's per-population (semantic-atom) supersession cosine bar.
# Default in the package is 0.92; we use the operator's hand-written floor (0.57) for the
# hand-written A4 claims. Captured + restored around the run (never leaked).
_SEMANTIC_THR_ENV = "ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD"
_HANDWRITTEN_COSINE_BAR = "0.57"

# Recall depth for the post-demotion query. k=10 matches the design's recall@10 surface and
# is large enough that a leaked shadow would be caught even if it ranked low.
_RECALL_K = 10


# --------------------------------------------------------------------------------------
# Dataset loading (with checksum verification — no silent drift)
# --------------------------------------------------------------------------------------

def _verify_and_load_cases() -> list[dict[str, Any]]:
    """Load ``cases.jsonl`` after asserting it matches the committed SHA256.

    A tampered or regenerated-but-uncommitted dataset fails loudly here rather than quietly
    changing a benchmark number. Returns the parsed case list.
    """
    if not _CASES_PATH.exists():
        raise FileNotFoundError(
            f"A4 dataset missing: {_CASES_PATH}. Run "
            f"`python benchmarks/data/a4_contradiction/generate.py` first.")
    raw = _CASES_PATH.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if _SUMS_PATH.exists():
        expected = _SUMS_PATH.read_text(encoding="utf-8").split()[0]
        if digest != expected:
            raise ValueError(
                f"A4 dataset checksum mismatch: {_CASES_PATH} is {digest}, "
                f"SHA256SUMS expects {expected}. Regenerate + recommit the dataset.")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


# --------------------------------------------------------------------------------------
# SAM/IA contradiction surfaces (imported from the installed package, not reimplemented)
# --------------------------------------------------------------------------------------

def _contradiction_module():
    """Import the installed contradiction package, re-reading the env-derived bars.

    The finder's per-population semantic bar is a module-level constant read from the env at
    import; ``importlib.reload`` re-executes config so a bar set just before this call is
    honored. Returns ``(facade_module, find_supersession_candidates, _pick_superseded)``.
    """
    import samia.runtime.contradiction as con
    importlib.reload(con)  # re-read ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD etc.
    from samia.runtime.contradiction import find_supersession_candidates, configure
    from samia.runtime.contradiction.passes import _pick_superseded
    return con, find_supersession_candidates, configure, _pick_superseded


# --------------------------------------------------------------------------------------
# The A4 run
# --------------------------------------------------------------------------------------

def _run_case(adapter: SamiaAdapter, case: dict[str, Any]) -> dict[str, Any]:
    """Run one belief-update case end to end and return its id-level observations.

    Steps (see module docstring): assert X + distractors -> detect supersession on not_X's
    claim -> deterministic loser pick -> assert ¬X + tombstone the loser -> recall the probe.
    Records the four booleans the scorer reads plus a small audit trail.
    """
    old = case["old"]
    new = case["new"]
    query = case["query"]

    adapter.reset()

    # 1. Assert X (the prior belief) + unrelated distractors; build the index.
    seed_items = [MemoryItem(id=old["id"], text=old["text"],
                             valid_from=old["valid_from"], source=old["source"])]
    seed_items += [MemoryItem(id=d["id"], text=d["text"]) for d in case["distractors"]]
    adapter.store(seed_items)
    adapter.build_index()

    mem = adapter._root  # the isolated temp root the adapter owns

    detected = False
    detect_score: float | None = None
    # 2. Detect the contradiction: pair not_X's parallel claim with the prior store.
    prior_env = adapter._apply_env()
    os.environ[_SEMANTIC_THR_ENV] = _HANDWRITTEN_COSINE_BAR
    try:
        con, find_supersession_candidates, configure, _pick_superseded = \
            _contradiction_module()
        configure(mem)
        cands = find_supersession_candidates(
            new["claim"], scope_nodes=None, memory_dir=mem,
            threshold=float(_HANDWRITTEN_COSINE_BAR))
        for c in cands:
            cand_stem = str(c["node_id"])
            cand_stem = cand_stem[:-3] if cand_stem.endswith(".md") else cand_stem
            if cand_stem == old["id"]:
                detected = True
                detect_score = float(c.get("score", 0.0))
                break
    finally:
        adapter._restore_env(prior_env)

    # 3. Assert ¬X (the updated belief) into the store; rebuild so it is recallable.
    adapter.store([MemoryItem(id=new["id"], text=new["text"],
                              valid_from=new["valid_from"], source=new["source"])])
    adapter.build_index()

    # 4. Deterministic loser/winner pick, then demote the loser (network-free, restorable).
    prior_env = adapter._apply_env()
    os.environ[_SEMANTIC_THR_ENV] = _HANDWRITTEN_COSINE_BAR
    pick_correct = False
    try:
        con, _find, configure, _pick_superseded = _contradiction_module()
        configure(mem)
        loser, winner = _pick_superseded(mem, old["id"], new["id"])
        pick_correct = (loser == old["id"] and winner == new["id"])
        # Demote whatever the rule chose as the loser via the vector tombstone (the
        # restorable, network-free half of SAM/IA's supersede cascade).
        from samia.core.vector import tombstone_node
        tombstone_node(mem, f"{loser}.md")
        demoted_id = loser
    finally:
        adapter._restore_env(prior_env)

    # 5. Query the probe AFTER demotion (semantic atom arm).
    ranked = adapter.recall(query, k=_RECALL_K)
    served_new_first = bool(ranked) and ranked[0] == new["id"]
    old_present = old["id"] in ranked

    # demote-correct: the OLD claim was the demotion target AND recall now serves the NEW
    # belief at rank 1 AND the OLD claim is gone from recall.
    demote_correct = bool(pick_correct and demoted_id == old["id"]
                          and served_new_first and not old_present)
    # shadow-persist: the demoted OLD claim still surfaces in recall (a leaked shadow).
    shadow_persist = bool(old_present)

    return {
        "case_id": case["case_id"],
        "detected": detected,
        "detect_score": detect_score,
        "pick_correct": pick_correct,
        "demoted_id": demoted_id,
        "served_new_first": served_new_first,
        "old_present": old_present,
        "demote_correct": demote_correct,
        "shadow_persist": shadow_persist,
        "ranked": ranked,
    }


def run(seed: int = 1337) -> dict[str, Any]:
    """Run the full A4 axis against SAM/IA and return the scored result dict.

    ``seed`` is accepted for harness-uniformity and recorded in the result; A4's content is a
    fixed reviewed dataset (no RNG draw at run time), so the seed does not change the numbers
    — the determinism comes from the fixed dataset + the pinned cached embedder.
    """
    cases = _verify_and_load_cases()
    raw_observations: list[dict[str, Any]] = []
    with SamiaAdapter() as adapter:
        for case in cases:
            raw_observations.append(_run_case(adapter, case))

    # Programmatic scoring (id-level; no judge).
    from data.a4_contradiction.score import score_cases
    scored = score_cases(raw_observations)

    scored["seed"] = seed
    scored["dataset_sha256"] = (
        _SUMS_PATH.read_text(encoding="utf-8").split()[0]
        if _SUMS_PATH.exists() else None)
    scored["config"] = {
        "semantic_cosine_bar": float(_HANDWRITTEN_COSINE_BAR),
        "semantic_cosine_bar_note": (
            "Hand-written A4 claims use SAM/IA's hand-written cosine floor (0.57) for the "
            "supersession finder's per-population bar, via "
            "ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD (default 0.92 is for template-"
            "backfilled atoms). Env captured + restored; package code unchanged."),
        "recall_k": _RECALL_K,
        "demote_mechanism": "samia.core.vector.tombstone_node (restorable)",
    }
    # Honest scope note: the LLM-judge-gated AUTO-supersede path is out of scope here.
    scored["auto_supersede_judge"] = {
        "exercised": False,
        "note": (
            "SAM/IA's auto-supersede (runtime.contradiction.passive_sweep) is gated behind "
            "an offline LLM REM judge (judge_contradictions), which a deterministic, "
            "network-free benchmark must not invoke (same constraint as A5's consolidation "
            "judge). A4 exercises every DETERMINISTIC arm of the pipeline — the embedding "
            "supersession finder, the _pick_superseded loser rule, and the restorable "
            "tombstone demote — and reports those. The judge-gated end-to-end auto path is "
            "measured-without-judge; no number is fabricated for it."),
    }
    scored["raw"] = raw_observations
    return scored


if __name__ == "__main__":
    result = run()
    # Print the headline + supporting metrics (drop the verbose raw rows for readability;
    # they remain in the returned dict for the harness to persist).
    summary = {k: v for k, v in result.items() if k != "raw"}
    print(json.dumps(summary, indent=2))
