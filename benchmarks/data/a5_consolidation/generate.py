"""Generator for the A5 (consolidation-gain) fixed dataset.

A5 measures recall@k **before vs after** a consolidation cycle (Δrecall). Per the
benchmark design this axis needs its OWN data, separate from A1 (retrieval) and A2
(retention) — conflating those populations is defect D6. So this generator emits a
self-contained corpus + probe set used by nothing else.

What the dataset is
-------------------
* A population of clean, unambiguous semantic facts. Each fact is on a distinct topic
  and carries one gold ``probe`` (a paraphrased question) plus its gold ``id`` — there is
  exactly one correct memory per probe (clean gold labels: defects D1/D2/D4).
* A subset of facts are deliberately authored as **near-duplicate clusters**: two atoms
  that state the same underlying claim in different words (e.g. a fact and its restatement
  from a second session). These clusters are what a consolidation/merge cycle is *supposed*
  to collapse — they are the material the A5 axis exercises the consolidation pass against.
  Each cluster names a single ``canonical`` gold id so the probe still has one unambiguous
  answer whether or not the duplicates were merged.

Determinism
-----------
Content is fully enumerated literal data (no RNG over text), and the only ordering step
uses a fixed seed, so regenerating always produces a byte-identical ``dataset.json``. The
companion ``SHA256SUMS`` pins the bytes; the task module refuses to run on a checksum
mismatch. No network, no model — this is pure data authoring.

Output
------
``dataset.json`` next to this file, plus ``SHA256SUMS`` covering it. Re-run with::

    python benchmarks/data/a5_consolidation/generate.py

and commit the result. The task reads ``dataset.json`` only; this script is the audit
trail for how those bytes were produced.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Fixed seed — the ONLY randomness is a deterministic shuffle of an otherwise fully
# enumerated corpus, so the dataset is reproducible bit-for-bit. Pinned here (not passed
# in) because the COMMITTED dataset must be a single fixed artifact, not seed-dependent.
SEED = 1337

# Schema version travels in the dataset so a future format change is detectable and the
# task can refuse data it does not understand.
SCHEMA_VERSION = 1

# k values the A5 metric reports recall@ at. Small, fixed, and well under the corpus size
# so a miss is a real miss (not a "k larger than the store" artifact).
K_VALUES = (1, 5, 10)


# Singleton facts — one atom, one topic, one gold probe each. These are NOT part of any
# near-duplicate cluster; they are the unambiguous baseline population the consolidation
# pass must leave untouched (a correct consolidation never harms a unique fact's recall).
# (id, text, probe, valid_from, source)
_SINGLETON_FACTS = [
    ("fact_telescope",
     "Lena assembled a reflecting telescope in her garage over the winter.",
     "What did Lena build during the winter?",
     "2022-12-10", "session_a"),
    ("fact_marathon",
     "Marcus finished the coastal marathon in just under four hours.",
     "How long did Marcus take to run the coastal marathon?",
     "2023-05-21", "session_b"),
    ("fact_bakery",
     "The corner bakery switched to a sourdough starter named Hank.",
     "What did the corner bakery name its sourdough starter?",
     "2023-02-14", "session_c"),
    ("fact_violin",
     "Sofia restrung her grandmother's violin before the recital.",
     "Whose violin did Sofia restring?",
     "2023-03-30", "session_d"),
    ("fact_greenhouse",
     "Omar grows heirloom tomatoes in a small backyard greenhouse.",
     "Where does Omar grow his heirloom tomatoes?",
     "2022-08-05", "session_e"),
    ("fact_lighthouse",
     "The old harbor lighthouse was repainted with red and white stripes.",
     "What colors was the harbor lighthouse repainted?",
     "2023-06-01", "session_f"),
    ("fact_chess",
     "Yuki won the regional chess tournament with a queen sacrifice.",
     "How did Yuki win the regional chess tournament?",
     "2023-04-18", "session_g"),
    ("fact_canoe",
     "Diego carved a cedar canoe and paddled it across the lake.",
     "What kind of wood did Diego use for his canoe?",
     "2022-09-22", "session_h"),
]

# Near-duplicate clusters — the material the consolidation/merge cycle targets. Each
# cluster is the SAME underlying claim authored twice (different phrasings, different
# sessions / valid_from), so a merge pass is *supposed* to collapse the pair. The
# ``canonical`` id is the single gold answer for the cluster's probe, so the probe stays
# unambiguous whether or not the duplicates were merged (defects D1/D2/D4).
#   probe        : the paraphrased question
#   canonical    : the gold id the probe scores against (the kept atom after a merge)
#   members      : [(id, text, valid_from, source), ...] — the near-duplicate atoms
_DUP_CLUSTERS = [
    {
        "cluster": "clu_apartment",
        "probe": "Which city did Priya move to for her new job?",
        "canonical": "dup_apartment_1",
        "members": [
            ("dup_apartment_1",
             "Priya relocated to Lisbon to start a job as a backend engineer.",
             "2023-01-09", "session_i"),
            ("dup_apartment_2",
             "Priya moved to Lisbon for a new backend engineering position.",
             "2023-01-15", "session_j"),
        ],
    },
    {
        "cluster": "clu_puppy",
        "probe": "What breed of dog did the Alvarez family adopt?",
        "canonical": "dup_puppy_1",
        "members": [
            ("dup_puppy_1",
             "The Alvarez family adopted a border collie puppy named Comet.",
             "2023-03-02", "session_k"),
            ("dup_puppy_2",
             "The Alvarez household took in a border collie pup they call Comet.",
             "2023-03-08", "session_l"),
        ],
    },
    {
        "cluster": "clu_summit",
        "probe": "Which peak did the climbing club reach in autumn?",
        "canonical": "dup_summit_1",
        "members": [
            ("dup_summit_1",
             "The climbing club summited Mount Aria in the autumn expedition.",
             "2022-10-11", "session_m"),
            ("dup_summit_2",
             "In autumn the climbing club reached the top of Mount Aria.",
             "2022-10-19", "session_n"),
        ],
    },
    {
        "cluster": "clu_novel",
        "probe": "What is the title of the novel Hana finished writing?",
        "canonical": "dup_novel_1",
        "members": [
            ("dup_novel_1",
             "Hana finished writing her debut novel titled Tidewater.",
             "2023-02-25", "session_o"),
            ("dup_novel_2",
             "Hana completed her first novel, which she named Tidewater.",
             "2023-03-04", "session_p"),
        ],
    },
]


def build_dataset() -> dict:
    """Assemble the full A5 dataset dict (deterministic; fixed seed shuffle only)."""
    items: list[dict] = []
    probes: list[dict] = []

    # Singleton facts: each is one stored item + one probe whose gold is that item.
    for fid, text, probe, vf, src in _SINGLETON_FACTS:
        items.append({
            "id": fid, "text": text, "valid_from": vf, "source": src,
            "trusted": True, "cluster": None,
        })
        probes.append({
            "probe": probe, "gold_id": fid, "cluster": None, "kind": "singleton",
        })

    # Near-duplicate clusters: every member is stored; the probe's gold is the canonical
    # id. The duplicates are the consolidation material (audit candidates), but they must
    # never become a *second* correct answer — gold is canonical-only.
    for clu in _DUP_CLUSTERS:
        for mid, text, vf, src in clu["members"]:
            items.append({
                "id": mid, "text": text, "valid_from": vf, "source": src,
                "trusted": True, "cluster": clu["cluster"],
            })
        probes.append({
            "probe": clu["probe"], "gold_id": clu["canonical"],
            "cluster": clu["cluster"], "kind": "duplicate_cluster",
        })

    # Deterministic store-order shuffle: a fixed-seed permutation of the item list so the
    # corpus is not trivially grouped by topic/cluster (a realistic interleaved store),
    # while staying reproducible. Probe order is left stable (it is the report order).
    rng = random.Random(SEED)
    rng.shuffle(items)

    return {
        "schema_version": SCHEMA_VERSION,
        "axis": "a5_consolidation",
        "seed": SEED,
        "k_values": list(K_VALUES),
        "description": (
            "A5 consolidation-gain corpus: clean unambiguous facts plus authored "
            "near-duplicate clusters. recall@k is scored BEFORE and AFTER a "
            "consolidation cycle; the metric is Delta recall. Each probe has exactly "
            "one gold id (canonical for clusters), so scoring is programmatic."
        ),
        "item_count": len(items),
        "probe_count": len(probes),
        "cluster_count": len(_DUP_CLUSTERS),
        "items": items,
        "probes": probes,
    }


def _write_json_stable(path: Path, obj: dict) -> bytes:
    """Serialize ``obj`` to ``path`` with stable, reproducible formatting; return bytes."""
    text = json.dumps(obj, indent=2, ensure_ascii=True, sort_keys=False) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def main() -> int:
    here = Path(__file__).resolve().parent
    dataset = build_dataset()
    dataset_path = here / "dataset.json"
    data = _write_json_stable(dataset_path, dataset)

    digest = hashlib.sha256(data).hexdigest()
    sums_path = here / "SHA256SUMS"
    sums_path.write_text(f"{digest}  dataset.json\n", encoding="utf-8")

    print(f"wrote {dataset_path} ({len(data)} bytes)")
    print(f"items={dataset['item_count']} probes={dataset['probe_count']} "
          f"clusters={dataset['cluster_count']}")
    print(f"sha256(dataset.json)={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
