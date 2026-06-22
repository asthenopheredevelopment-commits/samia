"""Generator for the A1 (retrieval-accuracy) fixed dataset.

A1 measures ONE thing: given N seeded facts and a paraphrased query per fact, is the
gold fact in the recall top-k (recall@{1,5,10}, MRR). It is deliberately *retrieval only*
and carries its OWN data, separate from A2 retention-after-delay (defect D6: the field's
most common error is conflating the two).

Design rules this generator enforces (defects D1/D2/D4 — clean, unambiguous gold):

* **One gold per query.** Every fact is on a *distinct topic* (a unique (subject, kind,
  detail) triple drawn without replacement), so exactly one stored fact answers each
  query and the gold id is never ambiguous. No two facts share a subject+kind pair, so a
  query can never legitimately match two memories.
* **Paraphrased queries, not string lookups.** Each query asks for the fact's detail using
  different words from the stored sentence, so recall exercises the semantic vector arm
  rather than a substring match. The query never contains the gold's distinguishing detail
  token verbatim.
* **Explicit gold + rationale per item.** Each item records its gold id and a one-line
  rationale ("query asks for <subject>'s <kind>; only <id> states it") so a third party can
  audit why that label is correct.
* **Deterministic + checksummed.** The whole set is a pure function of the seed: shuffling,
  selection and pairing all run off a single ``random.Random(seed)``. Running this file
  reproduces ``dataset.json`` byte-for-byte and re-writes ``SHA256SUMS``.

The facts are short, self-contained, single-claim sentences in the same neutral register as
SAM/IA's own ``test_semantic_recall`` fixtures. Nothing here is system-specific: the dataset
is plain JSON consumed by ``tasks/a1_retrieval.py`` through the ``MemoryAdapter`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# Default committed dataset parameters. The harness default N is 100; the generator can emit
# any N up to the topic-pool size for ad-hoc sizes, but the COMMITTED, checksummed dataset is
# fixed at this seed + size and must not drift.
DEFAULT_SEED = 1337
DEFAULT_N = 100
DATA_DIR = Path(__file__).resolve().parent

# -- Topic pool -----------------------------------------------------------------------------
# Each "kind" is a relation with: a fact template (how the stored sentence reads), a query
# template (a paraphrase that asks for the same detail in different words), and a pool of
# concrete, mutually distinct details. A fact = (subject, kind, detail). We draw distinct
# (subject, kind) pairs so no two facts collide, and a distinct detail per fact so the
# answer is unique. Templates avoid putting the detail token verbatim into the query.

_SUBJECTS = [
    "Maria", "David", "Priya", "Kenji", "Amara", "Luca", "Sofia", "Omar", "Hannah",
    "Diego", "Nadia", "Theo", "Yara", "Mateo", "Ingrid", "Rashid", "Elena", "Tariq",
    "Freya", "Hugo", "Leila", "Marcus", "Aisha", "Pablo", "Greta", "Noah", "Zara",
    "Felix", "Mei", "Ravi", "Clara", "Bruno", "Anika", "Owen", "Saanvi", "Viktor",
    "Lena", "Idris", "Camila", "Soren",
]

_KINDS: dict[str, dict] = {
    "pet": {
        "fact": "{subj} adopted a {detail} as a pet.",
        "query": "What kind of animal did {subj} take in?",
        "details": ["tabby cat", "beagle puppy", "green parrot", "spotted rabbit",
                    "goldfish", "leopard gecko", "miniature pony", "hedgehog",
                    "cockatiel", "tortoise", "ferret", "chinchilla"],
    },
    "car": {
        "fact": "{subj} bought a {detail} last year.",
        "query": "Which vehicle did {subj} purchase?",
        "details": ["red hatchback", "silver pickup truck", "black electric sedan",
                    "vintage convertible", "blue minivan", "yellow sports coupe",
                    "white station wagon", "grey crossover SUV", "green roadster",
                    "orange camper van", "teal city scooter", "maroon estate car"],
    },
    "city": {
        "fact": "{subj} moved to {detail} for work.",
        "query": "Where did {subj} relocate?",
        "details": ["Lisbon", "Osaka", "Nairobi", "Montreal", "Helsinki", "Bogota",
                    "Reykjavik", "Brisbane", "Krakow", "Casablanca", "Wellington",
                    "Tallinn"],
    },
    "job": {
        "fact": "{subj} now works as a {detail}.",
        "query": "What is {subj}'s profession these days?",
        "details": ["marine biologist", "pastry chef", "civil engineer",
                    "jazz pianist", "tax auditor", "wildlife photographer",
                    "speech therapist", "vineyard manager", "glass blower",
                    "air traffic controller", "forensic accountant", "ferry captain"],
    },
    "hobby": {
        "fact": "{subj} took up {detail} this spring.",
        "query": "What new pastime did {subj} start?",
        "details": ["rock climbing", "pottery", "bird watching", "stargazing",
                    "beekeeping", "watercolor painting", "kayaking", "calligraphy",
                    "orienteering", "model railroading", "quilting", "geocaching"],
    },
    "instrument": {
        "fact": "{subj} learned to play the {detail}.",
        "query": "Which musical instrument can {subj} play?",
        "details": ["cello", "trumpet", "accordion", "banjo", "harp", "clarinet",
                    "ukulele", "bassoon", "mandolin", "oboe", "sitar", "marimba"],
    },
    "dish": {
        "fact": "{subj}'s signature dish is {detail}.",
        "query": "What food is {subj} known for cooking?",
        "details": ["mushroom risotto", "lamb tagine", "seafood paella",
                    "spinach dumplings", "coconut curry", "beet borscht",
                    "plantain stew", "chestnut soup", "pumpkin gnocchi",
                    "fig tart", "miso ramen", "okra gumbo"],
    },
    "sport": {
        "fact": "{subj} competes in {detail} on weekends.",
        "query": "Which sport does {subj} play competitively?",
        "details": ["table tennis", "archery", "downhill skiing", "rowing",
                    "badminton", "squash", "trail running", "open-water swimming",
                    "speed climbing", "curling", "dressage", "fencing"],
    },
    "language": {
        "fact": "{subj} is studying {detail} this year.",
        "query": "Which language is {subj} currently learning?",
        "details": ["Portuguese", "Mandarin", "Swahili", "Finnish", "Korean",
                    "Hungarian", "Icelandic", "Arabic", "Vietnamese", "Greek",
                    "Turkish", "Welsh"],
    },
    "plant": {
        "fact": "{subj} is growing {detail} in the garden.",
        "query": "What is {subj} cultivating outdoors?",
        "details": ["heirloom tomatoes", "lavender", "bonsai maples", "chili peppers",
                    "climbing roses", "saffron crocus", "blueberry bushes", "mint",
                    "artichokes", "dahlias", "rhubarb", "sunflowers"],
    },
}


def _slug(text: str) -> str:
    """Filesystem-safe lower-case slug (the node id == MemoryItem.id == filename stem)."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def build_items(seed: int, n: int) -> list[dict]:
    """Build ``n`` distinct-topic retrieval items deterministically from ``seed``.

    Each item is ``{id, text, query, gold, rationale, kind}``. Draws distinct
    (subject, kind) pairs and a distinct detail per fact so the gold is unique. Raises if
    ``n`` exceeds the distinct-pair capacity (no ambiguous reuse is ever introduced to hit a
    larger N).
    """
    rng = random.Random(seed)

    # All distinct (subject, kind) pairs, then shuffle and take the first n. Per kind we also
    # consume details without replacement so two facts of the same kind never share a detail.
    pairs = [(s, k) for s in _SUBJECTS for k in _KINDS]
    capacity = sum(len(_KINDS[k]["details"]) for k in _KINDS)  # detail-limited capacity
    if n > capacity:
        raise ValueError(
            f"requested n={n} exceeds distinct-detail capacity {capacity}; "
            "the dataset must stay unambiguous (one gold per query) — do not reuse details")
    rng.shuffle(pairs)

    detail_pools: dict[str, list[str]] = {k: list(_KINDS[k]["details"]) for k in _KINDS}
    for k in detail_pools:
        rng.shuffle(detail_pools[k])

    items: list[dict] = []
    for subj, kind in pairs:
        if len(items) >= n:
            break
        # (subject, kind) pairs are already distinct (no reuse), so two facts never share a
        # query target. A subject may recur across DIFFERENT kinds: those have different
        # queries ("X's profession" vs "what animal X took in") and different details, so the
        # gold stays unique. The detail pool is consumed without replacement per kind, so no
        # two facts of the same kind share a detail either.
        if not detail_pools[kind]:
            continue
        detail = detail_pools[kind].pop()
        spec = _KINDS[kind]
        text = spec["fact"].format(subj=subj, detail=detail)
        query = spec["query"].format(subj=subj)
        item_id = f"a1_{_slug(subj)}_{kind}"
        items.append({
            "id": item_id,
            "text": text,
            "query": query,
            "gold": item_id,
            "kind": kind,
            "rationale": (
                f"query asks for {subj}'s {kind}; only {item_id} states it "
                f"({detail!r}); every fact has a distinct (subject, kind) topic and a "
                "distinct detail, so exactly one stored fact answers this query"),
        })

    if len(items) < n:
        raise ValueError(
            f"only produced {len(items)} unambiguous items for n={n} "
            "(ran out of distinct subject+kind pairs); enlarge the pools, do not reuse")
    # Stable order by id so the committed JSON is canonical regardless of draw order.
    items.sort(key=lambda d: d["id"])
    return items


def dataset_dict(seed: int, n: int) -> dict:
    """Assemble the full dataset object (metadata + items)."""
    return {
        "axis": "a1_retrieval",
        "description": (
            "Retrieval accuracy: N distinct-topic facts, one paraphrased query per fact; "
            "score whether the gold fact is in recall top-k. Retrieval only — separate "
            "data from A2 retention (defect D6). One unambiguous gold per query "
            "(defects D1/D2/D4)."),
        "seed": seed,
        "n": n,
        "metrics": ["recall@1", "recall@5", "recall@10", "mrr"],
        "scoring": "programmatic",
        "judge": "none (closed-form: gold id present in ranked recall list)",
        "items": build_items(seed, n),
    }


def _canonical_json(obj: dict) -> str:
    """Canonical JSON serialization (sorted keys, fixed separators) for stable checksums."""
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def write_dataset(seed: int = DEFAULT_SEED, n: int = DEFAULT_N,
                  out_dir: Path = DATA_DIR) -> Path:
    """Write ``dataset.json`` + ``SHA256SUMS`` to ``out_dir`` and return the dataset path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = dataset_dict(seed, n)
    text = _canonical_json(ds)
    ds_path = out_dir / "dataset.json"
    ds_path.write_text(text, encoding="utf-8")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out_dir / "SHA256SUMS").write_text(f"{digest}  dataset.json\n", encoding="utf-8")
    return ds_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the A1 retrieval fixed dataset.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    args = ap.parse_args()
    path = write_dataset(args.seed, args.n)
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {path} (n={args.n}, seed={args.seed})")
    print(f"sha256 {digest}  dataset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
