"""Generator for the A3 (temporal reasoning) FIXED benchmark dataset.

Emits a deterministic, seeded, checksummed dataset for the temporal-reasoning axis
described in ``BENCHMARK_DESIGN_v1.md`` (A3): *"what did I say about X most recently /
before Y"* — measured by **temporal-recall@k** and **ordering accuracy**.

Why a dedicated generator + committed dataset (the design's non-negotiables):

* **Deterministic / seeded.** Everything is built from a single integer seed and a fixed
  table of topics; there is no RNG draw whose value is unpinned. Re-running with the same
  seed reproduces byte-identical ``dataset.json``.
* **Versioned + checksummed.** ``dataset.json`` is written next to a ``SHA256SUMS`` so a
  third party re-runs the generator and confirms the bytes match. No network is touched.
* **Clean, unambiguous gold (defect D1/D2/D4).** Each topic is a chain of *dated* facts
  about ONE attribute (the city you live in, the phone you carry, ...). The facts are
  strictly ordered in time with **distinct** ``valid_from`` dates, each fact *supersedes*
  the previous value of that attribute, and the chains are on disjoint topics so no query
  is ambiguous between topics. Every query carries an explicit programmatic ``gold`` id
  and a human-readable ``rationale`` for why that id is correct.
* **Separate task/data from retrieval+retention (defect D6).** This dataset exists only
  for A3; it is never reused for A1 (retrieval) or A2 (retention). Its queries are
  *temporal* ("latest" / "as of date D"), not plain relevance lookups.

Dataset shape (``dataset.json``)::

    {
      "axis": "a3_temporal",
      "version": 1,
      "seed": <int>,
      "items":   [ {id, text, valid_from, source, topic, attribute, order}, ... ],
      "queries": [ {id, topic, kind, text, as_of?, gold, rationale,
                    open_ended_phrasings:[...]}, ... ]
    }

``items`` are stored verbatim as ``MemoryItem``s by the task; ``queries`` drive scoring.
Two query KINDS per topic, both with a single unambiguous programmatic gold id:

* ``most_recent`` — "the latest value of attribute A" → gold = the chronologically last
  fact in the topic chain (max ``valid_from``).
* ``as_of`` (the "before Y" case) — "the value of attribute A as of date D" → gold = the
  fact that was in force at D = the latest fact whose ``valid_from`` <= D. D is chosen to
  fall strictly *between* two updates so the gold is the EARLIER one (the still-current
  value at D), never the not-yet-true later one — that is the discriminating test.

Run::

    python benchmarks/data/a3_temporal/generate.py          # writes dataset.json + SHA256SUMS
    python benchmarks/data/a3_temporal/generate.py --check   # verify committed bytes match
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

# Single source of truth for the seed (kept here, not on the CLI default, so the committed
# bytes are reproducible from the file alone). The seed only labels the dataset version and
# orders deterministic choices; there is no unpinned random draw.
SEED = 1337
VERSION = 1
AXIS = "a3_temporal"

DATA_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIR / "dataset.json"
SUMS_PATH = DATA_DIR / "SHA256SUMS"


# Fixed topic table. Each topic is ONE attribute whose value changes over time; the chain
# is hand-authored so every value is unambiguous, every date distinct, and consecutive
# values are clearly different (the later fact supersedes the earlier). Phrasings avoid the
# stored wording so recall exercises semantics, not string match.
#
# Each chain entry is (valid_from, value_text). The chains live on disjoint subjects so a
# "where do I live" query can never be confused with a "what phone" query, etc.
_TOPICS: list[dict] = [
    {
        "topic": "residence",
        "attribute": "the city the speaker lives in",
        "subject": "I",
        "noun": "city",
        "chain": [
            ("2018-03-01", "I lived in Boston."),
            ("2020-07-15", "I moved to Chicago."),
            ("2022-02-10", "I relocated to Denver."),
            ("2024-09-05", "I settled in Seattle."),
        ],
    },
    {
        "topic": "phone",
        "attribute": "the phone the speaker carries",
        "subject": "I",
        "noun": "phone",
        "chain": [
            ("2019-01-20", "My phone was a Pixel 3."),
            ("2021-11-01", "My phone became a Pixel 6."),
            ("2023-10-12", "My phone is now a Pixel 8."),
        ],
    },
    {
        "topic": "job_title",
        "attribute": "the job title the speaker holds",
        "subject": "I",
        "noun": "job title",
        "chain": [
            ("2017-06-01", "I worked as a junior analyst."),
            ("2019-09-15", "I was promoted to senior analyst."),
            ("2022-04-01", "I became a team lead."),
            ("2025-01-10", "I took a role as engineering manager."),
        ],
    },
    {
        "topic": "car",
        "attribute": "the car the speaker drives",
        "subject": "I",
        "noun": "car",
        "chain": [
            ("2016-05-10", "I drove a red Honda Civic."),
            ("2020-12-01", "I switched to a silver Subaru Outback."),
            ("2024-03-22", "I now drive a blue Toyota RAV4."),
        ],
    },
    {
        "topic": "pet",
        "attribute": "the pet the speaker keeps",
        "subject": "I",
        "noun": "pet",
        "chain": [
            ("2018-08-08", "I had a tabby cat named Pixel."),
            ("2021-03-30", "I adopted a beagle named Comet."),
            ("2023-07-19", "I got a parrot named Mango."),
        ],
    },
    {
        "topic": "diet",
        "attribute": "the diet the speaker follows",
        "subject": "I",
        "noun": "diet",
        "chain": [
            ("2019-02-01", "I followed a vegetarian diet."),
            ("2021-06-15", "I switched to a pescatarian diet."),
            ("2024-01-05", "I went fully vegan."),
        ],
    },
]


def _slug(topic: str, order: int) -> str:
    """Stable, filesystem-safe node-id stem for a chain fact (caller id == node id)."""
    return f"a3_{topic}_{order:02d}"


def _query_id(topic: str, kind: str, suffix: str = "") -> str:
    base = f"a3q_{topic}_{kind}"
    return f"{base}_{suffix}" if suffix else base


def _midpoint(d1: str, d2: str) -> str:
    """A date strictly between two ISO dates (the as-of probe point).

    Why strictly between: an ``as_of`` query at this point must resolve to the EARLIER
    fact (the value still in force), which is the whole discriminator for "before Y" — the
    later fact's ``valid_from`` is after D so it was not yet true. The midpoint is a pure
    function of the two dates (no RNG), keeping the dataset deterministic.
    """
    a = _dt.date.fromisoformat(d1)
    b = _dt.date.fromisoformat(d2)
    mid = a + (b - a) // 2
    # Guard: never land exactly on a boundary (would make the gold ambiguous). With
    # distinct dates spanning months this never triggers, but keep the invariant explicit.
    if mid <= a:
        mid = a + _dt.timedelta(days=1)
    if mid >= b:
        mid = b - _dt.timedelta(days=1)
    return mid.isoformat()


def build_dataset() -> dict:
    """Construct the full A3 dataset dict deterministically from the fixed topic table."""
    items: list[dict] = []
    queries: list[dict] = []

    for topic in _TOPICS:
        name = topic["topic"]
        chain = topic["chain"]
        noun = topic["noun"]
        attribute = topic["attribute"]

        # --- facts (the dated chain) ---
        for order, (valid_from, text) in enumerate(chain):
            items.append({
                "id": _slug(name, order),
                "text": text,
                "valid_from": valid_from,
                "source": f"sess_{name}_{order:02d}",
                "topic": name,
                "attribute": attribute,
                "order": order,
            })

        last_order = len(chain) - 1
        last_id = _slug(name, last_order)
        last_date = chain[last_order][0]

        # --- most_recent query: gold = the chronologically last fact ---
        queries.append({
            "id": _query_id(name, "most_recent"),
            "topic": name,
            "kind": "most_recent",
            "text": f"What is the most recent {noun} for {attribute}?",
            "gold": last_id,
            "rationale": (
                f"The latest fact in the {name} chain is {last_id} "
                f"(valid_from {last_date}); 'most recent' = max valid_from."
            ),
            # Open-ended natural phrasings of the same question (for the pinned-judge
            # subset only; the programmatic gold above is the trustworthy answer).
            "open_ended_phrasings": [
                f"Right now, what is {attribute}?",
                f"What is the speaker's current {noun}?",
            ],
        })

        # --- as_of ("before Y") query per interior gap: gold = the still-in-force fact ---
        # One as_of probe per consecutive pair: D falls between fact[i] and fact[i+1], so
        # the value in force at D is fact[i] (gold), NOT fact[i+1] (not yet true). This is
        # the discriminating temporal test the axis exists to measure.
        for i in range(len(chain) - 1):
            as_of = _midpoint(chain[i][0], chain[i + 1][0])
            gold_id = _slug(name, i)
            queries.append({
                "id": _query_id(name, "as_of", f"{i:02d}"),
                "topic": name,
                "kind": "as_of",
                "text": f"As of {as_of}, what was {attribute}?",
                "as_of": as_of,
                "gold": gold_id,
                "rationale": (
                    f"At {as_of} the in-force fact is {gold_id} "
                    f"(valid_from {chain[i][0]}); the next update "
                    f"{_slug(name, i + 1)} (valid_from {chain[i + 1][0]}) "
                    f"is after {as_of}, so it was not yet true."
                ),
            })

    return {
        "axis": AXIS,
        "version": VERSION,
        "seed": SEED,
        "n_topics": len(_TOPICS),
        "n_items": len(items),
        "n_queries": len(queries),
        "items": items,
        "queries": queries,
    }


def _canonical_json(obj: dict) -> str:
    """Stable JSON serialization (sorted keys, fixed separators) for reproducible bytes."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_dataset() -> tuple[Path, str]:
    """Write ``dataset.json`` + ``SHA256SUMS`` and return (path, digest)."""
    payload = _canonical_json(build_dataset())
    DATASET_PATH.write_text(payload, encoding="utf-8")
    digest = _sha256(payload)
    SUMS_PATH.write_text(f"{digest}  dataset.json\n", encoding="utf-8")
    return DATASET_PATH, digest


def check_dataset() -> bool:
    """True iff the committed ``dataset.json`` matches a fresh deterministic build + its sum."""
    fresh = _canonical_json(build_dataset())
    if not DATASET_PATH.exists():
        print(f"MISSING: {DATASET_PATH}", file=sys.stderr)
        return False
    on_disk = DATASET_PATH.read_text(encoding="utf-8")
    if on_disk != fresh:
        print("MISMATCH: committed dataset.json differs from a fresh deterministic build",
              file=sys.stderr)
        return False
    if SUMS_PATH.exists():
        recorded = SUMS_PATH.read_text(encoding="utf-8").split()[0]
        if recorded != _sha256(on_disk):
            print("MISMATCH: SHA256SUMS does not match dataset.json", file=sys.stderr)
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate/verify the A3 temporal dataset.")
    ap.add_argument("--check", action="store_true",
                    help="verify the committed dataset matches a fresh build (no write)")
    args = ap.parse_args(argv)
    if args.check:
        ok = check_dataset()
        print("A3 dataset OK (bytes + checksum match)" if ok else "A3 dataset CHECK FAILED")
        return 0 if ok else 1
    path, digest = write_dataset()
    ds = build_dataset()
    print(f"wrote {path}")
    print(f"  topics={ds['n_topics']} items={ds['n_items']} queries={ds['n_queries']}")
    print(f"  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
