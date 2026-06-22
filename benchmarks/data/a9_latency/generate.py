"""Deterministic generator for the A9 (latency / scale) fixed dataset.

A9 measures *latency and scale*, not retrieval accuracy — but it still stores real
atoms and recalls with real queries so the wall-clock numbers reflect the actual
store -> index -> recall path, not a mock. It therefore reuses the **A1 fact shape**
(one unambiguous fact per memory: ``id`` / ``text`` / ``valid_from`` / ``source``,
each carrying an explicit gold id) at three store sizes (100 / 1000 / 10000).

What this produces
------------------
``facts.jsonl``   — 10000 fact records, one JSON object per line, in a fixed order.
                    The N=100 and N=1000 store sizes are the first 100 / 1000 lines of
                    this file (a prefix is a valid corpus because every fact is
                    independent and self-contained), so all three sizes share one
                    committed corpus with no duplication.
``probes.jsonl``  — 10000 query probes, one per fact, aligned by line number: probe on
                    line *i* targets the fact on line *i* with an explicit ``gold`` id.
                    A timing run samples a deterministic subset of these (see the task
                    module) so the latency measurement is bounded but representative,
                    and the gold lets the task confirm recall is not degenerate at scale.
``manifest.json`` — dataset metadata (seed, sizes, counts, schema, field semantics).
``SHA256SUMS``    — SHA256 of ``facts.jsonl`` / ``probes.jsonl`` / ``manifest.json`` so a
                    third party verifies they ran on the exact committed bytes.

Defect fixes baked in (per BENCHMARK_DESIGN_v1.md)
--------------------------------------------------
* **D1/D2/D4 — clean unambiguous gold.** Every fact is a unique ``(subject, attribute)``
  pair with a unique value drawn from disjoint, topically-separated vocabularies. No two
  facts share a subject *and* an attribute, so each probe has exactly one correct answer
  and no near-duplicate competitor. Subjects, attributes and values are generated, not
  hand-curated, but the construction guarantees global uniqueness (see ``_fact_text``).
* **D6 — retrieval vs retention are separate.** A9 is a pure latency axis: it never
  interleaves "delay" turns. The retention axis (A2) owns that; this dataset has no
  temporal-decay semantics. ``valid_from`` is present only because the fact shape carries
  it; A9 does not score on it.
* **D5 — no judge.** A9's metric is wall-clock (p50/p95 recall ms, ingest items/s),
  which is fully programmatic. There is no open-ended generation here, so no pinned judge
  is used or needed (the design reserves the judge for A3/A4/A7 only).

Determinism
-----------
A single fixed seed drives a stdlib ``random.Random`` instance; the vocabularies are
fixed lists; the output order is fixed. Regenerating on any machine yields byte-identical
files (verified by the committed ``SHA256SUMS``). No network, no system entropy.

Usage
-----
    python benchmarks/data/a9_latency/generate.py          # writes the dataset + sums
    python benchmarks/data/a9_latency/generate.py --check  # verify on-disk sums match

This module is import-safe: importing it exposes ``build_records`` / ``SEED`` / ``SIZES``
without writing anything (the task module imports those to know the schema and sizes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# --- fixed parameters (the dataset's identity) --------------------------------

#: The one seed that fixes the entire dataset. Changing it changes the dataset (and its
#: checksums); it is committed so the corpus is reproducible byte-for-byte.
SEED = 1337

#: The three store sizes the latency axis sweeps. The largest is the corpus size; the
#: smaller two are prefixes of it.
SIZES = (100, 1000, 10000)

#: Total facts generated == the largest size. Every size is a prefix of this corpus.
N_FACTS = max(SIZES)

DATA_DIR = Path(__file__).resolve().parent

# --- vocabularies (disjoint, topically-separated -> unambiguous gold) ---------
#
# A fact is "<subject> <verb-phrase for attribute> <value>." Uniqueness is guaranteed by
# construction: subject_i is unique per fact, and (attribute, value) are drawn so that no
# two facts collide on meaning. With a unique subject per fact, every probe ("what is
# <subject>'s <attribute>?") resolves to exactly one fact regardless of value overlap —
# the subject alone disambiguates. The varied attributes/values keep the embedding space
# spread out so the index is a realistic mix, not 10000 near-identical sentences.

# 60 given names x 60 surnames = 3600 base names; an index suffix makes every subject
# globally unique even past 3600 facts, while keeping each subject human-readable.
_GIVEN = [
    "Aaron", "Bianca", "Carlos", "Dahlia", "Elena", "Farid", "Greta", "Hassan",
    "Imani", "Jonas", "Kira", "Lucas", "Mira", "Nadia", "Omar", "Petra",
    "Quentin", "Rosa", "Samir", "Tara", "Ulrich", "Vera", "Wesley", "Xenia",
    "Yusuf", "Zara", "Anika", "Bruno", "Celia", "Dmitri", "Esme", "Felix",
    "Gloria", "Hugo", "Iris", "Javier", "Kasia", "Leon", "Marta", "Niko",
    "Olga", "Pablo", "Quinn", "Rania", "Sven", "Talia", "Umberto", "Vivian",
    "Walt", "Ximena", "Yara", "Zane", "Amara", "Bence", "Clara", "Diego",
    "Edith", "Fiona", "Goran", "Hana",
]
_SURNAME = [
    "Acosta", "Becker", "Castellano", "Dubois", "Eriksen", "Ferraro", "Gupta",
    "Halvorsen", "Ibarra", "Jensen", "Kovac", "Lindqvist", "Moreau", "Nawrocki",
    "Okonkwo", "Pereira", "Quaranta", "Rossi", "Sandoval", "Takeda", "Ueno",
    "Vargas", "Wojcik", "Xu", "Yamamoto", "Zielinski", "Andersson", "Baumann",
    "Costa", "Delacroix", "Egwu", "Fontaine", "Grimaldi", "Holm", "Ivanova",
    "Jaffe", "Klein", "Larsen", "Mancini", "Nieminen", "Ortiz", "Park",
    "Quintero", "Romano", "Schroder", "Tanaka", "Ustinov", "Valenzuela",
    "Weiss", "Xanthopoulos", "Yilmaz", "Zhao", "Abadi", "Brandt", "Cabrera",
    "Dvorak", "Engberg", "Farrugia", "Gallo", "Horvat",
]

# Each attribute = (slug, store-phrasing, query-phrasing). Store and query use DIFFERENT
# wording for the same attribute so recall exercises semantic matching, not string match
# (mirrors the Foundation smoke test's "avoid the exact stored words" rule).
_ATTRIBUTES = [
    ("pet", "keeps a pet {value}", "What pet does {subject} keep"),
    ("city", "lives in the city of {value}", "Which city is {subject}'s home"),
    ("job", "works as a {value}", "What is {subject}'s occupation"),
    ("hobby", "spends weekends on {value}", "What does {subject} do for fun"),
    ("car", "drives a {value}", "What vehicle does {subject} own"),
    ("instrument", "plays the {value}", "Which instrument does {subject} play"),
    ("dish", "always orders the {value}", "What is {subject}'s favorite dish"),
    ("sport", "trains for {value}", "Which sport does {subject} practice"),
    ("plant", "grows {value} on the balcony", "What does {subject} grow at home"),
    ("language", "is learning {value}", "Which language is {subject} studying"),
]

_PETS = ["tabby cat", "border collie", "cockatiel", "russian tortoise", "betta fish",
         "lop rabbit", "ferret", "leopard gecko", "miniature pony", "corn snake"]
_CITIES = ["Lyon", "Porto", "Tallinn", "Kyoto", "Valparaiso", "Aarhus", "Ljubljana",
           "Cebu", "Bergen", "Mendoza"]
_JOBS = ["marine biologist", "pastry chef", "civil engineer", "archivist",
         "wind-turbine technician", "cartographer", "speech therapist",
         "glassblower", "seismologist", "luthier"]
_HOBBIES = ["competitive bouldering", "watercolor painting", "amateur astronomy",
            "letterpress printing", "freshwater kayaking", "beekeeping",
            "vintage radio repair", "orchid breeding", "trail running", "origami"]
_CARS = ["teal hatchback", "diesel pickup", "convertible roadster", "electric scooter",
         "vintage motorcycle", "cargo bicycle", "hybrid sedan", "off-road buggy",
         "panel van", "three-wheeled tuk-tuk"]
_INSTRUMENTS = ["cello", "tenor saxophone", "hammered dulcimer", "bandoneon", "djembe",
                "harpsichord", "pan flute", "five-string banjo", "theremin", "marimba"]
_DISHES = ["mushroom risotto", "lamb tagine", "green curry", "borscht", "okonomiyaki",
           "ceviche", "spanakopita", "bibimbap", "ratatouille", "pho"]
_SPORTS = ["fencing", "open-water swimming", "archery", "speed skating", "handball",
           "rock climbing", "table tennis", "rowing", "biathlon", "ultimate frisbee"]
_PLANTS = ["cherry tomatoes", "lavender", "thai basil", "string-of-pearls succulents",
           "dwarf citrus", "carnivorous pitcher plants", "saffron crocus",
           "bonsai juniper", "rainbow chard", "passionflower vines"]
_LANGUAGES = ["Finnish", "Swahili", "Quechua", "Welsh", "Korean", "Basque", "Tagalog",
              "Icelandic", "Georgian", "Hungarian"]

_VALUE_POOLS = {
    "pet": _PETS, "city": _CITIES, "job": _JOBS, "hobby": _HOBBIES, "car": _CARS,
    "instrument": _INSTRUMENTS, "dish": _DISHES, "sport": _SPORTS, "plant": _PLANTS,
    "language": _LANGUAGES,
}


def _subject(idx: int) -> str:
    """A globally-unique, human-readable subject name for fact ``idx``.

    Combines a given name and a surname by index so the first 3600 subjects are distinct
    name pairs; a ``#NN`` suffix (the high digits of idx) keeps subjects unique past 3600
    without ever repeating a (given, surname, suffix) triple. The subject alone is the
    gold-disambiguating key, so its uniqueness is what guarantees one-fact-per-probe.
    """
    g = _GIVEN[idx % len(_GIVEN)]
    s = _SURNAME[(idx // len(_GIVEN)) % len(_SURNAME)]
    block = idx // (len(_GIVEN) * len(_SURNAME))
    return f"{g} {s}" if block == 0 else f"{g} {s} #{block:02d}"


def _fact_text(idx: int, attr: tuple[str, str, str], value: str) -> str:
    """Render the stored fact sentence for ``idx`` (subject + attribute phrasing + value)."""
    subject = _subject(idx)
    _, store_tmpl, _ = attr
    return f"{subject} {store_tmpl.format(value=value)}."


def _probe_text(idx: int, attr: tuple[str, str, str]) -> str:
    """Render the recall query for ``idx`` (different wording from the stored fact)."""
    subject = _subject(idx)
    _, _, query_tmpl = attr
    return f"{query_tmpl.format(subject=subject)}?"


def build_records(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Build the (facts, probes) record lists deterministically from ``rng``.

    Returns two equal-length lists aligned by index: ``facts[i]`` is the fact and
    ``probes[i]`` is the query whose single gold answer is ``facts[i]["id"]``. Each fact
    record carries the A1 fact shape (id / text / valid_from / source) plus the gold-key
    fields (subject / attribute / value) for audit. Probes carry the query text, the gold
    id, and a rationale (why that id is the unique answer) per the design's "explicit gold
    + rationale" rule.
    """
    facts: list[dict] = []
    probes: list[dict] = []
    for i in range(N_FACTS):
        attr = _ATTRIBUTES[i % len(_ATTRIBUTES)]
        slug = attr[0]
        # Deterministic value pick per fact; value overlap across facts is harmless
        # because the unique subject is the disambiguator.
        value = rng.choice(_VALUE_POOLS[slug])
        subject = _subject(i)
        fact_id = f"a9_fact_{i:05d}"
        # Spread synthetic valid_from dates across a year purely so the field is populated
        # realistically; A9 does not score on it (D6: no temporal semantics here).
        month = (i % 12) + 1
        day = (i % 27) + 1
        valid_from = f"2024-{month:02d}-{day:02d}"
        source = f"sess_{(i % 50):02d}"
        facts.append({
            "id": fact_id,
            "text": _fact_text(i, attr, value),
            "valid_from": valid_from,
            "source": source,
            "subject": subject,
            "attribute": slug,
            "value": value,
        })
        probes.append({
            "id": f"a9_probe_{i:05d}",
            "query": _probe_text(i, attr),
            "gold": fact_id,
            "rationale": (
                f"Only '{subject}' has a recorded {slug}; the subject name uniquely "
                f"identifies the single fact that answers this query."
            ),
        })
    return facts, probes


# --- serialization + checksums ------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write ``records`` as one compact JSON object per line (stable key order)."""
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=True, sort_keys=True))
            fh.write("\n")


def _sha256(path: Path) -> str:
    """Hex SHA256 of a file's bytes."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


_SUM_FILES = ("facts.jsonl", "probes.jsonl", "manifest.json")


def _write_sums(data_dir: Path) -> None:
    """Write SHA256SUMS over the three committed dataset files (sha256sum format)."""
    lines = []
    for name in _SUM_FILES:
        digest = _sha256(data_dir / name)
        lines.append(f"{digest}  {name}\n")
    (data_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def generate(data_dir: Path = DATA_DIR) -> dict:
    """Generate the full dataset + manifest + SHA256SUMS into ``data_dir``.

    Returns the manifest dict. Deterministic: a fresh seeded RNG drives every choice.
    """
    rng = random.Random(SEED)
    facts, probes = build_records(rng)

    _write_jsonl(data_dir / "facts.jsonl", facts)
    _write_jsonl(data_dir / "probes.jsonl", probes)

    manifest = {
        "axis": "a9_latency",
        "description": "Latency/scale: store->recall wall-clock at three store sizes.",
        "seed": SEED,
        "sizes": list(SIZES),
        "n_facts": N_FACTS,
        "n_probes": len(probes),
        "fact_shape": ["id", "text", "valid_from", "source"],
        "fact_gold_fields": ["subject", "attribute", "value"],
        "probe_shape": ["id", "query", "gold", "rationale"],
        "alignment": "facts.jsonl line i and probes.jsonl line i share a gold id",
        "size_semantics": "each size N uses the first N lines of facts.jsonl",
        "scoring": "wall-clock only (p50/p95 recall ms, ingest items/s); no judge",
        "embedder": "sentence-transformers/all-MiniLM-L6-v2 (384-d, CPU, cache-only)",
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_sums(data_dir)
    return manifest


def check(data_dir: Path = DATA_DIR) -> bool:
    """Verify on-disk files match the committed SHA256SUMS. Returns True iff all match."""
    sums_path = data_dir / "SHA256SUMS"
    if not sums_path.exists():
        print("SHA256SUMS missing — run the generator first.", file=sys.stderr)
        return False
    ok = True
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split()
        actual = _sha256(data_dir / name)
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            ok = False
        print(f"{status}  {name}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A9 latency dataset generator")
    parser.add_argument("--check", action="store_true",
                        help="verify on-disk files against the committed SHA256SUMS")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check() else 1
    manifest = generate()
    print(f"generated A9 dataset: {manifest['n_facts']} facts, "
          f"{manifest['n_probes']} probes, sizes={manifest['sizes']}")
    print(f"wrote: facts.jsonl probes.jsonl manifest.json SHA256SUMS -> {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
