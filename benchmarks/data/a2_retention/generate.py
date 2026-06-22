"""Generator for the A2 (retention / forgetting) fixed dataset.

A2 measures what survives the forgetting curve: seed a few **salient** facts plus a
larger field of **noise** facts, let time pass (the system's relevance-decay runs many
ticks), then re-query the salient facts *after the delay*. The two numbers are
``retention@delay`` (fraction of salient golds still recallable after the delay) and the
``noise-drop rate`` (fraction of noise atoms the forgetting curve evicted). A good memory
keeps the salient and sheds the noise.

Why this is its OWN dataset (defect D6)
---------------------------------------
Retention is NOT retrieval. A1 asks "with everything present, does the right memory rank
in top-k". A2 asks "after a delay that prunes the store, does the salient memory still
survive while the noise is forgotten". Conflating the two is the field's most common
defect, so this corpus is self-contained and is used by nothing else: salient facts carry
an explicit high salience; noise facts carry zero salience; every salient probe has one
unambiguous gold id (clean labels: defects D1/D2/D4).

What the dataset is
-------------------
* ``salient`` — a small set of important, distinctive facts. Each has one paraphrased
  ``probe`` and its gold ``id``. These are tagged ``salience: 1.0`` so the real
  relevance-decay surface dampens their decay and exempts them from auto-freeze: they are
  expected to PERSIST through the delay (the retention target).
* ``noise`` — a larger field of trivial, low-importance facts on unrelated everyday
  topics, tagged ``salience: 0.0``. These are expected to decay below the freeze threshold
  and be evicted by the forgetting curve (the noise-drop target). They are authored to be
  clearly distinct from every salient topic so a noise atom is never a second valid answer
  to a salient probe.
* ``delays`` — the delay points (in decay ticks) at which retention is measured, so the
  report shows the forgetting curve, not a single point.

The "delay" is exercised against the system's REAL decay pass (``samia.core.tier``): the
task ages every node's last-access into the past and runs ``decay_pass`` ``ticks`` times,
which dampens salient nodes (and exempts them from freeze) while letting zero-salience
noise decay below the freeze threshold and be archived out of the live store. The dataset
only fixes the corpus + labels + delay schedule; the task drives the decay surface.

Determinism
-----------
Content is fully enumerated literal data (no RNG over text); the only ordering step uses a
fixed seed, so regenerating produces a byte-identical ``dataset.json``. The companion
``SHA256SUMS`` pins the bytes; the task refuses to run on a checksum mismatch. No network,
no model — pure data authoring.

Output
------
``dataset.json`` next to this file, plus ``SHA256SUMS`` covering it. Re-run with::

    python benchmarks/data/a2_retention/generate.py

and commit the result. The task reads ``dataset.json`` only; this script is the audit
trail for how those bytes were produced.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Fixed seed — the ONLY randomness is a deterministic interleave of the salient + noise
# populations into one store order, so the committed dataset is reproducible bit-for-bit.
SEED = 1337

# Schema version travels in the dataset so a future format change is detectable and the
# task can refuse data it does not understand.
SCHEMA_VERSION = 1

# k values the A2 retention metric reports recall@ at. Small, fixed, and well under the
# corpus size so a miss is a real miss (not a "k larger than the store" artifact). A
# salient atom is "retained" at delay d if its gold id is still in the post-delay top-k.
K_VALUES = (1, 5, 10)

# Delay schedule, in decay-pass ticks (one tick = one application of the relevance-decay
# pass with the node's last-access aged past the warm-freshness window). Chosen to sample
# the forgetting curve from "barely aged" (noise mostly survives) through "fully pruned"
# (noise gone, salient retained). These are pure data; the task feeds them to the real
# decay surface. 0 is implicit (the pre-delay baseline the task always measures first).
DELAY_TICKS = (5, 10, 20, 40)

# The salience values written to the node frontmatter the decay surface reads. 1.0 puts a
# salient node above the freeze-exemption threshold (it persists); 0.0 leaves a noise node
# fully subject to decay + auto-freeze. These mirror the [0,1] salience field the system's
# own Tier-1 salience source writes; the task stamps them, the decay pass honors them.
SALIENT_SALIENCE = 1.0
NOISE_SALIENCE = 0.0


# Salient facts — important, distinctive, each with one unambiguous gold probe. These are
# the memories a good system must KEEP through the forgetting curve. Distinct topics, no
# overlap with any noise item, so each probe has exactly one correct answer (D1/D2/D4).
# (id, text, probe, valid_from, source)
_SALIENT_FACTS = [
    ("salient_allergy",
     "I am severely allergic to penicillin and carry an epinephrine auto-injector.",
     "What medication am I severely allergic to?",
     "2022-03-04", "medical_intake"),
    ("salient_safe_code",
     "The combination to the home safe is set to my late father's birth year, 1947.",
     "What is the home safe combination set to?",
     "2021-11-20", "household_record"),
    ("salient_emergency_contact",
     "My emergency contact is my sister Nadia, reachable at the lakeside clinic.",
     "Who is listed as my emergency contact?",
     "2022-07-09", "medical_intake"),
    ("salient_blood_type",
     "My blood type is O negative, which the transfusion clinic flagged as universal donor.",
     "What is my blood type?",
     "2020-05-15", "medical_intake"),
    ("salient_will_location",
     "My signed will is stored in the safety deposit box at the Harbor Street branch.",
     "Where is my signed will stored?",
     "2021-02-28", "legal_record"),
    ("salient_mortgage_rate",
     "I locked my mortgage at a fixed rate of 3.1 percent for thirty years.",
     "What fixed rate did I lock my mortgage at?",
     "2022-09-01", "financial_record"),
]

# Noise facts — trivial, low-importance everyday observations. These are the memories a
# good system should FORGET. Each is on a mundane topic clearly disjoint from every salient
# fact, so an evicted-or-surviving noise atom can never be a valid answer to a salient
# probe. Authored in a flat, interchangeable style — the point is that none is worth
# keeping. (id, text, valid_from, source)
_NOISE_FACTS = [
    ("noise_weather_tue", "The weather was mild and overcast on Tuesday afternoon.",
     "2023-01-03", "diary"),
    ("noise_lunch_sandwich", "I had a turkey sandwich for lunch at the corner deli.",
     "2023-01-04", "diary"),
    ("noise_bus_late", "The number 12 bus arrived four minutes late this morning.",
     "2023-01-05", "diary"),
    ("noise_umbrella", "Someone left a green umbrella by the office front door.",
     "2023-01-06", "diary"),
    ("noise_printer_paper", "The third-floor printer ran out of paper again before noon.",
     "2023-01-07", "diary"),
    ("noise_coffee_queue", "The coffee queue downstairs was unusually short today.",
     "2023-01-08", "diary"),
    ("noise_elevator_music", "They changed the elevator music to soft jazz this week.",
     "2023-01-09", "diary"),
    ("noise_parking_spot", "I parked in spot B14 instead of my usual B07 on Thursday.",
     "2023-01-10", "diary"),
    ("noise_plant_watered", "I watered the desk fern on the way out Friday evening.",
     "2023-01-11", "diary"),
    ("noise_pen_blue", "The blue pen on my desk finally ran dry mid-meeting.",
     "2023-01-12", "diary"),
    ("noise_window_open", "I left the kitchen window cracked open to air out the room.",
     "2023-01-13", "diary"),
    ("noise_mail_flyer", "A flyer for a new pizza place showed up in the mailbox.",
     "2023-01-14", "diary"),
    ("noise_dishwasher", "I ran the dishwasher on the short cycle after dinner.",
     "2023-01-15", "diary"),
    ("noise_stairs_count", "There are forty-two steps from the lobby to my floor.",
     "2023-01-16", "diary"),
    ("noise_clock_slow", "The kitchen wall clock was running two minutes slow.",
     "2023-01-17", "diary"),
    ("noise_socks_mismatch", "I wore one navy and one black sock by accident on Monday.",
     "2023-01-18", "diary"),
    ("noise_tea_kettle", "The kettle took a little longer than usual to whistle.",
     "2023-01-19", "diary"),
    ("noise_doormat", "The doormat was flipped over by the wind overnight.",
     "2023-01-20", "diary"),
]


def build_dataset() -> dict:
    """Assemble the full A2 dataset dict (deterministic; fixed-seed interleave only)."""
    items: list[dict] = []
    probes: list[dict] = []

    # Salient items: stored with high salience; each carries one probe whose gold is itself.
    for fid, text, probe, vf, src in _SALIENT_FACTS:
        items.append({
            "id": fid, "text": text, "valid_from": vf, "source": src,
            "trusted": True, "kind": "salient", "salience": SALIENT_SALIENCE,
        })
        probes.append({
            "probe": probe, "gold_id": fid, "kind": "salient",
        })

    # Noise items: stored with zero salience; they are the forgetting target, not probed
    # for recall. Their ids are tracked so the task can measure the noise-drop rate (how
    # many are no longer in the live store / no longer recallable after the delay).
    for nid, text, vf, src in _NOISE_FACTS:
        items.append({
            "id": nid, "text": text, "valid_from": vf, "source": src,
            "trusted": True, "kind": "noise", "salience": NOISE_SALIENCE,
        })

    # Deterministic store-order interleave: a fixed-seed permutation so the corpus is not
    # trivially grouped salient-then-noise (a realistic interleaved store, matching the
    # "interleave many turns" task shape), while staying reproducible. Probe order is left
    # stable (it is the report order).
    rng = random.Random(SEED)
    rng.shuffle(items)

    salient_ids = [f[0] for f in _SALIENT_FACTS]
    noise_ids = [n[0] for n in _NOISE_FACTS]

    return {
        "schema_version": SCHEMA_VERSION,
        "axis": "a2_retention",
        "seed": SEED,
        "k_values": list(K_VALUES),
        "delay_ticks": list(DELAY_TICKS),
        "salient_salience": SALIENT_SALIENCE,
        "noise_salience": NOISE_SALIENCE,
        "description": (
            "A2 retention/forgetting corpus: a few high-salience salient facts (each with "
            "one gold probe) plus a larger field of zero-salience noise facts. The task "
            "ages the store and runs the real relevance-decay pass for each delay in "
            "delay_ticks, then re-queries the salient probes. retention@delay is the "
            "fraction of salient golds still recallable in top-k; noise-drop is the "
            "fraction of noise atoms the forgetting curve evicted. Separate data from A1 "
            "(retrieval) — defect D6. Programmatic scoring on the id ranking; no judge."
        ),
        "item_count": len(items),
        "salient_count": len(salient_ids),
        "noise_count": len(noise_ids),
        "probe_count": len(probes),
        "salient_ids": salient_ids,
        "noise_ids": noise_ids,
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
    print(f"salient={dataset['salient_count']} noise={dataset['noise_count']} "
          f"probes={dataset['probe_count']} delays={dataset['delay_ticks']}")
    print(f"sha256(dataset.json)={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
