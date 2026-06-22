"""Generator for the A7 (distillation-fidelity) fixed dataset.

A7 measures whether SAM/IA's atomization preserves a source's *claims* when a verbose
source blob is distilled into atoms and then recalled. Per the benchmark design this axis
gets its OWN data, separate from A1 (retrieval) and A2 (retention) — conflating those
populations is defect D6. So this generator emits a self-contained corpus used by nothing
else: verbose multi-claim sources, each annotated with the exact claims a faithful
distillation must keep.

What the dataset is
-------------------
* A set of **verbose sources**. Each source is a short multi-sentence blob that bundles
  several distinct, checkable claims (a person + an attribute + a date + a place, etc.).
  The verbose form is deliberately wordy so the atomizer has real work to do.
* Each source carries an explicit list of **gold claims**. A claim is one atomic fact the
  faithful distillation must preserve, expressed two ways so scoring can be programmatic:
    - ``key_terms``  : the content tokens that MUST survive into some atom for the claim to
                       count as preserved (clean, unambiguous gold — defects D1/D2/D4).
    - ``probe``      : a paraphrased question used to recall the atom that should carry the
                       claim (exercises the real semantic recall path, not a string match).
    - ``forbidden``  : tokens whose appearance in the recalled atom would mean a *distorted*
                       claim (e.g. a wrong number/colour). Empty for most claims; present on
                       claims authored with a plausible distractor so precision is testable.
* A small **paraphrase subset** (``judge_eval: true`` claims). These are claims whose gold
  is stated in words that do NOT overlap the source surface form, so token coverage alone
  cannot decide preservation — this is the open-ended slice the pinned judge scores
  (design D5: programmatic primary, judge only for the genuinely open-ended subset).

Why two metrics
---------------
The primary metric is a programmatic **claim-preservation F1** computed from token coverage
of the gold claims against the distilled+recalled atoms (no model, fully reproducible). The
paraphrase subset is reported *separately* under a pinned local judge (fixed prompt,
temperature 0, saved transcript). Keeping them separate is the whole point of design rule
D5: a reader/judge effect can never contaminate the headline F1.

Determinism
-----------
Content is fully enumerated literal data (no RNG over text). The only ordering step uses a
fixed seed, so regenerating always produces a byte-identical ``dataset.json``. The companion
``SHA256SUMS`` pins the bytes; the task module refuses to run on a checksum mismatch. No
network, no model — this is pure data authoring.

Output
------
``dataset.json`` next to this file, plus ``SHA256SUMS`` covering it. Re-run with::

    python benchmarks/data/a7_distill/generate.py

and commit the result. The task reads ``dataset.json`` only; this script is the audit trail
for how those bytes were produced.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Fixed seed — the ONLY randomness is a deterministic shuffle of the otherwise fully
# enumerated source list, so the dataset is reproducible bit-for-bit. Pinned here (not
# passed in) because the COMMITTED dataset must be a single fixed artifact.
SEED = 1337

# Schema version travels in the dataset so a future format change is detectable and the
# task can refuse data it does not understand.
SCHEMA_VERSION = 1

# k values the A7 metric reports recall@ at when locating the atom that should carry a
# claim. Small and fixed; well under the atom population so a miss is a real miss.
K_VALUES = (1, 5, 10)


# Verbose sources. Each tuple is:
#   (source_id, valid_from, source_tag, verbose_text, [claims])
# where each claim is a dict:
#   {
#     "claim_id"  : stable id, unique within the source,
#     "key_terms" : [tokens that MUST survive into some atom — case-insensitive,
#                    matched as whole lowercased word substrings],
#     "probe"     : paraphrased recall question (no verbatim source phrasing),
#     "forbidden" : [tokens that, if present in the recalled atom, mean a DISTORTED claim],
#     "judge_eval": True for the paraphrase subset (open-ended; pinned judge), else absent,
#     "paraphrase": (judge_eval claims only) the gold claim restated in non-overlapping
#                    words — the statement the judge checks the atom against.
#   }
#
# Authoring rules (defects D1/D2/D4): every claim is unambiguous, every source's claims are
# on distinct facts, no two sources share a claim, and key_terms are content words that
# uniquely fix the claim (no stop-words, no terms shared across claims of one source).
_SOURCES = [
    (
        "src_pixel", "2023-04-01", "session_a",
        ("On the first of April 2023 Maria finally went down to the Springfield animal "
         "shelter and adopted a calico cat. She decided to name the cat Pixel. The shelter "
         "staff said Pixel had been waiting for a home for almost six months."),
        [
            {"claim_id": "pixel_species",
             "key_terms": ["calico", "cat"],
             "probe": "What kind of animal did Maria adopt?",
             "forbidden": ["dog", "kitten"]},
            {"claim_id": "pixel_name",
             "key_terms": ["pixel"],
             "probe": "What did Maria name her new cat?",
             "forbidden": []},
            {"claim_id": "pixel_shelter",
             "key_terms": ["springfield", "shelter"],
             "probe": "Where did Maria adopt the cat from?",
             "forbidden": []},
            # Paraphrase subset: the gold ("got a pet companion") shares no surface tokens
            # with the source, so only a judge can decide if the atom preserves it.
            {"claim_id": "pixel_adoption_act", "judge_eval": True,
             "key_terms": ["adopted"],
             "paraphrase": "Maria became the owner of a new pet companion.",
             "probe": "Did Maria take on responsibility for a new pet?",
             "forbidden": ["returned", "gave away"]},
        ],
    ),
    (
        "src_toyota", "2021-06-15", "session_b",
        ("David had been saving for two years, and in 2021 he bought a brand-new sedan. "
         "The car was a blue Toyota. He drives it to work every morning across the bridge."),
        [
            {"claim_id": "toyota_make",
             "key_terms": ["toyota"],
             "probe": "What make of car did David buy?",
             "forbidden": ["honda", "ford"]},
            {"claim_id": "toyota_colour",
             "key_terms": ["blue"],
             "probe": "What colour was David's new car?",
             "forbidden": ["red", "silver", "black"]},
            {"claim_id": "toyota_bodystyle",
             "key_terms": ["sedan"],
             "probe": "What body style was David's Toyota?",
             "forbidden": ["truck", "suv", "coupe"]},
        ],
    ),
    (
        "src_berlin", "2022-09-01", "session_c",
        ("Priya accepted an offer in the autumn of 2022. She had been interviewing for "
         "months. The role was a data engineering position, and it meant moving to Berlin. "
         "She started in September."),
        [
            {"claim_id": "berlin_role",
             "key_terms": ["data", "engineering"],
             "probe": "What job did Priya start in 2022?",
             "forbidden": ["scientist", "manager"]},
            {"claim_id": "berlin_city",
             "key_terms": ["berlin"],
             "probe": "Which city did Priya move to for the new job?",
             "forbidden": ["munich", "hamburg"]},
            {"claim_id": "berlin_relocate", "judge_eval": True,
             "key_terms": ["moving", "berlin"],
             "paraphrase": "Priya had to change the city she lived in for work.",
             "probe": "Did the new job require Priya to live somewhere else?",
             "forbidden": ["stayed", "remote"]},
        ],
    ),
    (
        "src_telescope", "2022-12-10", "session_d",
        ("Over the long winter Lena spent her weekends in the garage. Piece by piece she "
         "assembled a reflecting telescope from a kit. By spring she could see the rings of "
         "Saturn through it."),
        [
            {"claim_id": "telescope_what",
             "key_terms": ["reflecting", "telescope"],
             "probe": "What instrument did Lena build over the winter?",
             "forbidden": ["microscope", "refracting"]},
            {"claim_id": "telescope_where",
             "key_terms": ["garage"],
             "probe": "Where did Lena assemble the telescope?",
             "forbidden": ["kitchen", "shed"]},
            {"claim_id": "telescope_saw",
             "key_terms": ["saturn"],
             "probe": "What did Lena observe through her telescope?",
             "forbidden": ["mars", "jupiter"]},
        ],
    ),
    (
        "src_marathon", "2023-05-21", "session_e",
        ("Marcus trained all spring for the coastal marathon. On race day the weather was "
         "perfect. He crossed the finish line in just under four hours, a personal best."),
        [
            {"claim_id": "marathon_event",
             "key_terms": ["coastal", "marathon"],
             "probe": "What race did Marcus run in the spring?",
             "forbidden": ["sprint", "triathlon"]},
            {"claim_id": "marathon_time",
             "key_terms": ["four", "hours"],
             "probe": "Roughly how long did Marcus take to finish?",
             "forbidden": ["three", "five"]},
            {"claim_id": "marathon_pb", "judge_eval": True,
             "key_terms": ["personal", "best"],
             "paraphrase": "It was the fastest Marcus had ever run that distance.",
             "probe": "Was this Marcus's quickest time at this distance?",
             "forbidden": ["slowest", "worst"]},
        ],
    ),
    (
        "src_violin", "2023-03-30", "session_f",
        ("Before the recital Sofia carefully restrung an old violin. It was not just any "
         "violin — it had belonged to her grandmother. She tuned each string by ear."),
        [
            {"claim_id": "violin_action",
             "key_terms": ["restrung", "violin"],
             "probe": "What did Sofia do to the violin before the recital?",
             "forbidden": ["sold", "broke"]},
            {"claim_id": "violin_owner",
             "key_terms": ["grandmother"],
             "probe": "Whose violin was it originally?",
             "forbidden": ["mother", "aunt", "sister"]},
        ],
    ),
    (
        "src_greenhouse", "2022-08-05", "session_g",
        ("Omar is proud of his small backyard greenhouse. In it he grows heirloom "
         "tomatoes, the kind you cannot buy in shops. He waters them every evening."),
        [
            {"claim_id": "greenhouse_crop",
             "key_terms": ["heirloom", "tomatoes"],
             "probe": "What does Omar grow in his greenhouse?",
             "forbidden": ["peppers", "cucumbers"]},
            {"claim_id": "greenhouse_place",
             "key_terms": ["backyard", "greenhouse"],
             "probe": "Where does Omar grow his tomatoes?",
             "forbidden": ["balcony", "field"]},
        ],
    ),
    (
        "src_canoe", "2022-09-22", "session_h",
        ("Diego spent the late summer carving a canoe out of cedar. When it was finished he "
         "carried it down to the lake and paddled all the way across to the far shore."),
        [
            {"claim_id": "canoe_material",
             "key_terms": ["cedar"],
             "probe": "What wood did Diego use for his canoe?",
             "forbidden": ["oak", "pine", "birch"]},
            {"claim_id": "canoe_use",
             "key_terms": ["lake"],
             "probe": "Where did Diego paddle the canoe?",
             "forbidden": ["river", "ocean", "sea"]},
        ],
    ),
]


def build_dataset() -> dict:
    """Assemble the full A7 dataset dict (deterministic; fixed-seed source shuffle only)."""
    sources: list[dict] = []
    claim_total = 0
    judge_total = 0

    for sid, vf, tag, text, claims in _SOURCES:
        out_claims: list[dict] = []
        for c in claims:
            claim = {
                "claim_id": c["claim_id"],
                "key_terms": [t.lower() for t in c["key_terms"]],
                "probe": c["probe"],
                "forbidden": [t.lower() for t in c.get("forbidden", [])],
                "judge_eval": bool(c.get("judge_eval", False)),
            }
            if claim["judge_eval"]:
                # The judge checks the recalled atom against this restatement; it must be
                # present for every judge_eval claim (authoring invariant).
                claim["paraphrase"] = c["paraphrase"]
                judge_total += 1
            out_claims.append(claim)
            claim_total += 1
        sources.append({
            "id": sid,
            "valid_from": vf,
            "source": tag,
            "text": text,
            "claims": out_claims,
        })

    # Deterministic store-order shuffle of the SOURCES list so the corpus is not trivially
    # ordered, while staying reproducible. Claim order within a source is left stable (the
    # report order). The shuffle is on sources only — claims never move relative to their
    # source, so gold mapping is invariant.
    rng = random.Random(SEED)
    rng.shuffle(sources)

    return {
        "schema_version": SCHEMA_VERSION,
        "axis": "a7_distill",
        "seed": SEED,
        "k_values": list(K_VALUES),
        "description": (
            "A7 distillation-fidelity corpus: verbose multi-claim sources, each annotated "
            "with the gold claims a faithful atomization must preserve. The source is "
            "distilled by SAM/IA's offline atomizer, the atoms are stored and recalled, "
            "and claim-preservation F1 is scored programmatically from key-term coverage "
            "(primary, no model). A separate paraphrase subset (judge_eval claims) is "
            "scored by a pinned local judge with saved transcripts (open-ended only)."
        ),
        "source_count": len(sources),
        "claim_count": claim_total,
        "judge_claim_count": judge_total,
        "sources": sources,
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
    print(f"sources={dataset['source_count']} claims={dataset['claim_count']} "
          f"judge_claims={dataset['judge_claim_count']}")
    print(f"sha256(dataset.json)={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
