"""Generator for the A4 contradiction / belief-update dataset (fixed + checksummed).

A4 measures belief-update: a fact ``X`` is asserted, then later contradicted by ``¬X``
carrying *more evidence*, and the system is expected to demote the stale claim and serve
the updated belief. This module emits a deterministic, hand-curated dataset of independent
contradiction cases and writes it next to a SHA256 manifest so any third party regenerates
byte-identical data and verifies it offline.

Run::

    python benchmarks/data/a4_contradiction/generate.py

It writes ``cases.jsonl`` (one case per line) + ``SHA256SUMS`` in this directory. The task
module (``benchmarks/tasks/a4_contradiction.py``) loads ``cases.jsonl`` and asserts the
checksum before scoring, so a tampered or stale dataset fails loudly rather than silently
changing a number.

Determinism + defect fixes (see ``BENCHMARK_DESIGN_v1.md``):

* **D6 (retrieval != retention):** this is a *belief-update* task, not retrieval and not
  retention-after-delay. Each case carries its own pair + distractors; nothing here is
  interleaved over time or conflated with the A1/A2 data.
* **D1/D2/D4 (clean gold):** every case is a single-attribute flip (``X`` vs a *direct*
  ¬X on the same subject+attribute), so the "old belief" and "new belief" ids are
  unambiguous. Distractors are on distinct subjects/attributes so they can never be the
  correct demotion target. Each case states its gold (``new_id`` survives, ``old_id`` is
  demoted) and a one-line rationale.
* **D5 (no reader/judge confound):** the gold is an *id* check (which memory survives,
  which is demoted, what recall returns), scored programmatically — no generated prose is
  read, no LLM judge is in the loop. A4's metrics are exact-id, so the pinned judge is N/A
  (the task records that honestly).

There is no RNG-driven content here: the cases are an explicit, reviewed list (the seed
only fixes the *order* in which distractor pools are drawn, and that draw is itself
deterministic). "Seeded" for A4 means: same source list -> same file -> same checksum.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CASES_PATH = DATA_DIR / "cases.jsonl"
SUMS_PATH = DATA_DIR / "SHA256SUMS"

# Fixed seed recorded for provenance. A4's content is an explicit reviewed list (no random
# sampling), so the seed only documents the dataset revision; same list -> same checksum.
SEED = 1337
DATASET_VERSION = "a4-v1"

# --------------------------------------------------------------------------------------
# The contradiction cases. Each is a single-attribute belief flip on a distinct subject:
#   X      : the original claim (asserted first; earlier valid_from; weaker evidence).
#   not_X  : the updated claim (asserted later; later valid_from; MORE evidence) — the
#            belief that must survive and be served.
# The two differ on exactly one attribute value (city / role / color / count / ...), so a
# direct contradiction with a single, unambiguous correct demotion target. Subjects are
# unique across cases so cases never cross-contaminate when stored together.
#
# not_X is authored in two parts so the belief-update is faithful to how SAM/IA detects it:
#   notx_claim    : the terse, SUBJECT+ATTRIBUTE-parallel restatement with the NEW value
#                   ("Elena now lives in Lisbon."). It is the contradiction *signal* — it
#                   sits in the same embedding neighbourhood as X (measured cosine ~0.7-0.8,
#                   above SAM/IA's hand-written 0.57 supersession bar), so the embedding
#                   supersession finder pairs it with X.
#   notx_evidence : the corroboration that makes the update "more evidence" (lease/contract/
#                   invoice/...). This is what justifies demoting X rather than the reverse.
# The stored not_X node body = notx_claim + " " + notx_evidence. Detection runs on the
# claim text (the signal), exactly as an online belief-update write would carry the parallel
# assertion; the evidence + later valid_from drive the deterministic loser pick.
# --------------------------------------------------------------------------------------
_CASES: list[dict] = [
    {
        "case_id": "c01_city",
        "subject": "Elena",
        "attribute": "home city",
        "x_text": "Elena lives in Berlin.",
        "x_valid_from": "2021-01-10",
        "x_source": "s_offhand_remark",
        "notx_claim": "Elena now lives in Lisbon.",
        "notx_evidence": "Her lease, utility bills, and updated ID all list the "
                         "Lisbon address.",
        "notx_valid_from": "2023-08-01",
        "notx_source": "s_lease_documents",
        "query": "Where does Elena live now?",
        "rationale": "Lisbon supersedes Berlin: later date, documented evidence.",
    },
    {
        "case_id": "c02_role",
        "subject": "Marcus",
        "attribute": "job title",
        "x_text": "Marcus works as a junior analyst.",
        "x_valid_from": "2020-03-01",
        "x_source": "s_old_bio",
        "notx_claim": "Marcus now works as a senior manager.",
        "notx_evidence": "His promotion is confirmed by the company org chart and "
                         "his signed contract.",
        "notx_valid_from": "2023-05-15",
        "notx_source": "s_org_chart_contract",
        "query": "What is Marcus's current job title?",
        "rationale": "Senior manager supersedes junior analyst: later, contract-backed.",
    },
    {
        "case_id": "c03_color",
        "subject": "the Hartley house",
        "attribute": "exterior color",
        "x_text": "The Hartley house is painted white.",
        "x_valid_from": "2019-06-01",
        "x_source": "s_old_photo",
        "notx_claim": "The Hartley house is now painted blue.",
        "notx_evidence": "The contractor invoice and recent street-view photos "
                         "show the blue repaint.",
        "notx_valid_from": "2023-04-20",
        "notx_source": "s_invoice_photos",
        "query": "What color is the Hartley house?",
        "rationale": "Blue supersedes white: later repaint with invoice + photo evidence.",
    },
    {
        "case_id": "c04_count",
        "subject": "the Okonkwo family",
        "attribute": "number of children",
        "x_text": "The Okonkwo family has two children.",
        "x_valid_from": "2018-02-01",
        "x_source": "s_old_census_note",
        "notx_claim": "The Okonkwo family now has three children.",
        "notx_evidence": "The hospital birth record of their youngest confirms "
                         "the third child.",
        "notx_valid_from": "2022-11-30",
        "notx_source": "s_birth_record",
        "query": "How many children does the Okonkwo family have?",
        "rationale": "Three supersedes two: later birth record is direct evidence.",
    },
    {
        "case_id": "c05_team",
        "subject": "Priya",
        "attribute": "employer",
        "x_text": "Priya is employed by Nimbus Systems.",
        "x_valid_from": "2021-09-01",
        "x_source": "s_old_directory",
        "notx_claim": "Priya is now employed by Aurora Labs.",
        "notx_evidence": "Her new badge and the Aurora payroll record show the move.",
        "notx_valid_from": "2024-01-08",
        "notx_source": "s_badge_payroll",
        "query": "Who is Priya's current employer?",
        "rationale": "Aurora Labs supersedes Nimbus: later, payroll-backed move.",
    },
    {
        "case_id": "c06_diet",
        "subject": "Tomas",
        "attribute": "dietary preference",
        "x_text": "Tomas eats a standard omnivore diet.",
        "x_valid_from": "2019-01-01",
        "x_source": "s_casual_note",
        "notx_claim": "Tomas now eats a strict vegetarian diet.",
        "notx_evidence": "His dietitian and a year of logged vegetarian meal plans "
                         "confirm the change.",
        "notx_valid_from": "2023-02-14",
        "notx_source": "s_dietitian_logs",
        "query": "What is Tomas's diet now?",
        "rationale": "Vegetarian supersedes omnivore: later, dietitian-documented change.",
    },
    {
        "case_id": "c07_status",
        "subject": "the Riverside bridge",
        "attribute": "operational status",
        "x_text": "The Riverside bridge is open to traffic.",
        "x_valid_from": "2020-07-01",
        "x_source": "s_old_map",
        "notx_claim": "The Riverside bridge is now closed to traffic.",
        "notx_evidence": "The city engineering department issued an official "
                         "closure notice for structural repairs.",
        "notx_valid_from": "2023-09-12",
        "notx_source": "s_closure_notice",
        "query": "Is the Riverside bridge open or closed?",
        "rationale": "Closed supersedes open: later official engineering closure notice.",
    },
    {
        "case_id": "c08_owner",
        "subject": "the corner bakery",
        "attribute": "owner",
        "x_text": "The corner bakery is owned by Mrs. Dela Cruz.",
        "x_valid_from": "2017-05-01",
        "x_source": "s_old_signage",
        "notx_claim": "The corner bakery is now owned by Mr. Adeyemi.",
        "notx_evidence": "The notarized bill of sale and the new business license "
                         "record the ownership change.",
        "notx_valid_from": "2023-06-30",
        "notx_source": "s_bill_of_sale",
        "query": "Who currently owns the corner bakery?",
        "rationale": "Adeyemi supersedes Dela Cruz: later, notarized sale + license.",
    },
    {
        "case_id": "c09_season_pref",
        "subject": "Yuki",
        "attribute": "favorite season",
        "x_text": "Yuki's favorite season is summer.",
        "x_valid_from": "2018-08-01",
        "x_source": "s_old_chat",
        "notx_claim": "Yuki's favorite season is now winter.",
        "notx_evidence": "She has stated this repeatedly in recent conversations "
                         "and on her updated profile.",
        "notx_valid_from": "2024-02-01",
        "notx_source": "s_recent_profile",
        "query": "What is Yuki's favorite season now?",
        "rationale": "Winter supersedes summer: later, repeated + profile-confirmed.",
    },
    {
        "case_id": "c10_phone",
        "subject": "the Lindqvist clinic",
        "attribute": "contact phone number",
        "x_text": "The Lindqvist clinic's phone number is 555-0182.",
        "x_valid_from": "2019-04-01",
        "x_source": "s_old_flyer",
        "notx_claim": "The Lindqvist clinic's phone number is now 555-0467.",
        "notx_evidence": "The new number is listed on its current website and the "
                         "latest printed directory.",
        "notx_valid_from": "2023-10-05",
        "notx_source": "s_current_website",
        "query": "What is the Lindqvist clinic's phone number now?",
        "rationale": "555-0467 supersedes 555-0182: later, website + directory backed.",
    },
]

# --------------------------------------------------------------------------------------
# Distractor pool: unrelated, NON-contradicting facts on subjects that appear in NO case.
# They populate the store so recall must discriminate the updated belief from plausible
# neighbours; none of them is ever a valid demotion target (distinct subject+attribute),
# which keeps the gold unambiguous (D1/D2/D4). The assignment of distractors to cases is
# deterministic (round-robin by case order), so the dataset is reproducible.
# --------------------------------------------------------------------------------------
_DISTRACTORS: list[str] = [
    "The Watanabe garden grows tomatoes and basil every spring.",
    "Coastal Freight runs a daily route between the north and south depots.",
    "The Alvarez twins both play the violin in the youth orchestra.",
    "Greenfield Library is open until nine on weekday evenings.",
    "The Sorensen ferry carries bicycles at no extra charge.",
    "Old Mill Road was resurfaced with permeable asphalt this year.",
    "The Fenwick museum added a wing for maritime history.",
    "Captain Reyes pilots the harbor tour boat on weekends.",
    "The Delgado vineyard bottles a dry rosé each autumn.",
    "Northgate Station has six platforms and a glass roof.",
    "The Halvorsen choir rehearses on Thursday nights.",
    "Maple Court apartments share a rooftop vegetable plot.",
]

# How many distractors to attach to each case (kept small + fixed so the store stays a
# tight, reviewable population per case while still forcing discrimination).
_DISTRACTORS_PER_CASE = 3


def _build_cases() -> list[dict]:
    """Materialize the full case list with stable ids + a deterministic distractor draw.

    Each emitted case carries the three node payloads the task stores (old belief X, new
    belief not_X, and its distractors) plus the gold labels the scorer reads:
    ``old_id`` (must be demoted), ``new_id`` (must survive + be served), the probe
    ``query``, and a human ``rationale``. Distractors are drawn round-robin from the shared
    pool by case index so the assignment is fixed and never overlaps a case's own subject.
    """
    out: list[dict] = []
    pool_n = len(_DISTRACTORS)
    cursor = 0
    for case in _CASES:
        cid = case["case_id"]
        old_id = f"{cid}__x"          # the original claim node (to be demoted)
        new_id = f"{cid}__notx"       # the updated claim node (to survive)
        distractor_nodes = []
        for j in range(_DISTRACTORS_PER_CASE):
            text = _DISTRACTORS[(cursor + j) % pool_n]
            distractor_nodes.append({
                "id": f"{cid}__d{j}",
                "text": text,
            })
        cursor = (cursor + _DISTRACTORS_PER_CASE) % pool_n
        # not_X stored body = the parallel claim (detection signal) + its evidence.
        notx_body = f"{case['notx_claim']} {case['notx_evidence']}".strip()
        out.append({
            "case_id": cid,
            "subject": case["subject"],
            "attribute": case["attribute"],
            "old": {
                "id": old_id,
                "text": case["x_text"],
                "valid_from": case["x_valid_from"],
                "source": case["x_source"],
            },
            "new": {
                "id": new_id,
                "text": notx_body,
                # The terse parallel restatement of the new belief — the contradiction
                # signal the embedding supersession finder pairs against X. Carried
                # separately so detection runs on the signal, not the diluting evidence.
                "claim": case["notx_claim"],
                "valid_from": case["notx_valid_from"],
                "source": case["notx_source"],
            },
            "distractors": distractor_nodes,
            "query": case["query"],
            # Gold: the updated belief survives + is served; the original is demoted.
            "gold": {
                "survives_id": new_id,
                "demoted_id": old_id,
            },
            "rationale": case["rationale"],
        })
    return out


def _serialize(cases: list[dict]) -> str:
    """Render cases as deterministic JSONL (sorted keys, one object per line)."""
    lines = [json.dumps(c, sort_keys=True, ensure_ascii=True) for c in cases]
    return "\n".join(lines) + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_dataset() -> dict:
    """Write ``cases.jsonl`` + ``SHA256SUMS`` and return a small build summary."""
    cases = _build_cases()
    payload = _serialize(cases)
    CASES_PATH.write_text(payload, encoding="utf-8")
    digest = _sha256(payload)
    SUMS_PATH.write_text(f"{digest}  cases.jsonl\n", encoding="utf-8")
    return {
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "n_cases": len(cases),
        "cases_sha256": digest,
        "path": str(CASES_PATH),
    }


if __name__ == "__main__":
    summary = write_dataset()
    print(json.dumps(summary, indent=2))
