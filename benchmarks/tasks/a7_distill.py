"""A7 — Distillation fidelity (claim-preservation).

Task (deterministic, seeded)
----------------------------
For each verbose source in the fixed A7 dataset:

  1. **Distil.** Run SAM/IA's offline atomizer (``fact_extractor.extract_atoms_rule`` — the
     deterministic, network-free structural splitter) over the verbose source text. This is
     the real distillation surface the design names for A7 (``core/fact_extractor``,
     atomization): one wordy blob becomes one or more atoms.
  2. **Store.** Persist each distilled atom as a ``type: semantic`` node via the adapter and
     build the real MiniLM index (the same write/index path A1 uses).
  3. **Recall.** For each gold claim, probe the store with the claim's paraphrased question
     and take the top-k atom(s). This exercises the genuine semantic recall path, so a claim
     only counts as preserved if a faithful atom both survived distillation AND is
     retrievable — distillation fidelity end to end, not just "did the splitter emit text".
  4. **Score** (two separate metrics, never mixed — design rule D5):
       * **Primary, programmatic — claim-preservation F1.** No model. For every gold claim,
         the recalled atom text is checked for the claim's ``key_terms`` (all must survive)
         and its ``forbidden`` distractors (none may appear). A claim is *preserved* when its
         terms survive into a recalled atom with no distortion. F1 combines:
           - recall  = preserved gold claims / all gold claims (did distillation keep them),
           - precision = preserved gold claims / claims whose recalled atom asserted them
             (of the claims the atoms actually carried, how many were undistorted).
       * **Open-ended subset — pinned judge.** Only the dataset's ``judge_eval`` claims
         (paraphrased gold with no surface-token overlap) go to a pinned local judge with a
         fixed prompt at temperature 0; every transcript is saved. If the pinned judge model
         is not reachable, the judge metric is reported ``N/A`` with a reason — never
         fabricated (mirrors the A5 no-judge honesty rail).

Why A7 is its OWN task/data (defect D6)
---------------------------------------
A7 has a corpus and probes that exist nowhere else: *verbose* sources annotated with the
claims a distillation must keep. A1 (retrieval) stores already-atomic facts; A2 (retention)
measures decay after delay. A7 is the only axis that stores a *verbose* blob and asks
whether the atomized form still carries the source's claims. Conflating it with retrieval or
retention is the field's most common defect; the data is deliberately disjoint.

Grounding
---------
Every SAM/IA call routes through the installed package via the adapter and the public
``fact_extractor`` surface; nothing here reimplements SAM/IA behavior. The atomizer is the
deterministic rule splitter (no LLM, no network); the index + recall are the same MiniLM
path A1 uses; the judge (open-ended subset only) is the package's pinned local judge model.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make the harness package importable whether this module is run as a file
# (``python benchmarks/tasks/a7_distill.py``) or imported as
# ``benchmarks.tasks.a7_distill`` from the harness — same bootstrap the other axes use.
_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from adapters import MemoryItem, SamiaAdapter  # noqa: E402

# The offline, deterministic distillation surface named by the A7 design row
# (core/fact_extractor, atomization). extract_atoms_rule never reaches the network and is a
# pure function of its input, so the distillation step is fully reproducible.
from samia.core import fact_extractor as _fact_extractor


# ---------------------------------------------------------------------------
# Dataset loading + checksum guard (no network; fixed bytes)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "a7_distill"
_DATASET_PATH = _DATA_DIR / "dataset.json"
_SUMS_PATH = _DATA_DIR / "SHA256SUMS"


def _verify_and_load_dataset() -> dict:
    """Load the fixed A7 dataset, refusing to run if its bytes do not match SHA256SUMS.

    The committed dataset is the single source of truth for the gold claims; a checksum
    mismatch means the data was edited out from under the scorer, so the task fails loudly
    rather than scoring against unknown bytes (design non-negotiable #3).
    """
    raw = _DATASET_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = _SUMS_PATH.read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise RuntimeError(
            f"A7 dataset checksum mismatch: {digest} != {expected}. "
            f"Regenerate with data/a7_distill/generate.py and re-commit.")
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Programmatic claim-preservation primitives (no model)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercased content-token set of ``text`` for whole-word claim-term matching."""
    return set(_WORD_RE.findall(text.lower()))


def _claim_present(atom_text: str, key_terms: list[str]) -> bool:
    """True iff EVERY key term of the claim survives as a whole word in the atom text.

    All terms must be present (a claim is only preserved when its full content survives —
    a partial match is a dropped claim, not a kept one).
    """
    toks = _tokens(atom_text)
    return all(term in toks for term in key_terms)


def _claim_distorted(atom_text: str, forbidden: list[str]) -> bool:
    """True iff any forbidden distractor token appears as a whole word in the atom text.

    A distortion means the atom asserts a *wrong* version of the claim (wrong colour, wrong
    number, ...). Such a claim is neither preserved nor counted as a clean assertion.
    """
    if not forbidden:
        return False
    toks = _tokens(atom_text)
    return any(term in toks for term in forbidden)


# ---------------------------------------------------------------------------
# Pinned judge wrapper (open-ended paraphrase subset ONLY)
# ---------------------------------------------------------------------------

# Pinned judge model: the package's small fallback judge, fixed by name here so the open-
# ended subset is scored by ONE named local model (transcripts saved, re-scoreable). The
# judge runs at temperature 0 for determinism; it is a LOCAL model (no internet).
_PINNED_JUDGE_MODEL = "phi4-mini:latest"

# Fixed judge prompt for claim preservation. It is NOT the package's tool-call security
# prompt (that judges a different question); A7 pins its own claim-preservation prompt and
# parses the same VERDICT/CONFIDENCE/RATIONALE three-line contract the package's
# _parse_response understands. KEEP = the atom preserves the claim; DROP = it does not.
_JUDGE_PROMPT_TEMPLATE = (
    "You check whether a distilled memory atom preserves a source claim.\n"
    "Reply with EXACTLY three lines and nothing else:\n"
    "VERDICT: KEEP|DROP\n"
    "CONFIDENCE: <0.0-1.0>\n"
    "RATIONALE: <one short sentence>\n"
    "KEEP means the atom still asserts the claim (a paraphrase counts). DROP means the "
    "atom does not assert it or asserts a contradicting version.\n\n"
    "CLAIM: {claim}\n"
    "ATOM: {atom}\n"
)


@dataclass
class JudgeResult:
    available: bool
    model: str
    transcripts: list[dict] = field(default_factory=list)
    note: str = ""

    def keep_rate(self) -> float | None:
        if not self.available or not self.transcripts:
            return None
        kept = sum(1 for t in self.transcripts if t["verdict"] == "keep")
        return kept / len(self.transcripts)


def _judge_available() -> bool:
    """True iff the pinned local judge transport is reachable (best-effort, no exception)."""
    try:
        from samia.core import judge as _judge
        return bool(_judge._ollama_reachable(timeout_s=2.0))
    except Exception:
        return False


def _run_judge(claim_paraphrase: str, atom_text: str) -> tuple[str, float, str, str]:
    """Send one claim/atom pair to the pinned judge; return (verdict, conf, rationale, raw).

    Routes through the package's ollama transport with a FIXED prompt at temperature 0 (set
    on the payload here, overriding the transport default) so the judgment is deterministic
    and the model is pinned by name. The raw response is returned for the saved transcript.
    """
    from samia.core import judge as _judge

    prompt = _JUDGE_PROMPT_TEMPLATE.format(claim=claim_paraphrase, atom=atom_text)
    # Deterministic judge call: temperature 0, fixed model, low token budget. Built here
    # (not via _judge.judge) so the prompt is the A7 claim-preservation prompt and the
    # sampling is pinned to 0 — the package transport defaults to 0.1.
    import json as _json
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    payload = {
        "model": _PINNED_JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 1337, "num_predict": 120},
    }
    data = _json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(
        f"{_judge.OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=30.0) as r:
            raw = _json.loads(r.read().decode("utf-8")).get("response", "")
    except (_urlerr.URLError, OSError, _json.JSONDecodeError) as exc:
        return "unsure", 0.0, f"judge transport error: {type(exc).__name__}", ""
    verdict, conf, rationale = _judge._parse_response(raw)
    # _parse_response is tuned for allow/deny/unsure; normalise our KEEP/DROP from the raw.
    up = raw.upper()
    if "VERDICT: KEEP" in up or re.search(r"\bKEEP\b", up):
        verdict = "keep"
    elif "VERDICT: DROP" in up or re.search(r"\bDROP\b", up):
        verdict = "drop"
    else:
        verdict = "unsure"
    return verdict, conf, rationale, raw


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

@dataclass
class A7Result:
    """All A7 outputs: the programmatic F1 + per-claim detail + the judge subset summary."""
    axis: str
    seed: int
    source_count: int
    claim_count: int
    # programmatic (primary)
    claims_preserved: int
    claims_asserted: int        # claims whose recalled atom carried the terms at all
    claim_recall: float
    claim_precision: float
    claim_f1: float
    per_claim: list[dict]
    # open-ended (judge subset)
    judge: dict
    timing_s: float

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        return d


def run(adapter: SamiaAdapter | None = None, *, k: int = 5,
        run_judge: bool = True) -> A7Result:
    """Run the A7 distillation-fidelity task and return the scored result.

    Parameters
    ----------
    adapter:
        A SamiaAdapter to exercise. When None, one is created and closed for the run.
    k:
        Top-k atoms taken per claim probe when locating the atom that should carry a claim.
    run_judge:
        Whether to attempt the pinned-judge pass over the paraphrase subset. The judge is
        only invoked when its local transport is reachable; otherwise the judge metric is
        reported N/A with a reason (never fabricated).
    """
    dataset = _verify_and_load_dataset()
    t0 = time.time()

    own_adapter = adapter is None
    adapter = adapter or SamiaAdapter()
    try:
        adapter.reset()

        # --- 1+2: distil every verbose source and store the resulting atoms ---------
        # Each distilled atom gets an id derived from its source so the per-claim audit can
        # name where a kept/dropped claim came from. Atom ids are filesystem-safe slugs.
        atom_text_by_id: dict[str, str] = {}
        items: list[MemoryItem] = []
        for src in dataset["sources"]:
            atoms = _fact_extractor.extract_atoms_rule(src["text"])
            for i, atom in enumerate(atoms):
                aid = f"{src['id']}_atom{i:02d}"
                body = atom["body"]
                atom_text_by_id[aid] = body
                items.append(MemoryItem(
                    id=aid, text=body,
                    valid_from=atom.get("valid_from", "") or src.get("valid_from", ""),
                    source=src["source"]))
        adapter.store(items)
        adapter.build_index()

        # --- 3+4: probe each claim, score programmatically --------------------------
        per_claim: list[dict] = []
        preserved = 0      # gold claims kept faithfully (recall numerator)
        asserted = 0       # claims whose recalled atom carried the terms (precision denom)
        judge_jobs: list[tuple[dict, str]] = []   # (claim, recalled_atom_text)

        for src in dataset["sources"]:
            for claim in src["claims"]:
                ranked = adapter.recall(claim["probe"], k=k)
                # The atom that should carry this claim is the best-ranked atom whose text
                # contains the claim's key terms; if none of the top-k carry them, the claim
                # was not recalled (and, because the atoms are the only population, was not
                # faithfully distilled into a retrievable atom).
                hit_id = None
                hit_text = ""
                for aid in ranked:
                    txt = atom_text_by_id.get(aid, "")
                    if _claim_present(txt, claim["key_terms"]):
                        hit_id, hit_text = aid, txt
                        break
                # "asserted" = some recalled atom carried the claim's terms (present), so the
                # atoms made a claim on this fact at all. Used as the precision denominator
                # together with distortion checks.
                terms_present = hit_id is not None
                distorted = bool(hit_id) and _claim_distorted(hit_text, claim["forbidden"])
                is_preserved = terms_present and not distorted

                # A claim also counts toward "asserted" if its terms appeared in the top-1
                # atom even distorted — i.e. the atoms asserted *something* about this fact.
                # We approximate "asserted" as terms_present (the atom carried the claim's
                # content); a distorted assertion lowers precision without lowering recall's
                # denominator.
                if terms_present:
                    asserted += 1
                if is_preserved:
                    preserved += 1

                rec = {
                    "source_id": src["id"],
                    "claim_id": claim["claim_id"],
                    "judge_eval": claim.get("judge_eval", False),
                    "probe": claim["probe"],
                    "recalled": ranked[:k],
                    "hit_atom": hit_id,
                    "terms_present": terms_present,
                    "distorted": distorted,
                    "preserved": is_preserved,
                }
                per_claim.append(rec)

                if claim.get("judge_eval", False):
                    # The paraphrase subset is scored by the judge against the TOP recalled
                    # atom (best candidate the recall surfaced), regardless of token overlap.
                    top_text = atom_text_by_id.get(ranked[0], "") if ranked else ""
                    judge_jobs.append((claim, top_text))

        claim_count = len(per_claim)
        claim_recall = preserved / claim_count if claim_count else 0.0
        claim_precision = preserved / asserted if asserted else 0.0
        claim_f1 = (
            2 * claim_precision * claim_recall / (claim_precision + claim_recall)
            if (claim_precision + claim_recall) else 0.0)

        # --- open-ended subset: pinned judge over the paraphrase claims -------------
        judge_res = JudgeResult(available=False, model=_PINNED_JUDGE_MODEL)
        if run_judge and judge_jobs:
            if _judge_available():
                judge_res.available = True
                for claim, atom_text in judge_jobs:
                    verdict, conf, rationale, raw = _run_judge(
                        claim["paraphrase"], atom_text)
                    judge_res.transcripts.append({
                        "claim_id": claim["claim_id"],
                        "paraphrase": claim["paraphrase"],
                        "atom": atom_text,
                        "verdict": verdict,
                        "confidence": conf,
                        "rationale": rationale,
                        "raw": raw,
                    })
            else:
                judge_res.note = (
                    "pinned judge transport not reachable in this environment; "
                    "paraphrase subset reported N/A (not fabricated)")
        elif not judge_jobs:
            judge_res.note = "no judge_eval claims in dataset"
        else:
            judge_res.note = "judge pass disabled for this run (run_judge=False)"

        keep_rate = judge_res.keep_rate()
        judge_summary = {
            "model": judge_res.model,
            "available": judge_res.available,
            "claim_count": len(judge_jobs),
            "keep_rate": keep_rate,
            "note": judge_res.note,
            "transcripts": judge_res.transcripts,
        }

        return A7Result(
            axis=dataset["axis"],
            seed=dataset["seed"],
            source_count=dataset["source_count"],
            claim_count=claim_count,
            claims_preserved=preserved,
            claims_asserted=asserted,
            claim_recall=round(claim_recall, 4),
            claim_precision=round(claim_precision, 4),
            claim_f1=round(claim_f1, 4),
            per_claim=per_claim,
            judge=judge_summary,
            timing_s=round(time.time() - t0, 3),
        )
    finally:
        if own_adapter:
            adapter.close()


def main() -> int:
    """CLI: run A7 against SAM/IA and print the programmatic F1 + judge-subset summary."""
    res = run()
    print(f"axis={res.axis} seed={res.seed} sources={res.source_count} "
          f"claims={res.claim_count}")
    print(f"claim-preservation: preserved={res.claims_preserved}/{res.claim_count} "
          f"asserted={res.claims_asserted}")
    print(f"  recall={res.claim_recall}  precision={res.claim_precision}  "
          f"F1={res.claim_f1}")
    jr = res.judge
    if jr["available"]:
        print(f"  paraphrase judge ({jr['model']}): keep_rate={jr['keep_rate']} "
              f"over {jr['claim_count']} claims (transcripts saved)")
    else:
        print(f"  paraphrase judge: N/A — {jr['note']}")
    print(f"timing={res.timing_s}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
