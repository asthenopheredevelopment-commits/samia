# SAM/IA Capability Report (v1)

**What this is.** An independently-reproducible measurement of *what SAM/IA can do*, across
nine capability axes, produced by the harness in this directory against the **installed**
`samia` package (v1.0.0). It **replaces** the prior `SAMIA_BENCHMARK_COMPENDIUM_*` and
executive-summary documents. Methodology is `BENCHMARK_DESIGN_v1.md`; the raw machine record
is `results/scores.json` + `results/raw_<axis>.jsonl`; every dataset is committed and
checksummed (`data/SHA256SUMS`); one-command reproduction is `repro/` (Dockerfile + seeds).

**How to read it.** There is deliberately **no single aggregate "score"** — that is the one
number that hides which axis is weak. Each axis below states its task, its raw numbers, and
its honest caveat. Where SAM/IA's capability is partial or judge-gated, that is said plainly,
not papered over.

**Run provenance.** seed `1337` · adapter `samia` · package `samia 1.0.0` · python `3.12` ·
embedder `sentence-transformers/all-MiniLM-L6-v2` (CPU, 384-dim, cache-only) · no network at
score time · A9 sizes `100 / 1000 / 10000`. **Determinism: verified** — the benchmark was run
twice with `--check-determinism` and the scoring subtree (every programmatic metric and
gold-id outcome; A9 wall-clock latency legitimately excepted) was byte-identical across runs.

**Defects fixed (from our memory-benchmark defect audit).**
- **D6 — retrieval != retention (universal):** A1 (retrieval, all-present) and A2
  (retention-after-delay) are *separate tasks with separate data*; never conflated.
- **D5 — reader/judge confound:** programmatic exact-id scoring is primary on every axis; the
  pinned local judge runs only on the open-ended subset (A3 / A7; A4 is N/A) at temperature 0
  with saved transcripts, and never gates a programmatic number.
- **D1/D2/D4 — ambiguous/duplicate/incoherent gold:** every item carries one unambiguous gold
  id + rationale; each dataset is SHA256-pinned and the axis refuses to run on a mismatch.

---

## A1 — Retrieval accuracy

**Task.** Seed 100 distinct-topic facts; issue one paraphrased query per fact; is the gold
fact in the recall top-k? Surface: `core/semantic_recall.atom_retrieve` over the real MiniLM
`core/vector` index. Metric: recall@{1,5,10} + MRR, programmatic (gold-id position).

| metric | value |
|---|---|
| recall@1 | **0.95** |
| recall@5 | **1.00** |
| recall@10 | **1.00** |
| MRR | **0.97** |

**Caveat.** Strong, clean retrieval: every gold is in the top-5, 95/100 at rank 1. The 5
non-rank-1 cases are paraphrase queries where a sibling fact out-scores the gold by cosine but
the gold still ranks <=5 — a relevance-ranking nuance, not a miss. No judge (closed-form).

## A2 — Retention / forgetting

**Task.** Seed 6 salient + 18 noise atoms; age them and run SAM/IA's real relevance-decay
pass (`core/tier.decay_pass`, `auto_freeze=True`) for a delay schedule; after the delay, do
the *salient* memories survive while the *noise* is evicted? Two numbers per delay:
retention@k (salient golds still recallable) and noise-drop rate (noise atoms evicted from the
live store). Each delay is an independent fresh store. Programmatic; no judge.

| delay (ticks) | retention@{1,5,10} | noise-drop | salient live | noise live |
|---|---|---|---|---|
| 0 | 1.00 / 1.00 / 1.00 | 0.00 | 6/6 | 18/18 |
| 5 | 1.00 / 1.00 / 1.00 | 0.00 | 6/6 | 18/18 |
| 10 | 1.00 / 1.00 / 1.00 | 0.00 | 6/6 | 18/18 |
| 20 | 1.00 / 1.00 / 1.00 | **1.00** | 6/6 | 0/18 |
| 40 | 1.00 / 1.00 / 1.00 | **1.00** | 6/6 | 0/18 |

**Caveat.** This is the field's most-often-conflated pair, reported separately: SAM/IA keeps
**100% of salient memories at every delay** while shedding **100% of noise by delay 20**. The
forgetting is the genuine `decay_pass` + freeze eviction (salience-1.0 atoms are
freeze-exempt; zero-salience atoms decay past threshold and are archived out of `nodes/`). The
threshold is crossed between delay 10 and 20 — the curve is a step here because all noise
shares one decay rate, which is honest for this fixed corpus, not a smooth forgetting curve.

## A3 — Temporal reasoning

**Task.** Seed dated fact-chains (each topic = one attribute updated over time); ask
"most-recent value of A" and "value of A as-of date D". Surface: `core/temporal.query`
(bi-temporal interval recall) for the time filter + recency argmax within the relevance band.
Metric: temporal-recall@{1,3,5} + ordering accuracy, programmatic. 20 items, 20 queries.

| metric | value |
|---|---|
| temporal_recall@1 / @3 / @5 | **1.00 / 1.00 / 1.00** |
| MRR | **1.00** |
| ordering accuracy (per-topic chronology) | **1.00** |
| · by kind — most_recent (n=6) | @1 = 1.00 |
| · by kind — as_of (n=14) | @1 = 1.00 |

**Caveat.** Perfect on this dataset for both temporal question kinds, including the
"as-of-date" case that must drop a not-yet-true future update. Honest capability boundary:
`temporal.query` returns interval *membership* in node-id order, not recency-ranked — the
harness supplies the recency argmax/ordering over the dates SAM/IA recovers, so the metric
measures SAM/IA's date-fidelity + filter, with the argmax made explicit (see the task
docstring). Open-ended phrasings: a 12-item judge transcript is emitted for audit;
**`available_not_run`** (programmatic gold-id metrics are the trustworthy A3 numbers).

## A4 — Contradiction / belief-update

**Task.** Assert X; later assert not-X with more evidence; the system should demote the stale
claim and serve the updated belief. Surface: `runtime.contradiction.find_supersession_candidates`
(embedding detector) -> `_pick_superseded` (loser rule) -> `core.vector.tombstone_node`
(restorable demote) -> recall. 10 hand-written belief-flip cases. Programmatic (exact-id); no
judge (N/A for A4).

| metric | value |
|---|---|
| contradiction-detected | **10/10 (100%)** |
| loser-pick-correct | **10/10 (100%)** |
| demote-correct (new served rank-1, old gone) | **10/10 (100%)** |
| shadow-persist (old claim leaks back) | **0/10 (0%)** |

**Caveat.** Every deterministic arm of the supersession pipeline scores perfectly: the
contradiction is detected, the correct stale claim is chosen, demoted, and never leaks back.
**Reported scope limit:** SAM/IA's *auto*-supersede (`passive_sweep`) is gated behind an
offline LLM REM judge, which a deterministic network-free run must not invoke — so the
end-to-end *automatic* path is measured-without-judge (the judge-gated trigger is not
exercised here; no number is fabricated for it). The detector bar uses the package's
hand-written cosine floor (0.57) for these hand-written claims, env-set and restored.

## A5 — Consolidation gain

**Task.** Score recall@k *before* vs *after* a consolidation cycle (`MemoryAdapter.consolidate()`
-> `core.consolidation.audit_all` + index rebuild). 16 items (singletons + 4 near-duplicate
clusters), 12 probes. Metric: delta-recall per k, programmatic; no judge.

| metric | before | after | delta |
|---|---|---|---|
| recall@1 | 0.917 | 0.917 | **+0.000** |
| recall@5 | 1.000 | 1.000 | **+0.000** |
| recall@10 | 1.000 | 1.000 | **+0.000** |
| MRR | 0.958 | 0.958 | +0.000 |

consolidate summary: `{judge_applied: false, candidate_merges: 0}`

**Caveat — this is the honest one.** The end-to-end consolidation *gain* is **NOT measurable
under the determinism rules**, and is not faked. SAM/IA's actual atom *merge* is gated by an
offline REM judge (an LLM); a network-free deterministic benchmark must not invoke it. So A5
measures only the consolidation pass **available without the judge** (programmatic audit +
index rebuild), which is recall-**neutral** by design (delta ~ 0) — and a neutral result is
reported, not a fabricated lift. To measure true consolidation gain, run the judge-gated
merge offline and re-score; the harness is built to slot that in.

## A6 — Associative / multi-hop

**Task.** Wire a linked chain a->b->c into SAM/IA's real Hebbian graph by co-activating only
*adjacent* atoms (`core.bio.hebbian.hebbian_record` + `hebbian_consolidate`), then query the
head and expect the tail. Surface: `core.successor.need_vector` (query-local power iteration
over `edge_weights.json`). The chain atoms are deliberately NOT vector-similar head-to-tail,
so the tail is reachable only transitively. 6 probes (4 two-hop, 2 three-hop). Programmatic.

| metric | overall | 2-hop (n=4) | 3-hop (n=2) |
|---|---|---|---|
| recall@1 | **0.00** | 0.00 | 0.00 |
| recall@3 | **1.00** | 1.00 | 1.00 |
| recall@5 / @10 | 1.00 / 1.00 | 1.00 | 1.00 |
| MRR | 0.444 | 0.500 | 0.333 |
| reached-rate (tail got positive walk occupancy) | **1.00** | 1.00 | 1.00 |
| noise-leakage (noise atoms touched by the walk) | **0.00** | — | — |

**Caveat — honest weak spot.** Multi-hop association **works** (the tail is reached on every
probe, recall@3 = 1.00, zero noise leakage — the walk does not diffuse indiscriminately), but
it does **not** put the tail at rank 1 (recall@1 = 0.00, MRR ~ 0.44): the discounted-occupancy
walk ranks intermediate nodes above the terminal tail. So SAM/IA *can* traverse the chain but
does not surface the endpoint as the single best answer. This axis is reported at depth (by
hop count) precisely so this rank-1 gap is visible and not averaged away.

## A7 — Distillation fidelity

**Task.** Store a verbose source; distil it with SAM/IA's offline atomizer
(`core.fact_extractor.extract_atoms_rule`); recall each gold claim and check the atom
preserves it. 8 verbose sources, 22 gold claims. Primary metric: claim-preservation F1
(programmatic, key-term survival with no forbidden distortion). Open-ended subset: 3
paraphrase claims to the pinned judge.

| metric | value |
|---|---|
| claims preserved | **22/22** |
| claim-preservation recall | **1.00** |
| claim-preservation precision | **1.00** |
| **claim-preservation F1** | **1.00** |
| paraphrase judge (`phi4-mini:latest`, temp 0) | keep-rate **1.00** over 3 claims (transcripts saved) |

**Caveat.** Distillation is end-to-end faithful here: every gold claim survives atomization
into a *retrievable* atom with no distortion (the test requires the claim to both survive the
splitter AND be recallable). The judge subset is small (3 paraphrase claims) and ran because a
local judge daemon was up; its transcripts are saved and re-scoreable. If no judge daemon is
present, the paraphrase subset reports N/A and the programmatic F1 stands alone.

## A8 — Provenance / firewall

**Task.** Inject an untrusted/poisoned item next to a trusted one; quarantine the untrusted
item via the real forget primitive (`core.vector.tombstone_node`) and check it is kept out of
recall while the trusted fact survives. 10 probes, each run under both a no-firewall baseline
and a firewall condition. Programmatic set-membership; no judge.

| metric | firewall ON | baseline (no firewall) |
|---|---|---|
| poison-rejection % | **1.00** | — |
| false-trust % (poison served as truth) | **0.00** | 1.00 |
| trusted-retained % | **1.00** | 1.00 |

**Caveat.** The firewall is clean: with quarantine on, **0% of poisoned items are served and
100% of trusted facts are retained**. The baseline false-trust = 1.00 proves the poison is
genuinely recall-reachable, so the firewall-on rejection is attributable to the *quarantine*,
not to the poison being unfindable. Honest scope: SAM/IA's recall path is provenance-blind
(it ranks by cosine and does not read the `trusted` flag); the firewall measured here is the
explicit `tombstone_node` quarantine STEP applied to untrusted-source items, exactly as
`ia.forget_node` uses it — not an implicit property of recall.

## A9 — Latency / scale

**Task.** Store -> build-index -> recall at N = 100 / 1000 / 10000; wall-clock the whole
end-to-end path. Metric: ingest items/s + p50/p95 recall ms per store size, plus a gold-hit
sanity check so a fast-but-empty recall can't read as a latency win. Programmatic.

| N | ingest items/s | index build s | recall p50 ms | recall p95 ms | gold-hit rate |
|---|---|---|---|---|---|
| 100 | 695.0 | 0.14 | 5.5 | 6.8 | **1.00** |
| 1 000 | 805.4 | 1.21 | 6.9 | 8.1 | **0.99** |
| 10 000 | 730.7 | 13.34 | 72.7 | 79.0 | **0.80** |

**Caveat.** Latency is reported next to correctness on purpose. Recall stays single-digit-ms
through N=1000 and p95 < 80 ms at N=10000 on CPU; ingest holds ~700-800 items/s (embed-bound).
**The honest scale note is the gold-hit rate:** retrieval correctness is ~1.0 up to N=1000 but
falls to **0.80 at N=10000** — at ten-thousand atoms, one paraphrase probe in five no longer
finds its gold in the top-10. The latency numbers are valid (the recall path runs and mostly
hits), but accuracy degrades with scale; this is a real finding, surfaced rather than buried.
Wall-clock values vary by machine (that is the nature of a latency benchmark) — the timed
operations, probe sample, and gold-hit outcomes are reproducible; the milliseconds are not.

---

## Reproduce

```sh
# From the release root, with the installed package's interpreter:
python benchmarks/run_benchmark.py --seed 1337 --sizes 100,1000,10000

# Verify determinism (runs twice, asserts the scoring subtree is byte-identical):
python benchmarks/run_benchmark.py --seed 1337 --sizes 100,1000,10000 --check-determinism

# Fully sandboxed, one command: build the repro/ image and run it
# (see repro/Dockerfile for the exact build + run commands; pinned deps + cache-only
# embedder, no network at score time).
```

Datasets are checksummed in `data/SHA256SUMS`; the machine record is `results/scores.json`
(+ `results/raw_<axis>.jsonl`); the methodology is `BENCHMARK_DESIGN_v1.md`; the pinned
environment + seeds are `repro/seeds.toml`.

## Honest one-paragraph summary

SAM/IA is **strong** on retrieval (recall@5 = 1.0), retention-vs-forgetting (100% salient kept
/ 100% noise dropped), temporal reasoning (recall + ordering = 1.0), belief-update (100%
demote-correct, 0% shadow-persist), provenance firewall (0% false-trust, 100% trusted
retained), and distillation fidelity (F1 = 1.0). It is **partial** on multi-hop association
(the tail is always *reached* with zero noise leakage, but never ranked #1 — recall@3 = 1.0,
recall@1 = 0.0) and shows **accuracy decay at scale** (gold-hit 1.0 -> 0.80 from N=100 to
N=10000). Consolidation *gain* is **not measured** here by design — it is judge-gated and a
deterministic network-free run reports the judge-less pass as recall-neutral rather than
fabricating a lift. No aggregate number is offered; the per-axis numbers above are the report.
