# SAM/IA Capability Benchmark — Design v1
**Goal:** independently-verifiable, repeatable measurement of *what SAM/IA can do*. v1 reports SAM/IA's own
capabilities only — **no cross-system comparison yet** (the harness is built adapter-ready so other memory
systems can be plugged in later without changing tasks/metrics). This document + its harness **replace** the
prior `SAMIA_BENCHMARK_COMPENDIUM_*` / executive-summary / charts.

## Non-negotiables (what makes a number trustworthy)
1. **Runs against the INSTALLED package** (`pip install samia` in a clean venv — the test-install build), never the dev tree.
2. **Deterministic:** pinned deps, fixed RNG seeds, fixed embedder + judge model (by digest). Same inputs → same score.
3. **Versioned, checksummed datasets** committed under `benchmarks/data/` (SHA256 manifest). No network at score time.
4. **Programmatic scoring first** (exact-match / set-F1 / numeric tolerance). An LLM judge is used ONLY for open-ended
   recall, with a **pinned local model + fixed prompt**, and every judge transcript is saved (auditable, re-scoreable).
   — This deliberately avoids the reader/judge confound that contaminates LoCoMo/LongMemEval (per our defect audit).
5. **Everything published:** dataset, harness, seeds, raw per-item outputs, scores → a third party re-runs and matches.
6. **Adapter API** (`MemoryAdapter.store(items)` / `.recall(query, k)` / `.consolidate()` / `.reset()`): SAM/IA is the
   first adapter; the same tasks/metrics run against any future adapter unchanged (comparison-ready, comparison-deferred).

## Capability axes → task → metric → SAM/IA surface exercised
| # | Axis | Task (deterministic) | Metric | SAM/IA module |
|---|------|----------------------|--------|----------------|
| A1 | **Retrieval accuracy** | seed N facts; query for each; is the gold memory in top-k | recall@{1,5,10}, MRR | `core/semantic_recall`, `core/vector*` |
| A2 | **Retention / forgetting** | seed salient + noise facts; interleave many turns; re-query salient *after delay* | retention@delay vs noise-drop rate | `core/tier`, decay, `runtime/maintenanced` |
| A3 | **Temporal reasoning** | "what did I say about X *most recently / before Y*" | temporal-recall@k, ordering acc | `core/temporal*`, `temporal_recall_stc/sith` |
| A4 | **Contradiction / belief-update** | assert X; later assert ¬X with more evidence; query | demote-correct %, shadow-persist % | `core/gates`, `core/judge`, supersession |
| A5 | **Consolidation gain** | score recall@k **before vs after** a REM/merge cycle | Δrecall (consolidation lift) | `core/consolidation`, `tier2_merge`, `maintenanced` |
| A6 | **Associative / multi-hop** | seed linked chain a→b→c; query a, expect c | multi-hop recall@k | `core/chain` (Hebbian), `successor` |
| A7 | **Distillation fidelity** | store verbose source; recall; does the atom preserve the source claim | claim-preservation F1 (programmatic) | `core/fact_extractor`, atomization |
| A8 | **Provenance / firewall** | inject untrusted/poisoned items; verify they're quarantined, not recalled as truth | poison-rejection %, false-trust % | `core/gates`, `core/auditor`, `netconsent` |
| A9 | **Latency / scale** | store→recall at store sizes 10²/10³/10⁴; wall-clock | p50/p95 recall ms vs N, ingest items/s | end-to-end |

## Harness architecture
```
benchmarks/
  BENCHMARK_DESIGN_v1.md        ← this file (methodology)
  data/                         ← versioned task datasets + SHA256SUMS
  adapters/base.py              ← MemoryAdapter ABC (store/recall/consolidate/reset)
  adapters/samia_adapter.py     ← SAM/IA impl (imports the INSTALLED package)
  tasks/a1_retrieval.py ... a9_latency.py   ← one runnable module per axis (deterministic, seeded)
  run_benchmark.py              ← CLI: runs axes → writes results/raw_<axis>.jsonl + scores.json
  score.py                      ← programmatic scorers (+ pinned-judge wrapper, transcripts saved)
  results/                      ← raw outputs + scores (committed so others can diff)
  REPORT.md                     ← generated capability report (replaces the old compendium)
  repro/Dockerfile + seeds.toml ← pinned env + seeds for one-command reproduction
```
Run: `python benchmarks/run_benchmark.py --adapter samia --seed 1337 --sizes 100,1000,10000` → deterministic `scores.json` + `REPORT.md`.

## v1 build order
1. `adapters/base.py` + `adapters/samia_adapter.py` (against the installed package) → smoke: store/recall round-trips.
2. Dataset generators (seeded, checksummed) for A1–A8; A9 reuses A1 data at 3 sizes.
3. Task modules A1→A9, programmatic scorers, `run_benchmark.py`.
4. `REPORT.md` generator + `repro/` (Docker + seeds) → one-command repeat.
5. (later) additional adapters = the comparison phase — tasks/metrics unchanged.

## Locked scope (operator, v1)
- **All 9 axes** A1–A9.
- **Clean FIXED dataset that fixes known benchmark defects** (committed + checksummed), specifically:
  - **D6 (retrieval≠retention, universal):** A1 (retrieval) and A2 (retention-after-delay) are *separate* tasks with *separate* data — never conflated.
  - **D5 (reader/judge confound):** programmatic scoring is primary; the pinned judge runs only on A3/A4/A7 open-ended items, with saved transcripts so any reader effect is auditable + re-scoreable.
  - **D1/D2/D4 (dataset-specific):** hand-fixed gold labels, no ambiguous/duplicate/temporally-incoherent items; each item carries an explicit gold + rationale.
- **Programmatic scoring + one pinned reproducible judge** (local model by digest, fixed prompt) for the open-ended subset only.

## Honesty rails
- Report **raw numbers + the exact task definition** for every axis; no aggregate "score" that hides which axis is weak.
- Where SAM/IA has no capability for an axis, report it as N/A with the reason — don't fabricate.
- Distinguish **retrieval** (A1) from **retention** (A2) explicitly — conflating them is the field's most common defect.
