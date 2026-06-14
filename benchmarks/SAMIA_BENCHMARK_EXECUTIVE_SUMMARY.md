# SAM/IA Benchmark — Executive Summary

_Compiled 2026-06-13. A 1–2 page read of the SAM/IA competitive memory benchmark. Every number here is recomputed first-hand from the raw `results/*.jsonl`; nothing is estimated unless labelled. Full record, all caveats, and reproduction commands: **[SAMIA_BENCHMARK_COMPENDIUM_2026-06-13.md](SAMIA_BENCHMARK_COMPENDIUM_2026-06-13.md)**._

---

## The headline: equalize the extractor, SAM/IA leads at every tier

The earlier "mem0 0.48 vs SAM/IA 0.46" was an **unfair comparison** — mem0 used a frontier (glm-5.1) extractor while SAM/IA used a local 4B. Equalize the extraction model (the same local Qwen3-4B SAM/IA already uses) and SAM/IA leads at *both* tiers:

![Equalized 2x2: SAM/IA vs mem0 by extractor tier](charts/equalized_2x2.png)

| Overall accuracy (LoCoMo, n=100) | 4B extractor (local) | Frontier extractor (cloud) |
|---|---|---|
| **SAM/IA** | **0.47** | **0.59** |
| **mem0** | **0.37** | **0.48** |
| SAM/IA lead | **+0.10** | **+0.11** |

Two systems, architecturally parallel (both scale cleanly with extractor strength), but SAM/IA sits ~10 points higher at every tier. And **SAM/IA's local 4B (0.47) ties mem0's frontier cloud (0.48)** — SAM/IA reaches with a small local model what mem0 needs a frontier cloud model to reach.

## The structural advantage: SAM/IA degrades gracefully, mem0 collapses

On temporal questions, mem0 is 100% extraction-dependent — it stores only derived facts, so its accuracy **collapses 0.55 → 0.10** (−45) when the extractor weakens. SAM/IA **holds 0.70 → 0.50** (−20) because it retains the raw dated evidence turns underneath the extracted facts. This is a floor mem0 structurally lacks.

![Temporal accuracy by extractor: collapse vs hold](charts/temporal_degradation.png)

## The cost and provenance edge

| Dimension | SAM/IA | mem0 |
|---|---|---|
| Ingest LLM calls | **0** (local embedder only) | **272** recorded (+53 empty-retries) |
| Retrieval provenance | dia_id → source turn (auditable) | none (facts are derived) |
| Deletion / correction | keep source, deactivate node (restorable) | no restore (superseded facts deleted) |

SAM/IA buys its accuracy at ~zero ingest LLM cost *and* keeps the evidence chain mem0 discards — an axis the accuracy tables don't show: even where scores tie, SAM/IA is auditable and reversible.

## Statistical significance (honest)

- **Frontier tier is significant.** SAM/IA-frontier vs mem0-frontier: 15-vs-4 discordant split, **McNemar exact p = 0.019** (<0.05), risk difference +0.11.
- **4B tier is trending.** SAM/IA-4B vs mem0-4B: 19-vs-9 split, **p = 0.087** (not yet <0.05), risk difference +0.10 — same effect size as the frontier tier, but underpowered at n=100.
- **"Local 4B ties frontier cloud" is robust.** SAM/IA-4B vs mem0-frontier: 11-vs-12, **p = 1.000** — statistically indistinguishable.

(Wilson 95% CIs and full McNemar detail in Appendix B of the compendium.)

## One-line verdict

**At an equal extractor, SAM/IA beats mem0 by ~10 points on LoCoMo (0.47 vs 0.37 at 4B; 0.59 vs 0.48 at frontier), degrades gracefully where mem0 collapses (temporal 0.70→0.50 vs 0.55→0.10), retains the full evidence provenance mem0 discards, and pays ~0 ingest cost — with the remaining delta attributable to extraction quality, a roadmap lever, not architecture.**

## What's not yet proven

The 4B-tier +10 lead is **underpowered at n=100** (p=0.087; ~n≥200 needed to confirm at p<0.05); the frontier-tier advantage *is* significant (p=0.019). **LongMemEval was not run** to completion (harness built, oracle variant stopped at 20/120 as near-trivial; mem0 deliberately not run on LME). See the compendium's §9–§10 for the full omissions and caveats.

---

_Full record, derivations, slot studies, probes, reproduction commands and integrity anchor (manifest.json SHA256s): **[SAMIA_BENCHMARK_COMPENDIUM_2026-06-13.md](SAMIA_BENCHMARK_COMPENDIUM_2026-06-13.md)**. All figures recomputed from the cited `results/*.jsonl` on 2026-06-13._
