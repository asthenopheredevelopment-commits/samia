# SAM/IA Benchmark Compendium

_Compiled 2026-06-13. Unifies every memory-system benchmark run on SAM/IA and its competitors into one organized, explained record. Every number here is recomputed from the raw result files cited in-line; nothing is estimated unless labelled. Two benchmark families are covered: the **competitive QA benchmark** (LoCoMo, SAM/IA vs mem0 vs baselines) and the **temporal-recall calibration** series (intrinsic retrieval quality)._

**Raw-data roots (provenance):**
- Competitive: `~/Asthenosphere/benchmarks/memory_competitive_2026_06_08/` (raw `data/`, per-config `results/*.jsonl`, `results/results.csv` 900 rows, `manifest.json` with SHA256 + row counts, full `REPORT.md`).
- Temporal calibration: `~/Asthenosphere/benchmarks/temporal_recall_calibration_{,v2_,v3_balanced_,v3b_gamma_revalidation_}2026_06_11/` (`results.json` + `README.md` each).

---

## 1. Headline findings (read this first)

1. **At an equal extractor, SAM/IA beats mem0 by ~10 points** — the single most important result. The earlier "mem0 0.48 vs SAM/IA 0.46" was an *unfair* comparison: mem0 used a frontier (glm-5.1) extractor while SAM/IA used a local 4B. Equalize the extractor and SAM/IA leads at *both* tiers (table in §1 below, derivation in §4).
2. **The gap between the two systems is extraction quality, not architecture.** SAM/IA's score scales cleanly with extractor strength (0.47 → 0.59); so does mem0's (0.37 → 0.48). The architectures are roughly parallel — but SAM/IA sits ~10 points higher at every tier.
3. **SAM/IA degrades gracefully; mem0 collapses.** On temporal questions, mem0 is 100% extraction-dependent — its accuracy falls **0.55 → 0.10** when the extractor weakens. SAM/IA holds **0.70 → 0.50** because it retains the raw evidence turns underneath the extracted facts. This is SAM/IA's structural advantage: a floor mem0 doesn't have.
4. **SAM/IA pays ~0 ingest cost and keeps full provenance; mem0 pays ~272 LLM calls per corpus and discards source.** (§7.)
5. **The temporal-recall envelope is real on intrinsic recall:** the multiplicative need/STC/distinctiveness envelope triples recall@1 (0.20 → 0.60) and doubles MRR (0.37 → 0.74) on the calibration corpus (§8).

### The equalized 2×2 (LoCoMo, n=100, llama-3.3-70b reader+judge)

| Overall accuracy | 4B extractor (Qwen3-4B-Instruct local) | Frontier extractor (glm-5.1 cloud) |
|---|---|---|
| **SAM/IA** | **0.47** | **0.59** |
| **mem0** | **0.37** | **0.48** |
| SAM/IA lead | **+0.10** | **+0.11** |

Sources: `results/locomo_samia_ext_4b_baseline_percat20.jsonl` (0.47), `results/locomo_samia_frontier_percat20.jsonl` (0.59), `results/locomo_mem0_local_percat20.jsonl` (0.37), `results/locomo_llama70b_all_percat20.jsonl` → mem0 (0.48). All recomputed 2026-06-13.

![Equalized 2x2: SAM/IA vs mem0 by extractor tier](charts/equalized_2x2.png)

_Figure 1. The equalized 2×2. At an equal extractor SAM/IA leads at both tiers (+0.10 at 4B, +0.11 at frontier); SAM/IA's local 4B (0.47) ties mem0's frontier cloud (0.48). Rendered by `charts/make_charts.py`._

**Statistical honesty:** at n=100, the 95% binomial CI near 0.4–0.5 is ~±0.10. The equal-extractor +10 lead is at the edge of significance — a McNemar/paired test on the discordant pairs gives **p≈0.087** (trending, *not yet* p<0.05). The standalone fact that **SAM/IA's local-4B (0.47) ties mem0's frontier-cloud (0.48)** is itself strong (p=1.0, statistically indistinguishable) — SAM/IA reaches with a small local model what mem0 needs a frontier cloud model to reach.

---

## 2. Competitive benchmark — setup and controls

**Dataset.** LoCoMo (`snap-research/locomo`, 10 multi-session conversations, 1,986 QA), 5 categories: 1=multi-hop (282), 2=temporal (321), 3=open-domain (96), 4=single-hop (841), 5=adversarial/abstention (446). Stratified subsample of **20 per category = 100 questions/system**, seed 42, deterministic, logged in `manifest.json` (no silent caps).

**Systems under test** (only the memory layer varies — identical reader, judge, dataset, embedder, sample):

| System | What it is |
|---|---|
| `no_memory` | reader with zero context — the floor |
| `raw_vector_rag` | all-MiniLM-L6-v2 cosine top-12 over raw turns; no SAM/IA machinery |
| `samia_chainogram` | production `chainogram_retrieve` over per-conv stores; **cold-start** (no Hebbian history, no Tier-1 consolidation) |
| `full_context_oracle` | all turns concatenated (~27k tok) — reader ceiling where it fits |
| `mem0_oss` | mem0 v2.0.1, extraction-based memory, same embedder, same extraction gateway |

**Embedder parity.** All systems use `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU). Only retrieval orchestration differs.

**Two judge epochs (NOT cross-comparable).** Epoch 1 = `glm-5.1` (frontier, reader+judge) via opencode-go; its monthly quota wall on 2026-06-10 forced the boundary. Epoch 2 = `meta/llama-3.3-70b-instruct` (nvidia/groq, reader+judge, temp 0). The reader swap changed **45% of all answers** (only 220/400 paired rows identical), so cross-epoch *scores* are never compared — only within-epoch rankings and an explicit judge-vs-reader decomposition. The judge change alone is negligible: among the 220 identical-answer rows the two judges disagreed on **1 verdict (~0.5%)**.

---

## 3. Competitive results — both epochs

### Epoch 1 — reader+judge glm-5.1 (n=100/system)

| System | Overall | Multihop | Temporal | OpenDom | Singlehop | Adversarial | evid_hit | ctx tok |
|---|---|---|---|---|---|---|---|---|
| no_memory | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | — | 75 |
| raw_vector_rag | 0.33 | 0.20 | 0.40 | 0.30 | 0.70 | 0.05 | 0.54 | 546 |
| **samia_chainogram** | **0.37** | 0.25 | 0.40 | 0.45 | 0.65 | 0.10 | **0.66** | 4,907 |
| full_context_oracle | 0.56 | 0.50 | 0.65 | 0.60 | 0.90 | 0.15 | — | 27,114 |

Under a *strong* reader, SAM/IA beats RAG by 4 points and has the best retrieval recall measured (**evid_hit 0.66 vs 0.54**, +12).

### Epoch 2 — reader+judge llama-3.3-70b (n=100/system)

| System | Overall | Multihop | Temporal | OpenDom | Singlehop | Adversarial | evid_hit | ctx tok |
|---|---|---|---|---|---|---|---|---|
| no_memory | 0.02 | 0.00 | 0.00 | 0.05 | 0.00 | 0.05 | — | 136 |
| samia_chainogram (cold) | 0.37 | 0.45 | 0.20 | 0.35 | 0.70 | 0.15 | 0.66 | 4,965 |
| raw_vector_rag | 0.39 | 0.50 | 0.30 | 0.35 | 0.75 | 0.05 | 0.54 | 605 |
| full_context_oracle | 0.45 | 0.45 | 0.20 | 0.35 | 0.95 | 0.30 | — | 27,154 |
| **mem0_oss (frontier extract)** | **0.48** | 0.55 | 0.55 | 0.35 | 0.80 | 0.15 | n/a | 705 |

**Reader-capability interaction.** Under the weaker epoch-2 reader, context-heavy systems are punished: the oracle drops −11 (0.56→0.45) and SAM/IA-cold holds flat at 0.37, while mem0's pre-digested 705-token facts let it lead. The diagnosis: SAM/IA-cold *retrieves* the right turn (evid_hit 0.66) but defers synthesis to read-time; a weak reader can't extract the answer from raw prose. **This drove the fact-extract arc (§4).** Note `samia_chainogram` here is the *cold-start floor* — no atoms, no warm store.

---

## 4. The fact-extract arc and the equalization (the core story)

The epoch-2 finding — mem0 beats SAM/IA-cold because it pre-extracts facts at write time — was treated as an architecture signal, not a defeat. Three successive builds closed and then reversed the gap:

| Run | Overall | Temporal | ctx tok | Note |
|---|---|---|---|---|
| samia cold (epoch 2) | 0.37 | 0.20 | 4,965 | floor, no atoms |
| samia + atoms (P8) | 0.42 | 0.40 | 3,723 | write-time 4B extraction; temporal **doubled** |
| samia P2-final | 0.46 | 0.50 | 1,836 | +composer +entity-bridge; **beats oracle 0.45** on 7% of its context |
| mem0_oss (frontier extract) | 0.48 | 0.55 | 705 | reference |

P2-final left SAM/IA 2 points behind mem0 — but with one uncontrolled variable: **mem0's facts were extracted by frontier glm-5.1; SAM/IA's by a local 4B.** The equalization experiment (`runs/EQUALIZATION_STATE.md`) fixed that by re-ingesting mem0 with the *same* local Qwen3-4B (server on :8123, n_ctx 20480 to fit mem0's ~10.8k-token extraction prompt — documented deviation so the comparison stays about extraction *quality*, not truncation), and by running SAM/IA with frontier extraction for the top tier. Result = the §1 2×2:

- **Same 4B extractor:** SAM/IA **0.47** vs mem0 **0.37** (+10).
- **Same frontier extractor:** SAM/IA **0.59** vs mem0 **0.48** (+11).
- **Temporal degradation:** mem0 **0.55 → 0.10** (−45, total collapse); SAM/IA **0.70 → 0.50** (−20, graceful). mem0's temporal accuracy is *entirely* a function of extractor quality because it stores only derived facts; SAM/IA keeps the raw dated turns as a floor.

**Interpretation:** the two systems are architecturally parallel (both scale with the extractor), but SAM/IA is uniformly ~10 points higher and has a raw-evidence floor mem0 structurally lacks. The "mem0 wins" headline was an extractor-asymmetry artifact.

![Per-category accuracy at the frontier extractor](charts/category_frontier.png)

_Figure 2. Per-category accuracy with both systems at the frontier extractor (n=20/category). SAM/IA-frontier leads on multihop (0.75 vs 0.55), temporal (0.70 vs 0.55), open-domain (0.45 vs 0.35) and single-hop (0.90 vs 0.80); ties on adversarial (0.15). Per-category n=20 differences carry ±0.21 CI — read directionally, not as significant (see Appendix B)._

![Temporal accuracy by extractor: collapse vs hold](charts/temporal_degradation.png)

_Figure 3. The structural advantage. mem0's temporal accuracy collapses 0.55 → 0.10 (−45) when the extractor weakens, because it stores only derived facts; SAM/IA holds 0.70 → 0.50 (−20) on its raw-evidence floor._

### Residual-gap autopsy (why P2-final was still 2 behind at mixed extraction)
Per-question classification of mem0's wins over samia+atoms (0.48 vs 0.42, before equalization): **12 of mem0's 15 wins were already in SAM/IA's stores; 9 already in-context** — the reader drowned in a 3,723-token mixed context vs mem0's 705 tokens of pure facts. Only 3/15 were true extraction gaps (negations the 4B doesn't emit) and 3/15 retrieval-ranking. **~80% of the residual gap is one design decision:** chain-granularity loading with length-biased sum-scoring (built for a turn-only corpus). Budget-matching is a *negative result* — forcing SAMIA_BUDGET=1500 collapsed accuracy to 0.24 (chains load whole-or-nothing). Fix formalized as preplanner `memory_chainogram_atom_ranking`: mean-not-sum scoring, atom-chain preference, member caps.

---

## 5. Slot studies — which model slot actually moves the score

### Embedder sweep (baseline MiniLM-L6 = 0.47; `runs/SLOT_STUDY_STATE.md`)
| Embedder | dim | Overall | vs baseline |
|---|---|---|---|
| all-MiniLM-L6 (baseline) | 384 | 0.47 | — |
| bge-small | 384 | 0.48 | +1 |
| mpnet-base | 768 | 0.47 | 0 |
| bge-large | 1024 | 0.49 | +2 (best) |
| qwen3-4b-gguf mean-pool | 2560 | **0.10** | **−37 (non-viable)** |

**Verdict:** the embedder slot is ~flat on LoCoMo (≤+2); upgrades just shift the category mix. The qwen3-4b-as-embedder failure is instructive: instruct-decoder mean-pool space is **anisotropic** (real-store pairwise cosine mean 0.923/std 0.055 vs MiniLM 0.397/0.188) — no ranking signal, near-random top-k. A purpose-trained sentence encoder is required; a Qwen-family option would be the contrastive-trained Qwen3-Embedding-0.6B (named, not built).

![Embedder-slot sweep](charts/embedder_sweep.png)

_Figure 4. Embedder-slot sweep. Sentence-encoder swaps move overall accuracy ≤+0.02; the instruct-decoder mean-pool (qwen3-4b-gguf) collapses to 0.10 — non-viable, not a tuning gap._

### Judge-slot probe (`runs/SLOT_STUDY_STATE.md`, Arm B; 10 supersession + 10 compatible pairs)
| Judge backend | TPR | FPR |
|---|---|---|
| BitNet-2B (production default) | **0.0** | 0.0 |
| Qwen3-4B | **0.9** | 0.0 |

**Verdict:** a 4B judge slot is *transformative* — BitNet-2B emits no parseable JSON (judge silently inert; it also can't even load via the production `llama_cpp` path → falls to MockBackend), while Qwen3-4B turns the contradiction/supersession gate into a real high-precision filter (TPR 0→0.9 at FPR 0). The judge slot is where a bigger model pays off; the embedder slot is not.

### Extractor-size scaling (Round 2, GPU, `runs/SLOT_STUDY_STATE.md`)
| Extractor | Overall | atoms | Note |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 (baseline) | 0.47 | 3,311 | reproduces fixed_ln 0.47 exactly |
| Qwen3-8B (thinking-mode) | 0.45 | 1,711 | bigger, slower, *fewer* atoms, lower score |
| Mistral-7B-Instruct-v0.3 | — | 0 | extraction-contract failure (no parseable JSON) |
| frontier (glm-5.1) | **0.59** | — | the ceiling |

**Verdict:** extraction is **recipe/tuning-bound, not size-bound**. The 4B-Instruct-2507's extraction recipe beats a larger but differently-tuned 8B sibling. The only thing that buys the +12 jump to 0.59 is a frontier extractor — exactly the equalization lever.

---

## 6. Capability probes (deterministic, no LLM, production functions only)

| Probe | Result | What it proves |
|---|---|---|
| Temporal validity (`probe_temporal.json`) | **24/24** correct | `vector.query` ∩ `temporal.query` returns the time-valid fact version; raw RAG returns both versions (validity-blind) |
| Decay + recall repair (`probe_repair.json`) | **12/12 byte-exact** | nodes eroded below integrity 0.5, then anchor-repaired to integrity 1.0, body byte-identical to original |
| Supersession detection (`probe_supersession.json`) | **TPR 0.80 / FPR 0.10** | 8/10 supersession pairs caught, 1/10 false-flag (after retune) |

The supersession probe earned its keep: its *initial* TPR of 0.20 flushed **two silent production bugs** invisible in the head-to-head scoring — per-call HF-hub model reload (empty embeddings) and Jaccard-over-raw-frontmatter dilution. Operator retune (threshold 0.57, BitNet stage-2 on) lifted it to 0.80. The 2 remaining misses are MiniLM embedder-ceiling cases (cosine 0.45–0.49). A separate judge-probe showed a 4B LLM judge would lift supersession TPR to 0.9 (§5).

---

## 7. Cost and auditability ledger

| Dimension | SAM/IA | mem0_oss |
|---|---|---|
| Ingest LLM calls | **0** (local embedder only) | **272** recorded (+53 empty-retries) |
| Ingest wall time | seconds/conv | minutes/conv (2–571s/session) |
| Gateway dependency | none (local HF model) | opencode-go gateway; quota-bound |
| Stored artifact | raw turns (markdown nodes) + extracted atoms | LLM-extracted fact strings only |
| Retrieval provenance | dia_id → source turn (evid_hit measurable) | none (facts are derived) |
| Deletion / correction | keep source, deactivate node (restore path exists) | no restore (superseded facts deleted) |

SAM/IA buys its accuracy with ~zero ingest LLM cost *and* keeps the evidence chain mem0 discards. This is an axis the accuracy tables don't show: even where scores tie, SAM/IA is auditable and reversible.

---

## 8. Temporal-recall calibration series (intrinsic retrieval quality)

A separate family of runs measuring the time-aware recall envelope directly (vector recall on a temporal corpus, **no LLM judge** — pure recall@k / MRR). These calibrate the formula `score(c) = (S + 0.05·H + γ·TĈ)·(1 + λN·N̂ + λK·K̂ + λD·D̂)`.

| Run | Corpus | Key result |
|---|---|---|
| v1 (`..._2026_06_11`) | 100-node synthetic | baseline fwd-asym 0.94, recall@1 1.0 — best-θ all-zero (envelope is a no-op on a trivially-separable corpus; documents that the flags don't *hurt*) |
| v2 (`..._v2_...`) | 42-node, 6 topics | substrate build (SITH snapshots 42, Hebbian web formed 10) — wiring validation |
| **v3_balanced** (`..._v3_balanced_...`) | n=50 queries | **the headline:** best-θ {γ=0, λN=0.6, λK=2.0, λD=0.6} → recall@1 **0.60**, recall@5 0.90, recall@10 1.0, MRR **0.739**, vs **flag-off** recall@1 0.20, MRR 0.368 |
| v3b (`..._v3b_gamma_revalidation_...`) | reuses v3_balanced | re-validates γ/SITH under the live TC cosine-floor gate (`ASTHENOS_TEMPORAL_TC_COSINE_FLOOR`); γ is neutral-when-fueled (does not regress once the cosine floor blocks off-topic-but-recent additive TC) |

**Reading it:** on a corpus where retrieval is actually hard (v3_balanced), the multiplicative envelope **triples recall@1 (0.20 → 0.60)** and **doubles MRR (0.37 → 0.74)**. The dominant terms are need (λN=0.6) and STC (λK=2.0); γ (temporal-context drift) is held neutral by the cosine-floor safety gate. This is the intrinsic-recall evidence behind the temporal-recall formula that the competitive LoCoMo runs exercise end-to-end.

![Temporal recall@k envelope](charts/temporal_recall_at_k.png)

_Figure 5. Intrinsic temporal recall@k on v3_balanced (no LLM judge). The best envelope triples recall@1 (0.20 → 0.60); both curves converge to 1.0 by k=10, so the gain is in ranking the right node first._

---

## 9. What was NOT run (honest omissions)

- **LongMemEval:** harness built (`adapters/bench_lme.py`), oracle variant (500 q) started and **stopped at 20/120** on the first system — token conservation after the LoCoMo conclusion was reached; the oracle variant has near-trivial retrieval (it's mostly a reader test). **No LME results file exists.** mem0 was deliberately *not* run on LME (~500 ingest calls / ~3h for a non-discriminating variant). Logged as a skip, not a silent omission.
- **Formal judge cross-check:** a deepseek-v4-pro judge-agreement pass (`adapters/judge_agreement.py`) was planned but hit the glm-5.1 quota wall before completion; the epoch-1 judge error rate (~1.3% over a sampled cross-check) is estimated, not formally bounded.

---

## 10. Honest caveats (apply to every number above)

1. **Cross-epoch scores are not comparable.** Reader+judge both changed between epoch 1 and 2; 45% of answers differ. Only within-epoch rankings and the explicit reader-vs-judge decomposition are valid.
2. **Sample size.** n=100 → 95% CI ~±0.10 near 0.4; per-category n=20 → ±0.21. Category differences below 0.10, and overall differences below ~0.10, are weak.
3. **The equalization +10 lead is trending, not yet significant** (p≈0.087, n=100). The "SAM/IA-4B ties mem0-frontier" claim is the more robust standalone statement.
4. **Cold-start floor.** The 0.37 epoch-2 SAM/IA number is a *floor* (isolated per-conv stores, no Hebbian history, no Tier-1 consolidation). The atomized/P2/equalized runs are the representative numbers.
5. **Within-harness only.** These absolute scores are NOT comparable to published LoCoMo/LME leaderboards (different judges, samples, prompts).

---

## 11. One-line summary

**At an equal extractor, SAM/IA beats mem0 by ~10 points on LoCoMo (0.47 vs 0.37 at 4B; 0.59 vs 0.48 at frontier), degrades gracefully where mem0 collapses (temporal 0.70→0.50 vs 0.55→0.10), retains full evidence provenance mem0 discards, pays ~0 ingest cost, and its time-aware recall envelope triples recall@1 on the calibration corpus — with the remaining system-to-system delta attributable to extraction quality, a roadmap lever, not architecture.**

_All figures recomputed from the cited `results/*.jsonl` and `*/results.json` on 2026-06-13. Raw data, manifests, and adapters preserved at the two workspace roots in the header._

_Charts in this document are rendered by `charts/make_charts.py` (matplotlib, Agg backend) from the verified tables above — the script hard-codes the numbers, it does not recompute them. To regenerate: `python3 charts/make_charts.py` (writes the 5 PNGs; if matplotlib is unavailable it falls back to `charts/chart_data.csv` and the PNGs require `pip install matplotlib` + a re-run)._

---

## Appendix A — mem0 exact configuration (reproducibility)

This is the precise mem0 configuration used for the **0.37 equalized run** (`results/locomo_mem0_local_percat20.jsonl`). The whole point of the equalization was to run mem0's extractor on the *same* local Qwen3-4B that SAM/IA uses for atom extraction, so the head-to-head is architecture-only. Source: `adapters/mem0_local_adapter.py` + `runs/EQUALIZATION_STATE.md`.

**System:** mem0 v2.0.1 (open-source). Embedder, vector store and the retry shim are identical to the frontier-extracted mem0 run; only the extraction/update LLM changed.

Copy-pasteable `Memory.from_config(...)` block (verbatim from the adapter):

```python
cfg = {
    "llm": {"provider": "openai", "config": {
        "model": "qwen3-4b",
        "api_key": "local",                       # server ignores it
        "openai_base_url": "http://localhost:8123/v1",
        "max_tokens": 2000,                        # lowered from 4000 (see below)
        "temperature": 0.0}},
    "embedder": {"provider": "huggingface", "config": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dims": 384}},                   # SAME embedder as SAM/IA + raw_vector_rag
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": "locomo", "embedding_model_dims": 384,
        "path": str(store_dir / "qdrant")}},       # local, per-conversation
    "history_db_path": str(store_dir / "history.db"),
}
# custom_instructions: NONE — mem0's DEFAULT fact-extraction prompt is used.
# output_format: mem0 default JSON fact format.
```

**The honest answers to the three config questions reviewers ask:**

- **`temperature`: 0.0.** Deterministic extraction (matching the benchmark-wide temp-0 reader/judge policy in `manifest.json`).
- **`custom_instructions`: NONE.** mem0's *default* fact-extraction prompt was used unchanged — there were no custom instructions. This is the honest answer; the run deliberately did not hand-tune mem0's prompt.
- **`output_format`: mem0 default JSON fact format.** A shim (`_ShimmedLLM` / monkeypatched `OpenAILLM.generate_response`) strips `response_format={"type":"json_object"}` on the gateway — under that flag the 4B's reasoning tokens otherwise consume the whole budget and return empty completions — then retries: (1) without `response_format`, (2) with 3× `max_tokens`. Empty-after-retry is **counted, not silently dropped** (empty 4B extraction is exactly the data the experiment is after).

**The `max_tokens` lowering (4000 → 2000).** mem0's extraction prompt is ~10k tokens (max observed **10,772** across all 272 sessions). With the server at `n_ctx=20480`, `max_tokens=2000` never trips the context wall, and the shim's 3× empty-retry bump (→6000; 10,772 + 6,000 = 16,772 < 20,480) also fits. Extraction outputs (fact lists) are short, so 2000 is ample.

**Extraction server.** `llama.cpp` (`llama-cpp-python`'s OpenAI-compatible server) serving the Qwen3-4B-Instruct gguf at `127.0.0.1:8123`:

- Model: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` — the **same** Qwen3-4B SAM/IA uses for atom extraction.
- `--n_ctx 20480` — **REQUIRED.** `n_ctx 8192` fails extraction on *every* session (`context_length_exceeded` → 0 memories, a pure artifact). 20480 holds the max 10,772-token prompt plus the shim's headroom. This is a documented deviation so the comparison stays about extraction **quality**, not context truncation.
- `--n_gpu_layers 0` — CPU-only (mirrors SAM/IA's CPU gguf extraction and avoids contending with the ~5 GB already resident on the GPU).
- `--n_threads 12`, `--model_alias qwen3-4b`, launched `nice -n 10`.

**Vector store / embedder parity.** Qdrant, local, per-conversation. Embedder is `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — the **same** embedder as SAM/IA and `raw_vector_rag` (embedder parity is a benchmark-wide control; see §2). `SEARCH_K=12` matches the frontier mem0 adapter exactly.

---

## Appendix B — Statistical detail

All competitive figures are n=100 (20 questions × 5 categories), LoCoMo, epoch-2 llama-3.3-70b reader+judge. Computed first-hand from `results/*.jsonl`.

### Wilson 95% confidence intervals (overall accuracy)

| System / tier | Accuracy | Wilson 95% CI |
|---|---|---|
| SAM/IA — 4B extractor | 0.47 | [0.375, 0.567] |
| SAM/IA — frontier extractor | 0.59 | [0.492, 0.681] |
| mem0 — 4B extractor | 0.37 | [0.282, 0.468] |
| mem0 — frontier extractor | 0.48 | [0.385, 0.577] |

### McNemar exact paired tests (discordant pairs)

The paired comparisons below condition on the same 100 questions through both systems; only the discordant pairs (one system right, the other wrong) carry signal. "samia-only" = questions SAM/IA got right and the comparator got wrong; "comparator-only" = the reverse.

| Comparison | samia-only correct | comparator-only correct | exact p | risk difference | verdict |
|---|---|---|---|---|---|
| SAM/IA-4B vs mem0-4B | 19 | 9 | **0.087** | +0.10 | **trending** (not yet <0.05) |
| SAM/IA-4B vs mem0-frontier | 11 | 12 | **1.000** | ≈0.00 | **statistically indistinguishable** |
| SAM/IA-frontier vs mem0-frontier | 15 | 4 | **0.019** | +0.11 | **significant** (<0.05) |

### Reading the three tests

- **Frontier tier is significant.** SAM/IA-frontier vs mem0-frontier: 15-vs-4 discordant split, exact p=0.019, risk difference +0.11. The frontier-extractor advantage is established at this n.
- **4B tier is trending, not yet significant.** SAM/IA-4B vs mem0-4B: 19-vs-9 discordant split, exact p=0.087, risk difference +0.10. The direction is clear and the effect size matches the frontier tier, but it does not cross p<0.05 at n=100.
- **The "local 4B ties frontier cloud" result is robust.** SAM/IA-4B vs mem0-frontier: 11-vs-12, p=1.000 — statistically indistinguishable. SAM/IA reaches with a small local model what mem0 needs a frontier cloud model to reach. This is the strongest standalone paired statement in the set.

### Honest power note

At n=100 the frontier-tier advantage reaches significance (p=0.019); the 4B-tier +0.10 is **underpowered** at this n. The 19-vs-9 discordant split would need roughly **n ≥ 200** to confirm at p<0.05 (the effect is real-sized but the sample is half what it needs to be for that tier). Per-category figures are n=20 (±0.21 CI) and should not be over-read — treat the per-category bars (Figure 2) as directional, not significant.

---

## Appendix C — How to reproduce

Exact commands, pulled verbatim from `runs/EQUALIZATION_STATE.md`, `runs/CAMPAIGN_STATE.md`, and the launcher scripts. All paths are under `~/Asthenosphere/benchmarks/memory_competitive_2026_06_08/` (`$WS` below). Nothing here touches the live SAM/IA tree, settings, or daemon.

### C.1 — mem0 local extraction server (the equalized run)

```bash
WS=~/Asthenosphere/benchmarks/memory_competitive_2026_06_08
# 1. Provision the fresh venv (operator runs once; symlinks the compiled
#    llama_cpp CUDA build + pip-installs the pure-Python [server] deps):
bash $WS/runs/setup_mem0_local_venv.sh
# 2. Launch the CPU-only Qwen3-4B OpenAI server on :8123, n_ctx 20480:
bash $WS/runs/start_llama_server.sh         # PID -> runs/llama_server.pid
# 3. Sequential full ingest of all 10 LoCoMo convs -> runs/mem0_local_stores
#    (idempotent per-conv .ingested markers; resumable):
bash $WS/runs/run_local_ingest.sh           # progress -> runs/mem0_local_ingest.out
# 4. Bench the equalized mem0 (epoch-2 judge llama-3.3-70b is the default):
cd $WS/adapters && /usr/bin/python3 -u bench.py mem0_local 20
#    -> results/locomo_mem0_local_percat20.jsonl   (the 0.37 run)
# 5. Kill the server when done:
kill $(cat $WS/runs/llama_server.pid)
```

### C.2 — SAM/IA recall bench (env-var configured)

SAM/IA's temporal-recall envelope is configured by environment variables (formula in §8; mapping from `runs/CAMPAIGN_STATE.md`). General form: `python3 bench.py samia_recall_composed 20 <tag>`.

```bash
cd $WS/adapters
# B1 — λN regression guard (the canonical samia_fixed_ln 0.47 baseline):
ASTHENOS_TEMPORAL_WEIGHT=1 ASTHENOS_TEMPORAL_LAMBDA_N=0.6 \
  python3 bench.py samia_recall_composed 20 samia_fixed_ln
# B2 — retest negative-γ vs the TC-cosine-floor fix:
ASTHENOS_TEMPORAL_WEIGHT=1 ASTHENOS_TEMPORAL_GAMMA=0.6 \
  ASTHENOS_TEMPORAL_TC_COSINE_FLOOR=0.5 \
  python3 bench.py samia_recall_composed 20 samia_fixed_gamma
# B3 — fixed-γ default + λN:
ASTHENOS_TEMPORAL_WEIGHT=1 ASTHENOS_TEMPORAL_LAMBDA_N=0.6 \
  ASTHENOS_TEMPORAL_GAMMA=0.6 ASTHENOS_TEMPORAL_TC_COSINE_FLOOR=0.5 \
  python3 bench.py samia_recall_composed 20 samia_fixed_full
```

(The master flag `ASTHENOS_TEMPORAL_WEIGHT` defaults off; all envelope weights default to 0.0. The formula is `score(c) = (S_c + 0.05·H_c + γ·TĈ_c)·(1 + λN·N̂ + λK·K̂ + λD·D̂)`.)

### C.3 — regenerate the consolidated artifacts + charts

```bash
# Rebuild the one queryable table (results/results.csv) + manifest.json
# (SHA256 + per-file row counts) from results/*.jsonl + probes + datasets:
cd $WS/adapters && python3 make_artifacts.py
# Re-render the compendium figures (Agg backend; PNGs -> ../charts/):
python3 ~/Asthenosphere/benchmarks/charts/make_charts.py
```

### C.4 — integrity anchor

`manifest.json` is the integrity anchor: it records the **SHA256** and **row count** of every scored `results/*.jsonl` file plus the two source datasets (`data/locomo10.json`, `data/longmemeval_oracle.json`), the sample config (`per_cat=20`, `seed=42`, 100 sampled questions), both judge epochs with their endpoints, and the embedder. To verify a result file has not drifted, recompute its SHA256 and compare against the `files` block in `manifest.json`. `results/results.csv` (consolidated, with an `epoch` column keeping the two eras separable) is the single queryable form; both are regenerated by `make_artifacts.py` (C.3).
