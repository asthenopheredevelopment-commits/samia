# SAM/IA — Architecture and Internals

Contributor/internals companion to the top-level README. Covers the
component model with source references, the full module map, and how to
run the maintenance daemon (including the systemd unit).

---

## Components

SAM/IA implements a persistent memory layer with the following components,
all of which are present in the current codebase:

**Tiered store with relevance decay.** Nodes live in four relevance tiers:
hot, warm, cold, and frozen (`core/tier.py` `TIER_THRESHOLDS`). Relevance
decays at grade-specific rates. A node's grade (enriched, fertile, natural,
depleted, waste) modulates its decay rate. High-salience nodes decay more
slowly. Frozen nodes are archived to disk (`ia.freeze`) and can be thawed
on demand (`ia.thaw`). Note that `archived` is a `target_state` lifecycle
value, not a relevance tier — `decay_pass` skips nodes whose `target_state`
is `frozen` or `archived`.

**Chainogram episodic retrieval.** Memory is organized as chains —
directed graphs of nodes with bi-temporal edge intervals.
`chainogram_retrieve` in `core/context_extension/` (the `retrieval`
submodule) assembles a token-budget-constrained context from episodic
chains, using Hebbian co-activation edge weights to rank chains by relevance
to a query.

**Semantic atom arm with dual-process composer.** A second retrieval arm
(`core/semantic_recall.py`) serves `type:semantic` atoms — short, extracted
facts — via direct vector lookup. A composer joins the episodic chainogram
output with the semantic arm output into one labelled context without either
arm being aware of the other's population. The composer is flag-gated
(`ASTHENOS_SEMANTIC_ARM_ENABLED`).

**Fact extraction at consolidation.** The fact-extract producer
(`core/fact_extractor.py`) runs an extraction backend — a local LLM by
default (Qwen3-4B registry name), with a deterministic rule-splitter
fallback (and an optional Anthropic route) — over freeze and merge-abstract
offloads at consolidation time, writing `type:semantic` atoms to the memory
tree. Controlled by `ASTHENOS_FACT_EXTRACT_ENABLED` (default off). Atoms are
additive; no source node is deleted.

**Contradiction and supersession detection.** `runtime/contradiction/`
detects potential contradictions via embedding similarity. The candidate
cosine bar defaults to 0.57 (`ASTHENOS_CONTRADICTION_THRESHOLD`), with a
higher 0.92 bar for pairs involving a machine-generated semantic atom
(`ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD`); both are configurable. The
recall-first cosine bar is paired with an LLM judge gate that is on by
default (`ASTHENOS_CONTRADICTION_JUDGE`), which recovers precision.
`core/ia.py` implements restorable auto-supersede: a superseded node is
archived byte-exact and can be restored via `restore_node`. On the judge
probe corpus (2026-06-12), the Qwen3-4B judge measured TPR 0.9 / FPR 0.0.

**REM-cycle offline consolidation.** `runtime/rem_cycle/` implements a
WAKE/REM state machine. Decay/tiering (`tier.decay_tick`) runs continuously
on the scheduler in BOTH wake and REM — it is the forgetting curve and is
deliberately not REM-gated. The heavier strengthening/abstraction ops do run
only inside REM: vector index maintenance, consolidation surfacing, tier-2
merge, passive supersession, integrity repair, and replay. REM jobs yield
immediately to active use, and entry is event-driven — triggered by idle
pressure, not a bare timer.

**Content-integrity decay and repair with anchors.** A second, orthogonal
decay axis (`core/integrity/`) tracks the fraction of each node's content
that remains intact. Nodes erode slowly and character-by-character. A
pristine anchor stored at write time enables byte-exact recall-repair.
Erosion never proceeds on a node without a recoverable anchor (no
irrecoverable data loss in P1/P2). Generative reconstruction (for
anchor-absent nodes) is P3 and is off by default.

**Hebbian associative web.** `core/web_store.py` maintains a SQLite
co-activation edge graph (`edges.db`) that records weighted cross-chain
associations. The Hebbian web is the substrate the chainogram retrieval
scores against.

**Tier-1 hippocampal fast store.** `core/hippocampus/` implements an
engram store (held copies of recently active nodes, days-to-months) and a
ring store (capacity-bounded volatile pointer set, hours). kWTA sparse codes
provide pattern separation. Promotion from ring to engram is triggered by
access frequency or salience.

---

## Module Map

```
runtime/inference.py       — in-process LLM backend (LlamaCppBackend / MockBackend)
runtime/model_fetch.py     — model registry + license-notified auto-download
runtime/contradiction/     — embedding-similarity contradiction detection (package)
runtime/memory_guard.py    — pre-write validation pipeline
runtime/rem_cycle/         — WAKE/REM state machine (package)
runtime/rem_subscribers/   — REM-phase maintenance jobs (vector, atoms, erosion) (package)
runtime/maintenanced.py    — minimal maintenance daemon (scheduler + watcher + REM)
runtime/scheduler.py       — periodic job runner (decay, replay, gc, sm2)
runtime/watcher.py         — filesystem watcher (inotify/pyinotify/polling)
runtime/ipc.py             — JSON-line IPC protocol (client + server primitives)

core/tier.py               — relevance decay, tier classification
core/temporal.py           — bi-temporal query (valid_from / valid_to)
core/temporal_substrate.py — write-time temporal stamps (written_at, episode_seq)
core/temporal_recall_sith.py / successor.py / temporal_recall_stc.py /
core/temporal_distinctiveness.py
                           — temporal-recall envelope terms (SITH temporal
                             context, successor-representation need, synaptic
                             tagging, distinctiveness); flag-gated, default off
core/chain.py              — chain manifest I/O, edge-level temporal intervals
core/context_extension/    — chainogram_retrieve and budget primitives (package)
core/semantic_recall.py    — semantic atom retrieval arm + composer
core/integrity/            — content-integrity decay and anchor-based repair (package)
core/hippocampus/          — Tier-1 engram + ring store (package)
core/ia.py                 — compress, freeze, thaw, merge, forget, restore
core/web_store.py          — Hebbian co-activation edge graph (edges.db)
core/vector.py             — vector index (build + query); MiniLM-L6-v2 default,
                             embedder selectable via ASTHENOS_EMBED_MODEL with a
                             cross-embedder query guard
core/fact_extractor.py     — write-time fact extraction producer
core/consolidation.py      — merge candidate detection (Jaccard content-word)
core/mcp_server/           — MCP tool primitive backend functions (package);
                             the shipped `samia-mcp-server` console script is the
                             runnable stdio server (mcp_server_main.py)
```

Each of these packages exposes a re-export facade `__init__.py`, so the
import surface (`from samia.core.context_extension import X`) is unchanged.
Every function is callable in-process as a **library** (see Quickstart in
README), and `core/mcp_server/` provides the per-tool backend functions that
the shipped `samia-mcp-server` stdio server (`mcp_server_main.py`) exposes as
**MCP primitives** for agent harnesses (`claude mcp add asthenos-memory --
samia-mcp-server`).

---

## Maintenance Daemon

For autonomous upkeep — decay ticking, idle replay, REM-cycle
consolidation, index freshness — run the **maintenance daemon**:

```sh
python -m samia.runtime.maintenanced --memory-dir ~/samia_memory          # foreground
python -m samia.runtime.maintenanced --oneshot                            # single pass, exits
```

The `samia daemon run [...]` CLI is the supported wrapper around this module
— it forwards the same `--memory-dir` / `--interval` / `--oneshot` flags to
`maintenanced` (and offers `--sandboxed`, a fail-closed stub). The companion
subcommands `samia init`, `samia status`, and `samia mcp-server` round out
the console-script front-end; the module invocation above is the underlying
mechanism.

Single-instance locked per memory dir; SIGTERM stops it cleanly. A systemd
user unit is the natural way to keep it running:

```ini
# ~/.config/systemd/user/samia-maintenanced.service
[Unit]
Description=SAM/IA memory maintenance daemon
[Service]
ExecStart=/home/YOU/samia_venv/bin/python -m samia.runtime.maintenanced
Restart=on-failure
Environment=ASTHENOS_MEMORY_DIR=%h/samia_memory
[Install]
WantedBy=default.target
```

(`systemctl --user enable --now samia-maintenanced`.) The parent system's
full daemon — IPC server, skills, perception, voice — is not part of this
release; `maintenanced` is the memory-lifecycle core of it.
