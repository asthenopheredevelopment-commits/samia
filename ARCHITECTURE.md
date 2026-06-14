# SAM/IA — Architecture and Internals

Contributor/internals companion to the top-level README. Covers the
component model with source references, the full module map, and how to
run the maintenance daemon (including the systemd unit).

---

## Components

SAM/IA implements a persistent memory layer with the following components,
all of which are present in the current codebase:

**Tiered store with relevance decay.** Nodes live in five tiers: hot, warm,
cold, frozen, and archived. Relevance decays at grade-specific rates
(`core/tier.py`). A node's grade (enriched, fertile, natural, depleted,
waste) modulates its decay rate. High-salience nodes decay more slowly.
Frozen nodes are archived to disk and can be thawed on demand.

**Chainogram episodic retrieval.** Memory is organized as chains —
directed graphs of nodes with bi-temporal edge intervals.
`chainogram_retrieve` in `core/context_extension.py` assembles a
token-budget-constrained context from episodic chains, using Hebbian
co-activation edge weights to rank chains by relevance to a query.

**Semantic atom arm with dual-process composer.** A second retrieval arm
(`core/semantic_recall.py`) serves `type:semantic` atoms — short, extracted
facts — via direct vector lookup. A composer joins the episodic chainogram
output with the semantic arm output into one labelled context without either
arm being aware of the other's population. The composer is flag-gated
(`ASTHENOS_SEMANTIC_ARM_ENABLED`).

**Fact extraction at consolidation.** The fact-extract producer
(`core/fact_extractor.py`) runs a local LLM over session offloads at
consolidation time, writing `type:semantic` atoms to the memory tree.
Controlled by `ASTHENOS_FACT_EXTRACT_ENABLED` (default off). Atoms are
additive; no source node is deleted.

**Contradiction and supersession detection.** `runtime/contradiction.py`
detects potential contradictions via embedding similarity (cosine >= 0.75
threshold, configurable) with an optional LLM judge gate. `core/ia.py`
implements restorable auto-supersede: a superseded node is archived
byte-exact and can be restored via `restore_node`. The auto-supersession
probe measured TPR 0.80 / FPR 0.10 at the current threshold and embedding
model ceiling.

**REM-cycle offline consolidation.** `runtime/rem_cycle.py` implements a
WAKE/REM state machine. Offline maintenance jobs (vector index rebuild,
consolidation candidate detection, decay passes) run only during REM phase
and yield immediately to active use. Entry is event-driven: triggered by
idle pressure, not a bare timer.

**Content-integrity decay and repair with anchors.** A second, orthogonal
decay axis (`core/integrity.py`) tracks the fraction of each node's content
that remains intact. Nodes erode slowly and character-by-character. A
pristine anchor stored at write time enables byte-exact recall-repair.
Erosion never proceeds on a node without a recoverable anchor (no
irrecoverable data loss in P1/P2). Generative reconstruction (for
anchor-absent nodes) is P3 and is off by default.

**Hebbian associative web.** `core/web_store.py` maintains a SQLite
co-activation edge graph (`edges.db`) that records weighted cross-chain
associations. The Hebbian web is the substrate the chainogram retrieval
scores against.

**Tier-1 hippocampal fast store.** `core/hippocampus.py` implements an
engram store (held copies of recently active nodes, days-to-months) and a
ring store (capacity-bounded volatile pointer set, hours). kWTA sparse codes
provide pattern separation. Promotion from ring to engram is triggered by
access frequency or salience.

---

## Module Map

```
runtime/inference.py       — in-process LLM backend (LlamaCppBackend / MockBackend)
runtime/model_fetch.py     — model registry + license-notified auto-download
runtime/contradiction.py   — embedding-similarity contradiction detection
runtime/memory_guard.py    — pre-write validation pipeline
runtime/rem_cycle.py       — WAKE/REM state machine
runtime/rem_subscribers.py — REM-phase maintenance jobs (vector, atoms, erosion)
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
core/context_extension.py  — chainogram_retrieve and budget primitives
core/semantic_recall.py    — semantic atom retrieval arm + composer
core/integrity.py          — content-integrity decay and anchor-based repair
core/hippocampus.py        — Tier-1 engram + ring store
core/ia.py                 — compress, freeze, thaw, merge, forget, restore
core/web_store.py          — Hebbian co-activation edge graph (edges.db)
core/vector.py             — MiniLM-L6-v2 vector index (build + query)
core/fact_extractor.py     — write-time fact extraction producer
core/consolidation.py      — merge candidate detection (Jaccard content-word)
core/mcp_server.py         — MCP tool primitive layer (stdio MCP wrapper separate)
```

Every function is callable in-process as a **library** (see Quickstart in
README), and `core/mcp_server.py` exposes the operations as **MCP
primitives** for agent harnesses.

---

## Maintenance Daemon

For autonomous upkeep — decay ticking, idle replay, REM-cycle
consolidation, index freshness — run the **maintenance daemon**:

```sh
python -m samia.runtime.maintenanced --memory-dir ~/samia_memory          # foreground
python -m samia.runtime.maintenanced --oneshot                            # single pass, exits
```

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
