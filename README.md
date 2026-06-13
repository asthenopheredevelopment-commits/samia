# SAM/IA [Shallow Abyss Manifest/Infinite Abyss Compression]

Biologically-inspired persistent memory system for long-running AI agents and
personal knowledge work. Stores, retrieves, decays, and repairs memories across
sessions using a tiered architecture modeled on biological consolidation.

---

## Status

Single-operator production use. The system runs daily on one workstation and
has not been tested in multi-user or production-scale deployments.

**Benchmarking:** no first-party benchmark numbers ship with this
release. The system was developed against LoCoMo-style long-term-memory
workloads; we invite independent benchmarking — the retrieval entry points
(`chainogram_retrieve`, `semantic_recall.recall`) are stable seams to
harness against.

---

## What It Is

SAM/IA implements a persistent memory layer with the following components, all
of which are present in the current codebase:

**Tiered store with relevance decay.** Nodes live in five tiers: hot, warm,
cold, frozen, and archived. Relevance decays at grade-specific rates
(`core/tier.py`). A node's grade (enriched, fertile, natural, depleted, waste)
modulates its decay rate. High-salience nodes decay more slowly. Frozen nodes
are archived to disk and can be thawed on demand.

**Chainogram episodic retrieval.** Memory is organized as chains — directed
graphs of nodes with bi-temporal edge intervals. `chainogram_retrieve` in
`core/context_extension.py` assembles a token-budget-constrained context from
episodic chains, using Hebbian co-activation edge weights to rank chains by
relevance to a query.

**Semantic atom arm with dual-process composer.** A second retrieval arm
(`core/semantic_recall.py`) serves `type:semantic` atoms — short, extracted
facts — via direct vector lookup. A composer joins the episodic chainogram
output with the semantic arm output into one labelled context without
either arm being aware of the other's population. The composer is flag-gated
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
that remains intact. Nodes erode slowly and character-by-character. A pristine
anchor stored at write time enables byte-exact recall-repair. Erosion never
proceeds on a node without a recoverable anchor (no irrecoverable data loss
in P1/P2). Generative reconstruction (for anchor-absent nodes) is P3 and
is off by default.

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

## Architecture Overview

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

Every function is callable in-process as a **library** (see Quickstart), and
`core/mcp_server.py` exposes the operations as **MCP primitives** for agent
harnesses. For autonomous upkeep — decay ticking, idle replay, REM-cycle
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

---

## Requirements

**OS packages** (stock Ubuntu/Debian desktops ship without venv/pip):

```sh
sudo apt install -y python3-venv python3-pip curl
```

**Python**: 3.12+. Package dependencies (installed automatically by pip):

- `numpy`
- `sentence-transformers` with `all-MiniLM-L6-v2` (dim=384, CPU; ~90MB,
  downloads on first use from HuggingFace). Note: on CPU-only boxes install
  the CPU torch wheel FIRST to avoid ~1.5GB of unused CUDA wheels:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `[llm]` extra: `llama-cpp-python` for the local LLM arms (contradiction
  judge, fact extraction, synthesis). Without it everything non-LLM works;
  LLM-dependent features fall back to `MockBackend` and log a warning.
- `sqlite3` (Python stdlib)
- Optional: `inotify_simple` or `pyinotify` for low-latency filesystem watching
  (falls back to polling at 2s interval if absent)

No cloud credentials, no accounts, no telemetry. Everything runs locally.

**Complete inventory of network actions** (there are exactly three, all
one-time downloads, none automatic beyond what you invoke):

1. `pip install` — package dependencies from PyPI.
2. First `vector.build` — the MiniLM embedder (~90MB, Apache-2.0) from HuggingFace.
3. Optional model fetch (`model_fetch.fetch_model` or a missing-model LLM
   call) — the model's license is printed before download.

Nothing else touches the network — no phone-home, no update checks.

**Every engine download is operator-gated** by one knob, `ASTHENOS_MODEL_AUTOFETCH`:

- **unset (the default): no silent download ever.** At an interactive terminal
  the engine prints what/size/license/source and asks `Download? [y/N]`;
  with no terminal it refuses and names both remedies.
- **on-value (`1`)**: standing consent — downloads proceed without prompting
  (how agent/CI flows operate after clearing their own permission gate).
- **off-value (`0`)**: kill switch — every download refused in every mode,
  with a copy-pasteable manual-download instruction.

The MiniLM embedder loads local-only from cache first, so a box that has run
before is always silent and fast — consent is consulted only on a genuine
cache miss.

---

## Install

One command — installs the OS prerequisites, a venv, and SAM/IA, doing only
what's missing. **Idempotent**: re-running on a box that's already set up (or
half set up) is a fast no-op, not a rebuild.

```sh
bash install.sh                  # from a clone
```

Options (environment variables):

- `SAMIA_WITH_CLAUDE=1` — also install Claude Code (needed to *use* SAM/IA in the
  current version)
- `SAMIA_SERVICE=1` — install + enable the maintenance daemon as a systemd user
  service (`maintenanced`)
- `SAMIA_UPGRADE=1` — reinstall/upgrade an existing install
- `SAMIA_VENV=…` / `ASTHENOS_MEMORY_DIR=…` — relocate the venv / memory store

What it does — and only where absent: apt-installs `python3-venv`, `python3-pip`,
`python3-dev`, **`build-essential`** (the `[llm]` arm compiles `llama-cpp-python`
from source, so a compiler is required), `curl`, `git`; creates the venv;
`pip install`s `samia[llm]`; makes the memory directory.

---

## Quickstart (the manual steps install.sh automates)

```sh
sudo apt install -y python3-venv python3-pip python3-dev build-essential curl git
python3 -m venv ~/samia_venv && source ~/samia_venv/bin/activate
pip install '/path/to/samia[llm,test]'           # or plain: pip install /path/to/samia

export ASTHENOS_MEMORY_DIR=~/samia_memory        # default: ~/.local/share/samia/memory
export ASTHENOS_MODEL_AUTOFETCH=1                # standing consent for the model
                                                 # downloads below; unset = the
                                                 # engine asks at a tty and
                                                 # REFUSES in scripts (no silent
                                                 # downloads, ever)
```

Write a memory, index it, retrieve it via both arms (no LLM needed):

```python
import os
from pathlib import Path
from samia.core import frontmatter, vector, context_extension, semantic_recall

md = Path(os.environ["ASTHENOS_MEMORY_DIR"]).expanduser()

# an episodic node (served by the chainogram EVIDENCE arm)
fm = {"name": "first_fact", "description": "my first memory", "type": "reference"}
frontmatter.write_node(md / "nodes" / "first_fact.md", fm, list(fm.keys()),
                       "SAM/IA stored this on day one.\n")

# a semantic atom (served by the FACTS arm; at runtime the fact-extraction
# producer writes these for you — that path needs the local LLM)
fa = {"name": "sem_first_atom", "description": "my first semantic atom",
      "type": "semantic", "source": "s01", "valid_from": "2026-06-12",
      "tier": "warm"}
frontmatter.write_node(md / "nodes" / "sem_first_atom.md", fa, list(fa.keys()),
                       "The owner of this memory store likes dense little facts.\n")

vector.build(md)   # first run downloads MiniLM (Apache-2.0, ~90MB) — gated by
                   # ASTHENOS_MODEL_AUTOFETCH above; on a cold cache without it,
                   # non-interactive runs refuse rather than silently download

out = context_extension.chainogram_retrieve(md, "first memory", budget_tokens=2000)

os.environ["ASTHENOS_SEMANTIC_ARM_ENABLED"] = "1"
res = semantic_recall.recall(md, "what does the owner like?", budget_tokens=2000)
print(res["context"])   # KNOWN FACTS (atoms) + CONVERSATION EVIDENCE (episodic)
```

For the LLM arms (contradiction judge, fact extraction), either point
`ASTHENOS_FACT_EXTRACT_MODEL` / judge model envs at your own `.gguf`, or let
the built-in fetcher pull a default model (license notice printed first):

```python
from samia.runtime.model_fetch import fetch_model
fetch_model("Qwen3-4B-Instruct-2507-Q4_K_M")   # ~2.4GB, Apache-2.0
```

`core/mcp_server.py` exposes the same operations as MCP tool primitives for
agent harnesses; wire it to any stdio MCP wrapper.

---

## What Is Not Included

- **No personal corpus.** No memory nodes, chains, anchors, or history from
  the development environment are included. The package ships code only.
  Bring your own memory directory.
- **No bundled models.** The package ships no model weights. The LLM arms use
  a local `.gguf` you supply, or `runtime/model_fetch.py` can download a
  registry model on request (Qwen3-4B, Apache-2.0) — the model's license is
  printed before any download, and models are cached under
  `~/.local/share/asthenos/models/`. Set `ASTHENOS_MODEL_AUTOFETCH=0` to
  disable all downloading.
- **No cloud credentials.** API keys for optional cloud provider arms are not
  included and are never committed to the repository (verified: GATE1 scan,
  zero hard-coded secrets).
- **No Atoms UI.** The Topology Atlas 3D visualization and the Atoms control
  panel (Slint desktop app) are not part of this package.
- **The `runtime/` modules outside the memory core** — skills, preplanner,
  evolution engine, orchestrator, perception, voice I/O — are present in the
  source tree but are parts of a larger personal assistant system. Their
  configuration, dependencies, and integration points are not documented here.

---

## Known Limitations

- Single-operator tested. Concurrent multi-user writes to the same memory
  directory have not been tested and are likely unsafe.
- Cold-start performance (no prior Hebbian history, no Tier-1 consolidation)
  is the measured floor. Warm-store production performance is higher but has
  not been independently benchmarked.
- The supersession detector has a measured FPR of 0.10 at the current
  similarity threshold (0.57) and embedder (MiniLM-L6-v2). Two known miss
  cases involve semantically distant rephrasing at the embedding ceiling.
- Fact extraction quality scales with the extraction LLM. The default
  registry model is a 4B; a stronger local model improves the semantic-atom
  arm (especially write-time resolution of relative dates) at the cost of
  slower consolidation.
- The temporal-recall envelope terms (`ASTHENOS_TEMPORAL_*` flags) are
  shipped default-off; enable selectively and validate against your own
  workload.

## License

**PolyForm Noncommercial 1.0.0** (source-available). Free for personal,
research, educational, and any noncommercial use — experiment freely.
**Commercial use requires a paid license from the author** (see NOTICE).
This is deliberately not an OSI "open source" license: the trade is full
source transparency and free noncommercial use, with commercial users
supporting the project's development.

## Platform support

- **Linux** — developed and tested here; first-class.
- **macOS** — expected compatible (pure-Python + POSIX constructs that macOS
  provides; llama-cpp-python has Metal wheels). Untested — reports welcome.
- **Windows** — not natively supported yet: several modules use `fcntl`
  file-locking, which Windows lacks. **WSL2 works today.** Native support is
  a contained, contribution-friendly TODO (a locking shim + temp-path sweep).
