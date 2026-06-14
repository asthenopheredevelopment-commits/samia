# SAM/IA [Shallow Abyss Manifest/Infinite Abyss Compression]

Biologically-inspired persistent memory system for long-running AI agents and
personal knowledge work. Stores, retrieves, decays, and repairs memories across
sessions using a tiered architecture modeled on biological consolidation.

---

## Status

Single-operator production use. The system runs daily on one workstation and
has not been tested in multi-user or production-scale deployments.


---

## What It Is

SAM/IA implements a persistent memory layer with the following capabilities,
all present in the current codebase:

- **Tiered decay store** — five tiers (hot → archived); grade-specific decay
  rates; frozen nodes archive to disk and thaw on demand.
- **Chainogram episodic retrieval** — directed chain graphs with bi-temporal
  edge intervals; assembles token-budget-constrained context ranked by
  Hebbian co-activation weights.
- **Semantic atom arm + composer** — a second retrieval arm serving extracted
  `type:semantic` facts via vector lookup; a composer joins both arms into
  one labelled context (`ASTHENOS_SEMANTIC_ARM_ENABLED`).
- **Fact extraction at consolidation** — local LLM runs over session offloads
  at consolidation time, writing semantic atoms additively (default off;
  `ASTHENOS_FACT_EXTRACT_ENABLED`).
- **Contradiction / supersession detection** — embedding-similarity detection
  (cosine >= 0.75, configurable) with optional LLM judge; auto-supersede
  is restorable byte-exact via `restore_node`. Measured TPR 0.80 / FPR 0.10.
- **REM-cycle offline consolidation** — WAKE/REM state machine; maintenance
  jobs run only during REM and yield immediately to active use; entry is
  event-driven, not timer-based.
- **Content-integrity decay + anchor repair** — orthogonal decay axis erodes
  content slowly; byte-exact anchor repair from write-time snapshot; no
  irrecoverable loss in P1/P2.
- **Hebbian associative web** — SQLite co-activation edge graph (`edges.db`)
  that chainogram retrieval scores against.
- **Tier-1 hippocampal fast store** — engram store (days-to-months) + ring
  store (capacity-bounded, hours); kWTA sparse codes; promotion by access
  frequency or salience.

Full component internals and the module map: see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Running It

Every operation is callable in-process as a library and exposed as MCP
primitives via `core/mcp_server.py`. For autonomous upkeep — decay ticking,
idle replay, and REM-cycle consolidation — run the maintenance daemon:
`python -m samia.runtime.maintenanced`. Module map, component internals, and
the systemd unit: see [ARCHITECTURE.md](ARCHITECTURE.md).

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
- Optional: `inotify_simple` or `pyinotify` for low-latency filesystem
  watching (falls back to polling at 2s interval if absent)

No cloud credentials, no accounts, no telemetry. Everything runs locally.

**Network actions** — exactly three, all one-time, none automatic beyond
what you invoke:

1. `pip install` — package dependencies from PyPI.
2. First `vector.build` — the MiniLM embedder (~90MB, Apache-2.0) from
   HuggingFace.
3. Optional model fetch (`model_fetch.fetch_model` or a missing-model LLM
   call) — the model's license is printed before download.

Nothing else touches the network — no phone-home, no update checks.

**Every engine download is operator-gated** by `ASTHENOS_MODEL_AUTOFETCH`:

- **unset (the default)**: no silent download ever. At an interactive
  terminal the engine prints what/size/license/source and asks
  `Download? [y/N]`; with no terminal it refuses and names both remedies.
- **`1` (standing consent)**: downloads proceed without prompting — how
  agent/CI flows operate after clearing their own permission gate.
- **`0` (kill switch)**: every download refused in every mode, with a
  copy-pasteable manual-download instruction.

The MiniLM embedder loads local-only from cache first, so a box that has run
before is always silent and fast — consent is consulted only on a genuine
cache miss.

---

## Install

A guided installer. Run it and it **auto-detects your box** (GPU, systemd,
whether Claude Code is present), proposes a plan, and asks before it touches
anything. **Idempotent**: re-running on a box that's already set up (or half
set up) is a fast no-op, not a rebuild.

```sh
bash install.sh              # interactive: detect, show the plan, confirm, install
bash install.sh --dry-run    # print the resolved plan and change nothing
bash install.sh --yes        # non-interactive: accept all detected defaults
```

Piping it (`curl -fsSL https://raw.githubusercontent.com/asthenopheredevelopment-commits/samia/main/install.sh | bash`)
runs non-interactively against the detected defaults — no tty, no prompts.

**Override any choice with an environment variable.** A var that is set is
honored silently with no prompt, so scripted/CI runs are unaffected:

- `SAMIA_WITH_CLAUDE=1` — also install Claude Code (needed to *use* SAM/IA in the
  current version)
- `SAMIA_SERVICE=1` — install + enable the maintenance daemon as a systemd user
  service (`maintenanced`); `SAMIA_ENABLE_ARMS=1` also turns on the semantic arm
  + fact extraction in that service
- `SAMIA_CUDA=1` — rebuild llama-cpp-python with CUDA for GPU inference (see below)
- `SAMIA_UPGRADE=1` — reinstall/upgrade an existing install
- `SAMIA_VENV=…` / `ASTHENOS_MEMORY_DIR=…` — relocate the venv / memory store

What it does — and only where absent: apt-installs `python3-venv`, `python3-pip`,
`python3-dev`, **`build-essential`** (the `[llm]` arm compiles `llama-cpp-python`
from source, so a compiler is required), `curl`, `git`; creates the venv;
`pip install`s `samia[llm]`; makes the memory directory.

### GPU acceleration (optional, experimental)

SAM/IA **never requires a GPU** — the default build is CPU-only and runs
everywhere. With an NVIDIA GPU and a matching CUDA toolkit you can rebuild the
local-LLM backend for GPU offload (the REM-time judge / fact-extraction /
synthesis run far faster):

```sh
SAMIA_CUDA=1 bash install.sh        # rebuilds llama-cpp-python with CUDA
```

This builds llama-cpp-python against **your own** CUDA toolkit — install it from
your distro or NVIDIA to match your driver (SAM/IA does **not** bundle it; no
project ships the multi-GB proprietary toolkit). `install.sh` checks for `nvcc`
and skips the GPU build with a note if it's absent. Prefer not to compile?
llama-cpp-python publishes prebuilt CUDA wheels for common CUDA versions (no
toolkit needed) — though very new GPUs (Blackwell) may need the source build
until those wheels catch up.

GPU is **additive**: a CUDA build uses the GPU when one is present and falls
back to CPU automatically when it isn't (no GPU, VRAM exhaustion, driver
mismatch). Tune with `ASTHENOS_N_GPU_LAYERS` (`0` forces CPU; a positive N does
partial offload for limited VRAM). Verified on an RTX 5070 (CUDA 12.8 — Qwen3-4B
at ~150 tok/s vs ~6 tok/s on CPU, ~25×); other GPUs are expected to work but are
not yet tested.

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
This is deliberately not an OSI "open source" license.

## Platform support

- **Linux** — developed and tested here; first-class.
- **macOS** — expected compatible (pure-Python + POSIX constructs that macOS
  provides; llama-cpp-python has Metal wheels). Untested — reports welcome.
- **Windows** — not natively supported yet: several modules use `fcntl`
  file-locking, which Windows lacks. **WSL2 works today.** Native support is
  a contained, contribution-friendly TODO (a locking shim + temp-path sweep).
