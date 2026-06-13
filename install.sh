#!/usr/bin/env bash
# install.sh — one-command bootstrap for SAM/IA.
#
# From a clone:   bash install.sh
# Public (later): curl -fsSL https://raw.githubusercontent.com/asthenopheredevelopment-commits/samia/main/install.sh | bash
#
# Collapses every manual step the cold-metal rounds exposed: OS prerequisites
# (venv/pip/compiler/headers), a venv, the pip install (with the local-LLM
# arms), the memory dir, and — optionally — the maintenance daemon as a systemd
# user service. Idempotent and re-runnable.
#
# Tunables (env):
#   SAMIA_VENV=~/samia_venv              where the venv goes
#   ASTHENOS_MEMORY_DIR=~/.local/share/samia/memory   the memory store
#   SAMIA_WITH_LLM=1                     install the [llm] extra (local model arms)
#   SAMIA_SERVICE=0                      1 = install+enable the maintenanced service
#   SAMIA_WITH_CLAUDE=0                  1 = also install Claude Code (needed to USE SAM/IA)
#   SAMIA_UPGRADE=0                      1 = reinstall/upgrade samia even if already present
#   SAMIA_CUDA=0                         1 = rebuild llama-cpp-python with CUDA (GPU inference;
#                                            needs the CUDA toolkit / nvcc — CPU works without)
#   SAMIA_ENABLE_ARMS=0                  1 = turn on the semantic arm + fact extraction in the
#                                            systemd service (default off; needs SAMIA_SERVICE=1)
#   SAMIA_SRC=<pip spec>                 override source (default: this clone, else GitHub)
#
# IDEMPOTENT: every step checks first and does only what's missing. Re-running on
# a box that already has the prereqs / venv / samia / Claude Code is a fast no-op,
# not a rebuild. A fresh install happens only where something is actually absent.
set -uo pipefail

VENV="${SAMIA_VENV:-$HOME/samia_venv}"
MEMDIR="${ASTHENOS_MEMORY_DIR:-$HOME/.local/share/samia/memory}"
WITH_LLM="${SAMIA_WITH_LLM:-1}"
WITH_SERVICE="${SAMIA_SERVICE:-0}"
WITH_CLAUDE="${SAMIA_WITH_CLAUDE:-0}"
WITH_CUDA="${SAMIA_CUDA:-0}"
ENABLE_ARMS="${SAMIA_ENABLE_ARMS:-0}"
SRC="${SAMIA_SRC:-}"
REPO="https://github.com/asthenopheredevelopment-commits/samia.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"

say()   { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn()  { printf '   WARN: %s\n' "$*" >&2; }
# Network steps retry — flaky/CGNAT links (e.g. T-Mobile home) need it.
retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 3 ] && return 1; echo "   retry $n/3 ..."; sleep 5; done; }

# ---- 1. OS prerequisites (the only step that needs sudo) ---------------------
say "Checking OS prerequisites"
need=()
python3 -c 'import ensurepip' 2>/dev/null || need+=(python3-venv python3-pip)
command -v gcc  >/dev/null 2>&1 || need+=(build-essential)   # llama-cpp builds from source
command -v curl >/dev/null 2>&1 || need+=(curl)
command -v git  >/dev/null 2>&1 || need+=(git)
python3 -c 'import sysconfig,os; raise SystemExit(0 if os.path.exists(sysconfig.get_path("include")+"/Python.h") else 1)' 2>/dev/null || need+=(python3-dev)
if [ "${#need[@]}" -gt 0 ]; then
  echo "   installing: ${need[*]}"
  retry sudo apt-get update || warn "apt-get update failed"
  retry sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${need[@]}" \
    || { warn "could not install: ${need[*]} — install them and re-run"; exit 1; }
else
  echo "   all present"
fi

# ---- 2. virtualenv (create only if absent; reuse otherwise) -----------------
if [ -x "$VENV/bin/python" ]; then
  say "Reusing existing virtualenv at $VENV"
else
  say "Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi
retry "$VENV/bin/pip" -q install --upgrade pip || warn "pip self-upgrade failed (continuing)"

# ---- 3. install / update SAM/IA (skip if already present) -------------------
if [ -z "$SRC" ]; then
  if [ -n "$HERE" ] && [ -f "$HERE/pyproject.toml" ]; then SRC="$HERE"; else SRC="git+$REPO"; fi
fi
EXTRA=""; [ "$WITH_LLM" = "1" ] && EXTRA="[llm]"
CUR="$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("samia"))' 2>/dev/null || true)"
if [ -n "$CUR" ] && [ "${SAMIA_UPGRADE:-0}" != "1" ]; then
  say "samia $CUR already installed in this venv — skipping (set SAMIA_UPGRADE=1 to reinstall/upgrade)"
else
  [ -n "$CUR" ] && say "Upgrading samia ($CUR -> ${SRC}${EXTRA})" || say "Installing samia from ${SRC}${EXTRA}"
  UP=""; [ "${SAMIA_UPGRADE:-0}" = "1" ] && UP="--upgrade"
  retry "$VENV/bin/pip" install $UP "${SRC}${EXTRA}" \
    || { warn "pip install failed (network? for the private repo use a clone or 'gh auth')"; exit 1; }
fi
"$VENV/bin/python" -c 'import samia, samia.core.semantic_recall; print("   samia ready:", samia.__file__)'

# ---- 3b. optional: GPU build of llama-cpp-python ----------------------------
# Default install is CPU-only (portable). With a matching CUDA toolkit present,
# rebuild llama-cpp with GPU offload — much faster local-LLM (judge/extract/synth).
if [ "$WITH_CUDA" = "1" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    say "Rebuilding llama-cpp-python with CUDA (GPU inference)"
    # -DCMAKE_CUDA_ARCHITECTURES=native targets THIS box's GPU — required for
    # newer cards (e.g. Blackwell/sm_120 on an RTX 50-series); without it the
    # build can omit your arch and silently run on CPU. Verified on RTX 5070.
    retry env CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
        "$VENV/bin/pip" install --force-reinstall --no-cache-dir llama-cpp-python \
      || warn "CUDA build of llama-cpp-python failed (stays on the CPU build)"
  else
    warn "SAMIA_CUDA=1 but nvcc not found — install the CUDA toolkit, then re-run"
  fi
fi

# ---- 4. memory dir ----------------------------------------------------------
mkdir -p "$MEMDIR"

# ---- 5. optional: maintenance daemon as a systemd user service --------------
if [ "$WITH_SERVICE" = "1" ]; then
  say "Installing the maintenanced systemd user service"
  mkdir -p "$HOME/.config/systemd/user"
  {
    echo "[Unit]"
    echo "Description=SAM/IA memory maintenance daemon"
    echo "[Service]"
    echo "ExecStart=$VENV/bin/python -m samia.runtime.maintenanced"
    echo "Restart=on-failure"
    echo "Environment=ASTHENOS_MEMORY_DIR=$MEMDIR"
    if [ "$ENABLE_ARMS" = "1" ]; then       # opt-in: dual-process recall + atoms
      echo "Environment=ASTHENOS_SEMANTIC_ARM_ENABLED=1"
      echo "Environment=ASTHENOS_FACT_EXTRACT_ENABLED=1"
    fi
    echo "[Install]"
    echo "WantedBy=default.target"
  } > "$HOME/.config/systemd/user/samia-maintenanced.service"
  loginctl enable-linger "$USER" 2>/dev/null || warn "enable-linger failed (service runs only while logged in)"
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable --now samia-maintenanced 2>/dev/null || warn "could not start service (no user systemd?)"
fi

# ---- 6. optional: Claude Code (needed to USE SAM/IA in the current version) --
if [ "$WITH_CLAUDE" = "1" ]; then
  if command -v claude >/dev/null 2>&1; then
    say "Claude Code already present — skipping"
  else
    say "Installing Claude Code"
    # download-then-run (vet before execute) rather than piping a remote URL to a shell
    if retry curl -fsSL https://claude.ai/install.sh -o /tmp/claude_install.sh; then
      bash /tmp/claude_install.sh || warn "Claude Code installer failed — run it by hand later"
    else
      warn "could not download the Claude Code installer"
    fi
  fi
fi

say "Done"
cat <<DONE
  Activate the venv:   source $VENV/bin/activate
  Memory directory:    export ASTHENOS_MEMORY_DIR=$MEMDIR
  Allow model fetch:   export ASTHENOS_MODEL_AUTOFETCH=1   # gated; license printed first
  Quickstart + daemon usage: see README.md
DONE
