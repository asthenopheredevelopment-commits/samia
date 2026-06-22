#!/usr/bin/env bash
# install.sh — one-command bootstrap for SAM/IA.
#
# INTERACTIVE BY DEFAULT when stdin is a tty (prompts for each option).
# Non-interactive (no prompts) automatically when:
#   - piped: curl -fsSL .../install.sh | bash
#   - stdin is not a tty
#   - SAMIA_NONINTERACTIVE=1 is set
#   - --yes flag is passed
#
# From a clone:   bash install.sh   (recommended; works for a private repo too)
# Raw one-liner:  curl -fsSL https://raw.githubusercontent.com/asthenopheredevelopment-commits/samia/main/install.sh | bash
#                 (only resolves once the repo is PUBLIC — for a private repo,
#                  clone with credentials/'gh auth' and run from the clone)
#
# FLAGS:
#   -y, --yes     Accept all defaults, no prompts (same as SAMIA_NONINTERACTIVE=1)
#   --dry-run     Print the resolved plan + steps that WOULD run, then exit 0.
#                 No sudo, no writes, no installs.
#   -h, --help    Show this usage and exit 0.
#
# Collapses every manual step the cold-metal rounds exposed: OS prerequisites
# (venv/pip/compiler/headers), a venv, the pip install (with the local-LLM
# arms), the memory dir, and — optionally — the maintenance daemon as a systemd
# user service. Idempotent and re-runnable.
#
# TUNABLES (env vars always override prompts and detected defaults):
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
#   SAMIA_NONINTERACTIVE=1               suppress all prompts (same as --yes)
#
# RESOLUTION ORDER per option (highest wins):
#   1. Env var explicitly set in the environment (detected via ${VAR+set})
#   2. Interactive prompt (when stdin is a tty and not --yes / SAMIA_NONINTERACTIVE)
#   3. Auto-detected default (GPU presence, systemd usability, etc.)
#
# IDEMPOTENT: every step checks first and does only what's missing. Re-running on
# a box that already has the prereqs / venv / samia / Claude Code is a fast no-op,
# not a rebuild. A fresh install happens only where something is actually absent.
set -uo pipefail

# ---- Flag parsing ------------------------------------------------------------
# Must happen before any variable expansion so flags are seen before env defaults.
DRY_RUN=0
FORCE_YES=0
_usage() {
  cat <<USAGE
Usage: bash install.sh [OPTIONS]

  -y, --yes       Non-interactive: accept all detected defaults, no prompts.
  --dry-run       Print the resolved plan and the steps that WOULD run, then
                  exit 0. No writes, no sudo, no network calls.
  -h, --help      Show this message and exit 0.

Env var tunables (override prompts and detected defaults):
  SAMIA_VENV, ASTHENOS_MEMORY_DIR, SAMIA_WITH_LLM, SAMIA_SERVICE,
  SAMIA_WITH_CLAUDE, SAMIA_UPGRADE, SAMIA_CUDA, SAMIA_ENABLE_ARMS,
  SAMIA_SRC, SAMIA_NONINTERACTIVE

Examples:
  bash install.sh                          # interactive
  bash install.sh --yes                    # non-interactive, all defaults
  bash install.sh --dry-run                # see the plan without doing anything
  SAMIA_SERVICE=1 bash install.sh --yes    # non-interactive, enable service
  curl -fsSL .../install.sh | bash         # curl-pipe (auto non-interactive)
USAGE
}
for _arg in "$@"; do
  case "$_arg" in
    -y|--yes)      FORCE_YES=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    -h|--help)     _usage; exit 0 ;;
    *)             printf 'Unknown flag: %s\n\n' "$_arg" >&2; _usage >&2; exit 2 ;;
  esac
done

# ---- Interactivity gate ------------------------------------------------------
# Interactive = stdin is a real tty AND not forced non-interactive by flag or env.
# When piped (curl | bash) the tty check automatically makes this non-interactive,
# preserving CI/curl-pipe semantics with no special-casing required.
INTERACTIVE=0
if [ -t 0 ] && [ "$FORCE_YES" = "0" ] && [ "${SAMIA_NONINTERACTIVE:-}" = "" ]; then
  INTERACTIVE=1
fi

# ---- Helpers -----------------------------------------------------------------
say()   { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn()  { printf '   WARN: %s\n' "$*" >&2; }
# Network steps retry — flaky/CGNAT links (e.g. T-Mobile home) need it.
retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 3 ] && return 1; echo "   retry $n/3 ..."; sleep 5; done; }

# ask_yn "prompt text" "default (y|n)"
# In non-interactive mode: echoes the default and returns immediately.
# In interactive mode: prompts the user and accepts y/Y/yes/n/N/no or Enter.
ask_yn() {
  local prompt="$1" default="$2" answer
  if [ "$INTERACTIVE" = "0" ]; then
    echo "   $prompt [auto: $default]"
    [ "$default" = "y" ] && return 0 || return 1
  fi
  while true; do
    if [ "$default" = "y" ]; then
      printf '   %s [Y/n]: ' "$prompt"
    else
      printf '   %s [y/N]: ' "$prompt"
    fi
    read -r answer </dev/tty
    answer="${answer:-$default}"
    case "$answer" in
      y|Y|yes|YES) return 0 ;;
      n|N|no|NO)   return 1 ;;
      *) echo "   Please answer y or n." ;;
    esac
  done
}

# ask_path "prompt text" "default path"
# In non-interactive mode: echoes the default and returns it on stdout.
# In interactive mode: prompts; Enter accepts the default.
ask_path() {
  local prompt="$1" default="$2" answer
  if [ "$INTERACTIVE" = "0" ]; then
    echo "   $prompt [auto: $default]"
    printf '%s' "$default"
    return
  fi
  printf '   %s\n   [default: %s]: ' "$prompt" "$default" >/dev/tty
  read -r answer </dev/tty
  answer="${answer:-$default}"
  printf '%s' "$answer"
}

# ---- Static constants (no defaults yet) --------------------------------------
REPO="https://github.com/asthenopheredevelopment-commits/samia.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"

# ---- Detect "explicitly set in environment" BEFORE applying defaults ---------
# ${VAR+set} expands to the string "set" if VAR is defined (even if empty), and
# to "" if it is unset. This lets us distinguish a user who exported SAMIA_CUDA=0
# (explicit intent) from one who never set it at all (should use detected default).
_venv_from_env="${SAMIA_VENV+set}"
_memdir_from_env="${ASTHENOS_MEMORY_DIR+set}"
_llm_from_env="${SAMIA_WITH_LLM+set}"
_service_from_env="${SAMIA_SERVICE+set}"
_claude_from_env="${SAMIA_WITH_CLAUDE+set}"
_cuda_from_env="${SAMIA_CUDA+set}"
_arms_from_env="${SAMIA_ENABLE_ARMS+set}"
_upgrade_from_env="${SAMIA_UPGRADE+set}"
_src_from_env="${SAMIA_SRC+set}"

# ---- Auto-detect sensible defaults -------------------------------------------
# GPU/CUDA: ON only when nvcc is present AND nvidia-smi can see a GPU.
_cuda_detected=0
if command -v nvcc >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q 'GPU'; then
  _cuda_detected=1
fi

# Systemd user service: ON if systemctl --user show-environment succeeds.
_service_detected=0
if systemctl --user show-environment >/dev/null 2>&1; then
  _service_detected=1
fi

# Claude Code: ON (needs installing) if claude is absent; OFF if already present.
_claude_detected=0
if ! command -v claude >/dev/null 2>&1; then
  _claude_detected=1
fi

# WITH_LLM: ON by default (same as before); Arms: OFF by default.
_llm_detected=1
_arms_detected=0

# ---- Resolve each option (env > prompt > detected default) -------------------
# Path options: VENV and MEMDIR
if [ "$_venv_from_env" = "set" ]; then
  VENV="${SAMIA_VENV}"
  _venv_src="env"
else
  _venv_default="$HOME/samia_venv"
  if [ "$INTERACTIVE" = "1" ]; then
    VENV="$(ask_path "Virtualenv location" "$_venv_default")"
    _venv_src="answered"
  else
    VENV="$_venv_default"
    _venv_src="detected"
  fi
fi

if [ "$_memdir_from_env" = "set" ]; then
  MEMDIR="${ASTHENOS_MEMORY_DIR}"
  _memdir_src="env"
else
  _memdir_default="$HOME/.local/share/samia/memory"
  if [ "$INTERACTIVE" = "1" ]; then
    MEMDIR="$(ask_path "Memory directory" "$_memdir_default")"
    _memdir_src="answered"
  else
    MEMDIR="$_memdir_default"
    _memdir_src="detected"
  fi
fi

# Boolean options: WITH_LLM
if [ "$_llm_from_env" = "set" ]; then
  WITH_LLM="${SAMIA_WITH_LLM}"
  _llm_src="env"
else
  _llm_default="$( [ "$_llm_detected" = "1" ] && echo y || echo n )"
  if ask_yn "Install [llm] extras (local model arms)?" "$_llm_default"; then
    WITH_LLM=1; else WITH_LLM=0; fi
  _llm_src="$( [ "$INTERACTIVE" = "1" ] && echo answered || echo detected )"
fi

# WITH_CUDA
if [ "$_cuda_from_env" = "set" ]; then
  WITH_CUDA="${SAMIA_CUDA}"
  _cuda_src="env"
else
  _cuda_default="$( [ "$_cuda_detected" = "1" ] && echo y || echo n )"
  if ask_yn "Rebuild llama-cpp-python with CUDA (GPU inference)?" "$_cuda_default"; then
    WITH_CUDA=1; else WITH_CUDA=0; fi
  _cuda_src="$( [ "$INTERACTIVE" = "1" ] && echo answered || echo detected )"
fi

# WITH_SERVICE
if [ "$_service_from_env" = "set" ]; then
  WITH_SERVICE="${SAMIA_SERVICE}"
  _service_src="env"
else
  _service_default="$( [ "$_service_detected" = "1" ] && echo y || echo n )"
  if ask_yn "Install the maintenanced systemd user service?" "$_service_default"; then
    WITH_SERVICE=1; else WITH_SERVICE=0; fi
  _service_src="$( [ "$INTERACTIVE" = "1" ] && echo answered || echo detected )"
fi

# ENABLE_ARMS (only meaningful when WITH_SERVICE=1)
if [ "$_arms_from_env" = "set" ]; then
  ENABLE_ARMS="${SAMIA_ENABLE_ARMS}"
  _arms_src="env"
else
  _arms_default="$( [ "$_arms_detected" = "1" ] && echo y || echo n )"
  if ask_yn "Enable semantic arm + fact extraction in the service?" "$_arms_default"; then
    ENABLE_ARMS=1; else ENABLE_ARMS=0; fi
  _arms_src="$( [ "$INTERACTIVE" = "1" ] && echo answered || echo detected )"
fi

# WITH_CLAUDE
if [ "$_claude_from_env" = "set" ]; then
  WITH_CLAUDE="${SAMIA_WITH_CLAUDE}"
  _claude_src="env"
else
  _claude_default="$( [ "$_claude_detected" = "1" ] && echo y || echo n )"
  if ask_yn "Install Claude Code (required to use SAM/IA)?" "$_claude_default"; then
    WITH_CLAUDE=1; else WITH_CLAUDE=0; fi
  _claude_src="$( [ "$INTERACTIVE" = "1" ] && echo answered || echo detected )"
fi

# WITH_UPGRADE and SRC: env-only tunables (no auto-detect / no prompt)
WITH_UPGRADE="${SAMIA_UPGRADE:-0}"
SRC="${SAMIA_SRC:-}"

# ---- Plan summary (printed in all modes including --dry-run) -----------------
_yn() { [ "$1" = "1" ] && echo "ON" || echo "OFF"; }
say "Install plan"
printf '   Virtualenv:          %s  [%s]\n'  "$VENV"           "$_venv_src"
printf '   Memory directory:    %s  [%s]\n'  "$MEMDIR"         "$_memdir_src"
printf '   With LLM extras:     %-3s  [%s]\n' "$(_yn "$WITH_LLM")"   "$_llm_src"
printf '   CUDA llama-cpp:      %-3s  [%s]\n' "$(_yn "$WITH_CUDA")"  "$_cuda_src"
printf '   systemd service:     %-3s  [%s]\n' "$(_yn "$WITH_SERVICE")" "$_service_src"
printf '   Arms in service:     %-3s  [%s]\n' "$(_yn "$ENABLE_ARMS")" "$_arms_src"
printf '   Install Claude Code: %-3s  [%s]\n' "$(_yn "$WITH_CLAUDE")" "$_claude_src"
printf '   Upgrade existing:    %s\n'          "$(_yn "$WITH_UPGRADE")"

# ---- Dry-run: show what WOULD happen and exit --------------------------------
if [ "$DRY_RUN" = "1" ]; then
  # Compute which apt packages WOULD be needed (same detection, no apt call).
  _dry_need=()
  python3 -c 'import ensurepip' 2>/dev/null || _dry_need+=(python3-venv python3-pip)
  command -v gcc  >/dev/null 2>&1 || _dry_need+=(build-essential)
  command -v curl >/dev/null 2>&1 || _dry_need+=(curl)
  command -v git  >/dev/null 2>&1 || _dry_need+=(git)
  python3 -c 'import sysconfig,os; raise SystemExit(0 if os.path.exists(sysconfig.get_path("include")+"/Python.h") else 1)' 2>/dev/null || _dry_need+=(python3-dev)

  say "Steps that WOULD run (dry-run — no side effects)"
  _step=1
  if [ "${#_dry_need[@]}" -gt 0 ]; then
    printf '   %d. sudo apt-get install -y %s\n' "$_step" "${_dry_need[*]}"; _step=$((_step+1))
  else
    printf '   %d. OS prerequisites: all present — no apt call needed\n' "$_step"; _step=$((_step+1))
  fi
  if [ -x "$VENV/bin/python" ]; then
    printf '   %d. Reuse existing virtualenv at %s\n' "$_step" "$VENV"; _step=$((_step+1))
  else
    printf '   %d. python3 -m venv %s\n' "$_step" "$VENV"; _step=$((_step+1))
  fi
  printf '   %d. pip install samia%s (from %s)\n' "$_step" \
    "$( [ "$WITH_LLM" = "1" ] && echo "[llm]" || echo "" )" \
    "$( [ -n "$SRC" ] && echo "$SRC" || { [ -n "$HERE" ] && [ -f "$HERE/pyproject.toml" ] && echo "$HERE" || echo "git+$REPO"; } )"; _step=$((_step+1))
  if [ "$WITH_CUDA" = "1" ]; then
    printf '   %d. Rebuild llama-cpp-python with CUDA\n' "$_step"; _step=$((_step+1))
  fi
  printf '   %d. mkdir -p %s\n' "$_step" "$MEMDIR"; _step=$((_step+1))
  if [ "$WITH_SERVICE" = "1" ]; then
    printf '   %d. Write + enable samia-maintenanced.service (arms: %s)\n' "$_step" "$(_yn "$ENABLE_ARMS")"; _step=$((_step+1))
  fi
  if [ "$WITH_CLAUDE" = "1" ]; then
    if command -v claude >/dev/null 2>&1; then
      printf '   %d. Claude Code: already present — skip\n' "$_step"
    else
      printf '   %d. Download + run Claude Code installer\n' "$_step"
    fi
  fi
  printf '\n   Dry-run complete. No changes made.\n'
  exit 0
fi

# ---- Interactive final confirmation -----------------------------------------
# All prompts are done; user sees the full plan before anything runs.
if [ "$INTERACTIVE" = "1" ]; then
  printf '\n'
  if ! ask_yn "Proceed with install?" "y"; then
    echo "   Aborted."
    exit 0
  fi
fi

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
if [ -n "$CUR" ] && [ "$WITH_UPGRADE" != "1" ]; then
  say "samia $CUR already installed in this venv — skipping (set SAMIA_UPGRADE=1 to reinstall/upgrade)"
else
  [ -n "$CUR" ] && say "Upgrading samia ($CUR -> ${SRC}${EXTRA})" || say "Installing samia from ${SRC}${EXTRA}"
  UP=""; [ "$WITH_UPGRADE" = "1" ] && UP="--upgrade"
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
