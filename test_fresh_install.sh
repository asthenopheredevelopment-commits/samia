#!/usr/bin/env bash
# test_fresh_install.sh — SAM/IA cold-metal acceptance test (run on the fresh OS).
# Pass = every section prints OK and the final summary shows 0 failures.
set -uo pipefail
ST="$(cd "$(dirname "$0")" && pwd)"
echo "== SAM/IA fresh-install test =="; echo "staging: $ST"

# Bare-OS contract: every network action is operator-gated. Set
# SAMIA_TEST_ASSUME_YES=1 to pre-approve all of them (e.g. an agent that has
# already obtained approval through its own permission system).
confirm() {
  [ "${SAMIA_TEST_ASSUME_YES:-0}" = "1" ] && { echo "(pre-approved) $1"; return 0; }
  printf '%s [y/N] ' "$1"; read -r a; [ "$a" = "y" ] || [ "$a" = "Y" ]
}

echo "-- 0. OS prerequisites (stock Ubuntu desktop lacks venv/pip/curl)"
python3 --version || { echo "FAIL: no python3"; exit 1; }
if ! python3 -c "import ensurepip" 2>/dev/null; then
  echo "MISSING PREREQUISITES — run this, then rerun the test:"
  echo "    sudo apt install -y python3-venv python3-pip curl"
  exit 1
fi
# curl is not needed by SAM/IA itself, but later steps (e.g. installing
# Claude Code) use it — warn early rather than surprise later.
command -v curl >/dev/null || echo "NOTE: curl not found — fine for this test, but install it (sudo apt install -y curl) before any curl-fetched tooling installers (e.g. Claude Code)"

echo "-- 1. python + venv"
# clear any half-built venv from a prior failed run (round-1 residue lesson)
python3 -c "import shutil, os; shutil.rmtree(os.path.expanduser('~/samia_test_venv'), ignore_errors=True)"
python3 -m venv ~/samia_test_venv && source ~/samia_test_venv/bin/activate
pip -q install --upgrade pip

echo "-- 2. install the package (2 deps + extras; network required once)"
confirm "NETWORK: pip downloads samia's dependencies (numpy, sentence-transformers + torch, llama-cpp-python, pytest) from PyPI. Proceed?" \
  || { echo "DECLINED by operator — cannot continue without the package"; exit 1; }
pip install "$ST[llm,test]" || pip install "$ST" || { echo "FAIL: pip install"; exit 1; }
python3 -c "import samia, samia.core.semantic_recall; print('import OK:', samia.__file__)"

echo "-- 3. test suite from the installed tree (autofetch OFF: unit tests must never download models)"
ASTHENOS_MODEL_AUTOFETCH=0 python3 -m pytest "$ST/samia" -q | tail -2

echo "-- 4. quickstart from NOTHING (the README contract; no LLM required)"
confirm "NETWORK: first vector.build downloads the MiniLM embedder (~90MB, Apache-2.0) from HuggingFace. Proceed?" \
  || { echo "DECLINED by operator — steps 4-5 need the embedder; stopping here"; exit 0; }
# the script-level approval above becomes the engine's standing consent for
# this step (the engine's own gate would otherwise refuse: heredoc stdin != tty)
export ASTHENOS_MODEL_AUTOFETCH=1
export ASTHENOS_MEMORY_DIR=~/samia_test_store
python3 - <<'EOF'
import os
from pathlib import Path
from samia.core import frontmatter, vector, context_extension, semantic_recall
md = Path(os.environ["ASTHENOS_MEMORY_DIR"]).expanduser()
# episodic content: a reference node (served by the chainogram EVIDENCE arm)
fm = {"name": "first_fact", "description": "cold-metal test fact",
      "type": "reference"}
frontmatter.write_node(md / "nodes" / "first_fact.md", fm, list(fm.keys()),
                       "The cold-metal acceptance test wrote this node on a fresh OS.\n")
# semantic content: a hand-written atom (served by the FACTS arm — at runtime
# these are produced by fact extraction, which needs the LLM; writing one by
# hand keeps the fresh-install test LLM-free)
fa = {"name": "sem_first_atom", "description": "hand-written semantic atom",
      "type": "semantic", "source": "s01", "valid_from": "2026-06-12",
      "tier": "warm"}
frontmatter.write_node(md / "nodes" / "sem_first_atom.md", fa, list(fa.keys()),
                       "SAM/IA stores semantic atoms like this cold-metal fact atom.\n")
vector.build(md)   # first run downloads MiniLM (license: Apache-2.0)
out = context_extension.chainogram_retrieve(md, "cold-metal test", budget_tokens=2000)
assert "error" not in out, out
os.environ["ASTHENOS_SEMANTIC_ARM_ENABLED"] = "1"
res = semantic_recall.recall(md, "what did the acceptance test write?", budget_tokens=2000)
assert res.get("facts_n", 0) >= 1, res
assert "cold-metal fact atom" in res["context"], res["context"][:300]
print("quickstart OK — store created, indexed, both arms served content")
EOF

echo "-- 4b. maintenance daemon: one-shot pass on the test store (no network: kill-switch pinned)"
ASTHENOS_MODEL_AUTOFETCH=0 python3 -m samia.runtime.maintenanced \
  --memory-dir ~/samia_test_store --oneshot || { echo "FAIL: maintenanced oneshot"; exit 1; }
export ASTHENOS_MODEL_AUTOFETCH=1   # restore standing consent for step 5 (gated below)

echo "-- 5. model auto-fetch (downloads ~2.4GB Qwen3-4B; license printed first)"
if ! confirm "NETWORK: download Qwen3-4B (~2.4GB, Apache-2.0) from HuggingFace?"; then
  echo "DECLINED by operator — LLM arms stay on MockBackend; rerun step 5 anytime"
else
python3 - <<'EOF'
from samia.runtime.model_fetch import fetch_model
p = fetch_model("Qwen3-4B-Instruct-2507-Q4_K_M")
print("model present:", p)
from samia.runtime.inference import get_backend_for_model
b = get_backend_for_model(str(p))
print("backend:", type(b).__name__, "| completion:", b.complete("Say OK", max_tokens=8)[:40])
EOF
fi

echo "-- 6. hygiene: nothing outside sanctioned paths"
# grep -iv: case-insensitive so /Vault and /media/*/Vault mountpoints are
# filtered too (the staging tree may live under any of /vault, /Vault, or a
# removable /media/<user>/Vault mount — all are sanctioned, none is a leak).
find / -newer ~/samia_test_venv -path /proc -prune -o -path /sys -prune -o \
  -name "*.md" -path "*nodes*" -print 2>/dev/null | grep -iv "samia_test_store\|samia_test_venv\|vault" | head -5
echo "this install lives in ~/samia_test_venv — in a new shell run:  source ~/samia_test_venv/bin/activate"
echo "== DONE — compare against RUNBOOK pass criteria =="
