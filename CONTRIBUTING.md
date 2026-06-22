# Contributing to SAM/IA

Thanks for your interest. SAM/IA is a single-operator project run daily on one
workstation; it is not yet hardened for multi-user or production-scale use, so the
most valuable contributions right now are **bug reports, reproductions, and focused
fixes** rather than large features. Issues and PRs are both welcome.

## Development setup

```bash
git clone https://github.com/asthenopheredevelopment-commits/samia.git
cd samia
python -m venv .venv && source .venv/bin/activate
pip install "numpy<2" -e ".[test]"      # numpy<2 matches the ABI the stack is built against
# optional local-LLM arms (judge / fact-extraction / synthesis):
pip install -e ".[llm]"
```

## Running the tests

```bash
CUDA_VISIBLE_DEVICES="" pytest samia/ -q
```

The suite is CPU-only and the LLM arms are mocked. The semantic tests download the
small MiniLM sentence-transformers embedder once on first run (cached on disk
thereafter); one embedder-dependent test skips automatically if that embedder can't
be loaded (e.g. an offline cold cache). The real-backend LLM test is also skipped
unless `llama-cpp-python` is installed and `$SAMIA_TEST_GGUF` points at a `.gguf`
model — install the `[llm]` extra and set that variable to run it locally.

CI runs the same `pytest samia/` suite on every push and pull request, with one
difference: CI installs the `[llm]` extra and provides a tiny gguf via
`$SAMIA_TEST_GGUF`, so it additionally exercises one real-backend LLM load that the
plain `.[test]` local run skips. Please keep it green.

## Code conventions

SAM/IA's source follows a three-layer documentation convention, applied to every
file under `samia/`. If you touch a file, keep it consistent:

1. **Top docstring** — a one-line purpose, then `Layer 1 (Owns / Depends)` and
   `Layer 2 (What / Why)` sections, plus an optional `Layer 3 (Changelog)` section
   when the file has carve/refactor history worth noting (see `samia/core/temporal.py`
   for the shape — it carries all three).
2. **Inline What/Why sandwiches** — on a non-obvious block, an em-dash comment
   above (`# Name — What: …`) and, where it adds something, below (`# Name — Why: …`).
3. **Metadata footer** — the `# [Asthenosphere] <dotted.module>` block with
   `Phase / Layer / Role / Stability / ErrorModel / Depends / Exposes / Lines`.

Two hard rules that the project leans on heavily:

- **Refactors are zero-behavior-change.** Structure/style/docs edits must leave the
  docstring-stripped AST identical to before. If you split a module, preserve the
  public import surface via re-exports so no importer breaks.
- **No secrets in the tree.** The maintainer's clone runs a local `pre-push` gate
  (in `.git/hooks/`, not part of the tracked tree, so a fresh clone does **not**
  install it automatically) that blocks a push unless every check passes: it scans
  for credential-shaped strings, runs the three-layer convention check, and runs the
  CPU test suite. If you want the same gate on your clone, copy that hook into your
  own `.git/hooks/pre-push`. Whether or not you install it, keep secrets out of the
  tree and run `pytest samia/ -q` before opening a PR.

## Pull requests

- Keep PRs focused; one concern per PR.
- Include a test for a bug fix (a failing test that your change makes pass).
- Make sure `pytest samia/ -q` is green and your changes are behavior-preserving
  unless the PR is explicitly a behavior change (say so, and explain why).
- Describe **what** changed and **why** in the PR body.

## Reporting bugs

Open an issue with: what you did, what you expected, what happened, and the smallest
reproduction you can manage (a few lines against a temp `ASTHENOS_MEMORY_DIR` is ideal —
see `examples/quickstart.py`). Include your Python version and OS.
