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

The suite is CPU-only and self-contained (the LLM arms are mocked); one GPU-gated
test skips automatically when no GPU is present. The same suite runs in CI on every
push and pull request — please keep it green.

## Code conventions

SAM/IA's source follows a three-layer documentation convention, applied to every
file under `samia/`. If you touch a file, keep it consistent:

1. **Top docstring** — a one-line purpose, then `Layer 1 (Owns / Depends)` and
   `Layer 2 (What / Why)` sections (see `samia/core/temporal.py` for the shape).
2. **Inline What/Why sandwiches** — on a non-obvious block, an em-dash comment
   above (`# Name — What: …`) and, where it adds something, below (`# Name — Why: …`).
3. **Metadata footer** — the `# [Asthenosphere] <dotted.module>` block with
   `Phase / Layer / Role / Stability / ErrorModel / Depends / Exposes / Lines`.

Two hard rules that the project leans on heavily:

- **Refactors are zero-behavior-change.** Structure/style/docs edits must leave the
  docstring-stripped AST identical to before. If you split a module, preserve the
  public import surface via re-exports so no importer breaks.
- **No secrets in the tree.** A `pre-push` hook scans for credential-shaped strings
  and blocks the push if any are found.

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
