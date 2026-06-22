# Examples

A runnable example for SAM/IA. It is self-contained and uses the same in-process
library API that SAM/IA also exposes as MCP primitives.

The example below covers the write → index → recall loop. The library exposes a
wider primitive surface than it demonstrates — temporal queries, single-node
reads, salience tagging, block injection, and the forget/supersession tools, among
others (see `samia/core/mcp_server/`).

## quickstart.py

Store two memories, build the semantic index, and recall the relevant one — the
30-second tour of the write → index → recall loop.

```bash
# from a clone (no install needed — the script puts the repo root on the path):
python examples/quickstart.py

# or after installing the package:
pip install -e .
python examples/quickstart.py
```

First run downloads the ~90 MB sentence-transformers embedder (`all-MiniLM-L6-v2`,
a 384-dimensional model). By default it writes to a throwaway temp directory. Set
`ASTHENOS_MEMORY_DIR` to point it at a persistent location instead — when that
variable is already set in your environment it is used as-is and no temp dir is
created (the script only falls back to a temp dir when the variable is unset).

Expected output (the relevant memory ranks first; exact scores vary slightly):

```
Q: where did the user travel?

  0.54  paris-trip.md    Paris trip
  0.08  coffee-pref.md   Coffee preference
```

Scores are printed with two decimals (`{score:.2f}`); the exact cosine values
depend on the embedder version, so treat `0.54` / `0.08` as representative rather
than fixed.
