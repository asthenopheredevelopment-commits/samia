# Examples

Runnable examples for SAM/IA. Each is self-contained and uses the same in-process
library API that SAM/IA also exposes as MCP primitives.

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

First run downloads the ~80 MB sentence-transformers embedder (`all-MiniLM-L6-v2`).
It writes to a throwaway temp directory; set `ASTHENOS_MEMORY_DIR` to keep the
memories somewhere persistent.

Expected output (the relevant memory ranks first; exact scores vary slightly):

```
Q: where did the user travel?

  ~0.5  paris-trip.md    Paris trip
  ~0.1  coffee-pref.md   Coffee preference
```
