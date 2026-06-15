"""SAM/IA quickstart — store two memories, build the index, recall the relevant one.

    python examples/quickstart.py

First run downloads the ~80 MB sentence-transformers embedder (all-MiniLM-L6-v2).
Every call here is the same in-process library SAM/IA exposes as MCP primitives.
"""
import os
import sys
import pathlib
import tempfile

# Run straight from a clone (`python examples/quickstart.py`) without installing —
# put the repo root on the path. Harmless once you've `pip install`ed the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Point SAM/IA at a memory directory (created on first use). Omit this and it
# uses the default XDG location, ~/.local/share/samia/memory.
os.environ.setdefault("ASTHENOS_MEMORY_DIR", tempfile.mkdtemp(prefix="samia-quickstart-"))

from samia.core.paths import resolve_memory_root
from samia.core import vector
from samia.core.mcp_server import memory_write_node, memory_search

mem = resolve_memory_root()  # creates <dir>/nodes/ on first use

# 1. Store a couple of memories.
memory_write_node(
    mem,
    name="paris-trip",
    title="Paris trip",
    description="User visited Paris in spring 2024",
    body="The user traveled to Paris in April 2024 and visited the Louvre.",
    type_="semantic",
)
memory_write_node(
    mem,
    name="coffee-pref",
    title="Coffee preference",
    description="The user's coffee order",
    body="The user prefers oat-milk flat whites, no sugar.",
    type_="semantic",
)

# 2. Build the semantic index over what we just stored.
vector.build(mem)

# 3. Recall — semantic search ranks the relevant memory first.
print("Q: where did the user travel?\n")
for hit in memory_search(mem, "where did the user travel?", top_k=2):
    print(f"  {hit['score']:.2f}  {hit['node']:<16} {hit['title']}")

# Expected output (the relevant memory ranks first; exact scores vary slightly):
#   ~0.5  paris-trip.md    Paris trip
#   ~0.1  coffee-pref.md   Coffee preference
