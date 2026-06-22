"""SAM/IA quickstart — store two memories, build the index, recall the relevant one.

    python examples/quickstart.py

First run needs the ~90 MB sentence-transformers embedder (all-MiniLM-L6-v2). If it
isn't already in the HuggingFace cache the download is consent-gated: at an interactive
terminal you're asked '[y/N]', and a non-interactive run (piped / CI / no tty) is
REFUSED — it errors before recall unless you give standing consent with
ASTHENOS_MODEL_AUTOFETCH=1, or pre-download the model first
(e.g. `huggingface-cli download sentence-transformers/all-MiniLM-L6-v2`).
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

# For an unattended / CI run, give the embedder download standing consent up front:
# export ASTHENOS_MODEL_AUTOFETCH=1 before running (=0 is a kill switch that refuses
# every download). Left unset, the fetch asks at an interactive tty and is refused with
# no terminal. Uncomment to consent from inside the script:
# os.environ.setdefault("ASTHENOS_MODEL_AUTOFETCH", "1")

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

# Expected output — the load-bearing part is the ORDER: the relevant memory ranks
# first (higher score = more relevant). The printed score is a composite (term-index
# hits, re-ranking, and the engram/ring fast tiers fold on top of the embedder
# cosine), not a bare cosine, so treat the magnitudes as illustrative only:
#   higher  paris-trip.md    Paris trip
#   lower   coffee-pref.md   Coffee preference
