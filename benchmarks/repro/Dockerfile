# SAM/IA capability benchmark — one-command reproduction image.
#
# Builds a clean, pinned environment that installs the SAM/IA package the way a third party
# would (pip install of the release wheel, never a dev tree), pre-fetches the pinned MiniLM
# embedder into the image's HuggingFace cache so the benchmark runs CACHE-ONLY with NO network
# at score time, and runs all nine axes deterministically with the seeds in seeds.toml.
#
# Build (from the release ROOT so the build context includes the package + benchmarks):
#     docker build -f benchmarks/repro/Dockerfile -t samia-benchmark .
#
# Run (writes scores.json + the per-axis raw files into the container's results dir; mount a
# host directory to keep them):
#     docker run --rm -v "$PWD/benchmarks/results:/work/benchmarks/results" samia-benchmark
#
# Verify determinism (runs twice, asserts the scoring subtree is byte-identical):
#     docker run --rm samia-benchmark \
#         python benchmarks/run_benchmark.py --seed 1337 --sizes 100,1000,10000 --check-determinism
#
# The open-ended LLM-judge subset (A3/A4-N/A/A7) is NOT run in this image (no local judge
# daemon is bundled): the harness reports those subsets as "pending" and the programmatic
# metrics — the trustworthy numbers — reproduce exactly without it.

FROM python:3.12-slim

# Pinned dependency set (matches benchmarks/repro/seeds.toml [pins]). CPU-only torch keeps the
# image small and the run host-independent; the embedder is a 384-dim MiniLM on CPU.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

WORKDIR /work

# Install the pinned numerical + embedder stack first (its own layer, so it caches across
# rebuilds), CPU-only torch from the official CPU index.
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.12.1" \
        "numpy==2.5.0" \
        "transformers==5.12.1" \
        "sentence-transformers==5.6.0"

# Copy the release source (package + benchmarks + pyproject) and install the SAM/IA package
# from it — this is the INSTALLED package the benchmark must exercise, not a dev tree.
COPY pyproject.toml README.md LICENSE ./
COPY samia ./samia
COPY benchmarks ./benchmarks
RUN pip install .

# Pre-fetch the pinned embedder INTO THE IMAGE so the benchmark runs cache-only and never
# reaches the network at score time (determinism non-negotiable #2/#3). After this layer the
# model is resident in HF_HOME and ASTHENOS_MODEL_AUTOFETCH=0 will be satisfied from cache.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# No network is needed (or permitted) at score time; the adapter sets autofetch off itself,
# this just documents the contract for anyone reading the image.
ENV ASTHENOS_MODEL_AUTOFETCH=0

# Default command: the full deterministic benchmark with the pinned seed + sizes. Override the
# CMD to pass --check-determinism or an --axes subset.
CMD ["python", "benchmarks/run_benchmark.py", "--seed", "1337", "--sizes", "100,1000,10000"]
