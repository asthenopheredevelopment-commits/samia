"""SamiaAdapter — the MemoryAdapter implementation backed by the INSTALLED samia package.

This is the first (v1, only) adapter. It exercises SAM/IA the way the system's own tests
do: it plants ``type: semantic`` atom nodes under a memory root, builds the real MiniLM
vector index, and recalls through the semantic atom arm. Every call routes through public
``samia`` surfaces — no behavior is reimplemented here.

Grounding (read from the installed package, not invented):

* **Memory root** is selected by the ``ASTHENOS_MEMORY_DIR`` env var
  (``samia.core.paths.resolve_memory_root`` / ``ASTHENOS_MEMORY_DIR_ENV``). Pointing it at
  a fresh temp directory isolates a benchmark run completely — the live store is never
  touched. The root holds a ``nodes/`` subtree (the atom population) and a
  ``vector_index/`` subtree (the index).
* **A stored memory is a node file** ``<root>/nodes/<id>.md`` with frontmatter
  ``name`` + ``type: semantic`` (+ optional ``source`` / ``valid_from``) over a body —
  the exact atom shape SAM/IA's own ``test_semantic_recall`` plants and that
  ``semantic_recall._node_type`` / ``_atom_fields`` read. The node *filename* (``<id>.md``)
  is the id SAM/IA returns, so we use the caller's ``MemoryItem.id`` as the stem and map
  the returned ``<id>.md`` straight back to it.
* **Indexing** is ``samia.core.vector.build(root, rebuild=True)`` — re-embeds every node
  under ``nodes/`` into ``vector_index/embeddings.npy`` + ``manifest.json``. We rebuild
  after the final ``store`` of a batch (and lazily on first recall) so the index always
  reflects the current node set.
* **Retrieval** is the semantic atom arm: ``samia.core.semantic_recall.atom_retrieve``
  with ``ASTHENOS_SEMANTIC_ARM_ENABLED=1`` (the flag that turns the atom arm on; default
  OFF makes ``recall`` a chainogram passthrough that serves no atoms). ``atom_retrieve``
  runs the shared vector index, keeps only ``type: semantic`` hits, and returns ranked
  ``{node, score, ...}`` dicts. We map each ``node`` (``<id>.md``) back to the caller id.
* **Embedder** is the default ``sentence-transformers/all-MiniLM-L6-v2`` (CPU). It must be
  present in the local HuggingFace cache; we set ``ASTHENOS_MODEL_AUTOFETCH=0`` so a
  cache-miss fails loudly instead of reaching the network at score time (determinism rule).
* **Consolidation** uses ``samia.core.consolidation`` (the atom-minichain merge surfacer).
  Its merge *decisions* are gated by an offline REM judge that is out of scope for a
  deterministic, network-free benchmark, so ``consolidate`` here runs the programmatic
  audit + index rebuild and reports honestly that no judge-gated merge was applied (the A5
  task reads this and reports the consolidation-lift capability accordingly).

Determinism: the embedder is fixed by digest (MiniLM), the index build is a pure function
of the node set, and ``vector.query`` ranks by cosine — same nodes + same query → same
ranking. No randomness is introduced here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .base import MemoryAdapter, MemoryItem

# Env contracts owned by the installed package (read, not redefined here, so the adapter
# tracks the real names). Imported at module load so a missing package fails fast.
from samia.core.paths import ASTHENOS_MEMORY_DIR_ENV
from samia.core.semantic_recall import SEMANTIC_ARM_ENABLED_ENV
from samia.core import vector as _vector
from samia.core import semantic_recall as _semantic_recall

# The model-autofetch consent env (samia.core.netconsent). Set to "0" for a hard
# no-network guarantee at score time: a cache-miss embedder raises rather than downloading.
_MODEL_AUTOFETCH_ENV = "ASTHENOS_MODEL_AUTOFETCH"


def _atom_node_text(item: MemoryItem) -> str:
    """Render a MemoryItem as a SAM/IA ``type: semantic`` atom node file.

    Mirrors the atom shape SAM/IA's own tests plant (frontmatter ``name`` + ``type:
    semantic`` + optional ``source`` / ``valid_from``, then the body) so the node is read
    correctly by ``semantic_recall._node_type`` (population gate) and ``_atom_fields``.
    """
    lines = [f"name: {item.id}", "type: semantic"]
    if item.source:
        lines.append(f"source: {item.source}")
    if item.valid_from:
        lines.append(f"valid_from: {item.valid_from}")
    fm = "---\n" + "\n".join(lines) + "\n---\n"
    body = item.text.strip()
    return fm + body + "\n"


class SamiaAdapter(MemoryAdapter):
    """MemoryAdapter over the installed ``samia`` package, isolated to a temp memory root.

    Each instance owns a private temp directory used as ``ASTHENOS_MEMORY_DIR`` for the
    duration of every call, so concurrent or sequential runs never collide and the live
    store is never touched. ``reset`` wipes that directory back to empty.
    """

    name = "samia"

    def __init__(self, root: str | None = None) -> None:
        """Create the adapter and its isolated memory root.

        Parameters
        ----------
        root:
            Optional explicit directory to use as the memory root. When ``None`` (default)
            a fresh temp directory is created and owned by this adapter (removed on
            ``close``). Passing an explicit path is for debugging/inspection; that path is
            NOT auto-removed.
        """
        if root is None:
            self._root = Path(tempfile.mkdtemp(prefix="samia_bench_"))
            self._owns_root = True
        else:
            self._root = Path(root)
            self._owns_root = False
        (self._root / "nodes").mkdir(parents=True, exist_ok=True)
        # Pending = items stored since the last index build; the index is (re)built lazily
        # on the next recall so a multi-batch store does not re-embed after every batch.
        self._dirty = False

    # -- isolation plumbing ------------------------------------------------

    def _apply_env(self) -> dict:
        """Point samia at our isolated root + pin the embedder; return prior env to restore.

        Sets ASTHENOS_MEMORY_DIR (our temp root), ASTHENOS_SEMANTIC_ARM_ENABLED=1 (atom arm
        ON), and ASTHENOS_MODEL_AUTOFETCH=0 (no network at score time). Returns the prior
        values so ``_restore_env`` leaves the process environment exactly as it found it —
        the adapter must not leak its config onto an outer process or a live daemon.
        """
        keys = (ASTHENOS_MEMORY_DIR_ENV, SEMANTIC_ARM_ENABLED_ENV, _MODEL_AUTOFETCH_ENV)
        prior = {k: os.environ.get(k) for k in keys}
        os.environ[ASTHENOS_MEMORY_DIR_ENV] = str(self._root)
        os.environ[SEMANTIC_ARM_ENABLED_ENV] = "1"
        os.environ[_MODEL_AUTOFETCH_ENV] = "0"
        return prior

    @staticmethod
    def _restore_env(prior: dict) -> None:
        """Restore env keys captured by ``_apply_env`` to their prior values (or unset)."""
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # -- MemoryAdapter contract -------------------------------------------

    def store(self, items: list[MemoryItem]) -> list[str]:
        """Write each item as a ``type: semantic`` atom node; mark the index dirty.

        Returns the caller ids in order. The vector index is (re)built lazily on the next
        ``recall`` (or eagerly via ``build_index``), so storing N batches costs one embed
        pass, not N. Ids must be filesystem-safe stems (the task datasets use slug ids).
        """
        nodes_dir = self._root / "nodes"
        out: list[str] = []
        for it in items:
            (nodes_dir / f"{it.id}.md").write_text(_atom_node_text(it), encoding="utf-8")
            out.append(it.id)
        if items:
            self._dirty = True
        return out

    def build_index(self) -> None:
        """Force a full rebuild of the vector index over the current node set.

        Idempotent and safe to call eagerly; ``recall`` calls this automatically when the
        store has changed since the last build. Clears samia's per-process node-type cache
        so a rebuilt store is not served stale types.
        """
        prior = self._apply_env()
        try:
            _semantic_recall._clear_type_cache()
            _vector.build(self._root, rebuild=True)
            self._dirty = False
        finally:
            self._restore_env(prior)

    def recall(self, query: str, k: int = 10) -> list[str]:
        """Rank stored memory ids for ``query`` via the SAM/IA semantic atom arm.

        Builds the index first if the store changed since the last build, then calls
        ``semantic_recall.atom_retrieve`` (atom arm ON), which runs the real MiniLM vector
        query, keeps only ``type: semantic`` hits, and returns them ranked. We map each
        returned ``<id>.md`` node back to the caller id and truncate to ``k``. An empty
        store (or no index) yields ``[]`` — scored as a miss, not an error.
        """
        if not any((self._root / "nodes").glob("*.md")):
            return []
        if self._dirty:
            self.build_index()
        prior = self._apply_env()
        try:
            _semantic_recall._clear_type_cache()
            atoms = _semantic_recall.atom_retrieve(self._root, query, k=k)
        finally:
            self._restore_env(prior)
        out: list[str] = []
        for a in atoms:
            node = a.get("node") or ""
            stem = node[:-3] if node.endswith(".md") else node
            if stem:
                out.append(stem)
        return out[:k]

    def consolidate(self) -> dict:
        """Run the programmatic consolidation audit + rebuild; report honestly, no judge.

        ``samia.core.consolidation.audit_all`` surfaces near-duplicate atom-minichains, but
        the actual *merge* is gated by an offline REM judge (an LLM) that a deterministic,
        network-free benchmark must not invoke. So this performs the programmatic audit and
        rebuilds the index, and returns a summary marking ``judge_applied=False`` — the A5
        task reads this and reports consolidation-lift as measured-without-judge (no
        fabricated gain). Returns ``{}``-shaped summary when consolidation surfaces nothing.
        """
        prior = self._apply_env()
        try:
            summary: dict = {"adapter": self.name, "judge_applied": False}
            try:
                findings = _consolidation_audit(self._root)
                summary["candidate_merges"] = len(findings)
            except Exception as exc:  # consolidation surface is best-effort here
                summary["candidate_merges"] = 0
                summary["note"] = f"audit unavailable: {type(exc).__name__}"
            # Rebuild so any external state change is reflected; index stays consistent.
            _semantic_recall._clear_type_cache()
            _vector.build(self._root, rebuild=True)
            self._dirty = False
            return summary
        finally:
            self._restore_env(prior)

    def reset(self) -> None:
        """Wipe the isolated store back to empty (drop nodes/ + the vector index)."""
        for sub in ("nodes", "vector_index"):
            shutil.rmtree(self._root / sub, ignore_errors=True)
        (self._root / "nodes").mkdir(parents=True, exist_ok=True)
        self._dirty = False

    def close(self) -> None:
        """Remove the temp root if this adapter created it (cleanup at end of a run)."""
        if self._owns_root:
            shutil.rmtree(self._root, ignore_errors=True)

    # context-manager sugar so a task can `with SamiaAdapter() as a:` and auto-clean.
    def __enter__(self) -> "SamiaAdapter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _consolidation_audit(root: Path) -> list:
    """Run the package's atom-minichain consolidation audit over ``root`` (or [])."""
    from samia.core import consolidation as _con
    return _con.audit_all(root)
