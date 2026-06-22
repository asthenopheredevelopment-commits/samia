"""MemoryAdapter — the adapter contract every memory system implements for the benchmark.

The SAM/IA Capability Benchmark (see ``BENCHMARK_DESIGN_v1.md``) measures *what a memory
system can do* through one narrow, system-agnostic interface so the same tasks and metrics
run unchanged against any system. v1 ships exactly one adapter (SAM/IA); the abstraction
exists so additional systems can be plugged in later **without touching tasks or metrics**.

The contract is the four methods the design locks down:

    store(items)        ingest a batch of memories (returns the assigned ids, in order)
    recall(query, k)    rank stored memory ids by relevance to a query (best first)
    consolidate()       run the system's offline merge / REM / maintenance pass
    reset()             discard all state so the next task starts from an empty store

Design rules these signatures encode:

* **Retrieval (A1) and retention (A2) are separate tasks over separate data** (defect D6).
  This interface does NOT bake in a notion of "delay" — a retention task interleaves its
  own ``store`` batches and re-queries; the adapter just stores and recalls. The interface
  stays the same; only the *task* differs. Conflating the two is the field's most common
  defect and the contract is deliberately neutral about it.
* **Recall returns ranked IDS, not prose.** Programmatic scoring (recall@k, MRR, set-F1)
  is primary; it operates on the returned id ranking, never on generated text. This avoids
  the reader/judge confound (defect D5). The pinned judge, when used, scores the
  *open-ended* axes separately and is not part of this interface.
* **Every item carries an explicit, unambiguous gold id** (defect D1/D2/D4). The id the
  store assigns to an item is the same id a task scores against — there is one source of
  truth for "did the right memory come back".
* **Isolation is the adapter's responsibility.** A benchmark run must never read or write
  a live/production store. Implementations isolate per-run state (e.g. a temp directory)
  and ``reset`` returns to empty; the harness never has to know how.

This module is pure (stdlib only) so it imports with zero heavy dependencies; the concrete
adapters carry the system-specific imports.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    """One memory to be stored, with the gold metadata a task scores against.

    Attributes
    ----------
    id:
        Stable, caller-assigned identifier. This is the **gold label** a retrieval task
        checks for in the recall ranking, so it must be unique within a store and known to
        the task before scoring. The adapter must make ``recall`` return *this exact id*
        for the matching memory — adapters that have their own internal node ids are
        responsible for mapping back to it (see ``samia_adapter``).
    text:
        The memory content to ingest (the claim / fact / turn).
    valid_from:
        Optional ISO-ish date string (e.g. ``"2023-04-01"``) marking when the fact became
        true. Empty string means undated. Used by temporal axes (A3); ignored elsewhere.
    source:
        Optional provenance tag (e.g. a session id, or a trust label for the firewall
        axis A8). Empty string means no source. Adapters that support provenance/quarantine
        may route on it; adapters that do not simply store it as metadata.
    trusted:
        Provenance/firewall flag (A8). ``True`` = an ordinary trusted memory. ``False`` =
        an untrusted/poisoned item that the system is expected to quarantine rather than
        recall as truth. Adapters without a firewall store it like any other item and the
        A8 task reports the capability as absent.
    meta:
        Free-form per-item metadata a task may need to carry through store→recall→score
        (e.g. a multi-hop chain id, a contradiction pair id). The adapter is not required
        to interpret it; it exists so tasks stay self-describing.
    """

    id: str
    text: str
    valid_from: str = ""
    source: str = ""
    trusted: bool = True
    meta: dict = field(default_factory=dict)


class MemoryAdapter(abc.ABC):
    """The benchmark's memory-system contract: store / recall / consolidate / reset.

    Subclasses wrap one memory system. They MUST be deterministic given fixed inputs and a
    fixed environment (the harness pins seeds, the embedder, and forbids network at score
    time), and they MUST isolate all state per run so a benchmark never touches a live
    store. No method may reach the network at score time.
    """

    #: A short, stable identifier for this adapter (e.g. ``"samia"``). Used in result
    #: filenames and the report so a number is always attributable to a system+version.
    name: str = "abstract"

    @abc.abstractmethod
    def store(self, items: list[MemoryItem]) -> list[str]:
        """Ingest a batch of memories and return their ids in the same order as ``items``.

        The returned ids are the ids the caller passed in (``MemoryItem.id``); returning
        them confirms ingestion and lets a task assert the round-trip. Calling ``store``
        more than once is additive — items accumulate until ``reset``. An implementation
        may defer index construction (embedding, etc.) until the first ``recall`` so a
        multi-batch task does not pay to re-index after every batch, but after ``store``
        returns the items MUST be recallable on the next ``recall`` call.

        Determinism: storing the same items in the same order under the same environment
        must produce the same subsequent recall rankings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recall(self, query: str, k: int = 10) -> list[str]:
        """Return up to ``k`` stored memory ids ranked by relevance to ``query`` (best first).

        The result is a ranking of ``MemoryItem.id`` values, NOT generated prose — all
        primary scoring (recall@k, MRR, set-F1) reads this list directly, which is what
        keeps the benchmark free of the reader/judge confound. Fewer than ``k`` ids may be
        returned when the store holds fewer candidates or the system surfaces fewer; an
        empty list is a valid "nothing relevant / no index" answer (the task scores it as a
        miss, it is not an error).

        Quarantined / untrusted items (A8) MUST NOT appear here as if they were trusted
        truth; a system that quarantines them simply omits them from the ranking.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def consolidate(self) -> dict:
        """Run the system's offline consolidation (REM / merge / maintenance) pass once.

        Used by the consolidation-gain axis (A5), which scores ``recall`` *before* vs
        *after* this call to measure the lift a consolidation cycle provides. Returns a
        small JSON-serializable summary of what the pass did (counts, ids touched) for the
        audit trail — the metric is the recall delta, not this dict. A system with no
        consolidation step returns ``{}`` (or a dict noting the no-op) and the A5 task
        reports the capability as absent rather than fabricating a gain.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Discard ALL stored state so the next task starts from an empty, isolated store.

        After ``reset`` the store holds zero memories and ``recall`` returns ``[]`` until
        the next ``store``. This is what guarantees task independence and that no benchmark
        run leaves residue (in particular, that it never mutated a live/production store).
        Calling ``reset`` on an already-empty adapter is a no-op.
        """
        raise NotImplementedError
