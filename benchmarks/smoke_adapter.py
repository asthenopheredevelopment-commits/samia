"""Foundation smoke test: store 5 facts, recall one, assert the gold id comes back.

Run with the test venv's interpreter so the INSTALLED ``samia`` package is exercised:

    python benchmarks/smoke_adapter.py

This is the build-order step-1 round-trip check from ``BENCHMARK_DESIGN_v1.md``: it does
NOT measure any capability axis, it only proves the adapter contract works end to end
against the installed package (write atom nodes → build the real MiniLM index → recall
through the semantic atom arm → map ids back). Deterministic and network-free: the
embedder is the pinned MiniLM (cache-only; autofetch off in the adapter).

Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys

from adapters import MemoryItem, SamiaAdapter


# Five unambiguous facts (clean gold labels — defect D1/D2/D4). The query targets exactly
# one of them; the others are clear distractors on distinct topics so the gold is the
# unique best match.
_FACTS = [
    MemoryItem(id="fact_cat", text="Maria adopted a cat named Pixel in April 2023.",
               valid_from="2023-04-01", source="s01"),
    MemoryItem(id="fact_hike", text="Maria went hiking in the Alps last summer.",
               source="s01"),
    MemoryItem(id="fact_car", text="David bought a blue Toyota sedan in 2021.",
               valid_from="2021-06-15", source="s02"),
    MemoryItem(id="fact_job", text="Priya started a new job as a data engineer in Berlin.",
               valid_from="2022-09-01", source="s03"),
    MemoryItem(id="fact_recipe", text="The lasagna recipe calls for three cups of ricotta.",
               source="s04"),
]

# The probe and the single id it must return. The phrasing avoids the exact stored words
# so the test exercises semantic recall, not a string match.
_QUERY = "Which pet did Maria get?"
_GOLD_ID = "fact_cat"


def main() -> int:
    with SamiaAdapter() as adapter:
        adapter.reset()

        stored = adapter.store(_FACTS)
        assert stored == [f.id for f in _FACTS], (
            f"store should echo ids in order; got {stored}")

        ranked = adapter.recall(_QUERY, k=5)
        print(f"query:  {_QUERY!r}")
        print(f"gold:   {_GOLD_ID!r}")
        print(f"ranked: {ranked}")

        assert ranked, "recall returned no ids (empty store or no index?)"
        assert _GOLD_ID in ranked, (
            f"gold id {_GOLD_ID!r} not in recall top-5: {ranked}")
        # The gold should be the top hit for this unambiguous probe.
        assert ranked[0] == _GOLD_ID, (
            f"gold id {_GOLD_ID!r} not ranked first; got {ranked[0]!r} (full: {ranked})")

        # reset returns to empty.
        adapter.reset()
        assert adapter.recall(_QUERY, k=5) == [], "store not empty after reset"

    print("SMOKE OK: store 5 -> recall -> gold id returned at rank 1; reset clears store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
