"""samia.core.test_temporal_semantic — tests for the semantic temporal-query path in samia.core.temporal (BLOCKER fix).

What: tmp-store tests for samia.core.temporal.query(..., semantic=...). Plants a
  self-contained store (a few dated nodes), then exercises the semantic branch
  (1) WITH a real vector index built (CPU MiniLM via vector.build) and
  (2) with NO index built. Pins the fail-soft contract: the no-index case must
  return the non-semantic time scan plus a trailing {"note": ...} diagnostic and
  must NOT raise SystemExit (a public MCP caller must never get a process exit).
Why: temporal.query previously hard-imported the dev-tree-only memory_vector_index
  shim and called vector.query (which raises SystemExit when no index exists)
  without a guard. This test covers the refactored in-package import + fail-soft.

Depends: samia.core.{temporal, vector, frontmatter} (real embedder via vector.build
  — small node set so the MiniLM embed is cheap).
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import tempfile
import unittest
from pathlib import Path

from samia.core import temporal
from samia.core import vector


_HAS_EMBEDDER = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    return md


def _node(md: Path, stem: str, name: str, body: str,
          valid_from: str, valid_to: str = "null") -> None:
    fm = (f"---\nname: {name}\ntype: semantic\n"
          f"valid_from: {valid_from}\nvalid_to: {valid_to}\n---\n")
    (md / "nodes" / f"{stem}.md").write_text(fm + body + "\n", encoding="utf-8")


def _plant(md: Path) -> None:
    _node(md, "node_bbq_001", "barbecue brisket physics",
          "Low and slow smoking of brisket relies on collagen breakdown.",
          "2026-01-10")
    _node(md, "node_tax_001", "quarterly tax filing",
          "Estimated quarterly taxes are due in April for the prior period.",
          "2026-02-15")
    _node(md, "node_pet_001", "dog vaccination schedule",
          "Annual rabies booster keeps the dog's vaccination current.",
          "2026-03-20")


class TestTemporalSemanticNoIndex(unittest.TestCase):
    """semantic= with NO vector index must fail soft, not SystemExit."""

    def test_no_index_fails_soft_with_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _plant(md)
            # No vector.build -> no index. The semantic branch must catch the
            # SystemExit from vector.query and return the non-semantic scan.
            try:
                out = temporal.query(
                    md, at=None, since=None, range_pair=None,
                    semantic="barbecue", top_k=5)
            except SystemExit as exc:  # pragma: no cover - the bug we fixed
                self.fail(f"temporal.query(semantic=) raised SystemExit: {exc}")
            # Trailing diagnostic note present, naming the degradation.
            notes = [h for h in out if "note" in h]
            self.assertEqual(len(notes), 1, out)
            self.assertIn("semantic", notes[0]["note"])
            # The real (non-note) results are the time scan over all planted
            # nodes (valid intervals all open, no at/since/range filter).
            nodes = [h for h in out if "node" in h]
            self.assertEqual(len(nodes), 3, nodes)

    def test_no_index_non_semantic_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _plant(md)
            out = temporal.query(
                md, at=None, since=None, range_pair=None,
                semantic=None, top_k=5)
            # No semantic -> no note appended.
            self.assertTrue(all("note" not in h for h in out), out)
            self.assertEqual(len(out), 3, out)


@unittest.skipUnless(_HAS_EMBEDDER, "torch+transformers required for MiniLM build")
class TestTemporalSemanticWithIndex(unittest.TestCase):
    """semantic= with a real index built returns semantically ranked hits."""

    def test_with_index_uses_vector_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _plant(md)
            vector.build(md, rebuild=True)
            out = temporal.query(
                md, at=None, since=None, range_pair=None,
                semantic="smoking brisket low and slow", top_k=3)
            # No fail-soft note when the index exists.
            self.assertTrue(all("note" not in h for h in out), out)
            nodes = [h for h in out if "node" in h]
            self.assertTrue(nodes, "expected at least one semantic hit")
            # The bbq node should be present (it is the semantic match).
            self.assertIn("node_bbq_001.md", {h["node"] for h in nodes})

    def test_with_index_respects_temporal_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _plant(md)
            vector.build(md, rebuild=True)
            # since filter after all valid_from dates still keeps open-ended
            # nodes (valid_to=null); use a contains-style `at` to bound it.
            out = temporal.query(
                md, at=_dt.date(2026, 1, 11), since=None, range_pair=None,
                semantic="brisket", top_k=3)
            nodes = [h for h in out if "node" in h]
            # bbq node valid_from 2026-01-10, open-ended -> contains 2026-01-11.
            self.assertIn("node_bbq_001.md", {h["node"] for h in nodes})


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_temporal_semantic
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      BLOCKER fix — semantic temporal-query fail-soft (no SystemExit on missing index)
# Layer:      test (pytest)
# Role:       tests for samia.core.temporal.query(semantic=) — no-index branch fails soft with a trailing note (never SystemExit), non-semantic scan unaffected, real-index branch returns ranked + temporally filtered hits
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.temporal, samia.core.vector
# Exposes:    — (test module)
# Lines:      148
# ------------------------------------------------------------------------------
