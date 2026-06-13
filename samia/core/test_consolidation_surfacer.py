"""Tests for the consolidation surfacer's atom-mini-chain exclusion (BUG-2026-06-11).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for consolidation.audit_all's exclusion of the
             fact-extract atom mini-chains (fx_* filename / all-type:semantic
             members) from the EPISODIC consolidation surfacer scope.
    Depends: samia.core.consolidation, unittest, tempfile, json, pathlib.

Layer 2 (What / Why):
    What: the surfacer was built for episodic chains but had begun sweeping the
          fx_* atom mini-chains, closing a self-feeding loop (atoms -> fx chains
          -> surfacer -> merge_abstract -> fact re-extraction). These tests pin
          that an fx_ chain (and an all-semantic chain regardless of name) is
          SKIPPED, while a normal episodic chain still surfaces its near-dup pairs.
    Why:  a regression here re-opens the runaway loop. The exclusion is fail-soft
          (an unloadable chain stays surfaceable), so it never silently drops real
          episodic memory.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from samia.core import consolidation


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    return md


def _node(md: Path, name: str, body: str, type_: str) -> str:
    rel = f"nodes/{name}.md"
    fm = f"---\nname: {name}\ntype: {type_}\n---\n{body}\n"
    (md / rel).write_text(fm, encoding="utf-8")
    return rel


def _chain(md: Path, chain_id: str, members: list[tuple[str, str]]) -> None:
    """members: list of (addr, rel_file)."""
    payload = {
        "chain_id": chain_id,
        "members": [{"addr": a, "file": f, "tier": "warm"} for a, f in members],
    }
    (md / "chains" / f"{chain_id}.json").write_text(
        json.dumps(payload), encoding="utf-8")


# Two near-identical bodies so jaccard clears the 0.15 surfacing knee.
_BODY_A = "weights stream through an hdd onboard dram buffer dodging bus bandwidth"
_BODY_B = "weights stream through an hdd onboard dram buffer dodging pcie bandwidth"


class TestSurfacerExcludesAtomChains(unittest.TestCase):
    def test_fx_prefixed_chain_not_surfaced(self):
        """An fx_* mini-chain is excluded by the cheap id-prefix check."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            a = _node(md, "sem_a", _BODY_A, type_="semantic")
            b = _node(md, "sem_b", _BODY_B, type_="semantic")
            _chain(md, "fx_sem_a.md+sem_b", [("a1", a), ("b1", b)])
            findings = consolidation.audit_all(md)
            self.assertEqual(findings, [])

    def test_all_semantic_chain_not_surfaced_even_without_prefix(self):
        """Belt check: an all-type:semantic chain is excluded even without fx_."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            a = _node(md, "sem_c", _BODY_A, type_="semantic")
            b = _node(md, "sem_d", _BODY_B, type_="semantic")
            _chain(md, "atomchain_noprefix", [("a1", a), ("b1", b)])
            findings = consolidation.audit_all(md)
            self.assertEqual(findings, [])

    def test_episodic_chain_still_surfaced(self):
        """A normal episodic chain still surfaces its near-dup pairs."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            a = _node(md, "off_a", _BODY_A, type_="session_offload")
            b = _node(md, "off_b", _BODY_B, type_="session_offload")
            _chain(md, "session_chain", [("a1", a), ("b1", b)])
            findings = consolidation.audit_all(md)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["chain"], "session_chain")

    def test_mixed_tree_only_episodic_surfaces(self):
        """With both an fx_ chain and an episodic chain present, only the
        episodic one's pair is surfaced."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            sa = _node(md, "sem_e", _BODY_A, type_="semantic")
            sb = _node(md, "sem_f", _BODY_B, type_="semantic")
            _chain(md, "fx_sem_e.md+sem_f", [("a1", sa), ("b1", sb)])
            ea = _node(md, "off_c", _BODY_A, type_="session_offload")
            eb = _node(md, "off_d", _BODY_B, type_="session_offload")
            _chain(md, "session_chain2", [("a1", ea), ("b1", eb)])
            findings = consolidation.audit_all(md)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["chain"], "session_chain2")


if __name__ == "__main__":
    unittest.main()
