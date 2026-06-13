"""Tests for samia.core.tier — FEAT-2026-06-07 Tier-1 P5 salience-aware decay (D6 ii).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the P5 salience-aware relevance decay: step_relevance's
             salience-DAMPENED decrement (a high-salience node decays SLOWER than an
             equal-age low-salience node), decay_pass's FREEZE/eviction EXEMPTION (a
             node with salience >= SALIENCE_FREEZE_EXEMPT is NOT auto-frozen while an
             equal low-salience node IS), and the BACKWARD-COMPAT guarantee (a node
             with no salience field decays + auto-freezes EXACTLY as before).
    Depends: samia.core.tier, unittest, tempfile, pathlib (stdlib).

Layer 2 (What / Why):
    What: Verifies the EXISTING relevance/lifecycle decay is now salience-aware
          without changing the base days_since_access × grade behavior for non-salient
          nodes. The dampening and the exemption are the two D6-ii effects; they are
          DISTINCT from the separate granular content-integrity decay proposal (not
          tested here — P5 only modulates the existing relevance decay).
    Why:  D6 effect (ii) — high-salience memories must persist through the forgetting
          curve that would reclaim a trivial low-frequency node, but salience-0 nodes
          must behave byte-identically to the pre-P5 function (no regression). All tests
          use tempfile dirs and never touch the live memory tree.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import tier
from samia.core import web_store as _ws


def _write_decaying_node(memory_dir: Path, name: str, *, relevance: float,
                         last_access: str, salience: float | None = None,
                         tier_label: str = "cold",
                         grade: str = tier.DEFAULT_GRADE) -> Path:
    """Write a nodes/<name>.md with the fields decay_pass reads.

    A salience of None omits the field entirely (the backward-compat case).
    """
    nodes = memory_dir / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        "description: test node",
        "type: project",
        "chains: []",
        f"last_access: {last_access}",
        "access_count: 0",
        f"relevance: {relevance}",
        f"tier: {tier_label}",
        f"material_grade: {grade}",
    ]
    if salience is not None:
        lines.append(f"salience: {salience}")
    lines += ["---", f"{name} body", ""]
    p = nodes / f"{name}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


class TestSalienceDampenedDecay(unittest.TestCase):
    def test_high_salience_decays_slower_than_low(self):
        # Same old_rel, same age, same grade — only salience differs. The high-
        # salience node must lose LESS relevance over the same step (D6 ii).
        old_rel = 0.4
        days = 30  # stale regime (pull toward 0)
        low, _ = tier.step_relevance(old_rel, False, days, salience=0.0)
        high, _ = tier.step_relevance(old_rel, False, days,
                                      salience=1.0)
        # Both decay toward 0 (stale) so both DROP below old_rel...
        self.assertLess(low, old_rel)
        self.assertLess(high, old_rel)
        # ...but the high-salience node retains MORE relevance (decays slower).
        self.assertGreater(high, low)

    def test_salience_zero_is_byte_identical_to_pre_p5(self):
        # The explicit default (salience omitted) and salience=0.0 must equal the
        # legacy 4-arg behavior exactly — the backward-compat contract.
        for days in (0, 3, 7, 8, 30, 365):
            base, base_reason = tier.step_relevance(0.6, False, days)
            zero, zero_reason = tier.step_relevance(0.6, False, days, salience=0.0)
            self.assertEqual(base, zero)
            self.assertEqual(base_reason, zero_reason)

    def test_max_salience_still_decays_eventually(self):
        # Decay-everywhere: salience DAMPENS but never FREEZES the rate; a max-
        # salience node still loses some relevance (the number still moves).
        new, _ = tier.step_relevance(0.4, False, 30, salience=1.0)
        self.assertLess(new, 0.4)


class TestDecayPassSalience(unittest.TestCase):
    def test_high_salience_node_decays_slower_in_decay_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            # Two equal nodes (same relevance, same stale age) — only salience differs.
            _write_decaying_node(md, "low", relevance=0.4,
                                 last_access="2026-05-01", salience=0.0)
            _write_decaying_node(md, "high", relevance=0.4,
                                 last_access="2026-05-01", salience=1.0)
            trans = tier.decay_pass(md / "nodes", dry=False,
                                    today="2026-06-07", auto_freeze=False)
            by = {t["node"]: t for t in trans}
            self.assertIn("low", by)
            self.assertIn("high", by)
            # The high-salience node ends with a HIGHER new_rel (decayed slower).
            self.assertGreater(by["high"]["new_rel"], by["low"]["new_rel"])

    def test_freeze_exemption_keeps_high_salience_resident(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            # Both nodes are deeply stale + low relevance so they cross into frozen.
            _write_decaying_node(md, "trivial", relevance=0.02,
                                 last_access="2026-01-01", salience=0.0,
                                 tier_label="cold", grade="waste")
            _write_decaying_node(md, "critical", relevance=0.02,
                                 last_access="2026-01-01",
                                 salience=tier.SALIENCE_FREEZE_EXEMPT,
                                 tier_label="cold", grade="waste")
            # Stub the only seam that would reach the GLOBAL edges.db so the freeze
            # archives + unlinks locally without touching live ~/.local/share.
            with mock.patch.object(_ws, "forget_node_edges",
                                   return_value={"edges_deleted": 0}):
                trans = tier.decay_pass(md / "nodes", dry=False,
                                        today="2026-06-07", auto_freeze=True)
            by = {t["node"]: t for t in trans}
            # The trivial node was auto-frozen (its file is gone, archived).
            self.assertTrue(by["trivial"].get("frozen"))
            self.assertFalse((md / "nodes" / "trivial.md").exists())
            # The high-salience node is freeze-EXEMPT: NOT frozen, still resident.
            self.assertTrue(by["critical"].get("freeze_exempt"))
            self.assertFalse(by["critical"].get("frozen"))
            self.assertTrue((md / "nodes" / "critical.md").exists())

    def test_no_salience_node_auto_freezes_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            # No salience field at all (the pre-P5 corpus shape) — must auto-freeze.
            _write_decaying_node(md, "plain", relevance=0.02,
                                 last_access="2026-01-01", salience=None,
                                 tier_label="cold", grade="waste")
            with mock.patch.object(_ws, "forget_node_edges",
                                   return_value={"edges_deleted": 0}):
                trans = tier.decay_pass(md / "nodes", dry=False,
                                        today="2026-06-07", auto_freeze=True)
            by = {t["node"]: t for t in trans}
            self.assertTrue(by["plain"].get("frozen"))
            self.assertFalse(by["plain"].get("freeze_exempt"))
            self.assertFalse((md / "nodes" / "plain.md").exists())


if __name__ == "__main__":
    unittest.main()
