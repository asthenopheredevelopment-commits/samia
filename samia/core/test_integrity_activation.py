"""Tests for the integrity activation-wiring — FEAT-2026-06-07 granular-recall-repaired-decay.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the GRANULAR env-flag activation wiring that connects the
             already-built P1-P3 content-integrity mechanism to the daemon call sites:
               (1) DEFAULT-OFF / byte-identical — with ALL FOUR flags unset,
                   tier.decay_tick does NOT erode/freeze and mcp_server.memory_search does
                   NOT recall-repair (exactly the prior inert behavior);
               (2) REPAIR flag — ASTHENOS_INTEGRITY_REPAIR_ENABLED engages recall-repair
                   in memory_search (via integrity.repair_enabled());
               (3) DECAY flag — ASTHENOS_INTEGRITY_DECAY_ENABLED makes decay_tick erode;
               (4) FREEZE flag — ASTHENOS_INTEGRITY_FREEZE_ENABLED freezes a below-floor
                   node during the sweep;
               (5) INDEPENDENCE — decay-without-freeze erodes but does not freeze; repair-
                   without-decay heals but does not erode;
               (6) EXPLICIT OVERRIDE — an explicit param still overrides the env default;
               (7) RELEVANCE decay is UNCHANGED under every flag combination.
    Depends: samia.core.integrity, samia.core.tier, samia.core.mcp_server,
             samia.core.frontmatter, unittest, tempfile, pathlib, os, unittest.mock.
             monkeypatches os.environ; uses ONLY tempfile dirs; NEVER touches the live
             memory tree / ~/.local/share / the real global edges.db.

Layer 2 (What / Why):
    What: Verifies that each integrity activation flag wires its arm independently, that an
          all-unset environment is byte-identical to pre-wiring inert behavior, and that an
          explicit param overrides the flag — without ever activating anything by default.
    Why:  The mechanism was complete but inert because the daemon call sites defaulted the
          enabling FUNCTION PARAMS off and never passed them. This wiring resolves those
          params from the env flags; these tests pin the four flags as independent switches
          and lock the default-off guarantee. Produce-only: no daemon, no model, no live edits.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import integrity as I
from samia.core import tier as T
from samia.core import frontmatter as _fm
from samia.core import vector as vi


# Long body so a slow erosion rate drops a small but countable number of chars per pass.
_BODY = ("The operator's model of forgetting is granular and slow at the "
         "individual character level. A node erodes a little at a time, and "
         "recall repairs it faithfully from the pristine anchor.") * 3


def _write_node(memory_dir: Path, name: str, body: str, *,
                tier_label: str = "warm", last_access: str = "2026-01-01",
                salience: float | None = None, relevance: float = 0.5,
                with_anchor: bool = False) -> Path:
    """Write nodes/<name>.md; optionally pre-seed its pristine anchor."""
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
        f"address: {name}",
    ]
    if salience is not None:
        lines.append(f"salience: {salience}")
    lines += ["---", body, ""]
    p = nodes / f"{name}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    if with_anchor:
        fm, _o, b = _fm.read_node(p)
        I.write_anchor(memory_dir, name, b, fm)
    return p


def _erode_below_floor(memory_dir: Path, name: str) -> float:
    """Genuinely erode a node's BODY until its integrity is BELOW the floor."""
    p = memory_dir / "nodes" / f"{name}.md"
    fm, order, body = _fm.read_node(p)
    for _ in range(600):
        body, integ, n = I.erode(memory_dir, name, fm, order, body,
                                 days_since_recall=999, tier="cold", salience=0.0)
        if integ < I.INTEGRITY_FLOOR - 0.01:
            break
    _fm.write_node(p, fm, order, body)
    return I.get_integrity(_fm.read_node(p)[0])


def _clear_integrity_env(monkeyenv: dict) -> None:
    """Remove all four integrity flags from a dict-copy of os.environ."""
    for k in (I.INTEGRITY_REPAIR_ENABLED_ENV, I.INTEGRITY_DECAY_ENABLED_ENV,
              I.INTEGRITY_FREEZE_ENABLED_ENV, I.INTEGRITY_GENERATIVE_ENABLED_ENV):
        monkeyenv.pop(k, None)


class _EnvCase(unittest.TestCase):
    """Base: snapshot + restore the four integrity flags around each test (monkeypatch)."""

    def setUp(self):
        env = dict(os.environ)
        _clear_integrity_env(env)
        self._env_patch = mock.patch.dict(os.environ, env, clear=True)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)


# ---------------------------------------------------------------------------
# The env-reader helpers themselves (mirror generative_enabled()).
# ---------------------------------------------------------------------------


class TestFlagReaders(_EnvCase):
    def test_all_readers_default_off(self):
        self.assertFalse(I.repair_enabled())
        self.assertFalse(I.decay_enabled())
        self.assertFalse(I.freeze_enabled())

    def test_repair_reader_on(self):
        os.environ[I.INTEGRITY_REPAIR_ENABLED_ENV] = "1"
        self.assertTrue(I.repair_enabled())
        self.assertFalse(I.decay_enabled())
        self.assertFalse(I.freeze_enabled())

    def test_decay_reader_on(self):
        os.environ[I.INTEGRITY_DECAY_ENABLED_ENV] = "1"
        self.assertTrue(I.decay_enabled())
        self.assertFalse(I.repair_enabled())
        self.assertFalse(I.freeze_enabled())

    def test_freeze_reader_on(self):
        os.environ[I.INTEGRITY_FREEZE_ENABLED_ENV] = "1"
        self.assertTrue(I.freeze_enabled())
        self.assertFalse(I.repair_enabled())
        self.assertFalse(I.decay_enabled())

    def test_repair_reader_reuses_same_flag_as_p2_subscriber(self):
        # The P2 consolidation-repair subscriber must read the SAME flag (one switch).
        from samia.runtime import rem_subscribers as R
        self.assertEqual(I.INTEGRITY_REPAIR_ENABLED_ENV, R.INTEGRITY_REPAIR_ENABLED_ENV)
        os.environ[I.INTEGRITY_REPAIR_ENABLED_ENV] = "1"
        self.assertTrue(I.repair_enabled())
        self.assertTrue(R._integrity_repair_enabled())


# ---------------------------------------------------------------------------
# (1) DEFAULT-OFF: all four unset -> decay_tick inert, memory_search no-repair.
# ---------------------------------------------------------------------------


class TestDefaultOffByteIdentical(_EnvCase):
    def test_decay_tick_does_not_erode_or_freeze_with_no_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            before = (md / "nodes" / "n.md").read_text(encoding="utf-8")
            res = T.decay_tick(md, force=True)  # no flags, no explicit args
            self.assertTrue(res["fired"])
            self.assertEqual(res["n_integrity_eroded"], 0)
            # The body is untouched by the integrity axis (relevance frontmatter may move,
            # but no erosion sentinel is introduced into the served body).
            after = (md / "nodes" / "n.md").read_text(encoding="utf-8")
            self.assertNotIn(I.EROSION_SENTINEL, after)
            self.assertIn("granular", after)  # body content intact

    def test_memory_search_does_not_repair_with_no_flags(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n1", "Alpha alpha", with_anchor=True)
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query",
                                   lambda memory_dir, q, top_k=8: [dict(h) for h in main_hits]), \
                    mock.patch.object(I, "recall_repair") as rr:
                mcp.memory_search(md, "alpha", top_k=5,
                                  record_coactivation=False,
                                  include_coactivation_neighbors=False,
                                  include_engram=False, include_ring=False)
                rr.assert_not_called()


# ---------------------------------------------------------------------------
# (2) REPAIR flag -> recall-repair engages in memory_search.
# ---------------------------------------------------------------------------


class TestRepairFlagWiring(_EnvCase):
    def _run_search(self, md, expect_node="n1.md"):
        from samia.core import mcp_server as mcp
        main_hits = [{"score": 0.7, "node": expect_node, "title": "Alpha"}]
        calls = []
        with mock.patch.object(vi, "query",
                               lambda memory_dir, q, top_k=8: [dict(h) for h in main_hits]), \
                mock.patch.object(I, "recall_repair",
                                  side_effect=lambda m, n, **k: calls.append(n)):
            mcp.memory_search(md, "alpha", top_k=5,
                              record_coactivation=False,
                              include_coactivation_neighbors=False,
                              include_engram=False, include_ring=False)
        return calls

    def test_repair_flag_engages_recall_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n1", "Alpha alpha", with_anchor=True)
            os.environ[I.INTEGRITY_REPAIR_ENABLED_ENV] = "1"
            calls = self._run_search(md)
            self.assertIn("n1.md", calls)

    def test_explicit_false_overrides_repair_flag_on(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", with_anchor=True)
            os.environ[I.INTEGRITY_REPAIR_ENABLED_ENV] = "1"  # flag ON
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query",
                                   lambda memory_dir, q, top_k=8: [dict(h) for h in main_hits]), \
                    mock.patch.object(I, "recall_repair") as rr:
                mcp.memory_search(md, "alpha", top_k=5,
                                  record_coactivation=False,
                                  include_coactivation_neighbors=False,
                                  include_engram=False, include_ring=False,
                                  repair_integrity=False)  # explicit OFF overrides flag
                rr.assert_not_called()

    def test_explicit_true_overrides_repair_flag_off(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n1", "Alpha", with_anchor=True)
            # flag UNSET (default off)
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "Alpha"}]
            with mock.patch.object(vi, "query",
                                   lambda memory_dir, q, top_k=8: [dict(h) for h in main_hits]), \
                    mock.patch.object(I, "recall_repair") as rr:
                mcp.memory_search(md, "alpha", top_k=5,
                                  record_coactivation=False,
                                  include_coactivation_neighbors=False,
                                  include_engram=False, include_ring=False,
                                  repair_integrity=True)  # explicit ON overrides unset flag
                rr.assert_called()


# ---------------------------------------------------------------------------
# (3) DECAY flag -> decay_tick erodes.
# ---------------------------------------------------------------------------


class TestDecayFlagWiring(_EnvCase):
    def test_decay_flag_makes_decay_tick_erode(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            os.environ[I.INTEGRITY_DECAY_ENABLED_ENV] = "1"
            res = T.decay_tick(md, force=True)
            self.assertTrue(res["fired"])
            self.assertGreaterEqual(res["n_integrity_eroded"], 1)
            after = (md / "nodes" / "n.md").read_text(encoding="utf-8")
            self.assertIn(I.EROSION_SENTINEL, after)  # body genuinely eroded

    def test_explicit_false_overrides_decay_flag_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            os.environ[I.INTEGRITY_DECAY_ENABLED_ENV] = "1"  # flag ON
            res = T.decay_tick(md, force=True, erode_integrity=False)  # explicit OFF
            self.assertEqual(res["n_integrity_eroded"], 0)
            after = (md / "nodes" / "n.md").read_text(encoding="utf-8")
            self.assertNotIn(I.EROSION_SENTINEL, after)


# ---------------------------------------------------------------------------
# (4) FREEZE flag -> a below-floor node freezes during the sweep.
# ---------------------------------------------------------------------------


class TestFreezeFlagWiring(_EnvCase):
    def test_freeze_flag_freezes_below_floor_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_below_floor(md, "n")
            node_path = md / "nodes" / "n.md"
            self.assertTrue(node_path.exists())
            # Both DECAY + FREEZE on: the sweep runs (erodes) and the floor freezes.
            os.environ[I.INTEGRITY_DECAY_ENABLED_ENV] = "1"
            os.environ[I.INTEGRITY_FREEZE_ENABLED_ENV] = "1"
            T.decay_tick(md, force=True)
            # ia.freeze removes the node from nodes/ (reversible demotion, not deletion).
            self.assertFalse(node_path.exists())
            self.assertTrue((md / "archive").exists())

    def test_freeze_off_below_floor_node_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_below_floor(md, "n")
            node_path = md / "nodes" / "n.md"
            # DECAY on but FREEZE OFF: it keeps eroding, never freezes.
            os.environ[I.INTEGRITY_DECAY_ENABLED_ENV] = "1"
            T.decay_tick(md, force=True)
            self.assertTrue(node_path.exists())  # still resident (not frozen)


# ---------------------------------------------------------------------------
# (5) INDEPENDENCE — the four flags do not bleed into each other.
# ---------------------------------------------------------------------------


class TestFlagIndependence(_EnvCase):
    def test_decay_without_freeze_erodes_but_does_not_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_below_floor(md, "n")
            node_path = md / "nodes" / "n.md"
            os.environ[I.INTEGRITY_DECAY_ENABLED_ENV] = "1"  # decay only
            res = T.decay_tick(md, force=True)
            self.assertGreaterEqual(res["n_integrity_eroded"], 1)  # eroded
            self.assertTrue(node_path.exists())                    # NOT frozen

    def test_repair_without_decay_heals_but_does_not_erode(self):
        from samia.core import mcp_server as mcp
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n1", _BODY, with_anchor=True)
            # Genuinely erode the served body first (directly, not via the gated tick).
            p = md / "nodes" / "n1.md"
            fm, order, body = _fm.read_node(p)
            for _ in range(5):
                body, integ, n = I.erode(md, "n1", fm, order, body,
                                         days_since_recall=99, tier="cold", salience=0.0)
            _fm.write_node(p, fm, order, body)
            eroded_integrity = I.get_integrity(_fm.read_node(p)[0])
            self.assertLess(eroded_integrity, I.INTEGRITY_FULL)

            # REPAIR flag on, DECAY flag OFF: a recall heals (no further erosion).
            os.environ[I.INTEGRITY_REPAIR_ENABLED_ENV] = "1"
            main_hits = [{"score": 0.7, "node": "n1.md", "title": "n1"}]
            with mock.patch.object(vi, "query",
                                   lambda memory_dir, q, top_k=8: [dict(h) for h in main_hits]):
                mcp.memory_search(md, "alpha", top_k=5,
                                  record_coactivation=False,
                                  include_coactivation_neighbors=False,
                                  include_engram=False, include_ring=False)
            healed_integrity = I.get_integrity(_fm.read_node(p)[0])
            self.assertGreater(healed_integrity, eroded_integrity)  # healed
            # The served body is restored byte-exact from the anchor (no sentinels left).
            self.assertNotIn(I.EROSION_SENTINEL,
                             (md / "nodes" / "n1.md").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (7) RELEVANCE decay is UNCHANGED under every flag combination.
# ---------------------------------------------------------------------------


class TestRelevanceDecayUnaffected(_EnvCase):
    def _relevance_after_tick(self, md, name) -> float:
        return float(_fm.read_node(md / "nodes" / f"{name}.md")[0].get("relevance"))

    def test_relevance_decay_identical_across_flag_combos(self):
        # Run the SAME corpus + same `today` under each flag combo; relevance must match
        # the all-flags-off baseline exactly (the integrity axis never touches relevance).
        combos = [
            {},
            {I.INTEGRITY_DECAY_ENABLED_ENV: "1"},
            {I.INTEGRITY_REPAIR_ENABLED_ENV: "1"},
            {I.INTEGRITY_DECAY_ENABLED_ENV: "1", I.INTEGRITY_FREEZE_ENABLED_ENV: "1"},
            {I.INTEGRITY_DECAY_ENABLED_ENV: "1", I.INTEGRITY_REPAIR_ENABLED_ENV: "1",
             I.INTEGRITY_FREEZE_ENABLED_ENV: "1"},
        ]
        baseline = None
        for combo in combos:
            with tempfile.TemporaryDirectory() as tmp:
                md = Path(tmp)
                # A node that will decay but stays well above the freeze/tier floors so it
                # is not removed under any combo (keeps the relevance value comparable).
                _write_node(md, "n", _BODY, tier_label="warm",
                            last_access="2026-01-01", relevance=0.6, with_anchor=True)
                for k in (I.INTEGRITY_REPAIR_ENABLED_ENV, I.INTEGRITY_DECAY_ENABLED_ENV,
                          I.INTEGRITY_FREEZE_ENABLED_ENV):
                    os.environ.pop(k, None)
                os.environ.update(combo)
                T.decay_tick(md, force=True)
                rel = self._relevance_after_tick(md, "n")
                if baseline is None:
                    baseline = rel
                else:
                    self.assertAlmostEqual(rel, baseline, places=9,
                                           msg=f"relevance drifted under {combo}")


if __name__ == "__main__":
    unittest.main()
