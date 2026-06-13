"""Tests for samia.core.integrity P2 — FEAT-2026-06-07 granular-recall-repaired-decay.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for Phase P2 of the SECOND (content-fidelity) decay axis:
             (1) ANCHOR CAPTURE ON WRITE — a freshly-written node gains a PRISTINE anchor
                 (so it can thereafter erode + be faithfully repaired), and the anchor
                 captures the pristine body (NEVER the eroded served body);
             (2) CONSOLIDATION repair — consolidation_repair_pass PARTIALLY raises the
                 integrity of an eroded node (strength<1, strictly less than a full
                 recall_repair), anchor-first, cursor-tracked;
             (3) RECONCILIATION repair — a contradiction/merge READ partially repairs the
                 read node's integrity (partial_repair, anchor-first);
             (4) SALIENCE modulation — high-salience erodes slower via the LIVE salience
                 signal (live_salience reads bio.compute_salience);
             (5) the relevance/lifecycle decay is STILL untouched;
             (6) all P2 repair paths are gated/inert by default (REM + enable flags);
             (7) NO generative repair (anchor-first only) — no anchor => no-op.
    Depends: samia.core.integrity, samia.core.frontmatter, samia.core.merge_consumer,
             samia.runtime.rem_subscribers, samia.runtime.contradiction, unittest,
             tempfile, pathlib, unittest.mock (stdlib).

Layer 2 (What / Why):
    What: Verifies P2 — the engagement gate (anchor on write) + the two PARTIAL repair
          triggers (consolidation/reconciliation) + the live-salience erosion coupling —
          all anchor-first, all inert by default, never touching the relevance axis.
    Why:  The operator's model — recall heals fully, sleep + reconciliation heal partially
          what they touch; high-salience memories are durable. All tests use tempfile dirs
          and mock the embedder/inference; they NEVER touch the live memory tree /
          ~/.local/share / the global edges.db.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import integrity as I
from samia.core import frontmatter as _fm


# Long body so a slow rate erodes a small but countable number of chars.
_BODY = ("The operator's model of forgetting is granular and slow at the "
         "individual character level. A node erodes a little at a time, and "
         "recall repairs it faithfully from the pristine anchor.") * 3


def _write_node(memory_dir: Path, name: str, body: str, *,
                tier_label: str = "warm", last_access: str = "2026-05-01",
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


def _erode_some(memory_dir: Path, name: str, passes: int = 5,
                tier: str = "cold") -> float:
    """Erode + persist `passes` times; return the final on-disk integrity."""
    p = memory_dir / "nodes" / f"{name}.md"
    fm, order, body = _fm.read_node(p)
    for _ in range(passes):
        body, _i, _n = I.erode(memory_dir, name, fm, order, body,
                               days_since_recall=60, tier=tier)
    _fm.write_node(p, fm, order, body)
    return I.get_integrity(_fm.read_node(p)[0])


# ---------------------------------------------------------------------------
# (1) Anchor capture on the GENUINE write path
# ---------------------------------------------------------------------------


class TestAnchorCaptureOnWrite(unittest.TestCase):
    def test_capture_on_write_creates_anchor_then_node_can_erode(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold",
                            last_access="2026-01-01", with_anchor=False)
            self.assertFalse(I.has_anchor(md, "n"))  # P1 did NOT auto-capture
            fm, order, body = _fm.read_node(p)
            res = I.capture_on_write(md, "n", {"name": "n"}, body)
            self.assertTrue(res["captured"])
            self.assertTrue(I.has_anchor(md, "n"))
            # Now the node can erode (erosion is anchor-gated).
            new_body, new_int, n = I.erode(md, "n", fm, order, body,
                                           days_since_recall=60, tier="cold")
            self.assertGreater(n, 0)
            self.assertLess(new_int, 1.0)

    def test_capture_on_write_refreshes_on_genuine_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, with_anchor=False)
            I.capture_on_write(md, "n", {"name": "n"}, "first pristine body")
            self.assertEqual(I.read_anchor(md, "n"), "first pristine body")
            # A genuine re-write refreshes the anchor to the new pristine body.
            res = I.capture_on_write(md, "n", {"name": "n"}, "second pristine body")
            self.assertTrue(res["refreshed"])
            self.assertEqual(I.read_anchor(md, "n"), "second pristine body")

    def test_anchor_never_captured_from_eroded_served_body(self):
        # CRITICAL SAFETY: the erosion sweep must NEVER call ensure_anchor /
        # capture_on_write, or the anchor would be clobbered with degraded content.
        # We assert the anchor stays pristine across many erosion passes.
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            anchor_before = I.read_anchor(md, "n")
            # Run the erosion sweep (which persists the eroded body) several times.
            for _ in range(5):
                I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            anchor_after = I.read_anchor(md, "n")
            self.assertEqual(anchor_before, anchor_after)
            self.assertNotIn(I.EROSION_SENTINEL, anchor_after)
            self.assertEqual(anchor_after.rstrip(), _BODY.rstrip())

    def test_memory_write_node_captures_anchor(self):
        # The genuine write seam (memory_write_node) auto-captures the anchor, and the
        # captured anchor is the PRISTINE just-written body. Mock the capture-hook
        # internals (ring/salience/supersede) so the test never needs the embedder/db.
        from samia.core import mcp_server as _mcp
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            (md / "nodes").mkdir(parents=True, exist_ok=True)
            with mock.patch.object(_mcp, "_register_ring_and_salience",
                                   return_value={}), \
                 mock.patch.object(_mcp, "_online_supersede",
                                   return_value={"superseded": [], "recorded": []}):
                out = _mcp.memory_write_node(
                    md, name="genuine", title="Genuine Node",
                    description="real content", body="pristine operator content")
            self.assertEqual(out["written"], "genuine.md")
            self.assertTrue(out.get("anchor", {}).get("captured"))
            self.assertTrue(I.has_anchor(md, "genuine.md"))
            self.assertEqual(I.read_anchor(md, "genuine.md"),
                             "pristine operator content")


# ---------------------------------------------------------------------------
# (2) Consolidation repair (PARTIAL, anchor-first, < full recall)
# ---------------------------------------------------------------------------


class TestConsolidationRepair(unittest.TestCase):
    def test_partial_repair_is_partial_and_less_than_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            eroded = _erode_some(md, "n", passes=6)
            self.assertLess(eroded, 1.0)
            res = I.partial_repair(md, "n", trigger="consolidation")
            self.assertTrue(res["repaired"])
            self.assertTrue(res["partial"])
            self.assertTrue(res["anchor_faithful"])
            # PARTIAL: above the eroded value, but strictly below FULL (not a recall).
            self.assertGreater(res["new_integrity"], eroded)
            self.assertLess(res["new_integrity"], 1.0)
            # And strictly less than a FULL recall_repair would give from the same point.
            full = I.reconsolidate_integrity(eroded, I.RECALL_REPAIR_STRENGTH)
            self.assertLess(res["new_integrity"], full)

    def test_consolidation_repair_pass_partially_heals_eroded_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            eroded = _erode_some(md, "n", passes=6)
            res = I.consolidation_repair_pass(md)
            self.assertEqual(res["repaired"], 1)
            self.assertEqual(res["touched"], 1)
            healed = I.get_integrity(_fm.read_node(md / "nodes" / "n.md")[0])
            self.assertGreater(healed, eroded)   # partially healed
            self.assertLess(healed, 1.0)         # but NOT a full restore

    def test_consolidation_repair_pass_skips_pristine_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "pristine", _BODY, with_anchor=True)
            res = I.consolidation_repair_pass(md)
            self.assertEqual(res["touched"], 0)  # nothing eroded -> nothing to heal
            self.assertEqual(res["repaired"], 0)

    def test_consolidation_repair_pass_cursor_advances_and_wraps(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            for i in range(3):
                _write_node(md, f"n{i}", _BODY, tier_label="cold",
                            last_access="2026-01-01", with_anchor=True)
            # Budget 2 -> first slice processes 2, cursor at 2, work remains.
            r1 = I.consolidation_repair_pass(md, budget=2, cursor=0)
            self.assertEqual(r1["processed"], 2)
            self.assertEqual(r1["cursor"], 2)
            self.assertTrue(r1["work_remaining"])
            # Next slice processes the last one, cursor wraps to 0, no work remaining.
            r2 = I.consolidation_repair_pass(md, budget=2, cursor=r1["cursor"])
            self.assertEqual(r2["processed"], 1)
            self.assertEqual(r2["cursor"], 0)
            self.assertFalse(r2["work_remaining"])

    def test_consolidation_repair_no_anchor_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, with_anchor=False)
            res = I.partial_repair(md, "n", trigger="consolidation")
            self.assertFalse(res["repaired"])
            self.assertEqual(res["skipped"], "no-anchor")


# ---------------------------------------------------------------------------
# (3) Reconciliation repair via the contradiction/merge read seams
# ---------------------------------------------------------------------------


class TestReconciliationRepair(unittest.TestCase):
    def test_merge_dup_partially_repairs_surviving_node(self):
        from samia.core import merge_consumer as _mc
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            # Survivor (richer: higher access_count) + loser duplicate.
            ps = _write_node(md, "winner", _BODY, with_anchor=True)
            fm, order, body = _fm.read_node(ps)
            fm["access_count"] = 9
            _fm.write_node(ps, fm, order, body)
            I.write_anchor(md, "winner", _BODY + "\n", fm)
            _write_node(md, "loser", _BODY, with_anchor=False)
            eroded = _erode_some(md, "winner", passes=6, tier="cold")
            self.assertLess(eroded, 1.0)
            # Enable the merge feature; mock the supersede + provenance so no db/embedder.
            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_TIER2_MERGE_ENABLED": "1"}), \
                 mock.patch.object(_mc._ia, "forget_node",
                                   return_value={"superseded_archive": None}), \
                 mock.patch.object(_mc, "_add_provenance_edge",
                                   return_value={"written": False}):
                rec = _mc.merge_dup(md, "winner", "loser")
            self.assertEqual(rec["survivor"], "winner")
            healed = I.get_integrity(_fm.read_node(md / "nodes" / "winner.md")[0])
            self.assertGreater(healed, eroded)  # reconciliation partially healed it
            self.assertLess(healed, 1.0)        # PARTIAL (not a full recall)

    def test_reconciliation_repair_is_partial_trigger_tagged(self):
        # A reconciliation read uses partial_repair tagged trigger="reconciliation".
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            eroded = _erode_some(md, "n", passes=6)
            res = I.partial_repair(md, "n", trigger="reconciliation")
            self.assertTrue(res["repaired"])
            self.assertEqual(res["trigger"], "reconciliation")
            self.assertTrue(res["partial"])
            self.assertGreater(res["new_integrity"], eroded)
            self.assertLess(res["new_integrity"], 1.0)
            # The event log records the reconciliation trigger.
            log = md / "biomimetic" / "integrity_reconsolidation_log.jsonl"
            self.assertIn("reconciliation", log.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (4) Salience modulation via the LIVE signal
# ---------------------------------------------------------------------------


class TestLiveSalienceModulation(unittest.TestCase):
    def test_live_salience_reads_compute_salience(self):
        # live_salience must consult bio.compute_salience (the LIVE signal), not just
        # the static field. We patch compute_salience and assert it is used.
        from samia.core import bio as _bio
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, with_anchor=True, salience=0.1)
            with mock.patch.object(_bio, "compute_salience",
                                   return_value=0.77) as m:
                sal = I.live_salience(md, "n", {"salience": 0.1})
            m.assert_called_once()
            # The LIVE value (0.77) is used, NOT the stale static field (0.1).
            self.assertAlmostEqual(sal, 0.77, places=6)

    def test_live_salience_falls_back_to_field_when_signal_unavailable(self):
        from samia.core import bio as _bio
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            with mock.patch.object(_bio, "compute_salience",
                                   side_effect=RuntimeError("no embedder")):
                sal = I.live_salience(md, "n", {"salience": 0.42})
            self.assertAlmostEqual(sal, 0.42, places=6)  # graceful fallback

    def test_high_salience_erodes_slower_via_live_signal(self):
        # The erosion sweep must couple to the LIVE salience signal: with the live
        # signal returning HIGH for one node and LOW for another, the high node erodes
        # slower over the same number of sweeps.
        from samia.core import bio as _bio
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "hi", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _write_node(md, "lo", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)

            def _live(memory_dir, node, content=None, write=True, **kw):
                return 1.0 if str(node).startswith("hi") else 0.0

            with mock.patch.object(_bio, "compute_salience", side_effect=_live):
                for _ in range(3):
                    I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            hi = I.get_integrity(_fm.read_node(md / "nodes" / "hi.md")[0])
            lo = I.get_integrity(_fm.read_node(md / "nodes" / "lo.md")[0])
            self.assertLess(lo, 1.0)
            self.assertGreater(hi, lo)  # high (live) salience retained MORE integrity


# ---------------------------------------------------------------------------
# (5) The relevance/lifecycle decay is STILL untouched by P2
# ---------------------------------------------------------------------------


class TestRelevanceAxisStillUntouched(unittest.TestCase):
    def test_step_relevance_unchanged_by_p2(self):
        from samia.core import tier
        for days in (0, 3, 7, 8, 30, 365):
            new, reason = tier.step_relevance(0.6, False, days)
            zero, zero_reason = tier.step_relevance(0.6, False, days, salience=0.0)
            self.assertEqual(new, zero)
            self.assertEqual(reason, zero_reason)

    def test_consolidation_repair_does_not_change_relevance_or_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold",
                            last_access="2026-01-01", relevance=0.4, with_anchor=True)
            _erode_some(md, "n", passes=6)
            rel_before = _fm.read_node(p)[0]["relevance"]
            tier_before = _fm.read_node(p)[0]["tier"]
            I.consolidation_repair_pass(md)
            fm, _o, _b = _fm.read_node(p)
            self.assertEqual(fm["relevance"], rel_before)
            self.assertEqual(fm["tier"], tier_before)


# ---------------------------------------------------------------------------
# (6) All P2 repair paths are gated / inert by default
# ---------------------------------------------------------------------------


class TestP2GatedInertByDefault(unittest.TestCase):
    def test_consolidation_repair_subscriber_inert_without_enable_flag(self):
        from samia.runtime import rem_subscribers as _rs
        from samia.runtime import rem_cycle
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_some(md, "n", passes=6)
            rem_cycle.enter_rem(md, "test")
            # Enable flag UNSET -> the subscriber refuses (inert by default).
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(_rs.INTEGRITY_REPAIR_ENABLED_ENV, None)
                out = _rs._sub_integrity_repair(md)
            self.assertFalse(out.get("fired"))
            self.assertEqual(out.get("refused"), "not_enabled")
            # The eroded node was NOT repaired (still below pristine).
            self.assertLess(
                I.get_integrity(_fm.read_node(md / "nodes" / "n.md")[0]), 1.0)

    def test_consolidation_repair_subscriber_refuses_outside_rem(self):
        from samia.runtime import rem_subscribers as _rs
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, with_anchor=True)
            # WAKE (not in REM) -> refused even if enabled.
            with mock.patch.dict(os.environ,
                                 {_rs.INTEGRITY_REPAIR_ENABLED_ENV: "1"}):
                out = _rs._sub_integrity_repair(md)
            self.assertEqual(out.get("refused"), "not_in_rem")

    def test_consolidation_repair_subscriber_fires_when_enabled_in_rem(self):
        from samia.runtime import rem_subscribers as _rs
        from samia.runtime import rem_cycle
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            eroded = _erode_some(md, "n", passes=6)
            rem_cycle.enter_rem(md, "test")
            with mock.patch.dict(os.environ,
                                 {_rs.INTEGRITY_REPAIR_ENABLED_ENV: "1"}):
                out = _rs._sub_integrity_repair(md)
            self.assertTrue(out.get("fired"))
            healed = I.get_integrity(_fm.read_node(md / "nodes" / "n.md")[0])
            self.assertGreater(healed, eroded)

    def test_due_condition_inert_without_enable_flag(self):
        from samia.runtime import rem_subscribers as _rs
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, with_anchor=True)
            os.environ.pop(_rs.INTEGRITY_REPAIR_ENABLED_ENV, None)
            self.assertFalse(_rs._due_integrity_repair(md))
            with mock.patch.dict(os.environ,
                                 {_rs.INTEGRITY_REPAIR_ENABLED_ENV: "1"}):
                self.assertTrue(_rs._due_integrity_repair(md))


# ---------------------------------------------------------------------------
# (7) No generative repair — anchor-first only
# ---------------------------------------------------------------------------


class TestNoGenerativeRepair(unittest.TestCase):
    def test_partial_repair_never_guesses_without_anchor(self):
        # P2 is anchor-first ONLY — a node with no anchor is a no-op (no generative
        # reconstruction; that is P3).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, with_anchor=False)
            res = I.partial_repair(md, "n", trigger="reconciliation")
            self.assertFalse(res["repaired"])
            self.assertEqual(res["skipped"], "no-anchor")

    def test_partial_repair_restores_from_anchor_byte_exact_content(self):
        # The repaired served body is byte-exact to the ANCHOR (faithful, not a guess).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            anchor = I.read_anchor(md, "n")
            _erode_some(md, "n", passes=6)
            I.partial_repair(md, "n", trigger="consolidation")
            _f, _o, restored = _fm.read_node(md / "nodes" / "n.md")
            self.assertNotIn(I.EROSION_SENTINEL, restored)
            self.assertEqual(restored.rstrip(), anchor.rstrip())


if __name__ == "__main__":
    unittest.main()
