"""Tests for samia.core.integrity P3 — FEAT-2026-06-07 granular-recall-repaired-decay.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for Phase P3 (FINAL) of the SECOND (content-fidelity) decay axis:
             (1) TERMINAL FREEZE-AT-FLOOR (Q5a) — a node eroded below INTEGRITY_FLOOR
                 WITHOUT repair is FROZEN via the existing REVERSIBLE ia.freeze (NOT
                 deleted) and round-trips through ia.thaw (restorable);
             (2) SALIENCE EXEMPTION — a HIGH-salience node below the floor is NOT auto-
                 frozen (salience-exempt, consistent with the relevance path's P5);
             (3) GENERATIVE FALLBACK (Q1c/Q4a) — reconstructs ONLY when NO anchor remains
                 AND the flag is on (+ backend available), marked generative=true /
                 anchor_faithful=false / confabulation_risk=true;
             (4) ANCHOR-FIRST — when an anchor EXISTS, repair is anchor-first (the
                 generative backend is NEVER called);
             (5) SAFE NO-OP — generative disabled / unavailable -> no crash, no
                 fabrication (the same no-anchor no-op as P1/P2);
             (6) the relevance/lifecycle decay is STILL untouched by P3;
             (7) INERT BY DEFAULT — terminal_freeze OFF + generative flag OFF by default.
    Depends: samia.core.integrity, samia.core.frontmatter, samia.core.ia,
             samia.runtime.contradiction, unittest, tempfile, pathlib, os,
             unittest.mock (stdlib). MOCKs the inference backend — NEVER loads a model,
             NEVER touches the live memory tree / ~/.local/share / the global edges.db.

Layer 2 (What / Why):
    What: Verifies P3 — the integrity axis's terminal (freeze-at-floor, reversible, salience-
          exempt) and the last-resort generative reconstruction (no-anchor-only, gated,
          marked). All paths inert by default; the relevance axis is never touched.
    Why:  The operator's model — forgetting = demotion-to-frozen (reversible), and a body
          can be reconstructed only when its anchor is truly gone, only when explicitly
          enabled, and only with honest provenance. Produce-only: no daemon, no real model,
          tempfile dirs only.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import integrity as I
from samia.core import frontmatter as _fm
from samia.core import ia as _ia


# Long body so a slow rate erodes a small but countable number of chars per pass.
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


def _erode_to_just_above_floor(memory_dir: Path, name: str) -> float:
    """Genuinely erode a node's BODY until its integrity sits just above the floor.

    The integrity SCORE is derived from the actual eroded-character fraction of the served
    body (not a free-standing field), so to make the next sweep cross the floor we must
    erode the real body down close to it. Returns the resulting on-disk integrity.
    """
    p = memory_dir / "nodes" / f"{name}.md"
    fm, order, body = _fm.read_node(p)
    # Erode aggressively (many passes, cold + stale + zero salience) until just above floor.
    for _ in range(400):
        body, integ, n = I.erode(memory_dir, name, fm, order, body,
                                 days_since_recall=999, tier="cold", salience=0.0)
        if integ <= I.INTEGRITY_FLOOR + 0.02:
            break
    _fm.write_node(p, fm, order, body)
    return I.get_integrity(_fm.read_node(p)[0])


def _erode_below_floor(memory_dir: Path, name: str) -> float:
    """Genuinely erode a node's BODY until its integrity is already BELOW the floor.

    Used for the salience-exemption test: the node must be below the floor at sweep time so
    the exemption (not "didn't cross yet") is what spares it. Salience-0 erosion in this
    helper drops it; the sweep then runs with HIGH salience (which barely erodes), so the
    node stays below the floor and the exemption is the only reason it is not frozen.
    """
    p = memory_dir / "nodes" / f"{name}.md"
    fm, order, body = _fm.read_node(p)
    for _ in range(600):
        body, integ, n = I.erode(memory_dir, name, fm, order, body,
                                 days_since_recall=999, tier="cold", salience=0.0)
        if integ < I.INTEGRITY_FLOOR - 0.01:
            break
    _fm.write_node(p, fm, order, body)
    return I.get_integrity(_fm.read_node(p)[0])


# ---------------------------------------------------------------------------
# (1) Terminal freeze-at-floor — Q5a, reversible (NOT deletion)
# ---------------------------------------------------------------------------


class TestTerminalFreezeAtFloor(unittest.TestCase):
    def test_node_below_floor_without_repair_is_frozen_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            # Sit it just above the floor so a single erosion pass crosses it.
            _erode_to_just_above_floor(md, "n")
            node_path = md / "nodes" / "n.md"
            self.assertTrue(node_path.exists())

            recs = I.integrity_decay_pass(md, dry=False, today="2026-06-07",
                                          terminal_freeze=True)
            rec = next(r for r in recs if r["node"] == "n")
            self.assertLess(rec["new_integrity"], I.INTEGRITY_FLOOR)
            self.assertTrue(rec.get("terminal_freeze"))

            # FROZEN, not deleted: node file gone, reversible archive present.
            self.assertFalse(node_path.exists())
            frozen = md / "archive" / "n.frozen.json"
            self.assertTrue(frozen.exists())

    def test_terminal_freeze_round_trips_via_thaw(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_to_just_above_floor(md, "n")
            I.integrity_decay_pass(md, dry=False, today="2026-06-07",
                                   terminal_freeze=True)
            node_path = md / "nodes" / "n.md"
            self.assertFalse(node_path.exists())  # frozen
            # ia.thaw brings it back (restorable) — the terminal is reversible.
            _ia.thaw(md, "n")
            self.assertTrue(node_path.exists())
            fm, _o, body = _fm.read_node(node_path)
            self.assertEqual(str(fm.get("tier")), "warm")  # thawed back to warm

    def test_terminal_freeze_logs_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_to_just_above_floor(md, "n")
            I.integrity_decay_pass(md, dry=False, today="2026-06-07",
                                   terminal_freeze=True)
            log = md / "biomimetic" / "integrity_reconsolidation_log.jsonl"
            self.assertTrue(log.exists())
            events = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
            term = [e for e in events if e.get("event") == "terminal_freeze"]
            self.assertEqual(len(term), 1)
            self.assertEqual(term[0]["trigger"], "integrity_floor")
            self.assertTrue(term[0]["frozen"])

    def test_terminal_freeze_off_by_default_no_freeze(self):
        # INERT: terminal_freeze defaults OFF — a node below the floor is NOT frozen.
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_to_just_above_floor(md, "n")
            recs = I.integrity_decay_pass(md, dry=False, today="2026-06-07")  # default OFF
            rec = next(r for r in recs if r["node"] == "n")
            self.assertLess(rec["new_integrity"], I.INTEGRITY_FLOOR)
            self.assertNotIn("terminal_freeze", rec)
            # The node file still exists (no freeze) and was simply eroded further.
            self.assertTrue((md / "nodes" / "n.md").exists())
            self.assertFalse((md / "archive" / "n.frozen.json").exists())

    def test_node_above_floor_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="hot",  # hot erodes slowest
                        last_access="2026-06-06", with_anchor=True)
            # Fresh, hot, high integrity -> a single pass stays well above the floor.
            recs = I.integrity_decay_pass(md, dry=False, today="2026-06-07",
                                          terminal_freeze=True)
            for r in recs:
                if r["node"] == "n":
                    self.assertGreaterEqual(r["new_integrity"], I.INTEGRITY_FLOOR)
                    self.assertNotIn("terminal_freeze", r)
            self.assertTrue((md / "nodes" / "n.md").exists())


# ---------------------------------------------------------------------------
# (2) Salience exemption — consistent with the relevance path (P5)
# ---------------------------------------------------------------------------


class TestSalienceFreezeExemption(unittest.TestCase):
    def test_high_salience_below_floor_is_not_frozen(self):
        from samia.core import bio as _bio
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "vip", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_below_floor(md, "vip")  # already below floor at sweep time
            # LIVE salience well above the exemption threshold.
            high = I.salience_freeze_exempt() + 0.05
            with mock.patch.object(_bio, "compute_salience", return_value=high):
                recs = I.integrity_decay_pass(md, dry=False, today="2026-06-07",
                                              terminal_freeze=True)
            rec = next(r for r in recs if r["node"] == "vip")
            self.assertLess(rec["new_integrity"], I.INTEGRITY_FLOOR)  # below floor
            self.assertTrue(rec.get("freeze_exempt"))                 # but EXEMPT
            self.assertNotIn("terminal_freeze", rec)
            # Stays resident — NOT frozen.
            self.assertTrue((md / "nodes" / "vip.md").exists())
            self.assertFalse((md / "archive" / "vip.frozen.json").exists())

    def test_low_salience_below_floor_is_frozen(self):
        # Counterpart: a LOW-salience node below the floor IS frozen (exemption is the
        # only thing that spares it — the two axes' freeze policy is consistent).
        from samia.core import bio as _bio
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "plain", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_to_just_above_floor(md, "plain")
            with mock.patch.object(_bio, "compute_salience", return_value=0.0):
                recs = I.integrity_decay_pass(md, dry=False, today="2026-06-07",
                                              terminal_freeze=True)
            rec = next(r for r in recs if r["node"] == "plain")
            self.assertTrue(rec.get("terminal_freeze"))
            self.assertFalse((md / "nodes" / "plain.md").exists())  # frozen

    def test_exemption_threshold_tracks_tier_constant(self):
        # The integrity floor's exemption reads the SAME constant the relevance path uses.
        from samia.core import tier
        self.assertEqual(I.salience_freeze_exempt(), tier.SALIENCE_FREEZE_EXEMPT)


# ---------------------------------------------------------------------------
# (3) Generative fallback — no-anchor-only, gated, marked
# ---------------------------------------------------------------------------


class TestGenerativeFallback(unittest.TestCase):
    def _eroded_no_anchor_node(self, md: Path, name: str = "n") -> None:
        """A node that is eroded on disk but has NO anchor (the generative precondition)."""
        # Write WITH an anchor so erosion can run, erode it, then remove the anchor to
        # simulate a deeply-lost / never-snapshotted anchor.
        _write_node(md, name, _BODY, tier_label="cold",
                    last_access="2026-01-01", with_anchor=True)
        p = md / "nodes" / f"{name}.md"
        fm, order, body = _fm.read_node(p)
        for _ in range(6):
            body, _i, _n = I.erode(md, name, fm, order, body,
                                   days_since_recall=60, tier="cold")
        _fm.write_node(p, fm, order, body)
        # Remove the anchor -> no faithful repair source remains.
        I.anchor_path(md, name, fm).unlink()
        self.assertFalse(I.has_anchor(md, name, fm))

    def test_recall_generative_reconstructs_only_when_enabled_and_no_anchor(self):
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            self._eroded_no_anchor_node(md, "n")
            # MOCK the inference backend (NEVER load a real model).
            with mock.patch.object(_contra, "synthesis_enabled", return_value=True), \
                 mock.patch.object(_contra, "synthesize_node",
                                   return_value={"title": "T",
                                                 "body": "reconstructed body"}) as m, \
                 mock.patch.dict(os.environ,
                                 {I.INTEGRITY_GENERATIVE_ENABLED_ENV: "1"}):
                res = I.recall_repair(md, "n")
            self.assertTrue(res["repaired"])
            self.assertTrue(res["generative"])
            self.assertFalse(res["anchor_faithful"])
            self.assertTrue(res["confabulation_risk"])
            m.assert_called_once()
            # The served body is the reconstruction (not pristine, marked uncertain).
            _f, _o, body = _fm.read_node(md / "nodes" / "n.md")
            self.assertEqual(body.strip(), "reconstructed body")

    def test_generative_marked_in_log(self):
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            self._eroded_no_anchor_node(md, "n")
            with mock.patch.object(_contra, "synthesis_enabled", return_value=True), \
                 mock.patch.object(_contra, "synthesize_node",
                                   return_value={"title": "T", "body": "rebuilt"}), \
                 mock.patch.dict(os.environ,
                                 {I.INTEGRITY_GENERATIVE_ENABLED_ENV: "1"}):
                I.partial_repair(md, "n", trigger="consolidation")
            log = md / "biomimetic" / "integrity_reconsolidation_log.jsonl"
            events = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
            gen = [e for e in events if e.get("generative")]
            self.assertEqual(len(gen), 1)
            self.assertFalse(gen[0]["anchor_faithful"])
            self.assertTrue(gen[0]["confabulation_risk"])

    def test_partial_repair_generative_when_no_anchor(self):
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            self._eroded_no_anchor_node(md, "n")
            with mock.patch.object(_contra, "synthesis_enabled", return_value=True), \
                 mock.patch.object(_contra, "synthesize_node",
                                   return_value={"title": "T", "body": "partial rebuilt"}), \
                 mock.patch.dict(os.environ,
                                 {I.INTEGRITY_GENERATIVE_ENABLED_ENV: "1"}):
                res = I.partial_repair(md, "n", trigger="reconciliation")
            self.assertTrue(res["repaired"])
            self.assertTrue(res["generative"])
            self.assertTrue(res["partial"])
            self.assertEqual(res["trigger"], "reconciliation")


# ---------------------------------------------------------------------------
# (4) Anchor-first ALWAYS wins — generative never runs while an anchor exists
# ---------------------------------------------------------------------------


class TestAnchorFirstAlwaysWins(unittest.TestCase):
    def test_recall_with_anchor_never_calls_generative(self):
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            anchor = I.read_anchor(md, "n")
            # Erode the served body but KEEP the anchor.
            p = md / "nodes" / "n.md"
            fm, order, body = _fm.read_node(p)
            for _ in range(6):
                body, _i, _n = I.erode(md, "n", fm, order, body,
                                       days_since_recall=60, tier="cold")
            _fm.write_node(p, fm, order, body)
            # Generative ENABLED — but an anchor exists, so it must NOT be called.
            with mock.patch.object(_contra, "synthesis_enabled", return_value=True), \
                 mock.patch.object(_contra, "synthesize_node") as m, \
                 mock.patch.dict(os.environ,
                                 {I.INTEGRITY_GENERATIVE_ENABLED_ENV: "1"}):
                res = I.recall_repair(md, "n")
            m.assert_not_called()                       # anchor-first wins
            self.assertTrue(res["anchor_faithful"])
            self.assertFalse(res["generative"])
            # Served body restored byte-exact to the anchor (faithful, not a guess).
            _f, _o, restored = _fm.read_node(p)
            self.assertNotIn(I.EROSION_SENTINEL, restored)
            self.assertEqual(restored.rstrip(), anchor.rstrip())

    def test_partial_repair_with_anchor_never_calls_generative(self):
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            p = md / "nodes" / "n.md"
            fm, order, body = _fm.read_node(p)
            for _ in range(6):
                body, _i, _n = I.erode(md, "n", fm, order, body,
                                       days_since_recall=60, tier="cold")
            _fm.write_node(p, fm, order, body)
            with mock.patch.object(_contra, "synthesis_enabled", return_value=True), \
                 mock.patch.object(_contra, "synthesize_node") as m, \
                 mock.patch.dict(os.environ,
                                 {I.INTEGRITY_GENERATIVE_ENABLED_ENV: "1"}):
                res = I.partial_repair(md, "n", trigger="consolidation")
            m.assert_not_called()
            self.assertTrue(res["anchor_faithful"])
            self.assertFalse(res["generative"])


# ---------------------------------------------------------------------------
# (5) Safe no-op — generative disabled / unavailable
# ---------------------------------------------------------------------------


class TestGenerativeSafeNoOp(unittest.TestCase):
    def _eroded_no_anchor_node(self, md: Path, name: str = "n") -> None:
        _write_node(md, name, _BODY, tier_label="cold",
                    last_access="2026-01-01", with_anchor=True)
        p = md / "nodes" / f"{name}.md"
        fm, order, body = _fm.read_node(p)
        for _ in range(6):
            body, _i, _n = I.erode(md, name, fm, order, body,
                                   days_since_recall=60, tier="cold")
        _fm.write_node(p, fm, order, body)
        I.anchor_path(md, name, fm).unlink()

    def test_no_anchor_flag_off_is_noop(self):
        # Flag OFF (default) -> no-anchor stays the safe no-op (no fabrication).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            self._eroded_no_anchor_node(md, "n")
            os.environ.pop(I.INTEGRITY_GENERATIVE_ENABLED_ENV, None)
            before = (md / "nodes" / "n.md").read_text()
            res = I.recall_repair(md, "n")
            self.assertFalse(res["repaired"])
            self.assertEqual(res["skipped"], "no-anchor")
            # Body untouched — no fabrication.
            self.assertEqual((md / "nodes" / "n.md").read_text(), before)

    def test_no_anchor_flag_on_but_backend_unavailable_is_noop(self):
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            self._eroded_no_anchor_node(md, "n")
            # Flag ON, but the backend is UNAVAILABLE -> still a safe no-op.
            with mock.patch.object(_contra, "synthesis_enabled", return_value=False), \
                 mock.patch.dict(os.environ,
                                 {I.INTEGRITY_GENERATIVE_ENABLED_ENV: "1"}):
                self.assertFalse(I.generative_enabled())
                res = I.partial_repair(md, "n", trigger="consolidation")
            self.assertFalse(res["repaired"])
            self.assertEqual(res["skipped"], "no-anchor")

    def test_backend_returns_none_is_noop(self):
        # The backend is enabled but returns None (unparseable/empty) -> no-op, no crash.
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            self._eroded_no_anchor_node(md, "n")
            with mock.patch.object(_contra, "synthesis_enabled", return_value=True), \
                 mock.patch.object(_contra, "synthesize_node", return_value=None), \
                 mock.patch.dict(os.environ,
                                 {I.INTEGRITY_GENERATIVE_ENABLED_ENV: "1"}):
                res = I.recall_repair(md, "n")
            self.assertFalse(res["repaired"])
            self.assertEqual(res["skipped"], "no-anchor")

    def test_generative_enabled_false_when_flag_off(self):
        os.environ.pop(I.INTEGRITY_GENERATIVE_ENABLED_ENV, None)
        self.assertFalse(I.generative_enabled())


# ---------------------------------------------------------------------------
# (6) The relevance/lifecycle decay is STILL untouched by P3
# ---------------------------------------------------------------------------


class TestRelevanceAxisUntouchedByP3(unittest.TestCase):
    def test_step_relevance_unchanged(self):
        from samia.core import tier
        for days in (0, 3, 7, 8, 30, 365):
            new, reason = tier.step_relevance(0.6, False, days)
            zero, zero_reason = tier.step_relevance(0.6, False, days, salience=0.0)
            self.assertEqual(new, zero)
            self.assertEqual(reason, zero_reason)

    def test_terminal_freeze_does_not_change_relevance_step(self):
        # The integrity floor freeze reuses ia.freeze (lifecycle), but it does NOT alter
        # the relevance math — a surviving (above-floor) node keeps its relevance/tier.
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "survivor", _BODY, tier_label="hot",
                            last_access="2026-06-06", relevance=0.7, with_anchor=True)
            rel_before = _fm.read_node(p)[0]["relevance"]
            tier_before = _fm.read_node(p)[0]["tier"]
            I.integrity_decay_pass(md, dry=False, today="2026-06-07",
                                   terminal_freeze=True)
            fm, _o, _b = _fm.read_node(p)  # hot + fresh stays above floor (survives)
            self.assertEqual(fm["relevance"], rel_before)
            self.assertEqual(fm["tier"], tier_before)


# ---------------------------------------------------------------------------
# (7) Inert by default — both P3 mechanisms OFF unless opted-in
# ---------------------------------------------------------------------------


class TestP3InertByDefault(unittest.TestCase):
    def test_pass_default_does_not_freeze_and_does_not_call_generative(self):
        from samia.runtime import contradiction as _contra
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _erode_to_just_above_floor(md, "n")
            with mock.patch.object(_contra, "synthesize_node") as m:
                # Defaults: terminal_freeze OFF, generative flag OFF.
                os.environ.pop(I.INTEGRITY_GENERATIVE_ENABLED_ENV, None)
                I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            m.assert_not_called()
            self.assertTrue((md / "nodes" / "n.md").exists())  # not frozen
            self.assertFalse((md / "archive" / "n.frozen.json").exists())


if __name__ == "__main__":
    unittest.main()
