"""samia.core.test_merge_consumer — tests for the Tier-2 merge consumer P1 pick-winner dup-merge (FEAT-2026-06-07).

Layer 1 (Owns / Depends):
    Owns:    unit tests for merge_consumer's P1 surface — load_candidates,
             classify_pair (dup vs abstract via the jaccard fallback),
             pick_winner (richer survivor), merge_dup (RESTORABLE supersede +
             provenance edge), drain (backlog shrinks + work_remaining flips),
             the enable-flag gate, AND the rem_subscribers registration of
             "tier2_merge" at priority 22 + its refuse-outside-REM gate.
    Depends: samia.core.{merge_consumer,ia,frontmatter,consolidation,web_store},
             samia.runtime.{rem_cycle,rem_subscribers}, tempfile, unittest,
             json, sqlite3. EVERY test uses a tempdir memory root + a tempdir
             db_dir — NEVER the live ~/.local/share memory or the global
             edges.db, and NEVER a real embedder (no vector index is built, so
             classify_pair takes the deterministic jaccard fallback).

Layer 2 (What / Why):
    What: validates P1's Exit from the approved proposal:
          (a) a true-dup pair above the bar picks the RICHER winner + supersedes
              the loser RESTORABLY (restore_node round-trips the loser byte-exact)
              + lays a provenance edge survivor->loser;
          (b) a below-bar distinct-but-overlapping pair is NOT merged (recorded
              "abstract" for P2, both nodes still live);
          (c) draining a pair SHRINKS the candidate backlog (work_remaining goes
              false once the backlog is drained);
          (d) gated OFF (enable flag unset) -> no-op, nothing merged;
          (e) registered at priority 22 BETWEEN consolidation(20) and
              contradiction_passive(25);
          (f) the subscriber REFUSES outside REM.
    Why:  P1 is the missing DRAIN — without it REM's work_remaining stays true
          forever (the live sticky-REM symptom). Every merge must be reversible
          (the central false-positive-merge risk is acceptable ONLY because
          restore_node un-forgets byte-exact). A leak in the enable gate or the
          REM gate would run irreversible auto-merges outside operator control.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from samia.core import merge_consumer as mc
from samia.core import ia, web_store, consolidation
from samia.runtime import rem_cycle, rem_subscribers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(md: Path, name: str, body: str, **fm) -> None:
    """Write a node file with the given frontmatter + body."""
    lines = ["---", f"name: {name}"]
    for k, v in fm.items():
        lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append(body)
    (md / "nodes" / f"{name}.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


def _mem() -> Path:
    # What: mkdtemp + atexit-registered rmtree of every dir handed out. Why:
    #   mkdtemp does NOT auto-clean, so each call left a tier2_merge_test_* dir in
    #   /tmp and tripped the cold-metal zero-leftover hygiene gate. One atexit
    #   registration covers every `md = _mem()` call site and any test order.
    md = Path(tempfile.mkdtemp(prefix="tier2_merge_test_"))
    atexit.register(shutil.rmtree, md, ignore_errors=True)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    (md / "archive").mkdir(parents=True, exist_ok=True)
    return md


# Near-identical bodies => jaccard ~1.0 (DUP, >= 0.85 bar).
_DUP_BODY = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
             "lambda mu nu xi omicron pi rho sigma")
_DUP_BODY_RICH = _DUP_BODY + " tau upsilon phi chi psi omega extra detail here"

# Overlapping ~0.33 jaccard (surfaced, but ABSTRACT — below 0.85 bar).
_ABS_A = "alpha beta gamma delta rocket engine thrust nozzle combustion"
_ABS_B = "alpha beta gamma delta ocean tide salinity current wave"


def _write_candidates(md: Path, pairs: list[dict], threshold: float = 0.15) -> None:
    payload = {"generated": "x", "threshold": threshold, "candidates": pairs}
    (md / ".consolidation_candidates.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def _cand(a: str, b: str, sim: float) -> dict:
    return {"chain": "k", "a_addr": a, "a_file": f"nodes/{a}.md",
            "b_addr": b, "b_file": f"nodes/{b}.md", "similarity": sim}


class _Enabled:
    """Context manager: set ASTHENOS_TIER2_MERGE_ENABLED for the body."""

    def __init__(self, on: bool = True):
        self.on = on
        self._prev = None

    def __enter__(self):
        self._prev = os.environ.get("ASTHENOS_TIER2_MERGE_ENABLED")
        os.environ["ASTHENOS_TIER2_MERGE_ENABLED"] = "1" if self.on else "0"
        return self

    def __exit__(self, *a):
        if self._prev is None:
            os.environ.pop("ASTHENOS_TIER2_MERGE_ENABLED", None)
        else:
            os.environ["ASTHENOS_TIER2_MERGE_ENABLED"] = self._prev


# ---------------------------------------------------------------------------
# (a) true dup -> richer winner + RESTORABLE supersede + provenance edge
# ---------------------------------------------------------------------------


class TestPickWinnerDupMerge(unittest.TestCase):

    def test_picks_richer_winner(self):
        """Higher access_count then longer body wins."""
        md = _mem()
        _node(md, "A", _DUP_BODY, access_count=1)
        _node(md, "B", _DUP_BODY_RICH, access_count=5)  # richer
        survivor, loser = mc.pick_winner(md, "A", "B")
        self.assertEqual(survivor, "B")
        self.assertEqual(loser, "A")

    def test_dup_merge_restorable_and_provenance(self):
        """The loser is superseded RESTORABLY + a provenance edge is laid."""
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "A", _DUP_BODY, access_count=1)
            _node(md, "B", _DUP_BODY_RICH, access_count=5)  # survivor
            loser_text_before = (md / "nodes" / "A.md").read_text(encoding="utf-8")

            with _Enabled():
                rec = mc.merge_dup(md, "A", "B", db_dir=edb)

            self.assertEqual(rec["survivor"], "B")
            self.assertEqual(rec["loser"], "A")
            # Loser file gone, survivor still live.
            self.assertFalse((md / "nodes" / "A.md").exists())
            self.assertTrue((md / "nodes" / "B.md").exists())
            # Restorable archive exists.
            arc = md / "archive" / "A.superseded.json"
            self.assertTrue(arc.exists())
            archived = json.loads(arc.read_text(encoding="utf-8"))
            self.assertEqual(archived["superseded_by"], "B")
            # merged_from stamped on the survivor.
            from samia.core import frontmatter as fm
            parsed, _ = fm.parse((md / "nodes" / "B.md").read_text(encoding="utf-8"))
            self.assertIn("A", parsed[0].get("merged_from", []))
            # Provenance edge present in edges.db (temp db_dir).
            conn = sqlite3.connect(web_store._db_path(edb))
            row = conn.execute(
                "SELECT src_node, dst_node FROM edges WHERE ref_kind='provenance'"
            ).fetchone()
            conn.close()
            self.assertEqual(row, ("B.md", "A.md"))

            # RESTORE round-trips the loser byte-exact.
            res = ia.restore_node(md, "A")
            self.assertTrue(res["restored"])
            self.assertEqual((md / "nodes" / "A.md").read_text(encoding="utf-8"),
                             loser_text_before)


# ---------------------------------------------------------------------------
# (b) below-bar distinct-but-overlapping pair is NOT merged (left for P2)
# ---------------------------------------------------------------------------


class TestAbstractNotMerged(unittest.TestCase):

    def test_classify_below_bar_is_abstract(self):
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        # No vector index => jaccard fallback (~0.33 < 0.85).
        self.assertEqual(mc.classify_pair(md, "X", "Y", candidate_similarity=0.333),
                         "abstract")

    def test_drain_records_abstract_does_not_merge(self):
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "X", _ABS_A)
            _node(md, "Y", _ABS_B)
            _write_candidates(md, [_cand("X", "Y", 0.333)])
            with _Enabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            self.assertEqual(res["merged"], 0)
            self.assertEqual(res["recorded"], 1)
            # Both originals still live.
            self.assertTrue((md / "nodes" / "X.md").exists())
            self.assertTrue((md / "nodes" / "Y.md").exists())
            # Recorded to the P2 queue.
            q = md / "biomimetic" / "merge_candidates.jsonl"
            self.assertTrue(q.exists())
            rec = json.loads(q.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["mode"], "abstract")
            self.assertEqual(rec["status"], "pending")


# ---------------------------------------------------------------------------
# (c) draining shrinks the backlog -> work_remaining goes false
# ---------------------------------------------------------------------------


class TestDrainShrinksBacklog(unittest.TestCase):

    def test_drain_removes_pair_and_flips_work_remaining(self):
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "A", _DUP_BODY, access_count=1)
            _node(md, "B", _DUP_BODY_RICH, access_count=5)
            _write_candidates(md, [_cand("A", "B", 0.99)])
            self.assertEqual(len(mc.load_candidates(md)), 1)

            with _Enabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)

            self.assertEqual(res["merged"], 1)
            self.assertEqual(res["drained"], 1)
            self.assertEqual(res["remaining"], 0)
            self.assertFalse(res["work_remaining"])
            # The candidate file now has zero candidates (backlog drained).
            self.assertEqual(len(mc.load_candidates(md)), 0)

    def test_stale_pair_dropped_as_drained(self):
        """A pair whose node is already gone is dropped (the drain advancing)."""
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "A", _DUP_BODY)  # B.md never created
            _write_candidates(md, [_cand("A", "B", 0.99)])
            with _Enabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            self.assertEqual(res["merged"], 0)
            self.assertEqual(res["skipped"], 1)
            self.assertEqual(res["remaining"], 0)
            self.assertFalse(res["work_remaining"])


# ---------------------------------------------------------------------------
# (d) gated OFF (enable flag unset) -> no-op, nothing merged
# ---------------------------------------------------------------------------


class TestEnableGate(unittest.TestCase):

    def test_drain_noop_when_disabled(self):
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "A", _DUP_BODY, access_count=1)
            _node(md, "B", _DUP_BODY_RICH, access_count=5)
            _write_candidates(md, [_cand("A", "B", 0.99)])
            with _Enabled(on=False):
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            self.assertFalse(res["fired"])
            self.assertEqual(res["merged"], 0)
            # Nothing merged: both nodes live, candidate file untouched.
            self.assertTrue((md / "nodes" / "A.md").exists())
            self.assertTrue((md / "nodes" / "B.md").exists())
            self.assertEqual(len(mc.load_candidates(md)), 1)
            # work_remaining still reflects the undrained backlog.
            self.assertTrue(res["work_remaining"])


# ---------------------------------------------------------------------------
# (e)+(f) REM subscriber registration at priority 22 + refuse outside REM
# ---------------------------------------------------------------------------


class _RegistryIsolation(unittest.TestCase):
    def setUp(self) -> None:
        with rem_cycle._rem_subscribers_lock:
            self._saved = dict(rem_cycle._rem_subscribers)
            rem_cycle._rem_subscribers.clear()

    def tearDown(self) -> None:
        with rem_cycle._rem_subscribers_lock:
            rem_cycle._rem_subscribers.clear()
            rem_cycle._rem_subscribers.update(self._saved)


class TestSubscriberRegistration(_RegistryIsolation):

    def test_registered_at_priority_22(self):
        rem_subscribers.register_rem_subscribers()
        names = rem_cycle.registered_offline_ops()
        self.assertIn("tier2_merge", names)
        # Priority 22 => ordered AFTER consolidation(20), BEFORE contradiction(25).
        i_con = names.index("consolidation")
        i_t2 = names.index("tier2_merge")
        i_cp = names.index("contradiction_passive")
        self.assertLess(i_con, i_t2)
        self.assertLess(i_t2, i_cp)
        sub = rem_cycle._rem_subscribers["tier2_merge"]
        self.assertEqual(sub.priority, 22)
        self.assertEqual(sub.cursor_key, "tier2_merge")

    def test_subscriber_refuses_outside_rem(self):
        md = _mem()
        _node(md, "A", _DUP_BODY, access_count=1)
        _node(md, "B", _DUP_BODY_RICH, access_count=5)
        _write_candidates(md, [_cand("A", "B", 0.99)])
        # NOT in REM (no enter_rem) -> the gate refuses even when enabled.
        with _Enabled():
            res = rem_subscribers._sub_tier2_merge(md)
        self.assertFalse(res["fired"])
        self.assertEqual(res.get("refused"), "not_in_rem")
        # Nothing merged.
        self.assertTrue((md / "nodes" / "A.md").exists())
        self.assertTrue((md / "nodes" / "B.md").exists())

    def test_subscriber_drains_inside_rem(self):
        md = _mem()
        _node(md, "A", _DUP_BODY, access_count=1)
        _node(md, "B", _DUP_BODY_RICH, access_count=5)
        _write_candidates(md, [_cand("A", "B", 0.99)])
        rem_cycle.enter_rem(md, "test")
        with _Enabled():
            res = rem_subscribers._sub_tier2_merge(md)
        self.assertTrue(res["fired"])
        self.assertEqual(res["merged"], 1)
        # Cursor checkpointed with done=True (backlog drained).
        cur = rem_cycle.read_cursor(md, "tier2_merge")
        self.assertTrue(cur.get("done"))
        self.assertEqual(cur.get("remaining"), 0)

    def test_due_condition_double_gated(self):
        md = _mem()
        _write_candidates(md, [_cand("A", "B", 0.99)])
        # Disabled => not due even with candidates present.
        with _Enabled(on=False):
            self.assertFalse(rem_subscribers._due_tier2_merge(md))
        # Enabled + candidates => due.
        with _Enabled():
            self.assertTrue(rem_subscribers._due_tier2_merge(md))
        # Enabled + no candidates => not due.
        (md / ".consolidation_candidates.json").unlink()
        with _Enabled():
            self.assertFalse(rem_subscribers._due_tier2_merge(md))


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.test_merge_consumer
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-07 Tier-2 merge consumer P1
# Layer:      test (pytest)
# Role:       tests for samia.core.merge_consumer P1 — load_candidates,
#             classify_pair (jaccard fallback), pick_winner (richer survivor),
#             merge_dup (RESTORABLE supersede + provenance edge), drain (backlog
#             shrinks + work_remaining flips), the enable-flag gate, and the
#             rem_subscribers tier2_merge registration at priority 22 + its
#             refuse-outside-REM gate.
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.merge_consumer, samia.core.ia,
#             samia.core.frontmatter, samia.core.consolidation,
#             samia.core.web_store, samia.runtime.rem_cycle,
#             samia.runtime.rem_subscribers
# Exposes:    — (test module)
# Lines:      382
# --------------------------------------------------------------------------
