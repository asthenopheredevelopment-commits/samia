"""Tests for the Tier-2 merge consumer P2 — LLM-synthesized abstraction (gated).

Layer 1 (Owns / Depends):
    Owns:    unit tests for merge_consumer's P2 surface —
             synthesize_abstraction / synthesize_pending (PROPOSE a draft, no
             node created), confirm_abstraction (create the abstraction node +
             supersede BOTH sources RESTORABLY + provenance edges),
             reject_abstraction (changes nothing), the inference-disabled
             SAFE-NO-OP (pair stays pending), the NEVER-auto-applied guarantee,
             AND that the P1 dup auto-merge still works unchanged.
    Depends: samia.core.{merge_consumer,ia,frontmatter,web_store},
             samia.runtime.contradiction (synthesize_node — MOCKED, never a real
             model load), unittest, unittest.mock, tempfile, json, sqlite3.
             EVERY test uses a tempdir memory root + a tempdir db_dir — NEVER the
             live ~/.local/share memory or the global edges.db, and the inference
             call is ALWAYS mocked (no llama-cli / model loads).

Layer 2 (What / Why):
    What: validates P2's Exit from the approved proposal (Q2c OPERATOR-GATE):
          (a) an 'abstract' pair gets a PROPOSED abstraction — the draft is
              staged with the synthesized title+body, but NO node is created and
              NEITHER source is touched;
          (b) confirm_abstraction creates the abstraction node, supersedes BOTH
              sources RESTORABLY (restore_node round-trips each byte-exact), and
              adds provenance edges abstraction->each source;
          (c) reject_abstraction marks rejected and changes NOTHING (both
              originals stay live, no node created);
          (d) synthesis with inference DISABLED is a safe no-op — the pair stays
              pending, no crash, no proposal;
          (e) abstractions are NEVER auto-applied — without a confirm, no node is
              created and no source is superseded;
          (f) P1 dup auto-merge still works unchanged after P2 lands.
    Why:  abstractions create NEW content + can lose nuance, so the WHOLE point
          of P2 is the operator gate: a regression that auto-applies an
          abstraction (no confirm) silently rewrites real memory and loses
          nuance; a regression that fails to archive both sources on confirm
          makes the abstraction irreversible. The inference call is mocked so no
          model loads; edges.db is routed to a temp db_dir so the live store is
          untouched.
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
from unittest import mock

from samia.core import merge_consumer as mc
from samia.core import ia, web_store, frontmatter
from samia.runtime import contradiction as con


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(md: Path, name: str, body: str, **fm) -> None:
    lines = ["---", f"name: {name}"]
    for k, v in fm.items():
        lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append(body)
    (md / "nodes" / f"{name}.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


def _mem() -> Path:
    # What: mkdtemp + atexit-registered rmtree of every dir handed out. Why:
    #   mkdtemp does NOT auto-clean, so each call left a tier2_merge_p2_test_* dir
    #   in /tmp and tripped the cold-metal zero-leftover hygiene gate. One atexit
    #   registration covers every `md = _mem()` call site and any test order.
    md = Path(tempfile.mkdtemp(prefix="tier2_merge_p2_test_"))
    atexit.register(shutil.rmtree, md, ignore_errors=True)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    (md / "archive").mkdir(parents=True, exist_ok=True)
    return md


# Two distinct-but-overlapping notes (the P2 'abstract' tier).
_ABS_A = "alpha beta gamma delta rocket engine thrust nozzle combustion"
_ABS_B = "alpha beta gamma delta ocean tide salinity current wave"

# Near-identical bodies => DUP (for the P1-still-works check).
_DUP_BODY = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
_DUP_BODY_RICH = _DUP_BODY + " mu nu xi omicron extra detail here"

_SYNTH = {"title": "shared-greek-prefix concept",
          "body": "Both notes share the alpha-beta-gamma-delta opening; the "
                  "unified concept is the common prefix across distinct domains."}


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


def _queue_abstract(md: Path, a: str, b: str) -> str:
    """Record an 'abstract' pending pair (as P1's drain would). Returns its id."""
    mc._record_abstract(md, a, b, similarity=0.33)
    return mc._candidate_id(a, b)


# ---------------------------------------------------------------------------
# (a) an 'abstract' pair gets a PROPOSED abstraction (node NOT yet created)
# ---------------------------------------------------------------------------


class TestSynthesizeProposes(unittest.TestCase):

    def test_synthesize_stages_proposal_no_node_created(self):
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        cid = _queue_abstract(md, "X", "Y")

        with mock.patch.object(con, "synthesize_node", return_value=_SYNTH):
            res = mc.synthesize_abstraction(md, "X", "Y")

        self.assertEqual(res["status"], "proposed")
        self.assertEqual(res["candidate_id"], cid)
        # The candidate is now 'proposed' carrying the synthesized draft.
        cands = mc.list_abstraction_candidates(md)
        self.assertEqual(len(cands), 1)
        rec = cands[0]
        self.assertEqual(rec["status"], "proposed")
        self.assertEqual(rec["abstraction"]["title"], _SYNTH["title"])
        self.assertEqual(rec["abstraction"]["body"], _SYNTH["body"])
        self.assertEqual(rec["merged_from"], ["X", "Y"])
        # NO abstraction node exists yet, BOTH sources still live.
        abs_id = mc._new_abstraction_id(md, cid)
        # _new_abstraction_id returns the would-be id; nothing with that prefix exists.
        self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])
        self.assertTrue((md / "nodes" / "X.md").exists())
        self.assertTrue((md / "nodes" / "Y.md").exists())

    def test_synthesize_pending_proposes_batch(self):
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        _queue_abstract(md, "X", "Y")
        with _Enabled(), mock.patch.object(con, "synthesize_node",
                                           return_value=_SYNTH):
            res = mc.synthesize_pending(md, budget=10)
        self.assertTrue(res["fired"])
        self.assertEqual(res["proposed"], 1)
        self.assertEqual(res["pending"], 0)
        self.assertEqual(mc.list_abstraction_candidates(md)[0]["status"],
                         "proposed")

    def test_synthesize_pending_noop_when_disabled(self):
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        _queue_abstract(md, "X", "Y")
        with _Enabled(on=False), mock.patch.object(con, "synthesize_node",
                                                   return_value=_SYNTH):
            res = mc.synthesize_pending(md, budget=10)
        self.assertFalse(res["fired"])
        self.assertEqual(res["proposed"], 0)
        # Still pending (the enable gate held).
        self.assertEqual(mc.list_abstraction_candidates(md)[0]["status"],
                         "pending")


# ---------------------------------------------------------------------------
# (b) confirm creates the node + supersedes BOTH sources RESTORABLY + edges
# ---------------------------------------------------------------------------


class TestConfirmAbstraction(unittest.TestCase):

    def test_confirm_creates_node_supersedes_both_restorably(self):
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "X", _ABS_A)
            _node(md, "Y", _ABS_B)
            x_before = (md / "nodes" / "X.md").read_text(encoding="utf-8")
            y_before = (md / "nodes" / "Y.md").read_text(encoding="utf-8")
            cid = _queue_abstract(md, "X", "Y")
            with mock.patch.object(con, "synthesize_node", return_value=_SYNTH):
                mc.synthesize_abstraction(md, "X", "Y")

            rec = mc.confirm_abstraction(md, cid, db_dir=edb)
            self.assertTrue(rec["confirmed"])
            abs_id = rec["abstraction_id"]

            # The abstraction node now exists with the synthesized content.
            abs_path = md / "nodes" / f"{abs_id}.md"
            self.assertTrue(abs_path.exists())
            fm, _, body = frontmatter.read_node(abs_path)
            self.assertEqual(body.strip(), _SYNTH["body"])
            self.assertEqual(fm.get("merged_from"), ["X", "Y"])

            # BOTH sources superseded (files gone), both archived restorable.
            self.assertFalse((md / "nodes" / "X.md").exists())
            self.assertFalse((md / "nodes" / "Y.md").exists())
            for src in ("X", "Y"):
                arc = md / "archive" / f"{src}.superseded.json"
                self.assertTrue(arc.exists())
                self.assertEqual(
                    json.loads(arc.read_text())["superseded_by"], abs_id)
            self.assertEqual(set(rec["superseded"]), {"X", "Y"})

            # Provenance edges abstraction -> each source.
            conn = sqlite3.connect(web_store._db_path(edb))
            rows = conn.execute(
                "SELECT src_node, dst_node FROM edges WHERE ref_kind='provenance'"
            ).fetchall()
            conn.close()
            self.assertEqual(set(rows),
                             {(f"{abs_id}.md", "X.md"), (f"{abs_id}.md", "Y.md")})

            # restore_node round-trips EACH source byte-exact.
            self.assertTrue(ia.restore_node(md, "X")["restored"])
            self.assertTrue(ia.restore_node(md, "Y")["restored"])
            self.assertEqual((md / "nodes" / "X.md").read_text(encoding="utf-8"),
                             x_before)
            self.assertEqual((md / "nodes" / "Y.md").read_text(encoding="utf-8"),
                             y_before)

            # Candidate marked confirmed (resolved, off the unresolved list).
            self.assertEqual(mc.list_abstraction_candidates(md), [])

    def test_confirm_unproposed_is_refused(self):
        # A still-'pending' (un-synthesized) candidate cannot be confirmed.
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        cid = _queue_abstract(md, "X", "Y")
        res = mc.confirm_abstraction(md, cid)
        self.assertFalse(res["confirmed"])
        self.assertIn("proposed", res["error"])
        # Nothing created / superseded.
        self.assertTrue((md / "nodes" / "X.md").exists())
        self.assertTrue((md / "nodes" / "Y.md").exists())
        self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])


# ---------------------------------------------------------------------------
# (c) reject changes nothing
# ---------------------------------------------------------------------------


class TestRejectAbstraction(unittest.TestCase):

    def test_reject_changes_nothing(self):
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        cid = _queue_abstract(md, "X", "Y")
        with mock.patch.object(con, "synthesize_node", return_value=_SYNTH):
            mc.synthesize_abstraction(md, "X", "Y")

        res = mc.reject_abstraction(md, cid)
        self.assertTrue(res["rejected"])
        # Both originals still live, NO abstraction node, no archive.
        self.assertTrue((md / "nodes" / "X.md").exists())
        self.assertTrue((md / "nodes" / "Y.md").exists())
        self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])
        self.assertEqual(list((md / "archive").glob("*.superseded.json")), [])
        # The candidate is resolved (off the unresolved list).
        self.assertEqual(mc.list_abstraction_candidates(md), [])


# ---------------------------------------------------------------------------
# (d) synthesis with inference DISABLED is a safe no-op (pair stays pending)
# ---------------------------------------------------------------------------


class TestInferenceDisabledNoOp(unittest.TestCase):

    def test_synthesize_noop_when_inference_unavailable(self):
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        _queue_abstract(md, "X", "Y")
        # synthesize_node returns None when the backend is off/unavailable.
        with mock.patch.object(con, "synthesize_node", return_value=None):
            res = mc.synthesize_abstraction(md, "X", "Y")
        self.assertEqual(res["status"], "pending")
        # The pair is LEFT pending (no proposal, no crash).
        rec = mc.list_abstraction_candidates(md)[0]
        self.assertEqual(rec["status"], "pending")
        self.assertNotIn("abstraction", rec)
        # No node created, both sources untouched.
        self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])
        self.assertTrue((md / "nodes" / "X.md").exists())
        self.assertTrue((md / "nodes" / "Y.md").exists())

    def test_synthesis_enabled_tracks_judge_flag(self):
        # synthesis_enabled() must TRACK the judge flag AND inference availability:
        #   synthesis_enabled() == (_JUDGE_ENABLED and _inference_available()).
        # _JUDGE_ENABLED is read at import (its env default flipped 0->1 under
        # TUNE-2026-06-10), so test the tracking CONTRACT by patching the module
        # global to BOTH states — not by depending on the env default.
        # FIX-2026-06-10 (MEDIUM): the old assertion (== con._JUDGE_ENABLED) broke
        # when inference is unavailable (MockBackend) under the new default-on flag.
        saved = con._JUDGE_ENABLED
        try:
            for flag, env_val in ((False, "0"), (True, "1")):
                with mock.patch.dict(
                        os.environ,
                        {"ASTHENOS_CONTRADICTION_JUDGE": env_val}):
                    con._JUDGE_ENABLED = flag
                    expected = flag and con._inference_available()
                    self.assertEqual(con.synthesis_enabled(), expected)
                    # When synthesis is disabled (judge off OR no real backend),
                    # synthesize_node is a no-op (None) without mocking subprocess.
                    if not expected:
                        self.assertIsNone(con.synthesize_node("a", "b"))
        finally:
            con._JUDGE_ENABLED = saved


# ---------------------------------------------------------------------------
# (e) abstractions are NEVER auto-applied (no confirm => no node, no supersede)
# ---------------------------------------------------------------------------


class TestNeverAutoApplied(unittest.TestCase):

    def test_synthesize_alone_never_applies(self):
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        _queue_abstract(md, "X", "Y")
        with mock.patch.object(con, "synthesize_node", return_value=_SYNTH):
            mc.synthesize_abstraction(md, "X", "Y")
        # Proposed, but WITHOUT a confirm: no node, no supersede, no archive.
        self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])
        self.assertTrue((md / "nodes" / "X.md").exists())
        self.assertTrue((md / "nodes" / "Y.md").exists())
        self.assertEqual(list((md / "archive").glob("*.superseded.json")), [])

    def test_subscriber_synthesis_proposes_not_applies(self):
        # The REM subscriber's P2 step proposes; it never auto-creates a node.
        from samia.runtime import rem_cycle, rem_subscribers
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        # Queue an abstract pair (no .consolidation_candidates -> drain merges 0).
        _queue_abstract(md, "X", "Y")
        with rem_cycle._rem_subscribers_lock:
            saved = dict(rem_cycle._rem_subscribers)
            rem_cycle._rem_subscribers.clear()
        try:
            rem_cycle.enter_rem(md, "test")
            with _Enabled(), mock.patch.object(con, "synthesize_node",
                                               return_value=_SYNTH):
                res = rem_subscribers._sub_tier2_merge(md)
            self.assertTrue(res["fired"])
            self.assertEqual(res["synthesis"]["proposed"], 1)
            # PROPOSED only — no node materialized, both sources live.
            self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])
            self.assertTrue((md / "nodes" / "X.md").exists())
            self.assertTrue((md / "nodes" / "Y.md").exists())
            self.assertEqual(mc.list_abstraction_candidates(md)[0]["status"],
                             "proposed")
        finally:
            with rem_cycle._rem_subscribers_lock:
                rem_cycle._rem_subscribers.clear()
                rem_cycle._rem_subscribers.update(saved)


# ---------------------------------------------------------------------------
# (f) P1 dup auto-merge still works unchanged after P2 lands
# ---------------------------------------------------------------------------


class TestP1StillWorks(unittest.TestCase):

    def test_dup_merge_unchanged(self):
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "A", _DUP_BODY, access_count=1)
            _node(md, "B", _DUP_BODY_RICH, access_count=5)  # richer survivor
            a_before = (md / "nodes" / "A.md").read_text(encoding="utf-8")
            payload = {"generated": "x", "threshold": 0.15, "candidates": [
                {"chain": "k", "a_addr": "A", "a_file": "nodes/A.md",
                 "b_addr": "B", "b_file": "nodes/B.md", "similarity": 0.99}]}
            (md / ".consolidation_candidates.json").write_text(
                json.dumps(payload), encoding="utf-8")
            with _Enabled():
                res = mc.drain(md, budget=10, cursor=0, db_dir=edb)
            self.assertEqual(res["merged"], 1)
            self.assertFalse((md / "nodes" / "A.md").exists())
            self.assertTrue((md / "nodes" / "B.md").exists())
            # Restorable.
            self.assertTrue(ia.restore_node(md, "A")["restored"])
            self.assertEqual((md / "nodes" / "A.md").read_text(encoding="utf-8"),
                             a_before)


# ---------------------------------------------------------------------------
# MCP surface — list / confirm / reject delegate to the consumer (operator-only)
# ---------------------------------------------------------------------------


class TestMcpSurface(unittest.TestCase):

    def test_mcp_list_confirm_reject(self):
        from samia.core import mcp_server
        md = _mem()
        with tempfile.TemporaryDirectory() as edb:
            _node(md, "X", _ABS_A)
            _node(md, "Y", _ABS_B)
            cid = _queue_abstract(md, "X", "Y")
            with mock.patch.object(con, "synthesize_node", return_value=_SYNTH):
                mc.synthesize_abstraction(md, "X", "Y")

            # list surfaces the proposed candidate.
            listed = mcp_server.memory_merge_candidates(md)
            self.assertEqual(len(listed["candidates"]), 1)
            self.assertEqual(listed["candidates"][0]["candidate_id"], cid)

            # confirm via MCP creates + supersedes (route edges to temp db).
            with mock.patch("samia.core.web_store._DEFAULT_DB_PATH",
                            str(Path(edb) / "edges.db")):
                res = mcp_server.memory_confirm_merge(md, cid)
            self.assertTrue(res["confirmed"])
            self.assertFalse((md / "nodes" / "X.md").exists())
            self.assertEqual(mcp_server.memory_merge_candidates(md)["candidates"],
                             [])

    def test_mcp_reject_changes_nothing(self):
        from samia.core import mcp_server
        md = _mem()
        _node(md, "X", _ABS_A)
        _node(md, "Y", _ABS_B)
        cid = _queue_abstract(md, "X", "Y")
        with mock.patch.object(con, "synthesize_node", return_value=_SYNTH):
            mc.synthesize_abstraction(md, "X", "Y")
        res = mcp_server.memory_reject_merge(md, cid)
        self.assertTrue(res["rejected"])
        self.assertTrue((md / "nodes" / "X.md").exists())
        self.assertTrue((md / "nodes" / "Y.md").exists())
        self.assertEqual(list((md / "nodes").glob("merge-*.md")), [])


if __name__ == "__main__":
    unittest.main()
