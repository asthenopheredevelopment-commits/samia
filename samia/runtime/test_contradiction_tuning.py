"""Tests for TUNE-2026-06-08 contradiction-detector tuning.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the three coordinated usability fixes that make the
             (previously OFF/flooded) contradiction/supersession detector usable:
               (1) TYPE-SCOPING -- excluded_types()/is_excluded_node() exclude
                   episodic/experiential nodes (session_offload, bug) at every
                   detector enumeration/match site (find_supersession_candidates,
                   passive_sweep, bio.active_set); env-overridable.
               (2) DEDICATED FAST JUDGE -- the judge + synth route to a CACHED
                   small BitNet-2B backend (ASTHENOS_CONTRADICTION_JUDGE_MODEL)
                   via inference.get_backend_for_model, NOT the main Qwen-14B,
                   and stay fail-soft when unavailable.
               (3) DOUBLE-LOAD FIX -- inference.get_backend() is a load-once
                   singleton; repeated get/judge calls return the SAME instance.
    Depends: samia.runtime.{contradiction, inference}, samia.core.bio,
             unittest, unittest.mock, tempfile, os, pathlib.

Layer 2 (What / Why):
    What: verifies the candidate space collapse (excluded nodes are never
          checked nor matched), the env-override of the exclude set, the judge's
          dedicated cached backend routing + fail-soft, and the per-model-path
          singleton that kills the double 14B load.
    Why:  PRODUCE-ONLY -- every test uses a tempfile memory_dir and MOCKS the
          inference backend / frontmatter so NO 9GB or 1GB gguf loads. Activation
          still needs a daemon restart; these tests only prove the wiring.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.runtime import contradiction as con
from samia.runtime import inference as inf
from samia.core import bio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_node(md: Path, node_id: str, node_type: str | None) -> None:
    """Write a node with an optional `type` frontmatter field."""
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    type_line = f"type: {node_type}\n" if node_type is not None else ""
    (md / "nodes" / f"{node_id}.md").write_text(
        f"---\nname: {node_id}\n{type_line}valid_from: 2026-01-01\n---\n"
        f"shared overlapping body words for {node_id} here today\n",
        encoding="utf-8",
    )


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    return md


def setUpModule() -> None:  # noqa: N802 (unittest hook name)
    con._clear_type_cache()


# ---------------------------------------------------------------------------
# (1) TYPE-SCOPING — exclusion predicate + env override
# ---------------------------------------------------------------------------


class TestExcludedTypes(unittest.TestCase):
    def test_default_exclude_set(self):
        """Default exclude set is session_offload + bug."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASTHENOS_CONTRADICTION_EXCLUDE_TYPES", None)
            self.assertEqual(con.excluded_types(),
                             frozenset({"session_offload", "bug"}))

    def test_env_override(self):
        """The exclude set is env-overridable (live read)."""
        with mock.patch.dict(
            os.environ,
            {"ASTHENOS_CONTRADICTION_EXCLUDE_TYPES": "session_offload, log , bug"},
        ):
            self.assertEqual(con.excluded_types(),
                             frozenset({"session_offload", "log", "bug"}))

    def test_excluded_node_session_offload_by_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "n_episodic", "session_offload")
            self.assertTrue(con.is_excluded_node(md, "n_episodic"))
            self.assertTrue(con.is_excluded_node(md, "n_episodic.md"))

    def test_excluded_node_bug_by_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "n_bug", "bug")
            self.assertTrue(con.is_excluded_node(md, "n_bug"))

    def test_included_node_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "n_project", "project")
            self.assertFalse(con.is_excluded_node(md, "n_project"))

    def test_included_node_reference_is_content(self):
        """A transcript stored as reference IS content -> INCLUDED."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "n_ref", "reference")
            self.assertFalse(con.is_excluded_node(md, "n_ref"))

    def test_missing_type_included_conservatively(self):
        """No `type` field -> INCLUDED (never silently drop a real claim)."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "n_untyped", None)
            self.assertFalse(con.is_excluded_node(md, "n_untyped"))

    def test_unreadable_session_offload_excluded_by_name(self):
        """A typeless node named session_*_offload_* still excludes by name."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "session_2026_06_07_offload_abc", None)
            self.assertTrue(
                con.is_excluded_node(md, "session_2026_06_07_offload_abc"))

    def test_missing_node_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            self.assertFalse(con.is_excluded_node(md, "does_not_exist"))


# ---------------------------------------------------------------------------
# (1) TYPE-SCOPING — find_supersession_candidates drops excluded matches
# ---------------------------------------------------------------------------


class TestFinderDropsExcludedMatches(unittest.TestCase):
    @mock.patch.object(con, "find_contradiction_candidates")
    def test_excluded_candidate_not_matched(self, mock_find):
        """A session_offload candidate is dropped as a MATCH; an included one stays."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "epi", "session_offload")
            _write_node(md, "claim", "project")
            mock_find.return_value = [
                {"node_id": "epi.md", "title": "epi", "score": 0.97},
                {"node_id": "claim.md", "title": "claim", "score": 0.95},
            ]
            # No jaccard primitive interference: stub it to always pass.
            with mock.patch(
                "samia.core.consolidation.shingles", return_value={"x"}
            ), mock.patch(
                "samia.core.consolidation.jaccard", return_value=1.0
            ):
                out = con.find_supersession_candidates(
                    "incoming content", memory_dir=md)
            ids = {c["node_id"] for c in out}
            self.assertNotIn("epi.md", ids)
            self.assertIn("claim.md", ids)


# ---------------------------------------------------------------------------
# (1) TYPE-SCOPING — passive_sweep skips an excluded node-being-checked
# ---------------------------------------------------------------------------


class TestPassiveSweepSkipsExcluded(unittest.TestCase):
    def _patch_finder_judge(self, finder, judge):
        return (
            mock.patch.object(con, "find_supersession_candidates",
                              side_effect=lambda *a, **k: finder),
            mock.patch.object(con, "judge_contradictions",
                              side_effect=lambda *a, **k: judge),
        )

    def test_excluded_node_not_checked_included_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            # Two nodes: one episodic (excluded), one project (included).
            _write_node(md, "a_episodic", "session_offload")
            _write_node(md, "z_project", "project")
            with mock.patch.dict(
                os.environ, {"ASTHENOS_CONTRADICTION_ENABLED": "1"}
            ):
                f_p, j_p = self._patch_finder_judge([], [])
                with f_p as fmock, j_p:
                    out = con.passive_sweep(md, budget=10,
                                            cursor={"index": 0,
                                                    "__no_persist__": True})
            # The finder was called for the INCLUDED node only, never the excluded.
            checked_texts = [c.args[0] for c in fmock.call_args_list]
            self.assertTrue(any("z_project" in t for t in checked_texts))
            self.assertFalse(any("a_episodic" in t for t in checked_texts))
            self.assertEqual(out["skipped_excluded"], 1)


# ---------------------------------------------------------------------------
# (1) TYPE-SCOPING — bio.active_set drops excluded nodes from the locus
# ---------------------------------------------------------------------------


class TestActiveSetExcludesEpisodic(unittest.TestCase):
    def test_active_set_drops_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con._clear_type_cache()
            _write_node(md, "epi", "session_offload")
            _write_node(md, "claim", "project")
            _write_node(md, "new_write", "project")
            # Force the locus to contain both an episodic and a content node by
            # stubbing the neighbor/hot sources bio.active_set unions.
            with mock.patch.object(
                bio, "_recently_accessed_nodes",
                return_value=["epi.md", "claim.md"]
            ), mock.patch.object(
                bio, "_fast_engram_neighbors", return_value=[]
            ), mock.patch(
                "samia.core.web_store.coactivation_neighbors", return_value=[]
            ):
                locus = bio.active_set(md, ["new_write.md"])
            self.assertIn("claim.md", locus)
            self.assertNotIn("epi.md", locus)


# ---------------------------------------------------------------------------
# (2) DEDICATED FAST JUDGE — routes to JUDGE_MODEL backend, fail-soft
# ---------------------------------------------------------------------------


def _fake_backend(text, *, cls_name="LlamaCppBackend"):
    class _FakeBackend:
        def complete(self, prompt, *, max_tokens=256, temperature=0.0, stop=None):
            return text

    _FakeBackend.__name__ = cls_name
    _FakeBackend.__qualname__ = cls_name
    return _FakeBackend()


class TestJudgeRoutesToDedicatedBackend(unittest.TestCase):
    def test_judge_uses_get_backend_for_model_with_judge_path(self):
        """When the dedicated factory yields a REAL backend, the judge uses it
        and calls get_backend_for_model with the JUDGE_MODEL path."""
        judge_path = "/tmp/does_not_exist_judge_model.gguf"
        with mock.patch.dict(
            os.environ, {"ASTHENOS_CONTRADICTION_JUDGE_MODEL": judge_path}
        ):
            with mock.patch.object(
                inf, "get_backend_for_model",
                return_value=_fake_backend("real judge")
            ) as mfac:
                backend = con._judge_backend()
                self.assertEqual(type(backend).__name__, "LlamaCppBackend")
                mfac.assert_called_once_with(judge_path)

    def test_judge_falls_back_to_get_backend_when_model_absent(self):
        """When the dedicated factory yields a MockBackend (model absent), the
        judge falls back to get_backend() (the path existing tests mock)."""
        with mock.patch.object(
            inf, "get_backend_for_model", return_value=inf.MockBackend()
        ), mock.patch.object(
            inf, "get_backend", return_value=_fake_backend("fallback")
        ) as mgb:
            backend = con._judge_backend()
            self.assertEqual(type(backend).__name__, "LlamaCppBackend")
            mgb.assert_called()

    def test_inference_available_true_with_real_dedicated_backend(self):
        with mock.patch.object(
            con, "_judge_backend", return_value=_fake_backend("x")
        ):
            self.assertTrue(con._inference_available())

    def test_inference_available_false_with_mock(self):
        with mock.patch.object(
            con, "_judge_backend", return_value=inf.MockBackend()
        ):
            self.assertFalse(con._inference_available())

    def test_judge_fail_soft_when_backend_unavailable(self):
        """_infer_text -> None (records-only) when the judge backend is None."""
        with mock.patch.object(con, "_judge_backend", return_value=None):
            self.assertIsNone(con._infer_text("prompt", 64))

    def test_judge_fail_soft_on_mock_backend(self):
        with mock.patch.object(
            con, "_judge_backend", return_value=inf.MockBackend()
        ):
            self.assertIsNone(con._infer_text("prompt", 64))


# ---------------------------------------------------------------------------
# (3) DOUBLE-LOAD FIX — get_backend / get_backend_for_model is a singleton
# ---------------------------------------------------------------------------


class TestBackendSingleton(unittest.TestCase):
    def setUp(self):
        # Snapshot + clear the per-model cache so tests are isolated.
        self._saved = dict(inf._model_backend_cache)
        inf._model_backend_cache.clear()

    def tearDown(self):
        inf._model_backend_cache.clear()
        inf._model_backend_cache.update(self._saved)

    def test_same_instance_for_same_model_path(self):
        """get_backend_for_model returns the SAME instance on repeated calls
        for the same gguf -- the model loads at most once (double-load fix)."""
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "model.gguf"
            gguf.write_bytes(b"\x00")  # presence only; never loaded
            # Avoid importing real llama_cpp: stub LlamaCppBackend + the import.
            sentinel = _fake_backend("once")
            with mock.patch.object(
                inf, "LlamaCppBackend", return_value=sentinel
            ), mock.patch(
                "importlib.import_module", return_value=mock.MagicMock()
            ):
                b1 = inf.get_backend_for_model(str(gguf))
                b2 = inf.get_backend_for_model(str(gguf))
            self.assertIs(b1, b2)

    def test_get_backend_caches_main_model(self):
        """get_backend() returns the SAME instance on repeated calls (no
        re-instantiation of the main Qwen-14B)."""
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "qwen.gguf"
            gguf.write_bytes(b"\x00")
            sentinel = _fake_backend("main")
            with mock.patch.dict(
                os.environ, {"ASTHENOS_INFERENCE_MODEL": str(gguf)}
            ), mock.patch.object(
                inf, "LlamaCppBackend", return_value=sentinel
            ) as mctor, mock.patch(
                "importlib.import_module", return_value=mock.MagicMock()
            ):
                b1 = inf.get_backend()
                b2 = inf.get_backend()
            self.assertIs(b1, b2)
            # LlamaCppBackend constructed exactly ONCE (loaded once).
            self.assertEqual(mctor.call_count, 1)

    def test_missing_model_returns_mock_not_cached(self):
        with mock.patch.dict(
            os.environ, {"ASTHENOS_INFERENCE_MODEL": "/nope/missing.gguf"}
        ):
            b = inf.get_backend()
        self.assertEqual(type(b).__name__, "MockBackend")


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.test_contradiction_tuning
# Phase:      TUNE-2026-06-08 (type-scoping + dedicated judge backend + double-
#             load singleton fix)
# Layer:      test
# Stability:  v1.0
# ErrorModel: unittest assertions; tempfile dirs; mocked inference + frontmatter;
#             NO real model load (no gguf is ever opened by llama_cpp).
# Depends:    samia.runtime.contradiction, samia.runtime.inference,
#             samia.core.bio, unittest, unittest.mock, tempfile, os, pathlib.
# Exposes:    TestExcludedTypes, TestFinderDropsExcludedMatches,
#             TestPassiveSweepSkipsExcluded, TestActiveSetExcludesEpisodic,
#             TestJudgeRoutesToDedicatedBackend, TestBackendSingleton.
# --------------------------------------------------------------------------
