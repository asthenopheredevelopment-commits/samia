"""samia.runtime.test_contradiction — tests for samia.runtime.contradiction (AUD60 contradiction detection).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for embedding candidate finder, judge gate,
             abstraction synthesis, and memory guard integration.
    Depends: samia.runtime.contradiction, samia.runtime.inference,
             unittest, unittest.mock.

Layer 2 (What / Why):
    What: Validates the three AUD60 phases plus the FIX-2026-06-08 rewire of
          the judge gate + Tier-2 synthesis onto the in-process inference
          backend (samia.runtime.inference.get_backend), replacing the dead
          llama-cli subprocess. Covers: (1) embedding similarity returns
          candidates above threshold, (2) the judge parses structured JSON
          from a MOCKED backend and applies the confidence filter, (3) the
          judge FAILS SOFT (records-only / empty) when the backend is a
          MockBackend / raises / returns junk, (4) synthesize_node returns a
          {title, body} when the backend is mocked and None when unavailable,
          (5) NO subprocess/llama-cli path remains, and (6) check_contradiction
          orchestrates both phases and returns guard-compatible output.
    Why:  Contradiction detection is the last line of defense against memory
          inconsistency. The rewire makes flag 2 (ASTHENOS_CONTRADICTION_JUDGE)
          and merge-P2 abstraction actually functional in the daemon (where the
          backend is loaded in-process) while preserving the conservative
          fail-soft posture: a judge/synth error must NEVER block or corrupt a
          write. These tests MOCK the inference backend so no 9GB model loads.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import samia.runtime.contradiction as contra


def _fake_backend(text, *, cls_name="LlamaCppBackend"):
    """Build a stand-in inference backend whose complete() returns *text*.

    What: a lightweight object with a complete() method and a class name that is
          NOT "MockBackend" so contradiction._infer_text treats it as a real
          backend. *text* may be a string (returned) or an Exception (raised).
    Why:  the rewired judge/synth probe type(backend).__name__ != "MockBackend"
          and call backend.complete(prompt, max_tokens=..., temperature=...).
          This mock exercises that exact contract without loading any model.
    """
    class _FakeBackend:
        def complete(self, prompt, *, max_tokens=256, temperature=0.0, stop=None):
            if isinstance(text, Exception):
                raise text
            return text

    _FakeBackend.__name__ = cls_name
    _FakeBackend.__qualname__ = cls_name
    return _FakeBackend()


class TestFindCandidatesDisabled(unittest.TestCase):
    """Tests for candidate finder when disabled or unconfigured."""

    def test_returns_empty_when_disabled(self):
        """What: returns empty list when _ENABLED is False.
        Why: the default-off state must not interfere with writes."""
        original = contra._ENABLED
        try:
            contra._ENABLED = False
            reasons, meta = contra.check_contradiction({"text": "test"})
            self.assertEqual(reasons, [])
            self.assertEqual(meta, [])
        finally:
            contra._ENABLED = original

    def test_returns_empty_when_no_memory_dir(self):
        """What: returns empty when memory_dir is not configured.
        Why: graceful degradation when the daemon hasn't called configure()."""
        original_enabled = contra._ENABLED
        original_dir = contra._MEMORY_DIR
        try:
            contra._ENABLED = True
            contra._MEMORY_DIR = None
            candidates = contra.find_contradiction_candidates("test text")
            self.assertEqual(candidates, [])
        finally:
            contra._ENABLED = original_enabled
            contra._MEMORY_DIR = original_dir


class _NeutralDedicatedBackend:
    """Mixin: force the DEDICATED judge backend resolution to fall back to
    inference.get_backend() — the seam these tests already mock.

    What: setUp patches inference.get_backend_for_model to return MockBackend
          (so _judge_backend() takes its documented Mock->main-backend fallback)
          and clears the per-model backend cache.
    Why:  on a box where llama_cpp + a real judge gguf are present (the [llm]
          extra installed, or autofetch pulled the model), the dedicated factory
          returns a REAL backend that the get_backend mock never intercepts —
          9 tests fail exactly there (found cold-metal round 2, 2026-06-12).
          Neutralizing the factory restores the intended single mock seam in
          BOTH environments, and keeps unit tests from ever triggering the
          factory's autofetch download path.
    """

    def setUp(self):
        from samia.runtime import inference as _inf
        patcher = mock.patch.object(_inf, "get_backend_for_model",
                                    lambda *a, **k: _inf.MockBackend())
        patcher.start()
        self.addCleanup(patcher.stop)
        _inf._model_backend_cache.clear()
        self.addCleanup(_inf._model_backend_cache.clear)


class TestJudgeContradictions(_NeutralDedicatedBackend, unittest.TestCase):
    """Tests for the LLM judge gate (rewired to the in-process backend)."""

    def test_returns_empty_when_disabled(self):
        """What: returns empty list when _JUDGE_ENABLED is False.
        Why: judge is opt-in; must not run when disabled."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = False
            result = contra.judge_contradictions("new claim", [{"node_id": "n1"}])
            self.assertEqual(result, [])
        finally:
            contra._JUDGE_ENABLED = original

    def test_returns_empty_on_no_candidates(self):
        """What: returns empty when candidates list is empty.
        Why: nothing to judge."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            result = contra.judge_contradictions("new claim", [])
            self.assertEqual(result, [])
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_parses_judge_response(self, mock_get_backend):
        """What: parses structured JSON from the in-process backend output.
        Why: verifies the judge prompt -> backend.complete -> parse pipeline
             on the rewired in-process path (NOT a subprocess)."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend(json.dumps({
                "contradictions": [
                    {
                        "existing_claim_id": "node_abc",
                        "explanation": "User prefers dark vs light mode",
                        "confidence": 0.85,
                    }
                ]
            }))
            candidates = [{"node_id": "node_abc", "title": "UI preference"}]
            result = contra.judge_contradictions("user prefers light mode", candidates)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["existing_claim_id"], "node_abc")
            self.assertGreaterEqual(result[0]["confidence"], 0.7)
            mock_get_backend.assert_called()
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_parses_judge_response_with_preamble(self, mock_get_backend):
        """What: extracts the JSON block even when the model emits preamble text.
        Why: the parse logic must locate the first '{' (kept from the old path)."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            payload = json.dumps({"contradictions": [
                {"existing_claim_id": "n9", "explanation": "x", "confidence": 0.95},
            ]})
            mock_get_backend.return_value = _fake_backend(
                "Sure, here is my verdict:\n" + payload)
            result = contra.judge_contradictions("t", [{"node_id": "n9"}])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["existing_claim_id"], "n9")
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_parses_judge_response_with_trailing_text(self, mock_get_backend):
        """BUG-2026-06-11 judge-parse: a valid JSON object FOLLOWED BY trailing
        model commentary still parses (raw_decode stops at the closing brace).
        Why: ~95% of real judge outputs were discarded because json.loads choked
             on the appended prose ("Extra data") even though the JSON was valid."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            payload = json.dumps({"contradictions": [
                {"existing_claim_id": "n7", "explanation": "x", "confidence": 0.91},
            ]})
            mock_get_backend.return_value = _fake_backend(
                payload + "\n\nThis means the new claim conflicts. Hope that helps!")
            result = contra.judge_contradictions("t", [{"node_id": "n7"}])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["existing_claim_id"], "n7")
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_parses_judge_response_preamble_and_trailing(self, mock_get_backend):
        """Both a preamble AND trailing text around the JSON object parse."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            payload = json.dumps({"contradictions": [
                {"existing_claim_id": "n8", "explanation": "y", "confidence": 0.88},
            ]})
            mock_get_backend.return_value = _fake_backend(
                "Verdict:\n" + payload + "\nDone.")
            result = contra.judge_contradictions("t", [{"node_id": "n8"}])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["existing_claim_id"], "n8")
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_filters_low_confidence(self, mock_get_backend):
        """What: filters out contradictions below the confidence threshold.
        Why: low-confidence judgments are noise and should not flag writes."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend(json.dumps({
                "contradictions": [
                    {"existing_claim_id": "n1", "explanation": "maybe", "confidence": 0.3},
                    {"existing_claim_id": "n2", "explanation": "definitely", "confidence": 0.9},
                ]
            }))
            candidates = [
                {"node_id": "n1", "title": "A"},
                {"node_id": "n2", "title": "B"},
            ]
            result = contra.judge_contradictions("test", candidates)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["existing_claim_id"], "n2")
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_fail_soft_on_mock_backend(self, mock_get_backend):
        """What: judge FAILS SOFT (records-only empty) when get_backend() is a
                 MockBackend (the no-real-model fail-soft signal).
        Why: nothing must be auto-superseded off canned MockBackend text; this
             is the records-only posture that replaces 'llama-cli not on PATH'."""
        from samia.runtime.inference import MockBackend
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = MockBackend()
            result = contra.judge_contradictions("test", [{"node_id": "n1"}])
            self.assertEqual(result, [])
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_fail_soft_when_backend_raises(self, mock_get_backend):
        """What: judge FAILS SOFT (empty) when backend.complete() raises.
        Why: a backend error must never block or corrupt a write."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend(
                RuntimeError("model OOM"))
            result = contra.judge_contradictions("test", [{"node_id": "n1"}])
            self.assertEqual(result, [])
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend",
                side_effect=ImportError("llama_cpp absent"))
    def test_fail_soft_when_get_backend_raises(self, mock_get_backend):
        """What: judge FAILS SOFT (empty) when get_backend() itself raises.
        Why: an unloadable inference module must degrade to records-only."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            result = contra.judge_contradictions("test", [{"node_id": "n1"}])
            self.assertEqual(result, [])
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_fail_soft_on_junk_response(self, mock_get_backend):
        """What: judge FAILS SOFT (empty) when the backend returns non-JSON junk.
        Why: an unparseable verdict yields no contradictions (records-only)."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend("not json at all")
            result = contra.judge_contradictions("test", [{"node_id": "n1"}])
            self.assertEqual(result, [])
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_no_subprocess_used_for_judge(self, mock_get_backend):
        """What: asserts the judge does NOT shell out (no subprocess.run call),
                 i.e. a missing llama-cli binary no longer matters.
        Why: the rewire must leave NO dead llama-cli subprocess path. With
             subprocess.run patched to explode, the judge still completes via
             the in-process backend."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend(json.dumps({
                "contradictions": [
                    {"existing_claim_id": "nz", "explanation": "x", "confidence": 0.99},
                ]
            }))
            with mock.patch("subprocess.run",
                            side_effect=AssertionError(
                                "judge must not call subprocess.run")) as msr:
                result = contra.judge_contradictions("test", [{"node_id": "nz"}])
            msr.assert_not_called()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["existing_claim_id"], "nz")
        finally:
            contra._JUDGE_ENABLED = original

    def test_module_has_no_subprocess_attribute(self):
        """What: the contradiction module no longer imports subprocess.
        Why: a static guarantee that the dead llama-cli path is fully removed."""
        self.assertFalse(hasattr(contra, "subprocess"))


class TestSynthesizeNode(_NeutralDedicatedBackend, unittest.TestCase):
    """Tests for Tier-2 P2 abstraction synthesis (rewired to in-process backend)."""

    def test_returns_none_when_judge_flag_off(self):
        """What: synthesize_node returns None when the judge flag is off.
        Why: synthesis_enabled() gates on ASTHENOS_CONTRADICTION_JUDGE; off ->
             safe no-op so the merge consumer leaves the pair pending."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = False
            self.assertIsNone(contra.synthesize_node("a body", "b body"))
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_returns_title_body_when_mocked(self, mock_get_backend):
        """What: synthesize_node returns {title, body} from a mocked backend.
        Why: verifies the synthesis prompt -> backend.complete -> parse path."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend(json.dumps({
                "title": "Unified concept",
                "body": "Both notes describe the same higher-level idea.",
            }))
            out = contra.synthesize_node("note a text", "note b text")
            self.assertIsInstance(out, dict)
            self.assertEqual(out["title"], "Unified concept")
            self.assertIn("higher-level", out["body"])
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_synth_parses_with_trailing_text(self, mock_get_backend):
        """BUG-2026-06-11 judge-parse: synthesize_node also tolerates trailing
        commentary after the JSON object (same raw_decode fix as the judge)."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            payload = json.dumps({
                "title": "Unified concept",
                "body": "Both notes describe the same higher-level idea.",
            })
            mock_get_backend.return_value = _fake_backend(
                payload + "\n\nLet me know if you want it shorter.")
            out = contra.synthesize_node("note a", "note b")
            self.assertIsInstance(out, dict)
            self.assertEqual(out["title"], "Unified concept")
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_synth_returns_none_on_garbage(self, mock_get_backend):
        """Genuinely non-JSON synth output still falls back to None."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend("no json here at all")
            self.assertIsNone(contra.synthesize_node("a", "b"))
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_returns_none_on_mock_backend(self, mock_get_backend):
        """What: synthesize_node returns None when get_backend() is a MockBackend.
        Why: no real model -> safe no-op (records/pending), same fail-soft as judge."""
        from samia.runtime.inference import MockBackend
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = MockBackend()
            self.assertIsNone(contra.synthesize_node("a", "b"))
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_returns_none_when_backend_raises(self, mock_get_backend):
        """What: synthesize_node returns None when backend.complete() raises.
        Why: a synth error must never crash the merge consumer."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend(
                RuntimeError("model OOM"))
            self.assertIsNone(contra.synthesize_node("a", "b"))
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_returns_none_on_empty_body(self, mock_get_backend):
        """What: synthesize_node returns None when the synthesized body is empty.
        Why: an empty abstraction is not a usable node; leave the pair pending."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend(json.dumps({
                "title": "x", "body": "",
            }))
            self.assertIsNone(contra.synthesize_node("a", "b"))
        finally:
            contra._JUDGE_ENABLED = original


class TestSynthesisEnabled(_NeutralDedicatedBackend, unittest.TestCase):
    """Tests for synthesis_enabled() availability probe (rewired)."""

    def test_false_when_judge_flag_off(self):
        """What: False when ASTHENOS_CONTRADICTION_JUDGE is off.
        Why: the enable flag still gates synthesis."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = False
            self.assertFalse(contra.synthesis_enabled())
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_false_when_backend_is_mock(self, mock_get_backend):
        """What: False when flag is on but the backend is a MockBackend.
        Why: the availability probe now requires a REAL backend (not
             'is llama-cli on PATH')."""
        from samia.runtime.inference import MockBackend
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = MockBackend()
            self.assertFalse(contra.synthesis_enabled())
        finally:
            contra._JUDGE_ENABLED = original

    @mock.patch("samia.runtime.inference.get_backend")
    def test_true_when_flag_on_and_real_backend(self, mock_get_backend):
        """What: True when the flag is on AND a real backend is available.
        Why: this is the activated state (in-daemon, model configured)."""
        original = contra._JUDGE_ENABLED
        try:
            contra._JUDGE_ENABLED = True
            mock_get_backend.return_value = _fake_backend("ignored")
            self.assertTrue(contra.synthesis_enabled())
        finally:
            contra._JUDGE_ENABLED = original


class TestCheckContradiction(_NeutralDedicatedBackend, unittest.TestCase):
    """Tests for the orchestrator function."""

    def test_disabled_returns_empty(self):
        """What: returns empty tuples when disabled.
        Why: the default state must be no-op."""
        original = contra._ENABLED
        try:
            contra._ENABLED = False
            reasons, meta = contra.check_contradiction({"text": "anything"})
            self.assertEqual(reasons, [])
            self.assertEqual(meta, [])
        finally:
            contra._ENABLED = original

    @mock.patch.object(contra, "find_contradiction_candidates")
    def test_embedding_only_returns_reasons(self, mock_find):
        """What: returns embedding-based reasons when judge is disabled.
        Why: Phase 1 works standalone without Phase 2."""
        original_enabled = contra._ENABLED
        original_judge = contra._JUDGE_ENABLED
        try:
            contra._ENABLED = True
            contra._JUDGE_ENABLED = False
            mock_find.return_value = [
                {"node_id": "test_node", "title": "Test", "score": 0.82},
            ]
            reasons, meta = contra.check_contradiction({"text": "test"})
            self.assertEqual(len(reasons), 1)
            self.assertIn("contradiction_embedding", reasons[0])
            self.assertEqual(len(meta), 1)
            self.assertEqual(meta[0]["node_id"], "test_node")
        finally:
            contra._ENABLED = original_enabled
            contra._JUDGE_ENABLED = original_judge


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.runtime.test_contradiction
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD60 + FIX-2026-06-08 (judge/synth in-process inference rewire)
# Layer:      test (pytest)
# Role:       tests for samia.runtime.contradiction — embedding candidate finder,
#             LLM judge gate, Tier-2 synthesis, fail-soft posture, orchestrator
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.runtime.contradiction, samia.runtime.inference
# Exposes:    — (test module)
# Lines:      530
# --------------------------------------------------------------------------
