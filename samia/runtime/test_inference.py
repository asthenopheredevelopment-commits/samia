"""samia.runtime.test_inference — tests for samia.runtime.inference (backend protocol, mock, factory, IPC ops, telemetry).

Layer 1 (Owns / Depends):
    Owns:    inference unit tests, telemetry unit tests
    Depends: samia.runtime.inference, samia.runtime.ipc, unittest

Layer 2 (What / Why):
    What: Tests for MockBackend determinism, compare() tuple shape,
          get_backend() fallback logic (no env, no llama_cpp, missing model),
          IPC op registration and response shapes for judge/infer/inference_status,
          LlamaCppBackend skip-if-unavailable guard, TelemetryEmitter JSONL
          write/shape/error-resilience, telemetry emission from judge/infer ops,
          caller_hint propagation, inference_telemetry_status op.
    Why:  Inference is the foundation for AUD16-compatible contradiction
          detection.  Mock correctness ensures the runtime starts clean;
          factory tests verify graceful degradation across all failure modes.
          Telemetry tests (AUD28-28.2) ensure every inference call is logged
          for cost-attribution and that emission failures never disrupt ops.

AUD26 Phase 26.3 -- in-runtime inference.
AUD28 Phase 28.2 -- daemon telemetry emission.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time

# Real-backend test gate — needs llama-cpp-python installed AND a gguf model on
# disk (path in $SAMIA_TEST_GGUF). CI provides both; locally it skips unless set.
try:
    import llama_cpp as _llama_cpp  # noqa: F401
    _HAS_LLAMA_CPP = True
except Exception:
    _HAS_LLAMA_CPP = False
_TEST_GGUF = os.environ.get("SAMIA_TEST_GGUF", "")
_TEST_GGUF = _TEST_GGUF if (_TEST_GGUF and os.path.isfile(_TEST_GGUF)) else ""
import unittest
from pathlib import Path
from unittest import mock

from samia.runtime.inference import (
    MockBackend,
    LlamaCppBackend,
    TelemetryEmitter,
    get_backend,
    register_ops,
    _op_judge,
    _op_infer,
    _op_inference_status,
    _op_inference_telemetry_status,
    _build_telemetry_event,
    _deterministic_hash,
)


# ---------------------------------------------------------------------------
# Backend-registry isolation mixin
# ---------------------------------------------------------------------------


class _BackendRegistryIsolation(unittest.TestCase):
    """Snapshot + clear the inference module's GLOBAL backend caches around every
    test, restoring them afterward.

    What: clears inference._backend_registry and inference._model_backend_cache in
      setUp and restores the snapshot via addCleanup.
    Why: these two dicts are module-global. In an [llm]-present env, sibling tests
      (or any register_ops()/get_backend_for_model() call elsewhere in the suite)
      can leave a real backend keyed under "llama_cpp"/"bitnet"/a gguf path. The
      telemetry path resolves the backend through _resolve_backend() WHENEVER
      _backend_registry is non-empty (see _build_telemetry_event), so a leaked
      entry silently shadows the MockBackend a test monkeypatches onto
      inference._backend -- making the test's `backend`/`model_path` assertions
      depend on suite order. Clearing the registries per test makes every
      backend-sensitive test pass in BOTH llm and no-llm envs, any pytest order.
    """

    def setUp(self) -> None:
        super().setUp()
        import samia.runtime.inference as _inf
        _saved_reg = dict(_inf._backend_registry)
        _saved_cache = dict(_inf._model_backend_cache)
        _inf._backend_registry.clear()
        _inf._model_backend_cache.clear()

        def _restore() -> None:
            _inf._backend_registry.clear()
            _inf._backend_registry.update(_saved_reg)
            _inf._model_backend_cache.clear()
            _inf._model_backend_cache.update(_saved_cache)

        self.addCleanup(_restore)


# ---------------------------------------------------------------------------
# MockBackend tests
# ---------------------------------------------------------------------------


class TestMockBackendCompleteDeterministic(unittest.TestCase):
    """MockBackend.complete() returns the same output for the same prompt."""

    def test_same_prompt_same_result(self):
        backend = MockBackend()
        result_a = backend.complete("What is 6 times 7?")
        result_b = backend.complete("What is 6 times 7?")
        self.assertEqual(result_a, result_b)

    def test_different_prompts_may_differ(self):
        backend = MockBackend()
        # Not guaranteed to differ, but with different hashes
        # at least we know the hash path works.
        r1 = backend.complete("prompt alpha")
        r2 = backend.complete("prompt beta")
        # Both should be strings from the canned set.
        self.assertIsInstance(r1, str)
        self.assertIsInstance(r2, str)
        self.assertTrue(len(r1) > 0)

    def test_returns_string(self):
        backend = MockBackend()
        result = backend.complete("test")
        self.assertIsInstance(result, str)


class TestMockBackendCompareReturnsTuple(unittest.TestCase):
    """MockBackend.compare() returns (bool, str, float)."""

    def test_compare_shape(self):
        backend = MockBackend()
        result = backend.compare("The sky is blue", "The sky is green")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        contradicts, rationale, confidence = result
        self.assertIsInstance(contradicts, bool)
        self.assertIsInstance(rationale, str)
        self.assertIsInstance(confidence, float)

    def test_compare_deterministic(self):
        backend = MockBackend()
        r1 = backend.compare("A", "B")
        r2 = backend.compare("A", "B")
        self.assertEqual(r1, r2)

    def test_compare_confidence_range(self):
        backend = MockBackend()
        _, _, confidence = backend.compare("fact one", "fact two")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_is_loaded(self):
        backend = MockBackend()
        self.assertTrue(backend.is_loaded())


# ---------------------------------------------------------------------------
# get_backend() factory tests
# ---------------------------------------------------------------------------


class TestGetBackendFallsBackToMockWhenNoEnv(_BackendRegistryIsolation):
    """get_backend() returns MockBackend when ASTHENOS_INFERENCE_MODEL is unset."""

    def test_no_env_returns_mock(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure the env var is absent.
            os.environ.pop("ASTHENOS_INFERENCE_MODEL", None)
            backend = get_backend()
        self.assertIsInstance(backend, MockBackend)


class TestGetBackendFallsBackToMockWhenLlamaCppAbsent(_BackendRegistryIsolation):
    """get_backend() returns MockBackend when llama_cpp cannot be imported."""

    def test_import_error_returns_mock(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            fake_model = f.name
            f.write(b"fake gguf data")
        try:
            env = {"ASTHENOS_INFERENCE_MODEL": fake_model}
            with mock.patch.dict(os.environ, env):
                # Make importlib.import_module("llama_cpp") raise ImportError.
                with mock.patch("importlib.import_module", side_effect=ImportError("no llama_cpp")):
                    backend = get_backend()
            self.assertIsInstance(backend, MockBackend)
        finally:
            os.unlink(fake_model)


class TestGetBackendFallsBackToMockWhenModelPathMissing(_BackendRegistryIsolation):
    """get_backend() returns MockBackend when model path does not exist."""

    def test_missing_path_returns_mock(self):
        env = {"ASTHENOS_INFERENCE_MODEL": "/nonexistent/path/model.gguf"}
        with mock.patch.dict(os.environ, env):
            backend = get_backend()
        self.assertIsInstance(backend, MockBackend)


class TestGetBackendFallsBackToMockWhenNotGguf(_BackendRegistryIsolation):
    """get_backend() returns MockBackend when model path is not a .gguf file."""

    def test_non_gguf_returns_mock(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            fake_model = f.name
        try:
            env = {"ASTHENOS_INFERENCE_MODEL": fake_model}
            with mock.patch.dict(os.environ, env):
                backend = get_backend()
            self.assertIsInstance(backend, MockBackend)
        finally:
            os.unlink(fake_model)


# ---------------------------------------------------------------------------
# IPC op registration tests
# ---------------------------------------------------------------------------


class TestRegisterOpsAddsJudgeInferStatus(unittest.TestCase):
    """register_ops() registers exactly the three expected IPC ops."""

    def test_ops_registered(self):
        from samia.runtime import ipc
        from samia.runtime import inference as inf
        # Save and restore BOTH the op registry AND the backend registry to
        # avoid cross-test pollution. register_ops() populates
        # inference._backend_registry (e.g. a leaked MockBackend/BitNetBackend);
        # _resolve_backend prefers any registry entry over the module-level
        # _backend that later telemetry tests monkeypatch, so a leaked entry
        # silently shadows their broken backend (see the intra-file pollution
        # finding, 2026-06-11).
        saved = dict(ipc._op_registry)
        saved_be = dict(inf._backend_registry)
        try:
            # Clear any previous inference registrations.
            ipc._op_registry.pop("judge", None)
            ipc._op_registry.pop("infer", None)
            ipc._op_registry.pop("inference_status", None)
            ipc._op_registry.pop("inference_telemetry_status", None)

            backend = register_ops()
            self.assertIn("judge", ipc._op_registry)
            self.assertIn("infer", ipc._op_registry)
            self.assertIn("inference_status", ipc._op_registry)
            self.assertIn("inference_telemetry_status", ipc._op_registry)
            self.assertIsInstance(backend, MockBackend)
        finally:
            ipc._op_registry.clear()
            ipc._op_registry.update(saved)
            inf._backend_registry.clear()
            inf._backend_registry.update(saved_be)


# ---------------------------------------------------------------------------
# IPC op shape tests (calling handlers directly with MockBackend)
# ---------------------------------------------------------------------------


class TestJudgeOpViaMockReturnsAUD16ProtocolShape(_BackendRegistryIsolation):
    """judge op returns {contradicts, rationale, confidence} -- AUD16-compatible."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        inf._backend = MockBackend()

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend

    def test_judge_shape(self):
        result = _op_judge({"fact_a": "The sky is blue", "fact_b": "The sky is green"})
        self.assertIn("contradicts", result)
        self.assertIn("rationale", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["contradicts"], bool)
        self.assertIsInstance(result["rationale"], str)
        self.assertIsInstance(result["confidence"], float)

    def test_judge_requires_facts(self):
        with self.assertRaises(ValueError):
            _op_judge({"fact_a": "", "fact_b": "something"})
        with self.assertRaises(ValueError):
            _op_judge({"fact_a": "something", "fact_b": ""})
        with self.assertRaises(ValueError):
            _op_judge({})


class TestInferOpReturnsTextField(_BackendRegistryIsolation):
    """infer op returns {text: str}."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        inf._backend = MockBackend()

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend

    def test_infer_shape(self):
        result = _op_infer({"prompt": "Hello world"})
        self.assertIn("text", result)
        self.assertIsInstance(result["text"], str)
        self.assertTrue(len(result["text"]) > 0)

    def test_infer_requires_prompt(self):
        with self.assertRaises(ValueError):
            _op_infer({"prompt": ""})
        with self.assertRaises(ValueError):
            _op_infer({})


class TestInferenceStatusOpReportsBackend(_BackendRegistryIsolation):
    """inference_status op returns {backend, model_path, loaded}."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        inf._backend = MockBackend()

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend

    def test_status_shape(self):
        result = _op_inference_status({})
        self.assertEqual(result["backend"], "MockBackend")
        self.assertIsNone(result["model_path"])
        self.assertTrue(result["loaded"])


# ---------------------------------------------------------------------------
# LlamaCppBackend skip-if-unavailable
# ---------------------------------------------------------------------------


class TestLlamaCppBackendSkippedIfUnavailable(unittest.TestCase):
    """LlamaCppBackend constructor raises FileNotFoundError for missing model."""

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            LlamaCppBackend(model_path="/nonexistent/model.gguf")

    def test_non_gguf_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                LlamaCppBackend(model_path=path)
        finally:
            os.unlink(path)

    @unittest.skipUnless(_HAS_LLAMA_CPP and bool(_TEST_GGUF),
                         "needs llama-cpp-python + a gguf at $SAMIA_TEST_GGUF")
    def test_real_backend_loads(self):
        # Real-backend smoke: load an actual gguf and generate a few tokens.
        # Generic over the model — point SAMIA_TEST_GGUF at any gguf (a small LLM
        # today, a larger gguf later). CI provides a tiny model so this runs on
        # every push and exercises the real llama_cpp load+complete path.
        backend = LlamaCppBackend(model_path=_TEST_GGUF, n_ctx=512)
        out = backend.complete("The capital of France is", max_tokens=8)
        self.assertIsInstance(out, str)


# ---------------------------------------------------------------------------
# Telemetry tests (AUD28-28.2)
# ---------------------------------------------------------------------------


class TestTelemetryEmitterWritesJsonl(unittest.TestCase):
    """TelemetryEmitter.emit() writes valid JSONL to today's file."""

    def test_writes_one_line(self):
        with tempfile.TemporaryDirectory() as td:
            emitter = TelemetryEmitter(events_dir=Path(td))
            emitter.emit({"op": "judge", "success": True})
            today_path = emitter._today_path()
            self.assertTrue(today_path.exists())
            with open(today_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["op"], "judge")

    def test_appends_multiple_lines(self):
        with tempfile.TemporaryDirectory() as td:
            emitter = TelemetryEmitter(events_dir=Path(td))
            for i in range(5):
                emitter.emit({"seq": i})
            with open(emitter._today_path()) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 5)


class TestTelemetryEventShapeHasRequiredFields(_BackendRegistryIsolation):
    """_build_telemetry_event() returns all fields from the canonical schema."""

    _REQUIRED = {
        "ts", "op", "caller_hint", "prompt_chars", "response_chars",
        "latency_ms", "backend", "model_path", "success", "error",
    }

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved = inf._backend
        inf._backend = MockBackend()

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved

    def test_all_keys_present(self):
        event = _build_telemetry_event(
            "infer", {},
            prompt_chars=100, response_chars=50,
            latency_ms=12.3, success=True, error=None,
        )
        self.assertTrue(self._REQUIRED.issubset(event.keys()),
                        f"Missing keys: {self._REQUIRED - event.keys()}")

    def test_ts_is_iso_format(self):
        event = _build_telemetry_event(
            "judge", {},
            prompt_chars=10, response_chars=5,
            latency_ms=1.0, success=True, error=None,
        )
        # Should parse as a valid datetime.
        import datetime
        datetime.datetime.fromisoformat(event["ts"].replace("Z", "+00:00"))


class TestTelemetryEmitFailureDoesNotRaise(unittest.TestCase):
    """TelemetryEmitter.emit() swallows errors silently."""

    def test_unwritable_dir_does_not_raise(self):
        # Point at a path that cannot be created (under /proc).
        emitter = TelemetryEmitter(events_dir=Path("/proc/fake_asthenos_events"))
        # Must not raise.
        emitter.emit({"op": "test", "success": True})


class TestJudgeOpEmitsTelemetryOnSuccess(_BackendRegistryIsolation):
    """judge op emits a telemetry event on successful completion."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        self._saved_emitter = inf._emitter
        inf._backend = MockBackend()
        self._td = tempfile.TemporaryDirectory()
        inf._emitter = TelemetryEmitter(events_dir=Path(self._td.name))

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend
        inf._emitter = self._saved_emitter
        self._td.cleanup()

    def test_event_emitted(self):
        import samia.runtime.inference as inf
        _op_judge({"fact_a": "Sky is blue", "fact_b": "Sky is green"})
        today = inf._emitter._today_path()
        self.assertTrue(today.exists())
        with open(today) as f:
            event = json.loads(f.readline())
        self.assertEqual(event["op"], "judge")
        self.assertTrue(event["success"])
        self.assertIsNone(event["error"])
        self.assertGreater(event["prompt_chars"], 0)
        self.assertGreaterEqual(event["latency_ms"], 0)


class TestJudgeOpEmitsTelemetryOnError(_BackendRegistryIsolation):
    """judge op emits a telemetry event even when the backend raises."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        self._saved_emitter = inf._emitter
        # Use a backend whose compare() always raises.
        broken = MockBackend()
        broken.compare = lambda a, b: (_ for _ in ()).throw(RuntimeError("boom"))
        inf._backend = broken
        self._td = tempfile.TemporaryDirectory()
        inf._emitter = TelemetryEmitter(events_dir=Path(self._td.name))

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend
        inf._emitter = self._saved_emitter
        self._td.cleanup()

    def test_error_event_emitted(self):
        import samia.runtime.inference as inf
        with self.assertRaises(RuntimeError):
            _op_judge({"fact_a": "A", "fact_b": "B"})
        today = inf._emitter._today_path()
        self.assertTrue(today.exists())
        with open(today) as f:
            event = json.loads(f.readline())
        self.assertEqual(event["op"], "judge")
        self.assertFalse(event["success"])
        self.assertIn("boom", event["error"])


class TestInferOpEmitsTelemetry(_BackendRegistryIsolation):
    """infer op emits a telemetry event on success."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        self._saved_emitter = inf._emitter
        inf._backend = MockBackend()
        self._td = tempfile.TemporaryDirectory()
        inf._emitter = TelemetryEmitter(events_dir=Path(self._td.name))

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend
        inf._emitter = self._saved_emitter
        self._td.cleanup()

    def test_event_emitted(self):
        import samia.runtime.inference as inf
        _op_infer({"prompt": "Hello world"})
        today = inf._emitter._today_path()
        self.assertTrue(today.exists())
        with open(today) as f:
            event = json.loads(f.readline())
        self.assertEqual(event["op"], "infer")
        self.assertTrue(event["success"])
        self.assertEqual(event["prompt_chars"], len("Hello world"))


class TestCallerHintPropagated(_BackendRegistryIsolation):
    """caller_hint from request args appears in the telemetry event."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        self._saved_emitter = inf._emitter
        inf._backend = MockBackend()
        self._td = tempfile.TemporaryDirectory()
        inf._emitter = TelemetryEmitter(events_dir=Path(self._td.name))

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend
        inf._emitter = self._saved_emitter
        self._td.cleanup()

    def test_hint_in_event(self):
        import samia.runtime.inference as inf
        _op_judge({
            "fact_a": "X",
            "fact_b": "Y",
            "caller_hint": "oracle.judge_pass",
        })
        with open(inf._emitter._today_path()) as f:
            event = json.loads(f.readline())
        self.assertEqual(event["caller_hint"], "oracle.judge_pass")

    def test_hint_truncated_to_64_chars(self):
        import samia.runtime.inference as inf
        long_hint = "a" * 200
        _op_infer({"prompt": "test", "caller_hint": long_hint})
        with open(inf._emitter._today_path()) as f:
            event = json.loads(f.readline())
        self.assertEqual(len(event["caller_hint"]), 64)

    def test_no_hint_yields_null(self):
        import samia.runtime.inference as inf
        _op_infer({"prompt": "test"})
        with open(inf._emitter._today_path()) as f:
            event = json.loads(f.readline())
        self.assertIsNone(event["caller_hint"])


class TestInferenceTelemetryStatusOp(_BackendRegistryIsolation):
    """inference_telemetry_status op returns correct dir stats."""

    def setUp(self):
        super().setUp()
        import samia.runtime.inference as inf
        self._saved_backend = inf._backend
        self._saved_emitter = inf._emitter
        inf._backend = MockBackend()
        self._td = tempfile.TemporaryDirectory()
        inf._emitter = TelemetryEmitter(events_dir=Path(self._td.name))

    def tearDown(self):
        import samia.runtime.inference as inf
        inf._backend = self._saved_backend
        inf._emitter = self._saved_emitter
        self._td.cleanup()

    def test_status_shape(self):
        result = _op_inference_telemetry_status({})
        self.assertIn("events_dir", result)
        self.assertIn("today_count", result)
        self.assertIn("today_path", result)
        self.assertIn("today_size_bytes", result)
        self.assertIn("total_files", result)

    def test_count_increments(self):
        import samia.runtime.inference as inf
        _op_infer({"prompt": "one"})
        _op_infer({"prompt": "two"})
        result = _op_inference_telemetry_status({})
        self.assertEqual(result["today_count"], 2)
        self.assertGreater(result["today_size_bytes"], 0)
        self.assertEqual(result["total_files"], 1)


if __name__ == "__main__":
    unittest.main()

# [Asthenosphere] samia.runtime.test_inference
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD26.3 (in-runtime inference) + AUD28-28.2 (daemon telemetry)
# Layer:      test (pytest)
# Role:       tests for samia.runtime.inference — MockBackend determinism,
#             get_backend() fallbacks, IPC op registration/shapes, LlamaCpp guard,
#             telemetry emission/shape/resilience, caller_hint propagation
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.runtime.inference, samia.runtime.ipc
# Exposes:    — (test module)
# Lines:      633
# --------------------------------------------------------------------------
