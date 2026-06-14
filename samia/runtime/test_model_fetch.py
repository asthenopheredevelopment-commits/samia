"""samia.runtime.test_model_fetch — tests for samia.runtime.model_fetch (self-fetching gguf weights).

Layer 1 (Owns / Depends):
    Owns:    unit tests for the model registry + fetch_model() download path +
             the get_backend_for_model self-fetch wire.
    Depends: samia.runtime.model_fetch, samia.runtime.inference, unittest,
             unittest.mock, tempfile.

Layer 2 (What / Why):
    What: covers (1) registry lookup by logical name / filename / path;
          (2) existing-path passthrough (no download); (3) autofetch-disabled
          refusal carries a manual instruction; (4) a mocked download (urlopen
          patched) writes atomically, verifies, promotes .part -> final, and on
          a verification/network failure cleans up the .part and leaves NO
          partial; (5) get_backend_for_model triggers a fetch when the model is
          missing, with the backend load itself mocked.
    Why:  the self-fetch feature must NEVER leave a partial file, must refuse
          cleanly when gated off, and must fail soft (MockBackend) when a fetch
          is impossible -- these tests pin all four contracts without touching
          the network or downloading real multi-GB weights.

Layer 3 (Changelog):
    2026-06-10  FEAT       Initial suite.
    2026-06-12  SEC        Added TestConsentProtocol: the autofetch gate now
                            routes through samia.core.netconsent.consent --
                            env-on+non-tty downloads silently, env-unset+non-tty
                            refuses with NO network, env-unset+tty prompts (y
                            proceeds / n refuses), env-off refuses both modes.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.runtime import model_fetch
from samia.runtime import inference


class _FakeResp:
    """Minimal urlopen() context-manager stand-in over an in-memory payload."""

    def __init__(self, payload: bytes, content_length: bool = True):
        self._buf = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))} if content_length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)


class TestRegistry(unittest.TestCase):
    def test_known_seeded(self):
        # The two seeded models are present with the required fields.
        self.assertIn("Qwen3-4B-Instruct-2507-Q4_K_M", model_fetch.KNOWN)
        self.assertIn("BitNet-b1.58-2B-4T-i2_s", model_fetch.KNOWN)
        q = model_fetch.KNOWN["Qwen3-4B-Instruct-2507-Q4_K_M"]
        self.assertEqual(q["license"], "Apache-2.0")
        self.assertTrue(q["url"].startswith("https://huggingface.co/unsloth/"))
        b = model_fetch.KNOWN["BitNet-b1.58-2B-4T-i2_s"]
        self.assertEqual(b["license"], "MIT")
        # i2_s is explicitly flagged as not loadable by llama-cpp-python.
        self.assertFalse(b["llama_cpp_loadable"])

    def test_lookup_by_logical_name(self):
        e = model_fetch._registry_entry("Qwen3-4B-Instruct-2507-Q4_K_M")
        self.assertIsNotNone(e)
        self.assertEqual(e["filename"], "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

    def test_lookup_by_filename_and_path(self):
        by_file = model_fetch._registry_entry("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        by_path = model_fetch._registry_entry(
            "/some/where/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        self.assertIsNotNone(by_file)
        self.assertIs(by_file, by_path)

    def test_lookup_unknown_is_none(self):
        self.assertIsNone(model_fetch._registry_entry("no-such-model"))


class TestExistingPathPassthrough(unittest.TestCase):
    def test_existing_path_returned_unchanged_no_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "already.gguf"
            p.write_bytes(b"present")
            # urlopen must NOT be called when the path already exists.
            with mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   side_effect=AssertionError("must not download")):
                out = model_fetch.fetch_model(str(p))
            self.assertEqual(out, p)


class TestAutofetchDisabled(unittest.TestCase):
    def test_disabled_refuses_with_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.dict(os.environ,
                                 {model_fetch.AUTOFETCH_ENV: "0"}):
                with self.assertRaises(model_fetch.ModelFetchError) as cm:
                    model_fetch.fetch_model("Qwen3-4B-Instruct-2507-Q4_K_M")
            msg = str(cm.exception)
            self.assertIn(model_fetch.AUTOFETCH_ENV, msg)
            # The refusal tells the operator how to proceed manually.
            self.assertIn("manually place", msg.lower())

    def test_unknown_model_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)):
                with self.assertRaises(model_fetch.ModelFetchError) as cm:
                    model_fetch.fetch_model("totally-unknown-model")
            self.assertIn("unknown model", str(cm.exception).lower())


class TestDownloadMocked(unittest.TestCase):
    def setUp(self):
        # Pin autofetch ON for these mocked-download tests: they exercise the
        # fetch path itself, so an ambient ASTHENOS_MODEL_AUTOFETCH=0 (a valid
        # user/kit setting — the cold-metal kit sets it for the whole suite)
        # must not turn them into kill-switch tests. Found cold-metal round 2.
        env = mock.patch.dict(os.environ, {model_fetch.AUTOFETCH_ENV: "1"})
        env.start()
        self.addCleanup(env.stop)

    def _entry_with_size_window(self, payload: bytes) -> dict:
        # A throwaway registry entry whose size window accepts the test payload.
        return {
            "url": "https://huggingface.co/test/repo/resolve/main/m.gguf",
            "filename": "m.gguf",
            "license": "Apache-2.0",
            "license_url": "https://huggingface.co/test/repo",
            "size_min": 1,
            "size_max": len(payload) + 10,
        }

    def test_download_atomic_write_and_verify(self):
        payload = b"x" * 4096
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry_with_size_window(payload)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   return_value=_FakeResp(payload)):
                out = model_fetch.fetch_model("m")
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), payload)
            self.assertEqual(out.name, "m.gguf")
            # No .part left behind after a clean download.
            self.assertFalse((Path(tmp) / "m.gguf.part").exists())

    def test_verification_failure_cleans_part_no_partial(self):
        # Payload too small for the entry's size window -> verify fails.
        payload = b"tiny"
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry_with_size_window(payload)
            entry["size_min"] = len(payload) + 100  # force "too small"
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   return_value=_FakeResp(payload)):
                with self.assertRaises(model_fetch.ModelFetchError):
                    model_fetch.fetch_model("m")
            # Neither the final file nor the .part survive a failed verify.
            self.assertFalse((Path(tmp) / "m.gguf").exists())
            self.assertFalse((Path(tmp) / "m.gguf.part").exists())

    def test_network_failure_cleans_part_no_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry_with_size_window(b"y" * 100)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   side_effect=OSError("connection reset")):
                with self.assertRaises(model_fetch.ModelFetchError) as cm:
                    model_fetch.fetch_model("m")
            self.assertIn("connection reset", str(cm.exception))
            self.assertFalse((Path(tmp) / "m.gguf").exists())
            self.assertFalse((Path(tmp) / "m.gguf.part").exists())

    def test_sha256_pinned_entry_verifies(self):
        import hashlib
        payload = b"deadbeef" * 64
        sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            entry = {
                "url": "https://huggingface.co/test/repo/resolve/main/m.gguf",
                "filename": "m.gguf",
                "license": "MIT",
                "sha256": sha,
            }
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   return_value=_FakeResp(payload)):
                out = model_fetch.fetch_model("m")
            self.assertEqual(out.read_bytes(), payload)

    def test_sha256_mismatch_rejected(self):
        payload = b"z" * 256
        with tempfile.TemporaryDirectory() as tmp:
            entry = {
                "url": "https://huggingface.co/test/repo/resolve/main/m.gguf",
                "filename": "m.gguf",
                "license": "MIT",
                "sha256": "00" * 32,  # wrong
            }
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   return_value=_FakeResp(payload)):
                with self.assertRaises(model_fetch.ModelFetchError) as cm:
                    model_fetch.fetch_model("m")
            self.assertIn("sha256 mismatch", str(cm.exception))
            self.assertFalse((Path(tmp) / "m.gguf").exists())
            self.assertFalse((Path(tmp) / "m.gguf.part").exists())


class TestConsentProtocol(unittest.TestCase):
    """The download gate now routes through samia.core.netconsent.consent.

    These tests drive fetch_model() across the consent decision matrix WITHOUT
    pinning AUTOFETCH=1 in setUp (unlike TestDownloadMocked) so the env state is
    exactly what each case sets. urlopen is patched everywhere; a download is
    proven by the file landing (or a verify/network mock firing), a refusal by
    urlopen NEVER being called.
    """

    def _entry(self, payload: bytes) -> dict:
        return {
            "url": "https://huggingface.co/test/repo/resolve/main/m.gguf",
            "filename": "m.gguf",
            "license": "Apache-2.0",
            "license_url": "https://huggingface.co/test/repo",
            "size_min": 1,
            "size_max": len(payload) + 10,
        }

    def _env_without_knob(self) -> dict:
        return {k: v for k, v in os.environ.items()
                if k != model_fetch.AUTOFETCH_ENV}

    def test_env_on_non_tty_downloads_without_prompt(self):
        # (a) env explicitly "1" + non-tty -> downloads (mocked), NO prompt.
        payload = b"q" * 2048
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(payload)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.dict(os.environ,
                                 {model_fetch.AUTOFETCH_ENV: "1"}), \
                 mock.patch("sys.stdin.isatty", return_value=False), \
                 mock.patch("builtins.input",
                            side_effect=AssertionError("must not prompt")), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   return_value=_FakeResp(payload)) as uo:
                out = model_fetch.fetch_model("m")
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), payload)
            uo.assert_called_once()

    def test_env_unset_non_tty_refuses_no_network(self):
        # (b) env unset + non-tty -> refused, NO network attempted.
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(b"q" * 8)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.dict(os.environ, self._env_without_knob(),
                                 clear=True), \
                 mock.patch("sys.stdin.isatty", return_value=False), \
                 mock.patch("builtins.input",
                            side_effect=AssertionError("must not prompt")), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   side_effect=AssertionError(
                                       "must not download")) as uo:
                with self.assertRaises(model_fetch.ModelFetchError) as cm:
                    model_fetch.fetch_model("m")
            uo.assert_not_called()
            # Refusal carries a manual remedy.
            self.assertIn("manually place", str(cm.exception).lower())

    def test_env_unset_tty_yes_proceeds(self):
        # (c) env unset + tty + input 'y' -> proceeds (downloads).
        payload = b"q" * 1024
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(payload)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.dict(os.environ, self._env_without_knob(),
                                 clear=True), \
                 mock.patch("sys.stdin.isatty", return_value=True), \
                 mock.patch("builtins.input", return_value="y"), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   return_value=_FakeResp(payload)) as uo:
                out = model_fetch.fetch_model("m")
            self.assertTrue(out.exists())
            uo.assert_called_once()

    def test_env_unset_tty_no_refuses(self):
        # (d) env unset + tty + input 'n' -> refused, NO network.
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(b"q" * 8)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.dict(os.environ, self._env_without_knob(),
                                 clear=True), \
                 mock.patch("sys.stdin.isatty", return_value=True), \
                 mock.patch("builtins.input", return_value="n"), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   side_effect=AssertionError(
                                       "must not download")) as uo:
                with self.assertRaises(model_fetch.ModelFetchError):
                    model_fetch.fetch_model("m")
            uo.assert_not_called()

    def test_env_off_refuses_non_tty(self):
        # (e) env "0" -> refused in non-tty mode.
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(b"q" * 8)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.dict(os.environ,
                                 {model_fetch.AUTOFETCH_ENV: "0"}), \
                 mock.patch("sys.stdin.isatty", return_value=False), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   side_effect=AssertionError(
                                       "must not download")) as uo:
                with self.assertRaises(model_fetch.ModelFetchError):
                    model_fetch.fetch_model("m")
            uo.assert_not_called()

    def test_env_off_refuses_tty_without_prompt(self):
        # (e) env "0" -> refused even at a tty, with NO prompt (kill switch).
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(b"q" * 8)
            with mock.patch.object(model_fetch, "MODELS_DIR", Path(tmp)), \
                 mock.patch.object(model_fetch, "KNOWN", {"m": entry}), \
                 mock.patch.dict(os.environ,
                                 {model_fetch.AUTOFETCH_ENV: "0"}), \
                 mock.patch("sys.stdin.isatty", return_value=True), \
                 mock.patch("builtins.input",
                            side_effect=AssertionError("must not prompt")), \
                 mock.patch.object(model_fetch.urllib.request, "urlopen",
                                   side_effect=AssertionError(
                                       "must not download")) as uo:
                with self.assertRaises(model_fetch.ModelFetchError):
                    model_fetch.fetch_model("m")
            uo.assert_not_called()


class TestGetBackendSelfFetch(unittest.TestCase):
    def setUp(self):
        # Drop any cached backend so the fetch path is exercised fresh.
        inference._model_backend_cache.clear()
        # Pin autofetch ON (see TestDownloadMocked.setUp for why).
        env = mock.patch.dict(os.environ, {model_fetch.AUTOFETCH_ENV: "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_missing_model_triggers_fetch_then_loads(self):
        # The requested path is missing; fetch_model returns a real on-disk
        # .gguf; the backend load itself is mocked so no llama_cpp is needed.
        with tempfile.TemporaryDirectory() as tmp:
            fetched = Path(tmp) / "fetched.gguf"
            fetched.write_bytes(b"GGUF-bytes")
            sentinel = object()
            with mock.patch("samia.runtime.model_fetch.fetch_model",
                            return_value=fetched) as mfetch, \
                 mock.patch.object(inference, "LlamaCppBackend",
                                   return_value=sentinel) as mbackend, \
                 mock.patch("importlib.import_module"):
                out = inference.get_backend_for_model(str(Path(tmp) / "missing.gguf"))
            mfetch.assert_called_once()
            mbackend.assert_called_once()
            self.assertIs(out, sentinel)

    def test_fetch_failure_falls_soft_to_mock(self):
        # When self-fetch raises, the backend resolver returns MockBackend
        # (no regression vs the prior missing-model behavior).
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("samia.runtime.model_fetch.fetch_model",
                            side_effect=model_fetch.ModelFetchError("nope")):
                out = inference.get_backend_for_model(str(Path(tmp) / "missing.gguf"))
            self.assertIsInstance(out, inference.MockBackend)

    def test_existing_model_path_does_not_fetch(self):
        # An existing .gguf must never trigger a fetch.
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.gguf"
            real.write_bytes(b"GGUF")
            sentinel = object()
            with mock.patch("samia.runtime.model_fetch.fetch_model",
                            side_effect=AssertionError("must not fetch")), \
                 mock.patch.object(inference, "LlamaCppBackend",
                                   return_value=sentinel), \
                 mock.patch("importlib.import_module"):
                out = inference.get_backend_for_model(str(real))
            self.assertIs(out, sentinel)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.test_model_fetch
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-10 self-fetch + SEC-2026-06-12 consent protocol
# Layer:      test (pytest)
# Role:       tests for samia.runtime.model_fetch, samia.runtime.inference —
#             registry lookup, existing-path passthrough, autofetch-gate refusal,
#             atomic/verified mocked download, consent matrix, self-fetch wire
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.runtime.model_fetch, samia.runtime.inference
# Exposes:    — (test module)
# Lines:      429
# --------------------------------------------------------------------------
