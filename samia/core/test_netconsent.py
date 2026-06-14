"""samia.core.test_netconsent — tests for samia.core.netconsent.consent + helpers.

Layer 1 (Owns / Depends):
    Owns:    the consent-protocol decision-matrix tests (kill switch / standing
             consent / ask-if-tty / refuse-non-tty) + the env interpretation
             helper tests (env_explicit_on / env_explicit_off).
    Depends: samia.core.netconsent, unittest, unittest.mock.

Layer 2 (What / Why):
    What: pins the exact operator-approved semantics -- one knob
          (ASTHENOS_MODEL_AUTOFETCH): explicitly-off refuses in BOTH tty and
          non-tty modes; explicitly-on proceeds with no prompt; UNSET asks at a
          tty (y -> proceed, n/EOF -> refuse) and refuses with no tty while
          naming BOTH remedies. No branch touches the network -- consent() only
          decides; it never downloads. Also pins the core/vector integration:
          a cache HIT (local_files_only load succeeds) NEVER calls consent (the
          common path stays silent/fast); a cache MISS gates through consent and
          a declined fetch raises a RuntimeError naming the env remedy.
    Why:  the v1 release directive is "no silent downloads, ever". These tests
          lock the reversal of the old default-ON behavior so it cannot regress:
          an unset env must NEVER silently authorize a download.

Layer 3 (Changelog):
    2026-06-12  SEC        Initial suite for the consent protocol.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from samia.core import netconsent

ENV = netconsent.AUTOFETCH_ENV

# Standard consent() args reused across cases (content is irrelevant to the
# decision; only the env + tty state drive the branch).
_ARGS = dict(
    what="Test-Model",
    size_hint="~10MB",
    license_str="Apache-2.0",
    source="huggingface.co",
    manual_hint="place it at /tmp/x or set the env",
)


def _unset_env():
    """Patch os.environ with the autofetch knob REMOVED (the unset posture)."""
    return mock.patch.dict(os.environ, {}, clear=False) if ENV not in os.environ \
        else mock.patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                          if k != ENV}, clear=True)


class TestEnvHelpers(unittest.TestCase):
    def test_explicit_off_values(self):
        for v in ("0", "false", "no", "off", "OFF", " False "):
            with mock.patch.dict(os.environ, {ENV: v}):
                self.assertTrue(netconsent.env_explicit_off(), v)
                self.assertFalse(netconsent.env_explicit_on(), v)

    def test_explicit_on_values(self):
        for v in ("1", "true", "yes", "on", "ON", " Yes "):
            with mock.patch.dict(os.environ, {ENV: v}):
                self.assertTrue(netconsent.env_explicit_on(), v)
                self.assertFalse(netconsent.env_explicit_off(), v)

    def test_unset_is_neither(self):
        env = {k: v for k, v in os.environ.items() if k != ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(netconsent.env_explicit_on())
            self.assertFalse(netconsent.env_explicit_off())

    def test_empty_and_unrecognized_are_neither(self):
        # An empty / garbage value is present-but-meaningless: NOT a clear
        # on/off signal, so it falls through to the ask-if-tty default.
        for v in ("", "  ", "maybe", "2"):
            with mock.patch.dict(os.environ, {ENV: v}):
                self.assertFalse(netconsent.env_explicit_on(), repr(v))
                self.assertFalse(netconsent.env_explicit_off(), repr(v))


class TestKillSwitch(unittest.TestCase):
    """env explicitly off -> refuse in BOTH tty and non-tty modes, no prompt."""

    def test_off_non_tty_refuses(self):
        with mock.patch.dict(os.environ, {ENV: "0"}), \
             mock.patch("sys.stdin.isatty", return_value=False), \
             mock.patch("builtins.input",
                        side_effect=AssertionError("must not prompt")):
            self.assertFalse(netconsent.consent(**_ARGS))

    def test_off_tty_still_refuses_without_prompt(self):
        # Even at a tty, the kill switch refuses with NO prompt.
        with mock.patch.dict(os.environ, {ENV: "off"}), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input",
                        side_effect=AssertionError("must not prompt")):
            self.assertFalse(netconsent.consent(**_ARGS))


class TestStandingConsent(unittest.TestCase):
    """env explicitly on -> proceed with NO prompt, even non-tty (CI/agent)."""

    def test_on_non_tty_proceeds_no_prompt(self):
        with mock.patch.dict(os.environ, {ENV: "1"}), \
             mock.patch("sys.stdin.isatty", return_value=False), \
             mock.patch("builtins.input",
                        side_effect=AssertionError("must not prompt")):
            self.assertTrue(netconsent.consent(**_ARGS))

    def test_on_tty_proceeds_no_prompt(self):
        with mock.patch.dict(os.environ, {ENV: "true"}), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input",
                        side_effect=AssertionError("must not prompt")):
            self.assertTrue(netconsent.consent(**_ARGS))


class TestAskIfInteractive(unittest.TestCase):
    """env unset -> ask at a tty (y proceeds, n/EOF refuses)."""

    def _env_without_knob(self):
        return {k: v for k, v in os.environ.items() if k != ENV}

    def test_unset_tty_yes_proceeds(self):
        with mock.patch.dict(os.environ, self._env_without_knob(), clear=True), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value="y") as inp:
            self.assertTrue(netconsent.consent(**_ARGS))
            inp.assert_called_once()

    def test_unset_tty_yes_word_proceeds(self):
        with mock.patch.dict(os.environ, self._env_without_knob(), clear=True), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value="YES"):
            self.assertTrue(netconsent.consent(**_ARGS))

    def test_unset_tty_no_refuses(self):
        with mock.patch.dict(os.environ, self._env_without_knob(), clear=True), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value="n"):
            self.assertFalse(netconsent.consent(**_ARGS))

    def test_unset_tty_empty_refuses(self):
        # Default is No: a bare Enter refuses.
        with mock.patch.dict(os.environ, self._env_without_knob(), clear=True), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value=""):
            self.assertFalse(netconsent.consent(**_ARGS))

    def test_unset_tty_eof_refuses(self):
        # input() raising EOFError (piped/closed stdin under a tty fake) refuses.
        with mock.patch.dict(os.environ, self._env_without_knob(), clear=True), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(netconsent.consent(**_ARGS))


class TestRefuseNonInteractive(unittest.TestCase):
    """env unset + no tty -> refuse, naming BOTH remedies, no network."""

    def test_unset_non_tty_refuses(self):
        env = {k: v for k, v in os.environ.items() if k != ENV}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("sys.stdin.isatty", return_value=False), \
             mock.patch("builtins.input",
                        side_effect=AssertionError("must not prompt")):
            self.assertFalse(netconsent.consent(**_ARGS))

    def test_unset_non_tty_message_names_both_remedies(self):
        import io
        import contextlib

        env = {k: v for k, v in os.environ.items() if k != ENV}
        buf = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("sys.stdin.isatty", return_value=False), \
             contextlib.redirect_stderr(buf):
            netconsent.consent(**_ARGS)
        msg = buf.getvalue()
        # Both remedies: the env export AND the manual hint.
        self.assertIn(ENV, msg)
        self.assertIn("place it at /tmp/x", msg)


class TestNoticeAlwaysPrints(unittest.TestCase):
    """The what/size/license/source notice prints on every branch."""

    def _notice_for(self, env_val):
        import io
        import contextlib

        if env_val is None:
            env = {k: v for k, v in os.environ.items() if k != ENV}
            ctx = mock.patch.dict(os.environ, env, clear=True)
        else:
            ctx = mock.patch.dict(os.environ, {ENV: env_val})
        buf = io.StringIO()
        with ctx, mock.patch("sys.stdin.isatty", return_value=False), \
             contextlib.redirect_stderr(buf):
            netconsent.consent(**_ARGS)
        return buf.getvalue()

    def test_notice_on_kill_switch(self):
        out = self._notice_for("0")
        self.assertIn("Test-Model", out)
        self.assertIn("Apache-2.0", out)
        self.assertIn("huggingface.co", out)

    def test_notice_on_standing_consent(self):
        out = self._notice_for("1")
        self.assertIn("Test-Model", out)
        self.assertIn("~10MB", out)

    def test_notice_on_non_tty_refusal(self):
        out = self._notice_for(None)
        self.assertIn("Test-Model", out)
        self.assertIn("huggingface.co", out)


class TestVectorConsentIntegration(unittest.TestCase):
    """core/vector._ensure_model: cache HIT -> no consent; MISS -> gated.

    The transformers import inside _ensure_model is mocked at the load-helper
    boundary (_load_local_only) so no real model / torch weights are touched.
    """

    def setUp(self):
        # Reset the module-level model cache so each case exercises the load
        # path fresh (a prior real load in the suite would short-circuit).
        from samia.core import vector
        self.vector = vector
        self._saved = (vector._model, vector._tokenizer)
        vector._model = None
        vector._tokenizer = None

    def tearDown(self):
        self.vector._model, self.vector._tokenizer = self._saved

    def test_cache_hit_never_calls_consent(self):
        # local_files_only load succeeds -> the common path must be SILENT:
        # consent() is never invoked and no network/prompt is reached.
        fake_tok = object()
        fake_model = mock.Mock()
        with mock.patch.object(self.vector, "_load_local_only",
                               return_value=(fake_tok, fake_model)), \
             mock.patch.object(netconsent, "consent",
                               side_effect=AssertionError(
                                   "consent must not be called on cache hit")), \
             mock.patch.dict("sys.modules", {"torch": mock.Mock()}):
            self.vector._ensure_model()
        self.assertIs(self.vector._tokenizer, fake_tok)
        self.assertIs(self.vector._model, fake_model)
        fake_model.eval.assert_called_once()

    def test_cache_miss_declined_raises_runtime_error(self):
        # local_files_only returns None (cache miss) -> consent is consulted;
        # a declined consent raises a RuntimeError that names the env remedy.
        with mock.patch.object(self.vector, "_load_local_only",
                               return_value=None), \
             mock.patch.object(netconsent, "consent",
                               return_value=False) as cns, \
             mock.patch.dict("sys.modules", {"torch": mock.Mock()}):
            with self.assertRaises(RuntimeError) as cm:
                self.vector._ensure_model()
        cns.assert_called_once()
        self.assertIn(netconsent.AUTOFETCH_ENV, str(cm.exception))

    def test_cache_miss_approved_loads_via_network_path(self):
        # Cache miss + consent approved -> the network from_pretrained path runs.
        # We mock transformers so no real download happens.
        fake_tf = mock.Mock()
        fake_tf.AutoTokenizer.from_pretrained.return_value = object()
        loaded_model = mock.Mock()
        fake_tf.AutoModel.from_pretrained.return_value = loaded_model
        with mock.patch.object(self.vector, "_load_local_only",
                               return_value=None), \
             mock.patch.object(netconsent, "consent", return_value=True), \
             mock.patch.dict("sys.modules", {
                 "torch": mock.Mock(),
                 "transformers": fake_tf,
             }):
            self.vector._ensure_model()
        self.assertIs(self.vector._model, loaded_model)
        loaded_model.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.test_netconsent
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      SEC consent protocol — v1 "no silent downloads, ever"
# Layer:      test (pytest)
# Role:       tests for samia.core.netconsent — the one-knob (ASTHENOS_MODEL_
#             AUTOFETCH) decision matrix (kill switch / standing consent /
#             ask-if-tty / refuse-non-tty naming both remedies), the env
#             on/off helpers, the always-printed notice, and the core/vector
#             integration (cache HIT never consents; MISS gates, declined raises).
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.netconsent, samia.core.vector
# Exposes:    — (test module)
# Lines:      312
# --------------------------------------------------------------------------
