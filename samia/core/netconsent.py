"""samia.core.netconsent -- single owner of the SAM/IA download-consent gate.

Layer 1 (Owns / Depends):
    Owns:    AUTOFETCH_ENV, the env interpretation helpers (env_explicit_off,
             env_explicit_on), and consent() -- the one function every network
             fetch in the engine routes through before any byte moves.
    Depends: stdlib only (os, sys). NO third-party, NO samia.runtime import --
             lives in core/ so core/vector.py can gate its HF download without
             importing the runtime package.

Layer 2 (What / Why):
    What: consent(what, size_hint, license_str, source, manual_hint) -> bool
          decides whether a download may proceed, by ONE env knob
          (ASTHENOS_MODEL_AUTOFETCH) plus interactivity:
            - env explicitly OFF ("0"/"false"/"no"/"off") -> refuse (False).
            - env PRESENT with an on-value -> standing operator consent ->
              proceed (True) without prompting. This is what agent/CI flows
              export after clearing their own permission gate.
            - env NOT SET AT ALL (the default release posture):
                * interactive tty -> print what/size/license/source and prompt
                  "Download? [y/N]"; only a 'y'/'yes' proceeds.
                * non-tty -> refuse, naming BOTH remedies (export the env on, or
                  download manually to the path in manual_hint).
          The license notice for the specific download is ALWAYS printed before
          a True is returned (the caller still prints its own pre-byte notice;
          consent() prints the what/size/license/source summary as part of the
          prompt or, when standing consent is in effect, on its own line).
    Why:  operator directive for the v1 release: NO silent downloads, ever. The
          old gate was default-ON ("1") -- an unset env meant "just download".
          That is reversed here: unset now means "ask if a human is present,
          else refuse". A single module owns the exact semantics so the gguf
          fetcher (runtime/model_fetch) and the MiniLM embedder (core/vector)
          cannot drift apart -- one protocol, two callers, identical behavior.

Layer 3 (Changelog):
    2026-06-12  SEC        Initial. Extracted the autofetch gate into a core-
                            plane consent protocol (ask-if-tty / standing-env /
                            kill-switch). Replaces model_fetch._autofetch_enabled
                            for the consent decision; vector.py routes through it.
"""

from __future__ import annotations

import os
import sys

# AUTOFETCH_ENV -- What: the single env knob governing every download.
# AUTOFETCH_ENV -- Why: one name across the whole engine (the gguf fetcher and
#     the MiniLM embedder share it) so the operator has ONE lever: set it on for
#     standing consent / CI, set it off as a kill switch, or leave it unset for
#     the ask-if-interactive default. Same name model_fetch historically used,
#     so existing operator/kit settings keep working.
AUTOFETCH_ENV = "ASTHENOS_MODEL_AUTOFETCH"

# _OFF_VALUES / _ON_VALUES -- What: the recognized explicit env tokens.
# _OFF_VALUES / _ON_VALUES -- Why: case-insensitive, whitespace-stripped. Any
#     other present-but-unrecognized value is treated as NOT an explicit signal
#     (falls through to the ask-if-tty default) rather than guessed at -- we only
#     act on consent we can read unambiguously.
_OFF_VALUES = frozenset({"0", "false", "no", "off"})
_ON_VALUES = frozenset({"1", "true", "yes", "on"})

# _YES_REPLIES -- What: the interactive replies that authorize a download.
# _YES_REPLIES -- Why: explicit-yes only; the prompt default is No, so anything
#     that is not a clear yes (including EOF/empty) refuses.
_YES_REPLIES = frozenset({"y", "yes"})


def _env_raw() -> str | None:
    """Return the stripped/lowered env value, or None when the var is absent.

    What: distinguishes "not set at all" (None) from "set to empty/whitespace"
          (""), because the default posture (ask-if-tty) keys on genuine absence.
    Why:  the directive's three states are: explicitly-off, explicitly-on, and
          NOT-SET. An empty-string value is a present-but-meaningless setting; it
          is treated like an unrecognized token -> the ask-if-tty default, not a
          silent proceed.
    """
    val = os.environ.get(AUTOFETCH_ENV)
    if val is None:
        return None
    return val.strip().lower()


def env_explicit_off() -> bool:
    """True iff the env is PRESENT and set to a recognized off-value.

    What: "0"/"false"/"no"/"off" (case-insensitive) -> True; absent or any
          other value -> False.
    Why:  this is the kill switch. It outranks interactivity entirely: even at
          an interactive tty, an explicit off means refuse with no prompt.
    """
    raw = _env_raw()
    return raw is not None and raw in _OFF_VALUES


def env_explicit_on() -> bool:
    """True iff the env is PRESENT and set to a recognized on-value.

    What: "1"/"true"/"yes"/"on" (case-insensitive) -> True; absent or any
          other value -> False.
    Why:  standing operator consent. Agent/CI flows export this AFTER clearing
          their own permission gate, so the engine proceeds without a prompt
          that no human is there to answer.
    """
    raw = _env_raw()
    return raw is not None and raw in _ON_VALUES


def _print_notice(what: str, size_hint: str, license_str: str, source: str) -> None:
    """Print the what/size/license/source summary for a pending download.

    What: a compact multi-line notice to stderr identifying exactly what is
          about to be fetched and under what license, from where.
    Why:  the directive requires the license notice ALWAYS prints before bytes
          move. This summary is the consent-side notice; callers may add their
          own (model_fetch prints a second per-file notice in _download).
    """
    print(
        f"[netconsent] {what} -- {size_hint}, {license_str}\n"
        f"[netconsent]   source: {source}",
        file=sys.stderr,
        flush=True,
    )


def consent(
    what: str,
    size_hint: str,
    license_str: str,
    source: str,
    manual_hint: str,
) -> bool:
    """Decide whether a network download may proceed. Return True to proceed.

    Parameters
    ----------
    what : str
        Human label for the artifact (e.g. "MiniLM-L6-v2 embedder").
    size_hint : str
        Approximate size (e.g. "~90MB") for the operator's bandwidth decision.
    license_str : str
        License identifier shown before any bytes move (e.g. "Apache-2.0").
    source : str
        Source host/URL shown in the notice (e.g. "huggingface.co").
    manual_hint : str
        The non-tty / refusal message's manual remedy -- typically the exact
        local path to drop the file at so callers can skip the fetch entirely.

    Decision order (first match wins):
        1. env explicitly OFF  -> notice + refuse (kill switch; no prompt).
        2. env explicitly ON   -> notice + proceed (standing consent; no prompt).
        3. env NOT a clear on/off (absent or unrecognized):
             a. interactive tty -> notice + "Download? [y/N]"; y/yes -> proceed.
             b. non-tty         -> notice + refuse, naming BOTH remedies.

    The what/size/license/source notice prints (to stderr) on EVERY branch
    before this function returns -- the directive's "license notice always
    prints before bytes move" holds even on refusal.
    """
    # 1) Kill switch: explicit off refuses regardless of interactivity.
    if env_explicit_off():
        _print_notice(what, size_hint, license_str, source)
        print(
            f"[netconsent] {AUTOFETCH_ENV} is off: refusing to download {what}. "
            f"{manual_hint}",
            file=sys.stderr,
            flush=True,
        )
        return False

    # 2) Standing consent: explicit on proceeds silently (no human to prompt).
    if env_explicit_on():
        _print_notice(what, size_hint, license_str, source)
        return True

    # 3) Default posture (env not a clear signal): ask if a human is present.
    _print_notice(what, size_hint, license_str, source)
    if sys.stdin.isatty():
        try:
            reply = input(f"[netconsent] Download {what} now? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply in _YES_REPLIES:
            return True
        print(
            f"[netconsent] declined; not downloading {what}. {manual_hint}",
            file=sys.stderr,
            flush=True,
        )
        return False

    # Non-interactive with no standing consent: refuse, name BOTH remedies.
    print(
        f"[netconsent] no terminal and {AUTOFETCH_ENV} is not set: refusing to "
        f"download {what}. Either export {AUTOFETCH_ENV}=1 for standing consent, "
        f"or {manual_hint}",
        file=sys.stderr,
        flush=True,
    )
    return False


# ─────────────────────────────────────────────
# [netconsent] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-12  Status: active
# Role:       single owner of the download-consent gate for the SAM/IA engine
#             (kill-switch / standing-env-consent / ask-if-interactive default)
# Depends:    os, sys (stdlib only); NO samia.runtime import (core-plane safe)
# Gate:       ASTHENOS_MODEL_AUTOFETCH -- off=refuse, on=standing consent,
#             unset=ask-if-tty-else-refuse. License notice ALWAYS prints first.
# Callers:    samia.runtime.model_fetch.fetch_model, samia.core.vector._ensure_model
# ─────────────────────────────────────────────
