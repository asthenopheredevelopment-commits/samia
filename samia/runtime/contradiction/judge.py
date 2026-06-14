"""samia.runtime.contradiction.judge — Phase-2 LLM judge gate + P2 abstraction synth.

Layer 1 (Owns / Depends):
    Owns:    the dedicated cached judge backend resolution (_judge_backend,
             _inference_available, _infer_text), the tolerant first-JSON-object
             parser (_parse_first_json_object), the LLM contradiction judge
             (judge_contradictions), and the Tier-2 merge-consumer abstraction
             synthesizer (synthesis_enabled, synthesize_node).
    Depends: samia.runtime.inference (the in-process backend factory, optional,
             lazy), the package config leaf (the judge/synth budgets, the model
             default + path reader, the prompt templates, json + _log), and —
             through the PACKAGE FACADE — the patch-seam target _judge_backend and
             the facade-rebound flag _JUDGE_ENABLED.

Layer 2 (What / Why):
    What: the precision arm. The judge distinguishes "same topic, different claim"
          from "same topic, compatible claim"; the synthesizer folds a distinct-
          but-overlapping episodic pair into one semantic node. Both ride ONE
          dedicated, cached small backend (Qwen3-4B by default, NOT the slow 14B).
    Why:  embedding similarity is recall-first; the judge recovers precision. Both
          are fail-soft: a MockBackend / unavailable backend / parse failure
          collapses to the records-only ([]) / None no-op the callers rely on, so a
          judge or synth error never blocks or corrupts a write.

PATCH SEAMS (exemplar rule): _judge_backend, synthesis_enabled and _JUDGE_ENABLED are
    mock.patch.object(contradiction, ...) / facade-rebind targets that siblings within
    this module ALSO call/read (_inference_available + _infer_text reach _judge_backend;
    synthesize_node reaches synthesis_enabled; judge_contradictions + synthesis_enabled
    read _JUDGE_ENABLED), so those reach through the package facade so a package-level
    patch/rebind is honored.
"""

from __future__ import annotations

from typing import Any, Optional

# Shared leaf — the judge/synth token budgets, the model default + path reader, the
# two prompt templates, the package logger, and the re-exported json.
from . import config as _cfg


def _judge_backend() -> Any:
    """The DEDICATED, CACHED small backend the judge + synth use (fail-soft).

    What: builds (once) a backend for ASTHENOS_CONTRADICTION_JUDGE_MODEL
          (Qwen3-4B registry default) via inference.get_backend_for_model, which
          caches the LlamaCppBackend by model path so the small model loads a
          SINGLE time and is reused on every judge/synth call. When that factory
          is unavailable (older inference module) or the judge model is missing /
          not a .gguf / llama_cpp absent, it returns the MAIN backend
          (inference.get_backend()) so existing behavior + existing tests (which
          mock get_backend) are preserved. Any import error -> None.
    Why:  fix #2 -- the judge must NOT ride the slow 14B. A dedicated cached small
          backend keeps the passive sweep affordable while preserving the
          fail-soft contract (a MockBackend / unavailable backend -> the judge
          no-ops records-only/None exactly as before).
    """
    try:
        from samia.runtime import inference as _inf
    except Exception as exc:
        _cfg._log.debug("contradiction: inference module unavailable: %s", exc)
        return None
    factory = getattr(_inf, "get_backend_for_model", None)
    if factory is not None:
        try:
            dedicated = factory(_cfg._judge_model_path())
        except Exception as exc:
            _cfg._log.debug("contradiction: judge backend build failed: %s", exc)
            dedicated = None
        # Use the dedicated small backend ONLY when it is a REAL (non-mock)
        # backend -- i.e. the BitNet-2B gguf is present and loadable. When the
        # judge model is absent (the factory returns a MockBackend), fall back to
        # the main in-process backend so existing behavior + the existing tests
        # (which mock get_backend) are preserved.
        if dedicated is not None and type(dedicated).__name__ != "MockBackend":
            return dedicated
    # Fallback: the main in-process backend (and the path existing tests mock).
    try:
        return _inf.get_backend()
    except Exception as exc:
        _cfg._log.debug("contradiction: inference backend unavailable: %s", exc)
        return None


def _inference_available() -> bool:
    """True iff a REAL (non-mock) dedicated judge backend is loadable.

    What: asks _judge_backend() (the dedicated BitNet-2B small backend, cached)
          and reports whether it is anything OTHER than MockBackend (the fail-soft
          "no model configured" signal). Any import/init error -> False.
    Why:  the availability probe must reflect the DEDICATED judge backend being
          real, not the main 14B. _judge_backend() returns MockBackend when the
          judge model is unset / missing / llama_cpp absent (and the main fallback
          is also Mock), which is exactly the unavailable case the judge gate and
          synthesis must NO-OP on (records-only / None). Pure read.
    """
    # _judge_backend is a mock.patch.object seam (test_contradiction_tuning patches it
    # then calls _inference_available); reach it through the facade so the patch lands.
    from samia.runtime import contradiction as _pkg
    backend = _pkg._judge_backend()
    if backend is None:
        return False
    return type(backend).__name__ != "MockBackend"


def _infer_text(prompt: str, max_tokens: int) -> Optional[str]:
    """Generate text from the dedicated judge backend (or None, fail-soft).

    What: routes *prompt* through the DEDICATED small judge backend
          (_judge_backend().complete) -- Qwen3-4B by default, cached/loaded-once,
          NOT the slow main Qwen-14B -- and returns the raw completion text.
          Returns None when the backend is a MockBackend / unavailable /
          load-errored, or the call raises.
    Why:  the single inference entrypoint for both the judge and the abstraction
          synthesizer. Routing to a dedicated cached small backend keeps the
          passive sweep affordable AND stops the judge duplicating the 14B. A
          backend error must NEVER block or corrupt a write, so every failure
          collapses to None (the caller's records-only/None no-op).
    """
    # _judge_backend is a patch seam (test_contradiction_tuning patches it then calls
    # _infer_text); reach it through the facade so the patch is honored.
    from samia.runtime import contradiction as _pkg
    backend = _pkg._judge_backend()
    if backend is None:
        _cfg._log.debug("contradiction: judge backend unavailable")
        return None
    # MockBackend is the "no real model configured" fail-soft signal: treat it
    # exactly like an unavailable backend so nothing is auto-acted on canned text.
    if type(backend).__name__ == "MockBackend":
        _cfg._log.debug("contradiction: inference backend is MockBackend; skipping")
        return None
    try:
        return backend.complete(prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception as exc:
        _cfg._log.warning("contradiction: in-process inference call failed: %s", exc)
        return None


def _parse_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract the FIRST JSON object from an LLM completion (tolerates trailing text).

    What: find the first '{', then json.JSONDecoder().raw_decode from there —
          raw_decode parses ONE JSON value and stops, ignoring whatever the model
          appended after the closing brace. Returns the parsed object, or None when
          there is no '{' or the candidate region is not valid JSON.
    Why:  BUG-2026-06-11 judge-parse — ~95% of judge (and synth) outputs were
          discarded because the model emits trailing commentary AFTER a valid JSON
          object (e.g. `{...}\n\nThis means...`). The old json.loads(text[start:])
          fed that whole tail to the parser and raised "Extra data", throwing away
          good JSON. raw_decode keeps the genuinely-non-JSON failure path intact
          (returns None -> caller's existing fallback heuristics still run).
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _end = _cfg.json.JSONDecoder().raw_decode(text[start:])
    except (_cfg.json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def judge_contradictions(
    new_text: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the LLM judge on contradiction candidates.

    What: constructs a prompt with the new claim and candidate claims,
          routes it through the in-process inference backend
          (samia.runtime.inference.get_backend().complete), parses the
          structured JSON response.
    Why:  Phase 2 precision gate -- embedding similarity catches topical
          overlap, but the LLM judge distinguishes "same topic, different
          claim" from "same topic, compatible claim".

    Parameters
    ----------
    new_text : str
        The incoming write text.
    candidates : list of dicts
        Candidate nodes from find_contradiction_candidates.

    Returns
    -------
    list of dicts with keys: existing_claim_id, explanation, confidence.
    Only contradictions with confidence >= _JUDGE_CONFIDENCE_THRESHOLD.
    Empty list if the judge is disabled or fails.
    """
    # _JUDGE_ENABLED is rebound by tests on the package facade (con._JUDGE_ENABLED =
    # ...); read it through the facade so a facade-level rebind is honored.
    from samia.runtime import contradiction as _pkg
    if not _pkg._JUDGE_ENABLED:
        return []

    if not candidates:
        return []

    # What: format existing claims for the prompt.
    # Why: the judge needs both the claim ID and text to reference.
    claims_text = ""
    for i, c in enumerate(candidates, 1):
        claims_text += f"{i}. [{c['node_id']}] {c.get('title', '(no title)')}\n"

    prompt = _cfg._JUDGE_PROMPT_TEMPLATE.format(
        new_claim=new_text[:2000],
        existing_claims=claims_text,
    )

    # What: route through the in-process inference backend.
    # Why: the daemon already holds the loaded Qwen backend in-process (the
    #   passive sweep runs INSIDE the daemon), so we call get_backend().complete
    #   directly — no IPC round-trip, no llama-cli subprocess. _infer_text is the
    #   single fail-soft entrypoint: it returns None when the backend is a
    #   MockBackend / unavailable / load-errored or the call raises, which we map
    #   to the SAME empty (records-only) result as before. A judge error never
    #   blocks or corrupts a write.
    response_text = _infer_text(prompt, _cfg._JUDGE_INFER_MAX_TOKENS)
    if not response_text:
        return []

    # What: extract the FIRST JSON object from the response (the model may emit
    #   preamble before AND trailing commentary after the JSON block).
    # Why: BUG-2026-06-11 — raw_decode stops at the JSON object's closing brace
    #   and ignores trailing text, instead of json.loads choking on "Extra data"
    #   (which was discarding ~95% of otherwise-valid judge outputs).
    parsed = _parse_first_json_object(response_text)
    if parsed is None:
        _cfg._log.warning("contradiction: no parseable JSON in judge response")
        return []

    try:
        contradictions = parsed.get("contradictions", [])

        # What: filter by confidence threshold.
        # Why: low-confidence judgments are noise; only high-confidence
        #   contradictions should block or flag writes.
        return [
            c for c in contradictions
            if float(c.get("confidence", 0)) >= _cfg._JUDGE_CONFIDENCE_THRESHOLD
        ]
    except (KeyError, TypeError, ValueError) as exc:
        _cfg._log.warning("contradiction: judge response parse error: %s", exc)
        return []


def synthesis_enabled() -> bool:
    """True iff the local LLM synthesis backend is available to call.

    What: reuses the SAME enable flag as the LLM judge
          (ASTHENOS_CONTRADICTION_JUDGE) AND additionally requires a REAL
          in-process inference backend (get_backend() is not a MockBackend) —
          the synthesis call rides the judge's in-process inference plumbing, so
          it is available exactly when the judge can actually run a model.
    Why:  Tier-2 P2 — synthesis must be a SAFE NO-OP when inference is off
          (same conservative posture as the judge being disabled). The merge
          consumer checks this before attempting any abstraction, leaving the
          pair pending rather than crashing. Probing get_backend() (not
          "is llama-cli on PATH") is the rewired availability signal.
    """
    # _JUDGE_ENABLED is facade-rebound by tests; _inference_available rides the
    # _judge_backend patch seam — read both through the facade.
    from samia.runtime import contradiction as _pkg
    return _pkg._JUDGE_ENABLED and _pkg._inference_available()


def synthesize_node(text_a: str, text_b: str) -> Optional[dict[str, Any]]:
    """Synthesize one higher-level node from two source bodies (P2 abstraction).

    What: runs the SAME in-process inference backend judge_contradictions uses
          (samia.runtime.inference.get_backend().complete), with a synthesis
          prompt, and parses the structured JSON {title, body}. Returns that
          dict, or None when synthesis is disabled / the backend is unavailable /
          the response is unparseable.
    Why:  Tier-2 merge consumer P2 (Q1c/Q2c) — abstractive compression of a
          distinct-but-overlapping episodic pair into one semantic node. Reuses
          the existing inference entrypoint (no new model loader); the None
          return is the safe no-op the consumer relies on to leave the pair
          pending instead of crashing.

    Returns
    -------
    {"title": str, "body": str} or None.
    """
    # synthesis_enabled is a mock.patch.object seam; reach it through the facade so a
    # package-level patch (and the facade-rebound _JUDGE_ENABLED inside it) is honored.
    from samia.runtime import contradiction as _pkg
    if not _pkg.synthesis_enabled():
        return None

    prompt = _cfg._SYNTH_PROMPT_TEMPLATE.format(
        note_a=str(text_a)[:2000],
        note_b=str(text_b)[:2000],
    )

    # In-process inference (same backend as the judge). _infer_text fails soft to
    # None when the backend is a MockBackend / unavailable / load-errored or the
    # call raises — the merge consumer relies on that None to leave the pair
    # pending instead of crashing.
    response_text = _infer_text(prompt, _cfg._SYNTH_INFER_MAX_TOKENS)
    if not response_text:
        return None

    # BUG-2026-06-11 — same trailing-text tolerance as the judge: raw_decode the
    # first JSON object so a synth completion with appended commentary still parses.
    parsed = _parse_first_json_object(response_text)
    if parsed is None:
        _cfg._log.warning("contradiction: no parseable JSON in synth response")
        return None
    try:
        title = str(parsed.get("title", "")).strip()
        body = str(parsed.get("body", "")).strip()
    except (KeyError, TypeError, ValueError) as exc:
        _cfg._log.warning("contradiction: synth response parse error: %s", exc)
        return None
    if not body:
        _cfg._log.warning("contradiction: synth produced empty body")
        return None
    return {"title": title, "body": body}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.contradiction.judge
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.runtime.contradiction monolith
#             during modularization (the Phase-2 judge + P2 synth arm).
# Layer:      runtime (library helper, no daemon loop)
# Role:       Phase-2 LLM judge gate (judge_contradictions) + the Tier-2 P2
#             abstraction synthesizer (synthesize_node), both riding ONE dedicated
#             cached small backend (Qwen3-4B default, NOT the slow 14B) resolved via
#             _judge_backend / _inference_available / _infer_text, with the tolerant
#             _parse_first_json_object reader.
# Stability:  v0.4 — FIX-2026-06-08 in-process inference rewire (no subprocess);
#             BUG-2026-06-11 judge-parse raw_decode tolerance.
# ErrorModel: fail-soft — MockBackend / unavailable backend / parse failure collapses
#             to records-only ([]) / None; a judge or synth error never blocks a write.
# Depends:    .config (budgets + model default + prompts + json + _log);
#             samia.runtime.inference (lazy). Reaches _judge_backend / synthesis_enabled
#             / _inference_available / _JUDGE_ENABLED through the PACKAGE FACADE
#             (patch seams + facade-rebound flag).
# Exposes:    judge_contradictions, synthesis_enabled, synthesize_node (public);
#             _judge_backend (patch seam), _inference_available, _infer_text,
#             _parse_first_json_object (internal).
# Lines:      346
# --------------------------------------------------------------------------
