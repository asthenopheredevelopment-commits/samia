"""samia.runtime.inference -- protocol-abstracted LLM inference for SAM/IA.

Layer 1 (Owns / Depends):
    Owns:    InferenceBackend protocol, MockBackend, LlamaCppBackend,
             get_backend() factory, TelemetryEmitter (JSONL event logger),
             IPC op registration (judge, infer, inference_status,
             inference_telemetry_status)
    Depends: samia.runtime.ipc.register_op (plugin registration)

Layer 2 (What / Why):
    What: Defines the InferenceBackend protocol with three methods:
          complete(prompt, ...) -> str, compare(fact_a, fact_b) ->
          (bool, str, float), is_loaded() -> bool.  Two implementations:
          MockBackend (deterministic, always available) and LlamaCppBackend
          (lazy-imports llama_cpp, requires a .gguf model file).
          get_backend() factory reads ASTHENOS_INFERENCE_MODEL env var
          and falls back to MockBackend with a warning.  register_ops(ipc)
          wires four IPC ops: judge, infer, inference_status,
          inference_telemetry_status.
          TelemetryEmitter appends one JSON line per inference call to
          <events_dir>/YYYY-MM-DD.jsonl for cost-attribution against the
          Anthropic API counterfactual (viberank tracking).
    Why:  The memory runtime needs local inference for fact contradiction
          detection (AUD16-compatible judge) and general completion.
          Model-agnostic design lets the runtime start and pass tests with
          MockBackend, then activate a real model when the operator installs
          llama-cpp-python and downloads a GGUF.  Telemetry emission lets
          downstream ingest (AUD28-28.4) and the Atoms surface track
          inference volume, latency, and error rates without touching the
          daemon hot path.

Layer 3 (Changelog):
    2026-05-03  AUD26-26.3  Initial implementation.  MockBackend + LlamaCppBackend
                             + get_backend() + register_ops().
    2026-05-03  AUD28-28.2  TelemetryEmitter + telemetry wrappers on judge/infer
                             + caller_hint propagation + inference_telemetry_status op.
    2026-05-07  AUD82       inference_fallback_chain() + inference_reset_cuda IPC op.
                             Fallback chain: CPU qwen3 -> BitNet -> error.

Design doc: plans/sam_ia_runtime_design.md, section 1.2.
AUD26 Phase 26.3 -- in-runtime inference.
AUD28 Phase 28.2 -- daemon telemetry emission.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Protocol

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = logging.getLogger("samia.runtime.inference")

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class InferenceBackend(Protocol):
    """Abstract interface for an LLM inference backend."""

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str: ...

    def compare(self, fact_a: str, fact_b: str) -> tuple[bool, str, float]:
        """Return (contradicts, rationale, confidence).  AUD16-compatible."""
        ...

    def is_loaded(self) -> bool: ...


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------


def _deterministic_hash(text: str) -> int:
    """Return a stable integer hash for deterministic mock outputs."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class MockBackend:
    """Deterministic mock backend for tests and smoke runs.

    complete() returns canned text seeded by prompt hash.
    compare() returns a deterministic (bool, rationale, confidence) triple.
    """

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        h = _deterministic_hash(prompt)
        responses = [
            "The answer is 42.",
            "According to my analysis, this is correct.",
            "No contradictions detected.",
            "Further investigation is needed.",
        ]
        return responses[h % len(responses)]

    def compare(self, fact_a: str, fact_b: str) -> tuple[bool, str, float]:
        """Deterministic comparison based on input hashes."""
        combined = fact_a + "||" + fact_b
        h = _deterministic_hash(combined)
        contradicts = (h % 3) == 0  # ~33% contradiction rate
        confidence = round(0.5 + (h % 50) / 100.0, 2)
        if contradicts:
            rationale = f"Mock: facts appear contradictory (hash={h:#x})"
        else:
            rationale = f"Mock: facts appear consistent (hash={h:#x})"
        return contradicts, rationale, confidence

    def is_loaded(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# LlamaCppBackend
# ---------------------------------------------------------------------------

# Prompt template for the compare() method.
_COMPARE_SYSTEM = (
    "You are a fact-contradiction detector. Given two facts, determine whether "
    "they contradict each other. Reply ONLY with valid JSON: "
    '{"contradicts": true|false, "rationale": "<one sentence>", '
    '"confidence": <float 0.0-1.0>}'
)

_COMPARE_USER_TEMPLATE = (
    "Fact A: {fact_a}\n"
    "Fact B: {fact_b}\n\n"
    "Do these facts contradict each other? Reply with JSON only."
)

_CONTRADICTS_RE = re.compile(r'"contradicts"\s*:\s*(true|false)', re.I)
_RATIONALE_RE = re.compile(r'"rationale"\s*:\s*"([^"]*)"', re.I)
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)', re.I)


class LlamaCppBackend:
    """Backend using llama-cpp-python.  Lazy-imports at first call.

    Parameters
    ----------
    model_path : str
        Path to a .gguf model file.
    n_ctx : int
        Context window size (default 4096).
    n_gpu_layers : int
        GPU layers to offload (-1 = all).
    chat_format : str | None
        Chat template override (None = auto-detect from GGUF metadata).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        chat_format: str | None = None,
    ) -> None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        if not p.suffix == ".gguf":
            raise ValueError(f"model_path must be a .gguf file, got: {model_path}")

        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._chat_format = chat_format
        self._llm: Any = None  # llama_cpp.Llama instance, created lazily
        self._load_error: str | None = None
        self._load_error_ts: float = 0.0

    # AUD28.7 V1: error TTL for transient-failure recovery.
    # Why: a single transient init failure (GPU contention, OOM at startup) used
    # to cache permanently in self._load_error and force a daemon restart. Now
    # we clear after _LOAD_ERROR_TTL_S so the next call retries automatically.
    _LOAD_ERROR_TTL_S = 60.0

    def _ensure_loaded(self) -> Any:
        """Lazy-load the model.  Returns the Llama instance or raises."""
        if self._llm is not None:
            return self._llm
        if self._load_error is not None:
            err_age = time.monotonic() - self._load_error_ts
            if err_age < self._LOAD_ERROR_TTL_S:
                raise RuntimeError(self._load_error)
            # TTL elapsed — clear and retry below.
            _log.info("clearing cached load error after %.1fs (TTL=%.0fs); retrying",
                      err_age, self._LOAD_ERROR_TTL_S)
            self._load_error = None

        try:
            import llama_cpp  # type: ignore[import-untyped]
        except ImportError as exc:
            self._load_error = f"llama_cpp not importable: {exc}"
            self._load_error_ts = time.monotonic()
            raise ImportError(self._load_error) from exc

        def _load(ngl: int) -> Any:
            kwargs: dict[str, Any] = {
                "model_path": self._model_path,
                "n_ctx": self._n_ctx,
                "n_gpu_layers": ngl,
                "verbose": False,
            }
            if self._chat_format is not None:
                kwargs["chat_format"] = self._chat_format
            return llama_cpp.Llama(**kwargs)

        try:
            self._llm = _load(self._n_gpu_layers)
        except Exception as exc:
            # GPU ADDITIVE, CPU FALLBACK: SAM/IA never REQUIRES a GPU. If an
            # offloaded load fails (CUDA build on a box with no GPU, VRAM OOM, a
            # driver/toolkit mismatch), retry once on pure CPU before giving up.
            if self._n_gpu_layers != 0:
                _log.warning("GPU load failed (%s) — falling back to CPU "
                             "(n_gpu_layers=0)", exc)
                try:
                    self._llm = _load(0)
                    self._n_gpu_layers = 0
                except Exception as exc2:
                    self._load_error = f"Llama() init failed (GPU and CPU): {exc2}"
                    self._load_error_ts = time.monotonic()
                    raise RuntimeError(self._load_error) from exc2
            else:
                self._load_error = f"Llama() init failed: {exc}"
                self._load_error_ts = time.monotonic()
                raise RuntimeError(self._load_error) from exc

        _log.info("llama_cpp model loaded: %s (ctx=%d, gpu_layers=%d)",
                  self._model_path, self._n_ctx, self._n_gpu_layers)
        return self._llm

    def reset(self) -> dict[str, Any]:
        """AUD28.7 V1: operator-triggered reset for stuck loads.

        Clears any cached load error and forces a fresh init on next call.
        Does NOT unload an already-loaded model — only clears the stuck-error
        state. Returns a small status dict for observability via the IPC op.
        """
        had_error = self._load_error is not None
        self._load_error = None
        self._load_error_ts = 0.0
        loaded = self._llm is not None
        return {"cleared_error": had_error, "model_loaded": loaded}

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        llm = self._ensure_loaded()
        result = llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        return result["choices"][0]["text"]

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Templated chat completion (gguf's own chat template via llama_cpp).

        What: create_chat_completion with a system+user pair — the same call
          compare() already uses — returning the assistant text.
        Why: TUNE-2026-06-10 — instruct models given a RAW completion prompt
          ignore format instructions (the fact-extract smoke got prose instead
          of JSON, silently falling back to the rule splitter). The chat
          template is what makes instruct ggufs follow the system contract.
        """
        llm = self._ensure_loaded()
        result = llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"] or ""

    def compare(self, fact_a: str, fact_b: str) -> tuple[bool, str, float]:
        """Build a structured prompt, call the model, parse the JSON response."""
        llm = self._ensure_loaded()
        messages = [
            {"role": "system", "content": _COMPARE_SYSTEM},
            {"role": "user", "content": _COMPARE_USER_TEMPLATE.format(
                fact_a=fact_a, fact_b=fact_b,
            )},
        ]
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.0,
        )
        raw = result["choices"][0]["message"]["content"]

        # Try JSON parse first.
        try:
            parsed = json.loads(raw)
            contradicts = bool(parsed.get("contradicts", False))
            rationale = str(parsed.get("rationale", ""))[:200]
            confidence = float(parsed.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            return contradicts, rationale, confidence
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback: keyword/regex heuristic.
        contradicts = False
        cm = _CONTRADICTS_RE.search(raw or "")
        if cm:
            contradicts = cm.group(1).lower() == "true"

        rm = _RATIONALE_RE.search(raw or "")
        rationale = rm.group(1)[:200] if rm else (raw or "")[:200]

        cfm = _CONFIDENCE_RE.search(raw or "")
        try:
            confidence = float(cfm.group(1)) if cfm else 0.7
        except ValueError:
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        return contradicts, rationale, confidence

    def is_loaded(self) -> bool:
        return self._llm is not None and self._load_error is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# AUD29.3: Backend registry. Populated by register_ops(); inspected by
# _resolve_backend(). Maps short name → InferenceBackend instance. Each backend
# instantiates lazily — only the requested one materializes its model.
_backend_registry: dict[str, "InferenceBackend"] = {}

# FIX-2026-06-08 (double-load): per-model-path LlamaCppBackend cache.
# What: maps an absolute .gguf path -> the ONE LlamaCppBackend that wraps it.
# Why: the daemon logged the SAME Qwen-14B gguf loaded TWICE (~18GB) because
#   get_backend() built a fresh LlamaCppBackend on EVERY call -- register_ops()
#   built one (stored in _backend) and the contradiction judge's _infer_text
#   called get_backend() again, constructing a SECOND instance of the identical
#   model. Keying construction by model path makes a given gguf load at most
#   once: get_backend() and the dedicated judge backend both go through
#   get_backend_for_model(), so nothing double-loads the 14B (and the BitNet
#   judge model is likewise loaded a single time).
_model_backend_cache: dict[str, "LlamaCppBackend"] = {}


def _default_n_gpu_layers() -> int:
    """GPU layers to offload, default -1 (all). GPU is ADDITIVE: -1 is harmless on
    a CPU-only llama-cpp build (ignored -> runs CPU) and uses the GPU on a CUDA
    build when one is present. Override with ASTHENOS_N_GPU_LAYERS: 0 forces CPU
    even on a GPU build; a positive N does partial offload for limited VRAM."""
    v = os.environ.get("ASTHENOS_N_GPU_LAYERS", "").strip()
    if v:
        try:
            return int(v)
        except ValueError:
            _log.warning("ignoring non-int ASTHENOS_N_GPU_LAYERS=%r", v)
    return -1


def get_backend_for_model(
    model_path: str,
    *,
    n_ctx: int = 4096,
    n_gpu_layers: int | None = None,
    chat_format: str | None = None,
) -> "InferenceBackend":
    """Return the ONE cached LlamaCppBackend for *model_path* (load-once).

    What: builds a LlamaCppBackend for a SPECIFIC gguf and caches it keyed by
          the resolved absolute path, so repeated calls for the same model
          return the SAME instance (the model loads at most once). Falls back
          to MockBackend when the path is missing / not a .gguf / llama_cpp is
          unimportable -- the same fail-soft posture as get_backend().
    Why:  the singleton key that kills the double Qwen-14B load AND the home of
          the dedicated small judge backend (BitNet-2B): both the main backend
          and the judge resolve their model through this cache, so each distinct
          gguf has exactly one resident LlamaCppBackend.
    """
    if not model_path:
        return MockBackend()
    p = Path(model_path)
    if not p.exists():
        # SELF-FETCH (FEAT-2026-06-10): mirror the embedder's auto-download UX
        # for the gguf arms. If the missing path names a KNOWN model (or its
        # canonical copy is fetchable), pull it on demand; otherwise fail soft
        # to the existing missing-model -> MockBackend path so nothing regresses
        # when autofetch is disabled, the model is unknown, or the network is
        # unreachable.
        try:
            from samia.runtime import model_fetch
            fetched = model_fetch.fetch_model(model_path)
            p = Path(fetched)
            model_path = str(p)
        except Exception as exc:
            _log.warning(
                "inference: model not found at %s and self-fetch failed (%s), "
                "using MockBackend", model_path, exc,
            )
            return MockBackend()
        if not p.exists():
            _log.warning(
                "inference: model not found at %s, using MockBackend", model_path)
            return MockBackend()
    if p.suffix != ".gguf":
        _log.warning("inference: model path %s is not a .gguf file, using MockBackend",
                     model_path)
        return MockBackend()
    if n_gpu_layers is None:
        n_gpu_layers = _default_n_gpu_layers()
    key = str(p.resolve())
    cached = _model_backend_cache.get(key)
    if cached is not None:
        return cached
    try:
        backend = LlamaCppBackend(
            model_path=model_path, n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers, chat_format=chat_format,
        )
        # Verify the binding imports (model itself loads lazily on first call).
        import importlib
        importlib.import_module("llama_cpp")
    except ImportError:
        _log.warning("inference: llama_cpp not installed, using MockBackend")
        return MockBackend()
    except FileNotFoundError:
        _log.warning("inference: model not found at %s, using MockBackend", model_path)
        return MockBackend()
    except Exception as exc:
        _log.warning("inference: backend init failed (%s), using MockBackend", exc)
        return MockBackend()
    _model_backend_cache[key] = backend
    _log.info("inference: LlamaCppBackend cached for %s", key)
    return backend


def _resolve_backend(requested: Optional[str] = None) -> "InferenceBackend":
    """AUD29.3 selection chain: requested → ASTHENOS_INFERENCE_BACKEND env →
    first-loadable from registry → fallback to module _backend (LlamaCpp/Mock).

    Names: "llama_cpp", "bitnet", "npu", "mock". Case-insensitive.
    """
    candidates: list[str] = []
    if requested:
        candidates.append(str(requested).lower())
    env_default = os.environ.get("ASTHENOS_INFERENCE_BACKEND")
    if env_default:
        candidates.append(env_default.lower())
    # Stable preferred order if neither requested nor env is set.
    candidates.extend(["llama_cpp", "bitnet", "npu", "mock"])

    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        backend = _backend_registry.get(name)
        if backend is not None:
            return backend

    # Last resort: the legacy module-level _backend installed by register_ops().
    if _backend is not None:
        return _backend
    raise RuntimeError("no inference backend available")


def get_backend() -> InferenceBackend:
    """Return the cached main inference backend (load-once singleton).

    Tries LlamaCppBackend if ASTHENOS_INFERENCE_MODEL is set and points to
    an existing .gguf file and llama_cpp imports cleanly.  Falls back to
    MockBackend with a warning naming the specific reason.

    FIX-2026-06-08 (double-load): routes through get_backend_for_model(), which
    caches the LlamaCppBackend by model path. Previously every call (register_ops
    AND the contradiction judge's _infer_text) built a NEW LlamaCppBackend for
    the SAME Qwen-14B gguf, loading it twice (~18GB). With per-path caching a
    given model loads at most once regardless of how many call sites ask for it.
    """
    model_path = os.environ.get("ASTHENOS_INFERENCE_MODEL")
    if not model_path:
        _log.warning("inference: ASTHENOS_INFERENCE_MODEL not set, using MockBackend")
        return MockBackend()
    return get_backend_for_model(model_path)


# ---------------------------------------------------------------------------
# Telemetry emitter
# ---------------------------------------------------------------------------

# What: Appends one JSON line per inference call to a daily JSONL file for
#        cost-attribution and viberank tracking.
# Why:  Lock-free single-process writes; kernel guarantees write atomicity
#       for small payloads under PIPE_BUF on ext4.  Errors are caught so
#       the calling op is never disrupted by telemetry failures.

_DEFAULT_EVENTS_DIR = Path.home() / ".local" / "share" / "asthenos" / "inference_events"


class TelemetryEmitter:
    """Append-only JSONL writer for inference telemetry events."""

    def __init__(self, events_dir: Path | None = None) -> None:
        self._events_dir = events_dir or _DEFAULT_EVENTS_DIR

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    def _today_path(self) -> Path:
        """Return the JSONL path for today's UTC date."""
        return self._events_dir / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d") + ".jsonl")

    def emit(self, event: dict) -> None:
        """Append *event* as one JSON line to today's file.

        Errors are logged, never raised into the calling op.
        """
        try:
            self._events_dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event, separators=(",", ":"), default=str) + "\n"
            path = self._today_path()
            # O_APPEND makes each write atomic for lines < PIPE_BUF.
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        except Exception:
            _log.debug("telemetry emit failed", exc_info=True)

    def today_count(self) -> int:
        """Return the number of events in today's file (line count)."""
        path = self._today_path()
        if not path.exists():
            return 0
        try:
            with open(path, "r") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def today_size_bytes(self) -> int:
        """Return the byte size of today's file."""
        path = self._today_path()
        if not path.exists():
            return 0
        try:
            return path.stat().st_size
        except Exception:
            return 0

    def total_files(self) -> int:
        """Return total number of .jsonl files in the events dir."""
        if not self._events_dir.exists():
            return 0
        try:
            return sum(1 for p in self._events_dir.iterdir() if p.suffix == ".jsonl")
        except Exception:
            return 0


# Module-level emitter, set by register_ops().
_emitter: TelemetryEmitter | None = None


# ---------------------------------------------------------------------------
# IPC op registration
# ---------------------------------------------------------------------------

# Module-level backend instance, set by register_ops().
_backend: InferenceBackend | None = None


def _build_telemetry_event(
    op: str,
    args: dict[str, Any],
    *,
    prompt_chars: int,
    response_chars: int,
    latency_ms: float,
    success: bool,
    error: str | None,
) -> dict[str, Any]:
    """Build a telemetry event dict with the canonical schema.

    What: Assembles the event payload from op results + inference_status snapshot.
    Why:  Single construction point keeps the schema consistent across judge/infer.
          AUD29.3: prefers args["backend"] resolution so the actual backend used
          for THIS request is recorded (not just the module default).
    """
    requested = args.get("backend")
    actual_backend = _resolve_backend(requested) if (requested or _backend_registry) else _backend
    backend_name = type(actual_backend).__name__ if actual_backend else "unknown"
    model_path = getattr(actual_backend, "_model_path", None)

    # Caller-hint propagation: accept optional string, clamp to 64 chars.
    caller_hint = args.get("caller_hint")
    if isinstance(caller_hint, str):
        caller_hint = caller_hint[:64]
    else:
        caller_hint = None

    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "op": op,
        "caller_hint": caller_hint,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "latency_ms": round(latency_ms, 3),
        "backend": backend_name,
        "model_path": model_path,
        "success": success,
        "error": error,
    }


def _op_judge(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: judge(fact_a, fact_b) -> {ok, contradicts, rationale, confidence}.

    What: Validates inputs, calls backend.compare(), emits telemetry event.
    Why:  Telemetry wrapping records latency and success/error for every call
          so downstream ingest can compute cost-attribution against the API
          counterfactual.
    """
    fact_a = args.get("fact_a", "")
    fact_b = args.get("fact_b", "")
    if not fact_a or not fact_b:
        raise ValueError("judge requires non-empty fact_a and fact_b")
    backend = _resolve_backend(args.get("backend"))

    prompt_chars = len(fact_a) + len(fact_b)
    t0 = time.monotonic()
    try:
        contradicts, rationale, confidence = backend.compare(fact_a, fact_b)
        latency_ms = (time.monotonic() - t0) * 1000.0
        response_chars = len(rationale)
        if _emitter is not None:
            _emitter.emit(_build_telemetry_event(
                "judge", args,
                prompt_chars=prompt_chars, response_chars=response_chars,
                latency_ms=latency_ms, success=True, error=None,
            ))
        return {
            "contradicts": contradicts,
            "rationale": rationale,
            "confidence": confidence,
            # Surface the backend in-band so a caller can tell a real model
            # answer from MockBackend's canned/degraded output (e.g. when the
            # configured model path is missing and get_backend silently fell
            # back to MockBackend).
            "backend": type(backend).__name__,
        }
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000.0
        if _emitter is not None:
            _emitter.emit(_build_telemetry_event(
                "judge", args,
                prompt_chars=prompt_chars, response_chars=0,
                latency_ms=latency_ms, success=False, error=str(exc),
            ))
        raise


def _op_infer(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: infer(prompt, max_tokens=256, temperature=0.0) -> {text}.

    What: Validates inputs, calls backend.complete(), emits telemetry event.
    Why:  Same cost-attribution telemetry as judge; tracks prompt/response
          sizes for token-estimation in viberank scoring.
    """
    prompt = args.get("prompt", "")
    if not prompt:
        raise ValueError("infer requires a non-empty prompt")
    max_tokens = int(args.get("max_tokens", 256))
    temperature = float(args.get("temperature", 0.0))
    stop = args.get("stop")
    backend = _resolve_backend(args.get("backend"))

    prompt_chars = len(prompt)
    t0 = time.monotonic()
    try:
        text = backend.complete(
            prompt, max_tokens=max_tokens, temperature=temperature, stop=stop,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        response_chars = len(text)
        if _emitter is not None:
            _emitter.emit(_build_telemetry_event(
                "infer", args,
                prompt_chars=prompt_chars, response_chars=response_chars,
                latency_ms=latency_ms, success=True, error=None,
            ))
        # Surface the backend in-band so MockBackend's canned text cannot be
        # mistaken for real model output.
        return {"text": text, "backend": type(backend).__name__}
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000.0
        if _emitter is not None:
            _emitter.emit(_build_telemetry_event(
                "infer", args,
                prompt_chars=prompt_chars, response_chars=0,
                latency_ms=latency_ms, success=False, error=str(exc),
            ))
        raise


def _op_inference_status(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: inference_status() -> {backend, model_path, loaded}."""
    assert _backend is not None
    backend_name = type(_backend).__name__
    model_path = getattr(_backend, "_model_path", None)
    return {
        "backend": backend_name,
        "model_path": model_path,
        "loaded": _backend.is_loaded(),
    }


def _op_inference_reset(args: dict[str, Any]) -> dict[str, Any]:
    """AUD28.7 V1: clear cached load error so next inference call retries.

    Used when a transient init failure (GPU contention, OOM) cached a fatal
    error that the 60s TTL hasn't yet cleared. Operator-triggered. Does not
    unload an already-loaded model.
    """
    assert _backend is not None
    if hasattr(_backend, "reset"):
        return _backend.reset()
    return {"cleared_error": False, "model_loaded": _backend.is_loaded(),
            "note": f"{type(_backend).__name__} has no reset()"}


def _op_inference_telemetry_status(args: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: inference_telemetry_status() -> event dir stats.

    What: Returns events_dir path, today's event count / size, total files.
    Why:  Lets ingest verification (28.4) and Atoms surface confirm telemetry
          is flowing without reading the JSONL files directly.
    """
    if _emitter is None:
        return {
            "events_dir": str(_DEFAULT_EVENTS_DIR),
            "today_count": 0,
            "today_path": "",
            "today_size_bytes": 0,
            "total_files": 0,
        }
    return {
        "events_dir": str(_emitter.events_dir),
        "today_count": _emitter.today_count(),
        "today_path": str(_emitter._today_path()),
        "today_size_bytes": _emitter.today_size_bytes(),
        "total_files": _emitter.total_files(),
    }


def inference_fallback_chain(
    method: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """AUD82 Phase 3: fallback chain when GPU inference is unavailable.

    What: tries CPU-only qwen3 retry (via qwen3_backend.safe_call), then
          BitNet backend if registered, then returns an error response.
    Why:  when the circuit-breaker is open, callers need a structured
          fallback path rather than immediate failure. The chain degrades
          gracefully: CPU qwen3 is slow but functional; BitNet is the
          secondary backend; error is the last resort.

    Parameters
    ----------
    method : str
        Backend method name ('complete' or 'compare').
    *args, **kwargs :
        Arguments forwarded to the backend method.

    Returns
    -------
    dict with {ok: True, result: <value>, fallback: <str>} on success,
    or {ok: False, error: 'inference_unavailable', error_detail: <str>}.
    """
    # What: Step 1 -- CPU-only qwen3 retry via safe_call.
    # Why: qwen3_backend.safe_call already handles CPU retry logic (AUD82 Phase 1).
    try:
        from samia.runtime import qwen3_backend as _qwen3
        result = _qwen3.safe_call(method, *args, **kwargs)
        if result.get("ok"):
            return result
        _log.info("fallback_chain: qwen3 safe_call failed: %s", result.get("error_detail"))
    except Exception as exc:
        _log.info("fallback_chain: qwen3 safe_call exception: %s", exc)

    # What: Step 2 -- BitNet backend if registered.
    # Why: BitNet is a CPU-native backend; no CUDA dependency. If registered,
    #     it provides a secondary inference path.
    bitnet = _backend_registry.get("bitnet")
    if bitnet is not None:
        try:
            fn = getattr(bitnet, method, None)
            if fn is not None:
                bit_result = fn(*args, **kwargs)
                _log.info("fallback_chain: BitNet backend succeeded")
                return {"ok": True, "result": bit_result, "fallback": "bitnet"}
        except Exception as exc:
            _log.info("fallback_chain: BitNet backend failed: %s", exc)

    # What: Step 3 -- all fallbacks exhausted.
    # Why: return a structured error that callers (e.g. hypothesis.py) can
    #     handle gracefully instead of crashing.
    return {
        "ok": False,
        "error": "inference_unavailable",
        "error_detail": "all backends exhausted (qwen3 CPU + BitNet)",
    }


def _op_inference_reset_cuda(args: dict[str, Any]) -> dict[str, Any]:
    """AUD82 Phase 3: operator-driven reset of the CUDA circuit-breaker.

    What: resets the circuit-breaker state so GPU inference paths are
          re-enabled. Returns the prior breaker state and reset timestamp.
    Why:  after the operator fixes the underlying CUDA issue (driver update,
          GPU reset, etc.), they invoke this op to re-enable GPU inference
          without restarting the daemon.
    """
    from samia.runtime.inference_circuit_breaker import get_breaker
    import datetime

    breaker = get_breaker()
    prior_state = breaker.reset()
    return {
        "reset_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
        "prior_state": prior_state,
    }


def register_ops(
    events_dir: Path | None = None,
) -> InferenceBackend:
    """Create the inference backend, telemetry emitter, and register IPC ops.

    Called by daemon.py after IPC server creation.  Returns the backend
    instance for daemon-level access if needed.

    Parameters
    ----------
    events_dir : Path | None
        Override telemetry events directory (for tests).  Default:
        ~/.local/share/asthenos/inference_events/
    """
    global _backend, _emitter
    from samia.runtime.ipc import register_op

    _backend = get_backend()
    _emitter = TelemetryEmitter(events_dir=events_dir)

    # AUD29.3: register the default backend in the registry under its short
    # name so per-request `backend=...` kwargs can resolve. Future backends
    # (BitNet, NPU) call register_backend() at module-load time.
    backend_name = type(_backend).__name__
    short = {"LlamaCppBackend": "llama_cpp", "MockBackend": "mock"}.get(backend_name, "default")
    _backend_registry[short] = _backend

    # AUD29.2: optionally register BitNetBackend if the binary is present.
    try:
        import sys as _sys
        from samia.runtime import bitnet_backend as _bnb
        _bnb.register(_sys.modules[__name__])
    except Exception as _exc:  # never block daemon startup on optional backend
        _log.info("bitnet backend registration skipped: %s", _exc)

    # AUD29.1: optionally register NpuBackend if BBQ-Dev src is present.
    try:
        import sys as _sys
        from samia.runtime import npu_backend as _npb
        _npb.register(_sys.modules[__name__])
    except Exception as _exc:
        _log.info("npu backend registration skipped: %s", _exc)

    # AUD32: optionally register Qwen3-30B-A3B as 'qwen3_chat' backend
    # (used by the orchestrator). Default-OFF until ASTHENOS_QWEN3_CHAT_MODEL set.
    try:
        import sys as _sys
        from samia.runtime import qwen3_backend as _qwen3
        _qwen3.register(_sys.modules[__name__])
    except Exception as _exc:
        _log.info("qwen3_chat backend registration skipped: %s", _exc)

    register_op("judge", _op_judge)
    register_op("infer", _op_infer)
    register_op("inference_status", _op_inference_status)
    register_op("inference_reset", _op_inference_reset)
    register_op("inference_telemetry_status", _op_inference_telemetry_status)
    register_op("inference_backends", _op_inference_backends)
    register_op("inference_reset_cuda", _op_inference_reset_cuda)
    _log.info("inference ops registered (backend=%s, telemetry=%s)",
              backend_name, _emitter.events_dir)
    return _backend


def register_backend(name: str, backend: "InferenceBackend") -> None:
    """AUD29.3: external entry for additional backends to register themselves.

    Called by samia.runtime.bitnet_backend / npu_backend at module load. Uses
    short canonical names: "bitnet", "npu". Idempotent — re-registration replaces.
    """
    _backend_registry[name.lower()] = backend
    _log.info("inference: backend %r registered (%s)", name.lower(), type(backend).__name__)


def _op_inference_backends(args: dict[str, Any]) -> dict[str, Any]:
    """AUD29.3: list which backends are currently registered + their load state."""
    out = []
    for name, backend in _backend_registry.items():
        out.append({
            "name": name,
            "class": type(backend).__name__,
            "loaded": backend.is_loaded() if hasattr(backend, "is_loaded") else None,
            "model_path": getattr(backend, "_model_path", None),
        })
    env_default = os.environ.get("ASTHENOS_INFERENCE_BACKEND")
    return {"backends": out, "env_default": env_default}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.inference
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD82  (CUDA fallback chain + inference_reset_cuda op)
#        + FIX-2026-06-08: get_backend() is now a LOAD-ONCE singleton via the
#          per-model-path _model_backend_cache (get_backend_for_model). Kills the
#          double Qwen-14B load (register_ops + contradiction judge each built a
#          fresh LlamaCppBackend for the same gguf, ~18GB). A given gguf now loads
#          at most once; the dedicated BitNet-2B judge backend rides the same cache.
#          (lineage: AUD26-26.3 backends + AUD28-28.2 telemetry + AUD82 fallback.)
# Layer:      runtime (long-lived process)
# Role:       protocol-abstracted local LLM inference — Mock/LlamaCpp backends, a
#             load-once get_backend() factory, the CPU/BitNet fallback chain, JSONL
#             telemetry, and the judge/infer/status/reset IPC ops.
# Stability:  stable — v26.3+28.2+82+doubleload; protocol + factory + telemetry settled.
# ErrorModel: fail-soft — get_backend() falls back to MockBackend with a warning when
#             no real model is available; the fallback chain degrades CPU->BitNet->error;
#             IPC handlers return ok=False rather than raising; telemetry append is best-effort.
# Depends:    datetime, hashlib, json, logging, os, re, time, pathlib, typing (stdlib).
#             llama_cpp (lazy-imported, optional). samia.runtime.ipc (register_op).
# Exposes:    InferenceBackend, MockBackend, LlamaCppBackend, get_backend,
#             get_backend_for_model, register_backend, register_ops,
#             inference_fallback_chain, TelemetryEmitter.
# Lines:      991
# --------------------------------------------------------------------------
