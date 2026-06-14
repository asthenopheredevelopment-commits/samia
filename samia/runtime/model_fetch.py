"""samia.runtime.model_fetch -- self-fetching gguf LLM weights for SAM/IA.

Layer 1 (Owns / Depends):
    Owns:    KNOWN model registry (logical name -> source URL + license +
             size/sha sanity bound), MODELS_DIR resolver, fetch_model()
             (the on-demand downloader), ModelFetchError.
    Depends: stdlib only (hashlib, os, shutil, sys, urllib.request, pathlib).
             No third-party deps so the runtime can self-fetch on a clean box
             before llama-cpp-python / sentence-transformers are even imported.

Layer 2 (What / Why):
    What: Extends the embedder's "just works" download UX (sentence-transformers
          pulls all-MiniLM-L6-v2 on first use) to the gguf LLM arms. A small
          registry maps a LOGICAL model name to its HuggingFace `resolve` URL,
          its license, and a verification bound (sha256 when known, else a size
          sanity range). fetch_model(name_or_path) returns a local Path:
            - if the path already exists on disk, return it untouched;
            - if it is a KNOWN logical name (or a registry path whose file is
              missing), download it into the XDG models dir
              (~/.local/share/asthenos/models) atomically (.part -> rename),
              streaming progress to stderr, verifying size/sha, and printing a
              one-line license notice BEFORE the bytes move.
          The download is gated by the shared consent protocol in
          samia.core.netconsent, keyed on ASTHENOS_MODEL_AUTOFETCH:
            - explicitly OFF ("0"/"false"/"no"/"off") -> refuse with a
              copy-pasteable manual-download instruction (the kill switch);
            - explicitly ON ("1"/"true"/...) -> standing operator consent,
              proceed without prompting (the path agent/CI flows export after
              clearing their own permission gate);
            - NOT SET (the v1 release default) -> ask-if-interactive: at a tty,
              print what/size/license/source and prompt "Download? [y/N]"; with
              no tty, refuse and name BOTH remedies (export the env on, or place
              the file manually). NO silent download ever happens by default.
          Any failure raises ModelFetchError and never leaves a partial file.
    Why:  the public-release promise is "the memory system is self-fetching for
          any LLMs we use". The embedder already self-fetches; the contradiction
          judge / fact-extractor gguf arms did not, so a fresh install fell back
          to MockBackend with no path to recover except a manual curl. This
          module closes that gap while keeping the operator informed (license
          notice + progress) and the disk clean (atomic write, no partials).

Layer 3 (Changelog):
    2026-06-10  FEAT       Initial implementation. KNOWN registry (Qwen3-4B
                            Q4_K_M + BitNet-2B i2_s), fetch_model() with
                            autofetch gate, atomic write, progress, size/sha
                            verification, license notice.
    2026-06-12  SEC        Consent protocol: the bare _autofetch_enabled() gate
                            is replaced by samia.core.netconsent.consent(). Unset
                            env now ASKS at a tty / refuses non-tty instead of
                            silently downloading; explicit on = standing consent,
                            explicit off = kill switch. No silent downloads.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

from samia.core import netconsent

# ---------------------------------------------------------------------------
# XDG models dir
# ---------------------------------------------------------------------------

# What: the canonical on-disk home for fetched weights.
# Why:  matches where the rest of the runtime already looks for ggufs
#       (contradiction.py's BitNet path, the operator's cls-flags.conf model
#       paths) so a fetched file lands exactly where callers resolve it.
MODELS_DIR = Path.home() / ".local" / "share" / "asthenos" / "models"

# What: the env gate controlling on-demand download.
# Why:  re-exported from samia.core.netconsent so this module and the embedder
#       (core/vector.py) share ONE knob name. Semantics now live in netconsent:
#       off = refuse (kill switch), on = standing consent, unset = ask-if-tty /
#       refuse-non-tty (the v1 "no silent download" default). Re-exported (not
#       redefined) so callers/tests that read model_fetch.AUTOFETCH_ENV are
#       guaranteed the identical name.
AUTOFETCH_ENV = netconsent.AUTOFETCH_ENV


class ModelFetchError(RuntimeError):
    """Raised when a model cannot be fetched (gated off, network/verify fail).

    What: a single typed failure for every fetch_model() error path.
    Why:  callers (get_backend_for_model) catch ONE exception type to fail-soft
          to MockBackend; the message always carries the actionable next step.
    """


# ---------------------------------------------------------------------------
# KNOWN model registry
# ---------------------------------------------------------------------------

# What: logical model name -> {url, license, license_url, and ONE of
#       sha256 / (size_min, size_max)} sanity bound.
# Why:  the registry is the single source of truth for "where do our models
#       come from and how do we know we got the right bytes". `url` is the
#       HuggingFace `resolve` URL (direct file download). Verification prefers a
#       pinned sha256; when we only have an observed on-box size we record a
#       tolerant (size_min, size_max) window so a truncated/HTML-error download
#       is rejected without pinning to a single byte count that a re-quant could
#       legitimately shift.
KNOWN: dict[str, dict] = {
    # Qwen3-4B-Instruct-2507 Q4_K_M -- the validated chat/judge/extract model on
    # this box (unsloth GGUF repo, the source the operator downloaded from).
    # Apache-2.0. Observed on-box size 2,497,281,120 bytes.
    "Qwen3-4B-Instruct-2507-Q4_K_M": {
        "url": (
            "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/"
            "resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        ),
        "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "license": "Apache-2.0",
        "license_url": (
            "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF"
        ),
        # No pinned sha (unsloth re-quants periodically); tolerant size window
        # around the observed 2.497 GB catches truncation / HTML error pages.
        "size_min": 2_300_000_000,
        "size_max": 2_700_000_000,
    },
    # BitNet-b1.58-2B-4T i2_s -- MIT. Observed on-box size 1,187,801,280 bytes.
    # NOTE: llama-cpp-python CANNOT load the i2_s special quantization (it needs
    # bitnet.cpp). Kept in the registry so bitnet.cpp users can still self-fetch
    # it; the inference backends fall back to the Qwen judge model for
    # llama-cpp-python (see contradiction.py judge-model selection).
    "BitNet-b1.58-2B-4T-i2_s": {
        "url": (
            "https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/"
            "resolve/main/ggml-model-i2_s.gguf"
        ),
        "filename": "BitNet-b1.58-2B-4T-i2_s.gguf",
        "license": "MIT",
        "license_url": (
            "https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf"
        ),
        "size_min": 1_100_000_000,
        "size_max": 1_300_000_000,
        # bitnet.cpp only -- llama-cpp-python rejects the i2_s quant.
        "llama_cpp_loadable": False,
    },
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _registry_entry(name_or_path: str) -> Optional[dict]:
    """Return the KNOWN entry for a logical name or a registry filename/path.

    What: matches three shapes -> the same entry: the logical key
          ("Qwen3-4B-Instruct-2507-Q4_K_M"), the bare filename
          ("...gguf"), or any path whose basename is a registry filename.
    Why:  callers pass whatever they have (a logical name from a config, or a
          concrete gguf path from an env var); all roads that name a KNOWN model
          must resolve to its source entry so a missing file can be re-fetched.
    """
    if name_or_path in KNOWN:
        return KNOWN[name_or_path]
    base = Path(name_or_path).name
    for entry in KNOWN.values():
        if entry["filename"] == base or entry["filename"] == name_or_path:
            return entry
    # also allow the logical key to match a path's stem (".gguf" stripped)
    stem = Path(name_or_path).name
    if stem.endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    return KNOWN.get(stem)


def _target_path(entry: dict) -> Path:
    """The on-disk destination for a registry entry (MODELS_DIR / filename)."""
    return MODELS_DIR / entry["filename"]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify(path: Path, entry: dict) -> None:
    """Raise ModelFetchError unless *path* satisfies the entry's sanity bound.

    What: sha256 when the entry pins one, else a tolerant (size_min, size_max)
          window.
    Why:  a truncated download or an HTML error page must be rejected BEFORE the
          atomic rename promotes it to the canonical filename. The verifier runs
          on the .part file so a failure cleans up without ever exposing a bad
          model under the real name.
    """
    size = path.stat().st_size
    sha = entry.get("sha256")
    if sha:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got.lower() != sha.lower():
            raise ModelFetchError(
                f"sha256 mismatch for {path.name}: expected {sha}, got {got}"
            )
        return
    lo = entry.get("size_min")
    hi = entry.get("size_max")
    if lo is not None and size < lo:
        raise ModelFetchError(
            f"{path.name} too small ({size} bytes < {lo}); "
            f"download likely truncated or an error page"
        )
    if hi is not None and size > hi:
        raise ModelFetchError(
            f"{path.name} too large ({size} bytes > {hi}); "
            f"unexpected payload, refusing"
        )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download(entry: dict, dest: Path) -> Path:
    """Download entry['url'] -> dest atomically; verify; return dest.

    What: streams the URL to a sibling .part file, prints progress to stderr
          every ~5%, verifies the completed .part, then os.replace()-renames it
          onto *dest*. On ANY failure the .part is removed and ModelFetchError
          is raised -- never a partial file under the real name.
    Why:  atomic rename means a half-finished or interrupted fetch can never be
          mistaken for a complete model; verification-before-rename means a
          corrupt download is caught while still quarantined as .part.
    """
    url = entry["url"]
    host = _host(url)
    name = entry["filename"]
    license_name = entry.get("license", "unknown license")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    # Clean any stale partial from a prior aborted run.
    if part.exists():
        part.unlink()

    # License notice BEFORE the bytes move (operator consent / transparency).
    print(
        f"Fetching {name} ({license_name}) from {host}",
        file=sys.stderr,
        flush=True,
    )
    if entry.get("license_url"):
        print(f"  license: {entry['license_url']}", file=sys.stderr, flush=True)

    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 (known HF host)
            total = _content_length(resp)
            written = 0
            next_pct = 5
            with part.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = int(written * 100 / total)
                        if pct >= next_pct:
                            print(
                                f"  {name}: {pct}% "
                                f"({written}/{total} bytes)",
                                file=sys.stderr,
                                flush=True,
                            )
                            next_pct = (pct // 5 + 1) * 5
        # Verify the completed .part BEFORE promoting it.
        _verify(part, entry)
        os.replace(part, dest)
        print(f"  {name}: done -> {dest}", file=sys.stderr, flush=True)
        return dest
    except ModelFetchError:
        _cleanup(part)
        raise
    except Exception as exc:  # network / IO / verify
        _cleanup(part)
        raise ModelFetchError(
            f"failed to fetch {name} from {host}: {exc}"
        ) from exc


def _cleanup(part: Path) -> None:
    """Remove a .part file if present (best-effort, never raises)."""
    try:
        if part.exists():
            part.unlink()
    except OSError:
        pass


def _content_length(resp) -> int:
    """Total bytes from the response, or 0 when the header is absent."""
    try:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else 0
    except (AttributeError, TypeError, ValueError):
        return 0


def _host(url: str) -> str:
    """Bare host of a URL for the license notice (no urllib.parse import noise)."""
    from urllib.parse import urlparse

    return urlparse(url).netloc or url


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_model(name_or_path: str) -> Path:
    """Resolve *name_or_path* to a local gguf Path, downloading if needed.

    What:
      - if the given path exists on disk, return it unchanged;
      - else if it names a KNOWN model (logical name, bare filename, or a path
        whose basename is a registry filename), download it to MODELS_DIR and
        return the resulting Path;
      - else raise ModelFetchError (unknown + not present).
    The download is gated by the shared consent protocol (samia.core.netconsent,
    keyed on ASTHENOS_MODEL_AUTOFETCH): explicit-off refuses, explicit-on is
    standing consent, unset asks at a tty / refuses with no tty. Any refusal
    raises ModelFetchError with a copy-pasteable manual instruction.

    Why:  one call covers both "the file is already here" and "self-fetch it",
          so wiring sites stay trivial: ask for the model, get a real path or a
          clear error.
    """
    # 1) Already on disk under the exact path the caller gave -> use it.
    given = Path(name_or_path)
    if given.exists():
        return given

    # 2) Is it a KNOWN model we can fetch?
    entry = _registry_entry(name_or_path)
    if entry is None:
        raise ModelFetchError(
            f"unknown model '{name_or_path}' and no file at that path; "
            f"known models: {sorted(KNOWN)}"
        )

    dest = _target_path(entry)
    # The canonical destination may already exist even if the caller's path did
    # not (caller passed a logical name / a different dir).
    if dest.exists():
        return dest

    # Route the download decision through the shared consent protocol. The
    # manual_hint carries the exact dest + source so a refusal (kill switch,
    # declined prompt, or non-tty) always tells the operator how to proceed.
    manual_hint = (
        f"manually place the file at {dest} (source: {entry['url']}), "
        f"or set {AUTOFETCH_ENV}=1 to enable self-fetch."
    )
    approved = netconsent.consent(
        what=entry["filename"],
        size_hint=_size_hint(entry),
        license_str=entry.get("license", "unknown license"),
        source=_host(entry["url"]),
        manual_hint=manual_hint,
    )
    if not approved:
        raise ModelFetchError(
            f"refusing to auto-download {entry['filename']}. {manual_hint}"
        )

    return _download(entry, dest)


def _size_hint(entry: dict) -> str:
    """A human size hint for the consent notice from the entry's sanity bound.

    What: prefers the registry's size window midpoint (~N.N GB); falls back to
          "unknown size" when the entry only pins a sha256.
    Why:  the consent prompt should tell the operator roughly how big the pull
          is before they answer; the registry already records a size window for
          verification, so reuse it rather than hitting the network for a HEAD.
    """
    lo = entry.get("size_min")
    hi = entry.get("size_max")
    if lo is not None and hi is not None:
        mid_gb = (lo + hi) / 2 / 1_000_000_000
        return f"~{mid_gb:.1f}GB"
    if hi is not None:
        return f"<={hi / 1_000_000_000:.1f}GB"
    return "size unknown"


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.model_fetch
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      original SAM/IA runtime module — 2026-06-10 FEAT (KNOWN registry +
#             gated atomic fetch); 2026-06-12 SEC (consent protocol via
#             samia.core.netconsent replacing the bare autofetch gate).
# Layer:      runtime (library helper, no daemon loop)
# Role:       on-demand self-fetching of gguf LLM weights into MODELS_DIR (release
#             UX parity with the auto-downloading sentence-transformers embedder).
# Stability:  stable — KNOWN registry + consent-gated atomic fetch; no silent
#             download in the v1 release default (unset env asks-if-tty / refuses).
# ErrorModel: every failure raises ModelFetchError (gated-off refusal, unknown
#             model, network/IO, size/sha verify fail) and never leaves a partial
#             file — the .part is verified before the atomic rename and cleaned up
#             on any error; callers fail-soft to MockBackend on that one type.
# Depends:    hashlib, os, shutil, sys, urllib.request, urllib.parse, pathlib,
#             typing (stdlib). samia.core.netconsent (the consent gate).
# Exposes:    fetch_model, ModelFetchError, KNOWN, MODELS_DIR, AUTOFETCH_ENV.
# Gate:       samia.core.netconsent.consent on ASTHENOS_MODEL_AUTOFETCH
#             (off=refuse kill switch, on=standing consent, unset=ask-if-tty /
#             refuse-non-tty; v1 release default = NO silent download).
# Wired:      samia.runtime.inference.get_backend_for_model (fail-soft).
# Lines:      428
# --------------------------------------------------------------------------
