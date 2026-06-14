"""samia.runtime.contradiction -- AUD60 embedding-similarity contradiction detection.

Layer 1 (Owns / Depends):
    Owns:    Embedding-based contradiction candidate finder + optional LLM judge gate
             for memory writes, the CANONICAL unified supersession-candidate store,
             the PASSIVE REM-subscriber sweep, and the Tier-2 P2 abstraction
             synthesizer — split by responsibility into four submodules behind this
             re-export facade (the public import surface is byte-for-byte unchanged
             from the pre-split single module):
               - config   : the re-exported stdlib (json/logging/os, datetime/timezone,
                            Path/Any/Optional), the package logger _log + _now_iso, EVERY
                            tuning constant + prompt template, the SINGLE-OWNED mutable
                            state (_MEMORY_DIR / _TYPE_CACHE), and the live env/scoping
                            readers (is_enabled / auto_cosine_threshold / configure /
                            excluded_types / is_excluded_node / _node_type) — the
                            package's shared, single-owned leaf.
               - detection: the Phase-1 embedding finder (_embed_text / _load_index /
                            find_contradiction_candidates) + the P3a supersession finder
                            (_node_text_for_id / find_supersession_candidates).
               - store    : the R2 canonical supersession store (record + list-unresolved
                            + mark-confirmed/dismissed, dedup'd, crash-safe).
               - judge    : the Phase-2 LLM judge gate + the P2 synthesizer (the dedicated
                            cached backend _judge_backend / _inference_available /
                            _infer_text, _parse_first_json_object, judge_contradictions,
                            synthesis_enabled, synthesize_node).
               - passes   : the PASSIVE REM sweep (passive_sweep + helpers +
                            passive_has_work) and the AUD60 memory-guard integration
                            (check_contradiction).
    Depends: samia.core.vector (embedding infra, optional), samia.core.frontmatter
             (node reading), samia.core.consolidation (shingles/jaccard, optional),
             numpy (optional), samia.runtime.inference (in-process LLM judge backend,
             optional), and lazily samia.runtime.rem_cycle + samia.core.{temporal,ia,
             integrity,bio} — almost all reached via function-local imports in the
             submodules to keep the heavy/runtime deps off the package import path.

Layer 2 (What / Why):
    What: the three-phase contradiction pipeline + the supersession arms. Phase 1
          embedding similarity, Phase 2 LLM judge gate, Phase 3 memory-guard
          integration; plus the unified store, the passive sweep, and the P2 synth.
    Why:  the 1659-line monolith was split by RESPONSIBILITY (config / detection /
          store / judge / passes) with ZERO behavior change; this facade re-exports
          the FULL public surface so every importer (`from samia.runtime import
          contradiction` / `from samia.runtime.contradiction import X`) and every
          attribute reach-in (public OR private) is unaffected.

PATCH SEAMS + FACADE-REBOUND STATE (exemplar rule, HIGH blast radius — 24 importers):
    Several names are BOTH mock.patch.object(contradiction, ...) / facade-rebind
    targets AND called/read by a sibling submodule, so each sibling reaches them
    THROUGH this package facade (from samia.runtime import contradiction as _pkg;
    _pkg.<name>) so a package-level patch/rebind rebinds the attribute the caller
    actually reads:
      - the patch-seam FUNCTIONS: find_contradiction_candidates (passes/detection),
        find_supersession_candidates (passes), judge_contradictions (passes),
        list_supersession_candidates (passes), is_enabled (passes), _embed_text
        (detection's finder), _judge_backend (judge's _inference_available/_infer_text),
        synthesis_enabled (judge's synthesize_node);
      - the facade-rebound STATE: _ENABLED / _JUDGE_ENABLED (tests rebind on the
        facade; check_contradiction / judge_contradictions / synthesis_enabled read
        through the facade) and _MEMORY_DIR (configure() writes + the finders read
        through the facade, single-owned in config);
      - _log: the SHARED logger object (config._log == this facade's _log) whose
        .warning the live-isolation test patches in place — every arm logs through it.
    _TYPE_CACHE is single-owned in config; tests call con._TYPE_CACHE.clear() (in-place)
    on this re-exported object.

Public surface re-exported here (byte-for-byte the pre-split module — 26 names):
    re-exported imports : Any, Optional, Path, annotations, datetime, json, logging,
                          os, timezone
    functions           : auto_cosine_threshold, check_contradiction, configure,
                          excluded_types, find_contradiction_candidates,
                          find_supersession_candidates, is_enabled, is_excluded_node,
                          judge_contradictions, list_supersession_candidates,
                          mark_supersession_confirmed, mark_supersession_dismissed,
                          passive_has_work, passive_sweep, record_supersession_candidate,
                          synthesis_enabled, synthesize_node
Internal names also re-exported for direct test/importer/patch-seam access (NOT in
__all__): _log, _now_iso, _ENABLED, _JUDGE_ENABLED, _MEMORY_DIR, _TYPE_CACHE,
    _clear_type_cache, _node_type, _judge_model_path, _SEMANTIC_PAIR_THRESHOLD,
    _SUPERSESSION_JACCARD, plus the remaining module-level privates the monolith
    exposed (_embed_text, _load_index, _node_text_for_id, _supersession_path,
    _mark_supersession_candidate, _judge_backend, _inference_available, _infer_text,
    _parse_first_json_object, _list_node_ids, _node_field, _pick_superseded,
    _salience_guards_supersede) and the remaining tuning constants/templates.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

# Reload cascade — What: when THIS package facade is reloaded (importlib.reload of
#   samia.runtime.contradiction), re-execute the config submodule first so its
#   env-derived constants (_SEMANTIC_PAIR_THRESHOLD / _COSINE_THRESHOLD / the
#   exclude-types default / the passive budget, etc.) re-read the environment.
# Why: in the pre-split monolith importlib.reload(contradiction) re-ran the whole
#   module, re-reading every `float(os.environ.get(...))` constant; that contract is
#   relied on by test_semantic_threshold.test_env_override. After the split those
#   constants live in config, which a plain reload of this __init__ would NOT
#   re-execute (the `from .config import ...` below just rebinds config's stale
#   values). Detecting a reload via "config already imported" and reloading it keeps
#   the env-honored-on-reload behavior byte-for-byte. On the FIRST import config is
#   not yet in sys.modules, so this no-ops and config loads once via the import below.
_cfg_mod = _sys.modules.get(__name__ + ".config")
if _cfg_mod is not None:
    _importlib.reload(_cfg_mod)

# The shared leaf — the re-exported stdlib (json/logging/os + datetime/timezone +
# Path/Any/Optional; `annotations` rides the `from __future__` above), the package
# logger _log + _now_iso stamp, the SINGLE-OWNED mutable state (_MEMORY_DIR /
# _TYPE_CACHE), every tuning constant + prompt template, and the live env/scoping
# readers. json/logging/os/datetime/timezone/Path/Any/Optional are part of the public
# surface, so they must stay importable from the package facade.
from .config import (  # noqa: F401
    # re-exported stdlib (public surface)
    Any,
    Optional,
    Path,
    datetime,
    json,
    logging,
    os,
    timezone,
    # logger + stamp
    _log,
    _now_iso,
    # single-owned mutable state
    _MEMORY_DIR,
    _TYPE_CACHE,
    # constants + templates
    _ENABLED,
    _JUDGE_ENABLED,
    _COSINE_THRESHOLD,
    _SEMANTIC_PAIR_THRESHOLD,
    _JUDGE_CONFIDENCE_THRESHOLD,
    _MAX_CANDIDATES,
    _ONLINE_AUTO_COSINE,
    _SUPERSESSION_JACCARD,
    _SUPERSESSION_STORE,
    _PASSIVE_CURSOR_KEY,
    _PASSIVE_BUDGET,
    _JUDGE_INFER_MAX_TOKENS,
    _SYNTH_INFER_MAX_TOKENS,
    _JUDGE_MODEL_DEFAULT,
    _DEFAULT_EXCLUDE_TYPES,
    _JUDGE_PROMPT_TEMPLATE,
    _SYNTH_PROMPT_TEMPLATE,
    # scoping + env readers
    excluded_types,
    is_excluded_node,
    is_enabled,
    auto_cosine_threshold,
    configure,
    _node_type,
    _clear_type_cache,
    _judge_model_path,
)

# The Phase-1 embedding finder + the P3a supersession finder. _embed_text is a
# mock.patch.object seam (re-exported so the finder's facade reach + a package-level
# patch line up); _load_index / _node_text_for_id are re-exported for parity.
from .detection import (  # noqa: F401
    find_contradiction_candidates,
    find_supersession_candidates,
    _embed_text,
    _load_index,
    _node_text_for_id,
)

# The R2 canonical unified supersession store.
from .store import (  # noqa: F401
    record_supersession_candidate,
    list_supersession_candidates,
    mark_supersession_confirmed,
    mark_supersession_dismissed,
    _supersession_path,
    _mark_supersession_candidate,
)

# The Phase-2 LLM judge gate + the Tier-2 P2 synthesizer. _judge_backend is a
# mock.patch.object seam (the in-module _inference_available/_infer_text reach it
# through this facade); the rest are re-exported for direct test access + parity.
from .judge import (  # noqa: F401
    judge_contradictions,
    synthesis_enabled,
    synthesize_node,
    _judge_backend,
    _inference_available,
    _infer_text,
    _parse_first_json_object,
)

# The PASSIVE REM sweep + memory-guard integration.
from .passes import (  # noqa: F401
    passive_sweep,
    passive_has_work,
    check_contradiction,
    _list_node_ids,
    _node_field,
    _pick_superseded,
    _salience_guards_supersede,
)

# __all__ — the LOCALLY-owned PUBLIC names (the 26 the baseline records: 9 re-exported
# imports + 17 functions). The verify script diffs the full public surface (dir() minus
# underscore names) against the baseline; __all__ documents the intended export set and
# bounds `from ... import *` to exactly the pre-split public 26. (The private test/
# importer/patch-seam-reached names above are re-exported but intentionally NOT in
# __all__, mirroring the exemplars.)
__all__ = [
    # re-exported imports
    "Any", "Optional", "Path", "annotations", "datetime", "json", "logging",
    "os", "timezone",
    # functions
    "auto_cosine_threshold", "check_contradiction", "configure", "excluded_types",
    "find_contradiction_candidates", "find_supersession_candidates", "is_enabled",
    "is_excluded_node", "judge_contradictions", "list_supersession_candidates",
    "mark_supersession_confirmed", "mark_supersession_dismissed", "passive_has_work",
    "passive_sweep", "record_supersession_candidate", "synthesis_enabled",
    "synthesize_node",
]


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.contradiction
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD60 Phases 1-3 (embedding + judge + guard integration)
#             + FEAT-2026-06-07 P3a/P3b/R2 (supersession finder + CANONICAL unified
#               store) + P3c (PASSIVE REM-subscriber sweep) + Tier-2 P2 (abstraction
#               synthesis) + FIX-2026-06-08 (in-process inference rewire) + the
#               BUG-2026-06-07/06-11 fixes (by_row manifest read, judge-parse
#               raw_decode tolerance, supersession dedup) + TUNE-2026-06-08/06-10
#               (type-scoping, dedicated fast judge, recall-first bars).
#             + Phase-B modularization: the 1659-line monolith carved into a
#               re-export-preserving package (config/detection/store/judge/passes)
#               with ZERO behavior change; this __init__ re-exports the full public
#               surface so every importer + attribute reach-in is unaffected.
# Layer:      runtime (library helper, no daemon loop)
# Role:       re-export facade — the package's PUBLIC import surface, byte-for-byte
#             identical to the pre-split module. `from samia.runtime import
#             contradiction` / `from samia.runtime.contradiction import X` keeps
#             working for all 26 public names; the private test/importer/patch-seam-
#             reached names (_log/_now_iso/_ENABLED/_JUDGE_ENABLED/_MEMORY_DIR/
#             _TYPE_CACHE/_embed_text/_judge_backend/_inference_available/_infer_text/
#             _node_type/_clear_type_cache/_SEMANTIC_PAIR_THRESHOLD/_SUPERSESSION_JACCARD
#             + the remaining module-level privates + constants) are re-exported too.
# Stability:  stable — pure re-export; the implementation lives in the submodules.
#             v0.4 — all phases wired, default-off via env vars.
# ErrorModel: none here (import-time wiring only); each submodule footer documents its
#             own fail-open / fail-soft / gated-and-inert posture.
# Depends:    .config, .detection, .store, .judge, .passes.
# Exposes:    the public 26 (in __all__) + the private/patch-seam/state names above.
# Note:       PATCH SEAMS (high blast radius) — find_contradiction_candidates /
#             find_supersession_candidates / judge_contradictions /
#             list_supersession_candidates / is_enabled / _embed_text / _judge_backend
#             / synthesis_enabled are reached by sibling submodules THROUGH this facade
#             so a package-level mock.patch.object rebinds what the caller runs; the
#             facade-rebound flags _ENABLED / _JUDGE_ENABLED + the single-owned
#             _MEMORY_DIR are read/written through the facade too (config single-owns
#             them). The ONLINE/passive auto-supersede paths are GATED OFF by default
#             (ASTHENOS_CONTRADICTION_ENABLED).
# Lines:      261
# --------------------------------------------------------------------------
