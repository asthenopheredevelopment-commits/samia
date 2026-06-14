"""samia.runtime.contradiction.config — shared base of the contradiction package.

Layer 1 (Owns / Depends):
    Owns:    the module-top stdlib the monolith pulled in and that callers + tests
             reach THROUGH the package facade (json/logging/os, datetime/timezone,
             Path, Optional/Any, the `from __future__` annotations); the package
             logger _log + the _now_iso provenance stamp; EVERY tuning constant and
             prompt template (the scoring/threshold bars, the judge/synth budgets +
             model default, the store filename, the passive cursor key + budget, the
             exclude-type default, the two LLM prompt templates); the SINGLE-OWNED
             mutable module state (_MEMORY_DIR set by configure(), the _TYPE_CACHE
             node-type cache); and the small live env/scoping readers (is_enabled,
             auto_cosine_threshold, configure, excluded_types, _node_type,
             _clear_type_cache, is_excluded_node).
    Depends: samia.core.frontmatter (lazy, function-local in _node_type — the node
             read seam). json/logging/os/datetime/pathlib/typing from stdlib.

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — every sibling submodule imports
          its constants, its logger, its _now_iso, and its scoping helpers from here,
          and the mutable module state (_MEMORY_DIR / _TYPE_CACHE) is owned here so
          there is ONE copy of each. The flag/state names tests rebind on the package
          facade (_ENABLED / _JUDGE_ENABLED / _MEMORY_DIR) live here and are read by
          the siblings THROUGH the package facade so a facade-level rebind is honored.
    Why:  splitting the 1659-line monolith by responsibility (detection / store /
          judge / passes) leaves a shared base of imports + constants + the node-type
          cache + the env readers that all the arms need; concentrating them here
          keeps the import graph acyclic (config depends on nothing else in the
          package) and the tuning bars single-sourced.
"""

from __future__ import annotations

# Re-exported module-top names the monolith pulled in and other code (importers +
# tests) reaches THROUGH the package facade. The baseline records json/logging/os/
# datetime/timezone/Path/Any/Optional/annotations as part of the public surface, so
# they must stay importable from the package facade — they are owned here.
import json  # noqa: F401
import logging  # noqa: F401
import os  # noqa: F401
from datetime import datetime, timezone  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Optional  # noqa: F401

# _log — What: the single package logger. _log.warning is mock.patch.object'd on the
#   PACKAGE facade (test_memory_guard_live_isolation patches con._log.warning to
#   capture the passive sweep's summarized warnings), so every sibling must log
#   through THIS shared object (reached via config._log == the facade's _log) for the
#   in-place method patch to be seen.
_log = logging.getLogger("samia.runtime.contradiction")


def _now_iso() -> str:
    """UTC ISO-8601 timestamp for candidate provenance records."""
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Configuration constants (single-owned here; siblings import them through config)
# ---------------------------------------------------------------------------

# _ENABLED — What: master switch for contradiction detection.
# _ENABLED — Why: default-off so the feature doesn't impact write latency
#   until the operator explicitly enables it. The embedding model may not
#   be loaded, and numpy may not be available. Tests rebind this on the package
#   facade (contra._ENABLED = ...), so check_contradiction reads it through the
#   facade (not a stale local copy).
_ENABLED: bool = os.environ.get("ASTHENOS_CONTRADICTION_ENABLED", "0") == "1"

# _JUDGE_ENABLED — What: opt-in flag for the LLM judge gate.
# _JUDGE_ENABLED — Why: ON by default since TUNE-2026-06-10 — the lowered
#   cosine bar (below) is recall-first and NEEDS the BitNet stage-2 judge for
#   precision. Degrades gracefully (candidates-only) when the inference
#   backend is unavailable; latency rides the passive sweep, not hot writes.
#   Operator-directed 2026-06-10 (benchmark probe: TPR 0.2 at the old bar).
#   Tests rebind this on the package facade, so the judge/synth readers reach it
#   through the facade.
_JUDGE_ENABLED: bool = os.environ.get("ASTHENOS_CONTRADICTION_JUDGE", "1") == "1"

# _COSINE_THRESHOLD — What: minimum cosine similarity to flag a candidate.
# _COSINE_THRESHOLD — Why: TUNE-2026-06-10, operator-directed: 0.75 (AUD60)
#   missed 8/10 genuine paraphrased supersessions in the competitive-benchmark
#   probe (MiniLM paraphrase cosines run 0.49-0.95; title-prefix dilutes
#   further). 0.57 is recall-first; precision is recovered by the stage-2
#   BitNet judge (now default-on above). FPR at the candidate layer is bounded
#   by type-scoping (see TUNE-2026-06-08 below).
_COSINE_THRESHOLD: float = float(
    os.environ.get("ASTHENOS_CONTRADICTION_THRESHOLD", "0.57")
)

# _SEMANTIC_PAIR_THRESHOLD — What: the HIGHER cosine bar applied to any pair
#   involving a machine-generated semantic atom (type: semantic).
# _SEMANTIC_PAIR_THRESHOLD — Why: TUNE-2026-06-10 (2) — the fact-extract
#   backfill grew the scoped content corpus 129 -> 5,789 nodes; at the 0.57
#   recall-first bar that is 175,017 pairs (noise ocean: atoms share one
#   template, so their BASELINE mutual similarity sits in the 0.57-0.75 band
#   where hand-written paraphrased supersessions live). Atoms are deduped at
#   0.92 at creation, so >=0.92 survivors are genuinely actionable (~3.4k,
#   judge-drainable). Hand-written-only pairs keep the operator's 0.57.
#   (semantic_recall reaches this bar through the package facade.)
_SEMANTIC_PAIR_THRESHOLD: float = float(
    os.environ.get("ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD", "0.92")
)

# _JUDGE_CONFIDENCE_THRESHOLD — What: minimum LLM judge confidence to block.
# _JUDGE_CONFIDENCE_THRESHOLD — Why: 0.7 per AUD60 proposal. Below this,
#   the contradiction is flagged but not blocked.
_JUDGE_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ASTHENOS_CONTRADICTION_JUDGE_CONF", "0.7")
)

# _MAX_CANDIDATES — What: maximum candidates to send to the LLM judge.
# _MAX_CANDIDATES — Why: bounds inference cost per write (max 10 per AUD60).
_MAX_CANDIDATES: int = 10

# _ONLINE_AUTO_COSINE — What: cosine bar at/above which the ONLINE write path
#   AUTO-supersedes (no LLM judge) the exact case (same subject key + this cosine).
# _ONLINE_AUTO_COSINE — Why: the Q4-granularity decision — online has no judge,
#   so it auto-acts ONLY on the obvious exact-supersession case (default 0.92,
#   env-tunable). Weaker hits (0.75 <= cosine < 0.92) are recorded for the passive
#   judge, not auto-deleted. Made safe by reversibility via restore_node.
_ONLINE_AUTO_COSINE: float = float(
    os.environ.get("ASTHENOS_SUPERSESSION_AUTO_COSINE", "0.92")
)

# _MEMORY_DIR — What: root memory directory for node access.
# _MEMORY_DIR — Why: used by the embedding candidate finder to locate the vector
#   index and node files. SINGLE-OWNED here: configure() writes it through the
#   package facade and the finders read it through the package facade, so a test
#   that rebinds contra._MEMORY_DIR is honored (and there is never a second copy).
_MEMORY_DIR: Optional[Path] = None

# ---------------------------------------------------------------------------
# TUNE-2026-06-08: TYPE-SCOPING (the big lever)
# ---------------------------------------------------------------------------
#
# OPERATOR RULE: the contradiction/supersession detector EXCLUDES episodic /
# experiential records (a transcript of interactions / "this happened at time X"
# about experiences -- type session_offload, and bug = event records). It INCLUDES
# content / factual claims (project/reference/user/feedback). A song/doc transcript
# stored as reference IS content = included; the distinction is experiential-vs-
# content, mapped here onto the `type` frontmatter field.
#
# Empirically: cosine>=0.75 over the whole 2752-node index = ~807K candidate pairs
# (21% flood) because 2597 nodes are session_offload (episodic, self-similar).
# Scoping to contradictable types collapses that to ~152 nodes / ~393 pairs.
_DEFAULT_EXCLUDE_TYPES = "session_offload,bug"


def excluded_types() -> frozenset[str]:
    """The set of node `type` values the detector EXCLUDES (live env read).

    What: parses ASTHENOS_CONTRADICTION_EXCLUDE_TYPES (comma-separated, default
          "session_offload,bug") into a lowercased frozenset. Read each call so a
          test/daemon that sets the env after import sees the change.
    Why:  the type-scoping lever -- episodic/experiential records (session_offload
          transcripts, bug event records) are NOT contradictable content claims, so
          the detector must never enumerate or match them. Env-overridable so the
          operator can widen/narrow the exclusion without a code change.
    """
    raw = os.environ.get("ASTHENOS_CONTRADICTION_EXCLUDE_TYPES", _DEFAULT_EXCLUDE_TYPES)
    return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())


# _TYPE_CACHE — What: per-(memory_dir, node) cache of the resolved `type` field.
# _TYPE_CACHE — Why: passive_sweep + active_set + the finder each resolve a node's
#   type repeatedly across a sweep; reading frontmatter once and caching keeps the
#   scope check cheap (a sweep over thousands of nodes must not re-parse each file
#   per candidate). Keyed by (str(memory_dir), node-stem). Bounded by index size.
#   SINGLE-OWNED here; tests call con._TYPE_CACHE.clear() (in-place) on the facade,
#   which mutates THIS one dict object.
_TYPE_CACHE: dict[tuple[str, str], Optional[str]] = {}


def _clear_type_cache() -> None:
    """Drop the node-type cache (tests / after a node's type changes)."""
    _TYPE_CACHE.clear()


def _node_type(memory_dir: Path, node_id: str) -> Optional[str]:
    """Resolve a node's `type` frontmatter field (cached, lowercased).

    What: read nodes/<id>.md frontmatter and return its 'type' value lowercased,
          or None when the node is missing / unreadable / has no type. Cached per
          (memory_dir, stem) so a sweep resolves each node's type at most once.
    Why:  type-scoping needs the node's content/experiential class. Reading on
          demand + caching avoids loading every node up front and avoids re-parsing
          the same file across many candidate comparisons in one sweep.
    """
    stem = node_id[:-3] if node_id.endswith(".md") else node_id
    key = (str(memory_dir), stem)
    if key in _TYPE_CACHE:
        return _TYPE_CACHE[key]
    p = memory_dir / "nodes" / f"{stem}.md"
    val: Optional[str] = None
    if p.exists():
        # Lazy, function-local frontmatter import: keeps samia.core.frontmatter off
        # the package import path (the node read seam is only needed on a cache miss).
        try:
            from samia.core import frontmatter as _fm
            parsed, _ = _fm.parse(p.read_text(encoding="utf-8"))
            if parsed is not None:
                t = parsed[0].get("type")
                if isinstance(t, str) and t.strip():
                    val = t.strip().lower()
        except Exception:
            val = None
    _TYPE_CACHE[key] = val
    return val


def is_excluded_node(memory_dir: Path, node_id: str) -> bool:
    """True iff *node_id* is an excluded (episodic/experiential) node to SKIP.

    What: returns True when the node's resolved `type` is in excluded_types().
          Missing/unreadable type -> treat as INCLUDED (conservative: a real
          content claim is never silently dropped) EXCEPT the obvious
          session_offload case detectable from the FILENAME (the episodic
          transcripts are named session_*_offload_*), which stays excluded even
          when its frontmatter can't be parsed.
    Why:  the single predicate every detector enumeration/match site consults so
          the experiential-vs-content rule is applied identically online, in the
          passive sweep, and in the active-set. Conservative on ambiguity:
          excluding a genuine claim would silently disable the detector for it, so
          unknown-type defaults to INCLUDED.
    """
    excl = excluded_types()
    t = _node_type(memory_dir, node_id)
    if t is not None:
        return t in excl
    # Unreadable/typeless: include conservatively, EXCEPT obvious episodic by name.
    if "session_offload" in excl:
        stem = node_id[:-3] if node_id.endswith(".md") else node_id
        low = stem.lower()
        if low.startswith("session_") and "offload" in low:
            return True
    return False


def is_enabled() -> bool:
    """Live read of the ASTHENOS_CONTRADICTION_ENABLED master switch.

    What: returns True only when the operator has enabled contradiction/
          supersession detection (default OFF). Reads the env each call so a
          test/daemon that sets the flag after import sees the change.
    Why:  R8 produce-only gating — the ONLINE auto-supersede write-path behavior
          must stay inert until the operator enables it + restarts the daemon.
    """
    return os.environ.get("ASTHENOS_CONTRADICTION_ENABLED", "0") == "1"


def auto_cosine_threshold() -> float:
    """The online exact-supersession auto bar (default 0.92, env-tunable)."""
    return _ONLINE_AUTO_COSINE


def configure(memory_dir: Path) -> None:
    """Set the memory directory for contradiction detection.

    What: stores the memory_dir path for vector index lookups. Writes _MEMORY_DIR
          THROUGH the package facade so the single-owned state the finders read
          (also through the facade) is the one updated — a sibling submodule's local
          name would otherwise diverge from a test's contra._MEMORY_DIR rebind.
    Why:  the contradiction module needs to know where nodes live, but
          shouldn't hardcode paths. The daemon calls this during startup.
    """
    # Reach the package facade so the assignment lands on the attribute the finders
    # read (_pkg._MEMORY_DIR), keeping configure() and find_*_candidates in agreement
    # with any test that rebinds contra._MEMORY_DIR directly.
    from samia.runtime import contradiction as _pkg
    _pkg._MEMORY_DIR = memory_dir
    _log.info(
        "contradiction: configured memory_dir=%s enabled=%s judge=%s",
        memory_dir, _ENABLED, _JUDGE_ENABLED,
    )


# ---------------------------------------------------------------------------
# Phase 1/2/synth shared constants (the store filename, the jaccard floor, the
# passive cursor key + budget, the inference budgets + judge model default, and
# the two LLM prompt templates) — all single-owned here so every arm imports one.
# ---------------------------------------------------------------------------

# _SUPERSESSION_JACCARD — What: cheap lexical pre-filter floor before cosine.
# _SUPERSESSION_JACCARD — Why: reuse memory_guard's existing 0.25 jaccard smell
#   to bound the cosine candidate set (Q2 answered: jaccard stays as a cheap
#   pre-filter). Env-tunable; 0.25 mirrors memory_guard._CONTRADICTION_THRESHOLD.
_SUPERSESSION_JACCARD: float = float(
    os.environ.get("ASTHENOS_SUPERSESSION_JACCARD", "0.25")
)

# _SUPERSESSION_STORE — What: the single canonical candidate store filename.
# _SUPERSESSION_STORE — Why: R2 reconciliation — BOTH the old memory_guard
#   SUPERSESSION_LOG (run-1, ~/.local/share/.../memory_guard/) and this module
#   (run-2, <memory_dir>/biomimetic/) previously wrote supersession_candidates.jsonl
#   with DIFFERENT schemas. This module is now the ONE owner with ONE schema; the
#   surfacer and confirm/dismiss/list paths all route here.
_SUPERSESSION_STORE = "supersession_candidates.jsonl"

# _PASSIVE_CURSOR_KEY — What: the rem_cursors.json key the passive sweep
#   checkpoints its cursor under.
# _PASSIVE_CURSOR_KEY — Why: must match the cursor_key the REM registration
#   uses (rem_subscribers.register) so the driver reads the same resume point.
_PASSIVE_CURSOR_KEY = "contradiction_passive"

# _PASSIVE_BUDGET — What: default nodes processed per passive_sweep call.
# _PASSIVE_BUDGET — Why: REM is idle-budgeted; a bounded slice per cycle keeps a
#   single REM tick responsive and lets a full pass complete across many cycles.
_PASSIVE_BUDGET: int = int(os.environ.get("ASTHENOS_SUPERSESSION_PASSIVE_BUDGET", "20"))

# _JUDGE_INFER_MAX_TOKENS / _SYNTH_INFER_MAX_TOKENS — What: generation budgets
#   for the two in-process inference calls (judge verdict / abstraction synthesis).
# _JUDGE_INFER_MAX_TOKENS / _SYNTH_INFER_MAX_TOKENS — Why: mirror the old
#   llama-cli `-n` values (512 / 768) so the rewired in-process call asks for the
#   same amount of structured JSON output.
_JUDGE_INFER_MAX_TOKENS: int = 512
_SYNTH_INFER_MAX_TOKENS: int = 768

# ---------------------------------------------------------------------------
# TUNE-2026-06-08: DEDICATED FAST JUDGE BACKEND (BitNet-2B, not the main Qwen-14B)
# ---------------------------------------------------------------------------
#
# The judge runs once per candidate-bearing node across a whole-index passive
# sweep; on the main Qwen-14B CPU backend that is far too slow. Route the judge
# (and synthesize_node) to a DEDICATED small backend, loaded ONCE and cached,
# instead of the slow 14B. The dedicated backend rides inference's per-model-path
# cache (get_backend_for_model), so the judge model loads a single time and the
# judge never duplicates the 14B.
# Generic fallback: env supplies the real path on a configured box (cls-flags
# sets ASTHENOS_CONTRADICTION_JUDGE_MODEL); the literal here is only the
# unset-env default.
# DEFAULT SWAP (SLOT-STUDY 2026-06-12, operator-directed): the BitNet i2_s
# default never loads under stock llama-cpp-python (its int2/ternary kernel is
# bitnet.cpp-specific), so the unset-env judge silently fell to MockBackend —
# probe measured TPR 0.0. Qwen3-4B as judge measured TPR 0.9 / FPR 0.0 on the
# same probe corpus. The default is the REGISTRY LOGICAL NAME (not a path):
# get_backend_for_model -> fetch_model resolves it on disk or via the gated
# self-fetch, and it is the same model the fact extractor uses, so the cached
# backend is shared. BitNet remains selectable via env for bitnet.cpp setups.
_JUDGE_MODEL_DEFAULT = "Qwen3-4B-Instruct-2507-Q4_K_M"


def _judge_model_path() -> str:
    """The dedicated judge model (env-overridable; default = Qwen3-4B registry name)."""
    return os.environ.get("ASTHENOS_CONTRADICTION_JUDGE_MODEL", _JUDGE_MODEL_DEFAULT)


# _JUDGE_PROMPT_TEMPLATE — What: structured prompt for the LLM judge.
# _JUDGE_PROMPT_TEMPLATE — Why: instructs the model to output JSON with
#   contradiction confidence per candidate. Explicitly excludes temporal
#   changes from contradiction classification.
_JUDGE_PROMPT_TEMPLATE = """You are a memory consistency judge. Given a NEW claim and a list of EXISTING claims from a memory store, determine if the new claim contradicts any existing claim.

A contradiction means both claims cannot be true simultaneously. Temporal changes (preferences that evolved over time) are NOT contradictions.

NEW CLAIM:
{new_claim}

EXISTING CLAIMS:
{existing_claims}

Respond ONLY with a JSON object:
{{"contradictions": [{{"existing_claim_id": "...", "explanation": "...", "confidence": 0.0-1.0}}]}}

If there are no contradictions, respond: {{"contradictions": []}}"""

# _SYNTH_PROMPT_TEMPLATE — What: the synthesis prompt for the abstraction call.
# _SYNTH_PROMPT_TEMPLATE — Why: the merge consumer's P2 distinct-but-overlapping
#   pair needs a single higher-level semantic node that SUBSUMES both sources
#   (episodic->semantic). Asks for strict JSON {title, body} so the staged draft
#   has a clean shape; mirrors the judge's "respond ONLY with JSON" contract.
_SYNTH_PROMPT_TEMPLATE = """You are a memory abstraction synthesizer. Given TWO related memory notes, write ONE higher-level note that captures the shared concept both express, without losing the distinct detail each carries.

NOTE A:
{note_a}

NOTE B:
{note_b}

Respond ONLY with a JSON object:
{{"title": "<short title for the unified note>", "body": "<the synthesized higher-level note body>"}}"""


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.contradiction.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.runtime.contradiction monolith
#             during modularization (the shared config/constants leaf).
# Layer:      runtime (library helper, no daemon loop)
# Role:       shared base of the contradiction package — the re-exported stdlib
#             (json/logging/os/datetime/timezone/Path/Any/Optional), the package
#             logger _log + _now_iso stamp, EVERY tuning constant + prompt template,
#             the single-owned mutable state (_MEMORY_DIR / _TYPE_CACHE), and the
#             live env/scoping readers (is_enabled / auto_cosine_threshold /
#             configure / excluded_types / _node_type / _clear_type_cache /
#             is_excluded_node).
# Stability:  stable — pure constants + small readers; the implementation arms
#             (detection / store / judge / passes) all import their shared surface
#             from here so it is never duplicated.
# ErrorModel: _node_type is fail-soft (unreadable -> None, cached); is_excluded_node
#             is conservative (unknown-type -> INCLUDED except obvious episodic-by-
#             name). configure writes _MEMORY_DIR through the package facade so the
#             finders + any test rebind agree on one copy.
# Depends:    samia.core.frontmatter (lazy, function-local in _node_type).
#             json/logging/os/datetime/pathlib/typing (stdlib).
# Exposes:    is_enabled, auto_cosine_threshold, configure, excluded_types,
#             is_excluded_node (public); _log, _now_iso, _node_type,
#             _clear_type_cache, _judge_model_path, and every _* constant +
#             prompt template (single-owned, imported through config by siblings).
# Lines:      410
# --------------------------------------------------------------------------
