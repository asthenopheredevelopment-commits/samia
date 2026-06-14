"""samia.core.merge_consumer.config — shared constants, enable flag, candidate-id.

Layer 1 (Owns / Depends):
    Owns:    the module-level constants the whole package reads (the dup
             cosine/jaccard bars, the candidate/abstract/enqueued filenames, the
             provenance ref_kind, the guarded status), the live
             ASTHENOS_TIER2_MERGE_ENABLED read (is_enabled), and the
             order-independent candidate-id primitive (_candidate_id) shared by
             the abstraction store and the fact-extract enqueue.  Re-exports the
             three sibling-shared dependency modules (_con/_fm/_ia) so the carve
             imports them THROUGH one owner instead of each submodule re-importing.
    Depends: samia.core.consolidation (the surfacer schema + jaccard/shingles),
             samia.core.frontmatter (parse/serialize), samia.core.ia (the
             RESTORABLE supersede + event log + now-iso).  hashlib/json/os/Path
             from stdlib.

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — every sibling submodule
          imports its bars/filenames/flag from here, so the tunable surface lives
          in one place and is never duplicated.  _candidate_id is here (not in the
          abstraction submodule) because both the abstraction store AND the
          fact-extract enqueue derive the same per-pair key from it.
    Why:  splitting the 1175-line monolith by responsibility (candidate I/O,
          winner-merge, abstraction lifecycle, drain) leaves a shared base of
          constants + primitives that all four need; concentrating them here keeps
          the bars single-sourced and the import graph acyclic (config depends on
          nothing in the package).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from .. import consolidation as _con
from .. import frontmatter as _fm
from .. import ia as _ia

# _DUP_MERGE_COSINE — What: the HIGH cosine bar above which a pair is a TRUE
#   duplicate (Q2c AUTO pick-winner). Why: only the near-exact bulk merges
#   automatically; distinct-but-overlapping pairs (below the bar) are left for
#   P2's gated LLM-abstraction. Env-tunable; 0.92 per the proposal default.
_DUP_MERGE_COSINE: float = float(
    os.environ.get("ASTHENOS_DUP_MERGE_COSINE", "0.92")
)

# _DUP_MERGE_JACCARD — What: the dup bar on the surfacer's lexical jaccard
#   score, used when no vector index exists so cosine is unavailable. Why: the
#   surfacer scores pairs by jaccard (consolidation.py); when the embedding
#   index is absent the consumer must still classify deterministically. 0.85 is
#   a HIGH lexical-overlap bar (true near-duplicate prose), distinct from the
#   surfacer's 0.15 surfacing knee. Env-tunable.
_DUP_MERGE_JACCARD: float = float(
    os.environ.get("ASTHENOS_DUP_MERGE_JACCARD", "0.85")
)

_CANDIDATE_FILE = ".consolidation_candidates.json"
_ABSTRACT_LOG = "merge_candidates.jsonl"  # under biomimetic/ — P2's pending queue
_PROVENANCE_KIND = "provenance"

# _ENQUEUED_LOG — What: a tiny append-only done-set of candidate_ids already fed
#   to the fact-extract queue (one {"candidate_id","ts"} JSONL line per pair).
# Why: BUG-2026-06-11 runaway loop (enqueue side) — the surfacer re-presents the
#   SAME ~52 abstract pairs every REM cycle, so _enqueue_abstract_pair re-enqueued
#   them every drain (+~1,500 lines/hour to .fact_extract_queue.jsonl). The
#   candidate_id presence in merge_candidates.jsonl already dedups the recorded
#   pair; this is the belt-and-suspenders ledger so a pair is enqueued AT MOST
#   ONCE EVER even if its P2 record is later resolved/rewritten out of the
#   unresolved view. Under biomimetic/.
_ENQUEUED_LOG = "fact_extract_enqueued.jsonl"

# _GUARDED_STATUS — What: the abstraction-candidate status set when the P3 salience
#   guard fires on a DISTINCT high-salience source. Why: D6 effect (iii) / Q5a — a
#   distinct high-salience memory must NOT be auto-abstracted-away; it is SURFACED for
#   operator review (a terminal, listable status the operator resolves via confirm/
#   reject), never silently superseded. Distinct from "pending"/"proposed" so it is
#   visible as "needs review, salience-protected" and is not re-synthesized.
_GUARDED_STATUS = "guarded"


def is_enabled() -> bool:
    """Live read of the ASTHENOS_TIER2_MERGE_ENABLED master switch.

    What: True iff the operator has set ASTHENOS_TIER2_MERGE_ENABLED=1.
    Why:  Q5a — P1 is double-gated (REM + this flag), inert by default. A live
          read (not an import-time constant) lets a test or the daemon flip it
          without re-import, mirroring contradiction.is_enabled().
    """
    return os.environ.get("ASTHENOS_TIER2_MERGE_ENABLED", "0") == "1"


def _candidate_id(a_id: str, b_id: str) -> str:
    """Stable id for an abstract candidate pair (order-independent).

    What: "abs-<sha1(sorted(a,b))[:12]>" — deterministic so the same pair maps
          to the same candidate across drains (no duplicate proposals) and so
          confirm/reject can address it without an opaque counter.
    Why:  P2 confirm/reject (and the MCP surface) need a single addressable key
          per pair; deriving it from the sorted ids keeps it stable + dedup-able.
          Lives in config (not abstraction) because the fact-extract enqueue path
          keys its done-set off the SAME id.
    """
    key = "|".join(sorted((str(a_id), str(b_id))))
    return "abs-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.merge_consumer.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.merge_consumer monolith during modularization
# Layer:      core (pure library, no daemon dependency)
# Role:       shared base of the merge_consumer package — dup bars, store
#             filenames, provenance/guarded constants, the is_enabled flag, the
#             order-independent _candidate_id, and the re-exported _con/_fm/_ia
#             dependency modules every sibling imports through.
# Stability:  stable — pure constants + two side-effect-free helpers; the carve
#             changed no value (bars/filenames/flag byte-identical to the monolith).
# ErrorModel: none — is_enabled is a plain env read; _candidate_id never raises.
# Depends:    hashlib, os, pathlib, typing (stdlib). samia.core.consolidation,
#             samia.core.frontmatter, samia.core.ia (re-exported as _con/_fm/_ia).
# Exposes:    is_enabled, _candidate_id, the dup bars, the store filenames,
#             _PROVENANCE_KIND, _GUARDED_STATUS, _con/_fm/_ia, Any/Optional/Path.
# Lines:      126
# --------------------------------------------------------------------------
