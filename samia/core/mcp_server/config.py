"""samia.core.mcp_server.config — shared base of the MCP tool-server package.

Layer 1 (Owns / Depends):
    Owns:    the module-top imports the monolith pulled in and that callers reach
             THROUGH this module (the stdlib json/os/sqlite3 + datetime aliased _dt
             + Path + Any + the `from __future__` annotations), the aliased
             web_store dependency module (_ws) that the co-activation read uses, the
             four COACT_* read-back constants (the public co-activation tunables),
             and the two memory-dir path primitives every sibling derives its work
             from (_nodes_dir / _chains_dir).  Re-exports the aliased _ws so the
             carve imports the SAME web_store object through one owner.
    Depends: samia.core.web_store (re-exported as _ws — the edges.db path + the
             COACTIVATION ref_kind the neighbor read keys on).  json/os/sqlite3/
             datetime/pathlib/typing from stdlib.

Layer 2 (What / Why):
    What: the leaf of the package's dependency DAG — every sibling submodule
          imports its path primitives, the COACT_* constants, and the re-exported
          stdlib/_ws names from here, so the shared surface lives in one place and
          is never duplicated.  _nodes_dir / _chains_dir are here (not in the search
          submodule) because the search, write, chains, and context submodules all
          derive nodes/ and chains/ from the memory_dir through them.
    Why:  splitting the 1315-line monolith by responsibility (search / write /
          chains / bio_tools / context_tools) leaves a shared base of imports +
          path primitives + the co-activation constants that all the arms need;
          concentrating them here keeps the import graph acyclic (config depends on
          nothing else in the package) and the COACT_* tunables single-sourced.
"""

from __future__ import annotations

# Re-exported module-top names the monolith pulled in and other code imports THROUGH
# this package facade (json/os/sqlite3, datetime aliased _dt, Path, Any). `annotations`
# rides the `from __future__` above. The baseline records json/os/sqlite3/Any/Path/
# annotations as part of the public surface, so they must stay importable here.
import datetime as _dt  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import sqlite3  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

# The aliased web_store dependency module — single-owned here so the co-activation
# neighbor read in the search submodule reaches the SAME object (edges.db path +
# the COACTIVATION ref_kind) through one import path, never a duplicate alias.
from .. import web_store as _ws  # noqa: F401


# Co-activation read-back (FEAT-2026-06-05 Tier-0 D4) — conservative neighbor boost.
# What: the four tunables that bound the edges.db co-activation neighbor expansion on
#   the recall path — surface learned associations so they bias recall without ever
#   letting a neighbor outrank the hit that surfaced it or displacing anything real.
# Why: the Hebbian web was written but never read on the normal recall path (only a
#   default-off, failure-scoped seam). These constants open it generally but
#   conservatively; they are public (in the baseline surface) so a caller/test can
#   read or override the lambda/caps. Single-sourced here, imported by the search arm.
COACT_LAMBDA = 0.5        # neighbor pull = lambda * edge_weight (then clamped below parent)
COACT_MAX_NEIGHBORS = 3   # cap neighbors appended per search
COACT_PARENT_HITS = 5     # only expand neighbors of the top-N cosine/term hits
COACT_DELTA = 0.05        # parent-score haircut so a neighbor can't tie its parent


def _nodes_dir(memory_dir: Path) -> Path:
    # _nodes_dir — What: the nodes/ subdir of a memory_dir (where every node .md
    #   lives). Why: the single owner of the nodes/ join so the search/write/chains
    #   arms never each re-spell "memory_dir / 'nodes'"; one place to change the
    #   layout. Pure path join — never touches the filesystem, never raises.
    return memory_dir / "nodes"


def _chains_dir(memory_dir: Path) -> Path:
    # _chains_dir — What: the chains/ subdir of a memory_dir (where each chain .json
    #   lives). Why: the single owner of the chains/ join, mirroring _nodes_dir, so the
    #   chains arm and the get_chain enrichment derive it consistently. Pure path join.
    return memory_dir / "chains"


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.mcp_server.config
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.mcp_server monolith during
#             modularization (the shared module-top imports + path/co-activation base).
# Layer:      core (pure library, no daemon dependency)
# Role:       shared base of the mcp_server package — the re-exported stdlib/_dt/_ws
#             names every sibling reaches through, the four public COACT_* co-
#             activation tunables, and the _nodes_dir / _chains_dir path primitives.
# Stability:  stable — pure imports + constants + two side-effect-free path joins; the
#             carve changed no value (constants/primitives byte-identical to the monolith).
# ErrorModel: none — _nodes_dir / _chains_dir are pure path joins that never raise; no
#             filesystem touch at import or call.
# Depends:    datetime, json, os, sqlite3, pathlib, typing (stdlib). samia.core.web_store
#             (re-exported as _ws).
# Exposes:    _dt, json, os, sqlite3, Path, Any (re-exported imports); _ws; COACT_LAMBDA/
#             COACT_MAX_NEIGHBORS/COACT_PARENT_HITS/COACT_DELTA; _nodes_dir, _chains_dir.
# Lines:      95
# --------------------------------------------------------------------------
