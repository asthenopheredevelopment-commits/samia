"""samia.core.paths -- single source of truth for the SAM/IA memory root.

Layer 1 (Owns / Depends):
    Owns:    resolve_memory_root, ASTHENOS_MEMORY_DIR_ENV, _XDG_FALLBACK
    Depends: stdlib (os, logging, pathlib)

Layer 2 (What / Why):
    What: resolve_memory_root() returns the directory that holds the SAM/IA
          memory plane (the `nodes/`, `biomimetic/`, `chains/` subtrees). It
          tries three sources in order: (1) the ASTHENOS_MEMORY_DIR env var;
          (2) the legacy file-position derivation, BUT ONLY when that derived
          candidate actually looks like a memory root (a `nodes/` subdir
          exists under it); (3) an XDG fallback at ~/.local/share/samia/memory,
          created on first use with a one-time INFO log naming the env var.
    Why:  Modules that ship (bug_records, rem_cycle, hebbian_health) used to
          derive the memory root purely from their own file position
          (Path(__file__).resolve().parents[3]). That is correct ONLY in the
          dev layout (.../memory/tools/samia/<pkg>/<file>.py). In the staged
          release (.../staging/samia/...) parents[3] is the drive root, and in
          site-packages it is site-packages' parent -- so staged test runs
          literally wrote nodes/ and REM state onto the staging drive. This
          helper keeps every dev/daemon path byte-identical (clause 2 reuses
          the exact legacy candidate when nodes/ is present) while giving the
          release a safe, self-creating fallback instead of scribbling on the
          drive root.

Layer 3 (Changelog):
    2026-06-11  BUG-paths  Initial. Extracted the memory-root derivation out of
                           bug_records / rem_cycle / hebbian_health into one
                           layout-safe resolver (env -> verified-legacy -> XDG).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger("samia.core.paths")

# ASTHENOS_MEMORY_DIR_ENV -- What: the override env var name.
# ASTHENOS_MEMORY_DIR_ENV -- Why: a single named knob lets any deployment
#     (release, CI, container) point the memory plane wherever it wants without
#     touching code or relying on file position. Highest precedence.
ASTHENOS_MEMORY_DIR_ENV = "ASTHENOS_MEMORY_DIR"

# _LEGACY_PARENTS_DEPTH -- What: how many parents up from THIS file the dev
#     memory root sits.
# _LEGACY_PARENTS_DEPTH -- Why: paths.py lives at .../memory/tools/samia/core/
#     paths.py, so parents[0]=core, [1]=samia, [2]=tools, [3]=memory. This is
#     the SAME nesting depth the legacy bug_records (runtime/), rem_cycle
#     (runtime/) and hebbian_health (core/) sites used (all parents[3]), so the
#     candidate this produces is byte-identical to what they derived.
_LEGACY_PARENTS_DEPTH = 3

# _XDG_FALLBACK -- What: the release/last-resort memory root.
# _XDG_FALLBACK -- Why: XDG_DATA_HOME convention (default ~/.local/share),
#     matching the rest of the tree. Used only when neither the env var nor a
#     real legacy root is available -- e.g. a clean site-packages install.
_log_emitted_fallback = False  # one-time-log latch (module-scoped)


def _xdg_data_home() -> Path:
    """Return $XDG_DATA_HOME or its ~/.local/share default."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg)
    return Path.home() / ".local" / "share"


def resolve_memory_root(create: bool = True) -> Path:
    """Return the SAM/IA memory root directory, creating it if we fall back.

    Resolution order (first match wins):
        1. $ASTHENOS_MEMORY_DIR, if set (created with a nodes/ subdir if absent).
        2. The legacy file-position candidate (Path(__file__).parents[3]) IFF
           it already contains a nodes/ subdir -- preserves every existing
           dev/daemon path exactly, with no behavior change.
        3. XDG fallback ~/.local/share/samia/memory, created (with nodes/) on
           first use, accompanied by a one-time INFO log naming the env var.

    Parameters
    ----------
    create : bool
        When True (default), clauses 1 and 3 create the resolved root (and its
        nodes/ subdir) as a side effect. When False, NO directory is created --
        for callers that only need the path at *import* time (module-level
        NODES_DIR/MEMORY_DIR/_MEM_ROOT bindings) and must not write to $HOME or a
        read-only/sandbox HOME merely on import. Actual writers (write_node,
        save_chain, the runtime bootstrappers) create the dirs on first write.

    Side effect: with create=True, clauses 1 and 3 create directories (the
    env-named or fallback root, plus its nodes/ subdir). Clause 2 never creates
    anything, and create=False never creates anything.
    """
    global _log_emitted_fallback

    env_val = os.environ.get(ASTHENOS_MEMORY_DIR_ENV)
    if env_val:
        root = Path(env_val).expanduser()
        if create:
            (root / "nodes").mkdir(parents=True, exist_ok=True)
        return root

    candidate = Path(__file__).resolve().parents[_LEGACY_PARENTS_DEPTH]
    if (candidate / "nodes").is_dir():
        return candidate

    fallback = _xdg_data_home() / "samia" / "memory"
    if create:
        (fallback / "nodes").mkdir(parents=True, exist_ok=True)
    if not _log_emitted_fallback:
        _log.info(
            "memory root falling back to %s; set %s to override",
            fallback,
            ASTHENOS_MEMORY_DIR_ENV,
        )
        _log_emitted_fallback = True
    return fallback


# ─────────────────────────────────────────────
# [paths] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.core
# Version:    1.0.0  Updated: 2026-06-11  Status: active
# Role:       single layout-safe resolver for the SAM/IA memory root
#             (env -> verified-legacy file-position -> XDG fallback)
# Depends:    os, logging, pathlib (stdlib only)
# Note:       PRODUCE-ONLY at import; resolve_memory_root() may create the
#             env-named or XDG fallback root, but the verified-legacy clause
#             (the dev/daemon path) creates nothing and is byte-identical to
#             the prior parents[3] derivation.
# ─────────────────────────────────────────────
