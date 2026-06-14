"""samia — SAM/IA memory runtime package marker.

Layer 1 (Owns / Depends):
    Owns:    — (package marker; the namespace, no public API of its own).
    Depends: stdlib only (none at import; subpackages carry their own deps).
Layer 2 (What / Why):
    What: the top-level package for the SAM/IA memory runtime. Layout —
          samia.core    (pure library plane, importable, no daemon required) and
          samia.runtime (the long-lived process, Phase A1.1+).
    Why:  Phase A1 of the merged roadmap 2026-04-29 split the engine into an
          importable library plane and a daemon plane; this marker roots both so a
          consumer can `import samia.core` without dragging in the runtime.

Phase A1 of merged roadmap 2026-04-29. See plans/sam_ia_runtime_design.md.
"""

# ─────────────────────────────────────────────
# [Asthenosphere] samia
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      package marker (Phase A1 of merged roadmap 2026-04-29 — top-level
#             namespace rooting the core/ library and runtime/ daemon planes)
# Layer:      package marker (no plane of its own; subpackages carry the planes)
# Role:       top-level SAM/IA memory-runtime package namespace
# Stability:  stable (bare marker; changes only when the package layout changes)
# ErrorModel: none — import-time no-op; no logic to surface errors
# Depends:    — (stdlib only; subpackages own their dependencies)
# Exposes:    — (namespace only; public API lives in samia.core / samia.runtime)
# Lines:      28
# --------------------------------------------------------------------------
