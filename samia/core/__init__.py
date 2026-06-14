"""samia.core — pure library plane (package marker).

Layer 1 (Owns / Depends):
    Owns:    — (package marker; the submodules — ia, vector, temporal,
             fact_extractor, netconsent, frontmatter, etc. — own the public API).
    Depends: stdlib only at import; each submodule carries its own deps.
Layer 2 (What / Why):
    What: the pure-library plane of the SAM/IA engine. Modules here are extracted
          from the original memory/tools/*.py scripts; every public API is a pure
          function or a dataclass, importable with no daemon running.
    Why:  separating the library plane from the runtime daemon lets CLIs, the MCP
          server, and tests reuse the identical logic without a process. Acceptance
          (design doc §8.1): a script refactored to import from samia.core must
          produce byte-identical output to its pre-refactor version.
"""

# ─────────────────────────────────────────────
# [Asthenosphere] samia.core
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      package marker (the pure-library plane carved out in Phase A1 —
#             modules extracted from the original memory/tools/*.py scripts)
# Layer:      package marker (roots the core pure-library plane; no daemon)
# Role:       namespace for the SAM/IA pure-library plane (daemon-free APIs)
# Stability:  stable (bare marker; changes only with the core plane's layout)
# ErrorModel: none — import-time no-op; no logic to surface errors
# Depends:    — (stdlib only at import; submodules own their dependencies)
# Exposes:    — (namespace only; public API lives in the core.* submodules)
# Lines:      28
# --------------------------------------------------------------------------
