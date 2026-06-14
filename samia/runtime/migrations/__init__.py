"""samia.runtime.migrations -- one-time data migration scripts.

Layer 1 (Owns / Depends):
    Owns:    — (package marker; no public API of its own).
    Depends: — (stdlib only; the marker imports nothing).
Layer 2 (What / Why):
    What: package marker for the one-time data migration scripts.
    Why:  isolates migrations from runtime code; each migration is a standalone
          module with a dry-run-by-default CLI.
"""


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.migrations
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      original SAM/IA runtime module — package marker
# Layer:      runtime (library helper, no daemon loop)
# Role:       package marker grouping the one-time, dry-run-by-default data
#             migration scripts apart from the live runtime modules.
# Stability:  stable — empty package marker; carries no logic.
# ErrorModel: none — import-time marker only.
# Depends:    — (stdlib only).
# Exposes:    — (no public API; submodules are imported by name).
# Lines:      24
# --------------------------------------------------------------------------
