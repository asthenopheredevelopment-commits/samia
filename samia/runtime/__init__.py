"""samia.runtime — long-lived process plane for the SAM/IA memory system.

Layer 1 (Owns / Depends):
    Owns:    daemon lifecycle, AF_UNIX IPC, client library, CLI shim
    Depends: samia.core (library plane, optional — used by fallback paths)

Layer 2 (What / Why):
    What: Package marker that re-exports the three public classes callers need.
    Why:  Single import surface so downstream code writes
          ``from samia.runtime import SamiaClient`` rather than reaching
          into submodules.

Design doc: ~/Desktop/DinnerBell-BBQ-Dev/plans/sam_ia_runtime_design.md, section 1.2.
AUD26 Phase 26.1 — foundation (daemon + IPC + client + CLI).
"""

# GATE6 MEMORY-CORE carve: SamiaDaemon re-export removed (daemon.py does not ship)
from samia.runtime.client import SamiaClient   # noqa: F401

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────────────────────
# [Asthenosphere] samia.runtime
# phase: AUD26-26.1
# layer: runtime (long-lived process)
# ──────────────────────────────────────────────────────────────────────────────
