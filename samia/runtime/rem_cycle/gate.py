"""samia.runtime.rem_cycle.gate — the run-only-in-REM gate + its decorator form.

Layer 1 (Owns / Depends):
    Owns:    the single guard every migrated offline op consults at entry
             (gate_offline_op) and its decorator form (only_in_rem) that lets an op
             gate itself regardless of which caller invokes it.
    Depends: .config (_log_event, the re-exported deps via __init__), .state
             (is_rem / current_state — the phase read the gate keys on).

Layer 2 (What / Why):
    What: gate_offline_op returns True only when is_rem(mem); otherwise it LOGS a
          "refused — not in REM" no-op to rem_events.jsonl and returns False (a
          LOGGED no-op, never a silent drop). only_in_rem wraps an op so it returns
          a logged refusal dict outside REM and runs normally inside REM.
    Why:  Q5 — offline reconciliation refuses to run outside the sleep window so it
          never competes with active cognition. The gate travels with the op
          (root-cause gating), not with one registration site.

PATCH SEAM (exemplar rule): gate_offline_op is a `mock.patch.object(rem_cycle,
    "gate_offline_op", ...)` target (test_fact_extract_producer). The decorator
    only_in_rem is a SIBLING caller of it, so it reaches gate_offline_op THROUGH the
    package facade (`from samia.runtime import rem_cycle as _pkg; _pkg.gate_offline_op
    (...)`) — that way a patch on the package object takes effect for the wrapped op
    too, instead of binding the pre-patch module-local function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import functools

from .config import _log_event
from .state import current_state, is_rem


def gate_offline_op(mem: Path, op_name: str) -> bool:
    """The gate: True iff ``op_name`` may run now (i.e. the system is in REM).

    What: the single guard every migrated offline op consults at entry. Returns
          True only when is_rem(mem); otherwise it LOGS a "refused — not in REM"
          no-op to rem_events.jsonl and returns False (a LOGGED no-op, never a
          silent drop — risk-5).
    Why:  Q5 — offline reconciliation refuses to run outside the sleep window so
          it never competes with active cognition. The refusal is operator-
          visible so a "REM never enters" regression surfaces in the event log.
    """
    if is_rem(mem):
        return True
    _log_event(mem, "offline_refused", {"op": op_name, "phase": current_state(mem)["phase"]})
    return False


def only_in_rem(op_name: str | None = None):
    """Decorator form of the gate for offline-op functions whose first arg is mem.

    What: wraps ``fn(mem, ...)`` so it returns a logged refusal dict
          ({"fired": False, "refused": "not_in_rem", "op": name}) when called
          outside REM, and runs normally inside REM. ``op_name`` defaults to the
          wrapped function's __name__.
    Why:  lets a migrated op gate itself at its own entry — so it refuses
          regardless of WHICH caller (idle_pulse, scheduler.py, a manual call,
          a future caller) invokes it. The gate travels with the op (root-cause
          gating), not with one registration site.
    """
    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = op_name or getattr(fn, "__name__", "offline_op")

        @functools.wraps(fn)
        def _wrapped(mem: Path, *args: Any, **kwargs: Any) -> Any:
            # PATCH SEAM: reach gate_offline_op through the package facade so a
            # `mock.patch.object(rem_cycle, "gate_offline_op", ...)` patch applies to
            # the wrapped op too (never the pre-patch module-local function).
            from samia.runtime import rem_cycle as _pkg
            if not _pkg.gate_offline_op(Path(mem), name):
                return {"fired": False, "refused": "not_in_rem", "op": name}
            return fn(mem, *args, **kwargs)

        return _wrapped

    return _decorate


# ─────────────────────────────────────────────
# [Asthenosphere] samia.runtime.rem_cycle.gate
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      Phase B (2026-06-14): carved from the samia.rem_cycle monolith during
#             modularization
# Layer:      runtime (library helper, no daemon loop)
# Role:       the run-only-in-REM gate (gate_offline_op) + its decorator form
#             (only_in_rem). The single guard every migrated offline op consults at
#             entry; refuses outside REM with a LOGGED no-op (never a silent drop).
# Stability:  stable — behavior byte-identical to the monolith's gate section.
# ErrorModel: gate_offline_op never raises; a refusal is a logged False (the event
#             append is fail-soft in config). only_in_rem returns a refusal dict.
# Depends:    .config (_log_event), .state (is_rem, current_state); functools (stdlib);
#             the package facade (samia.runtime.rem_cycle) for the patch-seam reach.
# Exposes:    gate_offline_op, only_in_rem.
# Note:       PATCH SEAM — only_in_rem reaches gate_offline_op through the package
#             facade so a mock.patch.object on the package takes effect for the
#             decorated op (see module docstring).
# Lines:      103
# ─────────────────────────────────────────────
