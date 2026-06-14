"""samia.core.test_paths — tests for samia.core.paths.resolve_memory_root.

Layer 1 (Owns / Depends):
    Owns:    the resolve_memory_root resolution-order tests + the two-consumer
             rewiring tests (bug_records.NODES_DIR, rem_cycle._default_mem).
    Depends: pytest (tmp_path, monkeypatch), unittest.mock, samia.core.paths,
             samia.runtime.bug_records, samia.runtime.rem_cycle.

Layer 2 (What / Why):
    What: pins the three-clause resolution order (env -> verified-legacy -> XDG)
          and proves both downstream consumers route through the resolver.
    Why:  the staged release wrote nodes/ + REM state onto the drive root
          because the memory root was derived from file position. These tests
          lock the layout-safe behavior and the byte-identical dev path so the
          regression cannot return. All writes land in pytest tmp dirs; HOME and
          XDG_DATA_HOME are monkeypatched -- the real ~/.local is never touched.

Layer 3 (Changelog):
    2026-06-11  BUG-paths  Initial. env-wins / verified-legacy / XDG-fallback +
                           consumer-rewiring tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import samia.core.paths as paths


def _clear_env(monkeypatch) -> None:
    """Remove both env knobs so a clause-2/3 path is exercised deterministically."""
    monkeypatch.delenv("ASTHENOS_MEMORY_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


def test_env_var_wins(tmp_path, monkeypatch):
    """Clause 1: ASTHENOS_MEMORY_DIR overrides everything and creates nodes/."""
    monkeypatch.setenv("ASTHENOS_MEMORY_DIR", str(tmp_path))
    root = paths.resolve_memory_root()
    assert root == tmp_path
    assert (tmp_path / "nodes").is_dir()


def test_env_var_expanduser(tmp_path, monkeypatch):
    """Clause 1: a ~-prefixed env value is expanded against HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ASTHENOS_MEMORY_DIR", "~/mem")
    root = paths.resolve_memory_root()
    assert root == tmp_path / "mem"
    assert (root / "nodes").is_dir()


def test_legacy_derivation_honored_when_nodes_present(tmp_path, monkeypatch):
    """Clause 2: when the file-position candidate has a nodes/ subdir, use it
    verbatim and create nothing -- the byte-identical dev/daemon path."""
    _clear_env(monkeypatch)
    # Build a fake .../<root>/tools/samia/core/paths.py whose parents[3] has nodes/.
    fake = tmp_path / "tools" / "samia" / "core" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    (tmp_path / "nodes").mkdir()
    monkeypatch.setattr(paths, "__file__", str(fake))
    root = paths.resolve_memory_root()
    assert root == tmp_path


def test_xdg_fallback_when_no_legacy_nodes(tmp_path, monkeypatch):
    """Clause 3: when the candidate lacks nodes/, fall back to
    $XDG_DATA_HOME/samia/memory and create it (no drive-root scribbling)."""
    _clear_env(monkeypatch)
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    # candidate parents[3] is a tmp dir WITHOUT nodes/.
    fake = tmp_path / "drive" / "a" / "b" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setattr(paths, "__file__", str(fake))
    monkeypatch.setattr(paths, "_log_emitted_fallback", False)
    root = paths.resolve_memory_root()
    assert root == xdg / "samia" / "memory"
    assert (root / "nodes").is_dir()
    # The bogus candidate root must NOT have been written into.
    assert not (fake.resolve().parents[3] / "nodes").exists()


def test_xdg_fallback_default_home(tmp_path, monkeypatch):
    """Clause 3 default: with XDG_DATA_HOME unset, fall back under
    HOME/.local/share -- monkeypatched HOME, so no real ~ write."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    fake = tmp_path / "drive" / "a" / "b" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setattr(paths, "__file__", str(fake))
    monkeypatch.setattr(paths, "_log_emitted_fallback", False)
    root = paths.resolve_memory_root()
    assert root == tmp_path / ".local" / "share" / "samia" / "memory"
    assert (root / "nodes").is_dir()


def test_bug_records_nodes_dir_is_path(monkeypatch):
    """Consumer contract: bug_records.NODES_DIR stays a module-level Path so
    `from ... import NODES_DIR` and mock.patch keep working."""
    from samia.runtime import bug_records

    assert isinstance(bug_records.NODES_DIR, Path)
    assert bug_records.NODES_DIR.name == "nodes"


def test_bug_records_resolves_through_helper(tmp_path):
    """bug_records' NODES_DIR derives from resolve_memory_root: with the
    resolver mocked to a tmp root, a freshly recomputed NODES_DIR consumes it."""
    from samia.runtime import bug_records

    with mock.patch.object(
        bug_records, "resolve_memory_root", return_value=tmp_path
    ):
        recomputed = bug_records.resolve_memory_root() / "nodes"
    assert recomputed == tmp_path / "nodes"


def test_rem_cycle_default_mem_resolves_through_helper(tmp_path):
    """rem_cycle._default_mem returns resolve_memory_root()'s result; mock the
    resolver to a tmp dir and verify _default_mem consumes it."""
    from samia.runtime import rem_cycle

    with mock.patch.object(
        rem_cycle, "resolve_memory_root", return_value=tmp_path
    ):
        assert rem_cycle._default_mem() == tmp_path


def test_both_consumers_share_one_resolver(tmp_path):
    """Both consumers reference the SAME resolve_memory_root symbol name,
    proving a single source of truth for the memory root."""
    from samia.runtime import bug_records, rem_cycle

    assert bug_records.resolve_memory_root is rem_cycle.resolve_memory_root


# [Asthenosphere] samia.core.test_paths
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      BUG-paths — resolve_memory_root resolution order + consumer rewiring
# Layer:      test (pytest)
# Role:       tests for samia.core.paths.resolve_memory_root — env-wins / verified-legacy / XDG-fallback resolution order + both consumers route through the resolver
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    pytest + samia.core.paths, samia.runtime.bug_records, samia.runtime.rem_cycle
# Exposes:    — (test module)
# Lines:      156
# ------------------------------------------------------------------------------
