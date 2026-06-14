"""samia.runtime.migrations.test_aud61_target_state — tests for the AUD61 target_state migration.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for assign_target_state, validate_target_state,
             migrate_nodes (dry-run and apply).
    Depends: samia.runtime.migrations.aud61_target_state, samia.core.frontmatter,
             unittest, tempfile, pathlib, json (stdlib).

Layer 2 (What / Why):
    What: Validates the AUD61 migration heuristics (tier->state mapping),
          write-time validation rules (frozen/archived restrictions), and
          the dry-run/apply migration paths.
    Why:  Incorrect state assignment could silently freeze live nodes or
          allow writes to archived records. These tests catch heuristic
          regressions and lifecycle enforcement bugs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from samia.core import frontmatter as _fm
from samia.runtime.migrations.aud61_target_state import (
    VALID_TARGET_STATES,
    DEFAULT_TARGET_STATE,
    assign_target_state,
    validate_target_state,
    migrate_nodes,
)


class TestAssignTargetState(unittest.TestCase):
    """Tests for the assignment heuristic."""

    def test_hot_tier_is_live(self):
        """What: hot-tier nodes are assigned live.
        Why: hot nodes are actively used; live is the correct lifecycle."""
        self.assertEqual(assign_target_state({"tier": "hot"}, "nodes"), "live")

    def test_warm_tier_is_live(self):
        """What: warm-tier nodes are assigned live.
        Why: warm nodes are recent; live is the correct default."""
        self.assertEqual(assign_target_state({"tier": "warm"}, "nodes"), "live")

    def test_cold_tier_is_live(self):
        """What: cold-tier nodes default to live (conservative).
        Why: cold nodes may be re-accessed; freezing them prematurely
        would prevent tier_flow re-promotion."""
        self.assertEqual(assign_target_state({"tier": "cold"}, "nodes"), "live")

    def test_frozen_tier_is_frozen(self):
        """What: frozen-tier nodes are assigned frozen.
        Why: these are already inert in tier_flow; target_state confirms it."""
        self.assertEqual(assign_target_state({"tier": "frozen"}, "nodes"), "frozen")

    def test_archive_source_is_archived(self):
        """What: archive-sourced nodes are always archived.
        Why: files in archive/ are historical; they should never re-enter
        the active query set."""
        self.assertEqual(assign_target_state({"tier": "hot"}, "archive"), "archived")

    def test_missing_tier_is_live(self):
        """What: nodes without a tier field default to live.
        Why: conservative -- don't freeze nodes with incomplete metadata."""
        self.assertEqual(assign_target_state({}, "nodes"), "live")


class TestValidateTargetState(unittest.TestCase):
    """Tests for write-time target_state validation."""

    def test_none_is_ok(self):
        """What: missing target_state is accepted (will default to live).
        Why: backward compatibility during migration window."""
        ok, _ = validate_target_state(None)
        self.assertTrue(ok)

    def test_valid_states_accepted(self):
        """What: all three valid states pass validation.
        Why: basic correctness of the whitelist."""
        for state in VALID_TARGET_STATES:
            ok, _ = validate_target_state(state)
            self.assertTrue(ok, f"'{state}' should be valid")

    def test_invalid_state_rejected(self):
        """What: invalid state values are rejected.
        Why: typos or garbage must not pollute the lifecycle field."""
        ok, reason = validate_target_state("deleted")
        self.assertFalse(ok)
        self.assertIn("invalid", reason)

    def test_non_string_rejected(self):
        """What: non-string values are rejected.
        Why: target_state must be a string for frontmatter serialization."""
        ok, reason = validate_target_state(42)
        self.assertFalse(ok)

    def test_write_to_archived_rejected(self):
        """What: writes to archived nodes are rejected.
        Why: archived nodes are immutable historical records."""
        ok, reason = validate_target_state("live", current_state="archived")
        self.assertFalse(ok)
        self.assertIn("archived", reason)

    def test_write_to_frozen_rejected_without_override(self):
        """What: writes to frozen nodes are rejected without override.
        Why: frozen nodes are operator-protected."""
        ok, reason = validate_target_state("live", current_state="frozen")
        self.assertFalse(ok)
        self.assertIn("frozen", reason)

    def test_write_to_frozen_allowed_with_override(self):
        """What: writes to frozen nodes are allowed with override=True.
        Why: operator action can override the protection."""
        ok, reason = validate_target_state("live", current_state="frozen", override=True)
        self.assertTrue(ok)


class TestMigrateNodes(unittest.TestCase):
    """Integration tests for migrate_nodes (dry-run and apply)."""

    def _create_node(self, nodes_dir: Path, name: str, fm: dict) -> Path:
        """Helper: create a .md node with frontmatter."""
        order = list(fm.keys())
        path = nodes_dir / f"{name}.md"
        path.write_text(
            _fm.serialize(fm, order, "Test body content.\n"),
            encoding="utf-8",
        )
        return path

    def test_dry_run_does_not_write(self):
        """What: dry-run reports assignments without modifying files.
        Why: operator must review before committing changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = Path(tmpdir)
            nodes = mem / "nodes"
            nodes.mkdir()
            self._create_node(nodes, "test_node", {
                "name": "Test", "tier": "warm", "relevance": 0.6,
            })

            results = migrate_nodes(mem, apply=False)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["new_state"], "live")
            self.assertFalse(results[0]["applied"])

            # Verify file was NOT modified.
            fm, _, _ = _fm.read_node(nodes / "test_node.md")
            self.assertNotIn("target_state", fm)

    def test_apply_writes_target_state(self):
        """What: --apply mode writes target_state into node frontmatter.
        Why: this is the actual migration action."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = Path(tmpdir)
            nodes = mem / "nodes"
            nodes.mkdir()
            self._create_node(nodes, "warm_node", {
                "name": "Warm", "tier": "warm", "relevance": 0.6,
            })
            self._create_node(nodes, "frozen_node", {
                "name": "Frozen", "tier": "frozen", "relevance": 0.1,
            })

            results = migrate_nodes(mem, apply=True)
            self.assertEqual(len(results), 2)

            # Verify frontmatter was updated.
            fm_warm, _, _ = _fm.read_node(nodes / "warm_node.md")
            self.assertEqual(fm_warm["target_state"], "live")

            fm_frozen, _, _ = _fm.read_node(nodes / "frozen_node.md")
            self.assertEqual(fm_frozen["target_state"], "frozen")

    def test_archive_files_tagged_archived(self):
        """What: .frozen.json files in archive/ get target_state=archived.
        Why: archived nodes are historical; the migration encodes this."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = Path(tmpdir)
            archive = mem / "archive"
            archive.mkdir()
            data = {
                "frontmatter": {"name": "Old node", "tier": "frozen"},
                "body": "Historical content.",
            }
            (archive / "old_node.frozen.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )

            results = migrate_nodes(mem, apply=True)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["new_state"], "archived")

            # Verify JSON was updated.
            updated = json.loads(
                (archive / "old_node.frozen.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["frontmatter"]["target_state"], "archived")


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.runtime.migrations.test_aud61_target_state
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD61
# Layer:      test (pytest)
# Role:       tests for samia.runtime.migrations.aud61_target_state — tier->state
#             assignment heuristic, write-time validation, dry-run/apply migration
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.runtime.migrations.aud61_target_state, samia.core.frontmatter
# Exposes:    — (test module)
# Lines:      223
# --------------------------------------------------------------------------
