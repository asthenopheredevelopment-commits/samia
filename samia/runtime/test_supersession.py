"""Tests for the unified supersession-candidate store — FEAT-2026-06-07 P3b / R2.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the CANONICAL supersession store now owned by
             samia.runtime.contradiction: record_supersession_candidate (one
             schema, mode online|passive), list_supersession_candidates
             (unresolved-only), mark_supersession_confirmed / _dismissed
             (atomic resolve). Plus the R2 reconciliation check that the old
             surface-only memory_guard surfacer was removed.
    Depends: samia.runtime.contradiction, samia.runtime.memory_guard, unittest,
             tempfile, json, pathlib.

Layer 2 (What / Why):
    What: Verifies the single store records candidates with the unified schema
          ({new_id, old_id, cosine, jaccard, mode, ts, confirmed, dismissed}),
          lists only un-resolved ones, and resolves them atomically; and that
          memory_guard no longer carries the duplicate SUPERSESSION_LOG writer.
    Why:  R2 — the run-1 (memory_guard) and run-2 (contradiction) stores wrote
          the SAME filename with DIFFERENT schemas: a data-corruption landmine.
          The operator OVERRODE the surface-only Q4a design to online
          auto-supersede (made safe by reversibility via restore_node), so the
          surfacer was reconciled into ONE owner. Tests use tempfile memory_dirs
          only — they NEVER touch live ~/.local/share memory.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from samia.runtime import contradiction as con
import samia.runtime.memory_guard as mg


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    return md


class TestUnifiedStoreSchema(unittest.TestCase):
    def test_record_uses_one_schema_with_confirmed_dismissed(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            rec = con.record_supersession_candidate(
                md, "old", "new", cosine=0.81, jaccard=0.4, mode="online")
            # one schema: required fields present.
            for k in ("old_id", "new_id", "cosine", "jaccard", "mode", "ts",
                      "confirmed", "dismissed", "status"):
                self.assertIn(k, rec)
            self.assertEqual(rec["old_id"], "old.md")
            self.assertEqual(rec["new_id"], "new.md")
            self.assertEqual(rec["mode"], "online")
            self.assertFalse(rec["confirmed"])
            self.assertFalse(rec["dismissed"])
            # persisted to the canonical biomimetic path.
            line = (md / "biomimetic" / "supersession_candidates.jsonl"
                    ).read_text().strip()
            self.assertEqual(json.loads(line)["cosine"], 0.81)

    def test_passive_mode_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            rec = con.record_supersession_candidate(
                md, "o", "n", cosine=0.9, mode="passive")
            self.assertEqual(rec["mode"], "passive")
            self.assertIsNone(rec["jaccard"])


def _store_lines(md: Path) -> list[str]:
    p = md / "biomimetic" / "supersession_candidates.jsonl"
    if not p.exists():
        return []
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class TestRecordDedup(unittest.TestCase):
    """BUG-2026-06-11 — record_supersession_candidate skips a duplicate UNRESOLVED
    (old_id, new_id) append; a resolved prior row does NOT suppress re-detection."""

    def test_duplicate_unresolved_append_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "new", cosine=0.81,
                                              mode="passive")
            # the same pair re-detected next sweep — must NOT append again.
            con.record_supersession_candidate(md, "old", "new", cosine=0.82,
                                              mode="passive")
            self.assertEqual(len(_store_lines(md)), 1)
            # id normalization: with/without .md is the same pair.
            con.record_supersession_candidate(md, "old.md", "new.md", cosine=0.83,
                                              mode="online")
            self.assertEqual(len(_store_lines(md)), 1)

    def test_distinct_pairs_still_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "n1", cosine=0.8)
            con.record_supersession_candidate(md, "old", "n2", cosine=0.8)
            self.assertEqual(len(_store_lines(md)), 2)

    def test_resolved_pair_does_not_suppress_redetection(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "new", cosine=0.8)
            # operator dismisses it — that row is now resolved.
            con.mark_supersession_dismissed(md, "old", new_id="new")
            # the pair comes back later (re-detected) — a FRESH unresolved row
            # is appended (the dismissed row does not suppress it).
            con.record_supersession_candidate(md, "old", "new", cosine=0.85)
            lines = _store_lines(md)
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(con.list_supersession_candidates(md)), 1)


class TestListAndResolve(unittest.TestCase):
    def test_list_unresolved_then_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "new", cosine=0.8,
                                              mode="online")
            cands = con.list_supersession_candidates(md)
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0]["old_id"], "old.md")

            touched = con.mark_supersession_confirmed(md, "old", new_id="new")
            self.assertEqual(touched, 1)
            # confirmed → no longer un-resolved.
            self.assertEqual(con.list_supersession_candidates(md), [])
            # but still on disk (auditable) with confirmed=True.
            all_recs = con.list_supersession_candidates(md, unresolved_only=False)
            self.assertEqual(len(all_recs), 1)
            self.assertTrue(all_recs[0]["confirmed"])
            self.assertEqual(all_recs[0]["status"], "confirmed")

    def test_dismiss_resolves_without_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "new", cosine=0.8,
                                              mode="online")
            self.assertEqual(con.mark_supersession_dismissed(md, "old"), 1)
            self.assertEqual(con.list_supersession_candidates(md), [])
            recs = con.list_supersession_candidates(md, unresolved_only=False)
            self.assertTrue(recs[0]["dismissed"])

    def test_mark_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "new", cosine=0.8)
            self.assertEqual(con.mark_supersession_confirmed(md, "old"), 1)
            # already resolved → second confirm touches nothing.
            self.assertEqual(con.mark_supersession_confirmed(md, "old"), 0)

    def test_new_id_filter_targets_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            con.record_supersession_candidate(md, "old", "n1", cosine=0.8)
            con.record_supersession_candidate(md, "old", "n2", cosine=0.8)
            # confirm only the (old, n1) pair.
            self.assertEqual(con.mark_supersession_confirmed(md, "old",
                                                             new_id="n1"), 1)
            left = con.list_supersession_candidates(md)
            self.assertEqual(len(left), 1)
            self.assertEqual(left[0]["new_id"], "n2.md")


class TestR2Reconciliation(unittest.TestCase):
    def test_memory_guard_no_longer_owns_a_supersession_store(self):
        # The duplicate run-1 writer/markers must be gone (single owner).
        for name in ("SUPERSESSION_LOG", "_emit_supersession_candidate",
                     "supersession_candidates", "confirm_supersession_candidate",
                     "dismiss_supersession_candidate", "_supersession_candidates_from"):
            self.assertFalse(hasattr(mg, name),
                             f"memory_guard still exposes removed {name}")


if __name__ == "__main__":
    unittest.main()
