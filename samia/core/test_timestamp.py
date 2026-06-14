"""samia.core.test_timestamp — tests for samia.core.timestamp (AUD63 UTC timestamp utilities).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for now_utc, now_utc_iso, normalize_ts, assert_utc_iso.
    Depends: samia.core.timestamp, unittest, datetime (stdlib).

Layer 2 (What / Why):
    What: Validates the three AUD63 capabilities: (1) now_utc returns UTC-aware
          datetime, (2) normalize_ts converts naive/local-TZ to UTC, (3)
          assert_utc_iso rejects non-UTC strings.
    Why:  Timestamp consistency is foundational -- a bug here silently corrupts
          every downstream correlation. These tests catch regressions at the
          source.
"""

from __future__ import annotations

import datetime
import unittest

from samia.core.timestamp import (
    now_utc,
    now_utc_iso,
    normalize_ts,
    assert_utc_iso,
)


class TestNowUtc(unittest.TestCase):
    """Tests for now_utc() and now_utc_iso()."""

    def test_now_utc_is_aware(self):
        """What: now_utc returns a timezone-aware datetime.
        Why: naive datetimes are the root cause of AUD63 inconsistencies."""
        dt = now_utc()
        self.assertIsNotNone(dt.tzinfo)
        self.assertEqual(dt.tzinfo, datetime.timezone.utc)

    def test_now_utc_iso_ends_with_offset(self):
        """What: now_utc_iso returns a string ending with +00:00.
        Why: explicit offset is the AUD63 standard for persisted timestamps."""
        iso = now_utc_iso()
        self.assertTrue(iso.endswith("+00:00"), f"Expected +00:00 suffix, got: {iso}")

    def test_now_utc_iso_is_parseable(self):
        """What: the ISO string round-trips through fromisoformat.
        Why: a non-parseable timestamp defeats the purpose of standardization."""
        iso = now_utc_iso()
        dt = datetime.datetime.fromisoformat(iso)
        self.assertEqual(dt.tzinfo, datetime.timezone.utc)


class TestNormalizeTs(unittest.TestCase):
    """Tests for normalize_ts() on-read migration shim."""

    def test_already_utc_passthrough(self):
        """What: UTC timestamps pass through unchanged.
        Why: fast path -- no conversion overhead for compliant data."""
        ts = "2026-05-06T12:00:00+00:00"
        self.assertEqual(normalize_ts(ts), ts)

    def test_z_suffix_normalized(self):
        """What: Z suffix is normalized to +00:00.
        Why: both are valid ISO 8601 UTC, but we standardize on +00:00."""
        ts = "2026-05-06T12:00:00Z"
        result = normalize_ts(ts)
        self.assertTrue(result.endswith("+00:00"))
        self.assertNotIn("Z", result)

    def test_local_tz_offset_converted(self):
        """What: a -05:00 (CDT) timestamp is converted to UTC.
        Why: historical gating records used local-TZ; on-read must normalize."""
        ts = "2026-05-06T12:00:00-05:00"
        result = normalize_ts(ts)
        self.assertTrue(result.endswith("+00:00"), f"Expected UTC, got: {result}")
        # 12:00 CDT = 17:00 UTC
        self.assertIn("17:00:00", result)

    def test_naive_assumed_local(self):
        """What: naive timestamps are assumed local-TZ and converted.
        Why: pre-AUD63 writers used datetime.now() without timezone."""
        ts = "2026-05-06T12:00:00"
        result = normalize_ts(ts, assume_local=True)
        self.assertTrue(result.endswith("+00:00"), f"Expected UTC, got: {result}")

    def test_naive_assumed_utc(self):
        """What: with assume_local=False, naive timestamps get +00:00 directly.
        Why: some writers intended UTC but forgot the suffix."""
        ts = "2026-05-06T12:00:00"
        result = normalize_ts(ts, assume_local=False)
        self.assertIn("12:00:00", result)
        self.assertTrue(result.endswith("+00:00"))

    def test_empty_string_passthrough(self):
        """What: empty or None input returns unchanged.
        Why: fail-open -- don't crash on missing timestamps."""
        self.assertEqual(normalize_ts(""), "")
        self.assertIsNone(normalize_ts(None))

    def test_garbage_passthrough(self):
        """What: unparseable strings return unchanged.
        Why: fail-open for legacy data that doesn't match ISO 8601."""
        garbage = "not-a-timestamp"
        self.assertEqual(normalize_ts(garbage), garbage)


class TestAssertUtcIso(unittest.TestCase):
    """Tests for assert_utc_iso() write-path assertion."""

    def test_valid_utc_passes(self):
        """What: valid UTC timestamps pass through.
        Why: the assertion should not reject compliant data."""
        ts = "2026-05-06T12:00:00+00:00"
        self.assertEqual(assert_utc_iso(ts), ts)

    def test_valid_z_passes(self):
        """What: Z-suffixed timestamps also pass.
        Why: Z is a valid UTC designator in ISO 8601."""
        ts = "2026-05-06T12:00:00Z"
        self.assertEqual(assert_utc_iso(ts), ts)

    def test_naive_rejected(self):
        """What: naive timestamps raise ValueError.
        Why: write-path enforcement -- prevent new non-UTC writes."""
        with self.assertRaises(ValueError):
            assert_utc_iso("2026-05-06T12:00:00")

    def test_local_tz_rejected(self):
        """What: local-TZ timestamps raise ValueError.
        Why: only +00:00 or Z are acceptable for persisted data."""
        with self.assertRaises(ValueError):
            assert_utc_iso("2026-05-06T12:00:00-05:00")

    def test_empty_rejected(self):
        """What: empty string raises ValueError.
        Why: missing timestamps are a data integrity issue."""
        with self.assertRaises(ValueError):
            assert_utc_iso("")

    def test_context_in_error(self):
        """What: context string appears in the error message.
        Why: helps developers identify which write path produced the bad ts."""
        with self.assertRaises(ValueError) as ctx:
            assert_utc_iso("2026-05-06T12:00:00", context="tier.decay_tick")
        self.assertIn("tier.decay_tick", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_timestamp
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      AUD63 (UTC timestamp utilities)
# Layer:      test (pytest)
# Role:       tests for samia.core.timestamp — now_utc/now_utc_iso return UTC-aware + parseable, normalize_ts converts Z/local-TZ/naive to +00:00 (fail-open on garbage), assert_utc_iso rejects naive/local/empty with context in the error
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.timestamp
# Exposes:    — (test module)
# Lines:      166
# ------------------------------------------------------------------------------
