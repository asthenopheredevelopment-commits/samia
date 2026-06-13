"""samia.core.timestamp -- UTC timestamp utilities and on-read normalization.

Layer 1 (Owns / Depends):
    Owns:    now_utc(), normalize_ts(), assert_utc_iso() -- the canonical
             timestamp primitives for all Asthenosphere write and read paths.
    Depends: datetime, re (stdlib only).

Layer 2 (What / Why):
    What: Provides a single source-of-truth UTC timestamp function (now_utc)
          that all writers import instead of calling datetime.now() or the
          deprecated datetime.utcnow(). Includes an on-read normalization
          shim (normalize_ts) that converts legacy local-TZ or naive
          timestamps to UTC, and a write-path assertion (assert_utc_iso)
          that rejects non-UTC strings at serialization boundaries.
    Why:  AUD63 identified inconsistent timestamp conventions across
          subsystems (gating uses local-TZ, others use UTC, some are naive).
          This module enforces a single convention: store UTC with explicit
          +00:00 suffix. On-read normalization handles historical data
          without bulk rewrites. The write-path assertion catches new
          violations at runtime rather than allowing silent drift.

Design doc: AUD63_utc_timestamp_normalization.md
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Optional

_log = logging.getLogger("samia.core.timestamp")

# ---------------------------------------------------------------------------
# Phase 3: Canonical UTC timestamp factory
# ---------------------------------------------------------------------------

# _UTC -- What: timezone constant for UTC.
# _UTC -- Why: avoids repeated construction; used by now_utc() and
#   normalize_ts(). datetime.timezone.utc is the modern replacement for
#   the deprecated datetime.utcnow().
_UTC = datetime.timezone.utc


def now_utc() -> datetime.datetime:
    """Return the current time as a timezone-aware UTC datetime.

    What: wraps datetime.now(timezone.utc).
    Why:  single import point so every writer uses the same call. Avoids
          datetime.utcnow() (deprecated Python 3.12+) and datetime.now()
          (returns naive local time).
    """
    return datetime.datetime.now(_UTC)


def now_utc_iso(timespec: str = "seconds") -> str:
    """Return the current time as an ISO 8601 UTC string with +00:00 suffix.

    What: convenience wrapper combining now_utc() with isoformat().
    Why:  most callers want the string form for JSON serialization.
          The +00:00 suffix makes UTC explicit even for parsers that
          ignore the 'Z' convention.
    """
    return now_utc().isoformat(timespec=timespec)


# ---------------------------------------------------------------------------
# Phase 2: On-read normalization shim
# ---------------------------------------------------------------------------

# _OFFSET_RE -- What: regex to detect an explicit UTC offset in an ISO string.
# _OFFSET_RE -- Why: used by normalize_ts() to distinguish naive, local-TZ,
#   and UTC timestamps. Matches +HH:MM, -HH:MM, or Z at end of string.
_OFFSET_RE = re.compile(r"(?:[+-]\d{2}:\d{2}|Z)$")

# _LOCAL_TZ -- What: the operator's local timezone object (system default).
# _LOCAL_TZ -- Why: used by normalize_ts() to interpret naive timestamps as
#   local time (best-effort assumption for historical gating records that
#   used datetime.now() without timezone info).
try:
    _LOCAL_TZ = datetime.datetime.now().astimezone().tzinfo
except Exception:
    _LOCAL_TZ = None


def normalize_ts(ts_str: str, assume_local: bool = True) -> str:
    """Normalize a timestamp string to UTC ISO 8601 with +00:00 suffix.

    What: parses the input timestamp, converts to UTC if it has a non-UTC
          offset, and attaches +00:00 if naive. Returns the normalized
          ISO string.
    Why:  on-read migration for historical records that used local-TZ
          or naive timestamps. No bulk rewrite needed -- conversion
          happens transparently at the read boundary.

    Parameters
    ----------
    ts_str : str
        ISO 8601 timestamp string. May be naive, local-TZ, or already UTC.
    assume_local : bool
        If True and the timestamp is naive (no timezone info), assume it
        was written in the operator's local timezone and convert to UTC.
        If False, assume naive timestamps are already UTC (just attach
        +00:00).

    Returns
    -------
    str
        ISO 8601 string with explicit +00:00 suffix.

    Notes
    -----
    Returns the original string unchanged if parsing fails (fail-open).
    Logs a warning on conversion so the operator can track how many old
    records remain un-normalized.
    """
    if not ts_str or not isinstance(ts_str, str):
        return ts_str

    # What: fast path -- already UTC.
    # Why: avoids parsing overhead for the common case.
    if ts_str.endswith("+00:00") or ts_str.endswith("Z"):
        # Normalize Z to +00:00 for consistency.
        if ts_str.endswith("Z"):
            return ts_str[:-1] + "+00:00"
        return ts_str

    try:
        dt = datetime.datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        _log.debug("timestamp: could not parse '%s'; returning unchanged", ts_str)
        return ts_str

    if dt.tzinfo is not None:
        # What: has a non-UTC offset (e.g. -05:00 for CDT).
        # Why: convert to UTC and re-format.
        utc_dt = dt.astimezone(_UTC)
        _log.debug("timestamp: converted local-TZ '%s' -> UTC", ts_str)
        return utc_dt.isoformat(timespec="seconds") + (
            "" if "+00:00" in utc_dt.isoformat() else ""
        )

    # What: naive datetime -- no timezone info at all.
    # Why: assume local-TZ if assume_local=True (matches the gating ask.py
    #   behavior before M2 fix), otherwise treat as UTC.
    if assume_local and _LOCAL_TZ is not None:
        local_dt = dt.replace(tzinfo=_LOCAL_TZ)
        utc_dt = local_dt.astimezone(_UTC)
        _log.debug("timestamp: assumed local-TZ for naive '%s' -> UTC", ts_str)
        return utc_dt.isoformat(timespec="seconds")
    else:
        utc_dt = dt.replace(tzinfo=_UTC)
        return utc_dt.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Phase 3: Write-path assertion
# ---------------------------------------------------------------------------


def assert_utc_iso(ts_str: str, context: str = "") -> str:
    """Assert that a timestamp string is UTC ISO 8601 with explicit offset.

    What: validates the timestamp has +00:00 or Z suffix and is parseable.
    Why:  write-path guard that catches naive or local-TZ timestamps at
          the serialization boundary. Fails fast with ValueError rather
          than silently persisting an inconsistent timestamp.

    Parameters
    ----------
    ts_str : str
        The timestamp string to validate.
    context : str
        Optional context for the error message (e.g. caller name).

    Returns
    -------
    str
        The validated timestamp string (unchanged).

    Raises
    ------
    ValueError
        If the timestamp lacks an explicit UTC offset.
    """
    if not isinstance(ts_str, str) or not ts_str:
        raise ValueError(
            f"timestamp must be a non-empty string, got {type(ts_str).__name__}"
            f"{f' ({context})' if context else ''}"
        )

    if not (ts_str.endswith("+00:00") or ts_str.endswith("Z")):
        raise ValueError(
            f"timestamp must be UTC (end with +00:00 or Z), got '{ts_str}'"
            f"{f' ({context})' if context else ''}"
        )

    # What: verify it's actually parseable.
    # Why: catch garbage strings that happen to end with +00:00.
    try:
        check = ts_str.replace("Z", "+00:00")
        datetime.datetime.fromisoformat(check)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"timestamp not valid ISO 8601: '{ts_str}'"
            f"{f' ({context})' if context else ''}"
        ) from exc

    return ts_str


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.timestamp
# Phase:      AUD63 -- Phases 2-3 (on-read normalization + write enforcement)
# Layer:      core (pure library, no daemon dependency)
# Stability:  v1.0 -- foundational utility, no expected API changes
# ErrorModel: normalize_ts is fail-open (returns original on parse error);
#             assert_utc_iso is fail-fast (raises ValueError).
# Depends:    datetime, re, logging (stdlib).
# Exposes:    now_utc, now_utc_iso, normalize_ts, assert_utc_iso.
# Lines:      ~170
# --------------------------------------------------------------------------
