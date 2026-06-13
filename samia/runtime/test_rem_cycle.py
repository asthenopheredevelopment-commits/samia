"""Tests for samia.runtime.rem_cycle + samia.runtime.sleep_pressure (REM P1).

Layer 1 (Owns / Depends):
    Owns:    unit tests for the WAKE<->REM state machine (transitions persist +
             round-trip), the composite sleep-pressure metric (per-signal
             composition + absent-source = 0), the Q1 trigger (pressure AND idle
             OR the explicit force path, never a bare timer), and the THREE Q4
             wake paths (operator-activity -> wake_yield with no re-sleep;
             natural-drain + still-idle + work-remains -> snooze; max-duration ->
             wake_safety).
    Depends: samia.runtime.rem_cycle, samia.runtime.sleep_pressure, tempfile,
             unittest. All tests use tempdir memory roots — NEVER the live
             ~/.local/share memory or the global edges.db.

Layer 2 (What / Why):
    What: validates REM P1's contract from the approved proposal's Phase-1 Exit:
          (a) compute_pressure returns a normalized score + breakdown and flips
          sleep_needed at the threshold; (b) should_enter_rem requires pressure
          AND idle, OR the explicit flag; (c) the three wake paths each fire
          correctly; (d) state survives a simulated restart (persisted).
    Why:  REM is the sleep boundary gating ALL heavy offline memory work (P2).
          If entry/exit logic is wrong it either never rests (backlogs grow) or
          competes with active cognition (the swarm bug it exists to prevent).
"""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from samia.runtime import rem_cycle, sleep_pressure


def _mem() -> Path:
    """Fresh tempdir memory root, cleaned up at process exit.

    What: mkdtemp + an atexit-registered rmtree of every dir handed out. Why:
      mkdtemp does NOT auto-clean (only TemporaryDirectory does -- the old
      "auto-cleaned by tempfile" claim was false), so each call leaked a
      rem_test_* dir into /tmp and tripped the cold-metal zero-leftover hygiene
      gate. One atexit registration covers every `mem = _mem()` call site and
      any test order without editing each caller.
    """
    md = Path(tempfile.mkdtemp(prefix="rem_test_"))
    atexit.register(shutil.rmtree, md, ignore_errors=True)
    return md


def _write_consolidation_candidates(mem: Path, n: int) -> None:
    """Seed n near-dup candidates (the .consolidation_candidates.json source)."""
    p = mem / ".consolidation_candidates.json"
    p.write_text(json.dumps({
        "generated": "2026-06-07T00:00:00+00:00",
        "threshold": 0.85,
        "candidates": [{"a": str(i), "b": str(i + 1)} for i in range(n)],
    }))


def _write_offload(mem: Path, n: int) -> None:
    """Seed n session-offload state files (the offload-backlog source)."""
    d = mem / ".session_offload"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"sess{i}.json").write_text(json.dumps({"last_offload_byte": 1}))


# ---------------------------------------------------------------------------
# sleep_pressure — composition + absent sources
# ---------------------------------------------------------------------------


class TestSleepPressure(unittest.TestCase):

    def test_absent_sources_contribute_zero(self):
        """What: a bare memory dir (no clutter files) scores 0 with every signal
        marked absent. Why: a missing source must contribute 0, never crash."""
        mem = _mem()
        out = sleep_pressure.compute_pressure(mem)
        self.assertEqual(out["score"], 0.0)
        self.assertFalse(out["sleep_needed"])
        self.assertEqual(set(out["absent_sources"]), set(out["signals"].keys()))
        for sig in out["signals"].values():
            self.assertFalse(sig["present"])
            self.assertEqual(sig["contribution"], 0.0)

    def test_per_signal_composition_and_normalization(self):
        """What: two real signals compose into the sum; each normalizes to [0,1]
        against its own cap. Why: the composite is the honest owed-work measure."""
        mem = _mem()
        _write_offload(mem, 20)            # cap default 20 -> normalized 1.0
        _write_consolidation_candidates(mem, 300)  # cap 600 -> normalized 0.5
        out = sleep_pressure.compute_pressure(mem)
        self.assertTrue(out["signals"]["offload_backlog"]["present"])
        self.assertAlmostEqual(out["signals"]["offload_backlog"]["normalized"], 1.0)
        self.assertAlmostEqual(out["signals"]["near_dup_backlog"]["normalized"], 0.5)
        # sum = 1.0 + 0.5 (other four absent = 0)
        self.assertAlmostEqual(out["score"], 1.5)

    def test_normalized_saturates_at_one(self):
        """What: a raw count above its cap saturates at 1.0. Why: keeps the sum
        interpretable — no single raw-count signal can swamp the others."""
        mem = _mem()
        _write_offload(mem, 1000)  # way over cap 20
        out = sleep_pressure.compute_pressure(mem)
        self.assertEqual(out["signals"]["offload_backlog"]["normalized"], 1.0)
        self.assertLessEqual(out["score"], len(out["signals"]))

    def test_sleep_needed_flips_at_threshold(self):
        """What: sleep_needed is False below the threshold and True at/above it.
        Why: this boolean is the pressure half of the Q1 trigger."""
        mem = _mem()
        _write_consolidation_candidates(mem, 300)  # normalized 0.5
        below = sleep_pressure.compute_pressure(mem, threshold=1.0)
        self.assertFalse(below["sleep_needed"])
        at = sleep_pressure.compute_pressure(mem, threshold=0.5)
        self.assertTrue(at["sleep_needed"])

    def test_contradiction_backlog_counts_unresolved(self):
        """What: unresolved supersession rows feed the contradiction signal.
        Why: grounds the signal in the real P3 store, unresolved-only."""
        mem = _mem()
        bio = mem / "biomimetic"
        bio.mkdir(parents=True)
        store = bio / "supersession_candidates.jsonl"
        rows = [
            {"old_id": "a.md", "new_id": "b.md"},               # unresolved
            {"old_id": "c.md", "new_id": "d.md", "confirmed": True},  # resolved
            {"old_id": "e.md", "new_id": "f.md", "dismissed": True},  # resolved
        ]
        store.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out = sleep_pressure.compute_pressure(mem)
        sig = out["signals"]["contradiction_backlog"]
        self.assertTrue(sig["present"])
        self.assertEqual(sig["raw"], 1.0)  # only the unresolved row


# ---------------------------------------------------------------------------
# rem_cycle — state machine persistence + round-trip
# ---------------------------------------------------------------------------


class TestStateMachine(unittest.TestCase):

    def test_default_state_is_wake(self):
        """What: a never-slept memory dir reads WAKE. Why: cold start is awake."""
        mem = _mem()
        st = rem_cycle.current_state(mem)
        self.assertEqual(st["phase"], rem_cycle.WAKE)
        self.assertFalse(rem_cycle.is_rem(mem))

    def test_enter_and_wake_transitions_persist_and_round_trip(self):
        """What: enter_rem/wake persist to rem_state.json and survive a re-read
        (simulated restart — a fresh read sees the same phase). Why: the gate
        must outlast a daemon restart (Phase-1 Exit (d))."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test_enter")
        self.assertTrue((mem / "biomimetic" / "rem_state.json").exists())
        # Fresh read (no in-memory caching) == simulated restart round-trip.
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REM)
        self.assertTrue(rem_cycle.is_rem(mem))
        cycle = rem_cycle.current_state(mem)["cycle_id"]
        self.assertIsNotNone(cycle)
        rem_cycle.wake(mem, "operator_activity")
        st = rem_cycle.current_state(mem)
        self.assertEqual(st["phase"], rem_cycle.WAKE)
        self.assertIsNone(st["cycle_id"])
        self.assertEqual(st["last_wake_reason"], "operator_activity")

    def test_review_is_still_rem(self):
        """What: REVIEWING counts as in_rem (a REM sub-state). Why: mid-batch
        work is still legitimately resting until a true wake."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "x")
        rem_cycle.review(mem, "natural")
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REVIEWING)
        self.assertTrue(rem_cycle.is_rem(mem))

    def test_snooze_preserves_cycle_id(self):
        """What: re-entering REM from REVIEWING keeps the same cycle_id (a snooze
        resumes the cycle); from WAKE it starts a fresh one. Why: a snooze is the
        same rest continuing, not a new cycle."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "x")
        c1 = rem_cycle.current_state(mem)["cycle_id"]
        rem_cycle.review(mem, "natural")
        rem_cycle.enter_rem(mem, "snooze")
        self.assertEqual(rem_cycle.current_state(mem)["cycle_id"], c1)
        rem_cycle.wake(mem, "op")
        rem_cycle.enter_rem(mem, "fresh")
        self.assertNotEqual(rem_cycle.current_state(mem)["cycle_id"], c1)

    def test_events_logged(self):
        """What: transitions append to rem_events.jsonl. Why: operator-visible
        observability of the cycle (risk-5)."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "x")
        rem_cycle.wake(mem, "op")
        log = mem / "biomimetic" / "rem_events.jsonl"
        events = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        kinds = [e["event"] for e in events]
        self.assertIn("enter_rem", kinds)
        self.assertIn("wake", kinds)


# ---------------------------------------------------------------------------
# rem_cycle — Q1 trigger (pressure AND idle, OR explicit force; never a timer)
# ---------------------------------------------------------------------------


class TestTrigger(unittest.TestCase):

    def test_enters_only_when_pressure_and_idle(self):
        """What: should_enter_rem True only when pressure>=threshold AND idle.
        Why: pressure-without-idle competes with active cognition (the swarm
        bug); idle-without-pressure has nothing owed."""
        mem = _mem()
        _write_offload(mem, 30)  # saturates -> score >= default threshold 1.0
        idle_s = rem_cycle.IDLE_GATE_S + 10
        active_s = 1.0
        # pressure AND idle -> enter
        d = rem_cycle.should_enter_rem(mem, idle_seconds=idle_s)
        self.assertTrue(d["enter"])
        self.assertEqual(d["reason"], "pressure_and_idle")
        # pressure but NOT idle -> no enter
        d = rem_cycle.should_enter_rem(mem, idle_seconds=active_s)
        self.assertFalse(d["enter"])
        self.assertEqual(d["reason"], "not_idle")

    def test_no_pressure_no_enter_even_when_idle(self):
        """What: idle but zero pressure does not enter. Why: a bare idle-duration
        timer must never trip REM (feedback_scheduling_minimize_clocks)."""
        mem = _mem()  # no clutter -> score 0
        d = rem_cycle.should_enter_rem(mem, idle_seconds=rem_cycle.IDLE_GATE_S + 9999)
        self.assertFalse(d["enter"])
        self.assertEqual(d["reason"], "no_pressure")

    def test_explicit_force_enters_regardless(self):
        """What: the explicit force flag enters REM even with zero pressure and
        zero idleness. Why: Q1's on-demand trigger (risk-1 mitigation)."""
        mem = _mem()  # no pressure
        rem_cycle.request_sleep_now(mem)
        d = rem_cycle.should_enter_rem(mem, idle_seconds=0.0)
        self.assertTrue(d["enter"])
        self.assertTrue(d["forced"])
        self.assertEqual(d["reason"], "explicit_trigger")

    def test_rem_sleep_now_sets_flag_and_tick_enters(self):
        """What: rem_sleep_now flips the flag; tick then enters REM. Why: the
        thin trigger surface the IPC/MCP op wraps actually drives the machine."""
        mem = _mem()
        rem_cycle.rem_sleep_now(mem)
        res = rem_cycle.tick(mem, activity=False, idle_seconds=0.0)
        self.assertEqual(res["action"], "enter_rem")
        self.assertTrue(rem_cycle.is_rem(mem))


# ---------------------------------------------------------------------------
# rem_cycle — Q4 the THREE wake paths
# ---------------------------------------------------------------------------


class TestWakePaths(unittest.TestCase):

    def test_path_a_operator_activity_wakes_and_yields_no_resleep(self):
        """What: operator activity -> wake_yield, the cycle ENDS (->WAKE), and a
        follow-up tick does NOT auto-re-sleep (it re-evaluates from WAKE). Why:
        operator activity is the only TRUE end of a cycle (Q4 path a)."""
        mem = _mem()
        _write_offload(mem, 30)  # pressure high + idle below would otherwise sleep
        rem_cycle.enter_rem(mem, "x")
        res = rem_cycle.tick(mem, activity=True, idle_seconds=rem_cycle.IDLE_GATE_S + 5)
        self.assertEqual(res["action"], rem_cycle.WAKE_YIELD)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.WAKE)
        # Immediately ticking again with activity STILL present must not re-sleep.
        res2 = rem_cycle.tick(mem, activity=True, idle_seconds=1.0)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.WAKE)
        self.assertEqual(res2["action"], "stay_wake")

    def test_path_b_natural_drain_then_snooze_when_idle_and_work_remains(self):
        """What: in REM with pressure BELOW threshold but work still remaining and
        idle -> first tick enters REVIEWING; after the review window with still-
        idle + work-remains -> SNOOZE back into REM. Why: Q4's load-bearing
        refinement — a natural wake must not strand leftover work."""
        mem = _mem()
        # score 0.5 (one near-dup signal): work REMAINS (>0) but pressure is
        # BELOW the default threshold 1.0 -> a natural completion.
        _write_consolidation_candidates(mem, 300)
        self.assertTrue(sleep_pressure.compute_pressure(mem)["score"] > 0.0)
        self.assertFalse(sleep_pressure.compute_pressure(mem)["sleep_needed"])
        rem_cycle.enter_rem(mem, "x")
        idle_s = rem_cycle.IDLE_GATE_S + 5
        # Step 1: natural completion -> REVIEWING.
        r1 = rem_cycle.tick(mem, activity=False, idle_seconds=idle_s)
        self.assertEqual(r1["action"], rem_cycle.ENTER_REVIEWING)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REVIEWING)
        # Step 2: simulate the review window having elapsed by passing a `now`
        # past review_started_ts + REVIEW_WAIT_S. Still idle + work remains.
        rstart = float(rem_cycle.current_state(mem)["review_started_ts"])
        future = rstart + rem_cycle.REVIEW_WAIT_S + 1
        r2 = rem_cycle.tick(mem, activity=False, idle_seconds=idle_s, now=future)
        self.assertEqual(r2["action"], rem_cycle.SNOOZE)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REM)

    def test_path_b_review_window_open_keeps_reviewing(self):
        """What: inside the review window (before REVIEW_WAIT_S elapses) the
        machine stays REVIEWING, waiting for instructions. Why: the bounded
        brief-wait of the snooze gate."""
        mem = _mem()
        _write_consolidation_candidates(mem, 300)  # work remains, below threshold
        rem_cycle.enter_rem(mem, "x")
        idle_s = rem_cycle.IDLE_GATE_S + 5
        rem_cycle.tick(mem, activity=False, idle_seconds=idle_s)  # -> REVIEWING
        # `now` only a moment after review start: window still OPEN.
        rstart = float(rem_cycle.current_state(mem)["review_started_ts"])
        r = rem_cycle.tick(mem, activity=False, idle_seconds=idle_s, now=rstart + 1)
        self.assertEqual(r["action"], "reviewing_wait")
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REVIEWING)

    def test_path_b_settles_to_rest_when_no_work_remains(self):
        """What: after the review window with NO work remaining -> REST (stay in
        REM doing nothing), not a re-snooze. Why: prevents the busy snooze loop
        (risk-3) — once drained, settle into idle REM-rest."""
        mem = _mem()  # zero clutter -> work_remains False, pressure 0
        rem_cycle.enter_rem(mem, "x")
        idle_s = rem_cycle.IDLE_GATE_S + 5
        r1 = rem_cycle.tick(mem, activity=False, idle_seconds=idle_s)  # -> REVIEWING
        self.assertEqual(r1["action"], rem_cycle.ENTER_REVIEWING)
        rstart = float(rem_cycle.current_state(mem)["review_started_ts"])
        future = rstart + rem_cycle.REVIEW_WAIT_S + 1
        r2 = rem_cycle.tick(mem, activity=False, idle_seconds=idle_s, now=future)
        self.assertEqual(r2["action"], rem_cycle.REST)
        # Still in REM (resting), did not wake or re-snooze churn.
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REM)

    def test_path_b_rest_is_stable_no_churn(self):
        """What: once settled into idle REM-rest, a follow-up tick (still no work,
        still idle) stays REST in REM — it does NOT bounce back to REVIEWING.
        Why: the busy snooze/review loop guard (risk-3) — rest is a stable sink."""
        mem = _mem()  # zero clutter
        rem_cycle.enter_rem(mem, "x")
        idle_s = rem_cycle.IDLE_GATE_S + 5
        rem_cycle.tick(mem, activity=False, idle_seconds=idle_s)  # -> REVIEWING
        rstart = float(rem_cycle.current_state(mem)["review_started_ts"])
        rem_cycle.tick(mem, activity=False, idle_seconds=idle_s,
                       now=rstart + rem_cycle.REVIEW_WAIT_S + 1)  # -> REST (REM)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REM)
        # Follow-up tick: still nothing owed -> stays REST, phase unchanged.
        r3 = rem_cycle.tick(mem, activity=False, idle_seconds=idle_s)
        self.assertEqual(r3["action"], rem_cycle.REST)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REM)

    def test_path_c_max_duration_safety_wake(self):
        """What: when MAX_DURATION_S has elapsed since the phase began -> a
        safety wake (->WAKE), regardless of pressure/work. Why: the hard backstop
        so a misbehaving subscriber can never hold REM forever (Q4 path c)."""
        mem = _mem()
        _write_offload(mem, 30)  # high pressure + work remains: would normally stay
        rem_cycle.enter_rem(mem, "x")
        since = float(rem_cycle.current_state(mem)["since_ts"])
        future = since + rem_cycle.MAX_DURATION_S + 1
        r = rem_cycle.tick(mem, activity=False, idle_seconds=rem_cycle.IDLE_GATE_S + 5,
                           now=future)
        self.assertEqual(r["action"], rem_cycle.WAKE_SAFETY)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.WAKE)

    def test_high_pressure_stays_rem(self):
        """What: in REM with pressure ABOVE threshold + work remains + idle +
        no activity + under the cap -> STAY_REM (keep working). Why: the cycle
        continues while genuine reconciliation is owed."""
        mem = _mem()
        _write_offload(mem, 30)  # saturates -> above threshold
        rem_cycle.enter_rem(mem, "x")
        r = rem_cycle.evaluate(mem, activity=False,
                               idle_seconds=rem_cycle.IDLE_GATE_S + 5)
        self.assertEqual(r["action"], rem_cycle.STAY_REM)


# ---------------------------------------------------------------------------
# rem_cycle — idle gate + status surface
# ---------------------------------------------------------------------------


class TestIdleAndStatus(unittest.TestCase):

    def test_is_idle_threshold(self):
        """What: is_idle compares seconds-since-activity to IDLE_GATE_S; unknown
        idleness fails closed (not idle). Why: never sleep into active work."""
        self.assertTrue(rem_cycle.is_idle(rem_cycle.IDLE_GATE_S + 1))
        self.assertFalse(rem_cycle.is_idle(rem_cycle.IDLE_GATE_S - 1))
        self.assertTrue(rem_cycle.is_idle(idle_seconds=0.0) is False)  # 0s = active

    def test_rem_status_returns_state_and_gauge(self):
        """What: rem_status bundles the state + the full pressure breakdown.
        Why: the operator-visible health gauge the MCP/IPC surface wraps."""
        mem = _mem()
        _write_consolidation_candidates(mem, 100)
        out = rem_cycle.rem_status(mem)
        self.assertIn("state", out)
        self.assertIn("pressure", out)
        self.assertIn("signals", out["pressure"])
        self.assertEqual(out["state"]["phase"], rem_cycle.WAKE)


if __name__ == "__main__":
    unittest.main()


# ─────────────────────────────────────────────
# [test_rem_cycle] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.runtime
# Version:    1.0.0  Updated: 2026-06-07  Status: active
# Phase:      FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P1 tests)
# Role:       unit tests — state machine round-trip, sleep-pressure composition,
#             Q1 trigger, Q4 three wake paths (incl. the snooze)
# Depends:    json, tempfile, unittest; samia.runtime.rem_cycle, sleep_pressure
# ─────────────────────────────────────────────
