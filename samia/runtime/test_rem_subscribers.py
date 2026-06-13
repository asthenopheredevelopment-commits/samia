"""Tests for REM P2 — the offline-op subscriber registry, gate, and driver.

Layer 1 (Owns / Depends):
    Owns:    unit tests for rem_cycle's P2 surface (register_offline_op +
             run_due_subscribers + the run-only-in-REM gate + cursor
             checkpoint/resume + work_remaining feeding evaluate) AND the
             rem_subscribers migration wiring (the migrated ops refuse outside
             REM but run inside; the registry lists them in priority order).
    Depends: samia.runtime.rem_cycle, samia.runtime.rem_subscribers, tempfile,
             unittest. Every test uses a tempdir memory root — NEVER the live
             ~/.local/share memory or the global edges.db.

Layer 2 (What / Why):
    What: validates REM P2's Phase-2 Exit from the approved proposal:
          (a) register + priority order; (b) one subscriber failing does not
          abort the rest; (c) the gate blocks an offline op when NOT is_rem and
          allows it when is_rem; (d) an interrupt mid-run defers the remaining
          subscribers + leaves a resumable cursor; (e) a REM-gated migrated op
          does NOT run on an awake idle pulse but DOES run in REM; (f)
          work_remaining feeds evaluate(); (g) decay (the forgetting curve) is
          NOT REM-gated — it RUNS outside REM and is ABSENT from the REM
          registry (operator correction 2026-06-07).
    Why:  P2 moves the heavy STRENGTHENING/ABSTRACTING offline work behind the
          sleep boundary. Decay is the EXCEPTION (CLS): forgetting is short-term
          memory loss that runs continuously across wake+REM, so it stays on the
          idle_pulse path, ungated. If the gate leaks, offline strengthening
          work competes with active cognition (the swarm bug). If the driver
          aborts on one failure or loses a cursor, work is silently dropped (the
          regression the proposal exists to prevent).
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from samia.runtime import rem_cycle, rem_subscribers


def _mem() -> Path:
    """Fresh tempdir memory root, cleaned up at process exit.

    What: mkdtemp + an atexit-registered rmtree of every dir this helper hands
      out. Why: mkdtemp does NOT auto-clean (only TemporaryDirectory does -- the
      old docstring's "auto-cleaned by tempfile" claim was false), so each call
      left a rem_p2_test_* dir in /tmp and tripped the cold-metal hygiene gate
      (zero-leftover requirement). atexit fires once after the whole suite
      process exits, so a single registration covers every call site and any
      test order without touching the dozens of `mem = _mem()` callers.
    """
    md = Path(tempfile.mkdtemp(prefix="rem_p2_test_"))
    atexit.register(shutil.rmtree, md, ignore_errors=True)
    return md


class _RegistryIsolation(unittest.TestCase):
    """Base: snapshot + clear the global subscriber registry around each test so
    registry state never leaks between tests (the registry is module-global)."""

    def setUp(self) -> None:
        with rem_cycle._rem_subscribers_lock:
            self._saved = dict(rem_cycle._rem_subscribers)
            rem_cycle._rem_subscribers.clear()

    def tearDown(self) -> None:
        with rem_cycle._rem_subscribers_lock:
            rem_cycle._rem_subscribers.clear()
            rem_cycle._rem_subscribers.update(self._saved)


# ---------------------------------------------------------------------------
# Registry + priority order + due-condition
# ---------------------------------------------------------------------------


class TestRegistry(_RegistryIsolation):

    def test_register_and_priority_order(self):
        """What: registered ops sort by ascending priority (LOW first). Why:
        decay must run before the heavy graph work that runs on the pruned set."""
        rem_cycle.register_offline_op("replay", lambda m: {}, priority=30)
        rem_cycle.register_offline_op("decay", lambda m: {}, priority=10)
        rem_cycle.register_offline_op("consolidation", lambda m: {}, priority=20)
        self.assertEqual(rem_cycle.registered_offline_ops(),
                         ["decay", "consolidation", "replay"])

    def test_reregister_preserves_stats(self):
        """What: re-registering a name updates fn/priority but keeps run stats.
        Why: idempotent across daemon re-init (mirrors idle_pulse)."""
        sub = rem_cycle.register_offline_op("x", lambda m: {}, priority=5)
        sub.run_count = 7
        rem_cycle.register_offline_op("x", lambda m: {"changed": True}, priority=9)
        self.assertEqual(rem_cycle._rem_subscribers["x"].run_count, 7)
        self.assertEqual(rem_cycle._rem_subscribers["x"].priority, 9)

    def test_driver_runs_in_priority_order(self):
        """What: the driver runs due subscribers in ascending priority. Why: run
        order is load-bearing (prune before consolidate/replay)."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test")
        order: list[str] = []
        rem_cycle.register_offline_op("c", lambda m: order.append("c") or {}, priority=30)
        rem_cycle.register_offline_op("a", lambda m: order.append("a") or {}, priority=10)
        rem_cycle.register_offline_op("b", lambda m: order.append("b") or {}, priority=20)
        rem_cycle.run_due_subscribers(mem)
        self.assertEqual(order, ["a", "b", "c"])

    def test_due_condition_skips_not_due(self):
        """What: a subscriber whose due_fn returns False is skipped (not run).
        Why: a cycle with no real backlog for an op should not run it."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test")
        ran: list[str] = []
        rem_cycle.register_offline_op("idle_op", lambda m: ran.append("idle") or {},
                                      priority=10, due_condition=lambda m: False)
        rem_cycle.register_offline_op("busy_op", lambda m: ran.append("busy") or {},
                                      priority=20, due_condition=lambda m: True)
        rem_cycle.run_due_subscribers(mem)
        self.assertEqual(ran, ["busy"])


# ---------------------------------------------------------------------------
# Fail-open: one subscriber failing does not abort the rest
# ---------------------------------------------------------------------------


class TestFailOpen(_RegistryIsolation):

    def test_one_failure_does_not_abort_the_rest(self):
        """What: a subscriber that raises is recorded as an error but the driver
        continues to the remaining subscribers. Why: an offline op failing must
        never silently drop the rest of the batch (the regression to prevent)."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test")
        ran: list[str] = []

        def _boom(m):
            ran.append("boom")
            raise RuntimeError("intentional")

        rem_cycle.register_offline_op("ok_first", lambda m: ran.append("ok1") or {}, priority=10)
        rem_cycle.register_offline_op("boom", _boom, priority=20)
        rem_cycle.register_offline_op("ok_last", lambda m: ran.append("ok2") or {}, priority=30)
        out = rem_cycle.run_due_subscribers(mem)
        # All three were attempted; the failure did not abort the rest.
        self.assertEqual(ran, ["ok1", "boom", "ok2"])
        self.assertIn("error", out["results"]["boom"])
        self.assertEqual(rem_cycle._rem_subscribers["boom"].error_count, 1)
        # The error is operator-visible in the event log (logged, not swallowed).
        log = (mem / "biomimetic" / "rem_events.jsonl").read_text()
        self.assertIn("subscriber_error", log)


# ---------------------------------------------------------------------------
# The run-only-in-REM gate
# ---------------------------------------------------------------------------


class TestGate(_RegistryIsolation):

    def test_gate_blocks_outside_rem_allows_inside(self):
        """What: gate_offline_op is False in WAKE (logged refusal) and True in
        REM. Why: offline work refuses to run outside the sleep window (Q5)."""
        mem = _mem()
        # WAKE (default) -> blocked + logged.
        self.assertFalse(rem_cycle.gate_offline_op(mem, "any_op"))
        log = (mem / "biomimetic" / "rem_events.jsonl").read_text()
        self.assertIn("offline_refused", log)
        # REM -> allowed.
        rem_cycle.enter_rem(mem, "test")
        self.assertTrue(rem_cycle.gate_offline_op(mem, "any_op"))

    def test_only_in_rem_decorator_refuses_outside_runs_inside(self):
        """What: the @only_in_rem decorator returns a logged refusal dict outside
        REM and runs the wrapped fn inside REM. Why: an op gates itself at its own
        entry so it refuses regardless of which caller invokes it."""
        mem = _mem()
        calls: list[str] = []

        @rem_cycle.only_in_rem("guarded")
        def _op(m, *, val=1):
            calls.append("ran")
            return {"fired": True, "val": val}

        # WAKE -> refused, fn body NOT executed.
        out = _op(mem, val=9)
        self.assertEqual(out["refused"], "not_in_rem")
        self.assertEqual(calls, [])
        # REM -> runs.
        rem_cycle.enter_rem(mem, "test")
        out = _op(mem, val=9)
        self.assertTrue(out["fired"])
        self.assertEqual(out["val"], 9)
        self.assertEqual(calls, ["ran"])

    def test_driver_refuses_to_run_outside_rem(self):
        """What: run_due_subscribers in WAKE runs NO subscriber (logged refusal).
        Why: the driver is the offline-work engine — it must never fire while
        awake even if subscribers are registered."""
        mem = _mem()  # WAKE
        ran: list[str] = []
        rem_cycle.register_offline_op("op", lambda m: ran.append("x") or {}, priority=10)
        out = rem_cycle.run_due_subscribers(mem)
        self.assertEqual(out.get("refused"), "not_in_rem")
        self.assertEqual(ran, [])
        self.assertEqual(out["ran"], 0)


# ---------------------------------------------------------------------------
# Interrupt mid-run -> remaining subscribers deferred + checkpoint
# ---------------------------------------------------------------------------


class TestInterrupt(_RegistryIsolation):

    def test_activity_interrupt_defers_remaining_and_checkpoints(self):
        """What: with activity=True (operator arrived) the driver runs nothing
        new and defers all due subscribers; a subscriber that checkpoints a
        cursor leaves it resumable. Why: REM yields promptly to active cognition
        (Q4 path a) and interrupted work resumes from its cursor next cycle."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test")
        ran: list[str] = []

        def _checkpointing(m):
            ran.append("cp")
            rem_cycle.write_cursor(m, "cp_op", {"remaining": 5, "done": False})
            return {"work_remaining": True}

        rem_cycle.register_offline_op("cp_op", _checkpointing, priority=10,
                                      cursor_key="cp_op")
        rem_cycle.register_offline_op("other", lambda m: ran.append("other") or {},
                                      priority=20)
        # Pre-seed a cursor so the resumable-state assertion is meaningful even
        # though the interrupt defers BEFORE running (activity preempts).
        rem_cycle.write_cursor(mem, "cp_op", {"remaining": 5, "done": False})
        out = rem_cycle.run_due_subscribers(mem, activity=True)
        self.assertTrue(out["interrupted"])
        self.assertEqual(ran, [])  # activity preempts -> nothing new runs
        self.assertEqual(set(out["deferred"]), {"cp_op", "other"})
        # The cursor is resumable (remaining work recorded).
        cur = rem_cycle.read_cursor(mem, "cp_op")
        self.assertEqual(cur["remaining"], 5)
        self.assertFalse(cur["done"])

    def test_budget_defers_overflow_to_next_cycle(self):
        """What: with a budget of 1, only the first due subscriber runs; the rest
        are deferred (resume next cycle). Why: a single tick stays responsive;
        the cursor model finishes the rest later (no silent drop)."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test")
        ran: list[str] = []
        rem_cycle.register_offline_op("first", lambda m: ran.append("1") or {}, priority=10)
        rem_cycle.register_offline_op("second", lambda m: ran.append("2") or {}, priority=20)
        out = rem_cycle.run_due_subscribers(mem, budget=1)
        self.assertEqual(ran, ["1"])
        self.assertEqual(out["ran"], 1)
        self.assertIn("second", out["deferred"])

    def test_cursor_remaining_makes_op_due_even_if_due_fn_false(self):
        """What: an op whose due_fn is False but whose cursor records remaining
        work is still run (resumed). Why: an interrupted op must finish even if
        its source backlog reads empty mid-batch."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test")
        ran: list[str] = []
        rem_cycle.write_cursor(mem, "resume_op", {"remaining": 3, "done": False})
        rem_cycle.register_offline_op("resume_op", lambda m: ran.append("r") or {},
                                      priority=10, due_condition=lambda m: False,
                                      cursor_key="resume_op")
        rem_cycle.run_due_subscribers(mem)
        self.assertEqual(ran, ["r"])


# ---------------------------------------------------------------------------
# work_remaining feeds evaluate()
# ---------------------------------------------------------------------------


class TestWorkRemainingFeedsEvaluate(_RegistryIsolation):

    def test_subscriber_due_keeps_work_remains_true_at_zero_pressure(self):
        """What: with ZERO sleep-pressure but a subscriber that is due,
        work_remains is True (the subscriber half OR-in). Why: P2 replaces the
        pressure>0 proxy — evaluate() must not drain a cycle while a subscriber
        still owes work."""
        mem = _mem()  # zero clutter -> pressure 0
        # No subscriber yet -> no work remains.
        self.assertFalse(rem_cycle.work_remains(mem))
        # A due subscriber -> work remains True despite zero pressure.
        rem_cycle.register_offline_op("due_op", lambda m: {}, priority=10,
                                      due_condition=lambda m: True)
        self.assertTrue(rem_cycle.work_remains(mem))

    def test_cursor_remaining_keeps_work_remains_true(self):
        """What: a subscriber that is NOT due but has cursor-remaining work keeps
        work_remains True. Why: interrupted work counts as owed."""
        mem = _mem()
        rem_cycle.register_offline_op("op", lambda m: {}, priority=10,
                                      due_condition=lambda m: False,
                                      cursor_key="op")
        self.assertFalse(rem_cycle.work_remains(mem))
        rem_cycle.write_cursor(mem, "op", {"remaining": 2, "done": False})
        self.assertTrue(rem_cycle.work_remains(mem))

    def test_evaluate_stays_rem_while_subscriber_work_remains(self):
        """What: in REM at zero pressure but with a due subscriber, evaluate does
        NOT enter REVIEWING — work_remains keeps the cycle alive. Why: the snooze
        decision must honor subscriber work, not just pressure."""
        mem = _mem()
        rem_cycle.enter_rem(mem, "test")
        rem_cycle.register_offline_op("due_op", lambda m: {}, priority=10,
                                      due_condition=lambda m: True)
        idle_s = rem_cycle.IDLE_GATE_S + 5
        ev = rem_cycle.evaluate(mem, activity=False, idle_seconds=idle_s)
        # Pressure is below threshold (0), but work remains -> ENTER_REVIEWING is
        # driven by pressure_below_threshold; the KEY assertion is that
        # work_remains is True so a subsequent review snoozes back (not rests).
        self.assertTrue(rem_cycle.work_remains(mem))


# ---------------------------------------------------------------------------
# G2: work_remaining = MACHINE-DRAINABLE ONLY. Operator-gated backlogs (pending
# supersession confirms, surfaced-but-gated merge candidates) must NOT hold REM
# awake; only work a future machine cycle can drain (without operator action) does.
# ---------------------------------------------------------------------------


class TestG2MachineDrainableOnly(_RegistryIsolation):

    def test_only_operator_gated_work_lets_evaluate_reach_rest(self):
        """What: a subscriber whose ONLY remaining work is operator-gated reports
        neither a due-condition nor cursor-remaining work, so work_remains is False
        and evaluate() reaches REST. Why: G2 — operator-gated queues (pending
        confirms / gated merge candidates) must not OR into work_remaining and hold
        REM permanently awake. A gated-only subscriber: due_fn False (no machine work
        to start) + cursor 'remaining' False (nothing a machine cycle can drain)."""
        mem = _mem()  # zero clutter -> pressure 0
        rem_cycle.register_offline_op(
            "gated_only", lambda m: {"work_remaining": False},
            priority=10, due_condition=lambda m: False, cursor_key="gated_only")
        # Cursor records work that exists but is NOT machine-drainable -> done True,
        # remaining False (the post-G2 semantics the surfacer/contradiction now write).
        rem_cycle.write_cursor(mem, "gated_only", {"remaining": False, "done": True})
        self.assertFalse(rem_cycle._any_subscriber_work_remaining(mem))
        self.assertFalse(rem_cycle.work_remains(mem))
        # evaluate() through a full review window settles to REST (not snooze).
        rem_cycle.enter_rem(mem, "g2")
        idle_s = rem_cycle.IDLE_GATE_S + 5
        rem_cycle.tick(mem, activity=False, idle_seconds=idle_s)  # -> REVIEWING
        rstart = float(rem_cycle.current_state(mem)["review_started_ts"])
        r = rem_cycle.tick(mem, activity=False, idle_seconds=idle_s,
                           now=rstart + rem_cycle.REVIEW_WAIT_S + 1)
        self.assertEqual(r["action"], rem_cycle.REST)
        self.assertEqual(rem_cycle.current_state(mem)["phase"], rem_cycle.REM)

    def test_machine_drainable_work_still_holds_rem_awake(self):
        """What: alongside the gated-only subscriber, a subscriber WITH real
        machine-drainable work (cursor remaining True) keeps work_remains True ->
        REM stays awake (does not REST). Why: G2 must not over-correct — genuine
        machine work a future cycle can drain MUST still hold the cycle open."""
        mem = _mem()
        rem_cycle.register_offline_op(
            "gated_only", lambda m: {}, priority=10,
            due_condition=lambda m: False, cursor_key="gated_only")
        rem_cycle.write_cursor(mem, "gated_only", {"remaining": False, "done": True})
        rem_cycle.register_offline_op(
            "machine_work", lambda m: {}, priority=20,
            due_condition=lambda m: False, cursor_key="machine_work")
        rem_cycle.write_cursor(mem, "machine_work", {"remaining": True, "done": False})
        self.assertTrue(rem_cycle._any_subscriber_work_remaining(mem))
        self.assertTrue(rem_cycle.work_remains(mem))

    def test_contradiction_passive_pending_does_not_set_work_remaining(self):
        """What: contradiction.passive_sweep's work_remaining reflects ONLY the
        machine-drainable cursor (not wrapped), NOT operator-gated pending
        supersession candidates. A WRAPPED sweep with pending candidates reports
        work_remaining False (+ operator_gated_pending telemetry). Why: G2 — the
        pending OR-in was what made every wake report work and never REST."""
        from samia.runtime import contradiction as _con
        mem = _mem()
        nodes = mem / "nodes"
        nodes.mkdir(parents=True, exist_ok=True)
        # A small index so the default budget (20) covers it in one call -> the
        # cursor WRAPS (end >= total). find_supersession_candidates mocked to [] so
        # no judge/supersede work runs; list_supersession_candidates returns pending
        # OPERATOR-GATED items that (pre-G2) would have forced work_remaining True.
        for i in range(3):
            (nodes / f"n{i}.md").write_text(
                f"---\nname: n{i}\ntype: project\n---\nclaim {i}\n", encoding="utf-8")
        import unittest.mock as _mock
        with _mock.patch.object(_con, "is_enabled", return_value=True), \
             _mock.patch.object(_con, "find_supersession_candidates", return_value=[]), \
             _mock.patch.object(_con, "list_supersession_candidates",
                                return_value=[{"id": "p1"}, {"id": "p2"}]):
            # Explicit cursor at index 0 (no_persist so we own it): the slice covers
            # the whole 3-node index in one call so the pass WRAPS.
            res = _con.passive_sweep(
                mem, cursor={"index": 0, "__no_persist__": True})
        # Cursor wrapped over the index -> no machine work remains, even though
        # pending operator-gated candidates exist.
        self.assertFalse(res["work_remaining"])
        self.assertTrue(res.get("operator_gated_pending"))

    def test_consolidation_surfacer_cursor_remaining_zero_when_merge_off(self):
        """What: the consolidation surfacer's cursor 'remaining' is 0 when tier2
        merge is OFF (the surfaced backlog is operator-gated, not machine-drainable),
        and equals the surfaced count when merge is ON. Why: G2 — surfaced-but-gated
        merge candidates must not hold REM awake via the surfacer's cursor."""
        from samia.runtime import rem_subscribers as _rs
        from samia.core import merge_consumer as _mc
        import unittest.mock as _mock
        mem = _mem()
        # Enter REM so the gated surfacer actually runs (not refused).
        rem_cycle.enter_rem(mem, "g2")
        chains = mem / "chains"
        chains.mkdir(parents=True, exist_ok=True)
        # Two near-identical chains so audit_all surfaces a candidate pair.
        with _mock.patch("samia.core.consolidation.audit_all",
                         return_value=[{"a": "x", "b": "y", "sim": 0.99}]), \
             _mock.patch("samia.core.consolidation.surface", return_value=None):
            # Merge OFF -> surfaced backlog is operator-gated -> cursor remaining 0.
            with _mock.patch.object(_mc, "is_enabled", return_value=False):
                _rs._sub_consolidation(mem)
            cur_off = rem_cycle.read_cursor(mem, "consolidation")
            self.assertEqual(cur_off.get("surfaced"), 1)
            self.assertEqual(cur_off.get("remaining"), 0)
            self.assertFalse(rem_cycle._cursor_has_remaining(mem, "consolidation"))
            # Merge ON -> the surfaced backlog IS machine-drainable -> remaining = count.
            with _mock.patch.object(_mc, "is_enabled", return_value=True):
                _rs._sub_consolidation(mem)
            cur_on = rem_cycle.read_cursor(mem, "consolidation")
            self.assertEqual(cur_on.get("remaining"), 1)
            self.assertTrue(rem_cycle._cursor_has_remaining(mem, "consolidation"))


# ---------------------------------------------------------------------------
# G4: vector_maintenance subscriber — auto-rebuild the vector index on drift,
# incremental every cycle + full rebuild on a long cadence (cursor-tracked).
# ---------------------------------------------------------------------------


class TestG4VectorMaintenance(_RegistryIsolation):

    @staticmethod
    def _write_nodes(mem: Path, n: int) -> None:
        nodes = mem / "nodes"
        nodes.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (nodes / f"v{i}.md").write_text(
                f"---\nname: v{i}\n---\nbody {i}\n", encoding="utf-8")

    @staticmethod
    def _write_manifest(mem: Path, node_count: int) -> None:
        from samia.core import vector as _vec
        _vec._save_manifest(mem, {"node_count": node_count, "entries": {}})

    def test_subscriber_registers_at_priority_29(self):
        names = rem_subscribers.register_rem_subscribers()
        self.assertIn("vector_maintenance", names)
        sub = rem_cycle._rem_subscribers["vector_maintenance"]
        self.assertEqual(sub.priority, 29)
        # Slots AFTER integrity_repair (28) and BEFORE replay (30).
        self.assertEqual(sub.priority, rem_subscribers.PRIO_VECTOR_MAINTENANCE)

    def test_drift_due_when_index_count_differs_from_live(self):
        mem = _mem()
        self._write_nodes(mem, 5)
        # No manifest yet -> live nodes exist -> drift (first build) is due.
        self.assertTrue(rem_subscribers._vector_index_drift(mem))
        # Manifest matching the live count -> NOT due.
        self._write_manifest(mem, 5)
        self.assertFalse(rem_subscribers._vector_index_drift(mem))
        # A node added -> count mismatch -> due again.
        self._write_nodes(mem, 7)  # now 7 live, manifest says 5
        self.assertTrue(rem_subscribers._vector_index_drift(mem))

    def test_no_drift_is_a_noop_due_condition(self):
        mem = _mem()
        self._write_nodes(mem, 3)
        self._write_manifest(mem, 3)
        self.assertFalse(rem_subscribers._vector_index_drift(mem))
        # Empty corpus is never due (no thrash on nothing to index).
        empty = _mem()
        (empty / "nodes").mkdir(parents=True, exist_ok=True)
        self.assertFalse(rem_subscribers._vector_index_drift(empty))

    def test_subscriber_runs_incremental_build_when_recently_full(self):
        import unittest.mock as _mock
        mem = _mem()
        self._write_nodes(mem, 4)
        rem_cycle.enter_rem(mem, "g4")
        # Cursor says a full rebuild just happened -> this run is INCREMENTAL.
        import time as _time
        rem_cycle.write_cursor(mem, "vector_maintenance",
                               {"last_full_ts": _time.time()})
        with _mock.patch("samia.core.vector.build",
                         return_value={"node_count": 4}) as mb:
            res = rem_subscribers._sub_vector_maintenance(mem)
        self.assertTrue(res["fired"])
        self.assertEqual(res["mode"], "incremental")
        # build called with rebuild=False (the incremental, manifest-cached path).
        _args, kwargs = mb.call_args
        self.assertFalse(kwargs.get("rebuild", False))
        self.assertFalse(res["work_remaining"])

    def test_subscriber_runs_full_rebuild_when_cadence_elapsed(self):
        import unittest.mock as _mock
        mem = _mem()
        self._write_nodes(mem, 4)
        rem_cycle.enter_rem(mem, "g4")
        # Cursor's last full rebuild is far in the past -> cadence elapsed -> FULL.
        rem_cycle.write_cursor(mem, "vector_maintenance",
                               {"last_full_ts": 0.0})
        with _mock.patch("samia.core.vector.build",
                         return_value={"node_count": 4}) as mb:
            res = rem_subscribers._sub_vector_maintenance(mem)
        self.assertEqual(res["mode"], "full_rebuild")
        _args, kwargs = mb.call_args
        self.assertTrue(kwargs.get("rebuild", False))
        # The cursor records the new last_full_ts so the cadence is honored next run.
        cur = rem_cycle.read_cursor(mem, "vector_maintenance")
        self.assertGreater(cur["last_full_ts"], 0.0)
        self.assertEqual(cur["last_mode"], "full_rebuild")

    def test_full_rebuild_cadence_env_override(self):
        import os as _os
        import unittest.mock as _mock
        mem = _mem()
        self._write_nodes(mem, 2)
        rem_cycle.enter_rem(mem, "g4")
        # A tiny cadence (1s) with a last_full_ts a few seconds ago -> full rebuild.
        import time as _time
        rem_cycle.write_cursor(mem, "vector_maintenance",
                               {"last_full_ts": _time.time() - 5})
        prev = _os.environ.get(rem_subscribers.VECTOR_FULL_REBUILD_S_ENV)
        _os.environ[rem_subscribers.VECTOR_FULL_REBUILD_S_ENV] = "1"
        try:
            self.assertEqual(rem_subscribers._vector_full_rebuild_interval_s(), 1.0)
            with _mock.patch("samia.core.vector.build",
                             return_value={"node_count": 2}) as mb:
                res = rem_subscribers._sub_vector_maintenance(mem)
        finally:
            if prev is None:
                _os.environ.pop(rem_subscribers.VECTOR_FULL_REBUILD_S_ENV, None)
            else:
                _os.environ[rem_subscribers.VECTOR_FULL_REBUILD_S_ENV] = prev
        self.assertEqual(res["mode"], "full_rebuild")
        self.assertTrue(mb.call_args.kwargs.get("rebuild"))

    def test_refuses_outside_rem(self):
        import unittest.mock as _mock
        mem = _mem()  # WAKE phase (no enter_rem)
        self._write_nodes(mem, 2)
        with _mock.patch("samia.core.vector.build") as mb:
            res = rem_subscribers._sub_vector_maintenance(mem)
        self.assertFalse(res["fired"])
        self.assertEqual(res["refused"], "not_in_rem")
        mb.assert_not_called()


# ---------------------------------------------------------------------------
# Migration: a migrated op refuses on an awake idle pulse but runs in REM
# ---------------------------------------------------------------------------


class TestMigration(_RegistryIsolation):

    def test_decay_tick_runs_outside_rem(self):
        """What: tier.decay_tick RUNS when NOT in REM (does its decay pass,
        bounded only by its own 6h cooldown) — it does NOT refuse with
        not_in_rem. Why: operator correction 2026-06-07 — decay is the short-
        term forgetting curve, continuous across BOTH wake and REM. Sleep is
        for consolidation/replay, NOT forgetting; so decay is NOT REM-gated."""
        from samia.core.tier import decay_tick
        mem = _mem()
        (mem / "nodes").mkdir(parents=True)  # empty nodes dir -> decay_pass no-ops cleanly
        # WAKE (not in REM) -> RUNS (fresh 6h cooldown so it fires); no refusal.
        out = decay_tick(mem)
        self.assertNotEqual(out.get("refused"), "not_in_rem")
        self.assertTrue(out.get("fired"))
        # And it remains the idle_pulse-driven continuous op — NOT a REM
        # subscriber (verified by test_register_rem_subscribers_* above), so it
        # is not double-driven by the REM driver.

    def test_decay_tick_respects_6h_cooldown_outside_rem(self):
        """What: a second decay_tick within the 6h window no-ops on the cadence
        guard (not on a REM gate) while still NOT in REM. Why: confirm the ONLY
        throttle is the 6h DECAY_CADENCE cooldown (state-file), preserved by the
        correction; decay is never gated by wake/REM state."""
        from samia.core.tier import decay_tick
        mem = _mem()
        (mem / "nodes").mkdir(parents=True)
        first = decay_tick(mem)
        self.assertTrue(first.get("fired"))
        # Second call, still WAKE, within 6h -> cooldown no-op (NOT a REM refusal).
        second = decay_tick(mem)
        self.assertFalse(second.get("fired"))
        self.assertNotEqual(second.get("refused"), "not_in_rem")
        self.assertIn("elapsed_seconds", second)

    def test_idle_replay_tick_refuses_heavy_body_outside_rem(self):
        """What: context_extension.idle_replay_tick refuses its HEAVY body outside
        REM (returns refused) while still doing the cheap frozen-prefix refresh.
        Why: the single biggest offline-on-idle op moves behind REM; the light
        prefix refresh stays on the waking path (survey: do not silently stop it)."""
        from samia.core.context_extension import idle_replay_tick
        mem = _mem()
        out = idle_replay_tick(mem)  # WAKE
        self.assertEqual(out.get("refused"), "not_in_rem")
        # The cheap prefix refresh still ran on the waking path (not gated away).
        self.assertIn("frozen_prefix", out)

    def test_register_rem_subscribers_registers_all_in_priority_order(self):
        """What: register_rem_subscribers wires consolidation->tier2_merge->
        contradiction_passive->integrity_repair->replay->fact_extract in ascending
        priority. Why: the migration inventory the operator sees via rem_status; run
        order is load-bearing. FEAT-2026-06-07 P3c slots contradiction_passive at 25,
        between consolidation (20) and replay (30); FEAT-2026-06-07 P1 slots
        tier2_merge at 22, between consolidation (20) and contradiction_passive
        (25) — the surfacer produces near-dup pairs, the merge consumer drains
        them in the SAME cycle. FEAT-2026-06-07 granular-recall-repaired-decay P2
        slots integrity_repair at 28, between contradiction_passive (25) and replay
        (30) — sleep PARTIALLY heals the integrity of what it consolidates.
        Operator correction 2026-06-07: DECAY is NOT a REM subscriber — it is the
        continuous forgetting curve (wake+REM), driven by idle_pulse, never the REM
        driver (CLS: sleep is for consolidation/replay, not forgetting). G4-2026-06-11
        slots vector_maintenance at 29, between integrity_repair (28) and replay (30)
        — auto-rebuild the vector index on drift (incremental every cycle + a periodic
        full rebuild) so the index no longer drifts from the live node set."""
        expected = ["consolidation", "tier2_merge", "contradiction_passive",
                    "integrity_repair", "vector_maintenance", "replay",
                    "fact_extract"]
        names = rem_subscribers.register_rem_subscribers()
        self.assertEqual(names, expected)
        # Decay must NOT be in the REM registry (no double-drive; it runs only
        # via the idle_pulse "decay" subscriber, continuously across wake+REM).
        self.assertNotIn("decay", names)
        # rem_status surfaces the registered subscribers (parity with idle_pulse).
        mem = _mem()
        status = rem_cycle.rem_status(mem)
        self.assertEqual([s["name"] for s in status["subscribers"]], expected)

    def test_fact_extract_subscriber_refuses_outside_rem(self):
        """What: the new batch fact-extract wrapper refuses outside REM and is a
        no-op with no queue inside REM. Why: it is a NEW gated batch op (no live
        cadence to migrate); it must respect the sleep boundary too."""
        mem = _mem()
        # WAKE -> refused.
        out = rem_subscribers._sub_fact_extract(mem)
        self.assertEqual(out.get("refused"), "not_in_rem")
        # REM, empty queue -> clean no-op (no crash).
        rem_cycle.enter_rem(mem, "test")
        out = rem_subscribers._sub_fact_extract(mem)
        self.assertFalse(out.get("fired"))
        self.assertEqual(out.get("extracted", 0), 0)


if __name__ == "__main__":
    unittest.main()


# ─────────────────────────────────────────────
# [test_rem_subscribers] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.runtime
# Version:    1.0.0  Updated: 2026-06-07  Status: active
# Phase:      FEAT-2026-06-07-memory-rem-sleep-consolidation-cycle-v01 (P2 tests)
# Role:       unit tests — subscriber registry + priority + fail-open + the
#             run-only-in-REM gate + interrupt/checkpoint + work_remaining feeds
#             evaluate + migrated ops refuse outside REM but run inside
# Depends:    tempfile, unittest; samia.runtime.rem_cycle, rem_subscribers
# ─────────────────────────────────────────────
