"""Tests for samia.runtime.maintenanced (the release maintenance daemon).

Layer 1 (Owns / Depends):
    Owns:    unit tests for the minimal shippable maintenance spine —
             (a) the single-instance flock prevents a second start;
             (b) --oneshot exits 0 on a scratch store and writes scheduler state;
             (c) a clean SIGTERM shuts the running daemon down (lock released);
             (d) the scheduler jobs run with every OPTIONAL dep absent (the
                 carved release subset: opencode_drain / compact_index gone);
             (e) ASTHENOS_MODEL_AUTOFETCH=0 means a --oneshot never downloads a
                 model (the network-fetch path is asserted never-called).
    Depends: samia.runtime.maintenanced, samia.runtime.{scheduler,watcher,
             rem_cycle,model_fetch}, tempfile, subprocess, signal, unittest. All
             tests use tempdir memory roots — NEVER the live ~/.local/share
             memory tree, and NEVER a real store.

Layer 2 (What / Why):
    What: validates maintenanced's contract: it is a daemonless-release spine
          (scheduler + watcher + REM driving ONLY), single-instance, signal-clean,
          fail-soft against absent dev-only modules, and never touches the network
          at startup.
    Why:  maintenanced is the ONLY long-lived process the public release ships.
          If the lock is wrong two daemons double-drive REM; if --oneshot is not
          deterministic the cold-metal kit + CI cannot gate on it; if a missing
          optional module crashes a job the release maintenance silently dies; if
          startup downloads a model the "no network at startup" guarantee breaks.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from samia.runtime import maintenanced


def _scratch() -> Path:
    """Fresh tempdir memory root (NEVER the live store), cleaned at process exit.

    What: mkdtemp + an atexit-registered rmtree of every dir handed out. Why:
      mkdtemp does NOT auto-clean, so each call left a maint_test_* dir in /tmp
      and tripped the cold-metal zero-leftover hygiene gate. One atexit
      registration covers every `mem = _scratch()` call site and any test order
      without editing each caller.
    """
    md = Path(tempfile.mkdtemp(prefix="maint_test_"))
    atexit.register(shutil.rmtree, md, ignore_errors=True)
    return md


def _run_module(args: list[str], env_extra: dict[str, str] | None = None,
                timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run ``python -m samia.runtime.maintenanced`` in a child process.

    Inherits the current interpreter + a PYTHONPATH that makes ``samia``
    importable (the package root is three parents up from this file:
    .../<root>/samia/runtime/test_maintenanced.py -> <root>).
    """
    pkg_root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    # Belt-and-suspenders: a child must never reach the network or a real store.
    env.setdefault("ASTHENOS_MODEL_AUTOFETCH", "0")
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, "-m", "samia.runtime.maintenanced", *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=timeout)


# ---------------------------------------------------------------------------
# --oneshot: exits 0 + writes scheduler state
# ---------------------------------------------------------------------------


class TestOneshot(unittest.TestCase):

    def test_oneshot_exits_zero_and_writes_scheduler_state(self):
        """What: --oneshot on a scratch store returns 0 and persists scheduler
        state. Why: the cold-metal kit + CI gate on this single deterministic
        pass; it must complete cleanly and leave the state file."""
        mem = _scratch()
        proc = _run_module(["--oneshot", "--memory-dir", str(mem)],
                           env_extra={"ASTHENOS_MEMORY_DIR": str(mem)})
        self.assertEqual(proc.returncode, 0,
                         msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
        state = mem / ".runtime" / "scheduler_state.json"
        self.assertTrue(state.exists(),
                        f"scheduler_state.json not written; stdout={proc.stdout}")
        # The REM state machine also leaves a persisted state file after a tick.
        self.assertTrue((mem / "biomimetic" / "rem_state.json").exists(),
                        "rem_state.json not written by the REM tick")

    def test_oneshot_in_process_drives_all_three(self):
        """What: run_oneshot() in-process drives scheduler + watcher + REM and
        returns 0. Why: proves the three seams import and run together on a
        scratch store without spawning long-lived threads."""
        mem = _scratch()
        d = maintenanced.MaintenanceDaemon(mem)
        # acquire the lock the way main() does for the oneshot path
        self.assertTrue(d.acquire_lock())
        try:
            self.assertEqual(d.run_oneshot(), 0)
        finally:
            d.release_lock()
        self.assertTrue((mem / ".runtime" / "scheduler_state.json").exists())


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


class TestSingleInstanceLock(unittest.TestCase):

    def test_second_acquire_is_refused(self):
        """What: a second MaintenanceDaemon on the SAME store cannot take the
        lock while the first holds it. Why: two daemons would double-drive REM
        and race the scheduler state file."""
        mem = _scratch()
        first = maintenanced.MaintenanceDaemon(mem)
        second = maintenanced.MaintenanceDaemon(mem)
        self.assertTrue(first.acquire_lock())
        try:
            self.assertFalse(second.acquire_lock(),
                             "second daemon took the lock the first holds")
        finally:
            first.release_lock()
        # After release the lock is free again.
        self.assertTrue(second.acquire_lock())
        second.release_lock()

    def test_oneshot_refuses_while_held(self):
        """What: main(--oneshot) returns 1 (not 0) when another instance holds
        the lock. Why: even the single-pass path must honor single-instance."""
        mem = _scratch()
        holder = maintenanced.MaintenanceDaemon(mem)
        self.assertTrue(holder.acquire_lock())
        try:
            rc = maintenanced.main(["--oneshot", "--memory-dir", str(mem)])
            self.assertEqual(rc, 1)
        finally:
            holder.release_lock()


# ---------------------------------------------------------------------------
# Clean SIGTERM shutdown of a running daemon
# ---------------------------------------------------------------------------


class TestSignalShutdown(unittest.TestCase):

    def test_sigterm_shuts_down_cleanly(self):
        """What: a running daemon (child process) exits 0 on SIGTERM and frees
        the lock. Why: the release runs under systemd; SIGTERM is the normal
        stop and must be graceful (threads joined, lock released)."""
        mem = _scratch()
        pkg_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
        env["ASTHENOS_MODEL_AUTOFETCH"] = "0"
        env["ASTHENOS_MEMORY_DIR"] = str(mem)
        # Short interval so the tick loop is clearly alive before we signal.
        proc = subprocess.Popen(
            [sys.executable, "-m", "samia.runtime.maintenanced",
             "--memory-dir", str(mem), "--interval", "1"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            # Wait until the lockfile exists (daemon has started + locked).
            lock = mem / ".runtime" / "maintenanced.lock"
            deadline = time.time() + 20
            while not lock.exists() and time.time() < deadline:
                if proc.poll() is not None:
                    out, err = proc.communicate()
                    self.fail(f"daemon exited early rc={proc.returncode}\n"
                              f"stdout={out}\nstderr={err}")
                time.sleep(0.1)
            self.assertTrue(lock.exists(), "daemon never created its lockfile")
            # Let it run at least one tick.
            time.sleep(1.5)
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=20)
            self.assertEqual(proc.returncode, 0,
                             msg=f"stdout={out}\nstderr={err}")
            self.assertIn("exited cleanly", out)
            # Lock released (file removed on clean stop).
            self.assertFalse(lock.exists(), "lockfile not removed on clean stop")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


# ---------------------------------------------------------------------------
# Jobs run with all optional deps absent (the carved release subset)
# ---------------------------------------------------------------------------


class TestOptionalDepsAbsent(unittest.TestCase):
    """Simulate the carved release subset where opencode_drain + compact_index
    do NOT ship. The shipped seams must SKIP (not crash) when they are absent."""

    def setUp(self):
        # Hide the two dev-only modules so an import of them fails, exactly as in
        # the staged subset. We re-resolve the watcher's compact_index seam after.
        import importlib
        self._saved = {}
        for m in ("samia.core.opencode_drain", "samia.core.compact_index"):
            self._saved[m] = sys.modules.pop(m, None)
        # Block re-import by installing a finder that raises for these names.
        self._blocked = {"samia.core.opencode_drain", "samia.core.compact_index"}

        class _Blocker:
            def find_module(_self, name, path=None):
                return _self if name in self._blocked_ref else None

            def load_module(_self, name):
                raise ImportError(f"blocked (carved out): {name}")

            # importlib MetaPathFinder protocol (3.4+)
            def find_spec(_self, name, path=None, target=None):
                if name in self._blocked_ref:
                    raise ImportError(f"blocked (carved out): {name}")
                return None

        blocker = _Blocker()
        blocker._blocked_ref = self._blocked
        self._blocker = blocker
        # Invalidate BEFORE arming the blocker: invalidate_caches() can itself
        # trigger imports (editable-install finders re-import their targets),
        # and a blocked name raised mid-invalidation errors setUp in venvs.
        importlib.invalidate_caches()
        sys.meta_path.insert(0, blocker)

    def tearDown(self):
        try:
            sys.meta_path.remove(self._blocker)
        except ValueError:
            pass
        for m, mod in self._saved.items():
            if mod is not None:
                sys.modules[m] = mod
        import importlib
        importlib.invalidate_caches()

    def test_opencode_drain_resolves_none_when_absent(self):
        """What: scheduler's opencode_drain resolver returns None when the module
        is carved out. Why: the job then runs as a no-op stub, never crashes."""
        from samia.runtime import scheduler
        self.assertIsNone(scheduler._resolve_opencode_drain())

    def test_compact_index_skip_when_absent(self):
        """What: watcher's memory_md_regen action SKIPS (one debug log, no
        'failed' line) when compact_index is carved out. Why: a debounced regen
        for an unavailable backend is a no-op in the release, not an error."""
        from samia.runtime import watcher
        self.assertIsNone(watcher._resolve_compact_regenerate())
        mem = _scratch()
        (mem / "MEMORY.md").write_text("# x\n")
        logs: list[str] = []

        def logf(msg, *a):
            logs.append((msg % a) if a else msg)

        # Patch the resolved fn to None (simulating the absent build) and reset
        # the one-time skip latch so the skip-log is observable.
        watcher._COMPACT_REGENERATE_FN = None
        watcher._compact_skip_logged = False
        watcher._action_queue["memory_md_regen"] = time.monotonic() - 1
        watcher._dispatch_actions(mem, logf)
        self.assertFalse(any("failed" in line for line in logs),
                         f"a carved-out backend must not log a failure: {logs}")
        self.assertTrue(any("unavailable" in line for line in logs),
                        f"expected a skip log: {logs}")

    def test_oneshot_clean_with_deps_absent(self):
        """What: a full --oneshot subprocess succeeds even with the dev-only
        modules carved out of the importable path. Why: the release runs WITHOUT
        them; --oneshot must still exit 0. Uses a sitecustomize shim to block the
        two modules in the child so this matches the staged subset exactly."""
        mem = _scratch()
        # Write a small child driver that blocks the two modules then runs main.
        driver = mem / "_drive_absent.py"
        driver.write_text(
            "import sys\n"
            "class B:\n"
            "    names={'samia.core.opencode_drain','samia.core.compact_index'}\n"
            "    def find_spec(self,name,path=None,target=None):\n"
            "        if name in self.names:\n"
            "            raise ImportError('carved: '+name)\n"
            "        return None\n"
            "sys.meta_path.insert(0,B())\n"
            "from samia.runtime.maintenanced import main\n"
            "sys.exit(main())\n"
        )
        pkg_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
        env["ASTHENOS_MODEL_AUTOFETCH"] = "0"
        env["ASTHENOS_MEMORY_DIR"] = str(mem)
        proc = subprocess.run(
            [sys.executable, str(driver), "--oneshot", "--memory-dir", str(mem)],
            env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0,
                         msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")


# ---------------------------------------------------------------------------
# autofetch-off => no network (model_fetch download path never reached)
# ---------------------------------------------------------------------------


class TestNoNetworkAtStartup(unittest.TestCase):

    def test_oneshot_never_downloads_a_model(self):
        """What: a --oneshot pass on a scratch store never calls the model
        download path. Why: ASTHENOS_MODEL_AUTOFETCH=0 must mean zero downloads;
        more strongly, a bare maintenance pass loads NO model at all (LLM arms
        are lazy). We patch the only network sink (_download) and assert it is
        never invoked."""
        from samia.runtime import model_fetch

        calls: list[str] = []
        orig = model_fetch._download

        def _spy(entry, dest):
            calls.append(entry.get("filename", "?"))
            raise AssertionError("model_fetch._download must not run at startup")

        model_fetch._download = _spy  # type: ignore[assignment]
        try:
            mem = _scratch()
            with self._autofetch_off():
                d = maintenanced.MaintenanceDaemon(mem)
                self.assertTrue(d.acquire_lock())
                try:
                    rc = d.run_oneshot()
                finally:
                    d.release_lock()
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [],
                             "a model download was attempted during --oneshot")
        finally:
            model_fetch._download = orig  # type: ignore[assignment]

    def _autofetch_off(self):
        """Context manager forcing ASTHENOS_MODEL_AUTOFETCH=0 for the body."""
        import contextlib

        @contextlib.contextmanager
        def _cm():
            prev = os.environ.get("ASTHENOS_MODEL_AUTOFETCH")
            os.environ["ASTHENOS_MODEL_AUTOFETCH"] = "0"
            try:
                yield
            finally:
                if prev is None:
                    os.environ.pop("ASTHENOS_MODEL_AUTOFETCH", None)
                else:
                    os.environ["ASTHENOS_MODEL_AUTOFETCH"] = prev

        return _cm()


if __name__ == "__main__":
    unittest.main()


# ─────────────────────────────────────────────
# [test_maintenanced] — File Metadata
# Author:     code_warrior (CLI steward)  |  Project: Asthenosphere samia.runtime
# Version:    1.0.0  Updated: 2026-06-12  Status: active
# Role:       contract tests for the release maintenance daemon (single-instance
#             lock, --oneshot exit-0 + state write, clean SIGTERM, jobs run with
#             optional deps absent, autofetch-off => no network).
# Depends:    os, signal, subprocess, sys, tempfile, time, unittest, pathlib;
#             samia.runtime.{maintenanced,scheduler,watcher,rem_cycle,model_fetch}
# Note:       scratch tempdir stores ONLY — never the live memory tree, never a
#             real store, never the network.
# ─────────────────────────────────────────────
