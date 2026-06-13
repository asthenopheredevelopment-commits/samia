# idle_pulse daemon-resident tick loop — rollout

**Proposal:** `FEAT-2026-06-02-idle-pulse-daemon-resident-tick-loop-v01` (approved)
**Bug:** `bug_idle_pulse_hook_python_swarm` — fix #2 (fix #1 = flock + 30s guard, still live).

## What landed (additive, no coverage gap)

- **`samia/runtime/idle_pulse/__init__.py`** — subscriber registry (the 6 ticks),
  resident embedding model (loaded once via `vector._ensure_model()`),
  self-scheduled servicing loop (`IDLE_PULSE_LOOP_SECONDS`, default 30s), a
  coalescing `idle_pulse_nudge` op, and `idle_pulse_status`.
- **`daemon.py`** — `register_ops()` + `start_idle_pulse_loop()` in `run()`;
  `stop_idle_pulse_loop()` on shutdown.
- **`test_idle_pulse.py`** — 10 tests (registry / cadence / coalescing / fail-open
  isolation / op shapes). All green, no model load.
- **`hook_idle_pulse.sh.nudge`** — STAGED cheap nudge sender (mirrors
  `hook_heartbeat_tick.sh`). **Not active.**

The currently-live `hook_idle_pulse.sh` (fix #1: flock + 30s min-interval) is
**unchanged** and keeps covering maintenance until you complete Phase 5 below.

## Phase 5 — operator steps (gated on a daemon restart)

The running daemon (PID was ~3360) predates this code, so it does not yet have
the `idle_pulse_nudge` op. Do NOT swap the hook before restarting, or nudges
would hit a daemon that can't answer them.

1. **Restart the daemon** so it loads the new subsystem:
   `systemctl --user restart asthenos-runtime.service`
   (or however the daemon is managed — it runs `python3 -m samia.runtime.daemon`).
2. **Verify** the loop is up and the model is resident:
   `python3 -m samia.runtime.cli idle_pulse_status` (or via your usual IPC client)
   → expect `loop_running: true`, `model_resident: true`, 6 subscribers, and
   `last_run_age_s` advancing.
   Also confirm the daemon RSS rose ~880 MB once and is flat (resident model).
3. **Load test** (the bug's acceptance): run a multi-agent workflow and confirm
   **zero `python3 -` idle-pulse workers** spawn and RAM/swap stay stable. The
   `.hooks.log` should no longer emit per-call model loads.
4. **Switch the hook** to the cheap nudge sender (only after 1–3 pass):
   ```
   cd <tools>
   cp hook_idle_pulse.sh hook_idle_pulse.sh.guard-bak   # preserve fix #1
   cp hook_idle_pulse.sh.nudge hook_idle_pulse.sh
   ```
5. **Confirm** under load again: no python-fork-per-call; ticks still fire on
   cadence via the in-daemon loop (the nudge only adds event-freshness).

**Rollback:** `cp hook_idle_pulse.sh.guard-bak hook_idle_pulse.sh` restores fix #1.

## Tunables (env, optional)

- `IDLE_PULSE_LOOP_SECONDS` (30) — self-schedule interval.
- `IDLE_PULSE_GATE_CADENCE` (900) — gate_tick registry cadence (Q4 conservative
  15min; bounds gate regardless of per-run cost, so `gates.py` is untouched).
- `IDLE_PULSE_{AUDITOR,DOCS,DECAY,SUBAGENT_CLEANUP}_CADENCE` — per-tick cadences.

## Deferred (follow-ups, not blocking)

- **Phase 3** — explicit daemon-down liveness alert via availability/observer
  (Q5c). Ticks already pause idempotently when the daemon is down; the existing
  availability watchdog already probes daemon liveness, so this is incremental.
- **gate_tick measurement** — the 15min registry cadence is the safe default;
  measure the real per-run cost post-restart and tune `IDLE_PULSE_GATE_CADENCE`
  if it turns out cheap.
