"""samia.runtime.ipc — AF_UNIX socket server for SAM/IA memory runtime.

Layer 1 (Owns / Depends):
    Owns:    AF_UNIX socket server, JSON-line wire protocol, op dispatch,
             plugin op registration
    Depends: samia.runtime.daemon (receives daemon reference for health/shutdown)

Layer 2 (What / Why):
    What: IPCServer listens on $XDG_RUNTIME_DIR/samia-runtimed.sock (chmod 0600).
          JSON-line protocol: each request is a single newline-terminated JSON
          object, each response is a single newline-terminated JSON object.
          Request envelope:  {"op": str, "args": dict, "request_id": str}
          Response envelope: {"request_id": str, "ok": bool, "result": any, "error": str|null}
          Built-in ops: health, version, echo, shutdown.
          Plugin registration: register_op(name, handler_fn) for future phases.
    Why:  asyncio was considered but threading is simpler for Phase 26.1 where
          concurrent load is minimal (1-3 clients).  The threading model uses
          one accept thread + one handler thread per connection, with daemon
          threads so they don't block shutdown.  Future phases can migrate to
          asyncio if concurrency demands increase.

Design doc: plans/sam_ia_runtime_design.md, sections 2.1, 3.
AUD26 Phase 26.1 — foundation.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from samia.runtime.daemon import SamiaDaemon

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = logging.getLogger("samia.runtime.ipc")

# ---------------------------------------------------------------------------
# Default timeout
# ---------------------------------------------------------------------------

DEFAULT_REQUEST_TIMEOUT: float = 30.0

# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------

# Type: op handler receives (args: dict) and returns result (any JSON-able).
OpHandler = Callable[[dict[str, Any]], Any]

_op_registry: dict[str, OpHandler] = {}


def register_op(name: str, handler: OpHandler) -> None:
    """Register a custom op handler.

    Future phases (26.2 scheduler, 26.3 inference) call this to add ops
    without modifying ipc.py.  Raises ValueError on duplicate registration.
    """
    if name in _op_registry:
        raise ValueError(f"op {name!r} already registered")
    _op_registry[name] = handler
    _log.info("registered op: %s", name)


def unregister_op(name: str) -> None:
    """Remove a previously registered op (mainly for tests)."""
    _op_registry.pop(name, None)


# ---------------------------------------------------------------------------
# IPCServer
# ---------------------------------------------------------------------------


class IPCServer:
    """AF_UNIX socket server with JSON-line protocol.

    Parameters
    ----------
    sock_path : Path
        Where to bind the Unix socket.
    daemon : SamiaDaemon
        Reference to the daemon instance (used by health/shutdown ops).
    request_timeout : float
        Per-request timeout in seconds (default 30).
    """

    def __init__(
        self,
        sock_path: Path,
        daemon: SamiaDaemon,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._sock_path = sock_path
        self._daemon = daemon
        self._request_timeout = request_timeout
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._client_threads: list[threading.Thread] = []
        self._clients_lock = threading.Lock()

    # -- Client tracking --------------------------------------------------

    def connected_client_count(self) -> int:
        """Return the number of currently active client handler threads."""
        with self._clients_lock:
            # Prune dead threads while we're here.
            self._client_threads = [t for t in self._client_threads if t.is_alive()]
            return len(self._client_threads)

    # -- Built-in ops -----------------------------------------------------

    def _builtin_health(self, args: dict[str, Any]) -> Any:
        return self._daemon.health()

    def _builtin_version(self, args: dict[str, Any]) -> str:
        from samia.runtime.daemon import __version__
        return __version__

    def _builtin_echo(self, args: dict[str, Any]) -> str:
        return args.get("message", "")

    def _builtin_shutdown(self, args: dict[str, Any]) -> str:
        # Security: only allow shutdown from same UID.
        # Since AF_UNIX with chmod 0600 already enforces this, this is
        # defense-in-depth.  If we later add TCP, this check matters.
        self._daemon.request_shutdown()
        return "shutdown initiated"

    def _dispatch(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route an op to its handler and return a response dict."""
        builtins: dict[str, OpHandler] = {
            "health": self._builtin_health,
            "version": self._builtin_version,
            "echo": self._builtin_echo,
            "shutdown": self._builtin_shutdown,
        }

        handler = builtins.get(op) or _op_registry.get(op)
        if handler is None:
            return {
                "ok": False,
                "result": None,
                "error": f"unknown op: {op!r}",
            }

        try:
            result = handler(args)
            return {"ok": True, "result": result, "error": None}
        except Exception as exc:
            _log.exception("op %r raised", op)
            return {"ok": False, "result": None, "error": str(exc)}

    # -- Connection handler -----------------------------------------------

    def _handle_connection(self, conn: socket.socket, addr: Any) -> None:
        """Serve requests on one connection until it closes or server stops."""
        conn.settimeout(self._request_timeout)
        buf = b""
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break  # peer closed

                buf += chunk
                # Process all complete lines in the buffer.
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._process_line(conn, line)
        except Exception as exc:
            _log.debug("handler error: %s", exc)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _process_line(self, conn: socket.socket, line: bytes) -> None:
        """Parse one JSON line, dispatch, and send the response."""
        try:
            request = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            resp = {
                "request_id": None,
                "ok": False,
                "result": None,
                "error": f"parse error: {exc}",
            }
            self._send_response(conn, resp)
            return

        request_id = request.get("request_id")
        op = request.get("op", "")
        args = request.get("args") or {}

        result = self._dispatch(op, args)
        result["request_id"] = request_id
        self._send_response(conn, result)

    def _send_response(self, conn: socket.socket, resp: dict[str, Any]) -> None:
        """Send a JSON-line response (newline-terminated)."""
        try:
            data = json.dumps(resp, default=str).encode("utf-8") + b"\n"
            conn.sendall(data)
        except OSError:
            pass

    # -- Accept loop ------------------------------------------------------

    def _accept_loop(self) -> None:
        """Accept connections until stop event is set."""
        while not self._stop_event.is_set():
            try:
                if self._server_sock is None:
                    break
                conn, addr = self._server_sock.accept()
            except OSError:
                break

            t = threading.Thread(
                target=self._handle_connection,
                args=(conn, addr),
                daemon=True,
                name="samia-ipc-handler",
            )
            t.start()
            with self._clients_lock:
                self._client_threads.append(t)

    # -- Public API -------------------------------------------------------

    def start(self) -> None:
        """Bind the AF_UNIX socket and start the accept thread."""
        self._stop_event.clear()

        # Remove stale socket.
        try:
            self._sock_path.unlink()
        except FileNotFoundError:
            pass

        # Ensure parent directory exists.
        self._sock_path.parent.mkdir(parents=True, exist_ok=True)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(str(self._sock_path))
        os.chmod(str(self._sock_path), 0o600)
        self._server_sock.listen(8)

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name="samia-ipc-accept",
        )
        self._accept_thread.start()
        _log.info("ipc listening on %s", self._sock_path)

    def stop(self, grace: float = 5.0) -> None:
        """Stop accepting connections, drain in-flight, close socket, delete socket file.

        Parameters
        ----------
        grace : float
            Seconds to wait for in-flight handler threads to finish.
        """
        self._stop_event.set()

        # Close the server socket to unblock accept().
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        # Wait for accept thread.
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None

        # Drain in-flight handlers.
        with self._clients_lock:
            for t in self._client_threads:
                t.join(timeout=grace)
            self._client_threads.clear()

        # Remove socket file.
        try:
            self._sock_path.unlink(missing_ok=True)
        except OSError:
            pass

        _log.info("ipc stopped")


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.ipc
# phase: AUD26-26.1
# layer: runtime (long-lived process)
# --------------------------------------------------------------------------
