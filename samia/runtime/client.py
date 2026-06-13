"""samia.runtime.client — thin client library for talking to samia-runtimed.

Layer 1 (Owns / Depends):
    Owns:    SamiaClient class (connect, send, receive, convenience methods)
    Depends: nothing — pure stdlib (socket, json, uuid, threading)

Layer 2 (What / Why):
    What: SamiaClient connects to the daemon's AF_UNIX socket and speaks
          the JSON-line protocol.  Public methods: health(), version(),
          echo(message), shutdown(), call(op, **args).  Auto-reconnect on
          broken socket (single retry, then raise).  Context-manager support.
          Default timeout 30s, configurable.
    Why:  Every consumer of the daemon (MCP server, perception bridge, CLI,
          future BBQ recall path) needs a clean way to call ops.  This is
          the single client implementation they all use.

Wire protocol (matches samia.runtime.ipc):
    Request:  {"op": str, "args": dict, "request_id": str} + newline
    Response: {"request_id": str, "ok": bool, "result": any, "error": str|null} + newline

Design doc: plans/sam_ia_runtime_design.md, section 1.2.
AUD26 Phase 26.1 — foundation.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import uuid
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _default_sock_path() -> Path:
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "samia-runtimed.sock"
    return Path("/tmp") / "samia-runtimed.sock"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SamiaClientError(Exception):
    """Raised when the daemon returns an error response."""


class DaemonNotRunning(SamiaClientError):
    """Raised when the daemon socket is not reachable."""


# ---------------------------------------------------------------------------
# SamiaClient
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: float = 30.0


class SamiaClient:
    """Thin client for the SAM/IA runtime daemon.

    Parameters
    ----------
    sock_path : Path | None
        Path to the daemon's AF_UNIX socket.  Defaults to
        $XDG_RUNTIME_DIR/samia-runtimed.sock.
    timeout : float
        Socket timeout in seconds (default 30).
    """

    def __init__(
        self,
        sock_path: Path | str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._sock_path = Path(sock_path) if sock_path else _default_sock_path()
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    # -- Connection management --------------------------------------------

    def _connect(self) -> socket.socket:
        """Connect (or return existing connection)."""
        if self._sock is not None:
            return self._sock
        if not self._sock_path.exists():
            raise DaemonNotRunning(f"daemon socket not found: {self._sock_path}")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
        try:
            s.connect(str(self._sock_path))
        except (ConnectionRefusedError, OSError) as exc:
            s.close()
            raise DaemonNotRunning(f"cannot connect to daemon: {exc}") from exc
        self._sock = s
        return s

    def _disconnect(self) -> None:
        """Close the socket if open."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -- Wire helpers -----------------------------------------------------

    @staticmethod
    def _send(sock: socket.socket, request: dict[str, Any]) -> None:
        """Send a JSON-line request (newline-terminated)."""
        data = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
        sock.sendall(data)

    @staticmethod
    def _recv(sock: socket.socket) -> dict[str, Any]:
        """Read one JSON-line response (newline-terminated)."""
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("daemon closed connection")
            buf += chunk
            if b"\n" in buf:
                line, _ = buf.split(b"\n", 1)
                return json.loads(line.decode("utf-8"))

    # -- Core RPC ---------------------------------------------------------

    def call(self, op: str, **args: Any) -> Any:
        """Send an op to the daemon and return the result.

        Performs a single auto-reconnect if the socket is broken.
        Raises SamiaClientError on daemon error, DaemonNotRunning if
        the daemon is unreachable.
        """
        request = {
            "op": op,
            "args": args,
            "request_id": uuid.uuid4().hex,
        }

        with self._lock:
            for attempt in range(2):
                sock = self._connect()
                try:
                    self._send(sock, request)
                    resp = self._recv(sock)
                    break
                except (ConnectionError, OSError):
                    self._disconnect()
                    if attempt == 1:
                        raise DaemonNotRunning("connection broken after retry")
            else:
                raise DaemonNotRunning("connection broken after retry")

        # Validate response.
        if resp.get("request_id") != request["request_id"]:
            raise SamiaClientError(
                f"request_id mismatch: sent {request['request_id']!r}, "
                f"got {resp.get('request_id')!r}"
            )
        if not resp.get("ok"):
            raise SamiaClientError(resp.get("error") or "unknown error")

        return resp.get("result")

    # -- Convenience methods -----------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return daemon health dict (version, uptime, connected_clients, pid)."""
        return self.call("health")

    def version(self) -> str:
        """Return the daemon's version string."""
        return self.call("version")

    def echo(self, message: str) -> str:
        """Echo a message through the daemon (for testing)."""
        return self.call("echo", message=message)

    def shutdown(self) -> str:
        """Request graceful daemon shutdown."""
        return self.call("shutdown")

    # -- Lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the socket connection."""
        self._disconnect()

    def __enter__(self) -> SamiaClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# --------------------------------------------------------------------------
# [Asthenosphere] samia.runtime.client
# phase: AUD26-26.1
# layer: runtime (long-lived process)
# --------------------------------------------------------------------------
