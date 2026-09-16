"""Zero-cloud localhost HTTP server for the Memory Center.

The console is served with only the Python standard library
(``http.server``) and vanilla HTML/CSS/JS, so lightweight installs keep working.
It binds to ``127.0.0.1`` by default and never to a public interface unless the
operator explicitly passes a non-loopback ``--host``.

Security posture for a localhost console:

* Only ``GET`` is served; every other method is refused with 405.
* Static assets come from in-memory package constants, never from the
  filesystem, so there is no arbitrary static-read or path-traversal surface.
* Query parameters are parsed, validated, and clamped before use.
* JSON responses are serialized with ``allow_nan=False`` after recursive
  sanitization, and served with ``X-Content-Type-Options: nosniff``.
* A restrictive Content-Security-Policy (no inline scripts/styles, no remote
  origins) accompanies the document and asset responses.
* No raw database or adapter/model path is ever returned.
"""

from __future__ import annotations

import json
import ipaddress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .static import CLIENT_JS, INDEX_HTML, STYLES_CSS
from .views import ConsoleView, sanitize_payload


CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


class ConsoleHandler(BaseHTTPRequestHandler):
    """Routes GET requests to the read-only :class:`ConsoleView`."""

    server_version = "MPMConsole/0.1"

    @property
    def view(self) -> ConsoleView:
        return self.server.view  # type: ignore[attr-defined]

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence request logs
        return

    def _send(
        self,
        status: int,
        body: str,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(
            sanitize_payload(payload),
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        )
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.server.loopback_only and not self._valid_loopback_host():  # type: ignore[attr-defined]
            self._error(421, "invalid host for local console")
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            query = parse_qs(parsed.query, keep_blank_values=False, max_num_fields=64)
        except ValueError:
            self._error(400, "too many query parameters")
            return

        if path == "/":
            self._send(200, INDEX_HTML, "text/html; charset=utf-8", {"Content-Security-Policy": CSP})
            return
        if path == "/assets/console.css":
            self._send(200, STYLES_CSS, "text/css; charset=utf-8", {"Content-Security-Policy": CSP})
            return
        if path == "/assets/console.js":
            self._send(200, CLIENT_JS, "application/javascript; charset=utf-8", {"Content-Security-Policy": CSP})
            return

        if path == "/api/status":
            self._json(200, self.view.status())
            return
        if path == "/api/memories":
            self._json(
                200,
                self.view.list_memories(
                    q=_first(query, "q"),
                    status=_first(query, "status") or "all",
                    scope=_first(query, "scope"),
                    signal=_first(query, "signal") or "all",
                    limit=_parse_int(_first(query, "limit")),
                    offset=_parse_int(_first(query, "offset")),
                ),
            )
            return
        if path.startswith("/api/memories/"):
            memory_id = unquote(path[len("/api/memories/") :])
            detail = self.view.memory_detail(memory_id)
            if detail is None:
                self._error(404, "memory not found")
            else:
                self._json(200, detail)
            return
        if path == "/api/audit":
            types = query.get("types") or query.get("type")
            self._json(
                200,
                self.view.audit(
                    since=_parse_int(_first(query, "since")),
                    limit=_parse_int(_first(query, "limit")),
                    types=types,
                    newest=_parse_flag(_first(query, "newest")),
                ),
            )
            return
        if path == "/api/decisions":
            self._json(
                200,
                self.view.decisions(
                    op=_first(query, "op"),
                    version=_first(query, "version"),
                    target=_first(query, "target"),
                    limit=_parse_int(_first(query, "limit")),
                    offset=_parse_int(_first(query, "offset")),
                ),
            )
            return
        if path == "/api/checkpoints":
            self._json(200, self.view.checkpoints())
            return

        self._error(404, "not found")

    def _valid_loopback_host(self) -> bool:
        host_header = self.headers.get("Host", "")
        try:
            hostname = urlsplit("//" + host_header).hostname
        except ValueError:
            return False
        return bool(hostname and is_loopback(hostname))

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in {"/", "/assets/console.css", "/assets/console.js"}:
            self.send_response(200)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _method_not_allowed(self) -> None:
        self._error(405, "read-only console; only GET is supported")

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed


def make_server(store: Any, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Build a threaded HTTP server bound to ``store`` without starting it."""
    server = ThreadingHTTPServer((host, port), ConsoleHandler)
    server.daemon_threads = True
    server.view = ConsoleView(store)  # type: ignore[attr-defined]
    server.loopback_only = is_loopback(host)  # type: ignore[attr-defined]
    return server


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
