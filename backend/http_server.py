"""Serves the mirror UI and a tiny local JSON API from one process (no second terminal needed).

Bound to localhost only; requests with a foreign Host header are refused (DNS-rebinding guard).
"""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
import json
import logging
import threading

ALLOWED_HOSTS = ("localhost", "127.0.0.1", "::1")


class MirrorHandler(SimpleHTTPRequestHandler):
    api: dict[str, Callable[[], object]] = {}

    def _host_ok(self) -> bool:
        raw = (self.headers.get("Host") or "").strip()
        host = raw[1:raw.index("]")] if raw.startswith("[") and "]" in raw else raw.split(":")[0]
        return host in ALLOWED_HOSTS

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            self.send_error(403, "Forbidden host")
            return
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/"):
            handler = self.api.get(path)
            if not handler:
                self.send_error(404)
                return
            try:
                body = json.dumps(handler(), default=str).encode("utf-8")
            except Exception:
                logging.exception("API handler failed")
                self.send_error(500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            if path == "/api/export":
                self.send_header("Content-Disposition", 'attachment; filename="mirror-export.json"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # keep the console for the mirror's own log


def start_http_server(directory: Path, host: str, port: int, api: dict[str, Callable[[], object]]) -> ThreadingHTTPServer | None:
    handler_cls = type("BoundMirrorHandler", (MirrorHandler,), {"api": api})
    try:
        server = ThreadingHTTPServer((host, port), partial(handler_cls, directory=str(directory)))
    except OSError as exc:
        logging.error("Could not start the web server on %s:%s (%s). Is another mirror running? Set MIRROR_HTTP_PORT.", host, port, exc)
        return None
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    return server
