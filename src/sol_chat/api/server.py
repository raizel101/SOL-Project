"""SOL Chat local integration API. Standard library; loopback only.

No accounts, disk conversation history, external AI calls, or public deployment.
The separate engine searches the existing corpus and an installed local model.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sol_chat.config import (
    ConfigError,
    finite_json_float,
    load_config,
    local_origin,
    validate_config,
)
from sol_chat.core.engine import (
    Engine,
    diagnosis_boundary_trigger,
    persistent_distress_trigger,
    treatment_boundary_trigger,
    ungrounded_prediction_trigger,
    urgent_followup_trigger,
    urgent_trigger,
)

from .contract import build_openapi, render_docs
from .web_assets import UI_CSP, UI_PATHS, read_asset

API_VERSION = "1.0.0"
CHAT_PATHS = frozenset({"/api/chat", "/api/v1/chat"})
SEARCH_PATHS = frozenset({"/api/search", "/api/v1/search"})
GET_PATHS = (
    frozenset(
        {
            "/",
            "/health",
            "/health/live",
            "/health/ready",
            "/api/v1/status",
            "/api/v1/openapi.json",
            "/docs",
        }
    )
    | UI_PATHS
)
POST_PATHS = CHAT_PATHS | SEARCH_PATHS


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_json_constant(value):
    raise ValueError("Nonfinite JSON number")


def make_server(engine, host="127.0.0.1", port=0, allowed_origins=(), max_body_bytes=32768):
    if host != "127.0.0.1":
        raise ValueError("SOL Chat binds to 127.0.0.1 only; public/LAN hosting is not configured.")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Invalid local port.")
    if (
        isinstance(max_body_bytes, bool)
        or not isinstance(max_body_bytes, int)
        or not 1024 <= max_body_bytes <= 65536
    ):
        raise ValueError("Body limit must be from 1024 to 65536 bytes.")
    if not isinstance(allowed_origins, (list, tuple)) or len(allowed_origins) > 8:
        raise ValueError("At most eight explicit loopback origins are supported.")
    origins = {local_origin(origin) for origin in allowed_origins}
    generation_slot = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "SOLChat/1.0"
        sys_version = ""

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            # Do not print message bodies, questions, query strings or source text.
            pass

        def finish(self):
            # On Windows, immediately closing with an unread rejected body can
            # reset TCP before the JSON error reaches the client. Send FIN after
            # flushing, then discard a small, time-bounded amount of input.
            try:
                if not self.wfile.closed:
                    self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                self.connection.settimeout(0.05)
                deadline, consumed = time.monotonic() + 0.2, 0
                while consumed < 65536 and time.monotonic() < deadline:
                    chunk = self.rfile.read1(min(8192, 65536 - consumed))
                    if not chunk:
                        break
                    consumed += len(chunk)
            except (OSError, ValueError):
                pass
            finally:
                super().finish()

        def reply(self, status, value=None, extra_headers=None):
            ident = uuid.uuid4().hex[:16]
            if status == 204:
                body = b""
            else:
                payload = {"ok": status < 400, "request_id": ident}
                payload.update(value or {})
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.emit(status, body, "application/json; charset=utf-8", ident, extra_headers)

        def emit(self, status, body, content_type, ident=None, extra_headers=None, policy=None):
            self.close_connection = True
            ident = ident or uuid.uuid4().hex[:16]
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Request-ID", ident)
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                policy
                or "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            )
            origin = getattr(self, "headers", {}).get("Origin")
            if origin and origin in self.permitted_origins():
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            for key, item in (extra_headers or {}).items():
                self.send_header(key, item)
            self.end_headers()
            if body and self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, socket.timeout):
                    pass

        def fail(self, status, code, message, retryable=False, extra_headers=None):
            self.reply(
                status,
                {"error": {"code": code, "message": message, "retryable": retryable}},
                extra_headers,
            )

        def send_error(self, code, message=None, explain=None):
            # Base-handler parse/method errors must not emit HTML or echo paths.
            if getattr(self, "request_version", "HTTP/0.9") == "HTTP/0.9":
                self.request_version = "HTTP/1.1"
            self.fail(code, "http_request_error", "The HTTP request could not be processed.")

        def methods(self):
            if self.path in GET_PATHS:
                return "GET, OPTIONS"
            if self.path in POST_PATHS:
                return "POST, OPTIONS"
            return None

        def method_not_allowed(self):
            if not self.check_local_request():
                return
            methods = self.methods()
            if methods is None:
                self.fail(404, "not_found", "Unknown endpoint.")
            else:
                self.fail(
                    405,
                    "method_not_allowed",
                    "Use the documented HTTP method.",
                    extra_headers={"Allow": methods},
                )

        do_PUT = do_PATCH = do_DELETE = do_HEAD = do_TRACE = do_CONNECT = method_not_allowed

        def permitted_origins(self):
            actual = self.server.server_port
            return origins | {f"http://127.0.0.1:{actual}", f"http://localhost:{actual}"}

        def check_local_request(self):
            host_values = self.headers.get_all("Host", [])
            accepted = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if len(host_values) != 1 or host_values[0] not in accepted:
                self.fail(403, "host_not_allowed", "Use the explicit local SOL Chat address.")
                return False
            origin_values = self.headers.get_all("Origin", [])
            if len(origin_values) > 1 or (
                origin_values and origin_values[0] not in self.permitted_origins()
            ):
                self.fail(
                    403,
                    "origin_not_allowed",
                    "This local API does not permit the requesting website.",
                )
                return False
            return True

        def do_OPTIONS(self):
            if not self.check_local_request():
                return
            methods = self.methods()
            if methods is None:
                self.fail(404, "not_found", "Unknown endpoint.")
                return
            requested_method = self.headers.get("Access-Control-Request-Method")
            requested_headers = self.headers.get("Access-Control-Request-Headers", "")
            if requested_method and requested_method not in methods.split(", "):
                self.fail(
                    405,
                    "method_not_allowed",
                    "This preflight method is not supported.",
                    extra_headers={"Allow": methods},
                )
                return
            if any(
                header.strip().lower() != "content-type"
                for header in requested_headers.split(",")
                if header.strip()
            ):
                self.fail(
                    400,
                    "invalid_request",
                    "Only Content-Type is supported in browser preflight headers.",
                )
                return
            self.reply(
                204,
                extra_headers={
                    "Allow": methods,
                    "Access-Control-Allow-Methods": methods,
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Max-Age": "300",
                },
            )

        def do_GET(self):
            if not self.check_local_request():
                return
            if self.path not in GET_PATHS:
                self.method_not_allowed()
                return
            if self.path in UI_PATHS:
                try:
                    body, media_type = read_asset(self.path)
                except (OSError, ValueError):
                    self.fail(
                        503,
                        "asset_unavailable",
                        "A local interface file is unavailable. Restore the complete SOL application folder.",
                    )
                    return
                self.emit(200, body, media_type, policy=UI_CSP)
                return
            if self.path == "/":
                self.reply(
                    200,
                    {
                        "service": "sol-chat",
                        "api_version": API_VERSION,
                        "status": "running",
                        "endpoints": {
                            "chat": "/api/v1/chat",
                            "search": "/api/v1/search",
                            "status": "/api/v1/status",
                            "live": "/health/live",
                            "ready": "/health/ready",
                            "docs": "/docs",
                            "openapi": "/api/v1/openapi.json",
                            "interface": "/chat",
                        },
                        "privacy": {"conversation_storage": False, "local_only": True},
                        "student_facing_release_approved": False,
                    },
                )
                return
            if self.path == "/health/live":
                self.reply(200, {"service": "sol-chat", "status": "alive", "live": True})
                return
            base_url = f"http://127.0.0.1:{self.server.server_port}"
            if self.path == "/docs":
                self.emit(200, render_docs(base_url).encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/v1/openapi.json":
                self.emit(
                    200,
                    json.dumps(build_openapi(base_url), ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return
            try:
                result = engine.status(check_model=True)
                status = (
                    503 if self.path == "/health/ready" and result.get("ready") is not True else 200
                )
                if status == 503 and not result.get("error"):
                    result = dict(
                        result,
                        error={
                            "code": "service_not_ready",
                            "message": "The local index and installed model must both be available.",
                            "retryable": True,
                        },
                    )
                self.reply(status, result)
            except Exception:
                self.fail(
                    503, "health_unavailable", "The local engine status could not be checked.", True
                )

        def read_object(self):
            if self.headers.get("Transfer-Encoding"):
                self.fail(
                    400,
                    "unsupported_transfer_encoding",
                    "Chunked request bodies are not supported.",
                )
                return None
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                self.fail(411, "length_required", "Content-Length is required.")
                return None
            if (
                len(lengths) != 1
                or not 1 <= len(lengths[0]) <= 9
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
            ):
                self.fail(400, "invalid_length", "Invalid request body length.")
                return None
            length = int(lengths[0])
            if length > max_body_bytes:
                self.fail(413, "body_too_large", "Shorten the message and conversation history.")
                return None
            if (
                len(self.headers.get_all("Content-Type", [])) != 1
                or self.headers.get_content_type() != "application/json"
                or self.headers.get_content_charset("utf-8") not in {"utf-8", "utf8"}
                or self.headers.get("Content-Encoding", "identity").lower() != "identity"
            ):
                self.fail(415, "json_required", "Send application/json.")
                return None
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                obj = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=unique_json_object,
                    parse_constant=reject_json_constant,
                    parse_float=finite_json_float,
                )
            except (ValueError, UnicodeError, RecursionError, socket.timeout):
                self.fail(
                    400, "invalid_json", "The request must contain a complete UTF-8 JSON object."
                )
                return None
            if not isinstance(obj, dict):
                self.fail(400, "invalid_request", "The JSON body must be an object.")
                return None
            return obj

        def do_POST(self):
            if not self.check_local_request():
                return
            if self.path not in POST_PATHS:
                self.method_not_allowed()
                return
            obj = self.read_object()
            if obj is None:
                return
            is_chat = self.path in CHAT_PATHS
            fields = {"message", "history", "dry_run"} if is_chat else {"query", "limit"}
            if set(obj) - fields:
                self.fail(400, "invalid_request", "Unsupported request fields.")
                return
            message = obj.get("message" if is_chat else "query")
            if not isinstance(message, str) or not message.strip() or len(message) > 1200:
                self.fail(
                    400, "invalid_message", "Provide a nonempty message of at most 1200 characters."
                )
                return
            if is_chat:
                history = obj.get("history", [])
                dry_run = obj.get("dry_run", False)
                fixed_route = (
                    urgent_trigger(message)
                    or treatment_boundary_trigger(message)
                    or diagnosis_boundary_trigger(message)
                    or persistent_distress_trigger(message)
                    or ungrounded_prediction_trigger(message)
                    or urgent_followup_trigger(message, history)
                )
                if not fixed_route and (
                    not isinstance(history, list) or not isinstance(dry_run, bool)
                ):
                    self.fail(
                        400,
                        "invalid_request",
                        "history must be a list and dry_run must be Boolean.",
                    )
                    return
                # Fixed safety/boundary replies must not wait for generation.
                acquired = False if fixed_route else generation_slot.acquire(blocking=False)
                if not fixed_route and not acquired:
                    self.reply(
                        503,
                        {
                            "error": {
                                "code": "busy",
                                "message": "A local answer is already being generated. Try again shortly.",
                                "retryable": True,
                            }
                        },
                        {"Retry-After": "5"},
                    )
                    return
                try:
                    result = engine.chat(message, history=history, dry_run=dry_run)
                except Exception:
                    result = {
                        "error": {
                            "code": "engine_error",
                            "message": "The local engine could not complete this request.",
                            "retryable": True,
                        }
                    }
                finally:
                    if acquired:
                        generation_slot.release()
            else:
                limit = obj.get("limit")
                if limit is not None and (
                    isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 6
                ):
                    self.fail(400, "invalid_request", "limit must be an integer from 1 to 6.")
                    return
                try:
                    result = engine.search(message, limit=limit)
                except Exception:
                    result = {
                        "error": {
                            "code": "index_unavailable",
                            "message": "The local source index is unavailable.",
                            "retryable": True,
                        }
                    }
            error = result.get("error")
            status = 200
            if error:
                code = error.get("code", "engine_error")
                if code in {
                    "invalid_input",
                    "invalid_question",
                    "invalid_history",
                    "invalid_config",
                    "context_too_large",
                    "input_too_large",
                    "invalid_limit",
                    "question_too_long",
                    "history_too_long",
                    "prompt_too_long",
                    "invalid_request",
                }:
                    status = 400
                elif code in {"model_timeout", "ollama_timeout"}:
                    status = 504
                else:
                    status = 503
            self.reply(status, result)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="UTF-8 backend JSON; defaults to config/local.json in this project",
    )
    parser.add_argument("--port", type=int, help="Override the local port")
    parser.add_argument("--db", type=Path)
    parser.add_argument(
        "--allow-origin",
        action="append",
        help="Override configured origins (repeat for each loopback frontend)",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        index_path = (
            validate_config({"index_path": str(args.db)}, base_dir=Path.cwd()).index_path
            if args.db is not None
            else config.index_path
        )
        engine = Engine(db_path=index_path, config=config.engine)
        server = make_server(
            engine,
            host=config.host,
            port=config.port if args.port is None else args.port,
            allowed_origins=config.allowed_origins
            if args.allow_origin is None
            else args.allow_origin,
            max_body_bytes=config.max_body_bytes,
        )
    except (ConfigError, ValueError, TypeError):
        print(
            json.dumps(
                {
                    "event": "sol_chat_start_failed",
                    "error": {
                        "code": "invalid_config",
                        "message": "Check backend_config.json and local CLI limits. No server was started.",
                    },
                }
            ),
            flush=True,
        )
        return 2
    except OSError:
        print(
            json.dumps(
                {
                    "event": "sol_chat_start_failed",
                    "error": {
                        "code": "local_bind_failed",
                        "message": "The local port is unavailable. Stop the existing instance or choose another local port.",
                    },
                }
            ),
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "event": "sol_chat_started",
                "address": f"http://127.0.0.1:{server.server_port}",
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Local integration API only; no conversation text is written to disk.",
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
