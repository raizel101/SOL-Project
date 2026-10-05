"""Local SOL API contract and passive documentation, standard library only.

The canonical specification is stored as an immutable JSON string. Each call
returns a fresh object so a caller cannot mutate the contract for later clients.
Rendering documentation never contacts a service, calls a model, or runs script.
"""

from __future__ import annotations

import json
from html import escape
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
API_VERSION = "1.0.0"
MODES = (
    "retrieval",
    "local_rag",
    "prompt_preview",
    "urgent_support",
    "treatment_boundary",
    "human_support",
    "insufficient_sources",
    "supportive_abstention",
    "citation_check_failed",
    "support_response_withheld",
    "source_claim_withheld",
    "incomplete_response",
    "error",
)


def _local_base_url(value: str) -> str:
    if not isinstance(value, str) or any(character.isspace() for character in value):
        raise ValueError("The documentation base URL must be a local HTTP URL.")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid documentation base URL port.") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port is None
        or not 1 <= port <= 65535
    ):
        raise ValueError("Use an explicit http://127.0.0.1:PORT or http://localhost:PORT URL.")
    return value.rstrip("/")


def _ref(name: str) -> dict:
    return {"$ref": "#/components/schemas/" + name}


def _json_response(description: str, schema: dict) -> dict:
    return {
        "description": description,
        "content": {"application/json": {"schema": schema}},
        "headers": {
            "X-Request-ID": {
                "description": "Opaque identifier; no conversation content.",
                "schema": {"type": "string"},
            }
        },
    }


def _error_responses() -> dict:
    return {
        "400": _json_response(
            "Invalid input/JSON, duplicate object keys, nonfinite numbers (NaN/Infinity or exponent overflow), unsupported fields, or bounded context/history exceeded.",
            _ref("ErrorResponse"),
        ),
        "403": _json_response(
            "Host or browser Origin is not explicitly allowed.", _ref("ErrorResponse")
        ),
        "404": _json_response(
            "Unknown endpoint; use the exact path without a query string.", _ref("ErrorResponse")
        ),
        "405": {
            **_json_response("Wrong method for a known endpoint.", _ref("ErrorResponse")),
            "headers": {
                "Allow": {
                    "description": "Supported method(s) for this path, including OPTIONS.",
                    "schema": {"type": "string"},
                }
            },
        },
        "411": _json_response("Content-Length is required for POST.", _ref("ErrorResponse")),
        "413": _json_response(
            "Request exceeds the configured byte limit (default 32768 bytes).",
            _ref("ErrorResponse"),
        ),
        "415": _json_response(
            "POST requires one application/json Content-Type, an absent/UTF-8 charset, and absent/identity Content-Encoding; compressed or other-charset bodies are unsupported.",
            _ref("ErrorResponse"),
        ),
        "503": _json_response(
            "Local dependency/index failure, generation busy, or a withheld unverified answer. "
            "A fixed fallback may be present in answer; rejected model text is never returned.",
            _ref("ErrorResponse"),
        ),
        "504": _json_response("Local generation exceeded its time budget.", _ref("ErrorResponse")),
    }


def _make_template() -> dict:
    error = {
        "type": "object",
        "required": ["code", "message", "retryable"],
        "properties": {
            "code": {"type": "string"},
            "message": {"type": "string"},
            "retryable": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    offsets = {
        "type": "object",
        "required": ["start", "end_exclusive"],
        "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end_exclusive": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
        "description": "Exact Python Unicode-character offsets within the canonical extracted text record; end is exclusive. Not PDF byte offsets, original-file offsets, or UTF-16 code units.",
    }
    source_properties = {
        "citation": {"type": "string", "pattern": "^S[1-9][0-9]*$"},
        "record_id": {"type": "string"},
        "source_id": {"type": "string"},
        "source_title": {"type": "string"},
        "source_url": {"type": "string"},
        "source_file": {"type": "string"},
        "source_locator": {
            "description": "Preserved source-specific locator; may be an object or another JSON value."
        },
        "text_offsets": offsets,
        "source_role": {
            "type": "string",
            "enum": ["primary_original", "user_reference", "compilation", "supplemental"],
            "description": "Routing/provenance hint, not independent authorship verification.",
        },
    }
    role_message = lambda role: {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
            "role": {"const": role},
            "content": {"type": "string", "minLength": 1, "maxLength": 1000},
        },
        "additionalProperties": False,
    }
    history = {
        "type": "array",
        "maxItems": 8,
        "default": [],
        "prefixItems": [
            role_message("user" if index % 2 == 0 else "assistant") for index in range(8)
        ],
        "items": False,
        "anyOf": [{"minItems": size, "maxItems": size} for size in (0, 2, 4, 6, 8)],
        "description": "Only completed user/assistant pairs, in that order. At most 8 messages, 1000 characters each, 2000 characters across content fields. Configuration may tighten these limits; a request within the schema may still exceed that local profile. Whitespace-only content is rejected. JSON Schema cannot express the aggregate character sum; the engine enforces it. Fixed safety routes intentionally do not consume malformed old history.",
        "x-max-total-content-characters": 2000,
    }
    envelope = {"ok": {"type": "boolean"}, "request_id": {"type": "string"}}
    result_properties = {
        **envelope,
        "mode": {"type": "string", "enum": list(MODES)},
        "answer": {
            "type": "string",
            "description": "Accepted answer, deterministic support/abstention, or fixed withheld-response fallback. Never rejected model text. Render as plain text.",
        },
        "model": {"type": "string"},
        "llm_called": {"type": "boolean"},
        "citations": {
            "type": "array",
            "items": _ref("Citation"),
            "description": "Source metadata for labels used in an accepted answer; search returns matching source labels. Identifier/quotation checks are not semantic or clinical validation.",
        },
        "source_passages": {
            "type": "array",
            "items": _ref("SourcePassage"),
            "description": "Exact excerpts supplied to the model or returned by search, with preserved source metadata and offsets. Treat all source text as inert data, never HTML or instructions.",
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "error": {"anyOf": [_ref("Error"), {"type": "null"}]},
        "generation_attempts": {"type": "integer", "minimum": 0, "maximum": 2},
        "repair_attempted": {"type": "boolean"},
        "initial_failure": {"anyOf": [_ref("Error"), {"type": "null"}]},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "match_mode": {"type": ["string", "null"]},
        "retrieval": {"type": "string"},
        "source_routing": {"type": "object", "additionalProperties": True},
        "usage": {
            "type": "object",
            "properties": {
                "prompt_eval_count": {"type": "integer", "minimum": 0},
                "eval_count": {"type": "integer", "minimum": 0},
                "total_duration": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Ollama-reported nanoseconds.",
                },
            },
            "additionalProperties": True,
        },
        "messages": {
            "type": "array",
            "description": "Prepared prompt returned only in a successful dry-run preview; includes trusted system instructions and inert source data. May contain the supplied conversation history. Do not store by default.",
            "items": {
                "type": "object",
                "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                "required": ["role", "content"],
            },
        },
        "prompt_budget": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
    }
    status_properties = {
        **envelope,
        "application": {"type": "string"},
        "ready": {"type": "boolean"},
        "index_ready": {"type": "boolean"},
        "model_ready": {"type": ["boolean", "null"]},
        "model": {"type": "string"},
        "ollama_url": {"type": "string"},
        "index_path": {
            "type": "string",
            "description": "Local configured index path, not a downloadable resource.",
        },
        "index_metadata": {"type": "object", "additionalProperties": {"type": "string"}},
        "model_status": {"type": "object", "additionalProperties": True},
        "conversation_storage": {"type": "string"},
        "retrieval": {"type": "string"},
        "error": {"anyOf": [_ref("Error"), {"type": "null"}]},
    }
    schemas = {
        "Error": error,
        "ChatRequest": {
            "type": "object",
            "required": ["message"],
            "additionalProperties": False,
            "properties": {
                "message": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1200,
                    "description": "Non-whitespace user message; Python Unicode-character count. Configuration may impose a stricter engine limit.",
                },
                "history": history,
                "dry_run": {"type": "boolean", "default": False},
            },
            "description": "No system/tool/developer messages, client-selected model, sessions, source instructions, or arbitrary fields. dry_run suppresses model generation; fixed support/boundary routes may reply before retrieval.",
        },
        "SearchRequest": {
            "type": "object",
            "required": ["query"],
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1200,
                    "description": "Non-whitespace lexical search query. Configuration may impose a stricter engine limit.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "description": "Defaults to configured top_k_sources (normally 3). Boolean values are rejected.",
                },
            },
        },
        "Citation": {
            "type": "object",
            "properties": source_properties,
            "additionalProperties": True,
        },
        "SourcePassage": {
            "type": "object",
            "properties": {
                **source_properties,
                "text": {"type": "string"},
                "chunk_id": {"type": "string"},
                "snippet": {
                    "type": "string",
                    "description": "Search display snippet, not necessarily the exact full source excerpt.",
                },
                "bm25": {"type": "number"},
            },
            "additionalProperties": True,
        },
        "ChatResponse": {
            "type": "object",
            "required": ["ok", "request_id"],
            "properties": result_properties,
            "additionalProperties": True,
        },
        "ErrorResponse": {
            "type": "object",
            "required": ["ok", "request_id", "error"],
            "properties": {**result_properties, "ok": {"const": False}, "error": _ref("Error")},
            "additionalProperties": True,
        },
        "StatusResponse": {
            "type": "object",
            "required": ["ok", "request_id"],
            "properties": status_properties,
            "additionalProperties": True,
        },
        "ServiceResponse": {
            "type": "object",
            "required": ["ok", "request_id"],
            "properties": {
                **envelope,
                "service": {"type": "string"},
                "api_version": {"type": "string"},
                "status": {"type": "string"},
                "live": {"type": "boolean"},
                "endpoints": {"type": "object", "additionalProperties": True},
                "privacy": {
                    "type": "object",
                    "properties": {
                        "conversation_storage": {"const": False},
                        "local_only": {"const": True},
                    },
                    "additionalProperties": True,
                },
                "student_facing_release_approved": {"const": False},
            },
            "additionalProperties": True,
        },
    }
    chat_responses = {
        "200": _json_response(
            "Accepted response, preview, abstention, or deterministic safety/boundary support; inspect mode and llm_called.",
            _ref("ChatResponse"),
        ),
        **_error_responses(),
    }
    chat_responses["503"]["headers"]["Retry-After"] = {
        "description": "Present when busy; normally 5 seconds. Do not automatically retry nonretryable semantic/support failures.",
        "schema": {"type": "string"},
    }
    common_forbidden = {code: _error_responses()[code] for code in ("403", "405")}

    def get_operation(ident, summary, schema, description="", extra=None):
        return {
            "operationId": ident,
            "summary": summary,
            "description": description,
            "responses": {
                "200": _json_response(summary, schema),
                **common_forbidden,
                **(extra or {}),
            },
        }

    def post_operation(ident, summary, request_schema, description):
        return {
            "operationId": ident,
            "summary": summary,
            "description": description,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": _ref(request_schema)}},
            },
            "responses": chat_responses,
        }

    def asset_operation(ident, summary, media_type, description):
        return {
            "operationId": ident,
            "summary": summary,
            "description": description,
            "responses": {
                "200": {
                    "description": summary,
                    "content": {media_type: {"schema": {"type": "string"}}},
                },
                **common_forbidden,
                "503": _json_response(
                    "The fixed local UI asset is unavailable (asset_unavailable); no model call.",
                    _ref("ErrorResponse"),
                ),
            },
        }

    chat_description = (
        "Uses the existing local corpus and installed local model. One shared generation slot across legacy/versioned aliases; busy returns 503. "
        "Recognized urgent, treatment, diagnosis, persistent-distress and narrow prediction routes produce deterministic no-model replies. "
        "dry_run does not call the model. A citation-only correction can make at most one extra call within a shared budget of at most 120 seconds; configuration may tighten it. "
        "Validate HTTP status AND error; a 503 fallback is a withheld/unaccepted response, not a successful generation. No server chat persistence."
    )
    search_description = (
        "Read-only SQLite FTS5/BM25 keyword search of the broader corpus, including sources excluded from author-original chat evidence. "
        "No model call, training, fresh web scraping, transcription or semantic embeddings. Source references do not certify interpretation."
    )
    paths = {
        "/": {
            "get": get_operation(
                "serviceInfo",
                "Local service information and endpoint links",
                _ref("ServiceResponse"),
                "JSON service information remains available here; the local prototype interface is at /chat.",
            )
        },
        "/chat": {
            "get": asset_operation(
                "chatInterface",
                "Open the local chat prototype",
                "text/html",
                "Same-origin vanilla browser UI using the existing API/model/corpus. In-memory conversation only; no accounts, browser storage, CDN or automatic model generation. Educational prototype, not approved for student release.",
            )
        },
        "/assets/styles.css": {
            "get": asset_operation(
                "chatStyles",
                "Load fixed local chat styles",
                "text/css",
                "Allowlisted local stylesheet for /chat; no arbitrary file serving.",
            )
        },
        "/assets/app.js": {
            "get": asset_operation(
                "chatScript",
                "Load fixed local chat behavior",
                "text/javascript",
                "Allowlisted local script for /chat; sends user-initiated chat requests to the same-origin API.",
            )
        },
        "/assets/client-core.js": {
            "get": asset_operation(
                "chatClientCore",
                "Load fixed local response and history handling",
                "text/javascript",
                "Allowlisted local script for bounded history and response handling; no external dependencies.",
            )
        },
        "/health/live": {
            "get": get_operation(
                "liveness",
                "Check the HTTP process only",
                _ref("ServiceResponse"),
                "No index read or model check. A live process may not be ready.",
            )
        },
        "/health/ready": {
            "get": get_operation(
                "readiness",
                "Check the configured local index and installed model",
                _ref("StatusResponse"),
                "200 only when ready is true. Not a teaching-quality, safeguarding, privacy or clinical approval.",
                {
                    "503": _json_response(
                        "Index/model is unavailable or status cannot be checked; inspect ready and error.",
                        _ref("StatusResponse"),
                    )
                },
            )
        },
        "/api/v1/status": {
            "get": get_operation(
                "status",
                "Inspect engine status, including unready dependencies",
                _ref("StatusResponse"),
                "Normally 200 even when ready is false; use /health/ready for readiness status codes.",
                {"503": _json_response("Status could not be checked.", _ref("ErrorResponse"))},
            )
        },
        "/api/v1/chat": {
            "post": post_operation(
                "chat",
                "Return a bounded source-backed local reply",
                "ChatRequest",
                chat_description,
            )
        },
        "/api/v1/search": {
            "post": post_operation(
                "search",
                "Find exact source passages without generation",
                "SearchRequest",
                search_description,
            )
        },
        "/api/v1/openapi.json": {
            "get": get_operation(
                "openapi",
                "Get this raw OpenAPI 3.1 document",
                {"type": "object", "additionalProperties": True},
                "The contract is returned without the normal ok/request_id JSON envelope.",
            )
        },
        "/docs": {
            "get": {
                "operationId": "documentation",
                "summary": "Read passive local integration documentation",
                "description": "Self-contained HTML; no scripts, CDN resources or model requests.",
                "responses": {
                    "200": {
                        "description": "Local HTML documentation",
                        "content": {"text/html": {"schema": {"type": "string"}}},
                    },
                    **common_forbidden,
                },
            }
        },
        "/health": {
            "get": {
                **get_operation(
                    "legacyHealth",
                    "Legacy engine status",
                    _ref("StatusResponse"),
                    "Compatibility route: normally 200 even when ready is false; not a readiness probe.",
                    {"503": _json_response("Status could not be checked.", _ref("ErrorResponse"))},
                ),
                "deprecated": True,
            }
        },
        "/api/chat": {
            "post": {
                **post_operation(
                    "legacyChat",
                    "Compatibility alias of /api/v1/chat",
                    "ChatRequest",
                    chat_description,
                ),
                "deprecated": True,
            }
        },
        "/api/search": {
            "post": {
                **post_operation(
                    "legacySearch",
                    "Compatibility alias of /api/v1/search",
                    "SearchRequest",
                    search_description,
                ),
                "deprecated": True,
            }
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "SOL Chat local backend",
            "version": API_VERSION,
            "description": "Local-only educational RAG prototype using Sri Aurobindo / The Mother corpus references. No authentication, accounts, persistent conversations, training, public hosting or emergency monitoring. Other programs on this laptop can access the API. Source faithfulness and qualified teaching/mental-health/safeguarding review remain student-release blockers. All POSTs need Content-Length and one application/json Content-Type; charset must be absent/UTF-8, Content-Encoding absent/identity. Duplicate object keys and nonfinite numbers (NaN/Infinity or exponent overflow) are rejected. Default request-body limit 32768 bytes. Host must match the explicit loopback address/port; browser Origin must be explicitly allowed. Responses are non-streaming. Wrong methods on known routes return 405 with Allow. OPTIONS supports local CORS preflight; no wildcard CORS.",
        },
        "servers": [
            {
                "url": DEFAULT_BASE_URL,
                "description": "Only the explicitly configured loopback HTTP server is supported.",
            }
        ],
        "security": [],
        "paths": paths,
        "components": {"schemas": schemas},
    }


_SPEC_JSON = json.dumps(_make_template(), ensure_ascii=False, separators=(",", ":"))


def build_openapi(base_url: str = DEFAULT_BASE_URL) -> dict:
    """Return an isolated OpenAPI object for the actual local server address."""
    base_url = _local_base_url(base_url)
    result = json.loads(_SPEC_JSON)
    result["servers"][0]["url"] = base_url
    return result


def render_docs(base_url: str = DEFAULT_BASE_URL) -> str:
    """Render passive, offline HTML; no embedded scripts or active request UI."""
    spec = build_openapi(base_url)
    base_url = spec["servers"][0]["url"]
    rows = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            legacy = " (legacy)" if operation.get("deprecated") else ""
            rows.append(
                "<tr><td><code>"
                + escape(method.upper() + " " + path)
                + "</code>"
                + legacy
                + "</td><td>"
                + escape(operation["summary"])
                + "</td></tr>"
            )
    contract_url = escape(base_url + "/api/v1/openapi.json", quote=True)
    pretty_spec = escape(json.dumps(spec, ensure_ascii=False, indent=2))
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SOL local backend documentation</title>
<style>body{font-family:system-ui,sans-serif;line-height:1.6;color:#22303c;background:#f8fafb;max-width:960px;margin:3rem auto;padding:0 1.25rem}h1,h2{line-height:1.25}table{width:100%;border-collapse:collapse;background:#fff}td,th{text-align:left;border-bottom:1px solid #dde5eb;padding:.65rem}code,pre{font-family:ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f5;padding:1rem;border-radius:.5rem}aside{background:#fff4dd;border-left:4px solid #b68122;padding:1rem}a{color:#185c97}</style>
</head><body><h1>SOL Chat local backend</h1>
<p>Base address: <code>"""
        + escape(base_url)
        + """</code>. This page is passive API documentation. Open <code>/chat</code> on this server for the local chat prototype; it is not approved for student release.</p>
<aside>Educational prototype only. A citation is not proof of an interpretation. Source-faithfulness evaluation and qualified teaching, mental-health and safeguarding review remain required before student use. SOL cannot diagnose, monitor safety, call for help or replace human care.</aside>
<h2>Endpoints</h2><table><thead><tr><th>Route</th><th>Purpose</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table>
<h2>Requests and limits</h2>
<p>Send UTF-8 <code>application/json</code> with <code>Content-Length</code>. Use the exact endpoint path without a query string. A message/query has at most 1,200 Python Unicode characters. History is zero to four complete user/assistant pairs: at most eight messages, 1,000 characters per message and 2,000 characters total. Configuration may tighten these bounds. No extra fields or system/tool messages. Search <code>limit</code> is an integer 1–6. The default HTTP body limit is 32,768 bytes.</p>
<p>Duplicate JSON object keys and nonfinite numbers (NaN/Infinity or exponent overflow) are rejected. Send one Content-Type header; an explicit charset must be <code>utf-8</code> or <code>utf8</code>. Compressed bodies and other charsets are unsupported. Wrong HTTP methods on known routes return 405 with an <code>Allow</code> header.</p>
<pre>{"message":"What is integral education?","history":[],"dry_run":true}</pre>
<p>POST that example to <code>/api/v1/chat</code>. A dry-run inspects evidence and the prompt without generation; fixed support/boundary routes may answer without a preview. For reading-only search:</p>
<pre>{"query":"education attention interest","limit":3}</pre>
<h2>Response handling</h2>
<p>Parse JSON even on HTTP 503. Inspect <code>ok</code>, <code>error</code>, <code>mode</code> and <code>llm_called</code>. A fixed fallback in a withheld response is not an accepted model answer. Show it with a withheld/unaccepted label and warnings; never invent or display rejected model text. Do not automatically retry nonretryable semantic/support failures. Normal abstention and fixed support may legitimately have no citations and no model call.</p>
<p>There is one shared generation slot across legacy and versioned routes. Busy returns 503 with <code>Retry-After</code>. Retrieval, liveness and fixed safety routes do not require a generation slot. Generation is not streamed.</p>
<p>Render answer and source text as plain text. Show title, locator, exact excerpt and source URL where provided. Make links clickable only after validating HTTP(S) schemes. Offsets are exact Python Unicode-character ranges in the canonical extracted record, not page numbers or JavaScript UTF-16 indices. A source role is a routing hint, not independent authorship verification.</p>
<p>The local interface displays citations, source excerpts, warnings and response states. Conversation state stays in memory and can be cleared with a new conversation. It never truncates messages to fit history: a completed turn that exceeds the history limits requires a reset before the next message. Withheld or unverified responses are not promoted to accepted conversation history. A browser request cancellation, if used, does not cancel server model work.</p>
<h2>Local boundary</h2>
<p>This process binds only to <code>127.0.0.1</code>. No authentication, accounts or chat database are included. Other local programs can access it. Do not expose it using a tunnel, firewall rule or LAN/public bind. The bundled interface uses the same origin and needs no extra CORS setting. Only explicitly allowed loopback browser origins work; CORS is not authentication. The server does not persist chat or log bodies, but this does not verify all privacy behavior of your operating system or model service.</p>
<p>Index access is read-only. Starting this backend does not scrape websites, download a model or train model weights. Use the existing installed local Ollama model. Liveness checks the process; readiness checks local dependencies, not response quality.</p>
<h2>Complete contract</h2><p><a href="""
        + '"'
        + contract_url
        + '"'
        + """>Open the JSON specification</a>. The following embedded copy works without scripts or a CDN.</p><details><summary>OpenAPI 3.1 specification</summary><pre>"""
        + pretty_spec
        + """</pre></details>
</body></html>"""
    )


__all__ = ["API_VERSION", "DEFAULT_BASE_URL", "MODES", "build_openapi", "render_docs"]
