"""Loopback HTTP contract tests. FakeEngine never calls an LLM or reads the corpus."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import unittest

from sol_chat.api import server as SERVER

RECENT_URGENT_HISTORY = [
    {"role": "user", "content": "I want to hurt myself tonight."},
    {
        "role": "assistant",
        "content": "Please seek emergency support and ask a trusted person to be with you.",
    },
]


class FakeEngine:
    """An explicit seam: these tests measure transport, not model intelligence."""

    def __init__(self):
        self.calls = []
        self.chat_started = threading.Event()
        self.release_chat = threading.Event()
        self.block_chat = False
        self.result_override = None
        self.raise_error = False
        self.status_override = None
        self.raise_status = False

    def status(self, check_model=True):
        self.calls.append(("status", check_model))
        if self.raise_status:
            raise RuntimeError("INTERNAL_FIXTURE_SECRET")
        if self.status_override is not None:
            return self.status_override
        return {"mode": "status", "ready": True, "model": "fixture-local-model", "warnings": []}

    def search(self, question, limit=None):
        self.calls.append(("search", question, limit))
        return {"mode": "search", "results": [], "query": question, "warnings": []}

    def chat(self, question, history=None, dry_run=False):
        self.calls.append(("chat", question, history, dry_run))
        if question == "I want to harm myself":
            return {
                "mode": "urgent_support",
                "answer": "Fixture urgent human support.",
                "citations": [],
                "source_passages": [],
                "warnings": [],
                "llm_called": False,
            }
        if question == "What should I do now?" and history == RECENT_URGENT_HISTORY:
            return {
                "mode": "urgent_support",
                "answer": "Fixture urgent follow-up human support.",
                "citations": [],
                "source_passages": [],
                "warnings": [],
                "llm_called": False,
            }
        if question == "I take medicine for depression. Should I stop it and only meditate?":
            return {
                "mode": "treatment_boundary",
                "answer": "Fixture: discuss medication changes with the prescriber.",
                "citations": [],
                "source_passages": [],
                "warnings": [],
                "llm_called": False,
            }
        if (
            question
            == "I have felt low for weeks and cannot manage classes. Is this just spiritual weakness?"
        ):
            return {
                "mode": "human_support",
                "answer": "Fixture: seek qualified human support for persistent distress.",
                "citations": [],
                "source_passages": [],
                "warnings": [],
                "llm_called": False,
            }
        if question == "Does my child have ADHD because she cannot focus in class?":
            return {
                "mode": "human_support",
                "answer": "Fixture: qualified human assessment, not diagnosis from philosophy.",
                "citations": [],
                "source_passages": [],
                "warnings": [],
                "llm_called": False,
            }
        if question == "Give the Mother's exact quote proving that I will become rich next month.":
            return {
                "mode": "insufficient_sources",
                "answer": "Fixture: no invented personal wealth prediction.",
                "citations": [],
                "source_passages": [],
                "warnings": [],
                "llm_called": False,
            }
        if self.raise_error:
            raise RuntimeError("INTERNAL_FIXTURE_SECRET: do not send exceptions to clients")
        self.chat_started.set()
        if self.block_chat:
            if not self.release_chat.wait(5):
                raise RuntimeError("Fixture wait timed out")
        if self.result_override is not None:
            return self.result_override
        return {
            "mode": "prompt_preview" if dry_run else "local_rag",
            "answer": "Fixture response.",
            "citations": [],
            "source_passages": [],
            "warnings": [],
            "llm_called": not dry_run,
            "model": "fixture-local-model",
        }


class ServerContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = SERVER.make_server(
            self.engine,
            host="127.0.0.1",
            port=0,
            allowed_origins=("http://localhost:9999",),
            max_body_bytes=4096,
        )
        self.port = self.server.server_address[1]
        self.worker = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.worker.start()

    def tearDown(self):
        self.engine.release_chat.set()
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=3)

    def http(self, method, path, body=None, headers=None, raw=False, omit_length=False):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=4)
        body_bytes = (
            body
            if raw
            else (None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"))
        )
        if isinstance(body_bytes, str):
            body_bytes = body_bytes.encode("utf-8")
        custom_headers = dict(headers or {})
        if method == "POST" and "Content-Type" not in custom_headers:
            custom_headers["Content-Type"] = "application/json"
        connection.putrequest(method, path, skip_host=True)
        connection.putheader("Host", custom_headers.pop("Host", f"127.0.0.1:{self.port}"))
        for name, value in custom_headers.items():
            connection.putheader(name, value)
        if body_bytes is not None and not omit_length and "Content-Length" not in custom_headers:
            connection.putheader("Content-Length", str(len(body_bytes)))
        connection.endheaders(body_bytes)
        response = connection.getresponse()
        data = response.read()
        result_headers = dict(response.getheaders())
        status = response.status
        connection.close()
        parsed = (
            (
                json.loads(data.decode("utf-8"))
                if "application/json" in result_headers.get("Content-Type", "")
                else data.decode("utf-8")
            )
            if data
            else None
        )
        return status, result_headers, parsed

    def assert_error(self, result, expected_status):
        status, headers, data = result
        self.assertEqual(status, expected_status, data)
        self.assertFalse(data["ok"])
        self.assertIsInstance(data["request_id"], str)
        self.assertTrue(data["request_id"])
        self.assertIsInstance(data["error"]["code"], str)
        self.assertIsInstance(data["error"]["message"], str)
        self.assertIsInstance(data["error"]["retryable"], bool)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        return data

    def test_health_calls_local_engine_status(self):
        status, headers, data = self.http("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["model"], "fixture-local-model")
        self.assertTrue(data["request_id"])
        self.assertEqual(self.engine.calls, [("status", True)])
        self.assertIn("no-store", headers.get("Cache-Control", ""))

    def test_search_preserves_unicode_and_limit(self):
        query = "शिक्षा और concentration"
        status, _, data = self.http("POST", "/api/search", {"query": query, "limit": 3})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["ok"])
        self.assertEqual(self.engine.calls, [("search", query, 3)])

    def test_chat_preserves_history_and_dry_run(self):
        history = [
            {"role": "user", "content": "I am sad."},
            {"role": "assistant", "content": "Could you tell me more?"},
        ]
        status, _, data = self.http(
            "POST", "/api/chat", {"message": "An exam.", "history": history, "dry_run": True}
        )
        self.assertEqual(status, 200, data)
        self.assertFalse(data["llm_called"])
        self.assertEqual(self.engine.calls, [("chat", "An exam.", history, True)])

    def test_unknown_json_keys_rejected_before_engine(self):
        self.assert_error(
            self.http("POST", "/api/chat", {"message": "Hello", "model": "remote-cloud"}), 400
        )
        self.assertEqual(self.engine.calls, [])

    def test_invalid_field_types_rejected(self):
        bodies = [
            {"message": 42},
            {"message": "hello", "history": "not a list"},
            {"message": "hello", "dry_run": "true"},
        ]
        for body in bodies:
            with self.subTest(body=body):
                self.assert_error(self.http("POST", "/api/chat", body), 400)
        self.assertEqual(self.engine.calls, [])

    def test_search_boolean_limit_not_treated_as_integer(self):
        self.assert_error(self.http("POST", "/api/search", {"query": "peace", "limit": True}), 400)
        self.assertEqual(self.engine.calls, [])

    def test_non_object_json_rejected(self):
        for body in [[], "string", None]:
            with self.subTest(body=body):
                raw_body = json.dumps(body).encode("utf-8")
                self.assert_error(self.http("POST", "/api/chat", raw_body, raw=True), 400)
        self.assertEqual(self.engine.calls, [])

    def test_malformed_json_rejected(self):
        self.assert_error(self.http("POST", "/api/chat", b'{"message":', raw=True), 400)
        self.assertEqual(self.engine.calls, [])

    def test_content_type_required(self):
        self.assert_error(
            self.http(
                "POST", "/api/chat", {"message": "Hello"}, headers={"Content-Type": "text/plain"}
            ),
            415,
        )
        self.assertEqual(self.engine.calls, [])

    def test_missing_content_length_rejected(self):
        self.assert_error(self.http("POST", "/api/chat", b"{}", raw=True, omit_length=True), 411)
        self.assertEqual(self.engine.calls, [])

    def test_oversize_body_rejected_before_engine(self):
        self.assert_error(self.http("POST", "/api/chat", {"message": "x" * 5000}), 413)
        self.assertEqual(self.engine.calls, [])

    def test_transfer_encoding_not_supported(self):
        self.assert_error(
            self.http(
                "POST",
                "/api/chat",
                b"0\r\n\r\n",
                raw=True,
                headers={"Transfer-Encoding": "chunked"},
                omit_length=True,
            ),
            400,
        )
        self.assertEqual(self.engine.calls, [])

    def test_deep_json_returns_bounded_error(self):
        nested = b"[" * 1200 + b"0" + b"]" * 1200
        self.assert_error(self.http("POST", "/api/chat", nested, raw=True), 400)
        self.assertEqual(self.engine.calls, [])

    def test_engine_validation_errors_map_to_client_errors(self):
        for code in ["question_too_long", "history_too_long", "prompt_too_long", "invalid_request"]:
            with self.subTest(code=code):
                self.engine.result_override = {
                    "mode": "error",
                    "error": {"code": code, "message": "Fixture invalid input", "retryable": False},
                }
                self.assert_error(self.http("POST", "/api/chat", {"message": "hello"}), 400)

    def test_engine_timeout_maps_to_gateway_timeout(self):
        self.engine.result_override = {
            "mode": "error",
            "error": {"code": "model_timeout", "message": "Fixture timeout", "retryable": True},
        }
        self.assert_error(self.http("POST", "/api/chat", {"message": "hello"}), 504)

    def test_search_limit_exceeds_engine_contract(self):
        self.assert_error(self.http("POST", "/api/search", {"query": "peace", "limit": 7}), 400)
        self.assertEqual(self.engine.calls, [])

    def test_foreign_origin_rejected(self):
        self.assert_error(
            self.http(
                "POST",
                "/api/chat",
                {"message": "Hello"},
                headers={"Origin": "https://example.invalid"},
            ),
            403,
        )
        self.assertEqual(self.engine.calls, [])

    def test_rebinding_host_rejected(self):
        self.assert_error(
            self.http("GET", "/health", headers={"Host": "attacker.example:1234"}), 403
        )
        self.assertEqual(self.engine.calls, [])

    def test_wrong_loopback_port_rejected(self):
        self.assert_error(
            self.http("GET", "/health", headers={"Host": f"localhost:{self.port + 1}"}), 403
        )
        self.assertEqual(self.engine.calls, [])

    def test_explicit_allowed_origin(self):
        status, headers, data = self.http(
            "POST", "/api/search", {"query": "peace"}, headers={"Origin": "http://localhost:9999"}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:9999")
        self.assertNotEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertNotIn("Access-Control-Allow-Credentials", headers)

    def test_server_own_origin(self):
        origin = f"http://127.0.0.1:{self.port}"
        status, headers, data = self.http(
            "POST", "/api/search", {"query": "peace"}, headers={"Origin": origin}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), origin)

    def test_preflight_and_foreign_preflight(self):
        status, headers, data = self.http(
            "OPTIONS",
            "/api/chat",
            headers={
                "Origin": "http://localhost:9999",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        self.assertEqual(status, 204)
        self.assertIsNone(data)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:9999")
        self.assert_error(
            self.http("OPTIONS", "/api/chat", headers={"Origin": "https://example.invalid"}), 403
        )

    def test_arbitrary_file_and_unknown_route_not_served(self):
        for path in ["/knowledge_base.jsonl", "/../../local_llm_config.json", "/api/unknown"]:
            with self.subTest(path=path):
                self.assert_error(self.http("GET", path), 404)
        self.assertEqual(self.engine.calls, [])

    def test_busy_generation_is_bounded(self):
        self.engine.block_chat = True
        first_result = []
        first = threading.Thread(
            target=lambda: first_result.append(self.http("POST", "/api/chat", {"message": "First"}))
        )
        first.start()
        self.assertTrue(self.engine.chat_started.wait(2))
        result = self.http("POST", "/api/chat", {"message": "Second"})
        self.assert_error(result, 503)
        self.assertTrue(result[1].get("Retry-After"))
        self.engine.release_chat.set()
        first.join(timeout=3)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result[0][0], 200)
        self.assertEqual(len(self.engine.calls), 1)

    def test_urgent_support_bypasses_busy_generation_and_old_history(self):
        self.engine.block_chat = True
        first_result = []
        first = threading.Thread(
            target=lambda: first_result.append(self.http("POST", "/api/chat", {"message": "First"}))
        )
        first.start()
        try:
            self.assertTrue(self.engine.chat_started.wait(2))
            status, _, urgent = self.http(
                "POST",
                "/api/chat",
                {
                    "message": "I want to harm myself",
                    "history": "invalid old history",
                    "dry_run": "invalid old setting",
                },
            )
            self.assertEqual(status, 200, urgent)
            self.assertEqual(urgent["mode"], "urgent_support")
            self.assertFalse(urgent["llm_called"])
            self.assertEqual(urgent["citations"], [])
            ordinary = self.http("POST", "/api/chat", {"message": "Second ordinary question"})
            self.assert_error(ordinary, 503)
            self.assertTrue(ordinary[1].get("Retry-After"))
        finally:
            self.engine.release_chat.set()
            first.join(timeout=3)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result[0][0], 200)
        self.assertEqual(
            [call[1] for call in self.engine.calls], ["First", "I want to harm myself"]
        )

    def test_treatment_boundary_bypasses_busy_generation_and_old_history(self):
        self.engine.block_chat = True
        first_result = []
        first = threading.Thread(
            target=lambda: first_result.append(self.http("POST", "/api/chat", {"message": "First"}))
        )
        first.start()
        medication_question = "I take medicine for depression. Should I stop it and only meditate?"
        try:
            self.assertTrue(self.engine.chat_started.wait(2))
            status, _, boundary = self.http(
                "POST",
                "/api/chat",
                {
                    "message": medication_question,
                    "history": "invalid old history",
                    "dry_run": "invalid old setting",
                },
            )
            self.assertEqual(status, 200, boundary)
            self.assertEqual(boundary["mode"], "treatment_boundary")
            self.assertFalse(boundary["llm_called"])
            self.assertEqual(boundary["citations"], [])
            ordinary = self.http("POST", "/api/chat", {"message": "Second ordinary question"})
            self.assert_error(ordinary, 503)
            self.assertTrue(ordinary[1].get("Retry-After"))
        finally:
            self.engine.release_chat.set()
            first.join(timeout=3)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result[0][0], 200)
        self.assertEqual([call[1] for call in self.engine.calls], ["First", medication_question])

    def assert_fixed_route_bypasses_busy_generation(self, question, expected_mode):
        self.engine.block_chat = True
        first_result = []
        first = threading.Thread(
            target=lambda: first_result.append(self.http("POST", "/api/chat", {"message": "First"}))
        )
        first.start()
        try:
            self.assertTrue(self.engine.chat_started.wait(2))
            status, _, response = self.http(
                "POST",
                "/api/chat",
                {
                    "message": question,
                    "history": "invalid old history",
                    "dry_run": "invalid old setting",
                },
            )
            self.assertEqual(status, 200, response)
            self.assertEqual(response["mode"], expected_mode)
            self.assertFalse(response["llm_called"])
            self.assertEqual(response["citations"], [])
            self.assertEqual(response["source_passages"], [])
            ordinary = self.http("POST", "/api/chat", {"message": "Second ordinary question"})
            self.assert_error(ordinary, 503)
            self.assertTrue(ordinary[1].get("Retry-After"))
        finally:
            self.engine.release_chat.set()
            first.join(timeout=3)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result[0][0], 200)
        self.assertEqual([call[1] for call in self.engine.calls], ["First", question])

    def test_persistent_distress_bypasses_busy_generation_and_old_history(self):
        self.assert_fixed_route_bypasses_busy_generation(
            "I have felt low for weeks and cannot manage classes. Is this just spiritual weakness?",
            "human_support",
        )

    def test_diagnosis_boundary_bypasses_busy_generation_and_old_history(self):
        self.assert_fixed_route_bypasses_busy_generation(
            "Does my child have ADHD because she cannot focus in class?", "human_support"
        )

    def test_withheld_help_and_sources_survive_non_success_http_status(self):
        for mode in ("support_response_withheld", "source_claim_withheld"):
            with self.subTest(mode=mode):
                self.engine.result_override = {
                    "mode": mode,
                    "answer": "Fixture safe fallback, not rejected model text.",
                    "source_passages": [{"text": "Fixture evidence."}],
                    "citations": [],
                    "llm_called": True,
                    "warnings": ["Fixture withholding."],
                    "error": {"code": mode, "message": "Fixture withheld", "retryable": False},
                }
                status, _, data = self.http("POST", "/api/chat", {"message": "hello"})
                self.assertEqual(status, 503)
                self.assertFalse(data["ok"])
                self.assertEqual(data["answer"], self.engine.result_override["answer"])
                self.assertEqual(
                    data["source_passages"], self.engine.result_override["source_passages"]
                )
                self.assertFalse(data["error"]["retryable"])

    def test_unsupported_wealth_prediction_bypasses_busy_generation_and_old_history(self):
        self.assert_fixed_route_bypasses_busy_generation(
            "Give the Mother's exact quote proving that I will become rich next month.",
            "insufficient_sources",
        )

    def test_recent_urgent_followup_bypasses_busy_generation_with_valid_history(self):
        self.engine.block_chat = True
        first_result = []
        first = threading.Thread(
            target=lambda: first_result.append(self.http("POST", "/api/chat", {"message": "First"}))
        )
        first.start()
        try:
            self.assertTrue(self.engine.chat_started.wait(2))
            status, _, response = self.http(
                "POST",
                "/api/chat",
                {
                    "message": "What should I do now?",
                    "history": RECENT_URGENT_HISTORY,
                    "dry_run": False,
                },
            )
            self.assertEqual(status, 200, response)
            self.assertEqual(response["mode"], "urgent_support")
            self.assertFalse(response["llm_called"])
            self.assertEqual(response["citations"], [])
            self.assertEqual(response["source_passages"], [])
            ordinary = self.http("POST", "/api/chat", {"message": "Second ordinary question"})
            self.assert_error(ordinary, 503)
        finally:
            self.engine.release_chat.set()
            first.join(timeout=3)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result[0][0], 200)
        self.assertEqual(
            [call[1] for call in self.engine.calls], ["First", "What should I do now?"]
        )

    def test_recent_urgent_followup_with_invalid_raw_history_is_not_a_special_route(self):
        self.assert_error(
            self.http(
                "POST",
                "/api/chat",
                {"message": "What should I do now?", "history": "invalid raw history"},
            ),
            400,
        )
        self.assertEqual(self.engine.calls, [])

    def test_internal_exception_not_exposed(self):
        self.engine.raise_error = True
        data = self.assert_error(self.http("POST", "/api/chat", {"message": "Hello"}), 503)
        self.assertNotIn("INTERNAL_FIXTURE_SECRET", json.dumps(data))
        self.assertNotIn("Traceback", json.dumps(data))

    def test_public_binding_refused(self):
        with self.assertRaises((ValueError, OSError)):
            SERVER.make_server(FakeEngine(), host="0.0.0.0", port=0)

    def test_service_discovery_and_liveness_do_not_touch_engine(self):
        status, _, data = self.http("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(data["endpoints"]["chat"], "/api/v1/chat")
        self.assertFalse(data["privacy"]["conversation_storage"])
        self.assertFalse(data["student_facing_release_approved"])
        status, _, data = self.http("GET", "/health/live")
        self.assertEqual(status, 200)
        self.assertTrue(data["live"])
        self.assertEqual(self.engine.calls, [])

    def test_readiness_uses_actual_dependencies(self):
        status, _, data = self.http("GET", "/health/ready")
        self.assertEqual(status, 200)
        self.assertTrue(data["ready"])
        self.assertEqual(self.engine.calls, [("status", True)])

    def test_readiness_unavailable_is_503_but_status_is_200(self):
        self.engine.status_override = {
            "ready": False,
            "index_ready": True,
            "model_ready": False,
            "error": None,
        }
        data = self.assert_error(self.http("GET", "/health/ready"), 503)
        self.assertFalse(data["ready"])
        self.assertEqual(data["error"]["code"], "service_not_ready")
        for path in ("/health", "/api/v1/status"):
            with self.subTest(path=path):
                status, _, data = self.http("GET", path)
                self.assertEqual(status, 200)
                self.assertFalse(data["ready"])

    def test_readiness_exception_is_sanitized(self):
        self.engine.raise_status = True
        data = self.assert_error(self.http("GET", "/health/ready"), 503)
        self.assertNotIn("INTERNAL_FIXTURE_SECRET", json.dumps(data))

    def test_docs_are_passive_html_with_security_headers(self):
        status, headers, html = self.http("GET", "/docs")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("<iframe", html.lower())
        self.assertIn(f"http://127.0.0.1:{self.port}", html)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertEqual(self.engine.calls, [])

    def test_openapi_is_raw_spec_with_current_base_url(self):
        status, headers, data = self.http("GET", "/api/v1/openapi.json")
        self.assertEqual(status, 200)
        self.assertEqual(data["openapi"], "3.1.0")
        self.assertNotIn("ok", data)
        self.assertNotIn("request_id", data)
        self.assertEqual(data["servers"][0]["url"], f"http://127.0.0.1:{self.port}")
        self.assertIn("X-Request-ID", headers)
        self.assertEqual(set(data["paths"]), set(SERVER.GET_PATHS | SERVER.POST_PATHS))
        self.assertEqual(self.engine.calls, [])

    def test_versioned_chat_and_search_preserve_engine_contract(self):
        status, _, data = self.http("POST", "/api/v1/chat", {"message": "Hello", "dry_run": True})
        self.assertEqual(status, 200)
        self.assertFalse(data["llm_called"])
        status, _, data = self.http("POST", "/api/v1/search", {"query": "peace", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(self.engine.calls, [("chat", "Hello", [], True), ("search", "peace", 2)])

    def test_versioned_alias_shares_legacy_generation_slot(self):
        self.engine.block_chat = True
        first_result = []
        first = threading.Thread(
            target=lambda: first_result.append(self.http("POST", "/api/chat", {"message": "First"}))
        )
        first.start()
        try:
            self.assertTrue(self.engine.chat_started.wait(2))
            data = self.assert_error(self.http("POST", "/api/v1/chat", {"message": "Second"}), 503)
            self.assertEqual(data["error"]["code"], "busy")
            status, _, data = self.http(
                "POST",
                "/api/v1/chat",
                {"message": "I want to harm myself", "history": "old malformed history"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(data["mode"], "urgent_support")
            self.assertFalse(data["llm_called"])
            status, _, _ = self.http("POST", "/api/v1/search", {"query": "peace"})
            self.assertEqual(status, 200)
            status, _, _ = self.http("GET", "/health/live")
            self.assertEqual(status, 200)
        finally:
            self.engine.release_chat.set()
            first.join(timeout=3)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result[0][0], 200)

    def test_versioned_errors_have_same_validation_and_host_origin_checks(self):
        for path, body in (
            ("/api/v1/chat", {"message": "x", "model": "remote"}),
            ("/api/v1/search", {"query": "x", "limit": True}),
        ):
            with self.subTest(path=path):
                self.assert_error(self.http("POST", path, body), 400)
                self.assert_error(
                    self.http("POST", path, body, headers={"Origin": "https://foreign.example"}),
                    403,
                )
        for path in SERVER.GET_PATHS:
            with self.subTest(path=path):
                self.assert_error(self.http("GET", path, headers={"Host": "rebind.example"}), 403)
        self.assertEqual(self.engine.calls, [])

    def test_wrong_method_returns_json_405_and_allow(self):
        for method, path in (
            ("GET", "/api/v1/chat"),
            ("POST", "/health/live"),
            ("DELETE", "/docs"),
        ):
            with self.subTest(method=method, path=path):
                result = self.http(method, path, {})
                self.assert_error(result, 405)
                self.assertIn("Allow", result[1])
        self.assertEqual(self.engine.calls, [])

    def test_versioned_preflight_reports_route_specific_method(self):
        for path in SERVER.GET_PATHS | SERVER.POST_PATHS:
            method = "GET" if path in SERVER.GET_PATHS else "POST"
            with self.subTest(path=path):
                status, headers, data = self.http(
                    "OPTIONS",
                    path,
                    headers={
                        "Origin": "http://localhost:9999",
                        "Access-Control-Request-Method": method,
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
                self.assertEqual(status, 204)
                self.assertIsNone(data)
                self.assertEqual(headers["Access-Control-Allow-Methods"], method + ", OPTIONS")
        self.assertEqual(self.engine.calls, [])

    def test_disallowed_preflight_method_or_header_is_rejected(self):
        self.assert_error(
            self.http(
                "OPTIONS", "/api/v1/chat", headers={"Access-Control-Request-Method": "DELETE"}
            ),
            405,
        )
        self.assert_error(
            self.http(
                "OPTIONS",
                "/api/v1/chat",
                headers={"Access-Control-Request-Headers": "Authorization"},
            ),
            400,
        )

    def test_nonfinite_or_duplicate_json_is_rejected(self):
        for body in (
            b'{"message":"first","message":"second"}',
            b'{"query":"peace","limit":NaN}',
            b'{"message":"I want to harm myself","history":[1e309]}',
        ):
            with self.subTest(body=body):
                self.assert_error(self.http("POST", "/api/v1/chat", body, raw=True), 400)
        self.assertEqual(self.engine.calls, [])

    def test_excessive_length_header_has_bounded_error(self):
        result = self.http(
            "POST", "/api/v1/chat", b"{}", raw=True, headers={"Content-Length": "1" * 4500}
        )
        self.assert_error(result, 400)
        self.assertEqual(self.engine.calls, [])

    def test_non_utf8_declared_charset_rejected(self):
        self.assert_error(
            self.http(
                "POST",
                "/api/v1/chat",
                {"message": "Hello"},
                headers={"Content-Type": "application/json; charset=utf-16"},
            ),
            415,
        )

    def test_malformed_request_line_returns_bounded_json(self):
        connection = socket.create_connection(("127.0.0.1", self.port), timeout=3)
        try:
            connection.sendall(b"GET / HTTP/9.9\r\n\r\n")
            result = bytearray()
            while True:
                chunk = connection.recv(8192)
                if not chunk:
                    break
                result.extend(chunk)
        finally:
            connection.close()
        self.assertIn(b'"code": "http_request_error"', result)
        self.assertNotIn(b"<!DOCTYPE HTML>", result)
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
