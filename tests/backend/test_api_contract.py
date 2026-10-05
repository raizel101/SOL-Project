"""Offline contract/documentation checks; no model, corpus reads or HTTP calls."""

from __future__ import annotations

import unittest
from html.parser import HTMLParser

from sol_chat.api import contract as CONTRACT
from sol_chat.api import server as SERVER
from sol_chat.core.engine import Settings


class PassiveHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.styles = []
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, attrs))
        if tag == "style":
            self.in_style = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.styles.append(data)


class APIContractTests(unittest.TestCase):
    def test_openapi_version_and_local_server_address(self):
        spec = CONTRACT.build_openapi("http://localhost:9123/")
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertEqual(spec["info"]["version"], SERVER.API_VERSION)
        self.assertEqual(spec["servers"][0]["url"], "http://localhost:9123")
        self.assertEqual(spec["security"], [])

    def test_callers_cannot_mutate_subsequent_contracts(self):
        first = CONTRACT.build_openapi("http://localhost:9123")
        first["servers"][0]["url"] = "http://remote.invalid:80"
        first["paths"].clear()
        first["components"]["schemas"]["ChatRequest"]["properties"]["history"]["prefixItems"][0][
            "properties"
        ]["role"]["const"] = "system"
        second = CONTRACT.build_openapi()
        self.assertEqual(second["servers"][0]["url"], CONTRACT.DEFAULT_BASE_URL)
        self.assertTrue(second["paths"])
        self.assertEqual(
            second["components"]["schemas"]["ChatRequest"]["properties"]["history"]["prefixItems"][
                0
            ]["properties"]["role"]["const"],
            "user",
        )
        self.assertIsNot(first, second)
        self.assertIsNot(first["components"], second["components"])

    def test_all_references_resolve_and_operation_ids_are_unique(self):
        spec = CONTRACT.build_openapi()
        references, operation_ids = [], []

        def visit(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    references.append(value["$ref"])
                if "operationId" in value:
                    operation_ids.append(value["operationId"])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(spec)
        self.assertTrue(references)
        for reference in references:
            self.assertTrue(reference.startswith("#/"), reference)
            target = spec
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
            self.assertIsInstance(target, dict)
        self.assertEqual(len(operation_ids), len(set(operation_ids)))
        self.assertEqual(len(operation_ids), len(spec["paths"]))

    def test_spec_routes_and_methods_match_server_registry(self):
        paths = CONTRACT.build_openapi()["paths"]
        self.assertEqual(set(paths), SERVER.GET_PATHS | SERVER.POST_PATHS)
        for path in SERVER.GET_PATHS:
            self.assertEqual(set(paths[path]), {"get"}, path)
        for path in SERVER.POST_PATHS:
            self.assertEqual(set(paths[path]), {"post"}, path)
        self.assertEqual(SERVER.POST_PATHS, SERVER.CHAT_PATHS | SERVER.SEARCH_PATHS)

    def test_local_ui_routes_publish_fixed_media_types_and_json_errors(self):
        paths = CONTRACT.build_openapi()["paths"]
        media_types = {
            "/chat": "text/html",
            "/assets/styles.css": "text/css",
            "/assets/app.js": "text/javascript",
            "/assets/client-core.js": "text/javascript",
        }
        for path, media_type in media_types.items():
            with self.subTest(path=path):
                operation = paths[path]["get"]
                self.assertEqual(set(operation["responses"]["200"]["content"]), {media_type})
                self.assertEqual(
                    operation["responses"]["200"]["content"][media_type]["schema"],
                    {"type": "string"},
                )
                for code in ("403", "405", "503"):
                    self.assertEqual(
                        operation["responses"][code]["content"]["application/json"]["schema"],
                        {"$ref": "#/components/schemas/ErrorResponse"},
                    )
                self.assertIn("asset_unavailable", operation["responses"]["503"]["description"])
                self.assertIn("Allow", operation["responses"]["405"]["headers"])
        self.assertIn("application/json", paths["/"]["get"]["responses"]["200"]["content"])
        self.assertEqual(CONTRACT.API_VERSION, "1.0.0")
        self.assertIn("no accounts, browser storage", paths["/chat"]["get"]["description"])
        self.assertIn("not approved for student release", paths["/chat"]["get"]["description"])

    def test_history_schema_matches_default_engine_caps_and_roles(self):
        settings = Settings()
        history = CONTRACT.build_openapi()["components"]["schemas"]["ChatRequest"]["properties"][
            "history"
        ]
        self.assertEqual(history["maxItems"], settings.max_history_messages)
        self.assertEqual(history["x-max-total-content-characters"], settings.max_history_characters)
        self.assertEqual(len(history["prefixItems"]), settings.max_history_messages)
        self.assertIs(history["items"], False)
        self.assertEqual(
            [(entry["minItems"], entry["maxItems"]) for entry in history["anyOf"]],
            [(n, n) for n in (0, 2, 4, 6, 8)],
        )
        for index, entry in enumerate(history["prefixItems"]):
            self.assertEqual(entry["required"], ["role", "content"])
            self.assertIs(entry["additionalProperties"], False)
            self.assertEqual(
                entry["properties"]["role"]["const"], "user" if index % 2 == 0 else "assistant"
            )
            self.assertEqual(
                entry["properties"]["content"]["maxLength"], settings.max_history_message_characters
            )
        self.assertIn("Configuration may tighten", history["description"])

    def test_requests_publish_strict_fields_character_and_search_caps(self):
        schemas = CONTRACT.build_openapi()["components"]["schemas"]
        chat, search = schemas["ChatRequest"], schemas["SearchRequest"]
        self.assertEqual(set(chat["properties"]), {"message", "history", "dry_run"})
        self.assertEqual(set(search["properties"]), {"query", "limit"})
        self.assertEqual(chat["required"], ["message"])
        self.assertEqual(search["required"], ["query"])
        self.assertIs(chat["additionalProperties"], False)
        self.assertIs(search["additionalProperties"], False)
        self.assertEqual(chat["properties"]["message"]["maxLength"], 1200)
        self.assertEqual(search["properties"]["query"]["maxLength"], 1200)
        self.assertEqual(chat["properties"]["dry_run"]["type"], "boolean")
        self.assertEqual(search["properties"]["limit"]["type"], "integer")
        self.assertEqual(search["properties"]["limit"]["minimum"], 1)
        self.assertEqual(search["properties"]["limit"]["maximum"], 6)

    def test_html_has_no_active_or_remote_resources(self):
        base_url = "http://127.0.0.1:9123"
        html = CONTRACT.render_docs(base_url)
        parser = PassiveHTMLParser()
        parser.feed(html)
        links = []
        forbidden_tags = {
            "script",
            "iframe",
            "frame",
            "img",
            "link",
            "form",
            "object",
            "embed",
            "video",
            "audio",
            "source",
        }
        for tag, attrs in parser.elements:
            self.assertNotIn(tag, forbidden_tags)
            for name, value in attrs:
                self.assertFalse(name.lower().startswith("on"), name)
                self.assertNotIn(name, {"src", "srcset", "action", "formaction", "data"})
                if name == "href":
                    links.append(value)
                    self.assertEqual(value, base_url + "/api/v1/openapi.json")
                if name == "http-equiv":
                    self.assertNotEqual(value.lower(), "refresh")
        self.assertEqual(links, [base_url + "/api/v1/openapi.json"])
        for style in parser.styles:
            self.assertNotIn("@import", style.lower())
            self.assertNotIn("url(", style.lower())
        self.assertIn("&quot;openapi&quot;", html)
        self.assertIn("passive API documentation", html)
        self.assertIn("<code>/chat</code>", html)
        self.assertIn("not approved for student release", html)
        self.assertIn("Python Unicode characters", html)
        self.assertIn("never truncates messages", html)
        self.assertIn("reset before the next message", html)
        self.assertIn("not promoted to accepted conversation history", html)
        self.assertIn("does not cancel server model work", html)

    def test_unsafe_or_ambiguous_documentation_urls_are_rejected(self):
        unsafe = [
            None,
            True,
            8765,
            "",
            "https://localhost:8765",
            "http://example.com:8765",
            "http://localhost",
            "http://localhost:0",
            "http://localhost:65536",
            "http://localhost:bad",
            "http://localhost:8765/path",
            "http://localhost:8765/?query=1",
            "http://localhost:8765/#fragment",
            "http://user@localhost:8765",
            "http://user:password@localhost:8765",
            "http://[::1]:8765",
            "http://@localhost:8765",
            "http://localhost:",
            " http://localhost:8765",
            "http://localhost:8765\n",
        ]
        for value in unsafe:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CONTRACT.build_openapi(value)
                with self.assertRaises(ValueError):
                    CONTRACT.render_docs(value)

    def test_withheld_modes_error_contract_and_exact_offset_semantics(self):
        spec = CONTRACT.build_openapi()
        schemas = spec["components"]["schemas"]
        modes = schemas["ChatResponse"]["properties"]["mode"]["enum"]
        self.assertEqual(modes, list(CONTRACT.MODES))
        for mode in (
            "citation_check_failed",
            "support_response_withheld",
            "source_claim_withheld",
            "incomplete_response",
        ):
            self.assertIn(mode, modes)
        error_response = schemas["ErrorResponse"]
        self.assertEqual(error_response["properties"]["ok"], {"const": False})
        self.assertIn("error", error_response["required"])
        response_503 = spec["paths"]["/api/v1/chat"]["post"]["responses"]["503"]
        self.assertIn("rejected model text is never returned", response_503["description"])
        self.assertIn("Retry-After", response_503["headers"])
        offsets = schemas["SourcePassage"]["properties"]["text_offsets"]
        self.assertEqual(offsets["required"], ["start", "end_exclusive"])
        self.assertIn("Unicode-character", offsets["description"])
        self.assertIn("UTF-16", offsets["description"])

    def test_method_and_strict_json_transport_are_documented(self):
        spec = CONTRACT.build_openapi()
        for path, operations in spec["paths"].items():
            for operation in operations.values():
                with self.subTest(path=path):
                    self.assertIn("405", operation["responses"])
                    self.assertIn("Allow", operation["responses"]["405"]["headers"])
        description = spec["info"]["description"]
        self.assertIn("Duplicate object keys", description)
        self.assertIn("NaN/Infinity", description)
        self.assertIn("exponent overflow", description)
        self.assertIn("charset", description)
        self.assertIn("Content-Encoding", description)
        self.assertIn("UTF-8", description)
        for path in SERVER.POST_PATHS:
            responses = spec["paths"][path]["post"]["responses"]
            self.assertIn("UTF-8 charset", responses["415"]["description"])

    def test_legacy_aliases_are_marked_without_deprecating_v1(self):
        paths = CONTRACT.build_openapi()["paths"]
        for path in ("/health", "/api/chat", "/api/search"):
            for operation in paths[path].values():
                self.assertIs(operation["deprecated"], True)
        for path in ("/api/v1/chat", "/api/v1/search", "/api/v1/status"):
            for operation in paths[path].values():
                self.assertFalse(operation.get("deprecated", False))

    def test_service_contract_does_not_claim_student_release_or_storage(self):
        schemas = CONTRACT.build_openapi()["components"]["schemas"]
        properties = schemas["ServiceResponse"]["properties"]
        self.assertEqual(properties["student_facing_release_approved"], {"const": False})
        self.assertEqual(
            properties["privacy"]["properties"]["conversation_storage"], {"const": False}
        )
        self.assertEqual(properties["privacy"]["properties"]["local_only"], {"const": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
