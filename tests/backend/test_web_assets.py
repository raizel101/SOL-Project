"""Fixed local frontend serving with fake engine and zero inference."""

from __future__ import annotations

import http.client
import threading
import unittest
from pathlib import Path
from unittest import mock

from sol_chat.api import server, web_assets


class UntouchedEngine:
    def __getattr__(self, name):
        raise AssertionError("Static frontend serving must not call engine methods.")


class WebAssetsTests(unittest.TestCase):
    def setUp(self):
        self.server = server.make_server(UntouchedEngine(), port=0)
        self.worker = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=3)

    def request(self, path, method="GET", host=None, origin=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            headers = {"Host": host or f"127.0.0.1:{self.server.server_port}"}
            if origin:
                headers["Origin"] = origin
            connection.request(method, path, headers=headers)
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()

    def test_four_fixed_assets_have_correct_media_types_and_security(self):
        for path, (_, content_type) in web_assets.ASSETS.items():
            with self.subTest(path=path):
                status, headers, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertEqual(dict(headers)["Content-Type"], content_type)
                self.assertTrue(body)
                self.assertEqual(dict(headers)["Cache-Control"], "no-store")
                self.assertEqual(dict(headers)["X-Content-Type-Options"], "nosniff")
                policies = [
                    value for key, value in headers if key.lower() == "content-security-policy"
                ]
                self.assertEqual(policies, [web_assets.UI_CSP])
                self.assertIn("script-src 'self'", policies[0])
                self.assertIn("connect-src 'self'", policies[0])
                self.assertNotIn("unsafe-inline", policies[0])

    def test_only_explicit_paths_are_served(self):
        for path in (
            "/assets/../backend_config.json",
            "/assets/%2e%2e/server.py",
            "/web/index.html",
            "/chat?x=1",
            "/chat/",
            "/assets/app.js?x=1",
            "/assets/backend_config.json",
            "/web/",
            "/assets/",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)

    def test_frontend_still_requires_exact_local_host_and_origin(self):
        for path in web_assets.UI_PATHS:
            with self.subTest(path=path):
                self.assertEqual(self.request(path, host="rebind.invalid")[0], 403)
                self.assertEqual(self.request(path, origin="https://foreign.invalid")[0], 403)

    def test_post_is_method_error_and_preflight_is_get_only(self):
        for path in web_assets.UI_PATHS:
            with self.subTest(path=path):
                self.assertEqual(self.request(path, method="POST")[0], 405)
                status, headers, _ = self.request(path, method="OPTIONS")
                self.assertEqual(status, 204)
                self.assertEqual(dict(headers)["Access-Control-Allow-Methods"], "GET, OPTIONS")

    def test_missing_asset_returns_sanitized_json_not_filesystem_path(self):
        with mock.patch.object(
            server, "read_asset", side_effect=FileNotFoundError("PRIVATE_LOCAL_PATH")
        ):
            status, headers, body = self.request("/chat")
        self.assertEqual(status, 503)
        self.assertIn("application/json", dict(headers)["Content-Type"])
        self.assertIn(b'"code": "asset_unavailable"', body)
        self.assertNotIn(b"PRIVATE_LOCAL_PATH", body)

    def test_asset_reader_never_joins_arbitrary_paths(self):
        with mock.patch.object(Path, "open") as opener:
            with self.assertRaises(KeyError):
                web_assets.read_asset("/assets/../../server.py")
            opener.assert_not_called()

    def test_asset_reader_enforces_size_and_utf8(self):
        for payload in (b"x" * (web_assets.MAX_ASSET_BYTES + 1), b"\xff"):
            with (
                self.subTest(length=len(payload)),
                mock.patch.object(Path, "open", mock.mock_open(read_data=payload)),
            ):
                with self.assertRaises((ValueError, UnicodeError)):
                    web_assets.read_asset("/chat")

    def test_docs_remain_passive_and_script_disabled(self):
        status, headers, body = self.request("/docs")
        self.assertEqual(status, 200)
        self.assertNotIn(b"<script", body.lower())
        policy = dict(headers)["Content-Security-Policy"]
        self.assertNotIn("script-src 'self'", policy)
        self.assertIn("default-src 'none'", policy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
