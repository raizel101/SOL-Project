"""Configuration validation without model calls, installs or corpus writes."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sol_chat.api import server
from sol_chat.config import ConfigError, load_config, local_origin, validate_config
from sol_chat.paths import PROJECT_ROOT


class BackendConfigTests(unittest.TestCase):
    def test_shipped_configuration_matches_local_runtime(self):
        value = load_config()
        self.assertEqual(value.host, "127.0.0.1")
        self.assertEqual(value.port, 8765)
        self.assertEqual(value.engine["model"], "sol-chat")
        self.assertEqual(value.engine["ollama_base_url"], "http://127.0.0.1:11434")
        self.assertEqual(value.allowed_origins, ())
        self.assertEqual(
            value.index_path,
            (PROJECT_ROOT / "outputs/auro_guide_corpus/local_rag.sqlite").resolve(),
        )

    def test_relative_path_resolves_against_config_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text('{"index_path":"data/index.sqlite"}', encoding="utf-8")
            value = load_config(path)
            self.assertEqual(value.index_path, (Path(folder) / "data/index.sqlite").resolve())
            self.assertFalse(value.index_path.exists())  # No index was created.

    def test_default_when_optional_config_is_absent(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("sol_chat.config.DEFAULT_CONFIG", Path(folder) / "absent.json"):
                self.assertEqual(load_config().port, 8765)

    def test_missing_explicit_file_is_bounded_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ConfigError):
                load_config(Path(folder) / "absent.json")

    def test_unknown_fields_and_non_objects_rejected(self):
        for value in (None, [], "value", {"secret": "do not echo"}):
            with self.subTest(value=value), self.assertRaises(ConfigError) as failure:
                validate_config(value)
            self.assertNotIn("do not echo", str(failure.exception))

    def test_wrong_schema_versions_rejected(self):
        for value in (True, False, 1.0, 2, None, "1"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config({"schema_version": value})

    def test_public_lan_and_ambiguous_bindings_rejected(self):
        for value in ("0.0.0.0", "localhost", "::1", "192.168.1.1", None):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config({"host": value})

    def test_invalid_ports_and_body_limits_rejected(self):
        for field, values in (
            ("port", (True, -1, 65536, "8765", 1.5, None)),
            ("max_body_bytes", (True, 1023, 65537, "4096", None)),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ConfigError):
                    validate_config({field: value})

    def test_ephemeral_port_and_bounded_body_are_valid(self):
        value = validate_config({"port": 0, "max_body_bytes": 1024})
        self.assertEqual(value.port, 0)
        self.assertEqual(value.max_body_bytes, 1024)

    def test_origins_must_be_bounded_list(self):
        for value in (None, "http://localhost:5173", {}, ["http://localhost:5173"] * 9):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config({"allowed_origins": value})

    def test_origins_normalize_and_deduplicate(self):
        value = validate_config(
            {"allowed_origins": ["http://localhost:5173/", "http://localhost:5173"]}
        )
        self.assertEqual(value.allowed_origins, ("http://localhost:5173",))

    def test_foreign_malformed_and_credential_origins_rejected(self):
        values = (
            "*",
            "https://example.com",
            "http://localhost:bad",
            "http://[localhost",
            "http://user@localhost:5173",
            "http://localhost:5173/x",
            "http://localhost:5173?x=1",
            "http://localhost:5173#fragment",
            "http://localhost:5173\n",
            "http://@localhost:5173",
            "http://localhost:",
            "http://localhost:0",
            3,
            "http://localhost:" + "1" * 201,
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                local_origin(value)

    def test_empty_invalid_and_control_character_paths_rejected(self):
        for value in (None, "", 8, "abc\x00.sqlite", "\ud800", "x" * 4097):
            with self.subTest(value=repr(value)), self.assertRaises(ConfigError):
                validate_config({"index_path": value})

    def test_unc_network_and_device_paths_rejected_before_open(self):
        for value in (
            "//remote.invalid/share/index.sqlite",
            "\\\\remote.invalid\\share\\index.sqlite",
            "\\\\?\\C:\\index.sqlite",
            "\\\\.\\C:\\index.sqlite",
        ):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config({"index_path": value})
            with self.subTest(config=value), mock.patch.object(Path, "open") as opener:
                with self.assertRaises(ConfigError):
                    load_config(value)
                opener.assert_not_called()

    def test_engine_wrong_container_and_unknown_settings_rejected(self):
        for value in (None, [], "value", {"unexpected": "private value"}):
            with self.subTest(value=value), self.assertRaises(ConfigError) as failure:
                validate_config({"engine": value})
            self.assertNotIn("private value", str(failure.exception))

    def test_remote_cloud_and_malformed_engine_endpoints_rejected(self):
        values = (
            {"ollama_base_url": "https://example.com"},
            {"model": "some-cloud"},
            {"ollama_base_url": "http://[localhost"},
            {"ollama_base_url": None},
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config({"engine": value})

    def test_nonfinite_and_boolean_engine_numbers_rejected(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config({"engine": {"temperature": value}})

    def test_engine_cannot_widen_transport_caps(self):
        for field, value in (
            ("max_question_characters", 1201),
            ("max_history_messages", 9),
            ("max_history_characters", 2001),
            ("max_history_message_characters", 1001),
        ):
            with self.subTest(field=field), self.assertRaises(ConfigError):
                validate_config({"engine": {field: value}})

    def test_engine_can_tighten_caps(self):
        self.assertEqual(
            validate_config({"engine": {"max_history_messages": 2}}).engine["max_history_messages"],
            2,
        )

    def test_invalid_json_duplicates_and_oversize_configs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for value in (
                b"{",
                b"\xff",
                b'{"port":1,"port":2}',
                b'{"engine":{"temperature":NaN}}',
                b'{"engine":{"temperature":1e309}}',
                b" " * 65537,
            ):
                with self.subTest(value=value[:64]), self.assertRaises(ConfigError):
                    path.write_bytes(value)
                    load_config(path)

    def test_cli_config_failure_has_no_engine_or_server_side_effects(self):
        with (
            mock.patch.object(sys, "argv", ["server.py", "--config", "missing-fixture.json"]),
            mock.patch.object(server, "Engine") as engine,
            mock.patch.object(server, "make_server") as make,
            mock.patch("builtins.print") as printer,
        ):
            self.assertEqual(server.main(), 2)
            engine.assert_not_called()
            make.assert_not_called()
            payload = json.loads(printer.call_args.args[0])
            self.assertEqual(payload["error"]["code"], "invalid_config")
            self.assertNotIn("Traceback", str(printer.call_args))


if __name__ == "__main__":
    unittest.main(verbosity=2)
