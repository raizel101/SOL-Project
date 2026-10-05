"""Source-layout regressions; no corpus changes, model calls or deployments."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sol_chat import paths
from sol_chat.api import web_assets
from sol_chat.config import DEFAULT_CONFIG, load_config, validate_config
from sol_chat.core import engine, prompts, safety, settings, validation
from sol_chat.tools import reporting, verify_backend


class ProjectLayoutTests(unittest.TestCase):
    def test_default_paths_do_not_depend_on_terminal_directory(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with mock.patch.dict(os.environ, {"SOL_PROJECT_ROOT": ""}):
                    self.assertEqual(paths.find_project_root(), paths.PROJECT_ROOT)
                    self.assertEqual(
                        load_config().index_path, paths.CORPUS_DIR / "local_rag.sqlite"
                    )
                    self.assertEqual(
                        validate_config({}).index_path, paths.CORPUS_DIR / "local_rag.sqlite"
                    )
            finally:
                os.chdir(original_directory)
        self.assertEqual(DEFAULT_CONFIG, paths.PROJECT_ROOT / "config" / "local.json")

    def test_project_root_override_is_explicit_local_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"SOL_PROJECT_ROOT": directory}):
                self.assertEqual(paths.find_project_root(), Path(directory).resolve())
            with mock.patch.dict(
                os.environ, {"SOL_PROJECT_ROOT": str(Path(directory) / "missing")}
            ):
                with self.assertRaises(RuntimeError):
                    paths.find_project_root()
        for value in ("//remote.invalid/share/sol", "\\\\remote.invalid\\share\\sol"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"SOL_PROJECT_ROOT": value}),
            ):
                with self.assertRaises(RuntimeError):
                    paths.find_project_root()

    def test_static_routes_use_only_the_canonical_frontend(self):
        expected = {
            "/chat": paths.FRONTEND_DIR / "index.html",
            "/assets/styles.css": paths.FRONTEND_DIR / "styles" / "main.css",
            "/assets/app.js": paths.FRONTEND_DIR / "src" / "app.js",
            "/assets/client-core.js": paths.FRONTEND_DIR / "src" / "client-core.js",
        }
        self.assertEqual(set(web_assets.ASSETS), set(expected))
        for route, expected_path in expected.items():
            with self.subTest(route=route):
                asset_path, _ = web_assets.ASSETS[route]
                self.assertEqual(asset_path, expected_path)
                self.assertTrue(asset_path.is_file())
                self.assertIn(paths.FRONTEND_DIR, asset_path.parents)
                self.assertNotIn(paths.LEGACY_APP_DIR, asset_path.parents)

    def test_backend_modules_resolve_to_the_maintained_source_package(self):
        source_root = paths.PROJECT_ROOT / "src" / "sol_chat"
        modules = (
            "api.server",
            "api.contract",
            "api.web_assets",
            "config",
            "adapters.corpus",
            "core.engine",
            "core.errors",
            "core.settings",
            "core.validation",
            "core.safety",
            "core.relevance",
            "core.sources",
            "core.prompts",
            "core.policy",
            "core.responses",
            "tools.provision_model",
            "tools.verify_backend",
        )
        for name in modules:
            with self.subTest(module=name):
                module = importlib.import_module("sol_chat." + name)
                module_file = Path(module.__file__).resolve()
                self.assertIn(source_root, module_file.parents)
                self.assertNotIn(paths.LEGACY_APP_DIR, module_file.parents)

    def test_engine_has_one_maintained_implementation(self):
        definitions = []
        for source in (paths.PROJECT_ROOT / "src" / "sol_chat").rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            definitions.extend(
                source
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "Engine"
            )
        self.assertEqual(
            definitions, [paths.PROJECT_ROOT / "src" / "sol_chat" / "core" / "engine.py"]
        )
        self.assertEqual(engine.Engine.__module__, "sol_chat.core.engine")

    def test_compatibility_helper_exports_reference_real_feature_modules(self):
        self.assertIs(engine.Settings, settings.Settings)
        self.assertIs(engine._question, validation._question)
        self.assertIs(engine._history, validation._history)
        self.assertIs(engine.urgent_trigger, safety.urgent_trigger)
        self.assertIs(engine._personal_distress, safety._personal_distress)
        self.assertIs(engine._build_messages, prompts._build_messages)

    def test_reports_cannot_overwrite_source_corpus_or_historical_evidence(self):
        protected_folders = (
            paths.CORPUS_DIR,
            paths.LEGACY_APP_DIR,
            paths.CONFIG_DIR,
            paths.FRONTEND_DIR,
            paths.PROJECT_ROOT / "src",
            paths.PROJECT_ROOT / "scripts",
            paths.TESTS_DIR,
        )
        protected_files = [folder / "nested" / "diagnostic.json" for folder in protected_folders]
        protected_files.extend(
            (
                paths.LEGACY_APP_DIR / "verification_report.json",
                paths.LEGACY_APP_DIR / "backend_verification_report.json",
                paths.LEGACY_APP_DIR / "release_manifest.json",
                paths.CORPUS_DIR / "coverage_report.json",
                paths.CONFIG_DIR / "local.json",
                paths.REPORTS_DIR / ".." / "outputs" / "sol_chat" / "verification_report.json",
            )
        )
        self.assertIs(verify_backend.validate_report_path, reporting.validate_report_path)
        for target in protected_files:
            with (
                self.subTest(target=target),
                mock.patch.object(Path, "mkdir") as mkdir,
                mock.patch.object(Path, "write_text") as writer,
                mock.patch.object(Path, "replace") as replacer,
            ):
                with self.assertRaises(ValueError):
                    reporting.validate_report_path(target)
                with self.assertRaises(ValueError):
                    reporting.write_json_report(target, {"synthetic": True})
                mkdir.assert_not_called()
                writer.assert_not_called()
                replacer.assert_not_called()

    def test_new_diagnostics_have_an_explicit_json_destination(self):
        target = paths.REPORTS_DIR / "layout_fixture_not_written.json"
        self.assertEqual(reporting.validate_report_path(target), target.resolve())
        for invalid in (paths.REPORTS_DIR / "diagnostic.txt", paths.REPORTS_DIR / "diagnostic"):
            with self.subTest(target=invalid), self.assertRaises(ValueError):
                reporting.validate_report_path(invalid)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "new" / "fixture.json"
            payload = {"synthetic": True, "model_calls": 0}
            reporting.write_json_report(fixture, payload)
            self.assertEqual(json.loads(fixture.read_text(encoding="utf-8")), payload)
            self.assertFalse(fixture.with_name(fixture.name + ".tmp").exists())

    def test_legacy_shims_import_the_exact_canonical_module_object(self):
        mappings = {
            "server.py": "sol_chat.api.server",
            "api_contract.py": "sol_chat.api.contract",
            "backend_config.py": "sol_chat.config",
            "web_assets.py": "sol_chat.api.web_assets",
            "sol_engine.py": "sol_chat.core.engine",
            "provision_model.py": "sol_chat.tools.provision_model",
            "verify_backend.py": "sol_chat.tools.verify_backend",
            "verify_local.py": "sol_chat.tools.verify_local",
        }
        for index, (filename, module_name) in enumerate(mappings.items()):
            with self.subTest(filename=filename):
                expected = importlib.import_module(module_name)
                probe_name = "sol_layout_legacy_probe_" + str(index)
                spec = importlib.util.spec_from_file_location(
                    probe_name, paths.LEGACY_APP_DIR / filename
                )
                self.assertIsNotNone(spec)
                module = importlib.util.module_from_spec(spec)
                self.assertNotIn(probe_name, sys.modules)
                try:
                    sys.modules[probe_name] = module
                    spec.loader.exec_module(module)
                    self.assertIs(sys.modules[probe_name], expected)
                finally:
                    sys.modules.pop(probe_name, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
