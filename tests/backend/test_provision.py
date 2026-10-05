"""Provisioning regressions: fixtures only, no workers, downloads or live API."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sol_chat.tools import provision_model as provision

PROFILE = f'''FROM {provision.BASE_MODEL}
PARAMETER num_ctx 4096
PARAMETER num_predict 384
PARAMETER temperature 0.2
SYSTEM """Synthetic profile fixture, not a production assistant prompt."""
'''


class FakeFunction:
    def __init__(self, action):
        self.action = action
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.action(*args)


def fake_kernel(handle=123, exit_code=259, status_ok=True, times_ok=True):
    def exit_status(value, pointer):
        pointer._obj.value = exit_code
        return status_ok

    def process_times(value, creation, exit_, kernel_, user):
        creation._obj.dwHighDateTime = 3
        creation._obj.dwLowDateTime = 7
        return times_ok

    result = MagicMock()
    result.OpenProcess = FakeFunction(lambda rights, inherit, pid: handle)
    result.GetExitCodeProcess = FakeFunction(exit_status)
    result.GetProcessTimes = FakeFunction(process_times)
    result.CloseHandle = FakeFunction(lambda value: True)
    return result


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        (self.folder / "Modelfile").write_text(PROFILE, encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def api_fixture(self, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
        transport = MagicMock()
        transport.open.return_value = io.BytesIO(raw)
        return patch.object(provision, "opener", return_value=transport), transport

    def run_fixture(
        self,
        inventories,
        signatures=None,
        allow_resume=False,
        wait_pid=0,
        failures=None,
        package_present=False,
        package_exit=0,
    ):
        saved, operation_log = [], []
        self.package_calls = []
        package_script = self.folder.parents[1] / "scripts" / "package_project.py"
        original_is_file = Path.is_file

        def is_file(path):
            # Never depend on whether an unrelated real helper happens to exist
            # above the temporary fixture folder.
            return package_present if path == package_script else original_is_file(path)

        def package_run(command, **kwargs):
            self.assertEqual(command, [sys.executable, str(package_script)])
            self.assertEqual(kwargs, {"capture_output": True, "text": True, "timeout": 60})
            self.package_calls.append((command, kwargs))
            return SimpleNamespace(returncode=package_exit, stdout="synthetic packaging fixture")

        def save(state, **updates):
            state.update(updates)
            saved.append(dict(state))

        def operation(endpoint, payload, state):
            operation_log.append((endpoint, payload))

        @contextlib.contextmanager
        def unlocked():
            yield

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(provision, "HERE", self.folder))
            stack.enter_context(patch.object(provision, "save", side_effect=save))
            stack.enter_context(patch.object(provision, "exclusive_job", unlocked))
            stack.enter_context(
                patch.object(provision, "process_signature", side_effect=signatures or [None])
            )
            stack.enter_context(
                patch.object(provision, "installed_models", side_effect=inventories)
            )
            stack.enter_context(patch.object(provision, "stream_operation", side_effect=operation))
            fake_verify = stack.enter_context(
                patch.object(provision, "verify", return_value={"failures": failures or []})
            )
            model = stack.enter_context(patch.object(provision, "Engine"))
            model.return_value.status.return_value = {"ready": True}
            stack.enter_context(patch.object(provision.time, "sleep"))
            stack.enter_context(patch.object(provision.time, "monotonic", return_value=1))
            stack.enter_context(patch.object(Path, "is_file", is_file))
            stack.enter_context(patch.object(provision.subprocess, "run", side_effect=package_run))
            result = provision.run(wait_pid, allow_resume)
        return result, saved, operation_log, fake_verify

    def test_exact_profile_payload(self):
        with patch.object(provision, "HERE", self.folder):
            result = provision.profile_payload()
        self.assertEqual(result["model"], "sol-chat")
        self.assertEqual(result["from"], provision.BASE_MODEL)
        self.assertEqual(
            result["parameters"], {"num_ctx": 4096, "num_predict": 384, "temperature": 0.2}
        )
        self.assertTrue(result["stream"])
        self.assertIn("Synthetic profile fixture", result["system"])

    def test_default_profile_uses_config_but_relocation_fixture_stays_explicit(self):
        with patch.object(provision, "HERE", provision.LEGACY_APP_DIR):
            self.assertEqual(provision.profile_path(), provision.CONFIG_DIR / "Modelfile")
            self.assertNotEqual(provision.profile_path(), provision.LEGACY_APP_DIR / "Modelfile")
        with patch.object(provision, "HERE", self.folder):
            self.assertEqual(provision.profile_path(), self.folder / "Modelfile")

    def test_profile_rejects_extra_missing_duplicate_and_changed_settings(self):
        invalid = [
            PROFILE.replace(provision.BASE_MODEL, "another-model"),
            PROFILE + "FROM another-model\n",
            PROFILE + 'SYSTEM """Extra block"""\n',
            PROFILE + "TEMPLATE unexpected\n",
            PROFILE.replace("PARAMETER num_ctx 4096\n", ""),
            PROFILE.replace("PARAMETER num_predict 384", "PARAMETER num_ctx 4096"),
            PROFILE.replace("num_ctx 4096", "num_ctx 1000000"),
            PROFILE.replace("num_predict 384", "num_predict -4"),
            PROFILE.replace("temperature 0.2", "temperature nan"),
            PROFILE.replace("temperature 0.2", "temperature 1.0"),
            PROFILE.replace("Synthetic profile fixture, not a production assistant prompt.", " "),
        ]
        for text in invalid:
            with self.subTest(text=text):
                (self.folder / "Modelfile").write_text(text, encoding="utf-8")
                with (
                    patch.object(provision, "HERE", self.folder),
                    self.assertRaises((RuntimeError, ValueError)),
                ):
                    provision.profile_payload()

    def test_inventory_accepts_local_models_and_filters_cloud(self):
        fake, transport = self.api_fixture(
            {
                "models": [
                    {"name": provision.BASE_MODEL},
                    {"name": "remote", "remote_host": "external.invalid"},
                    {"name": "remote2", "remote_model": "something"},
                    {"name": "qwen-cloud"},
                ]
            }
        )
        with fake:
            models = provision.installed_models()
        self.assertEqual(set(models), {provision.BASE_MODEL})
        request = transport.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/tags")

    def test_inventory_rejects_invalid_and_oversized_responses(self):
        for value in (
            [],
            {"models": None},
            {"models": [5]},
            {"models": [{}]},
            {"models": [{"name": 4}]},
            b" " * (4 * 1024 * 1024 + 1),
        ):
            with self.subTest(value_type=type(value).__name__):
                fake, _ = self.api_fixture(value)
                with fake, self.assertRaises(RuntimeError):
                    provision.installed_models()

    def test_stream_progress_success_fixture(self):
        lines = [
            {"status": "pulling manifest"},
            {"status": "downloading", "total": 100, "completed": 20},
            {"status": "success"},
        ]
        fake, transport = self.api_fixture(
            b"".join(json.dumps(value).encode() + b"\n" for value in lines)
        )
        state, saved = {}, []
        with (
            fake,
            patch.object(
                provision, "save", side_effect=lambda obj, **updates: saved.append(updates)
            ),
            patch.object(provision.time, "monotonic", return_value=100),
        ):
            provision.stream_operation(
                "/api/pull", {"model": provision.BASE_MODEL, "stream": True}, state
            )
        self.assertEqual(saved[-1]["progress"]["status"], "success")
        request = transport.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/pull")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data)["model"], provision.BASE_MODEL)

    def test_stream_rejects_error_no_ack_and_oversized_lines(self):
        for body in (
            b'{"error":"synthetic failure"}\n',
            b'{"status":"downloading"}\n',
            b"x" * 65537 + b"\n",
        ):
            fake, _ = self.api_fixture(body)
            with (
                self.subTest(body_length=len(body)),
                fake,
                patch.object(provision, "save"),
                self.assertRaises(RuntimeError),
            ):
                provision.stream_operation("/api/create", {"model": "sol-chat"}, {})
        with self.assertRaises(ValueError):
            provision.stream_operation("/api/delete", {}, {})

    def test_stream_invalid_json_stops(self):
        fake, _ = self.api_fixture(b"not-json\n")
        with fake, self.assertRaises(ValueError):
            provision.stream_operation("/api/pull", {}, {})

    def test_default_never_downloads_missing_base(self):
        result, saved, operations, verify = self.run_fixture([{}])
        self.assertEqual(result, 1)
        self.assertEqual(operations, [])
        self.assertEqual(saved[-1]["status"], "failed")
        self.assertIn("downloading was not enabled", saved[-1]["error"]["message"])
        verify.assert_not_called()

    def test_missing_base_after_pull_ack_never_creates_alias(self):
        result, saved, operations, verify = self.run_fixture([{}, {}], allow_resume=True)
        self.assertEqual(result, 1)
        self.assertEqual([endpoint for endpoint, _ in operations], ["/api/pull"])
        self.assertIn("not listed locally", saved[-1]["error"]["message"])
        verify.assert_not_called()
        self.assertEqual(self.package_calls, [])

    def test_base_disappearing_before_create_never_creates_alias(self):
        result, saved, operations, verify = self.run_fixture([{provision.BASE_MODEL: {}}, {}])
        self.assertEqual(result, 1)
        self.assertEqual(operations, [])
        self.assertIn("creation was not attempted", saved[-1]["error"]["message"])
        verify.assert_not_called()

    def test_optional_archive_refresh_is_mocked_and_failure_reported(self):
        installed = {provision.BASE_MODEL: {}}
        for code in (0, 1):
            with self.subTest(package_exit=code):
                result, saved, _, _ = self.run_fixture(
                    [installed, installed], package_present=True, package_exit=code
                )
                self.assertEqual(len(self.package_calls), 1)
                self.assertEqual(saved[-1]["archive_refresh"]["exit_code"], code)
                self.assertEqual(result, 0 if code == 0 else 1)
                if code:
                    self.assertIn("integration_archive_refresh_failed", saved[-1]["failures"])

    def test_existing_alias_never_overwritten(self):
        for alias in ("sol-chat", "sol-chat:latest"):
            installed = {provision.BASE_MODEL: {}, alias: {}}
            result, saved, operations, verify = self.run_fixture([installed, installed])
            self.assertEqual(result, 1)
            self.assertEqual(operations, [])
            self.assertIn("not overwritten", saved[-1]["error"]["message"])
            verify.assert_not_called()

    def test_installed_base_creates_only_alias_and_runs_mock_checks(self):
        installed = {provision.BASE_MODEL: {}}
        result, saved, operations, verify = self.run_fixture([installed, installed])
        self.assertEqual(result, 0)
        self.assertEqual([endpoint for endpoint, _ in operations], ["/api/create"])
        self.assertEqual(saved[-1]["status"], "complete_human_review_required")
        verify.assert_called_once_with(live=True)

    def test_completed_setup_points_to_new_report_not_historical_evidence(self):
        installed = {provision.BASE_MODEL: {}}
        result, saved, operations, verify = self.run_fixture([installed, installed])
        self.assertEqual(result, 0)
        self.assertEqual([endpoint for endpoint, _ in operations], ["/api/create"])
        self.assertEqual(
            saved[-1]["verification_report"],
            str(provision.REPORTS_DIR / "model_verification_report.json"),
        )
        self.assertNotEqual(
            saved[-1]["verification_report"],
            str(provision.LEGACY_APP_DIR / "verification_report.json"),
        )
        verify.assert_called_once_with(live=True)

    def test_resume_only_after_observed_existing_process_stops(self):
        # Start signature=111, first poll same, then no running process. Every
        # operation is mocked; this does not launch or touch a real downloader.
        result, saved, operations, verify = self.run_fixture(
            [{}, {}, {provision.BASE_MODEL: {}}], [111, 111, None], True, 33700
        )
        self.assertEqual(result, 0)
        self.assertIn("waiting_for_existing_download", [item["status"] for item in saved])
        self.assertEqual([endpoint for endpoint, _ in operations], ["/api/pull", "/api/create"])
        verify.assert_called_once_with(live=True)

    def test_signature_change_does_not_follow_reused_pid(self):
        installed = {provision.BASE_MODEL: {}}
        result, saved, operations, _ = self.run_fixture(
            [installed, installed], [111, 222], False, 33700
        )
        self.assertEqual(result, 0)
        self.assertNotIn("waiting_for_existing_download", [item["status"] for item in saved])
        self.assertEqual([endpoint for endpoint, _ in operations], ["/api/create"])

    def test_failed_smoke_checks_do_not_claim_completion(self):
        installed = {provision.BASE_MODEL: {}}
        result, saved, _, _ = self.run_fixture(
            [installed, installed], failures=["synthetic_failure"]
        )
        self.assertEqual(result, 1)
        self.assertEqual(saved[-1]["status"], "live_smoke_checks_failed")

    @unittest.skipUnless(os.name == "nt", "Windows process/lock helpers")
    def test_windows_process_signature_is_read_only_and_closes_handles(self):
        kernel = fake_kernel()
        with (
            patch.object(provision.ctypes, "WinDLL", return_value=kernel),
            patch.object(provision.os, "kill", side_effect=AssertionError("No signaling allowed")),
        ):
            self.assertEqual(provision.process_signature(33700), (3 << 32) | 7)
        self.assertEqual(kernel.OpenProcess.calls, [(0x1000, False, 33700)])
        self.assertEqual(kernel.CloseHandle.calls, [(123,)])
        stopped = fake_kernel(exit_code=0)
        with patch.object(provision.ctypes, "WinDLL", return_value=stopped):
            self.assertIsNone(provision.process_signature(33700))
        self.assertEqual(stopped.CloseHandle.calls, [(123,)])

    @unittest.skipUnless(os.name == "nt", "Windows process/lock helpers")
    def test_unknown_process_inspection_errors_fail_closed(self):
        for code in (0, 5, 6, 123):
            with (
                self.subTest(code=code),
                patch.object(provision.ctypes, "WinDLL", return_value=fake_kernel(handle=0)),
                patch.object(provision.ctypes, "get_last_error", return_value=code),
                self.assertRaises(RuntimeError),
            ):
                provision.process_signature(33700)
        for code in (87, 1168):
            with (
                patch.object(provision.ctypes, "WinDLL", return_value=fake_kernel(handle=0)),
                patch.object(provision.ctypes, "get_last_error", return_value=code),
            ):
                self.assertIsNone(provision.process_signature(33700))

    @unittest.skipUnless(os.name == "nt", "Windows process/lock helpers")
    def test_os_lock_prevents_duplicate_and_releases_after_exception(self):
        with patch.object(provision, "HERE", self.folder):
            with self.assertRaisesRegex(ValueError, "fixture"):
                with provision.exclusive_job():
                    with self.assertRaisesRegex(RuntimeError, "already running"):
                        with provision.exclusive_job():
                            self.fail("Second worker unexpectedly acquired the lock")
                    raise ValueError("fixture")
            with provision.exclusive_job():
                pass  # Lock was released; only a temporary fixture file is used.

    def test_redirects_are_disabled(self):
        with self.assertRaisesRegex(RuntimeError, "redirect"):
            provision.NoRedirect().redirect_request(
                None, None, 302, "redirect", {}, "https://external.invalid/"
            )


if __name__ == "__main__":
    unittest.main()
