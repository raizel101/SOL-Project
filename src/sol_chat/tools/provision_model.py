"""Finish the authorized SOL model setup and run invented-prompt smoke tests.

Observes a known existing download before resuming it if interrupted. Uses the
installed loopback Ollama server; installs no software, publishes nothing, and
never overwrites an existing SOL alias. Run one copy only.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from ctypes import wintypes
from datetime import datetime, timezone

from sol_chat.core.engine import Engine
from sol_chat.paths import CONFIG_DIR, LEGACY_APP_DIR, REPORTS_DIR

from .verify_local import verify

HERE = LEGACY_APP_DIR
BASE_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
MODEL = "sol-chat"
BASE_URL = "http://127.0.0.1:11434"
STATE = HERE / "model_setup_status.json"


def profile_path():
    """Use the canonical profile; mocked relocation fixtures keep their own file."""
    return CONFIG_DIR / "Modelfile" if HERE == LEGACY_APP_DIR else HERE / "Modelfile"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Local model setup refuses HTTP redirects.")


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def now():
    return datetime.now(timezone.utc).isoformat()


def process_signature(pid):
    """Windows read-only process check; no os.kill(pid, 0) or termination."""
    if not pid:
        return None
    if os.name != "nt":
        raise RuntimeError("Watching an existing Windows download requires Windows.")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error in {87, 1168}:  # invalid/nonexistent process identifier
            return None
        raise RuntimeError(
            f"Cannot inspect existing downloader (Windows error {error}); refusing a duplicate."
        )
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise RuntimeError("Cannot inspect existing downloader exit status.")
        if code.value != 259:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            raise RuntimeError("Cannot inspect existing downloader creation time.")
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


@contextlib.contextmanager
def exclusive_job():
    """OS-held lock is automatically released if the worker stops."""
    with (HERE / "model_setup.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name != "nt":
            raise RuntimeError("This setup helper is for the user's Windows laptop.")
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(
                "A SOL setup worker is already running; no duplicate was started."
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def save(state, **updates):
    state.update(updates, updated_at_utc=now())
    temporary = STATE.with_name(STATE.name + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(STATE)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def installed_models():
    request = urllib.request.Request(BASE_URL + "/api/tags")
    with opener().open(request, timeout=10) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise RuntimeError("Local model inventory exceeds the response limit.")
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("models"), list):
        raise RuntimeError("Local model inventory must contain a models list.")
    models = {}
    for entry in obj["models"]:
        if not isinstance(entry, dict):
            raise RuntimeError("Invalid local model inventory entry.")
        name = entry.get("name", entry.get("model", ""))
        if not isinstance(name, str) or not name:
            raise RuntimeError("Local model inventory entry is missing its name.")
        if (
            entry.get("remote_host")
            or entry.get("remote_model")
            or re.search(r"(?:[:/-]cloud)(?:$|[:/-])", name, re.I)
        ):
            continue
        models[name] = entry
    return models


def stream_operation(endpoint, payload, state):
    if endpoint not in {"/api/pull", "/api/create"}:
        raise ValueError("Unknown setup endpoint")
    request = urllib.request.Request(
        BASE_URL + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    success, last_save = False, 0.0
    with opener().open(request, timeout=300) as response:
        while True:
            line = response.readline(65537)
            if not line:
                break
            if len(line) > 65536:
                raise RuntimeError("Local setup returned an oversized progress message.")
            value = json.loads(line.decode("utf-8"))
            if value.get("error"):
                raise RuntimeError(str(value["error"])[:500])
            success = value.get("status") == "success"
            if success or time.monotonic() - last_save >= 10:
                progress = {
                    key: value[key]
                    for key in ("status", "digest", "total", "completed")
                    if key in value
                }
                save(state, progress=progress)
                last_save = time.monotonic()
    if not success:
        raise RuntimeError("Local setup ended without a success acknowledgement.")


def profile_payload():
    text = profile_path().read_text(encoding="utf-8")
    profile = re.fullmatch(
        r'FROM ([^\r\n]+)\r?\n((?:PARAMETER [^\r\n]+\r?\n){3})SYSTEM """((?:(?!""").)*)"""\s*',
        text,
        re.S,
    )
    if not profile or profile.group(1) != BASE_MODEL:
        raise RuntimeError(
            "SOL Modelfile must contain exactly the shipped base, three parameters and one SYSTEM block."
        )
    parameters = {}
    for key, value in re.findall(r"^PARAMETER (\w+) ([^\r\n ]+)\r?$", profile.group(2), re.M):
        if key not in {"num_ctx", "num_predict", "temperature"} or key in parameters:
            raise RuntimeError("Unexpected model profile parameter.")
        parameters[key] = float(value) if key == "temperature" else int(value)
    if parameters != {"num_ctx": 4096, "num_predict": 384, "temperature": 0.2}:
        raise RuntimeError("SOL Modelfile parameters differ from the verified defaults.")
    if not profile.group(3).strip():
        raise RuntimeError("SOL system instructions cannot be empty.")
    return {
        "model": MODEL,
        "from": BASE_MODEL,
        "system": profile.group(3).strip(),
        "parameters": parameters,
        "stream": True,
    }


def run(wait_for_pid=0, allow_resume=False):
    state = {
        "application": "SOL Chat",
        "pid": os.getpid(),
        "base_model": BASE_MODEL,
        "model": MODEL,
        "started_at_utc": now(),
        "status": "starting",
        "wait_for_existing_download_pid": wait_for_pid,
        "profile_sha256": hashlib.sha256(profile_path().read_bytes()).hexdigest(),
        "software_installation": "none",
        "student_facing_release": "human review required",
    }
    with exclusive_job():
        try:
            save(state)
            signature = process_signature(wait_for_pid)
            deadline = time.monotonic() + 6 * 60 * 60
            while signature is not None and process_signature(wait_for_pid) == signature:
                if BASE_MODEL in installed_models():
                    break
                save(
                    state,
                    status="waiting_for_existing_download",
                    progress={
                        "status": "Existing downloader is still running; no duplicate launched."
                    },
                )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Existing download has not completed within six hours. Check the connection before resuming."
                    )
                time.sleep(15)
            models = installed_models()
            if BASE_MODEL not in models:
                if not allow_resume:
                    raise RuntimeError(
                        "The base model is not installed; model downloading was not enabled for this worker."
                    )
                save(state, status="downloading_or_resuming_base_model")
                stream_operation("/api/pull", {"model": BASE_MODEL, "stream": True}, state)
            save(state, status="creating_sol_model_profile")
            models = installed_models()
            if BASE_MODEL not in models:
                raise RuntimeError(
                    "The downloaded base model is not listed locally. SOL profile creation was not attempted."
                )
            if MODEL in models or MODEL + ":latest" in models:
                raise RuntimeError(
                    "SOL alias already exists. It was not overwritten; inspect it before choosing whether to use or replace it."
                )
            stream_operation("/api/create", profile_payload(), state)
            save(
                state,
                status="running_live_smoke_tests",
                runtime_status=Engine().status(check_model=True),
            )
            report = verify(live=True)
            package_script = HERE.parents[1] / "scripts" / "package_project.py"
            if package_script.is_file():
                packaged = subprocess.run(
                    [sys.executable, str(package_script)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if packaged.returncode:
                    report["failures"].append("integration_archive_refresh_failed")
                state["archive_refresh"] = {
                    "exit_code": packaged.returncode,
                    "output": packaged.stdout[-2000:],
                }
            save(
                state,
                status="live_smoke_checks_failed"
                if report["failures"]
                else "complete_human_review_required",
                verification_report=str(REPORTS_DIR / "model_verification_report.json"),
                failures=report["failures"],
                finished_at_utc=now(),
            )
            return 1 if report["failures"] else 0
        except Exception as exc:
            save(
                state,
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)[:1000]},
                finished_at_utc=now(),
            )
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int, default=0)
    parser.add_argument("--allow-resume-download", action="store_true")
    args = parser.parse_args()
    if args.wait_for_pid < 0:
        parser.error("PID must be nonnegative")
    try:
        return run(args.wait_for_pid, args.allow_resume_download)
    except Exception as exc:
        print(json.dumps({"status": "not_started", "message": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
