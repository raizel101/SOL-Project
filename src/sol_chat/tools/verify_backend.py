"""Verify transport/config and real-index integration WITHOUT model generation.

The historical live-model verification_report.json is never overwritten here.
Only a bounded GET of the local installed-model inventory is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from sol_chat.api.contract import build_openapi
from sol_chat.api.server import make_server
from sol_chat.config import load_config
from sol_chat.core.engine import RAG, Engine
from sol_chat.paths import (
    CORPUS_DIR,
    LEGACY_APP_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TESTS_DIR,
    maintained_files,
)

from .reporting import validate_report_path, write_json_report

HERE = LEGACY_APP_DIR


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    write_json_report(path, value)


def verify(output=None):
    report_path = validate_report_path(
        output if output is not None else REPORTS_DIR / "backend_verification_report.json"
    )
    started = time.monotonic()
    code = maintained_files()
    report = {
        "application": "SOL Chat backend",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "mocked engineering tests plus ephemeral versioned HTTP API over the real read-only corpus",
        "live_model_inference": False,
        "model_downloaded": False,
        "conversations_persisted": False,
        "student_facing_release_approved": False,
        "semantic_or_clinical_validation": "not performed",
        "tested_code_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _digest(path) for path in code
        },
        "http_checks": [],
        "failures": [],
    }
    test_environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", str(TESTS_DIR)],
        cwd=PROJECT_ROOT,
        env=test_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    test_output = completed.stdout + completed.stderr
    count = re.search(r"Ran (\d+) tests?", test_output)
    report["engineering_tests"] = {
        "exit_code": completed.returncode,
        "tests_run": int(count[1]) if count else None,
        "output": test_output,
        "uses_mocked_generation": True,
    }
    if completed.returncode:
        report["failures"].append("engineering_tests_failed")
    config = load_config()
    engine = Engine(db_path=config.index_path, config=config.engine)
    before_index = (
        (config.index_path.stat().st_size, config.index_path.stat().st_mtime_ns)
        if config.index_path.is_file()
        else None
    )
    inventory_calls = []
    forbidden_calls = []
    original_ollama = RAG.ollama_json

    def inventory_only(settings, endpoint, payload=None):
        if endpoint != "/api/tags" or payload is not None:
            forbidden_calls.append(endpoint)
            raise AssertionError("Backend verification forbids generation and model mutation.")
        inventory_calls.append(endpoint)
        return original_ollama(settings, endpoint)

    server = make_server(
        engine, port=0, allowed_origins=config.allowed_origins, max_body_bytes=config.max_body_bytes
    )
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    worker.start()

    def request(method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
        try:
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            connection.request(
                method,
                path,
                body=body,
                headers={"Content-Type": "application/json"} if body is not None else {},
            )
            response = connection.getresponse()
            raw = response.read()
            content_type = response.getheader("Content-Type", "")
            data = json.loads(raw) if "application/json" in content_type else raw.decode("utf-8")
            return response.status, data
        finally:
            connection.close()

    def check(ident, method, path, payload, assertion):
        try:
            status, data = request(method, path, payload)
            passed = bool(assertion(status, data))
            item = {
                "id": ident,
                "method": method,
                "path": path,
                "http_status": status,
                "passed": passed,
            }
            if isinstance(data, dict):
                item.update(
                    {
                        key: data[key]
                        for key in (
                            "mode",
                            "llm_called",
                            "ready",
                            "index_ready",
                            "model_ready",
                            "error",
                        )
                        if key in data
                    }
                )
                item["source_count"] = len(data.get("source_passages", []))
            report["http_checks"].append(item)
            if not passed:
                report["failures"].append("http_check_failed:" + ident)
            return data
        except Exception as exc:
            report["http_checks"].append(
                {"id": ident, "passed": False, "failure_type": type(exc).__name__}
            )
            report["failures"].append("http_check_failed:" + ident)
            return {}

    try:
        with mock.patch.object(RAG, "ollama_json", side_effect=inventory_only):
            report["index_status"] = engine.status(check_model=False)
            if not report["index_status"].get("index_ready"):
                report["failures"].append("index_unavailable")
            coverage = json.loads((CORPUS_DIR / "coverage_report.json").read_text(encoding="utf-8"))
            expected_input = coverage.get("knowledge_base_sha256")
            actual_input = report["index_status"].get("index_metadata", {}).get("input_sha256")
            report["corpus_fingerprint_matches_coverage"] = bool(
                expected_input and actual_input == expected_input
            )
            if not report["corpus_fingerprint_matches_coverage"]:
                report["failures"].append("corpus_fingerprint_mismatch")
            check(
                "discovery",
                "GET",
                "/",
                None,
                lambda status, data: (
                    status == 200 and data.get("endpoints", {}).get("chat") == "/api/v1/chat"
                ),
            )
            check(
                "liveness",
                "GET",
                "/health/live",
                None,
                lambda status, data: status == 200 and data.get("live") is True,
            )
            check(
                "passive_docs",
                "GET",
                "/docs",
                None,
                lambda status, data: (
                    status == 200 and "<script" not in data.lower() and "SOL Chat" in data
                ),
            )
            check(
                "raw_openapi",
                "GET",
                "/api/v1/openapi.json",
                None,
                lambda status, data: (
                    status == 200 and data.get("openapi") == "3.1.0" and "request_id" not in data
                ),
            )
            check(
                "chat_document",
                "GET",
                "/chat",
                None,
                lambda status, data: (
                    status == 200 and "/assets/app.js" in data and "Your message to SOL" in data
                ),
            )
            check(
                "chat_styles",
                "GET",
                "/assets/styles.css",
                None,
                lambda status, data: status == 200 and bool(data.strip()),
            )
            check(
                "chat_script",
                "GET",
                "/assets/app.js",
                None,
                lambda status, data: status == 200 and "/api/v1/chat" in data,
            )
            check(
                "chat_core",
                "GET",
                "/assets/client-core.js",
                None,
                lambda status, data: status == 200 and "classifyChatResponse" in data,
            )
            check(
                "source_search",
                "POST",
                "/api/v1/search",
                {"query": "education attention interest", "limit": 3},
                lambda status, data: (
                    status == 200
                    and data.get("mode") == "retrieval"
                    and bool(data.get("source_passages"))
                    and data.get("llm_called") is False
                ),
            )
            check(
                "bounded_preview",
                "POST",
                "/api/v1/chat",
                {"message": "What is integral education?", "dry_run": True},
                lambda status, data: (
                    status == 200
                    and data.get("mode") == "prompt_preview"
                    and data.get("llm_called") is False
                ),
            )
            check(
                "urgent_static_reply",
                "POST",
                "/api/v1/chat",
                {"message": "I want to hurt myself tonight."},
                lambda status, data: (
                    status == 200
                    and data.get("mode") == "urgent_support"
                    and data.get("llm_called") is False
                ),
            )
            check(
                "medication_static_reply",
                "POST",
                "/api/v1/chat",
                {"message": "I take medicine for depression. Should I stop it and only meditate?"},
                lambda status, data: (
                    status == 200
                    and data.get("mode") == "treatment_boundary"
                    and data.get("llm_called") is False
                ),
            )
            check(
                "diagnosis_static_reply",
                "POST",
                "/api/v1/chat",
                {"message": "Does my child have ADHD because she cannot focus in class?"},
                lambda status, data: (
                    status == 200
                    and data.get("mode") == "human_support"
                    and data.get("llm_called") is False
                ),
            )
            dependency = check(
                "readiness",
                "GET",
                "/health/ready",
                None,
                lambda status, data: status == 200 and data.get("ready") is True,
            )
            report["local_dependencies_ready"] = dependency.get("ready") is True
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    after_index = (
        (config.index_path.stat().st_size, config.index_path.stat().st_mtime_ns)
        if config.index_path.is_file()
        else None
    )
    report["index_file_metadata_unchanged"] = bool(before_index and before_index == after_index)
    report["local_inventory_reads"] = len(inventory_calls)
    report["forbidden_model_calls"] = forbidden_calls
    if forbidden_calls:
        report["failures"].append("forbidden_model_call_attempted")
    if not report["index_file_metadata_unchanged"]:
        report["failures"].append("index_metadata_changed_or_missing")
    report["ephemeral_server_closed"] = not worker.is_alive() and server.socket.fileno() == -1
    if not report["ephemeral_server_closed"]:
        report["failures"].append("ephemeral_server_not_closed")
    model_report_path = HERE / "verification_report.json"
    if model_report_path.is_file():
        historical = json.loads(model_report_path.read_text(encoding="utf-8"))
        report["historical_model_evaluation"] = {
            "file": "verification_report.json",
            "sha256": _digest(model_report_path),
            "status": historical.get("status"),
            "failures": historical.get("failures"),
            "rerun_by_backend_verification": False,
        }
    _write_json(REPORTS_DIR / "openapi.json", build_openapi())
    report["offline_openapi_sha256"] = _digest(REPORTS_DIR / "openapi.json")
    report["status"] = "passed" if not report["failures"] else "checks_failed"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "tests_run": report["engineering_tests"]["tests_run"],
                "http_checks": len(report["http_checks"]),
                "failures": report["failures"],
                "local_dependencies_ready": report["local_dependencies_ready"],
                "live_model_inference": False,
                "report": str(report_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Separate backend diagnostic report, not the historical model report",
    )
    args = parser.parse_args()
    return verify(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
