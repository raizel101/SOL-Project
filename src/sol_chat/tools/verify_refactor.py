"""Verify source organization against the preserved application archive.

Definition AST comparisons plus unchanged policy constants complement—not
replace—the regression tests. No model inference, model setup or index writes.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from sol_chat.paths import CORPUS_DIR, LEGACY_APP_DIR, PROJECT_ROOT, REPORTS_DIR

from .reporting import file_digest, write_json_report


def definitions(source: str) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def verify(node: Path, ruff: Path) -> dict:
    backend_path = REPORTS_DIR / "backend_verification_report.json"
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    if backend.get("status") != "passed":
        raise RuntimeError("Backend verification must pass first.")
    for name, expected in backend["tested_code_sha256"].items():
        if file_digest(PROJECT_ROOT / name) != expected:
            raise RuntimeError("Backend-tested source changed: " + name)
    baseline = PROJECT_ROOT / "outputs" / "sol_chat_integration.zip"
    with zipfile.ZipFile(baseline) as bundle:
        old_engine = bundle.read("sol_chat/sol_engine.py").decode("utf-8")
        previous_definitions = definitions(old_engine)
        current_definitions = {}
        for path in (PROJECT_ROOT / "src" / "sol_chat" / "core").glob("*.py"):
            for name, definition in definitions(path.read_text(encoding="utf-8")).items():
                if name in current_definitions:
                    raise RuntimeError("Duplicate core implementation: " + name)
                current_definitions[name] = definition
        changed = [
            name
            for name, definition in previous_definitions.items()
            if current_definitions.get(name) != definition
        ]
        if changed:
            raise RuntimeError("Core behavior definition changed: " + ", ".join(changed))
        historical = {}
        for filename in (
            "verification_report.json",
            "evaluation_history.json",
            "backend_verification_report.json",
            "ui_verification_report.json",
            "release_manifest.json",
        ):
            expected = bundle.read("sol_chat/" + filename)
            path = LEGACY_APP_DIR / filename
            if path.read_bytes() != expected:
                raise RuntimeError("Historical evidence changed: " + filename)
            historical[filename] = file_digest(path)
        old_assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(old_engine).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"SYSTEM_PROMPT", "REPAIR_INSTRUCTION"}
        }
        policy_path = PROJECT_ROOT / "src" / "sol_chat" / "core" / "policy.py"
        new_assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(policy_path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        if old_assignments != new_assignments:
            raise RuntimeError("Application prompt policy changed.")
    original_adapter = (CORPUS_DIR / "local_rag.py").read_text(encoding="utf-8")
    maintained_adapter = PROJECT_ROOT / "src" / "sol_chat" / "adapters" / "local_rag.py"
    if definitions(original_adapter) != definitions(maintained_adapter.read_text(encoding="utf-8")):
        raise RuntimeError("Adapter function behavior changed.")
    results = {}
    for name, command in (
        ("lint", [str(ruff), "check", "src", "scripts", "tests"]),
        ("format", [str(ruff), "format", "--check", "src", "scripts", "tests"]),
        ("frontend_helpers", [str(node), "tests/frontend/test_client_core.mjs"]),
    ):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=True,
        )
        results[name] = {"exit_code": result.returncode, "output": result.stdout + result.stderr}
    helpers = json.loads(results["frontend_helpers"]["output"])
    report = {
        "application": "SOL Chat structure refactor",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "source equivalence, formatting/lint, fixture and real-index HTTP checks",
        "core_definitions_ast_unchanged": len(previous_definitions),
        "adapter_definitions_ast_unchanged": len(definitions(original_adapter)),
        "prompt_policy_values_unchanged": True,
        "historical_evidence_unchanged_sha256": historical,
        "baseline_archive_sha256": file_digest(baseline),
        "backend_report_sha256": file_digest(backend_path),
        "engineering_tests": backend["engineering_tests"]["tests_run"],
        "real_index_http_checks": len(backend["http_checks"]),
        "frontend_helper_tests": helpers,
        "developer_checks": results,
        "index_metadata_unchanged_during_http_verification": backend[
            "index_file_metadata_unchanged"
        ],
        "live_model_inference": False,
        "models_downloaded": False,
        "source_code_is_single_canonical_implementation": True,
        "compatibility_wrappers_verified_by_regressions": True,
        "public_deployment_performed": False,
        "student_facing_release_approved": False,
        "semantic_or_clinical_validation": "not performed; failed historical model evaluation retained",
        "failures": [],
    }
    write_json_report(REPORTS_DIR / "refactor_verification_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=Path, required=True, help="Existing Node executable")
    parser.add_argument("--ruff", type=Path, required=True, help="Existing Ruff executable")
    args = parser.parse_args()
    result = verify(args.node.resolve(), args.ruff.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "engineering_tests": result["engineering_tests"],
                "core_definitions_unchanged": result["core_definitions_ast_unchanged"],
                "frontend_tests": result["frontend_helper_tests"]["tests_run"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
