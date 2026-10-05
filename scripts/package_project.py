"""Build a verified source-only handoff archive, never include corpus or weights."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = _bootstrap.PROJECT_ROOT
    report_path = root / "reports" / "backend_verification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or not report.get("tested_code_sha256"):
        raise RuntimeError("Run scripts/verify_backend.py successfully before packaging.")
    for name, expected in report["tested_code_sha256"].items():
        if digest(root / name) != expected:
            raise RuntimeError("Verified source changed; run verification again: " + name)
    refactor_path = root / "reports" / "refactor_verification_report.json"
    refactor = json.loads(refactor_path.read_text(encoding="utf-8"))
    if refactor.get("status") != "passed" or refactor.get("backend_report_sha256") != digest(
        report_path
    ):
        raise RuntimeError("Run scripts/verify_refactor.py against the current backend report.")
    names = {
        "README.md",
        "pyproject.toml",
        ".gitignore",
        ".editorconfig",
        ".env.example",
        "start_sol.cmd",
        "config/local.json",
        "config/Modelfile",
        "data/README.md",
        ".github/workflows/ci.yml",
    }
    # Restrict collection to source/documentation types. Never sweep future
    # .env files, data dumps, binary dependencies or model evaluations.
    source_types = {
        "src": {".py"},
        "frontend": {".html", ".css", ".js", ".md"},
        "tests": {".py", ".mjs", ".md"},
        "scripts": {".py", ".cmd"},
        "docs": {".md"},
    }
    for folder, extensions in source_types.items():
        for path in (root / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in extensions:
                names.add(path.relative_to(root).as_posix())
    names.add("reports/README.md")
    proof = root / "reports" / "structured_app_preview.jpg"
    if proof.is_file():
        names.add(proof.relative_to(root).as_posix())
    # Do not sweep optional model evaluations, logs or future private artifacts.
    for filename in (
        "backend_verification_report.json",
        "refactor_verification_report.json",
        "refactor_ui_observations.json",
        "openapi.json",
    ):
        path = root / "reports" / filename
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    for filename in (
        "server.py",
        "sol_engine.py",
        "api_contract.py",
        "backend_config.py",
        "web_assets.py",
        "verify_backend.py",
        "verify_local.py",
        "provision_model.py",
        "start_sol.cmd",
        "backend_config.json",
    ):
        names.add("outputs/sol_chat/" + filename)
    # Preserve evaluation provenance, not old implementation copies or inputs.
    for filename in ("verification_report.json", "evaluation_history.json"):
        path = root / "outputs" / "sol_chat" / filename
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    files = sorted(root / name for name in names)
    if any(root not in path.resolve().parents for path in files):
        raise RuntimeError("A packaged file resolves outside the project; review that link first.")
    manifest = {
        "application": "SOL Chat",
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "maintainable source checkout; not a public deployment or trained weights",
        "backend_verification_status": report["status"],
        "engineering_tests": report["engineering_tests"]["tests_run"],
        "real_index_http_checks": len(report["http_checks"]),
        "frontend_helper_tests": refactor["frontend_helper_tests"]["tests_run"],
        "source_equivalence_verified": True,
        "refactor_report_sha256": digest(refactor_path),
        "live_model_generation_by_refactor": False,
        "student_facing_release_approved": False,
        "historical_model_evaluation": report.get("historical_model_evaluation"),
        "external_dependencies": [
            "outputs/auro_guide_corpus/local_rag.sqlite and corpus metadata",
            "Ollama with installed sol-chat model",
        ],
        "data_and_weights_included": False,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    archive = root / "outputs" / "sol_chat_project.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, "sol-chat/" + path.relative_to(root).as_posix())
        bundle.writestr("sol-chat/release_manifest.json", manifest_bytes)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() or len(set(bundle.namelist())) != len(bundle.namelist()):
            raise RuntimeError("Archive integrity or unique-name check failed.")
        for entry in manifest["files"]:
            payload = bundle.read("sol-chat/" + entry["path"])
            if (
                len(payload) != entry["bytes"]
                or hashlib.sha256(payload).hexdigest() != entry["sha256"]
            ):
                raise RuntimeError("Archive checksum mismatch: " + entry["path"])
    print(
        json.dumps(
            {
                "archive": str(archive),
                "files": len(files) + 1,
                "bytes": archive.stat().st_size,
                "sha256": digest(archive),
                "verified": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
