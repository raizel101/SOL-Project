"""Central paths for a source checkout, independent of the terminal directory.

Data and historical reports are external artifacts. Nothing here creates,
downloads, copies or rebuilds them. An installed editable package resolves to
this checkout; a relocated installation can set SOL_PROJECT_ROOT explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    """Locate the checkout or a deliberately configured local project root."""
    configured = os.environ.get("SOL_PROJECT_ROOT")
    if configured:
        if configured.replace("\\", "/").startswith("//"):
            raise RuntimeError("SOL_PROJECT_ROOT must be a local checkout, not a network path.")
        candidate = Path(configured).resolve()
        if not candidate.is_dir():
            raise RuntimeError("SOL_PROJECT_ROOT does not point to an existing directory.")
        return candidate
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    raise RuntimeError("SOL checkout not found. Set SOL_PROJECT_ROOT to its project folder.")


PROJECT_ROOT = find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CORPUS_DIR = PROJECT_ROOT / "outputs" / "auro_guide_corpus"
LEGACY_APP_DIR = PROJECT_ROOT / "outputs" / "sol_chat"
REPORTS_DIR = PROJECT_ROOT / "reports"
TESTS_DIR = PROJECT_ROOT / "tests"


def maintained_files() -> list[Path]:
    """Files whose hashes establish which implementation a report tested."""
    sources = sorted((PROJECT_ROOT / "src" / "sol_chat").rglob("*.py"))
    tests = sorted(TESTS_DIR.rglob("test_*.py")) + sorted(TESTS_DIR.rglob("test_*.mjs"))
    frontend = [
        FRONTEND_DIR / "index.html",
        FRONTEND_DIR / "styles" / "main.css",
        FRONTEND_DIR / "src" / "app.js",
        FRONTEND_DIR / "src" / "client-core.js",
    ]
    scripts = sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    launchers = [PROJECT_ROOT / "start_sol.cmd", PROJECT_ROOT / "scripts" / "start_sol.cmd"]
    compatibility = [
        LEGACY_APP_DIR / name
        for name in (
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
        )
    ]
    return (
        sources
        + tests
        + frontend
        + scripts
        + launchers
        + compatibility
        + [PROJECT_ROOT / "pyproject.toml", CONFIG_DIR / "local.json", CONFIG_DIR / "Modelfile"]
    )
