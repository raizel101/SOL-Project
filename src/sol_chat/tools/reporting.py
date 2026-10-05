"""Safe diagnostic output shared by engineering and optional model verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sol_chat.paths import (
    CONFIG_DIR,
    CORPUS_DIR,
    FRONTEND_DIR,
    LEGACY_APP_DIR,
    PROJECT_ROOT,
    TESTS_DIR,
)


def file_digest(path: Path) -> str:
    """Hash a small source/report file, not the large corpus by default."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_report_path(path: Path | str) -> Path:
    """Never let a diagnostic overwrite source, configuration or old evidence."""
    target = Path(path).resolve()
    protected = (
        CORPUS_DIR,
        LEGACY_APP_DIR,
        CONFIG_DIR,
        FRONTEND_DIR,
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        TESTS_DIR,
    )
    if target.suffix.lower() != ".json" or any(
        target == folder or folder.resolve() in target.parents for folder in protected
    ):
        raise ValueError(
            "Choose a separate JSON report outside source, config, corpus and historical evidence."
        )
    return target


def write_json_report(path: Path | str, value: dict) -> None:
    """Atomically replace only a validated diagnostic report."""
    target = validate_report_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
