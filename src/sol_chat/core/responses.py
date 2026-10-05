"""Consistent JSON-compatible result envelopes."""

from __future__ import annotations


def _base(mode: str, model: str) -> dict:
    return {
        "mode": mode,
        "answer": "",
        "citations": [],
        "source_passages": [],
        "model": model,
        "llm_called": False,
        "warnings": [],
        "error": None,
        "generation_attempts": 0,
        "repair_attempted": False,
        "initial_failure": None,
    }
