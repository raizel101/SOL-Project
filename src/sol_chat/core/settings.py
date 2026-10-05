"""Immutable engine limits; validation never downloads a model."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sol_chat.adapters.corpus import RAG

from .errors import EngineError


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "sol-chat"
    num_ctx: int = 4096
    num_predict: int = 384
    temperature: float = 0.2
    top_k_sources: int = 3
    max_source_characters: int = 2200
    max_question_characters: int = 1200
    max_history_messages: int = 8
    max_history_characters: int = 2000
    max_history_message_characters: int = 1000
    request_timeout_seconds: int = 120
    keep_alive: str = "2m"

    @classmethod
    def from_dict(cls, values: dict | None) -> "Settings":
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise EngineError("invalid_config", "Engine configuration must be a JSON object.")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise EngineError(
                "invalid_config", "Unknown engine settings: " + ", ".join(sorted(unknown))
            )
        result = cls(**values)
        limits = {
            "num_ctx": (4096, 32768),
            "num_predict": (64, 1024),
            "top_k_sources": (1, 6),
            "max_source_characters": (300, 12000),
            "max_question_characters": (100, 4000),
            "max_history_messages": (0, 12),
            "max_history_characters": (0, 8000),
            "max_history_message_characters": (100, 4000),
            "request_timeout_seconds": (1, 180),
        }
        for key, (low, high) in limits.items():
            value = getattr(result, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise EngineError(
                    "invalid_config", f"{key} must be an integer from {low} to {high}."
                )
        if (
            isinstance(result.temperature, bool)
            or not isinstance(result.temperature, (int, float))
            or not 0 <= result.temperature <= 1
        ):
            raise EngineError("invalid_config", "temperature must be from 0 to 1.")
        if not isinstance(result.model, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}", result.model
        ):
            raise EngineError(
                "invalid_config", "model must name an already installed local Ollama model."
            )
        if re.search(r"(?:[:/-]cloud)(?:$|[:/-])", result.model, re.I):
            raise EngineError(
                "invalid_config", "Cloud models are disabled; use an installed local model."
            )
        try:
            RAG.loopback_url(result.ollama_base_url)
        except RAG.RagError as exc:
            raise EngineError("invalid_config", str(exc)) from exc
        if not isinstance(result.keep_alive, str) or not re.fullmatch(
            r"\d{1,3}[smh]", result.keep_alive
        ):
            raise EngineError("invalid_config", "keep_alive must be a duration such as 2m.")
        return result

    def ollama_config(self) -> dict:
        return {
            "ollama_base_url": self.ollama_base_url,
            "model": self.model,
            "request_timeout_seconds": self.request_timeout_seconds,
            "keep_alive": self.keep_alive,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
        }
