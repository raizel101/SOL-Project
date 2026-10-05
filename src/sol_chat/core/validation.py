"""Question and complete-turn history validation; no silent truncation."""

from __future__ import annotations

import re
from typing import Any

from .errors import EngineError
from .settings import Settings


def _question(value: Any, settings: Settings) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EngineError("invalid_question", "Please send a nonempty text question.")
    if len(value) > settings.max_question_characters:
        raise EngineError(
            "question_too_long",
            f"Keep your question within {settings.max_question_characters} characters.",
        )
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value):
        raise EngineError(
            "invalid_question", "The question contains unsupported control characters."
        )
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise EngineError(
            "invalid_question", "The question contains an invalid Unicode character."
        ) from exc
    return value.strip()


def _history(value: Any, settings: Settings) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EngineError(
            "invalid_history", "history must be a list of prior user/assistant messages."
        )
    if len(value) > settings.max_history_messages:
        raise EngineError(
            "history_too_long",
            f"Send at most {settings.max_history_messages} prior messages; omit the oldest complete turns.",
        )
    result, total = [], 0
    for number, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise EngineError(
                "invalid_history", "Each history message must contain only role and content."
            )
        expected = "user" if number % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise EngineError(
                "invalid_history",
                "History must alternate user/assistant, start with user, and contain no system, tool or developer roles.",
            )
        content = message["content"]
        if (
            not isinstance(content, str)
            or not content.strip()
            or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", content)
        ):
            raise EngineError(
                "invalid_history",
                "History content must be nonempty text without control characters.",
            )
        try:
            content.encode("utf-8")
        except UnicodeError as exc:
            raise EngineError(
                "invalid_history", "History content contains an invalid Unicode character."
            ) from exc
        if len(content) > settings.max_history_message_characters:
            raise EngineError(
                "history_too_long",
                f"Each prior message is limited to {settings.max_history_message_characters} characters.",
            )
        total += len(content)
        result.append({"role": expected, "content": content})
    if len(result) % 2:
        raise EngineError(
            "invalid_history", "History must end with assistant; send only complete previous turns."
        )
    if total > settings.max_history_characters:
        raise EngineError(
            "history_too_long",
            f"Prior messages total more than {settings.max_history_characters} characters; omit oldest complete turns.",
        )
    return result
