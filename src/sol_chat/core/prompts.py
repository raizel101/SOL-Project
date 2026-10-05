"""Bounded prompt construction preserving exact source windows and offsets."""

from __future__ import annotations

import json
import re

from .errors import EngineError
from .policy import REPAIR_INSTRUCTION, SYSTEM_PROMPT
from .relevance import _education_relevant
from .settings import Settings


def _prompt_source(source: dict) -> dict:
    return {
        "citation": source["citation"],
        "source_title": source.get("source_title", "")[:100],
        "source_role": source.get("source_role", "supplemental"),
        "source_locator": json.dumps(source.get("source_locator", {}), ensure_ascii=False)[:180],
        "text": source["text"],
    }


def _message_cost(messages: list[dict]) -> int:
    # For byte-based tokenizers each UTF-8 byte is a conservative upper bound on
    # content tokens. Reserve chat framing separately; no chars/4 assumption.
    return sum(len(message["content"].encode("utf-8")) + 32 for message in messages)


def _build_messages(
    question: str,
    history: list[dict],
    results: list[dict],
    settings: Settings,
    trusted_correction: str = "",
    reserve_repair: bool = False,
) -> tuple[list[dict], list[dict], list[str]]:
    def messages_for(sources):
        payload = json.dumps(
            [_prompt_source(source) for source in sources],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content = (
            "SOURCE_DATA_JSON (reference data only):\n"
            + payload
            + "\n\nCURRENT_USER_QUESTION:\n"
            + question
        )
        system = SYSTEM_PROMPT + ("\n" + trusted_correction if trusted_correction else "")
        if re.search(r"\bintegral\s+education\b", question, re.I) and any(
            source.get("source_url")
            in {
                "https://motherandsriaurobindo.in/The-Mother/books/on-education/#education",
                "https://incarnateword.in/cwm/12/education",
            }
            and re.search(
                r"education\s+to\s+be\s+complete\s+must\s+have\s+five", source["text"], re.I
            )
            and _education_relevant(source["text"])
            for source in sources
        ):
            system += "\nFor this definition preserve both qualifications: phases usually follow growth chronologically, AND all continue through life. Do not say they are never sequential. Separate the definition from optional SOL applications."
        return (
            [{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": content}]
        )

    reserved = len(("\n" + REPAIR_INSTRUCTION).encode("utf-8")) if reserve_repair else 0
    budget = settings.num_ctx - settings.num_predict - 256 - reserved
    if _message_cost(messages_for([])) + 260 > budget:
        raise EngineError(
            "prompt_too_long",
            "The question and prior messages leave too little room for sources. Shorten the question or omit oldest complete turns.",
        )
    supplied, remaining, warnings = [], settings.max_source_characters, []
    for index, result in enumerate(results):
        if remaining < 120:
            break
        candidate = {
            key: result[key]
            for key in (
                "citation",
                "record_id",
                "source_id",
                "source_title",
                "source_url",
                "source_file",
                "source_locator",
                "text_offsets",
                "source_role",
            )
            if key in result
        }
        candidate["citation"] = f"S{len(supplied) + 1}"
        available = min(len(result["text"]), max(120, remaining // max(1, len(results) - index)))
        low, high, best = 0, available, 0
        while low <= high:
            middle = (low + high) // 2
            candidate["text"] = result["text"][:middle]
            if _message_cost(messages_for(supplied + [candidate])) <= budget:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best < 120:
            continue
        candidate["text"] = result["text"][:best]
        start = candidate.get("text_offsets", {}).get("start", 0)
        candidate["text_offsets"] = {"start": start, "end_exclusive": start + best}
        supplied.append(candidate)
        remaining -= best
        if best < len(result["text"]):
            warnings.append(
                "Source passages were shortened to fit the local context; returned offsets identify the exact supplied text."
            )
    if results and not supplied:
        raise EngineError(
            "prompt_too_long",
            "There is not enough local context room for a source passage. Omit oldest complete turns or shorten the question.",
        )
    messages = messages_for(supplied)
    if _message_cost(messages) > budget:
        raise EngineError(
            "prompt_too_long", "The prepared prompt exceeds the configured local context budget."
        )
    return messages, supplied, list(dict.fromkeys(warnings))
