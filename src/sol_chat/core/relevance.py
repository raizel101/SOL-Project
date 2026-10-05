"""Conservative educational/career evidence gates, not semantic validation."""

from __future__ import annotations

import re


def _concentration_intent(question: str) -> bool:
    return bool(
        not re.search(
            r"\b(?:adhd|diagnos\w*|disorder|treatment|medicat\w*|medicin\w*|therap\w*|autis\w*|yogic|meditat\w*)\b",
            question,
            re.I,
        )
        and re.search(r"\b(?:concentrat\w*|attention|focus)\b", question, re.I)
        and re.search(
            r"\b(?:teacher|student|pupil|child|class\w*|stud\w*|learning|lesson\w*)\b",
            question,
            re.I,
        )
    )


def _concentration_relevant(text: str) -> bool:
    return bool(
        re.search(r"\b(?:attention|concentrat\w*)\b", text, re.I)
        and re.search(r"\binterest\b", text, re.I)
        and re.search(r"\b(?:child|children|educator|teach\w*)\b", text, re.I)
    )


def _education_claim_issue(answer: str, routing: dict) -> bool:
    focused = routing.get("focused_route") or {}
    return focused.get("concept") == "integral_education" and bool(
        re.search(
            r"\b(?:not in sequence|never sequential|not sequential|not chronological|do not follow.{0,30}chronolog\w*)\b",
            answer,
            re.I,
        )
    )


def _education_relevant(text: str) -> bool:
    return (
        all(
            re.search(r"\b" + aspect + r"\b", text, re.I)
            for aspect in ("physical", "vital", "mental", "psychic", "spiritual")
        )
        and bool(re.search(r"\busually\b.{0,180}\bchronologic\w*\b", text, re.I))
        and bool(re.search(r"\ball\b.{0,20}\bcontinue\b", text, re.I))
        and bool(re.search(r"\b(?:end of (?:his|their) life|throughout life)\b", text, re.I))
    )


def _career_intent(question: str) -> bool:
    text = question.casefold()
    if re.search(r"\b(?:career\w*|vocation\w*|profession\w*|occupation\w*|livelihood)\b", text):
        return True
    return bool(
        re.search(r"\b(?:choos\w*|choice\w*|decid\w*|decision\w*)\b", text)
        and re.search(
            r"\b(?:engineering|engineer|teaching|teacher|medicine|doctor|job\w*|work|degree|stud(?:y|ies)|destiny)\b",
            text,
        )
    )


def _career_relevant(text: str) -> bool:
    """Require nearby career, choice and work/aptitude anchors in ACTUAL text.

    This deliberately conservative keyword gate is not semantic validation.
    General education, national destiny and teacher qualifications alone are
    not accepted as personal vocational guidance.
    """
    normalized = " ".join(text.casefold().split())
    if re.search(
        r"\b(?:national destiny|destiny of (?:the )?(?:nation|india|country)|swaraj)\b", normalized
    ):
        return False
    domain = r"\b(?:careers?|vocations?|professions?|occupations?|livelihoods?)\b"
    choice = r"\b(?:choos\w*|choice\w*|decid\w*|decision\w*|select\w*|pursu\w*|enter(?:ing)?|find(?:ing)?|follow(?:ing)?)\b"
    fit = r"\b(?:aptitud\w*|interest\w*|abilit\w*|capacit\w*|skill\w*|talent\w*|inclination\w*|nature|work|jobs?|calling)\b"
    for match in re.finditer(domain, normalized):
        nearby = normalized[max(0, match.start() - 220) : match.end() + 220]
        if re.search(choice, nearby) and re.search(fit, nearby):
            return True
    return False
