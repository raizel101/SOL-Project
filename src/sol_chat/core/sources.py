"""Source provenance, deduplication and syntactic answer validation."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote, urlsplit


def _deduplicate_passages(sources: list[dict]) -> tuple[list[dict], int]:
    """Collapse near-identical editions without changing the retained text/offsets."""
    kept, fingerprints = [], []
    for source in sources:
        words = tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", source["text"]).casefold()))
        shingles = {words[index : index + 5] for index in range(max(0, len(words) - 4))}
        duplicate = False
        for previous, prior_shingles in fingerprints:
            if words == previous:
                duplicate = True
                break
            if min(len(words), len(previous)) < 30:
                continue
            ratio = min(len(words), len(previous)) / max(len(words), len(previous))
            if (
                min(len(words), len(previous)) >= 64
                and ratio >= 0.85
                and words[:64] == previous[:64]
            ):
                duplicate = True
                break
            union = shingles | prior_shingles
            intersection = shingles & prior_shingles
            if union and len(intersection) / len(union) >= 0.82:
                duplicate = True
                break
            if (
                ratio >= 0.70
                and min(len(shingles), len(prior_shingles))
                and len(intersection) / min(len(shingles), len(prior_shingles)) >= 0.95
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(source)
            fingerprints.append((words, shingles))
    return kept, len(sources) - len(kept)


def _citation(source: dict) -> dict:
    keys = (
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
    return {key: source[key] for key in keys if key in source}


def source_role(source: dict) -> str:
    """Conservative URL/title provenance routing, not passage-level authorship.

    Broad library search keeps every role. Default chat uses known primary
    collection locations and the five supplied local documents only. An editor's
    preface within an original edition can still need human authorship review.
    """
    url = source.get("source_url", "")
    title = source.get("source_title", "")
    url = url if isinstance(url, str) else ""
    title = title if isinstance(title, str) else ""
    local_ids = {
        "local-sol-reference",
        "local-core-principles-2",
        "local-auro-logo",
        "local-the-four-soul-forces",
        "local-mother-s-symbol",
    }
    if not url and source.get("source_id") in local_ids and source.get("source_file"):
        return "user_reference"
    try:
        parsed = urlsplit(url)
        host, path = (parsed.hostname or "").casefold(), unquote(parsed.path).casefold()
    except ValueError:
        return "supplemental"
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        return "supplemental"
    if re.search(r"\b(?:satprem|notebooks)\b", title.casefold()) or re.search(
        r"/(?:satprem|disciples|notebooks)(?:/|$)", path
    ):
        return "supplemental"
    if host in {"incarnateword.in", "www.incarnateword.in"}:
        if path.startswith("/compilations/"):
            return "compilation"
        if re.match(r"^/(?:cwsa|cwm|agenda)/\d+/[^/]+", path):
            return "primary_original"
    if host in {"motherandsriaurobindo.in", "www.motherandsriaurobindo.in"}:
        if re.match(r"^/(?:sri-aurobindo|the-mother)/books/", path):
            if "/compilations/" in path or "compilation" in title.casefold():
                return "compilation"
            return "primary_original"
    if host in {
        "sriaurobindoashram.org",
        "www.sriaurobindoashram.org",
        "library.sriaurobindoashram.org",
    }:
        if re.match(r"^/(?:sriaurobindo/cwsa\d*|mother/cwm\d*)/(?:chapter|text|\d+)", path):
            return "primary_original"
        title_is_cwsa = re.search(
            r"\b(?:cwsa|complete works of sri aurobindo|collected works of sri aurobindo)\b",
            title,
            re.I,
        )
        title_is_cwm = re.search(r"\b(?:cwm|collected works of the mother)\b", title, re.I)
        if (
            path.endswith(".pdf")
            and "/download/" in path
            and (
                ("/sriaurobindo/" in path and title_is_cwsa)
                or ("/mother/" in path and title_is_cwm)
            )
        ):
            return "primary_original"
    return "supplemental"


def _validate_answer(
    answer: str, sources: list[dict], question: str, history: list[dict]
) -> tuple[list[str], str | None]:
    allowed = {source["citation"] for source in sources}
    canonical = re.findall(r"\[(S[1-9]\d*)\]", answer)
    # Recognize citation-like malformed identifiers, including lower case,
    # leading zeros, spaces and comma-separated variants; none are silently fixed.
    citation_like = re.findall(r"\[\s*[Ss]\s*\d[^\]]*\]", answer)
    if any(not re.fullmatch(r"\[S[1-9]\d*\]", item) for item in citation_like):
        return [], "The response used a malformed source identifier."
    invalid = set(canonical) - allowed
    if invalid:
        return (
            [],
            "The response referred to source identifiers that were not supplied: "
            + ", ".join(sorted(invalid)),
        )
    if not canonical:
        return [], "The response did not include a source citation."
    normal = lambda text: " ".join(text.split())
    reference = {source["citation"]: normal(source["text"]) for source in sources}
    user_words = [normal(question)] + [
        normal(message["content"]) for message in history if message["role"] == "user"
    ]
    # Double/guillemet delimiters must be balanced; newline is ordinary
    # quotation content, not a way to escape attribution checks.
    expected = None
    pairs = {'"': '"', "“": "”", "«": "»"}
    delimiters = set(pairs) | set(pairs.values())
    for character in answer:
        if character not in delimiters:
            continue
        if expected is None:
            if character not in pairs:
                return [], "The response contains an unmatched quotation delimiter."
            expected = pairs[character]
        elif character == expected:
            expected = None
        else:
            return [], "The response contains mismatched or nested quotation delimiters."
    if expected is not None:
        return [], "The response contains an unmatched quotation delimiter."
    # Recognized double/guillemet forms, including multiline/short strings,
    # and adjacent-cited single-quote forms are checked conservatively.
    quotations = list(re.finditer(r'"([^"]*)"|“([^“”]*)”|«([^«»]*)»', answer))
    quotations += list(re.finditer(r"(?<!\w)[‘']([^’'\n]+)[’'](?!\w)\s*(?=\[S[1-9]\d*\])", answer))
    for match in quotations:
        quotation = normal(next(group for group in match.groups() if group is not None))
        if not quotation:
            return [], "The response contains an empty quotation."
        after = re.match(r"\s*\[(S[1-9]\d*)\]", answer[match.end() :])
        # An adjacent source citation ALWAYS means source attribution, even if
        # a user/history message supplied the same invented quotation.
        if after:
            if quotation not in reference.get(after.group(1), ""):
                return [], "A quotation was not an exact match to its adjacent cited passage."
            continue
        prefix = answer[max(0, match.start() - 80) : match.start()]
        explicit_echo = re.search(
            r"(?:you (?:said|wrote|mentioned)|your (?:words|message|question)(?: (?:were|was))?)\s*[:,]?\s*$",
            prefix,
            re.I,
        )
        attribution = re.search(
            r"\b(?:aurobindo|mother|author|teaching|passage|source)\b", prefix, re.I
        )
        if explicit_echo and not attribution and any(quotation in words for words in user_words):
            continue
        if not after:
            return [], "A quotation was not an exact match to its adjacent cited passage."
    return list(dict.fromkeys(canonical)), None
