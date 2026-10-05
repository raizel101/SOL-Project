"""Limited deterministic safety routing, not clinical assessment.

Keep the limitations visible. Changes require explicit regression fixtures and
qualified review; historical source text is never edited by these checks."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

from sol_chat.adapters.corpus import RAG

from .errors import EngineError
from .settings import Settings
from .validation import _history, _question


def urgent_trigger(question: str) -> bool:
    """Narrow phrase routing, not a diagnosis, risk score, or multilingual screen."""
    text = re.sub(r"\s+", " ", question.casefold().replace("’", "'"))
    # Remove only locally reported examples in clear academic/fictional context,
    # never the entire message: an essay question can also disclose present intent.
    academic_prefix = re.match(
        r"(?:for (?:an? )?(?:essay|research|novel)|a character|in (?:a|the) novel|what (?:does|is)|explain|define|write (?:an?|the))\b",
        text,
    )
    academic_topic = re.search(
        r"\b(?:meaning|definition|essay|suicide|suicidal|student says|character)\b", text
    )
    if academic_prefix and academic_topic:
        text = re.sub(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\'[^\'\n]+\'(?!\w)', " ", text)
        text = re.sub(
            r"\b(?:people|a character|the character|a student|the student) (?:say|says|said) (?:that )?[^.!?]*",
            " ",
            text,
        )
    patterns = (
        r"\bi (?:(?:want|intend|plan|am planning|am going) to |(?:will|might|may) )(?:kill|hurt|harm) myself\b",
        r"\bi(?:'m| am) (?:feeling )?suicidal\b",
        r"\bi (?:want|intend|plan|am planning|am going) to (?:end my life|die|commit suicide)\b",
        r"\bi(?:'m| am) thinking (?:about|of) (?:suicide|killing myself|ending my life)\b",
        r"\bi (?:have )?(?:taken|took) (?:an? )?overdose\b",
        r"\bi (?:have )?(?:swallowed|taken|took) (?:all|too many|a lot of) (?:my |the )?(?:pills|tablets|medicine)\b",
        r"\bi (?:can't|cannot|do not|don't) (?:keep|trust) myself safe\b",
        r"\bi (?:am|'m) in immediate danger\b",
        r"\bi (?:don't|do not) want to (?:live|be alive)\b",
        r"\bi (?:wish|hope) i (?:wouldn't|would not|won't|will not) wake up\b",
        r"\bi wish i (?:were|was) dead\b",
        r"\bi have no reason to (?:stay alive|live)\b",
        r"\b(?:everyone|my family) would be better off without me\b",
        r"\bi(?:'d| would) be better off dead\b",
        r"\b(?:my (?:friend|sister|brother|partner|child)|someone (?:here|with me)) (?:is|says (?:they are|they're)) (?:suicidal|in immediate danger)\b",
        r"\bmy (?:friend|sister|brother|partner|child|student) (?:wants|plans|is going) to (?:(?:kill|harm) (?:himself|herself|themselves)|die)\b",
        r"\bmy (?:friend|sister|brother|partner|child) (?:(?:just|already) )?(?:took|has taken) (?:an? )?overdose\b",
        r"\b(?:main|mai|mein) (?:khud ko |apne aap ko )?(?:marna|maar(?:na)?|suicide karna) (?:chahta|chahti|chahata|chahati) (?:hu|hoon|hun)\b",
        r"मैं (?:खुद को |अपने आप को )?(?:मारना|मरना|आत्महत्या करना) चाहत[ाी] हूँ",
        r"मैं (?:खुद को |अपने आप को )?नुकसान पहुँचा?ना चाहत[ाी] हूँ",
        r"\bmujhe apni jaan leni hai\b",
        r"मुझे जीना नहीं है",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def treatment_boundary_trigger(question: str) -> bool:
    """Narrow first-person medication-change request routing, not assessment."""
    text = re.sub(r"\s+", " ", question.casefold().replace("’", "'"))
    medicine = r"(?:medicin\w*|medicat\w*|prescri\w*|antidepressant\w*|pills?|tablets?)"
    if not re.search(r"\b(?:i|my|me)\b", text) or not re.search(r"\b" + medicine + r"\b", text):
        return False
    action = r"\b(?:stop|quit|discontinue|change|reduce|increase|adjust|skip|replace)\s+(?:taking\s+|using\s+)?"
    if re.search(
        action + r"(?:it\b|them\b|(?:my|the|these|this)\s+" + medicine + r"\b|" + medicine + r"\b)",
        text,
    ):
        return True
    # Recognize a name explicitly introduced as prescribed in this message;
    # this is not a medication dictionary or a treatment recommendation.
    named = re.findall(r"\bprescrib(?:ed|es)\s+(?:me\s+)?([a-z][a-z0-9-]{2,40})\b", text)
    return any(re.search(action + re.escape(name) + r"\b", text) for name in named)


def _personal_distress(question: str) -> bool:
    """Limited context hint; never a diagnosis or an assessment of risk."""
    text = re.sub(r"\s+", " ", question.casefold().replace("’", "'"))
    positive_distress = False
    personal = r"\bi(?:'m| am| feel| have| keep feeling)\b.{0,60}?\b(low|sad|depressed|down|hopeless|empty|distressed|anxious|overwhelmed)\b"
    for match in re.finditer(personal, text):
        preceding = text[max(match.start(), match.start(1) - 24) : match.start(1)]
        following = text[match.end(1) : match.end(1) + 30]
        if match.group(1) == "low" and re.match(
            r"\s+(?:scores?|marks?|grades?|income|battery|balance|temperature)\b", following
        ):
            continue
        if not re.search(r"\b(?:not|never|no longer)\b", preceding):
            positive_distress = True
            break
    if re.search(r"\bmy (?:depression|sadness|low mood)\b", text):
        positive_distress = True
    return positive_distress


def persistent_distress_trigger(question: str) -> bool:
    """Limited first-person duration/impairment rule, not a depression diagnosis."""
    if not _personal_distress(question):
        return False
    text = re.sub(r"\s+", " ", question.casefold().replace("’", "'"))
    duration = re.search(
        r"\b(?:for|over|during|past|these|lasted|lasting)\s+(?:(?:a|an|the|many|few|several|one|two|three|four|six|\d+)\s+)?(?:days?|weeks?|months?|years?)\b",
        text,
    )
    impairment = re.search(
        r"\b(?:can't|cannot|unable to|struggl\w* to)\b.{0,60}\b(?:manage|attend|study|work|classes|school|sleep|eat|function|bed)\b",
        text,
    )
    return bool(duration or impairment)


def _support_evidence_risk(text: str) -> bool:
    """Conservative lexical exclusions only for personal-distress chat evidence.

    Historical passages remain untouched and available in library search. This
    is not a judgment about the authors or a comprehensive safety classifier.
    """
    return bool(
        re.search(
            r"\b(?:subconscient|tamasic|vairagya|impurity|hostile forces|(?:spiritual|moral|nervous|vital) weakness)\b"
            r"|\b(?:call.{0,35}(?:divine|soul)|soul.{0,15}call)\b"
            r"|\b(?:weak\w*.{0,24}vital|dark\w*.{0,24}spirit|(?:deficien\w*|lack\w*).{0,24}faith)\b",
            text,
            re.I,
        )
    )


def _support_source_risk(source: dict) -> bool:
    # Dedicated historical suffering/depression discussions have not been
    # approved as personalized support evidence. Do not depend on a short
    # extracted prefix omitting their clinical-sounding/metaphysical context.
    metadata = (
        unquote(str(source.get("source_url", ""))).replace("-", " ")
        + " "
        + str(source.get("source_title", ""))
    )
    url = str(source.get("source_url", ""))
    educational_scope = (
        source.get("source_role") == "user_reference"
        or url.startswith("https://incarnateword.in/cwm/12/")
        or url.startswith("https://motherandsriaurobindo.in/The-Mother/books/on-education/")
    )
    # Personal letters and the broader spiritual library are not default
    # personalized emotional-support evidence. Scope is not human approval.
    return (
        not educational_scope
        or _support_evidence_risk(source["text"])
        or bool(
            re.search(
                r"\b(?:depress\w*|despond\w*|sorrow|suffering|sadness|difficulties of the path)\b",
                metadata,
                re.I,
            )
        )
    )


def _support_context(question: str, history: list[dict]) -> bool:
    if _personal_distress(question):
        return True
    if not history or not _personal_distress(history[-2]["content"]):
        return False
    # Only an explicit academic/topic-change question clears this narrow
    # carryforward. Prior assistant reassurances never establish safety.
    academic = re.match(
        r"(?:what|explain|define|compare|summari[sz]e|how did|who)\b", question, re.I
    )
    topic = re.search(
        r"\b(?:education|philosophy|teaching|mother|aurobindo|meditation|history|algebra|weather)\b",
        question,
        re.I,
    )
    return not bool(academic and topic)


def diagnosis_boundary_trigger(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:diagnose (?:me|my \w+)|(?:do i|does my (?:child|student|friend)) have (?:adhd|depression|anxiety|autism)|is my (?:child|student) (?:autistic|depressed))\b",
            question,
            re.I,
        )
    )


def _support_answer_issue(text: str) -> bool:
    # Fail closed on these recognized patterns even when negated. Such false
    # positives are safer to review than silently accepting personalized blame.
    return _support_evidence_risk(text) or bool(
        re.search(
            r"\byou (?:are|have|suffer from) (?:clinically |clinical )?(?:depression|depressed|an anxiety disorder)\b"
            r"|\byou (?:will|are guaranteed to) (?:pass|recover|be cured)\b",
            text,
            re.I,
        )
    )


def _supportive_abstention() -> str:
    return (
        "That sounds difficult. Feeling upset does not make you a failure. I couldn't find a suitable teaching in this question's retrieved passages, so I won't use historical spiritual explanations for how you feel. "
        "One optional next step is to tell a trusted person what happened and choose one small, manageable task together. If these feelings last or interfere with daily life, consider qualified professional support. What feels hardest right now?"
    )


def _urgent_message(question: str) -> dict:
    original = RAG.crisis_response()
    if re.search(
        r"\bmy (?:friend|sister|brother|partner|child|student)\b|\bsomeone (?:here|with me)\b",
        question,
        re.I,
    ):
        original = dict(
            original,
            answer="If you or someone else may be in immediate danger, contact local emergency services now and ask a trusted person physically nearby to help. If it is safe for you, stay with an at-risk person and do not promise to keep the danger secret. This chatbot cannot contact help or monitor anyone's safety. Is someone physically nearby who can help immediately?",
        )
    return original


def ungrounded_prediction_trigger(question: str) -> bool:
    """Narrow request for a quote proving personal future wealth, not all prediction."""
    text = question.casefold().replace("’", "'")
    requirements = (
        r"\b(?:mother|aurobindo)\b",
        r"\b(?:quote|quotation)\b",
        r"\b(?:prove|proves|proving|proof|guarantee|guarantees|guaranteed)\b",
        r"\b(?:i|me|my)\b",
        r"\b(?:rich|wealthy|wealth|millionaire)\b",
        r"\b(?:will|become|future|tomorrow|next month|next year)\b",
    )
    return all(re.search(pattern, text) for pattern in requirements)


def urgent_followup_trigger(question: Any, history: Any, settings: Settings | None = None) -> bool:
    """Safely recognize a narrow referential continuation of a prior urgent turn.

    No assistant statement is treated as a risk assessment. An intervening
    nonreferential user topic stops carryforward. Invalid history returns False
    so normal validation still rejects it; a current explicit crisis is separate.
    """
    settings = settings or Settings()
    try:
        question = _question(question, settings)
        history = _history(history, settings)
    except EngineError:
        return False
    phrases = {
        "what should i do now",
        "what do i do now",
        "what should i do",
        "what next",
        "what now",
        "what should i do next",
        "can you help",
        "can you help me",
        "can you help me now",
        "help me",
        "please help",
        "please help me",
        "what can i do now",
        "i need help now",
    }
    normalize = lambda value: re.sub(r"\s+", " ", value.casefold()).strip(" \t\r\n?.!")
    if normalize(question) not in phrases:
        return False
    for message in reversed(history):
        if message["role"] != "user":
            continue
        if urgent_trigger(message["content"]):
            return True
        if normalize(message["content"]) not in phrases:
            return False
    return False
