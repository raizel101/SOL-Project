"""Orchestrate read-only retrieval, bounded local generation and validation.

This module retains the public Engine API and imported helper aliases for
existing integrations. It never provisions models or saves conversations."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from sol_chat.adapters.corpus import RAG
from sol_chat.paths import CORPUS_DIR as CORPUS

from .errors import EngineError
from .policy import REPAIR_INSTRUCTION, SYSTEM_PROMPT
from .prompts import _build_messages, _message_cost, _prompt_source
from .relevance import (
    _career_intent,
    _career_relevant,
    _concentration_intent,
    _concentration_relevant,
    _education_claim_issue,
    _education_relevant,
)
from .responses import _base
from .safety import (
    _personal_distress,
    _support_answer_issue,
    _support_context,
    _support_evidence_risk,
    _support_source_risk,
    _supportive_abstention,
    _urgent_message,
    diagnosis_boundary_trigger,
    persistent_distress_trigger,
    treatment_boundary_trigger,
    ungrounded_prediction_trigger,
    urgent_followup_trigger,
    urgent_trigger,
)
from .settings import Settings
from .sources import _citation, _deduplicate_passages, _validate_answer, source_role
from .validation import _history, _question


class Engine:
    """Reusable local engine. Public operations return JSON-compatible objects.

    Constructor validation raises EngineError. Invalid request bodies and local
    service failures return mode='error' with an actionable structured error.
    No history or personal message is persisted by this engine.
    """

    def __init__(self, db_path: Path | str | None = None, config: dict | None = None):
        self.db_path = Path(db_path) if db_path is not None else CORPUS / "local_rag.sqlite"
        self.settings = Settings.from_dict(config)

    def _failure(self, exc: Exception) -> dict:
        if not isinstance(exc, EngineError):
            if isinstance(exc, RAG.RagError):
                exc = EngineError("local_service_unavailable", str(exc), True)
            elif isinstance(exc, (sqlite3.Error, OSError)):
                exc = EngineError(
                    "retrieval_unavailable",
                    "The local source index cannot be read. Check the index file and restart SOL Chat; the corpus has not been changed.",
                    True,
                )
            else:
                exc = EngineError(
                    "local_service_error",
                    "The local service returned an unexpected response. Check its logs and try again.",
                    True,
                )
        result = _base("error", self.settings.model)
        result["answer"] = exc.message
        result["error"] = exc.as_dict()
        return result

    def _check_model(self, timeout_seconds: float | None = None) -> dict:
        config = self.settings.ollama_config()
        config["request_timeout_seconds"] = min(5, config["request_timeout_seconds"])
        if timeout_seconds is not None:
            config["request_timeout_seconds"] = min(
                config["request_timeout_seconds"], timeout_seconds
            )
        return RAG.check_ollama(config)

    def _generate(self, messages: list[dict], timeout_seconds: float | None = None) -> dict:
        config = self.settings.ollama_config()
        if timeout_seconds is not None:
            config["request_timeout_seconds"] = min(
                config["request_timeout_seconds"], timeout_seconds
            )
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": {
                "num_ctx": self.settings.num_ctx,
                "num_predict": self.settings.num_predict,
                "temperature": self.settings.temperature,
            },
        }
        return RAG.ollama_json(config, "/api/chat", payload)

    def _retrieve(self, question: str, limit: int) -> dict:
        if not self.db_path.is_file():
            raise EngineError(
                "missing_index",
                "The local retrieval index is missing. Restore outputs/auro_guide_corpus/local_rag.sqlite before starting SOL Chat.",
            )
        retrieved = RAG.search(self.db_path, question, limit)
        for source in retrieved["results"]:
            source["source_role"] = source_role(source)
        return retrieved

    def _focused_education(self, question: str) -> dict | None:
        """A narrow, auditable lookup for the explicitly named integral-education concept.

        The Mother's canonical Education chapter defines its five aspects,
        although it need not contain every everyday-life word in the question.
        This is curated chapter routing, not semantic search or source invention.
        """
        if not re.search(r"\bintegral\s+education\b", question, re.I):
            return None
        urls = (
            "https://motherandsriaurobindo.in/The-Mother/books/on-education/#education",
            "https://incarnateword.in/cwm/12/education",
        )
        if not self.db_path.is_file():
            raise EngineError(
                "missing_index",
                "The local retrieval index is missing. Restore local_rag.sqlite before using SOL Chat.",
            )
        with contextlib.closing(
            sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM chunks WHERE source_url IN (?,?) ORDER BY id LIMIT 32", urls
            ).fetchall()
        candidates = []
        for row in rows:
            text = row["text"]
            definition = re.search(
                r"education\s+to\s+be\s+complete\s+must\s+have\s+five", text, re.I
            )
            if not definition:
                continue
            start, end = definition.start(), min(len(text), definition.start() + 1400)
            exact = text[start:end]
            # A recognized location alone is not enough: require the actual
            # five-aspect definition in the exact selected text.
            if not all(
                re.search(r"\b" + aspect + r"\b", exact, re.I)
                for aspect in ("physical", "vital", "mental", "psychic", "spiritual")
            ):
                continue
            source = {
                "citation": f"S{len(candidates) + 1}",
                "chunk_id": row["chunk_id"],
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "source_title": row["title"],
                "source_url": row["source_url"],
                "source_file": row["source_file"],
                "source_locator": json.loads(row["source_locator"]),
                "text_offsets": {
                    "start": row["start_char"] + start,
                    "end_exclusive": row["start_char"] + end,
                },
                "bm25": None,
                "snippet": exact[:240],
                "text": exact,
            }
            source["source_role"] = source_role(source)
            if source["source_role"] == "primary_original":
                candidates.append(source)
        if not candidates:
            return None
        return {
            "query": question,
            "search_terms": ["integral", "education"],
            "results": candidates,
            "match_mode": "focused canonical Education chapter and exact five-aspect definition",
            "retrieval": "read-only canonical chapter lookup; not semantic embeddings",
            "focused_route": {
                "concept": "integral_education",
                "method": "known primary chapter plus exact definition anchors",
                "candidate_count": len(candidates),
                "source_windows_preserve_record_offsets": True,
            },
        }

    def _focused_concentration(self, question: str) -> dict | None:
        """Known Mental Education paragraph, not general semantic retrieval."""
        if not _concentration_intent(question):
            return None
        if not self.db_path.is_file():
            raise EngineError(
                "missing_index", "The local retrieval index is missing. Restore local_rag.sqlite."
            )
        urls = (
            "https://incarnateword.in/cwm/12/mental-education",
            "https://motherandsriaurobindo.in/The-Mother/books/on-education/#mental-education",
        )
        with contextlib.closing(
            sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM chunks WHERE source_url IN (?,?) "
                "ORDER BY CASE WHEN record_id LIKE 'incarnate-%' THEN 0 ELSE 1 END, id LIMIT 64",
                urls,
            ).fetchall()
        candidates = []
        for row in rows:
            text = row["text"]
            anchor = re.search(
                r"Undeniably, what most impedes mental progress in children", text, re.I
            )
            if not anchor:
                continue
            finish = re.search(
                r"according to the need and the circumstances\.", text[anchor.start() :], re.I
            )
            if not finish:
                continue
            start, end = anchor.start(), anchor.start() + finish.end()
            exact = text[start:end]
            if (
                len(exact) > 1200
                or not _concentration_relevant(exact)
                or re.search(r"<[/!]?[a-z]", exact, re.I)
            ):
                continue
            source = {
                "citation": f"S{len(candidates) + 1}",
                "chunk_id": row["chunk_id"],
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "source_title": row["title"],
                "source_url": row["source_url"],
                "source_file": row["source_file"],
                "source_locator": json.loads(row["source_locator"]),
                "text_offsets": {
                    "start": row["start_char"] + start,
                    "end_exclusive": row["start_char"] + end,
                },
                "bm25": None,
                "snippet": exact[:240],
                "text": exact,
            }
            source["source_role"] = source_role(source)
            if source["source_role"] == "primary_original":
                candidates.append(source)
        if not candidates:
            return None
        return {
            "query": question,
            "search_terms": ["attention", "interest", "education"],
            "results": candidates,
            "match_mode": "focused canonical Mental Education paragraph with concentration anchors",
            "retrieval": "read-only canonical paragraph lookup; not semantic embeddings",
            "focused_route": {
                "concept": "educational_concentration",
                "method": "known primary chapter plus complete paragraph anchors",
                "candidate_count": len(candidates),
                "source_windows_preserve_record_offsets": True,
            },
        }

    def status(self, check_model: bool = True) -> dict:
        result = {
            "application": "SOL Chat",
            "ready": False,
            "index_ready": False,
            "model_ready": None,
            "model": self.settings.model,
            "ollama_url": self.settings.ollama_base_url,
            "index_path": str(self.db_path.resolve()),
            "conversation_storage": "not persisted",
            "retrieval": "read-only SQLite FTS5 BM25; not semantic embeddings",
            "context_tokens": self.settings.num_ctx,
            "max_response_tokens": self.settings.num_predict,
            "crisis_routing": "limited phrase-based prototype, not clinical assessment",
            "chat_source_roles": ["primary_original", "user_reference"],
            "career_relevance": "conservative word-pattern gate; not semantic validation",
            "treatment_routing": "limited first-person medication-change phrases; not clinical assessment",
            "persistent_distress_routing": "limited first-person duration/impairment phrases; not diagnosis",
            "prediction_routing": "limited requests for author quotes guaranteeing personal wealth",
            "error": None,
        }
        try:
            if not isinstance(check_model, bool):
                raise EngineError("invalid_request", "check_model must be true or false.")
            if not self.db_path.is_file():
                raise EngineError(
                    "missing_index",
                    "The local retrieval index is missing. Restore local_rag.sqlite; no index was created by this engine.",
                )
            with contextlib.closing(
                sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)
            ) as connection:
                metadata = dict(connection.execute("SELECT key,value FROM meta"))
                # Confirm the expected tables exist without a full corpus scan.
                connection.execute("SELECT id FROM chunks LIMIT 1").fetchone()
                connection.execute("SELECT rowid FROM chunks_fts LIMIT 1").fetchone()
            result["index_ready"] = True
            result["index_metadata"] = metadata
            if check_model:
                model = self._check_model()
                result["model_status"] = model
                result["model_ready"] = bool(model["installed"])
                if not model["installed"]:
                    result["error"] = EngineError(
                        "model_not_installed",
                        f"Local model {self.settings.model} is not installed. Select an already installed local model or create the SOL alias from your installed model; no model was downloaded.",
                    ).as_dict()
            result["ready"] = result["index_ready"] and result["model_ready"] is True
        except (EngineError, RAG.RagError, sqlite3.Error, OSError) as exc:
            result["error"] = self._failure(exc)["error"]
        return result

    def search(self, question: Any, limit: int | None = None) -> dict:
        try:
            question = _question(question, self.settings)
            limit = self.settings.top_k_sources if limit is None else limit
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 6:
                raise EngineError("invalid_limit", "Search limit must be an integer from 1 to 6.")
            retrieved = self._retrieve(question, limit)
            result = _base("retrieval", self.settings.model)
            result.update(
                source_passages=retrieved["results"],
                citations=[_citation(source) for source in retrieved["results"]],
                search_terms=retrieved["search_terms"],
                match_mode=retrieved.get("match_mode"),
                retrieval=retrieved["retrieval"],
            )
            if "fallback" in retrieved.get("match_mode", ""):
                result["warnings"].append(
                    "Only partial keyword matches were found; inspect passage relevance before using them."
                )
            return result
        except (EngineError, RAG.RagError, sqlite3.Error, OSError) as exc:
            return self._failure(exc)

    def chat(self, question: Any, history: Any = None, dry_run: bool = False) -> dict:
        llm_called = False
        attempts, initial_failure, result = 0, None, None
        deadline = time.monotonic() + min(120, self.settings.request_timeout_seconds)

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise EngineError(
                    "ollama_timeout",
                    "The local response's total time budget elapsed. Please try a narrower question.",
                    True,
                )
            return value

        def generate(messages):
            nonlocal attempts, llm_called
            timeout = remaining()
            attempts += 1
            llm_called = True
            response = self._generate(messages, timeout_seconds=timeout)
            remaining()  # Withhold late responses; never begin another call past the deadline.
            if not isinstance(response, dict) or response.get("done") is not True:
                raise EngineError(
                    "incomplete_response",
                    "Local Ollama did not return a completed answer. Try again or check its local logs.",
                    True,
                )
            if response.get("done_reason") == "length":
                raise EngineError(
                    "truncated_response",
                    "Ollama ended generation because its response-token limit was reached.",
                    True,
                )
            message = response.get("message", {})
            answer = message.get("content", "") if isinstance(message, dict) else ""
            if not isinstance(answer, str) or not answer.strip():
                raise EngineError(
                    "empty_response",
                    "The local model returned an empty answer. Try again or check its local logs.",
                    True,
                )
            if len(answer.encode("utf-8")) > 32000:
                raise EngineError(
                    "response_too_large",
                    "The local model response exceeded the output limit.",
                    True,
                )
            return answer, response

        try:
            question = _question(question, self.settings)
            # A new explicit danger disclosure takes priority even if an old
            # conversation has grown too large. No supplied history is consumed
            # or sent to a model on this deterministic route.
            if urgent_trigger(question):
                original = _urgent_message(question)
                result = _base("urgent_support", self.settings.model)
                result.update(
                    answer=original["answer"],
                    warnings=original["warnings"]
                    + [
                        "SOL's narrow English/Hindi phrase rules cannot reliably detect or assess emergencies. Seek human help; this chat is not monitored."
                    ],
                )
                return result
            if treatment_boundary_trigger(question):
                result = _base("treatment_boundary", self.settings.model)
                result.update(
                    answer="I can't decide whether you should stop or change prescribed medicine. Please discuss any changes with the clinician who prescribed it; don't change treatment based on this chatbot or replace it with meditation on this advice. You can ask them how reflective practices might complement your care. Would you like help preparing questions for them?",
                    warnings=[
                        "This is a limited phrase-based treatment-boundary response, not a clinical assessment. It cannot detect every medication or health-care concern."
                    ],
                )
                return result
            if diagnosis_boundary_trigger(question):
                result = _base("human_support", self.settings.model)
                result.update(
                    answer="This philosophy chatbot cannot diagnose you or another person. Please discuss the concern with a qualified health-care professional rather than inferring a condition from difficulty concentrating or from spiritual texts. A teacher can share specific observations without labeling the student. Would you like help preparing questions for a professional?",
                    warnings=[
                        "This is a limited explicit-diagnosis-request boundary, not medical assessment or comprehensive detection. No model was called."
                    ],
                )
                return result
            if persistent_distress_trigger(question):
                result = _base("human_support", self.settings.model)
                result.update(
                    answer="That sounds difficult to carry, especially when it lasts or interferes with daily life. You don't have to manage it alone. Consider speaking with a qualified mental-health professional or health-care provider, and tell a trusted teacher, family member or friend what has been happening. I can offer gentle reflection, but spiritual ideas are not clinical explanations, and this chat cannot diagnose or replace professional care. Is there someone you could reach out to today?",
                    warnings=[
                        "This limited first-person duration/impairment phrase rule is not a diagnosis or comprehensive distress detector. No model was called."
                    ],
                )
                return result
            if ungrounded_prediction_trigger(question):
                result = _base("insufficient_sources", self.settings.model)
                result.update(
                    answer="A quote cannot establish or guarantee your personal future wealth, and I won't invent one or attribute a prediction to the Mother or Sri Aurobindo. If you want, we can explore a specific authentic teaching about work or material life using source passages.",
                    warnings=[
                        "This is a limited rule for personal guaranteed-wealth quotation requests, not a detector for every unsupported prediction. No model was called."
                    ],
                )
                return result
            history = _history(history, self.settings)
            if urgent_followup_trigger(question, history, self.settings):
                preceding = next(
                    (
                        message["content"]
                        for message in reversed(history)
                        if message["role"] == "user" and urgent_trigger(message["content"])
                    ),
                    question,
                )
                original = _urgent_message(preceding)
                result = _base("urgent_support", self.settings.model)
                result.update(
                    answer=original["answer"],
                    warnings=original["warnings"]
                    + [
                        "This limited referential follow-up rule carries forward a recent urgent user disclosure; it is not an assessment of current safety."
                    ],
                )
                return result
            if not isinstance(dry_run, bool):
                raise EngineError("invalid_request", "dry_run must be true or false.")
            # A short referential follow-up can use the most recent user turn as
            # retrieval context; old assistant statements never become sources.
            retrieval_question = question
            if history and (
                len(RAG.search_terms(question)) < 3
                or re.search(r"\b(?:that|this|it|more|explain)\b", question.casefold())
            ):
                retrieval_question = history[-2]["content"] + " " + question
            # Search a larger candidate pool so highly ranked secondary commentary
            # cannot crowd every original passage out of the local prompt.
            retrieved = (
                self._focused_education(retrieval_question)
                or self._focused_concentration(retrieval_question)
                or self._retrieve(retrieval_question, 50)
            )
            candidate_roles = {
                role: sum(source["source_role"] == role for source in retrieved["results"])
                for role in ("primary_original", "user_reference", "compilation", "supplemental")
            }
            eligible = [
                source
                for source in retrieved["results"]
                if source["source_role"] == "primary_original"
            ]
            eligible += [
                source
                for source in retrieved["results"]
                if source["source_role"] == "user_reference"
            ]
            support_context = _support_context(question, history)
            support_rejected = 0
            if support_context:
                related = [source for source in eligible if not _support_source_risk(source)]
                support_rejected = len(eligible) - len(related)
                eligible = related
            concentration_gate = _concentration_intent(retrieval_question)
            concentration_rejected = 0
            if concentration_gate:
                related = [source for source in eligible if _concentration_relevant(source["text"])]
                concentration_rejected = len(eligible) - len(related)
                eligible = related
            career_gate = _career_intent(retrieval_question)
            career_rejected = 0
            if career_gate:
                related = [source for source in eligible if _career_relevant(source["text"])]
                career_rejected = len(eligible) - len(related)
                eligible = related
            eligible, duplicate_count = _deduplicate_passages(eligible)
            selected = [
                dict(source, citation=f"S{number + 1}")
                for number, source in enumerate(eligible[: self.settings.top_k_sources])
            ]
            source_routing = {
                "candidate_pool_limit": 50,
                "retrieved_candidates": len(retrieved["results"]),
                "candidate_roles": candidate_roles,
                "eligible_candidates": len(eligible),
                "excluded_candidates": candidate_roles["compilation"]
                + candidate_roles["supplemental"],
                "selected_before_prompt_budget": len(selected),
                "near_duplicate_candidates_removed": duplicate_count,
                "career_gate": {
                    "enabled": career_gate,
                    "candidate_passages_rejected": career_rejected,
                    "bounded_passages_rejected": 0,
                },
                "support_context": {
                    "enabled": support_context,
                    "candidate_passages_rejected": support_rejected,
                    "screen": "limited lexical exclusions, not clinical assessment",
                },
                "concentration_gate": {
                    "enabled": concentration_gate,
                    "candidate_passages_rejected": concentration_rejected,
                    "bounded_passages_rejected": 0,
                },
                "focused_route": retrieved.get("focused_route"),
            }
            if not selected:
                result = _base("insufficient_sources", self.settings.model)
                result.update(
                    answer="I couldn't find a useful original-author passage or supplied reference for this question in the local search results. Could you describe what feels difficult, or name the teaching you want to explore?",
                    search_terms=retrieved["search_terms"],
                    source_routing=source_routing,
                    warnings=[
                        "Default chat excludes compilations, secondary commentary and uncertain authorship. They remain available through library search; this does not mean no original teaching exists elsewhere in the collection."
                    ],
                )
                if career_gate:
                    result["answer"] = (
                        "I couldn't find an original-author passage about choosing a career that meets this question's relevance check. I won't use general education or national destiny as evidence for your personal choice. What interests and practical constraints matter to you? A trusted career adviser or teacher can help you compare your options."
                    )
                    result["warnings"].append(
                        "Career relevance is a conservative word-pattern check, not semantic proof; it can miss relevant passages. No model was called."
                    )
                elif concentration_gate:
                    result["answer"] = (
                        "I couldn't find a suitable concentration-specific educational passage in the retrieved evidence. I won't treat a general teaching passage as proof about attention or diagnose the student. Could you describe the learning task and what you have already tried?"
                    )
                elif support_context:
                    result.update(mode="supportive_abstention", answer=_supportive_abstention())
                    result["warnings"].append(
                        "Unsuitable historical cause/blame passages were excluded by a limited lexical rule. No model was called; this is not a comprehensive safety screen."
                    )
                return result
            messages, sources, prompt_warnings = _build_messages(
                question, history, selected, self.settings, reserve_repair=True
            )
            education_gate = bool(
                (retrieved.get("focused_route") or {}).get("concept") == "integral_education"
            )
            if education_gate and not all(
                _education_relevant(source["text"]) for source in sources
            ):
                result = _base("insufficient_sources", self.settings.model)
                result.update(
                    answer="The definition excerpts that fit this chat do not retain all five aspects and their chronological/lifelong qualifications. I won't explain the definition using unsupplied text. Please shorten the prior conversation or ask a narrower question.",
                    source_routing=source_routing,
                    warnings=[
                        "Exact bounded definition anchors were incomplete; no model was called."
                    ],
                )
                return result
            if concentration_gate:
                bounded_related = [
                    source for source in sources if _concentration_relevant(source["text"])
                ]
                source_routing["concentration_gate"]["bounded_passages_rejected"] = len(
                    sources
                ) - len(bounded_related)
                if not bounded_related:
                    result = _base("insufficient_sources", self.settings.model)
                    result.update(
                        answer="The exact excerpts that fit this chat do not retain the educational attention and interest anchors. I won't infer concentration guidance from unsupplied text.",
                        source_routing=source_routing,
                        warnings=[
                            "The bounded concentration relevance check failed; no model was called. This is not semantic or clinical validation."
                        ],
                    )
                    return result
                if len(bounded_related) != len(sources):
                    messages, sources, extra_warnings = _build_messages(
                        question, history, bounded_related, self.settings, reserve_repair=True
                    )
                    prompt_warnings.extend(extra_warnings)
            if career_gate:
                bounded_related = [source for source in sources if _career_relevant(source["text"])]
                source_routing["career_gate"]["bounded_passages_rejected"] = len(sources) - len(
                    bounded_related
                )
                if not bounded_related:
                    result = _base("insufficient_sources", self.settings.model)
                    result.update(
                        answer="The source excerpts that fit this chat's context do not provide clear career-choice evidence. I won't infer your vocation or destiny from an unrelated passage. Could you describe your interests and the options you're considering?",
                        source_routing=source_routing,
                        warnings=[
                            "Career anchors were absent from the exact bounded excerpts; unsupplied text was not used as evidence. No model was called. This keyword check can miss relevant material."
                        ],
                    )
                    return result
                if len(bounded_related) != len(sources):
                    messages, sources, extra_warnings = _build_messages(
                        question, history, bounded_related, self.settings, reserve_repair=True
                    )
                    prompt_warnings.extend(extra_warnings)
            result = _base("prompt_preview" if dry_run else "local_rag", self.settings.model)
            result.update(
                source_passages=sources,
                search_terms=retrieved["search_terms"],
                match_mode=retrieved.get("match_mode"),
                source_routing=source_routing,
                warnings=prompt_warnings
                + [
                    "Source identifiers and direct quotations are checked; claim accuracy, empathy and clinical safety still require human review."
                ],
            )
            if source_routing["excluded_candidates"]:
                result["warnings"].append(
                    "Compilations and supplemental commentary were excluded from this chat's author evidence; broad library search still includes them."
                )
            if duplicate_count:
                result["warnings"].append(
                    "Near-identical passages across editions were removed to preserve space for distinct evidence; retained offsets remain exact."
                )
            if career_gate:
                result["warnings"].append(
                    "Career passages passed a conservative word-pattern check, not semantic verification. SOL cannot determine a fixed destiny or choose your career."
                )
            if retrieved.get("focused_route"):
                result["warnings"].append(
                    "A narrow educational concept used a curated original chapter and exact anchored window, not general semantic relevance scoring."
                )
            if support_context:
                result["warnings"].append(
                    "Personal-distress evidence/output screening is limited lexical protection, not clinical assessment or guaranteed safe interpretation."
                )
            if any(source["source_role"] == "user_reference" for source in sources):
                result["warnings"].append(
                    "Supplied local documents are user references, not independently verified original-author teachings."
                )
            if "fallback" in retrieved.get("match_mode", ""):
                result["warnings"].append(
                    "Retrieval used partial keyword matches; the model must not force an unrelated teaching."
                )
            if dry_run:
                result["messages"] = messages
                result["prompt_budget"] = {
                    "conservative_input_token_upper_bound": _message_cost(messages),
                    "context_tokens": self.settings.num_ctx,
                    "reserved_response_tokens": self.settings.num_predict,
                    "reserved_framing_tokens": 256,
                    "reserved_repair_tokens": len(("\n" + REPAIR_INSTRUCTION).encode("utf-8")),
                }
                return result
            status = self._check_model(timeout_seconds=remaining())
            if not status["installed"]:
                raise EngineError(
                    "model_not_installed",
                    f"Local model {self.settings.model} is not installed. Start Ollama and select an already installed local model or create the SOL alias; SOL Chat never downloads a model.",
                    True,
                )
            answer, response = generate(messages)
            result["llm_called"] = True
            result["generation_attempts"] = attempts
            if support_context and _support_answer_issue(answer):
                result.update(
                    mode="support_response_withheld",
                    answer=_supportive_abstention(),
                    error=EngineError(
                        "support_response_withheld",
                        "Recognized blame, diagnosis or guaranteed-outcome language was withheld; no formatting correction was attempted.",
                    ).as_dict(),
                )
                return result
            if _education_claim_issue(answer, source_routing):
                result.update(
                    mode="source_claim_withheld",
                    answer="The generated explanation contradicted a qualification in the supplied definition, so I have withheld it. You can inspect the exact definition below or ask a narrower question.",
                    error=EngineError(
                        "source_claim_withheld",
                        "A recognized denial of the definition's usual chronological development was withheld; no semantic repair was attempted.",
                    ).as_dict(),
                )
                return result
            cited, validation_error = _validate_answer(answer, sources, question, history)
            if validation_error:
                initial_failure = EngineError(
                    "unverified_answer", validation_error[:500], True
                ).as_dict()
                result["initial_failure"] = initial_failure
                try:
                    repair_messages, repair_sources, extra_warnings = _build_messages(
                        question,
                        history,
                        sources,
                        self.settings,
                        trusted_correction=REPAIR_INSTRUCTION,
                    )
                    if career_gate and not all(
                        _career_relevant(source["text"]) for source in repair_sources
                    ):
                        raise EngineError(
                            "repair_evidence_changed",
                            "The repair prompt would lose a career relevance anchor.",
                        )
                    if concentration_gate and not all(
                        _concentration_relevant(source["text"]) for source in repair_sources
                    ):
                        raise EngineError(
                            "repair_evidence_changed",
                            "The repair prompt would lose an educational concentration anchor.",
                        )
                    if education_gate and not all(
                        _education_relevant(source["text"]) for source in repair_sources
                    ):
                        raise EngineError(
                            "repair_evidence_changed",
                            "The repair prompt would lose a definition qualification.",
                        )
                    if support_context and any(
                        _support_source_risk(source) for source in repair_sources
                    ):
                        raise EngineError(
                            "repair_evidence_changed",
                            "The repair evidence failed the personal-distress lexical screen.",
                        )
                except EngineError as exc:
                    result["warnings"].append(
                        "The single correction was skipped because a safe bounded source prompt could not be prepared: "
                        + exc.message
                    )
                else:
                    result["repair_attempted"] = True
                    result["warnings"].extend(extra_warnings)
                    result["warnings"].append(
                        "One bounded formatting/source-identifier correction was used. The rejected text was not replayed; this is not semantic or clinical validation."
                    )
                    answer, response = generate(repair_messages)
                    sources = repair_sources
                    result["source_passages"] = sources
                    result["generation_attempts"] = attempts
                    if support_context and _support_answer_issue(answer):
                        result.update(
                            mode="support_response_withheld",
                            answer=_supportive_abstention(),
                            error=EngineError(
                                "support_response_withheld",
                                "Recognized blame, diagnosis or guaranteed-outcome language in the correction was withheld; no further model call was made.",
                            ).as_dict(),
                        )
                        return result
                    if _education_claim_issue(answer, source_routing):
                        result.update(
                            mode="source_claim_withheld",
                            answer="The corrected explanation still contradicted a qualification in the supplied definition, so it was withheld. Please inspect the exact source passage below.",
                            error=EngineError(
                                "source_claim_withheld",
                                "A recognized definition contradiction was withheld; no further model call was made.",
                            ).as_dict(),
                        )
                        return result
                    cited, validation_error = _validate_answer(answer, sources, question, history)
            if validation_error:
                result.update(
                    mode="citation_check_failed",
                    answer="I couldn't verify the generated response against its supplied source identifiers and quotations, so I have withheld it. You can read the retrieved passages below or ask for a simple paraphrase.",
                    error=EngineError("unverified_answer", validation_error, True).as_dict(),
                )
                return result
            result.update(
                answer=answer.strip(),
                citations=[_citation(source) for source in sources if source["citation"] in cited],
                usage={
                    key: response[key]
                    for key in ("prompt_eval_count", "eval_count", "total_duration")
                    if key in response
                },
            )
            return result
        except (EngineError, RAG.RagError, sqlite3.Error, OSError, KeyError, TypeError) as exc:
            if (
                isinstance(exc, EngineError)
                and exc.code == "truncated_response"
                and result is not None
            ):
                result.update(
                    mode="incomplete_response",
                    llm_called=llm_called,
                    generation_attempts=attempts,
                    initial_failure=initial_failure,
                    answer="The local model reached its response limit, so I have withheld the potentially incomplete answer. Please ask a narrower question; the supplied source passages remain available below.",
                    error=exc.as_dict(),
                )
                return result
            failure = self._failure(exc)
            failure["llm_called"] = llm_called
            failure["generation_attempts"] = attempts
            failure["initial_failure"] = initial_failure
            failure["repair_attempted"] = bool(result and result.get("repair_attempted"))
            return failure


__all__ = [
    "Engine",
    "EngineError",
    "Settings",
    "SYSTEM_PROMPT",
    "urgent_trigger",
    "source_role",
    "treatment_boundary_trigger",
    "persistent_distress_trigger",
    "ungrounded_prediction_trigger",
    "urgent_followup_trigger",
    "diagnosis_boundary_trigger",
]
