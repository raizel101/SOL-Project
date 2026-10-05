# Architecture

SOL Chat is a retrieval-augmented generation (RAG) application. It searches an existing text corpus and passes bounded excerpts to an installed Ollama model. This repository does not contain newly trained model weights.

## Responsibilities

| Location | Responsibility |
| --- | --- |
| `src/sol_chat/api/` | HTTP requests, responses, fixed UI assets and the OpenAPI contract. |
| `src/sol_chat/core/` | Conversation orchestration, validated settings, bounded context, support routing and answer checks. |
| `src/sol_chat/adapters/` | Maintained read-only retrieval/local-model transport adapter; not application policy. |
| `src/sol_chat/config.py` | Strict JSON configuration and safe local-origin validation. |
| `src/sol_chat/paths.py` | Repository locations shared by entry points. |
| `frontend/` | Dependency-free browser interface, pure client helpers and styles. |
| `config/local.json` | Local runtime configuration; no credentials. |
| `scripts/` | Launching, diagnostics and explicit model setup entry points. |
| `src/sol_chat/tools/` | Diagnostic/setup implementations called by thin script entry points. |
| `tests/` | Engineering regression tests; inference/setup is mocked unless explicitly documented otherwise. |
| `docs/` | Architecture, development workflow, deployment boundaries and evaluation guidance. |
| `reports/` | New verification artifacts for the canonical source. |
| `outputs/auro_guide_corpus/` | Existing extracted texts, provenance reports and read-only retrieval database. Not application source. |
| `outputs/sol_chat/` | Compatibility entry points and preserved historical verification artifacts. |

Make ordinary changes in the canonical source directories, not in compatibility wrappers. Extraction scripts and old work artifacts are not part of the request-serving path.

The core separates `errors`, `settings`, `validation`, `safety`, `relevance`, `sources`, `prompts`, `policy` and `responses` from the `engine` that coordinates them. The API separates `server`, `contract` and `web_assets`. The small adapter implementation is maintained at `adapters/local_rag.py`; `adapters/corpus.py` exposes its shared `RAG` module binding. The original extraction copy under `outputs/auro_guide_corpus/local_rag.py` remains unchanged but is no longer a source-code dependency. A source-only checkout can import the application and run fixture tests without generated corpus files; actual retrieval still needs the external SQLite index.

## A chat turn

```text
Browser: validate message and completed history pairs
    → API: validate origin, exact route and bounded JSON body
    → Engine
        ├─ Recognized support/boundary → fixed reply (no LLM)
        └─ Read-only retrieval
            ├─ No suitable evidence → abstention (no LLM)
            └─ Eligible excerpts + bounded conversation → local Ollama
                → Answer checks → accepted answer or withheld fallback
    → Browser: plain-text response, warnings and inspectable source passages
```

Fixed support and abstention routes may return without calling the LLM. The response fields `mode`, `llm_called`, `error` and `ok` distinguish these states. A fixed fallback is not a generated, source-backed teaching.

## API contract

The primary operations remain `POST /api/v1/chat` and `POST /api/v1/search`. `/api/chat` and `/api/search` are compatibility aliases. `/chat` serves the interface; `/docs` is passive documentation; `/api/v1/openapi.json` serves the machine-readable contract. `GET /health/live` checks the HTTP process, while `GET /health/ready` checks the configured index and installed model.

Readiness establishes dependency availability, not answer quality. Ordinary generation shares one semaphore slot; recognized fixed support routes remain available while it is busy. Generation is non-streaming.

## Data and model boundaries

The frozen corpus remains under `outputs/auro_guide_corpus/`; the engine opens its SQLite index read-only. Search uses SQLite FTS5/BM25 keyword ranking, not semantic embeddings. Narrow curated education routes are exceptions to the general search path, not universal relevance checks. Text provenance, exact excerpt offsets and corpus coverage gaps must remain intact.

The corpus adapter also supplies the local Ollama transport. Configuration accepts loopback endpoints and installed local models; ordinary startup never pulls a model or rebuilds the corpus. No remote/cloud fallback is added by this structure.

The application has no accounts or saved conversations. Browser history stays in memory and is supplied by the client on each turn; it is not trusted teaching evidence. Clearing or reloading the page loses that conversation. Canceling a browser request does not stop model work already running on the server.

## Preserve these invariants

- Reference documents and retrieved text are untrusted data, never executable instructions.
- Render answers, metadata and source excerpts as text; validate source links before making them clickable.
- Preserve complete accepted history pairs. Never silently shorten a message, rewrite history or promote a withheld reply into accepted history.
- Treat historical spiritual language as philosophy, not medical evidence or a personalized diagnosis.
- Retain loopback binding, explicit origins, bounded requests, no arbitrary file serving and no request-body logging.
- Keep engineering verification separate from actual-model and human evaluation. A syntactically valid citation is not proof that its associated claim is faithful.

See [development](development.md), [deployment](deployment.md) and [safety and evaluation](safety-and-evaluation.md) before changing these boundaries.
