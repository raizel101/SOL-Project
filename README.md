# SOL Chat

A local, source-grounded reflection application using Sri Aurobindo and The
Mother's texts. Python backend, plain JavaScript frontend, existing Ollama model.
No model download or corpus rebuild is part of normal startup.

## Run

On this laptop, double-click **`start_sol.cmd` in this project folder**. Open
[SOL Chat](http://127.0.0.1:8765/chat). Run only one instance on port 8765.

With Python 3.10+ available:

```powershell
python scripts/run_sol.py
```

The installed `sol-chat` model must be available in Ollama, and the existing
`outputs/auro_guide_corpus/local_rag.sqlite` must be present. Configuration lives
in `config/local.json`; paths resolve beside that file, not your terminal folder.
Runtime code has no third-party Python dependencies and needs no Node build.

## Project layout

```text
src/sol_chat/       Backend package: API, engine modules, adapters and tools
frontend/          HTML, readable CSS and JavaScript; no generated bundle
config/            Local configuration and unchanged Ollama model profile
tests/             Backend regressions and pure frontend helper tests
scripts/           Thin start, test and verification entry points
docs/              Architecture, development, deployment and evaluation guides
data/              How to locate the external dataset; no duplicate corpus
reports/           New generated verification results, excluded from Git
outputs/           Existing corpus, historical reports/archives and legacy shims
work/              Historical extraction/debug artifacts, excluded from Git
```

`src/sol_chat/` and `frontend/` are the **single source of truth**. Old Python
entry points under `outputs/sol_chat/` delegate to the package so existing start
commands keep working. Do not add application features to those wrappers or edit
the old frontend delivery; the backend now serves the canonical `frontend/`.

## Develop and verify

```powershell
python scripts/test.py
node tests/frontend/test_client_core.mjs
python scripts/verify_backend.py
```

The first two use fixtures/mocked inference. The last also checks the real
read-only index and installed-model inventory, but forbids model generation.
New reports go in `reports/`; original evaluation evidence is preserved.

The structure-refactor verification passed 180 Python regressions, 14 real-index
HTTP checks, nine frontend helper tests, and lint/format checks. It also compared
32 core definitions and 17 adapter definitions against their preserved originals
without changing their behavior. See `reports/refactor_verification_report.json`;
this is engineering evidence, not a new real-model quality approval.

Optional developer tools, in an isolated virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff check src scripts tests
.venv\Scripts\ruff format --check src scripts tests
```

Ruff is a development-only dependency, pinned in `pyproject.toml`. Public
interfaces have typed boundaries/docstrings; module ownership and change/testing
workflow are described in [architecture](docs/architecture.md) and
[development](docs/development.md). CI runs fixture tests and formatting checks
without the private corpus or a model installation.

## Data, privacy and release status

The large dataset, model weights and original corpus archive were not moved or
duplicated. The small retrieval/transport adapter is maintained under
`src/sol_chat/adapters/`; the original extraction artifact remains unchanged.
Do not commit source documents, indexes, secrets or conversations to Git.

Chats remain in browser memory only; source text is displayed as plain text.
The app remains loopback-only. This organization refactor does not add public
authentication, hosted infrastructure, clinical assessment or student approval.
The preserved real-model evaluation contains a source-faithfulness failure;
passing engineering tests does not clear it. See
[evaluation](docs/safety-and-evaluation.md) and [deployment](docs/deployment.md).

Your intended public Vercel frontend/server-hosted model deployment is a separate
step requiring a chosen backend host and reviewed security/privacy/safeguarding
arrangements, especially for under-18 users. No service was published by this
refactor.
