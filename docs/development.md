# Development and maintenance

Run the commands below from the repository root. Runtime requirements are Python 3.10+, Ollama with the configured local model already installed, and the existing corpus under `outputs/auro_guide_corpus/`. Python application code uses the standard library. The frontend needs no Node runtime, package install or build step to serve; Node is used only for its helper tests.

## Start and stop

On Windows, double-click the root `start_sol.cmd` or `scripts/start_sol.cmd`. Keep its terminal open; stop that instance with Ctrl+C. The portable command is:

```powershell
python scripts/run_sol.py
```

Open [SOL Chat](http://127.0.0.1:8765/chat). [Readiness](http://127.0.0.1:8765/health/ready) should return HTTP 200 with `ready: true`. [API docs](http://127.0.0.1:8765/docs) and [OpenAPI](http://127.0.0.1:8765/api/v1/openapi.json) describe the request contract. Readiness does not certify answer quality.

If port 8765 is already in use, check whether the existing SOL instance is healthy before starting a duplicate. Do not terminate an unrelated process. A separate local port can be requested explicitly:

```powershell
python scripts/run_sol.py --port 8766
```

Starting SOL never downloads models, creates a replacement profile, saves conversations or rebuilds the corpus. Model setup scripts are explicit setup operations, not startup fixes.

## Configuration

Edit `config/local.json`, not Python constants, for local model/index/port settings. Unknown fields, nonfinite numbers, unsafe origins and public bind settings are rejected. Relative index paths resolve against the configuration file's directory rather than the terminal's current directory. The adapter source now lives in `src/sol_chat/adapters/`; the corpus index and provenance metadata remain external under `outputs/auro_guide_corpus/`.

```powershell
python scripts/run_sol.py --config config/local.json
```

The local UI and API share an origin. For a separate local development frontend, allow only its exact loopback origin using `--allow-origin`; never add wildcard CORS or public URLs to bypass the boundary.

Default input bounds are 1,200 Python Unicode characters per new message, complete alternating user/assistant history pairs, at most eight history messages, 1,000 characters each and 2,000 total. Backend configuration may tighten these API bounds, not widen them beyond the client contract. Keep Python and JavaScript character counting compatible, including emoji. A completed reply can be displayed in full even when it cannot fit the next turn's history; the client then requires an explicit reset rather than truncating it.

## Test layers

```powershell
python scripts/test.py
node tests/frontend/test_client_core.mjs
python scripts/verify_backend.py
```

The Python regression suite mocks model inference and provisioning. The pure Node suite checks text validation, history limits, accepted/withheld response classification and safe source URLs. The backend verifier starts and closes a temporary loopback server, reads the actual corpus and installed-model inventory, and forbids generation. It does not certify real-model answers.

`scripts/test.py` prepares the source-checkout import path and needs no corpus/model installation. Direct `python -m unittest discover -s tests -v` needs an editable installation or an explicitly configured `PYTHONPATH`; prefer the script for a fresh checkout. The no-generation backend verifier additionally needs the actual index, coverage metadata and local Ollama inventory.

For optional lint/format tooling, use an isolated virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff check src scripts tests
.venv\Scripts\ruff format --check src scripts tests
```

Ruff is a development-only dependency pinned in `pyproject.toml`. These are source-checkout/editable-package instructions, not a self-contained wheel deployment: frontend, configuration and external data stay at the repository root. A deliberately relocated package must point `SOL_PROJECT_ROOT` at its complete local checkout. `.env.example` documents environment variables only; the application does not automatically load a `.env` file.

New diagnostics belong under `reports/`. Historical model/backend/UI reports remain under `outputs/sol_chat/` and must not be overwritten or reclassified as results for new code. Compare report timestamps and tested-code hashes before relying on a status. The historical 169 Python / 14 HTTP / nine Node counts describe the earlier build; use the new report for current counts.

`scripts/verify_refactor.py` is a migration-specific equivalence check against the preserved integration archive and original extraction adapter. It additionally requires existing Node and Ruff executable paths (`--node` and `--ruff`) and a current backend report. Those historical comparison inputs are intentionally not bundled in a fresh source checkout; this check is not needed to run normal fixture tests. `scripts/package_project.py` builds the source handoff only after current backend and refactor verification reports pass. It excludes the full corpus and model weights; restore those separately to run retrieval on another machine.

Real-model evaluation is a separate operation. `scripts/verify_local.py --help` explains its options; `--live` deliberately calls the installed model on invented scenarios. Use a new report destination and inspect all failures, modes, sources and generation attempts. Ordinary engineering tests and startup must not implicitly run it or provision a model. Qualified human review remains a separate release step.

## Where to make changes

| Change | Primary location and checks |
| --- | --- |
| API request/response or route | `src/sol_chat/api/`; update contract and HTTP tests together. |
| Input or configuration rule | `core/validation.py`, `core/settings.py`, `config.py`; add boundary and invalid-input fixtures. |
| Safety/support routing | `core/safety.py` and `core/policy.py`; add direct, history, negation and busy-worker regressions. |
| Retrieval/source eligibility | `core/relevance.py`, `core/sources.py`, `adapters/local_rag.py`; verify provenance, unchanged excerpts and no-model abstention. Keep the shared binding in `adapters/corpus.py`. |
| Prompt/context budgeting | `core/prompts.py`; preserve source offsets, valid history and shared correction deadlines. |
| Engine coordination | `core/engine.py`; keep fixed routes and dependency errors independent of generation. |
| Frontend behavior | `frontend/src/`; add pure helper tests and browser checks for the changed state transition. |
| Layout/accessibility | `frontend/index.html`, `frontend/styles/main.css`; check desktop/mobile, keyboard use and readable warnings. |
| Diagnostics/setup | `src/sol_chat/tools/`; keep `scripts/` thin and model installation explicit. |

Use small named functions, explanatory error messages and docstrings for public boundaries. Comment why a restriction or unusual operation exists, rather than narrating obvious code. Keep I/O at adapters/entry points where practical. Do not introduce a general abstraction until a real second use requires it.

## Change checklist

- Reproduce the behavior with a small invented input and add a regression before or with the fix.
- Preserve strict request validation, local-only transport, exact source evidence, no hidden history edits and no personal-message logs.
- Update the corresponding contract/config/docs, not just the implementation.
- Run relevant tests, then the full engineering suite and no-generation backend verification.
- For UI changes, reload the browser and check accepted replies, sources, errors, cancellation/reset and responsive layout as applicable. Automated helpers alone do not verify the rendered interface.
- For prompt, retrieval, model, policy or corpus changes, schedule fresh real-model and qualified review before any release decision. Passing software tests does not erase historical model failures.

## Troubleshooting

| Symptom | Safe next check |
| --- | --- |
| `local_bind_failed` | Verify an existing local instance or choose another port. |
| `model_not_installed` / model unready | Check Ollama inventory and configured model name; do not rerun setup if the existing alias is present. |
| Missing/unreadable index | Check configured path and corpus files. Restore the expected data; do not create an empty replacement. |
| UI asset unavailable | Confirm the canonical frontend files exist; use the launcher from the repository, not a standalone copy of one Python file. |
| No suitable evidence | Inspect search/preview results. This is a retrieval/policy outcome, not necessarily a disconnected LLM. |
| Withheld response | Read its structured error and source context. Do not display rejected text or relax validation to make a test pass. |
| Busy generation / timeout | Offer an explicit retry later; do not add unlimited retries or more concurrent workers without capacity checks. |

Maintain a code backup and keep dataset snapshots/provenance reports separately. Corpus reassembly is its own workflow; this application refactor does not scrape, rewrite or transcribe additional source material.
