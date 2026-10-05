# Verification output

New diagnostics go here, outside maintained source. They are ignored by Git.
Reports record tested-code hashes; a source change makes old evidence historical.

The original `outputs/sol_chat/verification_report.json`, its evaluation history,
the pre-refactor integration ZIP and corpus reports remain unchanged. The known
source-faithfulness failure is not cleared by this code-structure refactor.

`python scripts/verify_backend.py` runs mocked engineering checks and read-only
HTTP/index integration. It forbids model inference/downloads. Optional actual
model evaluation is separate: `python scripts/verify_local.py --live`.
