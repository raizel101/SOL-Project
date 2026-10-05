"""Run engineering/integration verification without model generation."""

import _bootstrap  # noqa: F401

from sol_chat.tools.verify_backend import main

if __name__ == "__main__":
    raise SystemExit(main())
