"""Explicit model setup command. Ordinary launches never use this script."""

import _bootstrap  # noqa: F401

from sol_chat.tools.provision_model import main

if __name__ == "__main__":
    raise SystemExit(main())
