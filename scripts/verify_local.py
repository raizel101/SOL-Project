"""Optional invented-prompt model evaluation; --live explicitly enables inference."""

import _bootstrap  # noqa: F401

from sol_chat.tools.verify_local import main

if __name__ == "__main__":
    raise SystemExit(main())
