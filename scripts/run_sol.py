"""Start the local SOL application. No downloads or corpus rebuilds."""

import _bootstrap  # noqa: F401 - prepares the source checkout import path

from sol_chat.api.server import main

if __name__ == "__main__":
    raise SystemExit(main())
