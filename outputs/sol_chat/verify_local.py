"""Compatibility entry point. Maintain sol_chat.tools.verify_local, not this wrapper."""
from importlib import import_module
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
_implementation = import_module("sol_chat.tools.verify_local")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

# Preserve module identity so existing integrations and mocks use one code path.
sys.modules[__name__] = _implementation
