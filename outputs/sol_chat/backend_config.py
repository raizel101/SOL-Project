"""Compatibility entry point. Maintain sol_chat.config, not this wrapper."""
from importlib import import_module
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
_implementation = import_module("sol_chat.config")

if __name__ == "__main__":
    raise SystemExit("This compatibility module is imported by SOL; use scripts/run_sol.py.")

# Preserve module identity so existing integrations and mocks use one code path.
sys.modules[__name__] = _implementation
