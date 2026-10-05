"""Run source-checkout regression tests; no index, model or network needed."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node", type=Path, help="Existing Node executable to also run frontend tests"
    )
    args = parser.parse_args()
    root = _bootstrap.PROJECT_ROOT
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=root,
        env=environment,
        check=False,
    )
    if result.returncode or args.node is None:
        return result.returncode
    return subprocess.run(
        [str(args.node.resolve()), "tests/frontend/test_client_core.mjs"],
        cwd=root,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
