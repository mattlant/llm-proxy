from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(*command: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    shutil.rmtree(DIST, ignore_errors=True)
    run("npm", "run", "build:package", cwd=ROOT / "management_frontend")
    run(sys.executable, "-m", "build", "--sdist", "--no-isolation")
    run(sys.executable, "-m", "pytest", "tests/packaging/test_management_frontend_distribution.py", env={**os.environ, "MANAGEMENT_DISTRIBUTION_DIR": str(DIST)})
