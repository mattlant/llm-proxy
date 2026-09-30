#!/usr/bin/env python3
"""Offline installed-wheel smoke proof for the compliance distribution.

Requires a preseeded wheelhouse containing runtime/test dependencies.  The
child runs from a temporary directory, never from this repository checkout.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


CHILD = """
import asyncio
from importlib.metadata import entry_points

from llm_proxy_provider_compliance import observe_stream
from llm_proxy.provider_extensions import FinishReason, ResponseCompleted, ResponseStarted, TextCompleted, TextDelta, TextStarted, Usage

points = entry_points(group='llm_proxy.providers')
mock = next(point.load() for point in points if point.name == 'deterministic_mock')
assert mock.metadata.extension_id == 'deterministic-mock'

async def stream():
    yield ResponseStarted('installed', 'model')
    yield TextStarted('text')
    yield TextDelta('text', 'installed')
    yield TextCompleted('text')
    yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 1))

assert len(asyncio.run(observe_stream('installed-mock', stream()))) == 5
print('installed compliance and mock entry-point proof passed')
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if not args.wheelhouse.is_dir():
        raise SystemExit("wheelhouse does not exist")
    with tempfile.TemporaryDirectory(prefix="mat28-wheels-") as wheels, tempfile.TemporaryDirectory(prefix="mat28-venv-") as venv:
        for project in (root, root / "extensions/mock_provider", root / "extensions/provider_compliance"):
            subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", wheels], cwd=project, check=True)
        python = Path(venv) / "bin" / "python"
        subprocess.run([sys.executable, "-m", "venv", venv], check=True)
        subprocess.run([python, "-m", "pip", "install", "--no-index", "--find-links", str(args.wheelhouse), "--find-links", wheels, "llm-proxy==0.4.0", "llm-proxy-mock-provider==0.1.0", "llm-proxy-provider-compliance==0.1.0"], check=True)
        subprocess.run([python, "-c", CHILD], cwd="/tmp", check=True)


if __name__ == "__main__":
    main()
