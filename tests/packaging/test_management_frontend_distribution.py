from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pytest


def _assets(root: Path) -> set[str]:
    manifest = json.loads((root / "llm_proxy/static/management/asset-manifest.json").read_text())
    return set(manifest["assets"])


def test_sdist_derived_wheel_contains_and_serves_management_assets() -> None:
    repository = Path(__file__).resolve().parents[2]
    distribution_directory = os.environ.get("MANAGEMENT_DISTRIBUTION_DIR")
    if distribution_directory is None:
        pytest.skip("run through management_frontend npm run verify:distribution")
    dist = Path(distribution_directory)
    sdist = next(dist.glob("*.tar.gz"))
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        with tarfile.open(sdist) as archive:
            archive.extractall(temporary_path, filter="data")
        source = next(temporary_path.iterdir())
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation"], cwd=source, check=True)
        wheel = next((source / "dist").glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        for asset in _assets(repository):
            assert f"llm_proxy/static/management/{asset}" in names
        target = temporary_path / "target"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)], check=True)
        probe = """\
import asyncio, json, httpx
from pathlib import Path
from llm_proxy.interfaces.management_frontend import MANAGEMENT_STATIC_DIRECTORY, mount_management_frontend
from fastapi import FastAPI
async def main():
 app=FastAPI(); mount_management_frontend(app)
 async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
  assert (await client.get('/management/')).status_code == 200
  manifest=json.loads((MANAGEMENT_STATIC_DIRECTORY/'asset-manifest.json').read_text())
  asset=next(item for item in manifest['assets'] if item.endswith('.js'))
  assert (await client.get('/management/'+asset)).status_code == 200
asyncio.run(main())
"""
        environment = {**os.environ, "PYTHONPATH": str(target)}
        subprocess.run([sys.executable, "-c", probe], cwd=temporary_path, env=environment, check=True)
