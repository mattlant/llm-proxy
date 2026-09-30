from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_installed_wheel_starts_without_package_adjacent_configuration(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    wheelhouse = tmp_path / "wheelhouse"
    target = tmp_path / "target"
    cwd = tmp_path / "cwd"
    config_path = tmp_path / "test-config.yaml"
    wheelhouse.mkdir()
    target.mkdir()
    cwd.mkdir()
    config_path.write_bytes((repository / "tests/fixtures/test_config.yaml").read_bytes())

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheelhouse), str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheelhouse.glob("llm_proxy-*.whl"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    environment["LLM_PROXY_CONFIG"] = str(config_path)
    result = subprocess.run(
        [sys.executable, "-c", "import llm_proxy.app as app; print(app.configuration_store.snapshot.source_path)"],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(config_path)
    assert not (target / "llm_proxy" / "config.yaml").exists()


def test_wheel_build_ignores_stale_root_build_output(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    sandbox = tmp_path / "source"
    sandbox.mkdir()
    for path in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "build_backend.py"):
        source = repository / path
        if source.exists():
            shutil.copy2(source, sandbox / path)
    shutil.copytree(repository / "llm_proxy", sandbox / "llm_proxy")

    stale_package = sandbox / "build" / "lib" / "ollama_param_proxy"
    stale_package.mkdir(parents=True)
    (stale_package / "__init__.py").write_text("stale package\n")
    stale_package_info = sandbox / "ollama_param_proxy.egg-info"
    stale_package_info.mkdir()
    (stale_package_info / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: ollama-param-proxy\nVersion: 0.1.0\n"
    )

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(sandbox),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(wheelhouse.glob("llm_proxy-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata_text = archive.read(metadata).decode()

    assert not any(name.startswith("ollama_param_proxy/") for name in names)
    assert "Name: llm-proxy\n" in metadata_text
