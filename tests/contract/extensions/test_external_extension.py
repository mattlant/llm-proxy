from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from llm_proxy.extensions.kernel import discover_extensions


def test_external_extension_uses_only_public_sdk() -> None:
    source = (Path(__file__).parents[3] / "extensions/mock_capability/src/llm_proxy_mock_capability/extension.py").read_text()
    imports = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) and node.module}
    assert imports == {"llm_proxy.extensions"}


def test_external_extension_installs_and_is_discovered(tmp_path, monkeypatch) -> None:
    package = Path(__file__).parents[3] / "extensions/mock_capability"
    target = tmp_path / "site"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", str(target), str(package)], check=True, capture_output=True, text=True)
    monkeypatch.syspath_prepend(str(target))
    from importlib.metadata import entry_points
    catalog = discover_extensions(entry_points=entry_points().select(group="llm_proxy.extensions"))
    assert "deterministic.mock" in catalog.entries
