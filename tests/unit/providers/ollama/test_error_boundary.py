from __future__ import annotations

import ast
from pathlib import Path

from llm_proxy.providers.ollama.errors import UpstreamHttpError


def test_upstream_http_error_preserves_provider_boundary_details() -> None:
    error = UpstreamHttpError(502, "upstream unavailable", "rtx-3090")

    assert error.status_code == 502
    assert error.body == "upstream unavailable"
    assert error.provider == "rtx-3090"
    assert str(error) == "upstream unavailable"


def test_provider_imports_do_not_depend_on_inbound_http_frameworks() -> None:
    provider_root = Path(__file__).parents[4] / "llm_proxy" / "providers"
    for path in provider_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert all(
            (isinstance(node, ast.ImportFrom) and node.module != "fastapi")
            or (isinstance(node, ast.Import) and all(alias.name != "fastapi" for alias in node.names))
            for node in imports
        )
