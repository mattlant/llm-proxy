from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_PROJECTS = (
    "pyproject.toml",
    "extensions/mock_provider/pyproject.toml",
    "extensions/mock_capability/pyproject.toml",
    "extensions/provider_compliance/pyproject.toml",
    "tests/fixtures/pyproject.toml",
)


def _project_metadata(relative_path: str) -> dict[str, object]:
    with (REPOSITORY / relative_path).open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def test_requirements_txt_mirrors_root_runtime_dependencies() -> None:
    project = _project_metadata("pyproject.toml")
    declared = {str(Requirement(dependency)) for dependency in project["dependencies"]}
    requirements = {
        str(Requirement(line.strip()))
        for line in (REPOSITORY / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert requirements == declared


@pytest.mark.parametrize("relative_path", PACKAGE_PROJECTS)
def test_independently_packaged_project_requires_python_3123_or_newer(relative_path: str) -> None:
    minimum = Version("3.12.3")
    below_minimum = Version("3.12.2")
    project = _project_metadata(relative_path)
    requires_python = project.get("requires-python")
    assert requires_python is not None, f"{relative_path} does not declare requires-python"
    supported_versions = SpecifierSet(str(requires_python))
    assert minimum in supported_versions, f"{relative_path} excludes Python 3.12.3"
    assert below_minimum not in supported_versions, f"{relative_path} supports Python below 3.12.3"