from importlib.metadata import version
from pathlib import Path
import tomllib


def _package_version() -> str:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as file:
            return tomllib.load(file)["project"]["version"]
    return version("llm-proxy")


__version__ = _package_version()
