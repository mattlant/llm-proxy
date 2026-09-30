from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_proxy.application.errors import ModelNotExposedError
from llm_proxy.application.model_registry import ModelRegistry
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.models import InterfaceName


def registry(config_file: Path) -> ModelRegistry:
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["models"]["anthropic-only-model"] = {
        "upstream_model": "anthropic-upstream-model",
        "provider": "test-provider",
        "interfaces": ["anthropic"],
        "aliases": {"anthropic": ["anthropic-alias-one", "anthropic-alias-two"]},
        "parameters": {},
    }
    config_file.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = ConfigurationLoader().load_path(config_file)
    return ModelRegistry(config)


@pytest.mark.parametrize("alias", ["anthropic-alias-one", "anthropic-alias-two"])
def test_anthropic_aliases_resolve_without_exposing_other_interfaces(alias: str, config_file: Path) -> None:
    models = registry(config_file)
    resolved = models.resolve(InterfaceName.ANTHROPIC, alias)

    assert resolved.canonical_name == "anthropic-only-model"
    assert resolved.profile.upstream_model == "anthropic-upstream-model"
    with pytest.raises(ModelNotExposedError):
        models.resolve(InterfaceName.OPENAI, "anthropic-only-model")
    with pytest.raises(ModelNotExposedError):
        models.resolve(InterfaceName.OLLAMA, "anthropic-only-model")
