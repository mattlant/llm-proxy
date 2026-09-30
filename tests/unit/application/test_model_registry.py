from __future__ import annotations

import pytest

from llm_proxy.application.errors import (
    InterfaceDisabledError,
    ModelNotExposedError,
    ModelNotFoundError,
)
from llm_proxy.application.model_registry import ModelRegistry
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.models import InterfaceName

from tests.unit.configuration.test_validator import realistic_config


def registry(raw: dict | None = None) -> ModelRegistry:
    return ModelRegistry(ConfigurationLoader().load_mapping(raw or realistic_config()))


def test_canonical_and_case_insensitive_model_resolution():
    value = registry().resolve(InterfaceName.ANTHROPIC, "QWABLE")

    assert value.requested_name == "QWABLE"
    assert value.canonical_name == "qwable"
    assert value.profile.upstream_model.endswith("Qwable-v2-GGUF:Q4_K_M")


def test_interface_scoped_alias_resolves_to_canonical_profile():
    value = registry().resolve(InterfaceName.ANTHROPIC, "LOCAL-OPUS")

    assert value.canonical_name == "qwable"
    assert registry().aliases_for(InterfaceName.ANTHROPIC, "qwable") == ("local-opus",)


def test_openai_and_ollama_visibility_is_profile_scoped():
    service = registry()

    assert service.resolve(InterfaceName.OPENAI, "qwopus").canonical_name == "qwopus"
    assert service.resolve(InterfaceName.OLLAMA, "ornith").canonical_name == "ornith"
    assert [profile.name for profile in service.list_models(InterfaceName.ANTHROPIC)] == ["qwable"]
    assert [profile.name for profile in service.list_models(InterfaceName.OPENAI)] == ["qwopus", "ornith"]


def test_upstream_identifier_does_not_bypass_profile_lookup():
    with pytest.raises(ModelNotFoundError):
        registry().resolve(InterfaceName.ANTHROPIC, "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M")


def test_unknown_and_not_exposed_models_are_distinguished():
    with pytest.raises(ModelNotFoundError):
        registry().resolve(InterfaceName.OPENAI, "does-not-exist")
    with pytest.raises(ModelNotExposedError):
        registry().resolve(InterfaceName.OPENAI, "qwable")


def test_disabled_interface_is_rejected():
    raw = realistic_config()
    raw["interfaces"]["anthropic"]["enabled"] = False

    with pytest.raises(InterfaceDisabledError):
        registry(raw).resolve(InterfaceName.ANTHROPIC, "qwable")


def test_aliases_do_not_duplicate_listed_profiles_and_listing_order_is_stable():
    service = registry()

    assert [profile.name for profile in service.list_models(InterfaceName.OLLAMA)] == ["qwopus", "ornith"]
    assert all(profile.name != "local-opus" for profile in service.list_models(InterfaceName.ANTHROPIC))
