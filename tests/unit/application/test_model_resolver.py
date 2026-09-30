from __future__ import annotations

import pytest

from llm_proxy.application.model_registry import ModelRegistry
from llm_proxy.application.model_resolver import ModelResolver
from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.domain.messages import DeveloperRoleMode
from llm_proxy.provider_extensions import ProviderCapabilities

from tests.unit.configuration.test_validator import realistic_config


def resolver() -> ModelResolver:
    config = ConfigurationLoader().load_mapping(realistic_config())
    return ModelResolver(ModelRegistry(config), config)


def test_qwable_canonical_anthropic_resolution_is_complete_and_immutable():
    execution = resolver().resolve(InterfaceName.ANTHROPIC, "qwable")

    assert execution.requested_model == "qwable"
    assert execution.canonical_model == "qwable"
    assert execution.upstream_model == "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M"
    assert execution.interface is InterfaceName.ANTHROPIC
    assert execution.provider_instance_name == "rtx-3090"
    assert execution.parameters.temperature == 0.7
    with pytest.raises(AttributeError):
        execution.canonical_model = "other"  # type: ignore[misc]


def test_qwable_alias_resolves_to_same_complete_execution():
    execution = resolver().resolve(InterfaceName.ANTHROPIC, "local-opus")

    assert execution.canonical_model == "qwable"
    assert execution.upstream_model.endswith("Qwable-v2-GGUF:Q4_K_M")


def test_qwopus_openai_resolution_contains_canonical_parameters():
    execution = resolver().resolve(InterfaceName.OPENAI, "qwopus")

    assert execution.provider_instance_name == "rtx-3090"
    assert execution.upstream_model == "hf.co/barozp/Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M"
    assert execution.parameters.temperature == 1.0
    assert execution.parameters.repeat_penalty == 1.12
    assert execution.compatibility.developer_role_mode is DeveloperRoleMode.PRESERVE


def test_ornith_ollama_resolution_contains_compatibility_and_parameters():
    execution = resolver().resolve(InterfaceName.OLLAMA, "ornith")

    assert execution.provider_instance_name == "rtx-3090"
    assert execution.parameters.temperature == 0.6
    assert execution.compatibility.native_tools is True
    assert execution.compatibility.expose_thinking is False


def test_provider_qualified_passthrough_routes_without_listing_and_canonicalizes_provider_key():
    raw = realistic_config()
    raw["model_access"] = {"mode": "provider_passthrough", "listing_timeout_seconds": 1}
    config = ConfigurationLoader().load_mapping(raw)
    gateways = ProviderGatewayRegistry({"rtx-3090": object()}, config.providers, {"rtx-3090": ProviderCapabilities(completion=True, native_tools=False)})

    execution = ModelResolver(ModelRegistry(config), config, gateways).resolve(InterfaceName.OPENAI, "RTX-3090::opaque/Model:Q4")

    assert execution.canonical_model == "rtx-3090::opaque/Model:Q4"
    assert execution.upstream_model == "opaque/Model:Q4"
    assert not execution.compatibility.native_tools
