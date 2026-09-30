from __future__ import annotations

import pytest

from llm_proxy.application.errors import ModelNotExposedError
from llm_proxy.application.model_registry import ModelRegistry
from llm_proxy.application.model_resolver import ModelResolver
from llm_proxy.application.policy_context import RequestPolicyContext
from llm_proxy.application.policy_engine import PolicyEngine
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.validator import ConfigurationValidator
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import Message, MessageRole

from tests.unit.configuration.test_validator import realistic_config


def build_services():
    raw = realistic_config()
    raw["models"]["qwopus"]["interfaces"].append("anthropic")
    raw["policies"] = [
        {
            "name": "anthropic-fallback",
            "enabled": True,
            "match": {"models": ["local-opus"]},
            "actions": {"model": "qwopus", "parameters": {"temperature": 0.8}},
        }
    ]
    config = ConfigurationLoader().load_mapping(raw)
    ConfigurationValidator().validate(config)
    registry = ModelRegistry(config)
    resolver = ModelResolver(registry, config)
    return config, registry, resolver, PolicyEngine(config, registry)


def test_full_configuration_resolution_and_policy_proof():
    config, registry, resolver, policy_engine = build_services()

    qwable = resolver.resolve(InterfaceName.ANTHROPIC, "qwable")
    alias = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")
    assert qwable.canonical_model == "qwable"
    assert alias.canonical_model == "qwable"
    assert qwable.provider_instance_name == "rtx-3090"
    assert qwable.upstream_model.endswith("Qwable-v2-GGUF:Q4_K_M")

    with pytest.raises(ModelNotExposedError):
        resolver.resolve(InterfaceName.OPENAI, "qwable")

    qwopus = resolver.resolve(InterfaceName.OPENAI, "qwopus")
    ornith = resolver.resolve(InterfaceName.OLLAMA, "ornith")
    assert qwopus.provider_instance_name == "rtx-3090"
    assert qwopus.parameters.temperature == 1.0
    assert qwopus.parameters.repeat_penalty == 1.12
    assert ornith.provider_instance_name == "rtx-3090"
    assert ornith.upstream_model.endswith("Ornith-1.0-35B-GGUF:Q4_K_M")

    context = RequestPolicyContext(
        InterfaceName.ANTHROPIC,
        "local-opus",
        "qwable",
        (Message(MessageRole.USER, (TextContent("use the fallback"),)),),
        (),
        {},
    )
    final = policy_engine.apply(context, alias)
    assert final.requested_model == "local-opus"
    assert final.canonical_model == "qwopus"
    assert final.upstream_model.endswith("Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M")
    assert final.parameters.temperature == 0.8
    assert final.parameters.repeat_penalty == 1.12
    with pytest.raises(AttributeError):
        final.provider_instance_name = qwable.provider_instance_name  # type: ignore[misc]


def test_configuration_loaded_additive_matchers_apply_policy():
    raw = realistic_config()
    raw["policies"] = [{
        "name": "family-tool",
        "enabled": True,
        "match": {
            "model_contains_all": ["QW", "able"],
            "model_wildcard_any": ["*able"],
            "tool_wildcard_all": ["copilot/*", "*/github/read"],
        },
        "actions": {"parameters": {"temperature": 0.2}},
    }]
    config = ConfigurationLoader().load_mapping(raw)
    registry = ModelRegistry(config)
    resolver = ModelResolver(registry, config)
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")
    context = RequestPolicyContext(InterfaceName.ANTHROPIC, "local-opus", "qwable", (), ("copilot/github/read",), {})

    assert PolicyEngine(config, registry).apply(context, base).parameters.temperature == 0.2
