from __future__ import annotations

import pytest

from llm_proxy.application.policy_context import RequestPolicyContext
from llm_proxy.configuration.models import (
    InterfaceName,
    PolicyActions,
    PolicyMatch,
    SamplingParametersOverride,
)
from llm_proxy.application.model_registry import ModelRegistry
from llm_proxy.application.model_resolver import ModelResolver
from llm_proxy.application.policy_engine import PolicyEngine
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.validator import ConfigurationValidator
from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent, ToolResultContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.messages import DeveloperRoleMode

from tests.unit.configuration.test_validator import realistic_config


def test_policy_match_and_partial_actions_are_canonical_and_immutable():
    match = PolicyMatch(
        interfaces=frozenset({InterfaceName.ANTHROPIC}),
        models=frozenset({"qwable"}),
        text_contains_any=("urgent",),
        tools_all=("read_file",),
        metadata_equals={"tenant": "research"},
    )
    actions = PolicyActions(parameters=SamplingParametersOverride(temperature=0.0))

    assert match.interfaces == frozenset({InterfaceName.ANTHROPIC})
    assert match.metadata_equals == {"tenant": "research"}
    assert actions.parameters.temperature == 0.0
    with pytest.raises(AttributeError):
        actions.provider = "other"  # type: ignore[misc]


def test_policy_context_contains_canonical_fields_and_defensively_copies_metadata():
    metadata = {"tenant": {"name": "research"}}
    context = RequestPolicyContext(
        InterfaceName.ANTHROPIC,
        "local-opus",
        "qwable",
        (Message(MessageRole.USER, (TextContent("hello"),)),),
        ("read_file",),
        metadata,
    )
    metadata["tenant"]["name"] = "changed"

    assert context.interface is InterfaceName.ANTHROPIC
    assert context.requested_model == "local-opus"
    assert context.canonical_model == "qwable"
    assert context.tool_names == ("read_file",)
    assert context.metadata["tenant"] == {"name": "research"}


def test_policy_visible_text_context_can_keep_reasoning_distinct():
    context = RequestPolicyContext(
        InterfaceName.OPENAI,
        "qwopus",
        "qwopus",
        (
            Message(
                MessageRole.ASSISTANT,
                (ReasoningContent("private"), TextContent("visible")),
            ),
        ),
        (),
        {},
    )

    assert isinstance(context.messages[0].content[0], ReasoningContent)
    assert isinstance(context.messages[0].content[1], TextContent)


def test_empty_policy_actions_are_rejected():
    with pytest.raises(ValueError):
        PolicyActions()


def build_engine(policy_values: list[dict], mutate=None):
    raw = realistic_config()
    raw["policies"] = policy_values
    if mutate is not None:
        mutate(raw)
    config = ConfigurationLoader().load_mapping(raw)
    ConfigurationValidator().validate(config)
    registry = ModelRegistry(config)
    resolver = ModelResolver(registry, config)
    return PolicyEngine(config, registry), resolver


def context(requested="local-opus", canonical="qwable", messages=(), tools=(), metadata=None):
    return RequestPolicyContext(
        InterfaceName.ANTHROPIC,
        requested,
        canonical,
        tuple(messages),
        tuple(tools),
        metadata or {},
    )


def policy(name, match=None, actions=None, enabled=True):
    return {
        "name": name,
        "enabled": enabled,
        "match": match or {},
        "actions": actions or {"parameters": {"temperature": 0.8}},
    }


def test_no_policies_returns_base_unchanged_and_disabled_policy_is_ignored():
    engine, resolver = build_engine([],)
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")
    assert engine.apply(context(), base) is base


def test_policy_diagnostics_retain_order_and_only_changed_field_names():
    engine, resolver = build_engine([
        policy("model", actions={"model": "qwopus"}),
        policy("parameters", actions={"parameters": {"temperature": 0.2, "extra": {"private": "value"}}}),
    ], mutate=lambda raw: raw["models"]["qwopus"]["interfaces"].append("anthropic"))
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")
    result = engine.apply_with_diagnostics(context(), base)

    assert [entry.name for entry in result.applied_policies] == ["model", "parameters"]
    assert result.applied_policies[0].overridden_fields[0] == "model"
    assert result.applied_policies[1].overridden_fields == ("temperature", "parameters.extra(count=1)")

    engine, resolver = build_engine([policy("disabled", enabled=False)])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")
    assert engine.apply(context(), base) is base


def test_interface_canonical_model_and_alias_matching():
    engine, resolver = build_engine([
        policy("match", {"interfaces": ["anthropic"], "models": ["local-opus"]}, {"parameters": {"temperature": 0.4}})
    ])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    assert engine.apply(context(), base).parameters.temperature == 0.4


def test_visible_text_any_all_matching_excludes_reasoning_tool_arguments_and_results():
    messages = (
        Message(MessageRole.SYSTEM, (TextContent("visible system"),)),
        Message(MessageRole.USER, (TextContent("visible user"),)),
        Message(MessageRole.ASSISTANT, (
            ReasoningContent("secret reasoning"),
            TextContent("visible assistant"),
            ToolCallContent("call", "read_file", {"path": "secret argument"}),
        )),
        Message(MessageRole.TOOL, (ToolResultContent("call", (TextContent("secret result"),)),)),
    )
    engine, resolver = build_engine([
        policy("visible", {"text_contains_all": ["visible system", "visible user", "visible assistant"]}, {"parameters": {"temperature": 0.3}}),
    ])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")
    assert engine.apply(context(messages=messages), base).parameters.temperature == 0.3

    engine, resolver = build_engine([
        policy("hidden", {"text_contains_any": ["secret reasoning"]}, {"parameters": {"temperature": 0.3}}),
    ])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")
    assert engine.apply(context(messages=messages), base).parameters.temperature == 0.7


def test_tool_name_and_metadata_matching():
    engine, resolver = build_engine([
        policy("tools", {"tools_any": ["read_file"], "tools_all": ["read_file"], "metadata_equals": {"tenant": "research"}}, {"parameters": {"temperature": 0.2}}),
    ])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    assert engine.apply(context(tools=("read_file",), metadata={"tenant": "research"}), base).parameters.temperature == 0.2


def test_additive_model_and_tool_matchers_use_aliases_quantifiers_and_wildcards():
    engine, resolver = build_engine([
        policy(
            "match",
            {
                "model_contains_all": ["OPUS", "local"],
                "model_wildcard_all": ["local-*", "*opus"],
                "tool_wildcard_all": ["copilot/*", "*/github/read"],
            },
            {"parameters": {"temperature": 0.2}},
        ),
    ])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    assert engine.apply(context(tools=("copilot/github/read",)), base).parameters.temperature == 0.2
    assert engine.apply(context(tools=("Copilot/github/read",)), base).parameters.temperature == 0.7


def test_model_contains_all_is_order_independent_across_requested_and_canonical_names():
    engine, resolver = build_engine([
        policy("match", {"model_contains_all": ["local", "able"]}, {"parameters": {"temperature": 0.2}}),
    ])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    assert engine.apply(context(), base).parameters.temperature == 0.2


def test_later_policy_uses_original_context_after_model_action():
    def expose_qwopus(raw):
        raw["models"]["qwopus"]["interfaces"].append("anthropic")

    engine, resolver = build_engine([
        policy("replace", {"models": ["local-opus"]}, {"model": "qwopus"}),
        policy("original", {"model_contains_any": ["local"]}, {"parameters": {"temperature": 0.2}}),
    ], mutate=expose_qwopus)
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    assert engine.apply(context(), base).parameters.temperature == 0.2


def test_later_matching_policy_wins_while_partial_override_preserves_other_parameters():
    engine, resolver = build_engine([
        policy("first", actions={"parameters": {"temperature": 0.4, "top_p": 0.8}}),
        policy("second", actions={"parameters": {"temperature": 0.2}}),
    ])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    result = engine.apply(context(), base)
    assert result.parameters.temperature == 0.2
    assert result.parameters.top_p == 0.8


def test_model_override_re_resolves_profile_and_provider_override_changes_only_provider():
    def add_alternate(raw):
        raw["models"]["qwopus"]["interfaces"].append("anthropic")
        raw["providers"]["mirror"] = {
            "extension": "ollama",
            "config": {"base_url": "http://mirror.test:11434", "outbound_interface": "openai"},
            "enabled": True,
        }

    engine, resolver = build_engine([
        policy("model", actions={"model": "qwopus"}),
        policy("provider", actions={"provider": "mirror"}),
    ], mutate=add_alternate)
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    result = engine.apply(context(), base)
    assert result.canonical_model == "qwopus"
    assert result.upstream_model.endswith("Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M")
    assert result.parameters.temperature == 1.0
    assert result.provider_instance_name == "mirror"
    assert result.compatibility.developer_role_mode is DeveloperRoleMode.PRESERVE


def test_model_override_must_be_exposed_through_active_interface():
    engine, resolver = build_engine([policy("model", actions={"model": "qwopus"})])
    base = resolver.resolve(InterfaceName.ANTHROPIC, "local-opus")

    with pytest.raises(Exception):
        engine.apply(context(), base)
