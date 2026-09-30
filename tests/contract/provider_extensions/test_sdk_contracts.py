from __future__ import annotations

import pytest

from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import DeveloperRoleMode, Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
from llm_proxy.provider_extensions import (
    CompletionExecution,
    ProviderExecutionPolicy,
    SourceInterface,
    ProviderCapabilities,
    ProviderInstanceConfig,
    SdkCompatibility,
    validate_sdk_compatibility,
)


def test_public_execution_has_no_private_resolution_authority() -> None:
    execution = CompletionExecution(
        CompletionRequest("model", (Message(MessageRole.USER, (TextContent("hello"),)),)),
        "upstream", "instance", SamplingParameters(), "model",
    )
    assert execution.upstream_model == "upstream"
    assert not hasattr(execution, "model")
    assert not hasattr(execution, "provider")


def test_public_execution_exposes_only_immutable_provider_mapping_policy() -> None:
    policy = ProviderExecutionPolicy("openai_json_schema", expose_thinking=True, native_tools=True)
    execution = CompletionExecution(
        CompletionRequest("model", (Message(MessageRole.USER, (TextContent("hello"),)),)),
        "upstream", "instance", SamplingParameters(), "model", policy=policy,
    )
    assert execution.policy == policy
    assert execution.policy.structured_output_mode == "openai_json_schema"
    assert execution.policy.developer_role_mode is DeveloperRoleMode.PRESERVE
    with pytest.raises(ValueError, match="structured_output_mode"):
        ProviderExecutionPolicy("private-profile")
    with pytest.raises(TypeError, match="developer_role_mode"):
        ProviderExecutionPolicy(developer_role_mode="system")  # type: ignore[arg-type]


def test_public_execution_includes_stable_source_interface() -> None:
    request = CompletionRequest("model", (Message(MessageRole.USER, (TextContent("hello"),)),))
    execution = CompletionExecution(request, "upstream", "instance", SamplingParameters(), "model", source_interface=SourceInterface.ANTHROPIC)
    assert execution.source_interface is SourceInterface.ANTHROPIC


def test_provider_configuration_is_deeply_immutable_and_json_only() -> None:
    config = ProviderInstanceConfig("instance", "example.provider", {"nested": {"items": [1]}})
    with pytest.raises(TypeError):
        config.config["other"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        config.config["nested"]["items"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        ProviderInstanceConfig("instance", "example.provider", {"bad": object()})


@pytest.mark.parametrize(("declared", "installed", "expected"), [
    (SdkCompatibility("0.1.0", "0.1.999"), "0.1.0", True),
    (SdkCompatibility("0.1.0", "0.1.999"), "0.0.9", False),
    (SdkCompatibility("0.1.0", "0.1.999"), "0.2.0", False),
    (SdkCompatibility("0.1.0"), "0.1.1", True),
    (SdkCompatibility("invalid"), "0.1.0", False),
])
def test_sdk_compatibility_boundaries(declared: SdkCompatibility, installed: str, expected: bool) -> None:
    assert validate_sdk_compatibility(declared, installed).compatible is expected


def test_capabilities_enforce_parallel_tool_invariant() -> None:
    with pytest.raises(ValueError, match="requires native_tools"):
        ProviderCapabilities(parallel_tools=True)
