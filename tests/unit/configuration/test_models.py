from __future__ import annotations

from pathlib import Path

import pytest

from llm_proxy.configuration.models import (
    AuthenticationConfig,
    FileLoggingConfig,
    InterfaceConfig,
    InterfaceName,
    ProviderConfig,
    ServerConfig,
    ModelAliases,
    ModelCompatibility,
    ModelProfile,
)
from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.domain.messages import DeveloperRoleMode


def test_valid_server_configuration_is_frozen():
    server = ServerConfig("127.0.0.1", 11435, 0, 1.0, "INFO")

    assert server.host == "127.0.0.1"
    assert server.port == 11435
    with pytest.raises(AttributeError):
        server.port = 8080  # type: ignore[misc]


def test_file_logging_configuration_defaults_and_validation():
    assert FileLoggingConfig() == FileLoggingConfig(False, "./logs")
    assert FileLoggingConfig(True, "/var/tmp/llm-proxy").enabled is True
    with pytest.raises(TypeError, match="file_logging enabled"):
        FileLoggingConfig("true", "./logs")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="file_logging directory"):
        FileLoggingConfig(directory="  ")
    with pytest.raises(ValueError, match="file_logging directory"):
        FileLoggingConfig(directory=Path("./logs"))  # type: ignore[arg-type]


def test_server_file_logging_is_optional_and_immutable():
    server = ServerConfig("127.0.0.1", 11435, 0, 1.0, "INFO")
    assert server.file_logging == FileLoggingConfig()
    with pytest.raises(TypeError, match="file_logging must"):
        ServerConfig("127.0.0.1", 11435, 0, 1.0, "INFO", file_logging=False)  # type: ignore[arg-type]


def test_server_configuration_content_trace_policy_defaults_and_validates() -> None:
    assert ServerConfig("127.0.0.1", 11435, 0, 1.0, "INFO").payload_trace_include_content is True
    assert ServerConfig("127.0.0.1", 11435, 0, 1.0, "INFO", payload_trace_include_content=False).payload_trace_include_content is False
    with pytest.raises(TypeError, match="payload_trace_include_content"):
        ServerConfig("127.0.0.1", 11435, 0, 1.0, "INFO", payload_trace_include_content="false")  # type: ignore[arg-type]


def test_valid_interfaces_and_anthropic_authentication():
    interfaces = (
        InterfaceConfig(InterfaceName.OPENAI, True),
        InterfaceConfig(InterfaceName.OLLAMA, True),
        InterfaceConfig(
            InterfaceName.ANTHROPIC,
            True,
            AuthenticationConfig(allow_any_token=True, token=None),
            expose_thinking=True,
        ),
    )

    assert [interface.name for interface in interfaces] == [
        InterfaceName.OPENAI,
        InterfaceName.OLLAMA,
        InterfaceName.ANTHROPIC,
    ]
    assert interfaces[2].authentication == AuthenticationConfig(True, None)
    assert interfaces[2].expose_thinking is True


def test_provider_configuration_is_immutable_and_extension_owned():
    provider = ProviderConfig(
        "rtx-3090",
        "ollama",
        True,
        {"base_url": "http://provider.test:11434///", "outbound_interface": "openai"},
    )

    assert provider.extension_id == "ollama"
    assert provider.config["base_url"] == "http://provider.test:11434///"
    with pytest.raises(TypeError):
        provider.config["base_url"] = "http://other.test"  # type: ignore[index]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ServerConfig("", 11435, 0, 1, "INFO"),
        lambda: ServerConfig("127.0.0.1", 0, 0, 1, "INFO"),
        lambda: ServerConfig("127.0.0.1", 65536, 0, 1, "INFO"),
        lambda: ServerConfig("127.0.0.1", 11435, -1, 1, "INFO"),
        lambda: ServerConfig("127.0.0.1", 11435, 0, 0, "INFO"),
    ],
)
def test_invalid_server_values_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_invalid_anthropic_authentication_is_rejected():
    with pytest.raises(ValueError):
        AuthenticationConfig(allow_any_token=False, token=None)
    with pytest.raises(ValueError):
        AuthenticationConfig(allow_any_token=False, token="  ")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ProviderConfig("", "ollama", True, {}),
        lambda: ProviderConfig("local", "", True, {}),
        lambda: ProviderConfig("local", "ollama", "true", {}),  # type: ignore[arg-type]
        lambda: ProviderConfig("local", "ollama", True, {"bad": object()}),
    ],
)
def test_invalid_provider_values_are_rejected(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_provider_configuration_requires_mapping_with_string_keys():
    with pytest.raises(TypeError):
        ProviderConfig("local", "vllm", True, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ProviderConfig("local", "ollama", True, {1: "bad"})  # type: ignore[arg-type]


def test_model_profile_retains_exposure_aliases_parameters_and_compatibility():
    parameters = SamplingParameters(
        temperature=0.7,
        top_p=0.95,
        extra={"nested": {"enabled": True}},
    )
    profile = ModelProfile(
        name="qwable",
        upstream_model="hf.co/example/Qwable:Q4_K_M",
        provider="rtx-3090",
        interfaces=frozenset({InterfaceName.ANTHROPIC}),
        aliases=ModelAliases({InterfaceName.ANTHROPIC: ("local-opus", "claude-opus-local")}),
        parameters=parameters,
        compatibility=ModelCompatibility(expose_thinking=False, native_tools=True),
    )

    assert profile.interfaces == frozenset({InterfaceName.ANTHROPIC})
    assert profile.aliases.by_interface[InterfaceName.ANTHROPIC] == ("local-opus", "claude-opus-local")
    assert profile.parameters is parameters
    assert profile.parameters.extra["nested"] == {"enabled": True}
    assert profile.compatibility == ModelCompatibility(False, True)


def test_qwopus_can_be_exposed_through_openai_and_ollama():
    profile = ModelProfile(
        "qwopus",
        "hf.co/example/Qwopus:Q4_K_M",
        "rtx-3090",
        frozenset({InterfaceName.OPENAI, InterfaceName.OLLAMA}),
    )

    assert profile.interfaces == frozenset({InterfaceName.OPENAI, InterfaceName.OLLAMA})


def test_model_compatibility_defaults_and_validates_developer_role_mode():
    assert ModelCompatibility().developer_role_mode is DeveloperRoleMode.PRESERVE
    assert ModelCompatibility(developer_role_mode=DeveloperRoleMode.REJECT).developer_role_mode is DeveloperRoleMode.REJECT
    with pytest.raises(TypeError, match="developer_role_mode"):
        ModelCompatibility(developer_role_mode="system")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ModelProfile("", "upstream", "provider", frozenset({InterfaceName.OPENAI})),
        lambda: ModelProfile("profile", "", "provider", frozenset({InterfaceName.OPENAI})),
        lambda: ModelProfile("profile", "upstream", "", frozenset({InterfaceName.OPENAI})),
        lambda: ModelProfile("profile", "upstream", "provider", frozenset()),
        lambda: ModelAliases({InterfaceName.OPENAI: ("",)}),
    ],
)
def test_blank_profile_identity_exposure_or_aliases_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_case_folded_alias_duplicates_are_rejected():
    with pytest.raises(ValueError):
        ModelAliases({InterfaceName.ANTHROPIC: ("local-opus", "LOCAL-OPUS")})


def test_profile_name_cannot_be_repeated_as_an_alias_under_case_folding():
    with pytest.raises(ValueError):
        ModelProfile(
            "Qwable",
            "upstream",
            "provider",
            frozenset({InterfaceName.ANTHROPIC}),
            ModelAliases({InterfaceName.ANTHROPIC: ("qwable",)}),
        )


def test_non_json_compatible_parameter_extensions_are_rejected():
    with pytest.raises(ValueError):
        ModelProfile(
            "profile",
            "upstream",
            "provider",
            frozenset({InterfaceName.OPENAI}),
            parameters=SamplingParameters(extra={"invalid": object()}),
        )
