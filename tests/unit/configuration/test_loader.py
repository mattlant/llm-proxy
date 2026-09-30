from __future__ import annotations

from pathlib import Path

import pytest

from llm_proxy.configuration.loader import ConfigurationError, ConfigurationLoader
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.domain.messages import DeveloperRoleMode
from llm_proxy.observability.payload_trace import TraceMode


VALID_CONFIG = """
server:
  host: 127.0.0.1
  port: 11435
  timeout_seconds: 0
  config_reload_seconds: 1.0
  log_level: INFO
interfaces:
  openai:
    enabled: true
  ollama:
    enabled: true
  anthropic:
    enabled: true
    authentication:
      allow_any_token: true
      token: null
    expose_thinking: false
providers:
  test-provider:
    extension: ollama
    enabled: true
    config:
      base_url: http://provider.test:11434/
      outbound_interface: openai
models:
  qwable:
    upstream_model: hf.co/example/Qwable:Q4_K_M
    provider: test-provider
    interfaces: [anthropic]
    aliases:
      anthropic: [local-opus]
    parameters:
      temperature: 0.7
      top_p: 0.95
      extra:
        seed: 7
    compatibility:
      native_tools: true
policies: []
"""


def test_complete_valid_configuration_loads_to_immutable_typed_objects():
    config = ConfigurationLoader().load_text(VALID_CONFIG)

    assert config.server.port == 11435
    assert config.interfaces[InterfaceName.ANTHROPIC].enabled is True
    assert config.providers["test-provider"].extension_id == "ollama"
    assert config.providers["test-provider"].config["base_url"] == "http://provider.test:11434/"
    assert config.models["qwable"].aliases.by_interface[InterfaceName.ANTHROPIC] == ("local-opus",)
    assert config.models["qwable"].parameters.extra["seed"] == 7
    assert config.policies == ()
    assert config.models["qwable"].compatibility.developer_role_mode is DeveloperRoleMode.PRESERVE
    assert config.server.payload_trace_mode is TraceMode.DISABLED
    assert config.server.transport_debug is False
    assert config.server.payload_trace_include_content is True


@pytest.mark.parametrize("mode", ["disabled", "structured", "raw", "both"])
def test_payload_trace_setting_defaults_and_requires_mode(mode):
    assert ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", f"  log_level: INFO\n  payload_trace_mode: {mode}")).server.payload_trace_mode.value == mode
    with pytest.raises(ConfigurationError, match="payload_trace_mode"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", "  log_level: INFO\n  payload_trace_mode: true"))


def test_transport_debug_defaults_and_requires_boolean() -> None:
    assert ConfigurationLoader().load_text(VALID_CONFIG).server.transport_debug is False
    assert ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", "  log_level: INFO\n  transport_debug: true")).server.transport_debug is True
    with pytest.raises(ConfigurationError, match="transport_debug"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", "  log_level: INFO\n  transport_debug: nope"))


def test_file_logging_defaults_and_valid_nested_configuration() -> None:
    config = ConfigurationLoader().load_text(VALID_CONFIG)
    assert config.server.file_logging.enabled is False
    assert config.server.file_logging.directory == "./logs"
    config = ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", "  log_level: INFO\n  file_logging:\n    enabled: true\n    directory: ./diagnostics"))
    assert config.server.file_logging.enabled is True
    assert config.server.file_logging.directory == "./diagnostics"


@pytest.mark.parametrize(
    "fragment",
    [
        "file_logging: true",
        "file_logging:\n    enabled: nope",
        "file_logging:\n    directory: '  '",
        "file_logging:\n    unexpected: true",
    ],
)
def test_file_logging_invalid_nested_configuration_is_reported(fragment):
    with pytest.raises(ConfigurationError, match=r"server(?:\.file_logging)?"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", f"  log_level: INFO\n  {fragment}"))


def test_payload_trace_include_content_defaults_and_requires_boolean() -> None:
    assert ConfigurationLoader().load_text(VALID_CONFIG).server.payload_trace_include_content is True
    assert ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", "  log_level: INFO\n  payload_trace_include_content: false")).server.payload_trace_include_content is False
    with pytest.raises(ConfigurationError, match="payload_trace_include_content"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("  log_level: INFO", "  log_level: INFO\n  payload_trace_include_content: nope"))


@pytest.mark.parametrize("mode", ["preserve", "system", "reject"])
def test_developer_role_mode_loads_and_invalid_values_fail(mode):
    config = ConfigurationLoader().load_text(VALID_CONFIG.replace("      native_tools: true", f"      native_tools: true\n      developer_role_mode: {mode}"))
    assert config.models["qwable"].compatibility.developer_role_mode.value == mode

    with pytest.raises(ConfigurationError, match=r"developer_role_mode"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("      native_tools: true", "      native_tools: true\n      developer_role_mode: invalid"))


def test_minimal_valid_configuration_loads_with_empty_aliases_and_policies():
    config = ConfigurationLoader().load_mapping(
        {
            "server": {
                "host": "127.0.0.1",
                "port": 11435,
                "timeout_seconds": 0,
                "config_reload_seconds": 1,
                "log_level": "INFO",
            },
            "interfaces": {"openai": {"enabled": True}},
            "providers": {
                "local": {
                    "extension": "ollama",
                    "enabled": True,
                    "config": {"base_url": "http://provider.test:11434", "outbound_interface": "openai"},
                }
            },
            "models": {
                "model": {
                    "upstream_model": "model:latest",
                    "provider": "local",
                    "interfaces": ["openai"],
                }
            },
        }
    )

    assert config.models["model"].aliases.by_interface == {}
    assert config.policies == ()


@pytest.mark.parametrize("section", ["server", "providers", "models"])
def test_required_sections_are_reported(section):
    config = ConfigurationLoader().load_text(VALID_CONFIG)
    raw = {
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "timeout_seconds": config.server.timeout_seconds,
            "config_reload_seconds": config.server.config_reload_seconds,
            "log_level": config.server.log_level,
        },
        "interfaces": {"openai": {"enabled": True}},
        "providers": {"local": {"extension": "ollama", "config": {"base_url": "http://provider.test:11434", "outbound_interface": "openai"}, "enabled": True}},
        "models": {"model": {"upstream_model": "model", "provider": "local", "interfaces": ["openai"]}},
    }
    del raw[section]

    with pytest.raises(ConfigurationError, match=rf"\$\.{section}"):
        ConfigurationLoader().load_mapping(raw)


def test_unknown_top_level_and_nested_keys_fail():
    with pytest.raises(ConfigurationError, match=r"\$\.unexpected"):
        ConfigurationLoader().load_mapping({"unexpected": True})
    with pytest.raises(ConfigurationError, match=r"models\.qwable\.unknown"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("  qwable:\n", "  qwable:\n    unknown: true\n"))


def test_wrong_types_and_invalid_enums_report_nested_paths():
    with pytest.raises(ConfigurationError, match=r"server: port must be between"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("port: 11435", "port: nope"))
    with pytest.raises(ConfigurationError, match=r"providers\.test-provider.*extension_id"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("extension: ollama", "extension: '  '"))
    with pytest.raises(ConfigurationError, match=r"models\.qwable\.provider"):
        ConfigurationLoader().load_text(VALID_CONFIG.replace("provider: test-provider", "provider: '  '") )


def test_malformed_yaml_fails_with_configuration_error():
    with pytest.raises(ConfigurationError, match="malformed YAML"):
        ConfigurationLoader().load_text("server: [")


def test_load_path_reads_utf8_configuration(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID_CONFIG, encoding="utf-8")

    config = ConfigurationLoader().load_path(path)

    assert config.models["qwable"].name == "qwable"


def test_policy_contract_loads_matchers_and_partial_actions():
    import yaml

    raw = yaml.safe_load(VALID_CONFIG)
    raw["policies"] = [
        {
            "name": "research-temperature",
            "enabled": True,
            "match": {
                "interfaces": ["anthropic"],
                "models": ["qwable"],
                "model_contains_any": ["qw", "qw"],
                "model_contains_all": ["able", "qw"],
                "model_wildcard_any": ["*able"],
                "model_wildcard_all": ["qw*", "*able"],
                "text_contains_any": ["research"],
                "tools_all": ["read_file"],
                "tool_wildcard_any": ["read_*"],
                "tool_wildcard_all": ["*_file"],
                "metadata_equals": {"tenant": "research"},
            },
            "actions": {"parameters": {"temperature": 0.0}},
        }
    ]

    config = ConfigurationLoader().load_mapping(raw)

    assert config.policies[0].match.interfaces == {InterfaceName.ANTHROPIC}
    assert config.policies[0].match.model_contains_any == ("qw", "qw")
    assert config.policies[0].match.tool_wildcard_all == ("*_file",)
    assert config.policies[0].actions.parameters.temperature == 0.0


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("model_contains_any", "qwopus", r"model_contains_any"),
        ("model_wildcard_any", [1], r"model_wildcard_any\[0\]"),
        ("tool_wildcard_all", ["  "], r"tool_wildcard_all\[0\]"),
    ],
)
def test_policy_matcher_values_have_field_indexed_errors(field, value, path):
    import yaml

    raw = yaml.safe_load(VALID_CONFIG)
    raw["policies"] = [{"name": "invalid", "enabled": True, "match": {field: value}, "actions": {"parameters": {"temperature": 0.1}}}]

    with pytest.raises(ConfigurationError, match=path):
        ConfigurationLoader().load_mapping(raw)


def test_empty_or_unknown_policy_action_fails():
    import yaml

    raw = yaml.safe_load(VALID_CONFIG)
    raw["policies"] = [{"name": "empty", "enabled": True, "match": {}, "actions": {}}]
    with pytest.raises(ConfigurationError, match=r"policies\[0\].actions.*must not be empty"):
        ConfigurationLoader().load_mapping(raw)


@pytest.mark.parametrize("name", ("config.yaml", "config.example.yaml"))
def test_test_owned_configuration_files_load(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text(VALID_CONFIG, encoding="utf-8")

    config = ConfigurationLoader().load_path(path)

    assert config.providers["test-provider"].extension_id == "ollama"
