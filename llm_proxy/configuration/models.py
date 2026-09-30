from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from llm_proxy.domain.messages import DeveloperRoleMode
from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.observability.payload_trace import TraceMode


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python 3.10
    class StrEnum(str, Enum):
        pass


class InterfaceName(StrEnum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


class StructuredOutputMode(StrEnum):
    UNSUPPORTED = "unsupported"
    OPENAI_JSON_SCHEMA = "openai_json_schema"
    OLLAMA_FORMAT = "ollama_format"


class ModelAccessMode(StrEnum):
    CONFIGURED_ONLY = "configured_only"
    PROVIDER_PASSTHROUGH = "provider_passthrough"


class CapabilityFailureMode(StrEnum):
    REQUIRED = "required"
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True, slots=True)
class FileLoggingConfig:
    enabled: bool = False
    directory: str = "./logs"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("file_logging enabled must be a boolean")
        if not isinstance(self.directory, str) or not self.directory.strip():
            raise ValueError("file_logging directory must be non-blank")


@dataclass(frozen=True, slots=True)
class ModelAccessConfig:
    mode: ModelAccessMode = ModelAccessMode.CONFIGURED_ONLY
    listing_timeout_seconds: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ModelAccessMode):
            raise TypeError("mode must be a ModelAccessMode")
        if isinstance(self.listing_timeout_seconds, bool) or not isinstance(self.listing_timeout_seconds, int) or self.listing_timeout_seconds <= 0:
            raise ValueError("listing_timeout_seconds must be a positive integer")


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str
    port: int
    timeout_seconds: float
    config_reload_seconds: float
    log_level: str
    payload_trace_mode: TraceMode = TraceMode.DISABLED
    transport_debug: bool = False
    payload_trace_include_content: bool = True
    file_logging: FileLoggingConfig = field(default_factory=FileLoggingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be non-blank")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        if isinstance(self.config_reload_seconds, bool) or not isinstance(self.config_reload_seconds, (int, float)) or self.config_reload_seconds <= 0:
            raise ValueError("config_reload_seconds must be positive")
        if not isinstance(self.log_level, str) or not self.log_level.strip():
            raise ValueError("log_level must be non-blank")
        if not isinstance(self.payload_trace_mode, TraceMode):
            raise TypeError("payload_trace_mode must be a TraceMode")
        if not isinstance(self.transport_debug, bool):
            raise TypeError("transport_debug must be a boolean")
        if not isinstance(self.payload_trace_include_content, bool):
            raise TypeError("payload_trace_include_content must be a boolean")
        if not isinstance(self.file_logging, FileLoggingConfig):
            raise TypeError("file_logging must be a FileLoggingConfig")


@dataclass(frozen=True, slots=True)
class ManagementConfig:
    enabled: bool = False
    token_env: str = "LLM_PROXY_ADMIN_TOKEN"
    allow_remote: bool = False
    command_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.allow_remote, bool):
            raise TypeError("management enabled and allow_remote must be booleans")
        if not isinstance(self.token_env, str) or not self.token_env.strip():
            raise ValueError("management token_env must be non-blank")
        if isinstance(self.command_timeout_seconds, bool) or not isinstance(self.command_timeout_seconds, (int, float)) or self.command_timeout_seconds <= 0:
            raise ValueError("management command_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class AuthenticationConfig:
    allow_any_token: bool
    token: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.allow_any_token, bool):
            raise TypeError("allow_any_token must be a boolean")
        if self.token is not None and not isinstance(self.token, str):
            raise TypeError("token must be a string or null")
        if not self.allow_any_token and (self.token is None or not self.token.strip()):
            raise ValueError("token must be non-blank when allow_any_token is false")


@dataclass(frozen=True, slots=True)
class InterfaceConfig:
    name: InterfaceName
    enabled: bool
    authentication: AuthenticationConfig | None = None
    expose_thinking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, InterfaceName):
            raise TypeError("name must be an InterfaceName")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if self.authentication is not None and not isinstance(self.authentication, AuthenticationConfig):
            raise TypeError("authentication must be AuthenticationConfig or null")
        if not isinstance(self.expose_thinking, bool):
            raise TypeError("expose_thinking must be a boolean")


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    extension_id: str
    enabled: bool
    config: Mapping[str, Any] = field(default_factory=dict)
    listing_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("provider name must be non-blank")
        if not isinstance(self.extension_id, str) or not self.extension_id.strip():
            raise ValueError("provider extension_id must be non-blank")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not isinstance(self.listing_enabled, bool):
            raise TypeError("listing_enabled must be a boolean")
        if not isinstance(self.config, Mapping):
            raise TypeError("provider config must be a mapping")
        if not all(isinstance(key, str) and _is_json_value(value) for key, value in self.config.items()):
            raise ValueError("provider config must contain only JSON-compatible values with string keys")
        object.__setattr__(self, "config", _freeze_json(copy.deepcopy(dict(self.config))))


@dataclass(frozen=True, slots=True)
class ExtensionConfig:
    name: str
    extension_id: str
    enabled: bool
    config: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.extension_id, str) or not self.extension_id.strip(): raise ValueError("extension name and ID must be non-blank")
        if not isinstance(self.enabled, bool) or not isinstance(self.config, Mapping): raise TypeError("extension config is invalid")
        if not all(isinstance(key, str) and _is_json_value(value) for key, value in self.config.items()): raise ValueError("extension config must be JSON-compatible")
        object.__setattr__(self, "config", _freeze_json(copy.deepcopy(dict(self.config))))


@dataclass(frozen=True, slots=True)
class CapabilityBinding:
    id: str
    instance: str
    family: str
    version: int
    capability: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    on_failure: CapabilityFailureMode = CapabilityFailureMode.REQUIRED
    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), str) and getattr(self, name).strip() for name in ("id", "instance", "family", "capability")): raise ValueError("capability binding identity must be non-blank")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0: raise ValueError("capability version must be positive")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not 0 < self.timeout_seconds <= 600: raise ValueError("timeout_seconds must be in (0, 600]")
        if not isinstance(self.on_failure, CapabilityFailureMode): raise TypeError("on_failure must be CapabilityFailureMode")
        if not isinstance(self.arguments, Mapping) or not all(isinstance(key, str) and _is_json_value(value) for key, value in self.arguments.items()): raise ValueError("arguments must be JSON-compatible")
        object.__setattr__(self, "arguments", _freeze_json(copy.deepcopy(dict(self.arguments))))


@dataclass(frozen=True, slots=True)
class ModelCompatibility:
    expose_thinking: bool = False
    native_tools: bool = True
    structured_output: StructuredOutputMode = StructuredOutputMode.UNSUPPORTED
    developer_role_mode: DeveloperRoleMode = DeveloperRoleMode.PRESERVE

    def __post_init__(self) -> None:
        if not isinstance(self.expose_thinking, bool):
            raise TypeError("expose_thinking must be a boolean")
        if not isinstance(self.native_tools, bool):
            raise TypeError("native_tools must be a boolean")
        if not isinstance(self.structured_output, StructuredOutputMode):
            raise TypeError("structured_output must be a StructuredOutputMode")
        if not isinstance(self.developer_role_mode, DeveloperRoleMode):
            raise TypeError("developer_role_mode must be a DeveloperRoleMode")


@dataclass(frozen=True, slots=True)
class ModelAliases:
    by_interface: Mapping[InterfaceName, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.by_interface, Mapping):
            raise TypeError("by_interface must be a mapping")
        normalized: dict[InterfaceName, tuple[str, ...]] = {}
        for interface, aliases in self.by_interface.items():
            if not isinstance(interface, InterfaceName):
                raise TypeError("alias interface must be an InterfaceName")
            if not isinstance(aliases, (tuple, list)):
                raise TypeError("interface aliases must be a sequence")
            alias_values = tuple(aliases)
            for alias in alias_values:
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError("aliases must be non-blank strings")
            folded = [alias.casefold() for alias in alias_values]
            if len(folded) != len(set(folded)):
                raise ValueError(f"duplicate aliases for interface '{interface.value}'")
            normalized[interface] = alias_values
        object.__setattr__(self, "by_interface", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    upstream_model: str
    provider: str
    interfaces: frozenset[InterfaceName]
    aliases: ModelAliases = field(default_factory=lambda: ModelAliases({}))
    parameters: SamplingParameters = field(default_factory=SamplingParameters)
    compatibility: ModelCompatibility = field(default_factory=ModelCompatibility)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be non-blank")
        if not isinstance(self.upstream_model, str) or not self.upstream_model.strip():
            raise ValueError("upstream model must be non-blank")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider reference must be non-blank")
        if not isinstance(self.interfaces, (set, frozenset)):
            raise TypeError("interfaces must be a set of InterfaceName values")
        interfaces = frozenset(self.interfaces)
        if not interfaces:
            raise ValueError("profile must expose at least one interface")
        if not all(isinstance(interface, InterfaceName) for interface in interfaces):
            raise TypeError("interfaces must contain only InterfaceName values")
        if not isinstance(self.aliases, ModelAliases):
            raise TypeError("aliases must be ModelAliases")
        if not isinstance(self.parameters, SamplingParameters):
            raise TypeError("parameters must be SamplingParameters")
        if not isinstance(self.compatibility, ModelCompatibility):
            raise TypeError("compatibility must be ModelCompatibility")
        folded_name = self.name.casefold()
        if any(folded_name == alias.casefold() for aliases in self.aliases.by_interface.values() for alias in aliases):
            raise ValueError("profile name cannot also be an alias")
        if not all(_is_json_value(value) for value in self.parameters.extra.values()):
            raise ValueError("parameter extensions must contain only JSON-compatible values")
        object.__setattr__(self, "interfaces", interfaces)


@dataclass(frozen=True, slots=True)
class SamplingParametersOverride:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    extra: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("top_k", "repeat_last_n", "max_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.stop_sequences is not None:
            object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))
        if self.extra is not None:
            if not isinstance(self.extra, Mapping):
                raise TypeError("extra must be a mapping or null")
            if not all(_is_json_value(value) for value in self.extra.values()):
                raise ValueError("parameter extensions must contain only JSON-compatible values")
            object.__setattr__(self, "extra", MappingProxyType(copy.deepcopy(dict(self.extra))))


@dataclass(frozen=True, slots=True)
class PolicyMatch:
    interfaces: frozenset[InterfaceName] = frozenset()
    models: frozenset[str] = frozenset()
    model_contains_any: tuple[str, ...] = ()
    model_contains_all: tuple[str, ...] = ()
    model_wildcard_any: tuple[str, ...] = ()
    model_wildcard_all: tuple[str, ...] = ()
    text_contains_any: tuple[str, ...] = ()
    text_contains_all: tuple[str, ...] = ()
    tools_any: tuple[str, ...] = ()
    tools_all: tuple[str, ...] = ()
    tool_wildcard_any: tuple[str, ...] = ()
    tool_wildcard_all: tuple[str, ...] = ()
    metadata_equals: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(interface, InterfaceName) for interface in self.interfaces):
            raise TypeError("policy interfaces must contain InterfaceName values")
        for field_name in (
            "models",
            "model_contains_any",
            "model_contains_all",
            "model_wildcard_any",
            "model_wildcard_all",
            "text_contains_any",
            "text_contains_all",
            "tools_any",
            "tools_all",
            "tool_wildcard_any",
            "tool_wildcard_all",
        ):
            values = getattr(self, field_name)
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"policy {field_name} values must be non-blank strings")
            if field_name == "models":
                object.__setattr__(self, field_name, frozenset(values))
            else:
                object.__setattr__(self, field_name, tuple(values))
        if not isinstance(self.metadata_equals, Mapping):
            raise TypeError("metadata_equals must be a mapping")
        if not all(_is_json_value(value) for value in self.metadata_equals.values()):
            raise ValueError("metadata_equals values must be JSON-compatible")
        object.__setattr__(self, "interfaces", frozenset(self.interfaces))
        object.__setattr__(self, "metadata_equals", MappingProxyType(copy.deepcopy(dict(self.metadata_equals))))


@dataclass(frozen=True, slots=True)
class PolicyActions:
    model: str | None = None
    provider: str | None = None
    parameters: SamplingParametersOverride | None = None
    capabilities: tuple[CapabilityBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("policy model action must be non-blank")
        if self.provider is not None and (not isinstance(self.provider, str) or not self.provider.strip()):
            raise ValueError("policy provider action must be non-blank")
        if self.parameters is not None and not isinstance(self.parameters, SamplingParametersOverride):
            raise TypeError("policy parameters action must be SamplingParametersOverride or null")
        if not isinstance(self.capabilities, tuple) or not all(isinstance(value, CapabilityBinding) for value in self.capabilities):
            raise TypeError("capabilities must be CapabilityBinding values")
        if self.model is None and self.provider is None and self.parameters is None and not self.capabilities:
            raise ValueError("policy actions must not be empty")


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    name: str
    enabled: bool
    match: PolicyMatch
    actions: PolicyActions

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("policy name must be non-blank")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not isinstance(self.match, PolicyMatch):
            raise TypeError("match must be PolicyMatch")
        if not isinstance(self.actions, PolicyActions):
            raise TypeError("actions must be PolicyActions")


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    server: ServerConfig
    interfaces: Mapping[InterfaceName, InterfaceConfig]
    providers: Mapping[str, ProviderConfig]
    extensions: Mapping[str, ExtensionConfig] = field(default_factory=dict)
    models: Mapping[str, ModelProfile] = field(default_factory=dict)
    policies: tuple[PolicyConfig, ...] = ()
    management: ManagementConfig = field(default_factory=ManagementConfig)
    model_access: ModelAccessConfig = field(default_factory=ModelAccessConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.server, ServerConfig):
            raise TypeError("server must be ServerConfig")
        if not isinstance(self.management, ManagementConfig):
            raise TypeError("management must be ManagementConfig")
        if not isinstance(self.model_access, ModelAccessConfig):
            raise TypeError("model_access must be ModelAccessConfig")
        if not isinstance(self.interfaces, Mapping):
            raise TypeError("interfaces must be a mapping")
        if not isinstance(self.providers, Mapping):
            raise TypeError("providers must be a mapping")
        if not isinstance(self.extensions, Mapping) or not all(isinstance(key, str) and isinstance(value, ExtensionConfig) for key, value in self.extensions.items()): raise TypeError("extensions must map names to ExtensionConfig")
        if not isinstance(self.models, Mapping):
            raise TypeError("models must be a mapping")
        if not isinstance(self.policies, tuple):
            raise TypeError("policies must be a tuple")
        if not all(isinstance(key, InterfaceName) and isinstance(value, InterfaceConfig) for key, value in self.interfaces.items()):
            raise TypeError("interfaces must map InterfaceName to InterfaceConfig")
        if not all(isinstance(key, str) and isinstance(value, ProviderConfig) for key, value in self.providers.items()):
            raise TypeError("providers must map names to ProviderConfig")
        if not all(isinstance(key, str) and isinstance(value, ModelProfile) for key, value in self.models.items()):
            raise TypeError("models must map names to ModelProfile")
        if not all(isinstance(policy, PolicyConfig) for policy in self.policies):
            raise TypeError("policies must contain PolicyConfig values")
        object.__setattr__(self, "interfaces", MappingProxyType(dict(self.interfaces)))
        object.__setattr__(self, "providers", MappingProxyType(dict(self.providers)))
        object.__setattr__(self, "extensions", MappingProxyType(dict(self.extensions)))
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
