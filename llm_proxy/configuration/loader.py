from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.domain.parameters import CORE_PARAMETER_CATALOG, CompatibilityParameters, EffectiveParameters
from llm_proxy.domain.messages import DeveloperRoleMode

from .models import (
    AuthenticationConfig,
    CapabilityBinding, CapabilityFailureMode, ExtensionConfig,
    GatewayConfig,
    InterfaceConfig,
    InterfaceName,
    FileLoggingConfig,
    ManagementConfig,
    ModelAccessConfig,
    ModelAccessMode,
    ModelAliases,
    ModelCompatibility,
    ModelProfile,
    PolicyActions,
    PolicyConfig,
    PolicyMatch,
    ProviderConfig,
    StructuredOutputMode,
    TraceMode,
    SamplingParametersOverride,
    ServerConfig,
)


class ConfigurationError(Exception):
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


class ConfigurationLoader:
    def load_path(self, path: Path) -> GatewayConfig:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ConfigurationError("$", f"unable to read {path}: {error}") from error
        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as error:
            raise ConfigurationError("$", f"malformed YAML: {error}") from error
        return self.load_mapping(raw)

    def load_text(self, content: str) -> GatewayConfig:
        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as error:
            raise ConfigurationError("$", f"malformed YAML: {error}") from error
        return self.load_mapping(raw)

    def load_mapping(self, raw: Mapping[str, Any]) -> GatewayConfig:
        if not isinstance(raw, Mapping):
            raise ConfigurationError("$", "configuration root must be a mapping")
        self._require_keys(raw, {"server", "management", "interfaces", "providers", "extensions", "models", "policies", "model_access"}, "$")
        for section in ("server", "interfaces", "providers", "models"):
            if section not in raw:
                raise ConfigurationError(f"$.{section}", "section is required")
        policies_raw = raw.get("policies", [])
        return GatewayConfig(
            server=self._server(raw["server"], "server"),
            management=self._management(raw.get("management", {}), "management"),
            interfaces=self._interfaces(raw["interfaces"], "interfaces"),
            providers=self._providers(raw["providers"], "providers"),
            extensions=self._extensions(raw.get("extensions", {}), "extensions"),
            models=self._models(raw["models"], "models"),
            policies=self._policies(policies_raw, "policies"),
            model_access=self._model_access(raw.get("model_access", {}), "model_access"),
        )

    def _extensions(self, raw: Any, path: str) -> Mapping[str, ExtensionConfig]:
        result = {}
        for name, item in self._require_mapping(raw, path).items():
            value = self._require_mapping(item, f"{path}.{name}")
            self._require_keys(value, {"extension", "enabled", "config"}, f"{path}.{name}")
            try: result[name] = ExtensionConfig(name, self._required(value, "extension", f"{path}.{name}"), self._required(value, "enabled", f"{path}.{name}"), self._required(value, "config", f"{path}.{name}"))
            except (TypeError, ValueError) as error: raise ConfigurationError(f"{path}.{name}", str(error)) from error
        return result

    def _model_access(self, raw: Any, path: str) -> ModelAccessConfig:
        value = self._require_mapping(raw, path)
        self._require_keys(value, {"mode", "listing_timeout_seconds"}, path)
        try:
            return ModelAccessConfig(self._enum(ModelAccessMode, value.get("mode", "configured_only"), f"{path}.mode"), value.get("listing_timeout_seconds", 3))
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    def _management(self, raw: Any, path: str) -> ManagementConfig:
        value = self._require_mapping(raw, path)
        self._require_keys(value, {"enabled", "token_env", "allow_remote", "command_timeout_seconds"}, path)
        try:
            return ManagementConfig(
                enabled=value.get("enabled", False),
                token_env=value.get("token_env", "LLM_PROXY_ADMIN_TOKEN"),
                allow_remote=value.get("allow_remote", False),
                command_timeout_seconds=value.get("command_timeout_seconds", 10.0),
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    @staticmethod
    def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ConfigurationError(path, "must be a mapping")
        return value

    @staticmethod
    def _require_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
        for key in value:
            if not isinstance(key, str) or key not in allowed:
                raise ConfigurationError(f"{path}.{key}", "unknown configuration key")

    @staticmethod
    def _required(value: Mapping[str, Any], key: str, path: str) -> Any:
        if key not in value:
            raise ConfigurationError(f"{path}.{key}", "field is required")
        return value[key]

    @staticmethod
    def _enum(enum_type, value: Any, path: str):
        if not isinstance(value, str):
            raise ConfigurationError(path, "must be a string")
        try:
            return enum_type(value)
        except ValueError as error:
            raise ConfigurationError(path, f"unsupported value '{value}'") from error

    def _server(self, raw: Any, path: str) -> ServerConfig:
        value = self._require_mapping(raw, path)
        self._require_keys(value, {"host", "port", "timeout_seconds", "config_reload_seconds", "log_level", "payload_trace_mode", "transport_debug", "payload_trace_include_content", "file_logging"}, path)
        file_logging = self._require_mapping(value.get("file_logging", {}), f"{path}.file_logging")
        self._require_keys(file_logging, {"enabled", "directory"}, f"{path}.file_logging")
        try:
            return ServerConfig(
                self._required(value, "host", path),
                self._required(value, "port", path),
                self._required(value, "timeout_seconds", path),
                self._required(value, "config_reload_seconds", path),
                self._required(value, "log_level", path),
                self._enum(TraceMode, value.get("payload_trace_mode", "disabled"), f"{path}.payload_trace_mode"),
                value.get("transport_debug", False),
                value.get("payload_trace_include_content", True),
                FileLoggingConfig(file_logging.get("enabled", False), file_logging.get("directory", "./logs")),
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    def _interfaces(self, raw: Any, path: str) -> Mapping[InterfaceName, InterfaceConfig]:
        value = self._require_mapping(raw, path)
        interfaces: dict[InterfaceName, InterfaceConfig] = {}
        for key, item in value.items():
            item_path = f"{path}.{key}"
            interface = self._enum(InterfaceName, key, item_path)
            config = self._require_mapping(item, item_path)
            self._require_keys(config, {"enabled", "authentication", "expose_thinking"}, item_path)
            authentication = None
            if "authentication" in config and config["authentication"] is not None:
                auth_path = f"{item_path}.authentication"
                auth = self._require_mapping(config["authentication"], auth_path)
                self._require_keys(auth, {"allow_any_token", "token"}, auth_path)
                try:
                    authentication = AuthenticationConfig(
                        self._required(auth, "allow_any_token", auth_path),
                        self._required(auth, "token", auth_path),
                    )
                except (TypeError, ValueError) as error:
                    raise ConfigurationError(auth_path, str(error)) from error
            try:
                interfaces[interface] = InterfaceConfig(
                    interface,
                    self._required(config, "enabled", item_path),
                    authentication,
                    config.get("expose_thinking", False),
                )
            except (TypeError, ValueError) as error:
                raise ConfigurationError(item_path, str(error)) from error
        return interfaces

    def _providers(self, raw: Any, path: str) -> Mapping[str, ProviderConfig]:
        value = self._require_mapping(raw, path)
        providers: dict[str, ProviderConfig] = {}
        for name, item in value.items():
            item_path = f"{path}.{name}"
            config = self._require_mapping(item, item_path)
            self._require_keys(config, {"extension", "config", "enabled", "listing_enabled"}, item_path)
            try:
                providers[name] = ProviderConfig(
                    name,
                    self._required(config, "extension", item_path),
                    self._required(config, "enabled", item_path),
                    self._required(config, "config", item_path),
                    config.get("listing_enabled", False),
                )
            except (TypeError, ValueError) as error:
                raise ConfigurationError(item_path, str(error)) from error
        return providers

    def _models(self, raw: Any, path: str) -> Mapping[str, ModelProfile]:
        value = self._require_mapping(raw, path)
        models: dict[str, ModelProfile] = {}
        for name, item in value.items():
            item_path = f"{path}.{name}"
            config = self._require_mapping(item, item_path)
            self._require_keys(config, {"upstream_model", "provider", "interfaces", "aliases", "parameters", "compatibility"}, item_path)
            interfaces_raw = self._required(config, "interfaces", item_path)
            if not isinstance(interfaces_raw, (list, tuple)):
                raise ConfigurationError(f"{item_path}.interfaces", "must be a sequence")
            interfaces = frozenset(self._enum(InterfaceName, interface, f"{item_path}.interfaces[{index}]") for index, interface in enumerate(interfaces_raw))
            aliases = self._aliases(config.get("aliases", {}), f"{item_path}.aliases")
            parameters = self._parameters(config.get("parameters", {}), f"{item_path}.parameters")
            compatibility = self._compatibility(config.get("compatibility", {}), f"{item_path}.compatibility")
            upstream_model = self._required(config, "upstream_model", item_path)
            provider = self._required(config, "provider", item_path)
            if not isinstance(upstream_model, str) or not upstream_model.strip():
                raise ConfigurationError(f"{item_path}.upstream_model", "must be non-blank")
            if not isinstance(provider, str) or not provider.strip():
                raise ConfigurationError(f"{item_path}.provider", "must be non-blank")
            try:
                models[name] = ModelProfile(
                    name,
                    upstream_model,
                    provider,
                    interfaces,
                    aliases,
                    parameters,
                    compatibility,
                )
            except (TypeError, ValueError) as error:
                raise ConfigurationError(item_path, str(error)) from error
        return models

    def _aliases(self, raw: Any, path: str) -> ModelAliases:
        value = self._require_mapping(raw, path)
        aliases: dict[InterfaceName, tuple[str, ...]] = {}
        for key, item in value.items():
            interface = self._enum(InterfaceName, key, f"{path}.{key}")
            if not isinstance(item, (list, tuple)):
                raise ConfigurationError(f"{path}.{key}", "must be a sequence")
            aliases[interface] = tuple(item)
        try:
            return ModelAliases(aliases)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    def _parameters(self, raw: Any, path: str) -> SamplingParameters:
        value = self._require_mapping(raw, path)
        allowed = set(CORE_PARAMETER_CATALOG.names) | {"extra"}
        self._require_keys(value, allowed, path)
        stop_sequences = value.get("stop_sequences", ())
        if not isinstance(stop_sequences, (list, tuple)):
            raise ConfigurationError(f"{path}.stop_sequences", "must be a sequence")
        try:
            canonical = {name: value[name] for name in CORE_PARAMETER_CATALOG.names if name in value and name != "stop_sequences" and value[name] is not None}
            if stop_sequences:
                canonical["stop_sequences"] = tuple(stop_sequences)
            return EffectiveParameters(canonical, compatibility=CompatibilityParameters(value.get("extra", {}))).to_legacy()
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    def _compatibility(self, raw: Any, path: str) -> ModelCompatibility:
        value = self._require_mapping(raw, path)
        self._require_keys(value, {"expose_thinking", "native_tools", "structured_output", "developer_role_mode"}, path)
        try:
            return ModelCompatibility(
                value.get("expose_thinking", False),
                value.get("native_tools", True),
                self._enum(StructuredOutputMode, value.get("structured_output", "unsupported"), f"{path}.structured_output"),
                self._enum(DeveloperRoleMode, value.get("developer_role_mode", "preserve"), f"{path}.developer_role_mode"),
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    def _policies(self, raw: Any, path: str) -> tuple[PolicyConfig, ...]:
        if not isinstance(raw, (list, tuple)):
            raise ConfigurationError(path, "must be a sequence")
        policies: list[PolicyConfig] = []
        for index, item in enumerate(raw):
            item_path = f"{path}[{index}]"
            config = self._require_mapping(item, item_path)
            self._require_keys(config, {"name", "enabled", "match", "actions"}, item_path)
            try:
                policies.append(
                    PolicyConfig(
                        self._required(config, "name", item_path),
                        self._required(config, "enabled", item_path),
                        self._policy_match(self._required(config, "match", item_path), f"{item_path}.match"),
                        self._policy_actions(self._required(config, "actions", item_path), f"{item_path}.actions"),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ConfigurationError(item_path, str(error)) from error
        return tuple(policies)

    def _policy_match(self, raw: Any, path: str) -> PolicyMatch:
        value = self._require_mapping(raw, path)
        self._require_keys(value, {
            "interfaces", "models", "model_contains_any", "model_contains_all",
            "model_wildcard_any", "model_wildcard_all", "text_contains_any",
            "text_contains_all", "tools_any", "tools_all", "tool_wildcard_any",
            "tool_wildcard_all", "metadata_equals",
        }, path)

        def string_sequence(key: str) -> tuple[str, ...]:
            raw_values = value.get(key, ())
            if not isinstance(raw_values, (list, tuple)):
                raise ConfigurationError(f"{path}.{key}", "must be a sequence")
            values = tuple(raw_values)
            for index, item in enumerate(values):
                if not isinstance(item, str) or not item.strip():
                    raise ConfigurationError(f"{path}.{key}[{index}]", "must be a non-blank string")
            return values

        interface_values = string_sequence("interfaces")
        try:
            return PolicyMatch(
                interfaces=frozenset(self._enum(InterfaceName, item, f"{path}.interfaces[{index}]") for index, item in enumerate(interface_values)),
                models=frozenset(string_sequence("models")),
                model_contains_any=string_sequence("model_contains_any"),
                model_contains_all=string_sequence("model_contains_all"),
                model_wildcard_any=string_sequence("model_wildcard_any"),
                model_wildcard_all=string_sequence("model_wildcard_all"),
                text_contains_any=string_sequence("text_contains_any"),
                text_contains_all=string_sequence("text_contains_all"),
                tools_any=string_sequence("tools_any"),
                tools_all=string_sequence("tools_all"),
                tool_wildcard_any=string_sequence("tool_wildcard_any"),
                tool_wildcard_all=string_sequence("tool_wildcard_all"),
                metadata_equals=value.get("metadata_equals", {}),
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    def _policy_actions(self, raw: Any, path: str) -> PolicyActions:
        value = self._require_mapping(raw, path)
        self._require_keys(value, {"model", "provider", "parameters", "capabilities"}, path)
        parameters = None
        if "parameters" in value and value["parameters"] is not None:
            parameters = self._parameter_override(value["parameters"], f"{path}.parameters")
        try:
            return PolicyActions(value.get("model"), value.get("provider"), parameters, self._capability_bindings(value.get("capabilities", ()), f"{path}.capabilities"))
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error

    def _capability_bindings(self, raw: Any, path: str) -> tuple[CapabilityBinding, ...]:
        if not isinstance(raw, (list, tuple)): raise ConfigurationError(path, "must be a sequence")
        values = []
        for index, item in enumerate(raw):
            item_path = f"{path}[{index}]"; value = self._require_mapping(item, item_path)
            self._require_keys(value, {"id", "instance", "family", "version", "capability", "arguments", "timeout_seconds", "on_failure"}, item_path)
            try: values.append(CapabilityBinding(self._required(value, "id", item_path), self._required(value, "instance", item_path), self._required(value, "family", item_path), self._required(value, "version", item_path), self._required(value, "capability", item_path), value.get("arguments", {}), value.get("timeout_seconds", 30.0), self._enum(CapabilityFailureMode, value.get("on_failure", "required"), f"{item_path}.on_failure")))
            except (TypeError, ValueError) as error: raise ConfigurationError(item_path, str(error)) from error
        return tuple(values)

    def _parameter_override(self, raw: Any, path: str) -> SamplingParametersOverride:
        value = self._require_mapping(raw, path)
        allowed = set(CORE_PARAMETER_CATALOG.names) | {"extra"}
        self._require_keys(value, allowed, path)
        stop_sequences = value.get("stop_sequences")
        if stop_sequences is not None and not isinstance(stop_sequences, (list, tuple)):
            raise ConfigurationError(f"{path}.stop_sequences", "must be a sequence")
        try:
            fields = {name: value[name] for name in SamplingParametersOverride.__dataclass_fields__ if name in value}
            if stop_sequences is not None:
                fields["stop_sequences"] = tuple(stop_sequences)
            if "extra" in value:
                fields["extra"] = value["extra"]
            return SamplingParametersOverride(**fields)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(path, str(error)) from error
