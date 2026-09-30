from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.configuration.models import ProviderConfig
from llm_proxy.provider_extensions.management import ManagementCommandDescriptor, ProviderManagementCommandExecutor

from .command_validation import CommandSchemaError, compile_schema


class CommandRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ActivatedCommandProvider:
    provider_instance: str
    extension_id: str
    executor: ProviderManagementCommandExecutor
    commands: Mapping[str, ManagementCommandDescriptor]


def validate_command_executor(gateway: object, enabled: bool) -> None:
    if not enabled:
        return
    commands = getattr(gateway, "management_commands", None)
    execute = getattr(gateway, "execute_management_command", None)
    if not isinstance(commands, tuple) or not callable(execute):
        raise CommandRegistryError("management_commands capability requires a command executor")
    names: set[str] = set()
    for descriptor in commands:
        if not isinstance(descriptor, ManagementCommandDescriptor):
            raise CommandRegistryError("command executor has an invalid descriptor")
        if descriptor.name in names:
            raise CommandRegistryError(f"duplicate command name '{descriptor.name}'")
        names.add(descriptor.name)
        try:
            compile_schema(descriptor.input_schema)
            compile_schema(descriptor.output_schema)
        except CommandSchemaError as error:
            raise CommandRegistryError(f"invalid command schema for '{descriptor.name}'") from error


class ProviderCommandRegistry:
    def __init__(self, gateways: ProviderGatewayRegistry, providers: Mapping[str, ProviderConfig], extension_ids: Mapping[str, str], command_capabilities: Mapping[str, bool]) -> None:
        entries: dict[str, ActivatedCommandProvider] = {}
        self._known = frozenset(name for name, provider in providers.items() if provider.enabled)
        for name, provider in providers.items():
            if not provider.enabled or not command_capabilities.get(name, False):
                continue
            gateway = gateways.get(name)
            validate_command_executor(gateway, True)
            commands = {item.name: item for item in gateway.management_commands}
            entries[name] = ActivatedCommandProvider(name, extension_ids[name], gateway, MappingProxyType(commands))
        self._entries = MappingProxyType(entries)

    def has_instance(self, provider_instance: str) -> bool:
        return provider_instance in self._known

    def get(self, provider_instance: str) -> ActivatedCommandProvider | None:
        return self._entries.get(provider_instance)

    def count(self, provider_instance: str) -> int:
        entry = self.get(provider_instance)
        return len(entry.commands) if entry else 0
