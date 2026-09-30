"""Framework-neutral provider management-command SDK contracts."""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


JsonValue = None | bool | int | float | str | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
_COMMAND_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_INVOCATION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _freeze_json(value: Any, field_name: str) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} keys must be strings")
        return MappingProxyType({key: _freeze_json(item, field_name) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name) for item in value)
    raise TypeError(f"{field_name} must contain only JSON-compatible values")


class ManagementCommandMutability(StrEnum):
    READ_ONLY = "read_only"
    STATE_CHANGING = "state_changing"


class ManagementCommandPermission(StrEnum):
    INSPECT = "inspect"
    OPERATE = "operate"


@dataclass(frozen=True, slots=True)
class ManagementCommandDescriptor:
    name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    output_schema: Mapping[str, JsonValue]
    mutability: ManagementCommandMutability
    required_permission: ManagementCommandPermission

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _COMMAND_NAME.fullmatch(self.name):
            raise ValueError("command name must be a lowercase normalized identifier")
        if not isinstance(self.description, str) or not self.description.strip() or len(self.description) > 500:
            raise ValueError("command description must be non-blank and at most 500 characters")
        if not isinstance(self.mutability, ManagementCommandMutability) or not isinstance(self.required_permission, ManagementCommandPermission):
            raise TypeError("command mutability and permission must be management command enums")
        expected = ManagementCommandPermission.INSPECT if self.mutability is ManagementCommandMutability.READ_ONLY else ManagementCommandPermission.OPERATE
        if self.required_permission is not expected:
            raise ValueError("command mutability requires its corresponding permission")
        for name in ("input_schema", "output_schema"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze_json(copy.deepcopy(dict(value)), name))


@dataclass(frozen=True, slots=True)
class ManagementCommandContext:
    provider_instance: str
    invocation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider_instance, str) or not self.provider_instance.strip():
            raise ValueError("provider_instance must be non-blank")
        if not isinstance(self.invocation_id, str) or not _INVOCATION_ID.fullmatch(self.invocation_id):
            raise ValueError("invocation_id must be an opaque safe identifier")


class ProviderManagementCommandExecutor(Protocol):
    management_commands: tuple[ManagementCommandDescriptor, ...]

    async def execute_management_command(self, command_name: str, payload: JsonValue, context: ManagementCommandContext) -> JsonValue:
        ...
