from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool name must be non-empty")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("input_schema must be a mapping")
        schema = copy.deepcopy(dict(self.input_schema))
        object.__setattr__(self, "input_schema", MappingProxyType(schema))
