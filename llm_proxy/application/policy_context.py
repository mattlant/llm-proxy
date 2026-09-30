from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from llm_proxy.configuration.models import InterfaceName
from llm_proxy.domain.messages import Message
from llm_proxy.domain.requests import ExecutionControls


@dataclass(frozen=True, slots=True)
class RequestPolicyContext:
    interface: InterfaceName
    requested_model: str
    canonical_model: str
    messages: tuple[Message, ...]
    tool_names: tuple[str, ...]
    metadata: Mapping[str, Any]
    controls: ExecutionControls = field(default_factory=ExecutionControls)

    def __post_init__(self) -> None:
        if not isinstance(self.interface, InterfaceName):
            raise TypeError("interface must be an InterfaceName")
        if not isinstance(self.requested_model, str) or not self.requested_model.strip():
            raise ValueError("requested_model must be non-blank")
        if not isinstance(self.canonical_model, str) or not self.canonical_model.strip():
            raise ValueError("canonical_model must be non-blank")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tool_names", tuple(self.tool_names))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not isinstance(self.controls, ExecutionControls):
            raise TypeError("controls must be an ExecutionControls")
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))
