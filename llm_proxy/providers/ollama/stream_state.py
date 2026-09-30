from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from llm_proxy.domain.errors import ProviderProtocolError


@dataclass(slots=True)
class ToolCallStreamState:
    upstream_index: int
    block_id: str
    id_parts: list[str] = field(default_factory=list)
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)
    started: bool = False
    completed: bool = False

    @property
    def tool_call_id(self) -> str | None:
        return "".join(self.id_parts) or None

    @property
    def name(self) -> str | None:
        return "".join(self.name_parts) or None

    @property
    def arguments(self) -> str:
        return "".join(self.argument_parts)

    def append_id(self, fragment: str) -> None:
        self._require_active(fragment, "tool call ID")
        if self.started:
            raise ProviderProtocolError("tool call ID changed after start")
        self.id_parts.append(fragment)

    def append_name(self, fragment: str) -> None:
        self._require_active(fragment, "tool call name")
        if self.started:
            raise ProviderProtocolError("tool call name changed after start")
        self.name_parts.append(fragment)

    def append_arguments(self, fragment: str) -> None:
        self._require_active(fragment, "tool call arguments")
        self.argument_parts.append(fragment)

    def mark_started(self) -> None:
        if self.completed:
            raise ProviderProtocolError("tool call started after completion")
        if not self.tool_call_id or not self.name:
            raise ProviderProtocolError("tool call cannot start without an ID and name")
        self.started = True

    def mark_completed(self) -> None:
        if self.completed:
            raise ProviderProtocolError("tool call completed more than once")
        if not self.started:
            raise ProviderProtocolError("tool call cannot complete before start")
        self.completed = True

    def _require_active(self, fragment: str, field_name: str) -> None:
        if self.completed:
            raise ProviderProtocolError(f"{field_name} received after tool call completion")
        if not isinstance(fragment, str):
            raise TypeError(f"{field_name} fragments must be strings")


@dataclass(slots=True)
class ToolCallStreamStateStore:
    response_id: str
    states: dict[int, ToolCallStreamState] = field(default_factory=dict)
    ids: dict[str, int] = field(default_factory=dict)

    def state_for(self, upstream_index: int) -> ToolCallStreamState:
        if isinstance(upstream_index, bool) or not isinstance(upstream_index, int) or upstream_index < 0:
            raise ProviderProtocolError("tool call index must be a non-negative integer")
        return self.states.setdefault(
            upstream_index,
            ToolCallStreamState(upstream_index, f"tool_{upstream_index}"),
        )

    def append_id(self, upstream_index: int, fragment: str) -> ToolCallStreamState:
        state = self.state_for(upstream_index)
        state.append_id(fragment)
        return state

    def append_name(self, upstream_index: int, fragment: str) -> ToolCallStreamState:
        state = self.state_for(upstream_index)
        state.append_name(fragment)
        return state

    def append_arguments(self, upstream_index: int, fragment: str) -> ToolCallStreamState:
        state = self.state_for(upstream_index)
        state.append_arguments(fragment)
        return state

    def start_if_ready(self, upstream_index: int) -> ToolCallStreamState:
        state = self.state_for(upstream_index)
        if not state.name:
            raise ProviderProtocolError("tool call cannot start without an ID and name")
        if not state.started and not state.tool_call_id:
            fallback = self._fallback_id(upstream_index)
            state.id_parts.append(fallback)
        if state.tool_call_id:
            owner = self.ids.get(state.tool_call_id)
            if owner is not None and owner != upstream_index:
                raise ProviderProtocolError("tool call ID is shared by multiple upstream indices")
            self.ids[state.tool_call_id] = upstream_index
        if not state.started:
            state.mark_started()
        return state

    def _fallback_id(self, upstream_index: int) -> str:
        digest = hashlib.sha256(f"{self.response_id}:{upstream_index}".encode("utf-8")).hexdigest()[:16]
        return f"call_{digest}"
