from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BlockKind = Literal["text", "tool_use", "thinking"]


@dataclass(slots=True)
class AnthropicContentBlockState:
    canonical_block_id: str
    index: int
    kind: BlockKind
    tool_call_id: str | None = None
    tool_name: str | None = None
    fragments: list[str] = field(default_factory=list)
    stopped: bool = False


@dataclass(slots=True)
class AnthropicStreamState:
    response_id: str | None = None
    model: str | None = None
    next_content_index: int = 0
    blocks_by_id: dict[str, AnthropicContentBlockState] = field(default_factory=dict)
    started: bool = False
    completed: bool = False

    def start_message(self, response_id: str, model: str) -> None:
        if self.started or self.completed:
            raise ValueError("message has already started")
        self.response_id = response_id
        self.model = model
        self.started = True

    def start_block(
        self,
        canonical_block_id: str,
        kind: BlockKind,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> AnthropicContentBlockState:
        self._require_open_message()
        if canonical_block_id in self.blocks_by_id:
            raise ValueError("content block has already started")
        block = AnthropicContentBlockState(canonical_block_id, self.next_content_index, kind, tool_call_id, tool_name)
        self.blocks_by_id[canonical_block_id] = block
        self.next_content_index += 1
        return block

    def append_delta(self, canonical_block_id: str, fragment: str) -> AnthropicContentBlockState:
        block = self._require_open_block(canonical_block_id)
        block.fragments.append(fragment)
        return block

    def stop_block(self, canonical_block_id: str) -> AnthropicContentBlockState:
        block = self._require_open_block(canonical_block_id)
        block.stopped = True
        return block

    def complete_message(self) -> None:
        self._require_open_message()
        if any(not block.stopped for block in self.blocks_by_id.values()):
            raise ValueError("message completed with open content blocks")
        self.completed = True

    def _require_open_message(self) -> None:
        if not self.started:
            raise ValueError("message has not started")
        if self.completed:
            raise ValueError("message has already completed")

    def _require_open_block(self, canonical_block_id: str) -> AnthropicContentBlockState:
        self._require_open_message()
        block = self.blocks_by_id.get(canonical_block_id)
        if block is None:
            raise ValueError("content block has not started")
        if block.stopped:
            raise ValueError("content block has already stopped")
        return block
