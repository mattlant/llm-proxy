"""Provider-neutral canonical stream lifecycle validation."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from llm_proxy.provider_extensions import (
    ResponseCompleted, ResponseStarted, TextCompleted, TextDelta, TextStarted,
    ToolCallArgumentsDelta, ToolCallCompleted, ToolCallStarted,
)

from .adapter import ProviderComplianceViolation


async def observe_stream(provider_id: str, stream: AsyncIterator[Any]) -> tuple[Any, ...]:
    events: list[Any] = []
    blocks: dict[str, tuple[str, str | None]] = {}
    started = False
    completed = False
    async for event in stream:
        if completed:
            raise ProviderComplianceViolation(f"{provider_id}: streaming: event after ResponseCompleted ({type(event).__name__})")
        if isinstance(event, ResponseStarted):
            if started:
                raise ProviderComplianceViolation(f"{provider_id}: streaming: duplicate ResponseStarted")
            started = True
        elif not started:
            raise ProviderComplianceViolation(f"{provider_id}: streaming: {type(event).__name__} before ResponseStarted")
        elif isinstance(event, (TextStarted, ToolCallStarted)):
            if event.block_id in blocks:
                raise ProviderComplianceViolation(f"{provider_id}: streaming: duplicate block {event.block_id}")
            blocks[event.block_id] = ("text", None) if isinstance(event, TextStarted) else ("tool", event.tool_call_id)
        elif isinstance(event, TextDelta):
            if blocks.get(event.block_id, (None, None))[0] != "text":
                raise ProviderComplianceViolation(f"{provider_id}: streaming: TextDelta without open text block")
        elif isinstance(event, ToolCallArgumentsDelta):
            block = blocks.get(event.block_id)
            if block is None or block[0] != "tool":
                raise ProviderComplianceViolation(f"{provider_id}: streaming: ToolCallArgumentsDelta without open tool block")
            if block[1] != event.tool_call_id:
                raise ProviderComplianceViolation(f"{provider_id}: streaming: ToolCallArgumentsDelta tool call id differs from open block")
        elif isinstance(event, TextCompleted):
            if blocks.pop(event.block_id, (None, None))[0] != "text":
                raise ProviderComplianceViolation(f"{provider_id}: streaming: TextCompleted without open text block")
        elif isinstance(event, ToolCallCompleted):
            block = blocks.pop(event.block_id, None)
            if block is None or block[0] != "tool":
                raise ProviderComplianceViolation(f"{provider_id}: streaming: ToolCallCompleted without open tool block")
            if block[1] != event.tool_call_id:
                raise ProviderComplianceViolation(f"{provider_id}: streaming: ToolCallCompleted tool call id differs from open block")
        elif isinstance(event, ResponseCompleted):
            if blocks:
                raise ProviderComplianceViolation(f"{provider_id}: streaming: ResponseCompleted with open blocks")
            completed = True
        events.append(event)
    if not started:
        raise ProviderComplianceViolation(f"{provider_id}: streaming: missing ResponseStarted")
    if not completed:
        raise ProviderComplianceViolation(f"{provider_id}: streaming: missing ResponseCompleted")
    return tuple(events)
