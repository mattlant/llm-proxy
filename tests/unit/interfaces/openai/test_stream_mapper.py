from __future__ import annotations

import json

from llm_proxy.domain.events import ReasoningDelta, ResponseCompleted, ResponseStarted, TextDelta, ToolCallArgumentsDelta, ToolCallStarted
from llm_proxy.domain.events import ResponseFailed
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.responses import FinishReason, Usage
from llm_proxy.application.semantic_support import SemanticSupportResolver
from llm_proxy.configuration.models import InterfaceName


def completed(reason: FinishReason, usage: Usage):
    from llm_proxy.domain.events import ResponseCompleted
    event = ResponseCompleted(reason, None, usage)
    return ResponseCompleted(reason, None, usage, semantic_handling=SemanticSupportResolver().resolve_response(event, InterfaceName.OPENAI))
from llm_proxy.interfaces.openai.stream_mapper import OpenAIStreamMapper


async def test_projects_canonical_events_to_openai_sse() -> None:
    async def events():
        yield ResponseStarted("response-1", "upstream")
        yield TextDelta("text-1", "hello")
        yield ReasoningDelta("reasoning-1", "private")
        yield ToolCallStarted("tool-1", "call-1", "read")
        yield ToolCallArgumentsDelta("tool-1", "call-1", '{"path":')
        yield completed(FinishReason.TOOL_USE, Usage(2, 3))

    transcript = b"".join([chunk async for chunk in OpenAIStreamMapper().map_stream(events(), response_model="alias")]).decode()

    assert '"role":"assistant"' in transcript
    assert '"content":"hello"' in transcript
    assert '"reasoning_content":"private"' in transcript
    assert '"tool_calls":[{"index":0,"id":"call-1"' in transcript
    assert '"arguments":"{\\"path\\":"' in transcript
    assert '"finish_reason":"tool_calls"' in transcript
    assert transcript.endswith("data: [DONE]\n\n")
    chunks = [line[6:] for line in transcript.splitlines() if line.startswith("data: {")]
    assert json.loads(chunks[-1])["choices"] == []


async def test_ignores_late_failure_after_done() -> None:
    async def events():
        yield ResponseStarted("response-1", "upstream")
        yield completed(FinishReason.END_TURN, Usage(1, 1))
        yield ResponseFailed(ProviderUnavailableError("late"))

    transcript = b"".join([chunk async for chunk in OpenAIStreamMapper().map_stream(events(), response_model="alias")]).decode()

    assert transcript.endswith("data: [DONE]\n\n")
    assert "error" not in transcript


async def test_drains_source_after_completion_before_emitting_done() -> None:
    consumed_late_event = False

    async def events():
        nonlocal consumed_late_event
        yield ResponseStarted("response-1", "upstream")
        yield completed(FinishReason.END_TURN, Usage(1, 1))
        consumed_late_event = True
        yield TextDelta("text-1", "ignored")

    transcript = b"".join([chunk async for chunk in OpenAIStreamMapper().map_stream(events(), response_model="alias")]).decode()

    assert consumed_late_event is True
    assert '"content":"ignored"' not in transcript
    assert transcript.endswith("data: [DONE]\n\n")


async def test_projects_usage_updated_after_completion_before_done() -> None:
    completion = completed(FinishReason.END_TURN, Usage(None, None))

    async def events():
        yield ResponseStarted("response-1", "upstream")
        yield completion
        object.__setattr__(completion, "usage", Usage(7, 2))

    transcript = b"".join([chunk async for chunk in OpenAIStreamMapper().map_stream(events(), response_model="alias")]).decode()
    chunks = [json.loads(line[6:]) for line in transcript.splitlines() if line.startswith("data: {")]

    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    assert transcript.endswith("data: [DONE]\n\n")
