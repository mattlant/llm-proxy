from __future__ import annotations

from llm_proxy.domain.events import ResponseStarted, TextDelta
from llm_proxy.interfaces.openai.stream_mapper import OpenAIStreamMapper


async def test_cancelling_openai_projection_closes_canonical_stream() -> None:
    closed = False

    async def events():
        nonlocal closed
        try:
            yield ResponseStarted("response", "upstream")
            yield TextDelta("text", "first")
        finally:
            closed = True

    stream = OpenAIStreamMapper().map_stream(events(), response_model="alias")
    await anext(stream)
    await stream.aclose()

    assert closed is True
