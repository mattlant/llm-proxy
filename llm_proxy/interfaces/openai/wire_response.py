from __future__ import annotations

from collections.abc import AsyncIterator
from fastapi.responses import Response, StreamingResponse
from llm_proxy.provider_extensions import WireResponse, WireStream


def _headers(pairs):
    # Starlette's mapping API cannot retain duplicates; assign raw headers after construction.
    return [(name.decode("latin-1"), value.decode("latin-1")) for name, value in pairs]


def wire_response(value: WireResponse) -> Response:
    response = Response(value.body, status_code=value.status_code)
    response.raw_headers = list(value.headers)
    return response


async def relay_wire_stream(stream: WireStream) -> AsyncIterator[bytes]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await stream.aclose()


def wire_stream_response(stream: WireStream) -> StreamingResponse:
    response = StreamingResponse(relay_wire_stream(stream), status_code=stream.status_code)
    response.raw_headers = list(stream.headers)
    return response
