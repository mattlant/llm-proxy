from collections.abc import AsyncIterator
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.configuration.models import InterfaceName

from .dto import OpenAIChatCompletionRequest
from .error_mapper import OpenAIErrorMapper
from .request_mapper import OpenAIRequestMapper
from .response_mapper import OpenAIResponseMapper
from .stream_mapper import OpenAIStreamMapper
from .wire_request import OpenAIWireRequestMapper
from .wire_response import wire_response, wire_stream_response

router = APIRouter()
log = logging.getLogger("llm-proxy")


async def _with_first(first, events: AsyncIterator):
    try:
        yield first
        async for event in events:
            yield event
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        coordinator = request.app.state.execution_coordinator
        body = await request.body()
        wire_request = OpenAIWireRequestMapper().parse(body, tuple(request.headers.raw))
        wire_plan = coordinator.plan_wire(InterfaceName.OPENAI, wire_request)
        if wire_plan is not None:
            if wire_plan.execution.request.projection.stream:
                return wire_stream_response((await coordinator.stream(wire_plan)).stream)
            return wire_response((await coordinator.complete(wire_plan)).response)
        payload = OpenAIChatCompletionRequest.model_validate_json(body)
        completion_request = OpenAIRequestMapper().map_request(payload)
        if payload.stream:
            events = (await coordinator.stream(coordinator.plan_canonical(InterfaceName.OPENAI, completion_request))).events
            try:
                first = await anext(events)
            except StopAsyncIteration:
                raise RuntimeError("provider stream ended before response start")
            return StreamingResponse(
                OpenAIStreamMapper().map_stream(_with_first(first, events), response_model=payload.model),
                media_type="text/event-stream",
            )
        response = (await coordinator.complete(coordinator.plan_canonical(InterfaceName.OPENAI, completion_request))).response
        return JSONResponse(OpenAIResponseMapper().map_response(response, response_model=payload.model))
    except Exception as error:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("OpenAI chat completion failed")
        status, body = OpenAIErrorMapper().map_error(error)
        return JSONResponse(body, status_code=status)
