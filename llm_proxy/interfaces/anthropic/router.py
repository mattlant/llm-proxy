from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.configuration.models import InterfaceName
from llm_proxy.application.errors import InterfaceDisabledError

from .authentication import AnthropicAuthenticationError, AnthropicAuthenticator
from .dto import AnthropicMessagesRequest
from .error_mapper import AnthropicErrorMapper
from .headers import extract_anthropic_headers
from .request_mapper import AnthropicRequestMapper
from .response_mapper import AnthropicResponseMapper
from .stream_mapper import AnthropicStreamMapper

router = APIRouter()


async def _with_first(first, events: AsyncIterator):
    try:
        yield first
        async for event in events:
            yield event
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()


@router.post("/v1/messages")
async def messages(payload: AnthropicMessagesRequest, request: Request):
    try:
        headers = extract_anthropic_headers(request.headers)
        store = request.app.state.configuration_store
        store.reload()
        interface = store.snapshot.config.interfaces.get(InterfaceName.ANTHROPIC)
        if interface is None or not interface.enabled:
            raise InterfaceDisabledError(InterfaceName.ANTHROPIC.value, payload.model)
        if interface.authentication is None:
            raise AnthropicAuthenticationError("Invalid authentication credentials")
        AnthropicAuthenticator().authenticate(headers, interface.authentication)
        context_betas = tuple(beta for beta in headers.beta if beta.startswith("context-management-"))
        if payload.context_management is not None and context_betas and "context-management-2025-06-27" not in context_betas:
            raise ValueError("unsupported context-management beta")
        completion_request = AnthropicRequestMapper(expose_thinking=interface.expose_thinking).map_request(payload)
        coordinator = request.app.state.execution_coordinator
        if payload.stream:
            events = (await coordinator.stream(coordinator.plan_canonical(InterfaceName.ANTHROPIC, completion_request))).events
            try:
                first = await anext(events)
            except StopAsyncIteration:
                raise RuntimeError("provider stream ended before response start")
            return StreamingResponse(
                AnthropicStreamMapper().map_stream(_with_first(first, events), response_model=payload.model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        response = (await coordinator.complete(coordinator.plan_canonical(InterfaceName.ANTHROPIC, completion_request))).response
        return JSONResponse(AnthropicResponseMapper().map_response(response, response_model=payload.model))
    except Exception as error:
        status, body = AnthropicErrorMapper().map_error(error)
        return JSONResponse(body, status_code=status)
