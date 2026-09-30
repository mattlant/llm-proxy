from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.configuration.models import InterfaceName

from .dto import OllamaChatRequest, OllamaGenerateRequest, OllamaShowRequest
from .error_mapper import OllamaErrorMapper
from .request_mapper import OllamaRequestMapper
from .response_mapper import OllamaResponseMapper
from .stream_mapper import OllamaStreamMapper

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


@router.post("/api/chat")
async def chat(payload: OllamaChatRequest, request: Request):
    try:
        coordinator = request.app.state.execution_coordinator
        completion_request = OllamaRequestMapper().map_chat(payload)
        if payload.stream:
            events = (await coordinator.stream(coordinator.plan_canonical(InterfaceName.OLLAMA, completion_request))).events
            try:
                first = await anext(events)
            except StopAsyncIteration:
                raise RuntimeError("provider stream ended before response start")
            return StreamingResponse(OllamaStreamMapper().map_chat(_with_first(first, events), response_model=payload.model), media_type="application/x-ndjson")
        response = (await coordinator.complete(coordinator.plan_canonical(InterfaceName.OLLAMA, completion_request))).response
        return JSONResponse(OllamaResponseMapper().map_chat(response, response_model=payload.model))
    except Exception as error:
        status, body = OllamaErrorMapper().map_error(error)
        return JSONResponse(body, status_code=status)


@router.post("/api/generate")
async def generate(payload: OllamaGenerateRequest, request: Request):
    try:
        coordinator = request.app.state.execution_coordinator
        completion_request = OllamaRequestMapper().map_generate(payload)
        if payload.stream:
            events = (await coordinator.stream(coordinator.plan_canonical(InterfaceName.OLLAMA, completion_request))).events
            try:
                first = await anext(events)
            except StopAsyncIteration:
                raise RuntimeError("provider stream ended before response start")
            return StreamingResponse(OllamaStreamMapper().map_generate(_with_first(first, events), response_model=payload.model), media_type="application/x-ndjson")
        response = (await coordinator.complete(coordinator.plan_canonical(InterfaceName.OLLAMA, completion_request))).response
        return JSONResponse(OllamaResponseMapper().map_generate(response, response_model=payload.model))
    except Exception as error:
        status, body = OllamaErrorMapper().map_error(error)
        return JSONResponse(body, status_code=status)


@router.post("/api/show")
async def show(payload: OllamaShowRequest, request: Request) -> dict[str, Any]:
    try:
        store = request.app.state.configuration_store
        store.reload()
        reference = store.snapshot.registry.resolve(InterfaceName.OLLAMA, payload.model)
        profile = reference.profile
        capabilities = ["completion"]
        if profile.compatibility.native_tools:
            capabilities.append("tools")
        return {
            "name": payload.model,
            "model": profile.name,
            "provider": profile.provider,
            "capabilities": capabilities,
            "parameters": {
                "temperature": profile.parameters.temperature,
                "top_p": profile.parameters.top_p,
                "top_k": profile.parameters.top_k,
                "min_p": profile.parameters.min_p,
                "repeat_penalty": profile.parameters.repeat_penalty,
                "repeat_last_n": profile.parameters.repeat_last_n,
                "num_predict": profile.parameters.max_tokens,
                "stop": list(profile.parameters.stop_sequences),
            },
        }
    except Exception as error:
        status, body = OllamaErrorMapper().map_error(error)
        return JSONResponse(body, status_code=status)
