from __future__ import annotations

import pytest

from llm_proxy.configuration.models import StructuredOutputMode
from llm_proxy.provider_extensions import CompletionExecution, ProviderExecutionPolicy
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, ExecutionControls, JsonSchemaOutputConstraint, SamplingParameters
from llm_proxy.providers.ollama.request_mapper import OllamaOpenAIRequestMapper


def execution(mode: StructuredOutputMode) -> CompletionExecution:
    schema = {"type": "object", "$defs": {"choice": {"enum": ["one", "two"]}}, "properties": {"value": {"$ref": "#/$defs/choice"}}, "required": ["value"]}
    request = CompletionRequest(
        "local-opus",
        (Message(MessageRole.USER, (TextContent("title"),)),),
        controls=ExecutionControls(output_constraint=JsonSchemaOutputConstraint(schema)),
    )
    return CompletionExecution(
        request=request,
        upstream_model="upstream",
        provider_instance_name="provider",
        parameters=SamplingParameters(),
        response_model=request.model,
        policy=ProviderExecutionPolicy(structured_output_mode=mode.value),
    )


def test_maps_openai_json_schema_without_losing_schema_constructs() -> None:
    mapped = OllamaOpenAIRequestMapper().map_request(execution(StructuredOutputMode.OPENAI_JSON_SCHEMA))

    assert mapped["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {"type": "object", "$defs": {"choice": {"enum": ["one", "two"]}}, "properties": {"value": {"$ref": "#/$defs/choice"}}, "required": ["value"]},
        },
    }
    assert "effort" not in mapped


def test_maps_ollama_format_when_explicitly_configured() -> None:
    mapped = OllamaOpenAIRequestMapper().map_request(execution(StructuredOutputMode.OLLAMA_FORMAT))

    assert mapped["format"]["$defs"]["choice"]["enum"] == ["one", "two"]


def test_rejects_constraint_when_capability_is_unsupported() -> None:
    with pytest.raises(ValueError, match="does not support"):
        OllamaOpenAIRequestMapper().map_request(execution(StructuredOutputMode.UNSUPPORTED))
