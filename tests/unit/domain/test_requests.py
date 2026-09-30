from __future__ import annotations

import pytest

from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import (
    AutomaticToolChoice,
    CompletionRequest,
    NamedToolChoice,
    NoToolChoice,
    RequiredToolChoice,
    SamplingParameters,
)
from llm_proxy.domain.tools import ToolDefinition


def message() -> Message:
    return Message(MessageRole.USER, (TextContent("hello"),))


@pytest.mark.parametrize(
    "choice",
    [AutomaticToolChoice(), RequiredToolChoice(), NoToolChoice(), NamedToolChoice("lookup")],
)
def test_completion_request_supports_all_tool_choice_modes(choice):
    request = CompletionRequest("model", (message(),), tool_choice=choice)

    assert request.tool_choice == choice


def test_sampling_parameters_preserve_explicit_zero_and_absent_values():
    parameters = SamplingParameters(temperature=0.0, top_k=0, max_tokens=0)

    assert parameters.temperature == 0.0
    assert parameters.top_k == 0
    assert parameters.max_tokens == 0
    assert parameters.top_p is None


def test_sampling_parameters_preserve_stop_sequences_and_provider_extensions():
    parameters = SamplingParameters(stop_sequences=("END", "STOP"), extra={"mirostat": 2, "seed": 7})

    assert parameters.stop_sequences == ("END", "STOP")
    assert parameters.extra == {"mirostat": 2, "seed": 7}


def test_completion_request_preserves_tools_and_metadata_without_mutable_aliases():
    metadata = {"trace": {"id": "123"}}
    tool = ToolDefinition("lookup", "Find data", {"type": "object"})
    request = CompletionRequest("model", (message(),), tools=(tool,), metadata=metadata)
    metadata["trace"]["id"] = "changed"

    assert request.tools == (tool,)
    assert request.metadata["trace"] == {"id": "123"}


@pytest.mark.parametrize("value", [-1, -0.1])
def test_negative_sampling_counts_are_rejected(value):
    with pytest.raises(ValueError):
        SamplingParameters(max_tokens=value)


def test_empty_model_and_named_tool_choice_are_rejected():
    with pytest.raises(ValueError):
        CompletionRequest("", (message(),))
    with pytest.raises(ValueError):
        NamedToolChoice("")
