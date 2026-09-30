from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import get_type_hints

from llm_proxy.application.completion_gateway import CompletionGateway
from llm_proxy.domain.events import CompletionEvent
from llm_proxy.domain.responses import CompletionResponse
from llm_proxy.application.completion_execution import CompletionExecution


def test_completion_gateway_is_a_canonical_protocol() -> None:
    assert inspect.isclass(CompletionGateway)
    assert getattr(CompletionGateway, "_is_protocol", False)
    assert list(CompletionGateway.__annotations__) == []

    complete = CompletionGateway.complete
    stream = CompletionGateway.stream
    complete_annotations = get_type_hints(complete)
    stream_annotations = get_type_hints(stream)
    assert complete_annotations["execution"] is CompletionExecution
    assert complete_annotations["return"] is CompletionResponse
    assert stream_annotations["execution"] is CompletionExecution
    assert stream_annotations["return"] == AsyncIterator[CompletionEvent]
