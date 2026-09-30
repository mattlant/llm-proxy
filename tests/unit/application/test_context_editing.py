from __future__ import annotations

from llm_proxy.application.context_editing import apply_context_edits
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, ContextEditKind, ContextEditRequest, ContextManagementRequest, ExecutionControls


def test_clear_thinking_is_a_truthful_noop_for_local_suppressed_reasoning() -> None:
    request = CompletionRequest(
        "local-opus",
        (Message(MessageRole.USER, (TextContent("hello"),)),),
        controls=ExecutionControls(
            context_management=ContextManagementRequest((ContextEditRequest(ContextEditKind.CLEAR_THINKING, "all"),)),
        ),
    )

    application = apply_context_edits(request)

    assert application.request is request
    assert application.result.applied_edits == ()


def test_no_context_management_has_no_result() -> None:
    request = CompletionRequest("local-opus", (Message(MessageRole.USER, (TextContent("hello"),)),))

    assert apply_context_edits(request).result is None
