from __future__ import annotations

import pytest

from llm_proxy.domain.content import (
    ReasoningContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)


def test_mixed_content_preserves_order_and_reasoning_identity():
    blocks = (
        TextContent("visible before"),
        ToolCallContent("call-a", "read", {"path": "/a"}),
        ToolCallContent("call-b", "write", {"path": "/b"}),
        ReasoningContent("private reasoning"),
        TextContent("visible after"),
    )

    assert blocks[0].text == "visible before"
    assert [block.id for block in blocks[1:3]] == ["call-a", "call-b"]
    assert isinstance(blocks[3], ReasoningContent)
    assert isinstance(blocks[4], TextContent)


def test_tool_results_preserve_association_content_and_failure_state():
    success = ToolResultContent("call-a", (TextContent("ok"),))
    failure = ToolResultContent("call-b", (TextContent("permission denied"),), is_error=True)

    assert success.tool_call_id == "call-a"
    assert success.is_error is False
    assert failure.tool_call_id == "call-b"
    assert failure.content == (TextContent("permission denied"),)
    assert failure.is_error is True


def test_content_objects_are_frozen_and_mapping_inputs_are_defensively_copied():
    arguments = {"nested": {"value": 1}}
    call = ToolCallContent("call-a", "inspect", arguments)
    arguments["nested"]["value"] = 2

    assert call.arguments["nested"] == {"value": 1}
    with pytest.raises((AttributeError, TypeError)):
        call.id = "other"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["id", "name"])
def test_tool_call_rejects_empty_identity(field):
    values = {"id": "call-a", "name": "tool", "arguments": {}}
    values[field] = ""

    with pytest.raises(ValueError):
        ToolCallContent(**values)


def test_tool_result_rejects_empty_identity():
    with pytest.raises(ValueError):
        ToolResultContent("", (TextContent("result"),))
