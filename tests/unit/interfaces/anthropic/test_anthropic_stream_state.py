from __future__ import annotations

import pytest

from llm_proxy.interfaces.anthropic.stream_state import AnthropicStreamState


def test_tracks_stable_block_indexes_and_lifecycle() -> None:
    state = AnthropicStreamState()
    state.start_message("msg_1", "alias")
    text = state.start_block("text-1", "text")
    tool = state.start_block("tool-1", "tool_use", tool_call_id="call-1", tool_name="Read")
    state.append_delta("text-1", "hello")
    state.stop_block("text-1")
    state.stop_block("tool-1")
    state.complete_message()

    assert (text.index, tool.index) == (0, 1)
    assert state.completed is True


def test_rejects_invalid_stream_transitions() -> None:
    state = AnthropicStreamState()
    with pytest.raises(ValueError, match="has not started"):
        state.start_block("text", "text")
    state.start_message("msg", "model")
    state.start_block("text", "text")
    with pytest.raises(ValueError, match="already started"):
        state.start_block("text", "text")
    with pytest.raises(ValueError, match="open content"):
        state.complete_message()
    state.stop_block("text")
    with pytest.raises(ValueError, match="already stopped"):
        state.append_delta("text", "late")
