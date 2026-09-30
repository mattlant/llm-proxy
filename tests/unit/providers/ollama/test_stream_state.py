from __future__ import annotations

import pytest

from llm_proxy.domain.errors import ProviderProtocolError
from llm_proxy.providers.ollama.stream_state import ToolCallStreamState, ToolCallStreamStateStore


def test_single_tool_call_accumulates_fragments_and_starts_when_ready() -> None:
    store = ToolCallStreamStateStore("response-1")
    store.append_id(0, "call_")
    store.append_name(0, "Re")
    store.append_arguments(0, '{"path":')
    state = store.append_id(0, "1")
    store.append_name(0, "ad")

    assert state.started is False
    assert state.tool_call_id == "call_1"
    assert state.name == "Read"
    assert state.arguments == '{"path":'
    store.start_if_ready(0)
    assert state.started is True
    assert store.ids == {"call_1": 0}


def test_parallel_calls_and_out_of_order_indices_are_independent() -> None:
    store = ToolCallStreamStateStore("response-1")
    store.append_name(2, "Write")
    store.append_arguments(2, "{}")
    store.append_id(2, "call-2")
    store.append_id(0, "call-0")
    store.append_name(0, "Read")
    store.append_arguments(0, '{"path":"README.md"}')

    first = store.start_if_ready(0)
    second = store.start_if_ready(2)

    assert first.block_id == "tool_0"
    assert first.name == "Read"
    assert first.arguments == '{"path":"README.md"}'
    assert second.block_id == "tool_2"
    assert second.name == "Write"
    assert second.arguments == "{}"


def test_arguments_before_identity_are_buffered() -> None:
    store = ToolCallStreamStateStore("response-1")
    store.append_arguments(1, "{")
    store.append_arguments(1, '"path":"README.md"}')
    store.append_name(1, "Read")

    state = store.start_if_ready(1)

    assert state.arguments == '{"path":"README.md"}'


def test_missing_id_gets_deterministic_fallback() -> None:
    first_store = ToolCallStreamStateStore("response-1")
    second_store = ToolCallStreamStateStore("response-1")
    first_store.append_name(3, "Read")
    second_store.append_name(3, "Read")
    first = first_store.start_if_ready(3)
    second = second_store.start_if_ready(3)

    assert first.tool_call_id == second.tool_call_id
    assert first.tool_call_id is not None
    assert first.tool_call_id.startswith("call_")


def test_duplicate_index_identity_conflicts_are_rejected() -> None:
    store = ToolCallStreamStateStore("response-1")
    store.append_id(0, "call-0")
    store.append_name(0, "Read")
    store.start_if_ready(0)

    with pytest.raises(ProviderProtocolError, match="ID changed"):
        store.append_id(0, "other")


def test_name_changing_after_start_is_rejected() -> None:
    store = ToolCallStreamStateStore("response-1")
    store.append_id(0, "call-0")
    store.append_name(0, "Read")
    store.start_if_ready(0)

    with pytest.raises(ProviderProtocolError, match="name changed"):
        store.append_name(0, "Write")


def test_duplicate_tool_id_across_indices_is_rejected() -> None:
    store = ToolCallStreamStateStore("response-1")
    store.append_id(0, "same")
    store.append_name(0, "Read")
    store.start_if_ready(0)
    store.append_name(1, "Write")
    store.append_id(1, "same")

    with pytest.raises(ProviderProtocolError, match="shared"):
        store.start_if_ready(1)


def test_completed_state_rejects_late_fragments() -> None:
    state = ToolCallStreamState(0, "tool_0", ["call"], ["Read"])
    state.mark_started()
    state.mark_completed()

    with pytest.raises(ProviderProtocolError, match="after tool call completion"):
        state.append_arguments("late")
