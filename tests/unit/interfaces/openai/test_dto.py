from __future__ import annotations

import pytest

from llm_proxy.interfaces.openai.dto import OpenAIChatCompletionRequest


def test_openai_chat_completion_request_accepts_supported_protocol_fields() -> None:
    request = OpenAIChatCompletionRequest.model_validate({
        "model": "alias",
        "messages": [{"role": "system", "content": "be concise"}, {"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 100,
        "stop": ["DONE"],
        "stream": True,
        "top_k": 40,
        "min_p": 0.1,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
    })

    assert request.model == "alias"
    assert request.stop == ["DONE"]


def test_openai_chat_completion_request_ignores_unknown_compatibility_fields() -> None:
    request = OpenAIChatCompletionRequest.model_validate({
        "model": "alias",
        "messages": [{"role": "user", "content": "hello"}],
        "stream_options": {"include_usage": True},
    })

    assert request.model == "alias"


def test_openai_chat_completion_request_accepts_developer_messages() -> None:
    request = OpenAIChatCompletionRequest.model_validate({
        "model": "alias",
        "messages": [{"role": "developer", "content": "be concise"}],
    })

    assert request.messages[0].role == "developer"


def test_openai_chat_completion_request_accepts_structured_user_text_content() -> None:
    request = OpenAIChatCompletionRequest.model_validate({
        "model": "alias",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": " world"},
        ]}],
    })

    assert len(request.messages[0].content) == 2


@pytest.mark.parametrize("content", [[], [{"type": "image_url", "image_url": {"url": "x"}}]])
def test_openai_chat_completion_request_rejects_unsupported_structured_user_content(content) -> None:
    with pytest.raises(ValueError):
        OpenAIChatCompletionRequest.model_validate({
            "model": "alias",
            "messages": [{"role": "user", "content": content}],
        })
