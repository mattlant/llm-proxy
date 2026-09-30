import asyncio

import pytest

from llm_proxy.providers.ollama.wire_observer import OllamaWireStreamObserver


def test_observer_accepts_fragmented_usage_and_done() -> None:
    observer = OllamaWireStreamObserver("ollama")
    observer.observe(b'data: {"id":"r","model":"m","choices":[{"delta":{},"finish_reason":null}]}\n')
    observer.observe(b'\ndata: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\ndata: [DONE]\n\n')
    observer.finish()


def test_observer_error_can_be_isolated_by_relay() -> None:
    observer = OllamaWireStreamObserver("ollama")
    with pytest.raises(Exception):
        observer.observe(b"data: not-json\n\n")
