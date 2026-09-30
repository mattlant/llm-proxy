"""Best-effort structured observation for wire-relayed OpenAI SSE."""
from __future__ import annotations

import json

from llm_proxy.observability.payload_trace import current_recorder

from .stream_chunk_parser import OpenAIStreamParser
from .stream_parser import SseDecoder, SseRecord


class OllamaWireStreamObserver:
    def __init__(self, provider_name: str, *, expose_thinking: bool = False) -> None:
        self._provider_name = provider_name
        self._decoder = SseDecoder()
        self._parser = OpenAIStreamParser(expose_thinking=expose_thinking)

    def observe(self, chunk: bytes) -> None:
        for record in self._decoder.feed(bytes(chunk)):
            self._record(record.data)

    def finish(self) -> None:
        for record in self._decoder.finish():
            self._record(record.data)
        self._parser.finish()

    def _record(self, data: str) -> None:
        payload = data if data == "[DONE]" else json.loads(data)
        recorder = current_recorder()
        if recorder is not None:
            recorder.record("provider_stream_record", payload, provider=self._provider_name, stream=True)
        self._parser.feed(SseRecord(None, data))
