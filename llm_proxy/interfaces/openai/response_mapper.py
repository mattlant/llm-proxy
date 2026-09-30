from __future__ import annotations

import json
import time
from typing import Any

from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage
from llm_proxy.domain.semantic_support import EffectiveSemanticHandling, SemanticHandlingOutcome, SemanticIdentity, SemanticSupportPhase


class OpenAIResponseMapper:
    _finish_reasons = {
        FinishReason.END_TURN: "stop",
        FinishReason.STOP_SEQUENCE: "stop",
        FinishReason.MAX_TOKENS: "length",
        FinishReason.TOOL_USE: "tool_calls",
        FinishReason.REFUSAL: "content_filter",
        FinishReason.PAUSE_TURN: "stop",
    }

    @staticmethod
    def finish_reason_outcome(reason: FinishReason) -> SemanticHandlingOutcome:
        return SemanticHandlingOutcome.LOSSY if reason is FinishReason.PAUSE_TURN else SemanticHandlingOutcome.SUPPORTED

    @classmethod
    def validate_semantic_handling(cls, response: CompletionResponse) -> None:
        values = [item for item in response.semantic_handling if item.semantic is SemanticIdentity.FINISH_REASON and item.phase is SemanticSupportPhase.RESPONSE]
        if len(values) != 1 or values[0].outcome is not cls.finish_reason_outcome(response.finish_reason):
            raise ValueError("completion response finish-reason semantic handling is missing or inconsistent")

    def map_response(self, response: CompletionResponse, *, response_model: str) -> dict[str, Any]:
        self.validate_semantic_handling(response)
        message = self._map_message(response)
        result: dict[str, Any] = {
            "id": self.response_id(response.id),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response_model,
            "choices": [{"index": 0, "message": message, "finish_reason": self._finish_reasons[response.finish_reason]}],
        }
        usage = self.map_usage(response.usage)
        if usage is not None:
            result["usage"] = usage
        return result

    @staticmethod
    def response_id(value: str) -> str:
        return value if value.startswith("chatcmpl-") else f"chatcmpl-{value}"

    @staticmethod
    def map_usage(usage: Usage) -> dict[str, int | None] | None:
        if usage.input_tokens is None and usage.output_tokens is None:
            return None
        total = usage.input_tokens + usage.output_tokens if usage.input_tokens is not None and usage.output_tokens is not None else None
        return {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens, "total_tokens": total}

    @staticmethod
    def _map_message(response: CompletionResponse) -> dict[str, Any]:
        text = "".join(block.text for block in response.message.content if isinstance(block, TextContent))
        calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(dict(block.arguments), separators=(",", ":"))},
            }
            for block in response.message.content
            if isinstance(block, ToolCallContent)
        ]
        result: dict[str, Any] = {"role": "assistant", "content": text if text else None}
        reasoning = "".join(block.text for block in response.message.content if isinstance(block, ReasoningContent))
        if reasoning:
            result["reasoning_content"] = reasoning
        if calls:
            result["tool_calls"] = calls
        return result
