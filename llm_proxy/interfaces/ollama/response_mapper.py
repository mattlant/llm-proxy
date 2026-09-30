from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage
from llm_proxy.domain.semantic_support import SemanticHandlingOutcome, SemanticIdentity, SemanticSupportPhase


class OllamaResponseMapper:
    _done_reasons = {
        FinishReason.END_TURN: "stop",
        FinishReason.STOP_SEQUENCE: "stop",
        FinishReason.MAX_TOKENS: "length",
        FinishReason.TOOL_USE: "tool_calls",
        FinishReason.REFUSAL: "stop",
        FinishReason.PAUSE_TURN: "stop",
    }

    @staticmethod
    def finish_reason_outcome(reason: FinishReason) -> SemanticHandlingOutcome:
        return SemanticHandlingOutcome.LOSSY if reason in {FinishReason.PAUSE_TURN, FinishReason.REFUSAL} else SemanticHandlingOutcome.SUPPORTED

    @classmethod
    def validate_semantic_handling(cls, response: CompletionResponse) -> None:
        values = [item for item in response.semantic_handling if item.semantic is SemanticIdentity.FINISH_REASON and item.phase is SemanticSupportPhase.RESPONSE]
        if len(values) != 1 or values[0].outcome is not cls.finish_reason_outcome(response.finish_reason):
            raise ValueError("completion response finish-reason semantic handling is missing or inconsistent")

    def map_chat(self, response: CompletionResponse, *, response_model: str) -> dict[str, Any]:
        self.validate_semantic_handling(response)
        result: dict[str, Any] = {
            "model": response_model,
            "created_at": self.created_at(),
            "message": self.message(response),
            "done": True,
            "done_reason": self._done_reasons[response.finish_reason],
        }
        result.update(self.usage(response.usage))
        return result

    def map_generate(self, response: CompletionResponse, *, response_model: str) -> dict[str, Any]:
        self.validate_semantic_handling(response)
        result: dict[str, Any] = {
            "model": response_model,
            "created_at": self.created_at(),
            "response": self.text(response),
            "done": True,
            "done_reason": self._done_reasons[response.finish_reason],
        }
        result.update(self.usage(response.usage))
        return result

    @staticmethod
    def created_at() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def text(response: CompletionResponse) -> str:
        return "".join(block.text for block in response.message.content if isinstance(block, TextContent))

    def message(self, response: CompletionResponse) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.text(response)}
        calls = [
            {"function": {"name": block.name, "arguments": dict(block.arguments)}}
            for block in response.message.content
            if isinstance(block, ToolCallContent)
        ]
        if calls:
            message["tool_calls"] = calls
        return message

    @staticmethod
    def usage(usage: Usage) -> dict[str, int]:
        result: dict[str, int] = {}
        if usage.input_tokens is not None:
            result["prompt_eval_count"] = usage.input_tokens
        if usage.output_tokens is not None:
            result["eval_count"] = usage.output_tokens
        return result
