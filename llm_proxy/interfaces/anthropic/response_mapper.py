from __future__ import annotations

from typing import Any

from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.responses import CompletionResponse
from llm_proxy.domain.responses import FinishReason
from llm_proxy.domain.semantic_support import SemanticHandlingOutcome, SemanticIdentity, SemanticSupportPhase


class AnthropicResponseMapper:
    @staticmethod
    def finish_reason_outcome(reason: FinishReason) -> SemanticHandlingOutcome:
        return SemanticHandlingOutcome.SUPPORTED

    @classmethod
    def validate_semantic_handling(cls, response: CompletionResponse) -> None:
        values = [item for item in response.semantic_handling if item.semantic is SemanticIdentity.FINISH_REASON and item.phase is SemanticSupportPhase.RESPONSE]
        if len(values) != 1 or values[0].outcome is not cls.finish_reason_outcome(response.finish_reason):
            raise ValueError("completion response finish-reason semantic handling is missing or inconsistent")

    def map_response(self, response: CompletionResponse, *, response_model: str) -> dict[str, Any]:
        self.validate_semantic_handling(response)
        return {
            "id": self.response_id(response.id),
            "type": "message",
            "role": "assistant",
            "model": response_model,
            "content": [self._map_block(block) for block in response.message.content if isinstance(block, (TextContent, ToolCallContent))],
            "stop_reason": response.finish_reason.value,
            "stop_sequence": response.stop_sequence,
            "usage": {
                "input_tokens": response.usage.input_tokens or 0,
                "output_tokens": response.usage.output_tokens or 0,
            },
            **({"context_management": self._map_context_management(response.context_management)} if response.context_management is not None else {}),
        }

    @staticmethod
    def response_id(value: str) -> str:
        return value if value.startswith("msg_") else f"msg_{value}"

    @staticmethod
    def _map_block(block: TextContent | ToolCallContent) -> dict[str, Any]:
        if isinstance(block, TextContent):
            return {"type": "text", "text": block.text}
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.arguments)}

    @staticmethod
    def _map_context_management(result) -> dict[str, Any]:
        return {"applied_edits": [{"type": edit.kind.value} for edit in result.applied_edits]}
