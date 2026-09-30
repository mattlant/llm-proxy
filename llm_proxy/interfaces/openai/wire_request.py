"""Raw OpenAI chat-completions envelope used before DTO validation."""
from __future__ import annotations

import json
from collections.abc import Mapping

from llm_proxy.application.wire_planning import InboundWirePlanningRequest
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.provider_extensions import WireProtocolCapability, WireRequest, WireRequestProjection

from .semantic_mapping import OpenAISemanticMapper
from .parameters import parameters_from_wire_payload

OPENAI_CHAT_COMPLETIONS_CAPABILITY = WireProtocolCapability("openai.chat-completions", "1", "/v1/chat/completions")


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


class OpenAIWireRequestMapper:
    def parse(self, body: bytes, headers: tuple[tuple[bytes, bytes], ...]) -> InboundWirePlanningRequest:
        try:
            payload = json.loads(body.decode("utf-8"), object_pairs_hook=_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("invalid OpenAI JSON request") from error
        if not isinstance(payload, Mapping):
            raise ValueError("OpenAI request must be a JSON object")
        model = payload.get("model")
        stream = payload.get("stream", False)
        if not isinstance(model, str) or not model.strip() or not isinstance(stream, bool):
            raise ValueError("OpenAI request has invalid model or stream")
        params = parameters_from_wire_payload(payload)
        semantic = OpenAISemanticMapper()
        messages, tool_names = semantic.map_wire_messages(payload.get("messages")), semantic.map_wire_tool_names(payload.get("tools"))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        wire_request = WireRequest(OPENAI_CHAT_COMPLETIONS_CAPABILITY, body, tuple(headers), payload, WireRequestProjection(model, stream, params, messages, tool_names, metadata))
        return InboundWirePlanningRequest(wire_request, InterfaceName.OPENAI, model, stream, params, messages, tool_names, metadata)
