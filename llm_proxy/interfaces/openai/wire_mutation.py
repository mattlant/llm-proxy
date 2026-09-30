from __future__ import annotations

from llm_proxy.application.wire_planning import WireMutationDecision, WireMutationProjection
from llm_proxy.provider_extensions import WirePatch, WirePatchOperation, WireRequest
from .parameters import mutation_operations


class OpenAIWireMutationProjector:
    def project(self, request: WireRequest, decision: WireMutationDecision) -> WireMutationProjection:
        payload = request.payload
        operations = [WirePatchOperation("/model", decision.upstream_model)] if payload.get("model") != decision.upstream_model else []
        operations.extend(mutation_operations(payload, decision.parameters))
        if decision.stream and decision.requires_stream_usage:
            stream_options = payload.get("stream_options")
            if "stream_options" in payload and not isinstance(stream_options, dict):
                return WireMutationProjection(None, "stream_options_not_object")
            if not isinstance(stream_options, dict) or stream_options.get("include_usage") is not True:
                operations.append(WirePatchOperation("/stream_options/include_usage", True, create_missing_parent_object="stream_options" not in payload))
        return WireMutationProjection(WirePatch(tuple(operations)), None)
