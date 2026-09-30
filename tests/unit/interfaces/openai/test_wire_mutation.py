import json

import pytest

from llm_proxy.application.wire_planning import WireMutationDecision
from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.interfaces.openai.wire_request import OpenAIWireRequestMapper
from llm_proxy.interfaces.openai.wire_mutation import OpenAIWireMutationProjector


def test_projector_maps_parameters_and_stream_usage() -> None:
    request = OpenAIWireRequestMapper().parse(b'{"model":"alias","stream":true,"messages":[]}', ()).wire_request
    result = OpenAIWireMutationProjector().project(request, WireMutationDecision("upstream", SamplingParameters(temperature=0.2, stop_sequences=("END",), extra={"seed": 2}), True, True))

    assert result.fallback_reason is None
    assert json.loads(result.patch.apply(request)) == {"model": "upstream", "stream": True, "messages": [], "temperature": 0.2, "stop": ["END"], "seed": 2, "stream_options": {"include_usage": True}}


def test_projector_preserves_noop_and_reports_non_object_stream_options() -> None:
    projector = OpenAIWireMutationProjector()
    request = OpenAIWireRequestMapper().parse(b'{"model":"upstream","messages":[]}', ()).wire_request
    assert projector.project(request, WireMutationDecision("upstream", SamplingParameters(), False, False)).patch.apply(request) == request.body
    invalid = OpenAIWireRequestMapper().parse(b'{"model":"alias","stream":true,"stream_options":false,"messages":[]}', ()).wire_request
    assert projector.project(invalid, WireMutationDecision("upstream", SamplingParameters(), True, True)).fallback_reason == "stream_options_not_object"


def test_projector_rejects_extra_parameter_collisions() -> None:
    request = OpenAIWireRequestMapper().parse(b'{"model":"alias","messages":[]}', ()).wire_request
    with pytest.raises(ValueError, match="collide"):
        OpenAIWireMutationProjector().project(request, WireMutationDecision("upstream", SamplingParameters(extra={"temperature": 1}), False, False))
