from types import MappingProxyType

import pytest

from llm_proxy.application.wire_planning import (
    InboundWirePlanningRequest,
    PrivateWirePlanningRegistry,
    WireCapabilityIdentity,
    WireMutationProjection,
)
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.interfaces.openai.wire_request import OpenAIWireRequestMapper
from llm_proxy.interfaces.openai.wire_mutation import OpenAIWireMutationProjector
from llm_proxy.provider_extensions import WirePatch, WireProtocolCapability


def test_identity_ignores_behavioral_flags_and_registry_is_immutable() -> None:
    capability = WireProtocolCapability("test", "1", "/test")
    changed_flags = WireProtocolCapability("test", "1", "/test", requires_stream_usage=True)
    projector = OpenAIWireMutationProjector()
    registry = PrivateWirePlanningRegistry({WireCapabilityIdentity.from_capability(capability): projector})

    assert WireCapabilityIdentity.from_capability(changed_flags).route_label == "test@1"
    assert registry.get(changed_flags) is projector
    assert isinstance(registry._projectors, MappingProxyType)


def test_planning_input_freezes_metadata_and_projection_requires_one_result() -> None:
    inbound = OpenAIWireRequestMapper().parse(b'{"model":"alias","messages":[],"metadata":{"tenant":{"id":"a"}}}', ())
    metadata = {"tenant": {"id": "a"}}
    planned = InboundWirePlanningRequest(inbound.wire_request, InterfaceName.OPENAI, "alias", False, SamplingParameters(), (), (), metadata)
    metadata["tenant"]["id"] = "b"

    assert planned.metadata["tenant"]["id"] == "a"
    with pytest.raises(ValueError):
        WireMutationProjection(None, None)
    with pytest.raises(ValueError):
        WireMutationProjection(WirePatch(), "fallback")


def test_registry_rejects_invalid_entries() -> None:
    with pytest.raises(TypeError):
        PrivateWirePlanningRegistry({"not-an-identity": OpenAIWireMutationProjector()})
