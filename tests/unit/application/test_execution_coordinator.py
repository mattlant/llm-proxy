from __future__ import annotations

from types import SimpleNamespace

from llm_proxy.application.execution_coordinator import ExecutionCoordinator, merge_sampling_parameters
from llm_proxy.application.model_resolver import ResolvedModelExecution
from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.application.wire_planning import PrivateWirePlanningRegistry, WireCapabilityIdentity
from llm_proxy.application.wire_planning import InboundWirePlanningRequest
from llm_proxy.configuration.models import InterfaceName, ModelCompatibility
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
from llm_proxy.interfaces.openai.wire_request import OpenAIWireRequestMapper
from llm_proxy.interfaces.openai.wire_mutation import OpenAIWireMutationProjector
from llm_proxy.provider_extensions import ProviderCapabilities, WireProtocolCapability
from llm_proxy.provider_extensions.wire import WireResponse
from llm_proxy.observability.operational_logging import begin_operational_request, bind_operational_request, reset_operational_request


class StubResolver:
    def __init__(self, resolved): self.resolved = resolved
    def resolve(self, interface, requested_model):
        assert interface is InterfaceName.OPENAI and requested_model == "alias"
        return self.resolved


class StubPolicyEngine:
    def apply(self, context, base): return base


class StubStore:
    def __init__(self, snapshot): self.snapshot, self.reloads = snapshot, 0
    def reload(self): self.reloads += 1


class Gateway:
    def __init__(self): self.calls = []
    async def complete(self, execution): self.calls.append(("complete", execution)); return "canonical"
    def stream(self, execution): self.calls.append(("stream", execution)); return _empty_stream()
    async def complete_wire(self, execution): self.calls.append(("complete_wire", execution)); return WireResponse(200, (), b"wire")
    async def stream_wire(self, execution): raise AssertionError("not needed")


async def _empty_stream():
    if False: yield None


class CapabilityExecutor:
    async def execute(self, context, invocations):
        return None


def coordinator(gateway=None, registry=None):
    defaults = SamplingParameters(temperature=0.7, top_p=0.9, extra={"num_ctx": 4096})
    resolved = ResolvedModelExecution("alias", "canonical", "upstream", InterfaceName.OPENAI, "provider", defaults, ModelCompatibility())
    store = StubStore(SimpleNamespace(resolver=StubResolver(resolved), policy_engine=StubPolicyEngine()))
    gateway = gateway or Gateway()
    capability = WireProtocolCapability("openai.chat-completions", "1", "/v1/chat/completions")
    registry = registry or PrivateWirePlanningRegistry({WireCapabilityIdentity.from_capability(capability): OpenAIWireMutationProjector()})
    return ExecutionCoordinator(store, ProviderGatewayRegistry({"provider": gateway}, capabilities={"provider": ProviderCapabilities(wire_protocols=(capability,))}), CapabilityExecutor(), registry), store, gateway


async def test_canonical_plan_is_inert_and_dispatches_once() -> None:
    subject, store, gateway = coordinator()
    plan = subject.plan_canonical(InterfaceName.OPENAI, CompletionRequest("alias", (Message(MessageRole.USER, (TextContent("hello"),)),), parameters=SamplingParameters(temperature=0.2, extra={"seed": 1})))
    assert gateway.calls == []
    assert store.reloads == 1
    assert plan.execution.parameters == SamplingParameters(temperature=0.2, top_p=0.9, extra={"num_ctx": 4096, "seed": 1})
    assert (await subject.complete(plan)).response == "canonical"
    assert [call[0] for call in gateway.calls] == ["complete"]


async def test_wire_plan_is_inert_and_dispatches_once() -> None:
    subject, store, gateway = coordinator()
    plan = subject.plan_wire(InterfaceName.OPENAI, OpenAIWireRequestMapper().parse(b'{"model":"alias","messages":[]}', ()))
    assert plan is not None and gateway.calls == [] and store.reloads == 1
    assert (await subject.complete(plan)).response.body == b"wire"
    assert [call[0] for call in gateway.calls] == ["complete_wire"]


def test_wire_policy_uses_private_semantics_not_public_projection_mirrors() -> None:
    subject, _, _ = coordinator()
    inbound = OpenAIWireRequestMapper().parse(b'{"model":"alias","messages":[{"role":"user","content":"private"}]}', ())
    divergent = InboundWirePlanningRequest(
        inbound.wire_request.__class__(inbound.wire_request.capability, inbound.wire_request.body, inbound.wire_request.headers, inbound.wire_request.payload, inbound.wire_request.projection.__class__("other", False, SamplingParameters(), (), (), {})),
        InterfaceName.OPENAI, "alias", False, SamplingParameters(), inbound.messages, inbound.tool_names, inbound.metadata,
    )

    plan = subject.plan_wire(InterfaceName.OPENAI, divergent)

    assert plan is not None
    assert plan.context.requested_model == "alias"


def test_missing_projector_falls_back_before_wire_gateway_lookup() -> None:
    subject, _, gateway = coordinator(registry=PrivateWirePlanningRegistry({}))
    operational = begin_operational_request("openai")
    token = bind_operational_request(operational)
    try:
        assert subject.plan_wire(InterfaceName.OPENAI, OpenAIWireRequestMapper().parse(b'{"model":"alias","messages":[]}', ())) is None
        assert operational.fallback_reason == "wire_projector_unavailable"
        assert gateway.calls == []
    finally:
        reset_operational_request(token)


async def test_dispatch_uses_planned_resolution_after_configuration_changes() -> None:
    subject, store, gateway = coordinator()
    plan = subject.plan_canonical(InterfaceName.OPENAI, CompletionRequest("alias", (Message(MessageRole.USER, (TextContent("hello"),)),)))
    store.snapshot.resolver.resolved = ResolvedModelExecution("alias", "canonical", "new-upstream", InterfaceName.OPENAI, "provider", SamplingParameters(), ModelCompatibility())
    await subject.complete(plan)
    assert store.reloads == 1
    assert gateway.calls[0][1].upstream_model == "upstream"


def test_merge_sampling_parameters_retains_default_stop() -> None:
    assert merge_sampling_parameters(SamplingParameters(stop_sequences=("default",)), SamplingParameters(temperature=0.2)).stop_sequences == ("default",)
