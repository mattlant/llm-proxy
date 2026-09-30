from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

from llm_proxy.configuration.models import InterfaceName, StructuredOutputMode
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.domain.events import CompletionEvent, ResponseCompleted
from llm_proxy.domain.requests import CompletionRequest, ContextManagementResult, SamplingParameters
from llm_proxy.domain.parameters import EffectiveParameters, ParameterOverlay
from llm_proxy.domain.responses import CompletionResponse
from llm_proxy.observability.operational_logging import ExecutionMode, OperationalLogger, copy_trace_transaction_id, current_operational_request
from llm_proxy.provider_extensions import CompletionExecution, ProviderExecutionPolicy, SourceInterface, WireExecution
from llm_proxy.provider_extensions.wire import WireResponse, WireStream

from .context_editing import apply_context_edits
from .model_resolver import ResolvedModelExecution
from .policy_context import RequestPolicyContext
from .policy_engine import AppliedPolicy, PolicyApplication
from .provider_registry import ProviderGatewayRegistry
from .semantic_support import SemanticSupportResolver
from llm_proxy.domain.semantic_support import EffectiveSemanticHandling, SemanticHandlingOutcome
from .runtime import ActivationPolicy
from .wire_planning import InboundWirePlanningRequest, PrivateWirePlanningRegistry, WireCapabilityIdentity, WireMutationDecision
from .capabilities import CapabilityExecutor, CapabilityRequestContext
from llm_proxy.extensions.side_effect import SideEffectExecutionMode
from llm_proxy.observability.operational_logging import current_operational_request
from uuid import uuid4


def merge_sampling_parameters(defaults: SamplingParameters, requested: SamplingParameters) -> SamplingParameters:
    requested_effective = EffectiveParameters.from_legacy(requested)
    canonical = dict(requested_effective.canonical)
    if not requested.stop_sequences:
        canonical.pop("stop_sequences", None)
    return EffectiveParameters.from_legacy(defaults).apply(
        ParameterOverlay(canonical=canonical, compatibility=requested_effective.compatibility.values)
    ).to_legacy()


@dataclass(frozen=True, slots=True)
class PlanningContext:
    interface: InterfaceName
    requested_model: str
    stream: bool
    mode: ExecutionMode
    resolved: ResolvedModelExecution
    applied_policies: tuple[AppliedPolicy, ...]
    capability_context: CapabilityRequestContext
    invocations: tuple = ()
    semantic_handling: tuple[EffectiveSemanticHandling, ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalExecutionPlan:
    context: PlanningContext
    execution: CompletionExecution


@dataclass(frozen=True, slots=True)
class WireExecutionPlan:
    context: PlanningContext
    execution: WireExecution


@dataclass(frozen=True, slots=True)
class CanonicalCompletionResult:
    context: PlanningContext
    response: CompletionResponse


@dataclass(frozen=True, slots=True)
class WireCompletionResult:
    context: PlanningContext
    response: WireResponse


@dataclass(frozen=True, slots=True)
class CanonicalStreamResult:
    context: PlanningContext
    events: AsyncIterator[CompletionEvent]


@dataclass(frozen=True, slots=True)
class WireStreamResult:
    context: PlanningContext
    stream: WireStream


ExecutionPlan = CanonicalExecutionPlan | WireExecutionPlan


class ExecutionCoordinator:
    def __init__(self, configuration_store: ConfigurationStore, provider_gateways: ProviderGatewayRegistry, capability_executor: CapabilityExecutor, wire_planning_registry: PrivateWirePlanningRegistry, activation_policy: ActivationPolicy | None = None, semantic_support: SemanticSupportResolver | None = None) -> None:
        self._configuration_store = configuration_store
        self._provider_gateways = provider_gateways
        self._capability_executor = capability_executor
        self._wire_planning_registry = wire_planning_registry
        self._activation_policy = activation_policy or ActivationPolicy()
        self._semantic_support = semantic_support or SemanticSupportResolver()

    def plan_canonical(self, interface: InterfaceName, request: CompletionRequest) -> CanonicalExecutionPlan:
        if not isinstance(interface, InterfaceName) or not isinstance(request, CompletionRequest):
            raise TypeError("interface and request must be canonical values")
        context_application = apply_context_edits(request)
        request = context_application.request
        context = self._resolve_context(
            interface, request.model, request.stream,
            RequestPolicyContext(interface, request.model, request.model, request.messages, tuple(tool.name for tool in request.tools), request.metadata, request.controls),
            request.parameters, ExecutionMode.CANONICAL,
        )
        support_lookup = getattr(self._provider_gateways, "support", None)
        runtime_support = support_lookup(context.resolved.provider_instance_name) if callable(support_lookup) else None
        semantic_handling = self._semantic_support.resolve_request(request, context, runtime_support)
        rejected = next((item for item in semantic_handling if item.outcome is SemanticHandlingOutcome.REJECTED), None)
        if rejected is not None:
            raise InvalidCompletionRequest(f"selected execution does not support {rejected.semantic.value}")
        context = replace(context, semantic_handling=semantic_handling)
        self._record_route_decision(context, "canonical", semantic_handling)
        return CanonicalExecutionPlan(context, CompletionExecution(
            request, context.resolved.upstream_model, context.resolved.provider_instance_name, context.resolved.parameters, request.model,
            context_application.result, self._provider_policy(context.resolved), SourceInterface(interface.value),
        ))

    def plan_wire(self, interface: InterfaceName, request: InboundWirePlanningRequest) -> WireExecutionPlan | None:
        if not isinstance(interface, InterfaceName) or not isinstance(request, InboundWirePlanningRequest) or request.interface is not interface:
            return None
        capability_lookup = getattr(self._provider_gateways, "capabilities", None)
        wire_lookup = getattr(self._provider_gateways, "get_wire", None)
        if not callable(capability_lookup) or not callable(wire_lookup):
            self._record_wire_fallback("provider_wire_capability_unavailable")
            return None
        context = self._resolve_context(
            interface, request.requested_model, request.stream,
            RequestPolicyContext(interface, request.requested_model, request.requested_model, request.messages, request.tool_names, request.metadata, request.controls),
            request.parameters, ExecutionMode.PRESERVE,
        )
        capability = next((value for value in capability_lookup(context.resolved.provider_instance_name).wire_protocols if WireCapabilityIdentity.from_capability(value) == WireCapabilityIdentity.from_capability(request.wire_request.capability)), None)
        if capability is None:
            self._record_wire_fallback("resolved_provider_not_wire_capable")
            return None
        if context.resolved.compatibility.structured_output is not StructuredOutputMode.UNSUPPORTED:
            self._record_wire_fallback("structured_output_requires_canonical")
            return None
        projector = self._wire_planning_registry.get(request.wire_request.capability)
        if projector is None:
            self._record_wire_fallback("wire_projector_unavailable")
            return None
        if wire_lookup(context.resolved.provider_instance_name) is None:
            self._record_wire_fallback("wire_gateway_unavailable")
            return None
        projection = projector.project(request.wire_request, WireMutationDecision(context.resolved.upstream_model, context.resolved.parameters, request.stream, capability.requires_stream_usage))
        if projection.patch is None:
            self._record_wire_fallback(projection.fallback_reason or "wire_projection_unavailable")
            return None
        self._record_route_decision(context, WireCapabilityIdentity.from_capability(request.wire_request.capability).route_label)
        return WireExecutionPlan(context, WireExecution(request.wire_request, context.resolved.provider_instance_name, context.resolved.upstream_model, self._provider_policy(context.resolved), projection.patch))

    async def complete(self, plan: ExecutionPlan) -> CanonicalCompletionResult | WireCompletionResult:
        await self._execute_capabilities(plan.context)
        if isinstance(plan, CanonicalExecutionPlan):
            response = await self._provider_gateways.get(plan.execution.provider_instance_name).complete(plan.execution)
            if plan.execution.context_management is not None:
                response = replace(response, context_management=plan.execution.context_management)
            if isinstance(response, CompletionResponse):
                response = replace(response, semantic_handling=self._semantic_support.resolve_response(response, plan.context.interface))
            return CanonicalCompletionResult(plan.context, response)
        gateway = self._provider_gateways.get_wire(plan.execution.provider_instance_name)
        if gateway is None:
            raise RuntimeError("selected wire gateway is unavailable")
        return WireCompletionResult(plan.context, await gateway.complete_wire(plan.execution))

    async def stream(self, plan: ExecutionPlan) -> CanonicalStreamResult | WireStreamResult:
        await self._execute_capabilities(plan.context)
        if isinstance(plan, CanonicalExecutionPlan):
            events = self._provider_gateways.get(plan.execution.provider_instance_name).stream(plan.execution)
            return CanonicalStreamResult(plan.context, self._with_context_management(events, plan.execution.context_management, plan.context.interface))
        gateway = self._provider_gateways.get_wire(plan.execution.provider_instance_name)
        if gateway is None:
            raise RuntimeError("selected wire gateway is unavailable")
        return WireStreamResult(plan.context, await gateway.stream_wire(plan.execution))

    def _resolve_context(self, interface: InterfaceName, requested_model: str, stream: bool, policy_context: RequestPolicyContext, requested_parameters: SamplingParameters, mode: ExecutionMode) -> PlanningContext:
        self._configuration_store.reload()
        snapshot = self._configuration_store.snapshot
        assert_matches = getattr(self._provider_gateways, "assert_matches", None)
        if callable(assert_matches) and hasattr(snapshot, "config"):
            assert_matches(self._activation_policy.provider_identity(snapshot.config))
        base = snapshot.resolver.resolve(interface, requested_model)
        policy_context = replace(policy_context, canonical_model=base.canonical_model)
        application = self._apply_policy(snapshot.policy_engine, policy_context, replace(base, parameters=merge_sampling_parameters(base.parameters, requested_parameters)))
        operational = current_operational_request()
        request_id = operational.request_id if operational is not None else str(uuid4())
        trace = getattr(operational, "trace_transaction_id", None) if operational is not None else None
        capability_context = CapabilityRequestContext(request_id, trace, interface.value, SideEffectExecutionMode.CANONICAL if mode is ExecutionMode.CANONICAL else SideEffectExecutionMode.PRESERVE, requested_model, application.execution.canonical_model, application.execution.provider_instance_name, application.execution.upstream_model, stream)
        return PlanningContext(interface, requested_model, stream, mode, application.execution, application.applied_policies, capability_context, application.invocations)

    async def _execute_capabilities(self, context: PlanningContext) -> None:
        await self._capability_executor.execute(context.capability_context, context.invocations)

    @staticmethod
    def _provider_policy(resolved: ResolvedModelExecution) -> ProviderExecutionPolicy:
        return ProviderExecutionPolicy(resolved.compatibility.structured_output.value, resolved.compatibility.expose_thinking, resolved.compatibility.native_tools, resolved.compatibility.developer_role_mode)

    async def _with_context_management(self, events: AsyncIterator[CompletionEvent], result: ContextManagementResult | None, interface: InterfaceName) -> AsyncIterator[CompletionEvent]:
        try:
            async for event in events:
                if isinstance(event, ResponseCompleted):
                    updates = {"semantic_handling": self._semantic_support.resolve_response(event, interface)}
                    if result is not None:
                        updates["context_management"] = result
                    yield replace(event, **updates)
                else:
                    yield event
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    @staticmethod
    def _apply_policy(engine, context: RequestPolicyContext, base: ResolvedModelExecution) -> PolicyApplication:
        method = getattr(engine, "apply_with_diagnostics", None)
        return method(context, base) if callable(method) else PolicyApplication(engine.apply(context, base), ())

    @staticmethod
    def _record_wire_fallback(reason: str) -> None:
        context = current_operational_request()
        if context is not None:
            context.fallback_reason = reason

    @staticmethod
    def _record_route_decision(context: PlanningContext, route: str, semantic_handling: tuple[EffectiveSemanticHandling, ...] = ()) -> None:
        operational = current_operational_request()
        if operational is None:
            return
        copy_trace_transaction_id(operational)
        operational.interface, operational.requested_model, operational.upstream_model = context.interface.value, context.requested_model, context.resolved.upstream_model
        operational.provider, operational.mode, operational.stream, operational.route = context.resolved.provider_instance_name, context.mode, context.stream, route
        operational.policies = tuple((policy.name, policy.overridden_fields) for policy in context.applied_policies)
        operational.advisory_controls = tuple(item.semantic.value for item in semantic_handling if item.outcome is SemanticHandlingOutcome.ADVISORY)
        operational.semantic_handling = tuple(f"{item.semantic.value}:{item.outcome.value}:{item.scope.value}:{item.reason}" for item in semantic_handling)
        logger = OperationalLogger()
        logger.log_start(operational)
        logger.log_decision(operational)
