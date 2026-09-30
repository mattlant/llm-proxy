from __future__ import annotations

from dataclasses import dataclass, replace

from .policy_matching import (
    contains_casefold,
    equals_casefold,
    equals_sensitive,
    matches_all,
    matches_any,
    wildcard_casefold,
    wildcard_sensitive,
)
from llm_proxy.configuration.models import GatewayConfig, PolicyConfig, SamplingParametersOverride
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import MessageRole
from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.domain.parameters import CORE_PARAMETER_CATALOG, EffectiveParameters, ParameterOverlay

from .model_registry import ModelRegistry
from .model_resolver import ResolvedModelExecution
from .policy_context import RequestPolicyContext
from .capabilities import CapabilityInvocation


class PolicyEngine:
    def __init__(self, config: GatewayConfig, registry: ModelRegistry):
        self._config = config
        self._registry = registry

    def apply(
        self,
        context: RequestPolicyContext,
        base: ResolvedModelExecution,
    ) -> ResolvedModelExecution:
        return self.apply_with_diagnostics(context, base).execution

    def apply_with_diagnostics(
        self,
        context: RequestPolicyContext,
        base: ResolvedModelExecution,
    ) -> PolicyApplication:
        result = base
        applied: list[AppliedPolicy] = []
        invocations: list[CapabilityInvocation] = []
        for policy in self._config.policies:
            if not policy.enabled or not self._matches(policy, context):
                continue
            previous = result
            result = self._apply_actions(policy, context, result)
            fields = list(self._overridden_fields(previous, result))
            invocations.extend(CapabilityInvocation(policy.name, binding) for binding in policy.actions.capabilities)
            if policy.actions.capabilities: fields.append(f"capabilities(count={len(policy.actions.capabilities)})")
            applied.append(AppliedPolicy(policy.name, tuple(fields)))
        return PolicyApplication(result, tuple(applied), tuple(invocations))

    @staticmethod
    def _overridden_fields(before: ResolvedModelExecution, after: ResolvedModelExecution) -> tuple[str, ...]:
        fields: list[str] = []
        if before.upstream_model != after.upstream_model or before.canonical_model != after.canonical_model:
            fields.append("model")
        if before.provider_instance_name != after.provider_instance_name:
            fields.append("provider")
        before_effective = EffectiveParameters.from_legacy(before.parameters)
        after_effective = EffectiveParameters.from_legacy(after.parameters)
        for name in sorted(CORE_PARAMETER_CATALOG.names):
            if before_effective.canonical.get(name) != after_effective.canonical.get(name):
                fields.append(name)
        if before_effective.compatibility.values != after_effective.compatibility.values:
            fields.append(f"parameters.extra(count={len(after_effective.compatibility.values)})")
        return tuple(fields)

    def _matches(self, policy: PolicyConfig, context: RequestPolicyContext) -> bool:
        match = policy.match
        if match.interfaces and context.interface not in match.interfaces:
            return False
        model_names = (context.requested_model, context.canonical_model)
        if match.models and not matches_any(model_names, match.models, equals_casefold):
            return False
        if match.model_contains_any and not matches_any(model_names, match.model_contains_any, contains_casefold):
            return False
        if match.model_contains_all and not matches_all(model_names, match.model_contains_all, contains_casefold):
            return False
        if match.model_wildcard_any and not matches_any(model_names, match.model_wildcard_any, wildcard_casefold):
            return False
        if match.model_wildcard_all and not matches_all(model_names, match.model_wildcard_all, wildcard_casefold):
            return False

        visible_text = (self._visible_text(context),)
        if match.text_contains_any and not matches_any(visible_text, match.text_contains_any, contains_casefold):
            return False
        if match.text_contains_all and not matches_all(visible_text, match.text_contains_all, contains_casefold):
            return False
        if match.tools_any and not matches_any(context.tool_names, match.tools_any, equals_sensitive):
            return False
        if match.tools_all and not matches_all(context.tool_names, match.tools_all, equals_sensitive):
            return False
        if match.tool_wildcard_any and not matches_any(context.tool_names, match.tool_wildcard_any, wildcard_sensitive):
            return False
        if match.tool_wildcard_all and not matches_all(context.tool_names, match.tool_wildcard_all, wildcard_sensitive):
            return False
        return all(context.metadata.get(key) == value for key, value in match.metadata_equals.items())

    @staticmethod
    def _visible_text(context: RequestPolicyContext) -> str:
        parts: list[str] = []
        for message in context.messages:
            if message.role not in {MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT}:
                continue
            for block in message.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
        return "\n".join(parts)

    def _apply_actions(
        self,
        policy: PolicyConfig,
        context: RequestPolicyContext,
        current: ResolvedModelExecution,
    ) -> ResolvedModelExecution:
        actions = policy.actions
        result = current
        if actions.model is not None:
            replacement = self._registry.resolve(context.interface, actions.model)
            replacement_profile = replacement.profile
            result = replace(
                result,
                canonical_model=replacement.canonical_name,
                upstream_model=replacement_profile.upstream_model,
                provider_instance_name=replacement_profile.provider,
                parameters=replace(replacement_profile.parameters),
                compatibility=replacement_profile.compatibility,
            )
        if actions.provider is not None:
            provider = self._config.providers.get(actions.provider)
            if provider is None or not provider.enabled:
                raise ValueError(f"policy references unavailable provider '{actions.provider}'")
            result = replace(result, provider_instance_name=actions.provider)
        if actions.parameters is not None:
            result = replace(result, parameters=self._merge_parameters(result.parameters, actions.parameters))
        return result

    @staticmethod
    def _merge_parameters(
        current: SamplingParameters,
        override: SamplingParametersOverride,
    ) -> SamplingParameters:
        canonical = {
            name: getattr(override, name)
            for name in CORE_PARAMETER_CATALOG.names
            if hasattr(override, name) and getattr(override, name) is not None
        }
        compatibility = {} if override.extra is None else override.extra
        return EffectiveParameters.from_legacy(current).apply(ParameterOverlay(canonical=canonical, compatibility=compatibility)).to_legacy()


@dataclass(frozen=True, slots=True)
class AppliedPolicy:
    name: str
    overridden_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyApplication:
    execution: ResolvedModelExecution
    applied_policies: tuple[AppliedPolicy, ...]
    invocations: tuple[CapabilityInvocation, ...] = ()
