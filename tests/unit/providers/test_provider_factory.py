from __future__ import annotations

from llm_proxy.application.provider_factory import ProviderClientFactory
from llm_proxy.application.provider_registry import ProviderGatewayRegistry


class Gateway:
    pass


def test_factory_is_a_thin_registry_backed_lookup_adapter() -> None:
    gateway = Gateway()
    assert ProviderClientFactory(ProviderGatewayRegistry({"rtx-3090": gateway})).create("rtx-3090") is gateway


def test_factory_does_not_resolve_or_construct_models() -> None:
    factory = ProviderClientFactory(ProviderGatewayRegistry({}))
    assert not hasattr(factory, "resolver")
