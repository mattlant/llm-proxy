from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from llm_proxy.application.model_listing import ModelListingService
from llm_proxy.configuration.models import InterfaceName, ModelAccessMode
from llm_proxy.provider_extensions import ProviderCapabilities, ProviderListedModel, ProviderListingError, ProviderListingFailureCategory


@dataclass
class _Store:
    snapshot: object
    def reload(self): pass


class _Gateways:
    def __init__(self, responses): self.responses, self.calls = responses, []
    @property
    def instance_names(self): return tuple(self.responses)
    def capabilities(self, name): return ProviderCapabilities(model_listing=True)
    async def list_models(self, name, timeout):
        self.calls.append((name, timeout)); result = self.responses[name]
        if isinstance(result, Exception): raise result
        return result


def _service(responses, *, enabled=("alpha", "beta")):
    providers = {name: SimpleNamespace(enabled=True, listing_enabled=name in enabled, extension_id=f"ext-{name}") for name in responses}
    config = SimpleNamespace(model_access=SimpleNamespace(mode=ModelAccessMode.PROVIDER_PASSTHROUGH, listing_timeout_seconds=7), providers=providers)
    registry = SimpleNamespace(list_models=lambda interface: ())
    return ModelListingService(_Store(SimpleNamespace(config=config, registry=registry)), _Gateways(responses))


@pytest.mark.asyncio
async def test_listing_omits_opted_out_provider_and_orders_successful_records():
    service = _service({"beta": (ProviderListedModel("z"),), "alpha": (ProviderListedModel("b"), ProviderListedModel("a"))}, enabled=("alpha",))
    listed = await service.list_models(InterfaceName.OPENAI)
    assert [item.identity for item in listed] == ["alpha::a", "alpha::b"]
    assert service._gateways.calls == [("alpha", 7)]


@pytest.mark.asyncio
async def test_listing_isolates_partial_and_all_provider_failures():
    error = ProviderListingError("beta", ProviderListingFailureCategory.TIMEOUT, "timed out")
    service = _service({"alpha": (ProviderListedModel("ok"),), "beta": error})
    assert [item.identity for item in await service.list_models(InterfaceName.OPENAI)] == ["alpha::ok"]
    all_failed = _service({"alpha": error, "beta": error})
    assert await all_failed.list_models(InterfaceName.OPENAI) == ()
