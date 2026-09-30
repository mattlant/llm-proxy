from __future__ import annotations

import pytest

from llm_proxy.infrastructure.http_client import HttpResponse
from llm_proxy.provider_extensions import ProviderListingError, ProviderListingFailureCategory
from llm_proxy.providers.ollama.client import OllamaProviderClient


class _Transport:
    def __init__(self, response: HttpResponse):
        self.response = response
        self.calls: list[tuple[object, ...]] = []

    async def get_raw(self, url, headers, timeout, **kwargs):
        self.calls.append((url, headers, timeout, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_tags_listing_normalizes_name_and_preserves_get_tracing():
    transport = _Transport(HttpResponse(200, {}, b'{"models":[{"name":"qwen"},{"model":"llama"}]}'))

    models = await OllamaProviderClient("http://provider.test/", "gpu", transport=transport).list_models()

    assert [item.upstream_model for item in models] == ["qwen", "llama"]
    assert transport.calls == [("http://provider.test/api/tags", (), None, {"trace_provider": "gpu"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, category",
    [
        (HttpResponse(503, {}, b"{}"), ProviderListingFailureCategory.UNAVAILABLE),
        (HttpResponse(200, {}, b'{"models":[{"name":"same"},{"model":"same"}]}'), ProviderListingFailureCategory.INVALID_RESPONSE),
        (HttpResponse(200, {}, b'{"models":"not-a-list"}'), ProviderListingFailureCategory.INVALID_RESPONSE),
    ],
)
async def test_tags_listing_maps_status_and_malformed_payloads_to_typed_errors(response, category):
    with pytest.raises(ProviderListingError) as error:
        await OllamaProviderClient("http://provider.test", "gpu", transport=_Transport(response)).list_models()

    assert error.value.category is category
