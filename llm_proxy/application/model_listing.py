from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from llm_proxy.configuration.models import InterfaceName, ModelAccessMode
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.provider_extensions import ProviderListingError

from .provider_registry import ProviderGatewayRegistry

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ListedModel:
    identity: str
    upstream_model: str
    provider_instance_name: str


class ModelListingService:
    def __init__(self, store: ConfigurationStore, gateways: ProviderGatewayRegistry, logger: logging.Logger = log) -> None:
        self._store, self._gateways, self._logger = store, gateways, logger

    async def list_models(self, interface: InterfaceName) -> tuple[ListedModel, ...]:
        self._store.reload()
        snapshot = self._store.snapshot
        profiles = tuple(ListedModel(item.name, item.upstream_model, item.provider) for item in snapshot.registry.list_models(interface))
        if snapshot.config.model_access.mode is not ModelAccessMode.PROVIDER_PASSTHROUGH:
            return profiles
        eligible = [name for name, provider in snapshot.config.providers.items() if provider.enabled and provider.listing_enabled and name in self._gateways.instance_names and self._gateways.capabilities(name).model_listing]
        async def listed(name: str):
            started = time.monotonic()
            try:
                records = await self._gateways.list_models(name, snapshot.config.model_access.listing_timeout_seconds)
                extension = snapshot.config.providers[name].extension_id
                self._logger.info("provider_listing provider=%s extension=%s source=model_listing outcome=success duration_ms=%d count=%d", name, extension, (time.monotonic() - started) * 1000, len(records))
                return name, records
            except ProviderListingError as error:
                extension = snapshot.config.providers[name].extension_id
                self._logger.warning("provider_listing provider=%s extension=%s source=model_listing outcome=%s duration_ms=%d count=0", name, extension, error.category.value, (time.monotonic() - started) * 1000)
                return name, ()
        tasks = [asyncio.create_task(listed(name)) for name in eligible]
        try:
            values = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        discovered = [ListedModel(f"{name}::{item.upstream_model}", f"{name}::{item.upstream_model}", name) for name, records in values for item in records]
        discovered.sort(key=lambda item: (item.provider_instance_name.casefold(), item.identity.split("::", 1)[1]))
        return profiles + tuple(discovered)
