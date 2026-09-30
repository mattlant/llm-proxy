from __future__ import annotations

import logging

log = logging.getLogger(__name__)

class ApplicationLifecycle:
    def __init__(self) -> None:
        self._resources, self._closed = [], False
    def add(self, resource) -> None: self._resources.append(resource)
    async def aclose(self) -> None:
        if self._closed: return
        self._closed = True
        for resource in reversed(self._resources):
            try:
                close = getattr(resource, "aclose", None)
                if close is not None: await close()
            except Exception:
                log.warning("Lifecycle cleanup failed: type=%s", type(resource).__name__)
