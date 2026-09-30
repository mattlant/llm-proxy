from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llm_proxy.application.model_registry import ModelRegistry
from llm_proxy.application.model_resolver import ModelResolver
from llm_proxy.application.policy_engine import PolicyEngine
from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.application.runtime import ActivationPolicy, RuntimeDependencies

from .loader import ConfigurationError, ConfigurationLoader
from .models import GatewayConfig
from .validator import ConfigurationValidator

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    config: GatewayConfig
    registry: ModelRegistry
    resolver: ModelResolver
    policy_engine: PolicyEngine
    loaded_at: float
    source_path: Path
    source_mtime: float | None = None


class ConfigurationStore:
    def __init__(
        self,
        path: Path,
        loader: ConfigurationLoader,
        reload_interval_seconds: float,
        clock: Callable[[], float] = time.time,
        *,
        runtime_dependencies: RuntimeDependencies | None = None,
        activation_policy: ActivationPolicy | None = None,
    ):
        if reload_interval_seconds <= 0:
            raise ValueError("reload_interval_seconds must be positive")
        self.path = path
        self.loader = loader
        self.reload_interval_seconds = reload_interval_seconds
        self._clock = clock
        self._snapshot: ConfigurationSnapshot | None = None
        self._last_mtime: float | None = None
        self._last_check = float("-inf")
        self._last_rejected_reload: str | None = None
        self._runtime_dependencies = runtime_dependencies or RuntimeDependencies()
        self._activation_policy = activation_policy or ActivationPolicy()
        self.reload(force=True)

    @classmethod
    def from_initial_config(cls, path, loader, reload_interval_seconds, config, source_mtime, *, runtime_dependencies=None, activation_policy=None, clock=time.time):
        if reload_interval_seconds <= 0:
            raise ValueError("reload_interval_seconds must be positive")
        store = cls.__new__(cls)
        store.path, store.loader, store.reload_interval_seconds, store._clock = path, loader, reload_interval_seconds, clock
        store._snapshot = None
        store._last_mtime = source_mtime
        store._last_check = float("-inf")
        store._last_rejected_reload = None
        store._runtime_dependencies = runtime_dependencies or RuntimeDependencies()
        store._activation_policy = activation_policy or ActivationPolicy()
        store._snapshot = store.build_candidate(config, source_mtime=source_mtime)
        return store

    @property
    def snapshot(self) -> ConfigurationSnapshot:
        if self._snapshot is None:
            raise RuntimeError("configuration store has no active snapshot")
        return self._snapshot

    @property
    def last_rejected_reload(self) -> str | None:
        return self._last_rejected_reload

    @property
    def runtime_dependencies(self) -> RuntimeDependencies:
        return self._runtime_dependencies

    def build_candidate(self, config: GatewayConfig, source_mtime: float | None = None) -> ConfigurationSnapshot:
        ConfigurationValidator().validate(config)
        self._runtime_dependencies.validate_capabilities(config)
        registry = ModelRegistry(config)
        return ConfigurationSnapshot(
            config=config,
            registry=registry,
            resolver=ModelResolver(registry, config, self._runtime_dependencies.provider_gateways),
            policy_engine=PolicyEngine(config, registry),
            loaded_at=self._clock(),
            source_path=self.path,
            source_mtime=source_mtime,
        )

    def activate_candidate(self, snapshot: ConfigurationSnapshot) -> None:
        decision = self._activation_policy.classify(self.snapshot.config, snapshot.config)
        if decision.restart_required:
            raise ValueError(decision.message)
        self._snapshot = snapshot
        try:
            self._last_mtime = self.path.stat().st_mtime
        except OSError:
            pass

    def reload(self, force: bool = False) -> bool:
        now = self._clock()
        is_reload = self._snapshot is not None
        if not force and now - self._last_check < self.reload_interval_seconds:
            return False
        self._last_check = now

        try:
            mtime, config = self._load_stable()
            if not force and self._last_mtime == mtime:
                return False
            snapshot = self.build_candidate(config, source_mtime=mtime)
            if self._snapshot is not None and self._activation_policy.classify(self._snapshot.config, snapshot.config).restart_required:
                self._last_rejected_reload = self._activation_policy.classify(self._snapshot.config, snapshot.config).message
                return False
        except (OSError, ConfigurationError, ValueError, TypeError) as error:
            if self._snapshot is None:
                raise
            log.warning("Retaining previous configuration snapshot after reload failure: path=%s error=%s", self.path, error)
            self._last_rejected_reload = type(error).__name__
            return False

        self._snapshot = snapshot
        self._last_mtime = mtime
        self._last_rejected_reload = None
        if is_reload:
            log.info("Configuration reloaded successfully: path=%s", self.path)
        return True

    def _load_stable(self):
        for _ in range(3):
            before = self.path.stat().st_mtime
            config = self.loader.load_path(self.path)
            after = self.path.stat().st_mtime
            if before == after:
                return after, config
        raise ConfigurationError("$", "configuration source changed while loading")
