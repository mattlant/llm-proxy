from __future__ import annotations

import threading
from typing import Any, Mapping

from llm_proxy.configuration.atomic_writer import AtomicConfigurationWriter
from llm_proxy.configuration.loader import ConfigurationError, ConfigurationLoader
from llm_proxy.configuration.revision import configuration_mapping, configuration_revision, serialize_configuration
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.application.runtime import ActivationPolicy

from .contracts import CandidateResult
from .errors import ActivationFailed, ConfigurationStateConflict, ConfigurationStateIndeterminate, InvalidConfiguration, PersistenceFailed, RevisionConflict


class ConfigurationManagementService:
    def __init__(self, store: ConfigurationStore, writer: AtomicConfigurationWriter | None = None, activation_policy: ActivationPolicy | None = None) -> None:
        self._store = store
        self._writer = writer or AtomicConfigurationWriter()
        self._lock = threading.Lock()
        self._activation_policy = activation_policy or ActivationPolicy()

    @property
    def active_revision(self) -> str:
        return configuration_revision(self._store.snapshot.config)

    @property
    def persisted_revision(self) -> str:
        return configuration_revision(self._load_persisted())

    def validate(self, document: Mapping[str, Any]) -> CandidateResult:
        try:
            candidate = self._candidate(document)
        except (ConfigurationError, ValueError, TypeError) as error:
            return CandidateResult(False, None, False, (str(error),))
        return CandidateResult(True, configuration_revision(candidate.config), self._restart_required(candidate.config))

    def policy_snapshot(self) -> dict[str, Any]:
        with self._lock:
            persisted = self._load_persisted()
            if configuration_revision(persisted) != self.active_revision:
                raise ConfigurationStateConflict("active and persisted configuration differ; restore or restart before editing policies")
            return {"revision": configuration_revision(persisted), "policies": configuration_mapping(persisted)["policies"]}

    def validate_policies(self, policies: list[dict[str, Any]]) -> CandidateResult:
        try:
            candidate = self._policy_candidate(policies)
        except (ConfigurationError, ValueError, TypeError) as error:
            return CandidateResult(False, None, False, (str(error),))
        return CandidateResult(True, configuration_revision(candidate.config), False)

    def apply_policies(self, policies: list[dict[str, Any]], expected_revision: str) -> dict[str, Any]:
        with self._lock:
            persisted = self._load_persisted()
            persisted_revision = configuration_revision(persisted)
            if expected_revision != persisted_revision:
                raise RevisionConflict("configuration revision does not match")
            if persisted_revision != self.active_revision:
                raise ConfigurationStateConflict("active and persisted configuration differ; restore or restart before applying policies")
            try:
                candidate = self._policy_candidate(policies)
            except (ConfigurationError, ValueError, TypeError) as error:
                raise InvalidConfiguration(str(error)) from error
            candidate_revision = self._apply_reload_safe_candidate(candidate, persisted, persisted_revision, "policy")
            return {"active_revision": candidate_revision, "persisted_revision": candidate_revision, "restart_required": False, "policies": configuration_mapping(candidate.config)["policies"]}

    def model_profiles_snapshot(self) -> dict[str, Any]:
        with self._lock:
            persisted = self._load_persisted()
            if configuration_revision(persisted) != self.active_revision:
                raise ConfigurationStateConflict("active and persisted configuration differ; restore or restart before editing model profiles")
            return {"revision": configuration_revision(persisted), "models": configuration_mapping(persisted)["models"]}

    def validate_model_profiles(self, models: Mapping[str, Any]) -> CandidateResult:
        try:
            candidate = self._model_profiles_candidate(models)
        except (ConfigurationError, ValueError, TypeError) as error:
            return CandidateResult(False, None, False, (str(error),))
        return CandidateResult(True, configuration_revision(candidate.config), False)

    def apply_model_profiles(self, models: Mapping[str, Any], expected_revision: str) -> dict[str, Any]:
        with self._lock:
            persisted = self._load_persisted()
            persisted_revision = configuration_revision(persisted)
            if expected_revision != persisted_revision:
                raise RevisionConflict("configuration revision does not match")
            if persisted_revision != self.active_revision:
                raise ConfigurationStateConflict("active and persisted configuration differ; restore or restart before applying model profiles")
            try:
                candidate = self._model_profiles_candidate(models)
            except (ConfigurationError, ValueError, TypeError) as error:
                raise InvalidConfiguration(str(error)) from error
            candidate_revision = self._apply_reload_safe_candidate(candidate, persisted, persisted_revision, "model profile")
            return {"active_revision": candidate_revision, "persisted_revision": candidate_revision, "activation_class": "reload_safe", "models": configuration_mapping(candidate.config)["models"]}

    def update(self, document: Mapping[str, Any], expected_revision: str) -> dict[str, Any]:
        with self._lock:
            persisted = self._load_persisted()
            if expected_revision != configuration_revision(persisted):
                raise RevisionConflict("configuration revision does not match")
            try:
                candidate = self._candidate(document)
            except (ConfigurationError, ValueError, TypeError) as error:
                raise InvalidConfiguration(str(error)) from error
            revision = configuration_revision(candidate.config)
            try:
                self._writer.write(self._store.path, serialize_configuration(candidate.config))
            except OSError as error:
                raise PersistenceFailed("unable to persist configuration") from error
            restart_required = self._restart_required(candidate.config)
            if not restart_required:
                try:
                    self._store.activate_candidate(candidate)
                except Exception as error:
                    raise ActivationFailed("configuration persisted but activation failed") from error
            return self._result(revision, restart_required)

    def reload(self, expected_revision: str) -> dict[str, Any]:
        with self._lock:
            persisted = self._load_persisted()
            revision = configuration_revision(persisted)
            if expected_revision != revision:
                raise RevisionConflict("configuration revision does not match")
            candidate = self._candidate_from_config(persisted)
            restart_required = self._restart_required(candidate.config)
            if not restart_required:
                try:
                    self._store.activate_candidate(candidate)
                except Exception as error:
                    raise ActivationFailed("configuration activation failed") from error
            return self._result(revision, restart_required)

    def _load_persisted(self):
        try:
            return self._store.loader.load_path(self._store.path)
        except (ConfigurationError, OSError) as error:
            raise InvalidConfiguration("persisted configuration is invalid") from error

    def _candidate(self, document: Mapping[str, Any]):
        return self._candidate_from_config(self._store.loader.load_mapping(document))

    def _policy_candidate(self, policies: list[dict[str, Any]]):
        document = configuration_mapping(self._load_persisted())
        document["policies"] = policies
        return self._candidate(document)

    def _model_profiles_candidate(self, models: Mapping[str, Any]):
        if not isinstance(models, Mapping):
            raise ValueError("models are required")
        document = configuration_mapping(self._load_persisted())
        document["models"] = dict(models)
        for name, profile in document["models"].items():
            if not isinstance(profile, Mapping) or profile.get("name", name) != name:
                raise ValueError(f"models.{name}.name must match its stable profile key")
        return self._candidate(document)

    def _apply_reload_safe_candidate(self, candidate, persisted, persisted_revision: str, domain: str) -> str:
        previous = self._store.snapshot
        candidate_revision = configuration_revision(candidate.config)
        before = serialize_configuration(persisted)
        try:
            self._writer.write(self._store.path, serialize_configuration(candidate.config))
        except OSError as error:
            raise PersistenceFailed("unable to persist configuration") from error
        try:
            self._store.activate_candidate(candidate)
            if self.persisted_revision != candidate_revision or self.active_revision != candidate_revision:
                raise RuntimeError("candidate confirmation failed")
        except Exception as error:
            try:
                self._writer.write(self._store.path, before)
                self._store.activate_candidate(previous)
                if self.persisted_revision != persisted_revision or self.active_revision != persisted_revision:
                    raise RuntimeError("prior state confirmation failed")
            except Exception as rollback_error:
                raise ConfigurationStateIndeterminate(f"{domain} activation failed and prior state could not be confirmed") from rollback_error
            raise ActivationFailed(f"{domain} activation failed; prior state was restored") from error
        return candidate_revision

    def _candidate_from_config(self, config):
        return self._store.build_candidate(config)

    def _restart_required(self, config) -> bool:
        active = self._store.snapshot.config
        return self._activation_policy.classify(active, config).restart_required

    def _result(self, persisted_revision: str, restart_required: bool) -> dict[str, Any]:
        return {
            "active_revision": self.active_revision,
            "persisted_revision": persisted_revision,
            "restart_required": restart_required,
        }
