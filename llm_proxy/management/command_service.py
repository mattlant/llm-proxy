from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from llm_proxy.provider_extensions.management import ManagementCommandContext, ManagementCommandPermission

from .command_registry import ProviderCommandRegistry
from .command_validation import CommandSchemaError, compile_schema, validate_payload
from .errors import CommandFailed, CommandNotFound, CommandTimeout, InvalidCommandInput, InvalidCommandOutput, ManagementCommandsUnavailable, ManagementError, ProviderInstanceNotFound

log = logging.getLogger("llm-proxy")


class AdminCommandAuthorizer:
    def permits(self, permission: ManagementCommandPermission) -> bool:
        return permission in {ManagementCommandPermission.INSPECT, ManagementCommandPermission.OPERATE}


class CommandQueryService:
    def __init__(self, registry: ProviderCommandRegistry) -> None:
        self._registry = registry

    def list(self, provider_instance: str) -> dict[str, Any]:
        entry = self._registry.get(provider_instance)
        if entry is None:
            if self._registry.has_instance(provider_instance):
                raise ManagementCommandsUnavailable("management commands are unavailable for this provider instance")
            raise ProviderInstanceNotFound("provider instance was not found")
        return {"provider_instance": provider_instance, "extension_id": entry.extension_id, "commands": [
            {"name": item.name, "description": item.description, "input_schema": item.input_schema, "output_schema": item.output_schema, "mutability": item.mutability.value, "required_permission": item.required_permission.value}
            for item in sorted(entry.commands.values(), key=lambda value: value.name)
        ]}


class CommandDispatchService:
    def __init__(self, registry: ProviderCommandRegistry, timeout_seconds: float, authorizer: AdminCommandAuthorizer | None = None) -> None:
        self._registry, self._timeout, self._authorizer = registry, timeout_seconds, authorizer or AdminCommandAuthorizer()

    async def dispatch(self, provider_instance: str, command_name: str, payload: Any) -> dict[str, Any]:
        started = time.monotonic()
        audit = {"invocation_id": uuid.uuid4().hex, "provider_instance": provider_instance, "command": command_name, "extension_id": "unknown", "mutability": "unknown", "required_permission": "unknown"}
        try:
            entry = self._registry.get(provider_instance)
            if entry is None:
                if self._registry.has_instance(provider_instance):
                    raise ManagementCommandsUnavailable("management commands are unavailable for this provider instance")
                raise ProviderInstanceNotFound("provider instance was not found")
            audit["extension_id"] = entry.extension_id
            descriptor = entry.commands.get(command_name)
            if descriptor is None:
                raise CommandNotFound("management command was not found")
            audit["mutability"] = descriptor.mutability.value
            audit["required_permission"] = descriptor.required_permission.value
            if not self._authorizer.permits(descriptor.required_permission):
                from .errors import CommandUnauthorized
                raise CommandUnauthorized("command is not authorized")
            input_validator = compile_schema(descriptor.input_schema)
            validate_payload(input_validator, payload)
        except CommandSchemaError as error:
            failure = InvalidCommandInput(str(error))
            self._attach_audit(failure, audit, started)
            raise failure from error
        except ManagementError as error:
            self._attach_audit(error, audit, started)
            raise
        context = ManagementCommandContext(provider_instance, audit["invocation_id"])
        try:
            result = await asyncio.wait_for(entry.executor.execute_management_command(command_name, payload, context), timeout=self._timeout)
        except asyncio.TimeoutError as error:
            failure = CommandTimeout("management command timed out")
            self._attach_audit(failure, audit, started, timeout=True)
            raise failure from error
        except asyncio.CancelledError:
            log.info(
                "admin_command invocation_id=%s provider_instance=%s extension_id=%s command=%s mutability=%s required_permission=%s outcome=command_cancelled timeout=false cancelled=true duration_ms=%d",
                audit["invocation_id"], audit["provider_instance"], audit["extension_id"], audit["command"],
                audit["mutability"], audit["required_permission"], int((time.monotonic() - started) * 1000),
            )
            raise
        except Exception as error:
            failure = CommandFailed("management command failed")
            self._attach_audit(failure, audit, started)
            raise failure from error
        try:
            validate_payload(compile_schema(descriptor.output_schema), result)
        except CommandSchemaError as error:
            failure = InvalidCommandOutput("management command returned invalid output")
            self._attach_audit(failure, audit, started)
            raise failure from error
        return {"provider_instance": provider_instance, "command": command_name, "invocation_id": audit["invocation_id"], "result": result, "duration_ms": int((time.monotonic() - started) * 1000), "mutability": descriptor.mutability.value, "required_permission": descriptor.required_permission.value, "extension_id": entry.extension_id}

    @staticmethod
    def _attach_audit(error: ManagementError, audit: dict[str, str], started: float, timeout: bool = False) -> None:
        error.command_audit = {**audit, "outcome": error.code, "timeout": timeout, "cancelled": False, "duration_ms": int((time.monotonic() - started) * 1000)}
