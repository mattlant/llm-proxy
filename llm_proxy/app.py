from __future__ import annotations

import logging
import os
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .application.execution_coordinator import ExecutionCoordinator
from .application.semantic_support import SemanticProjectionRegistry, SemanticSupportResolver
from .application.wire_planning import PrivateWirePlanningRegistry, WireCapabilityIdentity
from .application.capabilities import CapabilityExecutor
from .application.lifecycle import ApplicationLifecycle
from .extensions.kernel import ExtensionInstanceRegistry, discover_extensions as discover_capability_extensions, activate_extension_instances
from .extensions.kernel import CapabilityFamilyAdapter
from .extensions.side_effect import SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, validate_side_effect
from .application.model_listing import ListedModel, ModelListingService
from .application.provider_registry import ProviderGatewayRegistry, activate_provider_instances, discover_extensions
from .application.runtime import ActivationPolicy, RuntimeDependencies
from .providers.ollama.extension import OllamaExtension
from .configuration.loader import ConfigurationLoader
from .configuration.models import InterfaceName
from .configuration.store import ConfigurationSnapshot, ConfigurationStore
from .interfaces.ollama.router import router as ollama_router
from .interfaces.openai.router import router as openai_router
from .interfaces.openai.wire_request import OPENAI_CHAT_COMPLETIONS_CAPABILITY
from .interfaces.openai.wire_mutation import OpenAIWireMutationProjector
from .interfaces.openai.error_mapper import OpenAIErrorMapper
from .interfaces.ollama.error_mapper import OllamaErrorMapper
from .interfaces.anthropic.error_mapper import AnthropicErrorMapper
from .interfaces.openai.response_mapper import OpenAIResponseMapper
from .interfaces.ollama.response_mapper import OllamaResponseMapper
from .interfaces.anthropic.response_mapper import AnthropicResponseMapper
from .interfaces.anthropic.router import router as anthropic_router
from .interfaces.admin.router import router as admin_router
from .interfaces.admin.error_mapper import management_error
from .interfaces.management_frontend import mount_management_frontend
from .management.authentication import AdminAuthenticator
from .management.configuration_service import ConfigurationManagementService
from .management.errors import ManagementError
from .management.errors import InvalidConfiguration
from .management.service import ManagementQueryService
from .management.command_registry import ProviderCommandRegistry
from .management.command_service import CommandDispatchService, CommandQueryService
from .observability.payload_trace import PayloadTraceRecorder
from .observability.operational_logging import (
    OperationalLogger, begin_operational_request, bind_operational_request,
    configure_logging, reset_operational_request,
)

CONFIG_PATH = Path(os.getenv("LLM_PROXY_CONFIG", Path.cwd() / "config.yaml"))

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm-proxy")


def _initial_store() -> ConfigurationStore:
    loader = ConfigurationLoader()
    initial = loader.load_path(CONFIG_PATH)
    return ConfigurationStore(CONFIG_PATH, loader, initial.server.config_reload_seconds)


configuration_store = _initial_store()
logging.getLogger().setLevel(configuration_store.snapshot.config.server.log_level.upper())
log.setLevel(configuration_store.snapshot.config.server.log_level.upper())


def _snapshot_for(store: ConfigurationStore) -> ConfigurationSnapshot:
    store.reload()
    return store.snapshot


def _wire_planning_registry() -> PrivateWirePlanningRegistry:
    return PrivateWirePlanningRegistry({
        WireCapabilityIdentity.from_capability(OPENAI_CHAT_COMPLETIONS_CAPABILITY): OpenAIWireMutationProjector(),
    })


def _semantic_projection_registry() -> SemanticProjectionRegistry:
    return SemanticProjectionRegistry({
        InterfaceName.OPENAI: OpenAIResponseMapper(),
        InterfaceName.OLLAMA: OllamaResponseMapper(),
        InterfaceName.ANTHROPIC: AnthropicResponseMapper(),
    })


def _metadata_router(snapshot: Callable[[], ConfigurationSnapshot], listing_service: ModelListingService | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/_proxy/health")
    async def health() -> dict[str, Any]:
        current = snapshot()
        config = current.config
        interface_order = {InterfaceName.OPENAI: 0, InterfaceName.OLLAMA: 1, InterfaceName.ANTHROPIC: 2}
        return {
            "ok": True,
            "config_file": str(current.source_path),
            "loaded_at": current.loaded_at,
            "interfaces": {name.value: value.enabled for name, value in config.interfaces.items()},
            "providers": {
                name: {
                    "extension": provider.extension_id,
                }
                for name, provider in config.providers.items()
                if provider.enabled
            },
            "models": {
                name: {
                    "interfaces": [interface.value for interface in sorted(profile.interfaces, key=interface_order.get)],
                    "provider": profile.provider,
                }
                for name, profile in config.models.items()
            },
        }

    @router.get("/api/tags")
    async def api_tags(request: Request) -> dict[str, Any]:
        service = getattr(request.app.state, "model_listing_service", listing_service)
        models = await service.list_models(InterfaceName.OLLAMA) if service else tuple(ListedModel(profile.name, profile.upstream_model, profile.provider) for profile in snapshot().registry.list_models(InterfaceName.OLLAMA))
        models = [{"name": item.identity, "model": item.upstream_model, "provider": item.provider_instance_name} for item in models]
        return {"models": models}

    @router.get("/v1/models")
    async def v1_models(request: Request) -> dict[str, Any]:
        service = getattr(request.app.state, "model_listing_service", listing_service)
        models = await service.list_models(InterfaceName.OPENAI) if service else tuple(ListedModel(profile.name, profile.upstream_model, profile.provider) for profile in snapshot().registry.list_models(InterfaceName.OPENAI))
        return {
            "object": "list",
            "data": [
                {"id": item.identity, "object": "model", "owned_by": item.provider_instance_name}
                for item in models
            ],
        }

    return router


@asynccontextmanager
async def _application_lifespan(app, provider_extensions, provided_gateways, capability_entry_points):
    lifecycle = ApplicationLifecycle()
    try:
        bootstrap = app.state.configuration_store
        bootstrap_snapshot = bootstrap.snapshot
        activation_policy = app.state.activation_policy
        catalog = discover_capability_extensions(entry_points=capability_entry_points)
        instances = await activate_extension_instances(catalog, bootstrap_snapshot.config.extensions.values(), (CapabilityFamilyAdapter(SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, validate_side_effect),))
        lifecycle.add(instances)
        gateways = provided_gateways or activate_provider_instances(
            provider_extensions,
            bootstrap_snapshot.config.providers,
            activation_policy.provider_identity,
        )
        lifecycle.add(gateways)
        runtime_dependencies = RuntimeDependencies(
            provider_gateways=gateways,
            extension_instances=instances,
        )
        live_store = ConfigurationStore.from_initial_config(
            bootstrap.path,
            bootstrap.loader,
            bootstrap.reload_interval_seconds,
            bootstrap_snapshot.config,
            bootstrap_snapshot.source_mtime,
            runtime_dependencies=runtime_dependencies,
            activation_policy=activation_policy,
            clock=bootstrap._clock,
        )
        app.state.configuration_store = live_store
        app.state.extension_catalog = catalog
        app.state.extension_instances = instances
        app.state.capability_executor = CapabilityExecutor(instances)
        app.state.provider_gateways = gateways
        app.state.management_configuration_service = ConfigurationManagementService(live_store, activation_policy=activation_policy)
        app.state.execution_coordinator = ExecutionCoordinator(live_store, gateways, app.state.capability_executor, _wire_planning_registry(), activation_policy, SemanticSupportResolver(_semantic_projection_registry()))
        app.state.model_listing_service = ModelListingService(app.state.configuration_store, gateways) if isinstance(gateways, ProviderGatewayRegistry) else None
        if app.state.configuration_store.snapshot.config.management.enabled:
            providers = app.state.configuration_store.snapshot.config.providers
            command_capabilities = {
                name: bool(provider_extensions.get(provider.extension_id) and provider_extensions.get(provider.extension_id).capabilities.management_commands)
                for name, provider in providers.items()
            }
            command_registry = ProviderCommandRegistry(gateways, providers, {name: provider.extension_id for name, provider in providers.items()}, command_capabilities)
            configuration_service = app.state.management_configuration_service
            app.state.provider_command_registry = command_registry
            app.state.command_query_service = CommandQueryService(command_registry)
            app.state.command_dispatch_service = CommandDispatchService(command_registry, app.state.configuration_store.snapshot.config.management.command_timeout_seconds)
            app.state.management_query_service = ManagementQueryService(live_store, provider_extensions, gateways, configuration_service, command_registry, activation_policy)
        yield
    finally:
        await lifecycle.aclose()


def create_app(
    configuration_store: ConfigurationStore | None = None,
    provider_gateways=None,
    extension_entry_points=None,
) -> FastAPI:
    store = configuration_store
    current = (lambda: _snapshot_for(store)) if store is not None else (lambda: _snapshot_for(globals()["configuration_store"]))
    initial = current()
    configure_logging(initial.config.server)
    payload_trace = PayloadTraceRecorder()
    extensions = discover_extensions((("built-in ollama", OllamaExtension()),))
    gateways = provider_gateways
    active_store = store or globals()["configuration_store"]
    activation_policy = ActivationPolicy()
    if isinstance(gateways, ProviderGatewayRegistry):
        if active_store.runtime_dependencies.provider_gateways is not gateways:
            raise ValueError("provider gateways must match the store runtime dependencies")
    app = FastAPI(title="LLM Proxy", version=__version__, lifespan=lambda value: _application_lifespan(value, extensions, gateways, extension_entry_points))
    app.state.configuration_store = active_store
    app.state.activation_policy = activation_policy
    app.state.provider_extension_registry = extensions
    app.state.provider_gateways = gateways
    snapshot = lambda: _snapshot_for(app.state.configuration_store)
    app.state.model_listing_service = ModelListingService(app.state.configuration_store, gateways) if isinstance(gateways, ProviderGatewayRegistry) else None
    if gateways is not None:
        app.state.capability_executor = CapabilityExecutor(ExtensionInstanceRegistry((), {}))
        app.state.execution_coordinator = ExecutionCoordinator(app.state.configuration_store, gateways, app.state.capability_executor, _wire_planning_registry(), activation_policy, SemanticSupportResolver(_semantic_projection_registry()))
    app.state.payload_trace_recorder = payload_trace

    error_mapper = OpenAIErrorMapper()
    ollama_error_mapper = OllamaErrorMapper()
    anthropic_error_mapper = AnthropicErrorMapper()

    def mapper_for(request: Request):
        if request.url.path == "/v1/messages":
            return anthropic_error_mapper
        return ollama_error_mapper if request.url.path.startswith("/api/") else error_mapper

    @app.middleware("http")
    async def trace_chat_completion_payloads(request: Request, call_next):
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            return await call_next(request)
        server_config = snapshot().config.server
        trace = payload_trace.begin(server_config.payload_trace_mode, server_config.payload_trace_include_content)
        tokens = payload_trace.bind(trace)
        outcome = "success"
        is_stream = False
        try:
            body = await request.body()
            payload_trace.record_raw_http(
                "client_request_raw", body, context=trace, method=request.method, url=str(request.url),
                path=request.url.path, query=request.scope.get("query_string", b""), headers=tuple(request.scope.get("headers", ())),
            )
            try:
                inbound = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                inbound = body.decode("utf-8", errors="replace")
            is_stream = isinstance(inbound, dict) and inbound.get("stream") is True
            payload_trace.record("client_request", inbound, context=trace)
            response = await call_next(request)
            if response.status_code >= 400:
                outcome = "error"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "error"
            raise

        original = response.body_iterator

        async def traced_body():
            nonlocal outcome
            stream_completed = not is_stream
            stream_started = False
            try:
                async for chunk in original:
                    raw_chunk = chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8")
                    if is_stream:
                        if not stream_started:
                            payload_trace.record_raw_http("client_stream_start_raw", context=trace, stream=True, status_code=response.status_code, headers=tuple(response.raw_headers))
                            stream_started = True
                        payload_trace.record_raw_http("client_stream_bytes_raw", raw_chunk, context=trace, stream=True)
                    else:
                        payload_trace.record_raw_http("client_response_raw", raw_chunk, context=trace, status_code=response.status_code, headers=tuple(response.raw_headers))
                    text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                    if text.startswith("data: "):
                        data = text[6:].strip()
                        try:
                            client_payload: Any = json.loads(data) if data != "[DONE]" else data
                        except json.JSONDecodeError:
                            client_payload = data
                        payload_trace.record("client_stream_record", client_payload, context=trace, stream=True)
                        if client_payload == "[DONE]":
                            stream_completed = True
                        elif isinstance(client_payload, dict) and "error" in client_payload:
                            outcome = "error"
                    else:
                        try:
                            client_payload = json.loads(text)
                        except json.JSONDecodeError:
                            client_payload = text
                        payload_trace.record("client_response", client_payload, context=trace)
                    yield chunk
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception:
                outcome = "error"
                raise
            finally:
                if is_stream and outcome == "success" and not stream_completed:
                    outcome = "incomplete"
                payload_trace.end(outcome, context=trace)
                payload_trace.reset(tokens)

        response.body_iterator = traced_body()
        return response

    @app.middleware("http")
    async def operational_completion_lifecycle(request: Request, call_next):
        interfaces = {
            "/v1/chat/completions": "openai",
            "/api/chat": "ollama",
            "/api/generate": "ollama",
            "/v1/messages": "anthropic",
        }
        interface = interfaces.get(request.url.path)
        if request.method != "POST" or interface is None:
            return await call_next(request)
        context = begin_operational_request(interface)
        token = bind_operational_request(context)
        try:
            response = await call_next(request)
        except asyncio.CancelledError:
            reset_operational_request(token)
            raise
        except Exception:
            reset_operational_request(token)
            raise
        original = response.body_iterator

        async def operational_body():
            outcome = "success" if response.status_code < 400 else "failure"
            try:
                async for chunk in original:
                    yield chunk
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception:
                outcome = "error"
                raise
            finally:
                OperationalLogger().log_completion(context, response.status_code, outcome)
                reset_operational_request(token)

        response.body_iterator = operational_body()
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        log.debug(
            "validation failed: path=%s errors=%s",
            request.url.path,
            [(e["loc"], e["msg"], e["type"]) for e in error.errors()],
        )     
        # Extract just the messages field errors, skip the full input payload
        messages_errors = [
            {
                "loc": e["loc"],
                "msg": e["msg"],
                "type": e["type"],
            }
            for e in error.errors()
            if "messages" in str(e["loc"])
        ]
        log.debug(
            "validation failed: path=%s messages_errors=%s",
            request.url.path,
            messages_errors,
        )           
        if request.url.path.startswith("/_admin/"):
            return await management_error(request, InvalidConfiguration("invalid management request"))
        status, payload = mapper_for(request).map_error(ValueError("invalid request"))
        return JSONResponse(payload, status_code=status)

    @app.exception_handler(Exception)
    async def interface_error(request: Request, error: Exception) -> JSONResponse:
        status, payload = mapper_for(request).map_error(error)
        return JSONResponse(payload, status_code=status)
    if initial.config.management.enabled:
        if initial.config.server.host not in {"127.0.0.1", "::1", "localhost"} and not initial.config.management.allow_remote:
            raise RuntimeError("management remote binding requires allow_remote=true")
        configuration_service = ConfigurationManagementService(app.state.configuration_store, activation_policy=activation_policy)
        management_gateways = gateways or ProviderGatewayRegistry({}, initial.config.providers)
        command_capabilities = {name: bool(extensions.get(provider.extension_id) and extensions.get(provider.extension_id).capabilities.management_commands) for name, provider in initial.config.providers.items()}
        command_registry = ProviderCommandRegistry(management_gateways, initial.config.providers, {name: provider.extension_id for name, provider in initial.config.providers.items()}, command_capabilities)
        app.state.management_configuration_service = configuration_service
        app.state.provider_command_registry = command_registry
        app.state.command_query_service = CommandQueryService(command_registry)
        app.state.command_dispatch_service = CommandDispatchService(command_registry, initial.config.management.command_timeout_seconds)
        app.state.management_query_service = ManagementQueryService(app.state.configuration_store, extensions, management_gateways, configuration_service, command_registry, activation_policy)
        app.state.admin_authenticator = AdminAuthenticator(initial.config.management.token_env)
        app.add_exception_handler(ManagementError, management_error)

        @app.middleware("http")
        async def audit_admin_requests(request: Request, call_next):
            if not request.url.path.startswith("/_admin/"):
                return await call_next(request)
            import time
            started = time.monotonic()
            response = await call_next(request)
            service = request.app.state.management_configuration_service
            try:
                active_revision = service.active_revision
                persisted_revision = service.persisted_revision
                restart_required = active_revision != persisted_revision
            except Exception:
                active_revision = persisted_revision = "unavailable"
                restart_required = False
            log.info(
                "admin operation=%s outcome=%s status=%s duration_ms=%d active_revision=%s persisted_revision=%s restart_required=%s",
                request.url.path.rsplit("/", 1)[-1],
                "success" if response.status_code < 400 else "failure",
                response.status_code,
                int((time.monotonic() - started) * 1000),
                active_revision,
                persisted_revision,
                restart_required,
            )
            command_audit = getattr(request.state, "command_audit", None)
            if command_audit is not None:
                log.info(
                    "admin_command invocation_id=%s provider_instance=%s extension_id=%s command=%s mutability=%s required_permission=%s outcome=%s timeout=%s cancelled=%s duration_ms=%d",
                    command_audit["invocation_id"], command_audit["provider_instance"], command_audit.get("extension_id", "unknown"),
                    command_audit["command"], command_audit["mutability"], command_audit["required_permission"],
                    command_audit.get("outcome", "success"), command_audit.get("timeout", False), command_audit.get("cancelled", False), command_audit["duration_ms"],
                )
            return response
        app.include_router(admin_router)
        mount_management_frontend(app)
    app.include_router(_metadata_router(snapshot, app.state.model_listing_service))
    current_snapshot = snapshot()
    if current_snapshot.config.interfaces.get(InterfaceName.OPENAI, None) and current_snapshot.config.interfaces[InterfaceName.OPENAI].enabled:
        app.include_router(openai_router)
    if current_snapshot.config.interfaces.get(InterfaceName.OLLAMA, None) and current_snapshot.config.interfaces[InterfaceName.OLLAMA].enabled:
        app.include_router(ollama_router)
    if current_snapshot.config.interfaces.get(InterfaceName.ANTHROPIC, None) and current_snapshot.config.interfaces[InterfaceName.ANTHROPIC].enabled:
        app.include_router(anthropic_router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    initial = _snapshot_for(configuration_store).config
    config = initial.server
    log.info("Starting LLM Proxy on http://%s:%s", config.host, config.port)
    if initial.management.enabled:
        host = f"[{config.host}]" if ":" in config.host and not config.host.startswith("[") else config.host
        log.info("Management UI available at http://%s:%s/management/", host, config.port)
    log.info("Configuration file: %s", CONFIG_PATH)
    uvicorn.run("llm_proxy.app:app", host=config.host, port=config.port, reload=False)


if __name__ == "__main__":
    main()
