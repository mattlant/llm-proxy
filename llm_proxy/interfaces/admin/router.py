from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Request

from llm_proxy.management.errors import InvalidConfiguration, ManagementError

router = APIRouter(prefix="/_admin/v1", tags=["admin"])


async def _authenticated(request: Request) -> None:
    await request.app.state.admin_authenticator.authenticate(request)


@router.get("/status", dependencies=[Depends(_authenticated)])
async def status(request: Request):
    return request.app.state.management_query_service.status()


@router.get("/session", dependencies=[Depends(_authenticated)])
async def session(request: Request):
    return {"permissions": request.app.state.admin_authenticator.permissions()}


@router.get("/configuration", dependencies=[Depends(_authenticated)])
async def configuration(request: Request):
    return request.app.state.management_query_service.configuration()

@router.get("/configuration/administration", dependencies=[Depends(_authenticated)])
async def configuration_administration(request: Request):
    return await request.app.state.management_query_service.configuration_administration()


@router.get("/extensions", dependencies=[Depends(_authenticated)])
async def extensions(request: Request):
    return request.app.state.management_query_service.extensions()


@router.get("/providers", dependencies=[Depends(_authenticated)])
async def providers(request: Request):
    return request.app.state.management_query_service.providers()


@router.get("/models", dependencies=[Depends(_authenticated)])
async def models(request: Request):
    return request.app.state.management_query_service.models()


@router.get("/providers/{provider_instance}/commands", dependencies=[Depends(_authenticated)])
async def provider_commands(provider_instance: str, request: Request):
    return request.app.state.command_query_service.list(provider_instance)


@router.post("/providers/{provider_instance}/commands/{command_name}", dependencies=[Depends(_authenticated)])
async def invoke_provider_command(provider_instance: str, command_name: str, request: Request, payload: Any = Body(...)):
    try:
        result = await request.app.state.command_dispatch_service.dispatch(provider_instance, command_name, payload)
    except ManagementError as error:
        request.state.command_audit = getattr(error, "command_audit", None)
        raise
    request.state.command_audit = result
    return {key: value for key, value in result.items() if key not in {"duration_ms", "mutability", "required_permission"}}


@router.post("/configuration/validate", dependencies=[Depends(_authenticated)])
async def validate(document: dict[str, Any], request: Request):
    return request.app.state.management_configuration_service.validate(document).as_dict()

@router.get("/configuration/policies", dependencies=[Depends(_authenticated)])
async def policies(request: Request):
    return request.app.state.management_configuration_service.policy_snapshot()

@router.post("/configuration/policies/validate", dependencies=[Depends(_authenticated)])
async def validate_policies(payload: dict[str, Any], request: Request):
    if set(payload) != {"policies"} or not isinstance(payload["policies"], list): raise InvalidConfiguration("policies are required")
    result = request.app.state.management_configuration_service.validate_policies(payload["policies"])
    return {"valid": result.valid, "revision": result.revision, "errors": list(result.errors)}

@router.put("/configuration/policies", dependencies=[Depends(_authenticated)])
async def apply_policies(payload: dict[str, Any], request: Request):
    if set(payload) != {"expected_revision", "policies"}: raise InvalidConfiguration("expected_revision and policies are required")
    expected_revision, policies = payload["expected_revision"], payload["policies"]
    if not isinstance(expected_revision, str) or not isinstance(policies, list): raise InvalidConfiguration("expected_revision and policies are required")
    return request.app.state.management_configuration_service.apply_policies(policies, expected_revision)

@router.get("/configuration/model-profiles", dependencies=[Depends(_authenticated)])
async def model_profiles(request: Request):
    return request.app.state.management_configuration_service.model_profiles_snapshot()

@router.post("/configuration/model-profiles/validate", dependencies=[Depends(_authenticated)])
async def validate_model_profiles(payload: dict[str, Any], request: Request):
    if set(payload) != {"models"} or not isinstance(payload["models"], dict):
        raise InvalidConfiguration("models are required")
    result = request.app.state.management_configuration_service.validate_model_profiles(payload["models"])
    return {"valid": result.valid, "revision": result.revision, "errors": list(result.errors)}

@router.put("/configuration/model-profiles", dependencies=[Depends(_authenticated)])
async def apply_model_profiles(payload: dict[str, Any], request: Request):
    if set(payload) != {"expected_revision", "models"} or not isinstance(payload["expected_revision"], str) or not isinstance(payload["models"], dict):
        raise InvalidConfiguration("expected_revision and models are required")
    return request.app.state.management_configuration_service.apply_model_profiles(payload["models"], payload["expected_revision"])


@router.put("/configuration", dependencies=[Depends(_authenticated)])
async def update(payload: dict[str, Any], request: Request):
    expected_revision = payload.get("expected_revision")
    document = payload.get("configuration")
    if not isinstance(expected_revision, str) or not isinstance(document, dict):
        raise InvalidConfiguration("expected_revision and configuration are required")
    return request.app.state.management_configuration_service.update(document, expected_revision)


@router.post("/configuration/reload", dependencies=[Depends(_authenticated)])
async def reload(payload: dict[str, Any], request: Request):
    expected_revision = payload.get("expected_revision")
    if not isinstance(expected_revision, str):
        raise InvalidConfiguration("expected_revision is required")
    return request.app.state.management_configuration_service.reload(expected_revision)
