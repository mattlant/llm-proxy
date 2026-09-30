"""Canonical configuration serialization and revision values."""
from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import yaml

from .models import GatewayConfig


def configuration_mapping(config: GatewayConfig) -> dict[str, Any]:
    """Return the public YAML shape without exposing dataclass internals."""
    def plain(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, dict) or hasattr(value, "items"):
            return {plain(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list, frozenset, set)):
            return [plain(item) for item in value]
        return value

    result = plain(config)
    result["interfaces"] = {key: value for key, value in result["interfaces"].items()}
    for value in result["interfaces"].values():
        value.pop("name")
    for value in result["providers"].values():
        value.pop("name")
        value["extension"] = value.pop("extension_id")
    for value in result.get("extensions", {}).values():
        value.pop("name")
        value["extension"] = value.pop("extension_id")
    for value in result["models"].values():
        value.pop("name")
        value["interfaces"] = sorted(value["interfaces"])
        value["aliases"] = value["aliases"]["by_interface"]
    for policy in result["policies"]:
        policy["match"]["interfaces"] = sorted(policy["match"]["interfaces"])
    return result


def serialize_configuration(config: GatewayConfig) -> str:
    return yaml.safe_dump(configuration_mapping(config), sort_keys=True, allow_unicode=True)


def configuration_revision(config: GatewayConfig) -> str:
    return hashlib.sha256(serialize_configuration(config).encode("utf-8")).hexdigest()
