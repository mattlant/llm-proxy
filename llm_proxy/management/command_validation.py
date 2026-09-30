from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import SchemaError

from llm_proxy.provider_extensions.management import JsonValue

_ALLOWED = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "oneOf", "anyOf", "allOf"}
_MAX_DEPTH = 16
_MAX_BYTES = 64 * 1024


class CommandSchemaError(ValueError):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _check_json(value: Any, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise CommandSchemaError("value exceeds maximum depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise CommandSchemaError("value must not contain NaN or infinity")
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CommandSchemaError("object keys must be strings")
        for item in value.values():
            _check_json(item, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_json(item, depth + 1)
        return
    raise CommandSchemaError("value must be JSON-compatible")


def _check_schema(schema: Mapping[str, JsonValue], depth: int = 0) -> None:
    _check_json(schema, depth)
    if depth > _MAX_DEPTH:
        raise CommandSchemaError("schema exceeds maximum depth")
    for key, value in schema.items():
        if key not in _ALLOWED:
            raise CommandSchemaError(f"unsupported schema keyword '{key}'")
        if key == "properties" and isinstance(value, Mapping):
            for nested in value.values():
                if not isinstance(nested, Mapping):
                    raise CommandSchemaError("property schemas must be objects")
                _check_schema(nested, depth + 1)
        elif key in {"items"} and isinstance(value, Mapping):
            _check_schema(value, depth + 1)
        elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, (list, tuple)):
            for nested in value:
                if not isinstance(nested, Mapping):
                    raise CommandSchemaError(f"{key} members must be objects")
                _check_schema(nested, depth + 1)


def compile_schema(schema: Mapping[str, JsonValue]) -> Draft202012Validator:
    try:
        encoded = json.dumps(_plain(schema), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CommandSchemaError("schema must be finite JSON") from error
    if len(encoded.encode()) > _MAX_BYTES:
        raise CommandSchemaError("schema exceeds maximum size")
    _check_schema(schema)
    try:
        normalized = _plain(schema)
        Draft202012Validator.check_schema(normalized)
    except SchemaError as error:
        raise CommandSchemaError("invalid JSON schema") from error
    return Draft202012Validator(normalized)


def validate_payload(validator: Draft202012Validator, payload: Any) -> None:
    try:
        encoded = json.dumps(_plain(payload), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CommandSchemaError("payload must be finite JSON") from error
    if len(encoded.encode()) > _MAX_BYTES:
        raise CommandSchemaError("payload exceeds maximum size")
    _check_json(payload)
    error = next(iter(validator.iter_errors(payload)), None)
    if error is not None:
        path = ".".join(str(item) for item in error.absolute_path) or "$"
        raise CommandSchemaError(f"{path}: {error.message[:200]}")
