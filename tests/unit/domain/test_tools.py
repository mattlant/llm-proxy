from __future__ import annotations

import pytest

from llm_proxy.domain.tools import ToolDefinition


def test_tool_schema_is_preserved_including_nested_refs_and_defs():
    schema = {
        "type": "object",
        "properties": {
            "request": {"$ref": "#/$defs/Request"},
        },
        "$defs": {
            "Request": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string", "default": "/"}},
            }
        },
        "required": ["request"],
        "additionalProperties": False,
    }

    tool = ToolDefinition("inspect", "Inspect a path", schema)

    assert tool.input_schema == schema
    assert tool.input_schema["properties"] == schema["properties"]
    assert tool.input_schema["$defs"] == schema["$defs"]


def test_tool_schema_is_defensively_copied():
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    tool = ToolDefinition("inspect", "Inspect a path", schema)
    schema["properties"]["path"]["type"] = "integer"

    assert tool.input_schema["properties"]["path"]["type"] == "string"
    with pytest.raises((AttributeError, TypeError)):
        tool.name = "other"  # type: ignore[misc]


def test_tool_definition_rejects_invalid_name_and_schema():
    with pytest.raises(ValueError):
        ToolDefinition("", "description", {})
    with pytest.raises(TypeError):
        ToolDefinition("inspect", "description", [])  # type: ignore[arg-type]
