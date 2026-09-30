import json

import pytest

from llm_proxy.interfaces.openai.wire_request import OpenAIWireRequestMapper
from llm_proxy.interfaces.openai.dto import OpenAIChatCompletionRequest
from llm_proxy.interfaces.openai.request_mapper import OpenAIRequestMapper
from llm_proxy.provider_extensions import WirePatch, WirePatchOperation


def test_wire_request_preserves_unknown_null_and_multimodal_values_when_patched() -> None:
    body = b'{"model":"alias","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]}],"vendor":{"null":null}}'
    request = OpenAIWireRequestMapper().parse(body, ((b"x-client", b"one"), (b"x-client", b"two")))

    result = WirePatch((WirePatchOperation("/model", "upstream"),)).apply(request.wire_request)

    assert json.loads(result) == {
        "model": "upstream",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}],
        "vendor": {"null": None},
    }
    assert request.wire_request.headers == ((b"x-client", b"one"), (b"x-client", b"two"))


def test_wire_patch_returns_original_bytes_without_mutations() -> None:
    body = b'{ "model" : "alias", "messages": [] }'
    request = OpenAIWireRequestMapper().parse(body, ())
    assert WirePatch().apply(request.wire_request) == body


def test_wire_patch_creates_only_explicit_stream_options_parent() -> None:
    request = OpenAIWireRequestMapper().parse(b'{"model":"alias","messages":[]}', ())
    assert json.loads(WirePatch((WirePatchOperation("/stream_options/include_usage", True, True),)).apply(request.wire_request))["stream_options"] == {"include_usage": True}


def test_wire_request_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError):
        OpenAIWireRequestMapper().parse(b'{"model":"a","model":"b"}', ())


def test_wire_semantics_match_canonical_mapping_and_ignore_unsupported_shapes() -> None:
    payload = {
        "model": "alias",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }
    canonical = OpenAIRequestMapper().map_request(OpenAIChatCompletionRequest.model_validate(payload))
    wire = OpenAIWireRequestMapper().parse(json.dumps(payload).encode(), ())

    assert wire.messages == canonical.messages
    assert wire.tool_names == tuple(tool.name for tool in canonical.tools)
    assert wire.wire_request.projection.messages == wire.messages
    unsupported = OpenAIWireRequestMapper().parse(b'{"model":"alias","messages":[{"role":"user","content":[{"type":"text","text":"match-me"},{"type":"image_url","image_url":{"url":"x"}}]}]}', ())
    assert unsupported.messages == ()
