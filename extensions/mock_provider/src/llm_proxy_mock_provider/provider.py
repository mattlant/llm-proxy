"""Scenario-driven gateway implementation; deliberately independent of gateway internals."""
from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from llm_proxy.provider_extensions import (
    CompletionExecution, CompletionGateway, CompletionResponse, FinishReason, ManagementCommandContext,
    ManagementCommandDescriptor, ManagementCommandMutability, ManagementCommandPermission, Message, MessageRole,
    ProviderFactory, ProviderInstanceConfig, ProviderUnavailableError, ResponseCompleted, ResponseStarted,
    ProviderListedModel,
    SourceInterface, TextCompleted, TextContent, TextDelta, TextStarted, ToolCallArgumentsDelta,
    ToolCallCompleted, ToolCallContent, ToolCallStarted, Usage,
)


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("scenario object keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise ValueError("scenario values must be JSON-compatible")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _subset(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(key in actual and (_subset(value, actual[key]) if isinstance(value, Mapping) and isinstance(actual[key], Mapping) else value == actual[key]) for key, value in expected.items())


def _visible_text(execution: CompletionExecution) -> str:
    return "".join(block.text for message in execution.request.messages for block in message.content if isinstance(block, TextContent))


def _tool_choice(execution: CompletionExecution) -> str:
    choice = execution.request.tool_choice
    return getattr(choice, "name", type(choice).__name__.removesuffix("ToolChoice").lower())


def _history(execution: CompletionExecution) -> tuple[Mapping[str, Any], ...]:
    result = []
    for message in execution.request.messages:
        content = []
        for block in message.content:
            if isinstance(block, TextContent):
                content.append(MappingProxyType({"type": "text", "text": block.text}))
            elif isinstance(block, ToolCallContent):
                content.append(MappingProxyType({"type": "tool_call", "name": block.name, "arguments": block.arguments}))
            else:
                content.append(MappingProxyType({"type": type(block).__name__}))
        result.append(MappingProxyType({"role": message.role.value, "content": tuple(content)}))
    return tuple(result)


def _require_json_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _validate_response(response: Mapping[str, Any]) -> None:
    allowed = {"id", "text", "content", "usage", "finish_reason", "stop_sequence", "latency_ms", "fragments", "error", "midstream_error"}
    if set(response) - allowed or ("text" not in response and "content" not in response):
        raise ValueError("scenario response is invalid")
    if "id" in response and (not isinstance(response["id"], str) or not response["id"]):
        raise ValueError("scenario response id is invalid")
    if "text" in response and not isinstance(response["text"], str):
        raise ValueError("scenario response text is invalid")
    if response.get("finish_reason", "end_turn") not in {item.value for item in FinishReason}:
        raise ValueError("scenario response finish_reason is invalid")
    if response.get("stop_sequence") is not None and not isinstance(response.get("stop_sequence"), str):
        raise ValueError("scenario response stop_sequence is invalid")
    latency = response.get("latency_ms", 0)
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0:
        raise ValueError("scenario response latency_ms is invalid")
    if not isinstance(response.get("error", False), bool) or not isinstance(response.get("midstream_error", False), bool):
        raise ValueError("scenario response failure flags are invalid")
    usage = response.get("usage", {})
    if not isinstance(usage, Mapping) or set(usage) - {"input_tokens", "output_tokens"}:
        raise ValueError("scenario response usage is invalid")
    for tokens in usage.values():
        if tokens is not None and (not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0):
            raise ValueError("scenario response usage is invalid")
    if "content" in response:
        content = response["content"]
        if not isinstance(content, (list, tuple)) or not content:
            raise ValueError("scenario response content is invalid")
        for item in content:
            _require_json_mapping(item, "scenario response content item")
            if item.get("type", "text") == "text":
                if set(item) != {"type", "text"} or not isinstance(item.get("text"), str):
                    raise ValueError("scenario text content is invalid")
            elif item.get("type") == "tool_call":
                if set(item) - {"type", "id", "name", "arguments"} or not isinstance(item.get("name"), str) or not item["name"] or not isinstance(item.get("arguments"), Mapping):
                    raise ValueError("scenario tool content is invalid")
                _freeze(item["arguments"])
            else:
                raise ValueError("scenario response content type is invalid")
    fragments = response.get("fragments", {})
    if not isinstance(fragments, Mapping) or not all(isinstance(key, str) and isinstance(value, (list, tuple)) and value and all(isinstance(fragment, str) and fragment for fragment in value) for key, value in fragments.items()):
        raise ValueError("scenario response fragments are invalid")


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    priority: int
    match: Mapping[str, Any]
    responses: tuple[Mapping[str, Any], ...]
    repeat: int | None


@dataclass(frozen=True, slots=True)
class Scenario:
    defaults: Mapping[str, Any]
    rules: tuple[Rule, ...]
    expectations: tuple[Mapping[str, Any], ...]
    models: tuple[str, ...] = ()


def parse_scenario(value: Any) -> Scenario:
    if not isinstance(value, Mapping) or set(value) - {"version", "defaults", "rules", "expectations", "models"}:
        raise ValueError("scenario must contain only version, defaults, rules, expectations, and models")
    if value.get("version") != 1:
        raise ValueError("scenario version must be 1")
    defaults = value.get("defaults", {})
    rules = value.get("rules", ())
    expectations = value.get("expectations", ())
    models = value.get("models", ())
    if not isinstance(defaults, Mapping) or not isinstance(rules, (list, tuple)) or not isinstance(expectations, (list, tuple)) or not isinstance(models, (list, tuple)) or not all(isinstance(model, str) and model.strip() for model in models) or len(models) != len(set(models)):
        raise ValueError("scenario defaults, rules, expectations, and models have invalid types")
    parsed: list[Rule] = []
    names: set[str] = set()
    for item in rules:
        if not isinstance(item, Mapping) or set(item) - {"id", "priority", "match", "responses", "repeat"}:
            raise ValueError("scenario rule has unsupported keys")
        identity, priority, match, responses, repeat = item.get("id"), item.get("priority", 0), item.get("match", {}), item.get("responses"), item.get("repeat")
        if not isinstance(identity, str) or not identity or identity in names or not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError("scenario rule id or priority is invalid")
        if not isinstance(match, Mapping) or set(match) - {"model", "response_model", "upstream_model", "source_interface", "stream", "text", "metadata", "tool_names", "tool_choice", "messages", "call_index"}:
            raise ValueError("scenario rule match is invalid")
        for key in ("model", "response_model", "upstream_model", "text", "tool_choice"):
            if key in match and (not isinstance(match[key], str) or not match[key]):
                raise ValueError("scenario rule match is invalid")
        if "source_interface" in match and match["source_interface"] not in {item.value for item in SourceInterface}:
            raise ValueError("scenario rule source_interface is invalid")
        if "stream" in match and not isinstance(match["stream"], bool):
            raise ValueError("scenario rule stream is invalid")
        if "call_index" in match and (not isinstance(match["call_index"], int) or isinstance(match["call_index"], bool) or match["call_index"] <= 0):
            raise ValueError("scenario rule call_index is invalid")
        if "tool_names" in match and (not isinstance(match["tool_names"], (list, tuple)) or not all(isinstance(name, str) and name for name in match["tool_names"])):
            raise ValueError("scenario rule tool_names is invalid")
        for key in ("metadata", "messages"):
            if key in match:
                _freeze(match[key])
        if not isinstance(responses, (list, tuple)) or not responses or not all(isinstance(response, Mapping) for response in responses):
            raise ValueError("scenario rule requires responses")
        for response in responses:
            _validate_response(response)
        if repeat is not None and (not isinstance(repeat, int) or isinstance(repeat, bool) or repeat <= 0):
            raise ValueError("scenario repeat must be a positive integer")
        names.add(identity)
        parsed.append(Rule(identity, priority, _freeze(match), tuple(_freeze(response) for response in responses), repeat))
    expectation_ids: set[str] = set()
    for item in expectations:
        if not isinstance(item, Mapping) or set(item) - {"id", "rule_id", "count"} or not isinstance(item.get("id"), str) or not item["id"] or item["id"] in expectation_ids or not isinstance(item.get("rule_id"), str) or item["rule_id"] not in names or not isinstance(item.get("count"), int) or isinstance(item["count"], bool) or item["count"] < 0:
            raise ValueError("scenario expectation is invalid")
        expectation_ids.add(item["id"])
    return Scenario(_freeze(defaults), tuple(parsed), tuple(_freeze(item) for item in expectations), tuple(models))


class DeterministicMockGateway(CompletionGateway):
    management_commands = (
        ManagementCommandDescriptor("replace_scenario", "Atomically replace this instance scenario", {"type": "object", "required": ["scenario"], "properties": {"scenario": {"type": "object"}}, "additionalProperties": False}, {"type": "object"}, ManagementCommandMutability.STATE_CHANGING, ManagementCommandPermission.OPERATE),
        ManagementCommandDescriptor("enqueue_response", "Queue a response for this instance", {"type": "object", "required": ["response"], "properties": {"response": {"type": "object"}}, "additionalProperties": False}, {"type": "object"}, ManagementCommandMutability.STATE_CHANGING, ManagementCommandPermission.OPERATE),
        ManagementCommandDescriptor("inspect_captured_requests", "Inspect sanitized captured requests", {"type": "object", "additionalProperties": False}, {"type": "object"}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT),
        ManagementCommandDescriptor("inspect_pending_expectations", "Inspect unmet expectations", {"type": "object", "additionalProperties": False}, {"type": "object"}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT),
        ManagementCommandDescriptor("reset_state", "Reset this instance state", {"type": "object", "additionalProperties": False}, {"type": "object"}, ManagementCommandMutability.STATE_CHANGING, ManagementCommandPermission.OPERATE),
        ManagementCommandDescriptor("verify_expectations", "Return deterministic expectation status", {"type": "object", "additionalProperties": False}, {"type": "object"}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT),
    )

    def __init__(self, config: ProviderInstanceConfig) -> None:
        scenario = config.config.get("scenario")
        self._scenario = parse_scenario(scenario) if scenario is not None else parse_scenario({"version": 1, "rules": []})
        self._revision = 1
        self._lock = asyncio.Lock()
        self._call_index = 0
        self._cursors: dict[str, int] = {}
        self._queue: deque[Mapping[str, Any]] = deque()
        capture_limit = config.config.get("capture_limit", 100)
        if not isinstance(capture_limit, int) or isinstance(capture_limit, bool) or not 1 <= capture_limit <= 1000:
            raise ValueError("capture_limit must be an integer between 1 and 1000")
        self._captures: deque[Mapping[str, Any]] = deque(maxlen=capture_limit)
        self._hits: dict[str, int] = {}
        self._in_flight: set[int] = set()
        redacted = config.config.get("redacted_metadata_keys", ())
        if not isinstance(redacted, (list, tuple)) or not all(isinstance(key, str) and key for key in redacted):
            raise ValueError("redacted_metadata_keys must be a sequence of non-blank strings")
        self._redacted_metadata_keys = frozenset(redacted)
        capture_max_bytes = config.config.get("capture_max_bytes", 16384)
        if not isinstance(capture_max_bytes, int) or isinstance(capture_max_bytes, bool) or not 128 <= capture_max_bytes <= 1048576:
            raise ValueError("capture_max_bytes must be an integer between 128 and 1048576")
        self._capture_max_bytes = capture_max_bytes

    async def list_models(self) -> tuple[ProviderListedModel, ...]:
        async with self._lock:
            return tuple(ProviderListedModel(model) for model in self._scenario.models)

    def _matches(self, rule: Rule, execution: CompletionExecution, index: int) -> bool:
        match = rule.match
        checks = {
            "model": execution.request.model, "upstream_model": execution.upstream_model,
            "response_model": execution.response_model,
            "source_interface": execution.source_interface.value, "stream": execution.request.stream,
            "text": _visible_text(execution), "call_index": index, "tool_choice": _tool_choice(execution),
            "tool_names": tuple(tool.name for tool in execution.request.tools),
            "messages": _history(execution),
        }
        for key, actual in checks.items():
            if key in match and match[key] != actual:
                return False
        if "metadata" in match and (not isinstance(match["metadata"], Mapping) or not _subset(match["metadata"], execution.request.metadata)):
            return False
        return "messages" not in match or match["messages"] == _history(execution)

    async def _reserve(self, execution: CompletionExecution) -> tuple[int, Mapping[str, Any], str]:
        async with self._lock:
            self._call_index += 1
            index = self._call_index
            metadata = {key: value for key, value in execution.request.metadata.items() if key not in self._redacted_metadata_keys}
            text = _visible_text(execution)
            truncated = False
            if len(text.encode("utf-8")) > self._capture_max_bytes:
                suffix = "...[truncated]"
                text = text.encode("utf-8")[:self._capture_max_bytes - len(suffix)].decode("utf-8", "ignore") + suffix
                truncated = True
            capture = MappingProxyType({"call_index": index, "model": execution.request.model, "response_model": execution.response_model, "upstream_model": execution.upstream_model, "source_interface": execution.source_interface.value, "stream": execution.request.stream, "text": text, "truncated": truncated, "tool_names": tuple(tool.name for tool in execution.request.tools), "metadata": _freeze(metadata), "messages": _history(execution)})
            self._captures.append(capture)
            if self._queue:
                self._in_flight.add(index)
                return index, self._queue.popleft(), "queued"
            candidates = [rule for rule in self._scenario.rules if self._matches(rule, execution, index) and (rule.repeat is None or self._hits.get(rule.id, 0) < rule.repeat)]
            if not candidates:
                raise ProviderUnavailableError("deterministic mock scenario has no matching response", provider=execution.provider_instance_name)
            rule = max(enumerate(candidates), key=lambda item: (item[1].priority, -next(i for i, known in enumerate(self._scenario.rules) if known.id == item[1].id)))[1]
            cursor = self._cursors.get(rule.id, 0)
            self._cursors[rule.id] = cursor + 1
            self._hits[rule.id] = self._hits.get(rule.id, 0) + 1
            self._in_flight.add(index)
            return index, rule.responses[cursor % len(rule.responses)], rule.id

    async def _release(self, index: int) -> None:
        async with self._lock:
            self._in_flight.discard(index)

    @staticmethod
    def _usage(response: Mapping[str, Any]) -> Usage:
        value = response.get("usage", {})
        if not isinstance(value, Mapping):
            raise ProviderUnavailableError("deterministic mock response usage is invalid")
        return Usage(value.get("input_tokens"), value.get("output_tokens"))

    @staticmethod
    def _content(response: Mapping[str, Any], index: int) -> tuple[Any, ...]:
        items = response.get("content")
        if items is None and "text" in response:
            items = ({"type": "text", "text": response["text"]},)
        if not isinstance(items, (list, tuple)) or not items:
            raise ProviderUnavailableError("deterministic mock response content is invalid")
        result = []
        for position, item in enumerate(items, 1):
            if not isinstance(item, Mapping):
                raise ProviderUnavailableError("deterministic mock response content is invalid")
            if item.get("type", "text") == "text" and isinstance(item.get("text"), str):
                result.append(TextContent(item["text"]))
            elif item.get("type") == "tool_call" and isinstance(item.get("name"), str) and isinstance(item.get("arguments", {}), Mapping):
                result.append(ToolCallContent(str(item.get("id", f"mock-tool-{index}-{position}")), item["name"], item.get("arguments", {})))
            else:
                raise ProviderUnavailableError("deterministic mock response content is invalid")
        return tuple(result)

    async def complete(self, execution: CompletionExecution) -> CompletionResponse:
        index, response, _ = await self._reserve(execution)
        try:
            if response.get("error"):
                raise ProviderUnavailableError("deterministic mock response failure", provider=execution.provider_instance_name)
            latency = response.get("latency_ms", 0)
            if latency:
                await asyncio.sleep(float(latency) / 1000)
            content = self._content(response, index)
            reason = FinishReason(response.get("finish_reason", "tool_use" if any(isinstance(item, ToolCallContent) for item in content) else "end_turn"))
            return CompletionResponse(str(response.get("id", f"mock-response-{index}")), execution.response_model, Message(MessageRole.ASSISTANT, content), reason, response.get("stop_sequence"), self._usage(response))
        finally:
            await self._release(index)

    async def stream(self, execution: CompletionExecution) -> AsyncIterator[Any]:
        index, response, _ = await self._reserve(execution)
        try:
            if response.get("error"):
                raise ProviderUnavailableError("deterministic mock response failure", provider=execution.provider_instance_name)
            latency = response.get("latency_ms", 0)
            if latency:
                await asyncio.sleep(float(latency) / 1000)
            response_id = str(response.get("id", f"mock-response-{index}"))
            yield ResponseStarted(response_id, execution.response_model)
            for position, item in enumerate(self._content(response, index), 1):
                block_id = f"mock-block-{index}-{position}"
                if isinstance(item, TextContent):
                    yield TextStarted(block_id)
                    fragments = response.get("fragments", {}).get(str(position), (item.text,))
                    for fragment in fragments:
                        yield TextDelta(block_id, fragment)
                    yield TextCompleted(block_id)
                else:
                    yield ToolCallStarted(block_id, item.id, item.name)
                    serialized = json.dumps(dict(item.arguments), separators=(",", ":"), sort_keys=True)
                    yield ToolCallArgumentsDelta(block_id, item.id, serialized)
                    yield ToolCallCompleted(block_id, item.id, item.arguments)
            if response.get("midstream_error"):
                raise ProviderUnavailableError("deterministic mock midstream failure", provider=execution.provider_instance_name)
            yield ResponseCompleted(FinishReason(response.get("finish_reason", "end_turn")), response.get("stop_sequence"), self._usage(response))
        finally:
            await self._release(index)

    async def execute_management_command(self, command_name: str, payload: Any, context: ManagementCommandContext) -> Any:
        async with self._lock:
            if command_name == "replace_scenario":
                self._scenario = parse_scenario(payload["scenario"]); self._revision += 1; self._cursors.clear(); self._hits.clear(); self._queue.clear()
                return {"revision": self._revision}
            if command_name == "enqueue_response":
                _validate_response(_require_json_mapping(payload["response"], "queued response"))
                self._queue.append(_freeze(payload["response"])); return {"queued": len(self._queue)}
            if command_name == "inspect_captured_requests":
                return {"requests": [_plain(item) for item in self._captures]}
            if command_name == "reset_state":
                self._call_index = 0; self._cursors.clear(); self._hits.clear(); self._queue.clear(); self._captures.clear(); self._in_flight.clear(); return {"revision": self._revision}
            pending = [{"id": item["id"], "remaining": max(0, item["count"] - self._hits.get(item.get("rule_id", item["id"]), 0))} for item in self._scenario.expectations]
            if command_name == "inspect_pending_expectations":
                return {"expectations": pending}
            if command_name == "verify_expectations":
                return {"satisfied": not any(item["remaining"] for item in pending), "expectations": pending}
        raise ValueError("unknown deterministic mock command")


class DeterministicMockFactory(ProviderFactory):
    def create(self, config: ProviderInstanceConfig) -> CompletionGateway:
        return DeterministicMockGateway(config)
