from __future__ import annotations

import contextvars
import logging
from datetime import datetime
from pathlib import Path
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .payload_trace import current_trace_context


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class ExecutionMode(StrEnum):
    CANONICAL = "canonical"
    PRESERVE = "preserve"


class LoggingConfigurationError(RuntimeError):
    pass


_active_file_handler: logging.FileHandler | None = None


def _create_launch_file_handler(directory: str) -> logging.FileHandler:
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for suffix in range(1000):
            suffix_text = "" if suffix == 0 else f"-{suffix}"
            candidate = path / f"llm-proxy-{timestamp}{suffix_text}.log"
            try:
                handler = logging.FileHandler(candidate, mode="x", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                return handler
            except FileExistsError:
                continue
        raise OSError("unable to allocate a unique launch log filename")
    except OSError as error:
        raise LoggingConfigurationError(f"unable to configure file logging directory '{directory}': {error}") from error


def _remove_active_file_handler() -> None:
    global _active_file_handler
    if _active_file_handler is None:
        return
    root = logging.getLogger()
    root.removeHandler(_active_file_handler)
    _active_file_handler.close()
    _active_file_handler = None


@dataclass(slots=True)
class OperationalRequestContext:
    request_id: str
    interface: str
    started_ns: int = field(default_factory=time.monotonic_ns)
    trace_transaction_id: str | None = None
    requested_model: str | None = None
    upstream_model: str | None = None
    provider: str | None = None
    mode: ExecutionMode | None = None
    stream: bool | None = None
    route: str | None = None
    policies: tuple[tuple[str, tuple[str, ...]], ...] = ()
    fallback_reason: str | None = None
    advisory_controls: tuple[str, ...] = ()
    semantic_handling: tuple[str, ...] = ()
    started_emitted: bool = False
    finished_emitted: bool = False


_context: contextvars.ContextVar[OperationalRequestContext | None] = contextvars.ContextVar(
    "operational_request_context", default=None
)


def begin_operational_request(interface: str) -> OperationalRequestContext:
    return OperationalRequestContext(uuid.uuid4().hex, interface)


def bind_operational_request(context: OperationalRequestContext):
    return _context.set(context)


def current_operational_request() -> OperationalRequestContext | None:
    return _context.get()


def reset_operational_request(token: contextvars.Token[OperationalRequestContext | None]) -> None:
    _context.reset(token)


def _safe_identifier(value: Any) -> str:
    text = str(value).replace("\\", "\\\\")
    text = "".join(ch if ch.isprintable() and not ch.isspace() else "_" for ch in text)
    return text


def _format_fields(fields: Mapping[str, Any]) -> str:
    return " ".join(f"{key}={_safe_identifier(value)}" for key, value in fields.items() if value is not None)


_SENSITIVE = ("authorization", "token", "secret", "password", "cookie", "api_key", "apikey", "access_key", "credential")
_SAFE_SCALAR = {"model", "stop_reason", "content_type", "response_format"}
_EXCLUDED = {"messages", "tools"}


def _summary(name: str, value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    from llm_proxy.domain.parameters import CORE_PARAMETER_CATALOG
    if name in CORE_PARAMETER_CATALOG.names and CORE_PARAMETER_CATALOG.is_numeric(name) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if name in _SAFE_SCALAR and isinstance(value, str):
        return value
    if name == "stop" and isinstance(value, list):
        return f"strings(count={len(value)},max_chars={max((len(item) for item in value if isinstance(item, str)), default=0)})"
    if isinstance(value, str):
        return f"string(chars={len(value)})"
    if isinstance(value, list):
        return f"list(count={len(value)})"
    if isinstance(value, Mapping):
        return f"object(keys={len(value)})"
    return type(value).__name__


_MAX_SUMMARY_CHARS = 2000


def summarize_effective_parameters(payload: Mapping[str, Any]) -> str:
    try:
        parts: list[str] = []
        skipped = 0
        for name in sorted(payload):
            folded = name.casefold()
            if name in _EXCLUDED or any(marker in folded for marker in _SENSITIVE):
                continue
            if len(parts) == 20:
                skipped += 1
                continue
            value = payload[name]
            if name == "stream_options" and isinstance(value, Mapping):
                parts.append(f"stream_options.include_usage={str(value.get('include_usage')).lower() if isinstance(value.get('include_usage'), bool) else _summary(name, value)}")
            elif name in {"format", "response_format", "tool_choice"}:
                parts.append(f"{name}={_summary(name, value)}")
            else:
                parts.append(f"{name}={_summary(name, value)}")
        joined = ",".join(parts)
        if skipped:
            joined += f",truncated={skipped}"
        if len(joined) > _MAX_SUMMARY_CHARS:
            joined = joined[:_MAX_SUMMARY_CHARS - len(",truncated=too_long")]
            joined += ",truncated=too_long"
        return joined
    except Exception:
        return "unavailable"


class OperationalLogger:
    def __init__(self) -> None:
        self._logger = logging.getLogger("llm-proxy")

    @staticmethod
    def _route_fields(context: OperationalRequestContext) -> dict[str, Any]:
        return {
            "request_id": context.request_id,
            "trace_transaction_id": context.trace_transaction_id,
            "interface": context.interface,
            "provider": context.provider,
            "mode": context.mode.value if context.mode else None,
            "route": context.route,
            "requested_model": context.requested_model,
            "upstream_model": context.upstream_model if context.upstream_model != context.requested_model else None,
            "stream": str(context.stream).lower() if context.stream is not None else None,
        }

    def log_start(self, context: OperationalRequestContext) -> None:
        if context.started_emitted:
            return
        context.started_emitted = True
        self._logger.info("completion_started %s", _format_fields(self._route_fields(context)))

    def log_decision(self, context: OperationalRequestContext) -> None:
        fields = self._route_fields(context)
        fields["policies"] = ";".join(name for name, _ in context.policies) or "none"
        fields["policy_overrides"] = ";".join(f"{name}:{','.join(overrides) or 'none'}" for name, overrides in context.policies) or "none"
        if context.mode is ExecutionMode.CANONICAL:
            fields["fallback_reason"] = context.fallback_reason
        if context.advisory_controls:
            fields["advisory_controls"] = ",".join(context.advisory_controls)
        if context.semantic_handling:
            fields["semantic_handling"] = ",".join(context.semantic_handling)
        self._logger.debug("completion_decision %s", _format_fields(fields))

    def log_effective_parameters(self, context: OperationalRequestContext | None, mode: ExecutionMode, payload: Mapping[str, Any]) -> None:
        if context is None:
            return
        fields = self._route_fields(context)
        fields["mode"] = mode.value
        fields["effective_parameters"] = summarize_effective_parameters(payload)
        self._logger.debug("completion_effective_parameters %s", _format_fields(fields))

    def log_completion(self, context: OperationalRequestContext, status_code: int, outcome: str) -> None:
        if not context.started_emitted or context.finished_emitted:
            return
        context.finished_emitted = True
        fields = self._route_fields(context)
        fields.update(status=status_code, outcome=outcome, duration_ms=max(0, (time.monotonic_ns() - context.started_ns) // 1_000_000))
        self._logger.info("completion_finished %s", _format_fields(fields))

    def log_capability(self, context, binding, outcome: str, duration_ms: int) -> None:
        fields = {"request_id": context.request_id, "trace_transaction_id": context.trace_transaction_id, "binding_id": binding.id, "extension_instance": binding.instance, "family": binding.family, "family_version": binding.version, "capability": binding.capability, "failure_mode": binding.on_failure.value, "outcome": outcome, "duration_ms": duration_ms}
        (self._logger.info if outcome == "success" else self._logger.warning)("extension_capability %s", _format_fields(fields))


def copy_trace_transaction_id(context: OperationalRequestContext) -> None:
    trace = current_trace_context()
    if trace is not None:
        context.trace_transaction_id = trace.transaction_id


def configure_logging(server_config: Any) -> None:
    global _active_file_handler
    level = logging._nameToLevel[server_config.log_level.upper()]
    logging.getLogger().setLevel(level)
    logging.getLogger("llm-proxy").setLevel(level)
    if server_config.transport_debug:
        httpx_level = httpcore_level = logging.DEBUG
    else:
        httpx_level = max(logging.INFO, level)
        httpcore_level = max(logging.WARNING, level)
    logging.getLogger("httpx").setLevel(httpx_level)
    logging.getLogger("httpcore").setLevel(httpcore_level)
    if server_config.file_logging.enabled:
        new_handler = _create_launch_file_handler(server_config.file_logging.directory)
        logging.getLogger().addHandler(new_handler)
        _remove_active_file_handler()
        _active_file_handler = new_handler
    else:
        _remove_active_file_handler()
