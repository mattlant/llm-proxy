import logging
import re

import pytest

from llm_proxy.configuration.models import FileLoggingConfig, ServerConfig
from llm_proxy.observability import operational_logging
from llm_proxy.observability.operational_logging import (
    ExecutionMode, OperationalLogger, begin_operational_request, configure_logging,
    LoggingConfigurationError, summarize_effective_parameters,
)


def config(level: str, transport_debug: bool = False) -> ServerConfig:
    return ServerConfig("127.0.0.1", 1, 0, 1, level, transport_debug=transport_debug)


@pytest.fixture(autouse=True)
def cleanup_file_handler():
    yield
    operational_logging._remove_active_file_handler()


def test_enabled_file_logging_creates_timestamped_handler_and_preserves_console(tmp_path):
    root = logging.getLogger()
    console_handlers = list(root.handlers)
    configure_logging(ServerConfig("127.0.0.1", 1, 0, 1, "INFO", file_logging=FileLoggingConfig(True, str(tmp_path / "nested"))))
    files = list((tmp_path / "nested").glob("*.log"))
    assert len(files) == 1
    assert re.fullmatch(r"llm-proxy-\d{8}-\d{6}(?:-\d+)?\.log", files[0].name)
    assert operational_logging._active_file_handler in root.handlers
    assert all(handler in root.handlers for handler in console_handlers)
    logging.getLogger("llm-proxy").info("file-destination-proof")
    operational_logging._active_file_handler.flush()
    assert "file-destination-proof" in files[0].read_text()


def test_file_logging_collision_uses_suffix_without_overwriting(tmp_path, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            from datetime import datetime
            return datetime(2026, 8, 27, 10, 42, 15)

    monkeypatch.setattr(operational_logging, "datetime", FixedDateTime)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    first = log_dir / "llm-proxy-20260827-104215.log"
    first.write_text("previous\n")
    configure_logging(ServerConfig("127.0.0.1", 1, 0, 1, "INFO", file_logging=FileLoggingConfig(True, str(log_dir))))
    assert first.read_text() == "previous\n"
    assert (log_dir / "llm-proxy-20260827-104215-1.log").exists()


def test_file_logging_failure_is_typed_and_does_not_fallback(tmp_path):
    target = tmp_path / "not-a-directory"
    target.write_text("x")
    with pytest.raises(LoggingConfigurationError, match="not-a-directory"):
        configure_logging(ServerConfig("127.0.0.1", 1, 0, 1, "INFO", file_logging=FileLoggingConfig(True, str(target))))


def test_failed_replacement_keeps_previous_file_handler(tmp_path):
    configure_logging(ServerConfig("127.0.0.1", 1, 0, 1, "INFO", file_logging=FileLoggingConfig(True, str(tmp_path / "good"))))
    previous = operational_logging._active_file_handler
    target = tmp_path / "not-a-directory"
    target.write_text("x")
    with pytest.raises(LoggingConfigurationError):
        configure_logging(ServerConfig("127.0.0.1", 1, 0, 1, "INFO", file_logging=FileLoggingConfig(True, str(target))))
    assert operational_logging._active_file_handler is previous
    assert previous in logging.getLogger().handlers


def test_disabling_file_logging_removes_only_feature_handler(tmp_path):
    configure_logging(ServerConfig("127.0.0.1", 1, 0, 1, "INFO", file_logging=FileLoggingConfig(True, str(tmp_path))))
    handler = operational_logging._active_file_handler
    configure_logging(config("INFO"))
    assert handler not in logging.getLogger().handlers
    assert operational_logging._active_file_handler is None


@pytest.mark.parametrize(
    ("level", "httpx_level", "httpcore_level"),
    [("DEBUG", logging.INFO, logging.WARNING), ("INFO", logging.INFO, logging.WARNING), ("WARNING", logging.WARNING, logging.WARNING), ("ERROR", logging.ERROR, logging.ERROR)],
)
def test_configure_logging_caps_dependency_levels(level, httpx_level, httpcore_level) -> None:
    configure_logging(config(level))
    assert logging.getLogger("httpx").level == httpx_level
    assert logging.getLogger("httpcore").level == httpcore_level


def test_transport_debug_is_explicit_escape_hatch() -> None:
    configure_logging(config("ERROR", transport_debug=True))
    assert logging.getLogger("llm-proxy").level == logging.ERROR
    assert logging.getLogger("httpx").level == logging.DEBUG
    assert logging.getLogger("httpcore").level == logging.DEBUG


def test_parameter_summary_is_bounded_and_never_reveals_payload_values() -> None:
    payload = {"messages": [{"content": "private"}], "api_key": "secret", "stop": ["very private"], "temperature": 0.2, "extra": "private"}
    summary = summarize_effective_parameters(payload)
    assert "private" not in summary
    assert "secret" not in summary
    assert "messages" not in summary
    assert "api_key" not in summary
    assert "temperature=0.2" in summary
    assert "stop=strings(count=1,max_chars=12)" in summary


def test_safe_scalars_show_actual_values() -> None:
    payload = {"model": "ornith", "stop_reason": "end_of_turn", "content_type": "text"}
    summary = summarize_effective_parameters(payload)
    assert "model=ornith" in summary
    assert "stop_reason=end_of_turn" in summary
    assert "content_type=text" in summary
    assert "string(chars=" not in summary


def test_summary_truncation_is_explicit_and_owned() -> None:
    long_value = "x" * 3000
    payload = {"model": long_value, "temperature": 0.5}
    summary = summarize_effective_parameters(payload)
    assert len(summary) <= 2000
    assert "truncated" in summary
    assert "model=" in summary


def test_summary_count_truncation_still_works() -> None:
    payload = {f"param_{i:03d}": "value" for i in range(25)}
    summary = summarize_effective_parameters(payload)
    assert "truncated=5" in summary


def test_numeric_parameters_show_actual_values_not_types() -> None:
    payload = {"n": 5, "temperature": 0.7, "top_p": 0.9, "top_k": 50}
    summary = summarize_effective_parameters(payload)
    assert "n=5" in summary
    assert "n=int" not in summary
    assert "temperature=0.7" in summary
    assert "top_p=0.9" in summary
    assert "top_k=50" in summary


def test_lifecycle_records_are_single_line_and_correlated(caplog) -> None:
    context = begin_operational_request("openai")
    context.provider = "provider"
    context.mode = ExecutionMode.CANONICAL
    context.route = "canonical"
    context.requested_model = "requested"
    context.upstream_model = "upstream"
    context.stream = False
    logger = OperationalLogger()
    with caplog.at_level(logging.DEBUG, logger="llm-proxy"):
        logger.log_start(context)
        logger.log_decision(context)
        logger.log_completion(context, 200, "success")
    messages = [record.getMessage() for record in caplog.records if record.name == "llm-proxy"]
    assert [message.split()[0] for message in messages] == ["completion_started", "completion_decision", "completion_finished"]
    assert all(f"request_id={context.request_id}" in message for message in messages)
