from __future__ import annotations

import codecs
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SseRecord:
    event: str | None
    data: str


class SseDecoder:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buffer = ""
        self._event: str | None = None
        self._data_lines: list[str] = []

    def feed(self, chunk: bytes) -> tuple[SseRecord, ...]:
        if not isinstance(chunk, bytes):
            raise TypeError("SSE chunks must be bytes")
        self._buffer += self._decoder.decode(chunk, final=False)
        return self._consume_lines()

    def finish(self) -> tuple[SseRecord, ...]:
        self._buffer += self._decoder.decode(b"", final=True)
        records = list(self._consume_lines())
        if self._buffer:
            self._process_line(self._buffer)
            self._buffer = ""
        record = self._emit_record()
        if record is not None:
            records.append(record)
        return tuple(records)

    def _consume_lines(self) -> tuple[SseRecord, ...]:
        records: list[SseRecord] = []
        while True:
            line_end = self._line_end(self._buffer)
            if line_end is None:
                break
            index, width = line_end
            line = self._buffer[:index]
            self._buffer = self._buffer[index + width:]
            if line == "":
                record = self._emit_record()
                if record is not None:
                    records.append(record)
            else:
                self._process_line(line)
        return tuple(records)

    @staticmethod
    def _line_end(value: str) -> tuple[int, int] | None:
        for index, character in enumerate(value):
            if character == "\n":
                return index, 1
            if character == "\r":
                if index + 1 == len(value):
                    return None
                return index, 2 if value[index + 1] == "\n" else 1
        return None

    def _process_line(self, line: str) -> None:
        if line.startswith(":"):
            return
        if ":" in line:
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""
        if field == "event":
            self._event = value
        elif field == "data":
            self._data_lines.append(value)

    def _emit_record(self) -> SseRecord | None:
        if self._event is None and not self._data_lines:
            return None
        record = SseRecord(self._event, "\n".join(self._data_lines))
        self._event = None
        self._data_lines = []
        return record
