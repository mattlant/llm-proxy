from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UpstreamHttpError(Exception):
    """Provider-internal HTTP failure, mapped before crossing the gateway boundary."""

    status_code: int
    body: str
    provider: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.body)
