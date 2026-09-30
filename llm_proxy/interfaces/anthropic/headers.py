from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from llm_proxy.domain.errors import InvalidCompletionRequest

_VERSION_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class AnthropicHeaders:
    version: str
    beta: tuple[str, ...]
    api_key: str | None
    bearer_token: str | None
    content_type: str

    @property
    def token(self) -> str | None:
        if self.api_key is not None and self.bearer_token is not None:
            if self.api_key != self.bearer_token:
                raise InvalidCompletionRequest("conflicting authentication credentials")
            return self.api_key
        return self.api_key or self.bearer_token


def extract_anthropic_headers(headers: Mapping[str, str]) -> AnthropicHeaders:
    values = {key.lower(): value for key, value in headers.items()}
    version = values.get("anthropic-version", "").strip()
    if not _VERSION_PATTERN.fullmatch(version):
        raise InvalidCompletionRequest("anthropic-version must be a YYYY-MM-DD version")

    content_type = values.get("content-type", "")
    if not content_type.lower().split(";", 1)[0].strip() == "application/json":
        raise InvalidCompletionRequest("Content-Type must be application/json")

    authorization = values.get("authorization", "").strip()
    bearer_token = None
    if authorization:
        scheme, _, credential = authorization.partition(" ")
        if scheme.lower() != "bearer" or not credential.strip():
            raise InvalidCompletionRequest("Authorization must use Bearer authentication")
        bearer_token = credential.strip()

    beta = tuple(dict.fromkeys(value.strip() for value in values.get("anthropic-beta", "").split(",") if value.strip()))
    api_key = values.get("x-api-key", "").strip() or None
    return AnthropicHeaders(version=version, beta=beta, api_key=api_key, bearer_token=bearer_token, content_type=content_type)
