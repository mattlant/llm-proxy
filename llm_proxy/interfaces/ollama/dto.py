from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OllamaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OllamaFunction(OllamaModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class OllamaTool(OllamaModel):
    type: Literal["function"]
    function: OllamaFunction


class OllamaToolCallFunction(OllamaModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class OllamaToolCall(OllamaModel):
    id: str | None = None
    function: OllamaToolCallFunction


class OllamaMessage(OllamaModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[OllamaToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_tool_fields(self) -> OllamaMessage:
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("tool_calls are valid only on assistant messages")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only on tool messages")
        if self.role == "assistant" and not self.content and not self.tool_calls:
            raise ValueError("assistant message requires content or tool_calls")
        return self


class OllamaOptions(OllamaModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = Field(default=None, ge=0)
    num_predict: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None


class OllamaChatRequest(OllamaModel):
    model: str = Field(min_length=1)
    messages: list[OllamaMessage] = Field(min_length=1)
    tools: list[OllamaTool] | None = None
    stream: bool = True
    options: OllamaOptions | None = None


class OllamaGenerateRequest(OllamaModel):
    model: str = Field(min_length=1)
    prompt: str
    system: str | None = None
    stream: bool = True
    options: OllamaOptions | None = None


class OllamaShowRequest(OllamaModel):
    model: str = Field(min_length=1)
