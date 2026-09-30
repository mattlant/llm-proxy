from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenAIFunctionDefinition(OpenAIModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class OpenAIToolDefinition(OpenAIModel):
    type: Literal["function"]
    function: OpenAIFunctionDefinition


class OpenAINamedToolChoiceFunction(OpenAIModel):
    name: str = Field(min_length=1)


class OpenAINamedToolChoice(OpenAIModel):
    type: Literal["function"]
    function: OpenAINamedToolChoiceFunction


class OpenAIToolCallFunction(OpenAIModel):
    name: str = Field(min_length=1)
    arguments: str


class OpenAIToolCall(OpenAIModel):
    id: str = Field(min_length=1)
    type: Literal["function"]
    function: OpenAIToolCallFunction


class OpenAISystemMessage(OpenAIModel):
    role: Literal["system"]
    content: str


class OpenAIDeveloperMessage(OpenAIModel):
    role: Literal["developer"]
    content: str


class OpenAITextContentPart(OpenAIModel):
    type: Literal["text"]
    text: str


class OpenAIUserMessage(OpenAIModel):
    role: Literal["user"]
    content: str | Annotated[list[OpenAITextContentPart], Field(min_length=1)]


class OpenAIAssistantMessage(OpenAIModel):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self) -> OpenAIAssistantMessage:
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant message requires content or tool_calls")
        return self


class OpenAIToolMessage(OpenAIModel):
    role: Literal["tool"]
    tool_call_id: str = Field(min_length=1)
    content: str


OpenAIMessage = OpenAISystemMessage | OpenAIDeveloperMessage | OpenAIUserMessage | OpenAIAssistantMessage | OpenAIToolMessage


class OpenAIChatCompletionRequest(OpenAIModel):
    # OpenAI-compatible clients add optional request extensions at different
    # cadences. Unknown extensions must not prevent a valid completion from
    # reaching the configured provider.
    model_config = ConfigDict(extra="ignore")
    model: str = Field(min_length=1)
    messages: list[OpenAIMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    stream: bool = False
    tools: list[OpenAIToolDefinition] | None = None
    tool_choice: Literal["auto", "required", "none"] | OpenAINamedToolChoice | None = None
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] | None = None
