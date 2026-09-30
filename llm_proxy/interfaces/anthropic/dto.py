from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnthropicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnthropicEphemeralCacheControl(AnthropicModel):
    type: Literal["ephemeral"]


class AnthropicTextBlock(AnthropicModel):
    type: Literal["text"]
    text: str = Field(min_length=1)
    cache_control: AnthropicEphemeralCacheControl | None = None


class AnthropicToolUseBlock(AnthropicModel):
    type: Literal["tool_use"]
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: dict[str, Any]


class AnthropicToolResultBlock(AnthropicModel):
    type: Literal["tool_result"]
    tool_use_id: str = Field(min_length=1)
    content: str | list[AnthropicTextBlock]
    is_error: bool = False
    cache_control: AnthropicEphemeralCacheControl | None = None


class AnthropicThinkingBlock(AnthropicModel):
    type: Literal["thinking"]
    thinking: str = Field(min_length=1)
    signature: str | None = None


class AnthropicRedactedThinkingBlock(AnthropicModel):
    type: Literal["redacted_thinking"]
    data: str = Field(min_length=1)


class AnthropicUnsupportedImageBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["image"]


class AnthropicUnsupportedDocumentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["document"]


AnthropicContentBlock = Annotated[
    AnthropicTextBlock
    | AnthropicToolUseBlock
    | AnthropicToolResultBlock
    | AnthropicThinkingBlock
    | AnthropicRedactedThinkingBlock
    | AnthropicUnsupportedImageBlock
    | AnthropicUnsupportedDocumentBlock,
    Field(discriminator="type"),
]


class AnthropicSystemTextBlock(AnthropicModel):
    type: Literal["text"]
    text: str = Field(min_length=1)
    cache_control: AnthropicEphemeralCacheControl | None = None


class AnthropicMessage(AnthropicModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicToolDefinition(AnthropicModel):
    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any]


class AnthropicAutoToolChoice(AnthropicModel):
    type: Literal["auto"]


class AnthropicAnyToolChoice(AnthropicModel):
    type: Literal["any"]


class AnthropicNoToolChoice(AnthropicModel):
    type: Literal["none"]


class AnthropicNamedToolChoice(AnthropicModel):
    type: Literal["tool"]
    name: str = Field(min_length=1)


AnthropicToolChoice = Annotated[
    AnthropicAutoToolChoice | AnthropicAnyToolChoice | AnthropicNoToolChoice | AnthropicNamedToolChoice,
    Field(discriminator="type"),
]


class AnthropicJsonOutputFormat(AnthropicModel):
    type: Literal["json_schema"]
    json_schema: dict[str, Any] = Field(alias="schema")

    @model_validator(mode="after")
    def validate_json_schema(self) -> AnthropicJsonOutputFormat:
        if not _is_json_value(self.json_schema):
            raise ValueError("schema must contain only JSON-compatible values")
        return self


class AnthropicOutputConfig(AnthropicModel):
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    format: AnthropicJsonOutputFormat | None = None


class AnthropicAdaptiveThinkingConfig(AnthropicModel):
    type: Literal["adaptive"]
    display: Literal["summarized", "omitted"]


class AnthropicEnabledThinkingConfig(AnthropicModel):
    type: Literal["enabled"]
    budget_tokens: int = Field(gt=0)
    display: Literal["summarized", "omitted"]


class AnthropicDisabledThinkingConfig(AnthropicModel):
    type: Literal["disabled"]
    display: Literal["summarized", "omitted"] = "omitted"


AnthropicThinkingConfig = Annotated[
    AnthropicAdaptiveThinkingConfig | AnthropicEnabledThinkingConfig | AnthropicDisabledThinkingConfig,
    Field(discriminator="type"),
]


class AnthropicClearThinkingContextEdit(AnthropicModel):
    type: Literal["clear_thinking_20251015"]
    keep: Literal["all"]


AnthropicContextEdit = Annotated[
    AnthropicClearThinkingContextEdit,
    Field(discriminator="type"),
]


class AnthropicContextManagement(AnthropicModel):
    edits: list[AnthropicContextEdit] = Field(min_length=1)


class AnthropicMessagesRequest(AnthropicModel):
    model: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    system: str | list[AnthropicSystemTextBlock] | None = None
    messages: list[AnthropicMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = Field(default=None, ge=0)
    stop_sequences: list[str] = Field(default_factory=list)
    stream: bool = False
    tools: list[AnthropicToolDefinition] = Field(default_factory=list)
    tool_choice: AnthropicToolChoice | None = None
    metadata: dict[str, Any] | None = None
    output_config: AnthropicOutputConfig | None = None
    thinking: AnthropicThinkingConfig | None = None
    context_management: AnthropicContextManagement | None = None

    @model_validator(mode="after")
    def validate_unique_tools(self) -> AnthropicMessagesRequest:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if self.metadata is not None and not _is_json_value(self.metadata):
            raise ValueError("metadata must contain only JSON-compatible values")
        if isinstance(self.thinking, AnthropicEnabledThinkingConfig) and self.thinking.budget_tokens >= self.max_tokens:
            raise ValueError("thinking budget_tokens must be less than max_tokens")
        return self


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False
