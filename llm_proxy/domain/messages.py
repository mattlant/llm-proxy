from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .content import ContentBlock, ReasoningContent, TextContent, ToolCallContent, ToolResultContent


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python 3.10
    class StrEnum(str, Enum):
        pass


class MessageRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class DeveloperRoleMode(StrEnum):
    PRESERVE = "preserve"
    SYSTEM = "system"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: tuple[ContentBlock, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole")
        content = tuple(self.content)
        if not content:
            raise ValueError("message content must not be empty")
        if not all(
            isinstance(block, (TextContent, ReasoningContent, ToolCallContent, ToolResultContent))
            for block in content
        ):
            raise TypeError("message content contains an unsupported block")
        object.__setattr__(self, "content", content)
