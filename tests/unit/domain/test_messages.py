from __future__ import annotations

import pytest

from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import DeveloperRoleMode, Message, MessageRole


def test_message_uses_ordered_tuple_content_and_domain_roles():
    message = Message(MessageRole.ASSISTANT, [TextContent("one"), TextContent("two")])

    assert message.role is MessageRole.ASSISTANT
    assert message.content == (TextContent("one"), TextContent("two"))


def test_message_rejects_empty_content():
    with pytest.raises(ValueError):
        Message(MessageRole.USER, ())


def test_message_rejects_non_domain_content():
    with pytest.raises(TypeError):
        Message(MessageRole.USER, ("text",))  # type: ignore[arg-type]


def test_message_is_frozen():
    message = Message(MessageRole.SYSTEM, (TextContent("instructions"),))

    with pytest.raises(AttributeError):
        message.role = MessageRole.USER  # type: ignore[misc]


def test_developer_role_and_modes_are_domain_contracts():
    assert Message(MessageRole.DEVELOPER, (TextContent("instructions"),)).role is MessageRole.DEVELOPER
    assert tuple(DeveloperRoleMode) == (
        DeveloperRoleMode.PRESERVE,
        DeveloperRoleMode.SYSTEM,
        DeveloperRoleMode.REJECT,
    )
