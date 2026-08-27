"""Shared pytest fixtures for the wardhook-core test suite.

Every fixture here is offline. The suite must pass with no API key set and no
network access, which is what lets CI run it on every push and lets a
contributor run it immediately after cloning.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from wardhook.core.rag.chunking import chunk_text
from wardhook.core.rag.store import InMemoryVectorStore


class ToolCallingFake(GenericFakeChatModel):
    """A fake chat model that accepts ``bind_tools``.

    ``GenericFakeChatModel`` raises ``NotImplementedError`` from ``bind_tools``,
    so it cannot drive the tool path. This subclass accepts the binding and
    returns itself, replaying whatever scripted messages it was given -- enough
    to exercise the full model/tool loop without a provider.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Accept a tool binding and return this model unchanged.

        Args:
            tools: Ignored; the scripted messages already decide the calls.
            **kwargs: Ignored.

        Returns:
            This model.
        """
        return self


@pytest.fixture
def fake_model() -> GenericFakeChatModel:
    """A model that replies once with a fixed string."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="Hello there.")]))


@pytest.fixture
def make_model():
    """Return a factory building a fake model from scripted replies."""

    def _make(*replies: str | AIMessage) -> GenericFakeChatModel:
        messages = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
        return GenericFakeChatModel(messages=iter(messages))

    return _make


@pytest.fixture
def make_tool_model():
    """Return a factory building a tool-capable fake model."""

    def _make(*replies: AIMessage) -> ToolCallingFake:
        return ToolCallingFake(messages=iter(list(replies)))

    return _make


@pytest.fixture
def sample_store() -> InMemoryVectorStore:
    """A small populated vector store covering two clearly distinct topics."""
    store = InMemoryVectorStore()
    store.add(
        chunk_text(
            "Storm damage claims carry a 500 excess under section 4 of the policy.",
            "policy.md",
        )
    )
    store.add(chunk_text("Our office is open from 9am to 5pm on weekdays.", "hours.md"))
    return store


@pytest.fixture
def echo_tool():
    """A trivial documented tool for exercising the tool-calling path."""

    def lookup_account(account_id: str) -> str:
        """Look up an account by its identifier."""
        return f"Account {account_id} is active."

    return lookup_account
