"""Fixtures for the cross-package composition tests.

Everything here needs two or more Wardhook packages installed at once, which is
exactly what the per-package suites are forbidden from assuming. These run only
in the combined environment (`make check`), never in a solo install.
"""

from __future__ import annotations

from typing import Any

import pytest

USAGE = {
    "input_tokens": 4200,
    "output_tokens": 180,
    "total_tokens": 4380,
    "input_token_details": {"cache_read": 3800},
}
METADATA = {"model_name": "claude-opus-5"}

POLICY = (
    "Section 4 -- Storm and flood damage. Storm damage claims carry a 500 "
    "excess. Flood damage carries a 1000 excess and requires a loss adjuster."
)


@pytest.fixture
def make_model():
    """Return a factory for a tool-capable fake model reporting real usage.

    The usage numbers matter: telemetry prices the uncached remainder, so a
    model that reports no cache detail cannot exercise the cost path at all.
    """

    def build(*replies: str, tool_call: tuple[str, dict[str, Any]] | None = None):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        class ToolCallingFake(GenericFakeChatModel):
            """A fake model that accepts ``bind_tools``."""

            def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
                """Accept the binding and return this model unchanged."""
                return self

        messages = []
        if tool_call is not None:
            name, arguments = tool_call
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": arguments, "id": "call-1"}],
                    usage_metadata=USAGE,
                    response_metadata=METADATA,
                )
            )
        messages.extend(
            AIMessage(content=reply, usage_metadata=USAGE, response_metadata=METADATA)
            for reply in replies
        )
        return ToolCallingFake(messages=iter(messages))

    return build


@pytest.fixture
def policy_store():
    """A vector store holding the policy wording the agent cites."""
    from wardhook.core import InMemoryVectorStore, chunk_text

    store = InMemoryVectorStore()
    store.add(chunk_text(POLICY, "policy.md"))
    return store


@pytest.fixture
def lookup_policy():
    """A documented tool the RBAC policy is allowed to permit or deny."""

    def lookup_policy(section: str) -> str:
        """Look up a section of the policy wording."""
        return f"Section {section}: storm damage carries a 500 excess."

    return lookup_policy


@pytest.fixture
def issue_refund():
    """A tool no role in these tests is permitted to call."""

    def issue_refund(claim_id: str) -> str:
        """Issue a refund against a claim."""
        return f"Refunded {claim_id}."

    return issue_refund
