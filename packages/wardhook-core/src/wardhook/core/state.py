"""The graph state passed between agent nodes.

:class:`AgentState` is the single dictionary that flows through every node of
an :class:`~wardhook.core.agent.AgentGraph`. LangGraph merges each node's
returned partial state into it, using the reducer annotations declared here.

Only ``messages`` is required. Every other key is optional and is populated
just by the nodes that are actually wired into a given graph: an agent built
without a retriever never sets ``context``, and one built without guardrails
never sets ``guardrail_events``.
"""

# NOTE: deliberately no `from __future__ import annotations` here.
# Under PEP 563 stringised annotations, typing_extensions.TypedDict cannot see
# through `NotRequired[...]` at class-creation time, so every key silently
# becomes required and LangGraph would demand a fully-populated state dict.
# PEP 604 unions and builtin generics used below are native from 3.10 anyway.

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

__all__ = ["AgentState", "Citation", "Principal"]


class Principal(TypedDict):
    """The identity on whose behalf an agent invocation is running.

    Guardrails read this to make role-based decisions -- most directly
    ``RoleBasedToolPolicy`` in ``wardhook-guardrails``, which matches
    ``roles`` against the tools a caller is permitted to invoke.

    Attributes:
        id: Stable identifier for the caller, recorded in audit trails.
        roles: Role names held by the caller.
        attributes: Free-form extra claims, such as tenant or region.
    """

    id: str
    roles: list[str]
    attributes: NotRequired[dict[str, Any]]


class Citation(TypedDict):
    """A single source attribution for retrieved context.

    Retrieval produces these as structured records rather than leaving the
    model to write source names into prose, so a caller can render or verify
    citations without parsing free text.

    Attributes:
        source: Where the chunk came from, usually a file path or URL.
        chunk_index: Position of the chunk within its source document.
        score: Similarity score against the query, higher being closer.
        text: The chunk text that was supplied to the model.
        metadata: Any additional metadata carried from the loader.
    """

    source: str
    chunk_index: int
    score: float
    text: str
    metadata: NotRequired[dict[str, Any]]


class AgentState(TypedDict):
    """State threaded through every node of an agent graph.

    Attributes:
        messages: The conversation so far. Annotated with LangGraph's
            ``add_messages`` reducer, so nodes return only the messages they
            add and LangGraph appends them.
        run_id: Unique id for this invocation, correlating traces with audit
            records.
        principal: Caller identity, if one was supplied.
        context: Retrieved chunks passed to the model this turn.
        citations: Source attributions for ``context``.
        guardrail_events: One record per guardrail decision, in the order the
            decisions were made. Annotated with :func:`operator.add` so records
            accumulate across nodes rather than the last node's list replacing
            everything before it. Text is excluded by design.
        blocked: ``True`` if a guardrail halted the run.
        block_reason: Why the run was halted, when ``blocked`` is ``True``.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    run_id: NotRequired[str]
    principal: NotRequired[Principal | None]
    context: NotRequired[list[Citation]]
    citations: NotRequired[list[Citation]]
    guardrail_events: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    blocked: NotRequired[bool]
    block_reason: NotRequired[str | None]
