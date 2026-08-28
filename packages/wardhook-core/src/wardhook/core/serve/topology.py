"""Read an agent's graph topology as plain, content-free data.

The shape of a Wardhook agent is already fully described by the graph it
compiled. :func:`read_topology` reads that description back -- every node, every
edge, which edges are conditional and what their labels are -- and returns it as
data this package owns, so nothing downstream has to know what a LangGraph
``Graph`` object looks like.

**It is configuration-accurate, and that is the whole point.**
:meth:`~wardhook.core.agent.AgentGraph._build` only adds the nodes the
configuration needs, so an agent with no retriever genuinely has no ``retrieve``
node and the topology genuinely has no box for one. A hand-drawn diagram would
lose exactly that property, which is the one thing a topology view is for.

**Nothing here is guessed.** Node names and edge labels are reported as the
graph declares them. What a custom node *means* is not inferred, because the
graph does not say -- and a diagram that invents semantics is worse than one
that admits it only knows names.

**Every read is defensive.** The served object may be a plain callable with an
``.invoke()`` method and no graph at all, which is a legitimate target for
:func:`~wardhook.core.serve.app.create_app`. That case reports
``available=False`` with a reason a human can act on, never an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Topology", "TopologyEdge", "TopologyNode", "read_topology"]


@dataclass(frozen=True, slots=True)
class TopologyNode:
    """One node in an agent's compiled graph.

    Attributes:
        key: The node's identifier, unique within the graph. LangGraph uses
            ``__start__`` and ``__end__`` for the two terminals.
        name: The node's display name. Usually equal to ``key``.
    """

    key: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this node.

        Returns:
            A dict with ``id`` and ``name``. The key is published as ``id``
            because that is what it is to a consumer drawing the graph.
        """
        return {"id": self.key, "name": self.name}


@dataclass(frozen=True, slots=True)
class TopologyEdge:
    """One directed edge between two nodes.

    Attributes:
        source: Key of the node the edge leaves.
        target: Key of the node the edge enters.
        label: The branch name for a conditional edge, such as ``"blocked"``,
            or ``None`` for an unconditional one.
        conditional: Whether a router decides at run time if this edge is taken.
    """

    source: str
    target: str
    label: str | None = None
    conditional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this edge.

        Returns:
            A dict with the source, target, label and conditional flag.
        """
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "conditional": self.conditional,
        }


@dataclass(frozen=True, slots=True)
class Topology:
    """An agent's graph structure, or an honest account of why it is unavailable.

    Attributes:
        nodes: Every node, in the order the graph reported them.
        edges: Every edge, in the order the graph reported them.
        mermaid: Mermaid source for the same graph, when the graph can produce
            it. Carried so a caller can render the diagram in a tool of their
            own choosing; Wardhook does not need it to draw the topology itself.
        available: Whether a topology could be read at all.
        reason: Why it could not be, when ``available`` is ``False``.
    """

    nodes: tuple[TopologyNode, ...] = ()
    edges: tuple[TopologyEdge, ...] = ()
    mermaid: str | None = None
    available: bool = True
    reason: str | None = None

    @property
    def keys(self) -> tuple[str, ...]:
        """The node keys, in graph order."""
        return tuple(node.key for node in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the whole topology.

        Returns:
            A dict with ``available``, ``reason``, ``nodes``, ``edges`` and
            ``mermaid``. Nothing in it derives from an agent's input or output;
            it describes configuration only.
        """
        return {
            "available": self.available,
            "reason": self.reason,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "mermaid": self.mermaid,
        }


def _unavailable(reason: str) -> Topology:
    """Build an empty topology carrying the reason it is empty.

    Args:
        reason: A human-readable explanation, phrased so a reader knows whether
            they need to do anything about it.

    Returns:
        A topology with ``available=False``.
    """
    return Topology(available=False, reason=reason)


def _mermaid_of(graph: Any) -> str | None:
    """Render Mermaid source for a graph, if it can.

    Args:
        graph: The graph object returned by ``get_graph()``.

    Returns:
        The Mermaid source, or ``None`` when the graph cannot produce it.
        Failure here is not an error: the diagram is an export convenience, and
        losing it must not cost the caller the topology itself.
    """
    drawer = getattr(graph, "draw_mermaid", None)
    if not callable(drawer):
        return None
    try:
        return str(drawer())
    # Broad on purpose: the diagram is an export convenience, so a renderer
    # that trips must not cost the caller the topology it came with.
    except Exception:
        return None


def read_topology(agent: Any) -> Topology:
    """Read an agent's graph structure.

    The chain ``agent.graph`` -> ``.get_graph()`` -> ``.nodes`` / ``.edges`` is
    walked with :func:`getattr` at every step, mirroring
    :meth:`~wardhook.core.agent.AgentGraph.trace`. Anything that does not expose
    a graph is reported rather than raising, because serving a plain callable is
    a supported use of this package.

    Args:
        agent: Any object. An :class:`~wardhook.core.agent.AgentGraph` yields a
            full topology; anything else yields ``available=False``.

    Returns:
        The topology, or an unavailable one naming why.

    Example:
        >>> from langchain_core.language_models.fake_chat_models import (
        ...     GenericFakeChatModel,
        ... )
        >>> from langchain_core.messages import AIMessage
        >>> from wardhook.core import AgentGraph
        >>> model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
        >>> topology = read_topology(AgentGraph(model=model))
        >>> topology.keys
        ('__start__', 'call_model', '__end__')
        >>> read_topology(object()).reason
        'object exposes no .graph attribute, so it has no topology to show.'
    """
    graph_holder = getattr(agent, "graph", None)
    if graph_holder is None:
        return _unavailable(
            f"{type(agent).__name__} exposes no .graph attribute, so it has no topology to show."
        )

    getter = getattr(graph_holder, "get_graph", None)
    if not callable(getter):
        return _unavailable(
            f"{type(agent).__name__}.graph has no get_graph() method, so its "
            f"structure cannot be read."
        )

    try:
        graph = getter()
    # Broad on purpose: this introspects a third-party compiled object, and a
    # dashboard must degrade to an honest message rather than a 500.
    except Exception as exc:
        return _unavailable(f"get_graph() raised {type(exc).__name__}: {exc}")

    raw_nodes = getattr(graph, "nodes", None) or {}
    raw_edges = getattr(graph, "edges", None) or ()

    nodes = tuple(
        TopologyNode(key=str(key), name=str(getattr(node, "name", key)))
        for key, node in dict(raw_nodes).items()
    )
    edges = tuple(
        TopologyEdge(
            source=str(getattr(edge, "source", "")),
            target=str(getattr(edge, "target", "")),
            label=None if (data := getattr(edge, "data", None)) is None else str(data),
            conditional=bool(getattr(edge, "conditional", False)),
        )
        for edge in raw_edges
    )
    return Topology(nodes=nodes, edges=edges, mermaid=_mermaid_of(graph))
