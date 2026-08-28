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
from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "Layout",
    "NodeBox",
    "Topology",
    "TopologyEdge",
    "TopologyNode",
    "layout",
    "read_topology",
    "render_svg",
]


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


# Layout geometry, in SVG user units. Chosen so a node box fits two lines of
# text: its name, and the per-node metrics the trace overlay writes underneath.
_NODE_W = 168.0
_NODE_H = 52.0
_H_GAP = 28.0
_V_GAP = 58.0
_MARGIN = 20.0
# Empty margin kept clear on each side of the boxes. Edges that cannot go
# straight down are routed through these gutters, which is what stops them
# from being drawn across a node box.
_GUTTER = 46.0

# LangGraph's two terminals, drawn differently because they are not work.
# Vertical half-gap between two ranks, and the horizontal offset a detouring
# edge uses when leaving or entering a box. Both stay inside empty space.
_BAND = _V_GAP / 2
_LANE = 46.0

_TERMINALS = frozenset({"__start__", "__end__"})


@dataclass(frozen=True, slots=True)
class NodeBox:
    """One node's position in a laid-out topology.

    Attributes:
        key: The node's key, matching :attr:`TopologyNode.key`.
        name: The node's display name.
        rank: How many nodes deep in the longest path from the start this is.
        x: Left edge, in SVG user units.
        y: Top edge, in SVG user units.
    """

    key: str
    name: str
    rank: int
    x: float
    y: float

    @property
    def centre_x(self) -> float:
        """Horizontal centre of the box."""
        return self.x + _NODE_W / 2

    @property
    def centre_y(self) -> float:
        """Vertical centre of the box."""
        return self.y + _NODE_H / 2


@dataclass(frozen=True, slots=True)
class Layout:
    """A topology with every node given a position.

    Attributes:
        boxes: One box per node, in the order the topology reported them.
        back_edges: Indices into the topology's edges that point backwards --
            the tool loop, for instance. They are routed around the side rather
            than through the diagram.
        width: Total canvas width, in SVG user units.
        height: Total canvas height, in SVG user units.
    """

    boxes: tuple[NodeBox, ...] = ()
    back_edges: frozenset[int] = frozenset()
    width: float = 0.0
    height: float = 0.0

    def box(self, key: str) -> NodeBox | None:
        """Find one node's box.

        Args:
            key: The node key to look for.

        Returns:
            The box, or ``None`` if the topology has no such node. An edge
            naming a node that does not exist is possible in a hand-built graph
            and must not crash the drawing.
        """
        return next((box for box in self.boxes if box.key == key), None)


def _depth_first_order(
    keys: tuple[str, ...], outgoing: dict[str, list[tuple[int, str]]]
) -> tuple[frozenset[int], list[str]]:
    """Classify back edges and produce a topological order.

    An edge is a back edge when it points at a node still open on the traversal
    stack -- the tool loop is exactly that. Removing them leaves a DAG, and the
    reverse of the finish order is a topological ordering of it, which is what
    makes the longest-path ranking below a single pass with no cycle guard.

    Args:
        keys: Node keys, in graph order. Traversal starts from the first and
            then from any node it did not reach, so an unreachable subgraph is
            still laid out rather than dropped.
        outgoing: Edges leaving each node, as ``(edge index, target)`` pairs.

    Returns:
        The set of back-edge indices, and the finish order (earliest first).
    """
    white, grey, black = 0, 1, 2
    colour = dict.fromkeys(keys, white)
    back: set[int] = set()
    finished: list[str] = []

    for root in keys:
        if colour[root] != white:
            continue
        colour[root] = grey
        stack: list[tuple[str, Iterator[tuple[int, str]]]] = [(root, iter(outgoing.get(root, ())))]
        while stack:
            node, pending = stack[-1]
            descended = False
            for index, target in pending:
                state = colour.get(target, black)
                if state == grey:
                    back.add(index)
                elif state == white:
                    colour[target] = grey
                    stack.append((target, iter(outgoing.get(target, ()))))
                    descended = True
                    break
            if not descended:
                colour[node] = black
                finished.append(node)
                stack.pop()

    return frozenset(back), finished


def layout(topology: Topology) -> Layout:
    """Assign every node a position, in ranks flowing top to bottom.

    The rank of a node is the length of the longest path reaching it, so a node
    always sits below everything that can precede it. That is the property that
    makes the picture readable: the eye follows the run.

    This is arithmetic, and deliberately so. No model is asked to lay anything
    out, describe anything, or infer what a node means; the graph already knows
    its own structure and this only draws what it says.

    Args:
        topology: The topology to lay out.

    Returns:
        The layout. An unavailable or empty topology yields an empty one.

    Example:
        >>> nodes = (TopologyNode("a", "a"), TopologyNode("b", "b"))
        >>> edges = (TopologyEdge("a", "b"), TopologyEdge("b", "a"))
        >>> placed = layout(Topology(nodes=nodes, edges=edges))
        >>> [(box.key, box.rank) for box in placed.boxes]
        [('a', 0), ('b', 1)]
        >>> sorted(placed.back_edges)
        [1]
    """
    keys = topology.keys
    if not keys:
        return Layout()

    outgoing: dict[str, list[tuple[int, str]]] = {key: [] for key in keys}
    for index, edge in enumerate(topology.edges):
        if edge.source in outgoing:
            outgoing[edge.source].append((index, edge.target))

    back, finished = _depth_first_order(keys, outgoing)

    rank = dict.fromkeys(keys, 0)
    for node in reversed(finished):
        for index, target in outgoing[node]:
            if index not in back and target in rank:
                rank[target] = max(rank[target], rank[node] + 1)

    columns: dict[int, list[str]] = {}
    for key in keys:
        columns.setdefault(rank[key], []).append(key)

    widest = max(len(row) for row in columns.values())
    row_span = widest * _NODE_W + (widest - 1) * _H_GAP
    names = {node.key: node.name for node in topology.nodes}

    boxes: list[NodeBox] = []
    for depth, row in sorted(columns.items()):
        span = len(row) * _NODE_W + (len(row) - 1) * _H_GAP
        left = _MARGIN + _GUTTER + (row_span - span) / 2
        for column, key in enumerate(row):
            boxes.append(
                NodeBox(
                    key=key,
                    name=names[key],
                    rank=depth,
                    x=left + column * (_NODE_W + _H_GAP),
                    y=_MARGIN + depth * (_NODE_H + _V_GAP),
                )
            )

    ordered = tuple(sorted(boxes, key=lambda box: keys.index(box.key)))
    return Layout(
        boxes=ordered,
        back_edges=back,
        # A gutter each side: back edges route right, edges that skip a rank
        # route left, and neither may be clipped.
        width=row_span + 2 * (_MARGIN + _GUTTER),
        height=len(columns) * _NODE_H + (len(columns) - 1) * _V_GAP + 2 * _MARGIN,
    )


def _edge_geometry(
    source: NodeBox, target: NodeBox, back: bool, width: float, index: int
) -> tuple[str, float, float, str]:
    """Route one edge, and say where its label goes.

    Three routes, because one route would be unreadable:

    * **Down one rank** -- a straight drop, the common case.
    * **Skipping ranks** -- out through the empty band below the source, along
      the left gutter, and back in through the band above the target. The
      ``blocked`` edge out of ``guard_input`` takes this route, and it is the
      edge a reader most wants to see clearly.
    * **Backwards** -- the same shape along the right gutter. The tool loop
      takes this route.

    The bands between ranks and the two gutters are empty by construction, so a
    detour through them cannot cross a node box. A single curve drawn box to box
    can and does, which is what this replaces.

    Args:
        source: Box the edge leaves.
        target: Box the edge enters.
        back: Whether this edge points backwards.
        width: Canvas width, used to find the right gutter.
        index: Position of this edge in the graph, used only to stagger the
            band a detour runs along. Two detours sharing one band would
            otherwise draw a few pixels of the same line; spreading them by
            index is deterministic, which matters more here than optimal,
            because the same configuration must always render identically.

    Returns:
        The SVG path data, the x and y for the edge's label, and the
        ``text-anchor`` that label should use.
    """
    if target.rank == source.rank + 1:
        drop_y = source.y + _NODE_H
        path = f"M{source.centre_x:.1f},{drop_y:.1f} L{target.centre_x:.1f},{target.y:.1f}"
        # Nudged off the line rather than centred on it, so a label never sits
        # on top of the edge it names.
        return (
            path,
            (source.centre_x + target.centre_x) / 2 + 6,
            (drop_y + target.y) / 2,
            "start",
        )

    if back or target.rank <= source.rank:
        gutter, lane = width - _MARGIN, _LANE
    else:
        gutter, lane = _MARGIN, -_LANE

    stagger = (index % 5 - 2) * 6.0
    exit_x = source.centre_x + lane
    entry_x = target.centre_x + lane
    band_out = source.y + _NODE_H + _BAND + stagger
    band_in = max(_MARGIN / 2, target.y - _BAND + stagger)

    path = (
        f"M{exit_x:.1f},{source.y + _NODE_H:.1f} "
        f"L{exit_x:.1f},{band_out:.1f} "
        f"L{gutter:.1f},{band_out:.1f} "
        f"L{gutter:.1f},{band_in:.1f} "
        f"L{entry_x:.1f},{band_in:.1f} "
        f"L{entry_x:.1f},{target.y:.1f}"
    )
    return path, (exit_x + gutter) / 2, band_out - 5, "middle"


def render_svg(topology: Topology) -> str:
    """Draw a topology as one inline SVG element.

    The diagram is built here rather than by a JavaScript renderer fetched from
    a CDN. Three things follow, all of them the point:

    * The page keeps working with no network, which is the environment this
      project exists for.
    * The drawing is ordinary Python and is therefore covered by the same test
      gate as everything else.
    * Each node carries a ``data-node`` attribute, which is the whole mechanism
      the trace overlay needs -- colouring a box by what it cost is setting an
      attribute on an element that is already there.

    Args:
        topology: The topology to draw.

    Returns:
        An ``<svg>`` element, or an empty string when there is nothing to draw.
        The caller decides what to show instead; this function does not invent
        a picture for an agent that has no graph.

    Example:
        >>> nodes = (TopologyNode("a", "a"), TopologyNode("b", "b"))
        >>> svg = render_svg(Topology(nodes=nodes, edges=(TopologyEdge("a", "b"),)))
        >>> svg.startswith("<svg")
        True
        >>> 'data-node="a"' in svg
        True
        >>> render_svg(Topology(available=False, reason="no graph"))
        ''
    """
    placed = layout(topology)
    if not placed.boxes:
        return ""

    paths: list[str] = []
    for index, edge in enumerate(topology.edges):
        source, target = placed.box(edge.source), placed.box(edge.target)
        if source is None or target is None:
            continue
        back = index in placed.back_edges
        data, label_x, label_y, anchor = _edge_geometry(source, target, back, placed.width, index)
        classes = "edge" + (" conditional" if edge.conditional else "")
        paths.append(f'<path class="{classes}" d="{data}" marker-end="url(#wh-arrow)"/>')
        if edge.label:
            paths.append(
                f'<text class="edge-label" x="{label_x:.1f}" y="{label_y:.1f}" '
                f'text-anchor="{anchor}">{escape(edge.label)}</text>'
            )

    groups: list[str] = []
    for box in placed.boxes:
        terminal = " terminal" if box.key in _TERMINALS else ""
        attr = escape(box.key, quote=True)
        groups.append(
            f'<g class="node{terminal}" data-node="{attr}">'
            f'<rect class="box" x="{box.x:.1f}" y="{box.y:.1f}" '
            f'width="{_NODE_W:.0f}" height="{_NODE_H:.0f}" rx="8"/>'
            f'<text class="label" x="{box.centre_x:.1f}" y="{box.centre_y - 3:.1f}" '
            f'text-anchor="middle">{escape(box.name)}</text>'
            f'<text class="metric" data-metric-for="{attr}" x="{box.centre_x:.1f}" '
            f'y="{box.centre_y + 14:.1f}" text-anchor="middle"></text>'
            f"</g>"
        )

    return (
        f'<svg class="topology" viewBox="0 0 {placed.width:.0f} {placed.height:.0f}" '
        f'width="{placed.width:.0f}" height="{placed.height:.0f}" role="img" '
        f'aria-label="Agent graph: {escape(", ".join(topology.keys), quote=True)}">'
        f'<defs><marker id="wh-arrow" viewBox="0 0 8 8" refX="7" refY="4" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path class="arrow" d="M0,0 L8,4 L0,8 z"/></marker></defs>'
        f'<g class="edges">{"".join(paths)}</g>'
        f'<g class="nodes">{"".join(groups)}</g>'
        f"</svg>"
    )
