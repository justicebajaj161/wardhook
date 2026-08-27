"""Wardhook observability: per-node tokens, cost, and latency for LangGraph agents.

Answers the question every team gets asked one week after an agent ships --
*what did that cost, and which step was slow?* -- without a hosted service, a
sidecar, or a vendor SDK.

This package has **no dependency on the rest of Wardhook**. It needs
``langchain-core`` to read token usage off model responses and ``typer`` for its
CLI, and nothing else. It never imports ``wardhook.core``.

When ``wardhook-core`` *is* present, :class:`Tracer` satisfies its structural
``TelemetryProtocol``, so it attaches with a keyword argument:

    >>> from wardhook.core import AgentGraph  # doctest: +SKIP
    >>> agent = AgentGraph(model="claude-opus-5", telemetry=True)  # doctest: +SKIP
    >>> agent.invoke("What excess applies to storm damage?")  # doctest: +SKIP
    >>> trace = agent.trace()  # doctest: +SKIP

Standalone, against a graph you already built:

    >>> from typing import TypedDict
    >>> from langgraph.graph import START, END, StateGraph
    >>> class State(TypedDict):
    ...     value: int
    >>> builder = StateGraph(State)
    >>> _ = builder.add_node("double", lambda s: {"value": s["value"] * 2})
    >>> _ = builder.add_edge(START, "double")
    >>> _ = builder.add_edge("double", END)
    >>> graph = builder.compile()
    >>> tracer = instrument(graph)
    >>> _ = graph.invoke({"value": 21})
    >>> trace = tracer.get_trace()
    >>> [step.node for step in trace.steps]
    ['double']
    >>> render_html(trace).startswith("<!doctype html>")
    True
"""

from wardhook.observability.callbacks import GraphTraceCallback, UsageCallback
from wardhook.observability.instrument import instrument, uninstrument
from wardhook.observability.models import TokenUsage, Trace, TraceStep
from wardhook.observability.pricing import (
    PRICES,
    PRICES_AS_OF,
    ModelPrice,
    UnknownModelWarning,
    estimate_cost,
    get_price,
    known_models,
    normalise_model_name,
    register_price,
)
from wardhook.observability.store import JSONLTraceStore, load_traces
from wardhook.observability.tracer import Tracer
from wardhook.observability.viewer.html import render_html

__version__ = "0.1.0"

__all__ = [
    "PRICES",
    "PRICES_AS_OF",
    "GraphTraceCallback",
    "JSONLTraceStore",
    "ModelPrice",
    "TokenUsage",
    "Trace",
    "TraceStep",
    "Tracer",
    "UnknownModelWarning",
    "UsageCallback",
    "__version__",
    "estimate_cost",
    "get_price",
    "instrument",
    "known_models",
    "load_traces",
    "normalise_model_name",
    "register_price",
    "render_html",
    "uninstrument",
]
