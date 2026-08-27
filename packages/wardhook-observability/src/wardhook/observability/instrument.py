"""Attach a tracer to a LangGraph graph you already built.

``wardhook-core`` reports node boundaries to a tracer itself. A graph you
assembled with plain LangGraph does not, so :func:`instrument` derives them
instead -- from the callback stream LangGraph already emits.

**Why callbacks rather than wrapping node functions.** Reaching into a compiled
graph and replacing each node's callable means depending on LangGraph's
internal layout, which is not a stable API and would break on upgrade. The
callback stream *is* a public interface, and it carries everything needed:
verified against langgraph 1.2 / langchain-core 1.6, each node's
``on_chain_start`` arrives with ``metadata["langgraph_node"]`` naming it, and
the matching ``on_chain_end`` shares its ``run_id``. Nothing internal is
touched, so this keeps working across LangGraph versions.

What :func:`instrument` does touch is the graph object's ``invoke`` family, so
callers do not have to remember to pass ``config={"callbacks": [...]}`` on every
call. That is a deliberate, small, reversible piece of monkey-patching --
:func:`uninstrument` puts it back.

Example:
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
    >>> graph.invoke({"value": 21})
    {'value': 42}
    >>> [step.node for step in tracer.get_trace().steps]
    ['double']
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

from wardhook.observability.callbacks import GraphTraceCallback
from wardhook.observability.tracer import Tracer

__all__ = ["instrument", "uninstrument"]

# Marks a graph as already instrumented and holds its tracer, so instrumenting
# twice returns the first tracer rather than double-counting every token.
_MARKER = "_wardhook_tracer"
_ORIGINALS = "_wardhook_originals"

# Methods worth wrapping. `stream` and `astream` return iterators, so they are
# passed through rather than awaited -- only `ainvoke` is a coroutine.
_SYNC_METHODS = ("invoke", "stream")
_ASYNC_METHODS = ("astream",)
_COROUTINE_METHODS = ("ainvoke",)


def _already_tracing(callbacks: Any, tracer: Tracer) -> bool:
    """Whether a config's callbacks already contain a handler for this tracer.

    This is what stops a graph being traced twice in one call. LangGraph's
    ``invoke`` is implemented on top of its own ``stream``, and both are
    wrapped here -- without this check the inner ``stream`` would attach a
    second handler and every node would be recorded, and billed, twice.

    Args:
        callbacks: Whatever the caller's config had under ``"callbacks"``.
        tracer: The tracer to look for.

    Returns:
        ``True`` if a handler reporting to ``tracer`` is already present.
    """
    if callbacks is None:
        return False
    handlers = callbacks if isinstance(callbacks, list) else getattr(callbacks, "handlers", ())
    return any(
        isinstance(handler, GraphTraceCallback) and handler.sink is tracer for handler in handlers
    )


def _with_tracing(config: Any, tracer: Tracer) -> dict[str, Any]:
    """Return a config carrying a tracing handler, adding one only if needed.

    A new :class:`~wardhook.observability.callbacks.GraphTraceCallback` is
    created per invocation because it holds the run's node bookkeeping; sharing
    one across concurrent calls would interleave their steps.

    Args:
        config: The caller's LangGraph config, possibly ``None``.
        tracer: The tracer to report to.

    Returns:
        A new config dict. The caller's own callbacks are preserved.
    """
    merged: dict[str, Any] = dict(config or {})
    existing = merged.get("callbacks")
    if _already_tracing(existing, tracer):
        return merged
    handler = GraphTraceCallback(tracer)
    if existing is None:
        merged["callbacks"] = [handler]
    elif isinstance(existing, list):
        merged["callbacks"] = [*existing, handler]
    else:
        # A CallbackManager rather than a plain list. Adding to it in place
        # would mutate the caller's object, so copy first where we can.
        adder = getattr(existing, "add_handler", None)
        if callable(adder):
            adder(handler, True)
        merged["callbacks"] = existing
    return merged


def instrument(graph: Any, tracer: Tracer | None = None) -> Tracer:
    """Record traces for every call to an existing compiled graph.

    Args:
        graph: A compiled LangGraph graph, or anything else exposing
            ``invoke``/``ainvoke``/``stream``/``astream``.
        tracer: The tracer to record into. A new one is created when omitted.

    Returns:
        The tracer now attached to the graph. Instrumenting an
        already-instrumented graph is a no-op that returns the original
        tracer, so this is safe to call from module scope that may be imported
        more than once.

    Raises:
        TypeError: If the object exposes none of the expected methods, which
            almost always means something other than a graph was passed.
    """
    existing: Tracer | None = getattr(graph, _MARKER, None)
    if existing is not None:
        return existing

    tracer = tracer if tracer is not None else Tracer()
    originals: dict[str, Any] = {}

    for name in (*_SYNC_METHODS, *_ASYNC_METHODS, *_COROUTINE_METHODS):
        original = getattr(graph, name, None)
        if not callable(original):
            continue
        originals[name] = original
        setattr(graph, name, _wrap(original, tracer, coroutine=name in _COROUTINE_METHODS))

    if not originals:
        raise TypeError(
            f"{type(graph).__name__} exposes none of invoke/ainvoke/stream/astream, "
            f"so there is nothing to instrument. Pass a compiled LangGraph graph."
        )

    setattr(graph, _ORIGINALS, originals)
    setattr(graph, _MARKER, tracer)
    return tracer


def uninstrument(graph: Any) -> None:
    """Remove tracing from a graph, restoring its original methods.

    Args:
        graph: A graph previously passed to :func:`instrument`. Doing this to
            an uninstrumented graph is a no-op.
    """
    originals: dict[str, Any] | None = getattr(graph, _ORIGINALS, None)
    if originals is None:
        return
    for name, original in originals.items():
        try:
            delattr(graph, name)
        except AttributeError:  # pragma: no cover - defensive
            setattr(graph, name, original)
    for attribute in (_ORIGINALS, _MARKER):
        with contextlib.suppress(AttributeError):
            delattr(graph, attribute)


def _wrap(original: Any, tracer: Tracer, *, coroutine: bool) -> Any:
    """Wrap one graph method so it traces itself.

    Args:
        original: The bound method being replaced.
        tracer: The tracer to report to.
        coroutine: Whether the method must be awaited. ``stream`` and
            ``astream`` return iterators and are passed straight through.

    Returns:
        The replacement callable.
    """
    if coroutine:

        @functools.wraps(original)
        async def async_wrapper(inputs: Any, config: Any = None, **kwargs: Any) -> Any:
            return await original(inputs, _with_tracing(config, tracer), **kwargs)

        return async_wrapper

    @functools.wraps(original)
    def wrapper(inputs: Any, config: Any = None, **kwargs: Any) -> Any:
        return original(inputs, _with_tracing(config, tracer), **kwargs)

    return wrapper
