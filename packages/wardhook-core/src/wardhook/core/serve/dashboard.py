"""A read-only JSON API describing what an agent *is* and what it *cost*.

:func:`create_dashboard` builds a small FastAPI application exposing three
endpoints:

* ``GET /api/topology``      -- the agent's graph, read from its own configuration.
* ``GET /api/runs``          -- one summary per recorded run.
* ``GET /api/runs/{run_id}`` -- one run's per-node timing, tokens and cost.

**It shows telemetry and configuration. It never shows content.** No prompt, no
model output, no retrieved chunk, no guardrail event body reaches this API. That
is not a policy applied on top of the data -- the telemetry model has no such
fields in it, and the projection below is an explicit allowlist so that it stays
that way even if one is added upstream. Rendering agent output here would turn
Wardhook into a tool that redacts personal data from the audit log and then
serves it over HTTP.

The question this design keeps having to answer is *"a node cost $0.40, so show
me the prompt"*. The answer is :attr:`run_id`, which appears on every run and
every step, and on every audit record the caller writes. Correlating the two is
a lookup in **the caller's own audit log**, where their redaction policy, their
retention rules and their access controls already apply. This API hands over the
key; it does not keep a second, unredacted copy of the lock.

**Nothing is imported from a sibling package.** The telemetry sink is read
structurally, exactly as :class:`~wardhook.core.agent.AgentGraph` reads
guardrails, so ``wardhook-core`` still installs and passes entirely on its own.
Two sink shapes are understood, because the two that ship with Wardhook do not
agree on method names:

* ``Tracer`` lists with ``traces()`` and looks up with ``get_trace(run_id)``.
* ``JSONLTraceStore`` lists with ``read()`` and looks up with ``read_one(run_id)``.

Duck-typing only the first pair would silently break the very mitigation the
second pair exists to provide, so both are supported.

Which one was found decides the reported ``mode``, and the mode is reported
rather than hidden: an in-memory tracer under ``uvicorn --workers 4`` only ever
sees its own process's traffic, and an observability tool that quietly drops
three quarters of the data is worse than one that says so.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query

from wardhook.core.serve.app import _describe
from wardhook.core.serve.topology import read_topology

__all__ = ["create_dashboard"]

# Method names understood on a telemetry sink, paired with the mode each implies.
# Order matters only in that the in-memory shape is checked first; the two sinks
# that ship with Wardhook expose one name each, never both.
_LIST_METHODS: tuple[tuple[str, str], ...] = (("traces", "memory"), ("read", "store"))
_LOOKUP_METHODS: tuple[str, ...] = ("get_trace", "read_one")

_MODE_NOTES: dict[str, str] = {
    "memory": (
        "In-memory tracer. This is one process's view: under multiple workers "
        "(uvicorn --workers N, gunicorn) each worker owns its own tracer, so "
        "roughly 1/N of traffic is visible here. Point the tracer at a shared "
        "JSONLTraceStore and pass that store as the dashboard's telemetry to "
        "see every run."
    ),
    "store": (
        "Shared trace store. Every run written to the file is visible here, "
        "including runs served by other worker processes."
    ),
    "none": (
        "No readable telemetry is attached, so no runs can be listed. Construct "
        "the agent with telemetry=True, or pass a sink to create_dashboard()."
    ),
}

# Largest page of runs the API will return in one response. A shared trace file
# is append-only and unbounded; returning all of it would make the endpoint
# unusable on exactly the deployment that needs it most.
_MAX_LIMIT = 1000
_DEFAULT_LIMIT = 100


def _optional_str(value: Any) -> str | None:
    """Coerce a value to text, preserving the difference between unset and empty.

    Args:
        value: Any value, possibly ``None``.

    Returns:
        ``None`` if the value was ``None``, otherwise its string form.
    """
    return None if value is None else str(value)


def _count(usage: Any, field: str) -> int:
    """Read one token count off a usage object.

    Args:
        usage: A token-usage object, or ``None`` for a node that called no model.
        field: The attribute to read.

    Returns:
        The count, or ``0`` when the usage or the field is absent. Guardrail and
        retrieval nodes legitimately have no usage at all.
    """
    return int(getattr(usage, field, 0) or 0)


def _project_step(step: Any) -> dict[str, Any]:
    """Project one trace step onto the fields this API publishes.

    **This function is the allowlist, and that is deliberate.** It names every
    field that reaches a browser. If a content-bearing field -- a prompt, a
    completion, a retrieved chunk -- is ever added to the upstream step type,
    this API does not begin serving it by accident; someone has to add it here
    on purpose, and this docstring is what they will read when they do.

    Args:
        step: Any object shaped like an observability ``TraceStep``.

    Returns:
        A JSON-serialisable dict of timing, tokens, cost and error only.
    """
    usage = getattr(step, "usage", None)
    return {
        "node": str(getattr(step, "node", "")),
        "run_id": str(getattr(step, "run_id", "")),
        "started_at": str(getattr(step, "started_at", "")),
        "latency_ms": float(getattr(step, "latency_ms", 0.0) or 0.0),
        "cost": float(getattr(step, "cost", 0.0) or 0.0),
        "model": _optional_str(getattr(step, "model", None)),
        "error": _optional_str(getattr(step, "error", None)),
        "tokens_in": _count(usage, "input_tokens"),
        "tokens_out": _count(usage, "output_tokens"),
        "cached_tokens": _count(usage, "cache_read_tokens"),
    }


def _project_trace(trace: Any) -> dict[str, Any]:
    """Project one whole trace, recomputing its totals from its steps.

    Totals are derived here rather than read from the trace, for the same reason
    :meth:`Trace.from_dict` recomputes them: a truncated or hand-edited trace
    then cannot claim totals its steps do not support.

    Args:
        trace: Any object shaped like an observability ``Trace``.

    Returns:
        A JSON-serialisable dict. ``metadata`` is the one caller-supplied field
        and is passed through verbatim, so -- as ``wardhook-observability``'s
        store already warns -- do not put user text in it.
    """
    steps = [_project_step(step) for step in getattr(trace, "steps", None) or ()]
    error = _optional_str(getattr(trace, "error", None))
    return {
        "run_id": str(getattr(trace, "run_id", "")),
        "started_at": str(getattr(trace, "started_at", "")),
        "latency_ms": float(getattr(trace, "latency_ms", 0.0) or 0.0),
        "metadata": dict(getattr(trace, "metadata", None) or {}),
        "error": error,
        "failed": error is not None or any(step["error"] is not None for step in steps),
        "totals": {
            "steps": len(steps),
            "tokens_in": sum(step["tokens_in"] for step in steps),
            "tokens_out": sum(step["tokens_out"] for step in steps),
            "cached_tokens": sum(step["cached_tokens"] for step in steps),
            "cost": sum(step["cost"] for step in steps),
        },
        "steps": steps,
    }


def _summarise(projected: dict[str, Any]) -> dict[str, Any]:
    """Reduce a projected trace to its summary form.

    Args:
        projected: The output of :func:`_project_trace`.

    Returns:
        The same dict without ``steps``. Derived by subtraction rather than
        rebuilt, so a summary can never carry a field the detail view lacks.
    """
    return {key: value for key, value in projected.items() if key != "steps"}


def _list_traces(sink: Any) -> tuple[list[Any], str]:
    """Read every run a sink is willing to list.

    Args:
        sink: The telemetry sink, or ``None``.

    Returns:
        A tuple of the traces (oldest first, as both known sinks report them)
        and the mode that reading them implies.
    """
    for name, mode in _LIST_METHODS:
        reader = getattr(sink, name, None)
        if callable(reader):
            return list(reader()), mode
    return [], "none"


def _find_trace(sink: Any, run_id: str) -> Any:
    """Look one run up on a sink.

    Args:
        sink: The telemetry sink, or ``None``.
        run_id: The run to find.

    Returns:
        The trace, or ``None`` when the sink cannot look runs up or does not
        hold that one.
    """
    for name in _LOOKUP_METHODS:
        lookup = getattr(sink, name, None)
        if callable(lookup):
            return lookup(run_id)
    return None


def _resolve_telemetry(agent: Any, telemetry: Any) -> Any:
    """Decide which telemetry sink the dashboard reads.

    Args:
        agent: The agent being described.
        telemetry: An explicitly supplied sink, or ``None`` to use the agent's.

    Returns:
        The sink, or ``None``. Passing a sink explicitly is what lets a caller
        point the dashboard at a shared :class:`JSONLTraceStore` while the agent
        keeps writing through an in-memory tracer -- the documented mitigation
        for the multi-worker limitation.
    """
    return telemetry if telemetry is not None else getattr(agent, "telemetry", None)


def create_dashboard(agent: Any, telemetry: Any = None) -> FastAPI:
    """Build the read-only dashboard API for an agent.

    Args:
        agent: The agent to describe. Any object works; one without a graph
            simply reports that it has no topology.
        telemetry: The sink to read runs from. Defaults to the agent's own
            ``telemetry`` attribute. Pass a shared trace store here to read
            every worker's runs rather than one process's.

    Returns:
        A FastAPI application, ready to mount or to serve on its own.

    Example:
        >>> from langchain_core.language_models.fake_chat_models import (
        ...     GenericFakeChatModel,
        ... )
        >>> from langchain_core.messages import AIMessage
        >>> from wardhook.core import AgentGraph
        >>> model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
        >>> app = create_dashboard(AgentGraph(model=model, name="demo"))
        >>> app.title
        'Wardhook Dashboard'
    """
    sink = _resolve_telemetry(agent, telemetry)
    app = FastAPI(
        title="Wardhook Dashboard",
        description=(
            "Read-only view of an agent's structure and what its runs cost. "
            "Serves telemetry and configuration only, never agent content."
        ),
    )

    @app.get("/api/topology", tags=["dashboard"])
    def topology() -> dict[str, Any]:
        """Describe the agent's graph and configuration.

        Returns:
            The topology, plus the same configuration summary ``GET /info``
            reports. Both are derived from the agent as configured, so an agent
            with no retriever genuinely reports no retrieval node.
        """
        return {
            "agent": str(_describe(agent)["name"]),
            "config": _describe(agent),
            **read_topology(agent).to_dict(),
        }

    @app.get("/api/runs", tags=["dashboard"])
    def runs(
        limit: int = Query(
            default=_DEFAULT_LIMIT,
            ge=1,
            le=_MAX_LIMIT,
            description="Maximum number of runs to return, newest first.",
        ),
    ) -> dict[str, Any]:
        """Summarise the recorded runs, newest first.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            The summaries, the total held, and which telemetry mode produced
            them. ``mode`` and ``mode_note`` are how the multi-worker limitation
            is disclosed rather than hidden.
        """
        traces, mode = _list_traces(sink)
        newest_first = list(reversed(traces))
        return {
            "mode": mode,
            "mode_note": _MODE_NOTES[mode],
            "total": len(newest_first),
            "returned": len(newest_first[:limit]),
            "runs": [_summarise(_project_trace(trace)) for trace in newest_first[:limit]],
        }

    @app.get("/api/runs/{run_id}", tags=["dashboard"])
    def run(run_id: str) -> dict[str, Any]:
        """Return one run's per-node timing, tokens and cost.

        Args:
            run_id: The run to look up.

        Returns:
            The projected trace.

        Raises:
            HTTPException: ``404`` when no such run is held. A run that has been
                evicted from a bounded in-memory ring is genuinely absent, so
                this is not an error the caller can fix by retrying.
        """
        trace = _find_trace(sink, run_id)
        if trace is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No trace for run_id {run_id!r}. It may have been evicted "
                    f"from the tracer's in-memory ring, or never recorded."
                ),
            )
        return _project_trace(trace)

    return app
