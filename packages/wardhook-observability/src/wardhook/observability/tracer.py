"""The trace sink: collects per-node timing, tokens, and cost for agent runs.

:class:`Tracer` satisfies ``wardhook.core.protocols.TelemetryProtocol``
structurally, so ``AgentGraph(telemetry=True)`` constructs one without either
package importing the other. It is equally usable on its own, driven either by
:func:`wardhook.observability.instrument` or by calling the lifecycle methods
directly.

**Built to sit in a long-lived server.** Three properties follow from that:

* *Bounded memory.* Completed traces are held in a ring of at most
  ``max_runs``. A process serving requests for a month must not accumulate a
  trace per request forever. Pass a ``store`` to keep everything on disk.
* *Thread safety.* Shared state is guarded by a lock, and the "which node is
  running right now" question is answered per-thread, so two concurrent
  requests cannot attribute each other's tokens.
* *No exception ever escapes into the agent.* Recording telemetry is not worth
  failing a user's request over.

Example:
    >>> tracer = Tracer()
    >>> tracer.start_run("r1", {"agent": "support"})
    >>> tracer.start_node("call_model", "r1")
    >>> tracer.record_usage(TokenUsage(input_tokens=900, output_tokens=120), "claude-opus-5")
    >>> tracer.end_node("call_model", "r1")
    >>> tracer.end_run("r1")
    >>> trace = tracer.get_trace()
    >>> trace.run_id, trace.total_tokens_out
    ('r1', 120)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any

from wardhook.observability.callbacks import UsageCallback
from wardhook.observability.models import TokenUsage, Trace, TraceStep
from wardhook.observability.pricing import estimate_cost

if TYPE_CHECKING:
    from wardhook.observability.store import JSONLTraceStore

__all__ = ["Tracer"]

# Name given to the step holding token usage that arrived while no node was
# open. Better than discarding it: an unattributed cost is still a cost.
UNGROUPED_NODE = "(ungrouped)"


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        A timezone-aware timestamp, e.g. ``2026-06-24T10:00:00.123456+00:00``.
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _OpenNode:
    """A node that has started but not finished."""

    node: str
    run_id: str
    started_at: str
    started_perf: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None


@dataclass
class _OpenRun:
    """A run that has started but not finished."""

    run_id: str
    started_at: str
    started_perf: float
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[TraceStep] = field(default_factory=list)


class Tracer:
    """Records what each node of an agent run cost and how long it took.

    Args:
        store: Optional :class:`~wardhook.observability.store.JSONLTraceStore`.
            When set, every completed trace is appended to it, so history
            outlives both the in-memory ring and the process.
        max_runs: How many completed traces to keep in memory. Older ones are
            evicted oldest-first.

    Attributes:
        UNGROUPED_NODE: Node name used for usage recorded outside any node.
    """

    def __init__(
        self,
        *,
        store: JSONLTraceStore | None = None,
        max_runs: int = 100,
    ) -> None:
        """Initialise the tracer. See the class docstring for arguments."""
        if max_runs < 1:
            raise ValueError(f"max_runs must be at least 1, got {max_runs}")
        self._store = store
        self._max_runs = max_runs
        self._lock = threading.Lock()
        self._local = threading.local()
        self._open: dict[str, _OpenRun] = {}
        self._completed: OrderedDict[str, Trace] = OrderedDict()
        self._last_run_id: str | None = None
        self._callbacks: list[Any] = [UsageCallback(self)]

    # ------------------------------------------------------------------
    # Per-thread bookkeeping
    # ------------------------------------------------------------------

    @property
    def _stack(self) -> list[_OpenNode]:
        """The nodes currently open on this thread, innermost last."""
        stack: list[_OpenNode] | None = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _current_run_id(self) -> str | None:
        """Best guess at which run this thread is working on.

        Returns:
            The run of the innermost open node, else the run this thread last
            started, else the most recently started run overall. The final
            fallback matters when a provider dispatches its callback on a
            worker thread that never saw ``start_run``.
        """
        if self._stack:
            return self._stack[-1].run_id
        thread_run: str | None = getattr(self._local, "run_id", None)
        if thread_run is not None:
            return thread_run
        with self._lock:
            return next(reversed(self._open), None) if self._open else None

    # ------------------------------------------------------------------
    # TelemetryProtocol
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Signal that an agent invocation has begun.

        Args:
            run_id: Unique identifier for this invocation.
            metadata: Free-form information about the run, typically the agent
                name and model. Recorded verbatim into the trace.
        """
        run = _OpenRun(
            run_id=run_id,
            started_at=_now_iso(),
            started_perf=perf_counter(),
            metadata=dict(metadata or {}),
        )
        self._local.run_id = run_id
        with self._lock:
            self._open[run_id] = run

    def end_run(self, run_id: str, error: str | None = None) -> None:
        """Signal that an agent invocation has finished.

        Any node still open is closed first, so an exception mid-node still
        produces a complete trace rather than a truncated one.

        Args:
            run_id: The identifier passed to :meth:`start_run`.
            error: Error description if the run failed, otherwise ``None``.
        """
        for open_node in [n for n in self._stack if n.run_id == run_id]:
            self.end_node(open_node.node, run_id, error="run ended before node completed")

        with self._lock:
            run = self._open.pop(run_id, None)
            if run is None:
                return
            trace = Trace(
                run_id=run.run_id,
                steps=tuple(run.steps),
                started_at=run.started_at,
                latency_ms=(perf_counter() - run.started_perf) * 1000.0,
                metadata=run.metadata,
                error=error,
            )
            self._completed[run_id] = trace
            self._completed.move_to_end(run_id)
            while len(self._completed) > self._max_runs:
                self._completed.popitem(last=False)
            self._last_run_id = run_id

        if getattr(self._local, "run_id", None) == run_id:
            self._local.run_id = None
        if self._store is not None:
            self._store.append(trace)

    def start_node(self, node: str, run_id: str) -> None:
        """Signal that a graph node has begun executing.

        Args:
            node: Name of the node.
            run_id: The current run identifier.
        """
        self._stack.append(
            _OpenNode(
                node=node,
                run_id=run_id,
                started_at=_now_iso(),
                started_perf=perf_counter(),
            )
        )

    def end_node(self, node: str, run_id: str, error: str | None = None) -> None:
        """Signal that a graph node has finished executing.

        Args:
            node: Name of the node.
            run_id: The current run identifier.
            error: Error description if the node raised, otherwise ``None``.
        """
        stack = self._stack
        open_node: _OpenNode | None = None
        for index in range(len(stack) - 1, -1, -1):
            if stack[index].node == node and stack[index].run_id == run_id:
                open_node = stack.pop(index)
                break
        if open_node is None:
            return

        step = TraceStep(
            node=open_node.node,
            run_id=open_node.run_id,
            started_at=open_node.started_at,
            latency_ms=(perf_counter() - open_node.started_perf) * 1000.0,
            usage=open_node.usage,
            cost=estimate_cost(open_node.model, open_node.usage),
            model=open_node.model,
            error=error,
        )
        with self._lock:
            run = self._open.get(run_id)
            if run is not None:
                run.steps.append(step)

    def callbacks(self) -> Sequence[Any]:
        """Return the LangChain callback handlers to attach to model calls.

        Returns:
            A single :class:`~wardhook.observability.callbacks.UsageCallback`,
            reused across calls so repeatedly invoking this does not churn
            handler objects.
        """
        return self._callbacks

    # ------------------------------------------------------------------
    # Recording and retrieval
    # ------------------------------------------------------------------

    def record_usage(self, usage: TokenUsage, model: str | None) -> None:
        """Attribute token usage to whichever node this thread has open.

        Usage arriving while no node is open is not discarded -- it lands on a
        synthetic :data:`UNGROUPED_NODE` step instead, because a cost you
        cannot attribute is still a cost you paid.

        Args:
            usage: The token counts to add.
            model: The model that produced them.
        """
        stack = self._stack
        if stack:
            open_node = stack[-1]
            open_node.usage = open_node.usage + usage
            open_node.model = open_node.model or model
            return

        run_id = self._current_run_id()
        if run_id is None:
            return
        with self._lock:
            run = self._open.get(run_id)
            if run is None:
                return
            for index, step in enumerate(run.steps):
                if step.node == UNGROUPED_NODE:
                    merged = step.usage + usage
                    run.steps[index] = TraceStep(
                        node=UNGROUPED_NODE,
                        run_id=run_id,
                        started_at=step.started_at,
                        latency_ms=0.0,
                        usage=merged,
                        cost=estimate_cost(step.model or model, merged),
                        model=step.model or model,
                    )
                    return
            run.steps.append(
                TraceStep(
                    node=UNGROUPED_NODE,
                    run_id=run_id,
                    started_at=_now_iso(),
                    latency_ms=0.0,
                    usage=usage,
                    cost=estimate_cost(model, usage),
                    model=model,
                )
            )

    def get_trace(self, run_id: str | None = None) -> Trace | None:
        """Return the trace for a run.

        Args:
            run_id: The run to look up. When omitted or ``None``, returns the
                most recently completed run -- which is what
                ``AgentGraph.trace()`` asks for.

        Returns:
            The trace, a partial trace if the run is still in flight, or
            ``None`` if the run is unknown or has been evicted.
        """
        with self._lock:
            target = run_id if run_id is not None else self._last_run_id
            if target is None:
                return None
            completed = self._completed.get(target)
            if completed is not None:
                return completed
            run = self._open.get(target)
            if run is None:
                return None
            return Trace(
                run_id=run.run_id,
                steps=tuple(run.steps),
                started_at=run.started_at,
                latency_ms=(perf_counter() - run.started_perf) * 1000.0,
                metadata=run.metadata,
            )

    def traces(self) -> list[Trace]:
        """Return every completed trace still held in memory, oldest first.

        Returns:
            The retained traces. At most ``max_runs`` of them.
        """
        with self._lock:
            return list(self._completed.values())

    def reset(self) -> None:
        """Discard all recorded and in-flight state.

        Does not touch a configured store -- traces already written to disk
        stay there.
        """
        with self._lock:
            self._open.clear()
            self._completed.clear()
            self._last_run_id = None
        self._local.stack = []
        self._local.run_id = None

    def __repr__(self) -> str:
        """Return a debug representation summarising what is recorded."""
        with self._lock:
            return (
                f"Tracer(completed={len(self._completed)}, in_flight={len(self._open)}, "
                f"max_runs={self._max_runs}, store={self._store is not None})"
            )
