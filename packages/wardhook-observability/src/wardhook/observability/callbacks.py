"""LangChain callback handlers that feed real token usage to a trace sink.

Two handlers live here, because the two ways this package is used need
different amounts of help:

* :class:`UsageCallback` is for when something else already knows the graph
  structure. ``wardhook-core`` calls ``start_node``/``end_node`` around every
  node itself, so the handler's only job is to read token usage off the model
  response and hand it over for attribution to whichever node is running.
* :class:`GraphTraceCallback` is for :func:`wardhook.observability.instrument`,
  which attaches to a graph nobody told us about. It derives node boundaries
  from LangGraph's own callback metadata, then does everything
  :class:`UsageCallback` does.

**Why token counts come from the provider.** Re-tokenising the prompt locally
would be an estimate, and a wrong one: it cannot see the system prompt the
provider prepends, tool-schema overhead, or which parts of the prompt were
served from cache. ``usage_metadata`` on the response is what the bill is
computed from, so it is what gets recorded here.

**These handlers never raise.** Telemetry sits directly in the request path of
somebody's production agent. A malformed response, a provider that reports
usage in an unexpected shape, or a bug in this file must degrade to a warning
and a missing number -- never to a failed agent run.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from wardhook.observability.models import TokenUsage

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

__all__ = ["GraphTraceCallback", "TraceSink", "UsageCallback", "usage_from_response"]

# LangGraph tags every node run with this metadata key. Verified against
# langgraph 1.2 / langchain-core 1.6: `on_chain_start` carries
# metadata["langgraph_node"], while `on_chain_end` carries only the run_id --
# which is why node names are remembered by run_id rather than re-read at the end.
_NODE_KEY = "langgraph_node"


class TraceSink(Protocol):
    """The part of :class:`~wardhook.observability.tracer.Tracer` these handlers use.

    Declared as a protocol so the handlers can be tested against a trivial
    recorder, and so no import cycle is needed between this module and the
    tracer that constructs it.
    """

    def start_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Begin a run. See :meth:`~wardhook.observability.tracer.Tracer.start_run`."""
        ...

    def end_run(self, run_id: str, error: str | None = None) -> None:
        """End a run. See :meth:`~wardhook.observability.tracer.Tracer.end_run`."""
        ...

    def start_node(self, node: str, run_id: str) -> None:
        """Begin a node. See :meth:`~wardhook.observability.tracer.Tracer.start_node`."""
        ...

    def end_node(self, node: str, run_id: str, error: str | None = None) -> None:
        """End a node. See :meth:`~wardhook.observability.tracer.Tracer.end_node`."""
        ...

    def record_usage(self, usage: TokenUsage, model: str | None) -> None:
        """Attribute token usage to the node currently executing."""
        ...


def usage_from_response(response: LLMResult) -> tuple[TokenUsage, str | None]:
    """Extract token usage and the model name from a LangChain result.

    Prefers ``usage_metadata`` on the returned message, which is the modern,
    provider-normalised shape and the only one carrying cache detail. Falls
    back to the older ``llm_output["token_usage"]`` dict so this still works
    against integrations that have not adopted the newer field.

    Args:
        response: The result handed to ``on_llm_end``.

    Returns:
        A ``(usage, model)`` pair. Both are empty or ``None`` when the response
        reports nothing usable, which is normal for fake models in tests.

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> from langchain_core.outputs import ChatGeneration, LLMResult
        >>> message = AIMessage(
        ...     content="hi",
        ...     usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        ... )
        >>> result = LLMResult(generations=[[ChatGeneration(message=message)]])
        >>> usage, model = usage_from_response(result)
        >>> usage.input_tokens, usage.output_tokens
        (12, 3)
    """
    llm_output = response.llm_output or {}
    model = llm_output.get("model_name") or llm_output.get("model")

    for batch in response.generations:
        for generation in batch:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            metadata = getattr(message, "response_metadata", None) or {}
            model = model or metadata.get("model_name") or metadata.get("model")
            raw = getattr(message, "usage_metadata", None)
            if raw:
                input_details = raw.get("input_token_details") or {}
                output_details = raw.get("output_token_details") or {}
                usage = TokenUsage(
                    input_tokens=int(raw.get("input_tokens", 0)),
                    output_tokens=int(raw.get("output_tokens", 0)),
                    cache_read_tokens=int(input_details.get("cache_read", 0)),
                    cache_write_tokens=int(input_details.get("cache_creation", 0)),
                    reasoning_tokens=int(output_details.get("reasoning", 0)),
                )
                return usage, model

    legacy = llm_output.get("token_usage") or {}
    if legacy:
        return (
            TokenUsage(
                input_tokens=int(legacy.get("prompt_tokens", 0)),
                output_tokens=int(legacy.get("completion_tokens", 0)),
            ),
            model,
        )
    return TokenUsage(), model


class UsageCallback(BaseCallbackHandler):
    """Reads token usage off model responses and reports it to a sink.

    Attach this when something else is already tracking which node is running
    -- which is exactly what ``wardhook-core`` does. The handler makes no
    attempt to work out the graph structure; it answers only "how many tokens
    did that call use, and on which model".

    Args:
        sink: The tracer to report to.

    Example:
        >>> class Recorder:
        ...     def __init__(self):
        ...         self.seen = []
        ...
        ...     def record_usage(self, usage, model):
        ...         self.seen.append((usage.output_tokens, model))
        >>> handler = UsageCallback(Recorder())
        >>> handler.raise_error
        False
    """

    # LangChain checks this before deciding whether to propagate an exception
    # raised inside a handler. Telemetry must not be able to fail a run.
    raise_error = False

    def __init__(self, sink: TraceSink) -> None:
        """Initialise the handler. See the class docstring for arguments."""
        super().__init__()
        self._sink = sink

    @property
    def sink(self) -> TraceSink:
        """The tracer this handler reports to.

        Exposed so a caller merging handlers into a config can tell whether one
        for a given tracer is already attached, and avoid adding a second.
        """
        return self._sink

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Record the token usage of a completed model call.

        Args:
            response: The provider's result.
            run_id: LangChain's id for this call. Unused: attribution is by
                whichever node the sink currently has open.
            parent_run_id: The enclosing run, if any. Unused, as above.
            **kwargs: Further LangChain callback arguments.
        """
        try:
            usage, model = usage_from_response(response)
            if not usage.is_empty or model:
                self._sink.record_usage(usage, model)
        except Exception as exc:  # pragma: no cover - defensive
            _warn_and_continue("read token usage from a model response", exc)


def _warn_and_continue(action: str, exc: Exception) -> None:
    """Report a telemetry failure without letting it escape.

    Args:
        action: What was being attempted, phrased to complete "failed to ...".
        exc: The exception that was swallowed.
    """
    warnings.warn(
        f"wardhook-observability failed to {action}: {type(exc).__name__}: {exc}. "
        f"The agent run is unaffected, but this part of the trace will be incomplete.",
        RuntimeWarning,
        stacklevel=3,
    )


class GraphTraceCallback(UsageCallback):
    """Derives run and node boundaries from LangGraph's own callback stream.

    Used by :func:`wardhook.observability.instrument` on a graph this package
    did not build and cannot wrap node-by-node. LangGraph emits a chain event
    per node with ``metadata["langgraph_node"]`` naming it; the outermost chain
    carries no such key and is treated as the run itself.

    Node names arrive only on ``on_chain_start``. ``on_chain_end`` identifies
    its chain by ``run_id`` alone, so the mapping is remembered on the way in
    and popped on the way out.

    Args:
        sink: The tracer to report to.
        run_id: Force a specific run id instead of adopting LangChain's uuid
            for the outermost chain. Used when the caller already has one.
    """

    def __init__(self, sink: TraceSink, run_id: str | None = None) -> None:
        """Initialise the handler. See the class docstring for arguments."""
        super().__init__(sink)
        self._forced_run_id = run_id
        self._run_id: str | None = None
        self._nodes: dict[UUID, str] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        inputs: dict[str, Any],  # noqa: ARG002
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        tags: list[str] | None = None,  # noqa: ARG002
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Open a run or a node, depending on which chain this is.

        Args:
            serialized: LangChain's description of the runnable. Unused.
            inputs: The chain's inputs. Deliberately unused -- they contain
                user text, and nothing here should copy that anywhere.
            run_id: LangChain's id for this chain.
            parent_run_id: The enclosing chain, if any. Unused.
            tags: LangChain tags such as ``graph:step:1``. Unused; the node
                name in ``metadata`` is the more stable signal.
            metadata: Carries ``langgraph_node`` on a node chain.
            **kwargs: Further LangChain callback arguments.
        """
        try:
            node = (metadata or {}).get(_NODE_KEY)
            if node is None:
                if self._run_id is None:
                    self._run_id = self._forced_run_id or str(run_id)
                    self._sink.start_run(self._run_id, {"source": "instrument"})
                return
            if self._run_id is None:
                # A node event with no enclosing run: possible if the handler
                # was attached partway through. Adopt the node's run as ours.
                self._run_id = self._forced_run_id or str(run_id)
                self._sink.start_run(self._run_id, {"source": "instrument"})
            self._nodes[run_id] = str(node)
            self._sink.start_node(str(node), self._run_id)
        except Exception as exc:  # pragma: no cover - defensive
            _warn_and_continue("open a trace step", exc)

    def on_chain_end(
        self,
        outputs: dict[str, Any],  # noqa: ARG002
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close whichever node or run this chain corresponds to.

        Args:
            outputs: The chain's outputs. Unused, for the same reason as
                ``inputs`` above.
            run_id: LangChain's id for the chain that just finished.
            parent_run_id: The enclosing chain, if any. Unused.
            **kwargs: Further LangChain callback arguments.
        """
        self._close(run_id, None)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close a node or run that ended in an exception.

        Args:
            error: The exception that ended the chain.
            run_id: LangChain's id for the failed chain.
            parent_run_id: The enclosing chain, if any. Unused.
            **kwargs: Further LangChain callback arguments.
        """
        self._close(run_id, f"{type(error).__name__}: {error}")

    def _close(self, run_id: UUID, error: str | None) -> None:
        """End the node or run identified by ``run_id``.

        Args:
            run_id: LangChain's id for the chain that finished.
            error: Failure description, or ``None`` on success.
        """
        try:
            node = self._nodes.pop(run_id, None)
            if node is not None and self._run_id is not None:
                self._sink.end_node(node, self._run_id, error)
            elif node is None and self._run_id == str(run_id):
                self._sink.end_run(self._run_id, error)
                self._run_id = None
        except Exception as exc:  # pragma: no cover - defensive
            _warn_and_continue("close a trace step", exc)
