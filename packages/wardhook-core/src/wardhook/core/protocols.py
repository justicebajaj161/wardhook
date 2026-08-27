"""Structural contracts that let optional Wardhook packages plug in.

This module is the seam that keeps the four Wardhook packages independently
installable. ``wardhook-core`` never imports ``wardhook.guardrails`` or
``wardhook.observability`` at module scope, and those packages never import
``wardhook.core`` at all. They meet here instead, through
:class:`typing.Protocol` definitions that describe *shapes* rather than
concrete classes.

The practical consequence is that anything satisfying the shape works. A
guardrail can come from ``wardhook-guardrails``, from your own codebase, or
from a third-party library, and :class:`~wardhook.core.agent.AgentGraph` will
drive it identically.

Two conventions keep the contract honest:

* **Context is a plain mapping, not a class.** Core passes a ``dict`` with
  documented keys, so an implementer never has to import a Wardhook type to
  read it.
* **Results are read by attribute, defensively.** Core uses
  :func:`read_guardrail_result` rather than isinstance checks, so a guardrail
  may return its own result type with its own extra fields.

Example:
    A minimal guardrail with no Wardhook dependency at all::

        class NoProfanity:
            name = "no-profanity"

            def on_input(self, text, context):
                if "damn" in text.lower():
                    return SimpleNamespace(
                        action="redact",
                        text=text.replace("damn", "****"),
                        reason="profanity",
                        rule="wordlist",
                    )
                return SimpleNamespace(action="allow", text=text, reason=None, rule=None)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "EmbeddingsProtocol",
    "GuardrailAction",
    "GuardrailDecision",
    "GuardrailProtocol",
    "GuardrailResultProtocol",
    "RetrieverProtocol",
    "TelemetryProtocol",
    "VectorStoreProtocol",
    "read_guardrail_result",
]


class GuardrailAction(str, Enum):
    """What a guardrail decided should happen to the text or call it inspected.

    This is a :class:`str` enum, so a guardrail from another package may return
    the plain string ``"block"`` and it will compare equal to
    :attr:`GuardrailAction.BLOCK`. That is deliberate: it means implementers do
    not need to import this enum to interoperate.

    Attributes:
        ALLOW: Let the text or tool call through unchanged.
        REDACT: Let it through, but substitute the modified text the guardrail
            returned. The run continues.
        BLOCK: Stop here. For input and output stages the graph short-circuits
            to the end; for a tool call the tool is never executed and the model
            receives a denial message instead.
    """

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class GuardrailDecision:
    """A normalised view of whatever a guardrail returned.

    Core converts every guardrail result into one of these via
    :func:`read_guardrail_result` so the rest of the runtime works against a
    single shape regardless of which package produced the original.

    Attributes:
        action: The normalised :class:`GuardrailAction`.
        text: The text to carry forward. For a redaction this is the modified
            text; otherwise it is the text that was passed in.
        reason: Human-readable explanation, surfaced in audit records.
        rule: Identifier of the specific rule that fired, such as an entity
            name or pattern id.
        guardrail: Name of the guardrail that produced the decision.
        details: Any additional structured data the guardrail attached.
    """

    __slots__ = ("action", "details", "guardrail", "reason", "rule", "text")

    def __init__(
        self,
        action: GuardrailAction,
        text: str,
        *,
        reason: str | None = None,
        rule: str | None = None,
        guardrail: str = "unknown",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise a decision. See class attributes for argument meanings."""
        self.action = action
        self.text = text
        self.reason = reason
        self.rule = rule
        self.guardrail = guardrail
        self.details: dict[str, Any] = dict(details or {})

    @property
    def blocked(self) -> bool:
        """Whether this decision stops the run."""
        return self.action is GuardrailAction.BLOCK

    @property
    def modified(self) -> bool:
        """Whether this decision changed the text."""
        return self.action is GuardrailAction.REDACT

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the decision.

        The decision text itself is deliberately omitted: it may contain the
        very PII a guardrail just redacted, and this dict is written to audit
        trails and traces.

        Returns:
            A dict with the guardrail name, action, reason, rule and details.
        """
        return {
            "guardrail": self.guardrail,
            "action": self.action.value,
            "reason": self.reason,
            "rule": self.rule,
            "details": self.details,
        }

    def __repr__(self) -> str:
        """Return a debug representation that never includes the text."""
        return (
            f"GuardrailDecision(guardrail={self.guardrail!r}, "
            f"action={self.action.value!r}, rule={self.rule!r})"
        )


@runtime_checkable
class GuardrailResultProtocol(Protocol):
    """The shape core expects back from a guardrail hook.

    Only ``action`` is strictly required; the remaining attributes are read
    with :func:`getattr` defaults, so a result object may omit them.
    """

    action: str
    """One of ``"allow"``, ``"redact"`` or ``"block"``."""


@runtime_checkable
class GuardrailProtocol(Protocol):
    """A policy object that inspects text and tool calls as an agent runs.

    Implementations are called at up to three points per run. Every hook is
    optional in practice: core probes for each with :func:`getattr` and skips
    any that is missing, so a guardrail that only cares about tool calls need
    only implement :meth:`on_tool_call`.

    The ``context`` mapping passed to every hook carries these documented keys:

    * ``run_id`` (:class:`str`) -- unique id for this agent invocation.
    * ``stage`` (:class:`str`) -- ``"input"``, ``"output"`` or ``"tool_call"``.
    * ``node`` (:class:`str`) -- the graph node currently executing.
    * ``principal`` (:class:`~collections.abc.Mapping` | ``None``) -- the caller
      identity, used for role-based decisions. Typically ``{"id": ...,
      "roles": [...]}``.
    """

    name: str
    """Stable identifier for this guardrail, used in audit records."""

    def on_input(self, text: str, context: Mapping[str, Any]) -> GuardrailResultProtocol:
        """Inspect user input before it reaches the model.

        Args:
            text: The incoming user text.
            context: Run context; see the class docstring for its keys.

        Returns:
            A result whose ``action`` decides whether the run continues.
        """
        ...

    def on_output(self, text: str, context: Mapping[str, Any]) -> GuardrailResultProtocol:
        """Inspect model output before it reaches the caller.

        Args:
            text: The model's response text.
            context: Run context; see the class docstring for its keys.

        Returns:
            A result whose ``action`` decides whether the response is returned,
            redacted, or withheld.
        """
        ...

    def on_tool_call(
        self,
        tool_name: str,
        tool_args: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> GuardrailResultProtocol:
        """Decide whether a tool call may execute.

        Args:
            tool_name: Name of the tool the model asked to call.
            tool_args: Arguments the model supplied.
            context: Run context; see the class docstring for its keys.

        Returns:
            A result whose ``action`` decides whether the tool runs. ``BLOCK``
            means the tool is never invoked and the model is told it was denied.
        """
        ...


@runtime_checkable
class TelemetryProtocol(Protocol):
    """A sink that records per-node timing, token, and cost information.

    ``wardhook-observability`` supplies the reference implementation, but any
    object with these methods works. Core calls the lifecycle hooks around each
    graph node and passes :meth:`callbacks` into every model invocation so the
    sink can read real token usage off the provider response.
    """

    def start_run(self, run_id: str, metadata: Mapping[str, Any] | None = None) -> None:
        """Signal that an agent invocation has begun.

        Args:
            run_id: Unique identifier for this invocation.
            metadata: Optional free-form information about the run.
        """
        ...

    def end_run(self, run_id: str, error: str | None = None) -> None:
        """Signal that an agent invocation has finished.

        Args:
            run_id: The identifier passed to :meth:`start_run`.
            error: Error description if the run failed, otherwise ``None``.
        """
        ...

    def start_node(self, node: str, run_id: str) -> None:
        """Signal that a graph node has begun executing.

        Args:
            node: Name of the node.
            run_id: The current run identifier.
        """
        ...

    def end_node(self, node: str, run_id: str, error: str | None = None) -> None:
        """Signal that a graph node has finished executing.

        Args:
            node: Name of the node.
            run_id: The current run identifier.
            error: Error description if the node raised, otherwise ``None``.
        """
        ...

    def callbacks(self) -> Sequence[Any]:
        """Return LangChain callback handlers to attach to model invocations.

        Returns:
            A sequence of handlers, possibly empty. Core passes these through
            to the model as ``config={"callbacks": [...]}``.
        """
        ...


@runtime_checkable
class EmbeddingsProtocol(Protocol):
    """Turns text into vectors. Matches the LangChain embeddings interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for indexing.

        Args:
            texts: Document texts to embed.

        Returns:
            One vector per input text, in the same order.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query text.

        Returns:
            The query's vector.
        """
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Stores embedded chunks and answers similarity queries over them."""

    def add(self, chunks: Sequence[Any]) -> list[str]:
        """Embed and store chunks.

        Args:
            chunks: :class:`~wardhook.core.rag.chunking.Chunk` objects to index.

        Returns:
            The assigned chunk ids, in input order.
        """
        ...

    def search(self, query: str, k: int = 4) -> list[Any]:
        """Return the ``k`` chunks most similar to ``query``.

        Args:
            query: Natural-language query text.
            k: Maximum number of results.

        Returns:
            Scored results, most similar first.
        """
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Fetches source-attributed context for a question."""

    def retrieve(self, query: str) -> list[Any]:
        """Return citable chunks relevant to ``query``.

        Args:
            query: The user's question.

        Returns:
            Retrieved results carrying source metadata for citation.
        """
        ...


def read_guardrail_result(
    result: Any,
    *,
    original_text: str,
    guardrail_name: str,
) -> GuardrailDecision:
    """Normalise any guardrail return value into a :class:`GuardrailDecision`.

    This is the function that makes cross-package duck typing safe. It accepts
    whatever a guardrail hands back -- a ``wardhook.guardrails`` result, a
    dataclass of your own, a plain dict, ``None``, or a bare boolean -- and
    produces one predictable shape.

    Unrecognised action strings are treated as ``ALLOW`` rather than raising,
    on the principle that a guardrail returning something odd should not take
    down a production agent. The unrecognised value is preserved in
    ``details["raw_action"]`` so the anomaly is still visible in audit records.

    Args:
        result: Whatever the guardrail's hook returned.
        original_text: The text passed into the hook, used as the fallback when
            the result carries no replacement text.
        guardrail_name: Name recorded on the decision.

    Returns:
        A normalised decision. ``None`` and ``True`` both mean allow; ``False``
        means block.

    Example:
        >>> read_guardrail_result(None, original_text="hi", guardrail_name="g").action
        <GuardrailAction.ALLOW: 'allow'>
        >>> d = read_guardrail_result(
        ...     {"action": "redact", "text": "[REDACTED]", "rule": "ssn"},
        ...     original_text="123-45-6789",
        ...     guardrail_name="pii",
        ... )
        >>> d.text, d.rule
        ('[REDACTED]', 'ssn')
    """
    if result is None or result is True:
        return GuardrailDecision(GuardrailAction.ALLOW, original_text, guardrail=guardrail_name)
    if result is False:
        return GuardrailDecision(
            GuardrailAction.BLOCK,
            original_text,
            reason="guardrail returned False",
            guardrail=guardrail_name,
        )

    if isinstance(result, Mapping):
        get = result.get
    else:

        def get(key: str, default: Any = None) -> Any:
            return getattr(result, key, default)

    raw_action = get("action", GuardrailAction.ALLOW)
    action_value = raw_action.value if isinstance(raw_action, Enum) else str(raw_action)

    details = dict(get("details", None) or {})
    try:
        action = GuardrailAction(action_value.lower())
    except ValueError:
        action = GuardrailAction.ALLOW
        details["raw_action"] = action_value

    text = get("text", None)
    return GuardrailDecision(
        action,
        original_text if text is None else str(text),
        reason=get("reason", None),
        rule=get("rule", None),
        guardrail=str(get("name", None) or guardrail_name),
        details=details,
    )
