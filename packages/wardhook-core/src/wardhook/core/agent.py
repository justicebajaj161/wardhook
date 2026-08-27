"""AgentGraph: the LangGraph agent runtime.

:class:`AgentGraph` assembles a LangGraph state machine from whichever pieces
you supply. The graph is built to fit: an agent with no tools, no guardrails and
no retriever compiles down to a single model call, and each optional feature
adds exactly the nodes it needs.

The full shape, when everything is attached::

    START -> guard_input -> retrieve -> call_model -> tools --+
                  |                        |   ^              |
                  |  (blocked)             |   +--------------+
                  v                        v
                 END                  guard_output -> END

Guardrails and telemetry arrive through the structural contracts in
:mod:`wardhook.core.protocols`, so neither ``wardhook-guardrails`` nor
``wardhook-observability`` is imported unless you actually ask for it. Core
never depends on its siblings.
"""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from wardhook.core.models import describe_model, resolve_model
from wardhook.core.protocols import GuardrailAction, GuardrailDecision, read_guardrail_result
from wardhook.core.rag.retriever import DEFAULT_RAG_INSTRUCTIONS, format_citations
from wardhook.core.state import AgentState, Principal
from wardhook.core.tools import normalize_tools, tool_names

__all__ = ["AgentGraph", "MissingIntegrationError"]

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
BLOCKED_INPUT_MESSAGE = "This request was blocked by a policy guardrail and was not processed."
BLOCKED_OUTPUT_MESSAGE = "The response was withheld by a policy guardrail."

GuardrailErrorPolicy = Literal["block", "allow", "raise"]


class MissingIntegrationError(ImportError):
    """Raised when an optional Wardhook package is requested but not installed."""


def _text_of(message: BaseMessage | None) -> str:
    """Return a message's text content, flattening multi-part content.

    Args:
        message: The message to read, or ``None``.

    Returns:
        The text, or an empty string when there is none. Multi-part content
        (a list of blocks, as vision-capable models return) is reduced to the
        concatenation of its text blocks, since guardrails inspect text.
    """
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def _last_of_type(messages: Sequence[BaseMessage], kind: type) -> BaseMessage | None:
    """Return the most recent message of a given class.

    Args:
        messages: Conversation history, oldest first.
        kind: The message class to look for.

    Returns:
        The last matching message, or ``None``.
    """
    for message in reversed(messages):
        if isinstance(message, kind):
            return message
    return None


def _is_auditable(decision: GuardrailDecision) -> bool:
    """Decide whether a guardrail decision belongs in the audit trail.

    Anything other than a clean allow is recorded. A guardrail that *raised* is
    recorded too, even when the configured error policy resolved it to an
    allow: a detector that crashed is precisely the event a compliance reviewer
    needs to see, and letting fail-open swallow it would hide the gap.

    Args:
        decision: The normalised decision.

    Returns:
        ``True`` if the decision should be recorded.
    """
    return decision.action is not GuardrailAction.ALLOW or bool(decision.details.get("error"))


def _add_node(builder: Any, name: str, fn: Callable[[AgentState], dict[str, Any]]) -> None:
    """Register a graph node.

    ``builder`` is typed as :data:`~typing.Any` on purpose. LangGraph's
    ``add_node`` overloads bind their state type to a protocol requiring
    ``__required_keys__`` on *instances*, which a TypedDict class cannot satisfy
    under mypy even though it is exactly the type LangGraph expects at runtime.
    Confining the looseness to this one function keeps every other builder call
    fully type-checked.

    Args:
        builder: The :class:`~langgraph.graph.StateGraph` under construction.
        name: Node name.
        fn: The node implementation.
    """
    builder.add_node(name, fn)


class AgentGraph:
    """A configurable LangGraph agent with optional governance and telemetry.

    Args:
        model: A chat model instance, a model name, or ``None`` for
            :data:`~wardhook.core.models.DEFAULT_MODEL`. See
            :func:`~wardhook.core.models.resolve_model`.
        tools: Tools the model may call. Accepts ``BaseTool`` instances and
            plain documented callables.
        guardrails: Objects satisfying
            :class:`~wardhook.core.protocols.GuardrailProtocol`. Typically from
            ``wardhook-guardrails``, but anything with the right shape works.
        telemetry: ``True`` to construct a tracer from ``wardhook-observability``,
            an object satisfying
            :class:`~wardhook.core.protocols.TelemetryProtocol` to use your own,
            or ``False`` to disable tracing entirely.
        system_prompt: System prompt for the model. Retrieval instructions are
            appended automatically when ``retriever`` is set.
        retriever: An object with ``context_for(query)``, usually
            :class:`~wardhook.core.rag.retriever.Retriever`. Adds a retrieval
            node ahead of the model.
        max_tool_iterations: Maximum model-tool round trips before the graph
            stops calling tools. This is the backstop against a model that
            loops on a failing tool forever.
        guardrail_error_policy: What to do when a guardrail itself raises.
            ``"block"`` (the default) treats the failure as a block, so a broken
            guardrail fails closed rather than silently letting traffic through.
            ``"allow"`` records the error and continues. ``"raise"`` propagates,
            which is usually what you want in development.
        checkpointer: Optional LangGraph checkpointer for persistence.
        name: Name used for the compiled graph and in trace metadata.
        **model_kwargs: Forwarded to the provider client when ``model`` is a
            name. Ignored when an instance is passed.

    Raises:
        MissingIntegrationError: If ``telemetry=True`` but
            ``wardhook-observability`` is not installed.
        ValueError: If ``max_tool_iterations`` is not positive.

    Example:
        >>> from langchain_core.language_models.fake_chat_models import (
        ...     GenericFakeChatModel,
        ... )
        >>> from langchain_core.messages import AIMessage
        >>> model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello there.")]))
        >>> agent = AgentGraph(model=model)
        >>> agent.invoke("hi")["output"]
        'Hello there.'
    """

    def __init__(
        self,
        model: Any = None,
        *,
        tools: Iterable[Any] | None = None,
        guardrails: Iterable[Any] | None = None,
        telemetry: Any = False,
        system_prompt: str | None = None,
        retriever: Any = None,
        max_tool_iterations: int = 10,
        guardrail_error_policy: GuardrailErrorPolicy = "block",
        checkpointer: Any = None,
        name: str = "wardhook-agent",
        **model_kwargs: Any,
    ) -> None:
        """Build and compile the agent graph. See the class docstring."""
        if max_tool_iterations <= 0:
            raise ValueError(f"max_tool_iterations must be positive, got {max_tool_iterations}")

        self.name = name
        self.model = resolve_model(model, **model_kwargs)
        self.tools = normalize_tools(tools)
        self.guardrails = list(guardrails or [])
        self.retriever = retriever
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_tool_iterations = max_tool_iterations
        self.guardrail_error_policy: GuardrailErrorPolicy = guardrail_error_policy
        self.telemetry = self._resolve_telemetry(telemetry)

        self._tools_by_name = {t.name: t for t in self.tools}
        self._model = self._bind_tools()
        self._graph = self._build(checkpointer)

    def _bind_tools(self) -> Any:
        """Bind the tool schemas to the model.

        Returns:
            The tool-aware model, or the plain model when no tools are attached.

        Raises:
            TypeError: If tools were supplied but the model cannot accept them.
                Chat models that do not support tool calling raise
                :class:`NotImplementedError` from ``bind_tools``, which is
                opaque at the call site; this turns it into an actionable message.
        """
        if not self.tools:
            return self.model
        binder = getattr(self.model, "bind_tools", None)
        if not callable(binder):
            raise TypeError(
                f"{type(self.model).__name__} has no bind_tools() method, so it "
                f"cannot use the {len(self.tools)} tool(s) supplied. Use a "
                f"tool-calling model, or construct the agent without tools."
            )
        try:
            return binder(self.tools)
        except NotImplementedError as exc:
            raise TypeError(
                f"{type(self.model).__name__} does not support tool calling. "
                f"Use a tool-calling model, or construct the agent without tools."
            ) from exc

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _resolve_telemetry(self, telemetry: Any) -> Any:
        """Turn the ``telemetry`` argument into a sink or ``None``.

        The import of ``wardhook.observability`` happens here rather than at
        module scope. That is what allows ``wardhook-core`` to be installed and
        used entirely on its own.

        Args:
            telemetry: ``True``, ``False``/``None``, or a telemetry object.

        Returns:
            A telemetry sink, or ``None`` when tracing is disabled.

        Raises:
            MissingIntegrationError: If ``True`` was passed but the
                observability package is not installed.
        """
        if telemetry is False or telemetry is None:
            return None
        if telemetry is True:
            try:
                from wardhook.observability import Tracer
            except ImportError as exc:
                raise MissingIntegrationError(
                    "telemetry=True needs the wardhook-observability package, which "
                    "is not installed. Install it with:\n\n"
                    "    pip install wardhook-observability\n\n"
                    "Or pass your own object implementing TelemetryProtocol."
                ) from exc
            return Tracer()
        return telemetry

    def _build(self, checkpointer: Any) -> Any:
        """Assemble and compile the state graph.

        Only the nodes this agent actually needs are added, so the compiled
        graph stays as small as the configuration allows.

        Args:
            checkpointer: Optional LangGraph checkpointer.

        Returns:
            The compiled graph.
        """
        builder = StateGraph(AgentState)
        has_guardrails = bool(self.guardrails)

        _add_node(builder, "call_model", self._trace("call_model", self._call_model))
        if self.retriever is not None:
            _add_node(builder, "retrieve", self._trace("retrieve", self._retrieve))
        if self.tools:
            _add_node(builder, "tools", self._trace("tools", self._execute_tools))
        if has_guardrails:
            _add_node(builder, "guard_input", self._trace("guard_input", self._guard_input))
            _add_node(builder, "guard_output", self._trace("guard_output", self._guard_output))

        after_input = "retrieve" if self.retriever is not None else "call_model"

        if has_guardrails:
            builder.add_edge(START, "guard_input")
            builder.add_conditional_edges(
                "guard_input",
                self._route_after_input,
                {"continue": after_input, "blocked": END},
            )
        else:
            builder.add_edge(START, after_input)

        if self.retriever is not None:
            builder.add_edge("retrieve", "call_model")

        finish = "guard_output" if has_guardrails else END
        if self.tools:
            builder.add_conditional_edges(
                "call_model",
                self._route_after_model,
                {"tools": "tools", "finish": finish},
            )
            builder.add_edge("tools", "call_model")
        else:
            builder.add_edge("call_model", finish)

        if has_guardrails:
            builder.add_edge("guard_output", END)

        return builder.compile(checkpointer=checkpointer, name=self.name)

    def _trace(
        self, node: str, fn: Callable[[AgentState], dict[str, Any]]
    ) -> Callable[[AgentState], dict[str, Any]]:
        """Wrap a node so the telemetry sink sees its start and end.

        Args:
            node: The node's name.
            fn: The node implementation.

        Returns:
            ``fn`` unchanged when telemetry is disabled, otherwise a wrapper
            that reports timing and any error.
        """
        if self.telemetry is None:
            return fn

        @functools.wraps(fn)
        def wrapper(state: AgentState) -> dict[str, Any]:
            run_id = state.get("run_id", "")
            self.telemetry.start_node(node, run_id)
            try:
                result = fn(state)
            except Exception as exc:
                self.telemetry.end_node(node, run_id, error=f"{type(exc).__name__}: {exc}")
                raise
            self.telemetry.end_node(node, run_id)
            return result

        return wrapper

    # ------------------------------------------------------------------
    # Guardrail plumbing
    # ------------------------------------------------------------------

    def _run_hook(
        self,
        guardrail: Any,
        hook_name: str,
        args: tuple[Any, ...],
        context: Mapping[str, Any],
        fallback_text: str,
    ) -> GuardrailDecision | None:
        """Invoke one guardrail hook and normalise whatever comes back.

        Args:
            guardrail: The guardrail object.
            hook_name: Which hook to call.
            args: Positional arguments preceding ``context``.
            context: The run context mapping.
            fallback_text: Text to carry when the result supplies none.

        Returns:
            The normalised decision, or ``None`` if this guardrail does not
            implement the hook (every hook is optional).

        Raises:
            Exception: Re-raises a guardrail's own error when
                ``guardrail_error_policy`` is ``"raise"``.
        """
        hook = getattr(guardrail, hook_name, None)
        if hook is None or not callable(hook):
            return None

        name = str(getattr(guardrail, "name", type(guardrail).__name__))
        try:
            raw = hook(*args, context)
        except Exception as exc:
            if self.guardrail_error_policy == "raise":
                raise
            action = (
                GuardrailAction.BLOCK
                if self.guardrail_error_policy == "block"
                else GuardrailAction.ALLOW
            )
            return GuardrailDecision(
                action,
                fallback_text,
                reason=f"guardrail raised {type(exc).__name__}: {exc}",
                rule="guardrail-error",
                guardrail=name,
                details={"error": True, "policy": self.guardrail_error_policy},
            )

        return read_guardrail_result(raw, original_text=fallback_text, guardrail_name=name)

    def _apply_text_guardrails(
        self, text: str, context: Mapping[str, Any], hook_name: str
    ) -> tuple[str, list[dict[str, Any]], GuardrailDecision | None]:
        """Run every guardrail's text hook in order, threading redactions through.

        Each guardrail sees the output of the previous one, so a redaction by an
        early guardrail is what a later one inspects. The chain stops at the
        first block.

        Args:
            text: The text to inspect.
            context: The run context mapping.
            hook_name: ``"on_input"`` or ``"on_output"``.

        Returns:
            A tuple of the possibly-redacted text, the audit records for every
            decision that was not a plain allow, and the blocking decision if
            there was one.
        """
        current = text
        events: list[dict[str, Any]] = []
        for guardrail in self.guardrails:
            decision = self._run_hook(guardrail, hook_name, (current,), context, current)
            if decision is None:
                continue
            if _is_auditable(decision):
                events.append({**decision.to_dict(), "stage": context.get("stage")})
            if decision.blocked:
                return current, events, decision
            if decision.modified:
                current = decision.text
        return current, events, None

    def _context(self, state: AgentState, stage: str, node: str) -> dict[str, Any]:
        """Build the context mapping handed to every guardrail hook.

        Args:
            state: Current graph state.
            stage: ``"input"``, ``"output"`` or ``"tool_call"``.
            node: The node currently executing.

        Returns:
            A plain dict, so implementers never import a Wardhook type to read it.
        """
        return {
            "run_id": state.get("run_id", ""),
            "stage": stage,
            "node": node,
            "principal": state.get("principal"),
            "agent": self.name,
        }

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _guard_input(self, state: AgentState) -> dict[str, Any]:
        """Screen the latest user message before it reaches the model.

        Args:
            state: Current graph state.

        Returns:
            A partial state update. A redaction replaces the user message in
            place (same id), so the model never sees the original text.
        """
        messages = state["messages"]
        last_human = _last_of_type(messages, HumanMessage)
        original = _text_of(last_human)
        context = self._context(state, "input", "guard_input")
        text, events, blocking = self._apply_text_guardrails(original, context, "on_input")

        update: dict[str, Any] = {"guardrail_events": events}
        if blocking is not None:
            reason = blocking.reason or f"blocked by {blocking.guardrail}"
            update["blocked"] = True
            update["block_reason"] = reason
            update["messages"] = [AIMessage(content=BLOCKED_INPUT_MESSAGE)]
            return update

        if text != original and last_human is not None:
            # add_messages replaces by id, so reusing the id swaps the message
            # rather than appending a second copy of the turn.
            update["messages"] = [HumanMessage(content=text, id=last_human.id)]
        return update

    def _guard_output(self, state: AgentState) -> dict[str, Any]:
        """Screen the model's reply before it reaches the caller.

        Args:
            state: Current graph state.

        Returns:
            A partial state update, replacing the reply in place if it was
            redacted or withheld.
        """
        if state.get("blocked"):
            return {}

        messages = state["messages"]
        last_ai = _last_of_type(messages, AIMessage)
        original = _text_of(last_ai)
        context = self._context(state, "output", "guard_output")
        text, events, blocking = self._apply_text_guardrails(original, context, "on_output")

        update: dict[str, Any] = {"guardrail_events": events}
        if blocking is not None:
            reason = blocking.reason or f"blocked by {blocking.guardrail}"
            update["blocked"] = True
            update["block_reason"] = reason
            if last_ai is not None:
                update["messages"] = [AIMessage(content=BLOCKED_OUTPUT_MESSAGE, id=last_ai.id)]
            return update

        if text != original and last_ai is not None:
            update["messages"] = [AIMessage(content=text, id=last_ai.id)]
        return update

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        """Fetch context for the latest question.

        Args:
            state: Current graph state.

        Returns:
            A partial state update carrying citation records. Retrieval runs
            once per invocation, not once per tool loop, because the question
            does not change between tool round trips.
        """
        if state.get("citations"):
            return {}
        query = _text_of(_last_of_type(state["messages"], HumanMessage))
        if not query:
            return {}
        _, citations = self.retriever.context_for(query)
        return {"citations": citations, "context": citations}

    def _system_message(self, state: AgentState) -> SystemMessage:
        """Build the system message for this turn.

        Args:
            state: Current graph state.

        Returns:
            The system message, with retrieval instructions and the numbered
            context block appended when retrieval produced anything.
        """
        parts = [self.system_prompt]
        citations = state.get("citations") or []
        if self.retriever is not None and citations:
            parts.append(DEFAULT_RAG_INSTRUCTIONS)
            parts.append("Context:\n" + format_citations(citations))
        return SystemMessage(content="\n\n".join(parts))

    def _call_model(self, state: AgentState) -> dict[str, Any]:
        """Invoke the model with the conversation so far.

        Args:
            state: Current graph state.

        Returns:
            A partial state update containing the model's reply.
        """
        messages: list[BaseMessage] = [self._system_message(state), *state["messages"]]
        config: RunnableConfig | None = None
        if self.telemetry is not None:
            callbacks = list(self.telemetry.callbacks())
            if callbacks:
                config = RunnableConfig(callbacks=callbacks)
        response = self._model.invoke(messages, config=config)
        return {"messages": [response]}

    def _execute_tools(self, state: AgentState) -> dict[str, Any]:
        """Run the tools the model asked for, subject to guardrail approval.

        A blocked call is never executed. The model is told it was denied via a
        normal tool result, which lets it recover by trying something else
        rather than failing the whole run.

        Args:
            state: Current graph state.

        Returns:
            A partial state update with one tool message per requested call.
        """
        last_ai = _last_of_type(state["messages"], AIMessage)
        tool_calls = list(getattr(last_ai, "tool_calls", None) or [])
        context = self._context(state, "tool_call", "tools")

        results: list[BaseMessage] = []
        events: list[dict[str, Any]] = []

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id", "")

            denial: GuardrailDecision | None = None
            for guardrail in self.guardrails:
                decision = self._run_hook(guardrail, "on_tool_call", (name, args), context, name)
                if decision is None:
                    continue
                if _is_auditable(decision):
                    events.append({**decision.to_dict(), "stage": "tool_call", "tool": name})
                if decision.blocked:
                    denial = decision
                    break

            if denial is not None:
                reason = denial.reason or f"denied by {denial.guardrail}"
                results.append(
                    ToolMessage(
                        content=(
                            f"Tool call denied by policy: {reason}. "
                            f"Do not retry this call; use another approach or "
                            f"tell the user it is not permitted."
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            tool = self._tools_by_name.get(name)
            if tool is None:
                results.append(
                    ToolMessage(
                        content=(
                            f"Unknown tool {name!r}. Available tools: "
                            f"{', '.join(tool_names(self.tools)) or 'none'}."
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            try:
                output = tool.invoke(args)
            except Exception as exc:
                # Returning the error as a tool result lets the model correct
                # itself; raising would abort a run that is often recoverable.
                results.append(
                    ToolMessage(
                        content=f"Tool {name!r} raised {type(exc).__name__}: {exc}",
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            results.append(
                ToolMessage(
                    content=output if isinstance(output, str) else repr(output),
                    tool_call_id=call_id,
                    name=name,
                )
            )

        return {"messages": results, "guardrail_events": events}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_input(self, state: AgentState) -> str:
        """Decide whether to continue after input screening.

        Args:
            state: Current graph state.

        Returns:
            ``"blocked"`` or ``"continue"``.
        """
        return "blocked" if state.get("blocked") else "continue"

    def _route_after_model(self, state: AgentState) -> str:
        """Decide whether to run tools or finish.

        Args:
            state: Current graph state.

        Returns:
            ``"tools"`` when the model requested a call and the iteration
            budget allows it, otherwise ``"finish"``.
        """
        messages = state["messages"]
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "finish"

        rounds = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
        if rounds > self.max_tool_iterations:
            return "finish"
        return "tools"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Any:
        """The compiled LangGraph graph, for streaming or visualisation."""
        return self._graph

    def _initial_state(
        self,
        user_input: Any,
        principal: Principal | Mapping[str, Any] | None,
        run_id: str,
    ) -> dict[str, Any]:
        """Normalise caller input into a starting state.

        Args:
            user_input: A string, a list of messages, or a mapping with an
                ``input``, ``question``, ``query`` or ``messages`` key. The
                mapping form is what makes an agent a valid ``wardhook-evals``
                target without an adapter.
            principal: Caller identity, if any.
            run_id: Identifier for this invocation.

        Returns:
            The initial graph state.

        Raises:
            TypeError: If the input shape is not recognised.
        """
        messages: list[BaseMessage]
        if isinstance(user_input, str):
            messages = [HumanMessage(content=user_input)]
        elif isinstance(user_input, BaseMessage):
            messages = [user_input]
        elif isinstance(user_input, Mapping):
            if "messages" in user_input:
                raw = user_input["messages"]
                messages = [
                    m if isinstance(m, BaseMessage) else HumanMessage(content=str(m)) for m in raw
                ]
            else:
                for key in ("input", "question", "query", "text", "prompt"):
                    if key in user_input:
                        messages = [HumanMessage(content=str(user_input[key]))]
                        break
                else:
                    raise TypeError(
                        "Mapping input must contain one of: 'messages', 'input', "
                        f"'question', 'query', 'text', 'prompt'. Got keys: "
                        f"{sorted(user_input)}."
                    )
            if principal is None:
                principal = user_input.get("principal")
        elif isinstance(user_input, Sequence):
            messages = [
                m if isinstance(m, BaseMessage) else HumanMessage(content=str(m))
                for m in user_input
            ]
        else:
            raise TypeError(
                f"Unsupported input type {type(user_input).__name__}. Pass a string, "
                f"a message, a list of messages, or a mapping."
            )

        return {
            "messages": messages,
            "run_id": run_id,
            "principal": principal,
            "citations": [],
            "context": [],
            "guardrail_events": [],
            "blocked": False,
            "block_reason": None,
        }

    def _format_result(self, state: Mapping[str, Any], run_id: str) -> dict[str, Any]:
        """Turn the final graph state into the invoke() return value.

        Args:
            state: The graph's final state.
            run_id: Identifier for this invocation.

        Returns:
            A plain, JSON-friendly dict.
        """
        messages = list(state.get("messages", []))
        final_ai = _last_of_type(messages, AIMessage)
        called: list[str] = [str(m.name) for m in messages if isinstance(m, ToolMessage) and m.name]
        return {
            "output": _text_of(final_ai),
            "messages": messages,
            "citations": list(state.get("citations") or []),
            "guardrail_events": list(state.get("guardrail_events") or []),
            "blocked": bool(state.get("blocked")),
            "block_reason": state.get("block_reason"),
            "tool_calls": called,
            "run_id": run_id,
            "model": describe_model(self.model),
        }

    def invoke(
        self,
        user_input: Any,
        *,
        principal: Principal | Mapping[str, Any] | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the agent to completion.

        Args:
            user_input: A string, a message, a list of messages, or a mapping
                with an ``input``/``question``/``messages`` key.
            principal: Caller identity for role-based guardrail decisions.
            run_id: Identifier for this invocation. Generated when omitted.
            config: Extra LangGraph runtime configuration.

        Returns:
            A dict with keys ``output``, ``messages``, ``citations``,
            ``guardrail_events``, ``blocked``, ``block_reason``, ``tool_calls``,
            ``run_id`` and ``model``.
        """
        run_id = run_id or uuid.uuid4().hex
        state = self._initial_state(user_input, principal, run_id)

        if self.telemetry is not None:
            self.telemetry.start_run(
                run_id, {"agent": self.name, "model": describe_model(self.model)}
            )
        try:
            final = self._graph.invoke(state, config=dict(config) if config else None)
        except Exception as exc:
            if self.telemetry is not None:
                self.telemetry.end_run(run_id, error=f"{type(exc).__name__}: {exc}")
            raise
        if self.telemetry is not None:
            self.telemetry.end_run(run_id)

        return self._format_result(final, run_id)

    async def ainvoke(
        self,
        user_input: Any,
        *,
        principal: Principal | Mapping[str, Any] | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Asynchronous counterpart to :meth:`invoke`.

        Args:
            user_input: As :meth:`invoke`.
            principal: As :meth:`invoke`.
            run_id: As :meth:`invoke`.
            config: As :meth:`invoke`.

        Returns:
            The same dict shape as :meth:`invoke`.
        """
        run_id = run_id or uuid.uuid4().hex
        state = self._initial_state(user_input, principal, run_id)

        if self.telemetry is not None:
            self.telemetry.start_run(
                run_id, {"agent": self.name, "model": describe_model(self.model)}
            )
        try:
            final = await self._graph.ainvoke(state, config=dict(config) if config else None)
        except Exception as exc:
            if self.telemetry is not None:
                self.telemetry.end_run(run_id, error=f"{type(exc).__name__}: {exc}")
            raise
        if self.telemetry is not None:
            self.telemetry.end_run(run_id)

        return self._format_result(final, run_id)

    def trace(self, run_id: str | None = None) -> Any:
        """Return the recorded trace for a run, if telemetry is attached.

        Args:
            run_id: The run to look up. Defaults to the most recent.

        Returns:
            The trace object, or ``None`` when no telemetry sink is attached or
            the sink does not expose ``get_trace``.
        """
        if self.telemetry is None:
            return None
        getter = getattr(self.telemetry, "get_trace", None)
        return getter(run_id) if callable(getter) else None

    def __repr__(self) -> str:
        """Return a debug representation summarising the configuration."""
        return (
            f"AgentGraph(name={self.name!r}, model={describe_model(self.model)!r}, "
            f"tools={len(self.tools)}, guardrails={len(self.guardrails)}, "
            f"retriever={self.retriever is not None}, telemetry={self.telemetry is not None})"
        )
