"""Tests for AgentGraph: graph shape, guardrail seams, tools, and RAG wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from wardhook.core.agent import BLOCKED_INPUT_MESSAGE, AgentGraph, MissingIntegrationError
from wardhook.core.protocols import GuardrailAction, read_guardrail_result
from wardhook.core.rag.retriever import Retriever

TOOL_CALL = [{"name": "lookup_account", "args": {"account_id": "A-1"}, "id": "call-1"}]


def allow(text):
    return SimpleNamespace(action="allow", text=text, reason=None, rule=None)


class Redactor:
    """Replaces a marker word, so the model provably never sees the original."""

    name = "redactor"

    def on_input(self, text, context):
        if "secret" in text:
            return SimpleNamespace(
                action="redact",
                text=text.replace("secret", "[REDACTED]"),
                reason="masked a marker word",
                rule="marker",
            )
        return allow(text)


class OutputRedactor:
    name = "output-redactor"

    def on_output(self, text, context):
        return SimpleNamespace(
            action="redact", text=text.upper(), reason="shouty policy", rule="upper"
        )


class Blocker:
    name = "blocker"

    def __init__(self, trigger="forbidden", stage="on_input"):
        self.trigger = trigger
        self.stage = stage

    def _decide(self, text):
        if self.trigger in text:
            return SimpleNamespace(
                action="block", text=text, reason="matched a prohibited term", rule="denylist"
            )
        return allow(text)

    def on_input(self, text, context):
        return self._decide(text) if self.stage == "on_input" else allow(text)

    def on_output(self, text, context):
        return self._decide(text) if self.stage == "on_output" else allow(text)


class RoleGate:
    """Approves tool calls only for callers holding a role."""

    name = "role-gate"

    def __init__(self, required="teller"):
        self.required = required

    def on_tool_call(self, tool_name, tool_args, context):
        roles = (context.get("principal") or {}).get("roles", [])
        if self.required in roles:
            return allow(tool_name)
        return SimpleNamespace(
            action="block", text=tool_name, reason=f"requires role {self.required}", rule="rbac"
        )


class ContextSpy:
    """Records the context mapping core hands to guardrails."""

    name = "spy"

    def __init__(self):
        self.seen = []

    def on_input(self, text, context):
        self.seen.append(dict(context))
        return allow(text)


class Exploding:
    name = "exploding"

    def on_input(self, text, context):
        raise RuntimeError("regex blew up")


class TestBareAgent:
    def test_answers_without_any_optional_features(self, fake_model):
        result = AgentGraph(model=fake_model).invoke("hi")
        assert result["output"] == "Hello there."
        assert result["blocked"] is False
        assert result["guardrail_events"] == []
        assert result["citations"] == []

    def test_works_with_no_wardhook_siblings_installed(self, fake_model):
        # The core package must never import its siblings at module scope.
        agent = AgentGraph(model=fake_model)
        assert agent.telemetry is None
        assert agent.guardrails == []

    def test_result_carries_a_run_id(self, fake_model):
        assert len(AgentGraph(model=fake_model).invoke("hi")["run_id"]) == 32

    def test_caller_supplied_run_id_is_honoured(self, fake_model):
        result = AgentGraph(model=fake_model).invoke("hi", run_id="run-42")
        assert result["run_id"] == "run-42"

    def test_rejects_a_non_positive_iteration_budget(self, fake_model):
        with pytest.raises(ValueError, match="max_tool_iterations"):
            AgentGraph(model=fake_model, max_tool_iterations=0)

    def test_repr_summarises_the_configuration(self, fake_model):
        assert "tools=0" in repr(AgentGraph(model=fake_model))


class TestInputShapes:
    @pytest.mark.parametrize(
        "payload",
        [
            "hi",
            {"input": "hi"},
            {"question": "hi"},
            {"query": "hi"},
            {"messages": ["hi"]},
            ["hi"],
        ],
    )
    def test_accepts_every_documented_input_shape(self, make_model, payload):
        # The mapping form is what makes an agent a valid wardhook-evals target
        # with no adapter in between.
        assert AgentGraph(model=make_model("ok")).invoke(payload)["output"] == "ok"

    def test_accepts_a_message_object(self, make_model):
        result = AgentGraph(model=make_model("ok")).invoke(HumanMessage(content="hi"))
        assert result["output"] == "ok"

    def test_mapping_without_a_known_key_is_rejected(self, fake_model):
        with pytest.raises(TypeError, match="Mapping input must contain"):
            AgentGraph(model=fake_model).invoke({"payload": "hi"})

    def test_unsupported_type_is_rejected(self, fake_model):
        with pytest.raises(TypeError, match="Unsupported input type"):
            AgentGraph(model=fake_model).invoke(3.14)

    def test_principal_can_travel_inside_the_mapping(self, make_model):
        spy = ContextSpy()
        agent = AgentGraph(model=make_model("ok"), guardrails=[spy])
        agent.invoke({"input": "hi", "principal": {"id": "u1", "roles": ["admin"]}})
        assert spy.seen[0]["principal"]["roles"] == ["admin"]


class TestInputGuardrails:
    def test_redaction_reaches_the_model_not_the_original(self, make_model):
        agent = AgentGraph(model=make_model("ack"), guardrails=[Redactor()])
        result = agent.invoke("my secret code")
        assert "[REDACTED]" in result["messages"][0].content
        assert "secret" not in result["messages"][0].content

    def test_redaction_replaces_rather_than_appends_the_turn(self, make_model):
        agent = AgentGraph(model=make_model("ack"), guardrails=[Redactor()])
        result = agent.invoke("my secret code")
        assert sum(isinstance(m, HumanMessage) for m in result["messages"]) == 1

    def test_blocking_short_circuits_before_the_model_runs(self, make_model):
        # The scripted model has no replies; reaching it would raise.
        agent = AgentGraph(model=make_model(), guardrails=[Blocker()])
        result = agent.invoke("something forbidden")
        assert result["blocked"] is True
        assert result["output"] == BLOCKED_INPUT_MESSAGE
        assert result["block_reason"] == "matched a prohibited term"

    def test_guardrails_chain_so_each_sees_the_previous_redaction(self, make_model):
        class SeesRedaction:
            name = "downstream"

            def __init__(self):
                self.observed = None

            def on_input(self, text, context):
                self.observed = text
                return allow(text)

        downstream = SeesRedaction()
        agent = AgentGraph(model=make_model("ok"), guardrails=[Redactor(), downstream])
        agent.invoke("my secret code")
        assert downstream.observed == "my [REDACTED] code"

    def test_events_accumulate_across_nodes(self, make_model):
        # Regression guard: without an accumulating reducer on the state field,
        # the output stage's events would overwrite the input stage's.
        agent = AgentGraph(
            model=make_model("some reply"), guardrails=[Redactor(), OutputRedactor()]
        )
        stages = [e["stage"] for e in agent.invoke("my secret code")["guardrail_events"]]
        assert stages == ["input", "output"]

    def test_events_exclude_the_inspected_text(self, make_model):
        agent = AgentGraph(model=make_model("ack"), guardrails=[Redactor()])
        event = agent.invoke("my secret code")["guardrail_events"][0]
        assert "text" not in event
        assert event["rule"] == "marker"

    def test_context_carries_the_documented_keys(self, make_model):
        spy = ContextSpy()
        AgentGraph(model=make_model("ok"), guardrails=[spy], name="acme").invoke(
            "hi", run_id="r1", principal={"id": "u9", "roles": []}
        )
        seen = spy.seen[0]
        assert seen["run_id"] == "r1"
        assert seen["stage"] == "input"
        assert seen["node"] == "guard_input"
        assert seen["agent"] == "acme"

    def test_a_guardrail_may_implement_only_the_hooks_it_needs(self, make_model):
        class ToolOnly:
            name = "tool-only"

            def on_tool_call(self, tool_name, tool_args, context):
                return allow(tool_name)

        assert AgentGraph(model=make_model("ok"), guardrails=[ToolOnly()]).invoke("hi")["output"]


class TestOutputGuardrails:
    def test_output_redaction_replaces_the_reply(self, make_model):
        agent = AgentGraph(model=make_model("quiet reply"), guardrails=[OutputRedactor()])
        assert agent.invoke("hi")["output"] == "QUIET REPLY"

    def test_output_blocking_withholds_the_reply(self, make_model):
        agent = AgentGraph(
            model=make_model("this is forbidden"),
            guardrails=[Blocker(stage="on_output")],
        )
        result = agent.invoke("hi")
        assert result["blocked"] is True
        assert "forbidden" not in result["output"]


class TestGuardrailErrorPolicy:
    def test_fails_closed_by_default(self, make_model):
        result = AgentGraph(model=make_model("ok"), guardrails=[Exploding()]).invoke("hi")
        assert result["blocked"] is True
        assert "RuntimeError" in result["block_reason"]

    def test_can_be_configured_to_fail_open(self, make_model):
        agent = AgentGraph(
            model=make_model("ok"), guardrails=[Exploding()], guardrail_error_policy="allow"
        )
        result = agent.invoke("hi")
        assert result["blocked"] is False
        assert result["guardrail_events"][0]["details"]["error"] is True

    def test_can_be_configured_to_propagate(self, make_model):
        agent = AgentGraph(
            model=make_model("ok"), guardrails=[Exploding()], guardrail_error_policy="raise"
        )
        with pytest.raises(RuntimeError, match="regex blew up"):
            agent.invoke("hi")


class TestTools:
    def test_executes_a_tool_and_feeds_the_result_back(self, make_tool_model, echo_tool):
        model = make_tool_model(
            AIMessage(content="", tool_calls=TOOL_CALL),
            AIMessage(content="Account A-1 is active."),
        )
        result = AgentGraph(model=model, tools=[echo_tool]).invoke("check A-1")
        assert result["tool_calls"] == ["lookup_account"]
        assert result["output"] == "Account A-1 is active."

    def test_rbac_denial_stops_the_call_without_executing_it(self, make_tool_model):
        executed = []

        def lookup_account(account_id: str) -> str:
            """Look up an account by its identifier."""
            executed.append(account_id)
            return "should never run"

        model = make_tool_model(
            AIMessage(content="", tool_calls=TOOL_CALL),
            AIMessage(content="I am not permitted to do that."),
        )
        agent = AgentGraph(model=model, tools=[lookup_account], guardrails=[RoleGate()])
        result = agent.invoke("check A-1", principal={"id": "u1", "roles": ["guest"]})

        assert executed == [], "a denied tool must never execute"
        denial = next(m for m in result["messages"] if m.type == "tool")
        assert "denied by policy" in denial.content
        assert result["guardrail_events"][0]["tool"] == "lookup_account"

    def test_rbac_allows_a_caller_holding_the_role(self, make_tool_model, echo_tool):
        model = make_tool_model(
            AIMessage(content="", tool_calls=TOOL_CALL),
            AIMessage(content="Account A-1 is active."),
        )
        agent = AgentGraph(model=model, tools=[echo_tool], guardrails=[RoleGate()])
        result = agent.invoke("check A-1", principal={"id": "u2", "roles": ["teller"]})
        assert result["tool_calls"] == ["lookup_account"]

    def test_a_raising_tool_is_reported_to_the_model_not_the_caller(self, make_tool_model):
        def lookup_account(account_id: str) -> str:
            """Look up an account by its identifier."""
            raise ValueError("upstream is down")

        model = make_tool_model(
            AIMessage(content="", tool_calls=TOOL_CALL),
            AIMessage(content="I could not reach the account service."),
        )
        result = AgentGraph(model=model, tools=[lookup_account]).invoke("check A-1")
        tool_message = next(m for m in result["messages"] if m.type == "tool")
        assert "upstream is down" in tool_message.content
        assert result["output"] == "I could not reach the account service."

    def test_unknown_tool_name_is_reported_back(self, make_tool_model, echo_tool):
        model = make_tool_model(
            AIMessage(content="", tool_calls=[{"name": "ghost", "args": {}, "id": "c1"}]),
            AIMessage(content="That tool does not exist."),
        )
        result = AgentGraph(model=model, tools=[echo_tool]).invoke("go")
        tool_message = next(m for m in result["messages"] if m.type == "tool")
        assert "Unknown tool" in tool_message.content

    def test_iteration_budget_stops_a_looping_model(self, make_tool_model, echo_tool):
        looping = [AIMessage(content="", tool_calls=TOOL_CALL) for _ in range(12)]
        agent = AgentGraph(
            model=make_tool_model(*looping), tools=[echo_tool], max_tool_iterations=3
        )
        result = agent.invoke("loop forever")
        assert len(result["tool_calls"]) <= 4

    def test_a_model_without_tool_support_fails_clearly(self, fake_model, echo_tool):
        with pytest.raises(TypeError, match="does not support tool calling"):
            AgentGraph(model=fake_model, tools=[echo_tool])


class TestRetrieval:
    def test_attaches_citations_to_the_result(self, make_model, sample_store):
        agent = AgentGraph(
            model=make_model("The excess is 500 [1]."), retriever=Retriever(sample_store, k=1)
        )
        result = agent.invoke("what excess applies to storm damage?")
        assert result["citations"][0]["source"] == "policy.md"
        assert "score" in result["citations"][0]

    def test_context_is_injected_into_the_system_prompt(self, make_model, sample_store):
        captured = {}

        class Recorder:
            def invoke(self, messages, config=None):
                captured["system"] = messages[0].content
                return AIMessage(content="ok")

        agent = AgentGraph(model=Recorder(), retriever=Retriever(sample_store, k=1))
        agent.invoke("storm damage excess")
        assert "[1] policy.md" in captured["system"]
        assert "Cite the passages" in captured["system"]

    def test_no_retriever_means_no_citations(self, make_model):
        assert AgentGraph(model=make_model("ok")).invoke("hi")["citations"] == []


class TestTelemetrySeam:
    def test_true_without_the_package_raises_an_actionable_error(self, fake_model, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "wardhook.observability":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(MissingIntegrationError, match="pip install wardhook-observability"):
            AgentGraph(model=fake_model, telemetry=True)

    def test_a_custom_sink_receives_the_node_lifecycle(self, make_model):
        class Sink:
            def __init__(self):
                self.events = []

            def start_run(self, run_id, metadata=None):
                self.events.append(("start_run", run_id))

            def end_run(self, run_id, error=None):
                self.events.append(("end_run", run_id))

            def start_node(self, node, run_id):
                self.events.append(("start_node", node))

            def end_node(self, node, run_id, error=None):
                self.events.append(("end_node", node))

            def callbacks(self):
                return []

        sink = Sink()
        AgentGraph(model=make_model("ok"), telemetry=sink).invoke("hi", run_id="r1")
        assert sink.events[0] == ("start_run", "r1")
        assert ("start_node", "call_model") in sink.events
        assert sink.events[-1] == ("end_run", "r1")

    def test_trace_is_none_without_telemetry(self, fake_model):
        assert AgentGraph(model=fake_model).trace() is None


class TestReadGuardrailResult:
    """The normaliser is what makes cross-package duck typing safe."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, GuardrailAction.ALLOW),
            (True, GuardrailAction.ALLOW),
            (False, GuardrailAction.BLOCK),
            ({"action": "redact"}, GuardrailAction.REDACT),
            ({"action": "BLOCK"}, GuardrailAction.BLOCK),
            (SimpleNamespace(action=GuardrailAction.BLOCK), GuardrailAction.BLOCK),
        ],
    )
    def test_normalises_every_supported_return_shape(self, value, expected):
        assert (
            read_guardrail_result(value, original_text="t", guardrail_name="g").action is expected
        )

    def test_an_unrecognised_action_allows_but_records_the_anomaly(self):
        # A guardrail returning nonsense must not take down a production agent,
        # but the anomaly still has to be visible in the audit trail.
        decision = read_guardrail_result(
            {"action": "explode"}, original_text="t", guardrail_name="g"
        )
        assert decision.action is GuardrailAction.ALLOW
        assert decision.details["raw_action"] == "explode"

    def test_falls_back_to_the_original_text(self):
        decision = read_guardrail_result(
            {"action": "allow"}, original_text="kept", guardrail_name="g"
        )
        assert decision.text == "kept"

    def test_to_dict_never_leaks_the_text(self):
        decision = read_guardrail_result(
            {"action": "redact", "text": "123-45-6789"}, original_text="x", guardrail_name="pii"
        )
        assert "123-45-6789" not in str(decision.to_dict())
        assert "123-45-6789" not in repr(decision)
