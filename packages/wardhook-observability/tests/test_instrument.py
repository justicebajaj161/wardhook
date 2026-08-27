"""Instrumenting a graph nobody told us about derives its nodes correctly."""

from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from wardhook.observability import Tracer, instrument, uninstrument
from wardhook.observability.callbacks import GraphTraceCallback

# LangGraph is deliberately NOT a dependency of wardhook-observability: node
# boundaries come from the langchain-core callback stream, and instrument()
# works against anything exposing .invoke(). These tests exercise the LangGraph
# path specifically, so they skip rather than fail in a standalone install.
pytest.importorskip("langgraph", reason="wardhook-observability does not depend on langgraph")

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: int
    text: str


def build_graph(fail_on: str | None = None):
    """A two-node graph, optionally rigged to raise in one of them."""

    def first(state: State) -> dict:
        if fail_on == "first":
            raise ValueError("boom")
        return {"value": state["value"] + 1}

    def second(state: State) -> dict:
        if fail_on == "second":
            raise ValueError("boom")
        return {"value": state["value"] * 2}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile()


class TestNodeDerivation:
    def test_nodes_are_discovered_from_langgraph_metadata(self):
        graph = build_graph()
        tracer = instrument(graph)

        assert graph.invoke({"value": 1, "text": ""})["value"] == 4

        trace = tracer.get_trace()
        assert [step.node for step in trace.steps] == ["first", "second"]
        assert trace.metadata == {"source": "instrument"}
        assert all(step.latency_ms >= 0 for step in trace.steps)

    def test_each_node_is_recorded_exactly_once(self):
        # Regression: LangGraph's `invoke` delegates to its own `stream`, and
        # both are wrapped. Without an idempotence check on the merged config,
        # a second handler attaches and every node is counted -- and billed --
        # twice.
        graph = build_graph()
        tracer = instrument(graph)
        graph.invoke({"value": 1, "text": ""})

        nodes = [step.node for step in tracer.get_trace().steps]
        assert nodes == ["first", "second"], f"nodes recorded more than once: {nodes}"

    def test_repeated_invocations_produce_separate_traces(self):
        graph = build_graph()
        tracer = instrument(graph)
        for _ in range(3):
            graph.invoke({"value": 1, "text": ""})
        assert len(tracer.traces()) == 3

    def test_a_failing_node_is_recorded_with_its_error(self):
        graph = build_graph(fail_on="second")
        tracer = instrument(graph)

        with pytest.raises(ValueError, match="boom"):
            graph.invoke({"value": 1, "text": ""})

        trace = tracer.get_trace()
        failed = [step for step in trace.steps if step.failed]
        assert [step.node for step in failed] == ["second"]
        assert "ValueError: boom" in failed[0].error

    def test_model_usage_is_attributed_to_the_calling_node(self, fake_model):
        model = fake_model(input_tokens=900, output_tokens=120)

        def ask(state: State) -> dict:
            return {"text": model.invoke("hi").content}

        builder = StateGraph(State)
        builder.add_node("ask", ask)
        builder.add_edge(START, "ask")
        builder.add_edge("ask", END)
        graph = builder.compile()

        tracer = instrument(graph)
        graph.invoke({"value": 0, "text": ""})

        step = next(s for s in tracer.get_trace().steps if s.node == "ask")
        assert (step.tokens_in, step.tokens_out) == (900, 120)
        assert step.cost == pytest.approx(900 * 5 / 1e6 + 120 * 25 / 1e6)

    def test_streaming_is_traced_too(self):
        graph = build_graph()
        tracer = instrument(graph)
        list(graph.stream({"value": 1, "text": ""}))
        assert [step.node for step in tracer.get_trace().steps] == ["first", "second"]

    def test_ainvoke_is_traced(self):
        graph = build_graph()
        tracer = instrument(graph)
        result = asyncio.run(graph.ainvoke({"value": 1, "text": ""}))
        assert result["value"] == 4
        assert [step.node for step in tracer.get_trace().steps] == ["first", "second"]


class TestAttachment:
    def test_instrumenting_twice_returns_the_first_tracer(self):
        graph = build_graph()
        first = instrument(graph)
        second = instrument(graph)
        assert first is second

        graph.invoke({"value": 1, "text": ""})
        assert [step.node for step in first.get_trace().steps] == ["first", "second"]

    def test_a_supplied_tracer_is_used(self):
        graph = build_graph()
        mine = Tracer(max_runs=5)
        assert instrument(graph, mine) is mine
        graph.invoke({"value": 1, "text": ""})
        assert mine.get_trace() is not None

    def test_the_callers_own_callbacks_survive(self):
        seen: list[str] = []

        class Counter(BaseCallbackHandler):
            def on_chain_start(self, serialized, inputs, **kwargs):
                seen.append("start")

        graph = build_graph()
        tracer = instrument(graph)
        graph.invoke({"value": 1, "text": ""}, config={"callbacks": [Counter()]})

        assert seen, "the caller's handler was dropped"
        assert [step.node for step in tracer.get_trace().steps] == ["first", "second"]

    def test_a_handler_for_this_tracer_is_never_added_twice(self):
        graph = build_graph()
        tracer = instrument(graph)
        # Passing our own handler in explicitly must not produce a second one.
        graph.invoke({"value": 1, "text": ""}, config={"callbacks": [GraphTraceCallback(tracer)]})
        assert [step.node for step in tracer.get_trace().steps] == ["first", "second"]

    def test_uninstrument_restores_the_original_methods(self):
        graph = build_graph()
        tracer = instrument(graph)
        uninstrument(graph)

        graph.invoke({"value": 1, "text": ""})
        assert tracer.get_trace() is None
        # And it can be re-instrumented afterwards.
        again = instrument(graph)
        graph.invoke({"value": 1, "text": ""})
        assert again.get_trace() is not None

    def test_uninstrumenting_a_plain_graph_is_a_no_op(self):
        uninstrument(build_graph())

    def test_instrumenting_something_that_is_not_a_graph_explains_itself(self):
        with pytest.raises(TypeError, match="nothing to instrument"):
            instrument(object())
