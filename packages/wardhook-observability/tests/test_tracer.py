"""The tracer attributes usage correctly, stays bounded, and never raises."""

from __future__ import annotations

import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from wardhook.observability import (
    JSONLTraceStore,
    TokenUsage,
    Tracer,
    UsageCallback,
)
from wardhook.observability.callbacks import usage_from_response
from wardhook.observability.tracer import UNGROUPED_NODE


def _llm_result(input_tokens=900, output_tokens=120, cache_read=0, model="claude-opus-5"):
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestLifecycle:
    def test_a_run_records_its_nodes_in_order(self, tracer):
        tracer.start_run("r1", {"agent": "support"})
        for node in ("guard_input", "retrieve", "call_model"):
            tracer.start_node(node, "r1")
            tracer.end_node(node, "r1")
        tracer.end_run("r1")

        trace = tracer.get_trace("r1")
        assert [step.node for step in trace.steps] == ["guard_input", "retrieve", "call_model"]
        assert trace.metadata == {"agent": "support"}
        assert not trace.failed

    def test_get_trace_with_no_argument_returns_the_latest_run(self, tracer):
        for run_id in ("r1", "r2"):
            tracer.start_run(run_id)
            tracer.end_run(run_id)
        # AgentGraph.trace() calls get_trace(None) positionally.
        assert tracer.get_trace(None).run_id == "r2"
        assert tracer.get_trace().run_id == "r2"

    def test_get_trace_returns_none_for_unknown_or_empty(self, tracer):
        assert tracer.get_trace() is None
        assert tracer.get_trace("never-happened") is None

    def test_an_in_flight_run_yields_a_partial_trace(self, tracer):
        tracer.start_run("r1")
        tracer.start_node("call_model", "r1")
        tracer.end_node("call_model", "r1")
        partial = tracer.get_trace("r1")
        assert [step.node for step in partial.steps] == ["call_model"]

    def test_node_and_run_errors_are_recorded(self, tracer):
        tracer.start_run("r1")
        tracer.start_node("tools", "r1")
        tracer.end_node("tools", "r1", error="ValueError: boom")
        tracer.end_run("r1", error="ValueError: boom")

        trace = tracer.get_trace("r1")
        assert trace.error == "ValueError: boom"
        assert trace.steps[0].error == "ValueError: boom"
        assert trace.failed

    def test_a_node_left_open_is_closed_by_end_run(self, tracer):
        # An exception mid-node must still produce a complete trace.
        tracer.start_run("r1")
        tracer.start_node("call_model", "r1")
        tracer.end_run("r1", error="boom")

        trace = tracer.get_trace("r1")
        assert [step.node for step in trace.steps] == ["call_model"]
        assert "before node completed" in trace.steps[0].error

    def test_ending_an_unknown_node_or_run_is_a_no_op(self, tracer):
        tracer.end_node("never-started", "r1")
        tracer.end_run("never-started")

    def test_latency_is_measured_and_positive(self, tracer):
        tracer.start_run("r1")
        tracer.start_node("call_model", "r1")
        tracer.end_node("call_model", "r1")
        tracer.end_run("r1")
        trace = tracer.get_trace("r1")
        assert trace.steps[0].latency_ms >= 0
        # The run wraps the node, so it cannot be the faster of the two.
        assert trace.latency_ms >= trace.steps[0].latency_ms


class TestUsageAttribution:
    def test_usage_lands_on_the_open_node_and_is_priced(self, tracer):
        tracer.start_run("r1")
        tracer.start_node("call_model", "r1")
        tracer.record_usage(TokenUsage(input_tokens=900, output_tokens=120), "claude-opus-5")
        tracer.end_node("call_model", "r1")
        tracer.end_run("r1")

        step = tracer.get_trace("r1").steps[0]
        assert step.tokens_out == 120
        assert step.model == "claude-opus-5"
        assert step.cost == pytest.approx(900 * 5 / 1e6 + 120 * 25 / 1e6)

    def test_two_calls_in_one_node_accumulate(self, tracer):
        tracer.start_run("r1")
        tracer.start_node("call_model", "r1")
        for _ in range(2):
            tracer.record_usage(TokenUsage(input_tokens=100, output_tokens=10), "claude-opus-5")
        tracer.end_node("call_model", "r1")
        tracer.end_run("r1")
        assert tracer.get_trace("r1").steps[0].tokens_in == 200

    def test_usage_outside_any_node_is_kept_not_dropped(self, tracer):
        # A cost you cannot attribute is still a cost you paid.
        tracer.start_run("r1")
        tracer.record_usage(TokenUsage(input_tokens=50, output_tokens=5), "claude-opus-5")
        tracer.record_usage(TokenUsage(input_tokens=50, output_tokens=5), "claude-opus-5")
        tracer.end_run("r1")

        trace = tracer.get_trace("r1")
        assert [step.node for step in trace.steps] == [UNGROUPED_NODE]
        assert trace.total_tokens_in == 100

    def test_usage_with_no_run_at_all_is_discarded_quietly(self, tracer):
        tracer.record_usage(TokenUsage(input_tokens=10), "claude-opus-5")
        assert tracer.get_trace() is None

    def test_concurrent_runs_do_not_mix_their_tokens(self):
        tracer = Tracer()
        barrier = threading.Barrier(2)

        def run(run_id, tokens):
            tracer.start_run(run_id)
            tracer.start_node("call_model", run_id)
            barrier.wait(timeout=5)  # force genuine interleaving
            tracer.record_usage(TokenUsage(input_tokens=tokens), "claude-opus-5")
            tracer.end_node("call_model", run_id)
            tracer.end_run(run_id)

        threads = [
            threading.Thread(target=run, args=("r1", 100)),
            threading.Thread(target=run, args=("r2", 900)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert tracer.get_trace("r1").total_tokens_in == 100
        assert tracer.get_trace("r2").total_tokens_in == 900


class TestCallbackIntegration:
    def test_usage_callback_feeds_the_open_node(self, tracer):
        handler = tracer.callbacks()[0]
        tracer.start_run("r1")
        tracer.start_node("call_model", "r1")
        handler.on_llm_end(_llm_result(cache_read=400), run_id=None)
        tracer.end_node("call_model", "r1")
        tracer.end_run("r1")

        step = tracer.get_trace("r1").steps[0]
        assert step.usage.cache_read_tokens == 400
        assert step.cost == pytest.approx((500 + 400 * 0.1) * 5 / 1e6 + 120 * 25 / 1e6)

    def test_callbacks_are_reused_not_rebuilt(self, tracer):
        assert tracer.callbacks()[0] is tracer.callbacks()[0]

    def test_a_broken_response_never_reaches_the_agent(self, tracer):
        class Exploding:
            @property
            def generations(self):
                raise RuntimeError("provider went sideways")

            llm_output = None

        handler = UsageCallback(tracer)
        with pytest.warns(RuntimeWarning, match="unaffected"):
            handler.on_llm_end(Exploding(), run_id=None)

    def test_handlers_do_not_let_langchain_propagate_their_errors(self, tracer):
        assert UsageCallback(tracer).raise_error is False

    def test_legacy_token_usage_shape_is_still_read(self):
        result = LLMResult(
            generations=[[]],
            llm_output={
                "model_name": "claude-haiku-4-5",
                "token_usage": {"prompt_tokens": 30, "completion_tokens": 7},
            },
        )
        usage, model = usage_from_response(result)
        assert (usage.input_tokens, usage.output_tokens, model) == (30, 7, "claude-haiku-4-5")

    def test_a_response_with_no_usage_yields_an_empty_usage(self):
        usage, model = usage_from_response(LLMResult(generations=[[]]))
        assert usage.is_empty and model is None


class TestRetentionAndStorage:
    def test_completed_traces_are_bounded(self):
        tracer = Tracer(max_runs=3)
        for index in range(10):
            tracer.start_run(f"r{index}")
            tracer.end_run(f"r{index}")

        assert len(tracer.traces()) == 3
        assert [trace.run_id for trace in tracer.traces()] == ["r7", "r8", "r9"]
        assert tracer.get_trace("r0") is None  # evicted, not resurrected

    def test_max_runs_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            Tracer(max_runs=0)

    def test_a_store_receives_every_completed_trace(self, tmp_path):
        store = JSONLTraceStore(tmp_path / "traces.jsonl")
        tracer = Tracer(store=store, max_runs=1)
        for index in range(3):
            tracer.start_run(f"r{index}")
            tracer.end_run(f"r{index}")

        # Memory is bounded; the file is not.
        assert len(tracer.traces()) == 1
        assert [trace.run_id for trace in store.read()] == ["r0", "r1", "r2"]

    def test_reset_clears_memory_but_not_the_store(self, tmp_path):
        store = JSONLTraceStore(tmp_path / "traces.jsonl")
        tracer = Tracer(store=store)
        tracer.start_run("r1")
        tracer.end_run("r1")
        tracer.reset()

        assert tracer.traces() == []
        assert tracer.get_trace() is None
        assert len(store.read()) == 1

    def test_repr_summarises_without_dumping_state(self, tracer):
        tracer.start_run("r1")
        tracer.end_run("r1")
        assert repr(tracer) == "Tracer(completed=1, in_flight=0, max_runs=100, store=False)"


class TestStore:
    def test_missing_file_reads_as_empty(self, tmp_path):
        assert JSONLTraceStore(tmp_path / "nope.jsonl").read() == []

    def test_read_one_finds_a_single_run(self, tmp_path, sample_trace):
        store = JSONLTraceStore(tmp_path / "t.jsonl")
        store.append(sample_trace)
        assert store.read_one("run-1").total_tokens_out == 120
        assert store.read_one("absent") is None
        assert len(store) == 1

    def test_parent_directories_are_created(self, tmp_path, sample_trace):
        store = JSONLTraceStore(tmp_path / "deep" / "nested" / "t.jsonl")
        store.append(sample_trace)
        assert store.path.exists()

    def test_blank_lines_are_skipped(self, tmp_path, sample_trace):
        store = JSONLTraceStore(tmp_path / "t.jsonl")
        store.append(sample_trace)
        store.path.write_text(store.path.read_text() + "\n\n", encoding="utf-8")
        assert len(store.read()) == 1

    def test_a_corrupt_line_names_its_line_number(self, tmp_path, sample_trace):
        store = JSONLTraceStore(tmp_path / "t.jsonl")
        store.append(sample_trace)
        with store.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        with pytest.raises(ValueError, match=r"t\.jsonl:2 is not valid JSON"):
            store.read()

    def test_repr_names_the_file(self, tmp_path):
        assert "t.jsonl" in repr(JSONLTraceStore(tmp_path / "t.jsonl"))
