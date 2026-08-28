"""Tests for the read-only dashboard API and the topology reader.

Every telemetry object here is a local fake. Importing ``wardhook.observability``
would make this suite fail in the standalone-install matrix, which is the check
that keeps the four packages independent -- and the dashboard's whole claim is
that it reads a sink structurally rather than depending on one.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from wardhook.core.agent import AgentGraph
from wardhook.core.rag.retriever import Retriever
from wardhook.core.rag.store import InMemoryVectorStore
from wardhook.core.serve.dashboard import create_dashboard
from wardhook.core.serve.topology import (
    Topology,
    TopologyEdge,
    TopologyNode,
    describe_agent,
    layout,
    read_topology,
    render_svg,
)


class Allower:
    name = "allower"

    def on_input(self, text, context):
        return SimpleNamespace(action="allow", text=text, reason=None, rule=None)


def usage(inp=0, out=0, cached=0):
    """A token-usage lookalike, matching observability's field names."""
    return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_tokens=cached)


def step(node, **overrides):
    """A TraceStep lookalike with sensible defaults."""
    fields = {
        "node": node,
        "run_id": "run-1",
        "started_at": "2026-08-27T10:00:00+00:00",
        "latency_ms": 12.0,
        "usage": usage(),
        "cost": 0.0,
        "model": None,
        "error": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def trace(run_id="run-1", steps=(), **overrides):
    """A Trace lookalike with sensible defaults."""
    fields = {
        "run_id": run_id,
        "steps": tuple(steps),
        "started_at": "2026-08-27T10:00:00+00:00",
        "latency_ms": 40.0,
        "metadata": {"agent": "demo", "model": "claude-opus-5"},
        "error": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class MemorySink:
    """A Tracer lookalike: lists with traces(), looks up with get_trace()."""

    def __init__(self, *traces):
        self._traces = list(traces)

    def traces(self):
        return list(self._traces)

    def get_trace(self, run_id=None):
        return next((t for t in self._traces if t.run_id == run_id), None)


class StoreSink:
    """A JSONLTraceStore lookalike: lists with read(), looks up with read_one()."""

    def __init__(self, *traces):
        self.path = "traces.jsonl"
        self._traces = list(traces)

    def read(self):
        return list(self._traces)

    def read_one(self, run_id):
        return next((t for t in self._traces if t.run_id == run_id), None)


class DeafSink:
    """A sink exposing neither read shape. Reported honestly, never crashed on."""


class Minimal:
    """A bare callable target: legitimate to serve, but it has no graph."""

    def invoke(self, text, **kwargs):
        return {"output": f"echo: {text}"}


@pytest.fixture
def full_agent(make_tool_model, echo_tool, sample_store):
    return AgentGraph(
        model=make_tool_model(AIMessage(content="ok")),
        tools=[echo_tool],
        guardrails=[Allower()],
        retriever=Retriever(sample_store),
        name="full",
    )


@pytest.fixture
def bare_agent(make_model):
    return AgentGraph(model=make_model("ok"), name="bare")


class TestReadTopology:
    def test_a_configured_agent_reports_every_node_it_built(self, full_agent):
        topology = read_topology(full_agent)
        assert topology.available is True
        assert set(topology.keys) == {
            "__start__",
            "guard_input",
            "retrieve",
            "call_model",
            "tools",
            "guard_output",
            "__end__",
        }

    def test_an_unconfigured_agent_genuinely_has_fewer_nodes(self, bare_agent, full_agent):
        # The whole value of a topology view is that it is configuration-accurate:
        # an agent with no retriever has no retrieve box because it has no
        # retrieve node, not because a template hid it.
        bare = read_topology(bare_agent)
        assert set(bare.keys) == {"__start__", "call_model", "__end__"}
        assert len(bare.nodes) < len(read_topology(full_agent).nodes)

    def test_conditional_edges_keep_their_branch_labels(self, full_agent):
        conditional = {
            (edge.source, edge.target, edge.label)
            for edge in read_topology(full_agent).edges
            if edge.conditional
        }
        assert ("guard_input", "__end__", "blocked") in conditional
        assert ("guard_input", "retrieve", "continue") in conditional
        assert ("call_model", "guard_output", "finish") in conditional
        # LangGraph drops a branch label that merely repeats its target node's
        # name, so the tools branch is conditional but unlabelled. Asserted
        # rather than worked around: the topology reports what the graph says.
        assert ("call_model", "tools", None) in conditional

    def test_unconditional_edges_carry_no_label(self, bare_agent):
        edges = {(e.source, e.target): e for e in read_topology(bare_agent).edges}
        assert edges[("__start__", "call_model")].label is None
        assert edges[("__start__", "call_model")].conditional is False

    def test_mermaid_source_is_carried_for_export(self, bare_agent):
        assert "call_model" in (read_topology(bare_agent).mermaid or "")

    def test_an_object_with_no_graph_is_reported_not_raised(self):
        topology = read_topology(Minimal())
        assert topology.available is False
        assert topology.nodes == ()
        assert "no .graph attribute" in (topology.reason or "")

    def test_a_graph_without_get_graph_is_reported(self):
        topology = read_topology(SimpleNamespace(graph=SimpleNamespace()))
        assert topology.available is False
        assert "no get_graph()" in (topology.reason or "")

    def test_a_raising_get_graph_becomes_a_reason_not_a_crash(self):
        def boom():
            raise RuntimeError("graph is corrupt")

        topology = read_topology(SimpleNamespace(graph=SimpleNamespace(get_graph=boom)))
        assert topology.available is False
        assert "RuntimeError: graph is corrupt" in (topology.reason or "")

    def test_a_graph_that_cannot_draw_still_yields_its_nodes(self):
        # Losing the export diagram must not cost the caller the topology.
        graph = SimpleNamespace(nodes={"a": SimpleNamespace(name="a")}, edges=[])
        topology = read_topology(SimpleNamespace(graph=SimpleNamespace(get_graph=lambda: graph)))
        assert topology.keys == ("a",)
        assert topology.mermaid is None

    def test_a_raising_draw_mermaid_still_yields_its_nodes(self):
        def boom():
            raise RuntimeError("renderer failed")

        graph = SimpleNamespace(nodes={"a": SimpleNamespace(name="a")}, edges=[], draw_mermaid=boom)
        topology = read_topology(SimpleNamespace(graph=SimpleNamespace(get_graph=lambda: graph)))
        assert topology.keys == ("a",)
        assert topology.mermaid is None

    def test_an_empty_graph_is_not_an_error(self):
        graph = SimpleNamespace(nodes=None, edges=None)
        topology = read_topology(SimpleNamespace(graph=SimpleNamespace(get_graph=lambda: graph)))
        assert topology.available is True
        assert topology.to_dict()["nodes"] == []

    def test_a_node_without_a_name_falls_back_to_its_key(self):
        graph = SimpleNamespace(nodes={"a": object()}, edges=[])
        topology = read_topology(SimpleNamespace(graph=SimpleNamespace(get_graph=lambda: graph)))
        assert topology.nodes[0].name == "a"

    def test_an_edge_without_endpoints_does_not_crash_the_read(self):
        graph = SimpleNamespace(nodes={}, edges=[object()])
        topology = read_topology(SimpleNamespace(graph=SimpleNamespace(get_graph=lambda: graph)))
        assert topology.edges[0].to_dict() == {
            "source": "",
            "target": "",
            "label": None,
            "conditional": False,
        }


class TestTopologyEndpoint:
    def test_it_reports_nodes_edges_and_configuration(self, full_agent):
        body = TestClient(create_dashboard(full_agent)).get("/api/topology").json()
        assert body["agent"] == "full"
        assert body["available"] is True
        assert {node["id"] for node in body["nodes"]} >= {"retrieve", "tools"}
        assert body["config"]["retrieval_enabled"] is True
        assert body["mermaid"].strip() != ""

    def test_an_agent_without_a_graph_gets_a_reason_not_a_500(self):
        body = TestClient(create_dashboard(Minimal())).get("/api/topology").json()
        assert body["available"] is False
        assert body["nodes"] == []
        assert "no .graph attribute" in body["reason"]


class TestRunsEndpoint:
    def test_no_telemetry_is_an_empty_list_not_an_error(self, bare_agent):
        # An agent constructed without telemetry is the common case, and it must
        # not turn the dashboard into an error page.
        body = TestClient(create_dashboard(bare_agent)).get("/api/runs").json()
        assert body["mode"] == "none"
        assert body["runs"] == []
        assert "No readable telemetry" in body["mode_note"]

    def test_a_sink_exposing_neither_read_shape_is_reported_honestly(self, bare_agent):
        client = TestClient(create_dashboard(bare_agent, telemetry=DeafSink()))
        assert client.get("/api/runs").json()["mode"] == "none"

    def test_an_in_memory_tracer_is_labelled_as_one_process_only(self, bare_agent):
        sink = MemorySink(trace("run-1"), trace("run-2"))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs").json()
        assert body["mode"] == "memory"
        assert "1/N of traffic" in body["mode_note"]

    def test_a_shared_store_is_labelled_as_complete(self, bare_agent):
        sink = StoreSink(trace("run-1"))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs").json()
        assert body["mode"] == "store"
        assert "every run" in body["mode_note"].lower()

    def test_runs_are_returned_newest_first(self, bare_agent):
        sink = MemorySink(trace("oldest"), trace("middle"), trace("newest"))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs").json()
        assert [run["run_id"] for run in body["runs"]] == ["newest", "middle", "oldest"]

    def test_the_limit_caps_the_page_but_the_total_stays_honest(self, bare_agent):
        sink = MemorySink(*[trace(f"run-{i}") for i in range(5)])
        body = (
            TestClient(create_dashboard(bare_agent, telemetry=sink))
            .get("/api/runs", params={"limit": 2})
            .json()
        )
        assert body["total"] == 5
        assert body["returned"] == 2
        assert len(body["runs"]) == 2

    def test_an_out_of_range_limit_is_rejected_by_validation(self, bare_agent):
        client = TestClient(create_dashboard(bare_agent, telemetry=MemorySink()))
        assert client.get("/api/runs", params={"limit": 0}).status_code == 422
        assert client.get("/api/runs", params={"limit": 100000}).status_code == 422

    def test_a_summary_carries_totals_but_never_the_steps(self, bare_agent):
        sink = MemorySink(
            trace("run-1", steps=[step("call_model", usage=usage(900, 120, 400), cost=0.0057)])
        )
        summary = (
            TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs").json()["runs"]
        )[0]
        assert "steps" not in summary
        assert summary["totals"] == {
            "steps": 1,
            "tokens_in": 900,
            "tokens_out": 120,
            "cached_tokens": 400,
            "cost": 0.0057,
        }

    def test_the_agents_own_tracer_is_used_when_none_is_passed(self, make_model):
        sink = MemorySink(trace("run-1"))
        agent = AgentGraph(model=make_model("ok"), telemetry=sink, name="wired")
        body = TestClient(create_dashboard(agent)).get("/api/runs").json()
        assert [run["run_id"] for run in body["runs"]] == ["run-1"]

    def test_an_explicit_sink_overrides_the_agents_own(self, make_model):
        # This is the documented multi-worker mitigation: the agent keeps writing
        # through its in-memory tracer while the dashboard reads the shared file.
        agent = AgentGraph(model=make_model("ok"), telemetry=MemorySink(trace("in-memory")))
        client = TestClient(create_dashboard(agent, telemetry=StoreSink(trace("on-disk"))))
        body = client.get("/api/runs").json()
        assert body["mode"] == "store"
        assert [run["run_id"] for run in body["runs"]] == ["on-disk"]


class TestOneRunEndpoint:
    def test_it_returns_per_node_timing_tokens_and_cost(self, bare_agent):
        sink = MemorySink(
            trace(
                "run-1",
                steps=[
                    step("guard_input", latency_ms=1.5),
                    step(
                        "call_model",
                        latency_ms=412.5,
                        usage=usage(900, 120, 400),
                        cost=0.0057,
                        model="claude-opus-5",
                    ),
                ],
            )
        )
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs/run-1")
        assert body.status_code == 200
        steps = body.json()["steps"]
        assert [s["node"] for s in steps] == ["guard_input", "call_model"]
        assert steps[1] == {
            "node": "call_model",
            "run_id": "run-1",
            "started_at": "2026-08-27T10:00:00+00:00",
            "latency_ms": 412.5,
            "cost": 0.0057,
            "model": "claude-opus-5",
            "error": None,
            "tokens_in": 900,
            "tokens_out": 120,
            "cached_tokens": 400,
        }

    def test_an_unknown_run_is_a_404_with_an_actionable_reason(self, bare_agent):
        client = TestClient(create_dashboard(bare_agent, telemetry=MemorySink()))
        response = client.get("/api/runs/nope")
        assert response.status_code == 404
        assert "evicted" in response.json()["detail"]

    def test_a_store_shaped_sink_is_looked_up_through_read_one(self, bare_agent):
        sink = StoreSink(trace("run-7", steps=[step("call_model")]))
        client = TestClient(create_dashboard(bare_agent, telemetry=sink))
        assert client.get("/api/runs/run-7").json()["run_id"] == "run-7"
        assert client.get("/api/runs/absent").status_code == 404

    def test_a_sink_that_cannot_look_runs_up_is_a_404_not_a_crash(self, bare_agent):
        client = TestClient(create_dashboard(bare_agent, telemetry=DeafSink()))
        assert client.get("/api/runs/anything").status_code == 404

    def test_a_failed_node_marks_the_whole_run_failed(self, bare_agent):
        sink = MemorySink(trace("run-1", steps=[step("tools", error="ValueError: bad args")]))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs/run-1")
        assert body.json()["failed"] is True
        assert body.json()["steps"][0]["error"] == "ValueError: bad args"

    def test_a_failed_run_with_healthy_steps_is_still_failed(self, bare_agent):
        sink = MemorySink(trace("run-1", steps=[step("call_model")], error="TimeoutError"))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs/run-1")
        assert body.json()["failed"] is True

    def test_a_healthy_run_is_not_failed(self, bare_agent):
        sink = MemorySink(trace("run-1", steps=[step("call_model")]))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs/run-1")
        assert body.json()["failed"] is False

    def test_a_node_that_called_no_model_reports_zero_tokens(self, bare_agent):
        # Guardrail and retrieval nodes have no usage at all; that is not an error.
        sink = MemorySink(trace("run-1", steps=[step("guard_input", usage=None)]))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs/run-1")
        assert body.json()["steps"][0]["tokens_in"] == 0

    def test_a_trace_with_no_steps_totals_to_zero(self, bare_agent):
        # steps and metadata are absent rather than empty, which is what a sink
        # from outside Wardhook may well hand over.
        sink = MemorySink(SimpleNamespace(run_id="run-1", steps=None, metadata=None))
        body = (
            TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs/run-1").json()
        )
        assert body["totals"] == {
            "steps": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cached_tokens": 0,
            "cost": 0,
        }
        assert body["metadata"] == {}


class TestContentNeverLeaks:
    def test_a_content_field_added_upstream_does_not_reach_the_api(self, bare_agent):
        # The load-bearing test for this whole feature. The projection in
        # dashboard.py is an allowlist, so a prompt field appearing on an
        # upstream step type cannot start being served by accident. If this test
        # ever fails, the dashboard has become a PII store.
        leaky_step = step("call_model")
        leaky_step.prompt = "My SSN is 123-45-6789"
        leaky_step.completion = "Your SSN 123-45-6789 is on file"
        leaky_trace = trace("run-1", steps=[leaky_step])
        leaky_trace.transcript = "the entire conversation"

        client = TestClient(create_dashboard(bare_agent, telemetry=MemorySink(leaky_trace)))
        for path in ("/api/runs", "/api/runs/run-1"):
            body = client.get(path).text
            assert "123-45-6789" not in body
            assert "prompt" not in body
            assert "completion" not in body
            assert "transcript" not in body

    def test_the_step_projection_is_a_closed_set_of_keys(self, bare_agent):
        # Pinned explicitly so that widening it is a deliberate, reviewed edit
        # rather than a side effect of a change somewhere upstream.
        sink = MemorySink(trace("run-1", steps=[step("call_model")]))
        body = TestClient(create_dashboard(bare_agent, telemetry=sink)).get("/api/runs/run-1")
        assert set(body.json()["steps"][0]) == {
            "node",
            "run_id",
            "started_at",
            "latency_ms",
            "cost",
            "model",
            "error",
            "tokens_in",
            "tokens_out",
            "cached_tokens",
        }

    def test_guardrail_events_are_not_exposed_anywhere(self, make_model):
        # Core returns guardrail_events from invoke(); the dashboard must not
        # surface them. They name the rule that fired, and on some guardrails
        # that is enough to infer the redacted value.
        agent = AgentGraph(model=make_model("ok"), guardrails=[Allower()], name="guarded")
        client = TestClient(create_dashboard(agent))
        payload = json.dumps(
            [client.get(p).json() for p in ("/api/topology", "/api/runs")],
        )
        assert "guardrail_events" not in payload


class TestRunIdIsTheJoinKey:
    def test_every_step_carries_the_run_id_it_belongs_to(self, bare_agent):
        # run_id is the documented way back to the caller's own audit log, which
        # is why the dashboard does not need a copy of the content.
        sink = MemorySink(trace("run-42", steps=[step("call_model", run_id="run-42")]))
        client = TestClient(create_dashboard(bare_agent, telemetry=sink))
        body = client.get("/api/runs/run-42").json()
        assert body["run_id"] == "run-42"
        assert {s["run_id"] for s in body["steps"]} == {"run-42"}


class TestAgainstTheRealAgent:
    def test_a_real_run_produces_a_trace_whose_nodes_are_in_the_topology(self, make_model):
        # An end-to-end check with a hand-rolled sink standing in for a tracer:
        # the join the trace overlay depends on has to hold for real node names,
        # not just for the ones the fixtures invent.
        class RecordingSink:
            """Enough of TelemetryProtocol to be driven by a real invocation."""

            def __init__(self):
                self.open_steps = []
                self.finished = []

            def start_run(self, run_id, metadata=None):
                self.open_steps = []

            def end_run(self, run_id, error=None):
                self.finished.append(trace(run_id, steps=list(self.open_steps)))

            def start_node(self, node, run_id):
                pass

            def end_node(self, node, run_id, error=None):
                self.open_steps.append(step(node, run_id=run_id))

            def callbacks(self):
                return []

            def traces(self):
                return list(self.finished)

            def get_trace(self, run_id=None):
                return next((t for t in self.finished if t.run_id == run_id), None)

        sink = RecordingSink()
        agent = AgentGraph(
            model=make_model("ok"), guardrails=[Allower()], telemetry=sink, name="real"
        )
        run_id = agent.invoke("hello")["run_id"]

        client = TestClient(create_dashboard(agent))
        topology_keys = {n["id"] for n in client.get("/api/topology").json()["nodes"]}
        run = client.get(f"/api/runs/{run_id}").json()

        assert run["steps"], "the run recorded no steps"
        assert {s["node"] for s in run["steps"]} <= topology_keys


class TestPageSelfContainment:
    def test_no_external_resource_is_referenced(self, full_agent):
        # The same assertion wardhook-observability makes about its static
        # viewer, mirrored here rather than shared. It cannot be shared: that
        # file lives in a sibling package and core must pass with no sibling
        # installed. So the promise is duplicated, because the promise matters
        # more than the duplication -- this page has to work in an air-gapped
        # network, which is exactly where a governance tool earns its place.
        page = TestClient(create_dashboard(full_agent)).get("/").text
        external = re.findall(r'(?:src|href)\s*=\s*["\'](?!#)([^"\']+)', page)
        assert external == [], f"page references external resources: {external}"
        for scheme in ("http://", "https://", "//cdn", "@import"):
            assert scheme not in page

    def test_every_script_on_the_page_is_inline(self, full_agent):
        # Stated as an invariant rather than a count, so it keeps holding as the
        # page grows: a script may exist, but it may never be fetched.
        page = TestClient(create_dashboard(full_agent)).get("/").text
        for tag in re.findall(r"<script[^>]*>", page):
            assert "src" not in tag, f"script loads an external file: {tag}"

    def test_the_page_is_served_as_html(self, bare_agent):
        response = TestClient(create_dashboard(bare_agent)).get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.text.startswith("<!doctype html>")

    def test_the_page_is_not_in_the_openapi_schema(self, bare_agent):
        # It is a page, not an API. Listing it as an endpoint would suggest a
        # machine-readable contract that does not exist.
        client = TestClient(create_dashboard(bare_agent))
        assert "/" not in client.get("/openapi.json").json()["paths"]


class TestPageContent:
    def test_it_names_the_agent_and_its_configuration(self, full_agent):
        page = TestClient(create_dashboard(full_agent)).get("/").text
        assert "<h1>full</h1>" in page
        assert "lookup_account" in page
        assert "allower" in page

    def test_an_agent_with_nothing_attached_says_so(self, bare_agent):
        page = TestClient(create_dashboard(bare_agent)).get("/").text
        assert page.count("none attached") == 2
        assert "disabled" in page

    def test_the_mode_banner_states_the_multi_worker_limitation(self, bare_agent):
        # An observability tool that quietly loses three quarters of the data is
        # worse than one that refuses to pretend otherwise.
        page = TestClient(create_dashboard(bare_agent, telemetry=MemorySink())).get("/").text
        assert 'class="banner memory"' in page
        assert "1/N of traffic" in page

    def test_a_shared_store_gets_the_reassuring_banner(self, bare_agent):
        page = TestClient(create_dashboard(bare_agent, telemetry=StoreSink())).get("/").text
        assert 'class="banner store"' in page

    def test_no_telemetry_gets_the_neutral_banner(self, bare_agent):
        page = TestClient(create_dashboard(bare_agent)).get("/").text
        assert 'class="banner none"' in page

    def test_the_footer_states_what_the_page_will_never_show(self, bare_agent):
        page = TestClient(create_dashboard(bare_agent)).get("/").text
        assert "never prompts, model output, retrieved context" in page
        assert "makes no network requests" in page


class TestPageEscaping:
    def test_a_guardrail_named_like_markup_survives_inert(self, make_model):
        # Guardrail and tool names come from user code. If a rendered name is
        # ever served over HTTP unescaped, that is stored cross-site scripting.
        class Nasty:
            name = "<script>alert(1)</script>"

            def on_input(self, text, context):
                return None

        agent = AgentGraph(model=make_model("ok"), guardrails=[Nasty()], name="x")
        page = TestClient(create_dashboard(agent)).get("/").text
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page

    def test_an_agent_named_like_markup_survives_inert(self, make_model):
        agent = AgentGraph(model=make_model("ok"), name='"><img onerror=alert(1)>')
        page = TestClient(create_dashboard(agent)).get("/").text
        assert "<img onerror" not in page
        assert "&lt;img onerror=alert(1)&gt;" in page


class TestLayout:
    def test_ranks_follow_the_longest_path(self, full_agent):
        placed = layout(read_topology(full_agent))
        rank = {box.key: box.rank for box in placed.boxes}
        assert rank["__start__"] == 0
        assert rank["guard_input"] < rank["retrieve"] < rank["call_model"]
        assert rank["call_model"] < rank["tools"]
        assert rank["__end__"] == max(rank.values())

    def test_the_tool_loop_is_recognised_as_a_back_edge(self, full_agent):
        # Without this the ranking would not terminate on a sensible answer:
        # tools would have to sit both above and below call_model.
        topology = read_topology(full_agent)
        placed = layout(topology)
        looped = {(topology.edges[i].source, topology.edges[i].target) for i in placed.back_edges}
        assert looped == {("tools", "call_model")}

    def test_nodes_sharing_a_rank_are_placed_side_by_side(self, full_agent):
        placed = layout(read_topology(full_agent))
        by_rank = {}
        for box in placed.boxes:
            by_rank.setdefault(box.rank, []).append(box)
        siblings = next(row for row in by_rank.values() if len(row) > 1)
        assert len({box.x for box in siblings}) == len(siblings)
        assert len({box.y for box in siblings}) == 1

    def test_an_empty_topology_lays_out_to_nothing(self):
        placed = layout(Topology(available=False, reason="no graph"))
        assert placed.boxes == ()
        assert (placed.width, placed.height) == (0.0, 0.0)

    def test_a_node_unreachable_from_the_start_is_still_placed(self):
        # A hand-built graph can have one. Dropping it silently would make the
        # picture a lie; this is the only kind of error a diagram can tell.
        nodes = (TopologyNode("a", "a"), TopologyNode("b", "b"), TopologyNode("orphan", "orphan"))
        placed = layout(Topology(nodes=nodes, edges=(TopologyEdge("a", "b"),)))
        assert {box.key for box in placed.boxes} == {"a", "b", "orphan"}

    def test_an_edge_naming_a_node_that_does_not_exist_is_ignored(self):
        nodes = (TopologyNode("a", "a"),)
        placed = layout(Topology(nodes=nodes, edges=(TopologyEdge("a", "ghost"),)))
        assert placed.box("a") is not None
        assert placed.box("ghost") is None

    def test_an_edge_leaving_a_node_that_does_not_exist_is_ignored(self):
        # The mirror of the case above. Both halves of an edge are checked,
        # because a hand-built graph can dangle at either end.
        nodes = (TopologyNode("a", "a"),)
        placed = layout(Topology(nodes=nodes, edges=(TopologyEdge("ghost", "a"),)))
        assert {box.key for box in placed.boxes} == {"a"}
        assert placed.back_edges == frozenset()

    def test_the_same_configuration_always_lays_out_identically(self, full_agent):
        # A diagram that moves between refreshes is one a reader cannot learn.
        first, second = layout(read_topology(full_agent)), layout(read_topology(full_agent))
        assert first == second


class TestRenderSvg:
    def test_every_node_is_addressable_by_name(self, full_agent):
        svg = render_svg(read_topology(full_agent))
        for node in read_topology(full_agent).nodes:
            assert f'data-node="{node.key}"' in svg
            assert f'data-metric-for="{node.key}"' in svg

    def test_a_conditional_edge_is_drawn_differently(self, full_agent):
        svg = render_svg(read_topology(full_agent))
        assert 'class="edge conditional"' in svg
        assert 'class="edge"' in svg
        assert ">blocked<" in svg

    def test_the_terminals_are_drawn_differently_from_the_work(self, full_agent):
        svg = render_svg(read_topology(full_agent))
        assert 'class="node terminal" data-node="__start__"' in svg
        assert 'class="node" data-node="call_model"' in svg

    def test_nothing_is_drawn_for_an_agent_with_no_graph(self):
        # No invented picture. The page says why instead.
        assert render_svg(read_topology(Minimal())) == ""

    def test_an_edge_to_a_missing_node_is_skipped_not_crashed_on(self):
        nodes = (TopologyNode("a", "a"),)
        svg = render_svg(Topology(nodes=nodes, edges=(TopologyEdge("a", "ghost"),)))
        assert "ghost" not in svg
        assert 'data-node="a"' in svg

    def test_a_back_edge_into_the_first_rank_stays_on_the_canvas(self):
        # The clamp that keeps a detour above the top rank from being drawn off
        # the top of the viewBox.
        nodes = (TopologyNode("a", "a"), TopologyNode("b", "b"))
        edges = (TopologyEdge("a", "b"), TopologyEdge("b", "a"))
        svg = render_svg(Topology(nodes=nodes, edges=edges))
        coordinates = [float(v) for v in re.findall(r"[ML](-?[\d.]+),(-?[\d.]+)", svg) for v in v]
        assert min(coordinates) >= 0, "a path leaves the canvas"

    def test_a_node_named_like_markup_is_inert_everywhere_it_appears(self):
        # A node name reaches four places: the aria-label, the data-node
        # attribute, the visible text, and the metric hook. Every one is
        # escaped, and the count is asserted so adding a fifth without escaping
        # it fails here rather than in someone's browser.
        nasty = "<script>x</script>"
        svg = render_svg(Topology(nodes=(TopologyNode(nasty, nasty),)))
        assert "<script>" not in svg
        assert svg.count("&lt;script&gt;x&lt;/script&gt;") == 4

    def test_an_edge_label_like_markup_is_inert(self):
        nodes = (TopologyNode("a", "a"), TopologyNode("b", "b"))
        edges = (TopologyEdge("a", "b", label="<b>x</b>", conditional=True),)
        svg = render_svg(Topology(nodes=nodes, edges=edges))
        assert "<b>x</b>" not in svg
        assert "&lt;b&gt;x&lt;/b&gt;" in svg


class TestTopologyOnThePage:
    def test_an_unconfigured_agent_draws_strictly_fewer_boxes(self, bare_agent, full_agent):
        # The claim this whole view exists to make: the picture is accurate to
        # the configuration, not to a template that hides unused parts.
        client_full = TestClient(create_dashboard(full_agent))
        client_bare = TestClient(create_dashboard(bare_agent))

        full_nodes = [n["id"] for n in client_full.get("/api/topology").json()["nodes"]]
        bare_nodes = [n["id"] for n in client_bare.get("/api/topology").json()["nodes"]]
        assert set(bare_nodes) < set(full_nodes)
        assert "retrieve" not in bare_nodes
        assert "tools" not in bare_nodes

        drawn_full = re.findall(r'data-node="([^"]+)"', client_full.get("/").text)
        drawn_bare = re.findall(r'data-node="([^"]+)"', client_bare.get("/").text)
        assert sorted(drawn_full) == sorted(full_nodes)
        assert sorted(drawn_bare) == sorted(bare_nodes)
        assert len(drawn_bare) < len(drawn_full)

    def test_the_page_says_how_many_nodes_and_edges_it_drew(self, bare_agent):
        assert "3 nodes, 2 edges" in TestClient(create_dashboard(bare_agent)).get("/").text

    def test_an_agent_with_no_graph_gets_the_reason_instead_of_a_diagram(self):
        page = TestClient(create_dashboard(Minimal())).get("/").text
        assert 'class="topology"' not in page
        assert "data-node=" not in page
        assert "no .graph attribute" in page

    def test_the_diagram_adds_no_external_resource(self, full_agent):
        # Re-asserted with the SVG present, because this is the step where a
        # vendored renderer would have been pulled in from a CDN.
        page = TestClient(create_dashboard(full_agent)).get("/").text
        assert "<svg" in page
        external = re.findall(r'(?:src|href)\s*=\s*["\'](?!#)([^"\']+)', page)
        assert external == []
        for scheme in ("http://", "https://", "//cdn", "@import"):
            assert scheme not in page


class TestTraceOverlay:
    def test_the_run_picker_appears_only_when_there_is_telemetry_to_pick_from(self, bare_agent):
        with_sink = TestClient(create_dashboard(bare_agent, telemetry=MemorySink())).get("/").text
        without = TestClient(create_dashboard(bare_agent)).get("/").text
        assert 'id="wh-run"' in with_sink
        assert 'id="wh-run"' not in without

    def test_the_script_ships_only_when_it_has_something_to_drive(self, bare_agent):
        with_sink = TestClient(create_dashboard(bare_agent, telemetry=MemorySink())).get("/").text
        without = TestClient(create_dashboard(bare_agent)).get("/").text
        assert with_sink.count("<script>") == 1
        assert "<script" not in without

    def test_the_run_id_field_is_present_and_copyable(self, bare_agent):
        # run_id is the documented way back to the caller's own audit log. It is
        # the reason the dashboard does not need a copy of the content, so it
        # has to be visible and easy to lift off the page.
        page = TestClient(create_dashboard(bare_agent, telemetry=MemorySink())).get("/").text
        assert 'id="wh-runid" readonly' in page
        assert 'id="wh-copy"' in page
        assert "run_id" in page

    def test_the_overlay_script_never_assigns_markup(self, bare_agent):
        # The invariant that makes it safe for the page to render values the API
        # returned: the script sets attributes and textContent, and builds nodes
        # with createElement. Nothing it touches can interpret a string as HTML.
        page = TestClient(create_dashboard(bare_agent, telemetry=MemorySink())).get("/").text
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            assert forbidden not in page

    def test_the_script_adds_no_external_resource(self, full_agent):
        sink = MemorySink(trace("run-1", steps=[step("call_model")]))
        page = TestClient(create_dashboard(full_agent, telemetry=sink)).get("/").text
        assert "<script>" in page
        external = re.findall(r'(?:src|href)\s*=\s*["\'](?!#)([^"\']+)', page)
        assert external == []
        for scheme in ("http://", "https://", "//cdn", "@import"):
            assert scheme not in page

    def test_every_node_the_overlay_can_paint_is_on_the_diagram(self, full_agent):
        # The join the overlay depends on, asserted on the server side where the
        # coverage gate can see it: a node key in the topology is exactly what a
        # step's node name has to match.
        client = TestClient(create_dashboard(full_agent))
        drawn = set(re.findall(r'data-metric-for="([^"]+)"', client.get("/").text))
        assert drawn == {node["id"] for node in client.get("/api/topology").json()["nodes"]}

    def test_a_step_carries_the_metrics_of_its_node(self, full_agent):
        # The payoff, stated as data: this node, in this run, cost this much.
        sink = MemorySink(
            trace(
                "run-1",
                steps=[
                    step("guard_input", latency_ms=4.0),
                    step(
                        "call_model",
                        latency_ms=412.5,
                        usage=usage(900, 120, 400),
                        cost=0.0057,
                        model="claude-opus-5",
                    ),
                ],
            )
        )
        client = TestClient(create_dashboard(full_agent, telemetry=sink))
        run = client.get("/api/runs/run-1").json()
        drawn = set(re.findall(r'data-metric-for="([^"]+)"', client.get("/").text))

        by_node = {s["node"]: s for s in run["steps"]}
        assert set(by_node) <= drawn, "a recorded node has no box to paint"
        assert by_node["call_model"]["cost"] == 0.0057
        assert by_node["call_model"]["tokens_in"] == 900
        assert by_node["call_model"]["latency_ms"] == 412.5
        assert by_node["guard_input"]["tokens_in"] == 0

    def test_an_unattributed_step_is_reported_rather_than_dropped(self, full_agent):
        # wardhook-observability parks usage that arrived while no node was open
        # on a synthetic "(ungrouped)" step. It has no box, and the overlay lists
        # it separately -- a cost you cannot attribute is still a cost you paid.
        sink = MemorySink(
            trace(
                "run-1",
                steps=[
                    step("call_model", cost=0.001),
                    step("(ungrouped)", latency_ms=0.0, usage=usage(300, 0), cost=0.0021),
                ],
            )
        )
        client = TestClient(create_dashboard(full_agent, telemetry=sink))
        nodes = [s["node"] for s in client.get("/api/runs/run-1").json()["steps"]]
        drawn = set(re.findall(r'data-metric-for="([^"]+)"', client.get("/").text))

        assert "(ungrouped)" in nodes
        assert "(ungrouped)" not in drawn
        assert client.get("/api/runs/run-1").json()["totals"]["cost"] == 0.0031

    def test_a_node_visited_twice_is_reported_twice_and_totals_agree(self, full_agent):
        # The tool loop returns to call_model on every round trip. Both visits
        # are in the trace and the run's totals include both, so the overlay has
        # to add them up rather than show the last one -- which is what it does.
        sink = MemorySink(
            trace(
                "run-1",
                steps=[
                    step("call_model", latency_ms=10.0, usage=usage(4200, 180), cost=0.0084),
                    step("tools", latency_ms=22.0),
                    step("call_model", latency_ms=8.0, usage=usage(4200, 180), cost=0.0084),
                ],
            )
        )
        run = TestClient(create_dashboard(full_agent, telemetry=sink)).get("/api/runs/run-1").json()
        visits = [s for s in run["steps"] if s["node"] == "call_model"]
        assert len(visits) == 2
        assert sum(s["tokens_in"] for s in visits) == run["totals"]["tokens_in"]
        assert sum(s["cost"] for s in visits) == run["totals"]["cost"]


class TestDescribesTheSystemDesign:
    def test_it_reports_how_retrieval_is_wired(self, full_agent):
        # "Retrieval: enabled" does not answer the question a developer has,
        # which is whether the thing is actually going to find anything.
        block = describe_agent(full_agent)["retrieval"]
        assert block["enabled"] is True
        assert block["retriever"] == "Retriever"
        assert block["store"] == "InMemoryVectorStore"
        assert block["embeddings"] == "HashingEmbeddings"
        assert block["top_k"] == 4
        assert block["indexed_chunks"] == 2

    def test_an_empty_index_is_visible_rather_than_implied(self, make_model):
        # An agent whose store is empty looks identical to a working one until
        # you notice it is answering from nothing. This is the tell.
        agent = AgentGraph(model=make_model("ok"), retriever=Retriever(InMemoryVectorStore()))
        assert describe_agent(agent)["retrieval"]["indexed_chunks"] == 0
        assert "0 chunks indexed" in TestClient(create_dashboard(agent)).get("/").text

    def test_a_store_that_cannot_be_sized_is_not_an_error(self, make_model):
        # A store from outside Wardhook need not implement __len__.
        class Unsized:
            def search(self, query, k=4):
                return []

        agent = AgentGraph(model=make_model("ok"), retriever=Retriever(Unsized()))
        assert describe_agent(agent)["retrieval"]["indexed_chunks"] is None
        assert "size unknown" in TestClient(create_dashboard(agent)).get("/").text

    def test_no_retriever_says_so_rather_than_showing_empty_rows(self, bare_agent):
        assert describe_agent(bare_agent)["retrieval"] == {"enabled": False}
        page = TestClient(create_dashboard(bare_agent)).get("/").text
        assert "not configured" in page
        assert "vector store" not in page

    def test_it_reports_the_orchestration_limits(self, full_agent):
        block = describe_agent(full_agent)["orchestration"]
        assert block["max_tool_iterations"] == 10
        assert block["guardrail_error_policy"] == "block"
        page = TestClient(create_dashboard(full_agent)).get("/").text
        assert "at most 10 round trips" in page
        assert "if a guardrail raises" in page

    def test_it_names_the_model(self, full_agent):
        assert describe_agent(full_agent)["model"] == "ToolCallingFake"

    def test_a_plain_callable_reports_nothing_rather_than_raising(self):
        # Serving a bare object with .invoke() is supported, so every one of
        # these reads has to survive an agent that has none of them.
        described = describe_agent(Minimal())
        assert described["model"] is None
        assert described["retrieval"] == {"enabled": False}
        assert described["orchestration"] == {
            "max_tool_iterations": None,
            "guardrail_error_policy": None,
        }
        assert TestClient(create_dashboard(Minimal())).get("/").status_code == 200


class TestBranding:
    def test_the_logo_is_drawn_inline_not_fetched(self, bare_agent):
        # A base64 PNG would arrive as src="data:..." and trip the page's own
        # no-external-resource check -- a check worth keeping sharper than the
        # logo. Drawn as SVG, it costs under a kilobyte and stays crisp.
        page = TestClient(create_dashboard(bare_agent)).get("/").text
        assert 'aria-label="Wardhook"' in page
        assert "<img" not in page
        assert re.findall(r'(?:src|href)\s*=\s*["\'](?!#)([^"\']+)', page) == []

    def test_the_logo_adapts_to_the_readers_theme(self, bare_agent):
        # The mark is stroked with a gradient of brand tokens, and the tokens
        # are redefined under prefers-color-scheme, so one page works on both.
        page = TestClient(create_dashboard(bare_agent)).get("/").text
        assert "--brand-1" in page
        assert "prefers-color-scheme: dark" in page
        assert page.count("--brand-1:") == 2, "brand colours need a dark variant"
