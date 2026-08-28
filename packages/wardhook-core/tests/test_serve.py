"""Tests for the FastAPI wrapper and the CLI target loader."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import typer
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from wardhook.core.agent import AgentGraph
from wardhook.core.serve.app import create_app
from wardhook.core.serve.cli import app as cli_app
from wardhook.core.serve.cli import load_target

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(result):
    """Return CLI output with colour, box drawing, and wrapping removed.

    CI forces colour and wraps Typer/rich output in a box, so a plain substring
    assertion passes locally and fails on a runner.

    Args:
        result: A ``CliRunner`` result.

    Returns:
        The visible text as one whitespace-normalised line.
    """
    text = _ANSI.sub("", result.output)
    for char in "\u2502\u256d\u256e\u2570\u256f\u2500":
        text = text.replace(char, " ")
    return " ".join(text.split())


def _write_agent_module(tmp_path, name="cli_agent_mod", body=None):
    """Write an importable module exposing a fake-model agent as ``agent``."""
    module = tmp_path / f"{name}.py"
    module.write_text(
        body
        or (
            "from langchain_core.language_models.fake_chat_models import GenericFakeChatModel\n"
            "from langchain_core.messages import AIMessage\n"
            "from wardhook.core import AgentGraph\n"
            "agent = AgentGraph(model=GenericFakeChatModel("
            "messages=iter([AIMessage(content='hi')])), name='cli-agent')\n"
        ),
        encoding="utf-8",
    )
    return module


class Denier:
    name = "denier"

    def on_input(self, text, context):
        return SimpleNamespace(action="block", text=text, reason="policy says no", rule="denylist")


@pytest.fixture
def client(make_model):
    agent = AgentGraph(model=make_model("Hello there."), name="test-agent")
    return TestClient(create_app(agent))


class TestApp:
    def test_health_reports_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_info_describes_the_agent(self, make_model, echo_tool, make_tool_model):
        agent = AgentGraph(
            model=make_tool_model(AIMessage(content="ok")),
            tools=[echo_tool],
            guardrails=[Denier()],
            name="described",
        )
        body = TestClient(create_app(agent)).get("/info").json()
        assert body["name"] == "described"
        assert body["tools"] == ["lookup_account"]
        assert body["guardrails"] == ["denier"]
        assert body["retrieval_enabled"] is False

    def test_invoke_returns_the_agent_output(self, client):
        response = client.post("/invoke", json={"input": "hi"})
        assert response.status_code == 200
        body = response.json()
        assert body["output"] == "Hello there."
        assert body["blocked"] is False
        assert len(body["run_id"]) == 32

    def test_a_blocked_run_is_a_200_not_an_error(self, make_model):
        # A guardrail firing is a successful policy evaluation, not a server
        # fault. Returning 4xx/5xx here would poison error-rate monitoring.
        agent = AgentGraph(model=make_model("unused"), guardrails=[Denier()])
        response = TestClient(create_app(agent)).post("/invoke", json={"input": "hi"})
        assert response.status_code == 200
        body = response.json()
        assert body["blocked"] is True
        assert body["block_reason"] == "policy says no"

    def test_empty_input_is_rejected_by_validation(self, client):
        assert client.post("/invoke", json={"input": ""}).status_code == 422

    def test_missing_input_is_rejected_by_validation(self, client):
        assert client.post("/invoke", json={}).status_code == 422

    def test_client_supplied_run_id_is_echoed_back(self, client):
        body = client.post("/invoke", json={"input": "hi", "run_id": "abc"}).json()
        assert body["run_id"] == "abc"

    def test_principal_reaches_the_guardrails(self, make_model):
        seen = {}

        class Spy:
            name = "spy"

            def on_input(self, text, context):
                seen["principal"] = context.get("principal")
                return SimpleNamespace(action="allow", text=text, reason=None, rule=None)

        agent = AgentGraph(model=make_model("ok"), guardrails=[Spy()])
        TestClient(create_app(agent)).post(
            "/invoke", json={"input": "hi", "principal": {"id": "u1", "roles": ["admin"]}}
        )
        assert seen["principal"]["roles"] == ["admin"]

    def test_an_agent_failure_becomes_a_500(self):
        class Broken:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("model exploded")

        response = TestClient(create_app(Broken()), raise_server_exceptions=False).post(
            "/invoke", json={"input": "hi"}
        )
        assert response.status_code == 500
        assert "model exploded" in response.json()["detail"]

    def test_serves_any_object_with_invoke(self):
        class Minimal:
            def invoke(self, text, **kwargs):
                return {"output": f"echo: {text}"}

        body = TestClient(create_app(Minimal())).post("/invoke", json={"input": "hi"}).json()
        assert body["output"] == "echo: hi"

    def test_a_non_dict_return_value_is_coerced(self):
        class Stringy:
            def invoke(self, text, **kwargs):
                return "plain string"

        body = TestClient(create_app(Stringy())).post("/invoke", json={"input": "hi"}).json()
        assert body["output"] == "plain string"

    def test_rejects_an_object_that_cannot_be_served(self):
        with pytest.raises(TypeError, match=r"no \.invoke"):
            create_app(object())

    def test_cors_is_off_unless_configured(self, client):
        headers = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
        response = client.options("/invoke", headers=headers)
        assert "access-control-allow-origin" not in {k.lower() for k in response.headers}

    def test_cors_can_be_enabled_explicitly(self, make_model):
        agent = AgentGraph(model=make_model("ok"))
        app = create_app(agent, cors_origins=["https://app.example"])
        response = TestClient(app).options(
            "/invoke",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers["access-control-allow-origin"] == "https://app.example"

    def test_openapi_schema_is_generated(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestLoadTarget:
    def test_loads_an_agent_attribute(self, tmp_path, monkeypatch):
        module = tmp_path / "sample_agent_mod.py"
        module.write_text(
            "from langchain_core.language_models.fake_chat_models import GenericFakeChatModel\n"
            "from langchain_core.messages import AIMessage\n"
            "from wardhook.core import AgentGraph\n"
            "agent = AgentGraph(model=GenericFakeChatModel("
            "messages=iter([AIMessage(content='hi')])))\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        assert hasattr(load_target("sample_agent_mod:agent"), "invoke")

    def test_calls_a_zero_argument_factory(self, tmp_path, monkeypatch):
        module = tmp_path / "factory_mod.py"
        module.write_text(
            "from langchain_core.language_models.fake_chat_models import GenericFakeChatModel\n"
            "from langchain_core.messages import AIMessage\n"
            "from wardhook.core import AgentGraph\n"
            "def build():\n"
            "    return AgentGraph(model=GenericFakeChatModel("
            "messages=iter([AIMessage(content='hi')])))\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        assert hasattr(load_target("factory_mod:build"), "invoke")

    def test_missing_colon_is_rejected(self):
        with pytest.raises(typer.BadParameter, match="module:attribute"):
            load_target("just_a_module")

    def test_unimportable_module_is_rejected(self):
        with pytest.raises(typer.BadParameter, match="Could not import"):
            load_target("no_such_module_xyz:agent")

    def test_missing_attribute_is_rejected(self):
        with pytest.raises(typer.BadParameter, match="has no attribute"):
            load_target("wardhook.core:not_a_real_attribute")

    def test_object_without_invoke_is_rejected(self):
        with pytest.raises(typer.BadParameter, match=r"no \.invoke"):
            load_target("wardhook.core:DEFAULT_MODEL")


class TestServeCommand:
    def test_starts_uvicorn_with_the_requested_bind(self, tmp_path, monkeypatch):
        # uvicorn.run blocks forever, so the command is driven with it replaced.
        # What is being checked is the wiring: the right app, host, and port.
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured = {}

        def fake_run(application, **kwargs):
            captured["app"] = application
            captured.update(kwargs)

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", fake_run)

        result = runner.invoke(cli_app, ["serve", "cli_agent_mod:agent", "--port", "9123"])

        assert result.exit_code == 0, plain(result)
        assert "Serving cli_agent_mod:agent on http://127.0.0.1:9123" in plain(result)
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 9123
        assert captured["reload"] is False

    def test_cors_origins_reach_the_application(self, tmp_path, monkeypatch):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured = {}

        import uvicorn

        monkeypatch.setattr(
            uvicorn, "run", lambda application, **_kw: captured.update(app=application)
        )

        result = runner.invoke(
            cli_app,
            [
                "serve",
                "cli_agent_mod:agent",
                "--cors-origin",
                "https://a.example",
                "--cors-origin",
                "https://b.example",
            ],
        )

        assert result.exit_code == 0, plain(result)
        client = TestClient(captured["app"])
        response = client.get("/health", headers={"Origin": "https://a.example"})
        assert response.headers["access-control-allow-origin"] == "https://a.example"

    def test_an_unloadable_target_fails_before_binding_a_port(self, monkeypatch):
        import uvicorn

        monkeypatch.setattr(
            uvicorn, "run", lambda *_a, **_k: pytest.fail("should never reach uvicorn")
        )
        result = runner.invoke(cli_app, ["serve", "no_such_module_xyz:agent"])
        assert result.exit_code != 0


class TestInfoCommand:
    def test_prints_the_agent_configuration_as_json(self, tmp_path, monkeypatch):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli_app, ["info", "cli_agent_mod:agent"])

        assert result.exit_code == 0, plain(result)
        described = json.loads(result.output)
        assert described["name"] == "cli-agent"

    def test_a_bad_target_is_reported_rather_than_traced(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli_app, ["info", "no_such_module_xyz:agent"])
        assert result.exit_code != 0
        assert "Could not import" in plain(result)


class TestFactoryLoading:
    def test_a_factory_that_raises_is_reported_with_its_cause(self, tmp_path, monkeypatch):
        _write_agent_module(
            tmp_path,
            name="exploding_factory_mod",
            body=("def build():\n    raise ValueError('missing config')\n"),
        )
        monkeypatch.chdir(tmp_path)

        with pytest.raises(typer.BadParameter, match="raised ValueError: missing config"):
            load_target("exploding_factory_mod:build")


class TestConsoleEntryPoint:
    def test_main_delegates_to_the_typer_app(self, monkeypatch):
        # `main` is what the `wardhook` console script actually calls, so a
        # break here is invisible to every other test and total for the user.
        from wardhook.core.serve import cli as cli_module

        called = []
        monkeypatch.setattr(cli_module, "app", lambda: called.append(True))
        cli_module.main()
        assert called == [True]


class TestCallableTargets:
    def test_a_target_rejecting_keyword_arguments_is_retried_positionally(self):
        # A bare callable is a legitimate target: anything with .invoke() works.
        # One that takes only the input must not 500 on principal/run_id.
        class PositionalOnly:
            def invoke(self, user_input):
                return {"output": f"echo: {user_input}"}

        client = TestClient(create_app(PositionalOnly()))
        response = client.post("/invoke", json={"input": "hello", "run_id": "r1"})

        assert response.status_code == 200
        assert response.json()["output"] == "echo: hello"


class TestDashboardMounting:
    def test_it_is_off_unless_asked_for(self, client):
        # The default a governance tool has to have: a debug UI that nobody
        # turned on is not running.
        assert client.get("/dashboard/").status_code == 404

    def test_it_mounts_when_asked_for(self, make_model):
        app = create_app(AgentGraph(model=make_model("ok"), name="mounted"), dashboard=True)
        assert TestClient(app).get("/dashboard/").status_code == 200
        assert TestClient(app).get("/dashboard/api/topology").json()["agent"] == "mounted"

    def test_mounting_it_does_not_disturb_the_agent_endpoints(self, make_model):
        app = create_app(AgentGraph(model=make_model("Hello there."), name="m"), dashboard=True)
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "ok"
        assert client.post("/invoke", json={"input": "hi"}).json()["output"] == "Hello there."

    def test_it_can_be_mounted_somewhere_else(self, make_model):
        app = create_app(
            AgentGraph(model=make_model("ok")), dashboard=True, dashboard_path="/_internal/ops"
        )
        client = TestClient(app)
        assert client.get("/_internal/ops/").status_code == 200
        assert client.get("/dashboard/").status_code == 404

    def test_the_environment_variable_can_turn_it_on(self, make_model, monkeypatch):
        monkeypatch.setenv("WARDHOOK_DASHBOARD", "1")
        app = create_app(AgentGraph(model=make_model("ok")))
        assert TestClient(app).get("/dashboard/").status_code == 200

    def test_an_explicit_false_beats_the_environment_variable(self, make_model, monkeypatch):
        # Otherwise an environment variable set for one service could turn the
        # UI on in another that had deliberately switched it off.
        monkeypatch.setenv("WARDHOOK_DASHBOARD", "1")
        app = create_app(AgentGraph(model=make_model("ok")), dashboard=False)
        assert TestClient(app).get("/dashboard/").status_code == 404

    def test_an_unrecognised_environment_value_leaves_it_off(self, make_model, monkeypatch):
        monkeypatch.setenv("WARDHOOK_DASHBOARD", "maybe")
        app = create_app(AgentGraph(model=make_model("ok")))
        assert TestClient(app).get("/dashboard/").status_code == 404

    def test_a_shared_sink_can_be_handed_to_the_dashboard(self, make_model):
        # The documented multi-worker mitigation, reachable from create_app:
        # the agent writes through its own tracer, the dashboard reads the file.
        class StoreSink:
            path = "traces.jsonl"

            def read(self):
                return []

        app = create_app(AgentGraph(model=make_model("ok")), dashboard=True, telemetry=StoreSink())
        assert TestClient(app).get("/dashboard/api/runs").json()["mode"] == "store"


class TestDashboardBindGuard:
    def _run(self, tmp_path, monkeypatch, *args):
        _write_agent_module(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured = {}

        import uvicorn

        monkeypatch.setattr(
            uvicorn, "run", lambda application, **kw: captured.update(app=application, **kw)
        )
        return runner.invoke(cli_app, ["serve", "cli_agent_mod:agent", *args]), captured

    def test_the_dashboard_url_is_printed_when_it_is_on(self, tmp_path, monkeypatch):
        result, captured = self._run(tmp_path, monkeypatch, "--dashboard", "--port", "9001")
        assert result.exit_code == 0, plain(result)
        assert "Dashboard at http://127.0.0.1:9001/dashboard/" in plain(result)
        assert TestClient(captured["app"]).get("/dashboard/").status_code == 200

    def test_nothing_is_mounted_or_announced_by_default(self, tmp_path, monkeypatch):
        result, captured = self._run(tmp_path, monkeypatch)
        assert result.exit_code == 0, plain(result)
        assert "Dashboard at" not in plain(result)
        assert TestClient(captured["app"]).get("/dashboard/").status_code == 404

    def test_a_non_loopback_bind_is_refused_before_a_port_is_opened(self, tmp_path, monkeypatch):
        # Two opt-ins, not one. Turning the dashboard on is one decision;
        # putting it on a network is a different and larger one.
        result, captured = self._run(tmp_path, monkeypatch, "--dashboard", "--host", "0.0.0.0")
        assert result.exit_code != 0
        assert captured == {}, "uvicorn was started despite the refusal"
        message = plain(result)
        assert "would put the dashboard on your network" in message
        assert "--dashboard-allow-remote" in message

    def test_the_second_opt_in_permits_it(self, tmp_path, monkeypatch):
        result, captured = self._run(
            tmp_path, monkeypatch, "--dashboard", "--host", "0.0.0.0", "--dashboard-allow-remote"
        )
        assert result.exit_code == 0, plain(result)
        assert captured["host"] == "0.0.0.0"
        assert TestClient(captured["app"]).get("/dashboard/").status_code == 200

    def test_a_non_loopback_bind_without_the_dashboard_is_untouched(self, tmp_path, monkeypatch):
        # The guard is about the dashboard, not about binding. Serving the agent
        # itself on 0.0.0.0 was already an explicit choice and stays allowed.
        result, captured = self._run(tmp_path, monkeypatch, "--host", "0.0.0.0")
        assert result.exit_code == 0, plain(result)
        assert captured["host"] == "0.0.0.0"

    def test_localhost_by_name_counts_as_loopback(self, tmp_path, monkeypatch):
        result, _ = self._run(tmp_path, monkeypatch, "--dashboard", "--host", "localhost")
        assert result.exit_code == 0, plain(result)

    def test_a_custom_mount_path_reaches_the_application(self, tmp_path, monkeypatch):
        result, captured = self._run(
            tmp_path, monkeypatch, "--dashboard", "--dashboard-path", "/ops"
        )
        assert result.exit_code == 0, plain(result)
        assert "Dashboard at http://127.0.0.1:8000/ops/" in plain(result)
        assert TestClient(captured["app"]).get("/ops/").status_code == 200
