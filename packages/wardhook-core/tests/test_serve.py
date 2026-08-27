"""Tests for the FastAPI wrapper and the CLI target loader."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from wardhook.core.agent import AgentGraph
from wardhook.core.serve.app import create_app
from wardhook.core.serve.cli import load_target


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
