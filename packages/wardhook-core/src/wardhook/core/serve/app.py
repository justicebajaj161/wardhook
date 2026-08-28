"""FastAPI application factory wrapping an agent.

:func:`create_app` turns any object with an ``.invoke()`` method into a REST
service. That includes an :class:`~wardhook.core.agent.AgentGraph`, a raw
compiled LangGraph graph, or a plain function you wrote yourself -- the server
does not care which, because it only ever calls ``.invoke()``.

Three endpoints are exposed:

* ``POST /invoke`` -- run the agent and return its result.
* ``GET  /health`` -- liveness probe for orchestrators.
* ``GET  /info``   -- the agent's configuration, for debugging a deployment.

A fourth thing can be mounted alongside them, and is off until asked for: the
read-only dashboard from :mod:`wardhook.core.serve.dashboard`. It is opt-in
because a governance tool must never ship a debug UI that becomes reachable in
production because nobody turned it off.

Guardrail decisions surface in the response as structured records, and a run
blocked by policy returns ``200`` with ``blocked: true`` rather than an error
status. A blocked request is a *successful* policy evaluation, not a server
fault, and conflating the two makes monitoring meaningless.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wardhook.core.serve.dashboard import create_dashboard, dashboard_enabled
from wardhook.core.serve.topology import describe_agent

__all__ = ["InvokeRequest", "InvokeResponse", "create_app"]


class InvokeRequest(BaseModel):
    """Body of a ``POST /invoke`` call.

    Attributes:
        input: The user's message.
        principal: Caller identity for role-based guardrail decisions, for
            example ``{"id": "u-17", "roles": ["claims-agent"]}``.
        run_id: Client-supplied run identifier. Generated when omitted.
    """

    input: str = Field(..., min_length=1, description="The user's message.")
    principal: dict[str, Any] | None = Field(
        default=None, description="Caller identity, e.g. {'id': ..., 'roles': [...]}."
    )
    run_id: str | None = Field(default=None, description="Optional client-supplied run id.")


class InvokeResponse(BaseModel):
    """Result of a ``POST /invoke`` call.

    Attributes:
        output: The agent's final text response.
        run_id: Identifier correlating this response with traces and audit records.
        blocked: Whether a guardrail halted the run.
        block_reason: Why it was halted, when ``blocked`` is true.
        citations: Source attributions for any retrieved context.
        guardrail_events: One record per non-allow guardrail decision.
        tool_calls: Names of the tools that executed, in order.
    """

    output: str
    run_id: str
    blocked: bool = False
    block_reason: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)


def create_app(
    agent: Any,
    *,
    title: str = "Wardhook Agent",
    version: str = "0.1.0",
    cors_origins: list[str] | None = None,
    dashboard: bool | None = None,
    dashboard_path: str = "/dashboard",
    telemetry: Any = None,
) -> FastAPI:
    """Wrap an agent in a FastAPI application.

    Args:
        agent: Any object exposing ``.invoke()``.
        title: OpenAPI title shown at ``/docs``.
        version: OpenAPI version string.
        cors_origins: Allowed CORS origins. Defaults to the comma-separated
            ``WARDHOOK_CORS_ORIGINS`` environment variable, or no CORS at all.
            CORS is off unless explicitly configured, so an agent is never
            browser-reachable from arbitrary origins by accident.
        dashboard: Whether to mount the read-only dashboard. Defaults to the
            ``WARDHOOK_DASHBOARD`` environment variable, and to off. It follows
            the same rule as ``cors_origins``: a feature that widens what the
            server exposes is never on unless somebody said so.
        dashboard_path: Where to mount it.
        telemetry: The sink the dashboard reads runs from. Defaults to the
            agent's own. Pass a shared trace store to see every worker's runs
            rather than one process's.

    Returns:
        The configured application.

    Raises:
        TypeError: If ``agent`` has no ``.invoke()`` method.

    Example:
        >>> from langchain_core.language_models.fake_chat_models import (
        ...     GenericFakeChatModel,
        ... )
        >>> from langchain_core.messages import AIMessage
        >>> from wardhook.core import AgentGraph
        >>> model = GenericFakeChatModel(messages=iter([AIMessage(content="pong")]))
        >>> app = create_app(AgentGraph(model=model))
        >>> app.title
        'Wardhook Agent'
    """
    if not hasattr(agent, "invoke"):
        raise TypeError(f"{type(agent).__name__} has no .invoke() method, so it cannot be served.")

    app = FastAPI(
        title=title,
        version=version,
        description="An agent served by wardhook-core.",
    )

    origins = cors_origins
    if origins is None:
        raw = os.getenv("WARDHOOK_CORS_ORIGINS", "").strip()
        origins = [o.strip() for o in raw.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Report that the service is up.

        Returns:
            A small status payload suitable for a liveness probe.
        """
        return {"status": "ok", "agent": str(describe_agent(agent)["name"])}

    @app.get("/info", tags=["ops"])
    def info() -> dict[str, Any]:
        """Describe the served agent's configuration.

        Returns:
            Tool names, guardrail names, and which optional features are on.
        """
        return describe_agent(agent)

    @app.post("/invoke", response_model=InvokeResponse, tags=["agent"])
    def invoke(request: InvokeRequest) -> InvokeResponse:
        """Run the agent against one user message.

        Args:
            request: The parsed request body.

        Returns:
            The agent's result. A run stopped by a guardrail returns ``200``
            with ``blocked: true``.

        Raises:
            HTTPException: ``500`` if the agent itself raised.
        """
        try:
            result = agent.invoke(
                request.input,
                principal=request.principal,
                run_id=request.run_id,
            )
        except TypeError:
            # A plain callable target may not accept our keyword arguments.
            result = agent.invoke(request.input)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Agent failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(result, dict):
            result = {"output": str(result)}

        return InvokeResponse(
            output=str(result.get("output", "")),
            run_id=str(result.get("run_id") or request.run_id or ""),
            blocked=bool(result.get("blocked", False)),
            block_reason=result.get("block_reason"),
            citations=list(result.get("citations") or []),
            guardrail_events=list(result.get("guardrail_events") or []),
            tool_calls=list(result.get("tool_calls") or []),
        )

    if dashboard_enabled(dashboard):
        app.mount(dashboard_path, create_dashboard(agent, telemetry))

    return app
