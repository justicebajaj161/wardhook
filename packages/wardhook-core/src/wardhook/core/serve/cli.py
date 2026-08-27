"""The `wardhook serve` command.

Turns an agent defined anywhere on your ``PYTHONPATH`` into a running HTTP
service with one command::

    wardhook serve myapp.agents:support_agent --port 8000

The target uses the standard ``module:attribute`` entry-point syntax. The
attribute may be an agent instance or a zero-argument factory that returns one;
a factory is called at startup, which is the right shape when construction
needs to read configuration or open a connection.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

__all__ = ["app", "load_target", "main"]

app = typer.Typer(
    name="wardhook",
    help="Serve and inspect Wardhook agents.",
    add_completion=False,
    no_args_is_help=True,
)


def load_target(target: str) -> Any:
    """Import an agent from a ``module:attribute`` string.

    Args:
        target: An entry-point reference such as ``myapp.agents:support_agent``.
            The attribute may be an agent or a zero-argument factory.

    Returns:
        The resolved agent.

    Raises:
        typer.BadParameter: If the string is malformed, the module or attribute
            cannot be found, a factory raises, or the result has no ``.invoke()``.
    """
    if ":" not in target:
        raise typer.BadParameter(
            f"Expected 'module:attribute', got {target!r}. For example: myapp.agents:support_agent"
        )
    module_name, _, attribute = target.partition(":")

    # Make the working directory importable so a local agent.py just works
    # without the caller having to set PYTHONPATH first.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise typer.BadParameter(
            f"Could not import module {module_name!r}: {exc}. "
            f"Check the name and that it is on your PYTHONPATH."
        ) from exc

    try:
        obj = getattr(module, attribute)
    except AttributeError as exc:
        available = ", ".join(n for n in dir(module) if not n.startswith("_"))[:200]
        raise typer.BadParameter(
            f"Module {module_name!r} has no attribute {attribute!r}. Available: {available}"
        ) from exc

    if not hasattr(obj, "invoke") and callable(obj):
        try:
            obj = obj()
        except Exception as exc:
            raise typer.BadParameter(
                f"Calling factory {target!r} raised {type(exc).__name__}: {exc}"
            ) from exc

    if not hasattr(obj, "invoke"):
        raise typer.BadParameter(
            f"{target!r} resolved to {type(obj).__name__}, which has no .invoke() method."
        )
    return obj


@app.command()
def serve(
    target: Annotated[str, typer.Argument(help="Agent to serve, as 'module:attribute'.")],
    host: Annotated[
        str, typer.Option("--host", "-h", help="Interface to bind.", envvar="WARDHOOK_HOST")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to bind.", envvar="WARDHOOK_PORT")
    ] = 8000,
    reload: Annotated[
        bool, typer.Option("--reload", help="Restart on code changes (development only).")
    ] = False,
    log_level: Annotated[str, typer.Option("--log-level", help="Uvicorn log level.")] = "info",
    cors_origin: Annotated[
        list[str] | None,
        typer.Option("--cors-origin", help="Allowed CORS origin. Repeatable."),
    ] = None,
) -> None:
    """Serve an agent over HTTP.

    Binds to localhost by default. Serving on ``0.0.0.0`` exposes the agent to
    your whole network, so that has to be an explicit choice.

    Args:
        target: The agent to serve, as ``module:attribute``.
        host: Interface to bind.
        port: Port to bind.
        reload: Whether to restart on code changes.
        log_level: Uvicorn log level.
        cors_origin: Allowed CORS origins; may be repeated.
    """
    import uvicorn

    from wardhook.core.serve.app import create_app

    agent = load_target(target)
    application = create_app(agent, cors_origins=list(cors_origin) if cors_origin else None)

    typer.echo(f"Serving {target} on http://{host}:{port}  (docs at /docs)")
    uvicorn.run(application, host=host, port=port, reload=reload, log_level=log_level)


@app.command()
def info(
    target: Annotated[str, typer.Argument(help="Agent to inspect, as 'module:attribute'.")],
) -> None:
    """Print an agent's configuration without starting a server.

    Useful for confirming that guardrails and tools are wired the way you
    expect before exposing the agent.

    Args:
        target: The agent to inspect, as ``module:attribute``.
    """
    import json

    from wardhook.core.serve.app import _describe

    typer.echo(json.dumps(_describe(load_target(target)), indent=2))


def main() -> None:
    """Console-script entry point for ``wardhook``."""
    app()
