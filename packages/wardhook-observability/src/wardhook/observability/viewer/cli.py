"""The `wardhook-trace` command.

Turns a JSON Lines trace file into a single HTML page you can open, attach to a
ticket, or publish as a CI artifact::

    wardhook-trace view traces/run-42.jsonl -o trace.html

There is also a terminal summary for when opening a browser is more ceremony
than the question deserves::

    wardhook-trace summary traces/run-42.jsonl
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from wardhook.observability.models import Trace
from wardhook.observability.store import load_traces
from wardhook.observability.viewer.html import render_html

__all__ = ["app", "main"]

app = typer.Typer(
    name="wardhook-trace",
    help="Inspect Wardhook trace files.",
    add_completion=False,
    no_args_is_help=True,
)


def _load(source: Path, run_id: str | None) -> list[Trace]:
    """Read a trace file, optionally narrowing to one run.

    Args:
        source: The JSON Lines trace file.
        run_id: A single run to keep, or ``None`` for all of them.

    Returns:
        The traces to render.

    Raises:
        typer.BadParameter: If the file is missing, malformed, or holds no run
            with the requested id.
    """
    try:
        traces = load_traces(source)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if run_id is None:
        return traces
    matching = [trace for trace in traces if trace.run_id == run_id]
    if not matching:
        available = ", ".join(trace.run_id for trace in traces[:5]) or "none"
        raise typer.BadParameter(
            f"No run {run_id!r} in {source}. First few run ids present: {available}"
        )
    return matching


@app.command()
def view(
    source: Annotated[Path, typer.Argument(help="JSON Lines trace file to render.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the HTML page.")
    ] = Path("trace.html"),
    run_id: Annotated[str | None, typer.Option("--run-id", help="Render only this run.")] = None,
    title: Annotated[
        str | None, typer.Option("--title", help="Heading and browser tab title.")
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the page after writing it.")
    ] = False,
) -> None:
    """Render a trace file to a self-contained HTML page."""
    traces = _load(source, run_id)
    page = render_html(traces, title=title or f"Wardhook trace - {source.name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")

    typer.echo(f"Wrote {output} ({len(traces)} run(s), {len(page):,} bytes, no external requests)")
    if open_browser:
        webbrowser.open(output.resolve().as_uri())


@app.command()
def summary(
    source: Annotated[Path, typer.Argument(help="JSON Lines trace file to summarise.")],
    run_id: Annotated[str | None, typer.Option("--run-id", help="Summarise only this run.")] = None,
) -> None:
    """Print a per-node breakdown of a trace file to the terminal."""
    traces = _load(source, run_id)
    if not traces:
        typer.echo(f"{source} contains no runs.")
        return

    for trace in traces:
        status = "  FAILED" if trace.failed else ""
        typer.echo(f"\nrun {trace.run_id}  {trace.started_at}{status}")
        typer.echo(f"  {'node':<18} {'ms':>9} {'in':>9} {'out':>9} {'cost':>11}")
        for step in trace.steps:
            typer.echo(
                f"  {step.node:<18} {step.latency_ms:>9.0f} {step.tokens_in:>9,} "
                f"{step.tokens_out:>9,} {step.cost:>11.5f}"
            )
        usage = trace.total_usage
        typer.echo(
            f"  {'TOTAL':<18} {trace.latency_ms:>9.0f} {usage.input_tokens:>9,} "
            f"{usage.output_tokens:>9,} {trace.total_cost:>11.5f}"
        )

    if len(traces) > 1:
        grand = sum(trace.total_cost for trace in traces)
        typer.echo(f"\n{len(traces)} runs, ${grand:.4f} total")


def main() -> None:
    """Entry point for the ``wardhook-trace`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
