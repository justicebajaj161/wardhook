"""The `wardhook-eval` command.

Two subcommands, matching the two questions a suite gets asked:

``run`` -- *does the agent pass its cases right now?*::

    wardhook-eval run cases.jsonl --target myapp.agents:support_agent -o run.json

``compare`` -- *did my change break anything that used to work?*::

    wardhook-eval compare run.json --baseline baseline.json   # exits 1 on regression

The split matters in CI. ``run`` exits non-zero on any failure, which is what
you want for a suite that is meant to be green. ``compare`` exits non-zero only
on a *regression*, which is what you want for a suite carrying known debt: it
blocks the change that broke something without blocking every build on failures
that were already there.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from wardhook.evals.baseline import compare
from wardhook.evals.case import CaseFormatError, load_cases
from wardhook.evals.report import CaseResult, EvalReport
from wardhook.evals.runner import EvalRunner

__all__ = ["app", "load_target", "main"]

app = typer.Typer(
    name="wardhook-eval",
    help="Run agent test cases and detect regressions.",
    add_completion=False,
    no_args_is_help=True,
)


def load_target(target: str) -> Any:
    """Import an evaluation target from a ``module:attribute`` string.

    Deliberately a copy of the equivalent helper in ``wardhook-core`` rather
    than an import of it. This package must work with no other Wardhook package
    installed, and duplicating fifteen lines is a smaller cost than a
    dependency that exists only to share them.

    Args:
        target: An entry-point reference such as ``myapp.agents:support_agent``.
            The attribute may be an agent or a zero-argument factory.

    Returns:
        The resolved target.

    Raises:
        typer.BadParameter: If the string is malformed, the module or attribute
            cannot be found, a factory raises, or the result has no
            ``.invoke()``.
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
        raise typer.BadParameter(f"Could not import module {module_name!r}: {exc}") from exc

    try:
        resolved = getattr(module, attribute)
    except AttributeError as exc:
        available = ", ".join(name for name in vars(module) if not name.startswith("_"))
        raise typer.BadParameter(
            f"Module {module_name!r} has no attribute {attribute!r}. Available: {available}"
        ) from exc

    if callable(resolved) and not callable(getattr(resolved, "invoke", None)):
        try:
            resolved = resolved()
        except Exception as exc:
            raise typer.BadParameter(f"Factory {target!r} raised: {exc}") from exc

    if not callable(getattr(resolved, "invoke", None)):
        raise typer.BadParameter(
            f"{target!r} resolved to {type(resolved).__name__}, which has no .invoke() method."
        )
    return resolved


def _read_cases(path: Path) -> list[Any]:
    """Load a case file, turning format errors into CLI errors.

    Args:
        path: The JSONL case file.

    Returns:
        The parsed cases.

    Raises:
        typer.BadParameter: If the file is missing or malformed.
    """
    try:
        return load_cases(path)
    except (FileNotFoundError, CaseFormatError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_report(path: Path, label: str) -> EvalReport:
    """Load a report file, turning errors into CLI errors.

    Args:
        path: The report file.
        label: What this file is, for the error message.

    Returns:
        The report.

    Raises:
        typer.BadParameter: If the file is missing or malformed.
    """
    try:
        return EvalReport.load(path)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(f"{label}: {exc}") from exc


def _echo_result(result: CaseResult) -> None:
    """Print one case's outcome as it completes.

    Args:
        result: The finished case.
    """
    mark = "PASS" if result.passed else "FAIL"
    line = f"  {mark}  {result.id}"
    if not result.passed:
        line += f"  -- {result.summary()}"
    typer.echo(line)


@app.command()
def run(
    cases: Annotated[Path, typer.Argument(help="JSONL file of test cases.")],
    target: Annotated[
        str, typer.Option("--target", "-t", help="Agent to evaluate, as 'module:attribute'.")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the report here as JSON.")
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Compare against this baseline straight after running."),
    ] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Only run cases with this tag. Repeatable.")
    ] = None,
    no_output: Annotated[
        bool,
        typer.Option(
            "--no-output", help="Omit agent output from the report, so it carries no data."
        ),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only print the summary.")] = False,
) -> None:
    """Run test cases against an agent.

    Exits non-zero if any case fails. Use `compare` instead when the suite
    carries known failures and only regressions should break the build.
    """
    parsed = _read_cases(cases)
    agent = load_target(target)
    runner = EvalRunner(agent, include_output=not no_output)

    if not quiet:
        typer.echo(f"Running {len(parsed)} case(s) against {target}\n")

    report = runner.run(
        parsed,
        tags=tag or None,
        on_result=None if quiet else _echo_result,
        metadata={"cases": str(cases), "target": target},
    )

    typer.echo(
        f"\n{report.passed}/{report.total} passed"
        f"  ({report.pass_rate:.0%})"
        + (f"  ${report.total_cost:.4f}" if report.total_cost else "")
        + f"  {report.duration_ms / 1000:.2f}s"
    )

    if output is not None:
        report.save(output)
        typer.echo(f"Report written to {output}")

    if baseline is not None:
        comparison = compare(report, _load_report(baseline, "baseline"))
        typer.echo(f"\nAgainst baseline: {comparison.summary()}")
        if comparison.has_regressions:
            for item in comparison.regressed:
                typer.echo(f"  REGRESSED  {item.id}  -- {item.detail}")
            raise typer.Exit(1)

    if report.failed:
        raise typer.Exit(1)


@app.command(name="compare")
def compare_command(
    current: Annotated[Path, typer.Argument(help="Report from the run just completed.")],
    baseline: Annotated[Path, typer.Option("--baseline", "-b", help="Saved reference report.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the comparison here as JSON.")
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Also fail on new cases that do not pass.")
    ] = False,
) -> None:
    """Compare a run against a baseline and fail on regressions.

    Exits 1 if any case that passed in the baseline now fails. Cases that were
    already failing do not fail the build -- that is the whole point of
    comparing rather than counting.
    """
    comparison = compare(
        _load_report(current, "current report"), _load_report(baseline, "baseline")
    )

    typer.echo(comparison.summary())
    for item in comparison.regressed:
        typer.echo(f"  REGRESSED     {item.id}  -- {item.detail}")
    for item in comparison.fixed:
        typer.echo(f"  FIXED         {item.id}")
    for item in comparison.new_failures:
        typer.echo(f"  NEW FAILURE   {item.id}  -- {item.detail}")
    for item in comparison.removed:
        typer.echo(f"  REMOVED       {item.id}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        typer.echo(f"Comparison written to {output}")

    if comparison.has_regressions:
        typer.echo(f"\n{len(comparison.regressed)} regression(s).")
        raise typer.Exit(1)
    if strict and comparison.new_failures:
        typer.echo(f"\n{len(comparison.new_failures)} new failing case(s), and --strict is set.")
        raise typer.Exit(1)
    typer.echo("\nNo regressions.")


@app.command()
def validate(
    cases: Annotated[Path, typer.Argument(help="JSONL file of test cases to check.")],
) -> None:
    """Check that a case file parses, without running anything.

    Useful as a fast pre-commit hook: a malformed case file should be caught
    before an expensive run starts, not halfway through one.
    """
    parsed = _read_cases(cases)
    tags = sorted({tag for case in parsed for tag in case.tags})
    criteria = sorted({name for case in parsed for name in case.expect})

    typer.echo(f"{cases}: {len(parsed)} case(s), OK")
    if criteria:
        typer.echo(f"  criteria used: {', '.join(criteria)}")
    if tags:
        typer.echo(f"  tags: {', '.join(tags)}")


def main() -> None:
    """Entry point for the ``wardhook-eval`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
