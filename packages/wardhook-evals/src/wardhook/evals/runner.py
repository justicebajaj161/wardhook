"""Executes test cases against any agent-like target.

The runner's whole design follows from one constraint: it must not know what it
is testing. A target is **anything with an** ``.invoke()`` **method** -- a
Wardhook ``AgentGraph``, a bare LangGraph graph, a LangChain chain, or a plain
function wrapped in a shim. Nothing here imports an agent framework, and this
package's only runtime dependency is its CLI library.

That is why :class:`~wardhook.evals.criteria.Outcome` exists: whatever the
target returns is flattened into one small record before any criterion sees it,
so criteria stay simple and targets stay unconstrained.

Two behaviours worth knowing:

* **A raising target fails its case, it does not abort the run.** One broken
  case out of two hundred should still leave you with a hundred and ninety-nine
  results and a report naming the broken one.
* **Cost is read back through duck typing.** If the target exposes ``.trace()``
  -- as an ``AgentGraph`` with telemetry attached does -- the run's cost is read
  from it, which is what makes the ``max_cost_usd`` criterion work. If it does
  not, cost is simply unknown, and nothing breaks.

Example:
    >>> from wardhook.evals import EvalCase
    >>> class Echo:
    ...     def invoke(self, text):
    ...         return {"output": f"you said {text}"}
    >>> report = EvalRunner(Echo()).run([EvalCase("a", "hi", {"contains": "you said hi"})])
    >>> report.passed, report.total
    (1, 1)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from wardhook.evals.case import EvalCase
from wardhook.evals.criteria import Outcome, evaluate
from wardhook.evals.report import CaseResult, EvalReport

__all__ = ["EvalRunner", "describe_target"]

# Keys a target might use for its final text, most specific first.
_TEXT_KEYS = ("output", "answer", "text", "response", "content", "result")


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def describe_target(target: Any) -> str:
    """Return a short, log-safe description of what is being evaluated.

    Args:
        target: The object under test.

    Returns:
        Its ``name`` if it has one, otherwise its class name.

    Example:
        >>> class Support:
        ...     name = "support-agent"
        >>> describe_target(Support())
        'support-agent'
    """
    name = getattr(target, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(target).__name__


def _text_of(value: Any) -> str:
    """Extract the final text from whatever a target returned.

    Args:
        value: The target's return value.

    Returns:
        The best available text. Falls back to ``str()`` rather than raising,
        because a target that returns something unexpected should produce a
        failing assertion, not a crashed run.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in _TEXT_KEYS:
            if key in value:
                return _text_of(value[key])
        return str(value)
    content = getattr(value, "content", None)
    if content is not None:
        return content if isinstance(content, str) else str(content)
    return str(value)


def _cost_of(target: Any, raw: Any) -> float | None:
    """Read the run's cost from the target's telemetry, if it has any.

    Args:
        target: The object under test.
        raw: Its return value, which may carry a ``run_id``.

    Returns:
        The cost in US dollars, or ``None`` when no telemetry is attached. All
        failures are swallowed: a missing cost must never fail a case.
    """
    tracer = getattr(target, "trace", None)
    if not callable(tracer):
        return None
    run_id = raw.get("run_id") if isinstance(raw, Mapping) else None
    try:
        trace = tracer(run_id)
    except Exception:
        return None
    cost = getattr(trace, "total_cost", None)
    return float(cost) if isinstance(cost, (int, float)) else None


class EvalRunner:
    """Runs test cases against a target and scores them.

    Args:
        target: Anything exposing ``.invoke()``.
        judge: Optional model for the ``llm_judge`` criterion. Never
            constructed implicitly -- an eval run should not start spending on
            API calls the caller did not ask for.
        include_output: Whether to record the agent's text in the report. On by
            default because it is what makes a failure diagnosable; turn it off
            when run files must not carry real data.
        principal: Default caller identity, used for cases that do not supply
            their own. Ignored by targets that do not accept one.

    Raises:
        TypeError: If the target has no ``.invoke()``, which is almost always
            a module path pointing at the wrong object.
    """

    def __init__(
        self,
        target: Any,
        *,
        judge: Any = None,
        include_output: bool = True,
        principal: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise the runner. See the class docstring for arguments."""
        if not callable(getattr(target, "invoke", None)):
            raise TypeError(
                f"{type(target).__name__} has no .invoke() method, so it cannot be "
                f"evaluated. Pass an agent, a compiled graph, or any object with "
                f"an invoke() method."
            )
        self.target = target
        self.judge = judge
        self.include_output = include_output
        self.principal = principal

    def _invoke(self, case: EvalCase) -> Any:
        """Call the target for one case, passing a principal where accepted.

        Args:
            case: The case being run.

        Returns:
            The target's return value.
        """
        principal = case.principal if case.principal is not None else self.principal
        if principal is not None:
            try:
                return self.target.invoke(case.input, principal=principal)
            except TypeError:
                # The target does not accept a principal. Retrying without one
                # is right: a plain function should still be testable by a case
                # that happens to carry an identity.
                pass
        return self.target.invoke(case.input)

    def run_case(self, case: EvalCase) -> CaseResult:
        """Run one case and score it.

        Args:
            case: The case to run.

        Returns:
            Its result. A target that raises produces a failing result carrying
            the exception, rather than propagating it.
        """
        started = perf_counter()
        try:
            raw = self._invoke(case)
            error: str | None = None
        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000.0
            return CaseResult(
                id=case.id,
                passed=False,
                latency_ms=elapsed,
                error=f"{type(exc).__name__}: {exc}",
                tags=case.tags,
            )

        elapsed = (perf_counter() - started) * 1000.0
        text = _text_of(raw)
        outcome = Outcome(
            text=text,
            blocked=bool(raw.get("blocked")) if isinstance(raw, Mapping) else False,
            tool_calls=tuple(str(name) for name in (raw.get("tool_calls") or ()))
            if isinstance(raw, Mapping)
            else (),
            latency_ms=elapsed,
            cost=_cost_of(self.target, raw),
            error=error,
            raw=raw,
            judge=self.judge,
        )

        criteria = tuple(evaluate(case.criteria(), outcome))
        return CaseResult(
            id=case.id,
            passed=all(result.passed for result in criteria),
            criteria=criteria,
            latency_ms=elapsed,
            cost=outcome.cost,
            blocked=outcome.blocked,
            output=text if self.include_output else "",
            tags=case.tags,
        )

    def run(
        self,
        cases: Iterable[EvalCase],
        *,
        tags: Sequence[str] | None = None,
        on_result: Callable[[CaseResult], None] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvalReport:
        """Run a set of cases and return a report.

        Args:
            cases: The cases to run.
            tags: When given, only run cases carrying one of these labels.
            on_result: Called after each case, for progress reporting.
            metadata: Free-form context recorded on the report, such as a
                git commit.

        Returns:
            The report.

        Example:
            >>> from wardhook.evals import EvalCase
            >>> class Fixed:
            ...     def invoke(self, text):
            ...         return "the excess is 500"
            >>> cases = [
            ...     EvalCase("a", "?", {"contains": "500"}),
            ...     EvalCase("b", "?", {"contains": "1000"}),
            ... ]
            >>> report = EvalRunner(Fixed()).run(cases)
            >>> report.passed, report.failed
            (1, 1)
            >>> report.failures[0].summary()
            "contains: output is missing ['1000']"
        """
        wanted = set(tags) if tags else None
        started_at = _now_iso()
        started = perf_counter()

        results: list[CaseResult] = []
        for case in cases:
            if wanted is not None and not wanted & set(case.tags):
                continue
            result = self.run_case(case)
            results.append(result)
            if on_result is not None:
                on_result(result)

        return EvalReport(
            results=tuple(results),
            target=describe_target(self.target),
            started_at=started_at,
            duration_ms=(perf_counter() - started) * 1000.0,
            metadata=dict(metadata or {}),
        )

    def __repr__(self) -> str:
        """Return a debug representation naming the target."""
        return f"EvalRunner(target={describe_target(self.target)!r})"
