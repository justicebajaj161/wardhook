"""Pluggable pass/fail criteria, and the normalised view they are given.

A criterion is a function of two arguments -- what the case expected, and what
the agent actually did -- returning a :class:`CriterionResult`. They live in a
registry so a project can add its own with :func:`register_criterion` without
subclassing anything or forking this package.

**Why a normalised :class:`Outcome`.** Criteria must work against a Wardhook
``AgentGraph``, a bare LangGraph graph, and a plain function, which return
wildly different shapes. The runner flattens whatever came back into one small
record, so a criterion never has to know what produced it. The raw response is
kept on :attr:`Outcome.raw` for :func:`json_path` and for criteria of your own.

**Text matching is case-insensitive by default.** ``contains`` and
``not_contains`` fold case, because asserting that a model said "excess" should
not fail when it said "Excess" -- that is a test that breaks for no reason.
Where case matters, use ``regex``, which does not fold.

Example:
    >>> outcome = Outcome(text="Storm damage carries a 500 excess.")
    >>> evaluate({"contains": ["500", "excess"]}, outcome)[0].passed
    True
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CRITERIA",
    "CriterionResult",
    "Outcome",
    "UnknownCriterionError",
    "evaluate",
    "get_criterion",
    "register_criterion",
]


class UnknownCriterionError(KeyError):
    """A case named a criterion that is not registered."""


@dataclass(frozen=True)
class CriterionResult:
    """The verdict of one criterion.

    Attributes:
        name: The criterion that ran.
        passed: Whether it was satisfied.
        detail: Human-readable explanation. Populated on failure so a report
            says *why*, not just that something failed.
    """

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        record: dict[str, Any] = {"name": self.name, "passed": self.passed}
        if self.detail:
            record["detail"] = self.detail
        return record


@dataclass
class Outcome:
    """What the agent actually did, flattened into one shape.

    Attributes:
        text: The agent's final text output.
        blocked: Whether a guardrail stopped the run.
        tool_calls: Names of tools the agent invoked.
        latency_ms: Wall time for the invocation.
        cost: Estimated cost in US dollars, when telemetry supplied one.
        error: Description of the exception the target raised, if it did.
        raw: The target's unmodified return value.
        judge: A model for the ``llm_judge`` criterion, supplied by the runner.
    """

    text: str = ""
    blocked: bool = False
    tool_calls: tuple[str, ...] = ()
    latency_ms: float = 0.0
    cost: float | None = None
    error: str | None = None
    raw: Any = None
    judge: Any = field(default=None, repr=False)


Criterion = Callable[[Any, Outcome], CriterionResult]

CRITERIA: dict[str, Criterion] = {}


def register_criterion(name: str, criterion: Criterion) -> None:
    """Add or replace a criterion in the registry.

    Args:
        name: The key cases will use inside ``expect``.
        criterion: A callable taking ``(expected, outcome)`` and returning a
            :class:`CriterionResult`.

    Example:
        >>> def shouty(expected, outcome):
        ...     return CriterionResult("shouty", outcome.text.isupper())
        >>> register_criterion("shouty", shouty)
        >>> evaluate({"shouty": True}, Outcome(text="HELLO"))[0].passed
        True
    """
    CRITERIA[name] = criterion


def get_criterion(name: str) -> Criterion:
    """Look up a criterion by name.

    Args:
        name: The criterion key.

    Returns:
        The registered callable.

    Raises:
        UnknownCriterionError: If nothing is registered under that name. The
            message lists what is available, because the usual cause is a typo
            in a case file.
    """
    try:
        return CRITERIA[name]
    except KeyError as exc:
        raise UnknownCriterionError(
            f"Unknown criterion {name!r}. Available: {', '.join(sorted(CRITERIA))}. "
            f"Register your own with wardhook.evals.register_criterion()."
        ) from exc


def evaluate(expect: Mapping[str, Any], outcome: Outcome) -> list[CriterionResult]:
    """Run every criterion a case asked for.

    Args:
        expect: The case's ``expect`` mapping.
        outcome: What the agent did.

    Returns:
        One result per criterion, in the order the case declared them.

    Raises:
        UnknownCriterionError: If the case names an unregistered criterion.
    """
    return [get_criterion(name)(expected, outcome) for name, expected in expect.items()]


def _as_list(value: Any) -> list[str]:
    """Coerce a string or sequence of strings to a list of strings.

    Args:
        value: A single value or a sequence of them.

    Returns:
        A list of strings. A bare string becomes a one-element list, so
        ``"contains": "500"`` and ``"contains": ["500"]`` behave identically.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _contains(expected: Any, outcome: Outcome) -> CriterionResult:
    """Every listed substring must appear in the output, ignoring case."""
    haystack = outcome.text.casefold()
    missing = [item for item in _as_list(expected) if item.casefold() not in haystack]
    return CriterionResult(
        "contains",
        not missing,
        "" if not missing else f"output is missing {missing}",
    )


def _not_contains(expected: Any, outcome: Outcome) -> CriterionResult:
    """No listed substring may appear in the output, ignoring case."""
    haystack = outcome.text.casefold()
    present = [item for item in _as_list(expected) if item.casefold() in haystack]
    return CriterionResult(
        "not_contains",
        not present,
        "" if not present else f"output unexpectedly contains {present}",
    )


def _regex(expected: Any, outcome: Outcome) -> CriterionResult:
    """Every listed pattern must match somewhere in the output. Case-sensitive."""
    unmatched: list[str] = []
    for pattern in _as_list(expected):
        try:
            if re.search(pattern, outcome.text) is None:
                unmatched.append(pattern)
        except re.error as exc:
            return CriterionResult("regex", False, f"invalid pattern {pattern!r}: {exc}")
    return CriterionResult(
        "regex",
        not unmatched,
        "" if not unmatched else f"no match for {unmatched}",
    )


def _equals(expected: Any, outcome: Outcome) -> CriterionResult:
    """The output must equal this exactly, ignoring surrounding whitespace."""
    actual = outcome.text.strip()
    wanted = str(expected).strip()
    return CriterionResult(
        "equals",
        actual == wanted,
        "" if actual == wanted else f"expected {wanted!r}, got {actual!r}",
    )


def _resolve_path(raw: Any, path: str) -> tuple[bool, Any]:
    """Walk a dotted path into a nested structure.

    Args:
        raw: The structure to walk, typically the target's return value.
        path: A dotted path such as ``citations.0.source``. Numeric segments
            index sequences.

    Returns:
        A ``(found, value)`` pair. ``found`` is ``False`` when the path does
        not exist, which is distinct from a path that exists and holds ``None``.
    """
    current = raw
    for segment in path.split("."):
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
        elif segment.lstrip("-").isdigit() and isinstance(current, Sequence):
            index = int(segment)
            if not -len(current) <= index < len(current):
                return False, None
            current = current[index]
        else:
            return False, None
    return True, current


def _json_path(expected: Any, outcome: Outcome) -> CriterionResult:
    """Each dotted path in the raw response must hold the given value."""
    if not isinstance(expected, Mapping):
        return CriterionResult(
            "json_path", False, "expected an object mapping dotted paths to values"
        )
    failures: list[str] = []
    for path, wanted in expected.items():
        found, actual = _resolve_path(outcome.raw, str(path))
        if not found:
            failures.append(f"{path} is absent")
        elif actual != wanted:
            failures.append(f"{path} is {actual!r}, expected {wanted!r}")
    return CriterionResult("json_path", not failures, "; ".join(failures))


def _tool_called(expected: Any, outcome: Outcome) -> CriterionResult:
    """Every named tool must have been invoked during the run."""
    called = {name.casefold() for name in outcome.tool_calls}
    missing = [name for name in _as_list(expected) if name.casefold() not in called]
    return CriterionResult(
        "tool_called",
        not missing,
        ""
        if not missing
        else f"expected {missing} to be called; actually called {list(outcome.tool_calls) or 'nothing'}",
    )


def _blocked(expected: Any, outcome: Outcome) -> CriterionResult:
    """Whether a guardrail was expected to stop the run."""
    wanted = bool(expected)
    return CriterionResult(
        "blocked",
        outcome.blocked == wanted,
        ""
        if outcome.blocked == wanted
        else f"expected blocked={wanted}, run was {'blocked' if outcome.blocked else 'allowed'}",
    )


def _max_latency_ms(expected: Any, outcome: Outcome) -> CriterionResult:
    """The run must complete within this many milliseconds."""
    limit = float(expected)
    ok = outcome.latency_ms <= limit
    return CriterionResult(
        "max_latency_ms",
        ok,
        "" if ok else f"took {outcome.latency_ms:.0f}ms, limit is {limit:.0f}ms",
    )


def _max_cost_usd(expected: Any, outcome: Outcome) -> CriterionResult:
    """The run must cost no more than this, when a cost is known.

    A run with no cost information passes: an unmeasured cost is not a failed
    budget, and failing here would make this criterion untestable on any target
    without telemetry attached.
    """
    limit = float(expected)
    if outcome.cost is None:
        return CriterionResult("max_cost_usd", True, "no cost reported; nothing to check")
    ok = outcome.cost <= limit
    return CriterionResult(
        "max_cost_usd",
        ok,
        "" if ok else f"cost ${outcome.cost:.5f}, limit is ${limit:.5f}",
    )


_JUDGE_PROMPT = """You are grading one response from an AI agent against a rubric.

Rubric:
{rubric}

Response:
{response}

Reply with exactly one word on the first line, PASS or FAIL, then one sentence
explaining why."""


def _llm_judge(expected: Any, outcome: Outcome) -> CriterionResult:
    """Grade the output against a rubric using a model.

    The rubric may be a plain string, or an object with a ``rubric`` key. The
    model comes from the runner's ``judge`` argument -- this criterion never
    constructs one itself, because silently spending money on an API call the
    caller did not configure would be a bad surprise.

    The judge is duck-typed: anything with ``.invoke()`` works, so this package
    imports no model library even to run an LLM-graded criterion.
    """
    rubric = expected.get("rubric") if isinstance(expected, Mapping) else expected
    if not rubric:
        return CriterionResult("llm_judge", False, "no rubric supplied")

    if outcome.judge is None:
        return CriterionResult(
            "llm_judge",
            False,
            "no judge model configured. Pass EvalRunner(target, judge=model), where "
            "model is any object with an .invoke() method -- a LangChain chat model, "
            "or your own wrapper. The optional 'wardhook-evals[judge]' extra installs "
            "langchain-core if you need one.",
        )

    prompt = _JUDGE_PROMPT.format(rubric=rubric, response=outcome.text)
    try:
        reply = outcome.judge.invoke(prompt)
    except Exception as exc:
        return CriterionResult(
            "llm_judge", False, f"judge model failed: {type(exc).__name__}: {exc}"
        )

    verdict = str(getattr(reply, "content", reply)).strip()
    passed = verdict.upper().startswith("PASS")
    return CriterionResult("llm_judge", passed, "" if passed else verdict[:200])


for _name, _fn in (
    ("contains", _contains),
    ("not_contains", _not_contains),
    ("regex", _regex),
    ("equals", _equals),
    ("json_path", _json_path),
    ("tool_called", _tool_called),
    ("blocked", _blocked),
    ("max_latency_ms", _max_latency_ms),
    ("max_cost_usd", _max_cost_usd),
    ("llm_judge", _llm_judge),
):
    register_criterion(_name, _fn)
