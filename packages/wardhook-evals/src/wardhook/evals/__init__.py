"""Wardhook evals: JSONL test cases, a pass/fail runner, and regression detection.

Answers the question that stops teams shipping prompt changes: *did that break
anything?* Not by eyeballing a diff of model output, but by running a suite and
comparing it to a saved baseline.

This package has **no dependency on the rest of Wardhook**, and none on any
agent framework. Its only runtime dependency is its CLI library. The runner
targets anything exposing ``.invoke()`` -- a Wardhook agent, a raw LangGraph
graph, a LangChain chain, or a plain object of your own:

    >>> class Support:
    ...     def invoke(self, question):
    ...         return {"output": "Storm damage carries a 500 excess."}
    >>> cases = [
    ...     EvalCase(
    ...         "excess-storm",
    ...         "What excess applies to storm damage?",
    ...         {"contains": ["500", "excess"]},
    ...     ),
    ...     EvalCase(
    ...         "no-invention",
    ...         "What excess applies to fire?",
    ...         {"not_contains": ["fire is covered"]},
    ...     ),
    ... ]
    >>> report = EvalRunner(Support()).run(cases)
    >>> f"{report.passed}/{report.total} passed"
    '2/2 passed'

Regression detection compares two reports and distinguishes *failing* from
*newly failing*, because "this was already broken" and "your change broke this"
call for different responses:

    >>> from wardhook.evals.report import CaseResult, EvalReport
    >>> before = EvalReport(results=(CaseResult("a", True), CaseResult("b", False)))
    >>> after = EvalReport(results=(CaseResult("a", False), CaseResult("b", False)))
    >>> comparison = compare(after, before)
    >>> comparison.summary()
    '1 regressed, 1 still_failing'
    >>> comparison.has_regressions
    True
"""

from wardhook.evals.baseline import (
    BaselineComparison,
    CaseComparison,
    Change,
    compare,
    compare_files,
)
from wardhook.evals.case import (
    CaseFormatError,
    EvalCase,
    dump_cases,
    load_cases,
    loads_cases,
)
from wardhook.evals.criteria import (
    CRITERIA,
    CriterionResult,
    Outcome,
    UnknownCriterionError,
    evaluate,
    get_criterion,
    register_criterion,
)
from wardhook.evals.report import CaseResult, EvalReport
from wardhook.evals.runner import EvalRunner, describe_target

__version__ = "0.1.1"

__all__ = [
    "CRITERIA",
    "BaselineComparison",
    "CaseComparison",
    "CaseFormatError",
    "CaseResult",
    "Change",
    "CriterionResult",
    "EvalCase",
    "EvalReport",
    "EvalRunner",
    "Outcome",
    "UnknownCriterionError",
    "__version__",
    "compare",
    "compare_files",
    "describe_target",
    "dump_cases",
    "evaluate",
    "get_criterion",
    "load_cases",
    "loads_cases",
    "register_criterion",
]
