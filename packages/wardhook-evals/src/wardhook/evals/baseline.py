"""Baseline comparison and regression detection.

The central idea of this module is that **"failing" and "newly failing" are
different facts and deserve different responses.** A suite with nine known
failures that stays at nine has not got worse; a suite that goes from zero to
one has. A gate that only reports a total pass count cannot tell those apart,
so it either blocks every build on pre-existing debt or blocks none of them.

Comparing a run against a saved baseline classifies every case into one of six
outcomes, and only :attr:`Change.REGRESSED` fails the build by default:

============== ======== ======= ==============================================
Change         Baseline Current Meaning
============== ======== ======= ==============================================
unchanged      pass     pass    Working, still working.
fixed          fail     pass    Someone repaired it. Update the baseline.
regressed      pass     fail    **Your change broke this.**
still_failing  fail     fail    Known debt. Not your fault, not fixed either.
added          absent   any     A new case. Not a regression by definition.
removed        any      absent  A case disappeared from the suite.
============== ======== ======= ==============================================

Example:
    >>> from wardhook.evals.report import CaseResult, EvalReport
    >>> before = EvalReport(results=(CaseResult("a", True), CaseResult("b", False)))
    >>> after = EvalReport(results=(CaseResult("a", False), CaseResult("b", False)))
    >>> comparison = compare(after, before)
    >>> comparison.has_regressions, [c.id for c in comparison.regressed]
    (True, ['a'])
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from wardhook.evals.report import EvalReport

__all__ = ["BaselineComparison", "CaseComparison", "Change", "compare", "compare_files"]


class Change(str, Enum):
    """How one case's result moved relative to the baseline.

    A :class:`str` enum, so a comparison serialises to plain JSON and can be
    matched against a bare string without importing this module.
    """

    UNCHANGED = "unchanged"
    FIXED = "fixed"
    REGRESSED = "regressed"
    STILL_FAILING = "still_failing"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True)
class CaseComparison:
    """How one case changed between two runs.

    Attributes:
        id: The case's identifier.
        change: Its classification.
        baseline_passed: Whether it passed in the baseline, or ``None`` if it
            was not in the baseline at all.
        current_passed: Whether it passed in the current run, or ``None`` if it
            was not run.
        detail: Why the current run failed, when it did.
    """

    id: str
    change: Change
    baseline_passed: bool | None = None
    current_passed: bool | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        record: dict[str, Any] = {"id": self.id, "change": self.change.value}
        if self.baseline_passed is not None:
            record["baseline_passed"] = self.baseline_passed
        if self.current_passed is not None:
            record["current_passed"] = self.current_passed
        if self.detail:
            record["detail"] = self.detail
        return record


@dataclass(frozen=True)
class BaselineComparison:
    """The result of comparing a run against a baseline.

    Attributes:
        comparisons: One entry per case seen in either run.
    """

    comparisons: tuple[CaseComparison, ...] = ()

    def _of(self, change: Change) -> list[CaseComparison]:
        """Return every comparison with the given classification."""
        return [item for item in self.comparisons if item.change is change]

    @property
    def regressed(self) -> list[CaseComparison]:
        """Cases that passed in the baseline and now fail."""
        return self._of(Change.REGRESSED)

    @property
    def fixed(self) -> list[CaseComparison]:
        """Cases that failed in the baseline and now pass."""
        return self._of(Change.FIXED)

    @property
    def still_failing(self) -> list[CaseComparison]:
        """Cases that failed before and fail now."""
        return self._of(Change.STILL_FAILING)

    @property
    def unchanged(self) -> list[CaseComparison]:
        """Cases that passed before and pass now."""
        return self._of(Change.UNCHANGED)

    @property
    def added(self) -> list[CaseComparison]:
        """Cases present in the current run but not the baseline."""
        return self._of(Change.ADDED)

    @property
    def removed(self) -> list[CaseComparison]:
        """Cases present in the baseline but not the current run."""
        return self._of(Change.REMOVED)

    @property
    def new_failures(self) -> list[CaseComparison]:
        """Added cases that fail.

        Not regressions -- nothing was broken, the case is simply new -- but
        worth surfacing, and what ``--strict`` fails the build on.
        """
        return [item for item in self.added if item.current_passed is False]

    @property
    def has_regressions(self) -> bool:
        """Whether anything that used to pass now fails."""
        return bool(self.regressed)

    def counts(self) -> dict[str, int]:
        """Return how many cases fell into each classification.

        Returns:
            A mapping from change name to count, including zeros, so a report
            has a stable shape.
        """
        return {change.value: len(self._of(change)) for change in Change}

    def summary(self) -> str:
        """Return a one-line summary of what moved.

        Returns:
            A compact description naming only the non-zero classifications.
        """
        parts = [f"{count} {name}" for name, count in self.counts().items() if count]
        return ", ".join(parts) or "nothing to compare"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the comparison."""
        return {
            "counts": self.counts(),
            "has_regressions": self.has_regressions,
            "cases": [item.to_dict() for item in self.comparisons],
        }


def _classify(baseline_passed: bool | None, current_passed: bool | None) -> Change:
    """Map a before/after pair to a classification.

    Args:
        baseline_passed: Whether the case passed in the baseline, or ``None``.
        current_passed: Whether it passed in the current run, or ``None``.

    Returns:
        The classification.
    """
    if baseline_passed is None:
        return Change.ADDED
    if current_passed is None:
        return Change.REMOVED
    if baseline_passed and current_passed:
        return Change.UNCHANGED
    if baseline_passed and not current_passed:
        return Change.REGRESSED
    if not baseline_passed and current_passed:
        return Change.FIXED
    return Change.STILL_FAILING


def compare(current: EvalReport, baseline: EvalReport) -> BaselineComparison:
    """Compare a run against a saved baseline.

    Cases are matched on :attr:`~wardhook.evals.case.EvalCase.id`, which is why
    ids must be stable: renaming one reads as a removal plus an addition, not
    as the same case changing.

    Args:
        current: The run just completed.
        baseline: The saved reference run.

    Returns:
        The classification of every case seen in either run, ordered by the
        current run first, then any baseline-only cases.
    """
    current_by_id = current.by_id()
    baseline_by_id = baseline.by_id()

    comparisons: list[CaseComparison] = []
    for case_id, result in current_by_id.items():
        before = baseline_by_id.get(case_id)
        comparisons.append(
            CaseComparison(
                id=case_id,
                change=_classify(None if before is None else before.passed, result.passed),
                baseline_passed=None if before is None else before.passed,
                current_passed=result.passed,
                detail=result.summary(),
            )
        )

    for case_id, before in baseline_by_id.items():
        if case_id not in current_by_id:
            comparisons.append(
                CaseComparison(
                    id=case_id,
                    # Classified rather than hardcoded, so _classify stays the
                    # single definition of what each transition means.
                    change=_classify(before.passed, None),
                    baseline_passed=before.passed,
                    current_passed=None,
                )
            )

    return BaselineComparison(comparisons=tuple(comparisons))


def compare_files(current: Any, baseline: Any) -> BaselineComparison:
    """Compare two report files by path.

    Args:
        current: Path to the run just completed.
        baseline: Path to the saved reference run.

    Returns:
        The comparison.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If either file is not valid JSON.
    """
    return compare(EvalReport.load(current), EvalReport.load(baseline))
