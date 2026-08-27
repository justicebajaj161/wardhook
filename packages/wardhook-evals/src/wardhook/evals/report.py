"""Run results and their serialisation.

A report is the record of one evaluation run: every case, whether it passed,
which criteria failed and why. It is written to JSON so it can be compared
against a saved baseline later (see :mod:`wardhook.evals.baseline`) and kept as
a CI artifact.

.. warning::
   **A report contains the agent's output by default.** That is what makes a
   failure diagnosable without re-running it -- but it means a run file from an
   agent handling real data carries real data. Pass ``include_output=False`` to
   :class:`~wardhook.evals.runner.EvalRunner` (or ``--no-output`` on the CLI)
   where that matters. Baseline comparison only needs the case id and its
   pass/fail, so a redacted report is still a perfectly good baseline.

Example:
    >>> from wardhook.evals.criteria import CriterionResult
    >>> result = CaseResult("a", True, (CriterionResult("contains", True),))
    >>> report = EvalReport(results=(result,))
    >>> report.passed, report.total, report.ok
    (1, 1, True)
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wardhook.evals.criteria import CriterionResult

__all__ = ["CaseResult", "EvalReport"]


@dataclass(frozen=True)
class CaseResult:
    """The outcome of one test case.

    Attributes:
        id: The case's identifier.
        passed: Whether every criterion was satisfied. A case with no criteria
            passes as long as the target did not raise.
        criteria: One result per criterion the case declared.
        latency_ms: Wall time for the invocation.
        cost: Estimated cost in US dollars, when telemetry supplied one.
        blocked: Whether a guardrail stopped the run.
        output: The agent's text output, unless suppressed.
        error: The exception the target raised, if it did.
        tags: The case's labels, carried through for filtering a report.
    """

    id: str
    passed: bool
    criteria: tuple[CriterionResult, ...] = ()
    latency_ms: float = 0.0
    cost: float | None = None
    blocked: bool = False
    output: str = ""
    error: str | None = None
    tags: tuple[str, ...] = ()

    @property
    def failures(self) -> list[CriterionResult]:
        """The criteria that did not pass."""
        return [result for result in self.criteria if not result.passed]

    def summary(self) -> str:
        """Return a one-line explanation of why this case failed.

        Returns:
            A short reason, or an empty string when the case passed.
        """
        if self.passed:
            return ""
        if self.error:
            return self.error
        return "; ".join(f"{r.name}: {r.detail}" for r in self.failures if r.detail) or "failed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        record: dict[str, Any] = {
            "id": self.id,
            "passed": self.passed,
            "criteria": [result.to_dict() for result in self.criteria],
            "latency_ms": round(self.latency_ms, 3),
        }
        if self.cost is not None:
            record["cost"] = round(self.cost, 8)
        if self.blocked:
            record["blocked"] = True
        if self.output:
            record["output"] = self.output
        if self.error is not None:
            record["error"] = self.error
        if self.tags:
            record["tags"] = list(self.tags)
        return record

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaseResult:
        """Rebuild a case result from :meth:`to_dict` output.

        Args:
            data: The serialised result.

        Returns:
            The reconstructed result.
        """
        return cls(
            id=str(data["id"]),
            passed=bool(data["passed"]),
            criteria=tuple(
                CriterionResult(
                    name=str(item.get("name", "")),
                    passed=bool(item.get("passed", False)),
                    detail=str(item.get("detail", "")),
                )
                for item in data.get("criteria", ())
            ),
            latency_ms=float(data.get("latency_ms", 0.0)),
            cost=data.get("cost"),
            blocked=bool(data.get("blocked", False)),
            output=str(data.get("output", "")),
            error=data.get("error"),
            tags=tuple(data.get("tags", ())),
        )


@dataclass(frozen=True)
class EvalReport:
    """The result of running a set of cases against one target.

    Attributes:
        results: One entry per case, in the order they ran.
        target: Description of what was evaluated.
        started_at: Wall-clock ISO-8601 timestamp of when the run began.
        duration_ms: Total wall time for the run.
        metadata: Free-form context, such as a git commit.
    """

    results: tuple[CaseResult, ...] = ()
    target: str = ""
    started_at: str = ""
    duration_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """How many cases ran."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """How many cases passed."""
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        """How many cases failed."""
        return self.total - self.passed

    @property
    def ok(self) -> bool:
        """Whether every case passed. An empty run is not a pass."""
        return self.total > 0 and self.failed == 0

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed, between 0.0 and 1.0."""
        return self.passed / self.total if self.total else 0.0

    @property
    def total_cost(self) -> float:
        """Estimated cost of the whole run, in US dollars."""
        return sum(result.cost or 0.0 for result in self.results)

    @property
    def failures(self) -> list[CaseResult]:
        """The cases that did not pass."""
        return [result for result in self.results if not result.passed]

    def by_id(self) -> dict[str, CaseResult]:
        """Index the results by case id.

        Returns:
            A mapping from case id to result, as baseline comparison needs.
        """
        return {result.id: result for result in self.results}

    def filter_tags(self, tags: Sequence[str]) -> EvalReport:
        """Return a report holding only cases carrying one of these tags.

        Args:
            tags: Labels to keep.

        Returns:
            A new report; this one is unchanged.
        """
        wanted = set(tags)
        kept = tuple(r for r in self.results if wanted & set(r.tags))
        return EvalReport(
            results=kept,
            target=self.target,
            started_at=self.started_at,
            duration_ms=self.duration_ms,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the whole run."""
        record: dict[str, Any] = {
            "target": self.target,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "cost": round(self.total_cost, 8),
            },
            "results": [result.to_dict() for result in self.results],
        }
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return record

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalReport:
        """Rebuild a report from :meth:`to_dict` output.

        The ``summary`` block is recomputed from the results rather than read
        back, so a hand-edited file cannot claim a pass rate its cases do not
        support.

        Args:
            data: The serialised report.

        Returns:
            The reconstructed report.
        """
        return cls(
            results=tuple(CaseResult.from_dict(item) for item in data.get("results", ())),
            target=str(data.get("target", "")),
            started_at=str(data.get("started_at", "")),
            duration_ms=float(data.get("duration_ms", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        """Write this report to a JSON file.

        Args:
            path: Destination. Parent directories are created.
        """
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        """Read a report from a JSON file.

        Args:
            path: The file to read.

        Returns:
            The report.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid JSON, naming the file.
        """
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"No report at {resolved}")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{resolved} is not valid JSON: {exc.msg}") from exc
        return cls.from_dict(payload)

    def __repr__(self) -> str:
        """Return a debug representation summarising the run."""
        return f"EvalReport(passed={self.passed}/{self.total}, target={self.target!r})"
