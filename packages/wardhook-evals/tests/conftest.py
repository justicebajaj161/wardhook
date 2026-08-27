"""Shared pytest fixtures for the wardhook-evals test suite."""

from __future__ import annotations

import pytest

from wardhook.evals import CaseResult, EvalCase, EvalReport


class ScriptedAgent:
    """A target with no framework behind it, which is the point.

    The runner must work against anything exposing ``.invoke()``, so the tests
    use the plainest possible object rather than a real agent.
    """

    name = "scripted-agent"

    def __init__(self, output="Storm damage carries a 500 excess.", **extra):
        self.output = output
        self.extra = extra
        self.calls: list[tuple] = []

    def invoke(self, text, principal=None):
        self.calls.append((text, principal))
        return {"output": self.output, "run_id": "r1", **self.extra}


@pytest.fixture
def agent() -> ScriptedAgent:
    """An agent returning one fixed answer."""
    return ScriptedAgent()


@pytest.fixture
def scripted():
    """Factory for agents with arbitrary response shapes."""
    return ScriptedAgent


@pytest.fixture
def cases() -> list[EvalCase]:
    """One passing case and one failing case."""
    return [
        EvalCase("excess-storm", "What excess applies to storm damage?", {"contains": ["500"]}),
        EvalCase("will-fail", "anything", {"contains": ["1000"]}),
    ]


@pytest.fixture
def make_report():
    """Build a report from ``{case_id: passed}``."""

    def build(**passed: bool) -> EvalReport:
        return EvalReport(
            results=tuple(CaseResult(id=name, passed=ok) for name, ok in passed.items())
        )

    return build
