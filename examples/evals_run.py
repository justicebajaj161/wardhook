"""Example: scoring an agent against test cases, and catching a regression.

Runs fully offline with no agent framework and no API key:

    python examples/evals_run.py

The point of this example is section 3. A real agent suite is never fully
green -- there are always cases nobody has had time to fix -- and a gate that
blocks on "any failure" blocks every build forever, so it gets switched off.
Comparing against a baseline separates "this was already broken" from "your
change broke this", and only fails on the second.
"""

from __future__ import annotations

from pathlib import Path

from wardhook.evals import EvalRunner, compare, load_cases

CASES = Path(__file__).parent / "data" / "claims_cases.jsonl"

POLICY_ANSWERS = {
    "storm": "Storm damage carries a 500 excess, and cover applies only where wind "
    "speeds exceeded 55 mph.",
    "flood": "Flood damage carries a 1000 excess and requires a loss adjuster to "
    "attend before settlement.",
    "condition": "Storm cover applies only where wind speeds exceeded 55 mph.",
    "subsidence": "I cannot find subsidence in this policy wording, so I cannot "
    "confirm whether it is covered.",
}


class ClaimsAgent:
    """A stand-in for a real agent, with no framework behind it.

    The runner targets anything exposing ``.invoke()``. Using the plainest
    possible object here is the point: an eval suite should outlive a rewrite
    of the agent it tests.

    Args:
        conflate_flood: Simulate the regression in section 3, where a prompt
            change makes the agent answer flood questions with storm figures.
    """

    name = "claims-assistant"

    def __init__(self, conflate_flood: bool = False) -> None:
        """Initialise the agent. See the class docstring for arguments."""
        self.conflate_flood = conflate_flood

    def invoke(self, question: str, principal: dict | None = None) -> dict:
        """Answer a question about the policy.

        Args:
            question: The customer's question.
            principal: The caller's identity. Refunds need a supervisor.

        Returns:
            A result dict in the shape ``wardhook-core`` returns.
        """
        lowered = question.lower()
        roles = (principal or {}).get("roles", [])

        if "refund" in lowered:
            if "supervisor" not in roles:
                return {"output": "I cannot issue refunds.", "blocked": True, "tool_calls": []}
            return {"output": "Refund issued.", "blocked": False, "tool_calls": ["issue_refund"]}

        if "flood" in lowered:
            key = "storm" if self.conflate_flood else "flood"
        elif "subsidence" in lowered:
            key = "subsidence"
        elif "condition" in lowered:
            key = "condition"
        else:
            key = "storm"

        return {
            "output": POLICY_ANSWERS[key],
            "blocked": False,
            "tool_calls": ["lookup_policy"],
            "run_id": "example",
        }


def section(title: str) -> None:
    """Print a section heading.

    Args:
        title: The heading text.
    """
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show(report) -> None:
    """Print a report as the CLI would.

    Args:
        report: The report to print.
    """
    for result in report.results:
        mark = "PASS" if result.passed else "FAIL"
        line = f"  {mark}  {result.id:<24}"
        if not result.passed:
            line += f"-- {result.summary()}"
        print(line)
    print(f"\n  {report.passed}/{report.total} passed ({report.pass_rate:.0%})")


def main() -> int:
    """Run every demonstration.

    Returns:
        A process exit code. Zero, because the regression in section 3 is
        deliberate -- this example demonstrates detection, it is not a test.
    """
    cases = load_cases(CASES)

    section("1. The case file")
    print(f"  {CASES.relative_to(Path.cwd()) if CASES.is_relative_to(Path.cwd()) else CASES}")
    for case in cases:
        print(f"    {case.id:<24} {', '.join(case.expect) or 'no criteria'}")
    print("\n  JSONL, one case per line, so a new case is a one-line diff and a")
    print("  changed expectation is a one-line diff.")

    section("2. Scoring the current agent")
    baseline_report = EvalRunner(ClaimsAgent()).run(cases)
    show(baseline_report)
    print("\n  `no-invention` is the interesting one: it asserts the agent does NOT")
    print("  claim cover the policy never mentions.")

    section("3. A prompt change ships, and something breaks")
    broken_report = EvalRunner(ClaimsAgent(conflate_flood=True)).run(cases)
    show(broken_report)

    comparison = compare(broken_report, baseline_report)
    print(f"\n  Against the baseline: {comparison.summary()}")
    for item in comparison.regressed:
        print(f"    REGRESSED  {item.id}  -- {item.detail}")

    print("\n  In CI this is the whole point:")
    print("    wardhook-eval run cases.jsonl --target myapp:agent -o run.json")
    print("    wardhook-eval compare run.json --baseline baseline.json   # exit 1")

    section("4. Why 'still failing' is not 'regressed'")
    debt = EvalRunner(ClaimsAgent(conflate_flood=True)).run(cases)
    unchanged = compare(debt, broken_report)
    print(f"  Comparing the broken run against a broken baseline: {unchanged.summary()}")
    print(f"  has_regressions: {unchanged.has_regressions}")
    print("\n  Known debt does not fail the build. Only a case that used to pass")
    print("  and now does not. That is what keeps the gate switched on.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
