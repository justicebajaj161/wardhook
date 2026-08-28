"""The CLI's exit codes are the contract; everything else is presentation."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from wardhook.evals.cli import app

runner = CliRunner()

# Typer renders errors through rich: inside a box, wrapped across lines, and --
# on a CI runner -- with ANSI colour escapes embedded mid-sentence. Asserting on
# the raw output therefore fails for reasons that have nothing to do with
# whether the message is right. This flattens all three.
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def plain(result):
    """Return CLI output with colour, box drawing, and wrapping removed.

    Args:
        result: A ``CliRunner`` result.

    Returns:
        The visible text as one whitespace-normalised line.
    """
    text = _ANSI.sub("", result.output)
    for char in "\u2502\u256d\u256e\u2570\u256f\u2500":
        text = text.replace(char, " ")
    return " ".join(text.split())


AGENT_MODULE = """\
class Fixed:
    name = "demo-agent"

    def __init__(self, output):
        self.output = output

    def invoke(self, text, principal=None):
        return {"output": self.output, "tool_calls": ["lookup_policy"], "blocked": False}


good = Fixed("Storm damage carries a 500 excess.")
bad = Fixed("I do not know.")


def factory():
    return good


not_an_agent = 42
"""

CASES = (
    '{"id": "excess-storm", "input": "?", '
    '"expect": {"contains": ["500"], "tool_called": "lookup_policy"}}\n'
    '{"id": "polite", "input": "?", "expect": {"contains": ["excess"]}, "tags": ["smoke"]}\n'
)


def workspace(tmp_path):
    """Write an importable agent module and a case file, and return the dir."""
    (tmp_path / "agent_mod.py").write_text(AGENT_MODULE, encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(CASES, encoding="utf-8")
    return tmp_path


class TestRun:
    def test_a_passing_run_exits_zero(self, tmp_path, monkeypatch):
        path = workspace(tmp_path)
        monkeypatch.chdir(path)

        result = runner.invoke(app, ["run", "cases.jsonl", "--target", "agent_mod:good"])

        assert result.exit_code == 0, result.output
        assert "2/2 passed" in result.output

    def test_a_failing_run_exits_one(self, tmp_path, monkeypatch):
        path = workspace(tmp_path)
        monkeypatch.chdir(path)

        result = runner.invoke(app, ["run", "cases.jsonl", "--target", "agent_mod:bad"])

        assert result.exit_code == 1
        assert "0/2 passed" in result.output
        assert "output is missing" in plain(result)

    def test_the_report_is_written_when_asked(self, tmp_path, monkeypatch):
        path = workspace(tmp_path)
        monkeypatch.chdir(path)

        result = runner.invoke(
            app, ["run", "cases.jsonl", "--target", "agent_mod:good", "-o", "run.json"]
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((path / "run.json").read_text(encoding="utf-8"))
        assert payload["summary"]["passed"] == 2
        assert payload["results"][0]["output"]

    def test_no_output_omits_the_agent_text(self, tmp_path, monkeypatch):
        path = workspace(tmp_path)
        monkeypatch.chdir(path)

        runner.invoke(
            app,
            ["run", "cases.jsonl", "-t", "agent_mod:good", "-o", "run.json", "--no-output"],
        )

        payload = json.loads((path / "run.json").read_text(encoding="utf-8"))
        assert "output" not in payload["results"][0]

    def test_tags_filter_the_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(workspace(tmp_path))
        result = runner.invoke(
            app, ["run", "cases.jsonl", "-t", "agent_mod:good", "--tag", "smoke"]
        )
        assert "1/1 passed" in result.output

    def test_quiet_suppresses_per_case_lines(self, tmp_path, monkeypatch):
        monkeypatch.chdir(workspace(tmp_path))
        result = runner.invoke(app, ["run", "cases.jsonl", "-t", "agent_mod:good", "-q"])
        assert "PASS" not in result.output
        assert "2/2 passed" in result.output

    def test_a_factory_target_is_called(self, tmp_path, monkeypatch):
        monkeypatch.chdir(workspace(tmp_path))
        result = runner.invoke(app, ["run", "cases.jsonl", "-t", "agent_mod:factory"])
        assert result.exit_code == 0, result.output

    def test_run_can_compare_against_a_baseline_inline(self, tmp_path, monkeypatch):
        path = workspace(tmp_path)
        monkeypatch.chdir(path)
        runner.invoke(app, ["run", "cases.jsonl", "-t", "agent_mod:good", "-o", "base.json"])

        result = runner.invoke(
            app, ["run", "cases.jsonl", "-t", "agent_mod:bad", "--baseline", "base.json"]
        )

        assert result.exit_code == 1
        assert "REGRESSED" in plain(result)


class TestTargetResolution:
    def test_a_malformed_target_explains_the_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(workspace(tmp_path))
        result = runner.invoke(app, ["run", "cases.jsonl", "-t", "agent_mod"])
        assert result.exit_code != 0
        assert "module:attribute" in plain(result)

    def test_a_missing_module_says_which(self, tmp_path, monkeypatch):
        monkeypatch.chdir(workspace(tmp_path))
        result = runner.invoke(app, ["run", "cases.jsonl", "-t", "nope:thing"])
        assert "Could not import module 'nope'" in plain(result)

    def test_a_missing_attribute_lists_what_is_there(self, tmp_path, monkeypatch):
        monkeypatch.chdir(workspace(tmp_path))
        result = runner.invoke(app, ["run", "cases.jsonl", "-t", "agent_mod:absent"])
        assert "has no attribute 'absent'" in plain(result)
        assert "good" in plain(result)

    def test_an_object_without_invoke_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(workspace(tmp_path))
        result = runner.invoke(app, ["run", "cases.jsonl", "-t", "agent_mod:not_an_agent"])
        assert "has no .invoke() method" in plain(result)

    def test_a_malformed_case_file_reports_its_line(self, tmp_path, monkeypatch):
        path = workspace(tmp_path)
        (path / "broken.jsonl").write_text('{"id": "a", "input": "x"}\n{oops\n', encoding="utf-8")
        monkeypatch.chdir(path)

        result = runner.invoke(app, ["run", "broken.jsonl", "-t", "agent_mod:good"])
        assert "broken.jsonl:2" in plain(result)


class TestCompare:
    def _reports(self, tmp_path, baseline_passed, current_passed):
        from wardhook.evals import CaseResult, EvalReport

        EvalReport(results=(CaseResult("a", baseline_passed),)).save(tmp_path / "base.json")
        EvalReport(results=(CaseResult("a", current_passed),)).save(tmp_path / "run.json")
        return tmp_path / "run.json", tmp_path / "base.json"

    def test_a_regression_exits_one(self, tmp_path):
        current, baseline = self._reports(tmp_path, True, False)
        result = runner.invoke(app, ["compare", str(current), "--baseline", str(baseline)])

        assert result.exit_code == 1
        assert "REGRESSED" in result.output
        assert "1 regression(s)" in result.output

    def test_known_debt_does_not_fail_the_build(self, tmp_path):
        # Failing in both runs is not a regression. This is the behaviour the
        # whole module exists for.
        current, baseline = self._reports(tmp_path, False, False)
        result = runner.invoke(app, ["compare", str(current), "--baseline", str(baseline)])

        assert result.exit_code == 0
        assert "still_failing" in result.output
        assert "No regressions" in result.output

    def test_a_fix_is_reported(self, tmp_path):
        current, baseline = self._reports(tmp_path, False, True)
        result = runner.invoke(app, ["compare", str(current), "--baseline", str(baseline)])
        assert result.exit_code == 0
        assert "FIXED" in result.output

    def test_strict_fails_on_a_new_failing_case(self, tmp_path):
        from wardhook.evals import CaseResult, EvalReport

        EvalReport(results=(CaseResult("a", True),)).save(tmp_path / "base.json")
        EvalReport(results=(CaseResult("a", True), CaseResult("b", False))).save(
            tmp_path / "run.json"
        )

        args = ["compare", str(tmp_path / "run.json"), "-b", str(tmp_path / "base.json")]
        assert runner.invoke(app, args).exit_code == 0
        strict = runner.invoke(app, [*args, "--strict"])
        assert strict.exit_code == 1
        assert "NEW FAILURE" in strict.output

    def test_the_comparison_can_be_written_to_json(self, tmp_path):
        current, baseline = self._reports(tmp_path, True, False)
        out = tmp_path / "cmp" / "comparison.json"
        runner.invoke(app, ["compare", str(current), "-b", str(baseline), "-o", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["has_regressions"] is True

    def test_a_missing_baseline_says_so(self, tmp_path):
        current, _ = self._reports(tmp_path, True, True)
        result = runner.invoke(app, ["compare", str(current), "-b", str(tmp_path / "absent.json")])
        assert result.exit_code != 0
        assert "baseline" in plain(result)


class TestValidate:
    def test_a_good_file_reports_its_criteria_and_tags(self, tmp_path):
        (tmp_path / "cases.jsonl").write_text(CASES, encoding="utf-8")
        result = runner.invoke(app, ["validate", str(tmp_path / "cases.jsonl")])

        assert result.exit_code == 0, result.output
        assert "2 case(s), OK" in result.output
        assert "contains" in result.output
        assert "smoke" in result.output

    def test_a_bad_file_fails(self, tmp_path):
        (tmp_path / "bad.jsonl").write_text('{"no_id": true}\n', encoding="utf-8")
        result = runner.invoke(app, ["validate", str(tmp_path / "bad.jsonl")])
        assert result.exit_code != 0
        assert "missing required key" in plain(result)


def test_the_console_script_entry_point_exists():
    from wardhook.evals.cli import main

    assert callable(main)


def test_no_arguments_shows_help():
    assert "Run agent test cases" in runner.invoke(app, []).output


class TestTargetResolutionFailures:
    def test_a_factory_that_raises_is_reported_rather_than_traced(self, tmp_path, monkeypatch):
        (tmp_path / "broken_mod.py").write_text(
            "def factory():\n    raise ValueError('missing config')\n", encoding="utf-8"
        )
        (tmp_path / "cases.jsonl").write_text(CASES, encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["run", "cases.jsonl", "--target", "broken_mod:factory"])

        assert result.exit_code != 0
        assert "missing config" in plain(result)


class TestBaselineComparison:
    def _report(self, tmp_path, monkeypatch, target, name):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / name
        runner.invoke(app, ["run", "cases.jsonl", "--target", target, "-o", str(path)])
        return path

    def test_a_run_matching_its_baseline_keeps_the_run_exit_code(self, tmp_path, monkeypatch):
        # No regression, so `run --baseline` must not invent a failure: the
        # exit code still reflects whether the cases themselves passed.
        path = workspace(tmp_path)
        baseline = self._report(path, monkeypatch, "agent_mod:good", "baseline.json")

        result = runner.invoke(
            app,
            ["run", "cases.jsonl", "--target", "agent_mod:good", "--baseline", str(baseline)],
        )

        assert result.exit_code == 0, plain(result)
        assert "Against baseline" in plain(result)
        assert "REGRESSED" not in plain(result)

    def test_a_regression_against_the_baseline_exits_one(self, tmp_path, monkeypatch):
        path = workspace(tmp_path)
        baseline = self._report(path, monkeypatch, "agent_mod:good", "baseline.json")

        result = runner.invoke(
            app,
            ["run", "cases.jsonl", "--target", "agent_mod:bad", "--baseline", str(baseline)],
        )

        assert result.exit_code == 1
        assert "REGRESSED" in plain(result)

    def test_compare_reports_a_case_that_left_the_suite(self, tmp_path, monkeypatch):
        # A shrinking suite must be visible; otherwise deleting a failing case
        # reads identically to fixing it.
        path = workspace(tmp_path)
        baseline = self._report(path, monkeypatch, "agent_mod:good", "baseline.json")

        (path / "cases.jsonl").write_text(CASES.splitlines(keepends=True)[0], encoding="utf-8")
        current = self._report(path, monkeypatch, "agent_mod:good", "current.json")

        result = runner.invoke(app, ["compare", str(current), "-b", str(baseline)])

        assert "REMOVED" in plain(result)
        assert "polite" in plain(result)


class TestValidateOutput:
    def test_a_suite_without_tags_still_lists_its_criteria(self, tmp_path, monkeypatch):
        (tmp_path / "untagged.jsonl").write_text(
            '{"id": "c1", "input": "?", "expect": {"contains": ["500"]}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["validate", "untagged.jsonl"])

        assert result.exit_code == 0, plain(result)
        assert "criteria used: contains" in plain(result)
        assert "tags:" not in plain(result)

    def test_a_suite_with_no_criteria_at_all_reports_neither_line(self, tmp_path, monkeypatch):
        (tmp_path / "bare.jsonl").write_text(
            '{"id": "c1", "input": "?", "expect": {}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["validate", "bare.jsonl"])

        assert result.exit_code == 0, plain(result)
        assert "1 case(s), OK" in plain(result)
        assert "criteria used" not in plain(result)


class TestConsoleEntryPoint:
    def test_main_delegates_to_the_typer_app(self, monkeypatch):
        from wardhook.evals import cli as cli_module

        called = []
        monkeypatch.setattr(cli_module, "app", lambda: called.append(True))
        cli_module.main()
        assert called == [True]
