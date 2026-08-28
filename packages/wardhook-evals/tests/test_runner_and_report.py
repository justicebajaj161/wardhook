"""The runner works against any shape of target and never aborts a run."""

from __future__ import annotations

from typing import ClassVar

import pytest

from wardhook.evals import CaseResult, EvalCase, EvalReport, EvalRunner, describe_target
from wardhook.evals.runner import _text_of


class TestTargets:
    def test_a_plain_object_with_invoke_is_enough(self):
        class Echo:
            def invoke(self, text):
                return {"output": f"you said {text}"}

        report = EvalRunner(Echo()).run([EvalCase("a", "hi", {"contains": "you said hi"})])
        assert report.passed == 1

    def test_a_target_returning_a_bare_string(self):
        class Plain:
            def invoke(self, text):
                return "the excess is 500"

        assert EvalRunner(Plain()).run([EvalCase("a", "?", {"contains": "500"})]).passed == 1

    def test_a_target_returning_a_message_object(self):
        class Message:
            content = "the excess is 500"

        class Model:
            def invoke(self, text):
                return Message()

        assert EvalRunner(Model()).run([EvalCase("a", "?", {"contains": "500"})]).passed == 1

    @pytest.mark.parametrize("key", ["output", "answer", "text", "response", "content", "result"])
    def test_common_response_keys_are_all_understood(self, key):
        class Shaped:
            def invoke(self, text):
                return {key: "the excess is 500"}

        assert EvalRunner(Shaped()).run([EvalCase("a", "?", {"contains": "500"})]).passed == 1

    def test_a_target_without_invoke_is_rejected_immediately(self):
        with pytest.raises(TypeError, match=r"has no \.invoke"):
            EvalRunner(object())

    def test_describe_target_prefers_a_name(self, agent):
        assert describe_target(agent) == "scripted-agent"
        assert describe_target(object()) == "object"


class TestRunning:
    def test_a_mixed_run_reports_both_outcomes(self, agent, cases):
        report = EvalRunner(agent).run(cases)
        assert (report.passed, report.failed, report.total) == (1, 1, 2)
        assert report.pass_rate == 0.5
        assert not report.ok
        assert [r.id for r in report.failures] == ["will-fail"]

    def test_a_raising_target_fails_its_case_without_aborting_the_run(self):
        class Flaky:
            def __init__(self):
                self.calls = 0

            def invoke(self, text):
                self.calls += 1
                if text == "boom":
                    raise ValueError("provider exploded")
                return {"output": "fine"}

        target = Flaky()
        report = EvalRunner(target).run(
            [
                EvalCase("ok-1", "a", {"contains": "fine"}),
                EvalCase("bad", "boom", {"contains": "fine"}),
                EvalCase("ok-2", "c", {"contains": "fine"}),
            ]
        )

        # The crucial property: the third case still ran.
        assert target.calls == 3
        assert report.passed == 2
        failure = report.by_id()["bad"]
        assert failure.error == "ValueError: provider exploded"
        assert failure.summary() == "ValueError: provider exploded"

    def test_a_case_with_no_criteria_passes_if_the_target_survives(self, agent):
        assert EvalRunner(agent).run([EvalCase("a", "x", {})]).passed == 1

    def test_the_principal_is_passed_through_when_accepted(self, agent):
        EvalRunner(agent).run([EvalCase("a", "x", {}, principal={"roles": ["agent"]})])
        assert agent.calls == [("x", {"roles": ["agent"]})]

    def test_a_target_that_rejects_a_principal_is_retried_without_one(self):
        class NoPrincipal:
            def invoke(self, text):
                return {"output": "ok"}

        report = EvalRunner(NoPrincipal()).run(
            [EvalCase("a", "x", {"contains": "ok"}, principal={"roles": ["agent"]})]
        )
        assert report.passed == 1

    def test_a_runner_level_principal_is_the_default(self, agent):
        EvalRunner(agent, principal={"roles": ["supervisor"]}).run([EvalCase("a", "x", {})])
        assert agent.calls == [("x", {"roles": ["supervisor"]})]

    def test_tags_filter_which_cases_run(self, agent):
        report = EvalRunner(agent).run(
            [
                EvalCase("smoke-1", "x", {}, tags=("smoke",)),
                EvalCase("slow-1", "x", {}, tags=("slow",)),
            ],
            tags=["smoke"],
        )
        assert [r.id for r in report.results] == ["smoke-1"]

    def test_on_result_is_called_per_case(self, agent, cases):
        seen: list[str] = []
        EvalRunner(agent).run(cases, on_result=lambda r: seen.append(r.id))
        assert seen == ["excess-storm", "will-fail"]

    def test_output_is_recorded_by_default_and_suppressible(self, agent, cases):
        assert "500" in EvalRunner(agent).run(cases).results[0].output
        assert EvalRunner(agent, include_output=False).run(cases).results[0].output == ""

    def test_blocked_and_tool_calls_are_extracted(self, scripted):
        target = scripted(blocked=True, tool_calls=["lookup_policy"])
        report = EvalRunner(target).run(
            [EvalCase("a", "x", {"blocked": True, "tool_called": "lookup_policy"})]
        )
        assert report.passed == 1
        assert report.results[0].blocked

    def test_repr_names_the_target(self, agent):
        assert repr(EvalRunner(agent)) == "EvalRunner(target='scripted-agent')"


class TestCostReadBack:
    def test_cost_is_read_from_the_targets_trace_when_it_has_one(self, agent):
        class Trace:
            total_cost = 0.0042

        agent.trace = lambda run_id=None: Trace()  # noqa: ARG005
        report = EvalRunner(agent).run([EvalCase("a", "x", {"max_cost_usd": 0.01})])
        assert report.results[0].cost == pytest.approx(0.0042)
        assert report.total_cost == pytest.approx(0.0042)
        assert report.passed == 1

    def test_a_budget_is_enforced_when_a_cost_is_known(self, agent):
        class Trace:
            total_cost = 0.5

        agent.trace = lambda run_id=None: Trace()  # noqa: ARG005
        report = EvalRunner(agent).run([EvalCase("a", "x", {"max_cost_usd": 0.01})])
        assert report.failed == 1
        assert "limit is" in report.results[0].summary()

    def test_a_target_without_telemetry_reports_no_cost(self, agent):
        assert EvalRunner(agent).run([EvalCase("a", "x", {})]).results[0].cost is None

    def test_a_trace_that_raises_does_not_fail_the_case(self, agent):
        def explode(run_id=None):
            raise RuntimeError("no such run")

        agent.trace = explode
        report = EvalRunner(agent).run([EvalCase("a", "x", {"contains": "500"})])
        assert report.passed == 1
        assert report.results[0].cost is None


class TestReport:
    def test_round_trips_through_json(self, tmp_path, agent, cases):
        report = EvalRunner(agent).run(cases)
        path = tmp_path / "nested" / "run.json"
        report.save(path)

        loaded = EvalReport.load(path)
        assert loaded.passed == report.passed
        assert loaded.by_id()["will-fail"].summary() == report.by_id()["will-fail"].summary()

    def test_the_summary_block_is_recomputed_not_trusted(self, tmp_path, agent, cases):
        # A hand-edited file must not be able to claim a pass rate its cases
        # do not support.
        path = tmp_path / "run.json"
        EvalRunner(agent).run(cases).save(path)
        text = path.read_text(encoding="utf-8").replace('"passed": 1', '"passed": 99', 1)
        path.write_text(text, encoding="utf-8")
        assert EvalReport.load(path).passed == 1

    def test_an_empty_report_is_not_ok(self):
        report = EvalReport()
        assert (report.total, report.passed, report.pass_rate, report.ok) == (0, 0, 0.0, False)

    def test_a_fully_passing_report_is_ok(self, make_report):
        assert make_report(a=True, b=True).ok

    def test_filter_tags_narrows_the_report(self):
        report = EvalReport(
            results=(
                CaseResult("a", True, tags=("smoke",)),
                CaseResult("b", True, tags=("slow",)),
            )
        )
        assert [r.id for r in report.filter_tags(["smoke"]).results] == ["a"]

    def test_a_missing_report_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No report at"):
            EvalReport.load(tmp_path / "absent.json")

    def test_a_corrupt_report_file_names_itself(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.json is not valid JSON"):
            EvalReport.load(path)

    def test_repr_summarises_the_run(self, agent, cases):
        assert repr(EvalRunner(agent).run(cases)) == (
            "EvalReport(passed=1/2, target='scripted-agent')"
        )


class TestCaseResultSerialisation:
    """Optional fields are omitted rather than nulled, for the same reason cases are."""

    def test_a_clean_pass_carries_only_the_core_keys(self):
        record = CaseResult(id="c1", passed=True, latency_ms=12.0).to_dict()
        assert set(record) == {"id", "passed", "criteria", "latency_ms"}

    def test_cost_blocked_output_error_and_tags_all_survive(self):
        record = CaseResult(
            id="c1",
            passed=False,
            latency_ms=12.0,
            cost=0.001234,
            blocked=True,
            output="the answer",
            error="RuntimeError: boom",
            tags=("smoke",),
        ).to_dict()

        assert record["cost"] == 0.001234
        assert record["blocked"] is True
        assert record["output"] == "the answer"
        assert record["error"] == "RuntimeError: boom"
        assert record["tags"] == ["smoke"]

    def test_a_zero_cost_is_still_reported(self):
        # 0.0 is a real measurement -- an unpriced model -- not a missing one.
        assert CaseResult(id="c1", passed=True, cost=0.0).to_dict()["cost"] == 0.0


class TestOutputCoercion:
    """A target can return anything. None of it may crash the runner."""

    def test_none_reads_as_empty_text(self):
        assert _text_of(None) == ""

    def test_a_mapping_without_a_known_key_is_stringified(self):
        assert "unexpected" in _text_of({"unexpected": "shape"})

    def test_a_nested_mapping_is_followed_through_the_known_key(self):
        assert _text_of({"output": {"output": "inner"}}) == "inner"

    def test_an_object_with_string_content_uses_it(self):
        class Message:
            content = "from content"

        assert _text_of(Message()) == "from content"

    def test_an_object_with_non_string_content_is_stringified(self):
        class Message:
            content: ClassVar = [{"type": "text", "text": "block"}]

        assert "block" in _text_of(Message())

    def test_anything_else_falls_back_to_str(self):
        assert _text_of(42) == "42"
