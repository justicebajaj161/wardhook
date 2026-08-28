"""Every registered criterion, including the ones with awkward edges."""

from __future__ import annotations

import pytest

from wardhook.evals import (
    CRITERIA,
    CriterionResult,
    Outcome,
    UnknownCriterionError,
    evaluate,
    get_criterion,
    register_criterion,
)
from wardhook.evals.criteria import _as_list


def check(name, expected, outcome):
    """Run one criterion directly."""
    return get_criterion(name)(expected, outcome)


class TestTextCriteria:
    def test_contains_accepts_a_string_or_a_list(self):
        outcome = Outcome(text="the excess is 500")
        assert check("contains", "500", outcome).passed
        assert check("contains", ["500", "excess"], outcome).passed

    def test_contains_folds_case(self):
        # Asserting the model said "excess" should not fail when it said
        # "Excess" -- that is a test that breaks for no reason.
        assert check("contains", "EXCESS", Outcome(text="the Excess is 500")).passed

    def test_contains_names_what_was_missing(self):
        result = check("contains", ["500", "1000"], Outcome(text="the excess is 500"))
        assert not result.passed
        assert "1000" in result.detail and "500" not in result.detail

    def test_not_contains_is_the_mirror_image(self):
        outcome = Outcome(text="Reference CLM-100045")
        assert check("not_contains", "123-45-6789", outcome).passed
        result = check("not_contains", ["CLM-100045"], outcome)
        assert not result.passed
        assert "unexpectedly contains" in result.detail

    def test_regex_does_not_fold_case(self):
        # regex is the escape hatch for when case actually matters.
        assert check("regex", r"\bExcess\b", Outcome(text="the Excess is 500")).passed
        assert not check("regex", r"\bExcess\b", Outcome(text="the excess is 500")).passed

    def test_regex_reports_an_invalid_pattern_rather_than_raising(self):
        result = check("regex", "([unclosed", Outcome(text="x"))
        assert not result.passed
        assert "invalid pattern" in result.detail

    def test_equals_ignores_surrounding_whitespace(self):
        assert check("equals", "yes", Outcome(text="  yes\n")).passed
        result = check("equals", "yes", Outcome(text="no"))
        assert not result.passed
        assert "expected 'yes', got 'no'" in result.detail


class TestStructuralCriteria:
    def test_tool_called_checks_the_invoked_tools(self):
        outcome = Outcome(tool_calls=("lookup_policy", "search_faq"))
        assert check("tool_called", "lookup_policy", outcome).passed
        assert check("tool_called", ["lookup_policy", "search_faq"], outcome).passed

    def test_tool_called_reports_what_actually_ran(self):
        result = check("tool_called", "issue_refund", Outcome(tool_calls=("lookup_policy",)))
        assert not result.passed
        assert "issue_refund" in result.detail and "lookup_policy" in result.detail

    def test_tool_called_says_so_when_nothing_ran(self):
        result = check("tool_called", "anything", Outcome())
        assert "actually called nothing" in result.detail

    def test_blocked_checks_both_directions(self):
        assert check("blocked", True, Outcome(blocked=True)).passed
        assert check("blocked", False, Outcome(blocked=False)).passed
        result = check("blocked", True, Outcome(blocked=False))
        assert not result.passed
        assert "run was allowed" in result.detail

    def test_json_path_walks_nested_structures(self):
        raw = {"citations": [{"source": "policy.md", "score": 0.9}], "blocked": False}
        outcome = Outcome(raw=raw)
        assert check("json_path", {"citations.0.source": "policy.md"}, outcome).passed
        assert check("json_path", {"blocked": False}, outcome).passed

    def test_json_path_supports_negative_indices(self):
        outcome = Outcome(raw={"items": ["a", "b", "z"]})
        assert check("json_path", {"items.-1": "z"}, outcome).passed

    def test_json_path_distinguishes_absent_from_wrong(self):
        outcome = Outcome(raw={"a": 1})
        assert "is absent" in check("json_path", {"nope": 1}, outcome).detail
        assert "expected 2" in check("json_path", {"a": 2}, outcome).detail

    def test_json_path_out_of_range_index_is_absent_not_an_error(self):
        outcome = Outcome(raw={"items": ["a"]})
        assert "is absent" in check("json_path", {"items.9": "x"}, outcome).detail

    def test_json_path_requires_an_object(self):
        assert not check("json_path", ["a.b"], Outcome(raw={})).passed


class TestBudgetCriteria:
    def test_max_latency_ms(self):
        assert check("max_latency_ms", 500, Outcome(latency_ms=120.0)).passed
        result = check("max_latency_ms", 100, Outcome(latency_ms=250.0))
        assert not result.passed
        assert "250ms" in result.detail

    def test_max_cost_usd(self):
        assert check("max_cost_usd", 0.01, Outcome(cost=0.005)).passed
        result = check("max_cost_usd", 0.001, Outcome(cost=0.02))
        assert not result.passed
        assert "limit is $0.00100" in result.detail

    def test_an_unmeasured_cost_passes(self):
        # An unmeasured cost is not a blown budget, and failing here would make
        # this criterion untestable against any target without telemetry.
        result = check("max_cost_usd", 0.01, Outcome(cost=None))
        assert result.passed
        assert "no cost reported" in result.detail


class TestLlmJudge:
    def test_without_a_judge_it_fails_with_install_instructions(self):
        result = check("llm_judge", "Is it polite?", Outcome(text="hi"))
        assert not result.passed
        assert "EvalRunner(target, judge=model)" in result.detail
        assert ".invoke() method" in result.detail

    def test_a_passing_verdict(self):
        class Judge:
            def invoke(self, prompt):
                assert "Is it polite?" in prompt
                return "PASS - it is courteous."

        assert check("llm_judge", "Is it polite?", Outcome(text="hi", judge=Judge())).passed

    def test_a_failing_verdict_carries_the_reason(self):
        class Judge:
            def invoke(self, prompt):
                return "FAIL - it was curt."

        result = check("llm_judge", "Is it polite?", Outcome(text="no", judge=Judge()))
        assert not result.passed
        assert "curt" in result.detail

    def test_a_message_object_reply_is_understood(self):
        class Reply:
            content = "PASS"

        class Judge:
            def invoke(self, prompt):
                return Reply()

        assert check("llm_judge", "rubric", Outcome(judge=Judge())).passed

    def test_a_rubric_may_be_an_object(self):
        class Judge:
            def invoke(self, prompt):
                return "PASS"

        assert check("llm_judge", {"rubric": "Is it polite?"}, Outcome(judge=Judge())).passed

    def test_an_empty_rubric_is_rejected(self):
        assert "no rubric" in check("llm_judge", "", Outcome()).detail

    def test_a_judge_that_raises_fails_the_case_rather_than_the_run(self):
        class Judge:
            def invoke(self, prompt):
                raise RuntimeError("rate limited")

        result = check("llm_judge", "rubric", Outcome(judge=Judge()))
        assert not result.passed
        assert "rate limited" in result.detail


class TestRegistry:
    def test_evaluate_runs_every_criterion_in_order(self):
        results = evaluate(
            {"contains": "500", "not_contains": "1000"}, Outcome(text="the excess is 500")
        )
        assert [r.name for r in results] == ["contains", "not_contains"]
        assert all(r.passed for r in results)

    def test_an_unknown_criterion_lists_the_available_ones(self):
        with pytest.raises(UnknownCriterionError, match="Unknown criterion 'contins'"):
            evaluate({"contins": "x"}, Outcome())
        with pytest.raises(UnknownCriterionError, match="max_cost_usd"):
            evaluate({"contins": "x"}, Outcome())

    def test_a_custom_criterion_can_be_registered(self):
        def shouty(expected, outcome):
            return CriterionResult("shouty", outcome.text.isupper() == bool(expected))

        try:
            register_criterion("shouty", shouty)
            assert evaluate({"shouty": True}, Outcome(text="HELLO"))[0].passed
            assert not evaluate({"shouty": True}, Outcome(text="hello"))[0].passed
        finally:
            CRITERIA.pop("shouty", None)

    def test_every_documented_criterion_is_registered(self):
        assert set(CRITERIA) >= {
            "contains",
            "not_contains",
            "regex",
            "equals",
            "json_path",
            "tool_called",
            "blocked",
            "max_latency_ms",
            "max_cost_usd",
            "llm_judge",
        }

    def test_criterion_results_serialise_without_empty_detail(self):
        assert CriterionResult("contains", True).to_dict() == {"name": "contains", "passed": True}


class TestValueCoercion:
    def test_a_bare_string_behaves_like_a_one_element_list(self):
        assert _as_list("500") == ["500"]

    def test_a_sequence_is_stringified_element_by_element(self):
        assert _as_list(["500", 600]) == ["500", "600"]

    def test_a_non_sequence_scalar_becomes_a_one_element_list(self):
        # A case file written by hand can carry `"contains": 500` without
        # quotes. Coercing beats failing on a JSON author's slip.
        assert _as_list(500) == ["500"]
        assert _as_list(True) == ["True"]
