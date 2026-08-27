"""Regression detection distinguishes "broken" from "newly broken"."""

from __future__ import annotations

from wardhook.evals import CaseResult, Change, EvalReport, compare, compare_files


class TestClassification:
    def test_the_four_core_transitions(self, make_report):
        baseline = make_report(stayed=True, broke=True, fixed=False, still_broken=False)
        current = make_report(stayed=True, broke=False, fixed=True, still_broken=False)

        comparison = compare(current, baseline)
        by_id = {item.id: item.change for item in comparison.comparisons}

        assert by_id == {
            "stayed": Change.UNCHANGED,
            "broke": Change.REGRESSED,
            "fixed": Change.FIXED,
            "still_broken": Change.STILL_FAILING,
        }

    def test_only_a_regression_fails_the_build(self, make_report):
        # The whole point: pre-existing debt must not block every change.
        baseline = make_report(known_bad=False)
        current = make_report(known_bad=False)

        comparison = compare(current, baseline)
        assert comparison.still_failing
        assert not comparison.has_regressions

    def test_a_regression_is_detected(self, make_report):
        comparison = compare(make_report(a=False), make_report(a=True))
        assert comparison.has_regressions
        assert [item.id for item in comparison.regressed] == ["a"]

    def test_a_new_case_is_added_not_regressed(self, make_report):
        comparison = compare(make_report(a=True, b=False), make_report(a=True))
        assert [item.id for item in comparison.added] == ["b"]
        assert not comparison.has_regressions

    def test_a_new_failing_case_is_surfaced_separately(self, make_report):
        comparison = compare(make_report(a=True, b=False), make_report(a=True))
        assert [item.id for item in comparison.new_failures] == ["b"]

    def test_a_new_passing_case_is_not_a_new_failure(self, make_report):
        comparison = compare(make_report(a=True, b=True), make_report(a=True))
        assert comparison.added and not comparison.new_failures

    def test_a_dropped_case_is_removed(self, make_report):
        comparison = compare(make_report(a=True), make_report(a=True, b=True))
        assert [item.id for item in comparison.removed] == ["b"]
        assert comparison.removed[0].baseline_passed is True
        assert comparison.removed[0].current_passed is None

    def test_comparing_a_run_to_itself_finds_nothing_new(self, make_report):
        report = make_report(a=True, b=False)
        comparison = compare(report, report)
        assert not comparison.has_regressions
        assert not comparison.added and not comparison.removed


class TestReporting:
    def test_counts_cover_every_classification(self, make_report):
        counts = compare(make_report(a=False), make_report(a=True)).counts()
        assert set(counts) == {change.value for change in Change}
        assert counts["regressed"] == 1
        assert counts["unchanged"] == 0

    def test_the_summary_names_only_what_moved(self, make_report):
        comparison = compare(make_report(a=False, b=True), make_report(a=True, b=True))
        assert comparison.summary() == "1 unchanged, 1 regressed"

    def test_an_empty_comparison_says_so(self):
        assert compare(EvalReport(), EvalReport()).summary() == "nothing to compare"

    def test_the_failure_detail_is_carried_through(self):
        from wardhook.evals.criteria import CriterionResult

        current = EvalReport(
            results=(
                CaseResult(
                    "a", False, (CriterionResult("contains", False, "output is missing ['500']"),)
                ),
            )
        )
        comparison = compare(current, EvalReport(results=(CaseResult("a", True),)))
        assert "output is missing" in comparison.regressed[0].detail

    def test_the_comparison_serialises(self, make_report):
        payload = compare(make_report(a=False), make_report(a=True)).to_dict()
        assert payload["has_regressions"] is True
        assert payload["counts"]["regressed"] == 1
        assert payload["cases"][0]["change"] == "regressed"

    def test_change_compares_equal_to_its_string(self):
        # A str enum, so a consumer can match on "regressed" without importing.
        assert Change.REGRESSED == "regressed"


class TestFiles:
    def test_compare_files_reads_two_reports(self, tmp_path, make_report):
        make_report(a=True).save(tmp_path / "baseline.json")
        make_report(a=False).save(tmp_path / "run.json")

        comparison = compare_files(tmp_path / "run.json", tmp_path / "baseline.json")
        assert comparison.has_regressions
