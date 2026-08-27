"""Tests for the compliance audit trail.

The recurring assertion here is that no record ever contains the data it is
auditing. That is the property the whole module exists to guarantee, so it is
checked from several angles rather than once.
"""

from __future__ import annotations

import json

import pytest

from wardhook.guardrails.audit import AuditLogger, diff_text
from wardhook.guardrails.pii import PIIRedactor
from wardhook.guardrails.rbac import RoleBasedToolPolicy

SSN_TEXT = "The claimant SSN is 123-45-6789 and email is alice@example.com"


class TestTextDiff:
    def test_describes_a_change_without_either_version(self):
        diff = diff_text(
            "SSN 123-45-6789",
            "SSN [US_SSN]",
            [{"entity": "US_SSN", "start": 4, "end": 15, "length": 11}],
        )
        assert diff.changed is True
        assert diff.entities == {"US_SSN": 1}
        assert "123-45-6789" not in str(diff.to_dict())

    def test_reports_no_change_for_identical_text(self):
        assert diff_text("same", "same").changed is False

    def test_records_the_length_delta(self):
        assert diff_text("abcdef", "ab").to_dict()["delta"] == -4

    def test_counts_repeated_entities(self):
        diff = diff_text(
            "a@b.com c@d.com",
            "[EMAIL] [EMAIL]",
            [
                {"entity": "EMAIL", "start": 0, "end": 7, "length": 7},
                {"entity": "EMAIL", "start": 8, "end": 15, "length": 7},
            ],
        )
        assert diff.entities == {"EMAIL": 2}

    def test_tolerates_partial_span_records(self):
        assert diff_text("a", "b", [{"start": 0}]).entities == {"UNKNOWN": 1}


class TestAuditLogging:
    def test_writes_one_json_object_per_line(self, audit_log):
        audit_log.log(guardrail="pii", action="redact", stage="output", run_id="r1")
        audit_log.log(guardrail="rbac", action="block", stage="tool_call", run_id="r1")
        lines = audit_log.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["run_id"] == "r1" for line in lines)

    def test_a_record_carries_the_fields_a_reviewer_needs(self, audit_log):
        audit_log.log(
            guardrail="pii-redactor",
            action="block",
            stage="output",
            run_id="run-7",
            rule="API_KEY",
            severity="critical",
            reason="credential in response",
            principal_id="u-42",
        )
        record = next(audit_log.read())
        assert record["guardrail"] == "pii-redactor"
        assert record["action"] == "block"
        assert record["stage"] == "output"
        assert record["rule"] == "API_KEY"
        assert record["severity"] == "critical"
        assert record["principal_id"] == "u-42"
        assert record["timestamp"].endswith("+00:00")

    def test_allows_are_not_recorded_by_default(self, audit_log):
        # An allow is the common case; logging every one buries the actions a
        # reviewer is looking for.
        audit_log.log(guardrail="pii", action="allow", stage="input")
        assert list(audit_log.read()) == []

    def test_allows_can_be_recorded_when_positive_evidence_is_required(self, tmp_path):
        logger = AuditLogger(tmp_path / "a.jsonl", record_allows=True)
        logger.log(guardrail="pii", action="allow", stage="input")
        assert len(list(logger.read())) == 1

    def test_an_unrecorded_allow_is_still_returned_to_the_caller(self, audit_log):
        event = audit_log.log(guardrail="pii", action="allow", stage="input")
        assert event.action == "allow"

    def test_in_memory_mode_writes_no_file(self):
        logger = AuditLogger()
        logger.log(guardrail="pii", action="block", stage="input")
        assert logger.path is None
        assert len(logger.events) == 1
        with pytest.raises(ValueError, match="in-memory only"):
            list(logger.read())

    def test_memory_retention_is_bounded(self):
        logger = AuditLogger(max_memory_events=5)
        for i in range(20):
            logger.log(guardrail="g", action="block", stage="input", run_id=str(i))
        assert len(logger.events) == 5
        assert logger.events[-1].run_id == "19"

    def test_json_keys_are_sorted_so_logs_diff_cleanly(self, audit_log):
        event = audit_log.log(guardrail="z", action="block", stage="input")
        keys = list(json.loads(event.to_json()).keys())
        assert keys == sorted(keys)

    def test_a_corrupt_line_does_not_destroy_the_rest_of_the_trail(self, audit_log):
        audit_log.log(guardrail="a", action="block", stage="input")
        with audit_log.path.open("a", encoding="utf-8") as handle:
            handle.write("{not valid json\n")
        audit_log.log(guardrail="b", action="block", stage="input")
        assert [r["guardrail"] for r in audit_log.read()] == ["a", "b"]

    def test_reading_a_log_that_does_not_exist_yet_is_empty(self, tmp_path):
        assert list(AuditLogger(tmp_path / "never-written.jsonl").read()) == []

    def test_parent_directories_are_created(self, tmp_path):
        logger = AuditLogger(tmp_path / "nested" / "deep" / "audit.jsonl")
        logger.log(guardrail="g", action="block", stage="input")
        assert logger.path.exists()


class TestNoLeakage:
    def test_a_pii_redaction_record_never_contains_the_pii(self, audit_log):
        result = PIIRedactor().on_output(SSN_TEXT, {})
        audit_log.record(result, stage="output", run_id="r1", before=SSN_TEXT)
        raw = audit_log.path.read_text(encoding="utf-8")
        assert "123-45-6789" not in raw
        assert "alice@example.com" not in raw

    def test_but_it_does_record_what_was_removed_and_from_where(self, audit_log):
        result = PIIRedactor().on_output(SSN_TEXT, {})
        event = audit_log.record(result, stage="output", run_id="r1", before=SSN_TEXT)
        assert event.diff["entities"] == {"US_SSN": 1, "EMAIL": 1}
        assert event.diff["changed"] is True
        assert all("start" in span and "length" in span for span in event.diff["spans"])

    def test_an_rbac_denial_record_carries_no_argument_values(self, audit_log, ctx):
        policy = RoleBasedToolPolicy({"agent": ["lookup_*"]})
        result = policy.on_tool_call(
            "issue_refund", {"iban": "GB33BUKB20201555555555"}, ctx("agent")
        )
        audit_log.record(result, stage="tool_call", run_id="r1", tool="issue_refund")
        assert "GB33BUKB20201555555555" not in audit_log.path.read_text(encoding="utf-8")

    def test_fingerprints_correlate_without_revealing(self, audit_log):
        digest = audit_log.fingerprint("123-45-6789")
        assert digest != "123-45-6789"
        assert len(digest) == 16
        assert audit_log.fingerprint("123-45-6789") == digest
        assert audit_log.fingerprint("123-45-6780") != digest

    def test_the_salt_makes_fingerprints_session_scoped(self):
        # Two loggers with different (default, random) salts must not produce
        # matching digests, or a fingerprint would be a portable identifier.
        assert AuditLogger().fingerprint("x") != AuditLogger().fingerprint("x")

    def test_a_stable_salt_enables_deliberate_cross_session_correlation(self, tmp_path):
        a = AuditLogger(tmp_path / "a.jsonl", fingerprint_salt="shared")
        b = AuditLogger(tmp_path / "b.jsonl", fingerprint_salt="shared")
        assert a.fingerprint("x") == b.fingerprint("x")


class TestRecordRun:
    def test_ingests_the_event_list_an_agent_returns(self, audit_log):
        # This is the bridge from wardhook-core's result dict, read
        # structurally so neither package imports the other.
        events = audit_log.record_run(
            [
                {
                    "guardrail": "pii-redactor",
                    "action": "redact",
                    "stage": "input",
                    "rule": "EMAIL",
                    "severity": "medium",
                    "reason": "redacted 1 occurrence",
                },
                {
                    "guardrail": "rbac-tool-policy",
                    "action": "block",
                    "stage": "tool_call",
                    "tool": "issue_refund",
                    "severity": "high",
                },
            ],
            run_id="run-9",
            principal_id="u-3",
        )
        assert [e.action for e in events] == ["redact", "block"]
        assert events[1].tool == "issue_refund"
        assert all(e.run_id == "run-9" and e.principal_id == "u-3" for e in events)

    def test_falls_back_to_the_stage_nested_in_details(self, audit_log):
        events = audit_log.record_run(
            [{"guardrail": "g", "action": "block", "details": {"stage": "output"}}],
            run_id="r1",
        )
        assert events[0].stage == "output"

    def test_an_empty_event_list_records_nothing(self, audit_log):
        assert audit_log.record_run([], run_id="r1") == []


class TestReport:
    def test_summarises_a_run_for_review(self, audit_log):
        redactor = PIIRedactor()
        for run in ("r1", "r2"):
            result = redactor.on_output(SSN_TEXT, {})
            audit_log.record(result, stage="output", run_id=run, before=SSN_TEXT)
        audit_log.log(guardrail="rbac", action="block", stage="tool_call", run_id="r1")

        report = audit_log.report()
        assert report["total_events"] == 3
        assert report["runs"] == 2
        assert report["by_action"] == {"block": 1, "redact": 2}
        assert report["by_stage"] == {"output": 2, "tool_call": 1}
        assert report["by_entity"] == {"EMAIL": 2, "US_SSN": 2}
        assert report["first_event"] <= report["last_event"]

    def test_an_empty_report_is_well_formed(self):
        report = AuditLogger().report()
        assert report["total_events"] == 0
        assert report["by_action"] == {}
        assert report["first_event"] is None

    def test_can_summarise_a_supplied_event_list(self, audit_log):
        audit_log.log(guardrail="a", action="block", stage="input")
        audit_log.log(guardrail="b", action="block", stage="input")
        assert audit_log.report(audit_log.events[:1])["total_events"] == 1
