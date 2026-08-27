"""Tests for entity packs, PII detection, checksum validators, and redaction."""

from __future__ import annotations

import pytest

from wardhook.guardrails.base import Severity
from wardhook.guardrails.entities import (
    BUILTIN_PACKS,
    EntityPack,
    EntityRule,
    PackNotFoundError,
    get_pack,
    register_pack,
)
from wardhook.guardrails.pii import PIIDetector, PIIRedactor, validate


class TestEntityPacks:
    @pytest.mark.parametrize("name", BUILTIN_PACKS)
    def test_every_builtin_pack_loads_and_compiles(self, name):
        pack = get_pack(name)
        assert pack.name == name
        assert pack.rules
        for rule in pack.rules:
            rule.compiled()

    @pytest.mark.parametrize("name", ["insurance", "healthcare", "fintech"])
    def test_domain_packs_inherit_the_universal_entities(self, name):
        # A domain pack extends `default` rather than replacing it, so an email
        # address is still caught whichever pack a deployment picks.
        assert "EMAIL" in get_pack(name).entity_names()

    def test_domain_packs_add_their_own_entities(self):
        assert "POLICY_NUMBER" in get_pack("insurance").entity_names()
        assert "NHS_NUMBER" in get_pack("healthcare").entity_names()
        assert "IBAN" in get_pack("fintech").entity_names()

    def test_unknown_pack_lists_what_is_available(self):
        with pytest.raises(PackNotFoundError, match="Known packs"):
            get_pack("nonexistent-pack")

    def test_none_resolves_to_the_default_pack(self):
        assert get_pack().name == "default"

    def test_an_instance_passes_through(self):
        pack = EntityPack(name="x", rules=())
        assert get_pack(pack) is pack

    def test_a_bare_rule_list_becomes_an_ad_hoc_pack(self):
        rules = [EntityRule(entity="X", pattern=r"\bxx\b")]
        assert get_pack(rules).entity_names() == ["X"]

    def test_filter_narrows_without_mutating_the_original(self):
        base = get_pack("default")
        narrowed = base.filter(include=["EMAIL"])
        assert narrowed.entity_names() == ["EMAIL"]
        assert len(base.rules) > 1

    def test_exclude_drops_entities(self):
        assert "EMAIL" not in get_pack("default").filter(exclude=["EMAIL"]).entity_names()

    def test_merge_lets_the_later_pack_win(self):
        a = EntityPack("a", (EntityRule(entity="X", pattern="a", severity=Severity.LOW),))
        b = EntityPack("b", (EntityRule(entity="X", pattern="b", severity=Severity.HIGH),))
        merged = a.merge(b)
        assert len(merged.rules) == 1
        assert merged.rules[0].severity is Severity.HIGH

    def test_invalid_regex_is_caught_at_construction(self):
        # Catching this when the pack loads beats catching it on whichever
        # request first happens to reach the rule.
        with pytest.raises(ValueError, match="invalid regex"):
            EntityPack("bad", (EntityRule(entity="X", pattern="[unclosed"),))

    def test_rule_missing_a_required_key_is_rejected(self):
        with pytest.raises(ValueError, match="missing required key"):
            EntityRule.from_dict({"entity": "X"})

    def test_unknown_severity_is_rejected(self):
        with pytest.raises(ValueError, match="unknown severity"):
            EntityRule.from_dict({"entity": "X", "pattern": "x", "severity": "apocalyptic"})

    def test_pack_missing_a_name_is_rejected(self):
        with pytest.raises(ValueError, match="missing a 'name'"):
            EntityPack.from_dict({"rules": []})

    def test_a_custom_pack_can_extend_a_builtin(self):
        pack = EntityPack.from_dict(
            {
                "name": "internal",
                "extends": "default",
                "rules": [{"entity": "EMPLOYEE_ID", "pattern": r"EMP-\d{6}", "severity": "medium"}],
            }
        )
        assert "EMPLOYEE_ID" in pack.entity_names()
        assert "EMAIL" in pack.entity_names()

    def test_yaml_round_trip(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "name: custom\nrules:\n  - entity: TICKET\n    pattern: 'TKT-\\d{4}'\n",
            encoding="utf-8",
        )
        assert EntityPack.from_yaml(path).entity_names() == ["TICKET"]

    def test_missing_yaml_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EntityPack.from_yaml(tmp_path / "absent.yaml")

    def test_non_mapping_yaml_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            EntityPack.from_yaml(path)

    def test_registering_over_an_existing_name_requires_intent(self):
        # Silently swapping a pack would change redaction behaviour process-wide
        # with no signal.
        register_pack(EntityPack("registry-test", ()))
        with pytest.raises(ValueError, match="already registered"):
            register_pack(EntityPack("registry-test", ()))
        register_pack(EntityPack("registry-test", ()), overwrite=True)


class TestValidators:
    @pytest.mark.parametrize(
        "number", ["4111 1111 1111 1111", "5500-0000-0000-0004", "378282246310005"]
    )
    def test_luhn_accepts_real_card_numbers(self, number):
        assert validate("luhn", number)

    @pytest.mark.parametrize("number", ["1234 5678 9012 3456", "4111 1111 1111 1112", "123"])
    def test_luhn_rejects_invalid_numbers(self, number):
        assert not validate("luhn", number)

    @pytest.mark.parametrize("iban", ["GB33BUKB20201555555555", "DE89370400440532013000"])
    def test_iban_accepts_valid_accounts(self, iban):
        assert validate("iban", iban)

    @pytest.mark.parametrize("iban", ["GB33BUKB20201555555556", "XX00NOTANIBAN", "GB"])
    def test_iban_rejects_invalid_accounts(self, iban):
        assert not validate("iban", iban)

    def test_nhs_accepts_a_valid_number(self):
        assert validate("nhs", "943 476 5919")

    @pytest.mark.parametrize("number", ["943 476 5918", "123456789", "0000000000"])
    def test_nhs_rejects_invalid_numbers(self, number):
        assert not validate("nhs", number)

    def test_no_validator_always_passes(self):
        assert validate(None, "anything at all")

    def test_unknown_validator_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown validator"):
            validate("astrology", "x")


class TestDetection:
    @pytest.mark.parametrize(
        ("text", "entity"),
        [
            ("write to alice@example.com", "EMAIL"),
            ("call 555-0142 now", "PHONE"),
            ("ssn 123-45-6789 on file", "US_SSN"),
            ("card 4111 1111 1111 1111", "CREDIT_CARD"),
            ("server at 192.168.1.24", "IP_ADDRESS"),
            ("key sk_live_abcdefgh12345678", "API_KEY"),
            ("token AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY"),
        ],
    )
    def test_detects_universal_entities(self, redactor, text, entity):
        assert entity in {m.entity for m in redactor.detector.detect(text)}

    @pytest.mark.parametrize(
        "text",
        [
            "the meeting is at 3pm",
            "order 12345 shipped",
            "section 4 clause 12 applies",
            "version 2.10.4 released",
            "",
        ],
    )
    def test_leaves_ordinary_text_alone(self, redactor, text):
        assert redactor.detector.detect(text) == []

    def test_a_checksum_rejects_a_card_shaped_non_card(self, redactor):
        assert "CREDIT_CARD" not in {
            m.entity for m in redactor.detector.detect("ref 1234 5678 9012 3456")
        }

    def test_context_words_gate_a_generic_shape(self):
        detector = PIIDetector(pack="healthcare", include=["MRN_BARE"])
        assert detector.detect("Patient number 4471902 admitted")
        assert not detector.detect("The order total was 4471902 cents")

    def test_overlapping_matches_resolve_to_the_most_specific(self):
        # A card number is also account-number-shaped; only one should be
        # reported, and it should be the card.
        matches = PIIDetector(pack="fintech").detect("pay with 4111 1111 1111 1111 today")
        assert [m.entity for m in matches] == ["CREDIT_CARD"]

    def test_resolution_is_deterministic_regardless_of_rule_order(self):
        text = "card 4111 1111 1111 1111"
        forward = PIIDetector(pack="fintech").detect(text)
        pack = get_pack("fintech")
        reversed_pack = EntityPack(name="rev", rules=tuple(reversed(pack.rules)))
        backward = PIIDetector(pack=reversed_pack).detect(text)
        assert [m.entity for m in forward] == [m.entity for m in backward]

    def test_matches_are_returned_in_document_order(self, redactor):
        matches = redactor.detector.detect("a@b.com then 555-0142 then c@d.com")
        assert [m.start for m in matches] == sorted(m.start for m in matches)

    def test_a_match_never_carries_the_matched_value(self, redactor):
        record = redactor.detector.detect("ssn 123-45-6789")[0].to_dict()
        assert "123-45-6789" not in str(record)
        assert record["entity"] == "US_SSN"

    def test_negative_context_window_is_rejected(self):
        with pytest.raises(ValueError, match="context_window"):
            PIIDetector(context_window=-1)


class TestRedaction:
    def test_replaces_every_occurrence(self, redactor):
        result = redactor.redact("mail a@b.com and c@d.com")
        assert result.text == "mail [EMAIL] and [EMAIL]"
        assert result.counts == {"EMAIL": 2}

    def test_clean_text_passes_through_unchanged(self, redactor):
        result = redactor.redact("nothing sensitive here")
        assert result.text == "nothing sensitive here"
        assert not result.found

    def test_reports_the_highest_severity_found(self, redactor):
        assert redactor.redact("a@b.com and card 4111 1111 1111 1111").max_severity is (
            Severity.CRITICAL
        )

    def test_summary_never_leaks_the_values(self, redactor):
        summary = redactor.redact("ssn 123-45-6789").to_dict()
        assert "123-45-6789" not in str(summary)
        assert summary["entities"] == {"US_SSN": 1}

    def test_different_packs_redact_the_same_text_differently(self):
        text = "Policy POL-889231 with IBAN GB33BUKB20201555555555"
        insurance = PIIRedactor(pack="insurance").redact(text).text
        fintech = PIIRedactor(pack="fintech").redact(text).text
        assert "[POLICY_NUMBER]" in insurance and "IBAN GB33" in insurance
        assert "[IBAN]" in fintech and "POL-889231" in fintech

    def test_stage_selection_is_honoured(self, redactor):
        output_only = PIIRedactor(on_stages=("output",))
        assert output_only.on_input("mail a@b.com", {}).allowed
        assert output_only.on_output("mail a@b.com", {}).modified

    def test_unknown_stage_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            PIIRedactor(on_stages=("middle",))

    def test_block_severity_escalates_a_redaction_to_a_block(self):
        guard = PIIRedactor(block_severity="critical")
        assert guard.on_output("key sk_live_abcdefgh12345678", {}).blocked
        assert guard.on_output("mail a@b.com", {}).modified

    def test_block_entities_escalates_by_name(self):
        guard = PIIRedactor(block_entities=["EMAIL"])
        assert guard.on_output("mail a@b.com", {}).blocked

    def test_include_narrows_what_is_detected(self):
        guard = PIIRedactor(include=["EMAIL"])
        assert guard.redact("a@b.com and 555-0142").text == "[EMAIL] and 555-0142"

    def test_the_guardrail_result_carries_no_pii(self, redactor):
        record = redactor.on_output("ssn 123-45-6789", {}).to_dict()
        assert "123-45-6789" not in str(record)
        assert record["action"] == "redact"

    def test_a_clean_result_reports_low_severity(self, redactor):
        # max_severity has to answer for the no-matches case too, and LOW is
        # the honest answer -- not an exception, and not the severity of a
        # match that never happened.
        result = redactor.redact("nothing sensitive here at all")
        assert not result.found
        assert result.max_severity is Severity.LOW
        assert result.counts == {}

    def test_nhs_number_whose_check_digit_computes_to_eleven(self):
        # The NHS checksum yields `11 - remainder`; when remainder is 0 that is
        # 11, which is not a digit and wraps to 0. Easy branch to get wrong,
        # and it silently rejects a whole class of valid numbers.
        detector = PIIDetector(pack="healthcare")
        matches = detector.detect("NHS number 1000000060")

        assert [match.entity for match in matches] == ["NHS_NUMBER"]
        assert matches[0].validated

    def test_nhs_number_with_a_check_digit_of_ten_is_rejected(self):
        # 10 is not expressible as a single check digit, so such a number
        # cannot be valid however plausible it looks.
        detector = PIIDetector(pack="healthcare")
        assert detector.detect("NHS number 1000000015") == []
