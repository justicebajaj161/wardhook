"""PII detection and redaction.

Detection is pattern-based, driven by whichever :class:`~wardhook.guardrails.
entities.EntityPack` you select. Two mechanisms keep the false-positive rate
usable on real text:

* **Checksum validators.** A regex matches anything card-shaped; a Luhn check
  is what separates a real card number from sixteen arbitrary digits. The same
  applies to IBANs (mod-97) and NHS numbers (modulus-11).
* **Context words.** Some entities are too generic to stand alone -- a bare
  six-digit medical record number is just a number. Those rules require a
  related term nearby before a match counts.

Overlapping matches are resolved deterministically, so a card number is never
also reported as a bank account.

Example:
    >>> redactor = PIIRedactor()
    >>> result = redactor.redact("Email alice@example.com or call 555-0142.")
    >>> result.text
    'Email [EMAIL] or call [PHONE].'
    >>> result.counts
    {'EMAIL': 1, 'PHONE': 1}
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from wardhook.guardrails.base import BaseGuardrail, GuardrailResult, Severity
from wardhook.guardrails.entities import EntityPack, EntityRule, get_pack

__all__ = ["PIIDetector", "PIIMatch", "PIIRedactor", "RedactionResult", "validate"]

_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def _all_same_digit(value: str) -> bool:
    """Report whether every digit in ``value`` is identical.

    Runs like ``0000000000`` and ``1111 1111 1111 1111`` satisfy the checksums
    they are shaped for, but in real documents they are placeholders, test
    fixtures, or form templates rather than live identifiers. Rejecting them
    trades a negligible amount of recall for a meaningful drop in false
    positives on exactly the kind of text these detectors run against.

    Args:
        value: Candidate text; non-digits are ignored.

    Returns:
        ``True`` if there are digits and they are all the same.
    """
    digits = [c for c in value if c.isdigit()]
    return bool(digits) and len(set(digits)) == 1


def _luhn(value: str) -> bool:
    """Check a number against the Luhn algorithm.

    Args:
        value: Candidate text; non-digits are ignored.

    Returns:
        ``True`` if the digits satisfy the Luhn checksum and are not a
        single repeated digit.
    """
    if _all_same_digit(value):
        return False
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _iban(value: str) -> bool:
    """Check an IBAN against its mod-97 checksum.

    Args:
        value: Candidate text; spaces and hyphens are ignored.

    Returns:
        ``True`` if the value is a structurally valid IBAN.
    """
    cleaned = re.sub(r"[\s-]", "", value).upper()
    if not 15 <= len(cleaned) <= 34 or not cleaned[:2].isalpha() or not cleaned[2:4].isdigit():
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    try:
        numeric = "".join(str(int(c, 36)) for c in rearranged)
    except ValueError:
        return False
    return int(numeric) % 97 == 1


def _nhs(value: str) -> bool:
    """Check a UK NHS number against its modulus-11 check digit.

    Args:
        value: Candidate text; non-digits are ignored.

    Returns:
        ``True`` if the ten digits satisfy the NHS check and are not a
        single repeated digit.
    """
    if _all_same_digit(value):
        return False
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 10:
        return False
    total = sum(digit * (10 - index) for index, digit in enumerate(digits[:9]))
    remainder = total % 11
    check = 11 - remainder
    if check == 11:
        check = 0
    if check == 10:  # 10 is not a valid check digit; the number is invalid.
        return False
    return check == digits[9]


_VALIDATORS = {"luhn": _luhn, "iban": _iban, "nhs": _nhs}


def validate(validator: str | None, value: str) -> bool:
    """Run a named checksum validator.

    Args:
        validator: One of ``luhn``, ``iban``, ``nhs``, or ``None``.
        value: The matched text.

    Returns:
        ``True`` if the value passes, or if no validator was requested.

    Raises:
        ValueError: If the validator name is unknown, which means a pack
            references a checksum this package does not implement.

    Example:
        >>> validate("luhn", "4111 1111 1111 1111")
        True
        >>> validate("luhn", "1234 5678 9012 3456")
        False
        >>> validate(None, "anything")
        True
    """
    if validator is None:
        return True
    try:
        check = _VALIDATORS[validator]
    except KeyError as exc:
        known = ", ".join(sorted(_VALIDATORS))
        raise ValueError(f"Unknown validator {validator!r}. Available: {known}.") from exc
    return check(value)


@dataclass(frozen=True, slots=True)
class PIIMatch:
    """One detected entity occurrence.

    The matched **value is deliberately not stored**. A match record travels
    into audit logs and traces, and a detector that persists the PII it found
    has defeated its own purpose. The span offsets and length are enough to
    describe precisely what changed.

    Attributes:
        entity: The entity identifier, such as ``US_SSN``.
        start: Start offset in the original text.
        end: End offset in the original text.
        length: Number of characters matched.
        severity: Risk level from the rule.
        validated: Whether a checksum confirmed the match. ``False`` here means
            the rule had no validator, not that a check failed -- failed checks
            are discarded rather than reported.
        replacement: The placeholder that will replace this span.
    """

    entity: str
    start: int
    end: int
    length: int
    severity: Severity
    validated: bool
    replacement: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record, free of the matched value."""
        return {
            "entity": self.entity,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "severity": self.severity.value,
            "validated": self.validated,
        }


@dataclass(slots=True)
class RedactionResult:
    """The outcome of redacting one piece of text.

    Attributes:
        text: The redacted text.
        matches: Every entity occurrence that was replaced, in document order.
        counts: How many occurrences were found per entity type.
        original_length: Length of the input text, for diff reporting.
    """

    text: str
    matches: list[PIIMatch] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    original_length: int = 0

    @property
    def found(self) -> bool:
        """Whether anything was detected."""
        return bool(self.matches)

    @property
    def max_severity(self) -> Severity:
        """The highest severity among the matches, or ``LOW`` if none."""
        if not self.matches:
            return Severity.LOW
        return max((m.severity for m in self.matches), key=lambda s: _SEVERITY_ORDER[s])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary, free of any matched value."""
        return {
            "entities": dict(self.counts),
            "match_count": len(self.matches),
            "max_severity": self.max_severity.value,
            "spans": [m.to_dict() for m in self.matches],
            "original_length": self.original_length,
            "redacted_length": len(self.text),
        }


class PIIDetector:
    """Finds entity occurrences in text using a configurable pack.

    Args:
        pack: A pack name, an :class:`EntityPack`, a sequence of
            :class:`EntityRule`, or ``None`` for the ``default`` pack.
        include: Restrict detection to these entity names.
        exclude: Skip these entity names.
        context_window: Characters either side of a match to search when a rule
            declares ``context_words``.

    Example:
        >>> detector = PIIDetector(pack="healthcare", include=["NHS_NUMBER"])
        >>> [m.entity for m in detector.detect("NHS number 943 476 5919 on file")]
        ['NHS_NUMBER']
        >>> detector.detect("NHS number 943 476 5918 on file")  # bad check digit
        []
    """

    def __init__(
        self,
        pack: str | EntityPack | Sequence[EntityRule] | None = None,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        context_window: int = 60,
    ) -> None:
        """Initialise the detector. See the class docstring for arguments."""
        if context_window < 0:
            raise ValueError(f"context_window must not be negative, got {context_window}")
        resolved = get_pack(pack)
        if include is not None or exclude is not None:
            resolved = resolved.filter(include=include, exclude=exclude)
        self.pack = resolved
        self.context_window = context_window
        self._compiled: list[tuple[EntityRule, re.Pattern[str]]] = [
            (rule, rule.compiled()) for rule in resolved.rules
        ]

    def _has_context(self, rule: EntityRule, text: str, start: int, end: int) -> bool:
        """Check whether a rule's required context words appear near a match.

        Args:
            rule: The rule that matched.
            text: The full text.
            start: Match start offset.
            end: Match end offset.

        Returns:
            ``True`` if the rule needs no context, or a required word is nearby.
        """
        if not rule.context_words:
            return True
        window = text[
            max(0, start - self.context_window) : min(len(text), end + self.context_window)
        ].lower()
        return any(word in window for word in rule.context_words)

    def detect(self, text: str) -> list[PIIMatch]:
        """Find every entity occurrence in ``text``.

        Args:
            text: The text to scan.

        Returns:
            Matches in document order, with overlaps resolved. An empty string
            yields an empty list.
        """
        if not text:
            return []

        candidates: list[PIIMatch] = []
        for rule, pattern in self._compiled:
            for found in pattern.finditer(text):
                value = found.group(0)
                if not validate(rule.validator, value):
                    continue
                if not self._has_context(rule, text, found.start(), found.end()):
                    continue
                candidates.append(
                    PIIMatch(
                        entity=rule.entity,
                        start=found.start(),
                        end=found.end(),
                        length=len(value),
                        severity=rule.severity,
                        validated=rule.validator is not None,
                        replacement=rule.redaction(),
                    )
                )
        return self._resolve_overlaps(candidates)

    @staticmethod
    def _resolve_overlaps(matches: list[PIIMatch]) -> list[PIIMatch]:
        """Drop matches that overlap a stronger one.

        Several rules can legitimately match the same span -- a card number is
        also account-number-shaped. Ranking by severity, then by span length,
        then by a checksum-confirmed match, picks the most specific reading and
        keeps the result deterministic regardless of rule order in the pack.

        Args:
            matches: All candidate matches.

        Returns:
            Non-overlapping matches in document order.
        """
        ranked = sorted(
            matches,
            key=lambda m: (
                -_SEVERITY_ORDER[m.severity],
                -m.length,
                not m.validated,
                m.start,
                m.entity,
            ),
        )
        kept: list[PIIMatch] = []
        for match in ranked:
            if any(match.start < k.end and k.start < match.end for k in kept):
                continue
            kept.append(match)
        return sorted(kept, key=lambda m: m.start)


class PIIRedactor(BaseGuardrail):
    """A guardrail that redacts, or optionally blocks on, detected PII.

    Args:
        pack: Entity pack name, instance, rules, or ``None`` for ``default``.
        include: Restrict detection to these entity names.
        exclude: Skip these entity names.
        on_stages: Which stages to police -- any of ``"input"`` and
            ``"output"``. Output-only is a common configuration: you accept
            whatever the user typed but guarantee nothing leaks back out.
        block_severity: Block instead of redact when a match at or above this
            severity is found. ``None`` (the default) always redacts. Use this
            for entities where seeing the text at all is the incident, such as
            a live credential.
        block_entities: Always block when one of these entities is found,
            regardless of severity.
        name: Identifier recorded in audit records.
        context_window: Characters either side of a match for context checks.

    Example:
        >>> guard = PIIRedactor(pack="fintech", on_stages=("output",))
        >>> guard.on_output("Card 4111 1111 1111 1111 on file.", {}).text
        'Card [CREDIT_CARD] on file.'
        >>> guard.on_input("Card 4111 1111 1111 1111 on file.", {}).allowed
        True
    """

    def __init__(
        self,
        pack: str | EntityPack | Sequence[EntityRule] | None = None,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        on_stages: Sequence[str] = ("input", "output"),
        block_severity: Severity | str | None = None,
        block_entities: Iterable[str] | None = None,
        name: str = "pii-redactor",
        context_window: int = 60,
    ) -> None:
        """Initialise the guardrail. See the class docstring for arguments."""
        super().__init__(name=name)
        self.detector = PIIDetector(
            pack, include=include, exclude=exclude, context_window=context_window
        )
        unknown = set(on_stages) - {"input", "output"}
        if unknown:
            raise ValueError(
                f"Unknown stage(s) {sorted(unknown)}. Valid stages: 'input', 'output'."
            )
        self.on_stages = tuple(on_stages)
        self.block_severity = Severity(block_severity) if block_severity else None
        self.block_entities = {e.upper() for e in (block_entities or ())}

    @property
    def pack(self) -> EntityPack:
        """The entity pack this guardrail is using."""
        return self.detector.pack

    def redact(self, text: str) -> RedactionResult:
        """Replace every detected entity with its placeholder.

        Args:
            text: The text to redact.

        Returns:
            The result, carrying the redacted text and span-level match records.

        Example:
            >>> PIIRedactor().redact("Reach me at bob@example.org").text
            'Reach me at [EMAIL]'
        """
        matches = self.detector.detect(text)
        if not matches:
            return RedactionResult(text=text, original_length=len(text))

        pieces: list[str] = []
        cursor = 0
        counts: dict[str, int] = {}
        for match in matches:
            pieces.append(text[cursor : match.start])
            pieces.append(match.replacement)
            cursor = match.end
            counts[match.entity] = counts.get(match.entity, 0) + 1
        pieces.append(text[cursor:])

        return RedactionResult(
            text="".join(pieces),
            matches=matches,
            counts=counts,
            original_length=len(text),
        )

    def _should_block(self, result: RedactionResult) -> bool:
        """Decide whether these findings warrant a block rather than a redaction.

        Args:
            result: The redaction result.

        Returns:
            ``True`` if the run should stop.
        """
        if any(m.entity.upper() in self.block_entities for m in result.matches):
            return True
        if self.block_severity is None:
            return False
        threshold = _SEVERITY_ORDER[self.block_severity]
        return any(_SEVERITY_ORDER[m.severity] >= threshold for m in result.matches)

    def _inspect(self, text: str, stage: str) -> GuardrailResult:
        """Run detection for one stage and build the guardrail result.

        Args:
            text: The text to inspect.
            stage: ``"input"`` or ``"output"``.

        Returns:
            An allow, redact, or block result.
        """
        if stage not in self.on_stages:
            return GuardrailResult.allow(text, name=self.name)

        result = self.redact(text)
        if not result.found:
            return GuardrailResult.allow(text, name=self.name)

        entities = ", ".join(sorted(result.counts))
        details = {**result.to_dict(), "stage": stage, "pack": self.pack.name}

        if self._should_block(result):
            return GuardrailResult.block(
                text,
                reason=f"blocked on sensitive entities: {entities}",
                rule=entities,
                name=self.name,
                severity=result.max_severity,
                details=details,
            )
        return GuardrailResult.redact(
            result.text,
            reason=f"redacted {len(result.matches)} occurrence(s): {entities}",
            rule=entities,
            name=self.name,
            severity=result.max_severity,
            details=details,
        )

    def on_input(self, text: str, context: Mapping[str, Any]) -> GuardrailResult:  # noqa: ARG002
        """Redact PII in user input.

        Args:
            text: The incoming text.
            context: Run context, unused by this guardrail.

        Returns:
            The result for the input stage.
        """
        return self._inspect(text, "input")

    def on_output(self, text: str, context: Mapping[str, Any]) -> GuardrailResult:  # noqa: ARG002
        """Redact PII in model output.

        Args:
            text: The model's response text.
            context: Run context, unused by this guardrail.

        Returns:
            The result for the output stage.
        """
        return self._inspect(text, "output")

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            f"PIIRedactor(name={self.name!r}, pack={self.pack.name!r}, "
            f"entities={len(self.pack.rules)}, stages={self.on_stages})"
        )
