"""The guardrail contract: results, actions, and the base class.

Everything in this package returns a :class:`GuardrailResult`. It is a plain
dataclass with no dependency on ``wardhook-core`` -- core reads its attributes
structurally, which is what lets the two packages compose while remaining
independently installable.

:class:`BaseGuardrail` implements all three hooks as allow-by-default, so a
subclass overrides only the stage it cares about. Subclassing is a convenience,
not a requirement: any object with the right method shapes works.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["Action", "BaseGuardrail", "GuardrailResult", "Severity"]


class Action(str, Enum):
    """What should happen to the inspected text or call.

    A :class:`str` enum, so ``Action.BLOCK == "block"`` is true. That means a
    consumer never needs to import this enum to interpret a result.

    Attributes:
        ALLOW: Proceed unchanged.
        REDACT: Proceed, but with the modified text on the result.
        BLOCK: Stop. The text is not forwarded, or the tool is not executed.
    """

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class Severity(str, Enum):
    """How serious a detection is, for triage in compliance review.

    Severity does not change what the guardrail *does* -- that is
    :class:`Action`. It exists so a reviewer reading a month of audit records
    can sort by risk instead of reading every line.

    Attributes:
        LOW: Noted, minimal risk. An internal reference or a low-value locator.
        MEDIUM: Identifying but not directly exploitable, such as an email.
        HIGH: Directly identifying or exploitable, such as a national ID.
        CRITICAL: Immediate exposure, such as a live credential.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class GuardrailResult:
    """The outcome of one guardrail inspecting one piece of text or one call.

    Attributes:
        action: What should happen next.
        text: The text to carry forward. For a redaction this is the modified
            text; otherwise it is what was passed in.
        reason: Human-readable explanation, surfaced in audit records.
        rule: Identifier of the rule that fired, such as an entity name.
        name: The guardrail that produced this result.
        severity: Risk level, for triage.
        details: Structured extras. Must never contain the matched values
            themselves -- audit records are written from this, and a log that
            stores the PII it detected has defeated its own purpose.
    """

    action: Action = Action.ALLOW
    text: str = ""
    reason: str | None = None
    rule: str | None = None
    name: str = "guardrail"
    severity: Severity = Severity.LOW
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        """Whether the run proceeds unchanged."""
        return self.action is Action.ALLOW

    @property
    def blocked(self) -> bool:
        """Whether the run stops here."""
        return self.action is Action.BLOCK

    @property
    def modified(self) -> bool:
        """Whether the text was changed."""
        return self.action is Action.REDACT

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this result.

        The ``text`` field is deliberately excluded. This dict feeds audit
        logs, and including the inspected text would write the very data the
        guardrail exists to protect.

        Returns:
            A dict safe to persist.
        """
        return {
            "guardrail": self.name,
            "action": self.action.value,
            "reason": self.reason,
            "rule": self.rule,
            "severity": self.severity.value,
            "details": dict(self.details),
        }

    @classmethod
    def allow(cls, text: str, *, name: str = "guardrail") -> GuardrailResult:
        """Build an allow result.

        Args:
            text: The text passing through unchanged.
            name: The guardrail's name.

        Returns:
            An allow result.
        """
        return cls(action=Action.ALLOW, text=text, name=name)

    @classmethod
    def redact(
        cls,
        text: str,
        *,
        reason: str,
        rule: str,
        name: str = "guardrail",
        severity: Severity = Severity.MEDIUM,
        details: Mapping[str, Any] | None = None,
    ) -> GuardrailResult:
        """Build a redaction result.

        Args:
            text: The **modified** text to carry forward.
            reason: Why the redaction happened.
            rule: Which rule fired.
            name: The guardrail's name.
            severity: Risk level.
            details: Structured extras, excluding any matched values.

        Returns:
            A redaction result.
        """
        return cls(
            action=Action.REDACT,
            text=text,
            reason=reason,
            rule=rule,
            name=name,
            severity=severity,
            details=dict(details or {}),
        )

    @classmethod
    def block(
        cls,
        text: str,
        *,
        reason: str,
        rule: str,
        name: str = "guardrail",
        severity: Severity = Severity.HIGH,
        details: Mapping[str, Any] | None = None,
    ) -> GuardrailResult:
        """Build a blocking result.

        Args:
            text: The text that was inspected, carried unchanged.
            reason: Why the run should stop.
            rule: Which rule fired.
            name: The guardrail's name.
            severity: Risk level.
            details: Structured extras, excluding any matched values.

        Returns:
            A blocking result.
        """
        return cls(
            action=Action.BLOCK,
            text=text,
            reason=reason,
            rule=rule,
            name=name,
            severity=severity,
            details=dict(details or {}),
        )


class BaseGuardrail:
    """Convenience base implementing all three hooks as allow-by-default.

    Subclass and override only the stage you care about. A guardrail that only
    polices tool calls implements :meth:`on_tool_call` and inherits pass-through
    behaviour for text.

    The unused arguments on the default hooks below are part of the interface
    contract every guardrail implements, not dead parameters, hence the
    ``noqa: ARG002`` markers.

    Args:
        name: Identifier recorded in audit records. Defaults to the class name.

    Example:
        >>> class NoShouting(BaseGuardrail):
        ...     def on_output(self, text, context):
        ...         if text.isupper():
        ...             return GuardrailResult.redact(
        ...                 text.capitalize(),
        ...                 reason="all caps",
        ...                 rule="shouting",
        ...                 name=self.name,
        ...             )
        ...         return GuardrailResult.allow(text, name=self.name)
        >>> guard = NoShouting()
        >>> guard.on_output("HELLO", {}).text
        'Hello'
        >>> guard.on_input("anything", {}).allowed
        True
    """

    name: str = "guardrail"

    def __init__(self, name: str | None = None) -> None:
        """Initialise the guardrail.

        Args:
            name: Override the recorded name.
        """
        if name is not None:
            self.name = name
        elif type(self).name == BaseGuardrail.name:
            self.name = type(self).__name__

    def on_input(self, text: str, context: Mapping[str, Any]) -> GuardrailResult:  # noqa: ARG002
        """Inspect user input. Allows by default.

        Args:
            text: The incoming text.
            context: Run context with ``run_id``, ``stage``, ``node`` and
                ``principal`` keys.

        Returns:
            An allow result.
        """
        return GuardrailResult.allow(text, name=self.name)

    def on_output(self, text: str, context: Mapping[str, Any]) -> GuardrailResult:  # noqa: ARG002
        """Inspect model output. Allows by default.

        Args:
            text: The model's response text.
            context: Run context; see :meth:`on_input`.

        Returns:
            An allow result.
        """
        return GuardrailResult.allow(text, name=self.name)

    def on_tool_call(
        self,
        tool_name: str,
        tool_args: Mapping[str, Any],  # noqa: ARG002
        context: Mapping[str, Any],  # noqa: ARG002
    ) -> GuardrailResult:
        """Decide whether a tool call may execute. Allows by default.

        Args:
            tool_name: The tool the model asked to call.
            tool_args: Arguments supplied by the model.
            context: Run context; see :meth:`on_input`.

        Returns:
            An allow result.
        """
        return GuardrailResult.allow(tool_name, name=self.name)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"{type(self).__name__}(name={self.name!r})"
