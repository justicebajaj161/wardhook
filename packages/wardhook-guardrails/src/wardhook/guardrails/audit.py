"""Structured audit logging for compliance review.

Every guardrail action -- redaction, block, or explicit allow -- becomes one
JSON object on one line. JSONL is the format because an audit trail is
append-only by nature, survives a crash mid-write with at most one bad line,
and can be tailed, grepped, and loaded incrementally without a parser holding
the whole file.

**The central design rule: an audit record never contains the data it is
auditing.** A log that stores the PII a redactor just removed has recreated the
exposure it was built to prevent, somewhere with a longer retention period and
usually weaker access control. Records therefore describe *what changed* --
entity type, character offsets, lengths -- and never *what it was*.

Where correlation genuinely matters ("did this same SSN appear in three
conversations?"), :class:`AuditLogger` can emit a salted fingerprint. The salt
is random per process by default, so fingerprints correlate within a session
and are useless afterwards; supply a stable salt only if you have decided that
cross-session correlation is worth the risk it carries.

Example:
    >>> import tempfile, pathlib
    >>> from wardhook.guardrails import PIIRedactor
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "audit.jsonl"
    >>> logger = AuditLogger(tmp)
    >>> result = PIIRedactor().on_output("SSN 123-45-6789", {"run_id": "r1"})
    >>> event = logger.record(
    ...     result, stage="output", run_id="r1", before="SSN 123-45-6789", after=result.text
    ... )
    >>> event.action
    'redact'
    >>> "123-45-6789" in tmp.read_text()
    False
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["AuditEvent", "AuditLogger", "TextDiff", "diff_text"]


@dataclass(frozen=True, slots=True)
class TextDiff:
    """A description of how text changed, carrying none of the text.

    Attributes:
        changed: Whether anything changed at all.
        before_length: Character count before.
        after_length: Character count after.
        spans: One record per replaced span: entity type, offsets and length.
        entities: Count of replacements per entity type.
    """

    changed: bool
    before_length: int
    after_length: int
    spans: tuple[dict[str, Any], ...] = ()
    entities: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "changed": self.changed,
            "before_length": self.before_length,
            "after_length": self.after_length,
            "delta": self.after_length - self.before_length,
            "spans": list(self.spans),
            "entities": dict(self.entities),
        }


def diff_text(
    before: str,
    after: str,
    spans: Sequence[Mapping[str, Any]] = (),
) -> TextDiff:
    """Describe a text change without recording either version.

    Args:
        before: The original text. Used only for its length.
        after: The modified text. Used only for its length.
        spans: Span records from a detector, each with ``entity``, ``start``,
            ``end`` and ``length`` keys.

    Returns:
        A diff safe to persist.

    Example:
        >>> d = diff_text(
        ...     "SSN 123-45-6789",
        ...     "SSN [US_SSN]",
        ...     [{"entity": "US_SSN", "start": 4, "end": 15, "length": 11}],
        ... )
        >>> d.changed, d.entities
        (True, {'US_SSN': 1})
    """
    counts: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for span in spans:
        entity = str(span.get("entity", "UNKNOWN"))
        counts[entity] = counts.get(entity, 0) + 1
        kept.append(
            {
                "entity": entity,
                "start": int(span.get("start", -1)),
                "end": int(span.get("end", -1)),
                "length": int(span.get("length", 0)),
            }
        )
    return TextDiff(
        changed=before != after,
        before_length=len(before),
        after_length=len(after),
        spans=tuple(kept),
        entities=counts,
    )


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One guardrail action, as it will appear in the log.

    Attributes:
        timestamp: UTC ISO-8601 timestamp of the decision.
        run_id: Correlates this event with a trace and with other events from
            the same agent invocation.
        guardrail: Which guardrail acted.
        action: ``allow``, ``redact`` or ``block``.
        stage: ``input``, ``output`` or ``tool_call``.
        rule: The specific rule that fired.
        severity: Risk level.
        reason: Human-readable explanation.
        principal_id: Identifier of the caller, when one was supplied.
        tool: The tool involved, for tool-call events.
        diff: How the text changed, if it changed.
        details: Structured extras from the guardrail.
        fingerprints: Optional salted digests for correlating repeat values.
    """

    timestamp: str
    run_id: str
    guardrail: str
    action: str
    stage: str
    rule: str | None = None
    severity: str = "low"
    reason: str | None = None
    principal_id: str | None = None
    tool: str | None = None
    diff: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)
    fingerprints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record with empty fields omitted."""
        record: dict[str, Any] = {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "guardrail": self.guardrail,
            "action": self.action,
            "stage": self.stage,
            "severity": self.severity,
        }
        for key, value in (
            ("rule", self.rule),
            ("reason", self.reason),
            ("principal_id", self.principal_id),
            ("tool", self.tool),
            ("diff", self.diff),
        ):
            if value is not None:
                record[key] = value
        if self.details:
            record["details"] = self.details
        if self.fingerprints:
            record["fingerprints"] = list(self.fingerprints)
        return record

    def to_json(self) -> str:
        """Return this event as one compact JSON line.

        Returns:
            JSON with sorted keys, so two logs of the same events diff cleanly.
        """
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class AuditLogger:
    """Writes guardrail actions to a JSONL audit trail.

    Args:
        path: Destination file. Parent directories are created. ``None`` keeps
            events in memory only, which is what tests and dry runs want.
        record_allows: Also log allow decisions. Off by default -- an allow is
            the overwhelmingly common case and logging every one buries the
            actions a reviewer is looking for. Turn it on where a regulator
            expects positive evidence that every request was screened.
        fingerprint_salt: Salt for correlation digests. Defaults to a random
            per-process value, so fingerprints correlate within a session and
            are meaningless across sessions. Pass a stable salt only if you
            have accepted the risk of cross-session correlation.
        max_memory_events: How many events to retain in memory for
            :meth:`report`. Bounded so a long-running process cannot grow
            without limit.

    Example:
        >>> logger = AuditLogger()  # in-memory only
        >>> logger.log(
        ...     guardrail="pii",
        ...     action="block",
        ...     stage="output",
        ...     run_id="r1",
        ...     reason="critical entity",
        ...     rule="API_KEY",
        ... ).action
        'block'
        >>> logger.report()["by_action"]
        {'block': 1}
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        record_allows: bool = False,
        fingerprint_salt: str | None = None,
        max_memory_events: int = 10_000,
    ) -> None:
        """Initialise the logger. See the class docstring for arguments."""
        self.path = Path(path).expanduser() if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record_allows = record_allows
        self._salt = fingerprint_salt if fingerprint_salt is not None else secrets.token_hex(16)
        self._events: list[AuditEvent] = []
        self._max_memory_events = max_memory_events
        # Guardrails can run concurrently across requests in a served agent,
        # and a torn line in an audit file is a corrupt record, not a warning.
        self._lock = threading.Lock()

    @property
    def events(self) -> list[AuditEvent]:
        """Return a copy of the retained in-memory events."""
        with self._lock:
            return list(self._events)

    def fingerprint(self, value: str) -> str:
        """Return a salted, truncated digest of a sensitive value.

        Args:
            value: The value to fingerprint.

        Returns:
            A 16-character hex digest. Salted so it cannot be checked against a
            precomputed table of, say, every possible SSN -- an unsalted hash of
            a low-entropy identifier is not anonymisation.
        """
        digest = hashlib.blake2b(
            value.encode("utf-8"), key=self._salt.encode("utf-8"), digest_size=8
        )
        return digest.hexdigest()

    def log(
        self,
        *,
        guardrail: str,
        action: str,
        stage: str,
        run_id: str = "",
        rule: str | None = None,
        severity: str = "low",
        reason: str | None = None,
        principal_id: str | None = None,
        tool: str | None = None,
        diff: TextDiff | None = None,
        details: Mapping[str, Any] | None = None,
        fingerprints: Iterable[str] = (),
    ) -> AuditEvent:
        """Record one guardrail action.

        Args:
            guardrail: Which guardrail acted.
            action: ``allow``, ``redact`` or ``block``.
            stage: ``input``, ``output`` or ``tool_call``.
            run_id: Correlation id for the agent invocation.
            rule: The rule that fired.
            severity: Risk level.
            reason: Human-readable explanation.
            principal_id: The caller's identifier.
            tool: Tool involved, for tool-call events.
            diff: How the text changed.
            details: Structured extras.
            fingerprints: Correlation digests.

        Returns:
            The event, whether or not it was written to disk. Allow events are
            still returned when ``record_allows`` is off, so a caller can
            inspect them without them entering the durable trail.
        """
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            run_id=run_id,
            guardrail=guardrail,
            action=action,
            stage=stage,
            rule=rule,
            severity=severity,
            reason=reason,
            principal_id=principal_id,
            tool=tool,
            diff=diff.to_dict() if diff else None,
            details=dict(details or {}),
            fingerprints=tuple(fingerprints),
        )

        if action == "allow" and not self.record_allows:
            return event

        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_memory_events:
                del self._events[: len(self._events) - self._max_memory_events]
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(event.to_json() + "\n")
                    handle.flush()
                    # An audit record that is still in the OS page cache when
                    # the process dies is an audit record that did not happen.
                    os.fsync(handle.fileno())
        return event

    def record(
        self,
        result: Any,
        *,
        stage: str,
        run_id: str = "",
        before: str | None = None,
        after: str | None = None,
        principal_id: str | None = None,
        tool: str | None = None,
    ) -> AuditEvent:
        """Record a guardrail result, deriving the diff automatically.

        Args:
            result: Anything carrying ``action``, ``name``/``guardrail``,
                ``rule``, ``reason`` and ``severity`` -- typically a
                :class:`~wardhook.guardrails.base.GuardrailResult`, but read
                structurally so results from other packages work too. A plain
                mapping is read by key, so one of ``wardhook-core``'s
                ``guardrail_events`` entries can be passed directly. Use
                :meth:`record_run` for a whole list of them.
            stage: ``input``, ``output`` or ``tool_call``.
            run_id: Correlation id for the agent invocation.
            before: Text before the guardrail ran, for the diff.
            after: Text after. Defaults to ``result.text``.
            principal_id: The caller's identifier.
            tool: Tool involved, for tool-call events.

        Returns:
            The recorded event.
        """

        # A Mapping is read by key and everything else by attribute. Core's
        # `guardrail_events` are dicts, and reaching for record() with one is
        # the obvious mistake: getattr finds nothing on a dict, so every event
        # would degrade to an "allow" and then be dropped by record_allows,
        # leaving an empty audit trail and no error. record_run() is still the
        # right call for a whole list; this makes the single-event case safe.
        def read(name: str, default: Any) -> Any:
            if isinstance(result, Mapping):
                value = result.get(name, default)
                return default if value is None else value
            return getattr(result, name, default)

        action = read("action", "allow")
        action_value = getattr(action, "value", action)
        severity = read("severity", "low")
        severity_value = getattr(severity, "value", severity)
        details = dict(read("details", None) or {})

        diff: TextDiff | None = None
        if before is not None:
            resolved_after = after if after is not None else read("text", before)
            diff = diff_text(before, resolved_after, details.get("spans", ()))

        # A result object names itself `name`; an event dict names it
        # `guardrail`, which is the key core writes.
        guardrail = read("name", None) or read("guardrail", "guardrail")

        return self.log(
            guardrail=str(guardrail),
            action=str(action_value),
            stage=stage,
            run_id=run_id,
            rule=read("rule", None),
            severity=str(severity_value),
            reason=read("reason", None),
            principal_id=principal_id,
            tool=tool,
            diff=diff,
            details=details,
        )

    def record_run(
        self,
        guardrail_events: Iterable[Mapping[str, Any]],
        *,
        run_id: str,
        principal_id: str | None = None,
    ) -> list[AuditEvent]:
        """Record the ``guardrail_events`` list returned by an agent invocation.

        This is the bridge from ``wardhook-core``'s result dict into a durable
        trail, and it reads the events structurally so no import of core is
        needed in either direction.

        Args:
            guardrail_events: The ``guardrail_events`` entries from an agent
                result.
            run_id: The invocation's run id.
            principal_id: The caller's identifier.

        Returns:
            The recorded events.

        Example:
            >>> logger = AuditLogger()
            >>> events = logger.record_run(
            ...     [
            ...         {
            ...             "guardrail": "pii",
            ...             "action": "redact",
            ...             "stage": "output",
            ...             "rule": "EMAIL",
            ...             "severity": "medium",
            ...         }
            ...     ],
            ...     run_id="r1",
            ... )
            >>> events[0].guardrail, events[0].action
            ('pii', 'redact')
        """
        recorded: list[AuditEvent] = []
        for entry in guardrail_events:
            details = dict(entry.get("details") or {})
            recorded.append(
                self.log(
                    guardrail=str(entry.get("guardrail", "guardrail")),
                    action=str(entry.get("action", "allow")),
                    stage=str(entry.get("stage") or details.get("stage") or "unknown"),
                    run_id=run_id,
                    rule=entry.get("rule"),
                    severity=str(entry.get("severity", "low")),
                    reason=entry.get("reason"),
                    principal_id=principal_id,
                    tool=entry.get("tool"),
                    details=details,
                )
            )
        return recorded

    def report(self, events: Sequence[AuditEvent] | None = None) -> dict[str, Any]:
        """Summarise events for compliance review.

        Args:
            events: Events to summarise. Defaults to the retained in-memory
                events.

        Entity counts are read from each event's recorded diff, falling back to
        the guardrail's own details. A caller who does not pass ``before`` to
        :meth:`record` still gets a populated ``by_entity``.

        Returns:
            Counts by action, stage, guardrail, severity and entity, plus the
            number of distinct runs touched and the time span covered.
        """
        source = list(events) if events is not None else self.events

        def tally(key: Any) -> dict[str, int]:
            counts: dict[str, int] = {}
            for event in source:
                value = str(key(event))
                counts[value] = counts.get(value, 0) + 1
            return dict(sorted(counts.items()))

        # Entity counts come from the diff when one was recorded, and otherwise
        # from the event's own details. The guardrail always reports what it
        # matched; only the diff is conditional on the caller passing `before`.
        # Reading the diff alone left `by_entity` empty for every caller who
        # did not -- an empty tally on a run that redacted two entities reads
        # as a broken summary rather than an omitted argument.
        entities: dict[str, int] = {}
        for event in source:
            counted = (event.diff or {}).get("entities") or (event.details or {}).get("entities")
            for entity, count in (counted or {}).items():
                entities[entity] = entities.get(entity, 0) + int(count)

        timestamps = sorted(e.timestamp for e in source)
        return {
            "total_events": len(source),
            "runs": len({e.run_id for e in source if e.run_id}),
            "by_action": tally(lambda e: e.action),
            "by_stage": tally(lambda e: e.stage),
            "by_guardrail": tally(lambda e: e.guardrail),
            "by_severity": tally(lambda e: e.severity),
            "by_entity": dict(sorted(entities.items())),
            "first_event": timestamps[0] if timestamps else None,
            "last_event": timestamps[-1] if timestamps else None,
        }

    def read(self) -> Iterator[dict[str, Any]]:
        """Iterate the records already written to the log file.

        Malformed lines are skipped rather than raising. A trail truncated by a
        crash should still be readable up to the damage, and refusing to parse
        the whole file because of one bad byte destroys more evidence than it
        protects.

        Yields:
            Each parsed record, in file order.

        Raises:
            ValueError: If this logger has no file path.
        """
        if self.path is None:
            raise ValueError("This AuditLogger is in-memory only; it has no file to read.")
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def __repr__(self) -> str:
        """Return a debug representation."""
        target = str(self.path) if self.path else "memory"
        return f"AuditLogger(path={target!r}, events={len(self._events)})"
