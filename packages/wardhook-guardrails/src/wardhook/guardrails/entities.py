r"""Config-driven entity packs, tuned per regulated domain.

"PII" is not one list. An insurance carrier cares about policy and claim
numbers; a hospital cares about medical record numbers and NHS numbers; a
fintech cares about IBANs and card numbers. Hard-coding a single global list
means every deployment either over-redacts or misses what actually matters.

An :class:`EntityPack` is a named set of :class:`EntityRule` definitions,
loadable from YAML or built in code. Four packs ship: ``default``,
``insurance``, ``healthcare`` and ``fintech``. Each domain pack extends
``default`` rather than replacing it, so an email address is still caught
whichever pack you choose.

Example:
    >>> pack = get_pack("fintech")
    >>> "IBAN" in pack.entity_names() and "EMAIL" in pack.entity_names()
    True

    >>> custom = EntityPack.from_dict(
    ...     {
    ...         "name": "internal",
    ...         "extends": "default",
    ...         "rules": [
    ...             {
    ...                 "entity": "EMPLOYEE_ID",
    ...                 "pattern": r"EMP-\d{6}",
    ...                 "severity": "medium",
    ...             }
    ...         ],
    ...     }
    ... )
    >>> "EMPLOYEE_ID" in {r.entity for r in custom.rules}
    True
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wardhook.guardrails.base import Severity

__all__ = [
    "BUILTIN_PACKS",
    "EntityPack",
    "EntityRule",
    "PackNotFoundError",
    "get_pack",
    "register_pack",
]

_PACKS_DIR = Path(__file__).parent / "packs"


class PackNotFoundError(KeyError):
    """Raised when a named entity pack cannot be resolved."""


@dataclass(frozen=True, slots=True)
class EntityRule:
    """One detectable entity type and how to recognise it.

    Attributes:
        entity: Stable identifier, upper snake case by convention, such as
            ``US_SSN``. Appears in audit records as the rule that fired.
        pattern: Regular expression matching the entity. Should include word
            boundaries where appropriate; the raw pattern is used unmodified.
        severity: Risk level recorded on detections.
        description: Human-readable explanation for documentation and review.
        validator: Name of a checksum validator to confirm a match, one of
            ``luhn``, ``iban`` or ``nhs``. Regex alone matches anything
            card-shaped; a checksum is what separates a real card number from
            sixteen arbitrary digits.
        replacement: Template used when redacting. ``{entity}`` is substituted.
        context_words: Words that must appear nearby for a match to count.
            Used for entities whose shape is too generic to stand alone, such
            as a bare six-digit medical record number.
    """

    entity: str
    pattern: str
    severity: Severity = Severity.MEDIUM
    description: str = ""
    validator: str | None = None
    replacement: str = "[{entity}]"
    context_words: tuple[str, ...] = ()

    def compiled(self) -> re.Pattern[str]:
        """Return the compiled, case-insensitive pattern.

        Returns:
            The compiled regex.

        Raises:
            ValueError: If the pattern does not compile, naming the entity so a
                broken custom pack is traceable to its rule.
        """
        try:
            return re.compile(self.pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"Rule {self.entity!r} has an invalid regex {self.pattern!r}: {exc}"
            ) from exc

    def redaction(self) -> str:
        """Return the placeholder that replaces a match.

        Returns:
            The rendered replacement, for example ``[US_SSN]``.
        """
        return self.replacement.format(entity=self.entity)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EntityRule:
        """Build a rule from a mapping, as loaded from YAML.

        Args:
            data: Mapping with at least ``entity`` and ``pattern``.

        Returns:
            The rule.

        Raises:
            ValueError: If a required key is missing or the severity is unknown.
        """
        for required in ("entity", "pattern"):
            if not data.get(required):
                raise ValueError(f"Entity rule is missing required key {required!r}: {dict(data)}")
        raw_severity = str(data.get("severity", "medium")).lower()
        try:
            severity = Severity(raw_severity)
        except ValueError as exc:
            valid = ", ".join(s.value for s in Severity)
            raise ValueError(
                f"Rule {data['entity']!r} has unknown severity {raw_severity!r}. Use one of: {valid}."
            ) from exc
        return cls(
            entity=str(data["entity"]),
            pattern=str(data["pattern"]),
            severity=severity,
            description=str(data.get("description", "")),
            validator=(str(data["validator"]) if data.get("validator") else None),
            replacement=str(data.get("replacement", "[{entity}]")),
            context_words=tuple(str(w).lower() for w in (data.get("context_words") or ())),
        )


@dataclass(frozen=True, slots=True)
class EntityPack:
    """A named collection of entity rules for one domain.

    Attributes:
        name: The pack's identifier.
        rules: The rules it contains.
        description: What the pack is for.
    """

    name: str
    rules: tuple[EntityRule, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        """Validate every rule's regex at construction time.

        Raises:
            ValueError: If any rule has an invalid pattern. Failing here beats
                failing on the first request that happens to reach the rule.
        """
        for rule in self.rules:
            rule.compiled()

    def entity_names(self) -> list[str]:
        """Return the entity identifiers in this pack, sorted."""
        return sorted(r.entity for r in self.rules)

    def filter(
        self, *, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None
    ) -> EntityPack:
        """Return a narrowed copy of this pack.

        Args:
            include: Keep only these entity names. ``None`` keeps all.
            exclude: Drop these entity names.

        Returns:
            A new pack. The original is unchanged.

        Example:
            >>> get_pack("default").filter(include=["EMAIL"]).entity_names()
            ['EMAIL']
        """
        kept = self.rules
        if include is not None:
            wanted = {e.upper() for e in include}
            kept = tuple(r for r in kept if r.entity.upper() in wanted)
        if exclude is not None:
            unwanted = {e.upper() for e in exclude}
            kept = tuple(r for r in kept if r.entity.upper() not in unwanted)
        return EntityPack(name=self.name, rules=kept, description=self.description)

    def merge(self, other: EntityPack, *, name: str | None = None) -> EntityPack:
        """Combine two packs, with ``other`` winning on conflicting entities.

        Args:
            other: The pack whose rules take precedence.
            name: Name for the result. Defaults to ``other``'s name.

        Returns:
            The merged pack.
        """
        by_entity = {r.entity: r for r in self.rules}
        by_entity.update({r.entity: r for r in other.rules})
        return EntityPack(
            name=name or other.name,
            rules=tuple(by_entity.values()),
            description=other.description or self.description,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EntityPack:
        """Build a pack from a mapping, as loaded from YAML.

        Args:
            data: Mapping with ``name``, optional ``extends``, and ``rules``.
                ``extends`` names a pack to inherit from, so a domain pack does
                not have to restate universal entities such as email addresses.

        Returns:
            The pack, with any inherited rules already merged in.

        Raises:
            ValueError: If ``name`` is missing or a rule is malformed.
            PackNotFoundError: If ``extends`` names an unknown pack.
        """
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError(f"Entity pack is missing a 'name': {dict(data)}")

        rules = tuple(EntityRule.from_dict(r) for r in (data.get("rules") or ()))
        pack = cls(name=name, rules=rules, description=str(data.get("description", "")))

        parent_name = data.get("extends")
        if parent_name:
            return get_pack(str(parent_name)).merge(pack, name=name)
        return pack

    @classmethod
    def from_yaml(cls, path: str | Path) -> EntityPack:
        """Load a pack from a YAML file.

        Args:
            path: Path to the YAML definition.

        Returns:
            The loaded pack.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a YAML mapping, or a rule is invalid.
        """
        import yaml

        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"No such entity pack file: {resolved}")
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"{resolved} must contain a YAML mapping, got {type(payload).__name__}"
            )
        return cls.from_dict(payload)


_REGISTRY: dict[str, EntityPack] = {}
BUILTIN_PACKS: tuple[str, ...] = ("default", "insurance", "healthcare", "fintech")
"""Names of the packs that ship with this package."""


def register_pack(pack: EntityPack, *, overwrite: bool = False) -> None:
    """Add a pack to the global registry so it can be fetched by name.

    Args:
        pack: The pack to register.
        overwrite: Allow replacing an existing pack of the same name.

    Raises:
        ValueError: If the name is taken and ``overwrite`` is ``False``.
            Silently replacing a pack would change redaction behaviour across
            an entire process with no signal.
    """
    if pack.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"An entity pack named {pack.name!r} is already registered. "
            f"Pass overwrite=True to replace it."
        )
    _REGISTRY[pack.name] = pack


def get_pack(pack: str | EntityPack | Sequence[EntityRule] | None = None) -> EntityPack:
    """Resolve a pack from a name, an instance, or a list of rules.

    Args:
        pack: One of:

            * ``None`` -- the ``default`` pack.
            * A name -- a registered pack, or a built-in loaded on first use.
            * An :class:`EntityPack` -- returned unchanged.
            * A sequence of :class:`EntityRule` -- wrapped in an ad-hoc pack.

    Returns:
        The resolved pack.

    Raises:
        PackNotFoundError: If a name matches no registered or built-in pack.

    Example:
        >>> get_pack("healthcare").name
        'healthcare'
        >>> get_pack().name
        'default'
    """
    if pack is None:
        return get_pack("default")
    if isinstance(pack, EntityPack):
        return pack
    if not isinstance(pack, str):
        return EntityPack(name="custom", rules=tuple(pack))

    if pack in _REGISTRY:
        return _REGISTRY[pack]

    candidate = _PACKS_DIR / f"{pack}.yaml"
    if candidate.exists():
        # Built-ins load lazily and are cached, so importing this module does
        # not read four YAML files you may never use.
        loaded = EntityPack.from_yaml(candidate)
        _REGISTRY[pack] = loaded
        return loaded

    known = sorted(set(_REGISTRY) | set(BUILTIN_PACKS))
    raise PackNotFoundError(
        f"No entity pack named {pack!r}. Known packs: {', '.join(known)}. "
        f"Register your own with register_pack(), or load one from YAML with "
        f"EntityPack.from_yaml()."
    )
