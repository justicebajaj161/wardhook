"""Wardhook guardrails: PII redaction, injection detection, RBAC, and audit logs.

Four independent policies that inspect what goes into and comes out of an LLM
agent, plus a compliance-grade audit trail recording every decision.

This package has **no dependency on the rest of Wardhook** -- not on
``wardhook-core``, not on LangChain, not on LangGraph. Its only runtime
dependency is PyYAML, for loading entity packs. Everything here is usable in a
plain function, a Flask view, or a batch job, not just inside an agent.

When ``wardhook-core`` *is* present, these classes satisfy its structural
guardrail contract and can be handed straight to ``AgentGraph``:

    >>> from wardhook.guardrails import PIIRedactor, RoleBasedToolPolicy
    >>> guardrails = [
    ...     PIIRedactor(pack="insurance"),
    ...     RoleBasedToolPolicy({"agent": ["lookup_*"]}),
    ... ]
    >>> [g.name for g in guardrails]
    ['pii-redactor', 'rbac-tool-policy']

Standalone, on any text at all:

    >>> PIIRedactor().redact("Reach me at alice@example.com").text
    'Reach me at [EMAIL]'
"""

from wardhook.guardrails.audit import AuditEvent, AuditLogger, TextDiff, diff_text
from wardhook.guardrails.base import Action, BaseGuardrail, GuardrailResult, Severity
from wardhook.guardrails.entities import (
    BUILTIN_PACKS,
    EntityPack,
    EntityRule,
    PackNotFoundError,
    get_pack,
    register_pack,
)
from wardhook.guardrails.injection import (
    INJECTION_SIGNALS,
    InjectionDetector,
    InjectionReport,
    SignalCategory,
)
from wardhook.guardrails.pii import PIIDetector, PIIMatch, PIIRedactor, RedactionResult
from wardhook.guardrails.rbac import RoleBasedToolPolicy, ToolPermission

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_PACKS",
    "INJECTION_SIGNALS",
    "Action",
    "AuditEvent",
    "AuditLogger",
    "BaseGuardrail",
    "EntityPack",
    "EntityRule",
    "GuardrailResult",
    "InjectionDetector",
    "InjectionReport",
    "PIIDetector",
    "PIIMatch",
    "PIIRedactor",
    "PackNotFoundError",
    "RedactionResult",
    "RoleBasedToolPolicy",
    "Severity",
    "SignalCategory",
    "TextDiff",
    "ToolPermission",
    "__version__",
    "diff_text",
    "get_pack",
    "register_pack",
]
