"""Shared pytest fixtures for the wardhook-guardrails test suite."""

from __future__ import annotations

import pytest

from wardhook.guardrails.audit import AuditLogger
from wardhook.guardrails.injection import InjectionDetector
from wardhook.guardrails.pii import PIIRedactor
from wardhook.guardrails.rbac import RoleBasedToolPolicy


@pytest.fixture
def redactor() -> PIIRedactor:
    """A redactor using the universal default pack."""
    return PIIRedactor()


@pytest.fixture
def injection() -> InjectionDetector:
    """An injection detector at its default threshold."""
    return InjectionDetector()


@pytest.fixture
def policy() -> RoleBasedToolPolicy:
    """A two-role policy: agents read, supervisors also refund."""
    return RoleBasedToolPolicy(
        {
            "agent": ["lookup_*", "search_*"],
            "supervisor": ["lookup_*", "search_*", "issue_refund"],
        }
    )


@pytest.fixture
def audit_log(tmp_path) -> AuditLogger:
    """A logger writing to a temporary JSONL file with a fixed salt."""
    return AuditLogger(tmp_path / "audit.jsonl", fingerprint_salt="test-salt")


@pytest.fixture
def ctx():
    """Return a factory for run-context mappings."""

    def _ctx(*roles: str, run_id: str = "run-1", principal_id: str = "u-1", **extra):
        return {
            "run_id": run_id,
            "stage": "tool_call",
            "node": "tools",
            "principal": {"id": principal_id, "roles": list(roles)},
            **extra,
        }

    return _ctx
