"""Role-based access control for tool calls.

An agent's tools are its blast radius. A support agent that can read a claim
and one that can issue a refund are the same code path with different
permissions, and the decision about which a given caller gets belongs outside
the model's judgement.

:class:`RoleBasedToolPolicy` maps roles to the tool patterns they may invoke and
evaluates each call against the ``principal`` carried in run context. It is
**deny-by-default**: a tool no role grants is denied, and a caller with no
recognised role gets nothing. Opening that up is possible but has to be an
explicit choice.

Patterns use shell-style globbing, so ``claims.*`` grants a whole namespace and
``*`` grants everything.

Example:
    >>> policy = RoleBasedToolPolicy(
    ...     {"agent": ["lookup_*"], "supervisor": ["lookup_*", "issue_refund"]}
    ... )
    >>> ctx = {"principal": {"id": "u1", "roles": ["agent"]}}
    >>> policy.on_tool_call("lookup_claim", {}, ctx).allowed
    True
    >>> policy.on_tool_call("issue_refund", {"amount": 500}, ctx).blocked
    True
    >>> policy.on_tool_call("issue_refund", {}, {"principal": {"roles": ["supervisor"]}}).allowed
    True
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any

from wardhook.guardrails.base import BaseGuardrail, GuardrailResult, Severity

__all__ = ["RoleBasedToolPolicy", "ToolPermission"]


class ToolPermission:
    """A compiled set of tool-name patterns granted to one role.

    Args:
        role: The role's name.
        patterns: Shell-style glob patterns of permitted tool names.
        deny: Patterns explicitly denied to this role. Denials win over grants,
            so a role can hold a broad grant with a carve-out.
    """

    __slots__ = ("deny", "patterns", "role")

    def __init__(
        self, role: str, patterns: Iterable[str], deny: Iterable[str] | None = None
    ) -> None:
        """Initialise a permission set. See the class docstring."""
        self.role = role
        self.patterns = tuple(patterns)
        self.deny = tuple(deny or ())

    def permits(self, tool_name: str) -> bool:
        """Check whether this role may call ``tool_name``.

        Args:
            tool_name: The tool the model asked to call.

        Returns:
            ``True`` if a grant matches and no denial does.
        """
        if any(fnmatchcase(tool_name, pattern) for pattern in self.deny):
            return False
        return any(fnmatchcase(tool_name, pattern) for pattern in self.patterns)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"ToolPermission(role={self.role!r}, allow={self.patterns}, deny={self.deny})"


class RoleBasedToolPolicy(BaseGuardrail):
    """Denies tool calls the caller's roles do not permit.

    Args:
        roles: Mapping of role name to permitted tool patterns. A value may be
            a list of patterns, or a mapping with ``allow`` and ``deny`` keys.
        default_roles: Roles assumed when the principal declares none. Empty by
            default, which means an unidentified caller can call nothing.
        allow_unlisted: Permit tools that no role's patterns mention. Defaults
            to ``False``. Leaving it ``False`` means adding a tool to an agent
            does not silently widen anyone's permissions -- a new tool is
            unreachable until a role explicitly grants it.
        allow_anonymous: Permit calls when run context carries no principal at
            all. Defaults to ``False``, so forgetting to pass a principal fails
            closed rather than granting unrestricted access.
        name: Identifier recorded in audit records.

    Raises:
        ValueError: If ``roles`` is empty, or a role's value is malformed.

    Example:
        >>> policy = RoleBasedToolPolicy(
        ...     {"admin": {"allow": ["*"], "deny": ["delete_*"]}},
        ... )
        >>> ctx = {"principal": {"roles": ["admin"]}}
        >>> policy.on_tool_call("read_ledger", {}, ctx).allowed
        True
        >>> policy.on_tool_call("delete_ledger", {}, ctx).blocked
        True
    """

    def __init__(
        self,
        roles: Mapping[str, Sequence[str] | Mapping[str, Sequence[str]]],
        *,
        default_roles: Sequence[str] = (),
        allow_unlisted: bool = False,
        allow_anonymous: bool = False,
        name: str = "rbac-tool-policy",
    ) -> None:
        """Initialise the policy. See the class docstring for arguments."""
        super().__init__(name=name)
        if not roles:
            raise ValueError(
                "RoleBasedToolPolicy needs at least one role. An empty policy "
                "denies every tool call, which is almost never intended."
            )

        self.permissions: dict[str, ToolPermission] = {}
        for role, spec in roles.items():
            if isinstance(spec, Mapping):
                allow = spec.get("allow", ())
                deny = spec.get("deny", ())
            elif isinstance(spec, str):
                raise ValueError(
                    f"Role {role!r} maps to a bare string {spec!r}. Use a list of "
                    f"patterns, for example [{spec!r}]."
                )
            else:
                allow, deny = spec, ()
            self.permissions[role] = ToolPermission(role, allow, deny)

        self.default_roles = tuple(default_roles)
        self.allow_unlisted = allow_unlisted
        self.allow_anonymous = allow_anonymous
        self._all_patterns = tuple(
            pattern for perm in self.permissions.values() for pattern in perm.patterns
        )

    def roles_for(self, context: Mapping[str, Any]) -> tuple[str, ...]:
        """Extract the caller's roles from run context.

        Args:
            context: Run context, expected to carry a ``principal`` mapping
                with a ``roles`` list.

        Returns:
            The caller's roles, falling back to ``default_roles``.
        """
        principal = context.get("principal") or {}
        if not isinstance(principal, Mapping):
            return self.default_roles
        raw = principal.get("roles") or ()
        roles = tuple(str(r) for r in raw) if not isinstance(raw, str) else (raw,)
        return roles or self.default_roles

    def permitted_tools(self, roles: Iterable[str]) -> list[str]:
        """List the patterns granted to a set of roles.

        Args:
            roles: Role names.

        Returns:
            The union of their allow patterns, sorted and de-duplicated.
        """
        granted = {
            pattern
            for role in roles
            if (perm := self.permissions.get(role)) is not None
            for pattern in perm.patterns
        }
        return sorted(granted)

    def check(self, tool_name: str, context: Mapping[str, Any]) -> GuardrailResult:
        """Decide whether the caller may invoke ``tool_name``.

        Args:
            tool_name: The tool requested by the model.
            context: Run context carrying the principal.

        Returns:
            An allow or block result. Denials carry the caller's roles and the
            patterns they do hold, so an audit reviewer can see not just that
            access was denied but what the caller was actually entitled to.
        """
        principal = context.get("principal")
        if not principal:
            if self.allow_anonymous:
                return GuardrailResult.allow(tool_name, name=self.name)
            return GuardrailResult.block(
                tool_name,
                reason=f"no principal supplied; {tool_name!r} requires an identified caller",
                rule="anonymous-denied",
                name=self.name,
                severity=Severity.HIGH,
                details={"tool": tool_name, "roles": [], "anonymous": True},
            )

        roles = self.roles_for(context)
        if any(
            (perm := self.permissions.get(role)) is not None and perm.permits(tool_name)
            for role in roles
        ):
            return GuardrailResult.allow(tool_name, name=self.name)

        # A tool no role mentions at all is a different situation from one a
        # role is explicitly denied, and allow_unlisted only covers the former.
        mentioned = any(fnmatchcase(tool_name, pattern) for pattern in self._all_patterns)
        if not mentioned and self.allow_unlisted:
            return GuardrailResult.allow(tool_name, name=self.name)

        principal_id = (
            principal.get("id", "unknown") if isinstance(principal, Mapping) else "unknown"
        )
        return GuardrailResult.block(
            tool_name,
            reason=(f"role(s) {list(roles) or ['none']} are not permitted to call {tool_name!r}"),
            rule="rbac-denied",
            name=self.name,
            severity=Severity.HIGH,
            details={
                "tool": tool_name,
                "principal_id": principal_id,
                "roles": list(roles),
                "permitted_patterns": self.permitted_tools(roles),
                "tool_is_listed": mentioned,
            },
        )

    def on_tool_call(
        self,
        tool_name: str,
        tool_args: Mapping[str, Any],  # noqa: ARG002
        context: Mapping[str, Any],
    ) -> GuardrailResult:
        """Enforce the policy for one tool call.

        Args:
            tool_name: The tool requested by the model.
            tool_args: The arguments supplied. Not inspected -- this policy
                gates on identity, not on argument values.
            context: Run context carrying the principal.

        Returns:
            An allow or block result.
        """
        return self.check(tool_name, context)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            f"RoleBasedToolPolicy(name={self.name!r}, roles={sorted(self.permissions)}, "
            f"allow_unlisted={self.allow_unlisted}, allow_anonymous={self.allow_anonymous})"
        )
