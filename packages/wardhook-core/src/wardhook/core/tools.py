"""Tool registration and normalisation helpers.

An agent's ``tools`` argument accepts a mix of shapes -- LangChain
:class:`~langchain_core.tools.BaseTool` instances, functions already decorated
with :func:`~langchain_core.tools.tool`, and plain Python callables. This module
flattens all of them into a single list of ``BaseTool`` so the rest of the
runtime, and any guardrail inspecting tool names, sees one consistent type.

Plain callables are wrapped automatically. Their docstring becomes the tool
description the model reads, so the docstring is load-bearing: an undocumented
callable is rejected rather than silently registered with an empty description
the model cannot act on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as make_tool

__all__ = ["ToolRegistrationError", "normalize_tools", "tool_names"]


class ToolRegistrationError(ValueError):
    """Raised when an object cannot be registered as an agent tool."""


def _wrap_callable(fn: Callable[..., Any]) -> BaseTool:
    """Wrap a plain callable as a LangChain tool.

    Args:
        fn: The function to expose to the model.

    Returns:
        The wrapped tool.

    Raises:
        ToolRegistrationError: If ``fn`` has no docstring, or LangChain cannot
            derive a schema from its signature.
    """
    name = getattr(fn, "__name__", repr(fn))
    if not (fn.__doc__ or "").strip():
        raise ToolRegistrationError(
            f"Tool {name!r} has no docstring. The docstring is what the model "
            f"reads to decide when to call the tool, so it is required. Add "
            f"one, or pass a pre-built BaseTool with an explicit description."
        )
    try:
        return make_tool(fn)
    except Exception as exc:
        raise ToolRegistrationError(
            f"Could not build a tool schema for {name!r}: {exc}. "
            f"Annotate every parameter with a type, or pass a pre-built BaseTool."
        ) from exc


def normalize_tools(tools: Iterable[Any] | None) -> list[BaseTool]:
    """Normalise mixed tool inputs into a list of ``BaseTool``.

    Args:
        tools: Any iterable of ``BaseTool`` instances and/or plain callables.
            ``None`` and empty iterables both yield an empty list.

    Returns:
        The normalised tools, in input order.

    Raises:
        ToolRegistrationError: If an entry is neither a tool nor a callable, or
            if two tools share a name. Duplicate names are rejected because the
            model addresses tools by name, so a collision would make one of the
            two silently unreachable.

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two integers.'''
        ...     return a + b
        >>> tools = normalize_tools([add])
        >>> tool_names(tools)
        ['add']
        >>> tools[0].description
        'Add two integers.'
    """
    if tools is None:
        return []

    normalized: list[BaseTool] = []
    seen: dict[str, str] = {}

    for entry in tools:
        if isinstance(entry, BaseTool):
            built = entry
        elif callable(entry):
            built = _wrap_callable(entry)
        else:
            raise ToolRegistrationError(
                f"Expected a BaseTool or a callable, got {type(entry).__name__}. "
                f"Wrap it with langchain_core.tools.tool() first."
            )

        if built.name in seen:
            raise ToolRegistrationError(
                f"Duplicate tool name {built.name!r}. Tools are addressed by "
                f"name, so two tools cannot share one."
            )
        seen[built.name] = built.name
        normalized.append(built)

    return normalized


def tool_names(tools: Sequence[BaseTool]) -> list[str]:
    """Return the names of ``tools``, in order.

    Args:
        tools: Normalised tools.

    Returns:
        Their names, suitable for logging or for an RBAC policy check.
    """
    return [t.name for t in tools]
