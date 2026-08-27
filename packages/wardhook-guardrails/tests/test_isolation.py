"""Guards the promise that this package stands alone.

wardhook-guardrails must be installable and useful with no other Wardhook
package present, and with no LangChain or LangGraph in the environment. That is
easy to state and easy to break with one convenient import, so it is checked
mechanically rather than by review.

The check is a static scan of the source rather than an import-time assertion,
because the development environment has all four packages installed -- an
import of a sibling would succeed there and the violation would go unnoticed
until someone installed this package on its own.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import wardhook.guardrails

PACKAGE_ROOT = Path(wardhook.guardrails.__file__).parent
SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))

FORBIDDEN_PREFIXES = (
    "wardhook.core",
    "wardhook.observability",
    "wardhook.evals",
    "langchain",
    "langgraph",
    "numpy",
    "pandas",
    "torch",
    "spacy",
    "transformers",
)


def _imported_modules(path: Path) -> set[str]:
    """Return every module name imported by a source file.

    Args:
        path: The file to parse.

    Returns:
        Dotted module names from both ``import x`` and ``from x import y``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_the_scan_actually_finds_source_files():
    assert len(SOURCE_FILES) >= 7, f"expected the package's modules, found {SOURCE_FILES}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_module_imports_a_sibling_or_a_heavy_dependency(path):
    offenders = {
        module
        for module in _imported_modules(path)
        if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    }
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. wardhook-guardrails must work "
        f"with nothing else from Wardhook installed, and without an ML stack."
    )


def test_the_only_third_party_runtime_dependency_is_pyyaml():
    third_party: set[str] = set()
    for path in SOURCE_FILES:
        for module in _imported_modules(path):
            root = module.split(".")[0]
            if root in sys.stdlib_module_names or root == "wardhook":
                continue
            third_party.add(root)
    assert third_party <= {"yaml"}, (
        f"unexpected third-party imports: {sorted(third_party - {'yaml'})}. "
        f"This package's dependency list is deliberately just PyYAML."
    )


def test_guardrails_satisfy_the_core_contract_without_importing_it():
    # Core reads guardrails structurally. Assert the shape here so a change to
    # a hook signature fails in this package's own suite, not only downstream.
    from wardhook.guardrails import InjectionDetector, PIIRedactor, RoleBasedToolPolicy

    guardrails = [
        PIIRedactor(),
        InjectionDetector(),
        RoleBasedToolPolicy({"agent": ["lookup_*"]}),
    ]
    context = {"run_id": "r1", "stage": "input", "node": "guard_input", "principal": None}

    for guardrail in guardrails:
        assert isinstance(guardrail.name, str) and guardrail.name
        for hook, args in (
            ("on_input", ("text",)),
            ("on_output", ("text",)),
            ("on_tool_call", ("tool", {})),
        ):
            result = getattr(guardrail, hook)(*args, context)
            assert hasattr(result, "action"), f"{guardrail.name}.{hook} result has no .action"
            assert str(getattr(result.action, "value", result.action)) in {
                "allow",
                "redact",
                "block",
            }
