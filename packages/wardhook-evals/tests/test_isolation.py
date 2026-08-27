"""Guards the promise that this package stands alone.

wardhook-evals must be installable and useful with no other Wardhook package
present, and with no agent framework at all. That is easy to state and easy to
break with one convenient import, so it is checked mechanically rather than by
review.

The check is a static AST scan rather than an import-time assertion, because
the development environment has all four packages installed -- an import of a
sibling would succeed there and the violation would go unnoticed until someone
installed this package on its own.

The strongest property here is the last test: the *only* third-party import in
the whole package is its CLI library. Even the LLM-graded criterion takes a
duck-typed model object rather than importing one, so evaluating an agent never
drags in the framework that agent was built with.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import wardhook.evals

PACKAGE_ROOT = Path(wardhook.evals.__file__).parent
SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))

FORBIDDEN_PREFIXES = (
    "wardhook.core",
    "wardhook.guardrails",
    "wardhook.observability",
    "langchain",
    "langchain_core",
    "langgraph",
    "numpy",
    "pandas",
    "torch",
)
# `click` is typer's own dependency and may legitimately be imported directly.
ALLOWED_THIRD_PARTY = {"typer", "click"}


def _imported_modules(path: Path) -> set[str]:
    """Return every module name imported by a source file.

    Args:
        path: The file to parse.

    Returns:
        Dotted module names from both ``import x`` and ``from x import y``.
        ``ast.walk`` descends into function bodies, so an import hidden inside
        a lazily-called function is caught too.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_the_package_imports_and_declares_a_version():
    assert wardhook.evals.__version__


def test_the_scan_actually_finds_source_files():
    # A real floor, not a truthiness check: if the scan silently stopped
    # finding modules, every test below would pass while checking nothing.
    assert len(SOURCE_FILES) >= 7, f"expected the package's modules, found {SOURCE_FILES}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_module_imports_a_sibling_or_an_agent_framework(path):
    offenders = {
        module
        for module in _imported_modules(path)
        if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    }
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. wardhook-evals must be able to "
        f"evaluate an agent without depending on what that agent was built with."
    )


def test_third_party_imports_stay_within_the_declared_dependencies():
    third_party: set[str] = set()
    for path in SOURCE_FILES:
        for module in _imported_modules(path):
            root = module.split(".")[0]
            if root in sys.stdlib_module_names or root == "wardhook":
                continue
            third_party.add(root)
    unexpected = third_party - ALLOWED_THIRD_PARTY
    assert not unexpected, (
        f"unexpected third-party imports: {sorted(unexpected)}. Add the "
        f"dependency to pyproject.toml deliberately, or drop the import."
    )


def test_the_llm_judge_criterion_needs_no_model_library():
    # The `judge` extra is a convenience for users who do not already have a
    # model library -- it must never be a hard requirement. Grading works
    # against any object with .invoke().
    from wardhook.evals import Outcome, get_criterion

    class Judge:
        def invoke(self, prompt):
            return "PASS - looks fine"

    result = get_criterion("llm_judge")("Is it polite?", Outcome(text="hi", judge=Judge()))
    assert result.passed


def test_the_runner_evaluates_a_target_built_from_nothing():
    # The end-to-end isolation claim: a plain object, no framework, no siblings.
    from wardhook.evals import EvalCase, EvalRunner

    class Bare:
        def invoke(self, text):
            return {"output": f"echo: {text}"}

    report = EvalRunner(Bare()).run([EvalCase("a", "hi", {"contains": "echo: hi"})])
    assert report.ok
