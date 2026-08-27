"""Guards the promise that this package stands alone.

wardhook-evals must be installable and useful with no other Wardhook
package present. That is easy to state and easy to break with one convenient
import, so it is checked mechanically rather than by review.

The check is a static AST scan rather than an import-time assertion, because
the development environment has all four packages installed -- an import of a
sibling would succeed there and the violation would go unnoticed until someone
installed this package on its own.

These tests are meaningful while the package is still being implemented: the
dependency boundary is the property most easily broken during the build.
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
    "langgraph",
    "numpy",
    "pandas",
    "torch",
)
ALLOWED_THIRD_PARTY = {"typer", "click"}


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


def test_the_package_imports_and_declares_a_version():
    assert wardhook.evals.__version__


def test_the_scan_actually_finds_source_files():
    assert SOURCE_FILES, "expected at least one module to scan"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_module_imports_a_sibling_or_a_heavy_dependency(path):
    offenders = {
        module
        for module in _imported_modules(path)
        if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    }
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. wardhook-evals must work "
        f"with nothing else from Wardhook installed."
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
