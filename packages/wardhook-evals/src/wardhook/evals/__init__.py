"""Wardhook evals: JSONL test cases, a pass/fail runner, and regression detection.

.. note::
   **This package is in progress.** The public API sketched in the package
   README is the design contract; the implementation follows. Only
   :data:`__version__` is exported today.

When complete, the runner will target any object exposing ``.invoke()`` -- a
Wardhook agent, a raw LangGraph graph, or a plain function -- so this package
depends on nothing else in the project.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
