"""Wardhook observability: per-node tokens, cost, and latency for LangGraph agents.

.. note::
   **This package is in progress.** The public API sketched in the package
   README is the design contract; the implementation follows. Only
   :data:`__version__` is exported today.

When complete, ``Tracer`` will satisfy the ``TelemetryProtocol`` declared in
``wardhook.core.protocols``, so ``AgentGraph(telemetry=True)`` picks it up
without either package importing the other.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
