"""The trace viewer: a self-contained HTML page and the CLI that writes it.

Kept in its own subpackage because it is the only part of
``wardhook-observability`` that depends on ``typer``, and the only part
concerned with presentation rather than measurement.

:func:`~wardhook.observability.viewer.html.render_html` is re-exported from the
top-level ``wardhook.observability`` namespace, so importing from here is
optional.
"""

from wardhook.observability.viewer.html import render_html

__all__ = ["render_html"]
