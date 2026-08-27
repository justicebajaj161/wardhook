"""Render traces to one self-contained HTML page.

The whole viewer is a single string of HTML with its CSS and JavaScript inlined.
There is no CDN link, no web font, no image request, and no server to run: the
file opens from a local path, from an email attachment, or from inside a CI
artifact, and behaves identically offline. That is a deliberate constraint --
a debugging tool that needs the network is useless in exactly the locked-down
environment where you most need to debug something.

.. warning::
   **Every interpolated value is escaped, and that is a security control.**
   Traces carry node names, model ids, error text, and caller-supplied metadata,
   all of which can contain characters that are markup. If a rendered trace is
   ever served over HTTP rather than opened locally, unescaped content is stored
   cross-site scripting. :func:`escape` is applied at every interpolation point
   and there is a test asserting a ``<script>`` tag survives the round trip
   inert.

Example:
    >>> from wardhook.observability.models import Trace
    >>> page = render_html(Trace("run-1"))
    >>> page.startswith("<!doctype html>")
    True
    >>> "run-1" in page
    True
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from wardhook.observability.models import Trace
from wardhook.observability.pricing import PRICES_AS_OF

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wardhook.observability.models import TraceStep

__all__ = ["render_html"]

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1a1d23; --muted: #6b7280; --line: #e5e7eb;
  --panel: #f9fafb; --bar: #3b82f6; --bar-soft: #dbeafe;
  --err-fg: #991b1b; --err-bg: #fef2f2; --err-line: #fecaca;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa1ab; --line: #262a31;
    --panel: #161a20; --bar: #60a5fa; --bar-soft: #1e3a5f;
    --err-fg: #fca5a5; --err-bg: #2a1416; --err-line: #7f1d1d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
.sub { color: var(--muted); font-size: .85rem; margin-bottom: 1.75rem; }
.run { border: 1px solid var(--line); border-radius: 10px; margin-bottom: 1.5rem; overflow: hidden; }
.run > summary {
  cursor: pointer; padding: .85rem 1rem; background: var(--panel);
  font-weight: 600; display: flex; gap: .75rem; align-items: baseline; flex-wrap: wrap;
}
.run > summary::-webkit-details-marker { display: none; }
.rid { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .9rem; }
.tags { margin-left: auto; font-weight: 400; color: var(--muted); font-size: .82rem; }
.body { padding: 1rem; }
.kpis { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 1rem; }
.kpi .v { font-size: 1.25rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .k { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; }
table { width: 100%; border-collapse: collapse; font-size: .87rem; }
th {
  text-align: left; font-size: .7rem; text-transform: uppercase; letter-spacing: .05em;
  color: var(--muted); padding: .4rem .5rem; border-bottom: 1px solid var(--line); font-weight: 600;
}
td { padding: .45rem .5rem; border-bottom: 1px solid var(--line); vertical-align: middle; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.failed td { background: var(--err-bg); color: var(--err-fg); }
.node { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.track { background: var(--bar-soft); border-radius: 3px; height: 9px; min-width: 90px; }
.fill { background: var(--bar); border-radius: 3px; height: 9px; }
.err {
  margin-top: .75rem; padding: .6rem .75rem; border: 1px solid var(--err-line);
  border-radius: 6px; background: var(--err-bg); color: var(--err-fg); font-size: .85rem;
}
.meta { margin-top: .85rem; color: var(--muted); font-size: .8rem; }
.meta code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
footer { color: var(--muted); font-size: .78rem; margin-top: 2rem; border-top: 1px solid var(--line); padding-top: .85rem; }
.empty { color: var(--muted); padding: 2rem; text-align: center; border: 1px dashed var(--line); border-radius: 10px; }
"""

# Expands or collapses every run at once. Deliberately trivial: it only ever
# toggles a boolean property on elements it selects itself, and never touches
# innerHTML, so no trace content can reach the DOM as markup through it.
_SCRIPT = """
document.addEventListener('click', function (event) {
  var button = event.target.closest('[data-toggle]');
  if (!button) return;
  var open = button.getAttribute('data-toggle') === 'open';
  document.querySelectorAll('details.run').forEach(function (node) { node.open = open; });
});
"""


def _fmt_int(value: int) -> str:
    """Format an integer with thousands separators.

    Args:
        value: The number to format.

    Returns:
        The formatted string, e.g. ``"12,400"``.
    """
    return f"{value:,}"


def _fmt_cost(value: float) -> str:
    """Format a dollar amount, keeping small costs legible.

    Args:
        value: US dollars.

    Returns:
        A string with enough decimal places to be meaningful. Sub-cent costs
        are common per node, and rounding them to two places would render an
        entire trace as ``$0.00``.
    """
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.5f}"
    return f"${value:.4f}"


def _fmt_ms(value: float) -> str:
    """Format a millisecond duration.

    Args:
        value: Milliseconds.

    Returns:
        Milliseconds under a second, otherwise seconds.
    """
    return f"{value:.0f}ms" if value < 1000 else f"{value / 1000:.2f}s"


def _step_row(step: TraceStep, longest_ms: float) -> str:
    """Render one step as a table row.

    Args:
        step: The step to render.
        longest_ms: Latency of the slowest step in the run, used to scale the
            bar. Relative bars make the bottleneck visible at a glance in a way
            a column of numbers does not.

    Returns:
        An HTML ``<tr>``.
    """
    width = (step.latency_ms / longest_ms * 100) if longest_ms > 0 else 0
    cached = step.usage.cache_read_tokens
    cache_cell = _fmt_int(cached) if cached else "&mdash;"
    row_class = ' class="failed"' if step.failed else ""
    error_title = f' title="{escape(step.error or "", quote=True)}"' if step.failed else ""
    return (
        f"<tr{row_class}{error_title}>"
        f'<td class="node">{escape(step.node)}</td>'
        f'<td><div class="track"><div class="fill" style="width:{width:.1f}%"></div></div></td>'
        f'<td class="num">{_fmt_ms(step.latency_ms)}</td>'
        f'<td class="num">{_fmt_int(step.tokens_in)}</td>'
        f'<td class="num">{_fmt_int(step.tokens_out)}</td>'
        f'<td class="num">{cache_cell}</td>'
        f'<td class="num">{_fmt_cost(step.cost)}</td>'
        f"<td>{escape(step.model or '&mdash;') if step.model else '&mdash;'}</td>"
        f"</tr>"
    )


def _render_run(trace: Trace, index: int) -> str:
    """Render one trace as a collapsible section.

    Args:
        trace: The trace to render.
        index: Position in the file. The first run is expanded by default so
            the page is useful without a click.

    Returns:
        An HTML ``<details>`` block.
    """
    usage = trace.total_usage
    longest = max((step.latency_ms for step in trace.steps), default=0.0)
    rows = "".join(_step_row(step, longest) for step in trace.steps) or (
        '<tr><td colspan="8" class="node">no steps recorded</td></tr>'
    )

    kpis = [
        ("steps", str(len(trace.steps))),
        ("latency", _fmt_ms(trace.latency_ms)),
        ("tokens in", _fmt_int(usage.input_tokens)),
        ("tokens out", _fmt_int(usage.output_tokens)),
        ("cached", _fmt_int(usage.cache_read_tokens)),
        ("cost", _fmt_cost(trace.total_cost)),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{escape(value)}</div>'
        f'<div class="k">{escape(key)}</div></div>'
        for key, value in kpis
    )

    error_html = (
        f'<div class="err"><strong>Run failed.</strong> {escape(trace.error)}</div>'
        if trace.error
        else ""
    )
    meta_html = ""
    if trace.metadata:
        pairs = " &middot; ".join(
            f"<code>{escape(str(key))}</code> {escape(str(value))}"
            for key, value in sorted(trace.metadata.items())
        )
        meta_html = f'<div class="meta">{pairs}</div>'

    status = " &middot; failed" if trace.failed else ""
    return (
        f'<details class="run"{" open" if index == 0 else ""}>'
        f'<summary><span class="rid">{escape(trace.run_id or "(no run id)")}</span>'
        f'<span class="tags">{escape(trace.started_at)}{status}</span></summary>'
        f'<div class="body">'
        f'<div class="kpis">{kpi_html}</div>'
        f"<table><thead><tr>"
        f"<th>node</th><th>latency</th><th>ms</th><th>in</th><th>out</th>"
        f"<th>cached</th><th>cost</th><th>model</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        f"{error_html}{meta_html}"
        f"</div></details>"
    )


def render_html(traces: Trace | Sequence[Trace], *, title: str = "Wardhook trace") -> str:
    """Render one or more traces to a complete, self-contained HTML document.

    Args:
        traces: A single :class:`~wardhook.observability.models.Trace` or a
            sequence of them.
        title: Document title, shown in the browser tab and as the heading.

    Returns:
        A full HTML document as a string. It references nothing external, so
        writing it to a file is all that is needed to view it.

    Example:
        >>> from wardhook.observability.models import Trace
        >>> "http://" in render_html(Trace("r1"))
        False
    """
    items = [traces] if isinstance(traces, Trace) else list(traces)

    total_cost = sum(trace.total_cost for trace in items)
    total_out = sum(trace.total_tokens_out for trace in items)
    failed = sum(1 for trace in items if trace.failed)

    if items:
        summary = (
            f"{len(items)} run{'s' if len(items) != 1 else ''} &middot; "
            f"{_fmt_int(total_out)} output tokens &middot; {_fmt_cost(total_cost)} total"
        )
        if failed:
            summary += f" &middot; {failed} failed"
        body = "".join(_render_run(trace, index) for index, trace in enumerate(items))
    else:
        summary = "no runs recorded"
        body = '<div class="empty">This trace file is empty.</div>'

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body><div class='wrap'>"
        f"<h1>{escape(title)}</h1>"
        f'<div class="sub">{summary}</div>'
        f"{body}"
        f"<footer>Generated by wardhook-observability. Costs are estimates from a price "
        f"table current as of {escape(PRICES_AS_OF)}; unpriced models are shown as $0. "
        f"This page makes no network requests.</footer>"
        f"</div><script>{_SCRIPT}</script></body></html>"
    )
