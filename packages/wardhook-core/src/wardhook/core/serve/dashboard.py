"""A read-only JSON API describing what an agent *is* and what it *cost*.

:func:`create_dashboard` builds a small FastAPI application exposing three
endpoints:

* ``GET /api/topology``      -- the agent's graph, read from its own configuration.
* ``GET /api/runs``          -- one summary per recorded run.
* ``GET /api/runs/{run_id}`` -- one run's per-node timing, tokens and cost.

**It shows telemetry and configuration. It never shows content.** No prompt, no
model output, no retrieved chunk, no guardrail event body reaches this API. That
is not a policy applied on top of the data -- the telemetry model has no such
fields in it, and the projection below is an explicit allowlist so that it stays
that way even if one is added upstream. Rendering agent output here would turn
Wardhook into a tool that redacts personal data from the audit log and then
serves it over HTTP.

The question this design keeps having to answer is *"a node cost $0.40, so show
me the prompt"*. The answer is :attr:`run_id`, which appears on every run and
every step, and on every audit record the caller writes. Correlating the two is
a lookup in **the caller's own audit log**, where their redaction policy, their
retention rules and their access controls already apply. This API hands over the
key; it does not keep a second, unredacted copy of the lock.

**Nothing is imported from a sibling package.** The telemetry sink is read
structurally, exactly as :class:`~wardhook.core.agent.AgentGraph` reads
guardrails, so ``wardhook-core`` still installs and passes entirely on its own.
Two sink shapes are understood, because the two that ship with Wardhook do not
agree on method names:

* ``Tracer`` lists with ``traces()`` and looks up with ``get_trace(run_id)``.
* ``JSONLTraceStore`` lists with ``read()`` and looks up with ``read_one(run_id)``.

Duck-typing only the first pair would silently break the very mitigation the
second pair exists to provide, so both are supported.

Which one was found decides the reported ``mode``, and the mode is reported
rather than hidden: an in-memory tracer under ``uvicorn --workers 4`` only ever
sees its own process's traffic, and an observability tool that quietly drops
three quarters of the data is worse than one that says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from wardhook.core.serve.app import _describe
from wardhook.core.serve.topology import read_topology, render_svg

__all__ = ["create_dashboard"]

_PAGE_TITLE = "Wardhook Dashboard"

# Method names understood on a telemetry sink, paired with the mode each implies.
# Order matters only in that the in-memory shape is checked first; the two sinks
# that ship with Wardhook expose one name each, never both.
_LIST_METHODS: tuple[tuple[str, str], ...] = (("traces", "memory"), ("read", "store"))
_LOOKUP_METHODS: tuple[str, ...] = ("get_trace", "read_one")

_MODE_NOTES: dict[str, str] = {
    "memory": (
        "In-memory tracer. This is one process's view: under multiple workers "
        "(uvicorn --workers N, gunicorn) each worker owns its own tracer, so "
        "roughly 1/N of traffic is visible here. Point the tracer at a shared "
        "JSONLTraceStore and pass that store as the dashboard's telemetry to "
        "see every run."
    ),
    "store": (
        "Shared trace store. Every run written to the file is visible here, "
        "including runs served by other worker processes."
    ),
    "none": (
        "No readable telemetry is attached, so no runs can be listed. Construct "
        "the agent with telemetry=True, or pass a sink to create_dashboard()."
    ),
}

# Largest page of runs the API will return in one response. A shared trace file
# is append-only and unbounded; returning all of it would make the endpoint
# unusable on exactly the deployment that needs it most.
_MAX_LIMIT = 1000
_DEFAULT_LIMIT = 100


# Copied from `wardhook.observability.viewer.html`, not imported from it.
# `wardhook-core` must install and pass with no sibling package present, and
# `make solo` is the check -- so the CSS is duplicated on purpose. That
# duplication is the price of independent installability, and it is a price
# worth paying: the alternative is a core package that cannot be installed
# without an observability package it does not otherwise need.
_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1a1d23; --muted: #6b7280; --line: #e5e7eb;
  --panel: #f9fafb; --bar: #3b82f6; --bar-soft: #dbeafe;
  --err-fg: #991b1b; --err-bg: #fef2f2; --err-line: #fecaca;
  --warn-fg: #92400e; --warn-bg: #fffbeb; --warn-line: #fde68a;
  --ok-fg: #065f46; --ok-bg: #ecfdf5; --ok-line: #a7f3d0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa1ab; --line: #262a31;
    --panel: #161a20; --bar: #60a5fa; --bar-soft: #1e3a5f;
    --err-fg: #fca5a5; --err-bg: #2a1416; --err-line: #7f1d1d;
    --warn-fg: #fcd34d; --warn-bg: #241c07; --warn-line: #78500a;
    --ok-fg: #6ee7b7; --ok-bg: #06231a; --ok-line: #0f5132;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: .95rem; margin: 0 0 .75rem; letter-spacing: -.005em; }
.sub { color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.card {
  border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.1rem;
  margin-bottom: 1.25rem; background: var(--panel);
}
.banner {
  border: 1px solid var(--line); border-left-width: 4px; border-radius: 6px;
  padding: .7rem .85rem; margin-bottom: 1.25rem; font-size: .85rem;
}
.banner.memory { border-color: var(--warn-line); background: var(--warn-bg); color: var(--warn-fg); }
.banner.store { border-color: var(--ok-line); background: var(--ok-bg); color: var(--ok-fg); }
.banner.none { border-color: var(--line); background: var(--panel); color: var(--muted); }
.facts { display: grid; grid-template-columns: 10rem 1fr; gap: .45rem 1rem; font-size: .87rem; }
.facts dt {
  color: var(--muted); font-size: .72rem; text-transform: uppercase;
  letter-spacing: .05em; padding-top: .18rem;
}
.facts dd { margin: 0; }
.chip {
  display: inline-block; border: 1px solid var(--line); border-radius: 999px;
  padding: .08rem .55rem; margin: 0 .3rem .3rem 0; font-size: .8rem; background: var(--bg);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.scroll { overflow-x: auto; }
.topology { display: block; margin: 0 auto; max-width: 100%; height: auto; }
.topology .box { fill: var(--bg); stroke: var(--line); stroke-width: 1.5; }
.topology .terminal .box { fill: var(--panel); stroke-dasharray: 3 3; }
.topology .label {
  fill: var(--fg); font: 600 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.topology .metric {
  fill: var(--muted); font: 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.topology .edge { fill: none; stroke: var(--muted); stroke-width: 1.4; opacity: .75; }
.topology .edge.conditional { stroke-dasharray: 5 4; }
.topology .arrow { fill: var(--muted); opacity: .75; }
.topology .edge-label {
  fill: var(--muted); font: 10px ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.legend { color: var(--muted); font-size: .78rem; margin: .9rem 0 0; }
.runbar { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin-bottom: .9rem; }
.runbar label {
  color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
}
select, button {
  font: inherit; font-size: .85rem; color: var(--fg); background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: .28rem .6rem;
}
button { cursor: pointer; }
button:hover { background: var(--panel); }
select { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; max-width: 22rem; }
#wh-status { color: var(--muted); font-size: .8rem; }
.selected { border-top: 1px solid var(--line); padding-top: .9rem; margin-bottom: .9rem; }
.selected[hidden] { display: none; }
.runid { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: .9rem; }
.runid .k {
  color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
}
#wh-runid {
  font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--fg);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: .28rem .6rem; min-width: 21rem; max-width: 100%;
}
.kpis { display: flex; flex-wrap: wrap; gap: 1.5rem; }
.kpi .v { font-size: 1.15rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .k {
  color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
}
.topology .node.failed .box { stroke: var(--err-line); stroke-width: 2; }
.topology .node.touched .label { fill: var(--fg); }
.topology .node:not(.touched) .label { opacity: .55; }
.off { color: var(--muted); }
.empty {
  color: var(--muted); padding: 1.5rem; text-align: center;
  border: 1px dashed var(--line); border-radius: 10px;
}
footer {
  color: var(--muted); font-size: .78rem; margin-top: 2rem;
  border-top: 1px solid var(--line); padding-top: .85rem;
}
"""


# The page's only script. It reads two JSON endpoints and then does exactly one
# thing to the document: sets attributes and textContent on elements the server
# already rendered. It never assigns innerHTML and never builds markup from a
# string, so no value returned by the API can reach the DOM as markup -- the
# same invariant `wardhook.observability`'s viewer holds, for the same reason.
#
# Colour is applied as fill-opacity over `var(--bar)` rather than as a computed
# colour, so the overlay follows the reader's light or dark theme without this
# script needing to know which one is in force.
_SCRIPT = """
(function () {
  var picker = document.getElementById('wh-run');
  if (!picker) return;
  var refresh = document.getElementById('wh-refresh');
  var status = document.getElementById('wh-status');
  var selected = document.getElementById('wh-selected');
  var runIdField = document.getElementById('wh-runid');
  var copy = document.getElementById('wh-copy');
  var kpis = document.getElementById('wh-kpis');
  var extra = document.getElementById('wh-extra');
  var base = window.location.pathname.replace(/\\/?$/, '/');

  function fmtMs(v) {
    return v < 1000 ? Math.round(v) + 'ms' : (v / 1000).toFixed(2) + 's';
  }
  function fmtCost(v) {
    if (!v) return '$0';
    return v < 0.01 ? '$' + v.toFixed(5) : '$' + v.toFixed(4);
  }
  function fmtInt(v) {
    return (v || 0).toLocaleString('en-US');
  }

  function clearOverlay() {
    document.querySelectorAll('.topology .node').forEach(function (node) {
      var box = node.querySelector('.box');
      box.style.fill = '';
      box.style.fillOpacity = '';
      node.classList.remove('failed');
      node.classList.remove('touched');
    });
    document.querySelectorAll('.topology .metric').forEach(function (label) {
      label.textContent = '';
    });
  }

  function kpi(key, value) {
    var wrap = document.createElement('div');
    wrap.className = 'kpi';
    var v = document.createElement('div');
    v.className = 'v';
    v.textContent = value;
    var k = document.createElement('div');
    k.className = 'k';
    k.textContent = key;
    wrap.appendChild(v);
    wrap.appendChild(k);
    return wrap;
  }

  // One node can execute several times in a run: the tool loop returns to
  // call_model on every round trip. Totalling them is the only honest answer --
  // showing the last execution alone would quietly under-report the node the
  // run's own totals say cost the most.
  function byNode(steps) {
    var seen = {};
    var order = [];
    steps.forEach(function (step) {
      var agg = seen[step.node];
      if (!agg) {
        agg = seen[step.node] = {
          node: step.node, latency_ms: 0, tokens_in: 0, tokens_out: 0,
          cost: 0, visits: 0, error: null
        };
        order.push(agg);
      }
      agg.latency_ms += step.latency_ms;
      agg.tokens_in += step.tokens_in;
      agg.tokens_out += step.tokens_out;
      agg.cost += step.cost;
      agg.visits += 1;
      if (step.error) agg.error = step.error;
    });
    return order;
  }

  function paint(run) {
    clearOverlay();
    runIdField.value = run.run_id;
    var aggregated = byNode(run.steps);
    var slowest = 0;
    aggregated.forEach(function (agg) {
      if (agg.latency_ms > slowest) slowest = agg.latency_ms;
    });

    var orphans = [];
    aggregated.forEach(function (agg) {
      var node = document.querySelector('.topology [data-node="' + agg.node + '"]');
      if (!node) {
        orphans.push(agg);
        return;
      }
      var share = slowest > 0 ? agg.latency_ms / slowest : 0;
      var box = node.querySelector('.box');
      box.style.fill = 'var(--bar)';
      box.style.fillOpacity = (0.07 + 0.5 * share).toFixed(3);
      node.classList.add('touched');
      if (agg.error) node.classList.add('failed');

      var parts = [];
      if (agg.visits > 1) parts.push('\\u00d7' + agg.visits);
      parts.push(fmtMs(agg.latency_ms));
      if (agg.tokens_in || agg.tokens_out) {
        parts.push(fmtInt(agg.tokens_in) + ' in / ' + fmtInt(agg.tokens_out) + ' out');
      }
      if (agg.cost) parts.push(fmtCost(agg.cost));
      node.querySelector('.metric').textContent = parts.join('  \\u00b7  ');
    });

    kpis.textContent = '';
    kpis.appendChild(kpi('steps', String(run.totals.steps)));
    kpis.appendChild(kpi('latency', fmtMs(run.latency_ms)));
    kpis.appendChild(kpi('tokens in', fmtInt(run.totals.tokens_in)));
    kpis.appendChild(kpi('tokens out', fmtInt(run.totals.tokens_out)));
    kpis.appendChild(kpi('cached', fmtInt(run.totals.cached_tokens)));
    kpis.appendChild(kpi('cost', fmtCost(run.totals.cost)));
    if (run.failed) kpis.appendChild(kpi('status', 'failed'));

    if (orphans.length) {
      var names = orphans.map(function (agg) {
        return agg.node + ' (' + fmtMs(agg.latency_ms) + ', ' + fmtCost(agg.cost) + ')';
      });
      extra.textContent =
        'Recorded but not on the diagram: ' + names.join(', ') +
        '. Token usage that arrived while no node was open is attributed here ' +
        'rather than discarded, because a cost you cannot attribute is still a ' +
        'cost you paid.';
    } else {
      extra.textContent = '';
    }
    selected.hidden = false;
  }

  function select(runId) {
    if (!runId) {
      clearOverlay();
      selected.hidden = true;
      return;
    }
    fetch(base + 'api/runs/' + encodeURIComponent(runId))
      .then(function (response) {
        if (!response.ok) throw new Error('run not found');
        return response.json();
      })
      .then(paint)
      .catch(function () {
        status.textContent = 'That run is no longer held. Refresh the list.';
      });
  }

  function load() {
    status.textContent = 'loading...';
    fetch(base + 'api/runs')
      .then(function (response) { return response.json(); })
      .then(function (data) {
        picker.textContent = '';
        picker.appendChild(new Option('- select a run -', ''));
        data.runs.forEach(function (run) {
          var label =
            run.run_id.slice(0, 12) + '  ' + fmtMs(run.latency_ms) +
            '  ' + fmtCost(run.totals.cost) + (run.failed ? '  failed' : '');
          picker.appendChild(new Option(label, run.run_id));
        });
        if (!data.runs.length) {
          status.textContent = 'No runs recorded yet. Invoke the agent, then refresh.';
          clearOverlay();
          selected.hidden = true;
          return;
        }
        status.textContent =
          data.returned < data.total
            ? 'showing ' + data.returned + ' of ' + data.total + ' runs'
            : data.total + (data.total === 1 ? ' run' : ' runs');
        picker.value = data.runs[0].run_id;
        select(picker.value);
      })
      .catch(function () {
        status.textContent = 'Could not read the run list.';
      });
  }

  picker.addEventListener('change', function () { select(picker.value); });
  refresh.addEventListener('click', load);
  copy.addEventListener('click', function () {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(runIdField.value);
    } else {
      runIdField.select();
    }
    status.textContent = 'run_id copied - look it up in your own audit log';
  });
  load();
})();
"""


def _optional_str(value: Any) -> str | None:
    """Coerce a value to text, preserving the difference between unset and empty.

    Args:
        value: Any value, possibly ``None``.

    Returns:
        ``None`` if the value was ``None``, otherwise its string form.
    """
    return None if value is None else str(value)


def _count(usage: Any, field: str) -> int:
    """Read one token count off a usage object.

    Args:
        usage: A token-usage object, or ``None`` for a node that called no model.
        field: The attribute to read.

    Returns:
        The count, or ``0`` when the usage or the field is absent. Guardrail and
        retrieval nodes legitimately have no usage at all.
    """
    return int(getattr(usage, field, 0) or 0)


def _project_step(step: Any) -> dict[str, Any]:
    """Project one trace step onto the fields this API publishes.

    **This function is the allowlist, and that is deliberate.** It names every
    field that reaches a browser. If a content-bearing field -- a prompt, a
    completion, a retrieved chunk -- is ever added to the upstream step type,
    this API does not begin serving it by accident; someone has to add it here
    on purpose, and this docstring is what they will read when they do.

    Args:
        step: Any object shaped like an observability ``TraceStep``.

    Returns:
        A JSON-serialisable dict of timing, tokens, cost and error only.
    """
    usage = getattr(step, "usage", None)
    return {
        "node": str(getattr(step, "node", "")),
        "run_id": str(getattr(step, "run_id", "")),
        "started_at": str(getattr(step, "started_at", "")),
        "latency_ms": float(getattr(step, "latency_ms", 0.0) or 0.0),
        "cost": float(getattr(step, "cost", 0.0) or 0.0),
        "model": _optional_str(getattr(step, "model", None)),
        "error": _optional_str(getattr(step, "error", None)),
        "tokens_in": _count(usage, "input_tokens"),
        "tokens_out": _count(usage, "output_tokens"),
        "cached_tokens": _count(usage, "cache_read_tokens"),
    }


def _project_trace(trace: Any) -> dict[str, Any]:
    """Project one whole trace, recomputing its totals from its steps.

    Totals are derived here rather than read from the trace, for the same reason
    :meth:`Trace.from_dict` recomputes them: a truncated or hand-edited trace
    then cannot claim totals its steps do not support.

    Args:
        trace: Any object shaped like an observability ``Trace``.

    Returns:
        A JSON-serialisable dict. ``metadata`` is the one caller-supplied field
        and is passed through verbatim, so -- as ``wardhook-observability``'s
        store already warns -- do not put user text in it.
    """
    steps = [_project_step(step) for step in getattr(trace, "steps", None) or ()]
    error = _optional_str(getattr(trace, "error", None))
    return {
        "run_id": str(getattr(trace, "run_id", "")),
        "started_at": str(getattr(trace, "started_at", "")),
        "latency_ms": float(getattr(trace, "latency_ms", 0.0) or 0.0),
        "metadata": dict(getattr(trace, "metadata", None) or {}),
        "error": error,
        "failed": error is not None or any(step["error"] is not None for step in steps),
        "totals": {
            "steps": len(steps),
            "tokens_in": sum(step["tokens_in"] for step in steps),
            "tokens_out": sum(step["tokens_out"] for step in steps),
            "cached_tokens": sum(step["cached_tokens"] for step in steps),
            "cost": sum(step["cost"] for step in steps),
        },
        "steps": steps,
    }


def _summarise(projected: dict[str, Any]) -> dict[str, Any]:
    """Reduce a projected trace to its summary form.

    Args:
        projected: The output of :func:`_project_trace`.

    Returns:
        The same dict without ``steps``. Derived by subtraction rather than
        rebuilt, so a summary can never carry a field the detail view lacks.
    """
    return {key: value for key, value in projected.items() if key != "steps"}


def _sink_mode(sink: Any) -> str:
    """Report which telemetry shape was found, without reading anything.

    Args:
        sink: The telemetry sink, or ``None``.

    Returns:
        ``"memory"``, ``"store"``, or ``"none"``. The mode is published rather
        than kept private because it is the difference between seeing all of an
        agent's traffic and seeing one worker process's share of it.
    """
    for name, mode in _LIST_METHODS:
        if callable(getattr(sink, name, None)):
            return mode
    return "none"


def _list_traces(sink: Any) -> list[Any]:
    """Read every run a sink is willing to list.

    Args:
        sink: The telemetry sink, or ``None``.

    Returns:
        The traces, oldest first, as both known sinks report them. An empty
        list when the sink cannot list runs -- which is not an error.
    """
    for name, _ in _LIST_METHODS:
        reader = getattr(sink, name, None)
        if callable(reader):
            return list(reader())
    return []


def _find_trace(sink: Any, run_id: str) -> Any:
    """Look one run up on a sink.

    Args:
        sink: The telemetry sink, or ``None``.
        run_id: The run to find.

    Returns:
        The trace, or ``None`` when the sink cannot look runs up or does not
        hold that one.
    """
    for name in _LOOKUP_METHODS:
        lookup = getattr(sink, name, None)
        if callable(lookup):
            return lookup(run_id)
    return None


def _resolve_telemetry(agent: Any, telemetry: Any) -> Any:
    """Decide which telemetry sink the dashboard reads.

    Args:
        agent: The agent being described.
        telemetry: An explicitly supplied sink, or ``None`` to use the agent's.

    Returns:
        The sink, or ``None``. Passing a sink explicitly is what lets a caller
        point the dashboard at a shared :class:`JSONLTraceStore` while the agent
        keeps writing through an in-memory tracer -- the documented mitigation
        for the multi-worker limitation.
    """
    return telemetry if telemetry is not None else getattr(agent, "telemetry", None)


def _enabled(flag: Any) -> str:
    """Render an on/off configuration flag.

    Args:
        flag: The flag's value.

    Returns:
        HTML reading ``enabled`` or ``disabled``.
    """
    return "enabled" if flag else '<span class="off">disabled</span>'


def _chips(values: Sequence[str], empty: str) -> str:
    """Render a list of names as escaped chips.

    Args:
        values: The names to render.
        empty: Text to show instead when there are none.

    Returns:
        HTML. Every name is escaped: tool and guardrail names come from user
        code and are therefore untrusted markup, exactly as trace content is in
        the static viewer, where the same escaping is documented as a security
        control. It matters more here, because this page really is served over
        HTTP rather than opened from a local file.
    """
    if not values:
        return f'<span class="off">{escape(empty)}</span>'
    return "".join(f'<span class="chip">{escape(str(value))}</span>' for value in values)


def _render_page(agent: Any, sink: Any) -> str:
    """Render the dashboard page for an agent.

    Args:
        agent: The agent being described.
        sink: The telemetry sink the dashboard reads, or ``None``.

    Returns:
        A complete HTML document. It references nothing external -- no CDN, no
        web font, no image -- so it behaves identically in an air-gapped
        network, which is the environment this project is aimed at.

    Example:
        >>> page = _render_page(object(), None)
        >>> page.startswith("<!doctype html>")
        True
        >>> "http" + "://" in page
        False
    """
    described = _describe(agent)
    name = str(described["name"])
    mode = _sink_mode(sink)
    script = f"<script>{_SCRIPT}</script>" if mode != "none" else ""

    facts = (
        f'<dt>type</dt><dd class="mono">{escape(str(described["type"]))}</dd>'
        f"<dt>tools</dt><dd>{_chips(described['tools'], 'none attached')}</dd>"
        f"<dt>guardrails</dt><dd>{_chips(described['guardrails'], 'none attached')}</dd>"
        f"<dt>retrieval</dt><dd>{_enabled(described['retrieval_enabled'])}</dd>"
        f"<dt>telemetry</dt><dd>{_enabled(described['telemetry_enabled'])}</dd>"
    )

    runbar = ""
    if mode != "none":
        runbar = (
            '<div class="runbar">'
            '<label for="wh-run">Run</label>'
            '<select id="wh-run"></select>'
            '<button id="wh-refresh" type="button">Refresh</button>'
            '<span id="wh-status"></span>'
            "</div>"
            '<div id="wh-selected" class="selected" hidden>'
            '<div class="runid"><span class="k">run_id</span>'
            '<input id="wh-runid" readonly value="">'
            '<button id="wh-copy" type="button">Copy</button>'
            "</div>"
            '<div class="kpis" id="wh-kpis"></div>'
            '<p class="legend" id="wh-extra"></p>'
            "</div>"
        )

    topology = read_topology(agent)
    diagram = render_svg(topology)
    if diagram:
        picture = (
            f'<div class="scroll">{diagram}</div>'
            f'<p class="legend">{len(topology.nodes)} nodes, {len(topology.edges)} edges. '
            f"A dashed edge is conditional: a router decides at run time whether it is "
            f"taken. This is the graph this agent actually compiled, so a feature it "
            f"is not configured for has no box here.</p>"
        )
    else:
        picture = f'<div class="empty">{escape(topology.reason or "No graph to draw.")}</div>'

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(_PAGE_TITLE)} &middot; {escape(name)}</title>"
        f'<style>{_STYLE}</style></head><body><div class="wrap">'
        f"<h1>{escape(name)}</h1>"
        f'<div class="sub">Structure and cost of this agent. '
        f"No prompts, no responses, no retrieved text.</div>"
        f'<div class="banner {escape(mode)}">{escape(_MODE_NOTES[mode])}</div>'
        f'<section class="card"><h2>Topology</h2>{runbar}{picture}</section>'
        f'<section class="card"><h2>Configuration</h2>'
        f'<dl class="facts">{facts}</dl></section>'
        f"<footer>Served by wardhook-core. This page shows telemetry and "
        f"configuration only -- never prompts, model output, retrieved context "
        f"or guardrail event bodies. Use a run's run_id to correlate it with "
        f"your own audit log, where your redaction policy applies. This page "
        f"makes no network requests.</footer>"
        f"</div>{script}</body></html>"
    )


def create_dashboard(agent: Any, telemetry: Any = None) -> FastAPI:
    """Build the read-only dashboard API for an agent.

    Args:
        agent: The agent to describe. Any object works; one without a graph
            simply reports that it has no topology.
        telemetry: The sink to read runs from. Defaults to the agent's own
            ``telemetry`` attribute. Pass a shared trace store here to read
            every worker's runs rather than one process's.

    Returns:
        A FastAPI application, ready to mount or to serve on its own.

    Example:
        >>> from langchain_core.language_models.fake_chat_models import (
        ...     GenericFakeChatModel,
        ... )
        >>> from langchain_core.messages import AIMessage
        >>> from wardhook.core import AgentGraph
        >>> model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
        >>> app = create_dashboard(AgentGraph(model=model, name="demo"))
        >>> app.title
        'Wardhook Dashboard'
    """
    sink = _resolve_telemetry(agent, telemetry)
    app = FastAPI(
        title="Wardhook Dashboard",
        description=(
            "Read-only view of an agent's structure and what its runs cost. "
            "Serves telemetry and configuration only, never agent content."
        ),
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        """Serve the dashboard page.

        Returns:
            A self-contained HTML document describing this agent.
        """
        return _render_page(agent, sink)

    @app.get("/api/topology", tags=["dashboard"])
    def topology() -> dict[str, Any]:
        """Describe the agent's graph and configuration.

        Returns:
            The topology, plus the same configuration summary ``GET /info``
            reports. Both are derived from the agent as configured, so an agent
            with no retriever genuinely reports no retrieval node.
        """
        return {
            "agent": str(_describe(agent)["name"]),
            "config": _describe(agent),
            **read_topology(agent).to_dict(),
        }

    @app.get("/api/runs", tags=["dashboard"])
    def runs(
        limit: int = Query(
            default=_DEFAULT_LIMIT,
            ge=1,
            le=_MAX_LIMIT,
            description="Maximum number of runs to return, newest first.",
        ),
    ) -> dict[str, Any]:
        """Summarise the recorded runs, newest first.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            The summaries, the total held, and which telemetry mode produced
            them. ``mode`` and ``mode_note`` are how the multi-worker limitation
            is disclosed rather than hidden.
        """
        mode = _sink_mode(sink)
        newest_first = list(reversed(_list_traces(sink)))
        return {
            "mode": mode,
            "mode_note": _MODE_NOTES[mode],
            "total": len(newest_first),
            "returned": len(newest_first[:limit]),
            "runs": [_summarise(_project_trace(trace)) for trace in newest_first[:limit]],
        }

    @app.get("/api/runs/{run_id}", tags=["dashboard"])
    def run(run_id: str) -> dict[str, Any]:
        """Return one run's per-node timing, tokens and cost.

        Args:
            run_id: The run to look up.

        Returns:
            The projected trace.

        Raises:
            HTTPException: ``404`` when no such run is held. A run that has been
                evicted from a bounded in-memory ring is genuinely absent, so
                this is not an error the caller can fix by retrying.
        """
        trace = _find_trace(sink, run_id)
        if trace is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No trace for run_id {run_id!r}. It may have been evicted "
                    f"from the tracer's in-memory ring, or never recorded."
                ),
            )
        return _project_trace(trace)

    return app
