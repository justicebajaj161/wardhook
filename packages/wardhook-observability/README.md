# wardhook-observability

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)

Per-node token count, cost estimate, and latency tracking for LangGraph agents,
with a self-contained HTML trace viewer.

Part of [Wardhook](https://github.com/justicebajaj161/wardhook).

## Install

```bash
pip install wardhook-observability
```

No account, no API key, no collector process, no sidecar. Nothing is sent
anywhere.

## Usage

With `wardhook-core`, tracing is one keyword argument:

```python
from wardhook.core import AgentGraph

agent = AgentGraph(model="claude-opus-5", telemetry=True)
agent.invoke("What excess applies to storm damage?")

trace = agent.trace()
for step in trace.steps:
    print(f"{step.node:14s} {step.latency_ms:6.0f}ms  {step.tokens_out:5d} tok  ${step.cost:.4f}")
```

```
guard_input        2ms      0 tok  $0.0000
retrieve          18ms      0 tok  $0.0000
call_model       412ms    120 tok  $0.0057
guard_output       3ms      0 tok  $0.0000
```

Standalone, against a graph you already built:

```python
from wardhook.observability import Tracer, instrument, render_html

instrument(my_existing_graph, tracer := Tracer())
my_existing_graph.invoke({"messages": [...]})
open("trace.html", "w").write(render_html(tracer.get_trace()))
```

And from the command line:

```bash
wardhook-trace view traces/run-42.jsonl -o trace.html
wardhook-trace summary traces/run-42.jsonl
```

## Keeping history

Point a tracer at a store and every completed run is appended to a JSON Lines
file. Memory stays bounded; the file does not.

```python
from wardhook.observability import JSONLTraceStore, Tracer

tracer = Tracer(store=JSONLTraceStore("traces/agent.jsonl"), max_runs=50)
```

## Correcting a price

The built-in table carries a `PRICES_AS_OF` date and is rendered on every
generated page, so a stale number is visible rather than silently trusted.
Unknown models cost `$0` and warn once. Override any rate without forking:

```python
from wardhook.observability import ModelPrice, register_price

register_price("claude-opus-5", ModelPrice(input_per_1m=4.50, output_per_1m=22.50))
```

## Design notes

- **Real usage, not estimates.** A LangChain callback handler reads
  `usage_metadata` off the provider response rather than re-tokenising. Local
  re-tokenisation cannot see the system prompt the provider prepends, tool
  schema overhead, or which tokens were served from cache.
- **Cache multipliers, applied once.** Cache reads bill at roughly 0.1× and
  writes at 1.25×. LangChain reports `input_tokens` as the *total*, cached
  tokens included, so the uncached remainder is derived before pricing — adding
  the cache buckets on top would bill them twice. A fully cached prompt costs
  0.1× the input rate here, not 1.1×.
- **`instrument()` reads callbacks, it does not patch nodes.** Replacing a
  compiled graph's node callables would depend on LangGraph internals and break
  on upgrade. The callback stream is a public interface and carries the node
  name on every step.
- **The viewer is one static HTML file** with inlined CSS and JS. No CDN, no
  web font, no network requests, no dashboard server to run. Every interpolated
  value is escaped — a trace carries user-influenced text, so that is a
  security control, not formatting.
- **Safe in a long-lived process.** Traces are held in a bounded ring, shared
  state is locked, node attribution is per-thread, and no callback can raise
  into your agent. Telemetry failing is never worth failing a request over.

## Links

- [Design decisions](../../docs/packages/observability.md)
- [Architecture overview](../../docs/architecture.md)
- [Main repository](https://github.com/justicebajaj161/wardhook)
