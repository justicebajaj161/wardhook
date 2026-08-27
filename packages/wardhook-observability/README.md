# wardhook-observability

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

Per-node token count, cost estimate, and latency tracking for LangGraph agents,
with a self-contained HTML trace viewer.

Part of [Wardhook](https://github.com/justicebajaj161/wardhook).

> **🚧 In progress.** This package is scaffolded but not yet implemented. Follow
> the [project status table](../../README.md#project-status) for progress.

## Planned API

```python
from wardhook.core import AgentGraph

agent = AgentGraph(model="claude-opus-5", telemetry=True)
agent.invoke("What excess applies to storm damage?")

trace = agent.trace()
for step in trace.steps:
    print(f"{step.node:14s} {step.latency_ms:6.0f}ms  {step.tokens_out:5d} tok  ${step.cost:.4f}")
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
```

## Design notes

- **Real usage, not estimates.** A LangChain callback handler reads
  `usage_metadata` off the provider response rather than re-tokenising.
- **Cost table includes cache multipliers** — cache reads bill at roughly 0.1×
  and writes at 1.25×, which dominates the arithmetic for any agent using prompt
  caching. Unknown models cost zero and warn rather than guessing.
- **The viewer is one static HTML file** with inlined CSS and JS. No CDN, no
  network requests, no dashboard server to run. It opens anywhere and is
  testable by asserting on the rendered markup.

## Links

- [Design decisions](../../docs/packages/observability.md)
- [Architecture overview](../../docs/architecture.md)
- [Main repository](https://github.com/justicebajaj161/wardhook)
