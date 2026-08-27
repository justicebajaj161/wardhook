# Architecture

Wardhook is four packages that compose but do not depend on each other. This
page explains how that works, why it was built that way, and what it costs.

For per-package design decisions, see [core](packages/core.md),
[guardrails](packages/guardrails.md), [observability](packages/observability.md),
and [evals](packages/evals.md).

## The constraint everything follows from

> Each package must be `pip install`-able alone and useful on its own, **and**
> `AgentGraph(guardrails=[...], telemetry=True)` must work when they are
> installed together.

Those two requirements pull in opposite directions. Satisfying both is the
single most important structural decision in the project, and nearly every
other choice is downstream of it.

## Why it matters

The packages have genuinely different audiences.

`wardhook-guardrails` is useful to someone with no agent at all — a Django view
that redacts PII before writing to a log, a batch job scrubbing a CSV, a Lambda
sanitising webhook payloads. If it dragged in LangGraph, LangChain, and an ML
stack, none of those people could use it. Its dependency list is **PyYAML**.

`wardhook-evals` is useful against anything with an `.invoke()` method,
including agents built with no Wardhook code whatsoever.

`wardhook-core` should be adoptable by someone who wants a LangGraph runtime
with RAG and does not care about governance at all.

A monorepo where everything imports everything serves none of them.

## Options considered

| Approach | Why not |
| --- | --- |
| Core depends on guardrails and observability | Kills standalone installability, the entire point of the split |
| Guardrails and observability depend on core | Anyone wanting PII redaction in a Flask app installs LangGraph |
| A shared `wardhook-common` base package | A fifth package for two protocol definitions, and every install pulls it |
| **Structural typing (chosen)** | Zero dependency in either direction |

## How it works

`wardhook-core` declares `typing.Protocol` shapes in
`wardhook/core/protocols.py` and duck-types anything it receives. The other
packages implement those shapes without importing core. Neither side knows the
other's types exist.

```
                          wardhook-core
        ┌────────────────────────────────────────────────┐
        │  protocols.py                                  │
        │    GuardrailProtocol   ◄╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐
        │    TelemetryProtocol   ◄╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐  │        ╎
        │    read_guardrail_result()                   ╎  │        ╎
        │                                              ╎  │        ╎
        │  agent.py :: AgentGraph                      ╎  │        ╎
        │    START → guard_input ─(blocked)→ END       ╎  │        ╎
        │              │                               ╎  │        ╎
        │              ▼                               ╎  │        ╎
        │           retrieve → call_model ⇄ tools      ╎  │        ╎
        │                          │                   ╎  │        ╎
        │                          ▼                   ╎  │        ╎
        │                     guard_output → END       ╎  │        ╎
        └──────────────────────────┬───────────────────╎──┘        ╎
                                   │                   ╎           ╎
              lazy import, only if │              structural   structural
              telemetry=True       ▼                match      match
                     ┌──────────────────────────┐    ╎           ╎
                     │ wardhook-observability   │╌╌╌╌┘           ╎
                     │   Tracer, get_trace()    │                ╎
                     └──────────────────────────┘                ╎
                     ┌──────────────────────────┐                ╎
                     │ wardhook-guardrails      │╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
                     │   PIIRedactor            │
                     │   InjectionDetector      │        no import of core
                     │   RoleBasedToolPolicy    │        in either direction
                     └──────────────────────────┘

                     ┌──────────────────────────┐
                     │ wardhook-evals           │  targets anything with
                     │   EvalRunner             │  .invoke() -- an AgentGraph,
                     │   baseline comparison    │  a raw graph, or a function
                     └──────────────────────────┘
```

### The three rules that keep it honest

**1. Core never imports a sibling at module scope.** `telemetry=True` triggers a
lazy import inside `AgentGraph._resolve_telemetry`, guarded so a missing package
produces an actionable message rather than a traceback:

```python
try:
    from wardhook.observability import Tracer
except ImportError as exc:
    raise MissingIntegrationError(
        "telemetry=True needs the wardhook-observability package ..."
    ) from exc
```

**2. Context is a plain `dict`, not a class.** Every guardrail hook receives a
mapping with documented keys — `run_id`, `stage`, `node`, `principal`, `agent`.
A guardrail author never imports a Wardhook type to read it.

**3. Results are normalised, not type-checked.** `read_guardrail_result()`
accepts whatever comes back — a dataclass, a dict, `None`, a bare boolean — and
produces one predictable `GuardrailDecision`. An unrecognised action string maps
to `ALLOW` while preserving the original in `details["raw_action"]`: a
third-party guardrail returning something odd must not take down a production
agent, but the anomaly must still be visible in the audit trail.

### What this buys you

Anything of the right shape works. No base class, no import, no dependency:

```python
class NoInternalCodenames:
    name = "no-codenames"

    def on_output(self, text, context):
        if "PROJECT_HALCYON" in text:
            return {
                "action": "redact",
                "text": text.replace("PROJECT_HALCYON", "[internal]"),
                "reason": "internal codename",
                "rule": "codename-list",
            }
        return {"action": "allow"}
```

The full contract is three optional hooks — `on_input`, `on_output`,
`on_tool_call` — each returning something with an `action` of `"allow"`,
`"redact"` or `"block"`. Core probes for each with `getattr` and skips any that
is missing, so a guardrail that only polices tool calls implements one method.

### What it costs

A static type checker cannot verify at the call site that a guardrail satisfies
the contract. That is a real loss and worth stating plainly.

It is mitigated three ways: `read_guardrail_result()` normalises defensively at
runtime; each package's suite pins the contract from its own side; and CI
installs each package alone and runs its suite there, which is the check that
actually matters. A type error in an unused code path is less dangerous than a
sibling import that only fails for the person who installed one package.

## The seam in practice

### Guardrails

Core calls the hooks and threads redactions through in order, so each guardrail
sees the previous one's output. The chain stops at the first block.

| Stage | Node | Blocking effect |
| --- | --- | --- |
| `input` | `guard_input` | Short-circuits to `END`; the model never runs |
| `output` | `guard_output` | Reply replaced with a withheld-message notice |
| `tool_call` | `tools` | Tool never executes; model receives a denial result |

A denied tool call becomes a normal tool result rather than an exception,
because an agent that can see a denial can often route around it. The same
applies to tools that raise.

**Guardrails fail closed.** If a guardrail itself raises,
`guardrail_error_policy="block"` (the default) treats it as a block. For a
governance tool, a crashed detector silently passing traffic is the worst
available outcome — it looks exactly like a clean run. The error is recorded
even under `"allow"`, so fail-open never means fail-silent.

### Telemetry

Core wraps each node to report `start_node` / `end_node`, and passes
`telemetry.callbacks()` into every model invocation so the sink can read real
token usage off the provider response rather than estimating it.

### Evals

`wardhook-evals` depends on nothing at all. Its runner targets any object with
`.invoke()`, which is why `AgentGraph.invoke()` accepts a mapping with an
`input` / `question` / `messages` key and returns a plain dict — an agent is a
valid eval target with no adapter in between.

## Data flow through a governed request

```
  user input
      │
      ▼
  guard_input      injection scored, PII redacted     ─┐
      │            (redaction replaces the turn         │
      ▼             in place, so the model never        │
  retrieve          sees the original)                  │
      │                                                 │  every decision
      ▼                                                 ├─ appended to
  call_model       tokens, cost, latency recorded        │  guardrail_events
      │                                                 │  (accumulating
      ▼                                                 │   reducer -- see
  tools            each call checked against RBAC;       │   note below)
      │            denials never execute                 │
      ▼                                                 │
  guard_output     PII redacted from the reply         ─┘
      │
      ▼
  { output, citations, guardrail_events, tool_calls, run_id, blocked }
                              │
                              ▼
                    AuditLogger.record_run()
                    → JSONL, one line per action,
                      containing no audited data
```

`guardrail_events` carries an `operator.add` reducer. Without it LangGraph
replaces the list on each node update, and the output stage would silently
overwrite everything the input stage recorded — the audit trail would show only
the last node that ran. This was a real bug, caught by a test that is still
there.

## Repository layout

```
wardhook/
├── packages/
│   ├── wardhook-core/          src/wardhook/core/
│   ├── wardhook-guardrails/    src/wardhook/guardrails/
│   ├── wardhook-observability/ src/wardhook/observability/
│   └── wardhook-evals/         src/wardhook/evals/
├── docs/          architecture, quickstart, one design page per package
├── examples/      one runnable script per package, plus a combined one
└── .github/       CI, issue and PR templates
```

### Namespace packaging

All four install into a shared `wardhook.` namespace via
[PEP 420](https://peps.python.org/pep-0420/). **No package contains a
`wardhook/__init__.py`** — that file's absence is what allows the four
distributions to merge into one import namespace at runtime.

The result is that `from wardhook.guardrails import PIIRedactor` works whether
you installed one package or all four, and `wardhook.__file__` is `None`
because the namespace is virtual.

### Tooling

One `ruff` and one `mypy` configuration in the root `pyproject.toml`, applied
uniformly. A `uv` workspace makes all four editable in one environment for
development. Each package still builds and publishes independently with
`hatchling`.

## How the boundary is enforced

Claims in a README rot. These are checked:

1. **AST scan per package.** `tests/test_isolation.py` parses every module and
   fails on an import of a sibling, LangChain, LangGraph, or a heavy numerical
   dependency. It is a *static* scan because the development environment has all
   four installed — an accidental import would succeed there and only break for
   someone who installed one package.
2. **Isolated CI matrix.** Each package is installed into an environment with no
   siblings, asserted to be alone, and its suite run there — across Python 3.10
   through 3.13.
3. **Offline tests.** No test touches the network or needs an API key, so the
   above runs on every push.

## Known limitations

- **Runtime contracts are not statically verified** across the seam. Discussed
  above.
- **The in-memory vector store** holds everything in one NumPy matrix. Adequate
  to roughly a hundred thousand chunks; past that, use a purpose-built store
  behind the same protocol.
- **Guardrail detection is pattern-based** and will miss things. That limitation
  is argued in full in the [guardrails design doc](packages/guardrails.md#limitations)
  rather than glossed over.
- **Multi-modal content is flattened** to its text blocks before guardrails see
  it. Images and audio are not inspected.
