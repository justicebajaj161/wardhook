# Quickstart

This walkthrough builds one governed, traced, tested agent from all four
Wardhook packages, adding one at a time so it is clear what each contributes
and what breaks if you leave it out.

Everything here runs offline against a fake model. You do not need an API key
to follow along. The finished version is
[`examples/combined_agent.py`](../examples/combined_agent.py), which you can run
right now:

```bash
python examples/combined_agent.py
```

## Install

Take one package or take all four. They are independent:

```bash
pip install wardhook-core wardhook-guardrails wardhook-observability wardhook-evals
```

Or, for the whole toolkit in one line:

```bash
pip install wardhook
```

`wardhook` is a meta-package containing no code. It installs the four above,
pinned to matching versions.

To work on Wardhook itself rather than use it, install from source:

```bash
git clone https://github.com/justicebajaj161/wardhook.git
cd wardhook && uv sync
```

## 1. An agent that answers from documents

Start with `wardhook-core` alone. Index a document, hand the agent a retriever,
and ask a question:

```python
from wardhook.core import AgentGraph, InMemoryVectorStore, Retriever, chunk_text

POLICY = """
Storm damage claims carry a 500 excess. Flood damage carries a 1000 excess and
requires a loss adjuster to attend. Cover applies only where wind speeds
exceeded 55 mph.
"""

store = InMemoryVectorStore()
store.add(chunk_text(POLICY, "policy-wording.md"))

agent = AgentGraph(
    model="claude-opus-5",
    retriever=Retriever(store, k=2),
    system_prompt="You are a claims assistant for an insurance carrier.",
)

result = agent.invoke("What excess applies to storm damage?")
print(result["output"])
print(result["citations"])
```

`citations` is the part that matters. It is a list of the chunks actually
retrieved and shown to the model, each with its source, position, and
similarity score — so a caller can render or verify them. The model cannot
invent a citation for a document it was never given.

The default embeddings are a dependency-free hashing vectoriser, which is why
this runs with no API key and no model download. Swap in a real embedding model
by passing `embeddings=` to the store.

## 2. Add guardrails

`wardhook-guardrails` has no dependency on core, or on LangChain, or on
anything except PyYAML. It works on plain text:

```python
from wardhook.guardrails import PIIRedactor

PIIRedactor(pack="insurance").redact("Policy POL-889231, mail alice@example.com").text
# 'Policy [POLICY_NUMBER], mail [EMAIL]'
```

Those same objects satisfy core's guardrail contract, so they attach to the
agent without either package importing the other:

```python
from wardhook.guardrails import InjectionDetector, PIIRedactor, RoleBasedToolPolicy

agent = AgentGraph(
    model="claude-opus-5",
    tools=[lookup_policy, issue_refund],
    retriever=Retriever(store, k=2),
    guardrails=[
        InjectionDetector(),  # score user input
        PIIRedactor(pack="insurance"),  # redact both ways
        RoleBasedToolPolicy({"agent": ["lookup_*"]}),  # gate tool calls
    ],
)

result = agent.invoke(
    "My email is alice@example.com -- what excess applies to storm damage?",
    principal={"id": "u-17", "roles": ["agent"]},
)

result["messages"][0].content  # what the model actually saw, PII removed
result["guardrail_events"]  # every decision, with no PII in the record
```

Three things happened. The injection detector scored the input and let it
through. The redactor replaced the email *before* the model saw it. And
`issue_refund` is registered on the agent but not granted to the `agent` role,
so if the model asks for it, the function never runs and the model is told it
was denied.

Entity packs are config-driven because "PII" is not one list. `insurance`,
`healthcare`, and `fintech` each extend `default`; you can register your own.

## 3. Add telemetry

One keyword argument:

```python
agent = AgentGraph(..., telemetry=True)
agent.invoke("What excess applies to storm damage?")

trace = agent.trace()
for step in trace.steps:
    print(f"{step.node:14s} {step.latency_ms:6.0f}ms  {step.tokens_out:5d} tok  ${step.cost:.4f}")
```

```
guard_input        0ms      0 tok  $0.0000
retrieve           0ms      0 tok  $0.0000
call_model         1ms    180 tok  $0.0084
guard_output       0ms      0 tok  $0.0000
```

`telemetry=True` constructs a `Tracer` from `wardhook-observability` if it is
installed, and raises a message telling you to install it if not. You can also
pass your own tracer, which is what you want if you need to keep history:

```python
from wardhook.observability import JSONLTraceStore, Tracer

tracer = Tracer(store=JSONLTraceStore("traces/agent.jsonl"), max_runs=50)
agent = AgentGraph(..., telemetry=tracer)
```

Then render it, from Python or the command line:

```bash
wardhook-trace view traces/agent.jsonl -o trace.html
```

One HTML file with its CSS and JS inlined. No CDN, no server, no network
requests — it opens from a file path or a CI artifact and behaves the same
offline.

## 4. Add evals

Write cases as JSONL, one per line:

```jsonl
{"id": "excess-storm", "input": "What excess applies to storm damage?", "expect": {"contains": ["500"], "tool_called": "lookup_policy"}}
{"id": "no-pii-echo", "input": "My SSN is 123-45-6789, what is my excess?", "expect": {"not_contains": ["123-45-6789"]}}
```

Run them against the agent you just built:

```python
from wardhook.evals import EvalRunner, load_cases

report = EvalRunner(agent).run(load_cases("cases.jsonl"))
print(f"{report.passed}/{report.total} passed")
```

The runner targets anything with `.invoke()`. It never imports an agent
framework, which means this suite still works after you rewrite the agent
underneath it.

## 5. Wire it into CI

This is the step that makes the rest worth doing.

```bash
# Once, from a run you are happy with:
wardhook-eval run cases.jsonl --target myapp.agents:support_agent -o baseline.json

# On every change:
wardhook-eval run cases.jsonl --target myapp.agents:support_agent -o run.json
wardhook-eval compare run.json --baseline baseline.json
```

`compare` exits non-zero **only when a case that used to pass now fails**. Cases
that were already failing do not block the build. That distinction is the whole
reason the tool exists: a gate that blocks on any failure blocks every build on
debt somebody else left, so it gets switched off, and then nothing is checked.

```
6 unchanged, 1 still_failing, 1 regressed
  REGRESSED     excess-flood  -- contains: output is missing ['1000']

1 regression(s).
```

Add `--strict` if you also want new failing cases to fail the build.

## Where to go next

- [Architecture overview](architecture.md) — how four packages compose without
  importing each other, and why it is done with protocols.
- Design decisions, one page per package:
  [core](packages/core.md) ·
  [guardrails](packages/guardrails.md) ·
  [observability](packages/observability.md) ·
  [evals](packages/evals.md)
- [`examples/`](../examples) — a runnable script per package, plus the combined
  one. All offline, all with no API key.

## A note on what this does not do

Guardrail detection is pattern- and heuristic-based: it will miss PII phrased
unusually and injections written in plain prose. Cost figures are estimates
from a dated price table. Eval criteria are mechanical and cannot tell you an
answer was right for the wrong reason.

These are real controls that raise the cost of a mistake and produce the
evidence a reviewer needs. They are not a guarantee, and they are not a
substitute for limiting what the agent can reach in the first place. Each
package's design page has a `Limitations` section that is specific about where
the edges are.
