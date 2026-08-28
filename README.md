# Wardhook

**Governance and observability for LangGraph agents — as four libraries you can adopt one at a time.**

[![CI](https://github.com/justicebajaj161/wardhook/actions/workflows/ci.yml/badge.svg)](https://github.com/justicebajaj161/wardhook/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2.svg)](https://mypy-lang.org)

[![PyPI core](https://img.shields.io/pypi/v/wardhook-core?label=wardhook-core)](https://pypi.org/project/wardhook-core/)
[![PyPI guardrails](https://img.shields.io/pypi/v/wardhook-guardrails?label=wardhook-guardrails)](https://pypi.org/project/wardhook-guardrails/)
[![PyPI observability](https://img.shields.io/pypi/v/wardhook-observability?label=wardhook-observability)](https://pypi.org/project/wardhook-observability/)
[![PyPI evals](https://img.shields.io/pypi/v/wardhook-evals?label=wardhook-evals)](https://pypi.org/project/wardhook-evals/)
[![PyPI meta](https://img.shields.io/pypi/v/wardhook?label=wardhook)](https://pypi.org/project/wardhook/)

---

## The problem

Teams shipping LLM agents into regulated domains — insurance, healthcare,
fintech — get asked three questions they usually cannot answer:

> *What personal data passed through this agent?*
> *What did it cost, and where?*
> *Did last week's prompt change break anything?*

The existing answers are a hosted SaaS platform or an all-or-nothing framework
you have to rebuild your agent inside. Wardhook is neither. It is four small
libraries, each answering one of those questions, each of which drops into an
agent you already own.

## The four packages

| Package | Answers | Runtime dependencies |
| --- | --- | --- |
| **[wardhook-core](packages/wardhook-core)** | LangGraph agent runtime, tool calling, RAG with real source citations, one-command FastAPI server | langgraph, langchain-core, fastapi |
| **[wardhook-guardrails](packages/wardhook-guardrails)** | PII redaction, prompt-injection scoring, tool RBAC, compliance audit trail | **PyYAML only** |
| **[wardhook-observability](packages/wardhook-observability)** | Per-node tokens, cost, and latency; static HTML trace viewer | langchain-core, typer |
| **[wardhook-evals](packages/wardhook-evals)** | JSONL test cases, pass/fail runner, baseline regression detection | typer only |
| **[wardhook](packages/wardhook)** | Meta-package: installs all four at a matching version. No code of its own. | the four above |

**Each installs and works completely alone.** That is not a claim in a README —
CI installs each package in an isolated environment with no siblings present and
runs its full suite there. If a package ever reaches for one of the others, the
build fails.

### Why not LangSmith, Langfuse, Presidio or NeMo Guardrails?

Because of two constraints those tools do not meet, and which this project is
built around. **`wardhook-guardrails` depends on PyYAML and nothing else** —
Presidio pulls spaCy plus its models, hundreds of megabytes, which is the
difference between fitting in a Lambda or an air-gapped container and not.
**Nothing leaves your machine** — LangSmith and Langfuse both want your traces
on their infrastructure, and the local dashboard above is the alternative to a
hosted console rather than a route to one.

If neither constraint binds for you, those tools are more mature and you should
probably use them. The honest scope of this one is: *PII redaction and cost
visibility in a LangGraph agent, offline, without pulling 300MB of ML
dependencies.*

## Install

Take one:

```bash
pip install wardhook-core            # the agent runtime
pip install wardhook-guardrails      # PII, injection, RBAC, audit
pip install wardhook-observability   # tokens, cost, latency
pip install wardhook-evals           # test cases and regression detection
```

Or take all four:

```bash
pip install wardhook                 # a meta-package; installs the four above
```

Nothing changes for the others either way. `wardhook` contains no code — it
exists so you can write one line instead of four, and it pins each package to an
exact version so `wardhook==0.1.0` means precisely the 0.1.0 set. Provider
clients ride along as extras: `wardhook[anthropic]`, `[openai]`, `[google]`,
`[chroma]`, `[judge]`, `[all]`.

## Quickstart

Guardrails on their own, with no agent framework anywhere. This runs exactly as
written after `pip install wardhook-guardrails`, including the output in the
comments:

```python
from wardhook.guardrails import PIIRedactor, AuditLogger

redactor, audit = PIIRedactor(pack="healthcare"), AuditLogger("audit.jsonl")

result = redactor.on_output("Patient MRN-4471902, contact bob@clinic.org", {})
audit.record(result, stage="output", run_id="req-17")

print(result.text)  # 'Patient [MRN], contact [EMAIL]'
print(audit.report())  # counts by action, stage, guardrail, severity, entity
```

All four composed into one governed, traced, testable agent. Unlike the block
above, this one sketches an integration rather than running as-is: `policy.md`,
`lookup_claim` and `issue_refund` are yours, and `model="claude-opus-5"` needs
`pip install "wardhook[anthropic]"` plus a key. For a version that runs right
now with no key and no setup, see [`examples/combined_agent.py`](examples/combined_agent.py).

```python
from wardhook.core import AgentGraph, InMemoryVectorStore, Retriever, chunk_text
from wardhook.guardrails import PIIRedactor, InjectionDetector, RoleBasedToolPolicy

store = InMemoryVectorStore()
store.add(chunk_text(open("policy.md").read(), "policy.md"))

agent = AgentGraph(
    model="claude-opus-5",
    tools=[lookup_claim, issue_refund],
    retriever=Retriever(store),
    guardrails=[
        InjectionDetector(),  # score user input
        PIIRedactor(pack="insurance"),  # redact both ways
        RoleBasedToolPolicy({"agent": ["lookup_*"]}),  # gate tool calls
    ],
    telemetry=True,  # tokens, cost, latency
)

result = agent.invoke(
    "What excess applies to storm damage on POL-889231?",
    principal={"id": "u-17", "roles": ["agent"]},
)

result["output"]  # the answer
result["citations"]  # [{'source': 'policy.md', 'chunk_index': 3, 'score': 0.71, ...}]
result["guardrail_events"]  # every policy decision, with no PII in the record
agent.trace()  # per-node tokens, cost, and latency
```

## The local dashboard

```bash
wardhook serve myapp.agents:support_agent --dashboard
# Serving myapp.agents:support_agent on http://127.0.0.1:8000  (docs at /docs)
# Dashboard at http://127.0.0.1:8000/dashboard/
```

One page, served from your own process, showing the agent's real graph with each
node shaded by what it actually cost. Nothing is fetched from a CDN and no trace
leaves the machine — the whole page is a few kilobytes of inlined HTML, CSS and
SVG, and it works with the network unplugged.

The diagram is derived from the compiled graph, so it is accurate to your
configuration rather than to a template: an agent with no retriever has no
`retrieve` box. Picking a run shades each node by its latency and writes its
tokens and cost underneath. A node visited more than once — `call_model`, on
every tool round trip — is totalled, so what a box says reconciles with the run
totals above it.

**It shows telemetry and configuration. It never shows content.** No prompt, no
model output, no retrieved chunk, no guardrail event body. That is not a filter
applied on the way out: the telemetry model has no such fields, and the API's
projection is an explicit allowlist with a test proving a content field added
upstream cannot start being served by accident. When you need the text behind a
cost, the page gives you the run's `run_id` — the same id that is on every audit
record you write — and you look it up in **your** audit log, where your
redaction and retention policy already apply.

**It is off by default and takes two opt-ins to reach a network.** Enabling it is
explicit (`--dashboard`, `WARDHOOK_DASHBOARD=1`, or `create_app(dashboard=True)`);
serving it on anything other than a loopback interface additionally requires
`--dashboard-allow-remote`. A governance tool must not ship a debug UI that
becomes reachable in production because nobody turned it off.

**Under multiple workers it sees one worker's traffic.** Each process owns its
own in-memory tracer, so `--workers 4` means roughly a quarter of the runs. The
page says which mode it is in rather than quietly under-reporting; point the
tracer at a shared `JSONLTraceStore` and hand that store to
`create_app(telemetry=store)` to see everything.

There is also a JSON API under the same mount — `/api/topology`, `/api/runs`,
`/api/runs/{run_id}` — if you would rather build your own view. `/api/topology`
includes Mermaid source for the same graph.

## How it composes without coupling

The interesting constraint is that `AgentGraph(guardrails=[...], telemetry=True)`
has to work while `wardhook-core` stays installable and useful with neither
sibling present.

Wardhook solves this with **structural typing across package boundaries**. Core
declares `typing.Protocol` shapes and duck-types whatever it is handed. The
other packages implement those shapes without importing core. Neither side
knows the other's types exist.

```
            wardhook-core                          wardhook-guardrails
      ┌───────────────────────────┐          ┌──────────────────────────────┐
      │  GuardrailProtocol        │◄╌╌╌╌╌╌╌╌╌│  PIIRedactor                 │
      │  TelemetryProtocol        │          │  InjectionDetector           │
      │                           │          │  RoleBasedToolPolicy         │
      │  ┌─────────────────────┐  │          └──────────────────────────────┘
      │  │  AgentGraph         │  │                 structural match,
      │  │                     │  │                 no import either way
      │  │  guard_input        │  │          ┌──────────────────────────────┐
      │  │    → retrieve       │  │◄╌╌╌╌╌╌╌╌╌│  wardhook-observability      │
      │  │    → call_model     │  │          │  Tracer, get_trace()         │
      │  │    ⇄ tools          │  │          └──────────────────────────────┘
      │  │    → guard_output   │  │
      │  └─────────────────────┘  │          ┌──────────────────────────────┐
      └───────────────────────────┘◄╌╌╌╌╌╌╌╌╌│  wardhook-evals              │
             any object with          targets │  EvalRunner, baselines       │
             .invoke() works                  └──────────────────────────────┘
```

The practical payoff: **anything of the right shape works**. A guardrail can
come from this repo, from your codebase, or from a third-party library, and the
runtime drives it identically:

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


agent = AgentGraph(model="claude-opus-5", guardrails=[NoInternalCodenames()])
```

No base class, no import, no dependency. Full rationale in
[docs/architecture.md](docs/architecture.md).

## Project status

Pre-1.0 and built in the open. All four packages are complete.

| Package | Status | Tests | Coverage |
| --- | --- | --- | --- |
| wardhook-core | ✅ Complete | 317 | 100% |
| wardhook-guardrails | ✅ Complete | 222 | 100% |
| wardhook-observability | ✅ Complete | 162 | 100% |
| wardhook-evals | ✅ Complete | 170 | 100% |
| wardhook (meta) | ✅ Complete | — | n/a |

Those numbers are generated, not transcribed: `make cov-table` reproduces this
table from a real run, and CI fails the build if coverage drops below 100%.
A further 11 cross-package tests in [`tests/`](tests) exercise the composition
itself, which no single package's suite can reach.
Coverage is **branch** coverage, which reads lower than line coverage. The few
genuinely unreachable defensive branches carry a `pragma: no cover` naming why,
rather than being deleted to flatter the number.

All five distributions are on PyPI as of 0.1.0. Releases run through
[PyPI Trusted Publishing](docs/releasing.md) on a version tag — there is no API
token stored anywhere in this repository.

To work on it rather than use it:

```bash
git clone https://github.com/justicebajaj161/wardhook.git
cd wardhook && uv sync && make check
```

## Development

```bash
make install    # create the dev environment with all four packages editable
make check      # lint + type-check + coverage-gated test, the gate CI runs
make solo       # prove each package installs and passes its suite entirely alone
make cov-table  # regenerate the Project status table above from a real run
make build      # wheels and sdists for all five, validated with twine
```

The whole test suite runs **fully offline against fake models**. No API key is
needed to contribute, and no test touches the network.

## Documentation

- [Architecture overview](docs/architecture.md) — the seam design and why structural typing
- [Quickstart](docs/quickstart.md) — all four packages into one working agent
- Design decisions, one page per package:
  [core](docs/packages/core.md) ·
  [guardrails](docs/packages/guardrails.md) ·
  [observability](docs/packages/observability.md) ·
  [evals](docs/packages/evals.md)
- [Releasing](docs/releasing.md) — how versions are cut and published
- [`examples/`](examples) — one runnable script per package plus a combined one,
  all offline, none needing an API key

## A note on what guardrails can and cannot do

Detection is pattern- and heuristic-based. It will miss PII phrased unusually or
in an unsupported language, and it will miss a novel injection written in plain
prose. These are real controls that raise the cost of a mistake and produce the
evidence a reviewer needs — not a guarantee, and not a substitute for limiting
what the agent can reach in the first place. That position is argued in full in
[the guardrails design doc](docs/packages/guardrails.md#limitations) rather than
buried.

### It has been measured, and the number is published

| | |
|---|---|
| Recall — labelled span covered by some rule | **84.3%** |
| Precision — detection landed on real PII | **94.2%** |
| Corpus | 2,076 labelled spans across 760 documents, all four packs |

Full tables, the method, and what these figures do *not* support are in
[`benchmarks/pii/`](benchmarks/pii). Reproduce with `make bench-pii`.

The corpus is generated from real-world identifier formats rather than from
Wardhook's own patterns — a corpus built out of the regexes under test would
score near 1.0 and measure nothing. It includes the written variants a pattern
may not anticipate, and 160 clean documents full of near-misses that must be
left alone.

**The number is published whatever it says**, and this one names specific
weaknesses: `DATE_OF_BIRTH` sits at 37% because the pattern matches
`DD/MM/YYYY` and not `1982-03-14`; `US_SSN` at 53% because it requires the
dashes; `SWIFT_BIC` over-redacts capitalised words near "wire". Those are
reproducible and fixable, and the figure was published before anything was
tuned so the next one can be compared against something nobody had an incentive
to flatter. A documented 84% you can check beats an undocumented claim you
cannot.

### Audit logging records blocks and redactions, not allows

`AuditLogger` defaults to `record_allows=False`. An allow is the overwhelmingly
common outcome, and logging every one buries the handful of actions a reviewer
opened the file to find. Turn it on where a regime expects positive evidence
that every request was screened:

```python
AuditLogger("audit.jsonl", record_allows=True)
```

This is a deliberate departure from a literal reading of "log every guardrail
action", made because a trail nobody can read is not a control. The reasoning
is in [the design doc](docs/packages/guardrails.md); the behaviour is one
constructor argument either way.

## Contributing

Issues and pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Security reports go through the private
channel in [SECURITY.md](SECURITY.md), not the issue tracker.

## License

MIT — see [LICENSE](LICENSE).
