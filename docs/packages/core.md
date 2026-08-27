# wardhook-core — design decisions

This page explains *why* core is shaped the way it is. For usage, see the
[package README](../../packages/wardhook-core/README.md).

## The problem

A LangGraph agent is easy to prototype and awkward to govern. The graph is
yours, the model client is yours, and anything you bolt on tends to assume it
owns the whole stack. Core's job is to be the runtime that *hosts* governance
without requiring it — useful alone, and the substrate the other three packages
attach to.

## Decision: composition through structural typing

**The constraint:** `AgentGraph(guardrails=[...], telemetry=True)` must work,
while `wardhook-core` remains installable and useful with neither sibling
present.

**Options considered:**

| Approach | Why not |
| --- | --- |
| Core depends on guardrails and observability | Kills standalone installability, the whole point of the split. |
| Guardrails and observability depend on core | Anyone wanting PII redaction in a non-LangGraph app drags in LangGraph. |
| A shared `wardhook-common` base package | A fifth package for two protocol definitions, and every install pulls it. |
| **Structural contracts (chosen)** | Zero dependency in either direction. |

Core declares `typing.Protocol` shapes in `wardhook/core/protocols.py` and
duck-types anything it receives. The other packages implement those shapes
without importing core. Neither side knows the other's types exist.

Three rules keep it honest:

1. **No sibling import at module scope.** `telemetry=True` triggers a lazy
   import inside `AgentGraph._resolve_telemetry`, guarded so a missing package
   produces `pip install wardhook-observability` rather than a traceback.
2. **Context is a plain `dict`, not a class.** A guardrail author never imports
   a Wardhook type to read `run_id` or `principal`.
3. **Results are normalised, not type-checked.** `read_guardrail_result()`
   accepts a dataclass, a dict, `None`, or a bare bool and produces one
   predictable `GuardrailDecision`.

**The cost:** a static type checker cannot verify a guardrail satisfies the
contract at the call site. We accept that and pin the contract with tests on
both sides of the seam instead — plus CI jobs that install each package alone
and run its suite, which is the check that actually matters.

## Decision: unrecognised guardrail actions allow, but are recorded

`read_guardrail_result()` maps an unknown action string to `ALLOW` rather than
raising, and stores the original in `details["raw_action"]`.

Raising would mean a third-party guardrail returning something unexpected takes
down a production agent. Silently allowing would hide it. Recording the anomaly
in the audit trail while keeping the agent alive is the only option that fails
usefully in both directions.

## Decision: guardrails fail closed by default

If a guardrail raises, `guardrail_error_policy="block"` (the default) treats it
as a block. For a governance tool, a crashed detector silently passing traffic
is the worst available outcome — it looks exactly like a clean run.

The error is recorded even under `"allow"`. A guardrail that crashed is
precisely what a compliance reviewer needs to see, so fail-open must not also
mean fail-silent. `"raise"` is available for development.

## Decision: the model interface is `.invoke()`, not `BaseChatModel`

`resolve_model()` accepts any object with an `.invoke()` method. This is what
makes the entire test suite runnable with no API key: a fake model is a first-
class citizen, not a mocking workaround. It also means community wrappers that
do not subclass `BaseChatModel` work without special-casing.

Passing a name (`"claude-opus-5"`) resolves through a small provider table, and
a missing provider package produces the exact `pip install` line to fix it.

## Decision: retrieval returns records, not prose

The alternative — asking the model to write source names into its answer — has
two failure modes: the model can cite a document it was never shown, and the
caller has to parse citations back out of free text.

Instead, `Retriever.context_for()` returns the numbered context block *and* the
citation list from a single search, so `[2]` in the answer is always
`citations[1]`. The records carry source, chunk index, score and text, and
travel in agent state alongside the answer.

## Decision: hashing embeddings as the default

`HashingEmbeddings` is a classical hashed bag-of-words vectoriser with no model
weights and no network calls, so `pip install wardhook-core` gives you a working
RAG pipeline immediately.

It matches on shared vocabulary, not meaning — it will not connect "car" to
"automobile". That limit is documented rather than hidden, and swapping in real
embeddings is a one-argument change.

One implementation detail worth stating: it hashes with BLAKE2b rather than
Python's built-in `hash()`, which is salted per process. An index saved in one
process must remain searchable in the next.

## Decision: a blocked run returns HTTP 200

The serve layer returns `200` with `blocked: true` when a guardrail stops a run.
A guardrail firing is a *successful* policy evaluation, not a server fault.
Returning 4xx or 5xx would make error-rate dashboards fire on correct behaviour
and make real failures impossible to spot.

## Decision: tool errors go to the model, not the caller

A tool that raises produces a `ToolMessage` describing the error rather than
propagating. An agent that can see a failure can often route around it; aborting
the run removes that chance. The same applies to guardrail denials — the model
is told it was denied and can try another approach.

`max_tool_iterations` (default 10) is the backstop against a model that loops on
a failing tool forever.

## Implementation notes worth knowing

**`state.py` deliberately omits `from __future__ import annotations`.** Under
PEP 563 stringised annotations, `typing_extensions.TypedDict` cannot see through
`NotRequired[...]` at class-creation time. Every key silently becomes required,
and LangGraph then demands a fully populated state dict. The file carries a
comment saying so.

**`guardrail_events` uses an `operator.add` reducer.** Without it, LangGraph
replaces the list on each node update, so the output stage's events would
overwrite the input stage's — and the audit trail would show only the last node
that ran.

**Graph shape is built to fit.** Nodes are added only when the corresponding
feature is configured. An agent with no tools, guardrails or retriever compiles
to a single model call.

## Known limitations

- The in-memory vector store holds everything in one NumPy matrix. Fine to
  roughly a hundred thousand chunks; past that, use a purpose-built store
  behind the same protocol.
- Streaming is available through `agent.graph` (the compiled LangGraph object)
  but `invoke()` itself is not incremental.
- Multi-modal content is flattened to its text blocks before guardrails see it.
  Image and audio content is not inspected.
