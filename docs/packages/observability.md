# wardhook-observability — design decisions

This page explains *why* the package is shaped the way it is. For what it does
and how to call it, see the [package README](../../packages/wardhook-observability/README.md).

## The problem

An agent ships. A week later someone asks three questions, usually in this
order: *what is this costing us*, *why is it slow*, and *which step is doing
that*. None of them is answerable from application logs, because the
interesting numbers — tokens, cache hits, per-node latency — exist only inside
the framework for the duration of the call.

The usual answer is a hosted observability platform. That is a reasonable
choice, and a bad default for a governance toolkit: it means shipping prompts
and completions to a third party, in exactly the regulated settings where the
rest of Wardhook is most useful. This package assumes the opposite — that the
data stays on your infrastructure, and that the tool must work offline, inside
a locked-down network, with nothing to run.

## Decision: read usage from the provider, never re-tokenise

The obvious way to count tokens is to run a tokeniser over the prompt. It is
also wrong, in three ways that all push the same direction: it cannot see the
system prompt the provider prepends, it does not know the token cost of the
tool schemas, and it has no idea which parts of the prompt were served from
cache. Every one of those makes a local count an *under*-estimate, which is the
worst direction for a number someone is budgeting against.

`usage_metadata` on the model response is what the provider's own bill is
computed from. Reading it is a callback handler rather than a tokeniser, and it
is exact.

The cost is that usage only exists if the integration reports it. Fake models
in tests report nothing, so the tracer records zeros — which is honest, and why
the test suite asserts on real usage supplied deliberately.

## Decision: price the uncached remainder, not the reported total

This is the subtlest thing in the package, and the easiest bug to ship.

LangChain reports `input_tokens` as the **sum of all input token types**, with
cache reads and cache writes already included. The natural-looking formula —
"total input at the full rate, plus cached tokens at 0.1×" — charges the cached
tokens twice, billing them at 1.1× the input rate instead of 0.1×. That is
eleven times too much for the cached portion, and cache reads are routinely
80–95% of the prompt on an agent that uses caching at all. On the 90%-cached
example in `examples/observability_trace.py` it makes the whole estimate 3.3×
too high, silently.

So the uncached remainder is derived first, and each bucket is priced at its
own multiplier:

```
uncached   = input_tokens - cache_read - cache_write
input_cost = (uncached + cache_read * 0.1 + cache_write * 1.25) * rate
```

A fully cached prompt therefore costs 0.1× the input rate, not 1.1× it. There
is a test asserting exactly that number, because the wrong answer is plausible
enough to survive a code review.

## Decision: an unknown model costs zero and says so

Three options were on the table for a model with no price entry: raise, guess a
plausible rate from a similar model, or report zero.

Raising is wrong — telemetry must not be able to fail an agent run. Guessing is
worse than useless: it puts a confident wrong number into a budget with no
signal that it is wrong, and someone will quote it in a planning meeting.

Zero plus a warning is the honest option. It is obviously not a real cost, the
warning names the model and how to fix it, and `register_price()` fixes it in
one line without forking. The warning fires once per model, because a per-node
cost calculation in a request loop would otherwise flood the logs.

The table also carries a `PRICES_AS_OF` date, exported and rendered in the
footer of every generated page. A hardcoded price table *will* go stale; the
least it can do is admit how old it is.

## Decision: `instrument()` reads the callback stream, not graph internals

To trace a graph the user already built, something has to know where each node
starts and stops. The direct approach is to reach into the compiled graph and
wrap each node's callable.

That was rejected because a compiled `Pregel` object's node layout is not a
public API. Code depending on it works until the next LangGraph release and
then breaks in a way that is hard to diagnose from the outside.

The callback stream *is* public, and it turns out to carry everything needed —
verified against langgraph 1.2 / langchain-core 1.6:

```
on_chain_start   metadata["langgraph_node"] = "call_model"   tags=["graph:step:2"]
on_chain_end     run_id only — no name
```

So node names are recorded on the way in, keyed by `run_id`, and popped on the
way out. Nothing internal is touched. The one thing `instrument()` does patch
is the graph's own `invoke`/`ainvoke`/`stream`/`astream`, so callers do not have
to remember `config={"callbacks": [...]}` on every call — and `uninstrument()`
reverses it.

That patching produced the one genuine bug found while building this: LangGraph
implements `invoke` on top of its own `stream`, and both are wrapped, so a
second handler attached on the inner call and **every node was counted, and
billed, twice**. The fix is an idempotence check on the merged config. A
regression test pins it.

## Decision: node attribution is per-thread

When a token-usage callback fires, the tracer must decide which node to
attribute it to. A single "current node" attribute would be correct in a script
and quietly wrong in a web server, where two requests interleave and each
would steal the other's tokens.

The open-node stack is therefore thread-local, and everything shared is behind
a lock. Two concurrent runs cannot contaminate each other; there is a test that
forces genuine interleaving with a barrier and asserts they do not.

Usage that arrives while no node is open is *not* dropped. It lands on a
synthetic `(ungrouped)` step, on the principle that a cost you cannot attribute
is still a cost you paid, and a number that quietly vanishes is worse than one
labelled awkwardly.

## Decision: memory is bounded by default

A tracer in a long-running server that keeps every trace is a memory leak with
extra steps. Completed traces live in a ring of `max_runs` (100 by default),
evicted oldest-first.

That makes the in-memory view deliberately partial, which is why
`JSONLTraceStore` exists: point a tracer at one and every completed run is
appended to disk as it finishes. Memory stays bounded, history does not.

JSON Lines was chosen over a database because traces are append-only, usually
read in bulk, and frequently end up in a pull-request diff during a prompt
change. Sorted keys mean two runs of the same agent diff line-for-line, and
`tail` and `jq` work without anything being installed.

## Decision: the viewer is one static file, and escaping is a security control

The trace viewer could have been a Streamlit app. It is a single HTML string
with its CSS and JavaScript inlined instead, because a debugging tool that
needs a server and a network is useless in exactly the locked-down environment
where debugging is hardest. The generated file opens from a local path, an
email attachment, or a CI artifact, and behaves identically offline. A test
asserts the rendered page contains no external `src` or `href` at all.

Every interpolated value goes through `html.escape`. That is not formatting:
traces carry node names, error text, model ids, and caller-supplied metadata,
all influenced by user input. If a rendered trace is ever served over HTTP
rather than opened locally, unescaped content is stored cross-site scripting.
The test for this parses the output and asserts no injected element or
attribute survives — a substring check is too weak, because correctly escaped
text legitimately *contains* the dangerous characters.

## Decision: two clocks, on purpose

`started_at` is wall-clock ISO-8601, because a persisted trace is only useful
if you can line it up against your application logs. `latency_ms` comes from
`time.perf_counter()`, which is monotonic.

Using wall clock for durations means an NTP correction or a daylight-saving
transition mid-run can produce a negative latency, and a negative latency in a
dashboard is worse than no dashboard. Using a monotonic clock for timestamps
means they correlate with nothing. Both are recorded because they answer
different questions.

## Limitations

- **Costs are estimates.** They come from a hardcoded table of first-party
  Anthropic API rates, and providers change prices. Partner platforms (Bedrock,
  Vertex AI) bill differently; their model ids resolve here, but the rate shown
  is the first-party one until you `register_price()` your own.
- **Only Anthropic models are priced out of the box.** Everything else costs
  `$0` and warns. Adding a model is one call, not a fork, but the table is not
  pretending to be universal.
- **Parallel branches within a single graph are approximated.** Node
  attribution is per-thread, so a graph that fans out across threads *inside*
  one run may attribute usage to whichever node that thread has open. Separate
  concurrent runs are handled correctly; parallel superstep branches sharing a
  thread are not distinguished.
- **Streaming token counts arrive at the end.** Usage is read from the final
  response, so a trace is complete only once the stream is exhausted.
- **The trace records structure, not content.** Node names, timings, tokens,
  cost — never prompts or completions. That is deliberate, and it means a trace
  cannot tell you *why* a step was slow, only that it was. Note that
  `metadata` is written verbatim, so do not put user text in it.

See the [architecture overview](../architecture.md) for how this package
attaches to `wardhook-core` without either importing the other.
