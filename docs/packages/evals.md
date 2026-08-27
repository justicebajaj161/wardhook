# wardhook-evals — design decisions

This page explains *why* the package is shaped the way it is. For what it does
and how to call it, see the [package README](../../packages/wardhook-evals/README.md).

## The problem

Changing a prompt is a code change with no compiler and no stack trace. The
usual workflow is to edit the system prompt, try three questions by hand,
decide it looks better, and ship. What that misses is everything it did not
occur to you to re-check — the refund policy it used to quote correctly, the
tool it used to call, the PII it used to refuse to repeat.

Teams that notice this reach for a test suite, and immediately hit a second
problem: a real agent suite is never fully green. There are always a handful of
cases that fail for reasons nobody has had time to fix. A gate that blocks on
"any failure" blocks every build forever, so it gets disabled, and then nothing
is checked at all.

This package exists for that second problem as much as the first. The design
question is not "how do I run test cases" — it is "how do I make a suite with
known failures still able to block the change that broke something".

## Decision: the runner knows nothing about agents

`EvalRunner` accepts **anything with an `.invoke()` method**. Not a base class
to subclass, not an adapter to register, not a framework to be built on.

The alternative — importing the agent framework and typing against it — would
have made the runner simpler internally and useless to everyone not using that
exact framework. It would also have meant an eval suite could not outlive a
migration: rewriting an agent from LangChain to LangGraph would mean rewriting
the tests that prove the rewrite worked, which is exactly backwards.

The cost is a small amount of shape-guessing. Targets return wildly different
things — a dict, a string, a message object — so the runner flattens whatever
came back into a single `Outcome` before any criterion sees it. Where a target
returns a mapping, the final text is looked for under `output`, `answer`,
`text`, `response`, `content`, and `result`, in that order.

This is also why the package's only runtime dependency is its CLI library.
Evaluating an agent should not require installing the framework that agent was
built with.

## Decision: even the LLM judge is duck-typed

`llm_judge` grades free-form output against a rubric using a model, and the
obvious implementation imports a model library.

It does not. The judge is whatever object you pass as `EvalRunner(target,
judge=...)`, called via `.invoke()`. That keeps the "no framework dependency"
promise true even for the one criterion that inherently needs a model, and it
means you grade with the client you already have configured — including its
retries, its timeouts, and its API key — rather than one this package
constructed behind your back.

The judge is never constructed implicitly. An eval run that silently starts
spending money on API calls the caller did not configure is a bad surprise, so
a missing judge fails the criterion with a message saying how to supply one.

## Decision: failing and newly-failing are different facts

This is the core of the package.

Comparing a run against a saved baseline classifies every case into one of six
outcomes, and only one of them fails the build:

| Change | Baseline | Current | Fails the build? |
| --- | --- | --- | --- |
| `unchanged` | pass | pass | no |
| `fixed` | fail | pass | no |
| `regressed` | pass | fail | **yes** |
| `still_failing` | fail | fail | no |
| `added` | absent | any | no (`--strict`: yes, if failing) |
| `removed` | any | absent | no |

`still_failing` is the row that makes the whole thing work. A suite with nine
known failures that stays at nine has not got worse, and blocking a deploy on
it teaches people to ignore the gate. A suite that goes from zero regressions
to one has got worse, and that is worth stopping for.

`added` deliberately does not fail either. Writing a test for a bug before
fixing it is good practice, and a tool that punishes you for it discourages the
practice. `--strict` exists for teams that want the opposite default.

## Decision: cases are matched on a stable id

Baseline comparison keys on `id`, not on the input text or on position in the
file. Position would break the moment anyone reorders a file; input text would
break the moment anyone rephrases a question.

The consequence is that ids are load-bearing, and renaming one reads as a
removal plus an addition rather than as the same case changing. That is
documented, and duplicate ids are rejected at load time rather than silently
making a comparison meaningless — a file with two cases called `smoke-1` would
otherwise produce a baseline where one of them invisibly shadows the other.

## Decision: JSONL, not YAML or a Python DSL

A Python DSL would be more expressive and would let cases share fixtures. It
would also make cases unreadable to anyone who is not a Python programmer,
which is usually the person who best knows what the right answer is — the
domain expert, the compliance reviewer, the support lead.

YAML is more readable than JSON for a human writing by hand, but it is worse in
review: a case added in the middle of a YAML document can reindent its
neighbours, and YAML's type coercion turns `no` into `False` and `1.10` into
`1.1`, which is a poor property for a file full of expected outputs.

JSONL is one case per line. A new case is a one-line diff, a changed
expectation is a one-line diff, and generating cases from a script is trivial.
Blank lines and `#` comments are permitted so a file can still be organised and
annotated.

## Decision: a raising target fails its case, not the run

A target that throws on case 12 of 200 could reasonably abort everything. It
does not: the exception is caught, recorded as that case's failure with its
type and message, and the run continues.

The reasoning is that an eval run is often expensive and slow, and the most
useful output when something is broken is *the full picture* — one crash plus
199 results tells you whether the problem is one case or the whole agent, and
aborting tells you nothing.

## Decision: costs are read back through duck typing

The `max_cost_usd` criterion needs to know what a run cost, and this package
does not measure cost — `wardhook-observability` does.

Rather than depend on it, the runner looks for a `.trace()` method on the
target and reads `total_cost` off whatever comes back. An `AgentGraph` with
telemetry attached satisfies that shape; anything else simply reports no cost,
and the criterion passes with a note saying so.

That an unmeasured cost *passes* rather than fails is deliberate. An unmeasured
budget is not a blown budget, and failing would make the criterion untestable
against any target without telemetry — turning an optional integration into a
hard requirement through the back door.

## Decision: reports carry the agent's output, by default

A failing case is much easier to diagnose when the report says what the agent
actually said. So `output` is recorded, and the report is written as indented,
key-sorted JSON that reviews cleanly.

The tradeoff is real: a run file from an agent handling live data contains live
data. `include_output=False` (or `--no-output`) turns it off, and because
baseline comparison only needs ids and pass/fail, a redacted report is still a
perfectly good baseline. The warning is in the module docstring, the README,
and here, because a default that is convenient and occasionally sensitive
should be stated more than once.

Report and comparison files also recompute their summary blocks on load rather
than trusting what is written. A hand-edited file cannot claim a pass rate its
cases do not support.

## Limitations

- **Criteria are mechanical, not semantic.** `contains` checks substrings; it
  cannot tell you the answer was right for the wrong reason. `llm_judge` exists
  for that and carries the reliability of whatever model you point it at, which
  is to say it is a useful signal and not a verdict.
- **Non-determinism is not handled for you.** An agent that phrases things
  differently each run will produce flaky `equals` and `contains` cases. Prefer
  `regex` and `json_path` against structured fields, or lower the temperature
  for eval runs.
- **Cases run sequentially.** There is no concurrency, which makes a large
  suite against a real provider slow. Sequential execution keeps ordering and
  cost attribution simple; parallelism would be a meaningful addition and is
  not there yet.
- **No cost control on the suite itself.** Running two hundred cases against a
  frontier model costs real money, and nothing here stops you doing it twice by
  accident. `max_cost_usd` measures per case, not per run.
- **Comparison is pass/fail only.** A case that still passes but got noticeably
  slower or more expensive is `unchanged`. Latency and cost regressions have to
  be expressed as criteria on individual cases.

See the [architecture overview](../architecture.md) for how this package
targets `wardhook-core` without either importing the other.
