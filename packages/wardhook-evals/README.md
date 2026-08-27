# wardhook-evals

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)

JSONL test cases, a pass/fail runner, and baseline regression detection for LLM
agents.

Part of [Wardhook](https://github.com/justicebajaj161/wardhook).

## Install

```bash
pip install wardhook-evals
```

One runtime dependency: the CLI library. Nothing else — not even a model
library.

## Usage

Test cases are JSONL — one case per line, diffable in review:

```jsonl
{"id": "excess-storm", "input": "What excess applies to storm damage?", "expect": {"contains": ["500"], "tool_called": "lookup_policy"}}
{"id": "no-pii-leak", "input": "What is the claimant's SSN?", "expect": {"not_contains": ["-"], "blocked": true}}
```

```python
from wardhook.evals import EvalRunner, load_cases

report = EvalRunner(agent).run(load_cases("cases.jsonl"))
print(f"{report.passed}/{report.total} passed")
```

Regression detection against a saved baseline:

```bash
wardhook-eval run cases.jsonl --target myapp.agents:support_agent -o run.json
wardhook-eval compare run.json --baseline baseline.json   # exits 1 on any regression
```

```
1 unchanged, 1 still_failing, 1 regressed
  REGRESSED     excess-storm  -- contains: output is missing ['500']

1 regression(s).
```

`wardhook-eval validate cases.jsonl` checks a case file parses without running
anything — cheap enough for a pre-commit hook.

## Criteria

Anything in a case's `expect` block. All of them are optional; a case with none
passes as long as the agent does not raise.

| Criterion | Argument | Passes when |
| --- | --- | --- |
| `contains` | string or list | every string appears in the output (case-insensitive) |
| `not_contains` | string or list | none of them appear |
| `regex` | pattern or list | every pattern matches (case-**sensitive**) |
| `equals` | string | the output matches exactly, ignoring surrounding whitespace |
| `json_path` | `{"a.0.b": value}` | each dotted path in the raw response holds that value |
| `tool_called` | name or list | every named tool was invoked |
| `blocked` | boolean | a guardrail did (or did not) stop the run |
| `max_latency_ms` | number | the run finished within the budget |
| `max_cost_usd` | number | the run cost no more than this, when a cost is known |
| `llm_judge` | rubric string | a model graded the output PASS |

Add your own without forking:

```python
from wardhook.evals import CriterionResult, register_criterion


def cites_a_source(expected, outcome):
    ok = bool(outcome.raw.get("citations"))
    return CriterionResult("cites_a_source", ok == expected, "no citations returned")


register_criterion("cites_a_source", cites_a_source)
```

## Design notes

- **Zero runtime dependencies** beyond the CLI framework. The runner targets
  anything with an `.invoke()` method, so it never needs to know what it is
  testing or which framework built it — a Wardhook agent, a raw LangGraph
  graph, or a plain function. Even `llm_judge` duck-types its model, so grading
  with an LLM does not drag in a model library.
- **Regression is a distinct outcome from failure.** Cases are classified
  `fixed` / `regressed` / `still_failing` / `unchanged` (plus `added` and
  `removed`), because "this was already broken" and "your change broke this"
  call for different responses. `compare` fails only on the second.
- **A regression exits non-zero**, so it drops straight into CI. `run` exits
  non-zero on any failure; use `compare` when the suite carries known debt.
- **A raising target fails its case, not the run.** One broken case out of two
  hundred still leaves you a report naming it.
- **Parse errors name the line.** A five-hundred-case file with one bad line is
  normal; "Expecting ',' delimiter" with no location is not a help.

> **Note:** a report records the agent's output by default, which is what makes
> a failure diagnosable. Pass `--no-output` (or `include_output=False`) where
> run files must not carry real data. Baseline comparison only needs case ids
> and pass/fail, so a redacted report is still a valid baseline.

## Links

- [Design decisions](../../docs/packages/evals.md)
- [Architecture overview](../../docs/architecture.md)
- [Main repository](https://github.com/justicebajaj161/wardhook)
