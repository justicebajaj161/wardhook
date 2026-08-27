# Example fixtures

Data used by the runnable examples in the parent directory.

| File | Used by | What it is |
| --- | --- | --- |
| `claims_cases.jsonl` | `examples/evals_run.py`, `examples/combined_agent.py` | Seven test cases for the example claims assistant, exercising text, tool, RBAC, and latency criteria |

The RAG examples (`core_rag_agent.py`, `combined_agent.py`) keep their sample
corpus inline as string constants rather than loading it from here. That is
deliberate: an example you can read top to bottom, without opening a second
file to find out what the agent was given, is a better example.

Case files support `#` comments and blank lines, so `claims_cases.jsonl` is
annotated in place.
