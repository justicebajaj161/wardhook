# wardhook-core

[![CI](https://github.com/justicebajaj161/wardhook/actions/workflows/ci.yml/badge.svg)](https://github.com/justicebajaj161/wardhook/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/wardhook-core.svg)](https://pypi.org/project/wardhook-core/)
[![Python](https://img.shields.io/pypi/pyversions/wardhook-core.svg)](https://pypi.org/project/wardhook-core/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

A LangGraph agent runtime with tool calling, retrieval with real source
citations, and a one-command FastAPI server.

Part of [Wardhook](https://github.com/justicebajaj161/wardhook). Works
completely on its own — it never imports the other Wardhook packages.

## Install

```bash
pip install wardhook-core

# with a model provider
pip install "wardhook-core[anthropic]"   # or [openai], [all]
```

## Usage

```python
from wardhook.core import AgentGraph, InMemoryVectorStore, Retriever, chunk_text

store = InMemoryVectorStore()
store.add(chunk_text(open("policy.md").read(), "policy.md"))

agent = AgentGraph(model="claude-opus-5", retriever=Retriever(store))
result = agent.invoke("What excess applies to storm damage?")

print(result["output"])
print(result["citations"][0]["source"])  # -> 'policy.md'
```

## What you get

**Provider-agnostic.** `AgentGraph` accepts any object with `.invoke()`. Pass a
`ChatAnthropic`, a `ChatOpenAI`, or a test double. Nothing in the base install
depends on a provider SDK, so you are never locked in.

```python
agent = AgentGraph(model=ChatOpenAI(model="gpt-4o"))  # instance
agent = AgentGraph(model="anthropic:claude-opus-5")  # name
agent = AgentGraph(model=my_fake)  # test double, no API key
```

**Tools from plain functions.** The docstring becomes the description the model
reads, so it is required rather than optional.

```python
def lookup_claim(claim_id: str) -> str:
    """Look up the current status of a claim by its identifier."""
    return db.claims.status(claim_id)


agent = AgentGraph(model="claude-opus-5", tools=[lookup_claim])
```

**Citations are structural, not parsed.** Retrieval returns records carrying
source, chunk position and score. You render or verify them directly; the model
cannot invent a citation for a document it was never shown.

```python
for c in result["citations"]:
    print(f"{c['source']} chunk {c['chunk_index']} (score {c['score']:.3f})")
```

**RAG that runs with no API key.** Document loading (PDF, Markdown, text),
recursive chunking with overlap, and a NumPy vector store. The default
embeddings are a classical hashing vectoriser — no model weights, no network —
so the pipeline works the moment you install it. Swap in real embeddings for
production; the interface is identical.

```python
store = InMemoryVectorStore(embeddings=OpenAIEmbeddings())
store.save("index")  # index.npz + index.json
store = InMemoryVectorStore.load("index", embeddings=OpenAIEmbeddings())
```

**Serve it in one command.**

```bash
wardhook serve myapp.agents:support_agent --port 8000
```

Exposes `POST /invoke`, `GET /health`, `GET /info`, and OpenAPI docs at `/docs`.
A production `Dockerfile` ships with the package.

## Composing with the rest of Wardhook

`AgentGraph` takes `guardrails=[...]` and `telemetry=True`, but core does not
depend on the packages that provide them. Both attach through structural
contracts in `wardhook.core.protocols`, so **any object of the right shape
works** — from Wardhook, from your codebase, or from somewhere else entirely.

```python
from wardhook.core import AgentGraph
from wardhook.guardrails import PIIRedactor, RoleBasedToolPolicy  # optional install

agent = AgentGraph(
    model="claude-opus-5",
    tools=[lookup_claim],
    guardrails=[PIIRedactor(pack="insurance"), RoleBasedToolPolicy(...)],
    telemetry=True,  # requires wardhook-observability
)
```

Writing your own takes no dependency at all:

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

The full contract is three optional hooks — `on_input`, `on_output`,
`on_tool_call` — each returning something with an `action` of `"allow"`,
`"redact"` or `"block"`. Implement only the ones you need.

## Notes

- **Guardrails fail closed.** If a guardrail raises, the run is blocked and the
  failure is recorded. Set `guardrail_error_policy="allow"` or `"raise"` to
  change that.
- **Denied tool calls never execute.** The model is told it was denied through a
  normal tool result, so it can recover instead of failing the run.
- **Tool errors go back to the model**, not to your caller — an agent that can
  see a failure can often work around it.

## Links

- [Design decisions](../../docs/packages/core.md)
- [Architecture overview](../../docs/architecture.md)
- [Main repository](https://github.com/justicebajaj161/wardhook)
