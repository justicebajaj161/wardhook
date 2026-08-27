# wardhook

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)

Governance and observability for LangGraph agents.

This package contains no code. It installs all four Wardhook packages at a
matching version, for when you want the whole toolkit:

```bash
pip install wardhook
```

Which is equivalent to:

```bash
pip install wardhook-core wardhook-guardrails wardhook-observability wardhook-evals
```

| Package | What it does |
| --- | --- |
| [wardhook-core](../wardhook-core) | LangGraph agent runtime, tool calling, RAG with real source citations, one-command FastAPI server |
| [wardhook-guardrails](../wardhook-guardrails) | PII redaction, prompt-injection scoring, tool RBAC, compliance audit trail |
| [wardhook-observability](../wardhook-observability) | Per-node tokens, cost, and latency; static HTML trace viewer |
| [wardhook-evals](../wardhook-evals) | JSONL test cases, pass/fail runner, baseline regression detection |

## Extras

Provider clients and optional backends, passed through to whichever package
owns them:

```bash
pip install "wardhook[anthropic]"   # langchain-anthropic, for a real model
pip install "wardhook[openai]"      # langchain-openai
pip install "wardhook[chroma]"      # Chroma vector store instead of the built-in one
pip install "wardhook[judge]"       # langchain-core, for the llm_judge eval criterion
pip install "wardhook[all]"         # all of the above
```

## Should you use this, or the individual packages?

**Use the individual packages** if you want one capability. Each installs and
works entirely alone — that is the point of the project, and CI proves it by
installing each one in an environment containing no other Wardhook package and
running its suite there. `wardhook-guardrails` in particular has a single
runtime dependency (PyYAML) and needs no agent framework at all.

**Use this package** if you want all four and would rather write one line. It
pins each dependency to an exact version, so `wardhook==0.1.0` means precisely
the 0.1.0 set. If you need to mix versions, depend on the packages directly.

## Links

- [Quickstart](../../docs/quickstart.md) — all four packages into one working agent
- [Architecture overview](../../docs/architecture.md) — how they compose without importing each other
- [Main repository](https://github.com/justicebajaj161/wardhook)
