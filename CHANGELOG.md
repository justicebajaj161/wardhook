# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

All four packages are versioned in lockstep while the project is pre-1.0.

## [Unreleased]

### Added

- **wardhook-core** — `AgentGraph`, a LangGraph agent runtime that builds only
  the nodes its configuration needs. Provider-agnostic model resolution
  (any object with `.invoke()`, or a name resolved through a provider table).
  Tool registration from plain documented callables. A RAG pipeline covering
  PDF/Markdown/text loading, recursive chunking with overlap, a dependency-free
  hashing embedder, a NumPy vector store with on-disk persistence, and retrieval
  that returns structured, citable source records. A FastAPI application factory
  and a `wardhook serve` command. A multi-stage, non-root `Dockerfile`.
- **wardhook-core** — `wardhook.core.protocols`, the structural contracts that
  let guardrails and telemetry attach without either side importing the other.
- **wardhook-guardrails** — Config-driven entity packs (`default`, `insurance`,
  `healthcare`, `fintech`) loadable from YAML with `extends` inheritance. PII
  detection and redaction with Luhn, IBAN, and NHS checksum validation plus
  context-word gating for generic shapes. A weighted-signal prompt-injection
  scorer across seven categories. Deny-by-default role-based access control for
  tool calls. A JSONL audit logger whose records describe what changed without
  ever containing the data being audited.
- **wardhook-observability** — `Tracer`, a telemetry sink satisfying core's
  `TelemetryProtocol`, recording per-node latency, token usage, and estimated
  cost. Token counts are read from the provider's own `usage_metadata` rather
  than re-tokenised locally. A cache-aware cost model that prices the uncached
  remainder, so cached tokens are not billed twice, over a price table carrying
  its own `PRICES_AS_OF` date and overridable with `register_price()`.
  `instrument()` traces a graph you already built by reading LangGraph's
  callback stream rather than patching its internals. `JSONLTraceStore` for
  durable history, a bounded in-memory ring for long-lived processes, and
  `render_html()` plus a `wardhook-trace` CLI producing a single self-contained
  HTML page that makes no network requests.
- **wardhook-evals** — A JSONL test-case format with line-numbered parse
  errors, and `EvalRunner`, which targets anything exposing `.invoke()` rather
  than any particular agent framework. Ten built-in criteria (`contains`,
  `not_contains`, `regex`, `equals`, `json_path`, `tool_called`, `blocked`,
  `max_latency_ms`, `max_cost_usd`, `llm_judge`) in an open registry, with
  `llm_judge` duck-typing its model so grading never pulls in a model library.
  Baseline comparison classifying every case `unchanged` / `fixed` /
  `regressed` / `still_failing` / `added` / `removed`, so a suite carrying known
  failures still blocks the change that broke something. A `wardhook-eval` CLI
  with `run`, `compare`, and `validate`; `compare` exits non-zero only on a
  regression.

### Notes

- Neither published package imports another. CI installs each in isolation and
  runs its suite there, so standalone installability is a checked property.
- The full test suite runs offline against fake models; no API key is required.

## [0.1.0] — unreleased

Initial release. Not yet published to PyPI.

[Unreleased]: https://github.com/justicebajaj161/wardhook/compare/main...HEAD
