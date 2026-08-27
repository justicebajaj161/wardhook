# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

All four packages are versioned in lockstep while the project is pre-1.0.

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-08-27

First release. Four independently installable packages plus a meta-package that
installs all four. Pre-1.0: the public API may still change, and the packages
are versioned in lockstep until it settles.

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
- **Docs** — an [architecture overview](docs/architecture.md) explaining the
  structural-typing seam, a [quickstart](docs/quickstart.md) building one agent
  from all four packages, a design-decisions page per package, and
  [releasing](docs/releasing.md).
- **Examples** — one runnable script per package plus `combined_agent.py` using
  all four, with shared fixtures in `examples/data/`. Every example runs offline
  against a fake model and needs no API key; CI executes all five on each push.
- **Release tooling** — `.github/workflows/release.yml` publishes all four
  packages to PyPI on a version tag using Trusted Publishing, so no API token is
  stored in this repository. It re-runs the full gate before building and
  refuses to publish if the tag and the declared versions disagree.
  `scripts/bump-version.sh` sets all eight version declarations at once.
- **wardhook** — a meta-package containing no code, so `pip install wardhook`
  brings in all four packages at matching versions. Extras pass through to
  whichever package owns them (`wardhook[anthropic]`, `[openai]`, `[chroma]`,
  `[judge]`, `[all]`). It ships no `wardhook/__init__.py`, which would shadow
  the PEP 420 namespace the four real packages share.

### Notes

- No package imports another. CI installs each of the four in isolation across
  Python 3.10-3.13 and runs its suite there, so standalone installability is a
  checked property rather than a claim. A separate job installs only the
  `wardhook` meta-package and asserts all four arrive at one version.
- The meta-package pins the four with `==` and is published after them, so
  `pip install wardhook` is never briefly unresolvable during a release.
- The full test suite runs offline against fake models; no API key is required.
- `AuditLogger` defaults to `record_allows=False`. This is a deliberate
  departure from a literal reading of "log every guardrail action" — an allow is
  the overwhelmingly common outcome, and logging every one buries what a
  reviewer opened the file to find. The behaviour is one constructor argument
  either way, and the reasoning is on the README and in the design doc.
- The build backend is pinned to `hatchling>=1.27,!=1.30.*,<1.32`: hatchling
  1.30.0 and 1.32.0 emit `Metadata-Version: 2.5`, which `packaging` rejects, so
  wheels built with them cannot be published.

[Unreleased]: https://github.com/justicebajaj161/wardhook/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/justicebajaj161/wardhook/releases/tag/v0.1.0
