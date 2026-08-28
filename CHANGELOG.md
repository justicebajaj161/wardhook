# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

All four packages are versioned in lockstep while the project is pre-1.0.

## [Unreleased]

Nothing yet.

## [0.2.0] — 2026-08-28

The local dashboard, and the first measured number for PII detection. Nothing
published changed shape: every entry below is additive, and `GET /info` gained
keys rather than losing any.

### Added

- **A read-only dashboard API in `wardhook-core`.**
  `wardhook.core.serve.create_dashboard(agent, telemetry=None)` builds a FastAPI
  application exposing `GET /api/topology`, `GET /api/runs` and
  `GET /api/runs/{run_id}`. The topology is read from the agent's own compiled
  graph, so it is accurate to the configuration rather than to a template: an
  agent with no retriever genuinely reports no `retrieve` node.
- **`wardhook.core.serve.read_topology(agent)`** and the `Topology`,
  `TopologyNode` and `TopologyEdge` records it returns. Anything without a
  graph — a plain callable with `.invoke()` is a supported target — reports
  `available=False` with a reason rather than raising.
- **`wardhook serve --dashboard`**, plus `create_app(dashboard=True)` and
  `WARDHOOK_DASHBOARD=1`. Off unless asked for; serving it on a non-loopback
  interface requires a second, separate `--dashboard-allow-remote`, and the
  refusal names the flag. `--dashboard-path` moves the mount point.
- **`wardhook.core.serve.describe_agent(agent)`**, which is the old private
  `_describe` from `serve/app.py`, moved beside `read_topology` and made public.
  Both answer "what is this agent, as plain data", and the application and the
  dashboard both need it — keeping it in either one would have made them import
  each other.
- **The dashboard describes the wiring, not just the graph.**
  `wardhook.core.serve.describe_agent()` — and therefore `GET /info`,
  `wardhook info` and the dashboard — now reports the model, the retriever,
  vector store and embeddings classes, the top-k and score threshold, **the
  number of chunks indexed**, the tool-iteration limit, and the guardrail error
  policy. All of it constructor arguments or an index size; none of it anything
  the agent processed. The chunk count is the one that matters in practice: an
  agent whose store is empty looks identical to a working one until it says
  `0 chunks indexed`.
- **A visual pass over the dashboard**, with the Wardhook mark drawn as inline
  SVG in the header — about 700 bytes, crisp at any size, and stroked with brand
  tokens that are redefined for dark mode, so one mark works on both themes. A
  base64 PNG would have been ~40x larger and would have tripped the page's own
  no-external-resource assertion.
- **The trace overlay.** Picking a run shades each node by how long it took and
  writes its tokens and cost underneath, so the architecture diagram and the
  bill are one picture. A node visited more than once — `call_model` on every
  tool round trip — is totalled rather than overwritten, so what a box says
  reconciles with the run's own totals. Steps with no box, such as the tracer's
  synthetic `(ungrouped)`, are listed rather than dropped.
- **`run_id` is shown prominently and is copyable,** because it is the join key
  back to your own audit log. That is the reason the dashboard does not need,
  and does not have, a second copy of the content.
- **The topology view.** The page draws the agent's graph as inline SVG,
  rendered server-side in pure Python: ranks by longest path, conditional edges
  dashed and labelled, and edges that loop back or skip a rank routed through
  the empty gutters rather than across a node box. No renderer is vendored and
  none is fetched, so the page stays a few kilobytes and keeps working offline.
  `draw_mermaid()` source is still published on `/api/topology` for anyone who
  would rather render it somewhere else.
- **A self-contained dashboard page** at the dashboard's mount root. One HTML
  document with its CSS inlined: no CDN, no web font, no image request, so it
  behaves identically in an air-gapped network. It names the agent, lists its
  tools and guardrails, and states in a banner which telemetry mode it is in.
  Every interpolated value is escaped — tool and guardrail names come from user
  code, and this page really is served over HTTP.

- **A measured PII benchmark**, in [`benchmarks/pii/`](benchmarks/pii), with
  `make bench-pii` to reproduce it. 84.3% recall and 94.2% precision over 2,076
  labelled spans in 760 documents across all four packs. The corpus is generated
  from real-world identifier formats rather than from the project's own regexes,
  and is committed so a reader can inspect exactly what was measured. Until now
  the answer to "what is the false-negative rate" was that nobody had measured
  it.

### Notes

- **The dashboard serves telemetry and configuration only, never content.** No
  prompt, completion, retrieved chunk or guardrail-event body reaches it. The
  step projection is an explicit allowlist, so a content field added to the
  telemetry model upstream cannot start being served by accident; there is a
  test pinning that. `run_id` is carried on every run and every step, and it is
  the intended join key back to your own audit log, where your redaction and
  retention policy already apply.
- **The benchmark number is published whatever it says, and nothing was tuned
  first.** It names specific weaknesses — `DATE_OF_BIRTH` at 37% because the
  pattern matches `DD/MM/YYYY` and not `1982-03-14`, `US_SSN` at 53% because it
  requires its dashes, `SWIFT_BIC` over-redacting capitalised words near "wire".
  Those are now in `docs/packages/guardrails.md` under Limitations with their
  figures. Fixing them is separate work, deliberately: the next result can be
  compared against a figure nobody had an incentive to flatter.
- **`make bench-pii` is not part of `make check`.** A benchmark is evidence to
  publish, not a gate to pass; wiring it into CI would create pressure to make
  the number go up, which is how a measurement stops being trustworthy.
- **The telemetry sink is read structurally, and both shapes are supported.**
  `Tracer` lists with `traces()`; `JSONLTraceStore` lists with `read()`. The API
  reports which one it found as a `mode`, because an in-memory tracer under
  multiple workers only sees its own process's traffic — a limitation that is
  disclosed rather than hidden.

## [0.1.1] — 2026-08-27

A pre-launch audit of every checkable claim in the README, plus the fixes it
turned up. No public API changed shape; two behaviours that silently produced
empty output now produce the right output.

### Fixed

- **`AuditLogger.record()` accepts a mapping.** `wardhook-core` returns its
  `guardrail_events` as dicts, but `record()` read its argument with `getattr`,
  which finds nothing on a dict. Every event degraded to an `allow` and was
  then discarded by the default `record_allows=False` — an empty audit trail,
  with no error, from the most obvious way to write one. Mappings are now read
  by key. `record_run()` remains the right call for a whole list.
- **`AuditLogger.report()` counts entities without a diff.** `by_entity` was
  read only from a recorded diff, which exists only when the caller passes
  `before=`. The README's own quickstart does not, so it printed
  `'by_entity': {}` on a run that had just redacted two entities. The count now
  falls back to the guardrail's own details, which already carried it.
- **`wardhook-core[google]` exists.** `resolve_model()` maps a `gemini-` prefix
  to `langchain-google-genai` and its error told you to install that extra, but
  no such extra was declared — so `pip install "wardhook-core[google]"` warned,
  installed nothing, and left you facing the identical error. Added to
  `wardhook-core`, to its `all`, and as a pass-through on the meta-package.
- **`.env.example` described four variables that do not exist.**
  `WARDHOOK_ENTITY_PACK`, `WARDHOOK_AUDIT_LOG`, `WARDHOOK_TRACE_DIR` and
  `WARDHOOK_PRICING_FILE` appeared nowhere in any source file, and
  `WARDHOOK_TARGET` — which the `Dockerfile` does read — was missing. The file
  now lists exactly what is read, and says why the others are constructor
  arguments rather than ambient environment state.

### Added

- **Coverage is 100% and gated.** `fail_under = 100` (branch coverage) fails
  the build on a drop, so the README's status table cannot drift from reality.
  The gaps this closed were real, not cosmetic: `AgentGraph.ainvoke`, the whole
  PDF loading path, `wardhook serve` and `wardhook info`, multi-block message
  content, provider-client construction failures, and the tracer's cross-thread
  paths all had no test at all.
- **`make cov-table`** regenerates the README's status table from a real run,
  so a reviewer can reproduce the numbers instead of trusting them.
- **Cross-package composition tests** in `tests/`, which the directory had
  promised in its docstring while containing none. They assert the claim the
  README leads with — guardrails, telemetry, and evals driving a core agent
  with no import between them — which no single package's suite can reach.
- `.github/ISSUE_TEMPLATE/config.yml`, disabling blank issues and pointing
  security reports at a private advisory.

### Changed

- `make check` now runs the coverage-gated suite, so it stays the same gate CI
  runs, as the README says it is.
- The README marks which quickstart block runs as-is and which sketches an
  integration; the second needs a `policy.md` and tools that are yours.

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
- **Release tooling** — `.github/workflows/release.yml` publishes all five
  packages to PyPI on a version tag using Trusted Publishing, so no API token is
  stored in this repository. It re-runs the full gate before building and
  refuses to publish if the tag and the declared versions disagree.
  `scripts/bump-version.sh` sets every version declaration at once -- each
  package's `pyproject.toml` and `__version__`, plus the meta-package's own
  version and each of its exact pins, including the ones inside extras.
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

[Unreleased]: https://github.com/justicebajaj161/wardhook/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/justicebajaj161/wardhook/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/justicebajaj161/wardhook/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/justicebajaj161/wardhook/releases/tag/v0.1.0
