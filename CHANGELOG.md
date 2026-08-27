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

### Notes

- Neither published package imports another. CI installs each in isolation and
  runs its suite there, so standalone installability is a checked property.
- The full test suite runs offline against fake models; no API key is required.

## [0.1.0] — unreleased

Initial release. Not yet published to PyPI.

[Unreleased]: https://github.com/justicebajaj161/wardhook/compare/main...HEAD
