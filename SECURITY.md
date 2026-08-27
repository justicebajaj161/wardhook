# Security Policy

Wardhook is a governance and observability toolkit. It sits directly in the path
of prompts, tool calls, and — by design — personally identifiable information.
Security issues here can have real downstream consequences, so we take them
seriously and ask that you report them privately.

## Supported Versions

Wardhook is pre-1.0. Security fixes land on the latest released minor version of
each package; there are no long-term support branches yet.

| Package                  | Supported |
| ------------------------ | --------- |
| `wardhook-core` 0.1.x        | Yes |
| `wardhook-guardrails` 0.1.x  | Yes |
| `wardhook-observability` 0.1.x | Yes |
| `wardhook-evals` 0.1.x       | Yes |
| Anything older           | No  |

## Reporting a Vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Report it through either channel below:

1. **GitHub Security Advisories (preferred)** — open a private report at
   <https://github.com/justicebajaj161/wardhook/security/advisories/new>.
   This keeps the discussion private until a fix ships.
2. **Email** — <adityabajaj161@gmail.com> with `[SECURITY]` in the subject line.

Please include, as far as you can determine it:

- Which package and version is affected
- A description of the issue and its impact
- Steps to reproduce, ideally a minimal proof of concept
- Any suggested remediation

**Please do not include real PII, production audit logs, or live credentials in
your report.** Synthetic examples are sufficient to demonstrate every class of
issue in this project, and a report is not a good place for regulated data.

## What to Expect

| Stage | Target |
| ----- | ------ |
| Acknowledgement of your report | Within 3 business days |
| Initial assessment and severity triage | Within 7 business days |
| Fix released, or a public status update if the work runs longer | Within 90 days |

We will keep you updated as the fix progresses, credit you in the advisory and
`CHANGELOG.md` unless you prefer to stay anonymous, and coordinate disclosure
timing with you. We ask that you give us a reasonable window to ship a fix
before publishing details.

## Scope

In scope — vulnerabilities in this repository's code, including:

- **Guardrail bypasses.** Input that defeats PII detection or redaction, or that
  slips past the prompt-injection scorer, when the guardrail is configured as
  documented.
- **Leakage through the audit log.** Any path by which raw secrets or
  unredacted PII are written to an audit record. Audit values are hashed by
  design; a case where they are not is a bug.
- **RBAC failures.** Any way to invoke a tool that the configured role policy
  should have denied.
- **Code execution or path traversal** via document ingestion, trace files, or
  eval case files.
- **Credential exposure**, including secrets leaking into traces, rendered trace
  HTML, or error messages.
- **Denial of service** in the served FastAPI endpoint or in the regex-based
  detectors (for example, catastrophic backtracking).

Out of scope:

- Vulnerabilities in upstream dependencies — report those to their maintainers,
  though we do want to know if Wardhook's usage makes an upstream issue
  materially worse.
- Behavior of the underlying LLM provider, including model outputs themselves.
- Missing hardening in the example scripts under `examples/`, which are
  illustrative and explicitly not production configurations.
- Results from automated scanners with no demonstrated exploit path.

## A Note on Detection Coverage

Wardhook's guardrails are pattern- and heuristic-based by design (see
`docs/packages/guardrails.md`). They will not catch every instance of PII or
every prompt injection, and that limitation is documented rather than hidden.
A missed detection is a **coverage gap** — please open a normal issue or pull
request with the example so we can extend the entity packs and test corpus.

A **security vulnerability** is different: it is a case where the tool reports
that it acted when it did not, writes data it promised to protect, or can be
manipulated into failing open. Those go through the private channel above.
