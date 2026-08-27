## What does this change?

<!-- A sentence or two on the behaviour that changes. -->

## Why?

<!-- The problem this solves. If it fixes an issue, link it: Fixes #123 -->

## Which packages does it touch?

- [ ] `wardhook-core`
- [ ] `wardhook-guardrails`
- [ ] `wardhook-observability`
- [ ] `wardhook-evals`
- [ ] Root docs, examples, or CI

## Checklist

- [ ] `make check` passes (lint, types, tests)
- [ ] New behaviour has tests; a fixed bug has a regression test
- [ ] Public functions and classes have Google-style docstrings
- [ ] **No new cross-package import.** A package must still work with none of
      the others installed — see `tests/test_isolation.py`
- [ ] No credentials, real PII, or customer data anywhere in the diff
- [ ] `CHANGELOG.md` updated under `[Unreleased]` if user-facing

## Anything reviewers should look at closely?

<!-- Trade-offs you made, alternatives you rejected, parts you are unsure about. -->
