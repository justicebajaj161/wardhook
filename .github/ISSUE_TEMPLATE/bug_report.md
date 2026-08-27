---
name: Bug report
about: Something behaves differently from what the documentation says
title: ''
labels: bug
assignees: ''
---

## What happened

<!-- What you observed. -->

## What you expected

<!-- What the docs or docstrings led you to expect. -->

## Reproduction

```python
# The smallest snippet that shows the problem.
# Please use synthetic data -- never paste real PII, credentials, or customer records.
```

## Environment

- Wardhook packages and versions: <!-- e.g. wardhook-guardrails 0.1.0 -->
- Python version:
- OS:
- Installed alone, or alongside other Wardhook packages?

## A note on missed detections

If a guardrail failed to catch something — a PII format it missed, an injection
that scored below the threshold — that is a **coverage gap**, and this is the
right place for it. Please include the example so it can be added to the test
corpus.

If instead the tool reported that it acted when it did not, or wrote data it
promised to protect, that is a security issue: please use the private channel in
[SECURITY.md](../../SECURITY.md) rather than this tracker.
