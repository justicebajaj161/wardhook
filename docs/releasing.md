# Releasing

All five distributions are versioned in lockstep while the project is pre-1.0,
and are published together from one tag: the four real packages, plus the
`wardhook` meta-package that installs them.

Releases go out through [PyPI Trusted Publishing][tp], so there is no API token
stored in this repository — nothing to leak, nothing to rotate. GitHub mints a
short-lived OIDC token for the workflow run and PyPI verifies it came from this
repository, this workflow file, and this environment.

[tp]: https://docs.pypi.org/trusted-publishers/

## One-time setup

Done once, before the first release. It is recorded here because the
constraints below are not obvious and are easy to hit again if a package is ever
added.

Add a **pending publisher** for each of the five projects at
<https://pypi.org/manage/account/publishing/>. PyPI caps you at three pending
publishers at a time, so the first release goes out in two passes — see
[The first release takes two passes](#the-first-release-takes-two-passes). Owner `justicebajaj161`,
repository `wardhook`, and workflow `release.yml` every time — but **a different
environment name for each**:

| PyPI project name | Environment name |
| --- | --- |
| `wardhook-core` | `pypi-core` |
| `wardhook-guardrails` | `pypi-guardrails` |
| `wardhook-observability` | `pypi-observability` |
| `wardhook-evals` | `pypi-evals` |
| `wardhook` | `pypi-meta` |

### Why the environments must differ

This is the part that catches people out, and it is worth understanding rather
than copying.

PyPI's uniqueness constraint on a *pending* publisher covers
**(owner, repository, workflow, environment)** and deliberately **excludes the
project name**. A pending publisher is allowed to create a project that does not
exist yet, so PyPI has to be able to decide which name to create — and two
pending publishers with identical external identity would be ambiguous.

All five of these packages publish from one workflow file in one repository, so
the environment is the only field left to tell them apart. Register a second one
without a distinct environment and PyPI rejects it:

> A pending trusted publisher matching this configuration has already been
> registered for a different project name.

Note also that the `pypi` shown greyed out in the Environment name box is
**placeholder text, not a value**. Leaving the field alone saves the publisher
with environment "(Any)", which collides with everything.

The constraint applies only while a project is *pending*. Once a project exists,
its publisher is attached to the project and the ambiguity is gone — but keeping
one environment per package is worth doing anyway, because it lets you require a
different reviewer for, say, `wardhook-core` than for the meta-package.

### The GitHub side

Create the five environments under *Settings → Environments*: `pypi-core`,
`pypi-guardrails`, `pypi-observability`, `pypi-evals`, `pypi-meta`. GitHub will
create them implicitly on first use, but making them yourself lets you add a
required reviewer — worth it, since a tag is easy to push by accident and a PyPI
release can only be yanked, never un-published.

To rehearse on <https://test.pypi.org>, register the same five there with
`testpypi-` prefixes instead (`testpypi-core`, …, `testpypi-meta`) and create
matching GitHub environments. The workflow's manual trigger targets either.

## Cutting a release

1. **Bump the version everywhere.** Eleven declarations: five
   `pyproject.toml` files, the `__version__` in each of the four packages'
   `__init__.py`, and the meta-package's exact pins on all four.

   ```bash
   ./scripts/bump-version.sh 0.2.0
   ```

   That sets all eight at once, because missing one is easy:

   ```bash
   grep -h '^version = ' packages/*/pyproject.toml | sort -u
   grep -h '^__version__' packages/*/src/wardhook/*/__init__.py | sort -u
   ```

   The release workflow refuses to build if the tag and the declared versions
   disagree, so a half-finished bump fails fast rather than shipping.

2. **Move `[Unreleased]` into a dated section** in
   [`CHANGELOG.md`](../CHANGELOG.md), following Keep a Changelog, and add the
   link reference at the bottom of the file.

3. **Check the gate locally** — the workflow re-runs all of this, but finding
   out here is faster:

   ```bash
   make check && make solo && make build
   ```

4. **Tag and push.**

   ```bash
   git tag -a v0.2.0 -m "Release 0.2.0"
   git push origin v0.2.0
   ```

5. **Approve the run** if you configured a required reviewer, and watch
   [Actions](https://github.com/justicebajaj161/wardhook/actions).

## What the workflow does

| Job | What it checks |
| --- | --- |
| `verify` | ruff, ruff format, mypy per package, the full test suite. A tag is not evidence the tree was green when it was cut. |
| `build` | Confirms every package declares the tagged version *and* that the meta-package's pins match it, builds each into its own directory, and runs `twine check`. |
| `publish` | One job per real package, each in its own environment and so with its own OIDC identity. `fail-fast` is off, so a failure on one does not cancel a sibling mid-upload. |
| `publish-meta` | `wardhook` last, after the four it pins. Publishing it first would leave a window in which `pip install wardhook` cannot resolve. |

## The first release takes two passes

PyPI allows **at most three *pending* trusted publishers per user**
(`warehouse/accounts/views.py`), and there are five projects here. That is a
spam control on unused registrations, not a queue — nobody at PyPI or GitHub
reviews or approves anything.

A publisher is "pending" only in the sense that its project does not exist yet.
The first successful upload creates the project and *reifies* the pending
publisher into an ordinary project-scoped one, which frees the slot. So:

1. Register three: `wardhook-core`, `wardhook-guardrails`,
   `wardhook-observability`. Leave `wardhook-evals` and `wardhook` for now —
   the meta-package publishes last anyway, so registering it early wastes a slot.
2. Tag and push. Three projects publish; `wardhook-evals` fails with no
   publisher configured, and `publish-meta` is skipped because `publish` did not
   fully succeed. That is the intended behaviour, not a broken release.
3. Register the remaining two. All three slots are free now.
4. **Re-run all jobs** on the same workflow run. The three already-published
   projects skip, `wardhook-evals` uploads, and `publish-meta` follows.

From the second release onwards this never happens again: all five projects
exist, their publishers are ordinary, and the pending limit is irrelevant.

## If a publish fails partway

PyPI does not allow overwriting a released version, and the five projects are
independent, so a partial release leaves some at the new version and some
behind.

Re-running is safe because both publish steps set `skip-existing: true`. That is
**not** the action's default — by default it fails on a file already present on
the index, which would make every re-run die on whatever had already succeeded.

If a released version is actually broken, **yank it rather than deleting it** —
a deleted version breaks anyone who already pinned it, while a yanked one stays
installable for existing pins and invisible to new resolutions. Then release a
patch version.

## Why `wardhook` publishes last

The meta-package pins each dependency with `==`, so it is only installable once
all four are on the index. `publish-meta` therefore waits on `publish`. If the
four succeed and the meta fails, re-running fills in just the meta — the reverse
order would have published something briefly unresolvable.

## A note on the build backend

`requires` in each `pyproject.toml` pins hatchling to `>=1.27,!=1.30.*,<1.32`.
That is not caution for its own sake: hatchling 1.30.0 and 1.32.0 emit
`Metadata-Version: 2.5`, which `packaging` rejects as invalid, which means
`twine check` fails and `twine upload` would too. Raise the upper bound when the
packaging toolchain accepts 2.5, and check `make build` still passes before you
do.
