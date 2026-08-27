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

Nothing is published until this is done. Until then,
[`.github/workflows/release.yml`](../.github/workflows/release.yml) is inert.

For each of the five projects — `wardhook-core`, `wardhook-guardrails`,
`wardhook-observability`, `wardhook-evals`, and `wardhook` — add a **pending
publisher** at <https://pypi.org/manage/account/publishing/>:

| Field | Value |
| --- | --- |
| PyPI project name | `wardhook-core` (and each of the other four) |
| Owner | `justicebajaj161` |
| Repository name | `wardhook` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Then create a `pypi` environment in the repository settings
(*Settings → Environments*). Adding a required reviewer there is worth doing:
it means a tag push cannot publish without a human approving the run.

Repeat with the environment named `testpypi` on <https://test.pypi.org> if you
want to rehearse. The workflow's manual trigger can target either.

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
| `publish` | One job per real package, each with its own OIDC identity. `fail-fast` is off, so a failure on one does not cancel a sibling mid-upload. |
| `publish-meta` | `wardhook` last, after the four it pins. Publishing it first would leave a window in which `pip install wardhook` cannot resolve. |

## If a publish fails partway

PyPI does not allow overwriting a released version, and the four projects are
independent, so a partial release leaves some at the new version and some
behind.

Re-running the workflow is safe: `pypa/gh-action-pypi-publish` skips a file that
already exists rather than failing, so only the missing projects upload. If a
released version is actually broken, **yank it rather than deleting it** — a
deleted version breaks anyone who already pinned it, while a yanked one stays
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
