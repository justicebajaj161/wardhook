# Contributing to Wardhook

Thanks for considering a contribution. This project is pre-1.0 and moving, so
issues, questions, and pull requests are all welcome.

## Development setup

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/justicebajaj161/wardhook.git
cd wardhook
make install          # uv sync -- all four packages, editable, plus dev tools
make check            # lint + type-check + test: the same gate CI runs
```

**No API key is required.** The entire suite runs offline against fake models,
and no test touches the network. If you find yourself needing a live provider to
make a test pass, that is a design problem with the test.

## Running things

| Command | What it does |
| --- | --- |
| `make check` | Lint, type-check, and test with the coverage gate — run this before opening a PR |
| `make test` | Every package's suite, including doctests |
| `make lint` | `ruff check` and `ruff format --check` |
| `make fmt` | Autofix lint findings and format |
| `make types` | `mypy`, each package independently |
| `make solo` | Install each package alone and run its suite there |
| `make cov-table` | Regenerate the README's status table from a real run |
| `make build` | Build and validate all five distributions |

To run one package: `uv run pytest packages/wardhook-guardrails -q`.

## The rule that matters most

**Each package must work with none of the others installed.**

This is the project's central constraint, and it is the easiest thing to break
with one convenient import. `wardhook-core` never imports its siblings at module
scope; the other three never import core at all. They meet through
`typing.Protocol` contracts in `wardhook/core/protocols.py`.

Two things enforce it:

- Each package has a `packages/<package>/tests/test_isolation.py` that parses
  every module's AST and fails on a forbidden import.
- CI installs each package into an environment with no siblings present and runs
  its suite there.

If you need something from another package, the answer is almost always to widen
a protocol rather than to add a dependency. If you think a dependency is genuinely
required, open an issue first — it is a significant architectural change.

## Coding style

- **Formatting and linting: `ruff`**, configured in the root `pyproject.toml`.
  There is no second formatter. Run `make fmt`.
- **Type hints on everything public.** `mypy` runs with
  `disallow_untyped_defs`, per package, in CI.
- **Google-style docstrings** on every public function, class, and module.
  Include an `Args:`, `Returns:`, and `Raises:` section where each applies.
- **Docstring examples are executed.** `pytest` runs with `--doctest-modules`,
  so an example that drifts out of date fails the build. This is deliberate.
- **Comment the *why*, not the *what*.** A comment explaining that a lookahead
  exists because the pattern otherwise reads a card number as a phone number is
  worth keeping. A comment restating the line below it is not.
- **No hardcoded secrets, ever.** Credentials come from the environment. See
  `.env.example`.

## Tests

- New behaviour needs a test. New *fixed bugs* especially need one.
- **Coverage is gated at 100%** (branch coverage), so an untested line fails the
  build. If a line genuinely cannot be reached through the public API, mark it
  `# pragma: no cover` with a comment saying why — do not delete a defensive
  guard to make the number go up.
- Prefer a small corpus over a single example for anything heuristic. The
  injection detector's thresholds were tuned against the corpus in
  `test_injection_and_rbac.py`, and that corpus is asserted as a whole so a
  future pattern change cannot quietly trade recall for precision.
- Test names should read as statements about behaviour:
  `test_a_denied_tool_never_executes`, not `test_rbac_2`.
- Guardrail tests should assert that audit records **do not** contain the data
  they are auditing. That property is the point of the module.

## Submitting a pull request

1. Fork and branch from `main`. Use a descriptive branch name.
2. Make your change, with tests.
3. Run `make check`. It must pass.
4. Write a commit message explaining *why*, not just what changed.
5. Open the PR and fill in the template.

Small, focused PRs get reviewed faster than large ones. If you are planning
something substantial, open an issue first so we can agree on the approach
before you spend the time.

## Adding an entity pack

Domain packs live in
`packages/wardhook-guardrails/src/wardhook/guardrails/packs/`. A new one should:

- `extends: default` rather than restating universal entities.
- Carry a `validator` (checksum) or `context_words` for any pattern whose shape
  alone would produce false positives.
- Come with tests covering both detections **and** near-misses that must not fire.

## Reporting bugs and security issues

Bugs go in the [issue tracker](https://github.com/justicebajaj161/wardhook/issues).

**Security vulnerabilities do not.** Use the private channel described in
[SECURITY.md](SECURITY.md). Note the distinction that file draws between a
*coverage gap* (a missed detection — please open a normal issue with the
example) and a *vulnerability* (the tool reports it acted when it did not, or
writes data it promised to protect).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you are expected to uphold it.
