# Wardhook for VS Code

Find personal data and prompt-injection risk in your code, without the text
ever leaving your machine.

This extension is a front end for
[`wardhook-guardrails`](https://pypi.org/project/wardhook-guardrails/), which
depends on PyYAML and nothing else. There is no account, no backend, and no
telemetry.

## What it does

- **Live diagnostics.** Squiggles on SSNs, card numbers, API keys and the rest
  as you type, with a severity that reflects how confident the detector is.
- **Quick fixes.** Replace one finding, or every finding in the file, with the
  library's own typed placeholder — `[US_SSN]`, `[CREDIT_CARD]`.
- **A findings panel** grouped by file, and a status-bar count.
- **`@wardhook` in Copilot Chat,** plus seven language-model tools that Copilot
  can call on its own to scan, redact, score injection risk, and explain why
  something was flagged.
- **Four built-in entity packs** — general, insurance, healthcare, fintech —
  or point it at your own YAML.

## The detector never hands over the secret

`PIIMatch` stores offsets, not the text it matched. A scan result says *"a
validated credit-card number occupies characters 38 to 57 of line 3"* and
nothing more, so the extension can draw a squiggle over a secret it was never
given. The test suite asserts this: a scan of a card number produces a
response whose JSON does not contain that number.

## Precision, and why the defaults are what they are

Pointing a PII detector at source code naively is unusable. Scanning the
Wardhook repository itself produces **89 matches across 66 files** — and 79 of
them are docstring examples and default config values: `127.0.0.1`,
`alice@example.com`, sample SSNs in doctests.

Only 10 matches are confirmed by a checksum, and all 10 are `critical`.

So the default is *report a match if a checksum validated it, **or** if it is
at least `high` severity* — and skip test directories, where sample data lives
on purpose.

| Filter | Diagnostics on this repo |
| --- | --- |
| Everything | 89 |
| `severity >= high` | 42 |
| **Default: validated or `>= high`, skipping tests** | **8** |
| `validatedOnly` | 10 |

Roughly 91% of the noise goes away and no real leak does. Turn
`wardhook.minSeverity` down to `medium` when you want everything.

## Performance

The analyzer is one long-lived Python process, not a subprocess per scan.

| | |
| --- | --- |
| Import cost, paid once at activation | ~160 ms |
| Scan round-trip, 38 KB file | p50 12 ms, p90 14 ms |
| Bulk throughput | ~3 MB/s |

Scans are debounced (150 ms by default) and a scan still in flight is
superseded when you keep typing.

## Requirements

- VS Code 1.95 or newer.
- **Python 3.10 or newer.** That is the only thing you have to provide.
- Copilot Chat is optional. Without it everything except the chat participant
  and the language-model tools still works.

You do **not** need to install anything by hand. On first run the extension
looks for an interpreter that can actually load the library — checking
virtualenvs beside your code, the Python extension's selection, and PATH — and
if none can, it offers to `pip install wardhook-guardrails` into a suitable
interpreter for you. One package, one dependency (PyYAML), a few seconds.

Two details that make that reliable:

- **Candidates are probed, not assumed.** `python3` on macOS is frequently the
  system 3.9, which cannot even import the library. Each candidate is asked to
  load an entity pack for real, and the first that manages it wins.
- **The probe loads a pack rather than just importing.** PyYAML is pulled in
  lazily when a pack is read, so an interpreter missing it imports cleanly and
  then fails on the first scan. The analyzer likewise builds the default pack
  before reporting itself ready, so "ready" means usable.

Inside a checkout of the Wardhook monorepo the package is found on `PYTHONPATH`
automatically, with no install at all.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `wardhook.pack` | `default` | Built-in entity pack. |
| `wardhook.customPackPath` | — | Your own pack YAML; overrides the above. |
| `wardhook.minSeverity` | `high` | Lowest severity reported. |
| `wardhook.validatedOnly` | `false` | Only checksum-confirmed matches. |
| `wardhook.excludeEntities` | `[]` | Entity types never reported. |
| `wardhook.excludeGlobs` | tests, `node_modules`, `.venv` | Paths never scanned. |
| `wardhook.scanOnType` | `true` | Re-scan while typing, not only on save. |
| `wardhook.debounceMs` | `150` | Idle time before re-scanning. |
| `wardhook.injectionThreshold` | `0.5` | Injection score treated as blocking. |
| `wardhook.pythonPath` | — | Pin an interpreter. |
| `wardhook.policyPackRepo` | — | `owner/repo` holding a shared pack. |

## GitHub sign-in

**Wardhook: Sign in to GitHub** uses the editor's own authentication provider,
so the OAuth flow and the token belong to VS Code — this extension never sees
a client secret and never stores a token. It is used for one thing: fetching a
shared entity pack from a private repo so a team scans against the same rules.

That traffic is inbound only. Nothing you scan is ever transmitted.

## Try it

Press <kbd>F5</kbd> to open the Extension Development Host on `demo/`, which
contains a file of planted fake secrets. See `demo/README.md` for the exact
findings to expect.

## Development

```
make ext-install    # npm dependencies
make ext-build      # tsc --noEmit, eslint, esbuild
make ext-test       # 27 tests in a real VS Code
make ext-package    # .vsix
```

The Python analyzer in `python/wardhook_sidecar.py` is linted by the same ruff
configuration as the four published packages, from the repository root.

## License

MIT.
