# wardhook-guardrails — design decisions

This page explains *why* the guardrails are built the way they are, and is
candid about what pattern matching can and cannot do. For usage, see the
[package README](../../packages/wardhook-guardrails/README.md).

## The problem

Three questions get asked of any agent that touches regulated data: what
personal information passed through it, could anyone talk it out of its
instructions, and who was allowed to make it *do* things. Answering them
usually means either buying a platform or writing the same regex file for the
fourth time.

## Decision: one runtime dependency

`wardhook-guardrails` depends on PyYAML and nothing else. Not LangChain, not
LangGraph, not spaCy, not a transformer.

This is the decision most of the others follow from. A redaction library that
drags in an ML stack cannot be dropped into a Lambda, a Django view, or a batch
job — and those are exactly the places PII shows up. The cost is real: an ML
NER model would catch names and addresses that regex cannot. That trade is
discussed under *Limitations* below rather than hidden.

A test in the suite parses every module's AST and fails if anything imports a
sibling package, LangChain, or a heavy numerical dependency. It is a static
scan rather than an import check because the development environment has all
four packages installed — an accidental import would succeed there and only
break for the person who installed this package alone.

## Decision: entity packs, not a global entity list

"PII" is not one list. Shipping a single global one guarantees every deployment
either over-redacts or misses what matters to it.

An `EntityPack` is a named rule set, loadable from YAML. The three domain packs
**extend** `default` rather than replacing it, so an email address is caught
whichever pack you pick and a domain pack only has to state what is actually
domain-specific.

The alternative — one pack with per-deployment enable flags — was rejected
because it makes the shared file grow without bound and turns every domain
addition into a change to a file everyone depends on.

## Decision: checksums and context words over more regex

Two mechanisms do most of the work in keeping false positives usable:

**Checksums.** A regex matches anything card-shaped. Luhn is what separates a
real card number from sixteen arbitrary digits, and financial and clinical text
is dense with digit runs. IBANs use mod-97, NHS numbers modulus-11.

**Context words.** Some entities are too generic to stand alone — a bare
six-digit medical record number is just a number. Those rules require a related
term within a configurable window. That trades recall for a large drop in false
positives, which is the right trade when the alternative is redacting every
six-digit number in a document.

One refinement worth recording: `0000000000` and `1111 1111 1111 1111` both
*pass* their checksums, but in real documents they are placeholders, form
templates, or test fixtures. Runs of a single repeated digit are rejected.

## Decision: audit records never contain the audited data

This is the rule the whole audit module is organised around.

A log that stores the PII a redactor just removed has recreated the exposure it
was built to prevent — in a file with longer retention and usually weaker
access control than the system it was protecting. Records therefore describe
**what changed**: entity type, character offsets, lengths, counts. Never the
values.

The test suite checks this from several angles, including asserting that a real
SSN and email never appear in a written log file.

Where correlation genuinely matters — *did this same value appear in three
conversations?* — `AuditLogger.fingerprint()` produces a salted digest. The salt
is random per process by default, so fingerprints correlate within a session
and are meaningless afterwards. An *unsalted* hash of a low-entropy identifier
is not anonymisation: there are only a billion possible SSNs, and a rainbow
table of all of them is an afternoon's work. Supplying a stable salt is
possible, but it is an explicit decision to accept cross-session correlation.

**JSONL, not a database.** An audit trail is append-only by nature, survives a
crash mid-write with at most one damaged line, and can be tailed, grepped, and
loaded incrementally. Each write is `fsync`ed — a record still sitting in the
page cache when the process dies is a record that did not happen.

**Allows are not logged by default.** An allow is the overwhelmingly common
case, and logging every one buries the actions a reviewer is looking for.
`record_allows=True` is there for regimes that require positive evidence that
every request was screened.

## Decision: weighted signal categories for injection, not a classifier

Prompt injection has no clean signature, so this is a scorer, not a detector.
Patterns are grouped into six categories, each contributing a weight; the
weights sum, saturate at 1.0, and are compared against a threshold.

**Weight is per category, not per hit.** Saying the same thing five times is not
five times more suspicious than saying it once, whereas tripping several
genuinely different attack shapes is. The high-confidence categories are
individually calibrated to clear the default 0.5 threshold; weaker ones must
co-occur.

`opaque_payload` (a long base64 blob) is deliberately weak at 0.3 — legitimate
messages carry attachments — but combined with `encoding_instruction`
("decode the following") it clears the threshold comfortably. That split came
directly from a measured false positive on *"Attached is the signed PDF, base64
encoded below."*

Two other patterns were narrowed for the same reason. `act as a ...` matched
*"act as a guarantor on this loan"*, and `show me the rules` matched *"show me
the rules for filing a claim"* — both ordinary business English. The current
patterns require an AI-persona object and a possessive respectively.

The corpus in the test suite is the calibration record: 14 attacks, 12 benign
messages, currently 14/14 caught with 0 false positives. Those numbers are
asserted, so a future pattern change that trades one for the other fails
visibly.

**Input-only by default.** Injection arrives in user input. Scanning model
output for these phrases produces a false positive every time the assistant
legitimately explains what prompt injection *is*.

## Decision: RBAC denies by default, in three separate ways

1. **A tool no role grants is denied.** Adding a tool to an agent does not
   silently widen anyone's permissions — it is unreachable until a role names it.
2. **A caller with no recognised role gets nothing.** `default_roles` is empty.
3. **A run with no principal at all is denied.** Forgetting to pass a principal
   fails closed rather than granting unrestricted access.

Each can be opened up — `allow_unlisted`, `default_roles`, `allow_anonymous` —
but each has to be an explicit decision in the constructor.

Denials win over grants, so a role can hold a broad `*` with a carve-out. A
denial record includes the patterns the caller *does* hold, so a reviewer sees
not just that access was refused but what the caller was entitled to.

**Arguments are not inspected.** This policy gates on identity, not on values.
A supervisor permitted to `issue_refund` may issue a refund of any size.
Value-based limits are a different control and belong in the tool itself.

## Limitations

Stated plainly, because a security control that oversells itself is worse than
one that does not exist.

- **Regex misses things an ML model would catch.** Names, addresses in
  unfamiliar formats, and PII phrased unusually or split across a sentence will
  slip through. The dependency-weight trade is deliberate; if you need NER,
  wrap one in the `BaseGuardrail` interface and add it to the list.
- **Injection detection is heuristic.** A novel attack in plain prose scores
  zero. Do not treat the score as a boundary you can rely on alone — reducing
  what the agent can *reach* matters more than detecting what it is *told*.
- **Non-English coverage is weak.** The patterns are English-language.
- **Context windows are character-based**, not sentence-aware, so an entity at
  the very start or end of a long document may not see its context word.
- **RBAC trusts the principal.** If your application can be tricked into
  passing the wrong roles, nothing here will catch it. Authenticate upstream.

The right framing: these are real controls that raise the cost of a mistake and
produce the evidence a reviewer needs. They are not a guarantee, and they do
not replace minimising the agent's reach in the first place.
