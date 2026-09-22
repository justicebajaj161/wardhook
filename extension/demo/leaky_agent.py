"""Sample agent prompt with deliberately planted secrets.

This file is the Wardhook extension's demo and test fixture. Every value below
is fake -- a reserved test credit-card number, a documentation-only AWS key id,
and invented contact details -- but each one is shaped so the detector treats
it exactly as it would treat the real thing.

Expect three findings at the default settings. The email and phone number are
detected too, but suppressed as medium-severity noise.
"""

CLAIM_PROMPT = """Customer Dana Reyes filed claim CLM-8891.
SSN on file: 796-30-8562. Card ending 4111 1111 1111 1111.
Contact: dana.reyes@acme-insure.com or 415-555-0142.
Summarise the claim for the adjuster."""

AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"


def build_prompt(notes: str) -> str:
    """Return the claim prompt with adjuster notes appended."""
    return f"{CLAIM_PROMPT}\n\nAdjuster notes: {notes}"
