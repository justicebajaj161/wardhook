#!/usr/bin/env python3
"""Generate the labelled corpus the PII benchmark is measured against.

**The values here are built from real-world formats, never from Wardhook's own
regexes.** That distinction is the whole point. A corpus derived from the
patterns under test would score close to 1.0 and measure nothing; every
generator below encodes how an identifier is actually written -- a US SSN with
and without its dashes, a date of birth in British, ISO and long-hand forms, a
Medicare beneficiary identifier both grouped and ungrouped -- because that is
what arrives in a real support ticket.

Every identifier is fabricated. Checksums are computed so the values are
*format*-valid (a Luhn-valid card, a mod-97-valid IBAN, a modulus-11-valid NHS
number), which is necessary to exercise the validators at all, but no value
belongs to anybody. Names, streets and domains are drawn from documentation
placeholders and reserved ranges.

The generator is seeded, so `corpus.jsonl` is reproducible byte for byte:

    python benchmarks/pii/generate_corpus.py

Run it after changing anything here, and commit the result alongside.
"""

from __future__ import annotations

import json
import random
import string
from collections.abc import Callable
from pathlib import Path

SEED = 20260828
DOCUMENTS_PER_DOMAIN = 150
NEGATIVES_PER_DOMAIN = 40
OUTPUT = Path(__file__).parent / "corpus.jsonl"

# --------------------------------------------------------------------------
# Checksum helpers, implemented from the public specifications.
# --------------------------------------------------------------------------


def luhn_check_digit(digits: str) -> str:
    """Return the Luhn check digit for a partial card number (ISO/IEC 7812)."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return str((10 - total % 10) % 10)


def iban_check_digits(country: str, bban: str) -> str:
    """Return the two mod-97 check digits for an IBAN (ISO 13616)."""
    rearranged = bban + country + "00"
    numeric = "".join(
        str(ord(char) - 55) if char.isalpha() else char for char in rearranged.upper()
    )
    return f"{98 - int(numeric) % 97:02d}"


def nhs_check_digit(digits: str) -> str | None:
    """Return the modulus-11 check digit for nine NHS digits, or None if invalid."""
    total = sum(int(char) * (10 - index) for index, char in enumerate(digits))
    remainder = 11 - total % 11
    if remainder == 11:
        return "0"
    if remainder == 10:
        return None
    return str(remainder)


# --------------------------------------------------------------------------
# Value generators. One entity may have several, because one identifier is
# written several ways in the wild and a detector meets all of them.
# --------------------------------------------------------------------------

FIRST = ["Ada", "Grace", "Alan", "Katherine", "Linus", "Barbara", "Dennis", "Radia"]
LAST = ["Lovelace", "Hopper", "Turing", "Johnson", "Torvalds", "Liskov", "Ritchie", "Perlman"]
DOMAINS = ["example.com", "example.org", "mail.example.co.uk", "corp.example.net"]
STREETS = ["Baker", "Elm", "Maple", "Pennsylvania", "Abbey", "Oxford", "Cedar", "Union"]
SUFFIX = ["Street", "St", "Avenue", "Ave", "Road", "Rd", "Lane", "Drive", "Boulevard"]
MBI_LETTERS = "ACDEFGHJKMNPQRTUVWXY"


def _digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(count))


def _upper(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(count))


def gen_email(rng: random.Random) -> str:
    first, last = rng.choice(FIRST).lower(), rng.choice(LAST).lower()
    style = rng.randrange(4)
    local = [
        f"{first}.{last}",
        f"{first[0]}{last}",
        f"{first}+claims",
        f"{first}_{last}{rng.randrange(10, 99)}",
    ][style]
    return f"{local}@{rng.choice(DOMAINS)}"


def gen_phone(rng: random.Random) -> str:
    # Reserved ranges: NANP 555-01xx, Ofcom 020 7946 0xxx, Indian test prefixes.
    return rng.choice(
        [
            f"+1 (415) 555-{rng.randrange(100, 200):04d}",
            f"415-555-{rng.randrange(100, 200):04d}",
            f"+44 20 7946 0{_digits(rng, 3)}",
            f"020 7946 0{_digits(rng, 3)}",
            f"+44 (0)161 496 0{_digits(rng, 3)}",
            f"+91 98765 {_digits(rng, 5)}",
        ]
    )


def gen_ssn(rng: random.Random) -> str:
    area = rng.randrange(100, 665)
    group, serial = rng.randrange(1, 100), rng.randrange(1, 10000)
    # Both forms are used on real forms; the SSA prints the dashed one.
    return rng.choice(
        [f"{area:03d}-{group:02d}-{serial:04d}", f"{area:03d}{group:02d}{serial:04d}"]
    )


def gen_credit_card(rng: random.Random) -> str:
    prefix, length = rng.choice([("4", 16), ("5" + str(rng.randrange(1, 6)), 16), ("34", 15)])
    body = prefix + _digits(rng, length - len(prefix) - 1)
    number = body + luhn_check_digit(body)
    style = rng.randrange(3)
    if style == 0:
        return number
    sep = " " if style == 1 else "-"
    return sep.join(number[i : i + 4] for i in range(0, len(number), 4))


def gen_dob(rng: random.Random) -> str:
    day, month, year = rng.randrange(1, 29), rng.randrange(1, 13), rng.randrange(1940, 2006)
    names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return rng.choice(
        [
            f"{day:02d}/{month:02d}/{year}",
            f"{day:02d}-{month:02d}-{year}",
            f"{year}-{month:02d}-{day:02d}",
            f"{names[month - 1]} {day}, {year}",
            f"{day} {names[month - 1][:3]} {year}",
        ]
    )


def gen_ip(rng: random.Random) -> str:
    # Documentation and private ranges only (RFC 5737, RFC 1918).
    return rng.choice(
        [
            f"192.0.2.{rng.randrange(1, 255)}",
            f"198.51.100.{rng.randrange(1, 255)}",
            f"10.{rng.randrange(0, 255)}.{rng.randrange(0, 255)}.{rng.randrange(1, 255)}",
        ]
    )


def gen_address(rng: random.Random) -> str:
    number, street, suffix = rng.randrange(1, 4000), rng.choice(STREETS), rng.choice(SUFFIX)
    return rng.choice(
        [
            f"{number} {street} {suffix}",
            f"{number} {street} {suffix}, Apt {rng.randrange(1, 40)}",
            f"Flat {rng.randrange(1, 20)}, {number} {street} {suffix}",
        ]
    )


def gen_passport(rng: random.Random) -> str:
    return rng.choice(
        [
            f"{_upper(rng, 1)}{_digits(rng, 8)}",
            f"{_upper(rng, 2)}{_digits(rng, 7)}",
            _digits(rng, 9),
        ]
    )


def gen_api_key(rng: random.Random) -> str:
    # Prefix only, with no `live`/`test`/`proj` segment. Those segments are what
    # make a string indistinguishable from a real Stripe or OpenAI credential,
    # and a corpus file full of those is one that secret scanners block, that
    # nobody can safely grep past, and that no reader can tell is fabricated.
    # The rule under test accepts the bare prefixes, so the pattern is exercised
    # at a realistic length either way.
    body = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(24))
    return rng.choice([f"sk_{body}", f"pk_{body}", f"rk_{body}"])


def gen_aws_key(rng: random.Random) -> str:
    body = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    return rng.choice(["AKIA", "ASIA"]) + body


def gen_bearer(rng: random.Random) -> str:
    body = "".join(rng.choice(string.ascii_letters + string.digits + "-._~") for _ in range(40))
    return f"Bearer {body}"


def gen_policy(rng: random.Random) -> str:
    return rng.choice([f"POL-{_digits(rng, 6)}", f"PLC/{_digits(rng, 8)}", f"P{_digits(rng, 7)}"])


def gen_claim(rng: random.Random) -> str:
    return rng.choice([f"CLM-{_digits(rng, 8)}", f"CLA/{_digits(rng, 7)}", f"C{_digits(rng, 9)}"])


def gen_adjuster(rng: random.Random) -> str:
    return rng.choice([f"ADJ-{_digits(rng, 5)}", f"LA{_digits(rng, 6)}"])


def gen_broker(rng: random.Random) -> str:
    body = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return rng.choice([f"BRK-{body}", f"BRK/{body}"])


def gen_ncd(rng: random.Random) -> str:
    return f"{rng.randrange(1, 15)} years no-claims"


def gen_plate(rng: random.Random) -> str:
    # UK current-style registration, DVLA format AA00 AAA.
    return f"{_upper(rng, 2)}{rng.randrange(10, 74):02d} {_upper(rng, 3)}"


def gen_vin(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # ISO 3779 excludes I, O, Q
    return "".join(rng.choice(alphabet) for _ in range(17))


def gen_mrn(rng: random.Random) -> str:
    return rng.choice([f"MRN {_digits(rng, 7)}", f"MRN-{_digits(rng, 8)}", f"MR#{_digits(rng, 7)}"])


def gen_mrn_bare(rng: random.Random) -> str:
    return _digits(rng, rng.randrange(6, 11))


def gen_nhs(rng: random.Random) -> str:
    while True:
        body = _digits(rng, 9)
        check = nhs_check_digit(body)
        if check is not None:
            number = body + check
            return rng.choice([f"{number[:3]} {number[3:6]} {number[6:]}", number])


def gen_icd(rng: random.Random) -> str:
    letter = rng.choice("ABCEFGHJKLMNPQRSTVWXYZ")
    code = f"{letter}{rng.randrange(0, 100):02d}"
    return rng.choice([code, f"{code}.{rng.randrange(0, 10)}"])


def gen_npi(rng: random.Random) -> str:
    return _digits(rng, 10)


def gen_medicare(rng: random.Random) -> str:
    letter = lambda: rng.choice(MBI_LETTERS)  # noqa: E731
    alnum = lambda: rng.choice(MBI_LETTERS + string.digits)  # noqa: E731
    mbi = (
        f"{rng.randrange(1, 10)}{letter()}{alnum()}{rng.randrange(0, 10)}"
        f"{letter()}{alnum()}{rng.randrange(0, 10)}{letter()}{letter()}"
        f"{rng.randrange(0, 10)}{rng.randrange(0, 10)}"
    )
    # CMS prints the grouped form on the card itself.
    return rng.choice([mbi, f"{mbi[:4]}-{mbi[4:7]}-{mbi[7:]}"])


def gen_health_plan(rng: random.Random) -> str:
    body = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    return rng.choice([f"HP-{body}", f"MBR/{body}", f"PLAN{body}"])


def gen_prescription(rng: random.Random) -> str:
    return rng.choice(
        [f"RX {_digits(rng, 7)}", f"RX-{_digits(rng, 8)}", f"SCRIPT#{_digits(rng, 7)}"]
    )


def gen_iban(rng: random.Random) -> str:
    country = rng.choice(["GB", "DE", "FR"])
    bban = {
        "GB": _upper(rng, 4) + _digits(rng, 14),
        "DE": _digits(rng, 18),
        "FR": _digits(rng, 23),
    }[country]
    iban = country + iban_check_digits(country, bban) + bban
    return rng.choice([iban, " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))])


def gen_bank_account(rng: random.Random) -> str:
    return _digits(rng, rng.randrange(8, 13))


def gen_cvv(rng: random.Random) -> str:
    return _digits(rng, rng.choice([3, 3, 4]))


def gen_crypto(rng: random.Random) -> str:
    b58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return rng.choice(
        [
            "1" + "".join(rng.choice(b58) for _ in range(33)),
            "bc1q" + "".join(rng.choice("023456789acdefghjklmnpqrstuvwxyz") for _ in range(38)),
            "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40)),
        ]
    )


def gen_swift(rng: random.Random) -> str:
    code = _upper(rng, 4) + rng.choice(["GB", "DE", "US"]) + _upper(rng, 2)
    return rng.choice([code, code + _upper(rng, 3)])


def gen_ein(rng: random.Random) -> str:
    return f"{rng.randrange(10, 100)}-{_digits(rng, 7)}"


def gen_sort_code(rng: random.Random) -> str:
    return rng.choice(
        [
            f"{rng.randrange(10, 99)}-{rng.randrange(10, 99)}-{rng.randrange(10, 99)}",
            f"{rng.randrange(10, 99)} {rng.randrange(10, 99)} {rng.randrange(10, 99)}",
        ]
    )


def gen_routing(rng: random.Random) -> str:
    return _digits(rng, 9)


GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "EMAIL": gen_email,
    "PHONE": gen_phone,
    "US_SSN": gen_ssn,
    "CREDIT_CARD": gen_credit_card,
    "DATE_OF_BIRTH": gen_dob,
    "IP_ADDRESS": gen_ip,
    "POSTAL_ADDRESS": gen_address,
    "PASSPORT": gen_passport,
    "API_KEY": gen_api_key,
    "AWS_ACCESS_KEY": gen_aws_key,
    "BEARER_TOKEN": gen_bearer,
    "POLICY_NUMBER": gen_policy,
    "CLAIM_NUMBER": gen_claim,
    "ADJUSTER_ID": gen_adjuster,
    "BROKER_CODE": gen_broker,
    "NCD_YEARS": gen_ncd,
    "LICENSE_PLATE": gen_plate,
    "VIN": gen_vin,
    "MRN": gen_mrn,
    "MRN_BARE": gen_mrn_bare,
    "NHS_NUMBER": gen_nhs,
    "ICD_CODE": gen_icd,
    "NPI": gen_npi,
    "MEDICARE_ID": gen_medicare,
    "HEALTH_PLAN_ID": gen_health_plan,
    "PRESCRIPTION_NUMBER": gen_prescription,
    "IBAN": gen_iban,
    "BANK_ACCOUNT": gen_bank_account,
    "CVV": gen_cvv,
    "CRYPTO_WALLET": gen_crypto,
    "SWIFT_BIC": gen_swift,
    "TAX_ID_EIN": gen_ein,
    "UK_SORT_CODE": gen_sort_code,
    "US_ROUTING_NUMBER": gen_routing,
}

# --------------------------------------------------------------------------
# Sentence templates. Where a rule requires a nearby context word, most
# templates supply one -- but not all of them do, because real correspondence
# does not always oblige. A missed identifier in a sentence without its cue is
# still a missed identifier, and hiding that would flatter the number.
# --------------------------------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "EMAIL": [
        "Please copy {v} on the reply.",
        "The claimant wrote in from {v}.",
        "Confirmation went to {v}.",
    ],
    "PHONE": [
        "Best contact number is {v}.",
        "I tried calling {v} twice this morning.",
        "Reachable on {v} after 6pm.",
    ],
    "US_SSN": [
        "Their SSN is {v}.",
        "Social security number {v} was given over the phone.",
        "Verified identity against {v}.",
    ],
    "CREDIT_CARD": [
        "Payment was taken on card {v}.",
        "The card ending in that number is {v}.",
        "Charge {v} for the excess.",
    ],
    "DATE_OF_BIRTH": [
        "Date of birth {v}.",
        "The patient was born {v}.",
        "DOB {v} matches the record.",
        "Their birthday is {v}.",
    ],
    "IP_ADDRESS": [
        "The request came from {v}.",
        "Logged from {v} at 09:12.",
        "Session originated at {v}.",
    ],
    "POSTAL_ADDRESS": [
        "The property is at {v}.",
        "Correspondence address: {v}.",
        "They have moved to {v}.",
    ],
    "PASSPORT": [
        "Passport {v} was supplied as ID.",
        "Travel document number {v}.",
        "Checked the passport, number {v}.",
    ],
    "API_KEY": ["The integration is using key {v}.", "Someone pasted {v} into the ticket."],
    "AWS_ACCESS_KEY": ["Access key {v} appears in the log.", "The credential {v} needs rotating."],
    "BEARER_TOKEN": ["The header was Authorization: {v}.", "Request carried {v}."],
    "POLICY_NUMBER": [
        "Policy {v} is in force.",
        "Quoting against {v}.",
        "The renewal for {v} is due.",
    ],
    "CLAIM_NUMBER": [
        "Claim {v} is with the adjuster.",
        "Opened claim {v} this morning.",
        "See {v} for the estimate.",
    ],
    "ADJUSTER_ID": ["Assigned to adjuster {v}.", "Loss adjuster {v} attended the site."],
    "BROKER_CODE": ["Placed through broker {v}.", "Intermediary code {v} on the schedule."],
    "NCD_YEARS": ["They declared {v} at inception.", "Protected {v} on the policy."],
    "LICENSE_PLATE": [
        "The vehicle registration is {v}.",
        "Plate {v} was recorded at the scene.",
        "Licence plate {v}.",
    ],
    "VIN": ["Vehicle identification number {v}.", "The VIN on the chassis reads {v}."],
    "MRN": ["Recorded under {v}.", "The chart is filed as {v}."],
    "MRN_BARE": [
        "Medical record number {v} in the system.",
        "Patient id {v} on the ward list.",
        "Chart number {v}.",
    ],
    "NHS_NUMBER": [
        "NHS number {v}.",
        "Registered with NHS number {v}.",
        "The number on file is {v}.",
    ],
    "ICD_CODE": [
        "Diagnosis coded as {v}.",
        "The ICD-10 entry is {v}.",
        "Diagnosed with {v} last spring.",
    ],
    "NPI": [
        "Referring provider NPI {v}.",
        "The national provider identifier is {v}.",
        "Provider number {v}.",
    ],
    "MEDICARE_ID": ["Medicare beneficiary identifier {v}.", "The MBI on the card is {v}."],
    "HEALTH_PLAN_ID": ["Member id {v} on the plan.", "Health plan reference {v}."],
    "PRESCRIPTION_NUMBER": ["Prescription {v} was dispensed Tuesday.", "Script {v} is on repeat."],
    "IBAN": ["Funds went to IBAN {v}.", "The account is {v}.", "Settlement to {v}."],
    "BANK_ACCOUNT": [
        "Account number {v} at the branch.",
        "The acct {v} was debited.",
        "Credit a/c {v}.",
    ],
    "CVV": [
        "The CVV given was {v}.",
        "Card security code {v} was read out.",
        "They quoted the CVC as {v}.",
    ],
    "CRYPTO_WALLET": ["Refund sent to wallet {v}.", "The address on file is {v}."],
    "SWIFT_BIC": [
        "Wire via SWIFT {v}.",
        "The BIC for the transfer is {v}.",
        "Bank code {v} on the instruction.",
    ],
    "TAX_ID_EIN": ["The employer EIN is {v}.", "Filed under {v}."],
    "UK_SORT_CODE": ["Sort code {v}.", "Paid into sort code {v}."],
    "US_ROUTING_NUMBER": [
        "ACH routing number {v}.",
        "The ABA routing is {v}.",
        "Wire routing {v}.",
    ],
}

DOMAIN_ENTITIES: dict[str, list[str]] = {
    "default": [
        "EMAIL",
        "PHONE",
        "US_SSN",
        "CREDIT_CARD",
        "DATE_OF_BIRTH",
        "IP_ADDRESS",
        "POSTAL_ADDRESS",
        "PASSPORT",
        "API_KEY",
        "AWS_ACCESS_KEY",
        "BEARER_TOKEN",
    ],
    "insurance": [
        "POLICY_NUMBER",
        "CLAIM_NUMBER",
        "ADJUSTER_ID",
        "BROKER_CODE",
        "NCD_YEARS",
        "LICENSE_PLATE",
        "VIN",
        "EMAIL",
        "PHONE",
        "POSTAL_ADDRESS",
        "DATE_OF_BIRTH",
    ],
    "healthcare": [
        "MRN",
        "MRN_BARE",
        "NHS_NUMBER",
        "ICD_CODE",
        "NPI",
        "MEDICARE_ID",
        "HEALTH_PLAN_ID",
        "PRESCRIPTION_NUMBER",
        "DATE_OF_BIRTH",
        "EMAIL",
        "PHONE",
    ],
    "fintech": [
        "IBAN",
        "BANK_ACCOUNT",
        "CVV",
        "CRYPTO_WALLET",
        "SWIFT_BIC",
        "TAX_ID_EIN",
        "UK_SORT_CODE",
        "US_ROUTING_NUMBER",
        "CREDIT_CARD",
        "EMAIL",
        "US_SSN",
    ],
}

OPENERS = [
    "Following up on the note from yesterday.",
    "Summary of the call, for the file.",
    "Handing this over before I go on leave.",
    "Adding the details the team asked for.",
    "Quick update after speaking with the customer.",
]

# Near-misses: strings that look like identifiers and are not. A detector that
# flags these is redacting a version number or an order reference, and the cost
# of that lands on whoever has to read the redacted text afterwards.
NEGATIVES = [
    "The build is on version 10.2.14.3 and the changelog is attached.",
    "Order 123456 shipped on Tuesday and arrived Thursday.",
    "We reviewed sections 4-11-2 of the handbook together.",
    "The meeting is 14/03 at 10am, no year given, room 402.",
    "Ticket ref ABC-99 was closed without action.",
    "Please read pages 100-250 before the review.",
    "Our SLA is 99.95 percent measured monthly.",
    "The invoice total came to 1,234.56 including tax.",
    "Batch 20260114 was released to staging at 3pm.",
    "Route 66 merchandise is out of stock again.",
    "Temperature held at 36.6 for the whole shift.",
    "The lift is out on floors 3-7 until Friday.",
]


def render(rng: random.Random, domain: str) -> dict[str, object]:
    """Build one labelled document for a domain."""
    entities = DOMAIN_ENTITIES[domain]
    chosen = rng.sample(entities, rng.randrange(2, min(5, len(entities)) + 1))
    parts: list[str] = [rng.choice(OPENERS)]
    spans: list[dict[str, object]] = []

    for entity in chosen:
        value = GENERATORS[entity](rng)
        sentence = rng.choice(TEMPLATES[entity])
        prefix = " ".join(parts) + " "
        offset = len(prefix) + sentence.index("{v}")
        parts.append(sentence.replace("{v}", value))
        spans.append({"entity": entity, "start": offset, "end": offset + len(value)})

    if rng.random() < 0.4:
        parts.append(rng.choice(NEGATIVES))

    return {"pack": domain, "text": " ".join(parts), "spans": spans}


def main() -> None:
    """Write the corpus to `corpus.jsonl`."""
    rng = random.Random(SEED)
    records: list[dict[str, object]] = []

    for domain in DOMAIN_ENTITIES:
        for _ in range(DOCUMENTS_PER_DOMAIN):
            records.append(render(rng, domain))
        for _ in range(NEGATIVES_PER_DOMAIN):
            body = " ".join(rng.sample(NEGATIVES, rng.randrange(2, 5)))
            records.append({"pack": domain, "text": body, "spans": []})

    for index, record in enumerate(records):
        record["id"] = f"{record['pack']}-{index:04d}"

    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    labelled = sum(len(r["spans"]) for r in records)  # type: ignore[arg-type]
    print(f"wrote {len(records)} documents and {labelled} labelled spans to {OUTPUT}")


if __name__ == "__main__":
    main()
