#!/usr/bin/env python3
"""Score `wardhook-guardrails` PII detection against the labelled corpus.

    python benchmarks/pii/run.py            # print the tables
    python benchmarks/pii/run.py --write    # and update results.md

**What is measured, and why in this shape.** A redactor's job is to cover the
sensitive substring. So the headline recall asks *was this gold span fully
covered by some detection*, whichever rule fired -- an NHS number caught by the
phone rule is mislabelled but not leaked. A second, stricter recall asks whether
the label was right too, because a wrong label degrades an audit trail even when
the redaction was correct.

Precision is reported the same way round. A detection overlapping no labelled
span is a false positive: it redacts something that was not personal data, and
that cost lands on whoever has to read the text afterwards. Detections that
land on real PII under the wrong name are counted separately rather than being
folded into either number.

The corpus is checked before it is scored. A span whose offsets no longer line
up with its text would silently deflate every figure here, so a misaligned
corpus is a hard error rather than a bad result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from wardhook.guardrails import PIIDetector

CORPUS = Path(__file__).parent / "corpus.jsonl"
RESULTS = Path(__file__).parent / "results.md"


@dataclass
class Tally:
    """Counts for one entity, one pack, or everything."""

    gold: int = 0
    covered: int = 0
    typed: int = 0
    exact: int = 0
    predictions: int = 0
    on_gold: int = 0
    on_gold_typed: int = 0
    spurious: int = 0

    def __iadd__(self, other: Tally) -> Tally:
        """Accumulate another tally into this one."""
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    @property
    def recall(self) -> float:
        """Fraction of labelled spans covered by some detection."""
        return self.covered / self.gold if self.gold else float("nan")

    @property
    def recall_typed(self) -> float:
        """Fraction covered by a detection carrying the right entity name."""
        return self.typed / self.gold if self.gold else float("nan")

    @property
    def precision(self) -> float:
        """Fraction of detections that landed on labelled data at all."""
        return self.on_gold / self.predictions if self.predictions else float("nan")


@dataclass
class Report:
    """Everything one run of the benchmark produced."""

    overall: Tally = field(default_factory=Tally)
    by_entity: dict[str, Tally] = field(default_factory=lambda: defaultdict(Tally))
    by_pack: dict[str, Tally] = field(default_factory=lambda: defaultdict(Tally))
    clean_documents: int = 0
    clean_false_positives: int = 0
    clean_words: int = 0


def load(path: Path) -> list[dict]:
    """Read the corpus and prove its labels still line up with its text.

    Args:
        path: The corpus file.

    Returns:
        The records.

    Raises:
        SystemExit: If the file is missing, or a span's offsets do not match a
            non-blank fragment of its own text. Scoring a misaligned corpus
            would produce numbers that look plausible and mean nothing.
    """
    if not path.exists():
        raise SystemExit(f"No corpus at {path}. Run generate_corpus.py first.")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        for span in record["spans"]:
            fragment = record["text"][span["start"] : span["end"]]
            if not fragment.strip() or fragment != fragment.strip():
                raise SystemExit(
                    f"{record['id']}: span {span} does not line up with the text "
                    f"({fragment!r}). Regenerate the corpus."
                )
    return records


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Whether two half-open spans share at least one character."""
    return a_start < b_end and b_start < a_end


def score(records: list[dict]) -> Report:
    """Run every pack's detector over its documents and tally the outcome.

    Args:
        records: The corpus.

    Returns:
        The report.
    """
    detectors = {pack: PIIDetector(pack=pack) for pack in sorted({r["pack"] for r in records})}
    report = Report()

    for record in records:
        text = record["text"]
        found = detectors[record["pack"]].detect(text)
        gold = record["spans"]

        if not gold:
            report.clean_documents += 1
            report.clean_false_positives += len(found)
            report.clean_words += len(text.split())
            continue

        for span in gold:
            tally = Tally(gold=1)
            covering = [
                match
                for match in found
                if match.start <= span["start"] and match.end >= span["end"]
            ]
            tally.covered = int(bool(covering))
            tally.typed = int(any(m.entity == span["entity"] for m in covering))
            tally.exact = int(
                any(
                    m.start == span["start"] and m.end == span["end"] and m.entity == span["entity"]
                    for m in covering
                )
            )
            report.by_entity[span["entity"]] += tally
            report.by_pack[record["pack"]] += tally
            report.overall += tally

        for match in found:
            tally = Tally(predictions=1)
            hits = [s for s in gold if _overlaps(match.start, match.end, s["start"], s["end"])]
            tally.on_gold = int(bool(hits))
            tally.on_gold_typed = int(any(s["entity"] == match.entity for s in hits))
            tally.spurious = int(not hits)
            report.by_entity[match.entity] += tally
            report.by_pack[record["pack"]] += tally
            report.overall += tally

    return report


def _pct(value: float) -> str:
    """Format a rate as a percentage, or a dash when undefined."""
    return "—" if value != value else f"{value * 100:.1f}%"


def render(report: Report) -> str:
    """Turn a report into markdown.

    Args:
        report: The scored report.

    Returns:
        The full report as markdown.
    """
    lines: list[str] = []
    overall = report.overall

    lines.append("## Headline")
    lines.append("")
    lines.append("| Measure | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Labelled spans | {overall.gold:,} |")
    lines.append(f"| **Recall — span covered by some rule** | **{_pct(overall.recall)}** |")
    lines.append(f"| Recall — covered *and* correctly named | {_pct(overall.recall_typed)} |")
    lines.append(f"| Recall — exact span and name | {_pct(overall.exact / overall.gold)} |")
    lines.append(f"| Detections | {overall.predictions:,} |")
    lines.append(
        f"| **Precision — detection landed on real PII** | **{_pct(overall.precision)}** |"
    )
    lines.append(
        f"| Detections on PII but under the wrong name | {overall.on_gold - overall.on_gold_typed:,} |"
    )
    lines.append(f"| Detections on nothing labelled | {overall.spurious:,} |")
    per_thousand = (
        report.clean_false_positives / report.clean_words * 1000 if report.clean_words else 0.0
    )
    lines.append(
        f"| False positives on {report.clean_documents} clean documents "
        f"| {report.clean_false_positives} ({per_thousand:.1f} per 1,000 words) |"
    )
    lines.append("")

    lines.append("## By pack")
    lines.append("")
    lines.append("| Pack | Spans | Recall | Recall (typed) | Precision |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for pack in sorted(report.by_pack):
        tally = report.by_pack[pack]
        lines.append(
            f"| {pack} | {tally.gold:,} | {_pct(tally.recall)} "
            f"| {_pct(tally.recall_typed)} | {_pct(tally.precision)} |"
        )
    lines.append("")

    lines.append("## By entity")
    lines.append("")
    lines.append("| Entity | Spans | Recall | Recall (typed) | Detections | Precision |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for entity in sorted(report.by_entity, key=lambda e: (report.by_entity[e].recall, e)):
        tally = report.by_entity[entity]
        lines.append(
            f"| `{entity}` | {tally.gold:,} | {_pct(tally.recall)} "
            f"| {_pct(tally.recall_typed)} | {tally.predictions:,} | {_pct(tally.precision)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Score the corpus and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update results.md as well.")
    args = parser.parse_args()

    report = render(score(load(CORPUS)))
    print(report)
    if args.write:
        RESULTS.write_text(
            "# PII detection benchmark — results\n\n"
            "Generated by `python benchmarks/pii/run.py --write`. Method, and what\n"
            "these numbers do and do not support, in [README.md](README.md).\n\n" + report,
            encoding="utf-8",
        )
        print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
