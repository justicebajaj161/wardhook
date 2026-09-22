# Demo workspace

`leaky_agent.py` carries planted, fake secrets so the extension has something
to find. Pressing **F5** opens this folder in the Extension Development Host,
so the findings appear immediately.

At the default settings three of the five detected items are reported:

| Entity | Severity | Validated | Reported |
| --- | --- | --- | --- |
| `US_SSN` | high | no | yes |
| `CREDIT_CARD` | critical | **yes** (Luhn) | yes |
| `AWS_ACCESS_KEY` | critical | no | yes |
| `EMAIL` | medium | no | no |
| `PHONE` | medium | no | no |

Set `wardhook.minSeverity` to `medium` to surface the other two, which is the
quickest way to show the precision filter doing real work.
