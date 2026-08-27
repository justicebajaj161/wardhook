"""Per-model token pricing, and the arithmetic that turns usage into dollars.

The price table below is **current as of 2026-06-24** (see :data:`PRICES_AS_OF`)
and covers Anthropic's published first-party API rates. Provider pricing moves,
and a cost estimate is only as trustworthy as the table behind it, so this
module is built around three rules:

1. **State the vintage.** :data:`PRICES_AS_OF` is exported and rendered in the
   trace viewer, so nobody has to guess how old a number is.
2. **Never guess.** An unrecognised model costs ``0.0`` and emits an
   :class:`UnknownModelWarning` once. Inventing a plausible rate for an unknown
   model would put a wrong number in a budget with no signal that it is wrong.
3. **Stay overridable.** :func:`register_price` lets you correct a rate, add a
   model, or price a self-hosted endpoint without forking this package.

Partner platforms (Amazon Bedrock, Google Vertex AI) bill at their own rates.
Model ids from those platforms still *resolve* here -- :func:`normalise_model_name`
strips the ``us.anthropic.`` style prefixes and dated suffixes -- but the
resulting figure is the first-party rate. Override it with
:func:`register_price` if you are billed elsewhere.

Example:
    >>> from wardhook.observability.models import TokenUsage
    >>> usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000)
    >>> round(estimate_cost("claude-opus-5", usage), 2)
    7.5
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wardhook.observability.models import TokenUsage

__all__ = [
    "PRICES",
    "PRICES_AS_OF",
    "ModelPrice",
    "UnknownModelWarning",
    "estimate_cost",
    "get_price",
    "known_models",
    "normalise_model_name",
    "register_price",
]

PRICES_AS_OF = "2026-06-24"
"""The date the built-in :data:`PRICES` table was last verified."""


class UnknownModelWarning(UserWarning):
    """Raised as a warning when a model has no entry in the price table.

    Deliberately a warning and not an exception: an agent should not fall over
    because its telemetry cannot price a model. The run continues, the cost is
    reported as ``0.0``, and the gap is visible rather than silent.
    """


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """What one model costs per million tokens.

    Attributes:
        input_per_1m: US dollars per million input tokens.
        output_per_1m: US dollars per million output tokens.
        cache_read_multiplier: Fraction of the input rate charged for tokens
            served from a prompt cache. Roughly ``0.1`` across Anthropic models.
        cache_write_multiplier: Multiple of the input rate charged for tokens
            written into a prompt cache. Roughly ``1.25`` for the default
            five-minute TTL; a one-hour TTL costs more, so override it if you
            use one.

    Example:
        >>> ModelPrice(5.0, 25.0).output_per_1m
        25.0
    """

    input_per_1m: float
    output_per_1m: float
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


# Anthropic first-party API rates, US dollars per million tokens.
PRICES: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-mythos-5": ModelPrice(10.00, 50.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

# Cloud vendors prefix model ids with a routing region and a vendor namespace.
_PROVIDER_PREFIXES = ("us.", "eu.", "apac.", "global.", "anthropic.")
# Dated snapshots: `-20251101` on the Claude API, `@20251101` on Vertex AI.
_DATE_SUFFIX = re.compile(r"[-@]\d{8}$")

# Models already warned about, so a per-node cost calculation in a hot loop
# does not emit thousands of identical warnings.
_WARNED: set[str] = set()


def normalise_model_name(model: str) -> str:
    """Reduce a provider-specific model id to its base name.

    Args:
        model: A model id, possibly carrying a cloud-provider prefix or a dated
            snapshot suffix.

    Returns:
        The lowercased base name used as a key in :data:`PRICES`.

    Example:
        >>> normalise_model_name("us.anthropic.claude-opus-5")
        'claude-opus-5'
        >>> normalise_model_name("claude-opus-4-5@20251101")
        'claude-opus-4-5'
    """
    name = model.strip().lower()
    changed = True
    while changed:
        changed = False
        for prefix in _PROVIDER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return _DATE_SUFFIX.sub("", name)


def get_price(model: str | None) -> ModelPrice | None:
    """Look up the price for a model.

    Args:
        model: A model id, in any of the forms :func:`normalise_model_name`
            accepts. ``None`` means no model was involved.

    Returns:
        The matching :class:`ModelPrice`, or ``None`` when the model is unknown
        or was not supplied. No warning is emitted here -- see
        :func:`estimate_cost`.

    Example:
        >>> get_price("claude-sonnet-5").input_per_1m
        2.0
        >>> get_price("some-local-llm") is None
        True
    """
    if not model:
        return None
    return PRICES.get(normalise_model_name(model))


def register_price(model: str, price: ModelPrice) -> None:
    """Add or override the price for a model.

    Use this to correct a rate that has moved, to price a model this package
    does not know about, or to bill at a partner platform's rates rather than
    Anthropic's first-party ones.

    Args:
        model: The model id. Normalised before storage, so registering
            ``"US.Anthropic.My-Model"`` also matches ``"my-model"``.
        price: The rates to apply.

    Example:
        >>> register_price("my-finetune", ModelPrice(0.5, 1.5))
        >>> get_price("my-finetune").output_per_1m
        1.5
    """
    name = normalise_model_name(model)
    PRICES[name] = price
    _WARNED.discard(name)


def known_models() -> list[str]:
    """Return every model this package can price, sorted.

    Returns:
        The normalised model names currently in :data:`PRICES`, including any
        added with :func:`register_price`.
    """
    return sorted(PRICES)


def estimate_cost(model: str | None, usage: TokenUsage) -> float:
    """Estimate what one model call cost, in US dollars.

    Cached tokens are the subtle part. LangChain reports ``input_tokens`` as
    the *total* input count, with cache reads and cache writes included in it.
    Billing the cache counts on top of that total would charge for them twice,
    so the uncached remainder is derived first and each bucket is then priced
    at its own multiplier::

        uncached = input_tokens - cache_read - cache_write
        input_cost = (uncached + cache_read * 0.1 + cache_write * 1.25) * rate

    Args:
        model: The model that produced this usage. ``None`` or an unrecognised
            id yields ``0.0``.
        usage: The token counts to price.

    Returns:
        The estimated cost in US dollars, or ``0.0`` when the model is unknown.

    Warns:
        UnknownModelWarning: Once per unrecognised model id, so a missing price
            is visible without flooding the logs.

    Example:
        >>> from wardhook.observability.models import TokenUsage
        >>> # A fully cached prompt bills at a tenth of the input rate.
        >>> cached = TokenUsage(input_tokens=1_000_000, cache_read_tokens=1_000_000)
        >>> round(estimate_cost("claude-opus-5", cached), 2)
        0.5
    """
    if not model:
        return 0.0

    price = get_price(model)
    if price is None:
        name = normalise_model_name(model)
        if name not in _WARNED:
            _WARNED.add(name)
            warnings.warn(
                f"No price is known for model {model!r}, so its cost is reported as 0.00. "
                f"The built-in table was current as of {PRICES_AS_OF} and covers: "
                f"{', '.join(known_models())}. Register a rate with "
                f"wardhook.observability.pricing.register_price() to fix this.",
                UnknownModelWarning,
                stacklevel=2,
            )
        return 0.0

    billable_input = (
        usage.uncached_input_tokens
        + usage.cache_read_tokens * price.cache_read_multiplier
        + usage.cache_write_tokens * price.cache_write_multiplier
    )
    input_cost = billable_input * price.input_per_1m / 1_000_000
    output_cost = usage.output_tokens * price.output_per_1m / 1_000_000
    return input_cost + output_cost
