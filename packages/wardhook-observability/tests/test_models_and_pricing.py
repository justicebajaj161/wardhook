"""Records round-trip losslessly, and the cost arithmetic is right."""

from __future__ import annotations

import pytest

from wardhook.observability import (
    ModelPrice,
    TokenUsage,
    Trace,
    TraceStep,
    UnknownModelWarning,
    estimate_cost,
    get_price,
    known_models,
    normalise_model_name,
    register_price,
)
from wardhook.observability.pricing import _WARNED, PRICES


class TestTokenUsage:
    def test_totals_and_addition(self, usage):
        assert usage.total_tokens == 1200
        doubled = usage + usage
        assert doubled.input_tokens == 2000
        assert doubled.reasoning_tokens == 100

    def test_uncached_input_excludes_both_cache_buckets(self, usage):
        # 1000 total input, of which 400 were read from cache and 100 written.
        assert usage.uncached_input_tokens == 500

    def test_uncached_input_never_goes_negative(self):
        # A provider reporting inconsistent counts must not produce a credit.
        broken = TokenUsage(input_tokens=10, cache_read_tokens=999)
        assert broken.uncached_input_tokens == 0

    def test_round_trip_through_dict(self, usage):
        assert TokenUsage.from_dict(usage.to_dict()) == usage

    def test_zero_fields_are_omitted_from_serialisation(self):
        assert TokenUsage(input_tokens=5).to_dict() == {"input_tokens": 5}

    def test_from_dict_tolerates_none_and_unknown_keys(self):
        assert TokenUsage.from_dict(None).is_empty
        assert TokenUsage.from_dict({"input_tokens": 3, "invented": 9}).input_tokens == 3

    def test_adding_a_non_usage_is_not_implemented(self):
        with pytest.raises(TypeError):
            _ = TokenUsage() + 5


class TestTraceRecords:
    def test_step_exposes_the_published_attribute_names(self):
        # The package README documents step.node/.latency_ms/.tokens_out/.cost.
        step = TraceStep("call_model", "r1", "t", 12.0, TokenUsage(input_tokens=9, output_tokens=4))
        assert (step.node, step.latency_ms, step.tokens_in, step.tokens_out) == (
            "call_model",
            12.0,
            9,
            4,
        )
        assert step.cost == 0.0 and not step.failed

    def test_trace_totals_sum_across_steps(self, sample_trace):
        assert sample_trace.total_tokens_in == 900
        assert sample_trace.total_tokens_out == 120
        assert sample_trace.total_cost == pytest.approx(0.0057)
        assert sample_trace.slowest_step.node == "call_model"

    def test_trace_round_trips_through_dict(self, sample_trace):
        assert Trace.from_dict(sample_trace.to_dict()) == sample_trace

    def test_totals_are_recomputed_not_trusted_on_load(self, sample_trace):
        # A hand-edited file must not be able to claim totals its steps do not
        # support -- the `totals` block is derived output, never input.
        payload = sample_trace.to_dict()
        payload["totals"]["cost"] = 999.0
        assert Trace.from_dict(payload).total_cost == pytest.approx(0.0057)

    def test_a_failing_step_makes_the_whole_trace_failed(self):
        step = TraceStep("tools", "r1", "t", 1.0, error="ValueError: boom")
        assert step.failed
        assert Trace("r1", (step,)).failed

    def test_empty_trace_has_no_slowest_step(self):
        assert Trace("r1").slowest_step is None


class TestPricing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("claude-opus-5", "claude-opus-5"),
            ("  Claude-Opus-5  ", "claude-opus-5"),
            ("us.anthropic.claude-opus-5", "claude-opus-5"),
            ("global.anthropic.claude-sonnet-5", "claude-sonnet-5"),
            ("claude-opus-4-5@20251101", "claude-opus-4-5"),
            ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
        ],
    )
    def test_provider_prefixes_and_dated_suffixes_are_stripped(self, raw, expected):
        assert normalise_model_name(raw) == expected

    def test_sonnet_5_is_priced_separately_from_sonnet_4_6(self):
        # These two are easy to conflate and are NOT the same rate.
        assert get_price("claude-sonnet-5") == ModelPrice(2.00, 10.00)
        assert get_price("claude-sonnet-4-6") == ModelPrice(3.00, 15.00)

    def test_simple_cost(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000)
        # 1M in at $5 + 100k out at $25/1M = 5.00 + 2.50
        assert estimate_cost("claude-opus-5", usage) == pytest.approx(7.50)

    def test_cached_tokens_are_not_billed_twice(self):
        # LangChain reports input_tokens as the TOTAL, cache reads included.
        # A fully cached prompt must cost 0.1x the input rate -- not 1.1x it.
        fully_cached = TokenUsage(input_tokens=1_000_000, cache_read_tokens=1_000_000)
        assert estimate_cost("claude-opus-5", fully_cached) == pytest.approx(0.50)

    def test_cache_writes_carry_the_premium_multiplier(self):
        written = TokenUsage(input_tokens=1_000_000, cache_write_tokens=1_000_000)
        assert estimate_cost("claude-opus-5", written) == pytest.approx(6.25)

    def test_mixed_cache_buckets(self):
        usage = TokenUsage(
            input_tokens=1_000_000,
            cache_read_tokens=600_000,
            cache_write_tokens=200_000,
            output_tokens=0,
        )
        # 200k uncached + 600k * 0.1 + 200k * 1.25 = 510k effective tokens at $5/1M.
        assert estimate_cost("claude-opus-5", usage) == pytest.approx(2.55)

    def test_unknown_model_costs_zero_and_warns_once(self):
        _WARNED.discard("mystery-model")
        usage = TokenUsage(input_tokens=1000, output_tokens=100)
        with pytest.warns(UnknownModelWarning, match="mystery-model"):
            assert estimate_cost("mystery-model", usage) == 0.0
        # The second call must stay silent, or a per-node loop floods the logs.
        with warnings_as_errors():
            assert estimate_cost("mystery-model", usage) == 0.0

    def test_no_model_costs_zero_without_warning(self):
        with warnings_as_errors():
            assert estimate_cost(None, TokenUsage(input_tokens=99)) == 0.0
            assert estimate_cost("", TokenUsage(input_tokens=99)) == 0.0

    def test_register_price_overrides_and_normalises(self):
        try:
            register_price("US.Anthropic.My-Finetune", ModelPrice(1.0, 2.0))
            assert get_price("my-finetune") == ModelPrice(1.0, 2.0)
            assert "my-finetune" in known_models()
            usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
            assert estimate_cost("my-finetune", usage) == pytest.approx(3.0)
        finally:
            PRICES.pop("my-finetune", None)

    def test_registering_clears_a_previous_unknown_model_warning(self):
        _WARNED.discard("later-known")
        with pytest.warns(UnknownModelWarning):
            estimate_cost("later-known", TokenUsage(input_tokens=1))
        try:
            register_price("later-known", ModelPrice(1.0, 1.0))
            assert "later-known" not in _WARNED
        finally:
            PRICES.pop("later-known", None)
            _WARNED.discard("later-known")


def warnings_as_errors():
    """Context manager turning any warning into an error."""
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield

    return _ctx()
