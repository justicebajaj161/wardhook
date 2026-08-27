"""Shared pytest fixtures for the wardhook-observability test suite."""

from __future__ import annotations

import pytest

from wardhook.observability import TokenUsage, Trace, Tracer, TraceStep


@pytest.fixture
def tracer() -> Tracer:
    """A fresh tracer holding nothing."""
    return Tracer()


@pytest.fixture
def usage() -> TokenUsage:
    """A usage with every field populated, so nothing is silently dropped."""
    return TokenUsage(
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=400,
        cache_write_tokens=100,
        reasoning_tokens=50,
    )


@pytest.fixture
def sample_trace() -> Trace:
    """A two-step trace: one retrieval node and one model call, plus a failure."""
    return Trace(
        run_id="run-1",
        started_at="2026-06-24T10:00:00+00:00",
        latency_ms=520.0,
        metadata={"agent": "support", "model": "claude-opus-5"},
        steps=(
            TraceStep(
                node="retrieve",
                run_id="run-1",
                started_at="2026-06-24T10:00:00+00:00",
                latency_ms=40.0,
            ),
            TraceStep(
                node="call_model",
                run_id="run-1",
                started_at="2026-06-24T10:00:00.040000+00:00",
                latency_ms=460.0,
                usage=TokenUsage(input_tokens=900, output_tokens=120, cache_read_tokens=400),
                cost=0.0057,
                model="claude-opus-5",
            ),
        ),
    )


@pytest.fixture
def fake_model():
    """Build a fake chat model that reports token usage like a real provider."""

    def build(
        text: str = "ok",
        *,
        input_tokens: int = 900,
        output_tokens: int = 120,
        cache_read: int = 0,
        model: str = "claude-opus-5",
        turns: int = 1,
    ):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        messages = [
            AIMessage(
                content=text,
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_token_details": {"cache_read": cache_read},
                },
                response_metadata={"model_name": model},
            )
            for _ in range(turns)
        ]
        return GenericFakeChatModel(messages=iter(messages))

    return build
