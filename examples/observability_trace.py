"""Example: per-node tokens, cost, and latency, plus a static HTML trace viewer.

Runs fully offline against a fake model, so no API key is needed:

    python examples/observability_trace.py

The point of this example is the arithmetic in section 2. Prompt caching is
where an agent's bill is actually decided, and the easy way to compute cost is
wrong: LangChain reports `input_tokens` as the total *including* cached tokens,
so billing the cache on top of it charges those tokens twice -- at 1.1x the
input rate instead of 0.1x, eleven times too much for that portion. Section 2
shows what that does to a realistic 90%-cached prompt.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from wardhook.observability import (
    JSONLTraceStore,
    TokenUsage,
    Tracer,
    estimate_cost,
    render_html,
)
from wardhook.observability.pricing import PRICES_AS_OF


def section(title: str) -> None:
    """Print a section heading.

    Args:
        title: The heading text.
    """
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def build_model(cache_read: int = 0):
    """Return a fake chat model that reports token usage like a real provider.

    Args:
        cache_read: How many input tokens to report as served from cache.

    Returns:
        A model instance. Imported lazily so the rest of this file runs even
        without langchain-core present.
    """
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Storm damage carries a 500 excess, subject to a 55mph wind check.",
                    usage_metadata={
                        "input_tokens": 4200,
                        "output_tokens": 180,
                        "total_tokens": 4380,
                        "input_token_details": {"cache_read": cache_read, "cache_creation": 0},
                    },
                    response_metadata={"model_name": "claude-opus-5"},
                )
            ]
        )
    )


def demo_per_node(tracer: Tracer) -> None:
    """Trace one agent run and print the per-node breakdown."""
    section("1. Where the time and the money actually went")
    from wardhook.core import AgentGraph

    agent = AgentGraph(model=build_model(), telemetry=tracer)
    agent.invoke("What excess applies to storm damage?")

    trace = tracer.get_trace()
    print(f"  {'node':<16}{'latency':>10}{'in':>9}{'out':>7}{'cost':>11}")
    for step in trace.steps:
        print(
            f"  {step.node:<16}{step.latency_ms:>9.1f}ms{step.tokens_in:>9,}"
            f"{step.tokens_out:>7,}{step.cost:>11.5f}"
        )
    print(f"  {'TOTAL':<16}{trace.latency_ms:>9.1f}ms{'':>9}{'':>7}{trace.total_cost:>11.5f}")
    print("\n  Nodes that never call a model cost nothing. That is worth seeing")
    print("  separately from the one that does.")


def demo_cache_arithmetic() -> None:
    """Show why cached tokens must not be billed twice."""
    section("2. The cache calculation that is easy to get wrong")
    rate_in, rate_out = 5.0, 25.0

    cold = TokenUsage(input_tokens=4200, output_tokens=180)
    warm = TokenUsage(input_tokens=4200, output_tokens=180, cache_read_tokens=3800)

    naive = (4200 * rate_in + 3800 * rate_in * 0.1) / 1e6 + 180 * rate_out / 1e6
    correct = estimate_cost("claude-opus-5", warm)

    print(f"  cold prompt, nothing cached      ${estimate_cost('claude-opus-5', cold):.5f}")
    print(f"  warm prompt, 3800 tokens cached  ${correct:.5f}")
    print(
        f"  the same, billed the wrong way   ${naive:.5f}   <- {naive / correct - 1:.0%} too high"
    )
    print("\n  input_tokens is the TOTAL, cached tokens included. Adding the cache")
    print("  on top charges them twice. wardhook-observability derives the")
    print("  uncached remainder first, so a fully cached prompt costs 0.1x -- not 1.1x.")


def demo_unknown_model() -> None:
    """Show that an unpriced model is reported honestly, not guessed at."""
    section("3. An unknown model costs zero and says so")
    import warnings

    usage = TokenUsage(input_tokens=1000, output_tokens=100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost = estimate_cost("some-self-hosted-llm", usage)

    print(f"  cost reported: ${cost:.5f}")
    print(f"  warning:       {str(caught[0].message).splitlines()[0] if caught else 'none'}")
    print(f"\n  The built-in table states its vintage ({PRICES_AS_OF}). Guessing a")
    print("  plausible rate would put a confident wrong number into a budget.")


def demo_viewer(tracer: Tracer, store: JSONLTraceStore) -> Path:
    """Render the trace to a self-contained HTML page.

    Args:
        tracer: The tracer holding the run.
        store: Where the trace was persisted.

    Returns:
        The path the page was written to.
    """
    section("4. One HTML file, no network")
    page = render_html(store.read(), title="Wardhook example trace")
    output = Path(tempfile.mkdtemp()) / "trace.html"
    output.write_text(page, encoding="utf-8")

    external = page.count("http://") + page.count("https://")
    print(f"  wrote {output}")
    print(f"  {len(page):,} bytes, {external} external references")
    print(f"  traces on disk: {len(store)}  (in memory: {len(tracer.traces())})")
    print("\n  Inlined CSS and JS, every value escaped. It opens from a file path,")
    print("  an email attachment, or a CI artifact, and behaves the same offline.")
    print("\n  From the command line, the same thing:")
    print(f"    wardhook-trace view {store.path} -o trace.html")
    return output


def main() -> int:
    """Run every demonstration.

    Returns:
        A process exit code.
    """
    store = JSONLTraceStore(Path(tempfile.mkdtemp()) / "traces.jsonl")
    tracer = Tracer(store=store)

    demo_per_node(tracer)
    demo_cache_arithmetic()
    demo_unknown_model()
    demo_viewer(tracer, store)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
