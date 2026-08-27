"""Example: all four packages composed into one governed, traced, tested agent.

Runs fully offline against a fake model, so no API key is needed:

    python examples/combined_agent.py

The point of this example is that none of these four packages imports another.
They meet through the structural contracts in `wardhook.core.protocols`: a
guardrail is anything with the right hooks, a telemetry sink is anything with
the right lifecycle methods, and the eval runner targets anything with
`.invoke()`. Delete any one package from your environment and the rest still
work -- which is what the CI matrix installing each one alone actually proves.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from wardhook.core import AgentGraph, InMemoryVectorStore, Retriever, chunk_text
from wardhook.evals import EvalRunner, compare, load_cases
from wardhook.guardrails import (
    AuditLogger,
    InjectionDetector,
    PIIRedactor,
    RoleBasedToolPolicy,
)
from wardhook.observability import JSONLTraceStore, Tracer, render_html

CASES = Path(__file__).parent / "data" / "claims_cases.jsonl"

POLICY = """\
Section 4 -- Storm and flood damage.
Storm damage claims carry a 500 excess. Flood damage carries a 1000 excess and
requires a loss adjuster to attend before settlement. Cover applies only where
wind speeds exceeded 55 mph, as recorded by the nearest Met Office station.

Section 5 -- Escape of water.
Escape of water carries a 250 excess. Trace and access costs are covered up to
5000. Damage caused by gradual seepage is excluded.
"""

ANSWER = (
    "Storm damage carries a 500 excess, and cover applies only where wind "
    "speeds exceeded 55 mph [1]."
)


def section(title: str) -> None:
    """Print a section heading.

    Args:
        title: The heading text.
    """
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def lookup_policy(section: str) -> str:
    """Look up a section of the policy wording.

    Args:
        section: The section to read, such as "storm" or "escape of water".

    Returns:
        The matching text from the policy.
    """
    return f"Policy section for {section!r}: see the wording above."


def issue_refund(claim_id: str, amount: float) -> str:
    """Issue a refund against a claim. Requires supervisor authority.

    Args:
        claim_id: The claim reference.
        amount: Amount to refund, in pounds.

    Returns:
        A confirmation line.
    """
    return f"Refunded {amount} against {claim_id}."


def build_model(*replies, tool_call: tuple[str, dict] | None = None):
    """Return a fake chat model reporting token usage like a real provider.

    Args:
        *replies: What the model should say, one per turn.
        tool_call: An optional ``(tool_name, arguments)`` the model asks for on
            its first turn, so the tool and RBAC paths can be exercised.

    Returns:
        A model instance. langchain-core is imported lazily so this file's
        top-level imports stay honest about what the example itself depends on.
    """
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class ToolCallingFake(GenericFakeChatModel):
        """A fake model that accepts ``bind_tools``.

        ``GenericFakeChatModel`` raises ``NotImplementedError`` from
        ``bind_tools``, so it cannot drive the tool path on its own. Accepting
        the binding and returning itself is enough to replay a scripted
        conversation through the full model/tool loop with no provider.
        """

        def bind_tools(
            self,
            tools,  # noqa: ARG002 - interface contract; the script decides the calls
            **kwargs,  # noqa: ARG002 - interface contract
        ):
            """Accept a tool binding and return this model unchanged.

            Args:
                tools: Ignored; the scripted messages decide the calls.
                **kwargs: Ignored.

            Returns:
                This model.
            """
            return self

    usage = {
        "input_tokens": 4200,
        "output_tokens": 180,
        "total_tokens": 4380,
        "input_token_details": {"cache_read": 3800},
    }
    metadata = {"model_name": "claude-opus-5"}

    messages = []
    if tool_call is not None:
        name, arguments = tool_call
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": arguments, "id": "call-1"}],
                usage_metadata=usage,
                response_metadata=metadata,
            )
        )
    messages.extend(
        AIMessage(content=reply, usage_metadata=usage, response_metadata=metadata)
        for reply in replies
    )
    return ToolCallingFake(messages=iter(messages))


def build_agent(tracer: Tracer, model=None, reply: str = ANSWER) -> AgentGraph:
    """Assemble the agent from all four packages.

    Args:
        tracer: The telemetry sink to attach.
        model: A prepared fake model. Built from ``reply`` when omitted.
        reply: What the fake model should answer.

    Returns:
        A configured agent.
    """
    store = InMemoryVectorStore()
    store.add(chunk_text(POLICY, "policy-wording.md", chunk_size=240, chunk_overlap=40))

    return AgentGraph(
        model=model if model is not None else build_model(reply),
        tools=[lookup_policy, issue_refund],
        retriever=Retriever(store, k=2),
        guardrails=[
            InjectionDetector(),
            PIIRedactor(pack="insurance"),
            RoleBasedToolPolicy({"agent": ["lookup_*"], "supervisor": ["lookup_*", "issue_*"]}),
        ],
        telemetry=tracer,
        system_prompt="You are a claims assistant for an insurance carrier.",
    )


def demo_one_request(tracer: Tracer, audit: AuditLogger) -> None:
    """Run one governed request and show what each package contributed."""
    section("1. One request, through all four packages")
    agent = build_agent(tracer)

    result = agent.invoke(
        "My email is alice@example.com -- what excess applies to storm damage on POL-889231?",
        principal={"id": "u-17", "roles": ["agent"]},
    )

    print(f"  answer     {result['output']}")
    print(f"\n  seen by the model (core + guardrails):\n    {result['messages'][0].content}")

    print("\n  citations (core):")
    for position, citation in enumerate(result["citations"], start=1):
        print(
            f"    [{position}] {citation['source']} chunk {citation['chunk_index']} "
            f"score {citation['score']:.3f}"
        )

    print("\n  guardrail events (guardrails):")
    for event in result["guardrail_events"]:
        print(f"    {event['stage']:<7} {event['guardrail']:<16} {event['action']}")
        audit.log(
            guardrail=event["guardrail"],
            action=event["action"],
            stage=event["stage"],
            run_id=result["run_id"],
            rule=event.get("rule"),
        )

    trace = tracer.get_trace()
    print("\n  trace (observability):")
    for step in trace.steps:
        print(
            f"    {step.node:<14}{step.latency_ms:>8.1f}ms{step.tokens_out:>6} tok"
            f"  ${step.cost:.5f}"
        )
    print(f"    {'TOTAL':<14}{trace.latency_ms:>8.1f}ms{'':>6}     ${trace.total_cost:.5f}")


def demo_rbac() -> None:
    """Show a tool call denied by role, without the tool ever running."""
    section("2. The same agent, a caller without authority")
    tracer = Tracer()
    # The model asks for `issue_refund`; RBAC decides whether it ever runs.
    model = build_model(
        "I am not able to issue a refund on this account.",
        tool_call=("issue_refund", {"claim_id": "CLM-100045", "amount": 5000}),
    )
    agent = build_agent(tracer, model=model)

    result = agent.invoke(
        "Issue a refund of 5000 against CLM-100045.",
        principal={"id": "u-17", "roles": ["agent"]},
    )
    print("  the model asked for   issue_refund(claim_id='CLM-100045', amount=5000)")
    for event in result["guardrail_events"]:
        if event["stage"] == "tool_call":
            print(f"  the policy said       {event['action'].upper()}  ({event['guardrail']})")
    print(f"  the answer            {result['output']}")

    refunded = "Refunded" in str([m.content for m in result["messages"]])
    print(f"  did the tool run?     {'yes' if refunded else 'no'}")
    print("\n  `issue_refund` is registered on the agent but not granted to this")
    print("  role. The model could ask for it; the function never executed, and")
    print("  the model was told it was denied rather than being left to guess.")


class ClaimsTarget:
    """The composed agent, wrapped so each case gets a fresh scripted run.

    A real target would be one long-lived ``AgentGraph``. A scripted fake model
    can only be replayed once, so a new one is built per question -- everything
    around it (retrieval, guardrails, telemetry, tool gating) is the real thing.

    Exposing ``.trace()`` is what lets the eval runner read cost back without
    importing ``wardhook-observability``: it duck-types that method, and the
    ``max_cost_usd`` criterion works or reports "no cost known".

    Args:
        conflate_flood: Simulate a prompt change that makes the agent answer
            flood questions with storm figures.
    """

    name = "claims-assistant"

    def __init__(self, conflate_flood: bool = False) -> None:
        """Initialise the target. See the class docstring for arguments."""
        self.conflate_flood = conflate_flood
        self.tracer = Tracer(max_runs=50)

    def _reply_for(self, question: str) -> str:
        """Pick the scripted answer for a question.

        Args:
            question: The customer's question.

        Returns:
            What the fake model should say.
        """
        lowered = question.lower()
        if "flood" in lowered and not self.conflate_flood:
            return (
                "Flood damage carries a 1000 excess and requires a loss adjuster "
                "to attend before settlement [1]."
            )
        return ANSWER

    def invoke(self, question: str, principal: dict | None = None) -> dict:
        """Answer one question through the fully composed agent.

        Args:
            question: The customer's question.
            principal: The caller's identity.

        Returns:
            The agent's result dict.
        """
        model = build_model(
            self._reply_for(question),
            tool_call=("lookup_policy", {"section": "storm and flood damage"}),
        )
        agent = build_agent(self.tracer, model=model)
        return agent.invoke(question, principal=principal or {"id": "u-17", "roles": ["agent"]})

    def trace(self, run_id: str | None = None):
        """Return the trace for a run, so cost criteria can read it.

        Args:
            run_id: The run to look up.

        Returns:
            The trace, or ``None``.
        """
        return self.tracer.get_trace(run_id)


def demo_evals() -> None:
    """Score the agent against the case file and detect a regression."""
    section("3. Scoring it, and catching a regression")
    cases = [case for case in load_cases(CASES) if "policy" in case.tags]
    print(f"  {len(cases)} case(s) tagged 'policy' from {CASES.name}\n")

    good = EvalRunner(ClaimsTarget()).run(cases)
    print(f"  current agent   {good.passed}/{good.total} passed")

    broken = EvalRunner(ClaimsTarget(conflate_flood=True)).run(cases)
    print(f"  after a change  {broken.passed}/{broken.total} passed")

    comparison = compare(broken, good)
    print(f"\n  {comparison.summary()}")
    for item in comparison.regressed:
        print(f"    REGRESSED  {item.id}  -- {item.detail}")
    print("\n  The eval runner never imported the agent framework. It targets")
    print("  anything with .invoke(), which is why a suite outlives a rewrite.")


def demo_artifacts(tracer: Tracer, audit: AuditLogger, out: Path) -> None:
    """Write the audit trail and the trace viewer to disk."""
    section("4. What you can hand to a reviewer")
    store = JSONLTraceStore(out / "traces.jsonl")
    for trace in tracer.traces():
        store.append(trace)

    page = out / "trace.html"
    page.write_text(render_html(store.read(), title="Combined example"), encoding="utf-8")

    report = audit.report()
    print(f"  audit trail   {audit.path}")
    print(f"                {report['total_events']} events, by action: {report['by_action']}")
    print(f"  trace viewer  {page}  ({page.stat().st_size:,} bytes, no network requests)")
    print("\n  The audit record says what changed and where -- never what it was.")


def main() -> int:
    """Run every demonstration.

    Returns:
        A process exit code.
    """
    out = Path(tempfile.mkdtemp())
    tracer = Tracer(max_runs=20)
    audit = AuditLogger(out / "audit.jsonl")

    demo_one_request(tracer, audit)
    demo_rbac()
    demo_evals()
    demo_artifacts(tracer, audit, out)

    section("What just happened")
    print("  core          ran the graph, retrieved with citations, called tools")
    print("  guardrails    scored injection, redacted PII both ways, gated the refund")
    print("  observability recorded per-node tokens, cost, and latency")
    print("  evals         scored the whole thing and caught a regression")
    print("\n  None of the four imports another. Each installs and passes its own")
    print("  suite alone -- see the `test` job in .github/workflows/ci.yml.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
