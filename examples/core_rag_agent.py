"""Example: a retrieval-augmented agent with real source citations.

Runs fully offline against a fake model, so no API key is needed:

    python examples/core_rag_agent.py

To run it against a real provider instead:

    pip install "wardhook-core[anthropic]"
    export ANTHROPIC_API_KEY=...
    python examples/core_rag_agent.py --live

The point of this example is the `citations` list. Retrieval returns structured
records carrying source, chunk position and similarity score, so a caller can
render or verify them directly -- the model cannot invent a citation for a
document it was never shown.
"""

from __future__ import annotations

import sys

from wardhook.core import AgentGraph, InMemoryVectorStore, Retriever, chunk_text

POLICY = """\
Section 4 -- Storm and flood damage.
Storm damage claims carry a 500 excess. Flood damage carries a 1000 excess and
requires a loss adjuster to attend before settlement. Cover applies only where
wind speeds exceeded 55mph, as recorded by the nearest Met Office station.

Section 5 -- Escape of water.
Escape of water carries a 250 excess. Trace and access costs are covered up to
5000. Damage caused by gradual seepage is excluded.
"""

HANDBOOK = """\
Our contact centre is open 9am to 5pm on weekdays and 9am to 1pm on Saturdays.
Claims may be registered online at any time. A claims handler will contact you
within two working days of registration.
"""


def build_store() -> InMemoryVectorStore:
    """Index the sample corpus.

    Returns:
        A store holding both documents, chunked and embedded. The default
        embeddings are a dependency-free hashing vectoriser, which is why this
        runs with no API key and no model download.
    """
    store = InMemoryVectorStore()
    store.add(chunk_text(POLICY, "policy-wording.md", chunk_size=240, chunk_overlap=40))
    store.add(chunk_text(HANDBOOK, "customer-handbook.md", chunk_size=240, chunk_overlap=40))
    return store


def build_model(live: bool):
    """Return the chat model to drive the agent.

    Args:
        live: Whether to use a real provider instead of a scripted fake.

    Returns:
        A model instance. The fake replays one scripted answer, which is enough
        to demonstrate the retrieval and citation path without a network call.
    """
    if live:
        return "claude-opus-5"

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=(
                        "Storm damage carries a 500 excess, and cover applies only where "
                        "wind speeds exceeded 55mph [1]."
                    )
                )
            ]
        )
    )


def main() -> int:
    """Run the example.

    Returns:
        A process exit code.
    """
    live = "--live" in sys.argv
    store = build_store()
    print(f"Indexed {len(store)} chunks from 2 documents.\n")

    agent = AgentGraph(
        model=build_model(live),
        retriever=Retriever(store, k=2),
        system_prompt="You are a claims assistant for an insurance carrier.",
    )

    question = "What excess applies to storm damage, and are there conditions?"
    print(f"Q: {question}\n")

    result = agent.invoke(question)

    print(f"A: {result['output']}\n")
    print("Sources actually retrieved and shown to the model:")
    for position, citation in enumerate(result["citations"], start=1):
        preview = citation["text"].strip().replace("\n", " ")[:66]
        print(
            f"  [{position}] {citation['source']} (chunk {citation['chunk_index']}, "
            f"score {citation['score']:.3f})"
        )
        print(f"      {preview}...")

    print(f"\nrun_id: {result['run_id']}   model: {result['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
