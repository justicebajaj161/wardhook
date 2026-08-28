"""Wardhook core: a LangGraph agent runtime with RAG and a one-command server.

This package is the runtime the other three Wardhook packages plug into, and it
works entirely on its own. Nothing here imports ``wardhook.guardrails`` or
``wardhook.observability``; they attach through the structural contracts in
:mod:`wardhook.core.protocols` only if you install and pass them.

Example:
    A retrieval-augmented agent with source citations::

        from wardhook.core import AgentGraph, InMemoryVectorStore, Retriever, chunk_text

        store = InMemoryVectorStore()
        store.add(chunk_text(open("policy.md").read(), "policy.md"))

        agent = AgentGraph(model="claude-opus-5", retriever=Retriever(store))
        result = agent.invoke("What excess applies to storm damage?")

        print(result["output"])
        for citation in result["citations"]:
            print(f"  {citation['source']} chunk {citation['chunk_index']}")
"""

from wardhook.core.agent import AgentGraph, MissingIntegrationError
from wardhook.core.models import DEFAULT_MODEL, ModelResolutionError, resolve_model
from wardhook.core.protocols import (
    EmbeddingsProtocol,
    GuardrailAction,
    GuardrailDecision,
    GuardrailProtocol,
    RetrieverProtocol,
    TelemetryProtocol,
    VectorStoreProtocol,
)
from wardhook.core.rag.chunking import Chunk, chunk_text
from wardhook.core.rag.embeddings import HashingEmbeddings
from wardhook.core.rag.loaders import Document, load_directory, load_document, load_to_chunks
from wardhook.core.rag.retriever import Retriever, format_citations
from wardhook.core.rag.store import InMemoryVectorStore, SearchResult
from wardhook.core.state import AgentState, Citation, Principal
from wardhook.core.tools import ToolRegistrationError, normalize_tools

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_MODEL",
    "AgentGraph",
    "AgentState",
    "Chunk",
    "Citation",
    "Document",
    "EmbeddingsProtocol",
    "GuardrailAction",
    "GuardrailDecision",
    "GuardrailProtocol",
    "HashingEmbeddings",
    "InMemoryVectorStore",
    "MissingIntegrationError",
    "ModelResolutionError",
    "Principal",
    "Retriever",
    "RetrieverProtocol",
    "SearchResult",
    "TelemetryProtocol",
    "ToolRegistrationError",
    "VectorStoreProtocol",
    "__version__",
    "chunk_text",
    "format_citations",
    "load_directory",
    "load_document",
    "load_to_chunks",
    "normalize_tools",
    "resolve_model",
]
