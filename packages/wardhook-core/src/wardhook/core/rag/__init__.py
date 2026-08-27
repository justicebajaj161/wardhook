"""Retrieval-augmented generation: ingest, chunk, embed, search, cite.

The four stages are deliberately separate modules with narrow interfaces, so
each can be replaced without disturbing the others. Swap the embeddings, swap
the store, or feed chunks in from your own pipeline entirely.

Example:
    >>> from wardhook.core.rag import InMemoryVectorStore, Retriever, chunk_text
    >>> store = InMemoryVectorStore()
    >>> _ = store.add(chunk_text("Storm cover carries a 500 excess.", "policy.md"))
    >>> Retriever(store, k=1).retrieve("storm excess")[0]["source"]
    'policy.md'
"""

from wardhook.core.rag.chunking import DEFAULT_SEPARATORS, Chunk, chunk_text
from wardhook.core.rag.embeddings import HashingEmbeddings, cosine_similarity, resolve_embeddings
from wardhook.core.rag.loaders import (
    SUPPORTED_SUFFIXES,
    Document,
    DocumentLoadError,
    load_directory,
    load_document,
    load_to_chunks,
)
from wardhook.core.rag.retriever import Retriever, format_citations, format_context
from wardhook.core.rag.store import InMemoryVectorStore, SearchResult

__all__ = [
    "DEFAULT_SEPARATORS",
    "SUPPORTED_SUFFIXES",
    "Chunk",
    "Document",
    "DocumentLoadError",
    "HashingEmbeddings",
    "InMemoryVectorStore",
    "Retriever",
    "SearchResult",
    "chunk_text",
    "cosine_similarity",
    "format_citations",
    "format_context",
    "load_directory",
    "load_document",
    "load_to_chunks",
    "resolve_embeddings",
]
