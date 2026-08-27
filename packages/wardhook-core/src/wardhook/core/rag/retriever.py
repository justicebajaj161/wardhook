"""Retrieval that returns citable, source-attributed chunks.

The distinguishing choice here is that citations are **structural**. Retrieval
returns records carrying source, position and score, and those records travel
in agent state alongside the answer. A caller renders or verifies them directly;
nobody has to parse source names back out of the model's prose, and the model
cannot invent a citation for a document it was never shown.

The context block handed to the model numbers each passage and labels it with
its source, so when the model writes "[1]" that marker maps onto a real record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from wardhook.core.rag.store import SearchResult

__all__ = ["DEFAULT_RAG_INSTRUCTIONS", "Retriever", "format_citations", "format_context"]

DEFAULT_RAG_INSTRUCTIONS = """\
Answer using only the numbered context passages below. Cite the passages you \
rely on with their bracketed numbers, for example [1] or [2][3]. If the context \
does not contain the answer, say so plainly rather than guessing."""
"""Instructions appended to the system prompt when a retriever is attached."""


def format_citations(citations: Sequence[Mapping[str, Any]]) -> str:
    """Render citation records as a numbered, source-labelled context block.

    Args:
        citations: Citation dicts as produced by
            :meth:`~wardhook.core.rag.store.SearchResult.to_citation`.

    Returns:
        The formatted block, or an empty string if ``citations`` is empty.

    Example:
        >>> print(
        ...     format_citations(
        ...         [{"source": "policy.md", "chunk_index": 3, "text": "Excess is 500."}]
        ...     )
        ... )
        [1] policy.md (chunk 3)
        Excess is 500.
    """
    if not citations:
        return ""
    return "\n\n".join(
        f"[{position}] {c.get('source', 'unknown')} (chunk {c.get('chunk_index', 0)})\n"
        f"{c.get('text', '')}"
        for position, c in enumerate(citations, start=1)
    )


def format_context(results: list[SearchResult]) -> str:
    """Render search results as a numbered, source-labelled context block.

    Args:
        results: Retrieved results, most relevant first.

    Returns:
        The formatted block, or an empty string if ``results`` is empty.

    Example:
        >>> from wardhook.core.rag.chunking import Chunk
        >>> hit = SearchResult(Chunk("Excess is 500.", "policy.md", 3), 0.82)
        >>> print(format_context([hit]))
        [1] policy.md (chunk 3)
        Excess is 500.
    """
    return format_citations([r.to_citation() for r in results])


class Retriever:
    """Fetches relevant context for a query and keeps it citable.

    Args:
        store: Any object with a ``search(query, k)`` method, such as
            :class:`~wardhook.core.rag.store.InMemoryVectorStore`.
        k: Maximum passages to return per query.
        score_threshold: Drop results scoring below this. The default of ``0.0``
            discards passages with no meaningful overlap while keeping anything
            weakly related, which suits the hashing fallback. Raise it when
            using a neural embedding model, where near-zero scores are rare and
            a higher floor is what actually filters noise.

    Raises:
        TypeError: If ``store`` has no ``search`` method.
        ValueError: If ``k`` is not positive.

    Example:
        >>> from wardhook.core.rag.chunking import chunk_text
        >>> from wardhook.core.rag.store import InMemoryVectorStore
        >>> store = InMemoryVectorStore()
        >>> _ = store.add(chunk_text("The deductible for storm damage is 500.", "policy.md"))
        >>> retriever = Retriever(store, k=1)
        >>> citations = retriever.retrieve("what is the storm deductible?")
        >>> citations[0]["source"]
        'policy.md'
    """

    def __init__(self, store: Any, *, k: int = 4, score_threshold: float = 0.0) -> None:
        """Initialise the retriever. See class docstring for arguments."""
        if not hasattr(store, "search"):
            raise TypeError(
                f"{type(store).__name__} is not a vector store; it has no search(query, k) method."
            )
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.store = store
        self.k = k
        self.score_threshold = score_threshold

    def search(self, query: str) -> list[SearchResult]:
        """Return raw search results above the score threshold.

        Args:
            query: The query text.

        Returns:
            Results, most relevant first.
        """
        results = self.store.search(query, k=self.k)
        return [r for r in results if r.score >= self.score_threshold]

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return citation records for a query.

        Args:
            query: The query text.

        Returns:
            Citation dicts matching :class:`~wardhook.core.state.Citation`,
            most relevant first. Empty when nothing clears the threshold.
        """
        return [result.to_citation() for result in self.search(query)]

    def context_for(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        """Return both the model-facing context block and its citation records.

        Producing both from a single search keeps the numbering in the block
        aligned with the order of the citation list, so ``[2]`` in the model's
        answer is always ``citations[1]``.

        Args:
            query: The query text.

        Returns:
            A ``(context_block, citations)`` pair. Both are empty when nothing
            relevant was found.
        """
        results = self.search(query)
        return format_context(results), [r.to_citation() for r in results]

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Retriever(k={self.k}, score_threshold={self.score_threshold})"
