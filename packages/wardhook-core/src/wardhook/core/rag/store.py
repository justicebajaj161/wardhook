"""Vector stores: an in-memory NumPy implementation and the protocol it satisfies.

:class:`InMemoryVectorStore` is the default. It holds every vector in a single
NumPy matrix and answers a query with one matrix-vector product, which is fast
and entirely adequate up to roughly a hundred thousand chunks -- comfortably
past the point where most agent knowledge bases sit.

Beyond that, swap in a purpose-built store. Anything satisfying
:class:`~wardhook.core.protocols.VectorStoreProtocol` works, so the retriever
above it does not change.

Persistence writes two files: an ``.npz`` for the matrix and a ``.json``
sidecar for the chunks. Keeping text out of the binary means you can inspect,
diff, or grep an index without loading NumPy.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wardhook.core.rag.chunking import Chunk
from wardhook.core.rag.embeddings import resolve_embeddings

__all__ = ["InMemoryVectorStore", "SearchResult"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A chunk returned by a similarity search, with its score.

    Attributes:
        chunk: The matched chunk, carrying its own source and position.
        score: Cosine similarity to the query, in ``[-1.0, 1.0]``. Higher is
            more similar.
    """

    chunk: Chunk
    score: float

    def to_citation(self) -> dict[str, Any]:
        """Return this result as a citation record.

        Returns:
            A dict matching :class:`~wardhook.core.state.Citation`, suitable for
            placing in agent state or returning to an API caller.
        """
        return {
            "source": self.chunk.source,
            "chunk_index": self.chunk.chunk_index,
            "score": round(self.score, 6),
            "text": self.chunk.text,
            "metadata": dict(self.chunk.metadata),
        }


class InMemoryVectorStore:
    """A NumPy-backed vector store with optional on-disk persistence.

    Args:
        embeddings: Any object with ``embed_documents`` and ``embed_query``.
            Defaults to :class:`~wardhook.core.rag.embeddings.HashingEmbeddings`,
            which needs no API key.

    Example:
        >>> from wardhook.core.rag.chunking import chunk_text
        >>> store = InMemoryVectorStore()
        >>> _ = store.add(chunk_text("Flood claims require a loss adjuster.", "policy.md"))
        >>> _ = store.add(chunk_text("Office opening hours are 9 to 5.", "hours.md"))
        >>> hits = store.search("who assesses a flood claim?", k=1)
        >>> hits[0].chunk.source
        'policy.md'
    """

    def __init__(self, embeddings: Any = None) -> None:
        """Initialise an empty store."""
        self.embeddings = resolve_embeddings(embeddings)
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"InMemoryVectorStore(chunks={len(self._chunks)}, embeddings={self.embeddings!r})"

    @property
    def chunks(self) -> list[Chunk]:
        """Return a copy of the indexed chunks, in insertion order."""
        return list(self._chunks)

    def add(self, chunks: Sequence[Chunk]) -> list[str]:
        """Embed and index chunks.

        Args:
            chunks: Chunks to add. An empty sequence is a no-op.

        Returns:
            The ids of the added chunks, in input order.

        Raises:
            ValueError: If the embeddings object returns vectors whose width
                does not match the vectors already stored. That means the store
                was built with different embeddings, and mixing the two would
                produce silently meaningless similarity scores.
        """
        if not chunks:
            return []

        vectors = np.asarray(
            self.embeddings.embed_documents([c.text for c in chunks]),
            dtype=np.float32,
        )
        if vectors.ndim != 2:
            raise ValueError(
                f"embed_documents must return a 2-D sequence of vectors, "
                f"got an array with {vectors.ndim} dimension(s)."
            )

        if self._matrix is None:
            self._matrix = vectors
        else:
            if vectors.shape[1] != self._matrix.shape[1]:
                raise ValueError(
                    f"Embedding width mismatch: store holds {self._matrix.shape[1]}-dim "
                    f"vectors but received {vectors.shape[1]}-dim. This store was built "
                    f"with a different embeddings model; rebuild it rather than mixing."
                )
            self._matrix = np.vstack([self._matrix, vectors])

        self._chunks.extend(chunks)
        return [c.id for c in chunks]

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        """Return the ``k`` chunks most similar to ``query``.

        Args:
            query: Natural-language query.
            k: Maximum number of results. Values above the store size simply
                return everything.

        Returns:
            Results sorted by descending similarity. Empty if the store is
            empty or ``k`` is not positive.
        """
        if self._matrix is None or not self._chunks or k <= 0:
            return []

        query_vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        norms = np.linalg.norm(self._matrix, axis=1)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            return []

        # Guard against zero-magnitude rows (a chunk with no recognisable
        # tokens) rather than emitting a divide-by-zero warning and NaNs.
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        scores = (self._matrix @ query_vector) / (safe_norms * query_norm)
        scores = np.where(norms == 0.0, -1.0, scores)

        top_k = min(k, len(self._chunks))
        # argpartition finds the top-k in linear time; only those k are sorted.
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = candidates[np.argsort(-scores[candidates])]
        return [
            SearchResult(chunk=self._chunks[int(i)], score=float(scores[int(i)])) for i in ordered
        ]

    def save(self, path: str | Path) -> Path:
        """Persist the store to disk.

        Writes ``<path>.npz`` for the vectors and ``<path>.json`` for the
        chunks, creating parent directories as needed.

        Args:
            path: Destination path, with or without a suffix.

        Returns:
            The path to the JSON sidecar.

        Raises:
            ValueError: If the store is empty; there is nothing to save.
        """
        if self._matrix is None:
            raise ValueError("Cannot save an empty store; add chunks first.")

        base = Path(path).expanduser().with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base.with_suffix(".npz"), vectors=self._matrix)
        sidecar = base.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "version": 1,
                    "dim": int(self._matrix.shape[1]),
                    "chunks": [c.to_dict() for c in self._chunks],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return sidecar

    @classmethod
    def load(cls, path: str | Path, embeddings: Any = None) -> InMemoryVectorStore:
        """Load a store previously written by :meth:`save`.

        Args:
            path: The path passed to :meth:`save`, with or without a suffix.
            embeddings: Embeddings to use for future queries. This **must** match
                whatever built the index; the stored vectors are meaningless
                under a different model.

        Returns:
            The restored store.

        Raises:
            FileNotFoundError: If either file is missing.
            ValueError: If the sidecar and vector file disagree on chunk count,
                which means one of the pair is stale.
        """
        base = Path(path).expanduser().with_suffix("")
        sidecar, vector_file = base.with_suffix(".json"), base.with_suffix(".npz")
        for required in (sidecar, vector_file):
            if not required.exists():
                raise FileNotFoundError(f"Missing store file: {required}")

        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in payload["chunks"]]
        with np.load(vector_file) as data:
            matrix = data["vectors"]

        if len(chunks) != matrix.shape[0]:
            raise ValueError(
                f"Corrupt store: {sidecar.name} holds {len(chunks)} chunks but "
                f"{vector_file.name} holds {matrix.shape[0]} vectors. The two "
                f"files are out of sync; rebuild the index."
            )

        store = cls(embeddings=embeddings)
        store._chunks = chunks
        store._matrix = matrix
        return store
