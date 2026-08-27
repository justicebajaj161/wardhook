"""Embedding backends, including a deterministic offline fallback.

Wardhook does not ship a neural embedding model. Anything satisfying
:class:`~wardhook.core.protocols.EmbeddingsProtocol` works, which includes every
LangChain embeddings class, so you can plug in whichever provider you already
pay for.

What ships instead is :class:`HashingEmbeddings` -- a classical hashed
bag-of-words vectoriser with no model weights, no network calls, and no API
key. It exists so the RAG pipeline is runnable and testable the moment the
package is installed. It is a real technique (the hashing trick) with real
limits: it matches on shared vocabulary, not meaning, so it will not connect
"car" to "automobile". Use it for tests, demos, and CI; use a proper embedding
model in production.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

__all__ = ["HashingEmbeddings", "cosine_similarity", "resolve_embeddings"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbeddings:
    """Deterministic, dependency-free embeddings via the hashing trick.

    Each token is hashed to a bucket and a sign, contributing sublinearly
    weighted counts to that bucket. The resulting vector is L2-normalised, so
    a dot product between two vectors is their cosine similarity.

    Hashing uses BLAKE2b rather than Python's built-in :func:`hash`, because
    ``hash()`` on strings is salted per process. A store embedded in one process
    must remain searchable in the next, so the hash has to be stable across
    interpreter restarts.

    Attributes:
        dim: Dimensionality of the produced vectors.

    Example:
        >>> embeddings = HashingEmbeddings(dim=64)
        >>> a = embeddings.embed_query("annual insurance premium")
        >>> b = embeddings.embed_query("premium insurance annual")
        >>> round(cosine_similarity(a, b), 6)  # word order is not encoded
        1.0
        >>> c = embeddings.embed_query("chocolate cake recipe")
        >>> cosine_similarity(a, c) < 0.2
        True
    """

    def __init__(self, dim: int = 256) -> None:
        """Initialise the vectoriser.

        Args:
            dim: Number of buckets. Larger values reduce hash collisions at the
                cost of memory; 256 is ample for demos and test corpora.

        Raises:
            ValueError: If ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        """Map a token to a bucket index and a sign.

        Args:
            token: A normalised token.

        Returns:
            The bucket index and either ``+1.0`` or ``-1.0``. The signed variant
            of the hashing trick keeps collisions from systematically inflating
            similarity, since colliding tokens cancel about half the time.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _vectorise(self, text: str) -> list[float]:
        """Turn text into a normalised vector.

        Args:
            text: Input text.

        Returns:
            An L2-normalised vector of length :attr:`dim`. Text with no
            recognisable tokens yields the zero vector.
        """
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            index, sign = self._bucket(token)
            vector[index] += sign
        # Sublinear scaling, the same intuition as log-scaled term frequency:
        # a term appearing ten times is more important than one appearing once,
        # but not ten times more important.
        vector = [math.copysign(math.log1p(abs(v)), v) for v in vector]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: Document texts.

        Returns:
            One vector per text, in input order.
        """
        return [self._vectorise(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Args:
            text: Query text.

        Returns:
            The query vector.
        """
        return self._vectorise(text)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"HashingEmbeddings(dim={self.dim})"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Similarity in ``[-1.0, 1.0]``. Returns ``0.0`` if either vector has zero
        magnitude, which is the sensible reading of "no shared direction" and
        avoids a division by zero.

    Raises:
        ValueError: If the vectors have different lengths.

    Example:
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
    """
    if len(a) != len(b):
        raise ValueError(f"Vector lengths differ: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def resolve_embeddings(embeddings: Any = None) -> Any:
    """Return a usable embeddings object, defaulting to the offline fallback.

    Args:
        embeddings: An object implementing ``embed_documents`` and
            ``embed_query``, or ``None`` to use :class:`HashingEmbeddings`.

    Returns:
        The embeddings object to use.

    Raises:
        TypeError: If ``embeddings`` is missing either required method. Failing
            here produces a far clearer error than a missing attribute surfacing
            mid-ingest.
    """
    if embeddings is None:
        return HashingEmbeddings()
    missing = [m for m in ("embed_documents", "embed_query") if not hasattr(embeddings, m)]
    if missing:
        raise TypeError(
            f"{type(embeddings).__name__} is not a valid embeddings object; "
            f"missing method(s): {', '.join(missing)}."
        )
    return embeddings
