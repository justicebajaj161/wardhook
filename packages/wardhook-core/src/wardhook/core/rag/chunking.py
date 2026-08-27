"""Recursive character chunking with overlap.

Splitting is recursive: the text is broken on the most semantically meaningful
separator that produces small enough pieces, falling back through progressively
weaker separators only where a piece is still too large. In practice that means
paragraphs stay whole where they fit, and only genuinely oversized runs of text
get cut mid-sentence.

Each chunk keeps its source and its position within that source, which is what
makes structural citations possible downstream -- the retriever never has to
guess where a passage came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DEFAULT_SEPARATORS", "Chunk", "chunk_text"]

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")
"""Separators tried in order, from strongest semantic boundary to weakest.

The final empty string is a deliberate backstop: it splits on raw character
count and guarantees termination for text with no whitespace at all, such as a
long base64 blob or minified source.
"""


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span of text with the provenance needed to cite it.

    Attributes:
        text: The chunk's text content.
        source: Where it came from -- a file path, URL, or caller-supplied label.
        chunk_index: Zero-based position of this chunk within its source.
        metadata: Any extra fields carried over from the loader, such as the
            page number for a PDF.
    """

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """A stable identifier combining source and position."""
        return f"{self.source}#{self.chunk_index}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the chunk."""
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Rebuild a chunk from :meth:`to_dict` output.

        Args:
            data: A previously serialised chunk.

        Returns:
            The reconstructed chunk.
        """
        return cls(
            text=data["text"],
            source=data["source"],
            chunk_index=int(data["chunk_index"]),
            metadata=dict(data.get("metadata") or {}),
        )


def _split_recursive(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """Split ``text`` into pieces no larger than ``chunk_size`` where possible.

    Args:
        text: Text to split.
        separators: Remaining separators to try, strongest first.
        chunk_size: Maximum size of a returned piece.

    Returns:
        Pieces in document order. A piece may exceed ``chunk_size`` only if the
        separator list was exhausted without a viable split.
    """
    if len(text) <= chunk_size:
        return [text] if text else []
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *rest = separators
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    pieces: list[str] = []
    for index, part in enumerate(parts):
        # Re-attach the separator we split on so the text round-trips.
        restored = part if index == len(parts) - 1 else part + separator
        if not restored:
            continue
        if len(restored) <= chunk_size:
            pieces.append(restored)
        else:
            pieces.extend(_split_recursive(restored, tuple(rest), chunk_size))
    return pieces


def _merge(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack pieces up to ``chunk_size``, carrying overlap between chunks.

    Args:
        pieces: Splits produced by :func:`_split_recursive`.
        chunk_size: Target maximum chunk length.
        chunk_overlap: Characters of trailing context repeated at the start of
            the next chunk.

    Returns:
        The merged chunk texts.
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            # Carry the tail of the finished chunk into the next one so a
            # sentence spanning a boundary is retrievable from either side.
            current = current[-chunk_overlap:] if chunk_overlap else ""
        current += piece

    if current.strip():
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    r"""Split ``text`` into overlapping, source-attributed chunks.

    Args:
        text: The document text to split.
        source: Label recorded on every chunk, used later for citation.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters repeated between consecutive chunks. Overlap
            keeps a passage retrievable even when it straddles a boundary.
        separators: Split boundaries to try, strongest first.
        metadata: Extra fields copied onto every chunk produced.

    Returns:
        The chunks, in document order. Empty or whitespace-only input yields an
        empty list.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or if ``chunk_overlap``
            is negative or not smaller than ``chunk_size``. Overlap at or above
            the chunk size would make no forward progress.

    Example:
        >>> chunks = chunk_text(
        ...     "First para.\n\nSecond para.", "notes.md", chunk_size=20, chunk_overlap=0
        ... )
        >>> [c.text.strip() for c in chunks]
        ['First para.', 'Second para.']
        >>> chunks[1].source, chunks[1].chunk_index
        ('notes.md', 1)
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must not be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size "
            f"({chunk_size}); otherwise chunking cannot make forward progress."
        )

    if not text.strip():
        return []

    pieces = _split_recursive(text, tuple(separators), chunk_size)
    merged = _merge(pieces, chunk_size, chunk_overlap)
    base_metadata = dict(metadata or {})

    return [
        Chunk(
            text=body,
            source=source,
            chunk_index=index,
            metadata=dict(base_metadata),
        )
        for index, body in enumerate(merged)
    ]
