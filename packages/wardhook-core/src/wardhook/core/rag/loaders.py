"""Document loaders for PDF, Markdown, and plain text.

Loading is deliberately thin. A loader's only job is to turn a file into text
plus provenance; splitting is :mod:`~wardhook.core.rag.chunking`'s concern and
embedding is the store's. Keeping them separate means you can swap any one of
the three without touching the others.

PDFs are read page by page, and the page number is preserved in each chunk's
metadata so a citation can point at the page a passage actually came from
rather than at the document as a whole.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wardhook.core.rag.chunking import Chunk, chunk_text

__all__ = [
    "SUPPORTED_SUFFIXES",
    "Document",
    "DocumentLoadError",
    "load_directory",
    "load_document",
    "load_to_chunks",
]

SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".pdf", ".md", ".markdown", ".txt", ".text", ".rst"}
)
"""File extensions :func:`load_document` knows how to read."""

_TEXT_SUFFIXES = SUPPORTED_SUFFIXES - {".pdf"}


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded document, before it has been split into chunks.

    Attributes:
        text: The full extracted text.
        source: Identifier for the document, normally its file path.
        metadata: Loader-supplied details such as page count or file size.
    """

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoadError(RuntimeError):
    """Raised when a document exists but cannot be read or decoded."""


def _load_pdf(path: Path) -> Document:
    """Extract text from a PDF, one page at a time.

    Args:
        path: Path to the PDF.

    Returns:
        The loaded document, with page boundaries marked in the text and the
        page count recorded in metadata.

    Raises:
        DocumentLoadError: If the file cannot be parsed as a PDF.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a base dependency
        raise DocumentLoadError(
            "Reading PDFs requires 'pypdf'. Install it with: pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(str(path))
        pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    except Exception as exc:
        raise DocumentLoadError(f"Could not read PDF {path}: {exc}") from exc

    # The page marker survives chunking, so a chunk that lands mid-document
    # still carries a hint of which page it came from.
    body = "\n\n".join(
        f"[page {number}]\n{content}" for number, content in pages if content.strip()
    )
    return Document(
        text=body,
        source=str(path),
        metadata={"file_type": "pdf", "page_count": len(pages)},
    )


def _load_text(path: Path) -> Document:
    """Read a UTF-8 text or Markdown file.

    Args:
        path: Path to the file.

    Returns:
        The loaded document.

    Raises:
        DocumentLoadError: If the file is not valid UTF-8.
    """
    try:
        body = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"{path} is not valid UTF-8. Convert it, or pass the text directly to chunk_text()."
        ) from exc
    return Document(
        text=body,
        source=str(path),
        metadata={"file_type": path.suffix.lstrip(".") or "txt"},
    )


def load_document(path: str | Path) -> Document:
    """Load a single document from disk.

    Args:
        path: Path to a file with a suffix in :data:`SUPPORTED_SUFFIXES`.

    Returns:
        The loaded document.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DocumentLoadError: If the suffix is unsupported, the path is a
            directory, or the contents cannot be read.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"No such document: {resolved}")
    if resolved.is_dir():
        raise DocumentLoadError(f"{resolved} is a directory; use load_directory() instead.")

    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(resolved)
    if suffix in _TEXT_SUFFIXES:
        return _load_text(resolved)

    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    raise DocumentLoadError(
        f"Unsupported file type {suffix!r} for {resolved}. Supported: {supported}."
    )


def load_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
    suffixes: Iterable[str] | None = None,
    skip_errors: bool = True,
) -> list[Document]:
    """Load every supported document in a directory.

    Args:
        directory: Directory to scan.
        recursive: Whether to descend into subdirectories.
        suffixes: Restrict loading to these extensions. Defaults to all of
            :data:`SUPPORTED_SUFFIXES`.
        skip_errors: If ``True``, an unreadable file is skipped rather than
            aborting the whole ingest. A corpus of a few hundred documents
            usually contains at least one broken PDF, and failing the entire
            run for it is rarely what you want.

    Returns:
        Loaded documents, sorted by path for reproducible ordering.

    Raises:
        NotADirectoryError: If ``directory`` is not a directory.
        DocumentLoadError: If a file fails to load and ``skip_errors`` is
            ``False``.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    wanted = {s.lower() for s in (suffixes or SUPPORTED_SUFFIXES)}
    paths = sorted(p for p in (root.rglob("*") if recursive else root.glob("*")) if p.is_file())

    documents: list[Document] = []
    for path in paths:
        if path.suffix.lower() not in wanted:
            continue
        try:
            documents.append(load_document(path))
        except (DocumentLoadError, OSError):
            if not skip_errors:
                raise
    return documents


def load_to_chunks(
    documents: Sequence[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split loaded documents into chunks, preserving each document's metadata.

    Args:
        documents: Documents to split.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters repeated between consecutive chunks.

    Returns:
        All chunks across all documents, in document order.

    Example:
        >>> docs = [Document(text="a b c", source="memo.txt", metadata={"team": "risk"})]
        >>> chunks = load_to_chunks(docs, chunk_size=100, chunk_overlap=0)
        >>> chunks[0].source, chunks[0].metadata["team"]
        ('memo.txt', 'risk')
    """
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_text(
                document.text,
                document.source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=document.metadata,
            )
        )
    return chunks
