"""
chunker.py

Splits extracted document text into overlapping chunks suitable
for embedding and vector-store ingestion.

Strategy: character-level sliding window with overlap.
This is intentionally simple and dependency-free.
A sentence-aware or token-aware splitter can be added later
if retrieval quality demands it.
"""

import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single piece of split document text with positional metadata."""

    text: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split *text* into overlapping character windows.

    Parameters
    ----------
    text:
        The full extracted document text.
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of characters shared between consecutive chunks.
        Must be strictly less than chunk_size.

    Returns
    -------
    list[str]
        Ordered list of text chunks. Empty input returns [].
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    step = chunk_size - chunk_overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start += step

    logger.debug(
        "chunk_text produced %d chunks "
        "(size=%d, overlap=%d, total_chars=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
        text_length,
    )

    return chunks


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    base_metadata: dict | None = None,
) -> list[Chunk]:
    """
    Same sliding-window split as chunk_text(), but returns rich
    Chunk objects that carry positional info and optional metadata.

    base_metadata is merged into every chunk (e.g. filename, doc_id).

    This is the function RAG/embedding layers should call so that
    Qdrant payloads can store meaningful provenance.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    text = text.strip()

    if not text:
        return []

    meta = base_metadata or {}
    chunks: list[Chunk] = []
    start = 0
    text_length = len(text)
    step = chunk_size - chunk_overlap
    index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunks.append(
                Chunk(
                    text=chunk_text_content,
                    chunk_index=index,
                    char_start=start,
                    char_end=end,
                    metadata=dict(meta),
                )
            )
            index += 1

        if end == text_length:
            break

        start += step

    logger.debug(
        "chunk_text_with_metadata produced %d chunks "
        "(size=%d, overlap=%d, total_chars=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
        text_length,
    )

    return chunks
