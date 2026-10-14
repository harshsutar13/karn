"""
ingestion.py

Document ingestion pipeline for the SIH26 RAG system.

Connects the existing independent modules into a single ingestion workflow:

    Document file
          |
          v
    load_document()               (documents/loader.py)
          |
          v
    chunk_text_with_metadata()    (documents/chunker.py)
          |
          v
    embed_texts()                 (embeddings/embedder.py)
          |
          v
    upsert_chunks() -> Qdrant     (rag/vector_store.py)

This module does NOT reimplement loading, chunking, or embedding.
It only orchestrates the existing modules.

Metadata captured per chunk:
    - document_id   (deterministic UUID-5 of filename)
    - filename
    - doc_type      (pdf / docx / txt)
    - chunk_index
    - char_start / char_end
    - department    (optional, caller-supplied)
    - extra         (any additional caller-supplied metadata)
"""

import hashlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from documents.loader import load_document
from documents.chunker import chunk_text_with_metadata, Chunk
from embeddings.embedder import embed_texts
from rag.vector_store import (
    get_client,
    create_collection_if_not_exists,
    upsert_chunks,
    ChunkRecord,
)
from config import QDRANT_COLLECTION

import re

def _strip_doc_header(text: str) -> str:
    """
    Cut everything before the first real section heading.

    Every P-101 doc starts with ~300 chars of identical metadata
    (title, doc number, revision, equipment, classification, status).
    That boilerplate dominates embeddings and makes every chunk look
    like every other chunk.

    Strategy: find the first line matching a numbered section
    ('1.', '1. Purpose', '# 1. ...') or a known top-level header,
    and return everything from that line onward.
    """
    if not text:
        return text

    # Patterns that mark the start of real content
    patterns = [
        re.compile(r'^\s*#\s*1\.'),              # markdown heading "1."
        re.compile(r'^\s*1\.\s+\w'),             # numbered "1. Purpose"
        re.compile(r'^\s*Subject\b', re.I),      # Approval note
        re.compile(r'^\s*Inspection Summary\b', re.I),
        re.compile(r'^\s*Historical Condition\b', re.I),
        re.compile(r'^\s*Objective\b', re.I),
        re.compile(r'^\s*Problem Statement\b', re.I),
        re.compile(r'^\s*Internal Maintenance Note\b', re.I),
        re.compile(r'^\s*Equipment Description\b', re.I),
    ]

    lines = text.splitlines()
    for i, line in enumerate(lines):
        for pat in patterns:
            if pat.match(line):
                return "\n".join(lines[i:]).strip()

    # Nothing matched — return original (defensive)
    return text

    return "\n".join(cleaned).strip()
logger = logging.getLogger(__name__)

# Stable namespace for deterministic document_id generation.
# document_id = UUID-5(namespace, filename) — same filename always
# maps to the same document_id, enabling deduplication on re-ingestion.
_DOC_ID_NAMESPACE = uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Summary returned after ingesting a single document."""
    filename: str
    document_id: str
    doc_type: str
    total_chars: int
    chunks_created: int
    chunks_upserted: int
    elapsed_load_s: float
    elapsed_chunk_s: float
    elapsed_embed_s: float
    elapsed_upsert_s: float

    @property
    def elapsed_total_s(self) -> float:
        return (
            self.elapsed_load_s
            + self.elapsed_chunk_s
            + self.elapsed_embed_s
            + self.elapsed_upsert_s
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_document_id(filename: str) -> str:
    """
    Generate a stable document_id from the filename using UUID-5.

    Same filename always produces the same UUID, which combined with
    chunk_index gives the deterministic point ID used in Qdrant.
    """
    return str(uuid.uuid5(_DOC_ID_NAMESPACE, filename))


def _doc_type_from_path(file_path: str) -> str:
    """Return 'pdf', 'docx', or 'txt' based on file extension."""
    suffix = Path(file_path).suffix.lower().lstrip(".")
    return suffix if suffix in {"pdf", "docx", "txt"} else "unknown"


# ---------------------------------------------------------------------------
# Core ingestion function
# ---------------------------------------------------------------------------

def ingest_document(
    file_path: str,
    collection_name: str = QDRANT_COLLECTION,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    department: str | None = None,
    extra_metadata: dict | None = None,
) -> IngestionResult:
    """
    Ingest a single document into the Qdrant vector store.

    The function is idempotent: re-ingesting the same file overwrites
    existing points (same deterministic document_id + chunk_index).

    Parameters
    ----------
    file_path:
        Absolute path to the document (.pdf, .docx, .txt).
    collection_name:
        Target Qdrant collection. Created if not exists.
    chunk_size:
        Character window size for chunking. Default: 1000.
    chunk_overlap:
        Character overlap between consecutive chunks. Default: 200.
    department:
        Optional department label stored in chunk metadata.
    extra_metadata:
        Optional dict of additional fields merged into every chunk's payload.

    Returns
    -------
    IngestionResult
        Timing and count information for the ingestion run.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    ValueError
        If the document produces no text or no chunks.
    ConnectionError
        If Qdrant is not reachable.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Document not found: {file_path!r}")

    filename = os.path.basename(file_path)
    document_id = _make_document_id(filename)
    doc_type = _doc_type_from_path(file_path)

    logger.info("Ingesting %r (id=%s, type=%s)", filename, document_id, doc_type)

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    text = load_document(file_path)
    text = _strip_doc_header(text)     # NEW
    elapsed_load = time.perf_counter() - t0

    if not text or not text.strip():
        raise ValueError(f"Document {filename!r} produced no extractable text.")

    logger.info("Loaded %d chars in %.2fs", len(text), elapsed_load)

    # ------------------------------------------------------------------
    # Step 2: Chunk
    # ------------------------------------------------------------------
    base_meta: dict = {"document_id": document_id, "filename": filename}
    if department:
        base_meta["department"] = department
    if extra_metadata:
        base_meta.update(extra_metadata)

    t0 = time.perf_counter()
    chunks: list[Chunk] = chunk_text_with_metadata(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_metadata=base_meta,
    )
    elapsed_chunk = time.perf_counter() - t0

    if not chunks:
        raise ValueError(f"Document {filename!r} produced no chunks.")

    logger.info("Created %d chunks in %.2fs", len(chunks), elapsed_chunk)

    # ------------------------------------------------------------------
    # Step 3: Embed
    # ------------------------------------------------------------------
    chunk_texts = [c.text for c in chunks]

    t0 = time.perf_counter()
    embeddings = embed_texts(chunk_texts)
    elapsed_embed = time.perf_counter() - t0

    logger.info(
        "Embedded %d chunks in %.2fs (%.2fs each)",
        len(embeddings), elapsed_embed, elapsed_embed / len(embeddings),
    )

    # ------------------------------------------------------------------
    # Step 4: Build ChunkRecords and upsert
    # ------------------------------------------------------------------
    records: list[ChunkRecord] = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append(
            ChunkRecord(
                document_id=document_id,
                filename=filename,
                doc_type=doc_type,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                embedding=embedding,
                extra_metadata=chunk.metadata,
            )
        )

    client = get_client()
    create_collection_if_not_exists(client, collection_name)

    t0 = time.perf_counter()
    upserted = upsert_chunks(client, records, collection_name=collection_name)
    elapsed_upsert = time.perf_counter() - t0

    logger.info("Upserted %d points in %.2fs", upserted, elapsed_upsert)

    return IngestionResult(
        filename=filename,
        document_id=document_id,
        doc_type=doc_type,
        total_chars=len(text),
        chunks_created=len(chunks),
        chunks_upserted=upserted,
        elapsed_load_s=elapsed_load,
        elapsed_chunk_s=elapsed_chunk,
        elapsed_embed_s=elapsed_embed,
        elapsed_upsert_s=elapsed_upsert,
    )
