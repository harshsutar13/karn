"""
retriever.py

Semantic retrieval layer for the SIH26 RAG pipeline.

Responsibility:
    Natural-language query
        -> embed query
        -> Qdrant semantic search
        -> ranked list of relevant chunks

NOT responsible for:
    - Generating an LLM answer  (see rag/pipeline.py)
    - Embedding document chunks (see embeddings/embedder.py)
    - Storing vectors           (see rag/vector_store.py)
    - Document loading/chunking (see documents/)
"""

import logging
import sys
import os
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings.embedder import embed_text
from rag.vector_store import (
    get_client,
    semantic_search,
    SearchResult,
    VECTOR_SIZE,
)
from config import QDRANT_COLLECTION


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with all the information the RAG pipeline
    needs to construct an LLM prompt and cite sources.
    """
    text: str
    score: float
    filename: str
    document_id: str
    doc_type: str
    chunk_index: int
    char_start: int
    char_end: int
    point_id: str
    extra: dict[str, Any]

    def as_context_string(self) -> str:
        """
        Format the chunk as a labelled context block for inclusion in
        an LLM prompt.  Example output:

            [Source: operations_manual.pdf | chunk 3 | score 0.87]
            The crude oil distillation unit operates at...
        """
        return (
            f"[Source: {self.filename} | chunk {self.chunk_index} "
            f"| score {self.score:.3f}]\n{self.text}"
        )


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 5,
    collection_name: str = QDRANT_COLLECTION,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most semantically relevant chunks for a natural-language query.

    Pipeline:
        query string
            -> embed_text()     (nomic-embed-text, 768-dim)
            -> semantic_search()  (Qdrant cosine similarity)
            -> list[RetrievedChunk]

    Parameters
    ----------
    query:
        The user's natural-language question or search query.
        Must be non-empty.
    top_k:
        Maximum number of chunks to return. Default: 5.
    collection_name:
        Qdrant collection to search. Default from config.
    score_threshold:
        Optional minimum cosine similarity (0.0–1.0).
        Chunks below this score are excluded.
        None = no filtering (return all top_k regardless of score).

    Returns
    -------
    list[RetrievedChunk]
        Ordered from most to least relevant. May be empty if the
        collection is empty or no chunks meet the threshold.

    Raises
    ------
    ValueError
        If query is empty or top_k < 1.
    ConnectionError
        If Qdrant is not reachable.
    """
    if not query or not query.strip():
        raise ValueError("retrieve: query must be a non-empty string.")

    if top_k < 1:
        raise ValueError(f"retrieve: top_k must be >= 1, got {top_k}.")

    logger.debug("Retrieving top %d chunks for query: %r", top_k, query[:80])

    # Step 1: embed the query
    query_vector = embed_text(query)

    if len(query_vector) != VECTOR_SIZE:
        raise ValueError(
            f"Query embedding has dimension {len(query_vector)}, "
            f"expected {VECTOR_SIZE}. Check embedding model configuration."
        )

    # Step 2: vector search
    client = get_client()
    raw_results: list[SearchResult] = semantic_search(
        client=client,
        query_vector=query_vector,
        top_k=top_k,
        collection_name=collection_name,
        score_threshold=score_threshold,
    )

    # Step 3: map to RetrievedChunk
    chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            text=r.text,
            score=r.score,
            filename=r.filename,
            document_id=r.document_id,
            doc_type=r.doc_type,
            chunk_index=r.chunk_index,
            char_start=r.char_start,
            char_end=r.char_end,
            point_id=r.point_id,
            extra=r.extra,
        )
        for r in raw_results
    ]

    logger.info(
        "Retrieved %d chunks for query %r (collection=%r)",
        len(chunks), query[:60], collection_name,
    )
    return chunks


def retrieve_as_context(
    query: str,
    top_k: int = 5,
    collection_name: str = QDRANT_COLLECTION,
    score_threshold: float | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """
    Convenience wrapper: retrieve chunks AND format them as a single
    context string ready for LLM prompt injection.

    Returns
    -------
    tuple[list[RetrievedChunk], str]
        (chunks, context_string)

        context_string is empty if no relevant chunks are found.
    """
    chunks = retrieve(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
        score_threshold=score_threshold,
    )

    if not chunks:
        return chunks, ""

    context = "\n\n".join(c.as_context_string() for c in chunks)
    return chunks, context
