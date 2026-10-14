"""
vector_store.py

Qdrant vector-store abstraction for the SIH26 AI Workbench.

Responsibilities:
    - Connect to the local Qdrant instance
    - Create / verify collections
    - Upsert chunks + embeddings with metadata
    - Semantic search by query vector

NOT responsible for:
    - Generating embeddings  (see embeddings/embedder.py)
    - Loading documents      (see documents/loader.py)
    - Chunking text          (see documents/chunker.py)
    - LLM generation         (see models/llm.py)

Point ID strategy:
    Deterministic UUID-5 derived from  document_id + str(chunk_index).
    UUID-5 uses a fixed namespace so identical (doc_id, chunk_index)
    pairs always hash to the same UUID. This means repeated ingestion
    of the same document overwrites (upserts) the existing point rather
    than creating a duplicate.
"""

import logging
import uuid
import sys
import os
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# Support running from backend/ or from backend/rag/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION


logger = logging.getLogger(__name__)

# Embedding dimension confirmed from nomic-embed-text live test.
# This is the only place this constant lives; do not scatter it.
VECTOR_SIZE = 768
DISTANCE = qmodels.Distance.COSINE

# Stable UUID namespace for deterministic point ID generation.
# Using a project-specific namespace avoids collisions with other systems.
_POINT_ID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_point_id(document_id: str, chunk_index: int) -> str:
    """
    Generate a deterministic UUID-5 for a (document_id, chunk_index) pair.

    The same inputs always produce the same UUID, ensuring that upserting
    the same document+chunk combination overwrites the existing Qdrant point
    instead of creating a duplicate.
    """
    stable_key = f"{document_id}::{chunk_index}"
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, stable_key))


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_client() -> QdrantClient:
    """
    Return a connected QdrantClient.

    Raises
    ------
    ConnectionError
        If Qdrant is not reachable at the configured host/port.
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        # Probe with a lightweight call to verify connectivity
        client.get_collections()
        logger.debug("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
        return client
    except Exception as exc:
        raise ConnectionError(
            f"Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. "
            f"Ensure Qdrant is running. Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Return True if the named collection already exists in Qdrant."""
    existing = [c.name for c in client.get_collections().collections]
    return collection_name in existing


def create_collection_if_not_exists(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION,
    vector_size: int = VECTOR_SIZE,
    distance: qmodels.Distance = DISTANCE,
) -> bool:
    """
    Create a Qdrant collection if it does not already exist.

    Parameters
    ----------
    client:
        Connected QdrantClient.
    collection_name:
        Name of the collection to create.
    vector_size:
        Dimensionality of stored vectors. Default: 768 (nomic-embed-text).
    distance:
        Distance metric. Default: Cosine.

    Returns
    -------
    bool
        True  if the collection was newly created.
        False if it already existed (no-op).
    """
    if collection_exists(client, collection_name):
        logger.info("Collection %r already exists — skipping creation.", collection_name)
        return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )
    logger.info(
        "Created collection %r (size=%d, distance=%s).",
        collection_name,
        vector_size,
        distance,
    )
    return True


def get_collection_info(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION,
) -> dict[str, Any]:
    """
    Return a summary dict of collection metadata.

    Useful for diagnostics and test assertions.
    """
    if not collection_exists(client, collection_name):
        raise ValueError(f"Collection {collection_name!r} does not exist.")

    info = client.get_collection(collection_name)
    # Note: vectors_count was removed in newer qdrant-client versions.
    # points_count is the reliable field for total stored points.
    return {
        "name": collection_name,
        "status": str(info.status),
        "points_count": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "distance": str(info.config.params.vectors.distance),
    }


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """
    Everything needed to store one chunk in Qdrant.

    Fields
    ------
    document_id:
        Stable identifier for the source document (e.g. filename hash
        or a UUID assigned at ingestion time). Used for deduplication.
    filename:
        Human-readable source filename.
    doc_type:
        e.g. "pdf", "docx", "txt".
    chunk_index:
        Zero-based position of this chunk within its document.
    text:
        The raw chunk text. Stored in the Qdrant payload so the RAG
        layer can retrieve it without a separate database lookup.
    char_start:
        Character offset of the chunk start in the original text.
    char_end:
        Character offset of the chunk end in the original text.
    embedding:
        The 768-dim float vector for this chunk.
    extra_metadata:
        Optional dict of additional fields (department, page, etc.)
        that are merged into the Qdrant payload.
    """
    document_id: str
    filename: str
    doc_type: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    embedding: list[float]
    extra_metadata: dict[str, Any] | None = None


def upsert_chunks(
    client: QdrantClient,
    records: list[ChunkRecord],
    collection_name: str = QDRANT_COLLECTION,
    expected_dim: int = VECTOR_SIZE,
) -> int:
    """
    Upsert a list of ChunkRecords into the Qdrant collection.

    Points use deterministic UUID-5 IDs so repeated ingestion of the
    same document overwrites existing points rather than duplicating them.

    Parameters
    ----------
    client:
        Connected QdrantClient.
    records:
        List of ChunkRecord objects with embeddings already populated.
    collection_name:
        Target collection (must already exist).
    expected_dim:
        Expected vector dimension. Any mismatch raises ValueError.

    Returns
    -------
    int
        Number of points upserted.

    Raises
    ------
    ValueError
        If records is empty, any embedding has wrong dimension, or
        the collection does not exist.
    """
    if not records:
        raise ValueError("upsert_chunks: records list is empty.")

    if not collection_exists(client, collection_name):
        raise ValueError(
            f"Collection {collection_name!r} does not exist. "
            "Call create_collection_if_not_exists() first."
        )

    points: list[qmodels.PointStruct] = []

    for i, rec in enumerate(records):
        # Validate embedding dimension
        if len(rec.embedding) != expected_dim:
            raise ValueError(
                f"records[{i}] has embedding dimension {len(rec.embedding)}, "
                f"expected {expected_dim}. Check your embedding model."
            )

        point_id = _make_point_id(rec.document_id, rec.chunk_index)

        payload: dict[str, Any] = {
            "document_id": rec.document_id,
            "filename": rec.filename,
            "doc_type": rec.doc_type,
            "chunk_index": rec.chunk_index,
            "text": rec.text,
            "char_start": rec.char_start,
            "char_end": rec.char_end,
        }

        if rec.extra_metadata:
            # extra_metadata values must be Qdrant-compatible scalars/lists
            payload.update(rec.extra_metadata)

        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=rec.embedding,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,  # Synchronous: ensures points are indexed before returning
    )

    logger.info(
        "Upserted %d points into collection %r.",
        len(points),
        collection_name,
    )
    return len(points)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single result returned by semantic_search()."""
    point_id: str
    score: float
    text: str
    document_id: str
    filename: str
    doc_type: str
    chunk_index: int
    char_start: int
    char_end: int
    extra: dict[str, Any]  # Any payload fields beyond the standard ones


_STANDARD_PAYLOAD_KEYS = {
    "document_id", "filename", "doc_type",
    "chunk_index", "text", "char_start", "char_end",
}


def semantic_search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    collection_name: str = QDRANT_COLLECTION,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """
    Perform semantic (vector) search against the Qdrant collection.

    Parameters
    ----------
    client:
        Connected QdrantClient.
    query_vector:
        768-dim float list (output of embed_text() on the user query).
    top_k:
        Number of nearest neighbours to return. Default: 5.
    collection_name:
        Collection to search. Default: QDRANT_COLLECTION.
    score_threshold:
        Optional minimum cosine similarity score (0-1). Results below
        this threshold are excluded. None means no filtering.

    Returns
    -------
    list[SearchResult]
        Ordered from most to least similar.

    Raises
    ------
    ValueError
        If query_vector is empty or top_k < 1.
    """
    if not query_vector:
        raise ValueError("semantic_search: query_vector must be non-empty.")

    if top_k < 1:
        raise ValueError(f"semantic_search: top_k must be >= 1, got {top_k}.")

    if not collection_exists(client, collection_name):
        raise ValueError(
            f"Collection {collection_name!r} does not exist."
        )

    # query_points() is the current API in qdrant-client >= 1.9.
    # It returns a QueryResponse; .points holds the ScoredPoint list.
    kwargs: dict = dict(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold

    response = client.query_points(**kwargs)
    hits = response.points

    results: list[SearchResult] = []
    for hit in hits:
        payload = hit.payload or {}
        extra = {k: v for k, v in payload.items() if k not in _STANDARD_PAYLOAD_KEYS}
        results.append(
            SearchResult(
                point_id=str(hit.id),
                score=hit.score,
                text=payload.get("text", ""),
                document_id=payload.get("document_id", ""),
                filename=payload.get("filename", ""),
                doc_type=payload.get("doc_type", ""),
                chunk_index=payload.get("chunk_index", -1),
                char_start=payload.get("char_start", -1),
                char_end=payload.get("char_end", -1),
                extra=extra,
            )
        )

    logger.debug(
        "semantic_search returned %d results for collection %r.",
        len(results),
        collection_name,
    )
    return results
