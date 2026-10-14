"""
embedder.py

Local embedding generation via Ollama.

Provides two public functions:

    embed_text(text)         -> list[float]
    embed_texts(texts)       -> list[list[float]]

Both call the configured Ollama embedding model and return raw
Python float lists, ready for insertion into Qdrant (or any other
vector store) without additional dependencies.

The module is intentionally Qdrant-free. Storing vectors is the
responsibility of the vector_store layer, not this module.
"""

import logging
import time
from typing import Any

import ollama

import sys
import os

# Support running from backend/ or from backend/embeddings/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMBEDDING_MODEL, OLLAMA_HOST


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_embedding(embedding: Any, source: str = "") -> list[float]:
    """
    Confirm that what Ollama returned is a non-empty list of numbers.

    Raises
    ------
    TypeError
        If the embedding is not a list.
    ValueError
        If the embedding is empty or contains non-numeric values.
    """
    if not isinstance(embedding, list):
        raise TypeError(
            f"Expected embedding to be a list, got {type(embedding).__name__}"
            + (f" (source: {source!r})" if source else "")
        )

    if not embedding:
        raise ValueError(
            "Received empty embedding vector"
            + (f" for: {source!r}" if source else "")
        )

    if not all(isinstance(v, (int, float)) for v in embedding):
        sample = [type(v).__name__ for v in embedding[:5]]
        raise ValueError(
            f"Embedding contains non-numeric values. "
            f"First 5 types: {sample}"
            + (f" (source: {source!r})" if source else "")
        )

    return [float(v) for v in embedding]


def _get_client() -> ollama.Client:
    """Return an Ollama client pointed at the configured host."""
    return ollama.Client(host=OLLAMA_HOST)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a single string.

    Parameters
    ----------
    text:
        The input text to embed. Must be a non-empty string.

    Returns
    -------
    list[float]
        Embedding vector. Length equals the model's output dimension.

    Raises
    ------
    ValueError
        If *text* is empty or whitespace-only.
    ollama.ResponseError
        If Ollama is unreachable or the model is not available.
    """
    if not text or not text.strip():
        raise ValueError("embed_text requires non-empty, non-whitespace input.")

    logger.debug("Embedding single text with model %r (len=%d)", EMBEDDING_MODEL, len(text))

    client = _get_client()

    try:
        response = client.embed(model=EMBEDDING_MODEL, input=text)
    except ollama.ResponseError as exc:
        logger.error("Ollama embed call failed: %s", exc)
        raise

    # ollama.Client.embed() returns an EmbedResponse.
    # The .embeddings field is a list of vectors (one per input).
    raw = response.embeddings

    if not raw or len(raw) == 0:
        raise ValueError(
            f"Ollama returned no embeddings for model {EMBEDDING_MODEL!r}. "
            "Ensure the model is pulled and supports embedding."
        )

    return _validate_embedding(raw[0], source=text[:80])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding vectors for a batch of strings.

    Calls Ollama once per text (Ollama's Python library currently
    sends one request per call; true batching is handled server-side
    when the API supports it). Each call is individually validated.

    Parameters
    ----------
    texts:
        Non-empty list of strings to embed. Each string must be
        non-empty and non-whitespace-only.

    Returns
    -------
    list[list[float]]
        One embedding vector per input string, in the same order.

    Raises
    ------
    ValueError
        If *texts* is empty, or any individual string is empty.
    ollama.ResponseError
        If Ollama is unreachable or the model is not available.
    """
    if not texts:
        raise ValueError("embed_texts requires a non-empty list of strings.")

    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise ValueError(
                f"embed_texts: texts[{i}] is empty or whitespace-only."
            )

    logger.debug(
        "Embedding %d texts with model %r", len(texts), EMBEDDING_MODEL
    )

    embeddings: list[list[float]] = []

    for i, text in enumerate(texts):
        vec = embed_text(text)
        embeddings.append(vec)
        logger.debug("  [%d/%d] dim=%d", i + 1, len(texts), len(vec))

    return embeddings


def get_embedding_dimension() -> int:
    """
    Return the vector dimension of the configured embedding model
    by embedding a probe sentence.

    Useful for configuring Qdrant collection schemas.
    """
    probe = "dimension probe"
    vec = embed_text(probe)
    return len(vec)
