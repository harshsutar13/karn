"""
pipeline.py

RAG (Retrieval-Augmented Generation) pipeline for SIH26.

Architecture:
    User question
          |
          v
    Query embedding (nomic-embed-text)
          |
          v
    Qdrant semantic retrieval
          |
          v
    Retrieved chunks + metadata
          |
          v
    Prompt construction (context injection)
          |
          v
    Local LLM (Gemma 4 12B via Ollama)
          |
          v
    Grounded answer + source citations

The pipeline explicitly grounds the LLM answer in the retrieved
document context.  If the retrieved context does not contain enough
information, the LLM is instructed to say so rather than hallucinate.

Responsibilities:
    - Orchestrate retrieval -> prompt -> LLM
    - Format grounded prompts
    - Return structured answers with sources

NOT responsible for:
    - Embedding generation   (embeddings/embedder.py)
    - Vector storage/search  (rag/vector_store.py)
    - Retrieval logic        (rag/retriever.py)
    - Raw LLM calls          (models/llm.py)
"""

import logging
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retriever import retrieve, RetrievedChunk
from models.llm import generate_response
from config import QDRANT_COLLECTION


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class SourceReference:
    """Metadata for one retrieved chunk used to construct the answer."""
    filename: str
    document_id: str
    chunk_index: int
    score: float
    char_start: int
    char_end: int
    text_preview: str  # First 200 chars of the chunk for display


@dataclass
class RAGResponse:
    """
    Structured response from the RAG pipeline.

    Fields
    ------
    answer:
        The LLM-generated answer grounded in retrieved context.
    sources:
        List of source chunks used to construct the answer.
    query:
        The original user question.
    chunks_retrieved:
        Number of chunks retrieved from Qdrant.
    context_used:
        Whether any relevant context was found and used.
    """
    answer: str
    sources: list[SourceReference]
    query: str
    chunks_retrieved: int
    context_used: bool


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_NO_CONTEXT_NOTICE = (
    "No relevant documents were found in the knowledge base for this query."
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a knowledgeable assistant for an oil refinery and industrial operations team.
You answer questions using ONLY the document context provided below.

RULES:
1. Base your answer strictly on the provided context. Do not invent facts.
2. If the context does not contain sufficient information to answer the question, say exactly: "The available documents do not contain sufficient information to answer this question."
3. Preserve technical accuracy — numbers, unit names, equipment names, and procedures must be quoted precisely as they appear in the context.
4. If multiple context passages are relevant, synthesize them coherently.
5. Cite the source document(s) at the end of your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def _build_prompt(question: str, chunks: list[RetrievedChunk], conversation: str | None = None) -> str:
    """
    Build a grounded RAG prompt from the question and retrieved chunks.

    conversation:
        Optional compact transcript of the prior turns ("user: ...\nassistant: ...").
        Helps resolve follow-up questions ("what about pump P-102?") without
        un-grounding the answer — the context rules still apply.
    """
    context_blocks = "\n\n".join(c.as_context_string() for c in chunks)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        context=context_blocks,
        question=_with_conversation(question, conversation),
    )


def _build_no_context_prompt(question: str, conversation: str | None = None) -> str:
    """
    Build a minimal prompt when no relevant context was retrieved.
    The LLM is instructed to acknowledge the gap.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(
        context=_NO_CONTEXT_NOTICE,
        question=_with_conversation(question, conversation),
    )


def _with_conversation(question: str, conversation: str | None) -> str:
    if not conversation:
        return question
    return f"{conversation}\n\nCURRENT QUESTION:\n{question}"


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def rag_query(
    question: str,
    top_k: int = 5,
    collection_name: str = QDRANT_COLLECTION,
    score_threshold: float | None = None,
    think: bool = False,
    conversation: str | None = None,
) -> RAGResponse:
    """
    Execute the full RAG pipeline for a user question.

    Parameters
    ----------
    question:
        The user's natural-language question.
    top_k:
        Maximum number of chunks to retrieve. Default: 5.
    collection_name:
        Qdrant collection to search.
    score_threshold:
        Optional minimum similarity score for retrieved chunks.
    conversation:
        Optional compact transcript of prior turns, so follow-up questions
        ("and its alarm threshold?") resolve against the right equipment.

    Returns
    -------
    RAGResponse
        Structured response with answer, sources, and metadata.

    Raises
    ------
    ValueError
        If question is empty.
    ConnectionError
        If Qdrant is not reachable.
    """
    if not question or not question.strip():
        raise ValueError("rag_query: question must be a non-empty string.")

    logger.info("RAG query: %r", question[:80])

    # -----------------------------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # -----------------------------------------------------------------------
    chunks = retrieve(
        query=question,
        top_k=top_k,
        collection_name=collection_name,
        score_threshold=score_threshold,
    )

    logger.info("Retrieved %d chunks", len(chunks))

    # -----------------------------------------------------------------------
    # Step 2: Build prompt
    # -----------------------------------------------------------------------
    context_used = len(chunks) > 0

    if context_used:
        prompt = _build_prompt(question, chunks, conversation=conversation)
    else:
        prompt = _build_no_context_prompt(question, conversation=conversation)
        logger.warning("No relevant chunks found for query: %r", question[:80])

    # -----------------------------------------------------------------------
    # Step 3: Generate grounded answer
    # -----------------------------------------------------------------------
    answer = generate_response(prompt, think=think)

    # -----------------------------------------------------------------------
    # Step 4: Build structured response
    # -----------------------------------------------------------------------
    sources = [
        SourceReference(
            filename=c.filename,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            score=c.score,
            char_start=c.char_start,
            char_end=c.char_end,
            text_preview=c.text[:1500],
        )
        for c in chunks
    ]

    return RAGResponse(
        answer=answer,
        sources=sources,
        query=question,
        chunks_retrieved=len(chunks),
        context_used=context_used,
    )
