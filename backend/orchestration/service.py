"""
orchestration/service.py

Thin orchestration service for SIH26.

Connects the capability router to the existing model/pipeline functions.
No HTTP calls, no agents, no LangGraph — just direct Python function calls.

Public API
----------
    handle_request(message, image_path=None, thinking=False, session_id=None)
        -> OrchestratorResponse
        Blocking call. Returns a completed OrchestratorResponse.

    handle_request_stream(message, image_path=None, thinking=False, session_id=None)
        -> Generator[str, None, None]
        Yields SSE-formatted strings.

SSE event types emitted by handle_request_stream
-------------------------------------------------
    data: {"type":"meta",     "capability":"...", "model":"..."}
    data: {"type":"thinking", "text":"..."}   <- reasoning token  (thinking=True only)
    data: {"type":"chunk",    "text":"..."}   <- answer token
    data: [DONE]

When thinking=False:
    Only "meta", "chunk", and [DONE] events are emitted.
When thinking=True:
    "thinking" events precede "chunk" events (Ollama phases: reason first, then answer).
    Vision always emits only "chunk" events (vision models don't think).

Session memory
--------------
When a session_id is supplied, the conversation is persisted to the local
memory store (data/memory.db) and prior turns + recent agent actions are
injected into every capability's prompt, so follow-up questions like
"and for pump P-102?" or "open the file you just generated" work.
"""

from __future__ import annotations
import json
import os
import sys
import logging
from dataclasses import dataclass
from typing import Generator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory import store as memory_store
from orchestration.router import classify_request
from models.registry import get_model

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    answer: str
    capability: str
    model: str


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _load_context(session_id: str | None) -> tuple[list[dict] | None, str | None]:
    """
    Load memory context for a session.

    Returns (history_prefix, transcript):
      history_prefix — Ollama message list for model layers that accept
                       a messages list (general, coding)
      transcript     — "user: .../assistant: ..." text for prompt-based
                       capabilities (RAG, vision)

    Both are None for fresh sessions.
    """
    if not session_id:
        return None, None
    try:
        return (
            memory_store.build_history_prefix(session_id),
            memory_store.conversation_transcript(session_id),
        )
    except Exception as exc:
        # Memory must never break answering — degrade to stateless.
        logger.warning("Memory load failed for session %r: %s", session_id, exc)
        return None, None


def _save_turn(session_id: str | None, message: str, answer: str,
               capability: str, model_name: str) -> None:
    if not session_id:
        return
    try:
        memory_store.add_message(session_id, "user", message, capability, model_name)
        memory_store.add_message(session_id, "assistant", answer, capability, model_name)
    except Exception as exc:
        logger.warning("Memory save failed for session %r: %s", session_id, exc)


def _stream_with_memory(
    gen: Generator[dict, None, None],
    session_id: str | None,
    message: str,
    capability: str,
    model_name: str,
) -> Generator[str, None, None]:
    """
    Wrap a typed-chunk generator: translate to SSE events while
    accumulating the answer, then persist the completed turn.
    On error mid-stream, the partial answer is saved so memory stays honest.
    """
    parts: list[str] = []
    try:
        for typed in gen:
            if typed.get("kind") == "chunk":
                parts.append(typed.get("text", ""))
            yield _sse_from_typed(typed)
    except Exception:
        partial = "".join(parts)
        _save_turn(session_id, message,
                   (partial + "\n\n[response interrupted]") if partial else "[no response — error]",
                   capability, model_name)
        raise
    _save_turn(session_id, message, "".join(parts), capability, model_name)


# ---------------------------------------------------------------------------
# Blocking (non-streaming) request handler
# ---------------------------------------------------------------------------

def handle_request(
    message: str,
    image_path: str | None = None,
    thinking: bool = False,
    session_id: str | None = None,
) -> OrchestratorResponse:
    """
    Route a user request to the correct capability and return the answer.

    Parameters
    ----------
    message:
        The user's natural-language request. Must be non-empty.
    image_path:
        Optional path to a local image. If provided, routes to vision.
    thinking:
        If True, enables extended reasoning where supported.
    session_id:
        Optional session id. When set, prior conversation + recent agent
        actions are injected as context and the turn is persisted.

    Returns
    -------
    OrchestratorResponse
        .answer     — model's response text
        .capability — which capability handled it
        .model      — which model was used
    """
    if not message or not message.strip():
        raise ValueError("handle_request: message must be non-empty.")

    history, transcript = _load_context(session_id)
    classification = classify_request(message, image_path, conversation=transcript)
    cap = classification.capability

    logger.info(
        "Orchestrator: capability=%r reason=%r thinking=%s session=%s",
        cap, classification.reason, thinking,
        bool(session_id),
    )

    # ---- VISION ------------------------------------------------------------
    # MVP policy: vision always runs in normal mode.
    # Vision models (qwen3-vl, etc.) don't produce thinking tokens, and the
    # thinking UI must not be shown for image requests regardless of the global
    # Thinking Mode toggle. Forcing think=False here centralises the policy so
    # every future vision model inherits it automatically.
    if cap == "vision":
        from models.vision import analyze_image
        model_name = get_model("vision").model_name
        prompt = f"{transcript}\n\n{message}" if transcript else message
        answer = analyze_image(image_path=image_path, prompt=prompt, think=False)
        _save_turn(session_id, message, answer, cap, model_name)
        return OrchestratorResponse(answer=answer, capability=cap, model=model_name)

    # ---- CODING ------------------------------------------------------------
    if cap == "coding":
        from models.coding import generate_code
        model_name = get_model("coding").model_name
        answer = generate_code(description=message, think=thinking, history=history)
        _save_turn(session_id, message, answer, cap, model_name)
        return OrchestratorResponse(answer=answer, capability=cap, model=model_name)

    # ---- DOCUMENT (RAG) ----------------------------------------------------
    if cap == "document":
        from rag.pipeline import rag_query
        model_name = get_model("general_reasoning").model_name
        try:
            rag_resp = rag_query(message, think=thinking, conversation=transcript)
        except ValueError as exc:
            if "question must be" in str(exc):
                raise
            logger.warning("RAG retrieval error (likely cold-start): %s", exc)
            answer = (
                "The document retrieval service encountered a temporary error. "
                "Please try again in a moment."
            )
            _save_turn(session_id, message, answer, cap, model_name)
            return OrchestratorResponse(answer=answer, capability=cap, model=model_name)
        _save_turn(session_id, message, rag_resp.answer, cap, model_name)
        return OrchestratorResponse(
            answer=rag_resp.answer,
            capability=cap,
            model=model_name,
        )

    # ---- GENERAL -----------------------------------------------------------
    from models.llm import generate_response
    model_name = get_model("general_reasoning").model_name
    answer = generate_response(message, think=thinking, history=history)
    _save_turn(session_id, message, answer, cap, model_name)
    return OrchestratorResponse(answer=answer, capability=cap, model=model_name)


# ---------------------------------------------------------------------------
# Streaming request handler (SSE generator)
# ---------------------------------------------------------------------------

def handle_request_stream(
    message: str,
    image_path: str | None = None,
    thinking: bool = False,
    session_id: str | None = None,
) -> Generator[str, None, None]:
    """
    Route a user request and yield SSE-formatted strings for real-time streaming.

    SSE events emitted (in order):
      1. meta     — capability + model name
      2. thinking — reasoning tokens (only when thinking=True and model supports it)
      3. chunk    — final answer tokens
      4. [DONE]   — stream sentinel

    The model layer (llm.py, coding.py, vision.py) yields typed dicts:
      {"kind": "thinking", "text": "..."}
      {"kind": "chunk",    "text": "..."}
    This function translates them to the corresponding SSE events, and
    persists the completed turn to session memory when session_id is set.
    """
    if not message or not message.strip():
        raise ValueError("handle_request_stream: message must be non-empty.")

    history, transcript = _load_context(session_id)
    classification = classify_request(message, image_path, conversation=transcript)
    cap = classification.capability

    logger.info(
        "Orchestrator stream: capability=%r reason=%r thinking=%s session=%s",
        cap, classification.reason, thinking,
        bool(session_id),
    )

    # ---- VISION ------------------------------------------------------------
    # MVP policy: vision always runs in normal mode (see handle_request above).
    if cap == "vision":
        from models.vision import analyze_image
        model_name = get_model("vision").model_name
        prompt = f"{transcript}\n\n{message}" if transcript else message
        yield _sse_meta(cap, model_name)
        gen = analyze_image(image_path=image_path, prompt=prompt, think=False, stream=True)
        yield from _stream_with_memory(gen, session_id, message, cap, model_name)
        yield _sse_done()
        return

    # ---- CODING ------------------------------------------------------------
    if cap == "coding":
        from models.coding import generate_code
        model_name = get_model("coding").model_name
        yield _sse_meta(cap, model_name)
        gen = generate_code(description=message, think=thinking, stream=True, history=history)
        yield from _stream_with_memory(gen, session_id, message, cap, model_name)
        yield _sse_done()
        return

    # ---- DOCUMENT (RAG) — retrieve first, then stream the answer -----------
    if cap == "document":
        from rag.retriever import retrieve
        from models.llm import generate_response
        from rag.pipeline import _build_prompt, _build_no_context_prompt
        from config import QDRANT_COLLECTION
        model_name = get_model("general_reasoning").model_name
        yield _sse_meta(cap, model_name)
        try:
            chunks = retrieve(query=message, collection_name=QDRANT_COLLECTION)
            if chunks:
                prompt = _build_prompt(message, chunks, conversation=transcript)
            else:
                prompt = _build_no_context_prompt(message, conversation=transcript)
            gen = generate_response(prompt, think=thinking, stream=True)
            yield from _stream_with_memory(gen, session_id, message, cap, model_name)
        except ValueError as exc:
            if "question must be" in str(exc):
                raise
            logger.warning("RAG retrieval error in stream: %s", exc)
            answer = "The document retrieval service encountered a temporary error. Please try again in a moment."
            _save_turn(session_id, message, answer, cap, model_name)
            yield _sse_chunk(answer)
        yield _sse_done()
        return

    # ---- GENERAL -----------------------------------------------------------
    from models.llm import generate_response
    model_name = get_model("general_reasoning").model_name
    yield _sse_meta(cap, model_name)
    gen = generate_response(message, think=thinking, stream=True, history=history)
    yield from _stream_with_memory(gen, session_id, message, cap, model_name)
    yield _sse_done()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_from_typed(typed: dict) -> str:
    """Translate a model-layer typed dict to the correct SSE event string."""
    kind = typed.get("kind", "chunk")
    text = typed.get("text", "")
    if kind == "thinking":
        return _sse_thinking(text)
    return _sse_chunk(text)

def _sse_meta(capability: str, model: str) -> str:
    payload = json.dumps({"type": "meta", "capability": capability, "model": model})
    return f"data: {payload}\n\n"

def _sse_chunk(text: str) -> str:
    payload = json.dumps({"type": "chunk", "text": text})
    return f"data: {payload}\n\n"

def _sse_thinking(text: str) -> str:
    payload = json.dumps({"type": "thinking", "text": text})
    return f"data: {payload}\n\n"

def _sse_done() -> str:
    return "data: [DONE]\n\n"
