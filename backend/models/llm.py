"""
models/llm.py

General-purpose LLM wrapper for SIH26 AI Workbench.
Routes all inference through models.manager.use_model so that only
one model is resident in VRAM at a time (needed on 6 GB GPUs).

Public API
----------
    generate_response(message, think=False, stream=False) -> str | Generator
        stream=False -> returns str (complete answer)
        stream=True  -> returns Generator[dict, None, None]
            Each yielded dict:
              {"kind": "chunk",    "text": "..."}   final answer token
              {"kind": "thinking", "text": "..."}   reasoning token (think=True only)

Ollama SDK note
---------------
Thinking content is on chunk.message.thinking (str | None).
Answer content is on chunk.message.content (str | None).
They arrive on separate chunks: thinking first, then content.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Generator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

from config import LLM_MODEL, OLLAMA_HOST
from models.manager import use_model

logger = logging.getLogger(__name__)


def _is_connection_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("connection", "refused", "10061", "10054"))


def generate_response(
    message: str,
    think: bool = False,
    stream: bool = False,
    history: list[dict] | None = None,
) -> "str | Generator[dict, None, None]":
    """
    Generate a response from the configured LLM via Ollama.

    The model is swapped into VRAM on demand via models.manager.

    history:
        Optional prior conversation as [{"role": "user"|"assistant"|"system",
        "content": str}, ...]. Prepended before the new user message so the
        model can resolve follow-up references ("that pump", "the file you
        made"). Pass None for single-turn requests.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    messages = list(history) if history else []
    messages.append({"role": "user", "content": message})

    if stream:
        return _stream_response(client, messages, think)

    try:
        with use_model(LLM_MODEL):
            response = client.chat(
                model=LLM_MODEL,
                messages=messages,
                think=think,
                keep_alive="2m",
            )
    except Exception as exc:
        if _is_connection_error(exc):
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_HOST}. "
                f"Ensure Ollama is running. Original: {exc}"
            ) from exc
        raise

    return response.message.content or ""


def _stream_response(
    client: ollama.Client,
    messages: list[dict],
    think: bool,
) -> Generator[dict, None, None]:
    """
    Internal generator — yields typed dicts from Ollama streaming chat.

    Wrapped in use_model so the swap happens once, before the stream
    starts. Once streaming has begun, no other request can swap the
    model away (the lock is held for the whole stream).
    """
    try:
        with use_model(LLM_MODEL):
            for raw in client.chat(
                model=LLM_MODEL,
                messages=messages,
                think=think,
                stream=True,
                keep_alive="2m",
            ):
                thinking_text = raw.message.thinking or ""
                content_text = raw.message.content or ""
                if thinking_text:
                    yield {"kind": "thinking", "text": thinking_text}
                if content_text:
                    yield {"kind": "chunk", "text": content_text}

    except Exception as exc:
        if _is_connection_error(exc):
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_HOST}. "
                f"Ensure Ollama is running. Original: {exc}"
            ) from exc
        raise