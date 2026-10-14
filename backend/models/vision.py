"""
models/vision.py

Vision capability for SIH26 AI Workbench.

Uses qwen3-vl:4b via Ollama's multimodal interface for:
  - Image description / visual analysis
  - Visual question answering (VQA)
  - Structured information extraction from images

Public API
----------
    analyze_image(image_path, prompt, think=False, stream=False) -> str | Generator[str]
        Send a local image + natural-language prompt to qwen3-vl:4b.
        stream=True returns a generator yielding text chunks.

Design principles
-----------------
- Model is NOT loaded until analyze_image() is called.
- No dependency on FastAPI, RAG, Qdrant, or the document loader.
- `think` parameter is accepted for API consistency but silently ignored —
  multimodal vision models do not use the Ollama think mode.
- Empty-response retry: if the model returns empty output on first call,
  one automatic retry is made with a simplified fallback prompt.
- num_predict is capped to prevent runaway generation.
- Technical values (equipment IDs, decimals, units) are preserved verbatim.
"""

from __future__ import annotations
import os
import sys
import base64
import logging
from typing import Generator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama
from config import OLLAMA_HOST, VISION_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOKENS =  1500

_FALLBACK_PROMPT = (
    "Look at this image carefully. In 3-5 sentences, describe what you see: "
    "the main subject, any visible text or labels, and any equipment or objects shown. "
    "Be concise. Do not explain your reasoning. Output only the description."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_model(img_b64: str, prompt: str) -> str:
    """Send one image + prompt (non-streaming). Returns raw response string."""
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        response = client.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"num_predict": _MAX_TOKENS, "num_ctx": 8192},
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if any(k in exc_str for k in ("connection", "refused", "10061", "10054")):
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_HOST}. "
                f"Ensure Ollama is running. Original: {exc}"
            ) from exc
        raise
    return (response.message.content or "").strip()


def _stream_model(img_b64: str, prompt: str) -> Generator[dict, None, None]:
    """Send one image + prompt (streaming). Yields typed chunk dicts.
    Vision models do not produce thinking tokens, so only
    {\"kind\": \"chunk\", \"text\": ...} dicts are emitted.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        for raw in client.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"num_predict": _MAX_TOKENS, "num_ctx": 8192},
            stream=True,
        ):
            text = raw.message.content or ""
            if text.strip():                     # skip empty / whitespace-only chunks
                yield {"kind": "chunk", "text": text}  # yield with original spacing
    except Exception as exc:
        exc_str = str(exc).lower()
        if any(k in exc_str for k in ("connection", "refused", "10061", "10054")):
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_HOST}. "
                f"Ensure Ollama is running. Original: {exc}"
            ) from exc
        raise


def _read_image_b64(image_path: str) -> str:
    """Read a local image file and return its base64 encoding."""
    abs_path = os.path.abspath(image_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Vision image not found: {abs_path!r}. "
            "Ensure the file exists and the path is correct."
        )
    logger.debug("Vision: reading image %s (%d bytes)", abs_path, os.path.getsize(abs_path))
    with open(abs_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
    # qwen3-vl:4b can enter an extended reasoning loop and consume all tokens
    # without emitting visible output if the prompt is too open-ended.
    # Append a directive suffix to keep it on track.
    prompt = (
        prompt.rstrip()
        + " Be concise. Do not explain your reasoning. Output only the answer."
    )
def analyze_image(
    image_path: str,
    prompt: str,
    think: bool = False,   # accepted for API consistency; ignored for vision models
    stream: bool = False,
):
    """
    Analyze a local image using qwen3-vl:4b via Ollama.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the image file.
    prompt:
        Natural-language question or instruction.
    think:
        Accepted for API consistency. Vision models do not use think mode;
        this parameter is silently ignored.
    stream:
        If True, returns a generator yielding text chunks as they arrive.
        If False (default), returns the complete response string.

    Returns
    -------
    str
        Complete model response (stream=False).
    Generator[str, None, None]
        Chunk generator (stream=True).

    Raises
    ------
    ValueError
        If image_path or prompt is empty, or both attempts return empty (non-stream).
    FileNotFoundError
        If image_path does not exist.
    ConnectionError
        If Ollama is not reachable.
    """
    if not image_path or not image_path.strip():
        raise ValueError("analyze_image: image_path must be a non-empty string.")
    if not prompt or not prompt.strip():
        raise ValueError("analyze_image: prompt must be a non-empty string.")

    img_b64 = _read_image_b64(image_path)
    logger.debug("Vision: calling %r (stream=%s).", VISION_MODEL, stream)

    if stream:
        return _stream_model(img_b64, prompt)

    # Non-streaming path: first attempt + fallback retry
    result = _call_model(img_b64, prompt)

    if not result:
        logger.warning(
            "Vision model %r returned empty response. Retrying with fallback prompt.",
            VISION_MODEL,
        )
        result = _call_model(img_b64, _FALLBACK_PROMPT)

    if not result:
        raise ValueError(
            f"Vision model {VISION_MODEL!r} returned an empty response for "
            f"{image_path!r} on both the primary and fallback prompts."
        )

    logger.debug("Vision: response length %d chars.", len(result))
    return result
