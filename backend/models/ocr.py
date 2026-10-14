"""
models/ocr.py

OCR capability for SIH26 AI Workbench.

Uses glm-ocr via Ollama's multimodal interface to extract text from images.

Public API
----------
    extract_text(image_path: str) -> str
        Read a local image, send it to glm-ocr, return clean extracted text.

Design principles
-----------------
- Model is NOT loaded until extract_text() is called.
- No dependency on FastAPI, RAG, Qdrant, or the document loader.
- Repetition protection: glm-ocr can occasionally loop its output.
  _dedup_repeated_blocks() detects and strips repetition before returning.
- num_predict is capped to avoid runaway generation.
- Technical values (decimals, units, equipment IDs) are preserved verbatim.

Supported input formats
-----------------------
Any image format Ollama / glm-ocr accepts: PNG, JPEG, BMP, TIFF.
The image is base64-encoded and sent inline in the chat payload.

Resource note
-------------
glm-ocr is a 2.2 GB model. On first call it loads into VRAM/RAM.
Subsequent calls in the same Ollama session are fast (~2-5s).
"""

import os
import sys
import base64
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama
from config import OLLAMA_HOST, OCR_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prompt that instructs the model to return plain text only.
_OCR_PROMPT = (
    "Extract all text from this image exactly as it appears. "
    "Return only the extracted text, with no commentary, no markdown, "
    "no code blocks, and no additional explanation."
)

# Cap token output to prevent runaway/repetition loops.
# 512 is sufficient for most OCR documents; raise if needed.
_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Repetition protection
# ---------------------------------------------------------------------------

def _dedup_repeated_blocks(text: str) -> str:
        # Strip markdown code fences — glm-ocr sometimes wraps its output in
    # ```markdown ... ``` when it thinks the content is code-adjacent.
    if "```" in text:
        # remove all fence lines
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("```")
        )
    """
    Detect and remove obvious repeated blocks in OCR output.

    GLM-OCR can occasionally loop, emitting the same block (a single line or
    a group of lines) many times. This happens in two observed forms:

      1. The whole output is one block repeated from the start.
      2. A block is repeated many times *anywhere* in the output
         (e.g. a plain-text version followed by N identical <table> copies),
         possibly separated by blank lines.

    Strategy
    --------
    A single linear scan over the non-empty lines.  For each position we try
    every block size (1 … remaining//2) and greedily collapse the largest
    block that repeats >= 3 times consecutively at that position.  The scan
    then advances past the collapsed run.  Where no repetition is found the
    current line is kept and the scan advances by 1.

    Threshold: >= 3 consecutive identical blocks (i.e. the block appears 3+
    times in a row).  This is conservative enough that legitimate OCR text
    (which rarely repeats an identical block three times) is left untouched.
    Technical values (decimals, units, equipment IDs) inside the kept copy
    are preserved verbatim.

    Parameters
    ----------
    text : str
        Raw OCR output from the model.

    Returns
    -------
    str
        De-duplicated text, stripped of leading/trailing whitespace.
    """
    # Strip blank lines — they are noise the model inserts between repeated
    # copies and would otherwise prevent block-matching.
    non_empty = [ln for ln in text.splitlines() if ln.strip()]

    if not non_empty:
        return text.strip()

    result: list[str] = []
    i = 0
    n = len(non_empty)

    while i < n:
        remaining = n - i
        collapsed = False

        # Try smallest block first — the minimal repeating unit is the right
        # one to collapse (e.g. 6 identical single lines → block_size=1 fires
        # before block_size=2 or 3 can produce a partial collapse).
        for block_size in range(1, remaining // 2 + 1):

            block = non_empty[i:i + block_size]

            # Count how many times this block repeats consecutively.
            reps = 1
            j = i + block_size
            while j + block_size <= n and non_empty[j:j + block_size] == block:
                reps += 1
                j += block_size

            if reps >= 3:
                logger.warning(
                    "OCR repetition detected: block of %d line(s) repeated "
                    "%d times. Keeping a single copy.",
                    block_size, reps,
                )
                result.extend(block)
                i = j          # skip all repetitions
                collapsed = True
                break

        if not collapsed:
            result.append(non_empty[i])
            i += 1

    return "\n".join(result).strip()




# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(image_path: str) -> str:
    """
    Extract text from a local image using glm-ocr via Ollama.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.
        Supported formats: PNG, JPEG, BMP, TIFF (whatever glm-ocr accepts).

    Returns
    -------
    str
        Extracted text, cleaned and de-duplicated.
        Preserves line breaks, technical values, and punctuation.

    Raises
    ------
    FileNotFoundError
        If image_path does not exist.
    ValueError
        If image_path is empty or Ollama returns an empty response.
    ConnectionError
        If Ollama is not reachable at the configured host.
    """
    # --- Validate input ---
    if not image_path or not image_path.strip():
        raise ValueError("extract_text: image_path must be a non-empty string.")

    abs_path = os.path.abspath(image_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"OCR image not found: {abs_path!r}. "
            "Ensure the file exists and the path is correct."
        )

    # --- Read and encode image ---
    logger.debug("Reading image: %s (%d bytes)", abs_path, os.path.getsize(abs_path))
    with open(abs_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("ascii")

    # --- Call glm-ocr ---
    client = ollama.Client(host=OLLAMA_HOST)
    logger.debug("Sending OCR request to model %r", OCR_MODEL)

    try:
        response = client.chat(
            model=OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _OCR_PROMPT,
                    "images": [img_b64],
                }
            ],
            options={"num_predict": _MAX_TOKENS},
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if "connection" in exc_str or "refused" in exc_str or "10061" in exc_str:
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_HOST}. "
                f"Ensure Ollama is running. Original: {exc}"
            ) from exc
        raise

    raw = response.message.content or ""

    if not raw.strip():
        raise ValueError(
            f"OCR model {OCR_MODEL!r} returned an empty response for {abs_path!r}. "
            "Check that the image is readable and the model is working correctly."
        )

    # --- Post-process ---
    cleaned = _dedup_repeated_blocks(raw)
    logger.debug("OCR complete. Raw %d chars -> cleaned %d chars.", len(raw), len(cleaned))

    return cleaned
