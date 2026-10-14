"""
api/vision.py

FastAPI router for the vision capability endpoint.

Uses qwen3-vl:4b via models/vision.py.

Endpoint:
    POST /api/vision/analyze  — analyse a local image with a natural-language prompt.

Design:
    All business logic stays in models/vision.py.
    This file only handles HTTP concerns: schema validation, error mapping,
    and response serialisation.

Resource note:
    qwen3-vl:4b is a 3.3 GB model.  Cold-start requests may take 16-33 seconds.
    Warm requests are typically 3-10 seconds.
"""

import os
import sys
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.vision import analyze_image
from models.registry import get_model

logger = logging.getLogger(__name__)
router = APIRouter()

# Retrieve model name from registry once at import time — no model loaded here.
_MODEL_NAME = get_model("vision").model_name


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class VisionAnalyzeRequest(BaseModel):
    image_path: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the local image file (PNG, JPEG, BMP, TIFF).",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language question or instruction, e.g. "
            "'Describe what you see.' or "
            "'What equipment is shown and what is its vibration value?'"
        ),
    )


class VisionAnalyzeResponse(BaseModel):
    answer: str
    model: str


# ---------------------------------------------------------------------------
# Error mapping helper  (same style as api/code.py)
# ---------------------------------------------------------------------------

def _handle_vision_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ConnectionError):
        logger.error("Ollama not reachable for vision: %s", exc)
        return HTTPException(
            status_code=503,
            detail=f"Vision model unavailable: {exc}",
        )
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Unexpected error in vision endpoint")
    return HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/api/vision/analyze", response_model=VisionAnalyzeResponse, tags=["Vision"])
def vision_analyze_endpoint(request: VisionAnalyzeRequest):
    """
    Analyse a local image using qwen3-vl:4b.

    Accepts any natural-language prompt:
    - Image description: "Describe what you see in this image."
    - Visual Q&A: "What equipment is shown and what is its vibration value?"
    - Field extraction: "List all labeled fields visible in this panel."

    The image must be accessible on the server's local filesystem.
    Supported formats: PNG, JPEG, BMP, TIFF.

    Responses may take 16-33 seconds on cold start and 3-10 seconds once warm.
    """
    try:
        answer = analyze_image(
            image_path=request.image_path,
            prompt=request.prompt,
        )
    except Exception as exc:
        raise _handle_vision_error(exc)

    return VisionAnalyzeResponse(answer=answer, model=_MODEL_NAME)
