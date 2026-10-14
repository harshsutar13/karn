"""
api/upload.py

Image upload endpoint for SIH26 AI Workbench.

POST /api/upload/image
    Accepts a multipart image upload.
    Saves to data/uploads/<uuid>.<ext> (temporary local storage).
    Returns {"file_path": "/absolute/path", "filename": "original.png"}.

The returned file_path can be passed directly to POST /api/ask or
POST /api/ask/stream as image_path to trigger vision routing.
"""

import os
import sys
import uuid
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)
router = APIRouter()

# Supported image MIME types
_ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/webp",
}

# Uploads directory — two levels up from api/ → project root / data / uploads
_UPLOADS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads")
)


class UploadResponse(BaseModel):
    file_path: str
    filename: str


@router.post(
    "/api/upload/image",
    response_model=UploadResponse,
    tags=["Upload"],
)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for vision analysis.

    Accepts: JPEG, PNG, BMP, TIFF, WebP.
    Returns the absolute server-side path to be passed as image_path
    in subsequent /api/ask or /api/ask/stream requests.
    """
    # Validate content type
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {content_type!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_TYPES))}"
            ),
        )

    # Determine extension from original filename or content type
    orig_name = file.filename or "upload"
    _, ext = os.path.splitext(orig_name)
    if not ext:
        ext_map = {
            "image/jpeg": ".jpg", "image/jpg": ".jpg",
            "image/png": ".png", "image/bmp": ".bmp",
            "image/tiff": ".tiff", "image/webp": ".webp",
        }
        ext = ext_map.get(content_type, ".png")

    # Ensure uploads directory exists
    os.makedirs(_UPLOADS_DIR, exist_ok=True)

    # Save with a UUID filename to avoid conflicts
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(_UPLOADS_DIR, save_name)

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with open(save_path, "wb") as fh:
            fh.write(contents)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save uploaded image: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    logger.info("Saved uploaded image: %s (%d bytes)", save_path, len(contents))
    return UploadResponse(file_path=save_path, filename=orig_name)
