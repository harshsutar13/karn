"""
api/ask.py

Unified /api/ask endpoint for SIH26.

Routes any request through the orchestrator to the correct capability:
    general   → Gemma 4 12B
    coding    → Qwen2.5-Coder 7B
    document  → RAG (Qdrant + Gemma)
    vision    → Qwen3-VL 4B

Endpoints
---------
POST /api/ask         — blocking, returns {answer, capability, model}
POST /api/ask/stream  — SSE streaming, same schema
"""

import os
import sys
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestration.service import handle_request, handle_request_stream

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's natural-language request.")
    image_path: str | None = Field(
        default=None,
        description="Optional absolute path to a local image. Triggers vision routing.",
    )
    thinking: bool = Field(
        default=False,
        description="Enable extended reasoning (think mode) where supported.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session id for persistent memory. When set, prior turns "
            "and recent agent actions are used as context and this turn is "
            "saved to the local memory store."
        ),
    )


class AskResponse(BaseModel):
    answer: str
    capability: str
    model: str


# ---------------------------------------------------------------------------
# Blocking endpoint (backward-compatible)
# ---------------------------------------------------------------------------

@router.post("/api/ask", response_model=AskResponse, tags=["Orchestrator"])
def ask_endpoint(request: AskRequest):
    """
    Unified ask endpoint (blocking).

    Automatically routes to the correct capability:
    - Vision if image_path is provided
    - Coding if the message involves code/programming
    - Document Q&A (RAG) if the message involves documents/equipment/maintenance
    - General chat otherwise

    The `thinking` field (default false) enables extended reasoning where supported.
    """
    try:
        result = handle_request(
            message=request.message,
            image_path=request.image_path,
            thinking=request.thinking,
            session_id=request.session_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ConnectionError as exc:
        logger.error("Ollama unreachable: %s", exc)
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error in /api/ask")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    return AskResponse(
        answer=result.answer,
        capability=result.capability,
        model=result.model,
    )


# ---------------------------------------------------------------------------
# Streaming endpoint (SSE)
# ---------------------------------------------------------------------------

@router.post("/api/ask/stream", tags=["Orchestrator"])
def ask_stream_endpoint(request: AskRequest):
    """
    Unified ask endpoint — Server-Sent Events (SSE) streaming.

    Same request schema as /api/ask. Returns a text/event-stream response.

    SSE event format:
        data: {"type":"meta","capability":"general","model":"gemma4:12b"}

        data: {"type":"chunk","text":"2 + 2 ="}

        data: {"type":"chunk","text":" 4"}

        data: [DONE]
    """
    def _generator():
        try:
            yield from handle_request_stream(
                message=request.message,
                image_path=request.image_path,
                thinking=request.thinking,
                session_id=request.session_id,
            )
        except FileNotFoundError as exc:
            import json
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
        except ValueError as exc:
            import json
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
        except ConnectionError as exc:
            import json
            logger.error("Ollama unreachable in stream: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'text': f'Model unavailable: {exc}'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            import json
            logger.exception("Unexpected error in /api/ask/stream")
            yield f"data: {json.dumps({'type': 'error', 'text': f'Internal error: {exc}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
