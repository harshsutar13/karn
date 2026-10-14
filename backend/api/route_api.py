"""api/route_api.py

Debug endpoint for the LLM-based router.
"""

from fastapi import APIRouter, Query
from orchestration.router import classify_request

router = APIRouter(prefix="/api/route", tags=["route"])


@router.get("/explain")
def explain(message: str = Query(..., min_length=1)) -> dict:
    result = classify_request(message)
    return {
        "message": message,
        "capability": result.capability,
        "reason": result.reason,
    }