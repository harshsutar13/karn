"""
api/capabilities.py

FastAPI router for capability/model registry endpoint.

Endpoint:
    GET /api/capabilities   — lists all registered capabilities and their
                              current model assignments, implementation status,
                              and provider info.

This is a read-only, zero-cost endpoint (no model calls).
Useful for health checks, UI model-selection menus, and monitoring.
"""

import os
import sys
import logging

from fastapi import APIRouter
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.registry import list_capabilities

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class CapabilityInfo(BaseModel):
    capability: str
    model: str
    provider: str
    implemented: bool
    notes: str


class CapabilitiesResponse(BaseModel):
    count: int
    capabilities: list[CapabilityInfo]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("/api/capabilities", response_model=CapabilitiesResponse, tags=["Capabilities"])
def get_capabilities():
    """
    Return the full list of registered AI capabilities and their current
    model assignments.

    Does not call any model — safe to call at any time.
    """
    raw = list_capabilities()
    return CapabilitiesResponse(
        count=len(raw),
        capabilities=[CapabilityInfo(**c) for c in raw],
    )
