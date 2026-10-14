"""
api/memory_api.py

Session-memory endpoints: what the agent remembers (conversation) and what
it did (actions). Backed by the local SQLite store — no external services.

Endpoints
---------
GET  /api/memory/sessions                  — list sessions with counts
GET  /api/memory/history?session_id=...    — messages + actions for a session
POST /api/memory/clear                     — wipe one session's memory
"""

from __future__ import annotations

import logging
import os
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory import store as memory_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/memory", tags=["memory"])


class ClearRequest(BaseModel):
    session_id: str


@router.get("/sessions")
def sessions() -> dict:
    """List all remembered sessions, most recently active first."""
    try:
        return {"sessions": memory_store.list_sessions()}
    except Exception as exc:
        logger.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
def history(
    session_id: str,
    limit: int = 50,
    include_actions: bool = True,
) -> dict:
    """
    Full memory for one session: conversation messages (oldest first)
    and, optionally, the actions the agent performed.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        sid = memory_store.normalize_session_id(session_id)
        return {
            "session_id": sid,
            "messages": memory_store.get_history(sid, limit=limit),
            "actions": (
                memory_store.get_recent_actions(sid, limit=limit)
                if include_actions else []
            ),
        }
    except Exception as exc:
        logger.exception("Failed to load history for %r", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/clear")
def clear(req: ClearRequest) -> dict:
    """Delete all memory (messages + actions) for one session."""
    if not req.session_id or not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        sid = memory_store.normalize_session_id(req.session_id)
        memory_store.clear_session(sid)
        return {"ok": True, "session_id": sid, "cleared": True}
    except Exception as exc:
        logger.exception("Failed to clear session %r", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc))
