"""
api/coding_agent_api.py

HTTP endpoints for the offline agentic CODING environment.

Endpoints:
    GET  /api/coding/tools                      — list the coding tools
    POST /api/coding/plan-and-execute           — run the workflow, JSON result
    POST /api/coding/plan-and-execute/stream    — run the workflow, SSE events
    GET  /api/coding/diff                       — current working diff vs snapshot

The stream emits the same event vocabulary as /api/agent/plan-and-execute/stream
(planning / plan / step_start / step_done / replan / final) plus a "phase"
event for snapshot / verify / debug transitions.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/coding", tags=["coding-agent"])


class CodingTaskRequest(BaseModel):
    task: str
    session_id: str | None = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


@router.get("/tools")
def list_tools() -> dict:
    """List the coding agent's tools with their argument schemas."""
    from tools.workspace import CODING_TOOL_SPECS
    return {"count": len(CODING_TOOL_SPECS), "tools": CODING_TOOL_SPECS}


@router.post("/plan-and-execute")
def plan_and_execute(req: CodingTaskRequest) -> dict:
    """Run the coding workflow synchronously and return the collected result."""
    if not req.task or not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")

    from tools.coding_agent import run_coding_task
    result = run_coding_task(req.task.strip())
    final = result["final"]
    if not final.get("ok") and final.get("error"):
        # still return 200 with the structured failure — the client renders it
        return {"ok": False, "task": req.task, "error": final["error"],
                "cycles": final.get("cycles", 0)}
    return {"ok": True, "task": req.task, **{k: v for k, v in final.items() if k != "ok"}}


def _remember(fn, *args, **kwargs):
    """Log to session memory; never let memory issues kill the stream."""
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.warning("Memory write failed during coding task", exc_info=True)


@router.post("/plan-and-execute/stream")
def plan_and_execute_stream(req: CodingTaskRequest):
    """Run the coding workflow, streaming progress as Server-Sent Events."""
    if not req.task or not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")

    from tools.coding_agent import stream_coding_task
    from memory import store as memory_store

    mem_sid = (memory_store.normalize_session_id(req.session_id)
               if req.session_id else None)
    if mem_sid:
        _remember(memory_store.add_message, mem_sid, "user", req.task.strip(),
                  "coding-agent", None)

    def _final_message(evt: dict) -> str:
        if evt.get("ok"):
            parts = [f"Coding agent completed: {req.task.strip()}"]
            summary = evt.get("summary") or evt.get("result")
            if isinstance(summary, str) and summary.strip():
                parts.append(summary.strip()[:500])
            return "\n".join(parts)
        return f"Coding agent failed: {evt.get('error') or 'unknown error'}"

    def _gen():
        try:
            for evt in stream_coding_task(req.task.strip()):
                if mem_sid and evt.get("type") == "step_done":
                    _remember(memory_store.log_action, mem_sid, "tool",
                              evt.get("tool"), evt.get("summary") or "",
                              bool(evt.get("ok")))
                if mem_sid and evt.get("type") == "final":
                    _remember(memory_store.log_action, mem_sid, "final",
                              "coding-agent",
                              ("task succeeded" if evt.get("ok")
                               else f"failed: {evt.get('error', '')}"),
                              bool(evt.get("ok")))
                    _remember(memory_store.add_message, mem_sid, "assistant",
                              _final_message(evt), "coding-agent", None)
                yield _sse(evt)
        except Exception as exc:  # pragma: no cover — safety net for the stream
            logger.exception("coding task stream crashed")
            yield _sse({"type": "error", "text": f"Internal error: {exc}"})
            yield _sse({"type": "final", "ok": False, "error": str(exc)})
        yield _sse_done()

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/diff")
def current_diff() -> dict:
    """Return the working diff against the task-start snapshot."""
    from tools.workspace import tool_get_diff
    res = tool_get_diff({}, {})
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail=res.get("error", "no snapshot"))
    return res["output"]
