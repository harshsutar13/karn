"""
api/planner_stream_api.py

Streaming version of /api/agent/plan-and-execute.

Emits Server-Sent Events as each step of the plan starts and finishes,
plus a final event with the output path / download URL.
"""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Generator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


class PlanStreamRequest(BaseModel):
    goal: str
    session_id: str | None = None


# Human-readable label for each tool, shown in the UI step list.
_TOOL_LABELS = {
    "rag_search":       "Searching the knowledge base",
    "extract_params":   "Extracting operating parameters",
    "run_python":       "Running Python in the sandbox",
    "write_docx":       "Generating Word document",
    "write_xlsx":       "Generating Excel spreadsheet",
    "write_pptx":       "Generating PowerPoint deck",
    "write_file":       "Saving text file",
    "read_file":        "Reading file",
    "read_xlsx":        "Reading spreadsheet",
    "edit_xlsx":        "Editing spreadsheet",
    "ocr_image":        "Extracting text via OCR",
    "analyze_image":    "Analyzing image",
    "scan_to_approval": "Running scan-to-approval workflow",
    "summarize":        "Summarizing content",
}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _short_summary(tool: str, output) -> str:
    """One-line summary of a step's output for the UI."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output[:120]
    if not isinstance(output, dict):
        return str(output)[:120]

    # Tool-specific summaries
    if tool == "rag_search":
        n = output.get("count", 0)
        return f"{n} chunks retrieved"
    if tool == "extract_params":
        keys = [k for k, v in output.items() if v is not None]
        return f"extracted: {', '.join(keys) if keys else 'none'}"
    if tool == "run_python":
        out = (output.get("stdout") or "").strip().replace("\n", " · ")
        return out[:120] if out else "ran successfully"
    if tool in ("write_docx", "write_xlsx", "write_pptx", "write_file"):
        return output.get("filename") or output.get("output_path", "")
    if tool == "summarize":
        return (output.get("summary") or "")[:120]
    if tool in ("ocr_image", "analyze_image"):
        return (str(output) or "")[:120]
    if tool == "scan_to_approval":
        return output.get("filename", "")

    # Generic: show first string value
    for v in output.values():
        if isinstance(v, str):
            return v[:120]
    return "done"


def _remember(fn, *args, **kwargs):
    """Log to session memory; never let memory issues kill the stream."""
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.warning("Memory write failed during plan execution", exc_info=True)


def _final_message(goal: str, final_output: dict | None, ok: bool, error: str | None) -> str:
    """Assistant-facing summary of a finished plan run (stored in memory)."""
    if not ok:
        return f"Agent plan failed: {error or 'unknown error'}"
    parts = [f"Agent plan completed: {goal}"]
    if isinstance(final_output, dict):
        fname = final_output.get("filename")
        if fname:
            parts.append(f"Generated file: {fname}")
        for key in ("summary", "result"):
            val = final_output.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip()[:500])
    return "\n".join(parts)


def _stream_plan(goal: str, session_id: str | None = None) -> Generator[str, None, None]:
    """The actual SSE generator."""
    from tools.planner import make_plan, replan, Plan
    from tools.executor import TOOL_REGISTRY, _resolve_args, MAX_REPLANS
    from memory import store as memory_store

    mem_sid = memory_store.normalize_session_id(session_id) if session_id else None
    if mem_sid:
        _remember(memory_store.add_message, mem_sid, "user", goal, "plan", None)

    # ---- 1. Plan ----
    yield _sse({"type": "planning"})

    try:
        plan = make_plan(goal)
    except Exception as exc:
        yield _sse({"type": "error", "text": f"Planning failed: {exc}"})
        yield _sse_done()
        return

    if not plan.steps:
        yield _sse({"type": "error",
                    "text": "Planner produced no usable steps."})
        yield _sse_done()
        return

    # Annotate each step with a UI label
    steps_for_ui = []
    for s in plan.steps:
        label = _TOOL_LABELS.get(s["tool"], s["tool"])
        steps_for_ui.append({
            "id": s["id"],
            "tool": s["tool"],
            "label": label,
        })

    yield _sse({
        "type": "plan",
        "reasoning": plan.reasoning,
        "steps": steps_for_ui,
    })

    # ---- 2. Execute with re-plan ----
    state: dict[str, dict] = {}
    current_plan: Plan = plan
    replan_count = 0

    for _attempt in range(MAX_REPLANS + 1):
        failed_step = None
        failed_reason = ""

        for step in current_plan.steps:
            sid = step["id"]
            tool_name = step["tool"]
            label = _TOOL_LABELS.get(tool_name, tool_name)

            # Emit start
            yield _sse({
                "type": "step_start",
                "id": sid,
                "tool": tool_name,
                "label": label,
            })

            raw_args = step.get("args", {})
            args = _resolve_args(raw_args, state)
            fn = TOOL_REGISTRY.get(tool_name)

            t0 = time.perf_counter()

            if not fn:
                res = {"ok": False, "error": f"unknown tool {tool_name}"}
            else:
                try:
                    res = fn(args, state)
                except Exception as e:
                    res = {"ok": False, "error": f"tool raised: {e}"}

            elapsed_ms = (time.perf_counter() - t0) * 1000
            state[sid] = res
            ok = bool(res.get("ok"))

            # Emit done
            yield _sse({
                "type": "step_done",
                "id": sid,
                "tool": tool_name,
                "ok": ok,
                "elapsed_ms": round(elapsed_ms),
                "summary": _short_summary(tool_name, res.get("output")) if ok else (res.get("error") or "")[:160],
            })

            if mem_sid:
                _remember(
                    memory_store.log_action, mem_sid, "tool", tool_name,
                    (_short_summary(tool_name, res.get("output")) if ok
                     else (res.get("error") or "")[:160]),
                    ok,
                )

            if not ok:
                failed_step = sid
                failed_reason = res.get("error", "step failed")
                break

        if failed_step is None:
            # Success
            final_output = {}
            if current_plan.steps:
                last_id = current_plan.steps[-1]["id"]
                final_output = state.get(last_id, {}).get("output", {}) or {}
            # Produce download_url if applicable
            download_url = None
            if isinstance(final_output, dict) and final_output.get("filename"):
                download_url = f"/api/deliverable/download/{final_output['filename']}"

            yield _sse({
                "type": "final",
                "ok": True,
                "replans": replan_count,
                "final_output": final_output,
                "download_url": download_url,
            })
            if mem_sid:
                _remember(memory_store.log_action, mem_sid, "final", "plan",
                          f"goal succeeded ({replan_count} replans, "
                          f"{len(current_plan.steps)} steps)", True)
                _remember(memory_store.add_message, mem_sid, "assistant",
                          _final_message(goal, final_output, True, None), "plan", None)
            yield _sse_done()
            return

        if replan_count >= MAX_REPLANS:
            yield _sse({
                "type": "final",
                "ok": False,
                "replans": replan_count,
                "error": f"Max re-plans reached. Last failure: {failed_reason}",
            })
            if mem_sid:
                _remember(memory_store.log_action, mem_sid, "final", "plan",
                          f"failed after {replan_count} replans: {failed_reason}", False)
                _remember(memory_store.add_message, mem_sid, "assistant",
                          _final_message(goal, None, False,
                                         f"max re-plans reached. Last failure: {failed_reason}"),
                          "plan", None)
            yield _sse_done()
            return

        # Re-plan
        replan_count += 1
        yield _sse({
            "type": "replan",
            "attempt": replan_count,
            "reason": failed_reason[:200],
        })

        try:
            new_plan = replan(goal, current_plan, failed_reason)
        except Exception as exc:
            yield _sse({"type": "error",
                        "text": f"Re-planner failed: {exc}"})
            yield _sse_done()
            return

        if not new_plan.steps:
            yield _sse({
                "type": "final",
                "ok": False,
                "replans": replan_count,
                "error": f"Planner could not recover. Failure: {failed_reason}",
            })
            if mem_sid:
                _remember(memory_store.log_action, mem_sid, "final", "plan",
                          f"could not recover: {failed_reason}", False)
                _remember(memory_store.add_message, mem_sid, "assistant",
                          _final_message(goal, None, False,
                                         f"planner could not recover. Failure: {failed_reason}"),
                          "plan", None)
            yield _sse_done()
            return

        # Emit the new plan for the UI
        new_steps_for_ui = []
        for s in new_plan.steps:
            new_steps_for_ui.append({
                "id": s["id"],
                "tool": s["tool"],
                "label": _TOOL_LABELS.get(s["tool"], s["tool"]),
            })
        yield _sse({
            "type": "plan",
            "reasoning": new_plan.reasoning,
            "steps": new_steps_for_ui,
            "is_replan": True,
        })

        current_plan = new_plan


@router.post("/plan-and-execute/stream")
def plan_and_execute_stream(req: PlanStreamRequest):
    if not req.goal or not req.goal.strip():
        raise HTTPException(status_code=400, detail="goal is required")

    return StreamingResponse(
        _stream_plan(req.goal.strip(), req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )