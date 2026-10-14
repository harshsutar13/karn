"""
api/planner_api.py

Planner-driven agent endpoint: LLM produces a plan, executor runs it.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tools.executor import plan_and_execute

router = APIRouter(prefix="/api/agent", tags=["agent"])


class PlanRequest(BaseModel):
    goal: str


@router.post("/plan-and-execute")
def plan_execute(req: PlanRequest) -> dict:
    if not req.goal or not req.goal.strip():
        raise HTTPException(status_code=400, detail="goal is required")

    result = plan_and_execute(req.goal.strip())

    # Build a download URL if the final step produced a docx
    download_url = None
    final = result.final_output or {}
    if isinstance(final, dict) and final.get("filename"):
        download_url = f"/api/deliverable/download/{final['filename']}"

    return {
        "goal": result.goal,
        "plan": result.plan,
        "replans": result.replans,
        "steps": result.step_results,
        "final_output": final,
        "download_url": download_url,
        "error": result.error,
    }