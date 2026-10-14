"""
api/sandbox_api.py

HTTP endpoints for the code sandbox.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tools.sandbox import run_python

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


class RunRequest(BaseModel):
    code: str
    timeout: int = 10


@router.post("/run")
def run(req: RunRequest) -> dict:
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="code is required")
    if req.timeout < 1 or req.timeout > 30:
        raise HTTPException(status_code=400, detail="timeout must be 1-30 seconds")

    result = run_python(req.code, timeout=req.timeout)
    return {
        "ok": result.ok,
        "exit_code": result.exit_code,
        "elapsed_ms": result.elapsed_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }