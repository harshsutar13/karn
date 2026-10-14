"""
api/deliverable.py

Endpoints that produce real file deliverables (docx).
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools.approval_note import generate_approval_note

router = APIRouter(prefix="/api/deliverable", tags=["deliverable"])


class ApprovalNoteRequest(BaseModel):
    case_id: str
    equipment_tag: str = "P-101"
    additional_context: str = ""


@router.post("/approval-note")
def create_approval_note(req: ApprovalNoteRequest) -> dict:
    if not req.case_id or not req.case_id.strip():
        raise HTTPException(status_code=400, detail="case_id is required")

    try:
        result = generate_approval_note(
            case_id=req.case_id.strip(),
            equipment_tag=req.equipment_tag.strip() or "P-101",
            additional_context=req.additional_context.strip(),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Approval note generation failed: {exc}")

    filename = os.path.basename(result.output_path)
    return {
        "output_path": result.output_path,
        "download_url": f"/api/deliverable/download/{filename}",
        "case_id": result.case_id,
        "sources": result.sources,
        "steps": result.steps,
        "draft_preview": result.draft_text[:400],
    }


@router.get("/download/{filename}")
def download(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    path = os.path.abspath(os.path.join(base, filename))

    if not path.startswith(base):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )