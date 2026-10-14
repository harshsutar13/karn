"""
api/agent_api.py

Multi-step agentic endpoint.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools.agent import analyze_case

router = APIRouter(prefix="/api/agent", tags=["agent"])


class AnalyzeRequest(BaseModel):
    case_id: str
    equipment_tag: str = "P-101"
    additional_context: str = ""


@router.post("/analyze-case")
def analyze(req: AnalyzeRequest) -> dict:
    if not req.case_id or not req.case_id.strip():
        raise HTTPException(status_code=400, detail="case_id is required")

    result = analyze_case(
        case_id=req.case_id.strip(),
        equipment_tag=req.equipment_tag.strip() or "P-101",
        additional_context=req.additional_context.strip(),
    )

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    filename = os.path.basename(result.output_path)
    return {
        "case_id": result.case_id,
        "equipment_tag": result.equipment_tag,
        "output_path": result.output_path,
        "download_url": f"/api/deliverable/download/{filename}",
        "steps": result.steps,
        "sources": result.sources,
        "sandbox": {
            "ok": result.calc_ok,
            "stdout": result.calc_stdout,
            "code": result.calc_code,
        },
        "draft_preview": result.draft_text[:600],
    }

from tools.agent import analyze_case, scan_to_approval


class ScanRequest(BaseModel):
    image_path: str
    case_id: str
    equipment_tag: str = "P-101"


@router.post("/scan-to-approval")
def scan_approval(req: ScanRequest) -> dict:
    if not req.image_path or not req.image_path.strip():
        raise HTTPException(status_code=400, detail="image_path is required")
    if not req.case_id or not req.case_id.strip():
        raise HTTPException(status_code=400, detail="case_id is required")
    import os
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=400, detail=f"image not found: {req.image_path}")

    result = scan_to_approval(
        image_path=req.image_path.strip(),
        case_id=req.case_id.strip(),
        equipment_tag=req.equipment_tag.strip() or "P-101",
    )

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    filename = os.path.basename(result.output_path)
    return {
        "case_id": result.case_id,
        "equipment_tag": result.equipment_tag,
        "image_path": result.image_path,
        "output_path": result.output_path,
        "download_url": f"/api/deliverable/download/{filename}",
        "steps": result.steps,
        "sources": result.sources,
        "ocr_preview": result.ocr_text[:400],
        "findings_preview": result.extracted_findings[:400],
    }