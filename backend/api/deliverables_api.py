"""
api/deliverables_api.py

List / download / delete generated deliverable files.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/deliverables", tags=["deliverables"])

# The outputs directory — must match what the writer tools use.
_OUTPUTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "outputs")
)


def _kind(name: str) -> str:
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    return ext or "file"


@router.get("")
def list_deliverables() -> dict:
    """
    Return every file in backend/outputs, newest first.
    """
    if not os.path.isdir(_OUTPUTS_DIR):
        return {"count": 0, "files": []}

    files = []
    for name in os.listdir(_OUTPUTS_DIR):
        full = os.path.join(_OUTPUTS_DIR, name)
        if not os.path.isfile(full):
            continue
        try:
            st = os.stat(full)
        except OSError:
            continue
        files.append({
            "name": name,
            "size": st.st_size,
            "modified": datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "kind": _kind(name),
            "download_url": f"/api/deliverable/download/{name}",
        })

    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"count": len(files), "files": files}


@router.delete("/{name}")
def delete_deliverable(name: str) -> dict:
    """
    Delete a file from the outputs directory.
    """
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    full = os.path.abspath(os.path.join(_OUTPUTS_DIR, name))
    if not full.startswith(_OUTPUTS_DIR):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(full)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")

    return {"ok": True, "deleted": name}