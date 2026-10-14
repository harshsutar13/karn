"""
api/ide_api.py

Backend API for the LOKI Code Studio / IDE.
Supports:
  - File tree exploration (workspace, project, or any custom folder on PC)
  - Common folder suggestions (Desktop, Documents, Downloads, Workspace, Project)
  - Folder validation
  - Full CRUD operations (create, read, update/save, delete, rename)
  - Safe code execution in isolated sandbox
  - LOKI AI actions (generate, explain, review, debug)
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import logging
from pathlib import Path
from typing import Literal

import base64
import ctypes
import getpass
import json
import mimetypes
import re
import string
import subprocess

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response, HTMLResponse
from pydantic import BaseModel, Field

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.sandbox import run_python
from models.coding import (
    generate_code,
    review_code,
    explain_code,
    debug_code,
    _call,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ide", tags=["ide"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "workspace"))
os.makedirs(WORKSPACE_ROOT, exist_ok=True)

# Directories to exclude from browsing
EXCLUDED_DIR_NAMES = {
    ".venv", "venv", "__pycache__", ".git", ".pytest_cache",
    ".agent_state", "node_modules", "logs", ".claude",
    ".idea", ".vscode", "dist", "build", ".next", "$RECYCLE.BIN",
    "System Volume Information"
}

# Max allowed file size for text editor reading (2 MB)
MAX_EDIT_FILE_SIZE = 2 * 1024 * 1024

TEXT_EXTENSIONS = {
    ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml",
    ".md", ".txt", ".rst", ".toml", ".cfg", ".ini",
    ".sql", ".sh", ".bat", ".ps1", ".env", ".gitignore",
    ".dockerignore", ".xml", ".csv"
}


# ---------------------------------------------------------------------------
# Path Helper & Validation
# ---------------------------------------------------------------------------

def _get_target_root(scope: str = "workspace", custom_root: str | None = None) -> str:
    if scope == "custom" and custom_root and custom_root.strip():
        resolved = os.path.abspath(custom_root.strip())
        if not os.path.isdir(resolved):
            raise HTTPException(
                status_code=400,
                detail=f"Folder does not exist or is not a directory: {custom_root}"
            )
        return resolved
    if scope == "project":
        return PROJECT_ROOT
    return WORKSPACE_ROOT


def _resolve_safe_path(rel_path: str, scope: str = "workspace", custom_root: str | None = None) -> str:
    root = _get_target_root(scope, custom_root)
    clean_rel = rel_path.strip().lstrip("/\\")
    target = os.path.abspath(os.path.join(root, clean_rel))

    # Path traversal check
    if os.path.commonpath([target, root]) != root:
        raise HTTPException(status_code=400, detail="Invalid path: outside allowed root directory")
    return target


def _to_rel_path(abs_path: str, scope: str = "workspace", custom_root: str | None = None) -> str:
    root = _get_target_root(scope, custom_root)
    rel = os.path.relpath(abs_path, root).replace("\\", "/")
    return "" if rel == "." else rel


def _encode_scope_key(scope: str, custom_root: str | None = None) -> str:
    if scope == "custom" and custom_root:
        clean = custom_root.strip()
        b64 = base64.urlsafe_b64encode(clean.encode("utf-8")).decode("ascii").rstrip("=")
        return f"c_{b64}"
    if scope == "project":
        return "project"
    return "workspace"


def _decode_scope_key(scope_key: str) -> tuple[str, str | None]:
    if scope_key == "project":
        return "project", None
    if scope_key.startswith("c_"):
        b64 = scope_key[2:]
        padded = b64 + "=" * (-len(b64) % 4)
        try:
            custom_root = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
            return "custom", custom_root
        except Exception:
            return "workspace", None
    return "workspace", None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FileNode(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int | None = None
    ext: str = ""
    children: list["FileNode"] | None = None


class FileContentResponse(BaseModel):
    path: str
    name: str
    content: str
    size: int
    read_only: bool = False
    scope: str = "workspace"
    root_path: str = ""


class SaveFileRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class CreateFileRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str = ""
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class CreateFolderRequest(BaseModel):
    path: str = Field(..., min_length=1)
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class RenameRequest(BaseModel):
    old_path: str = Field(..., min_length=1)
    new_path: str = Field(..., min_length=1)
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class DeleteRequest(BaseModel):
    path: str = Field(..., min_length=1)
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class ValidateFolderRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)


class RunCodeRequest(BaseModel):
    code: str
    timeout: int = Field(default=15, ge=1, le=60)
    file_path: str | None = None


class TerminalExecRequest(BaseModel):
    command: str = ""
    cwd: str | None = None
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None
    timeout: int = Field(default=45, ge=1, le=120)


class TerminalFixRequest(BaseModel):
    command: str = ""
    error_output: str = ""
    cwd: str = ""


class BatchCreateItem(BaseModel):
    path: str = Field(..., min_length=1)
    content: str = ""


class BatchCreateRequest(BaseModel):
    files: list[BatchCreateItem]
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class GlobalSearchRequest(BaseModel):
    query: str
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None
    case_sensitive: bool = False
    whole_word: bool = False
    is_regex: bool = False
    include_pattern: str = ""
    exclude_pattern: str = ""
    max_results: int = 300


class BatchReplaceRequest(BaseModel):
    query: str
    replacement: str = ""
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None
    case_sensitive: bool = False
    whole_word: bool = False
    is_regex: bool = False
    files: list[str] | None = None


class GitStatusRequest(BaseModel):
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class GitDiffRequest(BaseModel):
    file_path: str | None = None
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class GitCommitMessageRequest(BaseModel):
    hint: str = ""
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class GitCommitRequest(BaseModel):
    message: str = Field(..., min_length=1)
    files: list[str] | None = None
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class GitInitRequest(BaseModel):
    scope: Literal["workspace", "project", "custom"] = "workspace"
    custom_root: str | None = None


class GitPushFileRequest(BaseModel):
    file_path: str = Field(..., min_length=1, description="File name or path to push (e.g. 'p1.py')")
    commit_message: str | None = None
    git_workspace: str | None = None


class AIActionRequest(BaseModel):
    action: Literal["explain", "review", "debug", "generate"]
    code: str = ""
    prompt: str = ""
    error: str = ""
    language: str | None = None
    file_path: str | None = None


class InlineEditRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    selected_code: str = ""
    file_path: str = ""
    language: str = "python"
    context_before: str = ""
    context_after: str = ""



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _get_windows_drives() -> list[dict]:
    """Enumerate all active logical drives on Windows (C:, D:, E:, F:, USBs) with volume labels."""
    drives = []
    if sys.platform != "win32":
        return drives
    try:
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    except Exception:
        bitmask = 0

    for idx, letter in enumerate(string.ascii_uppercase):
        drive_path = f"{letter}:\\"
        has_drive = bool(bitmask & (1 << idx)) if bitmask else False
        if not has_drive:
            try:
                if os.path.exists(drive_path):
                    has_drive = True
            except Exception:
                has_drive = False

        if not has_drive:
            continue

        vol_name = ""
        drive_type = 3
        try:
            vol_buf = ctypes.create_unicode_buffer(1024)
            ctypes.windll.kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(drive_path),
                vol_buf, 1024, None, None, None, None, 0
            )
            vol_name = vol_buf.value
        except Exception:
            pass

        try:
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive_path)
        except Exception:
            pass

        icon = "💾"
        type_desc = "Local Disk"
        if drive_type == 2:
            icon = "🔌"
            type_desc = "Removable Disk"
        elif drive_type == 4:
            icon = "🌐"
            type_desc = "Network Drive"
        elif drive_type == 5:
            icon = "💿"
            type_desc = "Optical Drive"

        label = f"{vol_name} ({letter}:)" if vol_name else f"{type_desc} ({letter}:)"
        drives.append({
            "name": label,
            "letter": letter,
            "path": drive_path.replace("\\", "/"),
            "raw_path": drive_path,
            "type": "custom",
            "icon": icon,
            "is_drive": True
        })
    return drives


def _find_folder_candidates(folder_name: str) -> list[str]:
    """Search for a directory name across all drives and common folders."""
    clean_name = folder_name.strip().strip('"').strip("'")
    if not clean_name:
        return []

    # If it is already an absolute path or starts with a drive letter, check directly
    if os.path.isabs(clean_name) or (len(clean_name) >= 2 and clean_name[1] == ":"):
        if os.path.isdir(clean_name):
            return [os.path.abspath(clean_name).replace("\\", "/")]
        return []

    matches = []
    seen = set()

    drives = _get_windows_drives()
    drive_roots = [d["raw_path"] for d in drives]
    home = os.path.expanduser("~")
    roots_to_check = drive_roots + [
        os.path.join(home, "Desktop"),
        os.path.join(home, "Documents"),
        os.path.join(home, "Downloads"),
        home,
        WORKSPACE_ROOT,
        PROJECT_ROOT
    ]

    for root in roots_to_check:
        if not os.path.isdir(root):
            continue
        # Direct child check (e.g. D:\Java or C:\Java)
        candidate = os.path.join(root, clean_name)
        if os.path.isdir(candidate):
            norm = os.path.abspath(candidate).replace("\\", "/")
            if norm not in seen:
                seen.add(norm)
                matches.append(norm)

        # 1-level subfolder check
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if entry.is_dir() and entry.name not in EXCLUDED_DIR_NAMES and not entry.name.startswith("."):
                        sub_cand = os.path.join(entry.path, clean_name)
                        if os.path.isdir(sub_cand):
                            norm = os.path.abspath(sub_cand).replace("\\", "/")
                            if norm not in seen:
                                seen.add(norm)
                                matches.append(norm)
        except Exception:
            continue

    return matches


@router.get("/common-folders")
def get_common_folders():
    """Return all logical drives and common quick-access directories on the user's PC."""
    drives = _get_windows_drives()

    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    documents = os.path.join(home, "Documents")
    downloads = os.path.join(home, "Downloads")

    user_folders = []
    if os.path.isdir(desktop):
        user_folders.append({"name": "Desktop", "path": os.path.abspath(desktop).replace("\\", "/"), "type": "custom", "icon": "🖥️"})
    if os.path.isdir(documents):
        user_folders.append({"name": "Documents", "path": os.path.abspath(documents).replace("\\", "/"), "type": "custom", "icon": "📄"})
    if os.path.isdir(downloads):
        user_folders.append({"name": "Downloads", "path": os.path.abspath(downloads).replace("\\", "/"), "type": "custom", "icon": "⬇️"})

    workspace_folders = [
        {"name": "Mindcraft Workspace", "path": WORKSPACE_ROOT.replace("\\", "/"), "type": "workspace", "icon": "🏠"},
        {"name": "Mindcraft Project Root", "path": PROJECT_ROOT.replace("\\", "/"), "type": "project", "icon": "🗂️"},
    ]

    return {
        "drives": drives,
        "user_folders": user_folders,
        "workspace_folders": workspace_folders,
        "folders": drives + user_folders + workspace_folders
    }


@router.get("/resolve-folder-name")
def resolve_folder_name(name: str):
    """Search for matching folders across PC drives given a folder name from the browser picker."""
    candidates = _find_folder_candidates(name)
    return {
        "ok": True,
        "name": name,
        "matches": candidates
    }


@router.get("/list-subfolders")
def list_subfolders(path: str | None = None):
    """
    List subfolders for an interactive in-modal folder navigator.
    If path is empty or 'root', returns all logical drives on the PC.
    """
    if not path or path.strip() in {"", "/", "root", "drives"}:
        drives = _get_windows_drives()
        return {
            "ok": True,
            "current_path": "",
            "name": "This PC",
            "parent": None,
            "is_root": True,
            "folders": [
                {
                    "name": d["name"],
                    "path": d["path"],
                    "icon": d.get("icon", "💾"),
                    "is_drive": True
                }
                for d in drives
            ]
        }

    clean = path.strip().strip('"').strip("'")
    if clean.isalpha() and len(clean) == 1:
        clean = f"{clean.upper()}:\\"
    elif len(clean) == 2 and clean[1] == ":" and clean[0].isalpha():
        clean = f"{clean[0].upper()}:\\"

    resolved = os.path.abspath(clean)
    if not os.path.isdir(resolved):
        return {"ok": False, "error": f"Not a valid directory: {path}"}

    subfolders = []
    try:
        entries = sorted(os.scandir(resolved), key=lambda e: e.name.lower())
        for entry in entries:
            if entry.is_dir():
                name = entry.name
                if name in EXCLUDED_DIR_NAMES or name.startswith("$"):
                    continue
                if name.startswith(".") and name not in {".github"}:
                    continue
                subfolders.append({
                    "name": name,
                    "path": entry.path.replace("\\", "/"),
                    "icon": "📁"
                })
    except PermissionError:
        return {"ok": False, "error": f"Permission denied for {path}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    norm_resolved = resolved.replace("\\", "/")
    parent = os.path.dirname(resolved).replace("\\", "/")
    if parent == norm_resolved:
        parent = ""  # Parent of drive root (e.g. D:/) is This PC root

    return {
        "ok": True,
        "current_path": norm_resolved,
        "name": os.path.basename(resolved) or resolved,
        "parent": parent,
        "is_root": False,
        "folders": subfolders
    }


@router.post("/validate-folder")
def validate_folder(req: ValidateFolderRequest):
    """Validate that a folder exists on the user's PC and return its metadata."""
    clean = req.folder_path.strip().strip('"').strip("'")
    if not clean:
        return {"ok": False, "error": "Folder path cannot be empty"}

    # Handle drive letters like "f", "F", "f:", "F:", "f:/", "f:\\"
    if clean.isalpha() and len(clean) == 1:
        clean = f"{clean.upper()}:\\"
    elif len(clean) == 2 and clean[1] == ":" and clean[0].isalpha():
        clean = f"{clean[0].upper()}:\\"
    elif len(clean) == 3 and clean[1] == ":" and clean[2] in {"/", "\\"}:
        clean = f"{clean[0].upper()}:\\"

    resolved = os.path.abspath(clean)

    # If not found directly, check if user provided a folder name without drive prefix (e.g. "Java")
    if not os.path.exists(resolved):
        candidates = _find_folder_candidates(clean)
        if len(candidates) == 1:
            resolved = os.path.abspath(candidates[0])
        elif len(candidates) > 1:
            # Prioritize D: or active drive
            d_matches = [c for c in candidates if c.lower().startswith("d:")]
            if d_matches:
                resolved = os.path.abspath(d_matches[0])
            else:
                resolved = os.path.abspath(candidates[0])

    if not os.path.exists(resolved):
        return {"ok": False, "error": f"Path does not exist: {req.folder_path}"}
    if not os.path.isdir(resolved):
        return {"ok": False, "error": f"Path is a file, not a directory: {req.folder_path}"}

    try:
        # Check read access
        os.scandir(resolved)
    except PermissionError:
        return {"ok": False, "error": f"Permission denied to access folder: {req.folder_path}"}
    except Exception as exc:
        return {"ok": False, "error": f"Error accessing folder: {exc}"}

    name = os.path.basename(resolved) or resolved
    return {
        "ok": True,
        "path": resolved.replace("\\", "/"),
        "name": name,
        "parent": os.path.dirname(resolved).replace("\\", "/")
    }


@router.get("/tree")
def get_file_tree(
    scope: Literal["workspace", "project", "custom"] = "workspace",
    custom_root: str | None = None
) -> dict:
    """Return the nested file tree for the requested scope/folder."""
    root_dir = _get_target_root(scope, custom_root)

    def _build_tree(current_dir: str, depth: int = 0) -> list[dict]:
        # Limit deep recursion to 6 levels to keep UI super fast
        if depth > 6:
            return []
        nodes = []
        try:
            entries = sorted(os.scandir(current_dir), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return nodes

        for entry in entries:
            name = entry.name
            if name in EXCLUDED_DIR_NAMES:
                continue
            if name.startswith(".") and name not in {".env", ".gitignore"}:
                continue

            rel = _to_rel_path(entry.path, scope, custom_root)

            if entry.is_dir():
                children = _build_tree(entry.path, depth + 1)
                nodes.append({
                    "name": name,
                    "path": rel,
                    "is_dir": True,
                    "children": children
                })
            else:
                _, ext = os.path.splitext(name)
                try:
                    size = entry.stat().st_size
                except Exception:
                    size = 0
                nodes.append({
                    "name": name,
                    "path": rel,
                    "is_dir": False,
                    "size": size,
                    "ext": ext.lower()
                })
        return nodes

    tree = _build_tree(root_dir)
    return {
        "scope": scope,
        "root_name": os.path.basename(root_dir) or root_dir,
        "root_path": root_dir.replace("\\", "/"),
        "tree": tree
    }


@router.get("/file", response_model=FileContentResponse)
def read_file_content(
    path: str,
    scope: Literal["workspace", "project", "custom"] = "workspace",
    custom_root: str | None = None
):
    """Read text content of a file."""
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="Path is required")

    safe_path = _resolve_safe_path(path, scope, custom_root)
    if not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    stat = os.stat(safe_path)
    if stat.st_size > MAX_EDIT_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds editable limit ({stat.st_size} > {MAX_EDIT_FILE_SIZE} bytes)"
        )

    try:
        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")

    root_dir = _get_target_root(scope, custom_root)
    return FileContentResponse(
        path=path.replace("\\", "/"),
        name=os.path.basename(safe_path),
        content=content,
        size=stat.st_size,
        read_only=False,
        scope=scope,
        root_path=root_dir.replace("\\", "/")
    )


@router.post("/file")
def save_file(req: SaveFileRequest):
    """Save/update existing or new file content."""
    safe_path = _resolve_safe_path(req.path, req.scope, req.custom_root)

    # Ensure parent directory exists
    parent = os.path.dirname(safe_path)
    os.makedirs(parent, exist_ok=True)

    try:
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(req.content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    return {
        "ok": True,
        "path": req.path.replace("\\", "/"),
        "size": len(req.content.encode("utf-8")),
        "message": "File saved successfully"
    }


@router.post("/create-file")
def create_new_file(req: CreateFileRequest):
    """Create a new file. Fails if file already exists."""
    safe_path = _resolve_safe_path(req.path, req.scope, req.custom_root)

    if os.path.exists(safe_path):
        raise HTTPException(status_code=409, detail=f"File already exists: {req.path}")

    parent = os.path.dirname(safe_path)
    os.makedirs(parent, exist_ok=True)

    try:
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(req.content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create file: {exc}")

    return {"ok": True, "path": req.path.replace("\\", "/")}


@router.post("/create-folder")
def create_new_folder(req: CreateFolderRequest):
    """Create a new directory."""
    safe_path = _resolve_safe_path(req.path, req.scope, req.custom_root)

    if os.path.exists(safe_path):
        raise HTTPException(status_code=409, detail=f"Directory already exists: {req.path}")

    try:
        os.makedirs(safe_path, exist_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create directory: {exc}")

    return {"ok": True, "path": req.path.replace("\\", "/")}


@router.post("/rename")
def rename_item(req: RenameRequest):
    """Rename or move a file or directory."""
    old_safe = _resolve_safe_path(req.old_path, req.scope, req.custom_root)
    new_safe = _resolve_safe_path(req.new_path, req.scope, req.custom_root)

    if not os.path.exists(old_safe):
        raise HTTPException(status_code=404, detail=f"Source path not found: {req.old_path}")

    if os.path.exists(new_safe):
        raise HTTPException(status_code=409, detail=f"Destination already exists: {req.new_path}")

    # Ensure target parent directory exists
    os.makedirs(os.path.dirname(new_safe), exist_ok=True)

    try:
        os.rename(old_safe, new_safe)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rename failed: {exc}")

    return {"ok": True, "old_path": req.old_path, "new_path": req.new_path}


@router.post("/delete")
def delete_item(req: DeleteRequest):
    """Delete a file or folder."""
    safe_path = _resolve_safe_path(req.path, req.scope, req.custom_root)

    # Safety: Cannot delete the root directory itself
    root = _get_target_root(req.scope, req.custom_root)
    if os.path.samefile(safe_path, root):
        raise HTTPException(status_code=400, detail="Cannot delete root directory")

    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    try:
        if os.path.isdir(safe_path):
            shutil.rmtree(safe_path)
        else:
            os.remove(safe_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")

    return {"ok": True, "path": req.path}


@router.post("/batch-create")
def batch_create_files(req: BatchCreateRequest):
    """Create or overwrite multiple files at once across any formats and directories."""
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided to create")

    created = []
    errors = []

    for item in req.files:
        clean_path = item.path.strip()
        if not clean_path:
            continue
        try:
            safe_path = _resolve_safe_path(clean_path, req.scope, req.custom_root)
            parent = os.path.dirname(safe_path)
            os.makedirs(parent, exist_ok=True)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(item.content)
            created.append({
                "path": clean_path.replace("\\", "/"),
                "name": os.path.basename(safe_path),
                "size": len(item.content.encode("utf-8"))
            })
        except Exception as exc:
            errors.append({"path": clean_path, "error": str(exc)})

    return {
        "ok": len(errors) == 0,
        "created": created,
        "errors": errors,
        "total": len(req.files)
    }


BINARY_EXTENSIONS = {
    ".pyc", ".pyd", ".exe", ".dll", ".so", ".dylib", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".bmp",
    ".mp3", ".wav", ".ogg", ".mp4", ".mov", ".avi", ".webm",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".sqlite", ".db", ".sqlite3"
}


@router.post("/search")
def search_project(req: GlobalSearchRequest):
    """
    Search recursively across files in the active scope.
    Returns matching lines, line numbers, and offsets for highlighting.
    """
    import fnmatch
    query = (req.query or "").strip()
    if not query:
        return {"ok": True, "total_matches": 0, "total_files": 0, "query": "", "results": []}

    target_root = _get_target_root(req.scope, req.custom_root)
    flags = 0 if req.case_sensitive else re.IGNORECASE

    if req.is_regex:
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex: {e}")
    else:
        escaped = re.escape(query)
        if req.whole_word:
            escaped = rf"\b{escaped}\b"
        pattern = re.compile(escaped, flags)

    include_globs = [g.strip() for g in req.include_pattern.split(",") if g.strip()] if req.include_pattern else []
    exclude_globs = [g.strip() for g in req.exclude_pattern.split(",") if g.strip()] if req.exclude_pattern else []

    results = []
    total_matches = 0
    total_files = 0

    for root, dirs, files in os.walk(target_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]

        for file_name in sorted(files):
            if total_matches >= req.max_results:
                break

            _, ext = os.path.splitext(file_name)
            if ext.lower() in BINARY_EXTENSIONS:
                continue

            rel_file_path = os.path.relpath(os.path.join(root, file_name), target_root).replace("\\", "/")

            if include_globs and not any(fnmatch.fnmatch(file_name, g) or fnmatch.fnmatch(rel_file_path, g) for g in include_globs):
                continue
            if exclude_globs and any(fnmatch.fnmatch(file_name, g) or fnmatch.fnmatch(rel_file_path, g) for g in exclude_globs):
                continue

            full_path = os.path.join(root, file_name)
            try:
                if os.path.getsize(full_path) > 5 * 1024 * 1024:
                    continue

                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    file_matches = []
                    for line_idx, line in enumerate(f, start=1):
                        for m in pattern.finditer(line):
                            total_matches += 1
                            file_matches.append({
                                "line_number": line_idx,
                                "line_content": line.rstrip("\r\n"),
                                "match_start": m.start(),
                                "match_end": m.end()
                            })
                            if total_matches >= req.max_results:
                                break
                        if total_matches >= req.max_results:
                            break

                    if file_matches:
                        total_files += 1
                        results.append({
                            "file_path": rel_file_path,
                            "file_name": file_name,
                            "matches": file_matches
                        })
            except Exception:
                continue

    return {
        "ok": True,
        "total_matches": total_matches,
        "total_files": total_files,
        "query": query,
        "results": results
    }


@router.post("/batch-replace")
def batch_replace(req: BatchReplaceRequest):
    """
    Execute search and replace across files in the active scope.
    Supports single file or project-wide batch replacement.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    target_root = _get_target_root(req.scope, req.custom_root)
    flags = 0 if req.case_sensitive else re.IGNORECASE

    if req.is_regex:
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex: {e}")
    else:
        escaped = re.escape(query)
        if req.whole_word:
            escaped = rf"\b{escaped}\b"
        pattern = re.compile(escaped, flags)

    total_replaced = 0
    affected_files = []

    files_to_process = []
    if req.files:
        for f in req.files:
            clean = f.strip().lstrip("/\\")
            full_p = os.path.normpath(os.path.join(target_root, clean))
            if full_p.startswith(target_root) and os.path.isfile(full_p):
                files_to_process.append((clean, full_p))
    else:
        for root, dirs, files in os.walk(target_root):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]
            for file_name in files:
                _, ext = os.path.splitext(file_name)
                if ext.lower() in BINARY_EXTENSIONS:
                    continue
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, target_root).replace("\\", "/")
                files_to_process.append((rel_path, full_path))

    for rel_path, full_path in files_to_process:
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            new_content, count = pattern.subn(req.replacement, content)
            if count > 0:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                total_replaced += count
                affected_files.append(rel_path)
        except Exception as exc:
            logger.warning(f"Error replacing in {full_path}: {exc}")

    return {
        "ok": True,
        "replaced_occurrences": total_replaced,
        "affected_files": affected_files,
        "message": f"Successfully replaced {total_replaced} occurrence(s) across {len(affected_files)} file(s)."
    }


# ---------------------------------------------------------------------------
# Phase 5: Git Source Control & AI Commit Messages
# ---------------------------------------------------------------------------

HUNK_PATTERN = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", re.MULTILINE)


def _run_git(args: list[str], cwd: str, timeout: int = 20) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception as exc:
        return -1, "", str(exc)


def _parse_diff_hunks(diff_text: str) -> list[dict]:
    decorations = []
    for m in HUNK_PATTERN.finditer(diff_text):
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1

        if old_count == 0 and new_count > 0:
            for l in range(new_start, new_start + new_count):
                decorations.append({"line": l, "type": "added"})
        elif old_count > 0 and new_count == 0:
            del_line = max(1, new_start)
            decorations.append({"line": del_line, "type": "deleted"})
        else:
            for l in range(new_start, new_start + new_count):
                decorations.append({"line": l, "type": "modified"})
    return decorations


def _resolve_git_repo_path(scope: str = "workspace", custom_root: str | None = None) -> tuple[str, bool, str]:
    """
    Resolve the active git repository.
    Returns (cwd, is_external_workspace, repo_name).
    1. If the target root (e.g. custom folder) is a git repository, use it.
    2. Otherwise, use the user's configured external GIT_WORKSPACE (e.g. ~/Desktop/loki_git_workspace).
    """
    from config import GIT_WORKSPACE
    target = _get_target_root(scope, custom_root)
    code, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], target)
    if code == 0:
        return target, False, os.path.basename(target)

    # Check external GIT_WORKSPACE
    if GIT_WORKSPACE:
        if os.path.isdir(GIT_WORKSPACE):
            code_ws, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], GIT_WORKSPACE)
            if code_ws != 0:
                try:
                    from tools.git_pusher import init_git_workspace
                    init_git_workspace(GIT_WORKSPACE)
                    code_ws, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], GIT_WORKSPACE)
                except Exception:
                    pass
            if code_ws == 0:
                return GIT_WORKSPACE, True, os.path.basename(GIT_WORKSPACE)

    return target, False, os.path.basename(target)


@router.post("/git/status")
def git_status(req: GitStatusRequest):
    """
    Get git status for the current project root or external git workspace, including branch, staged/unstaged
    files, change count, and numstat additions/deletions.
    """
    cwd, is_external, repo_name = _resolve_git_repo_path(req.scope, req.custom_root)
    code, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if code != 0:
        return {
            "ok": True,
            "is_git_repo": False,
            "branch": "",
            "files": [],
            "total_changes": 0,
            "clean": True,
            "repo_name": repo_name,
            "repo_path": cwd,
            "is_external_workspace": is_external
        }

    # Detect current branch
    _, branch, _ = _run_git(["branch", "--show-current"], cwd)
    branch = branch.strip()
    if not branch:
        _, branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        branch = branch.strip() or "HEAD"

    # Status porcelain
    _, porcelain, _ = _run_git(["status", "--porcelain=v1", "-uall"], cwd)

    # Numstat diff
    numstat_map = {}
    _, numstat_out, _ = _run_git(["diff", "HEAD", "--numstat"], cwd)
    if not numstat_out.strip():
        _, numstat_out, _ = _run_git(["diff", "--numstat"], cwd)
    for line in numstat_out.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            ins = int(parts[0]) if parts[0].isdigit() else 0
            dels = int(parts[1]) if parts[1].isdigit() else 0
            f_path = parts[2].replace("\\", "/")
            numstat_map[f_path] = (ins, dels)

    files = []
    for line in porcelain.splitlines():
        if len(line) < 3:
            continue
        x = line[0]
        y = line[1]
        rel_path = line[3:].strip().replace("\\", "/")
        if " -> " in rel_path:
            rel_path = rel_path.split(" -> ")[-1]

        if x == "?" and y == "?":
            status = "U"
            label = "Untracked"
            staged = False
        elif x == "D" or y == "D":
            status = "D"
            label = "Deleted"
            staged = (x == "D")
        elif x == "A" or y == "A":
            status = "A"
            label = "Added"
            staged = (x == "A")
        elif x == "R" or y == "R":
            status = "R"
            label = "Renamed"
            staged = (x == "R")
        else:
            status = "M"
            label = "Modified"
            staged = (x != " ")

        ins, dels = numstat_map.get(rel_path, (0, 0))
        files.append({
            "path": rel_path,
            "file_name": os.path.basename(rel_path),
            "status": status,
            "status_label": label,
            "staged": staged,
            "insertions": ins,
            "deletions": dels
        })

    return {
        "ok": True,
        "is_git_repo": True,
        "branch": branch,
        "files": files,
        "total_changes": len(files),
        "clean": len(files) == 0,
        "repo_name": repo_name,
        "repo_path": cwd,
        "is_external_workspace": is_external
    }


@router.post("/git/diff")
def git_diff(req: GitDiffRequest):
    """
    Get git diff and parsed line gutter decorations for a specific file or the entire repository.
    """
    cwd, _, _ = _resolve_git_repo_path(req.scope, req.custom_root)
    code, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if code != 0:
        return {"ok": False, "error": "Not a git repository", "decorations": [], "raw_diff": ""}

    file_arg = ["--", req.file_path] if req.file_path else []

    # Get zero-context unified diff
    _, unstaged, _ = _run_git(["diff", "-U0"] + file_arg, cwd)
    _, staged, _ = _run_git(["diff", "--cached", "-U0"] + file_arg, cwd)
    raw_diff = (staged + ("\n" if staged and unstaged else "") + unstaged).strip()

    decorations = _parse_diff_hunks(raw_diff)

    return {
        "ok": True,
        "file_path": req.file_path or "",
        "raw_diff": raw_diff,
        "decorations": decorations
    }


@router.post("/git/commit-message")
def git_commit_message(req: GitCommitMessageRequest):
    """
    Analyze staged and unstaged git diffs using AI to generate a clean, conventional commit message.
    """
    cwd, _, _ = _resolve_git_repo_path(req.scope, req.custom_root)
    code, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if code != 0:
        raise HTTPException(status_code=400, detail="Not a git repository")

    _, status_out, _ = _run_git(["status", "--porcelain=v1", "-uall"], cwd)
    if not status_out.strip():
        return {
            "ok": True,
            "commit_message": "chore: no changes detected",
            "title": "chore: no changes detected",
            "body": ""
        }

    _, staged_diff, _ = _run_git(["diff", "--cached"], cwd)
    _, unstaged_diff, _ = _run_git(["diff"], cwd)
    combined_diff = (staged_diff + "\n" + unstaged_diff).strip()
    if not combined_diff:
        combined_diff = f"Untracked / Added files:\n{status_out}"

    if len(combined_diff) > 3000:
        combined_diff = combined_diff[:3000] + "\n...[truncated diff]"

    system_prompt = (
        "You are an expert software engineer and git version control specialist. "
        "Generate a concise, professional Conventional Commit message based on the provided git diff and changed files.\n"
        "RULES:\n"
        "1. First line MUST follow Conventional Commits format: <type>(<optional scope>): <imperative subject in lowercase> "
        "(types: feat, fix, refactor, style, docs, chore, perf, test).\n"
        "2. Keep the first line under 72 characters.\n"
        "3. Follow with an empty line, then 2-3 short bullet points starting with '- ' summarizing what changed.\n"
        "4. Output ONLY the commit message. No markdown code block fences around the entire response."
    )

    hint_note = f"\nUser hint: {req.hint.strip()}" if req.hint and req.hint.strip() else ""
    user_prompt = f"Git Status:\n{status_out}\n\nGit Diff:\n{combined_diff}{hint_note}"

    commit_msg = ""
    try:
        from models.coding import _call
        raw_msg = _call(system_prompt, user_prompt, think=False)
        commit_msg = raw_msg.strip()
        if commit_msg.startswith("```"):
            lines = commit_msg.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                commit_msg = "\n".join(lines[1:-1]).strip()
            elif commit_msg.startswith("```") and commit_msg.endswith("```"):
                commit_msg = commit_msg.strip("`").strip()
    except Exception as exc:
        logger.warning(f"Failed to generate commit message via LLM: {exc}")

    if not commit_msg:
        first_file = status_out.strip().splitlines()[0].split()[-1]
        _, ext = os.path.splitext(first_file)
        c_type = "feat(ui)" if ext in {".html", ".css", ".js"} else "feat(backend)" if ext == ".py" else "chore"
        commit_msg = f"{c_type}: update {first_file}\n\n- Updated {first_file}\n- Refined project configuration and components"

    parts = commit_msg.split("\n\n", 1)
    title = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""

    return {
        "ok": True,
        "commit_message": commit_msg,
        "title": title,
        "body": body
    }


@router.post("/git/commit")
def git_commit(req: GitCommitRequest):
    """
    Stage specified files (or all changes) and execute a git commit with the given message.
    """
    cwd, _, _ = _resolve_git_repo_path(req.scope, req.custom_root)
    code, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if code != 0:
        raise HTTPException(status_code=400, detail="Not a git repository")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Commit message cannot be empty")

    if req.files:
        for f in req.files:
            _run_git(["add", "--", f], cwd)
    else:
        _run_git(["add", "-A"], cwd)

    _run_git(["config", "user.name", "LOKI Developer"], cwd)
    _run_git(["config", "user.email", "loki@mindcraft.local"], cwd)

    code, stdout, stderr = _run_git(["commit", "-m", msg], cwd)
    if code != 0:
        if "nothing to commit" in (stdout + stderr).lower():
            return {"ok": True, "message": "Nothing to commit, working tree clean", "committed": False}
        return {"ok": False, "error": stderr or stdout, "committed": False}

    _, rev, _ = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    _, branch, _ = _run_git(["branch", "--show-current"], cwd)

    return {
        "ok": True,
        "committed": True,
        "commit_hash": rev.strip(),
        "branch": branch.strip(),
        "output": stdout.strip(),
        "message": f"Committed {rev.strip()} on {branch.strip()}"
    }


@router.post("/git/init")
def git_init(req: GitInitRequest):
    from config import GIT_WORKSPACE
    cwd = _get_target_root(req.scope, req.custom_root)
    real_cwd = os.path.realpath(cwd)
    real_proj = os.path.realpath(PROJECT_ROOT)
    
    # If the requested target is inside the LOKI project, initialize the external GIT_WORKSPACE instead
    if real_cwd == real_proj or real_cwd.startswith(real_proj + os.sep):
        target_dir = GIT_WORKSPACE
        note = " (LOKI project itself is kept Git-free per safety policy)"
    else:
        target_dir = cwd
        note = ""

    os.makedirs(target_dir, exist_ok=True)
    code, out, err = _run_git(["init"], target_dir)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"Failed to initialize git repository: {err or out}")
    _run_git(["config", "user.name", "LOKI Developer"], target_dir)
    _run_git(["config", "user.email", "loki@mindcraft.local"], target_dir)
    return {"ok": True, "message": f"Initialized Git repository in {target_dir}{note}", "workspace": target_dir}


@router.post("/git/push-file")
def git_push_file(req: GitPushFileRequest):
    """
    Push a single generated file (e.g. p1.py) to the separate external Git workspace.
    LOKI project itself is never touched or initialized with Git.
    """
    from tools.git_pusher import push_file_to_git
    res = push_file_to_git(
        file_path=req.file_path,
        commit_message=req.commit_message,
        git_workspace=req.git_workspace,
    )
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "Git push failed"))
    return res



@router.post("/run")
def run_code(req: RunCodeRequest):
    """Run code in the sandbox across multiple formats (Python, HTML live preview, JS/Node)."""
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="No code provided to execute")

    file_path = (req.file_path or "").lower()
    _, ext = os.path.splitext(file_path)

    # 1. HTML / Web format — Instant Live Web Preview
    if ext in {".html", ".htm"} or ("<html" in req.code.lower() and "<!doctype" in req.code.lower()):
        return {
            "ok": True,
            "exit_code": 0,
            "elapsed_ms": 1.0,
            "stdout": "[HTML Web Application Ready — Live Preview Running]",
            "stderr": "",
            "is_html": True,
            "html_content": req.code,
            "file_path": req.file_path or "index.html",
        }

    # 2. JavaScript / Node.js
    if ext in {".js", ".mjs"}:
        node_bin = shutil.which("node")
        if node_bin:
            import subprocess
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [node_bin, "-e", req.code],
                    capture_output=True,
                    text=True,
                    timeout=req.timeout
                )
                elapsed = (time.perf_counter() - t0) * 1000
                return {
                    "ok": proc.returncode == 0,
                    "exit_code": proc.returncode,
                    "elapsed_ms": round(elapsed, 1),
                    "stdout": proc.stdout or "[Program executed with no output]",
                    "stderr": proc.stderr or ""
                }
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "exit_code": -1,
                    "elapsed_ms": req.timeout * 1000,
                    "stdout": "",
                    "stderr": f"Execution timed out after {req.timeout}s"
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "exit_code": -1,
                    "elapsed_ms": 0,
                    "stdout": "",
                    "stderr": f"Node.js execution error: {exc}"
                }

    # 3. Default: Python sandbox execution
    result = run_python(req.code, timeout=req.timeout)
    return {
        "ok": result.ok,
        "exit_code": result.exit_code,
        "elapsed_ms": result.elapsed_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Integrated Developer Terminal API
# ---------------------------------------------------------------------------

@router.get("/terminal/info")
def terminal_info(
    scope: Literal["workspace", "project", "custom"] = "workspace",
    custom_root: str | None = None
):
    """Return initial terminal environment information."""
    target_root = _get_target_root(scope, custom_root)
    username = "User"
    try:
        username = getpass.getuser()
    except Exception:
        pass

    return {
        "ok": True,
        "cwd": target_root,
        "shell": "powershell",
        "user": username,
        "os": "Windows" if sys.platform.startswith("win") else sys.platform,
    }


def _clean_clixml(raw: str) -> str:
    """
    Remove PowerShell CLIXML serialization noise (such as progress records
    'Preparing modules for first use', etc.) and decode real error strings if present.
    """
    if not raw or "#< CLIXML" not in raw:
        return (raw or "").strip("\r\n")

    # If it is purely progress noise without errors, discard completely
    if '<Obj S="progress"' in raw and '<S S="Error"' not in raw:
        non_xml = re.sub(r'#<\s*CLIXML[\s\S]*?<\/Objs>', '', raw).strip("\r\n")
        return non_xml

    # Extract all real errors if encoded in CLIXML: <S S="Error">...</S>
    def _extract_clixml(match):
        xml_blob = match.group(0)
        error_chunks = re.findall(r'<S S="Error">(.*?)</S>', xml_blob, re.DOTALL)
        if not error_chunks:
            return ""
        cleaned = []
        for chunk in error_chunks:
            clean_chunk = re.sub(
                r'_x([0-9a-fA-F]{4})_',
                lambda m: chr(int(m.group(1), 16)),
                chunk
            )
            cleaned.append(clean_chunk)
        return "".join(cleaned).strip("\r\n")

    cleaned_text = re.sub(r'#<\s*CLIXML[\s\S]*?<\/Objs>', _extract_clixml, raw)
    return cleaned_text.strip("\r\n")


@router.post("/terminal/exec")
def terminal_exec(req: TerminalExecRequest):
    """
    Execute a shell command inside the user's active project or custom folder.
    Supports directory changes (cd), command output streaming, and error capture.
    """
    raw_cmd = (req.command or "").strip()

    # Determine starting working directory
    if req.cwd and req.cwd.strip():
        cand = os.path.abspath(req.cwd.strip())
        if os.path.isdir(cand):
            start_cwd = cand
        else:
            start_cwd = _get_target_root(req.scope, req.custom_root)
    else:
        start_cwd = _get_target_root(req.scope, req.custom_root)

    # Empty command
    if not raw_cmd:
        return {
            "ok": True,
            "exit_code": 0,
            "elapsed_ms": 0.0,
            "stdout": "",
            "stderr": "",
            "cwd": start_cwd,
            "is_clear": False,
        }

    # Handle cls / clear
    if raw_cmd.lower() in {"cls", "clear"}:
        return {
            "ok": True,
            "exit_code": 0,
            "elapsed_ms": 0.0,
            "stdout": "",
            "stderr": "",
            "cwd": start_cwd,
            "is_clear": True,
        }

    t0 = time.perf_counter()

    # Windows PowerShell execution with Base64 EncodedCommand
    if sys.platform.startswith("win"):
        ps_script = f"""
$ProgressPreference = 'SilentlyContinue'
$InformationPreference = 'SilentlyContinue'
$WarningPreference = 'Continue'
$ErrorActionPreference = 'Continue'
try {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}} catch {{}}
Set-Location -LiteralPath {json.dumps(start_cwd)}
try {{
    {raw_cmd}
}} catch {{
    [Console]::Error.WriteLine($_)
}} finally {{
    Write-Output "`n__LOKI_PWD__`n"
    (Get-Location).Path
}}
"""
        try:
            encoded = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                cwd=start_cwd,
                timeout=req.timeout,
                encoding="utf-8",
                errors="replace"
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            raw_stdout = proc.stdout or ""
            raw_stderr = proc.stderr or ""

            parts = raw_stdout.split("__LOKI_PWD__")
            cmd_output = _clean_clixml(parts[0].strip("\r\n"))
            new_cwd = parts[1].strip() if len(parts) > 1 and parts[1].strip() else start_cwd
            if not os.path.isdir(new_cwd):
                new_cwd = start_cwd

            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "elapsed_ms": elapsed,
                "stdout": cmd_output,
                "stderr": _clean_clixml(raw_stderr),
                "cwd": new_cwd,
                "is_clear": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "exit_code": -1,
                "elapsed_ms": req.timeout * 1000,
                "stdout": "",
                "stderr": f"Command timed out after {req.timeout} seconds",
                "cwd": start_cwd,
                "is_clear": False,
            }
        except Exception as exc:
            return {
                "ok": False,
                "exit_code": -1,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "stdout": "",
                "stderr": f"Execution error: {exc}",
                "cwd": start_cwd,
                "is_clear": False,
            }
    else:
        # Fallback for Linux / MacOS bash
        sh_cmd = f"cd {json.dumps(start_cwd)} && {raw_cmd} ; echo '__LOKI_PWD__' ; pwd"
        try:
            proc = subprocess.run(
                ["/bin/bash", "-c", sh_cmd],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                cwd=start_cwd,
                timeout=req.timeout,
                errors="replace"
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            raw_stdout = proc.stdout or ""
            raw_stderr = proc.stderr or ""
            parts = raw_stdout.split("__LOKI_PWD__")
            cmd_output = parts[0].strip("\r\n")
            new_cwd = parts[1].strip() if len(parts) > 1 and parts[1].strip() else start_cwd
            if not os.path.isdir(new_cwd):
                new_cwd = start_cwd

            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "elapsed_ms": elapsed,
                "stdout": cmd_output,
                "stderr": raw_stderr.strip("\r\n"),
                "cwd": new_cwd,
                "is_clear": False,
            }
        except Exception as exc:
            return {
                "ok": False,
                "exit_code": -1,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "stdout": "",
                "stderr": str(exc),
                "cwd": start_cwd,
                "is_clear": False,
            }


@router.post("/terminal/fix")
def terminal_fix(req: TerminalFixRequest):
    """
    Analyze a failed terminal command and generate:
    1. A concise, crystal-clear explanation of the error
    2. A suggested terminal fix command ready to run in 1-click
    """
    cmd = (req.command or "").strip()
    err = (req.error_output or "").strip()
    if not err and not cmd:
        raise HTTPException(status_code=400, detail="Command or error output is required")

    system_prompt = (
        "You are an expert DevOps engineer and CLI troubleshooting specialist in LOKI Code Studio. "
        "A user ran a shell command that failed or produced an error. "
        "Analyze the command and the exact error output, then output a JSON object with two fields:\n"
        "1. 'explanation': 1 to 2 clear sentences explaining why it failed and how to resolve it.\n"
        "2. 'suggested_command': The exact, copy-pasteable command the user should run to fix or retry the operation "
        "(e.g. 'pip install <missing_pkg>', 'npm install <pkg>', or corrected command syntax).\n"
        "Do NOT include markdown formatting or quotes around the JSON. Return valid JSON only."
    )

    user_prompt = f"""Failed Command:
{cmd}

Error Output:
{err}

Working Directory: {req.cwd or 'workspace'}
"""

    explanation = ""
    suggested_command = ""

    # Try LLM call first
    try:
        raw_res = _call(system_prompt, user_prompt, think=False)
        cleaned = raw_res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        explanation = data.get("explanation", "").strip()
        suggested_command = data.get("suggested_command", "").strip()
    except Exception as exc:
        logger.warning(f"Error calling coding model for terminal fix: {exc}")

    # Robust heuristic fallback if model returned empty or errored
    if not explanation or not suggested_command:
        if "No module named" in err:
            m = re.search(r"No module named ['\"]?([a-zA-Z0-9_-]+)['\"]?", err)
            if m:
                pkg = m.group(1)
                explanation = f"Python module '{pkg}' is not installed in the current environment."
                suggested_command = f"pip install {pkg}"
        elif "is not recognized as the name of a cmdlet" in err or "command not found" in err:
            explanation = "The requested command or executable could not be found in PATH or the current folder."
            parts = cmd.split()
            first_cmd = parts[0] if parts else ""
            if first_cmd in ["python3", "py"]:
                suggested_command = f"python {' '.join(parts[1:])}"
            elif first_cmd in ["pip3"]:
                suggested_command = f"pip {' '.join(parts[1:])}"
        elif "cannot find the file specified" in err or "No such file or directory" in err:
            explanation = "The specified file or script could not be located at the requested path."
        elif not explanation:
            explanation = f"Command exited with an error. Review the error trace above."

    return {
        "ok": True,
        "explanation": explanation,
        "suggested_command": suggested_command,
        "original_command": cmd,
    }


# ---------------------------------------------------------------------------
# Live Web Preview Engine (Serves real linked HTML, CSS, JS, Assets)
# ---------------------------------------------------------------------------

MIME_OVERRIDE = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
}


@router.get("/preview/{scope_key}")
@router.get("/preview/{scope_key}/{file_path:path}")
def serve_preview_file(scope_key: str, file_path: str = ""):
    """
    Serve real HTML, CSS, JavaScript, media, and assets directly from the active project root
    for realistic in-browser live execution without CORS or missing asset issues.
    """
    scope, custom_root = _decode_scope_key(scope_key)

    clean_path = file_path.strip().lstrip("/\\")
    if not clean_path:
        clean_path = "index.html"

    try:
        safe_path = _resolve_safe_path(clean_path, scope, custom_root)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Invalid path: {exc}")

    # If accessing a directory, look for index.html
    if os.path.isdir(safe_path):
        for candidate in ["index.html", "index.htm", "main.html"]:
            candidate_path = os.path.join(safe_path, candidate)
            if os.path.isfile(candidate_path):
                safe_path = candidate_path
                break
        else:
            raise HTTPException(status_code=404, detail=f"Directory '{clean_path}' contains no index.html")

    if not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail=f"File not found: {clean_path}")

    _, ext = os.path.splitext(safe_path)
    # For HTML files, inject the Sovereign IDE Preview Bridge to forward console.log/warn/error and exceptions
    if ext.lower() in [".html", ".htm"]:
        try:
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                html_text = f.read()

            bridge_script = """
<script id="__loki_preview_bridge">
(function() {
  function serialize(arg) {
    if (arg === null) return 'null';
    if (arg === undefined) return 'undefined';
    if (typeof arg === 'object') {
      try { return JSON.stringify(arg, null, 2); } catch(e) { return String(arg); }
    }
    return String(arg);
  }
  function sendConsole(level, args) {
    try {
      const text = Array.from(args).map(serialize).join(' ');
      window.parent.postMessage({
        type: 'LOKI_PREVIEW_CONSOLE',
        level: level,
        message: text,
        timestamp: new Date().toLocaleTimeString()
      }, '*');
    } catch(e) {}
  }
  const _log = console.log, _warn = console.warn, _error = console.error, _info = console.info;
  console.log = function(...args) { _log.apply(console, args); sendConsole('log', args); };
  console.warn = function(...args) { _warn.apply(console, args); sendConsole('warn', args); };
  console.error = function(...args) { _error.apply(console, args); sendConsole('error', args); };
  console.info = function(...args) { _info.apply(console, args); sendConsole('info', args); };

  window.addEventListener('error', function(e) {
    sendConsole('error', [(e.message || 'Script error') + (e.filename ? ' (' + e.filename + ':' + e.lineno + ')' : '')]);
  });
  window.addEventListener('unhandledrejection', function(e) {
    sendConsole('error', ['Unhandled Promise Rejection: ' + (e.reason ? (e.reason.stack || e.reason) : 'Unknown')]);
  });
})();
</script>
"""
            if "</body>" in html_text:
                html_text = html_text.replace("</body>", f"{bridge_script}\n</body>")
            elif "</html>" in html_text:
                html_text = html_text.replace("</html>", f"{bridge_script}\n</html>")
            else:
                html_text += f"\n{bridge_script}"

            return HTMLResponse(
                content=html_text,
                media_type="text/html; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        except Exception as exc:
            logger.warning(f"Failed to inject preview bridge into {safe_path}: {exc}")

    media_type = MIME_OVERRIDE.get(ext.lower())
    if not media_type:
        guessed, _ = mimetypes.guess_type(safe_path)
        media_type = guessed or "application/octet-stream"

    return FileResponse(
        safe_path,
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/preview-serve/{file_path:path}")
def serve_preview_file_by_query(
    file_path: str,
    scope: Literal["workspace", "project", "custom"] = "workspace",
    custom_root: str | None = None
):
    """Fallback query-parameter based preview endpoint."""
    scope_key = _encode_scope_key(scope, custom_root)
    return serve_preview_file(scope_key, file_path)


@router.post("/ai")
def ai_action(req: AIActionRequest):
    """
    Execute an AI coding assistant operation:
      - 'explain': Explain code in any language
      - 'review': Code review for quality & security
      - 'debug': Fix bug based on code & error traceback
      - 'generate': Write code for user prompt with multi-file and multi-format support
    """
    try:
        if req.action == "explain":
            if not req.code.strip():
                raise HTTPException(status_code=400, detail="Code is required to explain")
            result = explain_code(req.code)

        elif req.action == "review":
            if not req.code.strip():
                raise HTTPException(status_code=400, detail="Code is required to review")
            result = review_code(req.code, context=req.prompt)

        elif req.action == "debug":
            if not req.code.strip():
                raise HTTPException(status_code=400, detail="Code is required to debug")
            result = debug_code(req.code, error=req.error or req.prompt or "Check for issues")

        elif req.action == "generate":
            if not req.prompt.strip():
                raise HTTPException(status_code=400, detail="Prompt description is required")

            p_clean = req.prompt.strip()

            # -------------------------------------------------------------
            # Git Push & Compound Create + Push Orchestration
            # -------------------------------------------------------------
            wants_git_push = bool(re.search(r'\b(?:push(?:\s+it|\s+this)?\s+to\s+git|git\s+push)\b', p_clean, re.IGNORECASE))

            # Check for direct push command (e.g. 'Push p1.py to Git', 'git push p1.py', 'push it to git')
            direct_push_match = re.search(
                r'^\s*(?:git\s+push|push)\s+([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)(?:\s+to\s+(?:git|github|gitlab|remote))?\s*$',
                p_clean,
                re.IGNORECASE
            ) or re.search(
                r'^\s*(?:git\s+push|push)\s+(?:it|this|current\s+file)\s*(?:to\s+(?:git|github|gitlab|remote))?\s*$',
                p_clean,
                re.IGNORECASE
            )

            files_mentioned = re.findall(r'\b([a-zA-Z0-9_.-]+\.(?:py|js|ts|html|css|json|md|txt|sh))\b', p_clean, re.IGNORECASE)
            is_create = any(k in p_clean.lower() for k in ['create', 'write', 'generate', 'make', 'build', 'code', 'implement'])

            # Case 1: Compound "create <file> ... and push to git"
            if wants_git_push and is_create and files_mentioned:
                target_filename = files_mentioned[0]
                # Strip git push phrase so LLM focuses purely on generating the code
                clean_task = re.sub(
                    r'\s*(?:and\s+)?(?:then\s+)?(?:push(?:\s+it|\s+this)?\s+to\s+git|git\s+push(?:\s+it)?)\s*',
                    '',
                    p_clean,
                    flags=re.IGNORECASE
                ).strip()

                system_prompt = (
                    "You are an expert software engineer. "
                    f"Write clean, production-ready code for '{target_filename}'. "
                    "CRITICAL: Output ONLY the actual source code implementation. "
                    "Do NOT output shell scripts, bash commands, or git commands (like git init, git add, git commit). "
                    f"Wrap your complete code in a single markdown code block with header comment '# {target_filename}'."
                )

                from models.coding import _call
                generated_result = _call(
                    system_prompt=system_prompt,
                    user_message=f"Target File: {target_filename}\n\nTask: {clean_task}",
                    think=False
                )

                # Extract code content to save to disk
                code_match = re.search(r'```(?:[a-zA-Z0-9_.:-]*\n)?([\s\S]*?)```', generated_result)
                extracted_code = code_match.group(1).strip() if code_match else generated_result.strip()

                # Save file to workspace and backend/outputs
                save_paths = [
                    os.path.join(WORKSPACE_ROOT, target_filename),
                    os.path.join(PROJECT_ROOT, "backend", "outputs", target_filename),
                ]
                for sp in save_paths:
                    try:
                        os.makedirs(os.path.dirname(sp), exist_ok=True)
                        with open(sp, "w", encoding="utf-8") as f:
                            f.write(extracted_code)
                    except Exception as save_err:
                        logger.warning(f"Could not write generated file to {sp}: {save_err}")

                # Now push the generated file to the external Git workspace
                from tools.git_pusher import push_file_to_git
                push_res = push_file_to_git(target_filename)

                # Ensure generated code has a clean markdown code block
                if not generated_result.strip().startswith("```"):
                    lang = target_filename.split(".")[-1]
                    code_block_display = f"```{lang}:{target_filename}\n{extracted_code}\n```"
                else:
                    code_block_display = generated_result

                if push_res.get("ok"):
                    result_card = (
                        f"{code_block_display}\n\n"
                        f"---\n\n"
                        f"🚀 **LOKI Git Push Operation**\n\n"
                        f"• **Target File:** `{push_res['file']}`\n"
                        f"• **Saved To:** `{push_res['source_path']}`\n"
                        f"• **Git Workspace:** `{push_res['workspace']}`\n"
                        f"• **Staged:** `git add {push_res['file']}` (Success)\n"
                        f"• **Committed:** `{push_res['commit_summary']}`\n"
                        f"• **Push Status:** {push_res['push_status']}\n\n"
                        f"✅ {push_res['message']}"
                    )
                    return {"ok": True, "action": "generate", "result": result_card}
                else:
                    result_card = (
                        f"{code_block_display}\n\n"
                        f"---\n\n"
                        f"❌ **LOKI Git Push Error:**\n{push_res.get('error', 'Push failed')}"
                    )
                    return {"ok": True, "action": "generate", "result": result_card}

            # Case 2: Direct push of an existing file (e.g. "Push p1.py to Git")
            elif direct_push_match or (wants_git_push and not is_create):
                target_filename = None
                if direct_push_match and direct_push_match.lastindex:
                    target_filename = direct_push_match.group(1).strip()
                elif req.file_path:
                    target_filename = os.path.basename(req.file_path)
                elif files_mentioned:
                    target_filename = files_mentioned[0]

                if target_filename:
                    from tools.git_pusher import push_file_to_git
                    push_res = push_file_to_git(target_filename)
                    if push_res.get("ok"):
                        result_card = (
                            f"🚀 **LOKI Git Push Operation**\n\n"
                            f"• **Target File:** `{push_res['file']}`\n"
                            f"• **Source Path:** `{push_res['source_path']}`\n"
                            f"• **Git Workspace:** `{push_res['workspace']}`\n"
                            f"• **Staged:** `git add {push_res['file']}` (Success)\n"
                            f"• **Committed:** `{push_res['commit_summary']}`\n"
                            f"• **Push Status:** {push_res['push_status']}\n\n"
                            f"✅ {push_res['message']}"
                        )
                        return {"ok": True, "action": "generate", "result": result_card}
                    else:
                        err_msg = push_res.get("error", "Failed to push file")
                        result_card = f"❌ **LOKI Git Push Failed:**\n\n{err_msg}"
                        return {"ok": True, "action": "generate", "result": result_card}

            # Detect if prompt is asking for multi-file or specific language formats
            p_lower = req.prompt.lower()
            is_multifile = any(k in p_lower for k in [
                "files", "multiple files", "html and css", "frontend and backend",
                "index.html", "style.css", "app.js", "fullstack", "website", "project"
            ])

            multi_file_instruction = (
                "You are an expert full-stack software engineer and polyglot developer. "
                "Produce clean, production-ready code. "
                "CRITICAL: Do NOT output git commands or shell scripts (e.g. git init, git add, git commit) unless explicitly requested. Output the actual implementation code for the files. "
                "CRITICAL REQUIREMENT: If the request requires multiple files (or multiple formats such as HTML, CSS, JavaScript, Python, JSON, SQL), "
                "provide EACH file in its own distinct markdown code block with the exact filename clearly stated in a header comment or markdown title (e.g. `### filename.ext` or `<!-- filename.ext -->` or `// filename.ext` or `# filename.ext`). "
                "Make sure every file is complete, functional, and ready to save directly to disk."
            )

            from models.coding import _call
            target_lang = req.language or ("Fullstack / Multi-format" if is_multifile else "Auto-detect from request")
            result = _call(
                system_prompt=multi_file_instruction,
                user_message=f"Target: {target_lang}\n\nTask: {req.prompt}",
                think=False
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

        return {"ok": True, "action": req.action, "result": result}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("AI action failed in IDE API")
        return {"ok": False, "error": str(exc)}


@router.post("/inline-edit")
def inline_edit(req: InlineEditRequest):
    """
    Perform a targeted inline code transformation (Cursor / Windsurf Ctrl+K style).
    Takes a snippet of code (with optional surrounding context) and returns the modified replacement.
    """
    try:
        from models.coding import _call

        system_prompt = (
            "You are an expert AI code editor and refactoring engine embedded in a developer IDE. "
            "You receive an instruction and a target code snippet (with surrounding context). "
            "Your task is to rewrite the target code according to the instruction, or generate the requested code. "
            "CRITICAL RULES:\n"
            "1. Output ONLY the raw replacement code. "
            "2. Do NOT wrap output in markdown code blocks (NO ``` or ```language). "
            "3. Do NOT include explanations, greetings, or notes before or after the code. "
            "4. Preserve consistent indentation, naming style, and conventions from the surrounding context. "
            "5. If generating new code without selection, output ONLY the code to insert."
        )

        user_msg = (
            f"Language: {req.language}\n"
            f"File: {req.file_path or 'active file'}\n"
            f"Instruction: {req.instruction}\n\n"
        )
        if req.context_before.strip():
            user_msg += f"--- Context Before Selection ---\n{req.context_before[-500:]}\n\n"
        user_msg += f"--- Target Selected Code to Replace ---\n{req.selected_code or '(No selection, cursor insertion)'}\n\n"
        if req.context_after.strip():
            user_msg += f"--- Context After Selection ---\n{req.context_after[:500]}\n"

        result = _call(
            system_prompt=system_prompt,
            user_message=user_msg,
            think=False
        )

        clean = result.strip()
        # Clean any accidental markdown backticks
        if clean.startswith("```"):
            lines = clean.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                clean = "\n".join(lines[1:-1])
            elif clean.startswith("```") and clean.endswith("```"):
                clean = clean.strip("`")

        return {
            "ok": True,
            "replacement": clean,
            "original": req.selected_code
        }
    except Exception as exc:
        logger.exception("Inline edit failed in IDE API")
        return {"ok": False, "error": str(exc), "replacement": "", "original": req.selected_code}

