"""
backend/tools/git_pusher.py

Isolated Git push capability for LOKI Co-Pilot.

SAFETY CRITICAL:
  - NEVER initializes Git in the LOKI project (d:\\mindcraft).
  - NEVER puts the LOKI project itself on Git.
  - ONLY pushes generated/output files (e.g. "p1.py", "p2.py") by copying
    them to a completely separate Git workspace.

Workflow:
  1. Find the generated file (e.g. in outputs/, workspace/, root, or Downloads).
  2. Copy it to the separate Git workspace.
  3. Run 'git add <file>'.
  4. Commit it with an automatic or custom message.
  5. Run 'git push' (with graceful handling if no remote is configured yet).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import logging
from typing import Optional

from config import GIT_WORKSPACE, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Search paths for generated/output files
DEFAULT_SEARCH_DIRS = [
    os.path.join(PROJECT_ROOT, "backend", "outputs"),
    os.path.join(PROJECT_ROOT, "workspace"),
    PROJECT_ROOT,
    os.path.join(os.path.expanduser("~"), "Downloads"),
]


def is_safety_violation(target_workspace: str) -> bool:
    """
    Returns True if target_workspace is inside or identical to PROJECT_ROOT.
    Ensures the LOKI project is NEVER put on Git.
    """
    real_ws = os.path.realpath(os.path.abspath(target_workspace))
    real_proj = os.path.realpath(os.path.abspath(PROJECT_ROOT))
    return real_ws == real_proj or real_ws.startswith(real_proj + os.sep)


def find_generated_file(file_arg: str) -> Optional[str]:
    """
    Locate a generated file by name or relative path.
    Searches:
      1. Direct path (if absolute or relative to cwd/PROJECT_ROOT)
      2. backend/outputs/<filename>
      3. workspace/<filename>
      4. PROJECT_ROOT/<filename>
      5. ~/Downloads/<filename>
      6. Recursive search under backend/outputs and workspace/
    """
    if not file_arg or not file_arg.strip():
        return None

    clean = file_arg.strip().strip("'\"`")
    # In case user says "p1.py to git", trim trailing words
    tokens = clean.split()
    if tokens:
        clean = tokens[0]

    # 1. Direct path
    if os.path.isabs(clean) and os.path.isfile(clean):
        return os.path.abspath(clean)

    rel_proj = os.path.abspath(os.path.join(PROJECT_ROOT, clean))
    if os.path.isfile(rel_proj):
        return rel_proj

    # 2. Look in candidates by basename
    base_name = os.path.basename(clean)
    for search_dir in DEFAULT_SEARCH_DIRS:
        if os.path.isdir(search_dir):
            candidate = os.path.join(search_dir, base_name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    # 3. Recursive search in backend/outputs
    outputs_dir = os.path.join(PROJECT_ROOT, "backend", "outputs")
    if os.path.isdir(outputs_dir):
        for root, _, files in os.walk(outputs_dir):
            if base_name in files:
                return os.path.abspath(os.path.join(root, base_name))

    # 4. Recursive search in workspace
    ws_dir = os.path.join(PROJECT_ROOT, "workspace")
    if os.path.isdir(ws_dir):
        for root, _, files in os.walk(ws_dir):
            if base_name in files:
                return os.path.abspath(os.path.join(root, base_name))

    return None


def _run_git_cmd(cmd: list[str], cwd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Execute a git command inside the target workspace directory."""
    try:
        proc = subprocess.run(
            ["git"] + cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"Git command timed out after {timeout} seconds"
    except Exception as exc:
        return -1, "", str(exc)


def init_git_workspace(workspace_path: str) -> tuple[bool, str]:
    """
    Ensure the external Git workspace directory exists and has a git repository.
    Strictly verifies that workspace_path is outside LOKI project.
    """
    if is_safety_violation(workspace_path):
        return False, (
            f"SAFETY VIOLATION: Cannot initialize Git inside LOKI project '{PROJECT_ROOT}'. "
            "Git workspace must be an external directory."
        )

    os.makedirs(workspace_path, exist_ok=True)
    git_dir = os.path.join(workspace_path, ".git")

    if not os.path.isdir(git_dir):
        rc, out, err = _run_git_cmd(["init"], cwd=workspace_path)
        if rc != 0:
            return False, f"git init failed in '{workspace_path}': {err or out}"
        logger.info("Initialized new external Git workspace at %s", workspace_path)

    # Ensure committer name & email exist locally so git commit never blocks
    rc_name, name_out, _ = _run_git_cmd(["config", "user.name"], cwd=workspace_path)
    if rc_name != 0 or not name_out.strip():
        _run_git_cmd(["config", "user.name", "LOKI Co-Pilot"], cwd=workspace_path)

    rc_email, email_out, _ = _run_git_cmd(["config", "user.email"], cwd=workspace_path)
    if rc_email != 0 or not email_out.strip():
        _run_git_cmd(["config", "user.email", "loki@mindcraft.local"], cwd=workspace_path)

    return True, "Workspace ready"


def push_file_to_git(
    file_path: str,
    commit_message: Optional[str] = None,
    git_workspace: Optional[str] = None,
) -> dict:
    """
    Find a generated file, copy it to the separate Git workspace,
    git add, git commit, and git push.

    Args:
        file_path: Filename (e.g. 'p1.py') or path to the file.
        commit_message: Optional custom commit message.
        git_workspace: Optional custom Git workspace (defaults to GIT_WORKSPACE).

    Returns:
        Structured dict with execution results for each step.
    """
    # 0. Safety & workspace resolution
    target_ws = os.path.abspath(git_workspace or GIT_WORKSPACE)

    if is_safety_violation(target_ws):
        return {
            "ok": False,
            "error": (
                f"SAFETY VIOLATION: Refusing to operate inside LOKI project '{PROJECT_ROOT}'. "
                "The LOKI project itself must never be put on Git. "
                f"Use a separate external directory such as '{GIT_WORKSPACE}'."
            ),
        }

    # 1. Find the generated file
    src_file = find_generated_file(file_path)
    if not src_file:
        return {
            "ok": False,
            "error": (
                f"File '{file_path}' not found in outputs, workspace, project, or Downloads. "
                "Please make sure the file exists before pushing."
            ),
            "searched_dirs": DEFAULT_SEARCH_DIRS,
        }

    # 2. Ensure external Git workspace exists and is initialized
    ws_ok, ws_msg = init_git_workspace(target_ws)
    if not ws_ok:
        return {"ok": False, "error": ws_msg}

    filename = os.path.basename(src_file)
    dest_file = os.path.join(target_ws, filename)

    try:
        shutil.copy2(src_file, dest_file)
    except Exception as exc:
        return {"ok": False, "error": f"Failed to copy '{src_file}' to '{dest_file}': {exc}"}

    # 3. git add <filename>
    rc_add, add_out, add_err = _run_git_cmd(["add", filename], cwd=target_ws)
    if rc_add != 0:
        return {"ok": False, "error": f"git add failed: {add_err or add_out}"}

    # 4. git commit -m <message>
    msg = commit_message.strip() if commit_message and commit_message.strip() else f"Add {filename} via LOKI Co-Pilot"
    rc_commit, commit_out, commit_err = _run_git_cmd(["commit", "-m", msg], cwd=target_ws)

    committed = False
    commit_summary = ""
    if rc_commit == 0:
        committed = True
        commit_summary = commit_out.split("\n")[0] if commit_out else "Committed successfully"
    elif "nothing to commit" in (commit_out + commit_err).lower() or "working tree clean" in (commit_out + commit_err).lower():
        committed = True
        commit_summary = "Already up to date (no new changes to commit)"
    else:
        return {
            "ok": False,
            "error": f"git commit failed: {commit_err or commit_out}",
            "workspace": target_ws,
            "file": filename,
        }

    # 5. git push
    rc_push, push_out, push_err = _run_git_cmd(["push"], cwd=target_ws, timeout=30)
    pushed = False
    combined_push_output = (push_out + " " + push_err).strip()

    if rc_push == 0:
        pushed = True
        push_status = push_out.split("\n")[0] if push_out else "Pushed to remote successfully"
    else:
        # Check if remote is simply unconfigured or branch has no upstream
        lower_push = combined_push_output.lower()
        if "no configured push destination" in lower_push or "fatal: no remote" in lower_push or "no upstream branch" in lower_push:
            pushed = False
            push_status = (
                "Committed to local Git workspace. Remote repository not configured yet. "
                f"To push to GitHub/GitLab, open '{target_ws}' and run: 'git remote add origin <repo_url>' and 'git push -u origin main'."
            )
        else:
            pushed = False
            push_status = f"Push failed ({combined_push_output})"

    return {
        "ok": True,
        "file": filename,
        "source_path": src_file,
        "workspace": target_ws,
        "dest_path": dest_file,
        "added": True,
        "committed": committed,
        "commit_message": msg,
        "commit_summary": commit_summary,
        "pushed": pushed,
        "push_status": push_status,
        "message": (
            f"Successfully copied '{filename}' to Git workspace '{target_ws}', "
            f"staged with 'git add', and committed."
            + (f" Remote push: {push_status}" if pushed else f" Note: {push_status}")
        ),
    }
