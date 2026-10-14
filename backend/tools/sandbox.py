"""
tools/sandbox.py

Isolated Python execution with timeout.

Security posture (MVP, air-gapped host):
  - Runs in a temp directory that is deleted after execution
  - Uses `python -I -u` (isolated mode: no PYTHONPATH, no user env, no
    user site-packages). NOTE: -S is deliberately NOT used so that the
    pre-installed whitelist (numpy, pandas, openpyxl, sympy) is visible
    to agent-generated code.
  - Hard 10s timeout
  - Captures stdout/stderr separately
  - Does NOT block network — the host is assumed air-gapped
    (see /api/sovereignty/status for the proof)

For production, swap `subprocess.run` for a container/VM boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: float
    script_path: str  # temp path, deleted after run


def run_python(code: str, timeout: int = 10) -> SandboxResult:
    """
    Execute Python source code in an isolated temp directory.

    Parameters
    ----------
    code : str
        The Python source to run.
    timeout : int
        Seconds before the process is force-killed.

    Returns
    -------
    SandboxResult
    """
    import time

    if not code or not code.strip():
        return SandboxResult(
            ok=False, stdout="", stderr="Empty code", exit_code=-1,
            elapsed_ms=0.0, script_path="",
        )

    with tempfile.TemporaryDirectory(prefix="sih26_sbx_") as tmp:
        script_name = f"run_{uuid.uuid4().hex[:8]}.py"
        script_path = os.path.join(tmp, script_name)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        t0 = time.perf_counter()
        try:
            # -I = isolated (no PYTHONPATH, no user site)
            # -u = unbuffered output (so we get stdout even on crash)
            # NOTE: -S removed so whitelisted libraries are importable.
            proc = subprocess.run(
                [sys.executable, "-I", "-u", script_path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    # Minimal env — no proxies, no tokens, no home
                    "PATH": os.environ.get("PATH", ""),
                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                    "TEMP": tmp,
                    "TMP": tmp,
                },
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return SandboxResult(
                ok=(proc.returncode == 0),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
                elapsed_ms=round(elapsed, 1),
                script_path=script_path,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return SandboxResult(
                ok=False,
                stdout=(
                    (exc.stdout or b"").decode("utf-8", "replace")
                    if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                ),
                stderr=f"TIMEOUT after {timeout}s",
                exit_code=-1,
                elapsed_ms=round(elapsed, 1),
                script_path=script_path,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return SandboxResult(
                ok=False, stdout="", stderr=f"Execution error: {exc}",
                exit_code=-1, elapsed_ms=round(elapsed, 1), script_path=script_path,
            )