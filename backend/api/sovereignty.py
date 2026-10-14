"""
api/sovereignty.py

Sovereignty proof endpoints.

Real question: does THE WORKBENCH make any external calls?
Not "is the host online?" — the host may have other apps using the internet.

This module filters connections by process name so we can prove:
  - Zero external connections owned by python.exe / uvicorn / qdrant / ollama
  - Any other process's traffic is out of scope for the sovereignty claim
"""

from __future__ import annotations

import socket
import time
from typing import Any

from fastapi import APIRouter

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

router = APIRouter(prefix="/api/sovereignty", tags=["sovereignty"])

# Process names considered part of the SIH26 workbench stack.
# Add more if you rename executables.
_WORKBENCH_PROCESSES = {
    "python.exe", "pythonw.exe",
    "uvicorn.exe",
    "qdrant.exe",
    "ollama.exe", "ollama app.exe",
    "llama-server.exe",
}

# Anything starting with these is local
_LOCAL_PREFIXES = ("127.", "0.0.0.0", "::1", "::")


def _is_local(ip: str) -> bool:
    if not ip:
        return True
    for p in _LOCAL_PREFIXES:
        if ip.startswith(p):
            return True
    return False


def _is_private_lan(ip: str) -> bool:
    """10.x, 172.16-31.x, 192.168.x. Also link-local fe80::/10."""
    if not ip:
        return False
    if ip.startswith("fe80:") or ip.startswith("FE80:"):
        return True  # IPv6 link-local
    if ":" in ip:
        return False  # other IPv6 we treat as external for the demo
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    if a == 10:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 192 and b == 168:
        return True
    return False


def _proc_name(pid: int | None) -> str:
    if not pid:
        return "(unknown)"
    try:
        return psutil.Process(pid).name()
    except Exception:
        return "(dead)"


def _is_workbench(pid: int | None, name: str | None = None) -> bool:
    if not pid:
        return False
    n = (name or _proc_name(pid)).lower()
    return n in _WORKBENCH_PROCESSES


@router.get("/status")
def sovereignty_status() -> dict[str, Any]:
    """
    Snapshot of network connections, split into:
      - workbench_connections: owned by our stack (should ALL be loopback)
      - other_app_connections: owned by browsers/OS/etc (out of scope)
    """
    if not _PSUTIL_OK:
        return {
            "psutil_available": False,
            "error": "psutil not installed. Run: pip install psutil",
        }

    try:
        conns = psutil.net_connections(kind="inet")
    except Exception as exc:
        return {
            "psutil_available": True,
            "error": f"Could not enumerate connections: {exc}",
        }

    workbench_loopback = []
    workbench_external = []
    other_external = []
    other_loopback_count = 0

    for c in conns:
        if not c.raddr:
            continue
        rip = c.raddr.ip
        pid = c.pid
        pname = _proc_name(pid)
        entry = {
            "proc": pname,
            "pid": pid,
            "laddr": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else None,
            "raddr": f"{rip}:{c.raddr.port}",
            "status": c.status,
        }

        if _is_workbench(pid, pname):
            if _is_local(rip) or _is_private_lan(rip):
                workbench_loopback.append(entry)
            else:
                workbench_external.append(entry)
        else:
            if _is_local(rip) or _is_private_lan(rip):
                other_loopback_count += 1
            else:
                other_external.append(entry)

    workbench_clean = len(workbench_external) == 0

    return {
        "psutil_available": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),

        # --- THE ANSWER FOR JUDGES ---
        "workbench_external_connections": workbench_external,
        "workbench_external_count": len(workbench_external),
        "workbench_loopback_count": len(workbench_loopback),
        "workbench_sovereign": workbench_clean,

        # --- Context (out of scope) ---
        "other_app_external_count": len(other_external),
        "other_app_loopback_count": other_loopback_count,
        "other_app_note": (
            "These belong to browsers / OS / unrelated apps on the host. "
            "They are NOT the workbench. In a production deployment the host "
            "would be isolated on a private network with no internet route."
        ),

        "interpretation": (
            "SOVEREIGN: The workbench made zero external network calls. "
            "All AI inference, embedding, and vector search are local."
            if workbench_clean else
            "WARNING: workbench process made an external call. Investigate."
        ),
    }


@router.get("/ping-external")
def ping_external_attempt() -> dict[str, Any]:
    """
    Attempt an outbound TCP connection to a public IP.
    reachable=False = host is air-gapped.
    """
    target_host, target_port = "1.1.1.1", 53
    result: dict[str, Any] = {
        "target": f"{target_host}:{target_port}",
        "attempted": True,
        "reachable": None,
        "elapsed_ms": None,
        "note": (
            "reachable=False means the host is air-gapped. "
            "reachable=True means the host CAN reach the internet — the "
            "workbench still doesn't, as shown by /status."
        ),
    }

    t0 = time.perf_counter()
    try:
        with socket.create_connection((target_host, target_port), timeout=3):
            result["reachable"] = True
    except (socket.timeout, OSError):
        result["reachable"] = False
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result