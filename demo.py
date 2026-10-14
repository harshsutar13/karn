"""
demo.py

SIH26 Sovereign AI Workbench — end-to-end demo.

Runs 7 demos in sequence covering:
  1. Sovereignty — proof of zero external calls
  2. LLM routing — smart capability classifier
  3. RAG Q&A    — grounded answer with citations
  4. Sandbox    — verified code execution
  5. Scan-to-approval — OCR → approval note .docx
  6. Planner-agent    — LLM planning + re-planning → .xlsx + .pptx
  7. Vision     — read a refinery P&ID

Usage:
    cd D:\\SIH26
    .venv\\Scripts\\python.exe demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND = ROOT / "backend"
API = "http://127.0.0.1:8000"

sys.path.insert(0, str(BACKEND))

VISION_IMAGE_CANDIDATES = [
    ROOT / "data" / "uploads" / "21727dedb3b7476e94d2afb279e8b1ca.jpg",
    ROOT / "data" / "uploads" / "b70fbea5ff9b413082f8dc980e3b038a.png",
    ROOT / "data" / "ocr_test.png",
]


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(title: str, step: str = "") -> None:
    bar = "=" * 74
    print()
    print(f"{CYAN}{bar}{RESET}")
    if step:
        print(f"{CYAN}{BOLD}  {step}  —  {title}{RESET}")
    else:
        print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[!]{RESET}   {msg}")


def err(msg: str) -> None:
    print(f"  {RED}[ERR]{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: float = 10.0) -> dict | None:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def http_post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = "(no body)"
        return {"__error__": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"__error__": str(e)}


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight() -> bool:
    banner("Pre-flight checks", "STEP 0")
    all_ok = True

    caps = http_get(f"{API}/api/capabilities")
    if caps:
        ok(f"FastAPI reachable at {API} — {caps.get('count', '?')} capabilities")
    else:
        err(f"FastAPI NOT reachable. Run: python run_project.py")
        all_ok = False

    sov = http_get(f"{API}/api/sovereignty/status")
    if sov:
        ok(f"Sovereignty: workbench external = {sov.get('workbench_external_count')}, "
           f"sovereign = {sov.get('workbench_sovereign')}")
    else:
        warn("Sovereignty endpoint not available")

    return all_ok


# ---------------------------------------------------------------------------
# DEMO 1 — Sovereignty
# ---------------------------------------------------------------------------

def demo_1_sovereignty() -> None:
    banner("Sovereignty proof — no external calls", "DEMO 1")

    sov = http_get(f"{API}/api/sovereignty/status")
    if not sov:
        warn("Endpoint unavailable.")
        return

    print()
    info(f"Hostname:      {sov.get('hostname')}")
    info(f"Timestamp:     {sov.get('timestamp')}")
    print()
    print(f"  {BOLD}Workbench processes:{RESET}")
    print(f"    external connections:  {sov.get('workbench_external_count')}")
    print(f"    loopback (local):      {sov.get('workbench_loopback_count')}")
    col = GREEN if sov.get('workbench_sovereign') else RED
    print(f"    SOVEREIGN:             {col}{sov.get('workbench_sovereign')}{RESET}")
    print()
    info(f"Other apps on host (out of scope): {sov.get('other_app_external_count')} external")
    print()
    print(f"  {DIM}{sov.get('interpretation')}{RESET}")


# ---------------------------------------------------------------------------
# DEMO 2 — LLM routing
# ---------------------------------------------------------------------------

def demo_2_routing() -> None:
    banner("LLM-based capability routing", "DEMO 2")

    samples = [
        ("What is the vibration reading on P-101?",        "document"),
        ("Write a Python function to parse CSV files.",    "coding"),
        ("Hello, how are you today?",                      "general"),
        ("Summarize the inspection report for P-101.",     "document"),
    ]

    for msg, expected in samples:
        url = f"{API}/api/route/explain?" + urllib.parse.urlencode({"message": msg})
        r = http_get(url, timeout=15)
        if not r:
            warn(f"(route endpoint missing — showing classifier via ask)")
            continue
        got = r.get("capability")
        reason = r.get("reason", "")
        marker = GREEN + "OK" + RESET if got == expected else YELLOW + "??" + RESET
        print(f"  [{marker}]  {got:8}  {msg[:50]:<52}  ({reason[:50]})")


# ---------------------------------------------------------------------------
# DEMO 3 — RAG Q&A
# ---------------------------------------------------------------------------

def demo_3_rag_qa() -> None:
    banner("RAG Q&A — grounded answer with citations", "DEMO 3")

    question = (
        "What is the current vibration reading on P-101 and how does it "
        "compare to the historical baseline?"
    )
    print(f"  {BOLD}Question:{RESET} {question}")
    print()
    info("POST /api/ask (15-40s)")

    t0 = time.time()
    resp = http_post(f"{API}/api/ask", {"message": question}, timeout=120)
    dt = time.time() - t0

    if "__error__" in resp:
        err(resp["__error__"])
        return

    print()
    print(f"  {BOLD}Answer:{RESET}")
    for line in (resp.get("answer") or "").split("\n"):
        print(f"    {line}")

    print()
    info(f"capability={resp.get('capability')}  model={resp.get('model')}  ({dt:.1f}s)")


# ---------------------------------------------------------------------------
# DEMO 4 — Sandbox
# ---------------------------------------------------------------------------

def demo_4_sandbox() -> None:
    banner("Sandbox — verified pump-power calculation", "DEMO 4")

    code = """\
rho = 994
g = 9.81
Q = 90 / 3600
H = 32
eta = 0.75
hydraulic_kw = rho * g * Q * H / 1000
input_kw = hydraulic_kw / eta
print(f'Hydraulic power: {hydraulic_kw:.2f} kW')
print(f'Input power:     {input_kw:.2f} kW')
assert abs(hydraulic_kw - 7.80) < 0.05
assert abs(input_kw - 10.40) < 0.05
print('Verification: PASS')
"""
    print(f"  {BOLD}Running in isolated sandbox:{RESET}")
    for line in code.strip().split("\n"):
        print(f"    {DIM}{line}{RESET}")

    print()
    resp = http_post(f"{API}/api/sandbox/run", {"code": code, "timeout": 10})

    if "__error__" in resp:
        err(resp["__error__"])
        return

    ok(f"ok={resp.get('ok')}  exit={resp.get('exit_code')}  {resp.get('elapsed_ms'):.0f}ms")
    print(f"  {BOLD}stdout:{RESET}")
    for line in (resp.get("stdout") or "").split("\n"):
        print(f"    {line}")


# ---------------------------------------------------------------------------
# DEMO 5 — Scan-to-approval
# ---------------------------------------------------------------------------

def demo_5_scan_approval() -> None:
    banner("Scan → OCR → approval note (.docx)", "DEMO 5")

    image = next((p for p in VISION_IMAGE_CANDIDATES if p.is_file()), None)
    if not image:
        warn("No test image found; skipping.")
        return

    print(f"  {BOLD}Scanned image:{RESET} {image}")
    info("POST /api/agent/scan-to-approval (60-120s)")

    t0 = time.time()
    resp = http_post(
        f"{API}/api/agent/scan-to-approval",
        {"image_path": str(image), "case_id": "CASE-P101-DEMO", "equipment_tag": "P-101"},
        timeout=300,
    )
    dt = time.time() - t0

    if "__error__" in resp:
        err(resp["__error__"])
        return

    print()
    for step in resp.get("steps", []):
        print(f"    {step}")

    print()
    print(f"  {BOLD}Deliverable:{RESET} {resp.get('output_path')}")
    print(f"  {BOLD}Download:   {RESET} {resp.get('download_url')}")
    info(f"Total: {dt:.1f}s")


# ---------------------------------------------------------------------------
# DEMO 6 — Planner-driven agent
# ---------------------------------------------------------------------------

def demo_6_planner() -> None:
    banner("Planner-driven agent — LLM plans, executes, re-plans", "DEMO 6")

    goal = (
        "Analyze P-101 vibration trends and produce an Excel spreadsheet "
        "with the current vs historical values, plus a PowerPoint deck "
        "summarizing the findings."
    )
    print(f"  {BOLD}Goal:{RESET} {goal}")
    print()
    info("POST /api/agent/plan-and-execute (90-240s)")
    info("The LLM plans the steps; the executor runs them; failures trigger re-planning.")

    t0 = time.time()
    resp = http_post(
        f"{API}/api/agent/plan-and-execute",
        {"goal": goal},
        timeout=600,
    )
    dt = time.time() - t0

    if "__error__" in resp:
        err(resp["__error__"])
        return

    plan = resp.get("plan", {})
    print()
    print(f"  {BOLD}Reasoning:{RESET}")
    print(f"    {plan.get('reasoning', '(none)')}")

    print()
    print(f"  {BOLD}Planned steps:{RESET}")
    for s in plan.get("steps", []):
        print(f"    {s['id']:8}  {s['tool']}")

    print()
    print(f"  {BOLD}Execution:{RESET}")
    for s in resp.get("steps", []):
        status = "OK" if s.get("ok") else "FAIL"
        col = GREEN if s.get("ok") else RED
        print(f"    [{col}{status}{RESET}] {s['id']:8} ({s['tool']})")
        if s.get("tool") in ("write_xlsx", "write_pptx", "write_docx") and s.get("ok"):
            out = s.get("output") or {}
            if out.get("filename"):
                print(f"        -> {out['filename']}")

    print()
    print(f"  {BOLD}Replans:{RESET} {resp.get('replans')}")
    print(f"  {BOLD}Error:  {RESET} {resp.get('error') or 'none'}")
    info(f"Total: {dt:.1f}s")


# ---------------------------------------------------------------------------
# DEMO 7 — Vision
# ---------------------------------------------------------------------------

def demo_7_vision() -> None:
    banner("Multimodal — vision reads a refinery P&ID", "DEMO 7")

    image = next((p for p in VISION_IMAGE_CANDIDATES if p.is_file()), None)
    if not image:
        warn("No test image found; skipping.")
        return

    print(f"  {BOLD}Image:{RESET} {image}")
    info("POST /api/vision/analyze (15-40s)")

    resp = http_post(
        f"{API}/api/vision/analyze",
        {
            "image_path": str(image),
            "prompt": "Describe what you see. What equipment, text, or condition is shown?",
        },
        timeout=180,
    )

    if "__error__" in resp:
        err(resp["__error__"])
        return

    text = (resp.get("answer") or resp.get("result")
            or resp.get("description") or json.dumps(resp))
    print()
    print(f"  {BOLD}Vision output:{RESET}")
    for line in text.split("\n")[:25]:
        print(f"    {line}")
def demo_8_new_capability() -> None:
    banner("New capability added by config alone", "DEMO 8")

    caps = http_get(f"{API}/api/capabilities")
    if not caps:
        warn("Capabilities endpoint not reachable.")
        return

    print()
    print(f"  {BOLD}Current capabilities ({caps.get('count')}):{RESET}")
    for c in caps.get("capabilities", []):
        print(f"    {c.get('capability'):20} -> {c.get('model')}")

    print()
    info("The 'summarization' and 'coding_fast' capabilities were added by")
    info("editing models/registry.py only — no changes to the API, router,")
    info("executor, or any other module. Restart FastAPI and they appear.")

    print()
    info("Testing the summarize tool via the planner (10-30s)...")

    resp = http_post(
        f"{API}/api/agent/plan-and-execute",
                        {
            "goal": (
                "First use rag_search to find information about the P-101 "
                "vibration issue. Then use the summarize tool to summarize "
                "that retrieved content. Finally, use write_file to save the "
                "summary to D:/SIH26/backend/outputs/p101_summary.txt"
            )
        },
        timeout=180,
    )

    if "__error__" in resp:
        err(resp["__error__"])
        return

    print()
    print(f"  {BOLD}Planned steps:{RESET}")
    for s in resp.get("plan", {}).get("steps", []):
        print(f"    {s['id']:8}  {s['tool']}")

    print()
    print(f"  {BOLD}Execution:{RESET}")
    for s in resp.get("steps", []):
        status = "OK" if s.get("ok") else "FAIL"
        col = GREEN if s.get("ok") else RED
        print(f"    [{col}{status}{RESET}] {s['id']:8} ({s['tool']})")

    print()
    print(f"  {BOLD}Replans:{RESET} {resp.get('replans')}")
    print(f"  {BOLD}Error:  {RESET} {resp.get('error') or 'none'}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print()
    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║        SIH26 SOVEREIGN AI WORKBENCH  —  LIVE DEMO                ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{DIM}  All models, embeddings, and vector search execute locally.{RESET}")
    print(f"{DIM}  Nothing leaves this machine.{RESET}")

    if not preflight():
        print()
        err("Pre-flight failed. Start services with: python run_project.py")
        return 1

    demos = [
        demo_1_sovereignty,
        demo_2_routing,
        demo_3_rag_qa,
        demo_4_sandbox,
        demo_5_scan_approval,
        demo_6_planner,
        demo_7_vision,
        demo_8_new_capability,
    ]

    def _unload():
        """Ask Ollama to evict any resident models so the next demo has full VRAM."""
        try:
            import sys as _sys
            from pathlib import Path as _P
            _sys.path.insert(0, str(_P(__file__).parent / "backend"))
            from models.manager import unload_all
            unload_all()
        except Exception:
            pass

    try:
        for d in demos:
            # Evict models before VRAM-hungry demos
            if d.__name__ in ("demo_5_scan_approval", "demo_6_planner", "demo_7_vision","demo_8_new_capability",):
                info("Evicting resident models from VRAM...")
                _unload()
                time.sleep(1)
            d()
    except KeyboardInterrupt:
        print()
        warn("Interrupted by user.")
        return 130

    banner("DEMO COMPLETE")
    print()
    ok("All capabilities exercised.")
    print(f"  {BOLD}Artifacts in:{RESET} {BACKEND / 'outputs'}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())