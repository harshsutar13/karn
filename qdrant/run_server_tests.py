"""
run_server_tests.py

Start all services (Ollama, Qdrant, FastAPI) then run an API test script.

Usage:
    python run_server_tests.py <test_script.py>
"""

import subprocess
import sys
import time
import urllib.request
import os
import runpy

OLLAMA_EXE   = r"C:\Users\Admin\AppData\Local\Programs\Ollama\ollama.exe"
QDRANT_EXE   = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.exe"
QDRANT_DIR   = r"C:\WORK DRIVE\SIH26\qdrant"
BACKEND_DIR  = r"C:\WORK DRIVE\SIH26\backend"
VENV_UVICORN = r"C:\WORK DRIVE\SIH26\.venv\Scripts\uvicorn.exe"

OLLAMA_URL   = "http://localhost:11434"
QDRANT_URL   = "http://localhost:6333"
API_URL      = "http://localhost:8000"

QDRANT_LOG   = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.log"
OLLAMA_LOG   = r"C:\WORK DRIVE\SIH26\qdrant\ollama.log"
API_LOG      = r"C:\WORK DRIVE\SIH26\qdrant\api_server.log"


def is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_service(name: str, exe: str, args: list, cwd: str, log: str,
                   check_url: str, settle: int = 0):
    if is_ready(check_url):
        print(f"[{name}] Already running at {check_url}")
        return None
    print(f"[{name}] Starting...")
    log_fh = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen([exe] + args, cwd=cwd,
                            stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"[{name}] PID {proc.pid} — waiting...")
    for i in range(30):
        time.sleep(1)
        if is_ready(check_url):
            if settle:
                print(f"[{name}] Port up — settling {settle}s...")
                time.sleep(settle)
            print(f"[{name}] Ready at {check_url} (took {i+1}s)")
            return proc
        sys.stdout.write(f"\r[{name}] Waiting... {i+1}s")
        sys.stdout.flush()
    print(f"\n[{name}] ERROR: did not become ready. Check {log}")
    proc.terminate()
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <test_script.py>")
        sys.exit(1)

    test_script = sys.argv[1]

    # 1. Ollama
    ensure_service("ollama", OLLAMA_EXE, ["serve"],
                   cwd=os.path.dirname(OLLAMA_EXE),
                   log=OLLAMA_LOG, check_url=OLLAMA_URL, settle=3)

    # 2. Qdrant
    ensure_service("qdrant", QDRANT_EXE, ["--disable-telemetry"],
                   cwd=QDRANT_DIR, log=QDRANT_LOG, check_url=QDRANT_URL)

    # 3. FastAPI (uvicorn)
    ensure_service(
        "fastapi", VENV_UVICORN,
        ["main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=BACKEND_DIR,
        log=API_LOG,
        check_url=API_URL,
    )

    print(f"\n[runner] All services ready. Running: {test_script}\n")
    runpy.run_path(test_script, run_name="__main__")
    print("\n[runner] Done. Services left running.")
