"""
run_all_tests.py

Utility: start Ollama + Qdrant, then run a test script.

Usage:
    python run_all_tests.py <path_to_test_script.py>
"""

import subprocess
import sys
import time
import urllib.request
import os
import runpy

OLLAMA_EXE  = r"C:\Users\Admin\AppData\Local\Programs\Ollama\ollama.exe"
QDRANT_EXE  = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.exe"
QDRANT_DIR  = r"C:\WORK DRIVE\SIH26\qdrant"
QDRANT_LOG  = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.log"
OLLAMA_URL  = "http://localhost:11434"
QDRANT_URL  = "http://localhost:6333"


def is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_service(name: str, exe: str, args: list, cwd: str, log: str, check_url: str):
    if is_ready(check_url):
        print(f"[{name}] Already running at {check_url}")
        return None
    print(f"[{name}] Starting...")
    log_fh = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [exe] + args,
        cwd=cwd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    print(f"[{name}] PID {proc.pid} — waiting...")
    for i in range(25):
        time.sleep(1)
        if is_ready(check_url):
            # Extra settle time for Ollama: it accepts connections before
            # its model runtime is fully initialised, causing ReadError 10054
            # on the first request if we hit it too quickly.
            settle = 4 if name == "ollama" else 0
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
    if not os.path.isfile(test_script):
        print(f"Test script not found: {test_script}")
        sys.exit(1)

    ollama_proc = ensure_service(
        "ollama", OLLAMA_EXE, ["serve"],
        cwd=os.path.dirname(OLLAMA_EXE),
        log=r"C:\WORK DRIVE\SIH26\qdrant\ollama.log",
        check_url=OLLAMA_URL,
    )

    qdrant_proc = ensure_service(
        "qdrant", QDRANT_EXE, ["--disable-telemetry"],
        cwd=QDRANT_DIR,
        log=QDRANT_LOG,
        check_url=QDRANT_URL,
    )

    print(f"\n[runner] Both services ready. Running: {test_script}\n")
    try:
        runpy.run_path(test_script, run_name="__main__")
    finally:
        print("\n[runner] Test complete. Services left running for reuse.")
