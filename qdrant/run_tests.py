"""
run_tests.py

Utility: ensure Qdrant is running, then execute a test module.

Usage:
    python run_tests.py <path_to_test_script.py>

This script:
1. Checks if Qdrant is already responding on port 6333.
2. If not, starts qdrant.exe as a detached subprocess.
3. Waits up to 15 seconds for Qdrant to be ready.
4. Imports and runs the requested test script via runpy.
5. The Qdrant process continues running until this script exits
   (since we keep a reference to it).
"""

import subprocess
import sys
import time
import urllib.request
import os
import runpy

QDRANT_EXE = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.exe"
QDRANT_DIR = r"C:\WORK DRIVE\SIH26\qdrant"
QDRANT_URL = "http://localhost:6333"
QDRANT_LOG = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.log"


def is_qdrant_ready() -> bool:
    try:
        with urllib.request.urlopen(QDRANT_URL, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_qdrant() -> subprocess.Popen | None:
    """Start Qdrant if not already running. Returns the process or None."""
    if is_qdrant_ready():
        print("[qdrant] Already running at http://localhost:6333")
        return None

    print("[qdrant] Starting Qdrant...")
    log_fh = open(QDRANT_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [QDRANT_EXE, "--disable-telemetry"],
        cwd=QDRANT_DIR,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    print(f"[qdrant] PID {proc.pid} — waiting for HTTP...")
    for i in range(20):
        time.sleep(1)
        if is_qdrant_ready():
            print(f"[qdrant] Ready at http://localhost:6333 (took {i+1}s)")
            return proc
        sys.stdout.write(f"\r[qdrant] Waiting... {i+1}s")
        sys.stdout.flush()

    print("\n[qdrant] ERROR: Qdrant did not become ready after 20s")
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

    qdrant_proc = ensure_qdrant()

    try:
        # Run the test script in the same interpreter
        runpy.run_path(test_script, run_name="__main__")
    finally:
        if qdrant_proc is not None:
            # Let it keep running for reuse — comment out to auto-stop
            print("[qdrant] Leaving Qdrant running for subsequent use.")
            # qdrant_proc.terminate()
