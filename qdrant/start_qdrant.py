"""
start_qdrant.py

Launches the Qdrant binary as a persistent background process
and verifies it is responding on port 6333.

Usage:
    python start_qdrant.py
"""

import subprocess
import sys
import time
import urllib.request
import urllib.error
import os

QDRANT_EXE = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.exe"
QDRANT_DIR = r"C:\WORK DRIVE\SIH26\qdrant"
QDRANT_URL = "http://localhost:6333"
LOG_FILE    = r"C:\WORK DRIVE\SIH26\qdrant\qdrant.log"


def is_running() -> bool:
    try:
        with urllib.request.urlopen(QDRANT_URL, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    if is_running():
        print("Qdrant is already running at http://localhost:6333")
        return 0

    print(f"Starting Qdrant from {QDRANT_DIR} ...")
    log_fh = open(LOG_FILE, "w", encoding="utf-8")

    proc = subprocess.Popen(
        [QDRANT_EXE, "--disable-telemetry"],
        cwd=QDRANT_DIR,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    print(f"Qdrant process started (PID {proc.pid}). Waiting for HTTP...")

    for attempt in range(20):
        time.sleep(1)
        if is_running():
            print(f"Qdrant ready at {QDRANT_URL} (took {attempt+1}s)")
            return 0
        sys.stdout.write(f"\r  Waiting... {attempt+1}s")
        sys.stdout.flush()

    print(f"\nQdrant did not become ready after 20s. Check log: {LOG_FILE}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
