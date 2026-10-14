"""
run_project.py

Single master launcher for the SIH26 AI Workbench MVP.

Responsibilities:
1. Dynamically detect project paths (virtualenv, models, executables)
2. Set Ollama VRAM guard env vars (single model resident)
3. Start or reuse local Ollama service (http://localhost:11434)
4. Verify required MVP models exist in Ollama
5. Start or reuse local Qdrant vector store (http://localhost:6333)
6. Start or reuse FastAPI server with frontend UI (http://localhost:8000)
7. Run fast lightweight MVP verification (GET /, GET /api/capabilities, POST /api/ask)
8. Automatically open the Web UI in the default browser
9. Keep services alive and handle clean shutdown on Ctrl+C without orphan processes

Usage:
    python run_project.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
QDRANT_DIR = os.path.join(PROJECT_ROOT, "qdrant")
QDRANT_EXE = os.path.join(QDRANT_DIR, "qdrant.exe")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
API_URL = "http://localhost:8000"

# Models the launcher REQUIRES to be present in Ollama.
# These must match what config.py actually uses.
REQUIRED_MODELS = [
    "llama3.2:latest",       # main LLM
    "qwen2.5-coder:7b",      # coding
    "nomic-embed-text",      # embeddings for RAG
    "qwen3-vl:4b",           # vision
    "glm-ocr:latest",        # OCR
    # Optional — uncomment if your app needs them for vision/ocr:
    # "qwen3-vl:4b",
    # "glm-ocr:latest",
    # "qwen2.5-coder:7b",
]

# VRAM guards — set before spawning Ollama. Prevents CUDA crash on 6 GB GPUs.
OLLAMA_ENV = {
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_KEEP_ALIVE": "2m",
}

# Timeouts (seconds)
OLLAMA_TIMEOUT = 60
QDRANT_TIMEOUT = 30
FASTAPI_TIMEOUT = 180     # was 30 — model cold-start can take a while

_started_processes: dict[str, subprocess.Popen] = {}


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def is_service_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def find_ollama_executable() -> str | None:
    found = shutil.which("ollama") or shutil.which("ollama.exe")
    if found and os.path.isfile(found):
        return found

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        cand = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(cand):
            return cand

    cand = os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")
    if os.path.isfile(cand):
        return cand

    return None


def match_model(required: str, installed_tags: list[str]) -> bool:
    req_base, _, req_tag = required.lower().partition(":")
    for inst in installed_tags:
        inst_base, _, inst_tag = inst.lower().partition(":")
        if req_base == inst_base:
            if not req_tag or req_tag == "latest":
                return True
            if inst_tag == req_tag:
                return True
    return False


# ---------------------------------------------------------------------------
# Service Starters
# ---------------------------------------------------------------------------

def step_1_ollama() -> None:
    sys.stdout.write("[1/5] Checking Ollama ............ ")
    sys.stdout.flush()

    if is_service_ready(OLLAMA_URL):
        print("READY", flush=True)
        return

    exe = find_ollama_executable()
    if not exe:
        print("FAILED", flush=True)
        print("[ERROR] Ollama executable not found.", flush=True)
        print("Ensure Ollama is installed from https://ollama.com and available in PATH.", flush=True)
        sys.exit(1)

    log_file = os.path.join(LOGS_DIR, "ollama.log")
    log_fh = open(log_file, "a", encoding="utf-8")

    # Merge VRAM guard env vars with current environment
    env = os.environ.copy()
    env.update(OLLAMA_ENV)

    try:
        proc = subprocess.Popen(
            [exe, "serve"],
            cwd=os.path.dirname(exe),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception as exc:
        print("FAILED", flush=True)
        print(f"[ERROR] Failed to launch Ollama: {exc}", flush=True)
        sys.exit(1)

    _started_processes["ollama"] = proc

    for _ in range(OLLAMA_TIMEOUT):
        time.sleep(1)
        if is_service_ready(OLLAMA_URL):
            time.sleep(1)
            print("READY", flush=True)
            return

    print("FAILED", flush=True)
    print(f"[ERROR] Ollama did not respond within {OLLAMA_TIMEOUT}s. Check log: {log_file}", flush=True)
    cleanup_and_exit(1)


def step_2_models() -> None:
    sys.stdout.write("[2/5] Checking models ............ ")
    sys.stdout.flush()

    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            installed = [m.get("name", "") for m in data.get("models", [])]
    except Exception as exc:
        print("FAILED", flush=True)
        print(f"[ERROR] Failed to query Ollama model tags: {exc}", flush=True)
        cleanup_and_exit(1)

    missing = [m for m in REQUIRED_MODELS if not match_model(m, installed)]

    if missing:
        print("FAILED", flush=True)
        print(f"[ERROR] Missing required MVP model(s): {', '.join(missing)}", flush=True)
        print("To install, run:", flush=True)
        for m in missing:
            print(f"    ollama pull {m}", flush=True)
        cleanup_and_exit(1)

    print("READY", flush=True)


def step_3_qdrant() -> None:
    sys.stdout.write("[3/5] Starting Qdrant ........... ")
    sys.stdout.flush()

    if is_service_ready(QDRANT_URL):
        print("READY", flush=True)
        return

    if not os.path.isfile(QDRANT_EXE):
        print("FAILED", flush=True)
        print(f"[ERROR] Qdrant executable not found at: {QDRANT_EXE}", flush=True)
        cleanup_and_exit(1)

    log_file = os.path.join(LOGS_DIR, "qdrant.log")
    log_fh = open(log_file, "a", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            [QDRANT_EXE],                 # no deprecated flags
            cwd=QDRANT_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        print("FAILED", flush=True)
        print(f"[ERROR] Failed to launch Qdrant: {exc}", flush=True)
        cleanup_and_exit(1)

    _started_processes["qdrant"] = proc

    for _ in range(QDRANT_TIMEOUT):
        time.sleep(1)
        if is_service_ready(QDRANT_URL):
            print("READY", flush=True)
            return

    print("FAILED", flush=True)
    print(f"[ERROR] Qdrant did not respond within {QDRANT_TIMEOUT}s. Check log: {log_file}", flush=True)
    cleanup_and_exit(1)


def step_4_fastapi() -> None:
    sys.stdout.write("[4/5] Starting FastAPI .......... ")
    sys.stdout.flush()

    if is_service_ready(API_URL):
        print("READY", flush=True)
        return

    # Use the Python interpreter running this launcher.
    # This follows the environment selected by VS Code or the terminal.
    python_exe = sys.executable

    cmd = [
        python_exe,
        "-m", "uvicorn",
        "main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]

    log_file = os.path.join(LOGS_DIR, "api_server.log")
    log_fh = open(log_file, "a", encoding="utf-8")

    env = os.environ.copy()
    env.update(OLLAMA_ENV)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=BACKEND_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception as exc:
        log_fh.close()
        print("FAILED", flush=True)
        print(f"[ERROR] Failed to launch FastAPI server: {exc}", flush=True)
        cleanup_and_exit(1)

    _started_processes["fastapi"] = proc

    for _ in range(FASTAPI_TIMEOUT):
        time.sleep(1)
        if is_service_ready(API_URL):
            print("READY", flush=True)
            return

    print("FAILED", flush=True)
    print(
        f"[ERROR] FastAPI server did not become ready within "
        f"{FASTAPI_TIMEOUT}s. Check log: {log_file}",
        flush=True
    )
    cleanup_and_exit(1)


def step_5_verify_mvp() -> None:
    sys.stdout.write("[5/5] Checking MVP API .......... ")
    sys.stdout.flush()

    # 1. Root UI
    try:
        req = urllib.request.Request(API_URL)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print("FAILED", flush=True)
                print(f"[ERROR] Root endpoint returned status {resp.status}", flush=True)
                cleanup_and_exit(1)
    except Exception as exc:
        print("FAILED", flush=True)
        print(f"[ERROR] Root UI health check failed: {exc}", flush=True)
        cleanup_and_exit(1)

    # 2. Capabilities
    try:
        req = urllib.request.Request(f"{API_URL}/api/capabilities")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if resp.status != 200 or "capabilities" not in data:
                print("FAILED", flush=True)
                print("[ERROR] Capabilities endpoint returned invalid response.", flush=True)
                cleanup_and_exit(1)
    except Exception as exc:
        print("FAILED", flush=True)
        print(f"[ERROR] Capabilities endpoint failed: {exc}", flush=True)
        cleanup_and_exit(1)

    # 3. /api/ask — just check it responds 200 with an 'answer' key.
    # Don't gate on capability == 'general' (router behavior can evolve).
    try:
        payload = json.dumps({"message": "What is 2 + 2?"}).encode("utf-8")
        req = urllib.request.Request(
            f"{API_URL}/api/ask",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            ans_data = json.loads(resp.read())
            if resp.status != 200 or "answer" not in ans_data:
                print("FAILED", flush=True)
                print(f"[ERROR] /api/ask verification failed. Response: {ans_data}", flush=True)
                cleanup_and_exit(1)
    except Exception as exc:
        print("FAILED", flush=True)
        print(f"[ERROR] /api/ask test request failed: {exc}", flush=True)
        cleanup_and_exit(1)

    print("READY", flush=True)


# ---------------------------------------------------------------------------
# Cleanup & Signal Handling
# ---------------------------------------------------------------------------

def cleanup_and_exit(code: int = 0) -> None:
    if _started_processes:
        print("\nStopping services started by launcher...", flush=True)
        for name, proc in _started_processes.items():
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        print("All launcher services stopped.", flush=True)
    sys.exit(code)


def signal_handler(sig, frame):
    print("\nReceived exit signal.", flush=True)
    cleanup_and_exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60, flush=True)
    print("              LOKI an AI WORKBENCH MVP", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    step_1_ollama()
    step_2_models()
    step_3_qdrant()
    step_4_fastapi()
    step_5_verify_mvp()

    print(flush=True)
    print("=" * 60, flush=True)
    print("                  MVP IS READY", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    print("Web UI:      http://localhost:8000", flush=True)
    print("API:         http://localhost:8000/api/ask", flush=True)
    print("Capabilities: http://localhost:8000/api/capabilities", flush=True)
    print(flush=True)
    print("Press Ctrl+C to stop the project.", flush=True)
    print(flush=True)

    def _open_browser():
        try:
            webbrowser.open("http://localhost:8000")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        while True:
            time.sleep(1)
            for name, proc in list(_started_processes.items()):
                if proc.poll() is not None:
                    print(f"\n[ALERT] Service '{name}' exited unexpectedly with code "
                          f"{proc.returncode}.", flush=True)
                    cleanup_and_exit(1)
    except KeyboardInterrupt:
        cleanup_and_exit(0)


if __name__ == "__main__":
    main()
