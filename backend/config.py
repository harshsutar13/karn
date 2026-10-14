import os


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

# --- LLM -------------------------------------------------------------------
# RTX 3050 Laptop has 6 GB VRAM. gemma4:12b needs ~7.6 GB just for weights,
# which crashes llama-server with 0xc0000409 + "CUDA out of memory" whenever
# anything else (embedding/vision) is loaded. llama3.2 (2 GB) fits safely.
#
# If you want to try gemma4:12b again later (single-model, nothing else
# resident), change the default back to "gemma4:12b".
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:latest"
)
PLANNER_MODEL = os.getenv(
    "PLANNER_MODEL",
    "llama3.2:latest"
)
# --- Coding ----------------------------------------------------------------
# qwen2.5-coder:7b (4.7 GB) is the installed coding model and fits the GPU.
CODING_MODEL = os.getenv(
    "CODING_MODEL",
    "qwen2.5-coder:7b"
)

# --- Embedding -------------------------------------------------------------
# Small (274 MB). Kept warm; safe alongside LLM.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

# --- OCR -------------------------------------------------------------------
# 2.2 GB. Only loaded on demand.
OCR_MODEL = os.getenv(
    "OCR_MODEL",
    "glm-ocr:latest"
)

# --- Vision ----------------------------------------------------------------
# 3.3 GB. Only loaded on demand. Swap-friendly alongside LLM (manager
# unloads the LLM first before loading vision).
VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "qwen3-vl:4b"
)

# --- Qdrant ----------------------------------------------------------------
QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost"
)

QDRANT_PORT = int(os.getenv(
    "QDRANT_PORT",
    "6333"
))

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "sih26_documents"
)

# --- Memory ----------------------------------------------------------------
# Persistent session memory (conversation + agent actions) for follow-up
# questions like "open the file you just created". Local SQLite only.
MEMORY_DB_PATH = os.getenv(
    "MEMORY_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "memory.db")
)

# --- Project Root ----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# --- Git workspace ---------------------------------------------------------
# A SEPARATE folder (outside this project) where generated output files
# are copied before being committed and pushed.
# Override with the GIT_WORKSPACE env var if you place it elsewhere.
GIT_WORKSPACE = os.getenv(
    "GIT_WORKSPACE",
    os.path.join(os.path.expanduser("~"), "Desktop", "loki_git_workspace")
)

