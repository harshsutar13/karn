"""
models/registry.py

Model capability registry for SIH26 AI Workbench.

Purpose
-------
Decouples the rest of the system from specific model names.
The orchestrator, RAG pipeline, and API routes should reference
capabilities (e.g. "general_reasoning", "coding") rather than
hard-coding model names like "gemma4:12b".

Design
------
- `ModelConfig` — a lightweight dataclass describing one model.
- `REGISTRY` — the single source of truth mapping capability → ModelConfig.
- `get_model(capability)` — the only function external code needs to call.
- The registry is populated from config.py values, which in turn read
  from environment variables, so the whole system is reconfigurable
  without touching source code.

Capabilities defined
--------------------
  general_reasoning  →  Gemma 4 12B     (default general LLM)
  coding             →  Qwen2.5-Coder 7B (code generation / analysis)
  embedding          →  nomic-embed-text (vector embeddings, 768-dim)

Future capabilities (registered as placeholders):
  vision             →  TBD (local VLM)
  ocr                →  TBD (local OCR engine)

Resource note
-------------
This registry does NOT load or initialise any model.
Models are loaded on-demand by their respective capability modules.
Only one large model should be in VRAM at a time on the RTX 3060 (6 GB).
"""

import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LLM_MODEL, CODING_MODEL, EMBEDDING_MODEL, OCR_MODEL, VISION_MODEL, OLLAMA_HOST


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable description of a model registered for a capability.

    Fields
    ------
    capability:
        Human-readable capability name.  Used as the registry key.
    model_name:
        The exact model tag as used by Ollama (e.g. "gemma4:12b").
    provider:
        Where the model runs.  Currently always "ollama".
    host:
        Ollama host URL.
    context_length:
        Approximate context window in tokens.  Used for chunking decisions.
    notes:
        Free-form notes for documentation / future reference.
    """
    capability: str
    model_name: str
    provider: str = "ollama"
    host: str = OLLAMA_HOST
    context_length: int = 4096
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ModelConfig] = {

    "general_reasoning": ModelConfig(
        capability="general_reasoning",
        model_name=LLM_MODEL,
        context_length=4096,
        notes=(
            "Primary general-purpose LLM. Used for RAG answer generation, "
            "summarisation, and reasoning. Partially GPU-offloaded on RTX 3060."
        ),
    ),

    "coding": ModelConfig(
        capability="coding",
        model_name=CODING_MODEL,
        context_length=8192,
        notes=(
            "Coding specialist. Used for code generation, review, debugging, "
            "and SQL generation. Large model — offload from VRAM before loading "
            "another large model."
        ),
    ),

    "embedding": ModelConfig(
        capability="embedding",
        model_name=EMBEDDING_MODEL,
        context_length=8192,
        notes=(
            "Text embedding model. Produces 768-dimensional vectors. "
            "Used for semantic search and RAG retrieval."
        ),
    ),

    # ---- Placeholders for future capabilities --------------------------------
    # These are defined so the orchestrator can reference them,
    # even before the underlying modules are implemented.

    "vision": ModelConfig(
        capability="vision",
        model_name=VISION_MODEL,
        provider="ollama",
        context_length=4096,
        notes=(
            "Vision-language model (qwen3-vl:4b) via Ollama multimodal interface. "
            "Accepts image + text prompt, returns natural-language analysis. "
            "Supports image description, visual Q&A, and structured extraction."
        ),
    ),

    "ocr": ModelConfig(
        capability="ocr",
        model_name=OCR_MODEL,
        provider="ollama",
        context_length=2048,
        notes=(
            "OCR model (glm-ocr) via Ollama multimodal interface. "
            "Accepts image input, returns extracted plain text. "
            "Output is post-processed to remove repetition loops."
        ),
    ),    
    "summarization": ModelConfig(
        capability="summarization",
        model_name=LLM_MODEL,           # reuse llama3.2 — no new model needed
        context_length=8192,
        notes=(
            "Summarization specialist. Same underlying model as "
            "general_reasoning, registered separately to demonstrate that "
            "new capabilities can be added by config edit alone. Used for "
            "condensing long documents into short summaries."
        ),
    ),

    "coding_fast": ModelConfig(
        capability="coding_fast",
        model_name="llama3.2:latest",   # fast path for simple code tasks
        context_length=4096,
        notes=(
            "Fast coding path for short/simple code tasks. Uses llama3.2 "
            "instead of qwen2.5-coder for lower latency. Registered as a "
            "distinct capability without code changes."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(capability: str) -> ModelConfig:
    """
    Return the ModelConfig for the requested capability.

    Parameters
    ----------
    capability:
        One of the keys in REGISTRY (e.g. "general_reasoning", "coding").

    Returns
    -------
    ModelConfig
        Frozen dataclass with model name, host, and metadata.

    Raises
    ------
    KeyError
        If the capability is not registered.
    ValueError
        If the capability is registered but has no model assigned yet
        (i.e. placeholder entries with empty model_name).
    """
    if capability not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(
            f"Unknown capability {capability!r}. "
            f"Available capabilities: {available}"
        )

    cfg = REGISTRY[capability]

    if not cfg.model_name:
        raise ValueError(
            f"Capability {capability!r} is registered as a placeholder "
            f"but has no model assigned. Check registry.py."
        )

    return cfg


def list_capabilities() -> list[dict]:
    """
    Return a summary list of all registered capabilities.
    Suitable for use in a health/status API endpoint.
    """
    return [
        {
            "capability": cfg.capability,
            "model": cfg.model_name or "(not assigned)",
            "provider": cfg.provider,
            "implemented": bool(cfg.model_name),
            "notes": cfg.notes,
        }
        for cfg in REGISTRY.values()
    ]
