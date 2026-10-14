"""
models/manager.py

Ensures only one Ollama model is resident in VRAM at a time.
Designed for GPUs with limited VRAM (e.g. RTX 3050 6GB).
"""

from __future__ import annotations

import logging
import threading

import ollama

from config import OLLAMA_HOST

logger = logging.getLogger(__name__)

_client = ollama.Client(host=OLLAMA_HOST)
_lock = threading.Lock()
_current_model: str | None = None

KEEP_ALIVE = "2m"

# Model names that are embedding-only — must be warmed up with .embed(),
# not .generate(). Anything with "embed" in the name is treated as such.
def _is_embedding_model(name: str) -> bool:
    return "embed" in name.lower()


def _warm_up(model: str) -> None:
    """Force Ollama to load `model` into VRAM now."""
    if _is_embedding_model(model):
        _client.embed(model=model, input="warmup", keep_alive=KEEP_ALIVE)
    else:
        _client.generate(model=model, prompt="", keep_alive=KEEP_ALIVE)


def _unload(model: str) -> None:
    """Force Ollama to evict a model from VRAM."""
    try:
        # Both generate() and embed() unload with keep_alive=0.
        if _is_embedding_model(model):
            _client.embed(model=model, input="", keep_alive=0)
        else:
            _client.generate(model=model, prompt="", keep_alive=0)
        logger.info("Unloaded model %r from VRAM", model)
    except Exception as exc:
        logger.warning("Failed to unload %r: %s", model, exc)


class use_model:
    """
    Context manager: guarantees `model` is loaded before the block runs.
    If a different model is currently resident, unloads it first.
    Thread-safe.
    """

    def __init__(self, model: str):
        self.model = model

    def __enter__(self):
        global _current_model
        _lock.acquire()

        try:
            if _current_model is not None and _current_model != self.model:
                logger.info("Swapping model: %r -> %r", _current_model, self.model)
                _unload(_current_model)
                _current_model = None

            _warm_up(self.model)
            _current_model = self.model
            return self

        except Exception:
            _lock.release()
            raise

    def __exit__(self, exc_type, exc, tb):
        _lock.release()
        return False


def current_model() -> str | None:
    return _current_model


def unload_all() -> None:
    global _current_model
    with _lock:
        if _current_model is not None:
            _unload(_current_model)
            _current_model = None