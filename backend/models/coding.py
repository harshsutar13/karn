"""
models/coding.py

Coding capability for SIH26 AI Workbench.

Uses Qwen2.5-Coder (7B) via Ollama for:
  - Code generation from a natural-language description
  - Code review and critique
  - Code explanation
  - Debugging assistance
  - SQL generation

Design principles
-----------------
- Model is NOT loaded until a function in this module is actually called.
- `think=False` is used for normal requests (fast path).
  Set think=True only for complex multi-step reasoning if needed later.
- The caller supplies a system prompt (or uses the provided defaults)
  so that this module stays reusable across different task types.
- All functions return plain strings. Structured parsing is the caller's
  responsibility.

Resource note
-------------
Qwen2.5-Coder 7B fits within the available GPU memory more comfortably.
Do not load it simultaneously with Gemma.
The orchestrator is responsible for serialising access.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama
from config import OLLAMA_HOST
from models.registry import get_model

logger = logging.getLogger(__name__)

# Retrieve config once at import time — does NOT start the model
_cfg = get_model("coding")


# ---------------------------------------------------------------------------
# Default system prompts
# ---------------------------------------------------------------------------

_SYSTEM_GENERATE = (
    "You are an expert software engineer. "
    "When given a task description, produce clean, working code. "
    "Add brief inline comments where helpful. "
    "Do not add unnecessary boilerplate or imports. "
    "Return only the code unless asked to explain."
)

_SYSTEM_REVIEW = (
    "You are a senior code reviewer. "
    "Analyse the provided code for correctness, clarity, security, and performance. "
    "Be concise and specific. List issues, then suggest improvements."
)

_SYSTEM_EXPLAIN = (
    "You are a patient technical mentor. "
    "Explain what the provided code does in plain language. "
    "Describe the overall purpose, then walk through key steps."
)

_SYSTEM_DEBUG = (
    "You are a debugging expert. "
    "Identify the root cause of the issue described, then provide a fix. "
    "Show the corrected code and explain the change."
)

_SYSTEM_SQL = (
    "You are a SQL expert. "
    "Generate correct, efficient SQL for the described task. "
    "Use standard SQL unless a specific database is mentioned. "
    "Return only the SQL query."
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _call(
    system_prompt: str,
    user_message: str,
    think: bool = False,
    stream: bool = False,
    history: list[dict] | None = None,
):
    """
    Call the coding model with a system + user message pair.

    Parameters
    ----------
    system_prompt:
        Instruction context for the model.
    user_message:
        The actual user request.
    think:
        If True, enables extended reasoning (slower, higher quality).
    stream:
        If True, yields text chunks instead of returning a single string.

    Returns
    -------
    str | Generator[dict]
        The model's text response (str) or a typed-chunk generator (stream=True).
        Stream dicts: {"kind": "chunk", "text": ...} or {"kind": "thinking", "text": ...}

    Raises
    ------
    ConnectionError
        If Ollama is not reachable.
    """
    from typing import Generator
    client = ollama.Client(host=OLLAMA_HOST)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    logger.debug(
        "Calling coding model %r (think=%s, stream=%s), user msg: %r",
        _cfg.model_name, think, stream, user_message[:80],
    )

    if stream:
        def _gen():
            try:
                for raw in client.chat(
                    model=_cfg.model_name,
                    messages=messages,
                    think=think,
                    stream=True,
                ):
                    thinking_text = (raw.message.thinking or "")
                    content_text  = (raw.message.content  or "")
                    if thinking_text:
                        yield {"kind": "thinking", "text": thinking_text}
                    if content_text:
                        yield {"kind": "chunk", "text": content_text}
            except Exception as exc:
                if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                    raise ConnectionError(
                        f"Cannot reach Ollama at {OLLAMA_HOST}. "
                        f"Ensure Ollama is running. Original: {exc}"
                    ) from exc
                raise
        return _gen()

    try:
        response = client.chat(
            model=_cfg.model_name,
            messages=messages,
            think=think,
        )
    except Exception as exc:
        if "connection" in str(exc).lower() or "refused" in str(exc).lower():
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_HOST}. "
                f"Ensure Ollama is running. Original: {exc}"
            ) from exc
        raise

    return response.message.content or ""


# ---------------------------------------------------------------------------
# Public API — one function per task type
# ---------------------------------------------------------------------------

def generate_code(description: str, language: str = "Python", think: bool = False, stream: bool = False, history: list[dict] | None = None):
    """
    Generate code for a natural-language task description.

    Parameters
    ----------
    description:
        What the code should do (e.g. "Parse a CSV file and compute column means").
    language:
        Target programming language. Default: "Python".
    think:
        Enable extended reasoning for complex tasks.
    history:
        Optional prior conversation messages, so follow-up requests like
        "now add error handling to that function" work.

    Returns
    -------
    str
        Generated code (and optionally brief explanation if the model adds one).
    """
    if not description or not description.strip():
        raise ValueError("generate_code: description must be non-empty.")

    prompt = f"Language: {language}\n\nTask: {description}"
    return _call(_SYSTEM_GENERATE, prompt, think=think, stream=stream, history=history)


def review_code(code: str, context: str = "") -> str:
    """
    Review the provided code for quality, correctness, and security.

    Parameters
    ----------
    code:
        The code to review.
    context:
        Optional extra context (e.g. "This is a FastAPI endpoint handler.").

    Returns
    -------
    str
        Code review with specific issues and suggested improvements.
    """
    if not code or not code.strip():
        raise ValueError("review_code: code must be non-empty.")

    parts = []
    if context:
        parts.append(f"Context: {context}\n")
    parts.append(f"Code to review:\n```\n{code}\n```")
    return _call(_SYSTEM_REVIEW, "\n".join(parts))


def explain_code(code: str) -> str:
    """
    Explain what the provided code does in plain language.

    Parameters
    ----------
    code:
        The code to explain.

    Returns
    -------
    str
        Plain-language explanation.
    """
    if not code or not code.strip():
        raise ValueError("explain_code: code must be non-empty.")

    return _call(_SYSTEM_EXPLAIN, f"```\n{code}\n```")


def debug_code(code: str, error: str) -> str:
    """
    Identify and fix a bug in the provided code.

    Parameters
    ----------
    code:
        The buggy code.
    error:
        The error message or description of the problem.

    Returns
    -------
    str
        Root cause analysis and corrected code.
    """
    if not code or not code.strip():
        raise ValueError("debug_code: code must be non-empty.")
    if not error or not error.strip():
        raise ValueError("debug_code: error description must be non-empty.")

    prompt = f"Error / Problem:\n{error}\n\nCode:\n```\n{code}\n```"
    return _call(_SYSTEM_DEBUG, prompt)


def generate_sql(description: str, schema: str = "") -> str:
    """
    Generate a SQL query for a natural-language description.

    Parameters
    ----------
    description:
        What the query should do.
    schema:
        Optional table/column definitions to guide the model.

    Returns
    -------
    str
        SQL query string.
    """
    if not description or not description.strip():
        raise ValueError("generate_sql: description must be non-empty.")

    parts = []
    if schema:
        parts.append(f"Schema:\n{schema}\n")
    parts.append(f"Task: {description}")
    return _call(_SYSTEM_SQL, "\n".join(parts))
