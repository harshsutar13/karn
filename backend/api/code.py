"""
api/code.py

FastAPI router for the coding capability endpoints.

Uses Qwen2.5-Coder (7B) via models/coding.py.

Endpoints:
    POST /api/code/generate   — generate code from a natural-language task
    POST /api/code/review     — review provided code for quality and issues
    POST /api/code/explain    — explain what code does in plain language
    POST /api/code/debug      — identify and fix a bug
    POST /api/code/sql        — generate a SQL query from a description

Design:
    All business logic stays in models/coding.py.
    This file only handles HTTP concerns: schema validation, error mapping,
    and response serialisation.

Resource note:
    Qwen2.5-Coder 7B is the installed coding model. Requests may take
    depending on whether the model is already warm in memory.
    The think=False default keeps responses fast.
"""

import os
import sys
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.coding import (
    generate_code,
    review_code,
    explain_code,
    debug_code,
    generate_sql,
)
from models.registry import get_model

logger = logging.getLogger(__name__)
router = APIRouter()

# Expose which model is backing this router (for documentation/headers)
_MODEL_NAME = get_model("coding").model_name


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CodeGenerateRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Natural-language task description")
    language: str = Field(default="Python", description="Target programming language")
    think: bool = Field(default=False, description="Enable extended reasoning (slower, higher quality)")


class CodeReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Code to review")
    context: str = Field(default="", description="Optional context about the code")


class CodeExplainRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Code to explain")


class CodeDebugRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Buggy code")
    error: str = Field(..., min_length=1, description="Error message or description of the problem")


class SQLGenerateRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Natural-language description of the query")
    db_schema: str = Field(default="", description="Optional table/column definitions (e.g. 'users(id, name, email)')")


class CodeResponse(BaseModel):
    result: str
    model: str


# ---------------------------------------------------------------------------
# Error handler helper
# ---------------------------------------------------------------------------

def _handle_coding_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ConnectionError):
        logger.error("Ollama not reachable for coding: %s", exc)
        return HTTPException(status_code=503, detail=f"Coding model unavailable: {exc}")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Unexpected error in coding endpoint")
    return HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/api/code/generate", response_model=CodeResponse, tags=["Coding"])
def code_generate_endpoint(request: CodeGenerateRequest):
    """
    Generate code for a natural-language task description.

    Uses Qwen2.5-Coder 7B. Responses may take 15-90 seconds on first load.
    Set think=true for complex tasks requiring deeper reasoning (even slower).
    """
    try:
        result = generate_code(
            description=request.description,
            language=request.language,
            think=request.think,
        )
    except Exception as exc:
        raise _handle_coding_error(exc)
    return CodeResponse(result=result, model=_MODEL_NAME)


@router.post("/api/code/review", response_model=CodeResponse, tags=["Coding"])
def code_review_endpoint(request: CodeReviewRequest):
    """
    Review code for correctness, clarity, security, and performance.
    """
    try:
        result = review_code(code=request.code, context=request.context)
    except Exception as exc:
        raise _handle_coding_error(exc)
    return CodeResponse(result=result, model=_MODEL_NAME)


@router.post("/api/code/explain", response_model=CodeResponse, tags=["Coding"])
def code_explain_endpoint(request: CodeExplainRequest):
    """
    Explain what the provided code does in plain language.
    """
    try:
        result = explain_code(code=request.code)
    except Exception as exc:
        raise _handle_coding_error(exc)
    return CodeResponse(result=result, model=_MODEL_NAME)


@router.post("/api/code/debug", response_model=CodeResponse, tags=["Coding"])
def code_debug_endpoint(request: CodeDebugRequest):
    """
    Identify the root cause of a bug and return a corrected version.
    """
    try:
        result = debug_code(code=request.code, error=request.error)
    except Exception as exc:
        raise _handle_coding_error(exc)
    return CodeResponse(result=result, model=_MODEL_NAME)


@router.post("/api/code/sql", response_model=CodeResponse, tags=["Coding"])
def code_sql_endpoint(request: SQLGenerateRequest):
    """
    Generate a SQL query from a natural-language description.
    Optionally supply schema information for more accurate results.
    """
    try:
        result = generate_sql(description=request.description, schema=request.db_schema)
    except Exception as exc:
        raise _handle_coding_error(exc)
    return CodeResponse(result=result, model=_MODEL_NAME)
