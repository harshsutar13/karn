"""
test_models.py

Tests for the model registry and capability modules.

Tests:
    REGISTRY TESTS (no model calls — pure config/logic)
    1. Registry imports correctly
    2. get_model() returns correct config for each implemented capability
    3. get_model() raises KeyError for unknown capability
    4. get_model() raises ValueError for placeholder (vision/ocr)
    5. list_capabilities() returns all 5 entries
    6. config.py values are reflected correctly

    GENERAL REASONING TESTS (calls Gemma — fast)
    7. generate_response() returns a non-empty string
    8. generate_response() uses the correct model (gemma4:12b)

    CODING TESTS (calls Qwen3-Coder — may be slow)
    9. generate_code() returns non-empty code for a simple task
   10. review_code() returns non-empty review text
   11. explain_code() returns non-empty explanation
   12. generate_sql() returns SQL-looking output
   13. generate_code() raises ValueError for empty description
   14. debug_code() raises ValueError for missing error description

Run with:
    $env:PYTHONUTF8="1"
    & "C:\\WORK DRIVE\\SIH26\\.venv\\Scripts\\python.exe" `
        "C:\\WORK DRIVE\\SIH26\\qdrant\\run_all_tests.py" `
        "C:\\WORK DRIVE\\SIH26\\backend\\models\\test_models.py"
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.registry import get_model, list_capabilities, REGISTRY, ModelConfig
from models.llm import generate_response
from config import LLM_MODEL, CODING_MODEL, EMBEDDING_MODEL

SEP = "=" * 60


def print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ============================================================
# REGISTRY TESTS — no model calls
# ============================================================

def test_registry_imports():
    print_section("TEST 1 -- Registry imports correctly")
    assert REGISTRY is not None
    assert isinstance(REGISTRY, dict)
    print(f"  Capabilities registered: {list(REGISTRY.keys())}")
    print("  [PASS] Registry imports OK")


def test_get_model_implemented():
    print_section("TEST 2 -- get_model() returns correct config for implemented capabilities")
    for cap in ("general_reasoning", "coding", "embedding"):
        cfg = get_model(cap)
        assert isinstance(cfg, ModelConfig), f"Expected ModelConfig for {cap!r}"
        assert cfg.model_name, f"model_name is empty for {cap!r}"
        print(f"  {cap:20s} → {cfg.model_name}")
    print("  [PASS] All implemented capabilities return ModelConfig with model_name")


def test_get_model_unknown():
    print_section("TEST 3 -- get_model() raises KeyError for unknown capability")
    try:
        get_model("does_not_exist")
        raise AssertionError("Expected KeyError was not raised")
    except KeyError as exc:
        print(f"  KeyError raised: {exc}")
        print("  [PASS] Unknown capability raises KeyError")


def test_get_model_placeholder():
    print_section("TEST 4 -- get_model() raises ValueError for placeholder capabilities")
    for cap in ("vision", "ocr"):
        try:
            get_model(cap)
            raise AssertionError(f"Expected ValueError for placeholder {cap!r}")
        except ValueError as exc:
            print(f"  {cap}: ValueError raised correctly — {exc}")
    print("  [PASS] Placeholder capabilities raise ValueError")


def test_list_capabilities():
    print_section("TEST 5 -- list_capabilities() returns all 5 entries")
    caps = list_capabilities()
    assert len(caps) == 5, f"Expected 5, got {len(caps)}"
    for c in caps:
        assert "capability" in c
        assert "model" in c
        assert "implemented" in c
        print(f"  {c['capability']:20s} implemented={c['implemented']}  model={c['model']}")
    print("  [PASS] list_capabilities() returned 5 entries")


def test_config_reflected():
    print_section("TEST 6 -- config.py values reflected correctly in registry")
    assert get_model("general_reasoning").model_name == LLM_MODEL, (
        f"general_reasoning model mismatch: expected {LLM_MODEL!r}"
    )
    assert get_model("coding").model_name == CODING_MODEL, (
        f"coding model mismatch: expected {CODING_MODEL!r}"
    )
    assert get_model("embedding").model_name == EMBEDDING_MODEL, (
        f"embedding model mismatch: expected {EMBEDDING_MODEL!r}"
    )
    print(f"  general_reasoning → {get_model('general_reasoning').model_name}")
    print(f"  coding            → {get_model('coding').model_name}")
    print(f"  embedding         → {get_model('embedding').model_name}")
    print("  [PASS] Config values correctly reflected in registry")


# ============================================================
# GENERAL REASONING TESTS (Gemma)
# ============================================================

def test_general_reasoning_response():
    print_section("TEST 7 -- generate_response() returns non-empty string (Gemma)")
    t0 = time.perf_counter()
    result = generate_response("Reply with exactly the word PONG and nothing else.")
    elapsed = time.perf_counter() - t0
    print(f"  Response : {result!r}")
    print(f"  Time     : {elapsed:.2f}s")
    assert isinstance(result, str) and len(result.strip()) > 0
    print("  [PASS] generate_response() returned non-empty string")


def test_general_reasoning_model_name():
    print_section("TEST 8 -- general_reasoning uses correct model name from config")
    cfg = get_model("general_reasoning")
    assert cfg.model_name == LLM_MODEL
    print(f"  Model: {cfg.model_name}")
    print("  [PASS] Model name matches config")


# ============================================================
# CODING TESTS (Qwen3-Coder)
# ============================================================

def test_generate_code():
    print_section("TEST 9 -- generate_code() returns code for a simple task")
    t0 = time.perf_counter()
    result = generate_code(
        description="Write a Python function that returns the factorial of n using recursion.",
        language="Python",
    )
    elapsed = time.perf_counter() - t0
    print(f"  Time    : {elapsed:.2f}s")
    print(f"  Output  :\n")
    for line in result.strip().split("\n")[:20]:
        print(f"    {line}")
    assert isinstance(result, str) and len(result.strip()) > 10
    assert "def " in result or "function" in result.lower()
    print("  [PASS] generate_code() returned code-looking output")


def test_review_code():
    print_section("TEST 10 -- review_code() returns non-empty review")
    sample = "def add(a, b):\n    return a + b\n"
    t0 = time.perf_counter()
    result = review_code(sample, context="Simple utility function")
    elapsed = time.perf_counter() - t0
    print(f"  Time    : {elapsed:.2f}s")
    print(f"  Review  : {result[:200]!r}")
    assert isinstance(result, str) and len(result.strip()) > 10
    print("  [PASS] review_code() returned non-empty review")


def test_explain_code():
    print_section("TEST 11 -- explain_code() returns non-empty explanation")
    sample = "result = [x**2 for x in range(10) if x % 2 == 0]"
    t0 = time.perf_counter()
    result = explain_code(sample)
    elapsed = time.perf_counter() - t0
    print(f"  Time       : {elapsed:.2f}s")
    print(f"  Explanation: {result[:300]!r}")
    assert isinstance(result, str) and len(result.strip()) > 10
    print("  [PASS] explain_code() returned non-empty explanation")


def test_generate_sql():
    print_section("TEST 12 -- generate_sql() returns SQL-looking output")
    t0 = time.perf_counter()
    result = generate_sql(
        description="Select all employees hired after 2020 ordered by last name.",
        schema="employees(id, first_name, last_name, hire_date, department_id)",
    )
    elapsed = time.perf_counter() - t0
    print(f"  Time  : {elapsed:.2f}s")
    print(f"  SQL   : {result.strip()!r}")
    result_upper = result.upper()
    assert "SELECT" in result_upper or "select" in result.lower()
    print("  [PASS] generate_sql() returned SQL-looking output")


def test_generate_code_empty_raises():
    print_section("TEST 13 -- generate_code() raises ValueError for empty description")
    try:
        generate_code("")
        raise AssertionError("Expected ValueError not raised")
    except ValueError as exc:
        print(f"  ValueError raised: {exc}")
        print("  [PASS] Empty description raises ValueError")


def test_debug_code_missing_error_raises():
    print_section("TEST 14 -- debug_code() raises ValueError for empty error description")
    try:
        debug_code("x = 1 / 0", "")
        raise AssertionError("Expected ValueError not raised")
    except ValueError as exc:
        print(f"  ValueError raised: {exc}")
        print("  [PASS] Missing error description raises ValueError")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Import here so the import itself is part of the test run
    from models.coding import (
        generate_code, review_code, explain_code,
        generate_sql, debug_code,
    )

    print(f"\n{SEP}")
    print("  MODEL CAPABILITY TEST SUITE")
    print(SEP)
    print(f"  LLM_MODEL      : {LLM_MODEL}")
    print(f"  CODING_MODEL   : {CODING_MODEL}")
    print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")

    # Registry tests (no model calls)
    test_registry_imports()
    test_get_model_implemented()
    test_get_model_unknown()
    test_get_model_placeholder()
    test_list_capabilities()
    test_config_reflected()

    # Gemma tests
    test_general_reasoning_response()
    test_general_reasoning_model_name()

    # Qwen tests (may be slow on first call)
    print(f"\n{SEP}")
    print("  CODING TESTS (Qwen2.5-Coder 7B — may be slow on first call)")
    print(SEP)
    test_generate_code()
    test_review_code()
    test_explain_code()
    test_generate_sql()
    test_generate_code_empty_raises()
    test_debug_code_missing_error_raises()

    print(f"\n{SEP}")
    print("  ALL MODEL CAPABILITY TESTS PASSED")
    print(SEP)
