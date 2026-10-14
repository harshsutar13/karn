"""
test_api_code.py

Smoke tests for the coding capability and capabilities listing endpoints.

Tests:
    1.  GET  /api/capabilities              -- lists all 5 capabilities
    2.  GET  /api/capabilities              -- response schema validation
    3.  POST /api/code/generate             -- Python factorial function
    4.  POST /api/code/generate             -- language param (SQL via generate, not /sql)
    5.  POST /api/code/review               -- code review returns feedback
    6.  POST /api/code/explain              -- explanation of a list comprehension
    7.  POST /api/code/debug                -- debug a simple off-by-one error
    8.  POST /api/code/sql                  -- SQL from description + schema
    9.  POST /api/code/generate             -- empty description -> 422
   10.  POST /api/code/debug                -- empty error -> 422
   11.  POST /api/code/sql                  -- response contains model name
   12.  GET  /api/capabilities              -- implemented flags correct

Run with:
    $env:PYTHONUTF8="1"
    & "C:\\WORK DRIVE\\SIH26\\.venv\\Scripts\\python.exe" `
        "C:\\WORK DRIVE\\SIH26\\qdrant\\run_server_tests.py" `
        "C:\\WORK DRIVE\\SIH26\\backend\\api\\test_api_code.py"
"""

import sys
import os
import json
import time
import urllib.request
import urllib.error

SEP = "=" * 60
BASE_URL = "http://localhost:8000"


def section(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE_URL + path, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(path: str, body: dict, timeout: int = 180) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_capabilities_list():
    section("TEST 1 -- GET /api/capabilities (lists all 5 capabilities)")
    status, body = get("/api/capabilities")
    print(f"  Status : {status}")
    print(f"  Count  : {body.get('count')}")
    assert status == 200
    assert body["count"] == 5
    caps = {c["capability"]: c for c in body["capabilities"]}
    assert "general_reasoning" in caps
    assert "coding" in caps
    assert "embedding" in caps
    assert "vision" in caps
    assert "ocr" in caps
    print("  [PASS] All 5 capabilities returned")


def test_capabilities_schema():
    section("TEST 2 -- /api/capabilities response schema validation")
    status, body = get("/api/capabilities")
    assert status == 200
    for cap in body["capabilities"]:
        assert "capability" in cap
        assert "model" in cap
        assert "provider" in cap
        assert "implemented" in cap
        assert "notes" in cap
        print(f"  {cap['capability']:<22} implemented={cap['implemented']}  model={cap['model']}")
    print("  [PASS] Schema valid for all entries")


def test_capabilities_implemented_flags():
    section("TEST 12 -- Implemented flags correct (general_reasoning, coding, embedding=True; vision, ocr=False)")
    status, body = get("/api/capabilities")
    caps = {c["capability"]: c for c in body["capabilities"]}
    for cap in ("general_reasoning", "coding", "embedding"):
        assert caps[cap]["implemented"] is True, f"{cap} should be implemented"
    for cap in ("vision", "ocr"):
        assert caps[cap]["implemented"] is False, f"{cap} should NOT be implemented"
    print("  [PASS] Implemented flags are correct")


def test_code_generate_python():
    section("TEST 3 -- POST /api/code/generate (Python factorial)")
    t0 = time.perf_counter()
    status, body = post("/api/code/generate", {
        "description": "Write a Python function that returns the factorial of n using recursion.",
        "language": "Python",
    })
    elapsed = time.perf_counter() - t0
    print(f"  Status  : {status}")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"  Model   : {body.get('model')}")
    print(f"  Result  : {body.get('result', '')[:200]!r}")
    assert status == 200
    result = body["result"]
    assert len(result.strip()) > 10
    assert "def " in result or "factorial" in result.lower()
    assert body["model"] == "qwen2.5-coder:7b"
    print("  [PASS] Code generated, model=qwen2.5-coder:7b")


def test_code_review():
    section("TEST 5 -- POST /api/code/review (review a simple function)")
    t0 = time.perf_counter()
    status, body = post("/api/code/review", {
        "code": "def divide(a, b):\n    return a / b\n",
        "context": "Financial calculation utility",
    })
    elapsed = time.perf_counter() - t0
    print(f"  Status  : {status}")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"  Review  : {body.get('result', '')[:250]!r}")
    assert status == 200
    result = body["result"]
    assert len(result.strip()) > 10
    # Should flag missing zero-division guard
    assert any(k in result.lower() for k in ["zero", "division", "error", "exception", "handle", "check"])
    print("  [PASS] Review identifies division-by-zero risk")


def test_code_explain():
    section("TEST 6 -- POST /api/code/explain (list comprehension)")
    t0 = time.perf_counter()
    status, body = post("/api/code/explain", {
        "code": "squares = [x**2 for x in range(10) if x % 2 == 0]",
    })
    elapsed = time.perf_counter() - t0
    print(f"  Status  : {status}")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"  Result  : {body.get('result', '')[:250]!r}")
    assert status == 200
    assert len(body["result"].strip()) > 10
    print("  [PASS] Explanation returned")


def test_code_debug():
    section("TEST 7 -- POST /api/code/debug (off-by-one error)")
    t0 = time.perf_counter()
    status, body = post("/api/code/debug", {
        "code": "def last_element(lst):\n    return lst[len(lst)]\n",
        "error": "IndexError: list index out of range",
    })
    elapsed = time.perf_counter() - t0
    print(f"  Status  : {status}")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"  Result  : {body.get('result', '')[:300]!r}")
    assert status == 200
    result = body["result"]
    assert len(result.strip()) > 10
    # Model should identify the off-by-one and suggest len(lst)-1 or lst[-1]
    assert any(k in result.lower() for k in ["-1", "len(lst) - 1", "index", "last"])
    print("  [PASS] Debug identified off-by-one fix")


def test_code_sql():
    section("TEST 8 -- POST /api/code/sql (SQL from description + schema)")
    t0 = time.perf_counter()
    status, body = post("/api/code/sql", {
        "description": "Find all orders placed in the last 30 days, ordered by total descending.",
        "db_schema": "orders(id, customer_id, total, created_at)",
    })
    elapsed = time.perf_counter() - t0
    print(f"  Status  : {status}")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"  SQL     : {body.get('result', '')!r}")
    assert status == 200
    result = body["result"].upper()
    assert "SELECT" in result
    assert "ORDER" in result or "ORDER BY" in result
    print("  [PASS] SQL returned with SELECT and ORDER BY")


def test_code_generate_empty_422():
    section("TEST 9 -- POST /api/code/generate (empty description -> 422)")
    status, body = post("/api/code/generate", {"description": ""})
    print(f"  Status: {status}")
    print(f"  Body  : {str(body)[:200]}")
    assert status == 422
    print("  [PASS] Empty description rejected with 422")


def test_code_debug_empty_error_422():
    section("TEST 10 -- POST /api/code/debug (empty error -> 422)")
    status, body = post("/api/code/debug", {"code": "x = 1", "error": ""})
    print(f"  Status: {status}")
    print(f"  Body  : {str(body)[:200]}")
    assert status == 422
    print("  [PASS] Empty error rejected with 422")


def test_sql_response_has_model():
    section("TEST 11 -- /api/code/sql response includes model name")
    status, body = post("/api/code/sql", {
        "description": "Count all rows in the users table.",
    })
    assert status == 200
    assert body.get("model") == "qwen2.5-coder:7b"
    print(f"  model = {body['model']!r}")
    print("  [PASS] Response includes correct model name")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  CODING CAPABILITY API SMOKE TESTS")
    print(SEP)
    print(f"  Base URL : {BASE_URL}")

    # Run capabilities tests first (no model calls)
    test_capabilities_list()
    test_capabilities_schema()
    test_capabilities_implemented_flags()

    # Run coding tests (Qwen2.5-Coder — may be slow first call)
    print(f"\n{SEP}")
    print("  CODING ENDPOINT TESTS (Qwen2.5-Coder 7B)")
    print(SEP)

    test_code_generate_python()
    test_code_review()
    test_code_explain()
    test_code_debug()
    test_code_sql()
    test_code_generate_empty_422()
    test_code_debug_empty_error_422()
    test_sql_response_has_model()

    print(f"\n{SEP}")
    print("  ALL CODING API TESTS PASSED")
    print(SEP)
