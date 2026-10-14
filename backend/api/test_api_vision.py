"""
test_api_vision.py

Smoke tests for POST /api/vision/analyze and related endpoints.

Tests:
     1. POST /api/vision/analyze  — valid image + description -> 200
     2. POST /api/vision/analyze  — visual Q&A -> 200
     3. Response contains "P-101A"
     4. Response contains "7.1"
     5. Missing image path -> 404
     6. Empty image_path -> 422
     7. Empty prompt -> 422
     8. Response contains model "qwen3-vl:4b"
     9. GET  /api/chat          — existing chat still works -> 200
    10. GET  /api/capabilities  — capability endpoint still works -> 200
    11. Vision capability is reported as implemented
    12. POST /api/vision/analyze — response schema has "answer" and "model" fields

Run with:
    $env:PYTHONUTF8="1"
    & "C:\\WORK DRIVE\\SIH26\\.venv\\Scripts\\python.exe" `
        "C:\\WORK DRIVE\\SIH26\\qdrant\\run_server_tests.py" `
        "C:\\WORK DRIVE\\SIH26\\backend\\api\\test_api_vision.py"
"""

import sys
import os
import json
import time
import urllib.request
import urllib.error

SEP = "=" * 60
BASE_URL = "http://localhost:8000"
VISION_IMAGE = r"C:\WORK DRIVE\SIH26\data\vision_test.png"
MISSING_IMAGE = r"C:\WORK DRIVE\SIH26\data\no_such_image_xyz.png"


def section(t: str) -> None:
    print(f"\n{SEP}\n  {t}\n{SEP}")


def get(path: str, timeout: int = 10) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as r:
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

def test_vision_description():
    section("TEST 1 -- POST /api/vision/analyze (description prompt) -> 200")
    t0 = time.perf_counter()
    status, body = post("/api/vision/analyze", {
        "image_path": VISION_IMAGE,
        "prompt": "Describe what you see in this image.",
    })
    elapsed = time.perf_counter() - t0
    print(f"  Status : {status}")
    print(f"  Time   : {elapsed:.1f}s")
    print(f"  Answer : {body.get('answer', '')[:200]!r}")
    assert status == 200, f"Expected 200, got {status}. Body: {body}"
    assert body.get("answer"), "Expected non-empty 'answer' field"
    print("  [PASS] Description prompt returned 200 with answer")


def test_vision_vqa():
    section("TEST 2 -- POST /api/vision/analyze (visual Q&A) -> 200")
    t0 = time.perf_counter()
    status, body = post("/api/vision/analyze", {
        "image_path": VISION_IMAGE,
        "prompt": "What equipment is shown and what is its vibration value?",
    })
    elapsed = time.perf_counter() - t0
    print(f"  Status : {status}")
    print(f"  Time   : {elapsed:.1f}s")
    print(f"  Answer : {body.get('answer', '')!r}")
    assert status == 200, f"Expected 200, got {status}. Body: {body}"
    # Store for tests 3 & 4
    return body


def test_answer_contains_pump(body: dict):
    section("TEST 3 -- Response contains 'P-101A'")
    answer = body.get("answer", "")
    found = "P-101A" in answer or "p-101" in answer.lower()
    print(f"  Answer snippet: {answer[:150]!r}")
    print(f"  P-101A found  : {'YES' if found else 'NO'}")
    assert found, f"Expected 'P-101A' in answer. Got:\n{answer!r}"
    print("  [PASS] P-101A recognized")


def test_answer_contains_vibration(body: dict):
    section("TEST 4 -- Response contains '7.1'")
    answer = body.get("answer", "")
    found = "7.1" in answer
    print(f"  Answer snippet: {answer[:150]!r}")
    print(f"  7.1 found     : {'YES' if found else 'NO'}")
    assert found, f"Expected '7.1' in answer. Got:\n{answer!r}"
    print("  [PASS] Vibration value 7.1 recognized")


def test_missing_image():
    section("TEST 5 -- Missing image path -> 404")
    status, body = post("/api/vision/analyze", {
        "image_path": MISSING_IMAGE,
        "prompt": "Describe this image.",
    })
    print(f"  Status : {status}")
    print(f"  Detail : {body.get('detail', str(body))[:200]}")
    assert status == 404, f"Expected 404, got {status}. Body: {body}"
    print("  [PASS] Missing image returns 404")


def test_empty_image_path():
    section("TEST 6 -- Empty image_path -> 422")
    status, body = post("/api/vision/analyze", {
        "image_path": "",
        "prompt": "Describe this image.",
    })
    print(f"  Status : {status}")
    print(f"  Detail : {str(body)[:200]}")
    assert status == 422, f"Expected 422, got {status}. Body: {body}"
    print("  [PASS] Empty image_path rejected with 422")


def test_empty_prompt():
    section("TEST 7 -- Empty prompt -> 422")
    status, body = post("/api/vision/analyze", {
        "image_path": VISION_IMAGE,
        "prompt": "",
    })
    print(f"  Status : {status}")
    print(f"  Detail : {str(body)[:200]}")
    assert status == 422, f"Expected 422, got {status}. Body: {body}"
    print("  [PASS] Empty prompt rejected with 422")


def test_model_in_response():
    section("TEST 8 -- Response contains model 'qwen3-vl:4b'")
    status, body = post("/api/vision/analyze", {
        "image_path": VISION_IMAGE,
        "prompt": "What is shown here?",
    }, timeout=120)
    model = body.get("model", "")
    print(f"  Status : {status}")
    print(f"  Model  : {model!r}")
    assert status == 200, f"Expected 200, got {status}"
    assert model == "qwen3-vl:4b", f"Expected 'qwen3-vl:4b', got {model!r}"
    print("  [PASS] Response model field is 'qwen3-vl:4b'")


def test_existing_chat():
    section("TEST 9 -- GET /api/chat still works -> 200")
    status, body = post("/api/chat", {"message": "Say HELLO"}, timeout=60)
    print(f"  Status   : {status}")
    print(f"  Response : {body.get('response', str(body))[:100]!r}")
    assert status == 200, f"Expected 200, got {status}"
    print("  [PASS] /api/chat endpoint unaffected")


def test_capabilities_endpoint():
    section("TEST 10 -- GET /api/capabilities still works -> 200")
    status, body = get("/api/capabilities")
    print(f"  Status : {status}")
    print(f"  Count  : {body.get('count')}")
    assert status == 200, f"Expected 200, got {status}"
    assert body.get("count") == 5
    print("  [PASS] /api/capabilities endpoint unaffected")


def test_vision_capability_implemented():
    section("TEST 11 -- Vision capability is reported as implemented")
    status, body = get("/api/capabilities")
    caps = {c["capability"]: c for c in body.get("capabilities", [])}
    vision = caps.get("vision", {})
    print(f"  vision entry: {vision}")
    assert vision.get("implemented") is True, (
        f"Expected vision implemented=True, got: {vision}"
    )
    assert vision.get("model") == "qwen3-vl:4b", (
        f"Expected model 'qwen3-vl:4b', got: {vision.get('model')}"
    )
    print("  [PASS] Vision capability shown as implemented with correct model")


def test_response_schema():
    section("TEST 12 -- Response schema has 'answer' and 'model' fields")
    status, body = post("/api/vision/analyze", {
        "image_path": VISION_IMAGE,
        "prompt": "What equipment status is shown?",
    }, timeout=120)
    print(f"  Status : {status}")
    print(f"  Keys   : {list(body.keys())}")
    assert status == 200
    assert "answer" in body, f"Missing 'answer' field. Got keys: {list(body.keys())}"
    assert "model"  in body, f"Missing 'model' field.  Got keys: {list(body.keys())}"
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert isinstance(body["model"],  str) and body["model"].strip()
    print(f"  answer : {body['answer'][:100]!r}")
    print(f"  model  : {body['model']!r}")
    print("  [PASS] Response schema correct")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  VISION API SMOKE TESTS")
    print(SEP)
    print(f"  Base URL    : {BASE_URL}")
    print(f"  Vision image: {VISION_IMAGE}")

    # --- Regression: existing endpoints ---
    section("REGRESSION — Existing endpoints")
    test_existing_chat()
    test_capabilities_endpoint()
    test_vision_capability_implemented()

    # --- Vision endpoint: validation (no model calls) ---
    section("VALIDATION TESTS (Pydantic — no model calls)")
    test_empty_image_path()
    test_empty_prompt()

    # --- Vision endpoint: live model tests ---
    section("LIVE VISION TESTS (qwen3-vl:4b via Ollama)")
    test_vision_description()
    vqa_body = test_vision_vqa()
    test_answer_contains_pump(vqa_body)
    test_answer_contains_vibration(vqa_body)
    test_missing_image()
    test_model_in_response()
    test_response_schema()

    print(f"\n{SEP}")
    print("  ALL VISION API TESTS PASSED  (12/12)")
    print(SEP)
