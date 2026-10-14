"""
test_api.py

Smoke-test the FastAPI endpoints.

Tests:
    1. GET  /                   -- health check
    2. POST /api/chat           -- existing LLM endpoint (must still work)
    3. GET  /api/rag/collection -- collection not found before ingestion
    4. POST /api/documents/ingest -- upload refinery_maintenance.txt
    5. GET  /api/rag/collection -- check points are indexed
    6. POST /api/rag/query      -- grounded question with answer in document
    7. POST /api/rag/query      -- question with no context (refusal behavior)
    8. POST /api/rag/query      -- bad request (empty question -> 422)

Run with:
    $env:PYTHONUTF8="1"
    & "C:\\WORK DRIVE\\SIH26\\.venv\\Scripts\\python.exe" `
        "C:\\WORK DRIVE\\SIH26\\qdrant\\run_server_tests.py" `
        "C:\\WORK DRIVE\\SIH26\\backend\\api\\test_api.py"
"""


import sys
import os
import json
import time
import urllib.request
import urllib.error
import urllib.parse

SEP = "=" * 60
BASE_URL = "http://localhost:8000"
TEST_DOC = r"C:\WORK DRIVE\SIH26\data\documents\refinery_maintenance.txt"
TEST_COLLECTION = "sih26_api_test"


def print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def get(path: str) -> tuple[int, dict]:
    url = BASE_URL + path
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_json(path: str, body: dict) -> tuple[int, dict]:
    url = BASE_URL + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_multipart_file(path: str, file_path: str, extra_fields: dict | None = None) -> tuple[int, dict]:
    """Simple multipart/form-data POST for file upload."""
    import email.generator
    import io
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase

    boundary = "BOUNDARY_SIH26_TEST"
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as fh:
        file_data = fh.read()

    body_parts = []

    # File field
    body_parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: text/plain\r\n\r\n'
    )
    body = "".join(body_parts).encode() + file_data + b"\r\n"

    # Extra form fields
    for key, val in (extra_fields or {}).items():
        body += (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f'{val}\r\n'
        ).encode()

    body += f'--{boundary}--\r\n'.encode()

    url = BASE_URL + path
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body_str = e.read()
        try:
            return e.code, json.loads(body_str)
        except Exception:
            return e.code, {"detail": body_str.decode(errors="replace")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health():
    print_section("TEST 1 -- GET / (health check)")
    status, body = get("/")
    print(f"  Status: {status}")
    print(f"  Body  : {body}")
    assert status == 200, f"Expected 200, got {status}"
    assert body.get("status") == "online"
    print("  [PASS] Health check OK")


def test_existing_chat():
    print_section("TEST 2 -- POST /api/chat (existing endpoint must still work)")
    status, body = post_json("/api/chat", {"message": "Say 'hello' in one word."})
    print(f"  Status  : {status}")
    print(f"  Response: {body.get('response', '')[:100]!r}")
    assert status == 200, f"Expected 200, got {status}"
    assert "response" in body
    assert len(body["response"]) > 0
    print("  [PASS] /api/chat still working")


def test_collection_not_found():
    print_section("TEST 3 -- GET /api/rag/collection (collection not yet created)")
    status, body = get(f"/api/rag/collection?collection={TEST_COLLECTION}")
    print(f"  Status: {status}")
    print(f"  Body  : {body}")
    assert status == 404, f"Expected 404 for nonexistent collection, got {status}"
    print("  [PASS] 404 returned for nonexistent collection")


def test_document_ingest():
    print_section("TEST 4 -- POST /api/documents/ingest")
    print(f"  File: {TEST_DOC}")

    t0 = time.perf_counter()
    status, body = post_multipart_file(
        "/api/documents/ingest",
        TEST_DOC,
        extra_fields={"collection": TEST_COLLECTION, "department": "Operations"},
    )
    elapsed = time.perf_counter() - t0

    print(f"  Status          : {status}")
    print(f"  Response        : {json.dumps(body, indent=4)}")
    print(f"  Elapsed         : {elapsed:.2f}s")

    assert status == 200, f"Expected 200, got {status}\nBody: {body}"
    assert body["chunks_upserted"] > 0
    assert body["collection"] == TEST_COLLECTION
    print(f"  [PASS] Ingested {body['chunks_upserted']} chunks from {body['filename']}")
    return body


def test_collection_after_ingest():
    print_section("TEST 5 -- GET /api/rag/collection (after ingestion)")
    status, body = get(f"/api/rag/collection?collection={TEST_COLLECTION}")
    print(f"  Status : {status}")
    print(f"  Body   : {body}")
    assert status == 200, f"Expected 200, got {status}"
    assert body["points_count"] > 0
    assert body["vector_size"] == 768
    print(f"  [PASS] Collection has {body['points_count']} points, vector_size=768")


def test_rag_query_with_context():
    print_section("TEST 6 -- POST /api/rag/query (answer exists in document)")
    payload = {
        "question": "What is the bearing temperature alarm setpoint for pump P-101A?",
        "top_k": 4,
        "collection": TEST_COLLECTION,
    }
    print(f"  Question: {payload['question']!r}")

    t0 = time.perf_counter()
    status, body = post_json("/api/rag/query", payload)
    elapsed = time.perf_counter() - t0

    print(f"  Status   : {status}")
    print(f"  Time     : {elapsed:.2f}s")
    print(f"  Context  : {body.get('context_used')}")
    print(f"  Chunks   : {body.get('chunks_retrieved')}")
    print(f"  Answer   : {body.get('answer', '')[:300]!r}")
    print(f"  Sources  : {len(body.get('sources', []))} items")
    for s in body.get("sources", []):
        print(f"    [{s['chunk_index']}] {s['filename']} score={s['score']:.4f}")

    assert status == 200, f"Expected 200, got {status}"
    assert body["context_used"] is True
    assert "85" in body["answer"] or "temperature" in body["answer"].lower()
    assert len(body["sources"]) > 0
    print("  [PASS] RAG returned grounded answer with sources")


def test_rag_query_no_context():
    print_section("TEST 7 -- POST /api/rag/query (no context — refusal)")
    payload = {
        "question": "What movies are playing at the cinema this weekend?",
        "top_k": 3,
        "score_threshold": 0.80,
        "collection": TEST_COLLECTION,
    }
    print(f"  Question: {payload['question']!r}")

    t0 = time.perf_counter()
    status, body = post_json("/api/rag/query", payload)
    elapsed = time.perf_counter() - t0

    print(f"  Status   : {status}")
    print(f"  Time     : {elapsed:.2f}s")
    print(f"  Context  : {body.get('context_used')}")
    print(f"  Chunks   : {body.get('chunks_retrieved')}")
    print(f"  Answer   : {body.get('answer', '')[:300]!r}")

    assert status == 200
    if not body["context_used"]:
        print("  [PASS] No context retrieved — model correctly signaled absence")
    else:
        answer_lower = body["answer"].lower()
        refusal = any(k in answer_lower for k in [
            "not contain", "insufficient", "not found", "no information", "available documents"
        ])
        print(f"  [NOTE] Some context retrieved; LLM refusal={refusal}")


def test_bad_request():
    print_section("TEST 8 -- POST /api/rag/query (empty question -> 422)")
    status, body = post_json("/api/rag/query", {"question": ""})
    print(f"  Status: {status}")
    print(f"  Body  : {str(body)[:200]}")
    assert status == 422, f"Expected 422 Unprocessable Entity for empty question, got {status}"
    print("  [PASS] Empty question correctly rejected with 422")


def test_invalid_file_type():
    print_section("TEST 9 -- POST /api/documents/ingest (invalid file type -> 400)")
    # Upload a .py file — not in the allowed set
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as tmp:
        tmp.write(b"print('hello')")
        tmp_path = tmp.name

    try:
        status, body = post_multipart_file(
            "/api/documents/ingest",
            tmp_path,
            extra_fields={"collection": TEST_COLLECTION},
        )
    finally:
        os.unlink(tmp_path)

    print(f"  Status: {status}")
    print(f"  Body  : {str(body)[:200]}")
    assert status == 400, f"Expected 400 for unsupported file type, got {status}"
    print("  [PASS] Unsupported file type correctly rejected with 400")


def test_reingest_idempotency():
    print_section("TEST 10 -- POST /api/documents/ingest (repeated upload is idempotent)")

    # Get current count
    _, before = get(f"/api/rag/collection?collection={TEST_COLLECTION}")
    count_before = before.get("points_count", -1)
    print(f"  Points before re-ingest: {count_before}")

    # Re-upload the same document
    status, body = post_multipart_file(
        "/api/documents/ingest",
        TEST_DOC,
        extra_fields={"collection": TEST_COLLECTION},
    )
    assert status == 200, f"Expected 200, got {status}\nBody: {body}"

    # Check count is unchanged
    _, after = get(f"/api/rag/collection?collection={TEST_COLLECTION}")
    count_after = after.get("points_count", -1)
    print(f"  Points after re-ingest : {count_after}")

    assert count_after == count_before, (
        f"Duplicate points created on re-ingest! Before={count_before}, After={count_after}"
    )
    print("  [PASS] Re-ingestion is idempotent — point count unchanged")


def cleanup():
    """Delete the test collection."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from rag.vector_store import get_client, collection_exists
        client = get_client()
        if collection_exists(client, TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)
            print(f"\n  [Cleanup] Deleted test collection {TEST_COLLECTION!r}")
    except Exception as exc:
        print(f"\n  [Cleanup] Warning: {exc}")


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  FastAPI RAG ENDPOINT SMOKE TESTS")
    print(SEP)
    print(f"  Base URL    : {BASE_URL}")
    print(f"  Test doc    : {TEST_DOC}")
    print(f"  Collection  : {TEST_COLLECTION}")

    try:
        test_health()
        test_existing_chat()
        test_collection_not_found()
        test_document_ingest()
        test_collection_after_ingest()
        test_rag_query_with_context()
        test_rag_query_no_context()
        test_bad_request()
        test_invalid_file_type()
        test_reingest_idempotency()
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        cleanup()
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        cleanup()
        sys.exit(1)
    finally:
        cleanup()

    print(f"\n{SEP}")
    print("  ALL API TESTS PASSED")
    print(SEP)

