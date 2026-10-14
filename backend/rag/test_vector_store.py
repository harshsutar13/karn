"""
test_vector_store.py

End-to-end test for backend/rag/vector_store.py.

Pipeline exercised:
    embed_text()                    (embeddings/embedder.py)
        -> upsert_chunks()          (rag/vector_store.py)
        -> semantic_search()        (rag/vector_store.py)

Tests:
    1. Connection to Qdrant
    2. Collection creation
    3. Upsert 3 synthetic chunks
    4. Search and verify semantic relevance
    5. Repeated upsert — no duplicate points
    6. Error handling

Run with:
    $env:PYTHONUTF8="1"
    & "C:\\WORK DRIVE\\SIH26\\.venv\\Scripts\\python.exe" `
        "C:\\WORK DRIVE\\SIH26\\backend\\rag\\test_vector_store.py"
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings.embedder import embed_text
from rag.vector_store import (
    get_client,
    create_collection_if_not_exists,
    collection_exists,
    get_collection_info,
    upsert_chunks,
    semantic_search,
    ChunkRecord,
    VECTOR_SIZE,
)
from config import QDRANT_COLLECTION

SEP = "=" * 60

# Use a dedicated test collection to avoid polluting production data
TEST_COLLECTION = "sih26_test"

# Synthetic refinery chunks (text as specified in the requirements)
CHUNK_A = "The centrifugal pump is showing abnormal vibration. Maintenance inspection is required."
CHUNK_B = "Compressor discharge pressure should remain within the approved operating range."
CHUNK_C = "The employee cafeteria serves lunch at noon."

QUERY = "The refinery pump has high vibration."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# TEST 1 — Connection
# ---------------------------------------------------------------------------

def test_connection():
    print_section("TEST 1 -- Connection to Qdrant")
    t0 = time.perf_counter()
    client = get_client()
    elapsed = time.perf_counter() - t0
    print(f"  Connected to Qdrant in {elapsed:.3f}s")
    print(f"  Host : {os.environ.get('QDRANT_HOST', 'localhost')}")
    print(f"  Port : {os.environ.get('QDRANT_PORT', '6333')}")
    collections = client.get_collections().collections
    print(f"  Existing collections: {[c.name for c in collections]}")
    print("  [PASS] Connection successful")
    return client


# ---------------------------------------------------------------------------
# TEST 2 — Collection creation
# ---------------------------------------------------------------------------

def test_collection_creation(client):
    print_section("TEST 2 -- Collection creation")

    # Clean slate: delete test collection if it exists from a previous run
    if collection_exists(client, TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)
        print(f"  Deleted pre-existing collection {TEST_COLLECTION!r} for clean test")

    created = create_collection_if_not_exists(client, TEST_COLLECTION)
    assert created is True, "Expected True (newly created)"
    print(f"  Collection {TEST_COLLECTION!r} created")

    # Idempotent — calling again should NOT raise and should return False
    created_again = create_collection_if_not_exists(client, TEST_COLLECTION)
    assert created_again is False, "Expected False (already existed)"
    print(f"  Idempotent call returned False correctly")

    info = get_collection_info(client, TEST_COLLECTION)
    print(f"  Collection info: {info}")
    assert info["vector_size"] == VECTOR_SIZE, (
        f"Wrong vector size: expected {VECTOR_SIZE}, got {info['vector_size']}"
    )
    print("  [PASS] Collection created, idempotency verified, vector_size=768 confirmed")
    return info


# ---------------------------------------------------------------------------
# TEST 3 — Upsert
# ---------------------------------------------------------------------------

def test_upsert(client) -> list[float]:
    print_section("TEST 3 -- Embed and upsert 3 chunks")

    texts = [CHUNK_A, CHUNK_B, CHUNK_C]
    doc_id = "test_doc_synthetic_001"

    print(f"  Embedding {len(texts)} chunks via nomic-embed-text...")
    t0 = time.perf_counter()
    embeddings = [embed_text(t) for t in texts]
    elapsed = time.perf_counter() - t0
    print(f"  Embedding time: {elapsed:.2f}s total ({elapsed/len(texts):.2f}s each)")

    records = [
        ChunkRecord(
            document_id=doc_id,
            filename="synthetic_test.txt",
            doc_type="txt",
            chunk_index=i,
            text=texts[i],
            char_start=i * 100,
            char_end=i * 100 + len(texts[i]),
            embedding=embeddings[i],
            extra_metadata={"department": "Testing", "test_run": True},
        )
        for i in range(len(texts))
    ]

    t0 = time.perf_counter()
    count = upsert_chunks(client, records, collection_name=TEST_COLLECTION)
    elapsed = time.perf_counter() - t0
    print(f"  Upserted {count} points in {elapsed:.3f}s")

    assert count == 3, f"Expected 3 upserted points, got {count}"

    info = get_collection_info(client, TEST_COLLECTION)
    print(f"  Collection now has {info['points_count']} points")
    assert info["points_count"] == 3, (
        f"Expected 3 points in collection, got {info['points_count']}"
    )
    print("  [PASS] 3 points upserted and confirmed in collection")

    # Return query embedding for next test
    print(f"\n  Embedding query: {QUERY!r}")
    query_vec = embed_text(QUERY)
    print(f"  Query vector dim: {len(query_vec)}")
    return query_vec


# ---------------------------------------------------------------------------
# TEST 4 — Semantic search
# ---------------------------------------------------------------------------

def test_semantic_search(client, query_vec: list[float]):
    print_section("TEST 4 -- Semantic search")
    print(f"  Query : {QUERY!r}")
    print(f"  top_k : 3")

    t0 = time.perf_counter()
    results = semantic_search(
        client,
        query_vector=query_vec,
        top_k=3,
        collection_name=TEST_COLLECTION,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Search time: {elapsed:.3f}s")
    print(f"  Results ({len(results)}):")

    for i, r in enumerate(results):
        print(f"\n  [{i}] score={r.score:.6f}")
        print(f"       text={r.text!r}")
        print(f"       doc_id={r.document_id}")
        print(f"       chunk_index={r.chunk_index}")
        print(f"       filename={r.filename}")
        print(f"       chars={r.char_start}-{r.char_end}")
        print(f"       extra={r.extra}")
        print(f"       point_id={r.point_id}")

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # Verify top result is the pump/vibration chunk (Chunk A)
    top_result = results[0]
    assert CHUNK_A in top_result.text, (
        f"Expected pump/vibration chunk as top result.\n"
        f"Got: {top_result.text!r}"
    )
    assert top_result.score > results[1].score, (
        "Results not sorted by score descending"
    )

    # Verify metadata is preserved
    assert top_result.filename == "synthetic_test.txt"
    assert top_result.doc_type == "txt"
    assert top_result.extra.get("department") == "Testing"

    print(f"\n  Top result is pump/vibration chunk: score={top_result.score:.6f} [CORRECT]")
    print("  [PASS] Semantic search returned correct top result with metadata")


# ---------------------------------------------------------------------------
# TEST 5 — Repeated upsert (no duplicates)
# ---------------------------------------------------------------------------

def test_no_duplicates(client):
    print_section("TEST 5 -- Repeated upsert does not create duplicates")

    doc_id = "test_doc_synthetic_001"
    texts = [CHUNK_A, CHUNK_B, CHUNK_C]

    print("  Re-upserting the same 3 chunks with identical document_id + chunk_index...")
    embeddings = [embed_text(t) for t in texts]
    records = [
        ChunkRecord(
            document_id=doc_id,
            filename="synthetic_test.txt",
            doc_type="txt",
            chunk_index=i,
            text=texts[i],
            char_start=i * 100,
            char_end=i * 100 + len(texts[i]),
            embedding=embeddings[i],
        )
        for i in range(len(texts))
    ]

    count = upsert_chunks(client, records, collection_name=TEST_COLLECTION)
    print(f"  Second upsert returned: {count} points processed")

    info = get_collection_info(client, TEST_COLLECTION)
    print(f"  Collection still has: {info['points_count']} points")

    assert info["points_count"] == 3, (
        f"Duplicate points created! Expected 3, got {info['points_count']}. "
        "Point ID deduplication is not working."
    )
    print("  [PASS] No duplicates created — deterministic UUIDs working correctly")


# ---------------------------------------------------------------------------
# TEST 6 — Error handling
# ---------------------------------------------------------------------------

def test_error_handling(client):
    print_section("TEST 6 -- Error handling")

    # Empty records list
    try:
        upsert_chunks(client, [], collection_name=TEST_COLLECTION)
        assert False, "Expected ValueError"
    except ValueError as exc:
        print(f"  [PASS] upsert_chunks([]) raised ValueError: {exc}")

    # Wrong embedding dimension
    bad_embedding = [0.0] * 64  # Wrong: should be 768
    bad_record = ChunkRecord(
        document_id="bad_doc",
        filename="bad.txt",
        doc_type="txt",
        chunk_index=0,
        text="Some text",
        char_start=0,
        char_end=9,
        embedding=bad_embedding,
    )
    try:
        upsert_chunks(client, [bad_record], collection_name=TEST_COLLECTION)
        assert False, "Expected ValueError for wrong dimension"
    except ValueError as exc:
        print(f"  [PASS] Wrong embedding dim raised ValueError: {exc}")

    # Empty query vector
    try:
        semantic_search(client, query_vector=[], collection_name=TEST_COLLECTION)
        assert False, "Expected ValueError"
    except ValueError as exc:
        print(f"  [PASS] Empty query_vector raised ValueError: {exc}")

    # Invalid top_k
    try:
        semantic_search(client, query_vector=[0.0] * 768, top_k=0, collection_name=TEST_COLLECTION)
        assert False, "Expected ValueError"
    except ValueError as exc:
        print(f"  [PASS] top_k=0 raised ValueError: {exc}")

    # Non-existent collection
    try:
        semantic_search(client, query_vector=[0.0] * 768, collection_name="does_not_exist_xyz")
        assert False, "Expected ValueError"
    except ValueError as exc:
        print(f"  [PASS] Non-existent collection raised ValueError: {exc}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    print(f"\n{SEP}")
    print("  VECTOR STORE TEST SUITE")
    print(SEP)
    print(f"  Qdrant host      : {os.environ.get('QDRANT_HOST', 'localhost')}")
    print(f"  Qdrant port      : {os.environ.get('QDRANT_PORT', '6333')}")
    print(f"  Test collection  : {TEST_COLLECTION}")
    print(f"  Vector size      : {VECTOR_SIZE}")
    print(f"  Distance metric  : Cosine")
    print(f"  Test chunks      : 3")
    print(f"  Query            : {QUERY!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_summary()

    try:
        client = test_connection()
        test_collection_creation(client)
        query_vec = test_upsert(client)
        test_semantic_search(client, query_vec)
        test_no_duplicates(client)
        test_error_handling(client)
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
    except ConnectionError as exc:
        print(f"\n[ERROR] Qdrant not reachable: {exc}")
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n[ERROR] Unexpected: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up: delete the test collection so each run is fresh
        try:
            from qdrant_client import QdrantClient as _QC
            _c = _QC(host="localhost", port=6333, timeout=5)
            if collection_exists(_c, TEST_COLLECTION):
                _c.delete_collection(TEST_COLLECTION)
                print(f"\n  [Cleanup] Deleted test collection {TEST_COLLECTION!r}")
        except Exception:
            pass  # Best-effort cleanup

    print(f"\n{SEP}")
    print("  ALL TESTS PASSED")
    print(SEP)
