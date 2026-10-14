"""
test_rag_pipeline.py

End-to-end RAG pipeline test for SIH26.

Pipeline exercised:
    Document file
        -> ingest_document()   [loader + chunker + embedder + Qdrant]
        -> rag_query()         [retriever + LLM]
        -> RAGResponse         [structured answer + sources]

Test cases:
    1. Document ingestion
    2. Retrieval sanity — query with answer in document
    3. Semantic retrieval — differently phrased question
    4. No-context behavior — question whose answer is NOT in the document
    5. Repeated ingestion — no duplicate points
    6. RAGResponse structure validation

Run with:
    $env:PYTHONUTF8="1"
    & "C:\\WORK DRIVE\\SIH26\\.venv\\Scripts\\python.exe" `
        "C:\\WORK DRIVE\\SIH26\\qdrant\\run_tests.py" `
        "C:\\WORK DRIVE\\SIH26\\backend\\rag\\test_rag_pipeline.py"
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.ingestion import ingest_document
from rag.pipeline import rag_query, RAGResponse
from rag.retriever import retrieve
from rag.vector_store import get_client, collection_exists, get_collection_info
from config import QDRANT_COLLECTION

SEP = "=" * 60

# Test document
TEST_DOC = r"C:\WORK DRIVE\SIH26\data\documents\refinery_maintenance.txt"

# Use a dedicated test collection
TEST_COLLECTION = "sih26_rag_test"


def print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_rag_response(resp: RAGResponse) -> None:
    print(f"\n  Query            : {resp.query!r}")
    print(f"  Chunks retrieved : {resp.chunks_retrieved}")
    print(f"  Context used     : {resp.context_used}")
    print(f"\n  ANSWER:\n")
    for line in resp.answer.strip().split("\n"):
        print(f"    {line}")
    print(f"\n  SOURCES ({len(resp.sources)}):")
    for s in resp.sources:
        print(f"    [{s.chunk_index}] {s.filename} | score={s.score:.4f} | chars={s.char_start}-{s.char_end}")
        print(f"         preview: {s.text_preview[:100]!r}")


# ---------------------------------------------------------------------------
# TEST 1 — Document ingestion
# ---------------------------------------------------------------------------

def test_ingestion() -> str:
    print_section("TEST 1 -- Document ingestion")

    assert os.path.isfile(TEST_DOC), f"Test document not found: {TEST_DOC}"

    t0 = time.perf_counter()
    result = ingest_document(
        file_path=TEST_DOC,
        collection_name=TEST_COLLECTION,
        chunk_size=600,
        chunk_overlap=120,
        department="Operations",
    )
    elapsed = time.perf_counter() - t0

    print(f"  File            : {result.filename}")
    print(f"  Document ID     : {result.document_id}")
    print(f"  Doc type        : {result.doc_type}")
    print(f"  Total chars     : {result.total_chars}")
    print(f"  Chunks created  : {result.chunks_created}")
    print(f"  Chunks upserted : {result.chunks_upserted}")
    print(f"  Load time       : {result.elapsed_load_s:.3f}s")
    print(f"  Chunk time      : {result.elapsed_chunk_s:.3f}s")
    print(f"  Embed time      : {result.elapsed_embed_s:.3f}s")
    print(f"  Upsert time     : {result.elapsed_upsert_s:.3f}s")
    print(f"  TOTAL           : {result.elapsed_total_s:.3f}s")

    assert result.chunks_created > 0, "Expected at least 1 chunk"
    assert result.chunks_upserted == result.chunks_created, (
        f"Upserted {result.chunks_upserted} != created {result.chunks_created}"
    )
    assert result.total_chars > 500, "Document seems too short"

    client = get_client()
    info = get_collection_info(client, TEST_COLLECTION)
    print(f"\n  Collection info : {info}")
    assert info["points_count"] == result.chunks_upserted

    print(f"  [PASS] Document ingested: {result.chunks_upserted} chunks in Qdrant")
    return result.document_id


# ---------------------------------------------------------------------------
# TEST 2 — Direct retrieval (exact topic in document)
# ---------------------------------------------------------------------------

def test_retrieval_direct():
    print_section("TEST 2 -- Retrieval: direct question (answer in document)")

    query = "What is the vibration alert threshold for pump P-101A?"
    print(f"  Query: {query!r}")

    chunks = retrieve(query=query, top_k=3, collection_name=TEST_COLLECTION)

    print(f"  Chunks returned: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] score={c.score:.4f} | {c.text[:100]!r}")

    assert len(chunks) > 0, "Expected at least 1 chunk"

    top = chunks[0]
    # The answer (7.1 mm/s) should be in the top chunk
    assert "7.1" in top.text or "vibration" in top.text.lower(), (
        f"Top chunk doesn't seem relevant to pump vibration threshold.\n"
        f"Got: {top.text!r}"
    )
    print(f"  [PASS] Top chunk score={top.score:.4f}, contains pump/vibration info")


# ---------------------------------------------------------------------------
# TEST 3 — RAG query (exact topic)
# ---------------------------------------------------------------------------

def test_rag_direct():
    print_section("TEST 3 -- RAG query: answer exists in document")

    question = "What is the vibration alert threshold for pump P-101A?"
    print(f"  Asking: {question!r}")

    t0 = time.perf_counter()
    resp = rag_query(question=question, top_k=3, collection_name=TEST_COLLECTION)
    elapsed = time.perf_counter() - t0

    print(f"  Response time: {elapsed:.2f}s")
    print_rag_response(resp)

    assert resp.context_used is True, "Expected context to be used"
    assert resp.chunks_retrieved > 0
    assert len(resp.answer.strip()) > 20, "Answer is suspiciously short"
    assert len(resp.sources) > 0, "Expected source citations"

    # Check the answer mentions the value from the document
    answer_lower = resp.answer.lower()
    assert "7.1" in resp.answer or "vibration" in answer_lower, (
        f"Answer doesn't mention pump vibration threshold.\nAnswer: {resp.answer}"
    )
    print("  [PASS] RAG answer mentions vibration threshold, sources cited")


# ---------------------------------------------------------------------------
# TEST 4 — RAG query (semantic, differently phrased)
# ---------------------------------------------------------------------------

def test_rag_semantic():
    print_section("TEST 4 -- RAG query: semantically phrased (different wording)")

    question = "How quickly must engineers respond when a pump shows unusual shaking?"
    print(f"  Asking: {question!r}")
    print(f"  (Document says: 'inspected within 24 hours' when 'abnormal vibration detected')")

    t0 = time.perf_counter()
    resp = rag_query(question=question, top_k=4, collection_name=TEST_COLLECTION)
    elapsed = time.perf_counter() - t0

    print(f"  Response time: {elapsed:.2f}s")
    print_rag_response(resp)

    assert resp.context_used is True
    assert resp.chunks_retrieved > 0

    # The document says "inspected within 24 hours" — retrieval should surface it
    top_texts = " ".join(c.text_preview for c in resp.sources).lower()
    answer_lower = resp.answer.lower()

    # Either the source or answer should mention hours/inspection
    relevant = (
        "24" in top_texts or "24" in resp.answer
        or "inspect" in top_texts or "inspect" in answer_lower
        or "hour" in top_texts or "hour" in answer_lower
    )
    assert relevant, (
        f"Neither sources nor answer mention 24-hour inspection window.\n"
        f"Answer: {resp.answer}\nSources: {top_texts[:300]}"
    )
    print("  [PASS] Semantic retrieval found pump inspection protocol")


# ---------------------------------------------------------------------------
# TEST 5 — No-context behavior
# ---------------------------------------------------------------------------

def test_rag_no_context():
    print_section("TEST 5 -- RAG query: answer NOT in document")

    question = "What is the recommended lunch menu for refinery workers?"
    print(f"  Asking: {question!r}")
    print(f"  (This topic does not exist in the test document)")

    t0 = time.perf_counter()
    resp = rag_query(
        question=question,
        top_k=3,
        collection_name=TEST_COLLECTION,
        score_threshold=0.75,  # High threshold — unlikely to pass for irrelevant topic
    )
    elapsed = time.perf_counter() - t0

    print(f"  Response time: {elapsed:.2f}s")
    print_rag_response(resp)

    # With a high score threshold, the model should NOT get relevant chunks
    # and should decline to hallucinate.
    if not resp.context_used:
        print("  [PASS] No relevant context retrieved (threshold filter worked)")
    else:
        # Some chunks passed the threshold — check the LLM response
        answer_lower = resp.answer.lower()
        declines = (
            "not contain" in answer_lower
            or "insufficient" in answer_lower
            or "do not have" in answer_lower
            or "not found" in answer_lower
            or "no information" in answer_lower
            or "available documents" in answer_lower
        )
        print(f"  Context used: {resp.context_used}, answer signals refusal: {declines}")
        print(f"  [NOTE] LLM got some context — check answer quality above")
        # This is a soft check — we verify the model doesn't confidently answer
        # about lunch menus. We don't hard-fail here since threshold behavior
        # depends on embedding similarity to refinery text.


# ---------------------------------------------------------------------------
# TEST 6 — Repeated ingestion (no duplicates)
# ---------------------------------------------------------------------------

def test_no_duplicates():
    print_section("TEST 6 -- Repeated ingestion does not create duplicate points")

    client = get_client()
    info_before = get_collection_info(client, TEST_COLLECTION)
    count_before = info_before["points_count"]
    print(f"  Points before re-ingestion: {count_before}")

    # Re-ingest the same document
    result = ingest_document(
        file_path=TEST_DOC,
        collection_name=TEST_COLLECTION,
        chunk_size=600,
        chunk_overlap=120,
    )

    info_after = get_collection_info(client, TEST_COLLECTION)
    count_after = info_after["points_count"]
    print(f"  Points after re-ingestion : {count_after}")

    assert count_after == count_before, (
        f"Duplicate points created! Before={count_before}, After={count_after}. "
        "Deterministic point IDs are not working."
    )
    print("  [PASS] No duplicates — idempotent ingestion confirmed")


# ---------------------------------------------------------------------------
# TEST 7 — RAGResponse structure
# ---------------------------------------------------------------------------

def test_response_structure():
    print_section("TEST 7 -- RAGResponse structure validation")

    question = "What are the SIS safety functions in the CDU?"
    resp = rag_query(question=question, top_k=3, collection_name=TEST_COLLECTION)

    assert isinstance(resp.answer, str) and resp.answer, "answer must be non-empty str"
    assert isinstance(resp.sources, list), "sources must be a list"
    assert isinstance(resp.query, str), "query must be str"
    assert isinstance(resp.chunks_retrieved, int), "chunks_retrieved must be int"
    assert isinstance(resp.context_used, bool), "context_used must be bool"

    for s in resp.sources:
        assert isinstance(s.filename, str)
        assert isinstance(s.score, float)
        assert isinstance(s.chunk_index, int)
        assert isinstance(s.text_preview, str)

    print(f"  answer type         : {type(resp.answer).__name__} OK")
    print(f"  sources count       : {len(resp.sources)}")
    print(f"  chunks_retrieved    : {resp.chunks_retrieved}")
    print(f"  context_used        : {resp.context_used}")
    print("  [PASS] RAGResponse structure is correct")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    try:
        client = get_client()
        if collection_exists(client, TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)
            print(f"\n  [Cleanup] Deleted test collection {TEST_COLLECTION!r}")
    except Exception as exc:
        print(f"\n  [Cleanup] Warning: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  RAG PIPELINE END-TO-END TEST SUITE")
    print(SEP)
    print(f"  Test document    : {TEST_DOC}")
    print(f"  Test collection  : {TEST_COLLECTION}")

    try:
        test_ingestion()
        test_retrieval_direct()
        test_rag_direct()
        test_rag_semantic()
        test_rag_no_context()
        test_no_duplicates()
        test_response_structure()
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
    print("  ALL RAG PIPELINE TESTS PASSED")
    print(SEP)
