"""
test_embedder.py

Verification suite for backend/embeddings/embedder.py.

Tests:
    1. Single embedding — structure and dimension
    2. Batch embeddings — count and dimension consistency
    3. Semantic sanity check — cosine similarity ordering

Run with:
    $env:PYTHONUTF8="1"
    & "C:\\WORK DRIVE\\SIH26\\.venv\\Scripts\\python.exe" `
        "C:\\WORK DRIVE\\SIH26\\backend\\embeddings\\test_embedder.py"
"""

import math
import sys
import os
import time

# Allow import from backend/ regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings.embedder import embed_text, embed_texts, get_embedding_dimension
from config import EMBEDDING_MODEL

SEP = "=" * 60


# ---------------------------------------------------------------------------
# Pure-stdlib cosine similarity (no numpy)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two equal-length float vectors
    using only the Python standard library.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def fmt_vec(vec: list[float], n: int = 6) -> str:
    """Show first n values of a vector for readability."""
    sample = ", ".join(f"{v:.6f}" for v in vec[:n])
    return f"[{sample}, ... ] (dim={len(vec)})"


# ---------------------------------------------------------------------------
# TEST 1 — Single embedding
# ---------------------------------------------------------------------------

def test_single_embedding() -> int:
    """
    Returns
    -------
    int
        The embedding dimension (needed by subsequent tests).
    """
    print_section("TEST 1 -- Single embedding")

    sentence = "The centrifugal pump is showing abnormal vibration."
    print(f"  Input : {sentence!r}")

    t0 = time.perf_counter()
    vec = embed_text(sentence)
    elapsed = time.perf_counter() - t0

    print(f"  Elapsed         : {elapsed:.3f}s")
    print(f"  Return type     : {type(vec).__name__}")
    print(f"  Vector dimension: {len(vec)}")
    print(f"  First 6 values  : {fmt_vec(vec)}")

    # Assertions
    assert isinstance(vec, list), (
        f"Expected list, got {type(vec).__name__}"
    )
    assert len(vec) > 0, "Embedding vector is empty"
    assert all(isinstance(v, float) for v in vec), (
        "Not all values are floats"
    )

    print("  [PASS] Single embedding: type, length, and numeric content verified")
    return len(vec)


# ---------------------------------------------------------------------------
# TEST 2 — Batch embeddings
# ---------------------------------------------------------------------------

def test_batch_embeddings(expected_dim: int) -> None:
    print_section("TEST 2 -- Batch embeddings")

    texts = [
        "The centrifugal pump is showing abnormal vibration.",
        "High vibration has been detected in the refinery pump.",
        "The employee cafeteria serves lunch at noon.",
        "Predictive maintenance reduces unplanned shutdowns in oil refineries.",
        "Crude oil distillation separates fractions by boiling point.",
    ]

    print(f"  Input texts     : {len(texts)}")

    t0 = time.perf_counter()
    embeddings = embed_texts(texts)
    elapsed = time.perf_counter() - t0

    print(f"  Elapsed (total) : {elapsed:.3f}s")
    print(f"  Elapsed per text: {elapsed / len(texts):.3f}s")
    print(f"  Embeddings count: {len(embeddings)}")

    for i, (txt, emb) in enumerate(zip(texts, embeddings)):
        print(f"  [{i}] dim={len(emb)}  preview={txt[:55]!r}")

    # Assertions
    assert len(embeddings) == len(texts), (
        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
    )

    dims = [len(e) for e in embeddings]
    assert len(set(dims)) == 1, (
        f"Inconsistent dimensions across embeddings: {dims}"
    )
    assert dims[0] == expected_dim, (
        f"Batch dimension {dims[0]} differs from single-embed dimension {expected_dim}"
    )

    print(f"  [PASS] Batch: count={len(embeddings)}, all dims={dims[0]}")


# ---------------------------------------------------------------------------
# TEST 3 — Semantic sanity check
# ---------------------------------------------------------------------------

def test_semantic_sanity() -> None:
    print_section("TEST 3 -- Semantic sanity check")

    text_a = "The centrifugal pump is showing abnormal vibration."
    text_b = "High vibration has been detected in the refinery pump."
    text_c = "The employee cafeteria serves lunch at noon."

    print(f"  A: {text_a!r}")
    print(f"  B: {text_b!r}")
    print(f"  C: {text_c!r}")
    print()

    t0 = time.perf_counter()
    vec_a = embed_text(text_a)
    vec_b = embed_text(text_b)
    vec_c = embed_text(text_c)
    elapsed = time.perf_counter() - t0

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    print(f"  sim(A, B) = {sim_ab:.6f}  [pump vibration vs pump vibration]")
    print(f"  sim(A, C) = {sim_ac:.6f}  [pump vibration vs cafeteria lunch]")
    print(f"  Embed time: {elapsed:.3f}s for 3 texts")
    print()

    # Semantic correctness: A and B should be closer than A and C
    passed = sim_ab > sim_ac
    delta = sim_ab - sim_ac

    print(f"  sim(A,B) > sim(A,C) ? {'YES' if passed else 'NO'} (delta={delta:+.6f})")

    assert passed, (
        f"Semantic sanity FAILED: sim(A,B)={sim_ab:.6f} is NOT > sim(A,C)={sim_ac:.6f}. "
        "The embedding model may not be working correctly."
    )

    print("  [PASS] Semantic sanity: related texts are closer than unrelated ones")


# ---------------------------------------------------------------------------
# TEST 4 — Error handling
# ---------------------------------------------------------------------------

def test_error_handling() -> None:
    print_section("TEST 4 -- Error handling (empty input)")

    # embed_text with empty string
    try:
        embed_text("")
        assert False, "Expected ValueError for empty string"
    except ValueError as exc:
        print(f"  [PASS] embed_text('') raised ValueError: {exc}")

    # embed_text with whitespace only
    try:
        embed_text("   \n\t  ")
        assert False, "Expected ValueError for whitespace-only"
    except ValueError as exc:
        print(f"  [PASS] embed_text(whitespace) raised ValueError: {exc}")

    # embed_texts with empty list
    try:
        embed_texts([])
        assert False, "Expected ValueError for empty list"
    except ValueError as exc:
        print(f"  [PASS] embed_texts([]) raised ValueError: {exc}")

    # embed_texts with a blank item inside
    try:
        embed_texts(["valid text", ""])
        assert False, "Expected ValueError for blank item in list"
    except ValueError as exc:
        print(f"  [PASS] embed_texts(['valid', '']) raised ValueError: {exc}")


# ---------------------------------------------------------------------------
# Summary header
# ---------------------------------------------------------------------------

def print_summary_header() -> None:
    print(SEP)
    print("  EMBEDDER TEST SUITE")
    print(SEP)
    print(f"  Embedding model : {EMBEDDING_MODEL}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_summary_header()

    try:
        dim = test_single_embedding()
        test_batch_embeddings(expected_dim=dim)
        test_semantic_sanity()
        test_error_handling()
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected exception: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print(f"\n{SEP}")
    print("  ALL TESTS PASSED")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(SEP)
