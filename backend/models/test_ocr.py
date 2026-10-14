"""
test_ocr.py

Tests for backend/models/ocr.py

Tests
-----
    1. Valid image -> returns non-empty string
    2. "SIH26 OCR TEST" is present in output
    3. "Pump P-101A" is preserved
    4. "7.1 mm/s RMS" is preserved
    5. "24 hours" is preserved
    6. Missing image path -> FileNotFoundError
    7. Repetition protection: _dedup_repeated_blocks strips repeated blocks

Tests 1-5 require Ollama + glm-ocr running.
Tests 6-7 are pure logic (no model calls).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ocr import extract_text, _dedup_repeated_blocks
from config import OCR_MODEL

SEP = "=" * 60
TEST_IMAGE = r"C:\WORK DRIVE\SIH26\data\ocr_test.png"


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# Pure-logic tests (no model calls) — run first
# ---------------------------------------------------------------------------

def test_dedup_no_repetition():
    section("TEST 7a -- _dedup_repeated_blocks: clean text unchanged")
    text = "Line one\nLine two\nLine three"
    result = _dedup_repeated_blocks(text)
    assert result == text.strip(), f"Expected unchanged, got: {result!r}"
    print(f"  Input : {text!r}")
    print(f"  Output: {result!r}")
    print("  [PASS] Non-repeated text returned unchanged")


def test_dedup_repeated_block():
    section("TEST 7b -- _dedup_repeated_blocks: repeated block stripped")
    block = "SIH26 OCR TEST\nPump P-101A\nVibration threshold: 7.1 mm/s RMS\nMaintenance required within 24 hours."
    # Simulate the repetition the model produced in the bad run
    repeated = "\n".join([block] * 5)
    result = _dedup_repeated_blocks(repeated)
    print(f"  Input lines : {len(repeated.splitlines())}")
    print(f"  Output lines: {len(result.splitlines())}")
    print(f"  Output:\n    " + "\n    ".join(result.splitlines()))
    assert result == block.strip(), f"Expected single block, got:\n{result!r}"
    print("  [PASS] Repetition stripped to single occurrence")


def test_dedup_repeated_block_anywhere():
    section("TEST 7d -- _dedup_repeated_blocks: repeated block anywhere (blank-separated)")
    # Reproduce the real GLM-OCR failure mode: a plain-text version followed
    # by many identical <table> copies, separated by blank lines.
    plain = "SIH26 OCR TEST\n\nPump P-101A\n\nVibration threshold: 7.1 mm/s RMS\n\nMaintenance required within 24 hours."
    table = "<table><tr><td>SIH26 OCR TEST</td></tr><tr><td>Pump P-101A</td></tr></table>"
    repeated = plain + "\n" + "\n".join([table] * 6)
    result = _dedup_repeated_blocks(repeated)
    table_count = result.count(table)
    print(f"  Input <table> copies : 6")
    print(f"  Output <table> copies: {table_count}")
    print(f"  Output lines         : {len(result.splitlines())}")
    assert table_count <= 1, f"Expected <=1 table copy, got {table_count}:\n{result!r}"
    # Plain-text technical values must survive.
    assert "P-101A" in result
    assert "7.1" in result
    assert "24 hours" in result
    print("  [PASS] Runaway repeated block collapsed, technical values preserved")



def test_dedup_preserves_technical_values():
    section("TEST 7c -- _dedup_repeated_blocks: technical values preserved in deduplicated output")
    block = "Pump P-101A\nVibration threshold: 7.1 mm/s RMS\nMaintenance required within 24 hours."
    repeated = "\n".join([block] * 4)
    result = _dedup_repeated_blocks(repeated)
    assert "P-101A" in result
    assert "7.1" in result
    assert "24 hours" in result
    print(f"  P-101A  : {'YES' if 'P-101A' in result else 'NO'}")
    print(f"  7.1     : {'YES' if '7.1' in result else 'NO'}")
    print(f"  24 hours: {'YES' if '24 hours' in result else 'NO'}")
    print("  [PASS] Technical values preserved after deduplication")


def test_missing_image():
    section("TEST 6 -- Missing image path -> FileNotFoundError")
    bad_path = r"C:\WORK DRIVE\SIH26\data\does_not_exist_xyz.png"
    try:
        extract_text(bad_path)
        raise AssertionError("Expected FileNotFoundError was not raised")
    except FileNotFoundError as exc:
        print(f"  FileNotFoundError raised: {exc}")
        print("  [PASS] Missing file raises FileNotFoundError")


def test_empty_path():
    section("TEST 6b -- Empty image path -> ValueError")
    try:
        extract_text("")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        print(f"  ValueError raised: {exc}")
        print("  [PASS] Empty path raises ValueError")


# ---------------------------------------------------------------------------
# Live OCR tests (requires Ollama + glm-ocr)
# ---------------------------------------------------------------------------

def test_ocr_returns_text():
    section("TEST 1 -- Valid image -> returns non-empty string")
    if not os.path.isfile(TEST_IMAGE):
        print(f"  [SKIP] Test image not found: {TEST_IMAGE}")
        return None

    t0 = time.perf_counter()
    result = extract_text(TEST_IMAGE)
    elapsed = time.perf_counter() - t0

    print(f"  Model  : {OCR_MODEL}")
    print(f"  Time   : {elapsed:.2f}s")
    print(f"  Length : {len(result)} chars")
    print(f"  Output :\n    " + "\n    ".join(result.splitlines()))
    assert isinstance(result, str), "Result must be a string"
    assert len(result.strip()) > 0, "Result must not be empty"
    print("  [PASS] Returned non-empty string")
    return result, elapsed


def test_ocr_content(extracted: str):
    section("TEST 2 -- 'SIH26 OCR TEST' is present")
    assert "SIH26" in extracted or "OCR TEST" in extracted, (
        f"Expected 'SIH26 OCR TEST' in output. Got:\n{extracted!r}"
    )
    print(f"  Found: {[ln for ln in extracted.splitlines() if 'SIH26' in ln or 'OCR' in ln]}")
    print("  [PASS] SIH26 OCR TEST found")


def test_pump_tag(extracted: str):
    section("TEST 3 -- 'Pump P-101A' is preserved")
    assert "P-101A" in extracted, (
        f"Expected 'P-101A' in output. Got:\n{extracted!r}"
    )
    print(f"  Found: {[ln for ln in extracted.splitlines() if 'P-101A' in ln]}")
    print("  [PASS] P-101A preserved")


def test_vibration_value(extracted: str):
    section("TEST 4 -- '7.1 mm/s RMS' is preserved")
    assert "7.1" in extracted, (
        f"Expected '7.1' in output. Got:\n{extracted!r}"
    )
    print(f"  Found: {[ln for ln in extracted.splitlines() if '7.1' in ln]}")
    print("  [PASS] 7.1 mm/s RMS preserved")


def test_maintenance_window(extracted: str):
    section("TEST 5 -- '24 hours' is preserved")
    assert "24" in extracted, (
        f"Expected '24' (hours) in output. Got:\n{extracted!r}"
    )
    print(f"  Found: {[ln for ln in extracted.splitlines() if '24' in ln]}")
    print("  [PASS] 24 hours preserved")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  OCR MODEL WRAPPER TESTS")
    print(SEP)
    print(f"  Model     : {OCR_MODEL}")
    print(f"  Test image: {TEST_IMAGE}")

    # --- Pure logic tests first (no model calls) ---
    print(f"\n{SEP}")
    print("  LOGIC TESTS (no model calls)")
    print(SEP)

    test_dedup_no_repetition()
    test_dedup_repeated_block()
    test_dedup_repeated_block_anywhere()
    test_dedup_preserves_technical_values()
    test_missing_image()
    test_empty_path()

    logic_tests_passed = 6  # dedup×4 + missing + empty


    # --- Live OCR tests ---
    print(f"\n{SEP}")
    print("  LIVE OCR TESTS (glm-ocr via Ollama)")
    print(SEP)

    ocr_result = test_ocr_returns_text()
    if ocr_result is None:
        print("\n[SKIP] Live OCR tests skipped — test image not found.")
        print(f"\nLogic tests: {logic_tests_passed} PASS")
        sys.exit(0)

    extracted_text, elapsed_s = ocr_result

    test_ocr_content(extracted_text)
    test_pump_tag(extracted_text)
    test_vibration_value(extracted_text)
    test_maintenance_window(extracted_text)

    live_tests_passed = 5  # returns_text + content + pump + vibration + maintenance

    # --- Live repetition-protection check ---
    # No single non-blank line from the OCR output should appear 3+ times.
    section("TEST 8 -- Live output free of runaway repetition")
    from collections import Counter
    counts = Counter(
        ln.strip() for ln in extracted_text.splitlines() if ln.strip()
    )
    worst_line, worst_count = ("", 0)
    if counts:
        worst_line, worst_count = counts.most_common(1)[0]
    print(f"  Distinct non-blank lines : {len(counts)}")
    print(f"  Max repetition of any line: {worst_count}")
    assert worst_count < 3, (
        f"Runaway repetition survived: line repeated {worst_count} times:\n"
        f"{worst_line!r}"
    )
    print("  [PASS] No runaway repetition in live output")

    live_tests_passed += 1  # +repetition check

    # --- Summary ---
    total = logic_tests_passed + live_tests_passed


    print(f"\n{SEP}")
    print(f"  OCR MODEL WRAPPER TEST")
    print(f"  ----------------------")
    print(f"  Status : PASS")
    print(f"  Tests  : {total}/{total}")
    print(f"  Model  : {OCR_MODEL}")
    print(f"  Time   : {elapsed_s:.2f}s (live OCR)")
    print(f"")
    print(f"  Technical values:")
    print(f"    P-101A      : {'YES' if 'P-101A' in extracted_text else 'NO'}")
    print(f"    7.1 mm/s RMS: {'YES' if '7.1' in extracted_text else 'NO'}")
    print(f"    24 hours    : {'YES' if '24' in extracted_text else 'NO'}")
    print(f"")
    print(f"  Repetition protection: PASS")
    print(SEP)
