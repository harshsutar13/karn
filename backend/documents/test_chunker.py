"""
test_chunker.py

Verifies chunker.py against:
  1. The existing small test.txt
  2. A larger synthetic refinery document (generated inline)

Run from:  C:\\WORK DRIVE\\SIH26\\backend\\documents\\
Command :  python test_chunker.py
"""

import sys
import os

# Allow running directly from the documents/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from documents.loader import load_document
from documents.chunker import chunk_text, chunk_text_with_metadata


SEP = "=" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_chunks(chunks: list[str], label: str = "") -> None:
    if label:
        print(f"\n--- {label} ---")
    for i, chunk in enumerate(chunks):
        preview = chunk[:120].replace("\n", " ")
        print(f"  [{i}] len={len(chunk):>5}  {preview!r}")


# ---------------------------------------------------------------------------
# Test 1: small test.txt
# ---------------------------------------------------------------------------

def test_small_document() -> None:
    print_section("TEST 1 — Small document (test.txt)")

    file_path = r"C:\WORK DRIVE\SIH26\data\documents\test.txt"
    text = load_document(file_path)
    print(f"  Document length: {len(text)} chars")

    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    print(f"  Chunks produced : {len(chunks)}")
    print_chunks(chunks, "chunk_text output")

    # With a 423-char document and chunk_size=1000 we expect exactly 1 chunk
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    print("\n  [PASS] chunk count correct for small document")


# ---------------------------------------------------------------------------
# Test 2: larger synthetic refinery document
# ---------------------------------------------------------------------------

SYNTHETIC_REFINERY_DOC = """\
Crude Oil Distillation Unit — Operations Manual (Excerpt)

1. Introduction

The Crude Oil Distillation Unit (CDU) is the first and most critical processing unit
in a petroleum refinery. Raw crude oil, received via pipeline or tanker, is heated and
separated into fractions based on boiling point. These fractions feed downstream units
including the Fluidised Catalytic Cracking (FCC) unit, the Hydrocracker, the Reformer,
and the Visbreaker.

2. Feed Pre-treatment

Before entering the Atmospheric Distillation Tower (ADT), crude oil passes through the
Desalter. The Desalter removes salts, sediment, and water by mixing the crude with
wash water and then using an electrostatic field to coalesce the water droplets. Target
salt content at desalter outlet is below 1 PTB (pounds per thousand barrels).

Key instrumentation:
- Flow transmitters on crude feed line (FT-101, FT-102)
- Temperature elements on desalter inlet/outlet (TE-201, TE-202)
- Pressure transmitters on electrostatic grid (PT-301)

3. Preheating Train

Crude oil passes through a series of heat exchangers known collectively as the preheat
train. Heat is recovered from hot product streams and pump-around circuits. Typical
preheat train outlet temperature is 260–290 °C before the crude furnace.

Critical maintenance points:
- Shell-and-tube exchangers should be inspected for fouling every 18 months.
- Vibration analysis should be performed on all pump motors quarterly.
- Thermographic surveys of furnace tubes should be performed annually.

4. Atmospheric Distillation Tower (ADT)

The ADT operates at near-atmospheric pressure (slightly above 1 atm to prevent air
ingress). Vapourised fractions rise through the tower and are drawn off at sidecuts:
- Overhead: LPG and naphtha (boiling range: up to ~175 °C)
- Kerosene sidecut: 150–260 °C
- Gas oil sidecut: 230–360 °C
- Atmospheric residue (long residue): >360 °C, drawn from the bottom

Tower internals include valve trays and structured packing sections. Tray efficiency
must be monitored; a delta-pressure increase across tray sections can indicate flooding
or fouling.

5. Vacuum Distillation Unit (VDU)

Atmospheric residue from the ADT bottom is further processed in the VDU. Operating at
20–100 mmHg absolute pressure, the VDU recovers additional distillates:
- Vacuum gas oil (VGO): feed to the FCC or Hydrocracker
- Vacuum residue: feed to the Visbreaker or Bitumen unit

Steam ejectors maintain the vacuum. Any loss of vacuum can result in thermal cracking
inside the column and severe fouling.

6. Predictive Maintenance Programme

The refinery has adopted an integrated predictive maintenance (PdM) programme covering
the CDU and VDU assets. Key techniques employed:

6.1 Vibration Analysis
All rotating equipment — pumps, compressors, fans — is fitted with online vibration
sensors. Data is streamed to the Plant Asset Management (PAM) system at 1-second
intervals. Alarm thresholds are set per ISO 10816-3.

6.2 Oil Analysis
Lube oil samples from gearboxes and bearing housings are drawn monthly and analysed
for:
- Viscosity
- Total Acid Number (TAN)
- Wear metal content (Fe, Cu, Cr, Al, Pb)
- Water content (Karl Fischer)

6.3 Infrared Thermography
Infrared cameras are used quarterly to inspect:
- Furnace tubes for hotspots
- Refractory lining degradation
- Electrical switchgear

6.4 Ultrasonic Testing
Piping and vessel wall thickness measurements are taken annually using UT probes to
monitor corrosion rates, especially in high-velocity areas and at injection points.

7. Safety and Permit-to-Work

No maintenance activity may begin without a valid Permit-to-Work (PTW) issued by the
Area Authority. Work on live hydrocarbon systems additionally requires:
- Hot Work Permit (for cutting, welding)
- Confined Space Entry Permit
- Isolation Certificate signed by the Operations Shift Supervisor

8. Emergency Shutdown (ESD) System

The CDU and VDU are protected by a Safety Instrumented System (SIS) conforming to
IEC 61511. Safety Instrumented Functions (SIFs) include:
- SIF-CDU-01: High furnace tube skin temperature -> fuel gas trip
- SIF-CDU-02: High ADT overhead pressure -> pressure relief + shutdown
- SIF-VDU-01: Loss of vacuum -> steam ejector alarm -> VDU feed isolation

Functional Safety assessments (SIL determination) are reviewed every 5 years or
following any Safety Integrity Level (SIL) deviation event.

9. Conclusion

Reliable operation of the CDU/VDU complex is fundamental to overall refinery
throughput and profitability. Proactive maintenance, rigorous inspection, and
adherence to PTW and ESD procedures are the primary defences against unplanned
shutdowns, product quality excursions, and safety incidents.
"""


def test_large_document() -> None:
    print_section("TEST 2 — Large synthetic refinery document")

    text = SYNTHETIC_REFINERY_DOC
    print(f"  Document length: {len(text)} chars")

    # --- chunk_text ---
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    print(f"  chunk_text (size=500, overlap=100) -> {len(chunks)} chunks")
    print_chunks(chunks, "chunk_text output")

    assert len(chunks) > 1, "Expected multiple chunks for a large document"

    # --- chunk_text_with_metadata ---
    rich_chunks = chunk_text_with_metadata(
        text,
        chunk_size=500,
        chunk_overlap=100,
        base_metadata={
            "filename": "synthetic_refinery.txt",
            "doc_type": "manual",
            "department": "Operations",
        },
    )

    print(f"\n  chunk_text_with_metadata -> {len(rich_chunks)} chunks")
    for c in rich_chunks:
        print(
            f"  [{c.chunk_index}] chars={c.char_start}-{c.char_end}"
            f"  meta={c.metadata}  preview={c.text[:60]!r}"
        )

    # Verify chunk count is consistent
    assert len(rich_chunks) == len(chunks), (
        f"chunk_text and chunk_text_with_metadata returned different counts: "
        f"{len(chunks)} vs {len(rich_chunks)}"
    )

    # Verify overlap: adjacent chunks should share some text
    for i in range(len(rich_chunks) - 1):
        a = rich_chunks[i].text
        b = rich_chunks[i + 1].text
        # At chunk_overlap=100, the last 100 chars of chunk i should appear
        # near the start of chunk i+1 (strip() can shift by a few chars)
        overlap_text = a[-90:]
        assert overlap_text in b, (
            f"Overlap not found between chunk {i} and chunk {i + 1}.\n"
            f"  End of [{i}]: {a[-100:]!r}\n"
            f"  Start of [{i+1}]: {b[:200]!r}"
        )

    print("\n  [PASS] overlap verified between consecutive chunks")


# ---------------------------------------------------------------------------
# Test 3: Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases() -> None:
    print_section("TEST 3 — Edge cases")

    # Empty string
    result = chunk_text("", chunk_size=500, chunk_overlap=100)
    assert result == [], f"Expected [] for empty input, got {result}"
    print("  [PASS] empty string -> []")

    # Whitespace only
    result = chunk_text("   \n\n  ", chunk_size=500, chunk_overlap=100)
    assert result == [], f"Expected [] for whitespace-only input, got {result}"
    print("  [PASS] whitespace-only string -> []")

    # Text shorter than chunk_size
    short = "Short text."
    result = chunk_text(short, chunk_size=500, chunk_overlap=100)
    assert len(result) == 1 and result[0] == short, (
        f"Expected ['{short}'], got {result}"
    )
    print("  [PASS] text shorter than chunk_size -> 1 chunk")

    # Invalid overlap
    try:
        chunk_text("hello", chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        print("  [PASS] chunk_overlap >= chunk_size raises ValueError")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(SEP)
    print("  CHUNKER TEST SUITE")
    print(SEP)

    try:
        test_small_document()
        test_large_document()
        test_edge_cases()
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)

    print(f"\n{SEP}")
    print("  ALL TESTS PASSED")
    print(SEP)
