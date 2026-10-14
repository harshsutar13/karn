"""
Standalone Qwen3-VL:4b vision test.

Three tests:
    A — Image description: can the model describe what it sees?
    B — Visual Q&A: can it answer a specific question about the image?
    C — Structured extraction: can it extract key-value pairs accurately?

No project imports. Pure Ollama API calls.
Does NOT modify any SIH26 project files.
"""
import base64
import json
import time
import sys
import os
import urllib.request

MODEL      = "qwen3-vl:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"
IMAGE_PATH = r"C:\WORK DRIVE\SIH26\data\vision_test.png"

SEP = "=" * 60


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# Load + encode image once
with open(IMAGE_PATH, "rb") as f:
    IMG_B64 = base64.b64encode(f.read()).decode()

print(f"\n{SEP}")
print(f"  QWEN3-VL 4B STANDALONE VISION TEST")
print(SEP)
print(f"  Model : {MODEL}")
print(f"  Image : {IMAGE_PATH} ({os.path.getsize(IMAGE_PATH)} bytes)")


def ask(question: str) -> tuple[str, float]:
    """Send one vision question, return (response_text, elapsed_seconds)."""
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": question,
            "images": [IMG_B64],
        }],
        "stream": False,
        "options": {"num_predict": 300},
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    text = resp.get("message", {}).get("content", "").strip()
    return text, elapsed


# ---------------------------------------------------------------------------
# Test A: Image description
# ---------------------------------------------------------------------------
section("TEST A -- Image description")
print("  Prompt: 'Describe what you see in this image in 2-4 sentences.'")
try:
    text_a, t_a = ask("Describe what you see in this image in 2-4 sentences.")
    print(f"  Time  : {t_a:.2f}s")
    print("  Output:")
    for ln in text_a.split("\n"):
        print(f"    {ln}")
    pass_a = any(kw in text_a.lower() for kw in [
        "pump", "equipment", "status", "panel", "warning", "vibration", "p-101"
    ])
    print(f"  Result: {'PASS' if pass_a else 'PARTIAL'}")
except Exception as e:
    text_a, t_a, pass_a = str(e), 0.0, False
    print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Test B: Visual Q&A
# ---------------------------------------------------------------------------
section("TEST B -- Visual Q&A")
print("  Prompt: 'What equipment is shown and what is its vibration value?'")
try:
    text_b, t_b = ask(
        "What equipment is shown in this image and what is its vibration value?"
    )
    print(f"  Time  : {t_b:.2f}s")
    print("  Output:")
    for ln in text_b.split("\n"):
        print(f"    {ln}")
    has_pump = "P-101A" in text_b or "pump" in text_b.lower() or "p-101" in text_b.lower()
    has_vibr = "7.1" in text_b
    pass_b   = has_pump and has_vibr
    print(f"  P-101A recognized   : {'YES' if has_pump else 'NO'}")
    print(f"  7.1 mm/s recognized : {'YES' if has_vibr else 'NO'}")
    print(f"  Result: {'PASS' if pass_b else 'FAIL'}")
except Exception as e:
    text_b, t_b, pass_b = str(e), 0.0, False
    has_pump = has_vibr = False
    print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Test C: Structured extraction
# ---------------------------------------------------------------------------
section("TEST C -- Structured extraction")
print("  Prompt: 'List all key-value pairs visible in this image, one per line.'")
try:
    text_c, t_c = ask(
        "List all the labeled fields and their values visible in this image, one per line."
    )
    print(f"  Time  : {t_c:.2f}s")
    print("  Output:")
    for ln in text_c.split("\n"):
        print(f"    {ln}")
    has_equip  = "P-101A" in text_c or "Pump" in text_c or "pump" in text_c.lower()
    has_status = "Warning" in text_c or "warning" in text_c.lower()
    has_vibr_c = "7.1" in text_c
    has_maint  = "24" in text_c
    pass_c     = has_equip and has_vibr_c
    print(f"  Equipment (P-101A) : {'YES' if has_equip else 'NO'}")
    print(f"  Status (Warning)   : {'YES' if has_status else 'NO'}")
    print(f"  Vibration (7.1)    : {'YES' if has_vibr_c else 'NO'}")
    print(f"  Maintenance (24h)  : {'YES' if has_maint else 'NO'}")
    print(f"  Result: {'PASS' if pass_c else 'FAIL'}")
except Exception as e:
    text_c, t_c, pass_c = str(e), 0.0, False
    has_equip = has_status = has_vibr_c = has_maint = False
    print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
overall = pass_a and pass_b and pass_c

section("FINAL REPORT")
print(f"  QWEN3-VL 4B STANDALONE TEST")
print(f"  ----------------------------")
print(f"  Model: {MODEL}")
print(f"  Status: {'PASS' if overall else 'PARTIAL/FAIL'}")
print(f"")
print(f"  Test A -- Image description : {'PASS' if pass_a else 'FAIL'}")
print(f"  Test B -- Visual Q&A        : {'PASS' if pass_b else 'FAIL'}")
print(f"  Test C -- Structured extract: {'PASS' if pass_c else 'FAIL'}")
print(f"")
print(f"  Response times:")
print(f"    Test A: {t_a:.2f}s")
print(f"    Test B: {t_b:.2f}s")
print(f"    Test C: {t_c:.2f}s")
print(f"")
print(f"  Equipment recognized : {'YES' if has_pump else 'NO'}")
print(f"  Vibration recognized : {'YES' if has_vibr else 'NO'}")
print(f"")
print(f"  Overall vision capability: {'PASS' if overall else 'PARTIAL/FAIL'}")
print(SEP)
