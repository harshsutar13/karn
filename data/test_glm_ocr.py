"""
Standalone GLM-OCR test.
No project imports — pure Ollama API call with a base64-encoded image.
"""
import base64
import time
import sys
import os

IMAGE_PATH = r"C:\WORK DRIVE\SIH26\data\ocr_test.png"
MODEL      = "glm-ocr:latest"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Read and base64-encode the image
with open(IMAGE_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

print(f"Model   : {MODEL}")
print(f"Image   : {IMAGE_PATH} ({os.path.getsize(IMAGE_PATH)} bytes)")
print(f"Sending OCR request...\n")

import urllib.request, json

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Extract all text from this image exactly as it appears.",
            "images": [img_b64],
        }
    ],
    "stream": False,
}

data = json.dumps(payload).encode()
req  = urllib.request.Request(OLLAMA_URL, data=data,
                              headers={"Content-Type": "application/json"})

t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    sys.exit(1)
elapsed = time.perf_counter() - t0

extracted = resp.get("message", {}).get("content", "").strip()

print("=" * 50)
print(f"  GLM-OCR STANDALONE TEST RESULT")
print("=" * 50)
print(f"  Status  : {'PASS' if extracted else 'FAIL (empty response)'}")
print(f"  Time    : {elapsed:.2f}s")
print(f"\n  Extracted text:\n")
for line in extracted.split("\n"):
    print(f"    {line}")

# Check for key technical values
checks = {
    "pump P-101A"        : "P-101A" in extracted,
    "7.1 mm/s RMS"       : "7.1" in extracted,
    "24 hours"           : "24" in extracted,
    "SIH26 OCR TEST"     : "SIH26" in extracted or "OCR" in extracted,
    "Vibration threshold": "Vibration" in extracted or "vibration" in extracted.lower(),
}
print(f"\n  Technical value checks:")
all_pass = True
for label, result in checks.items():
    status = "YES" if result else "NO"
    if not result:
        all_pass = False
    print(f"    {label:<25} : {status}")

print(f"\n  Technical values preserved: {'YES' if all_pass else 'PARTIAL/NO'}")
print("=" * 50)
