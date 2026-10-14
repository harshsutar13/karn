"""Quick helper: generate vision_test.png and verify it is a valid PNG."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from test_qwen3_vl import create_vision_test_png

OUT = r"C:\WORK DRIVE\SIH26\data\vision_test.png"
create_vision_test_png(OUT)

# Verify PNG signature
with open(OUT, "rb") as f:
    sig = f.read(8)
assert sig == b"\x89PNG\r\n\x1a\n", f"Bad PNG signature: {sig!r}"
print(f"  PNG signature: OK")
print(f"  File size    : {os.path.getsize(OUT)} bytes")
print("  PNG generation: PASS")
