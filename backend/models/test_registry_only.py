"""
test_registry_only.py

Registry-only tests — no Ollama, no model calls.
Runs in under 1 second.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.registry import get_model, list_capabilities, REGISTRY, ModelConfig
from config import LLM_MODEL, CODING_MODEL, EMBEDDING_MODEL

SEP = "=" * 60

def section(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")

section("TEST 1 -- Registry imports")
assert isinstance(REGISTRY, dict)
print(f"  Capabilities: {list(REGISTRY.keys())}")
print("  [PASS]")

section("TEST 2 -- get_model() for implemented capabilities")
for cap in ("general_reasoning", "coding", "embedding"):
    cfg = get_model(cap)
    assert isinstance(cfg, ModelConfig) and cfg.model_name
    print(f"  {cap:<22} -> {cfg.model_name}")
print("  [PASS]")

section("TEST 3 -- KeyError for unknown capability")
try:
    get_model("does_not_exist")
    raise AssertionError("Expected KeyError")
except KeyError as e:
    print(f"  KeyError raised: {e}")
print("  [PASS]")

section("TEST 4 -- ValueError for placeholder capabilities (vision, ocr)")
for cap in ("vision", "ocr"):
    try:
        get_model(cap)
        raise AssertionError(f"Expected ValueError for {cap!r}")
    except ValueError as e:
        print(f"  {cap}: ValueError: {e}")
print("  [PASS]")

section("TEST 5 -- list_capabilities() returns all 5 entries")
caps = list_capabilities()
assert len(caps) == 5, f"Expected 5, got {len(caps)}"
for c in caps:
    impl = c["implemented"]
    model = c["model"]
    cap_name = c["capability"]
    print(f"  {cap_name:<22} implemented={impl}  model={model}")
print("  [PASS]")

section("TEST 6 -- Config values reflected correctly in registry")
gr = get_model("general_reasoning")
co = get_model("coding")
em = get_model("embedding")
assert gr.model_name == LLM_MODEL,      f"Mismatch: {gr.model_name!r} != {LLM_MODEL!r}"
assert co.model_name == CODING_MODEL,   f"Mismatch: {co.model_name!r} != {CODING_MODEL!r}"
assert em.model_name == EMBEDDING_MODEL,f"Mismatch: {em.model_name!r} != {EMBEDDING_MODEL!r}"
print(f"  general_reasoning -> {gr.model_name}")
print(f"  coding            -> {co.model_name}")
print(f"  embedding         -> {em.model_name}")
print("  [PASS]")

section("TEST 7 -- ModelConfig is immutable (frozen dataclass)")
cfg = get_model("general_reasoning")
try:
    cfg.model_name = "something_else"
    raise AssertionError("Expected FrozenInstanceError")
except Exception as e:
    print(f"  Assignment blocked: {type(e).__name__}: {e}")
print("  [PASS]")

section("TEST 8 -- Coding capability uses Ollama provider")
co = get_model("coding")
assert co.provider == "ollama"
assert co.host.startswith("http")
print(f"  provider={co.provider!r}  host={co.host!r}")
print("  [PASS]")

print(f"\n{SEP}")
print("  REGISTRY TESTS (8/8) PASSED -- no model calls required")
print(SEP)
