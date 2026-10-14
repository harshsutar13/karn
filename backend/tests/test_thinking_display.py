"""
tests/test_thinking_display.py

Tests for the thinking-display improvement:
  - Ollama SDK exposes chunk.message.thinking (separate from chunk.message.content)
  - llm.py streaming yields {"kind":"thinking", "text":...} for reasoning tokens
  - llm.py streaming yields {"kind":"chunk",    "text":...} for answer tokens
  - coding.py streaming yields same typed dicts
  - vision.py streaming yields only "chunk" dicts (no thinking)
  - service.py translates typed dicts to SSE "thinking" / "chunk" events
  - /api/ask/stream SSE includes thinking events when thinking=True
  - /api/ask/stream SSE has NO thinking events when thinking=False
  - Existing /api/ask non-stream still works
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# Helpers
# ============================================================================

def make_chunk(thinking_text=None, content_text=None):
    """Build a fake Ollama streaming chunk with given thinking/content fields."""
    msg = type("Msg", (), {
        "thinking": thinking_text,
        "content":  content_text,
    })()
    return type("Chunk", (), {"message": msg})()


# ============================================================================
# 1. llm.py streaming — typed dict output
# ============================================================================

class TestLLMStreamTypedDicts:
    def _make_client(self, chunks):
        class FakeClient:
            def chat(self_, **kwargs):
                yield from chunks
        return FakeClient()

    def test_thinking_chunks_emitted_as_thinking_kind(self, monkeypatch):
        """Reasoning tokens → {"kind": "thinking", "text": ...}"""
        from models.llm import generate_response
        client = self._make_client([
            make_chunk(thinking_text="Let me think..."),
            make_chunk(thinking_text=" step 2"),
            make_chunk(content_text="The answer is 4"),
        ])
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: client)

        results = list(generate_response("test", think=True, stream=True))
        thinking = [r for r in results if r["kind"] == "thinking"]
        chunks   = [r for r in results if r["kind"] == "chunk"]

        assert len(thinking) == 2
        assert thinking[0]["text"] == "Let me think..."
        assert thinking[1]["text"] == " step 2"
        assert len(chunks) == 1
        assert chunks[0]["text"] == "The answer is 4"

    def test_content_chunks_emitted_as_chunk_kind(self, monkeypatch):
        """Answer tokens → {"kind": "chunk", "text": ...}"""
        from models.llm import generate_response
        client = self._make_client([
            make_chunk(content_text="Hello"),
            make_chunk(content_text=" world"),
        ])
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: client)

        results = list(generate_response("test", think=False, stream=True))
        assert all(r["kind"] == "chunk" for r in results)
        assert "".join(r["text"] for r in results) == "Hello world"

    def test_no_thinking_when_think_false(self, monkeypatch):
        """When think=False, no thinking dicts emitted even if model surprises us."""
        from models.llm import generate_response
        # Model should not send thinking when think=False, but test defensively
        client = self._make_client([
            make_chunk(content_text="2 + 2 = 4"),
        ])
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: client)

        results = list(generate_response("test", think=False, stream=True))
        assert not any(r["kind"] == "thinking" for r in results)

    def test_empty_chunks_skipped(self, monkeypatch):
        """Chunks with both thinking and content as None/empty are skipped."""
        from models.llm import generate_response
        client = self._make_client([
            make_chunk(thinking_text=None, content_text=None),
            make_chunk(thinking_text="",   content_text=""),
            make_chunk(content_text="ok"),
        ])
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: client)

        results = list(generate_response("test", stream=True))
        assert results == [{"kind": "chunk", "text": "ok"}]

    def test_thinking_and_content_in_same_raw_chunk(self, monkeypatch):
        """If a chunk has BOTH thinking and content, both are emitted."""
        from models.llm import generate_response
        client = self._make_client([
            make_chunk(thinking_text="reason", content_text="answer"),
        ])
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: client)

        results = list(generate_response("test", stream=True))
        assert {"kind": "thinking", "text": "reason"} in results
        assert {"kind": "chunk",    "text": "answer"}  in results


# ============================================================================
# 2. coding.py streaming — same typed dict format
# ============================================================================

class TestCodingStreamTypedDicts:
    def test_coding_thinking_chunks(self, monkeypatch):
        """Coding model thinking tokens → {"kind": "thinking", "text": ...}"""
        from models.coding import generate_code
        client_calls = []

        class FakeClient:
            def chat(self_, **kwargs):
                client_calls.append(kwargs)
                yield make_chunk(thinking_text="Planning code...")
                yield make_chunk(content_text="def factorial(n):")
                yield make_chunk(content_text="    return 1 if n == 0 else n * factorial(n-1)")

        monkeypatch.setattr("models.coding.ollama.Client", lambda host: FakeClient())

        results = list(generate_code("factorial function", think=True, stream=True))
        thinking_texts = [r["text"] for r in results if r["kind"] == "thinking"]
        chunk_texts    = [r["text"] for r in results if r["kind"] == "chunk"]

        assert thinking_texts == ["Planning code..."]
        assert "def factorial" in "".join(chunk_texts)

    def test_coding_no_thinking_when_off(self, monkeypatch):
        """When think=False, coding model emits only chunk dicts."""
        from models.coding import generate_code

        class FakeClient:
            def chat(self_, **kwargs):
                yield make_chunk(content_text="def f(): pass")

        monkeypatch.setattr("models.coding.ollama.Client", lambda host: FakeClient())

        results = list(generate_code("simple function", stream=True))
        assert all(r["kind"] == "chunk" for r in results)


# ============================================================================
# 3. vision.py streaming — only chunk dicts (no thinking)
# ============================================================================

class TestVisionStreamTypedDicts:
    def test_vision_emits_only_chunk_dicts(self, monkeypatch):
        """Vision model never emits thinking dicts."""
        from models.vision import analyze_image

        class FakeClient:
            def chat(self_, **kwargs):
                yield make_chunk(content_text="Pump P-101A")
                yield make_chunk(content_text=" vibration 7.1 mm/s")

        monkeypatch.setattr("models.vision.ollama.Client", lambda host: FakeClient())

        # Mock file I/O (isfile, getsize, open, base64)
        monkeypatch.setattr("models.vision.os.path.isfile", lambda p: True)
        monkeypatch.setattr("models.vision.os.path.getsize", lambda p: 4)  # debug log
        monkeypatch.setattr("builtins.open", lambda p, m: type("F", (), {
            "__enter__": lambda s: type("B", (), {"read": lambda s: b"fake"})(),
            "__exit__": lambda *a: None,
        })())
        monkeypatch.setattr("models.vision.base64.b64encode", lambda b: b"ZmFrZQ==")

        results = list(analyze_image("fake.png", "describe", stream=True))
        assert len(results) == 2
        assert all(r["kind"] == "chunk" for r in results)
        assert not any(r["kind"] == "thinking" for r in results)
        # Spaces in token text are now preserved (only empty chunks skipped)
        assert results[1]["text"] == " vibration 7.1 mm/s"


# ============================================================================
# 4. service.py SSE helpers
# ============================================================================

class TestSSEHelpers:
    def test_sse_thinking_format(self):
        """_sse_thinking emits type='thinking' SSE event."""
        from orchestration.service import _sse_thinking
        line = _sse_thinking("I need to reason about this.")
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[6:].strip())
        assert payload["type"] == "thinking"
        assert payload["text"] == "I need to reason about this."

    def test_sse_from_typed_thinking(self):
        """_sse_from_typed routes thinking kind to _sse_thinking."""
        from orchestration.service import _sse_from_typed
        line = _sse_from_typed({"kind": "thinking", "text": "hmm"})
        payload = json.loads(line[6:].strip())
        assert payload["type"] == "thinking"

    def test_sse_from_typed_chunk(self):
        """_sse_from_typed routes chunk kind to _sse_chunk."""
        from orchestration.service import _sse_from_typed
        line = _sse_from_typed({"kind": "chunk", "text": "hello"})
        payload = json.loads(line[6:].strip())
        assert payload["type"] == "chunk"

    def test_sse_from_typed_defaults_to_chunk(self):
        """_sse_from_typed with unknown kind defaults to chunk."""
        from orchestration.service import _sse_from_typed
        line = _sse_from_typed({"kind": "unknown", "text": "x"})
        payload = json.loads(line[6:].strip())
        assert payload["type"] == "chunk"


# ============================================================================
# 5. service.py handle_request_stream — thinking events in SSE output
# ============================================================================

class TestHandleRequestStreamThinking:
    def _parse_events(self, sse_lines):
        events = []
        for line in sse_lines:
            if not line.startswith("data: "): continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                events.append({"type": "__done__"})
                continue
            try: events.append(json.loads(payload))
            except: pass
        return events

    def test_thinking_events_present_when_thinking_true(self, monkeypatch):
        """When thinking=True, stream yields thinking events before chunk events."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        def fake_generate(msg, think=False, stream=False, history=None):
            if stream:
                return iter([
                    {"kind": "thinking", "text": "Let me think..."},
                    {"kind": "chunk",    "text": "The answer is 4"},
                ])
            return "The answer is 4"
        # generate_response is lazily imported inside handle_request_stream;
        # patch at the source module so the lazy `from models.llm import ...` picks it up.
        monkeypatch.setattr("models.llm.generate_response", fake_generate)

        events = self._parse_events(list(svc.handle_request_stream("2+2?", thinking=True)))

        types = [e["type"] for e in events]
        assert "thinking" in types
        assert "chunk" in types
        # thinking must come before chunk
        thinking_idx = types.index("thinking")
        chunk_idx    = types.index("chunk")
        assert thinking_idx < chunk_idx

    def test_no_thinking_events_when_thinking_false(self, monkeypatch):
        """When thinking=False, stream yields NO thinking events."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        def fake_generate(msg, think=False, stream=False, history=None):
            if stream:
                return iter([{"kind": "chunk", "text": "The answer is 4"}])
            return "The answer is 4"
        monkeypatch.setattr("models.llm.generate_response", fake_generate)

        events = self._parse_events(list(svc.handle_request_stream("2+2?", thinking=False)))
        assert not any(e.get("type") == "thinking" for e in events)
        assert any(e.get("type") == "chunk" for e in events)

    def test_stream_ends_with_done(self, monkeypatch):
        """SSE stream always ends with [DONE]."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        def fake_generate(msg, think=False, stream=False, history=None):
            return iter([{"kind": "chunk", "text": "ok"}]) if stream else "ok"
        monkeypatch.setattr("models.llm.generate_response", fake_generate)

        events = self._parse_events(list(svc.handle_request_stream("hello")))
        assert events[-1]["type"] == "__done__"


# ============================================================================
# 6. FastAPI /api/ask/stream — thinking events in HTTP SSE
# ============================================================================

class TestAskStreamThinkingHTTP:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    def test_stream_with_thinking_true_includes_thinking_type(self, client, monkeypatch):
        """POST /api/ask/stream thinking=true → SSE contains type=thinking events."""
        import orchestration.service as svc

        def fake_stream(msg, image_path=None, thinking=False):
            yield 'data: {"type":"meta","capability":"general","model":"gemma4:12b"}\n\n'
            if thinking:
                yield 'data: {"type":"thinking","text":"I need to reason..."}\n\n'
            yield 'data: {"type":"chunk","text":"The answer is 4"}\n\n'
            yield "data: [DONE]\n\n"

        monkeypatch.setattr(svc, "handle_request_stream", fake_stream)

        resp = client.post("/api/ask/stream", json={"message": "2+2?", "thinking": True})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        events = []
        for line in body.splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                try: events.append(json.loads(line[6:]))
                except: pass

        types = {e["type"] for e in events}
        assert "thinking" in types
        assert "chunk" in types

    def test_stream_with_thinking_false_no_thinking_events(self, client, monkeypatch):
        """POST /api/ask/stream thinking=false → no type=thinking events."""
        import orchestration.service as svc

        def fake_stream(msg, image_path=None, thinking=False):
            yield 'data: {"type":"meta","capability":"general","model":"gemma4:12b"}\n\n'
            yield 'data: {"type":"chunk","text":"The answer is 4"}\n\n'
            yield "data: [DONE]\n\n"

        monkeypatch.setattr(svc, "handle_request_stream", fake_stream)

        resp = client.post("/api/ask/stream", json={"message": "2+2?", "thinking": False})
        assert resp.status_code == 200

        body = resp.text
        for line in body.splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                try:
                    evt = json.loads(line[6:])
                    assert evt.get("type") != "thinking", f"Unexpected thinking event: {evt}"
                except json.JSONDecodeError:
                    pass


# ============================================================================
# 7. Regression — /api/ask non-stream still works
# ============================================================================

class TestRegression:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    def test_ask_non_stream_backward_compat(self, client, monkeypatch):
        """/api/ask (non-stream) still returns {answer, capability, model}."""
        import orchestration.service as svc
        monkeypatch.setattr(
            svc, "handle_request",
            lambda msg, image_path=None, thinking=False:
                svc.OrchestratorResponse("The answer is 4", "general", "gemma4:12b")
        )
        resp = client.post("/api/ask", json={"message": "2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The answer is 4"
        assert data["capability"] == "general"
        assert data["model"] == "gemma4:12b"

    def test_ask_no_thinking_field_still_works(self, client, monkeypatch):
        """/api/ask without thinking field defaults correctly."""
        import orchestration.service as svc
        received = {}
        def fake_handle(msg, image_path=None, thinking=False):
            received["thinking"] = thinking
            return svc.OrchestratorResponse("ok", "general", "gemma4:12b")
        monkeypatch.setattr(svc, "handle_request", fake_handle)
        client.post("/api/ask", json={"message": "hi"})
        assert received["thinking"] == False

    def test_capabilities_still_200(self, client):
        """/api/capabilities still returns 200."""
        resp = client.get("/api/capabilities")
        assert resp.status_code == 200
        assert "capabilities" in resp.json()


# ============================================================================
# 8. Vision normal-mode policy — vision always runs with thinking=False
# ============================================================================

class TestVisionNormalModePolicy:
    """
    MVP policy: capability=vision always calls analyze_image(think=False),
    regardless of the global thinking flag passed by the caller.
    The service layer enforces this; the vision model never receives think=True.
    """

    def _vision_mocks(self, monkeypatch, chunks=None):
        """Apply standard vision file-IO and client mocks."""
        if chunks is None:
            chunks = [make_chunk(content_text="Pump P-101A")]

        class FakeClient:
            def __init__(self, _chunks):
                self._chunks = _chunks
            def chat(self_, **kwargs):
                yield from self_._chunks

        monkeypatch.setattr(
            "models.vision.ollama.Client",
            lambda host: FakeClient(chunks)
        )
        monkeypatch.setattr("models.vision.os.path.isfile", lambda p: True)
        monkeypatch.setattr("models.vision.os.path.getsize", lambda p: 4)
        monkeypatch.setattr("builtins.open", lambda p, m: type("F", (), {
            "__enter__": lambda s: type("B", (), {"read": lambda s: b"fake"})(),
            "__exit__": lambda *a: None,
        })())
        monkeypatch.setattr("models.vision.base64.b64encode", lambda b: b"ZmFrZQ==")

    def _route_to_vision(self, monkeypatch):
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "vision", "reason": ""})()
        )

    def _parse_events(self, sse_lines):
        events = []
        for line in sse_lines:
            if not line.startswith("data: "): continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                events.append({"type": "__done__"})
                continue
            try: events.append(json.loads(payload))
            except: pass
        return events

    def test_vision_stream_thinking_true_produces_no_thinking_events(self, monkeypatch):
        """Even with thinking=True, vision stream emits zero 'thinking' SSE events."""
        import orchestration.service as svc
        self._route_to_vision(monkeypatch)
        self._vision_mocks(monkeypatch)

        events = self._parse_events(
            list(svc.handle_request_stream("describe", image_path="fake.png", thinking=True))
        )
        assert not any(e.get("type") == "thinking" for e in events), \
            "Vision must not emit thinking events"
        assert any(e.get("type") == "chunk" for e in events), \
            "Vision must emit at least one chunk event"
        assert events[-1]["type"] == "__done__"

    def test_vision_stream_thinking_false_produces_no_thinking_events(self, monkeypatch):
        """With thinking=False, vision stream also emits zero 'thinking' events."""
        import orchestration.service as svc
        self._route_to_vision(monkeypatch)
        self._vision_mocks(monkeypatch)

        events = self._parse_events(
            list(svc.handle_request_stream("describe", image_path="fake.png", thinking=False))
        )
        assert not any(e.get("type") == "thinking" for e in events)
        assert any(e.get("type") == "chunk" for e in events)

    def test_vision_meta_event_identifies_capability(self, monkeypatch):
        """Vision stream always starts with meta capability=vision."""
        import orchestration.service as svc
        self._route_to_vision(monkeypatch)
        self._vision_mocks(monkeypatch)

        events = self._parse_events(
            list(svc.handle_request_stream("describe", image_path="fake.png", thinking=True))
        )
        meta_events = [e for e in events if e.get("type") == "meta"]
        assert len(meta_events) == 1
        assert meta_events[0]["capability"] == "vision"

    def test_general_thinking_true_still_produces_thinking_events(self, monkeypatch):
        """General capability with thinking=True still produces thinking events (not affected)."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        def fake_generate(msg, think=False, stream=False, history=None):
            if stream:
                return iter([
                    {"kind": "thinking", "text": "reasoning..."},
                    {"kind": "chunk",    "text": "The answer"},
                ])
            return "The answer"
        monkeypatch.setattr("models.llm.generate_response", fake_generate)

        events = self._parse_events(
            list(svc.handle_request_stream("2+2?", thinking=True))
        )
        assert any(e.get("type") == "thinking" for e in events), \
            "General capability must still produce thinking events when thinking=True"
        assert any(e.get("type") == "chunk" for e in events)


# ============================================================================
# 9. Vision streaming whitespace preservation
# ============================================================================

class TestVisionStreamingWhitespace:
    """
    Verify that inter-token leading spaces are preserved.
    vision.py must check emptiness with text.strip() but yield original `text`
    so that tokenised output like ['The', ' image', ' shows'] concatenates
    to 'The image shows' (not 'Theimageshows').
    """

    def test_leading_spaces_preserved_in_chunks(self, monkeypatch):
        """Tokens with leading spaces are yielded as-is."""
        from models.vision import analyze_image

        tokens = ["The", " image", " shows", " a", " pump"]

        class FakeClient:
            def chat(self_, **kwargs):
                for t in tokens:
                    yield make_chunk(content_text=t)

        monkeypatch.setattr("models.vision.ollama.Client", lambda host: FakeClient())
        monkeypatch.setattr("models.vision.os.path.isfile", lambda p: True)
        monkeypatch.setattr("models.vision.os.path.getsize", lambda p: 4)
        monkeypatch.setattr("builtins.open", lambda p, m: type("F", (), {
            "__enter__": lambda s: type("B", (), {"read": lambda s: b"fake"})(),
            "__exit__": lambda *a: None,
        })())
        monkeypatch.setattr("models.vision.base64.b64encode", lambda b: b"ZmFrZQ==")

        results = list(analyze_image("fake.png", "describe", stream=True))
        reconstructed = "".join(r["text"] for r in results)

        assert reconstructed == "The image shows a pump", \
            f"Expected 'The image shows a pump', got: {repr(reconstructed)}"

    def test_pump_vibration_tokens_preserve_spaces(self, monkeypatch):
        """Equipment description tokens concatenate correctly."""
        from models.vision import analyze_image

        tokens = ["Pump", " P-101A", " has", " a", " vibration", " value"]

        class FakeClient:
            def chat(self_, **kwargs):
                for t in tokens:
                    yield make_chunk(content_text=t)

        monkeypatch.setattr("models.vision.ollama.Client", lambda host: FakeClient())
        monkeypatch.setattr("models.vision.os.path.isfile", lambda p: True)
        monkeypatch.setattr("models.vision.os.path.getsize", lambda p: 4)
        monkeypatch.setattr("builtins.open", lambda p, m: type("F", (), {
            "__enter__": lambda s: type("B", (), {"read": lambda s: b"fake"})(),
            "__exit__": lambda *a: None,
        })())
        monkeypatch.setattr("models.vision.base64.b64encode", lambda b: b"ZmFrZQ==")

        results = list(analyze_image("fake.png", "describe", stream=True))
        reconstructed = "".join(r["text"] for r in results)

        assert reconstructed == "Pump P-101A has a vibration value", \
            f"Expected 'Pump P-101A has a vibration value', got: {repr(reconstructed)}"

    def test_whitespace_only_chunks_are_skipped(self, monkeypatch):
        """Chunks that are purely whitespace/empty are still filtered out."""
        from models.vision import analyze_image

        class FakeClient:
            def chat(self_, **kwargs):
                yield make_chunk(content_text="Word")
                yield make_chunk(content_text="   ")   # whitespace-only — should be skipped
                yield make_chunk(content_text="")      # empty — should be skipped
                yield make_chunk(content_text=" next")

        monkeypatch.setattr("models.vision.ollama.Client", lambda host: FakeClient())
        monkeypatch.setattr("models.vision.os.path.isfile", lambda p: True)
        monkeypatch.setattr("models.vision.os.path.getsize", lambda p: 4)
        monkeypatch.setattr("builtins.open", lambda p, m: type("F", (), {
            "__enter__": lambda s: type("B", (), {"read": lambda s: b"fake"})(),
            "__exit__": lambda *a: None,
        })())
        monkeypatch.setattr("models.vision.base64.b64encode", lambda b: b"ZmFrZQ==")

        results = list(analyze_image("fake.png", "describe", stream=True))
        assert len(results) == 2, f"Expected 2 chunks (skipped empty/ws), got {len(results)}"
        reconstructed = "".join(r["text"] for r in results)
        assert reconstructed == "Word next"
