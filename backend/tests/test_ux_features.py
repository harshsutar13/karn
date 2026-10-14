"""
tests/test_ux_features.py

Unit + integration tests for the SIH26 Chat UX upgrade:
  - Thinking mode flows through all layers
  - /api/ask backward-compat (no thinking field)
  - /api/ask with thinking=True/False
  - Upload endpoint saves file and returns path
  - /api/ask/stream yields SSE format
  - Regression: /api/capabilities, /api/ask non-stream

Tests are designed to:
  - Use mocks for unit-level tests (no model calls)
  - Make minimal live calls only for final integration verification
"""

import os
import sys
import json
import uuid
import tempfile
import io
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ===========================================================================
# 1. UNIT: thinking param flows through llm.py
# ===========================================================================

class TestLLMThinkParam:
    def test_generate_response_accepts_think_false(self, monkeypatch):
        """generate_response() accepts think=False without error."""
        from models.llm import generate_response
        calls = []
        class FakeResp:
            class message:
                content = "ok"
        def fake_chat(**kwargs):
            calls.append(kwargs)
            return FakeResp()
        class FakeClient:
            def chat(self, **kwargs): return fake_chat(**kwargs)
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: FakeClient())
        result = generate_response("hello", think=False)
        assert result == "ok"
        assert calls[0].get("think") == False

    def test_generate_response_accepts_think_true(self, monkeypatch):
        """generate_response() passes think=True to ollama."""
        from models.llm import generate_response
        calls = []
        class FakeResp:
            class message:
                content = "deep answer"
        class FakeClient:
            def chat(self, **kwargs):
                calls.append(kwargs)
                return FakeResp()
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: FakeClient())
        result = generate_response("explain quantum", think=True)
        assert result == "deep answer"
        assert calls[0].get("think") == True

    def test_generate_response_backward_compat(self, monkeypatch):
        """generate_response('msg') with no extra args still works."""
        from models.llm import generate_response
        class FakeResp:
            class message:
                content = "compat"
        class FakeClient:
            def chat(self, **kwargs): return FakeResp()
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: FakeClient())
        result = generate_response("test")
        assert result == "compat"

    def test_generate_response_stream_yields_chunks(self, monkeypatch):
        """generate_response(stream=True) yields text chunks."""
        from models.llm import generate_response
        class FakeChunk:
            def __init__(self, text):
                self.message = type("M", (), {"content": text})()
        class FakeClient:
            def chat(self, **kwargs):
                for t in ["Hello", " world", "!"]:
                    yield FakeChunk(t)
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: FakeClient())
        chunks = list(generate_response("test", stream=True))
        assert chunks == ["Hello", " world", "!"]

    def test_generate_response_stream_skips_empty(self, monkeypatch):
        """Stream generator skips empty chunks."""
        from models.llm import generate_response
        class FakeChunk:
            def __init__(self, text):
                self.message = type("M", (), {"content": text})()
        class FakeClient:
            def chat(self, **kwargs):
                for t in ["A", "", "B", ""]:
                    yield FakeChunk(t)
        monkeypatch.setattr("models.llm.ollama.Client", lambda host: FakeClient())
        chunks = list(generate_response("test", stream=True))
        assert chunks == ["A", "B"]


# ===========================================================================
# 2. UNIT: thinking flows through service.py
# ===========================================================================

class TestOrchestratorThinking:
    def test_handle_request_passes_thinking_general(self, monkeypatch):
        """thinking=True is passed to generate_response for general cap."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": "test"})()
        )
        called_with = {}
        def fake_generate(msg, think=False, stream=False):
            called_with["think"] = think
            return "answer"
        monkeypatch.setattr("orchestration.service.generate_response", fake_generate)
        svc.handle_request("hello", thinking=True)
        assert called_with["think"] == True

    def test_handle_request_passes_thinking_false(self, monkeypatch):
        """thinking=False is passed through (default)."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        called_with = {}
        def fake_generate(msg, think=False, stream=False):
            called_with["think"] = think
            return "ans"
        monkeypatch.setattr("orchestration.service.generate_response", fake_generate)
        svc.handle_request("hello", thinking=False)
        assert called_with["think"] == False

    def test_handle_request_default_no_thinking(self, monkeypatch):
        """handle_request with no thinking arg defaults to False."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        called_with = {}
        def fake_generate(msg, think=False, stream=False):
            called_with["think"] = think
            return "ans"
        monkeypatch.setattr("orchestration.service.generate_response", fake_generate)
        svc.handle_request("hello")  # no thinking kwarg
        assert called_with["think"] == False


# ===========================================================================
# 3. UNIT: SSE helper functions
# ===========================================================================

class TestSSEHelpers:
    def test_sse_meta_format(self):
        """_sse_meta returns correctly formatted SSE line."""
        from orchestration.service import _sse_meta
        line = _sse_meta("general", "gemma4:12b")
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[6:].strip())
        assert payload["type"] == "meta"
        assert payload["capability"] == "general"
        assert payload["model"] == "gemma4:12b"

    def test_sse_chunk_format(self):
        """_sse_chunk returns correctly formatted SSE line."""
        from orchestration.service import _sse_chunk
        line = _sse_chunk("hello world")
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[6:].strip())
        assert payload["type"] == "chunk"
        assert payload["text"] == "hello world"

    def test_sse_done_format(self):
        """_sse_done returns correct sentinel."""
        from orchestration.service import _sse_done
        assert _sse_done() == "data: [DONE]\n\n"


# ===========================================================================
# 4. UNIT: handle_request_stream yields correct SSE structure
# ===========================================================================

class TestHandleRequestStream:
    def test_stream_general_yields_meta_chunks_done(self, monkeypatch):
        """Stream for general capability yields meta → chunks → done."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        def fake_generate(msg, think=False, stream=False):
            if stream:
                return iter(["2 ", "+ ", "2 = ", "4"])
            return "4"
        monkeypatch.setattr("orchestration.service.generate_response", fake_generate)

        events = list(svc.handle_request_stream("What is 2+2?"))
        assert events[0].startswith("data: ")
        meta = json.loads(events[0][6:].strip())
        assert meta["type"] == "meta"
        assert meta["capability"] == "general"

        chunk_events = [e for e in events if "chunk" in e]
        assert len(chunk_events) >= 1
        texts = [json.loads(e[6:].strip())["text"] for e in chunk_events]
        assert "".join(texts) == "2 + 2 = 4"

        assert events[-1] == "data: [DONE]\n\n"

    def test_stream_passes_thinking(self, monkeypatch):
        """handle_request_stream passes thinking=True to generate_response."""
        import orchestration.service as svc
        monkeypatch.setattr(
            "orchestration.router.classify_request",
            lambda msg, img=None: type("C", (), {"capability": "general", "reason": ""})()
        )
        called_with = {}
        def fake_generate(msg, think=False, stream=False):
            called_with["think"] = think
            return iter(["ok"]) if stream else "ok"
        monkeypatch.setattr("orchestration.service.generate_response", fake_generate)
        list(svc.handle_request_stream("test", thinking=True))
        assert called_with["think"] == True


# ===========================================================================
# 5. UNIT: /api/ask FastAPI schema — thinking field
# ===========================================================================

class TestAskEndpointSchema:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    def test_ask_backward_compat_no_thinking(self, client, monkeypatch):
        """POST /api/ask without thinking field still works."""
        import orchestration.service as svc
        monkeypatch.setattr(
            svc, "handle_request",
            lambda msg, image_path=None, thinking=False:
                svc.OrchestratorResponse("ok", "general", "gemma4:12b")
        )
        resp = client.post("/api/ask", json={"message": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "ok"
        assert data["capability"] == "general"
        assert data["model"] == "gemma4:12b"

    def test_ask_with_thinking_false(self, client, monkeypatch):
        """POST /api/ask with thinking=false works."""
        import orchestration.service as svc
        received = {}
        def fake_handle(msg, image_path=None, thinking=False):
            received["thinking"] = thinking
            return svc.OrchestratorResponse("ans", "general", "gemma4:12b")
        monkeypatch.setattr(svc, "handle_request", fake_handle)
        resp = client.post("/api/ask", json={"message": "hello", "thinking": False})
        assert resp.status_code == 200
        assert received["thinking"] == False

    def test_ask_with_thinking_true(self, client, monkeypatch):
        """POST /api/ask with thinking=true passes thinking to orchestrator."""
        import orchestration.service as svc
        received = {}
        def fake_handle(msg, image_path=None, thinking=False):
            received["thinking"] = thinking
            return svc.OrchestratorResponse("deep", "general", "gemma4:12b")
        monkeypatch.setattr(svc, "handle_request", fake_handle)
        resp = client.post("/api/ask", json={"message": "explain this", "thinking": True})
        assert resp.status_code == 200
        assert received["thinking"] == True

    def test_ask_stream_endpoint_exists(self, client, monkeypatch):
        """POST /api/ask/stream returns 200 and text/event-stream."""
        import orchestration.service as svc
        def fake_stream(msg, image_path=None, thinking=False):
            yield 'data: {"type":"meta","capability":"general","model":"gemma4:12b"}\n\n'
            yield 'data: {"type":"chunk","text":"4"}\n\n'
            yield "data: [DONE]\n\n"
        monkeypatch.setattr(svc, "handle_request_stream", fake_stream)
        resp = client.post("/api/ask/stream", json={"message": "2+2?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ===========================================================================
# 6. UNIT: Upload endpoint
# ===========================================================================

class TestUploadEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    def test_upload_valid_png(self, client, tmp_path):
        """POST /api/upload/image with a PNG returns file_path and filename."""
        # Create a minimal 1×1 PNG (valid header bytes)
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
            b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
            b'\xd4n\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        resp = client.post(
            "/api/upload/image",
            files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "file_path" in data
        assert data["filename"] == "test.png"
        assert os.path.isfile(data["file_path"])
        # Cleanup
        os.unlink(data["file_path"])

    def test_upload_unsupported_type_returns_415(self, client):
        """POST /api/upload/image with a non-image returns 415."""
        resp = client.post(
            "/api/upload/image",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert resp.status_code == 415

    def test_upload_image_routes_to_vision(self, client, tmp_path, monkeypatch):
        """After upload, passing file_path to /api/ask triggers vision."""
        import orchestration.service as svc
        received = {}
        def fake_handle(msg, image_path=None, thinking=False):
            received["image_path"] = image_path
            received["cap"] = "vision"
            return svc.OrchestratorResponse("P-101A seen", "vision", "qwen3-vl:4b")
        monkeypatch.setattr(svc, "handle_request", fake_handle)

        # First upload a real (minimal) PNG
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
            b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
            b'\xd4n\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        upload_resp = client.post(
            "/api/upload/image",
            files={"file": ("img.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert upload_resp.status_code == 200
        file_path = upload_resp.json()["file_path"]

        # Now POST to /api/ask with image_path set
        ask_resp = client.post("/api/ask", json={
            "message": "What do you see?",
            "image_path": file_path,
        })
        assert ask_resp.status_code == 200
        assert received.get("image_path") == file_path
        os.unlink(file_path)


# ===========================================================================
# 7. REGRESSION: existing endpoints unchanged
# ===========================================================================

class TestRegression:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    def test_capabilities_still_works(self, client):
        """/api/capabilities returns 200 and has expected keys."""
        resp = client.get("/api/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "capabilities" in data
        caps = {c["name"] for c in data["capabilities"]}
        assert "general" in caps
        assert "coding" in caps
        assert "vision" in caps

    def test_root_returns_html(self, client):
        """GET / returns HTML containing SIH26."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "SIH26" in resp.text

    def test_ask_returns_required_fields(self, client, monkeypatch):
        """/api/ask non-stream response has answer, capability, model."""
        import orchestration.service as svc
        monkeypatch.setattr(
            svc, "handle_request",
            lambda msg, image_path=None, thinking=False:
                svc.OrchestratorResponse("test answer", "general", "gemma4:12b")
        )
        resp = client.post("/api/ask", json={"message": "hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) >= {"answer", "capability", "model"}
