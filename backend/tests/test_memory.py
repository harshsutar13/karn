"""
tests/test_memory.py

Offline tests for the persistent session-memory system:

  1. SQLite store (messages, actions, context builders) — no Ollama needed
  2. Orchestration: history injection + turn persistence + graceful
     degradation when the memory store fails
  3. HTTP endpoints: /api/ask (session_id), /api/memory/*

All tests monkeypatch the LLM boundary, so nothing here needs Ollama,
Qdrant, or any other service running.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory import store as memory_store


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def memdb(tmp_path, monkeypatch):
    """Point the memory store at a throwaway SQLite file."""
    db = tmp_path / "memory.db"
    monkeypatch.setattr(memory_store, "MEMORY_DB_PATH", str(db))
    return db


@pytest.fixture
def general_cap(monkeypatch):
    """Force the router to always classify as 'general'."""
    def fake_classify(message, image_path=None, **kwargs):
        return type("C", (), {"capability": "general", "reason": "test"})()
    monkeypatch.setattr("orchestration.service.classify_request", fake_classify)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace models.llm.generate_response with a deterministic fake."""
    received = {}

    def fake_generate(message, think=False, stream=False, history=None):
        received["message"] = message
        received["history"] = history
        if stream:
            return iter([
                {"kind": "chunk", "text": "mem "},
                {"kind": "chunk", "text": "answer"},
            ])
        return "mem answer"

    monkeypatch.setattr("models.llm.generate_response", fake_generate)
    return received


@pytest.fixture
def client(monkeypatch, memdb):
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)


# ===========================================================================
# 1. Store: session ids
# ===========================================================================

class TestNormalizeSessionId:
    def test_none_becomes_default(self):
        assert memory_store.normalize_session_id(None) == "default"

    def test_empty_becomes_default(self):
        assert memory_store.normalize_session_id("   ") == "default"

    def test_sanitises_unsafe_characters(self):
        assert memory_store.normalize_session_id('a b";drop') == "a_b__drop"

    def test_caps_length(self):
        sid = memory_store.normalize_session_id("x" * 200)
        assert len(sid) == memory_store.MAX_SESSION_ID_LEN


# ===========================================================================
# 2. Store: messages
# ===========================================================================

class TestMessages:
    def test_roundtrip_oldest_first(self, memdb):
        memory_store.add_message("s1", "user", "hello")
        memory_store.add_message("s1", "assistant", "hi there")
        memory_store.add_message("s1", "user", "bye")

        history = memory_store.get_history("s1")
        assert [m["role"] for m in history] == ["user", "assistant", "user"]
        assert history[0]["content"] == "hello"
        assert history[-1]["content"] == "bye"

    def test_unknown_session_is_empty(self, memdb):
        assert memory_store.get_history("nope") == []

    def test_rejects_bad_role(self, memdb):
        with pytest.raises(ValueError):
            memory_store.add_message("s1", "system", "not allowed")

    def test_skips_empty_content(self, memdb):
        memory_store.add_message("s1", "user", "   ")
        assert memory_store.get_history("s1") == []

    def test_sessions_are_isolated(self, memdb):
        memory_store.add_message("s1", "user", "one")
        memory_store.add_message("s2", "user", "two")
        assert [m["content"] for m in memory_store.get_history("s1")] == ["one"]
        assert [m["content"] for m in memory_store.get_history("s2")] == ["two"]

    def test_limit_returns_most_recent(self, memdb):
        for i in range(5):
            memory_store.add_message("s1", "user", f"m{i}")
        history = memory_store.get_history("s1", limit=2)
        assert [m["content"] for m in history] == ["m3", "m4"]


# ===========================================================================
# 3. Store: actions
# ===========================================================================

class TestActions:
    def test_log_and_fetch(self, memdb):
        memory_store.log_action("s1", "tool", "excel", "Created report.xlsx")
        actions = memory_store.get_recent_actions("s1")
        assert len(actions) == 1
        assert actions[0]["tool"] == "excel"
        assert actions[0]["summary"] == "Created report.xlsx"
        assert actions[0]["ok"] == 1

    def test_failed_action_flag(self, memdb):
        memory_store.log_action("s1", "final", None, "plan crashed", ok=False)
        assert memory_store.get_recent_actions("s1")[0]["ok"] == 0

    def test_summary_truncated_to_500(self, memdb):
        memory_store.log_action("s1", "tool", "t", "x" * 900)
        assert len(memory_store.get_recent_actions("s1")[0]["summary"]) == 500

    def test_actions_context_block(self, memdb):
        memory_store.log_action("s1", "tool", "word", "Wrote plan.docx")
        memory_store.log_action("s1", "final", None, "Goal complete")
        block = memory_store.actions_context_block("s1")
        assert block is not None
        assert "- word: Wrote plan.docx" in block
        assert "- final: Goal complete" in block

    def test_actions_context_block_none_when_empty(self, memdb):
        assert memory_store.actions_context_block("s1") is None

    def test_failed_action_marked_in_block(self, memdb):
        memory_store.log_action("s1", "tool", "ppt", "boom", ok=False)
        assert "(FAILED)" in memory_store.actions_context_block("s1")


# ===========================================================================
# 4. Store: context builders
# ===========================================================================

class TestContextBuilders:
    def test_history_prefix_none_for_fresh_session(self, memdb):
        assert memory_store.build_history_prefix("fresh") is None

    def test_history_prefix_includes_turns_and_actions(self, memdb):
        memory_store.add_message("s1", "user", "make a report")
        memory_store.add_message("s1", "assistant", "done")
        memory_store.log_action("s1", "tool", "excel", "Created report.xlsx")

        prefix = memory_store.build_history_prefix("s1")
        assert prefix is not None
        assert prefix[0]["role"] == "system"
        assert "Created report.xlsx" in prefix[0]["content"]
        assert {"role": "user", "content": "make a report"} in prefix

    def test_history_prefix_without_actions(self, memdb):
        memory_store.add_message("s1", "user", "hi")
        memory_store.log_action("s1", "tool", "t", "did a thing")
        prefix = memory_store.build_history_prefix("s1", include_actions=False)
        assert all(m["role"] != "system" for m in prefix)

    def test_transcript_none_for_fresh_session(self, memdb):
        assert memory_store.conversation_transcript("fresh") is None

    def test_transcript_format(self, memdb):
        memory_store.add_message("s1", "user", "q1")
        memory_store.add_message("s1", "assistant", "a1")
        t = memory_store.conversation_transcript("s1")
        assert t.startswith("RECENT CONVERSATION:")
        assert "user: q1" in t
        assert "assistant: a1" in t

    def test_build_llm_messages_fresh(self, memdb):
        msgs = memory_store.build_llm_messages("fresh", "hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_build_llm_messages_appends_new_turn(self, memdb):
        memory_store.add_message("s1", "user", "earlier")
        msgs = memory_store.build_llm_messages("s1", "new question")
        assert msgs[-1] == {"role": "user", "content": "new question"}
        assert {"role": "user", "content": "earlier"} in msgs


# ===========================================================================
# 5. Store: session management
# ===========================================================================

class TestSessionManagement:
    def test_clear_session_removes_messages_and_actions(self, memdb):
        memory_store.add_message("s1", "user", "hello")
        memory_store.log_action("s1", "tool", "t", "did")
        memory_store.add_message("s2", "user", "keep me")

        memory_store.clear_session("s1")
        assert memory_store.get_history("s1") == []
        assert memory_store.get_recent_actions("s1") == []
        assert len(memory_store.get_history("s2")) == 1

    def test_list_sessions(self, memdb):
        memory_store.add_message("s1", "user", "one")
        memory_store.add_message("s1", "user", "two")
        memory_store.add_message("s2", "user", "three")

        sessions = {s["session_id"]: s for s in memory_store.list_sessions()}
        assert sessions["s1"]["message_count"] == 2
        assert sessions["s2"]["message_count"] == 1


# ===========================================================================
# 6. Orchestration: memory injection + persistence
# ===========================================================================

class TestOrchestrationMemory:
    def test_turn_persisted_after_blocking_request(self, memdb, general_cap, fake_llm):
        from orchestration.service import handle_request
        result = handle_request("hello", session_id="s1")
        assert result.answer == "mem answer"

        history = memory_store.get_history("s1")
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "hello"
        assert history[1]["content"] == "mem answer"
        assert history[1]["capability"] == "general"

    def test_history_injected_on_followup(self, memdb, general_cap, fake_llm):
        from orchestration.service import handle_request
        handle_request("first question", session_id="s1")
        handle_request("follow up", session_id="s1")

        history = fake_llm["history"]
        assert history is not None
        assert {"role": "user", "content": "first question"} in history
        assert {"role": "assistant", "content": "mem answer"} in history

    def test_no_history_for_fresh_session(self, memdb, general_cap, fake_llm):
        from orchestration.service import handle_request
        handle_request("hello", session_id="brand-new")
        assert fake_llm["history"] is None

    def test_no_session_means_no_memory_calls(self, memdb, general_cap, fake_llm):
        from orchestration.service import handle_request
        handle_request("hello")  # no session_id
        assert fake_llm["history"] is None
        assert memory_store.list_sessions() == []

    def test_stream_persists_turn(self, memdb, general_cap, fake_llm):
        from orchestration.service import handle_request_stream
        events = list(handle_request_stream("stream question", session_id="s1"))

        assert events[0].startswith("data: ")
        assert events[-1] == "data: [DONE]\n\n"
        chunks = [e for e in events if '"chunk"' in e]
        assert len(chunks) == 2

        history = memory_store.get_history("s1")
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[1]["content"] == "mem answer"

    def test_memory_failure_does_not_break_answering(self, memdb, general_cap,
                                                     fake_llm, monkeypatch):
        from orchestration.service import handle_request

        def boom(*args, **kwargs):
            raise RuntimeError("sqlite exploded")

        monkeypatch.setattr(memory_store, "build_history_prefix", boom)
        monkeypatch.setattr(memory_store, "conversation_transcript", boom)
        monkeypatch.setattr(memory_store, "add_message", boom)

        result = handle_request("hello", session_id="s1")
        assert result.answer == "mem answer"


# ===========================================================================
# 7. HTTP endpoints
# ===========================================================================

class TestMemoryEndpoints:
    def test_ask_then_history_roundtrip(self, client, general_cap, fake_llm):
        resp = client.post("/api/ask", json={
            "message": "hello", "session_id": "web-1",
        })
        assert resp.status_code == 200
        assert resp.json()["answer"] == "mem answer"

        hist = client.get("/api/memory/history", params={"session_id": "web-1"})
        assert hist.status_code == 200
        data = hist.json()
        assert [m["role"] for m in data["messages"]] == ["user", "assistant"]

    def test_history_missing_session_id_rejected(self, client):
        resp = client.get("/api/memory/history")
        assert resp.status_code == 422  # FastAPI: required query param missing

    def test_sessions_lists_known_sessions(self, client, general_cap, fake_llm):
        client.post("/api/ask", json={"message": "hi", "session_id": "web-2"})
        resp = client.get("/api/memory/sessions")
        assert resp.status_code == 200
        ids = [s["session_id"] for s in resp.json()["sessions"]]
        assert "web-2" in ids

    def test_clear_wipes_session(self, client, general_cap, fake_llm):
        client.post("/api/ask", json={"message": "hi", "session_id": "web-3"})
        resp = client.post("/api/memory/clear", json={"session_id": "web-3"})
        assert resp.status_code == 200
        assert resp.json()["cleared"] is True

        hist = client.get("/api/memory/history", params={"session_id": "web-3"})
        assert hist.json()["messages"] == []
        assert hist.json()["actions"] == []

    def test_ask_without_session_id_backward_compat(self, client, general_cap, fake_llm):
        resp = client.post("/api/ask", json={"message": "old client"})
        assert resp.status_code == 200
        assert resp.json()["capability"] == "general"


# ===========================================================================
# 8. Planner stream helpers
# ===========================================================================

class TestPlannerStreamHelpers:
    def test_final_message_success_with_file(self):
        from api.planner_stream_api import _final_message
        msg = _final_message(
            "make a report", {"filename": "report.xlsx", "summary": "3 sheets"}, True, None
        )
        assert "Agent plan completed" in msg
        assert "report.xlsx" in msg

    def test_final_message_failure(self):
        from api.planner_stream_api import _final_message
        msg = _final_message("goal", None, False, "tool exploded")
        assert "Agent plan failed" in msg
        assert "tool exploded" in msg

    def test_remember_swallows_errors(self):
        from api.planner_stream_api import _remember

        def boom():
            raise RuntimeError("nope")

        _remember(boom)  # must not raise
