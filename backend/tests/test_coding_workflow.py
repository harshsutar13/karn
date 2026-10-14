"""
tests/test_coding_workflow.py

End-to-end offline tests for the PLAN -> EXECUTE -> TEST -> DEBUG coding
workflow (tools/coding_agent.py) and its HTTP layer (api/coding_agent_api.py).

The planner LLM is replaced with scripted JSON responses by monkeypatching
tools.planner._call_planner_llm, so these tests run with Ollama DOWN.
File operations run against the throwaway `project` fixture (tmp_path).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import coding_agent, planner, workspace
from tools.coding_agent import run_coding_task, stream_coding_task
from tools.workspace import CODING_TOOL_SPECS


# ---------------------------------------------------------------------------
# Scripted LLM material
# ---------------------------------------------------------------------------

BUGGY_MODULE = (
    "def is_palindrome(s):\n"
    '    cleaned = "".join(c.lower() for c in s if c.isalnum())\n'
    "    return cleaned != cleaned[::-1]\n"
)

TEST_MODULE = (
    "from tools.string_utils import is_palindrome\n"
    "\n"
    "def test_simple():\n"
    '    assert is_palindrome("racecar")\n'
    "\n"
    "def test_mixed():\n"
    '    assert is_palindrome("A man, a plan, a canal: Panama")\n'
    "\n"
    "def test_not_palindrome():\n"
    '    assert not is_palindrome("hello")\n'
)

FIX_PATCH = (
    "--- a/backend/tools/string_utils.py\n"
    "+++ b/backend/tools/string_utils.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def is_palindrome(s):\n"
    '     cleaned = "".join(c.lower() for c in s if c.isalnum())\n'
    "-    return cleaned != cleaned[::-1]\n"
    "+    return cleaned == cleaned[::-1]\n"
)

PLAN_1 = {
    "reasoning": "Create the module and tests, then run them.",
    "steps": [
        {"id": "step_1", "tool": "create_file",
         "args": {"path": "backend/tools/string_utils.py", "content": BUGGY_MODULE}},
        {"id": "step_2", "tool": "create_file",
         "args": {"path": "backend/tests/test_string_utils.py", "content": TEST_MODULE}},
        {"id": "step_3", "tool": "run_test",
         "args": {"target": "backend/tests/test_string_utils.py"}},
    ],
}

REPLAN_1 = {
    "reasoning": "The comparison is inverted; flip it and re-run the tests.",
    "steps": [
        {"id": "step_1", "tool": "apply_patch", "args": {"patch": FIX_PATCH}},
        {"id": "step_2", "tool": "run_test",
         "args": {"target": "backend/tests/test_string_utils.py"}},
    ],
}

REPLAN_STUB = {
    "reasoning": "Re-run the tests.",
    "steps": [
        {"id": "step_1", "tool": "run_test",
         "args": {"target": "backend/tests/test_string_utils.py"}},
    ],
}


def _scripted(responses):
    """Fake tools.planner._call_planner_llm returning canned JSON strings."""
    it = iter(responses)

    def fake(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("planner LLM called more times than scripted")

    return fake


def _events_by_type(result):
    evts = {}
    for e in result["events"]:
        evts.setdefault(e["type"], []).append(e)
    return evts


# ---------------------------------------------------------------------------
# Workflow: plan -> execute -> test -> debug -> success
# ---------------------------------------------------------------------------

class TestWorkflowSuccess:
    def test_success_after_one_debug_cycle(self, project, monkeypatch):
        monkeypatch.setattr(
            planner, "_call_planner_llm",
            _scripted([json.dumps(PLAN_1), json.dumps(REPLAN_1)]),
        )

        result = run_coding_task("Add is_palindrome to string_utils with tests")

        final = result["final"]
        assert final["ok"], result["events"]

        # exactly one debug cycle (initial plan + one replan)
        assert final["cycles"] == 1

        # the bug was actually fixed on disk
        fixed = (project / "backend" / "tools" / "string_utils.py").read_text(encoding="utf-8")
        assert "cleaned == cleaned[::-1]" in fixed

        # final test summary: all 3 tests pass
        tests = final["final_output"]["tests"]
        assert tests, final
        assert tests[-1]["passed"] == 3
        assert tests[-1]["failed"] == 0

        # event stream shape
        by_type = _events_by_type(result)
        assert len(by_type["plan"]) == 2          # initial + debug plan
        assert by_type["plan"][1].get("is_replan") is True
        assert by_type["replan"][0]["attempt"] == 1
        phases = [p["phase"] for p in by_type["phase"]]
        assert "snapshot" in phases and "verify" in phases and "debug" in phases
        for e in by_type["step_done"]:
            assert isinstance(e["ok"], bool)
            assert "elapsed_ms" in e and "summary" in e

        # working diff reflects the task
        out = final["final_output"]
        assert "backend/tools/string_utils.py" in out["files_added"]
        assert "backend/tests/test_string_utils.py" in out["files_added"]
        assert out["insertions"] >= 6

        # snapshot exists for later /api/coding/diff calls
        assert (project / ".agent_state" / "snapshot").is_dir()

    def test_unknown_tools_are_dropped(self, project, monkeypatch):
        plan = {
            "reasoning": "explore then create",
            "steps": [
                {"id": "step_1", "tool": "list_files", "args": {"path": "backend"}},
                {"id": "step_2", "tool": "invented_tool", "args": {}},
                {"id": "step_3", "tool": "create_file",
                 "args": {"path": "backend/tools/marker.py", "content": "M = 1\n"}},
            ],
        }
        monkeypatch.setattr(planner, "_call_planner_llm", _scripted([json.dumps(plan)]))
        result = run_coding_task("create a marker module")
        final = result["final"]
        assert final["ok"], result["events"]
        assert (project / "backend" / "tools" / "marker.py").exists()

    def test_clean_plan_no_debug_needed(self, project, monkeypatch):
        plan = {
            "reasoning": "Just create a file, no tests required.",
            "steps": [
                {"id": "step_1", "tool": "create_file",
                 "args": {"path": "backend/tools/plain.py", "content": "P = 7\n"}},
                {"id": "step_2", "tool": "get_diff", "args": {}},
            ],
        }
        monkeypatch.setattr(planner, "_call_planner_llm", _scripted([json.dumps(plan)]))
        result = run_coding_task("create plain.py")
        final = result["final"]
        assert final["ok"], result["events"]
        assert final["cycles"] == 0
        by_type = _events_by_type(result)
        assert "replan" not in by_type


# ---------------------------------------------------------------------------
# Workflow: bounded debug loop
# ---------------------------------------------------------------------------

class TestWorkflowFailure:
    def test_max_debug_cycles(self, project, monkeypatch):
        responses = [json.dumps(PLAN_1)] + [json.dumps(REPLAN_STUB)] * 3
        monkeypatch.setattr(planner, "_call_planner_llm", _scripted(responses))

        result = run_coding_task("Add is_palindrome to string_utils with tests")

        final = result["final"]
        assert not final["ok"]
        assert "Max debug cycles" in final["error"]
        assert final["cycles"] == 3

    def test_planner_unreachable(self, project, monkeypatch):
        def boom(prompt):
            raise ConnectionError("Cannot reach Ollama at http://localhost:11434")

        monkeypatch.setattr(planner, "_call_planner_llm", boom)
        result = run_coding_task("anything")

        final = result["final"]
        assert not final["ok"]
        assert "Planning failed" in final["error"]
        by_type = _events_by_type(result)
        assert by_type["error"][0]["text"].startswith("Planning failed")
        # nothing was written
        assert not (project / "backend" / "tools" / "string_utils.py").exists()

    def test_malformed_plan_json(self, project, monkeypatch):
        monkeypatch.setattr(planner, "_call_planner_llm",
                            _scripted(["this is not json at all"]))
        result = run_coding_task("anything")
        assert not result["final"]["ok"]
        assert "Planning failed" in result["final"]["error"]

    def test_empty_task_rejected(self, project):
        result = run_coding_task("   ")
        assert not result["final"]["ok"]


# ---------------------------------------------------------------------------
# Debug-loop diagnostics quality
# ---------------------------------------------------------------------------

class TestDebugDiagnostics:
    def test_replan_prompt_contains_failure_and_diff(self, project, monkeypatch):
        captured = []

        def fake(prompt: str) -> str:
            captured.append(prompt)
            if len(captured) == 1:
                return json.dumps(PLAN_1)
            return json.dumps(REPLAN_1)

        monkeypatch.setattr(planner, "_call_planner_llm", fake)
        result = run_coding_task("Add is_palindrome to string_utils with tests")
        assert result["final"]["ok"], result["events"]

        assert len(captured) == 2
        replan_prompt = captured[1]
        # the debug prompt carries the failing test evidence ...
        assert "test_simple" in replan_prompt or "test_mixed" in replan_prompt
        assert "failed" in replan_prompt
        # ... and the working diff
        assert "+++ b/backend/tools/string_utils.py" in replan_prompt


# ---------------------------------------------------------------------------
# HTTP layer (fully offline)
# ---------------------------------------------------------------------------

class TestCodingAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    def test_tools_endpoint(self, client):
        r = client.get("/api/coding/tools")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == len(CODING_TOOL_SPECS)
        assert set(data["tools"]) == set(CODING_TOOL_SPECS)

    def test_stream_endpoint_end_to_end(self, client, project, monkeypatch):
        monkeypatch.setattr(
            planner, "_call_planner_llm",
            _scripted([json.dumps(PLAN_1), json.dumps(REPLAN_1)]),
        )
        r = client.post(
            "/api/coding/plan-and-execute/stream",
            json={"task": "Add is_palindrome to string_utils with tests"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        events = []
        done_seen = False
        for line in r.text.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload == "[DONE]":
                done_seen = True
                continue
            events.append(json.loads(payload))

        assert done_seen
        finals = [e for e in events if e.get("type") == "final"]
        assert len(finals) == 1
        assert finals[0]["ok"] is True
        assert finals[0]["cycles"] == 1
        types = [e["type"] for e in events]
        assert "step_start" in types and "step_done" in types
        assert "phase" in types

    def test_sync_endpoint(self, client, project, monkeypatch):
        monkeypatch.setattr(planner, "_call_planner_llm",
                            _scripted([json.dumps(PLAN_1), json.dumps(REPLAN_1)]))
        r = client.post(
            "/api/coding/plan-and-execute",
            json={"task": "Add is_palindrome to string_utils with tests"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["cycles"] == 1

    def test_diff_endpoint_after_task(self, client, project, monkeypatch):
        monkeypatch.setattr(
            planner, "_call_planner_llm",
            _scripted([json.dumps(PLAN_1), json.dumps(REPLAN_1)]),
        )
        client.post("/api/coding/plan-and-execute",
                    json={"task": "Add is_palindrome to string_utils with tests"})

        r = client.get("/api/coding/diff")
        assert r.status_code == 200
        data = r.json()
        assert "backend/tools/string_utils.py" in data["files_added"]
        assert "+++" in data["diff"]

    def test_diff_endpoint_no_snapshot(self, client, project):
        r = client.get("/api/coding/diff")
        assert r.status_code == 404
        assert "snapshot" in r.json()["detail"]

    def test_empty_task_rejected(self, client):
        for url in ("/api/coding/plan-and-execute", "/api/coding/plan-and-execute/stream"):
            r = client.post(url, json={"task": "  "})
            assert r.status_code == 400
