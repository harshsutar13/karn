"""Regression tests for simple Office-document planner requests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.planner import Plan, make_plan, replan
from tools import executor


def test_hello_world_word_request_uses_docx_writer_without_sandbox(monkeypatch):
    """A basic Word request must never need model-generated Python."""
    monkeypatch.setattr(
        "tools.planner._call_planner_llm",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    plan = make_plan("Create Word file with 'Hello World' text and big heading.")

    assert [step["tool"] for step in plan.steps] == ["write_docx"]
    assert plan.steps[0]["args"]["title"] == "Hello World"
    assert plan.steps[0]["args"]["output_filename"] == "hello_world.docx"


def test_generic_word_request_is_one_docx_step_without_llm(monkeypatch):
    monkeypatch.setattr(
        "tools.planner._call_planner_llm",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    plan = make_plan("Create Word file with given text.")

    assert [step["tool"] for step in plan.steps] == ["write_docx"]
    assert plan.steps[0]["args"]["sections"] == [
        {"heading": "given text", "body": "given text"}
    ]


def test_simple_word_replan_stays_on_the_docx_writer(monkeypatch):
    monkeypatch.setattr(
        "tools.planner._call_planner_llm",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    plan = replan(
        "Create Word file with given text.",
        Plan(reasoning="bad", steps=[]),
        "write_docx sections must be a non-empty list",
    )

    assert [step["tool"] for step in plan.steps] == ["write_docx"]


def test_write_file_without_path_saves_a_managed_text_deliverable(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(executor, "_ALLOWED_FILE_DIRS", [str(tmp_path)])

    result = executor.tool_write_file({"content": "Hello World"}, {})

    assert result["ok"]
    output = result["output"]
    assert output["filename"].endswith(".txt")
    assert (tmp_path / output["filename"]).read_text(encoding="utf-8") == "Hello World"
