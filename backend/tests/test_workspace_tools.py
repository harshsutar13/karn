"""
tests/test_workspace_tools.py

Offline unit tests for tools/workspace.py — the custom coding tools of the
agentic coding environment. No Ollama / Qdrant / network needed.

All file operations run against the throwaway `project` fixture (tmp_path),
via monkeypatched tools.workspace.PROJECT_ROOT.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import workspace
from tools.workspace import (
    CODING_TOOL_REGISTRY,
    CODING_TOOL_SPECS,
    take_snapshot,
    tool_apply_patch,
    tool_create_file,
    tool_get_diff,
    tool_list_files,
    tool_read_file,
    tool_run_command,
    tool_run_test,
    tool_search_code,
)


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_matches_specs(self):
        assert set(CODING_TOOL_REGISTRY) == set(CODING_TOOL_SPECS)
        for name, spec in CODING_TOOL_SPECS.items():
            assert spec["description"], name
            assert "args" in spec, name

    def test_requested_tools_present(self):
        expected = {
            "list_files", "read_file", "search_code", "create_file",
            "apply_patch", "run_test", "run_command", "get_diff",
        }
        assert expected == set(CODING_TOOL_REGISTRY)


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

class TestListFiles:
    def test_lists_tree(self, project):
        res = tool_list_files({"path": "backend", "depth": 2}, {})
        assert res["ok"], res
        paths = [e["path"] for e in res["output"]["entries"]]
        assert "backend/tools" in paths
        assert "backend/tools/app.py" in paths
        assert "backend/tests/conftest.py" in paths

    def test_prunes_noise_dirs(self, project):
        res = tool_list_files({"path": "backend", "depth": 3}, {})
        assert res["ok"]
        paths = [e["path"] for e in res["output"]["entries"]]
        assert "__pycache__" not in paths
        assert not any("pycache" in p for p in paths)

    def test_root_listing_default_path(self, project):
        res = tool_list_files({}, {})
        assert res["ok"]
        paths = [e["path"] for e in res["output"]["entries"]]
        assert "README.md" in paths
        assert "frontend" in paths
        assert "logs" not in paths  # excluded root dir

    def test_pattern_filter(self, project):
        res = tool_list_files({"path": "backend", "depth": 2, "pattern": "*.py"}, {})
        assert res["ok"]
        entries = res["output"]["entries"]
        assert all(e["path"].endswith(".py") for e in entries)
        assert "backend/tools/app.py" in [e["path"] for e in entries]

    def test_rejects_outside_root(self, project):
        res = tool_list_files({"path": "../elsewhere"}, {})
        assert not res["ok"]
        assert "not allowed" in res["error"] or "cannot resolve" in res["error"]

    def test_missing_dir(self, project):
        res = tool_list_files({"path": "nope"}, {})
        assert not res["ok"]


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_numbered_lines(self, project):
        res = tool_read_file({"path": "backend/tools/app.py"}, {})
        assert res["ok"], res
        out = res["output"]
        assert out["total_lines"] == 2
        assert out["text"].splitlines()[0] == "1\tdef add(a, b):"
        assert out["text"].splitlines()[1] == "2\t    return a + b"

    def test_line_range(self, project):
        res = tool_read_file({"path": "backend/tools/app.py", "start_line": 2, "end_line": 2}, {})
        assert res["ok"]
        assert res["output"]["text"] == "2\t    return a + b"
        assert res["output"]["start_line"] == 2
        assert res["output"]["end_line"] == 2

    def test_backend_relative_and_abs_paths(self, project):
        rel = tool_read_file({"path": "tools/app.py"}, {})
        assert rel["ok"]
        abs_path = os.path.join(str(project), "backend", "tools", "app.py")
        absl = tool_read_file({"path": abs_path}, {})
        assert absl["ok"]
        assert absl["output"]["text"] == rel["output"]["text"]

    def test_missing_file(self, project):
        res = tool_read_file({"path": "backend/tools/nope.py"}, {})
        assert not res["ok"]
        assert "not found" in res["error"]

    def test_outside_root_rejected(self, project):
        res = tool_read_file({"path": str(project.parent / "some_file.txt")}, {})
        assert not res["ok"]
        assert "not allowed" in res["error"]

    def test_excluded_dir_rejected(self, project):
        res = tool_read_file({"path": "backend/__pycache__/app.cpython-310.pyc"}, {})
        assert not res["ok"]
        assert "not allowed" in res["error"]

    def test_start_beyond_eof(self, project):
        res = tool_read_file({"path": "backend/tools/app.py", "start_line": 99}, {})
        assert not res["ok"]
        assert "beyond end of file" in res["error"]


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------

class TestSearchCode:
    def test_literal_search(self, project):
        res = tool_search_code({"query": "return a + b"}, {})
        assert res["ok"], res
        matches = res["output"]["matches"]
        assert len(matches) == 1
        assert matches[0]["path"] == "backend/tools/app.py"
        assert matches[0]["line"] == 2

    def test_case_insensitive_default(self, project):
        res = tool_search_code({"query": "HELLO WORLD"}, {})
        assert res["ok"]
        assert res["output"]["count"] == 1

    def test_regex_search_with_glob(self, project):
        res = tool_search_code({"query": r"def \w+\(", "regex": True, "glob": "*.py"}, {})
        assert res["ok"]
        assert res["output"]["count"] == 1
        assert res["output"]["matches"][0]["path"] == "backend/tools/app.py"

    def test_scope_limits_results(self, project):
        res = tool_search_code({"query": "hello", "path": "frontend"}, {})
        assert res["ok"]
        assert res["output"]["count"] == 1
        assert res["output"]["matches"][0]["path"] == "frontend/index.html"

    def test_skips_excluded_dirs(self, project):
        res = tool_search_code({"query": "noise"}, {})
        assert res["ok"]
        assert res["output"]["count"] == 0  # logs/ is excluded

    def test_context_lines(self, project):
        res = tool_search_code({"query": "return a + b", "context": 1}, {})
        assert res["ok"]
        assert "def add" in res["output"]["matches"][0]["context"]

    def test_no_results(self, project):
        res = tool_search_code({"query": "zzz_not_there_zzz"}, {})
        assert res["ok"]
        assert res["output"]["count"] == 0

    def test_invalid_regex(self, project):
        res = tool_search_code({"query": "([", "regex": True}, {})
        assert not res["ok"]
        assert "invalid regex" in res["error"]


# ---------------------------------------------------------------------------
# create_file
# ---------------------------------------------------------------------------

class TestCreateFile:
    def test_creates_new_file(self, project):
        content = "def square(x):\n    return x * x\n"
        res = tool_create_file(
            {"path": "backend/tools/mathx.py", "content": content}, {})
        assert res["ok"], res
        assert res["output"]["lines"] == 2
        with open(os.path.join(str(project), "backend", "tools", "mathx.py"),
                  encoding="utf-8") as f:
            assert f.read() == content

    def test_refuses_existing_file(self, project):
        res = tool_create_file(
            {"path": "backend/tools/app.py", "content": "x"}, {})
        assert not res["ok"]
        assert "already exists" in res["error"]
        # original untouched
        res2 = tool_read_file({"path": "backend/tools/app.py"}, {})
        assert res2["output"]["total_lines"] == 2

    def test_refuses_outside_root(self, project):
        res = tool_create_file({"path": "../evil.py", "content": "x"}, {})
        assert not res["ok"]
        assert "not allowed" in res["error"]

    def test_creates_missing_dirs(self, project):
        res = tool_create_file(
            {"path": "backend/tools/nested/deep/mod.py", "content": "X = 1\n"}, {})
        assert res["ok"]

    def test_requires_content(self, project):
        res = tool_create_file({"path": "backend/tools/x.py"}, {})
        assert not res["ok"]


# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------

APP_PATCH = """--- a/backend/tools/app.py
+++ b/backend/tools/app.py
@@ -1,2 +1,6 @@
 def add(a, b):
     return a + b
+
+
+def sub(a, b):
+    return a - b
"""


class TestApplyPatch:
    def test_basic_patch(self, project):
        res = tool_apply_patch({"patch": APP_PATCH}, {})
        assert res["ok"], res
        assert res["output"]["files"] == ["backend/tools/app.py"]
        assert res["output"]["hunks_applied"] == 1
        text = open(os.path.join(str(project), "backend", "tools", "app.py"),
                    encoding="utf-8").read()
        assert "def sub(a, b):" in text
        assert "def add(a, b):\n    return a + b" in text

    def test_multihunk_patch(self, project):
        tool_apply_patch({"patch": APP_PATCH}, {})
        patch2 = (
            "--- a/backend/tools/app.py\n"
            "+++ b/backend/tools/app.py\n"
            "@@ -1,4 +1,4 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 0\n"
            "@@ -5,6 +5,6 @@\n"
            " def sub(a, b):\n"
            "-    return a - b\n"
            "+    return a - b - 0\n"
        )
        res = tool_apply_patch({"patch": patch2}, {})
        assert res["ok"], res
        text = open(os.path.join(str(project), "backend", "tools", "app.py"),
                    encoding="utf-8").read()
        assert "return a + b + 0" in text
        assert "return a - b - 0" in text

    def test_mismatch_returns_actual_content(self, project):
        bad = (
            "--- a/backend/tools/app.py\n"
            "+++ b/backend/tools/app.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def totally_different_function():\n"
            "-    return nothing\n"
            "+    return something\n"
        )
        res = tool_apply_patch({"patch": bad}, {})
        assert not res["ok"]
        err = res["error"]
        assert "did not match" in err
        assert "ACTUAL FILE CONTENT" in err
        assert "def add(a, b):" in err  # ground truth for the debug loop
        # file untouched
        res2 = tool_read_file({"path": "backend/tools/app.py"}, {})
        assert res2["output"]["total_lines"] == 2

    def test_whitespace_fuzz_match(self, project):
        # write a file whose lines carry trailing spaces
        with open(os.path.join(str(project), "backend", "tools", "app.py"),
                  "w", encoding="utf-8") as f:
            f.write("def add(a, b):   \n    return a + b   \n")
        patch = (
            "--- a/backend/tools/app.py\n"
            "+++ b/backend/tools/app.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b\n"
        )
        res = tool_apply_patch({"patch": patch}, {})
        assert res["ok"], res

    def test_new_file_via_dev_null(self, project):
        patch = (
            "--- /dev/null\n"
            "+++ b/backend/tools/fresh.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+A = 1\n"
            "+B = 2\n"
        )
        res = tool_apply_patch({"patch": patch}, {})
        assert res["ok"], res
        text = open(os.path.join(str(project), "backend", "tools", "fresh.py"),
                    encoding="utf-8").read()
        assert text == "A = 1\nB = 2\n"

    def test_crlf_file_preserved(self, project):
        path = os.path.join(str(project), "backend", "tools", "app.py")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("def add(a, b):\r\n    return a + b\r\n")
        patch = (
            "--- a/backend/tools/app.py\n"
            "+++ b/backend/tools/app.py\n"
            "@@ -1,2 +1,6 @@\n"
            " def add(a, b):\n"
            "     return a + b\n"
            "+\n"
            "+def mul(a, b):\n"
            "+    return a * b\n"
        )
        res = tool_apply_patch({"patch": patch}, {})
        assert res["ok"], res
        raw = open(path, "rb").read().decode("utf-8")
        assert "\r\n" in raw
        assert "def mul" in raw

    def test_lenient_undercounted_hunk(self, project):
        # LLMs routinely emit @@ counts that undercount the body.
        # The parser must keep consuming explicit +/-/space lines.
        patch = (
            "--- a/backend/tools/app.py\n"
            "+++ b/backend/tools/app.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def add(a, b):\n"
            "     return a + b\n"
            "+\n"
            "+def mul(a, b):\n"
            "+    return a * b\n"
        )
        res = tool_apply_patch({"patch": patch}, {})
        assert res["ok"], res
        text = open(os.path.join(str(project), "backend", "tools", "app.py"),
                    encoding="utf-8").read()
        assert "def mul(a, b):" in text
        assert "return a * b" in text

    def test_blank_line_ends_completed_hunk(self, project):
        # Two file sections separated by a blank line: the blank must NOT
        # be swallowed as context of the first hunk.
        patch = (
            "--- a/backend/tools/app.py\n"
            "+++ b/backend/tools/app.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 1\n"
            "\n"
            "--- a/backend/tools/helpers.py\n"
            "+++ b/backend/tools/helpers.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 42\n"
            "+VALUE = 43\n"
        )
        res = tool_apply_patch({"patch": patch}, {})
        assert res["ok"], res
        app = open(os.path.join(str(project), "backend", "tools", "app.py"),
                   encoding="utf-8").read()
        helpers = open(os.path.join(str(project), "backend", "tools", "helpers.py"),
                       encoding="utf-8").read()
        assert "return a + b + 1" in app
        assert "VALUE = 43" in helpers

    def test_requires_hunks(self, project):
        res = tool_apply_patch({"patch": "just some text\nno diff here"}, {})
        assert not res["ok"]
        assert "no hunks" in res["error"]

    def test_outside_root_rejected(self, project):
        patch = "--- a/../evil.py\n+++ b/../evil.py\n@@ -1 +1 @@\n-a\n+b\n"
        res = tool_apply_patch({"patch": patch}, {})
        assert not res["ok"]
        assert "not allowed" in res["error"]

    def test_missing_file(self, project):
        patch = "--- a/backend/tools/ghost.py\n+++ b/backend/tools/ghost.py\n@@ -1 +1 @@\n-a\n+b\n"
        res = tool_apply_patch({"patch": patch}, {})
        assert not res["ok"]
        assert "not found" in res["error"]


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------

class TestRunTest:
    def test_passing_target(self, project):
        tool_create_file({
            "path": "backend/tests/test_ok.py",
            "content": "def test_ok():\n    assert 1 + 1 == 2\n",
        }, {})
        res = tool_run_test({"target": "backend/tests/test_ok.py"}, {})
        assert res["ok"], res
        out = res["output"]
        assert out["passed"] == 1
        assert out["failed"] == 0
        assert out["failures"] == []
        assert out["target"] == "backend/tests/test_ok.py"

    def test_failing_target(self, project):
        tool_create_file({
            "path": "backend/tests/test_bad.py",
            "content": "def test_bad():\n    assert 1 + 1 == 3\n",
        }, {})
        res = tool_run_test({"target": "backend/tests/test_bad.py"}, {})
        assert not res["ok"]
        out = res["output"]
        assert out["failed"] == 1
        assert out["passed"] == 0
        assert any("test_bad" in f for f in out["failures"])
        assert "assert" in out["stdout_tail"]

    def test_missing_target(self, project):
        res = tool_run_test({"target": "backend/tests/test_missing.py"}, {})
        assert not res["ok"]

    def test_requires_target(self, project):
        res = tool_run_test({}, {})
        assert not res["ok"]
        assert "requires" in res["error"]


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_py_compile_allowed(self, project):
        res = tool_run_command(
            {"command": "python -m py_compile backend/tools/app.py"}, {})
        assert res["ok"], res

    def test_pytest_allowed(self, project):
        tool_create_file({
            "path": "backend/tests/test_cmd.py",
            "content": "def test_cmd():\n    assert True\n",
        }, {})
        res = tool_run_command(
            {"command": "python -m pytest backend/tests/test_cmd.py -q"}, {})
        assert res["ok"], res

    def test_disallowed_program(self, project):
        res = tool_run_command({"command": "rm -rf everything"}, {})
        assert not res["ok"]
        assert "allowlist" in res["error"]

    def test_disallowed_python_module(self, project):
        res = tool_run_command({"command": "python -c 'print(1)'"}, {})
        assert not res["ok"]
        assert "python -m" in res["error"]

    def test_shell_metacharacters_rejected(self, project):
        for cmd in ("ls | wc", "cat x > y", "echo a; echo b", "ls && dir"):
            res = tool_run_command({"command": cmd}, {})
            assert not res["ok"], cmd
            assert "disallowed shell character" in res["error"]

    def test_ls_allowed(self, project):
        res = tool_run_command({"command": "ls backend/tools"}, {})
        assert res["ok"], res
        assert "app.py" in res["output"]["stdout"]

    def test_requires_command(self, project):
        res = tool_run_command({}, {})
        assert not res["ok"]


# ---------------------------------------------------------------------------
# snapshot & get_diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_no_snapshot_error(self, project):
        res = tool_get_diff({}, {})
        assert not res["ok"]
        assert "no snapshot" in res["error"]

    def test_snapshot_and_diff(self, project):
        snap = take_snapshot()
        assert snap["files"] >= 4

        # modify, add, delete
        tool_apply_patch({"patch": APP_PATCH}, {})
        tool_create_file({"path": "backend/tools/new_mod.py", "content": "X = 1\n"}, {})
        os.remove(os.path.join(str(project), "backend", "tools", "helpers.py"))

        res = tool_get_diff({}, {})
        assert res["ok"], res
        out = res["output"]
        assert out["files_changed"] == ["backend/tools/app.py"]
        assert out["files_added"] == ["backend/tools/new_mod.py"]
        assert out["files_deleted"] == ["backend/tools/helpers.py"]
        assert out["insertions"] >= 4
        assert out["deletions"] >= 1
        assert "--- a/backend/tools/app.py" in out["diff"]
        assert "+++ b/backend/tools/app.py" in out["diff"]
        assert "+def sub(a, b):" in out["diff"]
        assert "-VALUE = 42" in out["diff"]

    def test_clean_tree_empty_diff(self, project):
        take_snapshot()
        res = tool_get_diff({}, {})
        assert res["ok"]
        assert res["output"]["diff"] == ""
        assert res["output"]["files_changed"] == []
