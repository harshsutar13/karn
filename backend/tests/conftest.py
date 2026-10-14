"""conftest.py — adds the backend directory to sys.path for all tests.

Also provides the `project` fixture used by the coding-agent tests: a tiny
throwaway project tree (tmp_path) that the workspace tools operate on by
monkeypatching tools.workspace.PROJECT_ROOT / BACKEND_ROOT. This keeps the
tests fully offline and side-effect free on the real repository.
"""
import sys
import os

# backend/ is the parent of the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tools import workspace


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Build a minimal project tree and point the workspace tools at it."""
    root = str(tmp_path)

    _write(os.path.join(root, "backend", "tools", "app.py"),
           "def add(a, b):\n"
           "    return a + b\n")
    _write(os.path.join(root, "backend", "tools", "helpers.py"),
           "VALUE = 42\n")
    _write(os.path.join(root, "frontend", "index.html"),
           "<html>\n<p>hello world</p>\n</html>\n")
    _write(os.path.join(root, "README.md"), "# demo project\n")

    # conftest for the throwaway tests, mirroring the real backend/tests one
    _write(os.path.join(root, "backend", "tests", "conftest.py"),
           "import sys, os\n"
           "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))\n")

    # noise that every tool must prune
    os.makedirs(os.path.join(root, "backend", "__pycache__"), exist_ok=True)
    _write(os.path.join(root, "backend", "__pycache__", "app.cpython-310.pyc"), "junk")
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    _write(os.path.join(root, "logs", "api.log"), "noise")

    monkeypatch.setattr(workspace, "PROJECT_ROOT", root)
    monkeypatch.setattr(workspace, "BACKEND_ROOT", os.path.join(root, "backend"))
    return tmp_path
