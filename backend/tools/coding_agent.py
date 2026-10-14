"""
tools/coding_agent.py

Offline agentic CODING workflow: PLAN -> EXECUTE -> TEST -> DEBUG.

Loop:
    1. SNAPSHOT the source tree (so get_diff can show the working diff;
       the project is not a git repository).
    2. PLAN: the local LLM emits a JSON plan of coding-tool steps
       (list_files / read_file / search_code / create_file / apply_patch /
        run_test / run_command / get_diff).
    3. EXECUTE the steps sequentially; later steps may reference earlier
       outputs via {"ref": "step_N"}.
    4. TEST/VERIFY: results of any run_test steps are collected, and every
       .py file the plan touched is automatically syntax-checked with
       py_compile.
    5. DEBUG: on failure, the LLM receives the failing tests, tracebacks and
       the current working diff, and produces a corrective plan. Bounded by
       MAX_DEBUG_CYCLES.

Emits the same event vocabulary as api/planner_stream_api.py
(planning / plan / step_start / step_done / replan / final) plus a
"phase" event, so the frontend can reuse its plan renderer.

All LLM calls go through tools.planner._call_planner_llm, which tests can
monkeypatch to run fully offline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Generator

from tools import planner
from tools import workspace
from tools.executor import _resolve_args
from tools.workspace import (
    CODING_TOOL_REGISTRY,
    CODING_TOOL_SPECS,
    take_snapshot,
    tool_get_diff,
    tool_run_command,
)

logger = logging.getLogger(__name__)

MAX_DEBUG_CYCLES = 3
MAX_STEPS_PER_PLAN = 8
_DEBUG_DIFF_CHARS = 6000


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are the PLANNER of an OFFLINE coding agent working inside a local project.

Given a TASK, output a JSON object with TWO keys:
"reasoning" (short string) and "steps" (list).
Each step has EXACTLY these keys: "id", "tool", "args".

TOOLS YOU MAY USE:
{tools}

STRICT RULES:
1. Output ONLY the JSON object. No markdown fences, no text before/after.
2. "tool" MUST be one of the tools above. Do not invent tools.
3. EXPLORE FIRST: when the task touches existing code, start with
   list_files / search_code / read_file before editing anything.
4. Create NEW files with create_file. Modify EXISTING files ONLY with
   apply_patch, and only after reading them with read_file.
5. If the plan changes code, it MUST include a run_test step (write a test
   file first with create_file if no test exists), followed by get_diff.
6. Keep the plan to 2-6 steps. Use "step_1", "step_2" strings to reference
   earlier outputs.
7. All string values must be wrapped in double quotes.

EXAMPLE for task "add a function is_palindrome to the utils module with tests":
{
  "reasoning": "Find the utils module, read it, append the function, add tests, run them.",
  "steps": [
    {"id": "step_1", "tool": "search_code", "args": {"query": "def is_palindrome", "glob": "*.py"}},
    {"id": "step_2", "tool": "read_file", "args": {"path": "backend/tools/utils.py"}},
    {"id": "step_3", "tool": "apply_patch", "args": {"patch": "--- a/backend/tools/utils.py\\n+++ b/backend/tools/utils.py\\n@@ ..."}},
    {"id": "step_4", "tool": "create_file", "args": {"path": "backend/tests/test_utils.py", "content": "..."}},
    {"id": "step_5", "tool": "run_test", "args": {"target": "backend/tests/test_utils.py"}},
    {"id": "step_6", "tool": "get_diff", "args": {}}
  ]
}

TASK:
{task}

YOUR JSON:
"""

_REPLAN_SYSTEM = """\
You are the PLANNER of an OFFLINE coding agent.

The previous plan FAILED. Output a REVISED JSON object with keys
"reasoning" (string) and "steps" (list of {{"id", "tool", "args"}}).

TOOLS:
{tools}

TASK:
{task}

PREVIOUS PLAN:
{prev_plan}

WHAT FAILED:
{failure}

CURRENT WORKING DIFF (changes made so far):
{diff}

RULES:
1. Output ONLY JSON. No fences, no preamble.
2. FIX the root cause. Prefer apply_patch with accurate context read from
   the failure details; re-read files with read_file if unsure.
3. Include a run_test step that re-runs the failing tests, then get_diff.
4. Keep it to 1-4 steps.
5. All string values in double quotes.

YOUR JSON:
"""


# ---------------------------------------------------------------------------
# Planning (LLM-backed, monkeypatchable via tools.planner)
# ---------------------------------------------------------------------------

def _tools_block() -> str:
    lines = []
    for name, spec in CODING_TOOL_SPECS.items():
        desc = spec["description"].split(".")[0]
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _clean_steps(steps: list) -> list[dict]:
    cleaned = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        tool = s.get("tool")
        if tool not in CODING_TOOL_REGISTRY:
            logger.warning("Coding planner used unknown tool %r; dropping step", tool)
            continue
        cleaned.append({
            "id": s.get("id") or f"step_{i + 1}",
            "tool": tool,
            "args": s.get("args") or {},
        })
    return cleaned[:MAX_STEPS_PER_PLAN]


def make_coding_plan(task: str) -> dict:
    """Ask the LLM for a coding plan. Returns {"reasoning", "steps"}."""
    prompt = _PLANNER_SYSTEM.replace("{tools}", _tools_block()).replace("{task}", task)
    raw = planner._call_planner_llm(prompt)
    obj = planner._extract_json_obj(raw)
    if not obj or "steps" not in obj:
        raise ValueError(f"planner returned malformed JSON: {raw[:300]}")
    steps = _clean_steps(obj.get("steps") or [])
    if not steps:
        raise ValueError(f"planner produced no usable steps: {raw[:300]}")
    return {"reasoning": obj.get("reasoning", ""), "steps": steps, "raw": raw}


def replan_coding(task: str, prev_plan: dict, failure: str, diff: str) -> dict:
    """Ask the LLM for a corrective plan after a failure."""
    prompt = (
        _REPLAN_SYSTEM
        .replace("{tools}", _tools_block())
        .replace("{task}", task)
        .replace("{prev_plan}", json.dumps(
            {"reasoning": prev_plan.get("reasoning", ""),
             "steps": prev_plan.get("steps", [])}, indent=2))
        .replace("{failure}", failure[:3000])
        .replace("{diff}", diff[:_DEBUG_DIFF_CHARS] or "(no changes yet)")
    )
    raw = planner._call_planner_llm(prompt)
    obj = planner._extract_json_obj(raw)
    if not obj or "steps" not in obj:
        raise ValueError(f"re-planner returned malformed JSON: {raw[:300]}")
    steps = _clean_steps(obj.get("steps") or [])
    if not steps:
        raise ValueError(f"re-planner produced no usable steps: {raw[:300]}")
    return {"reasoning": obj.get("reasoning", ""), "steps": steps, "raw": raw}


# ---------------------------------------------------------------------------
# Step execution helpers
# ---------------------------------------------------------------------------

def _step_label(tool: str) -> str:
    return {
        "list_files":  "Exploring the workspace",
        "read_file":   "Reading file",
        "search_code": "Searching code",
        "create_file": "Creating file",
        "apply_patch": "Applying patch",
        "run_test":    "Running tests",
        "run_command": "Running command",
        "get_diff":    "Reviewing working diff",
    }.get(tool, tool)


def _short_summary(tool: str, output) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output[:120]
    if not isinstance(output, dict):
        return str(output)[:120]
    if tool == "list_files":
        return f"{output.get('count', 0)} entries"
    if tool == "read_file":
        return (f"{output.get('total_lines', 0)} lines · showing "
                f"{output.get('start_line')}-{output.get('end_line')}")
    if tool == "search_code":
        return f"{output.get('count', 0)} matches in {output.get('files_matched', 0)} files"
    if tool == "create_file":
        return f"created {output.get('rel_path', '')} ({output.get('lines', 0)} lines)"
    if tool == "apply_patch":
        return f"patched {', '.join(output.get('files', []))}"
    if tool == "run_test":
        return (f"{output.get('passed', 0)} passed, {output.get('failed', 0)} failed"
                if output.get("summary") is not None else "ran")
    if tool == "run_command":
        return f"exit {output.get('exit_code')}"
    if tool == "get_diff":
        return (f"+{output.get('insertions', 0)} -{output.get('deletions', 0)} "
                f"({len(output.get('files_changed', []))} changed, "
                f"{len(output.get('files_added', []))} added)")
    for v in output.values():
        if isinstance(v, str):
            return v[:120]
    return "done"


def _touched_py_files(records: list[dict]) -> list[str]:
    """Project-relative .py paths created/patched by executed steps."""
    files: list[str] = []
    for rec in records:
        tool = rec.get("tool")
        out = rec.get("output")
        if not isinstance(out, dict):
            continue
        if tool == "create_file" and out.get("rel_path"):
            files.append(out["rel_path"])
        elif tool == "apply_patch":
            for f in out.get("files", []):
                files.append(str(f).replace("\\", "/"))
    return sorted({f for f in files if f.endswith(".py")})


def _syntax_check(py_files: list[str]) -> dict:
    """py_compile every touched .py file. Returns a tool-result dict."""
    root = workspace.PROJECT_ROOT
    existing = [f for f in py_files if os.path.isfile(os.path.join(root, *f.split("/")))]
    if not existing:
        return {"ok": True, "output": {"checked": 0, "note": "no touched .py files"}}
    cmd = "python -m py_compile " + " ".join(existing)
    res = tool_run_command({"command": cmd, "timeout": 60}, {})
    out = res.get("output") or {}
    out.setdefault("checked", len(existing))
    return {"ok": res.get("ok", False), "output": out, "error": res.get("error")}


def _collect_test_results(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("tool") == "run_test" and r.get("ok") is not None]


def _failure_reason(records: list[dict], syntax: dict | None) -> str:
    parts: list[str] = []
    for rec in records:
        if rec.get("ok") is False and rec.get("tool") != "run_test":
            parts.append(f"step {rec.get('id')} ({rec.get('tool')}) failed: {rec.get('error', '')[:500]}")
    if syntax and not syntax.get("ok"):
        parts.append(f"syntax check failed: {syntax.get('error', '')[:800]}")
    for rec in _collect_test_results(records):
        if not rec.get("ok"):
            out = rec.get("output") or {}
            fails = "; ".join(out.get("failures", [])[:5])
            parts.append(
                f"run_test {out.get('target', '')}: {out.get('passed', 0)} passed / "
                f"{out.get('failed', 0)} failed. Failing: {fails}. "
                f"Output tail: {(out.get('stdout_tail') or '')[:1500]}"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The workflow generator
# ---------------------------------------------------------------------------

def stream_coding_task(task: str) -> Generator[dict, None, None]:
    """Run the full coding workflow, yielding event dicts (SSE payloads)."""
    task = (task or "").strip()
    if not task:
        yield {"type": "error", "text": "task is required"}
        yield {"type": "final", "ok": False, "error": "empty task"}
        return

    # ---- 0. Snapshot ----
    snap = take_snapshot()
    yield {"type": "phase", "phase": "snapshot",
           "text": f"Snapshot taken ({snap['files']} source files)"}

    # ---- 1. Plan ----
    yield {"type": "planning"}
    try:
        plan = make_coding_plan(task)
    except Exception as exc:
        logger.warning("Coding planner failed: %s", exc)
        yield {"type": "error", "text": f"Planning failed: {exc}"}
        yield {"type": "final", "ok": False, "error": f"Planning failed: {exc}"}
        return

    def _plan_event(p: dict, is_replan: bool = False) -> dict:
        return {
            "type": "plan",
            "reasoning": p.get("reasoning", ""),
            "steps": [
                {"id": s["id"], "tool": s["tool"], "label": _step_label(s["tool"])}
                for s in p["steps"]
            ],
            "is_replan": is_replan,
        }

    yield _plan_event(plan)

    state: dict[str, dict] = {}
    all_records: list[dict] = []       # full history (UI, touched files)
    current_plan = plan
    cycle = 0

    while True:
        # ---- 2. Execute ----
        # Records are reset per cycle: a test that failed in an earlier
        # (already-debugged) cycle must not doom the current one.
        records: list[dict] = []
        for step in current_plan["steps"]:
            sid = step["id"]
            tool_name = step["tool"]
            fn = CODING_TOOL_REGISTRY[tool_name]
            args = _resolve_args(step.get("args", {}) or {}, state)

            yield {"type": "step_start", "id": sid, "tool": tool_name,
                   "label": _step_label(tool_name)}
            t0 = time.perf_counter()
            try:
                res = fn(args, state) if args is not None else fn({}, state)
            except Exception as e:
                res = {"ok": False, "error": f"tool raised: {e}"}
            elapsed = round((time.perf_counter() - t0) * 1000)

            ok = bool(res.get("ok"))
            rec = {
                "id": sid, "tool": tool_name, "ok": ok,
                "output": res.get("output"),
                "elapsed_ms": elapsed,
            }
            if not ok:
                rec["error"] = res.get("error", "step failed")
            records.append(rec)
            all_records.append(rec)
            state[sid] = res

            yield {
                "type": "step_done", "id": sid, "tool": tool_name, "ok": ok,
                "elapsed_ms": elapsed,
                "summary": (_short_summary(tool_name, res.get("output")) if ok
                            else (res.get("error") or "step failed")[:200]),
            }
            if not ok:
                break

        # ---- 3. Verify ----
        yield {"type": "phase", "phase": "verify", "text": "Verifying changes"}
        touched = _touched_py_files(all_records)
        syntax = None
        if touched:
            check_id = f"check_{cycle + 1}"
            yield {"type": "step_start", "id": check_id, "tool": "run_command",
                   "label": "Syntax-checking changed files"}
            t0 = time.perf_counter()
            syntax = _syntax_check(touched)
            elapsed = round((time.perf_counter() - t0) * 1000)
            rec = {"id": check_id, "tool": "run_command", "ok": bool(syntax.get("ok")),
                   "output": syntax.get("output"), "elapsed_ms": elapsed}
            if not syntax.get("ok"):
                rec["error"] = syntax.get("error")
            records.append(rec)
            all_records.append(rec)
            yield {
                "type": "step_done", "id": check_id, "tool": "run_command",
                "ok": bool(syntax.get("ok")), "elapsed_ms": elapsed,
                "summary": (f"syntax OK ({len(touched)} file(s))"
                            if syntax.get("ok")
                            else (syntax.get("error") or "syntax error")[:200]),
            }

        failure = _failure_reason(records, syntax)

        if not failure:
            diff_res = tool_get_diff({}, {})
            diff_out = diff_res.get("output") or {}
            tests = _collect_test_results(records)
            test_summary = [
                {"target": (t.get("output") or {}).get("target"),
                 "passed": (t.get("output") or {}).get("passed", 0),
                 "failed": (t.get("output") or {}).get("failed", 0)}
                for t in tests
            ]
            yield {
                "type": "final", "ok": True, "cycles": cycle,
                "final_output": {
                    "tests": test_summary,
                    "files_changed": diff_out.get("files_changed", []),
                    "files_added": diff_out.get("files_added", []),
                    "files_deleted": diff_out.get("files_deleted", []),
                    "insertions": diff_out.get("insertions", 0),
                    "deletions": diff_out.get("deletions", 0),
                    "diff": diff_out.get("diff", ""),
                },
            }
            return

        # ---- 4. Debug ----
        if cycle >= MAX_DEBUG_CYCLES:
            yield {
                "type": "final", "ok": False, "cycles": cycle,
                "error": f"Max debug cycles ({MAX_DEBUG_CYCLES}) reached. Last failure:\n{failure[:800]}",
            }
            return

        cycle += 1
        yield {"type": "phase", "phase": "debug",
               "text": f"Debug cycle {cycle}: re-planning against failure"}
        yield {"type": "replan", "attempt": cycle, "reason": failure[:200]}

        diff_res = tool_get_diff({}, {})
        diff_text = (diff_res.get("output") or {}).get("diff", "")

        try:
            new_plan = replan_coding(task, current_plan, failure, diff_text)
        except Exception as exc:
            yield {"type": "error", "text": f"Re-planning failed: {exc}"}
            yield {"type": "final", "ok": False, "cycles": cycle,
                   "error": f"Re-planning failed: {exc}. Last failure:\n{failure[:600]}"}
            return

        yield _plan_event(new_plan, is_replan=True)
        current_plan = new_plan


# ---------------------------------------------------------------------------
# Non-streaming wrapper (used by tests / sync callers)
# ---------------------------------------------------------------------------

def run_coding_task(task: str) -> dict:
    """Collect all events from stream_coding_task into a summary dict."""
    events: list[dict] = []
    final: dict | None = None
    for evt in stream_coding_task(task):
        events.append(evt)
        if evt.get("type") == "final":
            final = evt
    return {"events": events, "final": final or {"ok": False, "error": "no final event"}}
