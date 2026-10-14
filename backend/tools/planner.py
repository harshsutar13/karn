"""
tools/planner.py

LLM-driven planner for the SIH26 agent.

Produces a JSON plan of tool calls given a goal.
Also supports re-planning given a failure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import ollama

from config import LLM_MODEL, PLANNER_MODEL, OLLAMA_HOST
from models.llm import generate_response
from models.manager import use_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Available tools
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {
    "rag_search": {
        "description": "Search the local knowledge base (SOPs, manuals, reports).",
        "args": {
            "query": "str — natural language search query",
            "top_k": "int (optional, default 5)",
        },
        "returns": "list of {filename, score, text} chunks",
    },
    "extract_params": {
        "description": "Extract operating parameters (rho, g, Q, H, eta) from RAG results.",
        "args": {"source": "str — name of prior step output to read (e.g. 'step_1')"},
        "returns": "dict of numeric parameters",
    },
    "run_python": {
        "description": "Execute Python code in an isolated sandbox. Use for calculations.",
        "args": {
            "code": "str — Python source code",
            "timeout": "int (optional, default 10)",
        },
        "returns": "{ok, stdout, stderr, exit_code}",
    },
    "write_docx": {
        "description": "Write a .docx deliverable from structured sections.",
        "args": {
            "title": "str",
            "sections": "list of {heading, body}",
            "output_filename": "str",
        },
        "returns": "{output_path, filename}",
    },
    "write_xlsx": {
        "description": "Write an Excel .xlsx file from structured tables.",
        "args": {
            "title": "str — workbook title",
            "output_filename": "str — e.g. 'analysis.xlsx'",
            "sheets": "list of {sheet_name, headers (list), rows (list of lists)}",
        },
        "returns": "{output_path, filename}",
    },
    "write_pptx": {
        "description": "Write a PowerPoint .pptx deck with title slide and content slides.",
        "args": {
            "title": "str — deck title",
            "subtitle": "str (optional)",
            "output_filename": "str — e.g. 'summary.pptx'",
            "slides": "list of {title, bullets (list of str), notes (optional)}",
        },
        "returns": "{output_path, filename}",
    },
    "read_file": {
        "description": (
            "Read a text file from an allowed directory (backend/outputs, "
            "backend/docs, data/, or user's Downloads folder). Returns the full text content."
        ),
        "args": {"path": "str — path to the file"},
        "returns": "{path, bytes, chars, text}",
    },
    "write_file": {
        "description": (
            "Write text content to a file inside an allowed directory. "
            "Pass a simple filename (e.g., 'script.py') to save in backend/outputs. "
            "Files are automatically saved to the outputs directory."
        ),
        "args": {
            "path": "str — simple filename like 'calc.py' or 'data.txt' (saved to outputs directory)",
            "content": "str — the text to write (use 'step_1.content' to reference prior step)",
        },
        "returns": "{path, bytes, chars}",
    },
    "edit_code_file": {
        "description": (
            "Edit/modify an existing code file. Use this when the user asks to "
            "edit, modify, update, patch, or add functionality to existing code. "
            "Reads the file, generates modified version with requested changes, "
            "and writes it back. Can access files in outputs directory or Downloads folder."
        ),
        "args": {
            "path": "str — filename or path (e.g., 'p2.py', 'C:/Users/Admin/Downloads/script.py')",
            "instructions": "str — what changes to make (e.g., 'add error handling', 'add a square root function')",
        },
        "returns": "{path, original_lines, modified_lines, changes_made}",
    },
    "read_xlsx": {
        "description": (
            "Read an existing Excel .xlsx file and return all sheets with their "
            "headers and rows."
        ),
        "args": {"path": "str — path to the .xlsx file"},
        "returns": "{path, sheets: [{name, headers, rows, row_count, col_count}]}",
    },
    "edit_xlsx": {
        "description": (
            "Modify an existing Excel .xlsx file. Supports setting individual "
            "cells, appending rows, and adding new sheets."
        ),
        "args": {
            "path": "str — path to the .xlsx file",
            "set_cells": "list of {sheet, cell ('B3'), value}",
            "append_rows": "list of {sheet, values: [...]}",
            "add_sheet": "list of sheet names to create",
        },
        "returns": "{path, sheets, changes}",
    },
    "ocr_image": {
        "description": "Run OCR on a local image and return extracted text.",
        "args": {"image_path": "str — absolute path to the image"},
        "returns": "str — extracted text",
    },
    "analyze_image": {
        "description": "Run vision model on a local image for description or understanding.",
        "args": {
            "image_path": "str — absolute path to the image",
            "prompt": "str — what to ask about the image",
        },
        "returns": "str — natural language analysis",
    },
    "scan_to_approval": {
        "description": (
            "COMPLETE end-to-end workflow for a scanned inspection report: runs "
            "OCR, extracts findings, retrieves SOPs, AND writes a maintenance "
            "approval note as a Word file — all in one step. Do NOT add extra "
            "OCR / rag_search / write_docx steps after calling this."
        ),
        "args": {
            "image_path": "str — absolute path to the scanned image",
            "case_id": "str — case identifier (e.g. 'CASE-P101-001')",
            "equipment_tag": "str (optional, default 'P-101')",
        },
        "returns": "{output_path, filename, ocr_preview, findings_preview, sources, steps}",
    },
    "summarize": {
        "description": (
            "Summarize a block of text into a short prose paragraph. "
            "USE THIS whenever the goal asks to 'summarize', 'sum up', "
            "'condense', or 'give a brief overview'. Returns plain text — "
            "NOT a document. Pair with write_file if the summary must be saved."
        ),
        "args": {
            "text": "str — the content to summarize (use 'step_1' to reference a prior step)",
            "max_words": "int (optional, default 120)",
        },
        "returns": "{model, capability, max_words, summary}",
    },
    "generate_content": {
        "description": (
            "Generate original text content using the LLM. "
            "USE THIS when the goal asks to write an essay, article, story, letter, "
            "description, or any other original content. Returns the generated text."
        ),
        "args": {
            "prompt": "str — what to generate (e.g. 'Write an essay about Elon Musk')",
            "max_words": "int (optional, default 500) — approximate word count",
        },
        "returns": "{content: str — the generated text}",
    },
    "git_push": {
        "description": (
            "Copy a generated output file to the Git workspace, commit it, and push. "
            "USE THIS when the user says 'push X to git', 'commit X', 'push X'. "
            "Only works for files already in backend/outputs or Downloads."
        ),
        "args": {
            "file": "str — filename to push, e.g. 'p1.py' or 'report.docx'",
            "message": "str (optional) — commit message; auto-generated if omitted",
        },
        "returns": "{file, source, workspace, commit, push}",
    },
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are the PLANNER for an industrial AI workbench.

Given a GOAL, output a JSON object with TWO keys: "reasoning" (a short string)
and "steps" (a list). Each step has EXACTLY these three keys: "id", "tool", "args".

TOOLS YOU MAY USE:
{tools}

STRICT RULES:
1. Output ONLY the JSON object. No markdown fences, no text before or after.
2. Each step's "tool" MUST be one of the tools listed above. Do not invent tools.
3. Each step's "args" MUST be a plain object. Do not invent extra keys inside args.
4. Use the STRING "step_1", "step_2", etc. to reference a prior step's output.
5. Keep the plan to 2-4 steps. Prefer fewer steps.
6. UNDERSTAND USER INTENT CAREFULLY:
   - "read/check/show/display/view" = ONLY use read_file (1 step)
   - "edit/modify/update/add/change/improve" = use edit_code_file
   - "create/write/generate NEW file" = use generate_content + write_file
7. If the goal asks to GENERATE content (essay, article, story, etc.),
   FIRST use generate_content to create the text, THEN use write_docx to save it.
   Example: [generate_content with prompt, then write_docx using step_1.content]
8. If the goal asks to WRITE/SAVE CODE to a file:
   FIRST use generate_content with a code prompt, THEN use write_file to save it.
   DO NOT add run_python step unless explicitly asked to "run" or "execute" the code.
9. If the goal mentions a Word file with given text (not generated), end with write_docx.
   If it mentions Excel, end with write_xlsx.
   If it mentions slides, end with write_pptx.
   If it asks to summarize, use the summarize tool — not write_docx.
10. ALL string values must be wrapped in double quotes. Write "step_1", not step_1.
11. NEVER use run_python to create, validate, or format a Word, Excel, or
    PowerPoint deliverable. Use write_docx, write_xlsx, or write_pptx directly.
    run_python is only for calculations or when explicitly asked to execute code.

EXAMPLE 1 for goal "summarize the P-101 vibration docs":
{
  "reasoning": "Search docs, then summarize.",
  "steps": [
    {"id": "step_1", "tool": "rag_search", "args": {"query": "P-101 vibration", "top_k": 5}},
    {"id": "step_2", "tool": "summarize", "args": {"text": "step_1", "max_words": 120}}
  ]
}

EXAMPLE 2 for goal "read p3.py file" or "check what's in calc.py" or "show me the code in script.py":
{
  "reasoning": "User wants to READ/VIEW a file only, not edit it.",
  "steps": [
    {"id": "step_1", "tool": "read_file", "args": {"path": "p3.py"}}
  ]
}

EXAMPLE 3 for goal "create a word file, write an essay about elon musk":
{
  "reasoning": "Generate essay content using LLM, then save to Word document.",
  "steps": [
    {"id": "step_1", "tool": "generate_content", "args": {"prompt": "Write an informative essay about Elon Musk", "max_words": 500}},
    {"id": "step_2", "tool": "write_docx", "args": {"title": "Essay About Elon Musk", "sections": [{"heading": "Elon Musk", "body": "step_1.content"}], "output_filename": "elon_musk_essay.docx"}}
  ]
}

EXAMPLE 4 for goal "write python code for simple calculator in calc.py file":
{
  "reasoning": "Generate calculator code, then save to file. Do NOT run it.",
  "steps": [
    {"id": "step_1", "tool": "generate_content", "args": {"prompt": "Write complete Python code for a simple calculator with add, subtract, multiply, divide functions", "max_words": 300}},
    {"id": "step_2", "tool": "write_file", "args": {"path": "calc.py", "content": "step_1.content"}}
  ]
}

EXAMPLE 5 for goal "edit p2.py add error handling and input validation":
{
  "reasoning": "Modify existing p2.py file to add requested features.",
  "steps": [
    {"id": "step_1", "tool": "edit_code_file", "args": {"path": "p2.py", "instructions": "Add comprehensive error handling (try-except blocks) and input validation to ensure user inputs are valid numbers"}}
  ]
}

EXAMPLE 6 for goal "create fact.py with factorial code then push to git":
{
  "reasoning": "Generate factorial code, save to file, then push to git.",
  "steps": [
    {"id": "step_1", "tool": "generate_content", "args": {"prompt": "Write Python code: factorial from 0 to n", "max_words": 300}},
    {"id": "step_2", "tool": "write_file", "args": {"path": "fact.py", "content": "step_1.content"}},
    {"id": "step_3", "tool": "git_push", "args": {"file": "fact.py"}}
  ]
}

GOAL:
{goal}

YOUR JSON:
"""

_REPLAN_SYSTEM = """\
You are the PLANNER for an industrial AI workbench.

A previous plan failed. Output a REVISED JSON object with keys
"reasoning" (string) and "steps" (list of {"id", "tool", "args"}).

TOOLS:
{tools}

Original GOAL:
{goal}

Previous plan that failed:
{prev_plan}

Why it failed:
{failure}

STRICT RULES:
1. Output ONLY JSON. No fences, no preamble.
2. Do NOT repeat the exact step that failed.
3. If a step used an unavailable library or a missing file, DROP that step.
4. Keep 2-3 steps.
5. All string values must be in double quotes.

YOUR JSON:
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    reasoning: str
    steps: list[dict] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict:
        return {"reasoning": self.reasoning, "steps": self.steps}


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json_obj(text: str) -> dict | None:
    """
    Extract the first JSON object from an LLM response.

    Robust to: markdown fences, leading/trailing prose, doubled braces,
    and unquoted identifier values (e.g. `step_1` where `"step_1"` is expected).
    """
    if not text:
        return None

    # Strip markdown fences
    text = re.sub(r"```(?:json|JSON)?\s*", "", text)
    text = text.replace("```", "").strip()



    # Find the first opening brace
    start = text.find("{")
    if start < 0:
        return None

    # Try to parse the whole tail first (fast path for clean responses)
    tail = text[start:]
    try:
        return json.loads(tail)
    except json.JSONDecodeError:
        pass

    # Balanced-brace scan
    depth = 0
    in_string = False
    escape = False
    end = None

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        # Truncated JSON — try to close it
        candidate = tail
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if in_string:
            candidate += '"'
        candidate += "]" * max(0, open_brackets)
        candidate += "}" * max(0, open_braces)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    candidate = text[start:end]

    # Try as-is
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Repair unquoted identifiers like `step_1` → `"step_1"`
    repaired = re.sub(
        r'(:\s*)(step_\w+(?:\.\w+)?)(\s*[,}\]])',
        r'\1"\2"\3',
        candidate,
    )
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair trailing commas
    repaired2 = re.sub(r",(\s*[}\]])", r"\1", repaired)
    try:
        return json.loads(repaired2)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _tools_block() -> str:
    """Compact tool list for the planner prompt."""
    lines = []
    for name, spec in AVAILABLE_TOOLS.items():
        desc = spec["description"].split(".")[0]
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _call_planner_llm(prompt: str) -> str:
    """Call the planner model via streaming."""
    client = ollama.Client(host=OLLAMA_HOST)
    chunks: list[str] = []
    with use_model(PLANNER_MODEL):
        for chunk in client.chat(
            model=PLANNER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            keep_alive="2m",
            stream=True,
            options={
                "num_predict": 4096,
                "num_ctx": 8192,
                "temperature": 0.1,
                "top_p": 0.9,
            },
        ):
            piece = chunk.message.content or ""
            if piece:
                chunks.append(piece)
    return "".join(chunks)


def _clean_steps(steps: list) -> list[dict]:
    """Validate and normalize a list of step dicts."""
    cleaned = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        tool = s.get("tool")
        if tool not in AVAILABLE_TOOLS:
            logger.warning("Planner used unknown tool %r; dropping step", tool)
            continue
        cleaned.append({
            "id": s.get("id") or f"step_{i+1}",
            "tool": tool,
            "args": s.get("args") or {},
        })
    return cleaned


def _simple_word_document_plan(goal: str) -> Plan | None:
    """Return a deterministic plan for a standalone Word-file request.

    Word creation is already a first-class tool.  Sending a simple request to
    the general planner creates opportunities for it to invent irrelevant text
    files, input documents, slide decks, or Python package imports.  Requests
    that only ask to create a Word file therefore run as one ``write_docx``
    step.  Document-analysis workflows (OCR, RAG, reports, and approvals) are
    deliberately left to the full planner.

    IMPORTANT: If the request asks to GENERATE content (essay, article, story, etc.),
    this function returns None so the full planner can generate the content via LLM.
    """
    goal_lower = goal.lower()
    if not re.search(r"\b(create|make|write|generate)\b", goal_lower):
        return None
    if not ("word" in goal_lower or ".docx" in goal_lower):
        return None
    if re.search(r"\b(ocr|scan|image|photo|approval|rag|sop|manual|report|analysis|summari[sz]|read)\b", goal_lower):
        return None

    # Check if the request is asking to GENERATE content (essay, article, story, etc.)
    # If so, return None to let the full planner handle LLM content generation
    if re.search(r"\b(essay|article|story|blog|letter|email|content|paragraph|description|about)\b", goal_lower):
        return None

    # Quoted content is the clearest signal of the text the user wants in the
    # document.  For an unquoted "given text" request, retain that literal
    # phrase rather than fabricating extra content.
    quoted = re.findall(r"['\"]([^'\"]{1,500})['\"]", goal)
    body = quoted[-1].strip() if quoted else "given text"
    title = body if len(body) <= 80 else "Generated Word Document"
    filename_stem = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    output_filename = f"{filename_stem or 'document'}.docx"

    return Plan(
        reasoning="Create the requested Word document with the dedicated writer.",
        steps=[
            {
                "id": "step_1",
                "tool": "write_docx",
                "args": {
                    "title": title,
                    "sections": [
                        {"heading": title, "body": body},
                    ],
                    "output_filename": output_filename,
                },
            },
        ],
    )


# ---------------------------------------------------------------------------
# LLM-powered multi-step intent classifier
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """\
You are an intent classifier for an AI coding assistant.
The user's goal may contain MULTIPLE actions chained together (e.g. "create X then push to git").

Extract ALL actions in order and return them as a JSON array called "steps".

Each step has:
  - "intent": one of the intents below
  - "file_name": the filename involved (or null)
  - "instructions": details about what to do (or null)

INTENTS:
- READ_FILE    : read/view/show/check/display an existing file
- EDIT_FILE    : modify/update/add to/patch/fix/refactor an EXISTING named file
- CREATE_FILE  : create a NEW code file (blank or with content)
- CREATE_WORD  : create a Word document (.docx)
- CREATE_EXCEL : create an Excel spreadsheet (.xlsx)
- CREATE_SLIDES: create a PowerPoint presentation (.pptx)
- SUMMARIZE    : summarize content
- RAG_SEARCH   : search the knowledge base
- RUN_CODE     : execute/run code
- GIT_PUSH     : push/commit a file to Git
- GENERAL      : none of the above

KEY RULES:
- "add X to <file>" = EDIT_FILE (modifying existing), NOT CREATE_FILE
- "create/write/generate <description> in <file>.py" = CREATE_FILE with instructions = the description
- "create blank/empty file" = CREATE_FILE, instructions = null
- "push/commit <file> to git/github" = GIT_PUSH
- "then push it to git" after a create step → GIT_PUSH using the same file_name
- Connectors like "then", "and then", "after that", "also" signal a new step

Output ONLY valid JSON:
{"steps": [{"intent": "...", "file_name": "...", "instructions": "..."}, ...]}

EXAMPLES:

"create fact.py containing factorial code then push to git"
→ {"steps": [
    {"intent": "CREATE_FILE", "file_name": "fact.py", "instructions": "write Python code to find factorial of numbers from 0 to n"},
    {"intent": "GIT_PUSH",    "file_name": "fact.py", "instructions": null}
  ]}

"write a calculator in calc.py and push it to github"
→ {"steps": [
    {"intent": "CREATE_FILE", "file_name": "calc.py", "instructions": "write a complete Python calculator with add, subtract, multiply, divide"},
    {"intent": "GIT_PUSH",    "file_name": "calc.py", "instructions": null}
  ]}

"edit p2.py add error handling then push to git"
→ {"steps": [
    {"intent": "EDIT_FILE", "file_name": "p2.py", "instructions": "add error handling"},
    {"intent": "GIT_PUSH",  "file_name": "p2.py", "instructions": null}
  ]}

"read p3.py"
→ {"steps": [{"intent": "READ_FILE", "file_name": "p3.py", "instructions": null}]}

"push p1.py to git"
→ {"steps": [{"intent": "GIT_PUSH", "file_name": "p1.py", "instructions": null}]}

GOAL: {goal}
JSON:"""


def _classify_intent(goal: str) -> list[dict]:
    """Use the LLM to classify ALL intents in the goal (may be multi-step)."""
    prompt = _INTENT_SYSTEM.replace("{goal}", goal)
    client = ollama.Client(host=OLLAMA_HOST)
    chunks = []
    with use_model(PLANNER_MODEL):
        for chunk in client.chat(
            model=PLANNER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            keep_alive="2m",
            stream=True,
            options={"num_predict": 512, "temperature": 0.0},
        ):
            piece = chunk.message.content or ""
            if piece:
                chunks.append(piece)
    raw = "".join(chunks)
    obj = _extract_json_obj(raw)
    if not obj:
        return []
    # Return the steps list; fall back to wrapping a single-intent response
    if "steps" in obj:
        return obj["steps"]
    if "intent" in obj:
        return [obj]
    return []


def _build_steps_for_intent(intent: dict, goal: str, last_file: str = "") -> list[dict] | None:
    """Convert a single classified intent into planner steps."""
    kind         = intent.get("intent", "")
    file_name    = (intent.get("file_name") or last_file or "").strip()
    instructions = (intent.get("instructions") or "").strip()

    if kind == "READ_FILE" and file_name:
        return [{"id": "PLACEHOLDER", "tool": "read_file",
                 "args": {"path": file_name}}]

    if kind == "EDIT_FILE" and file_name:
        return [{"id": "PLACEHOLDER", "tool": "edit_code_file",
                 "args": {"path": file_name, "instructions": instructions or goal}}]

    if kind == "CREATE_FILE" and file_name:
        is_blank = (
            not instructions or
            any(w in goal.lower() for w in ["blank", "empty", "just create", "only create"])
        )
        if is_blank:
            return [{"id": "PLACEHOLDER", "tool": "write_file",
                     "args": {"path": file_name, "content": ""}}]

        lang = ("Python"     if file_name.endswith(".py")   else
                "JavaScript" if file_name.endswith(".js")   else
                "Java"       if file_name.endswith(".java") else "code")
        return [
            {"id": "PLACEHOLDER", "tool": "generate_content",
             "args": {
                 "prompt": (
                     f"Write complete, executable {lang} code that does the following:\n"
                     f"{instructions}\n\n"
                     f"Rules:\n"
                     f"- Output ONLY raw code, no markdown fences, no prose\n"
                     f"- Use clean functions with docstrings\n"
                     f"- No import of non-standard libraries\n"
                     f"- The code must run without modification"
                 ),
                 "max_words": 400,
             }},
            {"id": "PLACEHOLDER", "tool": "write_file",
             "args": {"path": file_name, "content": "step_N.content"}},
        ]

    if kind == "RUN_CODE" and file_name:
        return [
            {"id": "PLACEHOLDER", "tool": "read_file",  "args": {"path": file_name}},
            {"id": "PLACEHOLDER", "tool": "run_python", "args": {"code": "step_N.text", "timeout": 10}},
        ]

    if kind == "GIT_PUSH":
        # file_name already resolved from last_file if classifier returned empty
        if not file_name:
            return None
        return [{"id": "PLACEHOLDER", "tool": "git_push",
                 "args": {"file": file_name}}]

    return None


def _direct_plan_from_intents(goal: str, intents: list[dict]) -> Plan | None:
    """Build a multi-step Plan from a list of classified intents."""
    if not intents:
        return None

    all_steps: list[dict] = []
    step_counter = 0
    last_file = ""  # track the most recently referenced file across steps

    for intent in intents:
        # inherit last seen file if this intent's file_name is empty
        raw_steps = _build_steps_for_intent(intent, goal, last_file=last_file)
        if raw_steps is None:
            return None  # Unknown intent — fall through to full planner

        # Update last_file for subsequent intents
        if intent.get("file_name"):
            last_file = intent["file_name"].strip()

        for s in raw_steps:
            step_counter += 1
            sid = f"step_{step_counter}"

            # Resolve back-references: "step_N" → actual previous step id
            args = {}
            for k, v in s["args"].items():
                if isinstance(v, str) and "step_N" in v:
                    prev_sid = f"step_{step_counter - 1}"
                    args[k] = v.replace("step_N", prev_sid)
                else:
                    args[k] = v

            all_steps.append({"id": sid, "tool": s["tool"], "args": args})

    if not all_steps:
        return None

    reasoning = " → ".join(
        f"{i.get('intent', '?')}({i.get('file_name', '')})" for i in intents
    )
    return Plan(reasoning=reasoning, steps=all_steps)




def make_plan(goal: str) -> Plan:
    """Ask the LLM to produce a plan for the given goal."""
    # 1. Fast deterministic shortcut for simple Word documents
    direct_plan = _simple_word_document_plan(goal)
    if direct_plan:
        return direct_plan

    # 2. LLM-powered multi-step intent classifier
    try:
        intents = _classify_intent(goal)
        logger.info("Intents classified: %s", intents)
        intent_plan = _direct_plan_from_intents(goal, intents)
        if intent_plan:
            return intent_plan
    except Exception as exc:
        logger.warning("Intent classifier failed (%s); falling through to planner", exc)

    # 3. Full LLM planner for everything else
    prompt = (
        _PLANNER_SYSTEM
        .replace("{tools}", _tools_block())
        .replace("{goal}", goal)
    )
    logger.info("Planner called for goal: %r", goal[:80])

    raw = _call_planner_llm(prompt)
    obj = _extract_json_obj(raw)

    if not obj or "steps" not in obj:
        logger.warning("Planner returned malformed plan; raw=%r", raw[:400])
        return Plan(reasoning="(planner returned malformed JSON)", steps=[], raw=raw)

    return Plan(
        reasoning=obj.get("reasoning", ""),
        steps=_clean_steps(obj.get("steps") or []),
        raw=raw,
    )


def replan(goal: str, prev_plan: Plan, failure: str) -> Plan:
    """Ask the LLM for a revised plan after a failure."""
    direct_plan = _simple_word_document_plan(goal)
    if direct_plan:
        return direct_plan

    prompt = (
        _REPLAN_SYSTEM
        .replace("{tools}", _tools_block())
        .replace("{goal}", goal)
        .replace("{prev_plan}", json.dumps(prev_plan.to_dict(), indent=2))
        .replace("{failure}", failure)
    )
    logger.info("Re-planning after failure: %s", failure[:200])

    raw = _call_planner_llm(prompt)
    obj = _extract_json_obj(raw)

    if not obj or "steps" not in obj:
        return Plan(reasoning="(re-plan malformed)", steps=[], raw=raw)

    return Plan(
        reasoning=obj.get("reasoning", ""),
        steps=_clean_steps(obj.get("steps") or []),
        raw=raw,
    )
