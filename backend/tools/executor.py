"""
tools/executor.py

Runs a Plan step by step. Each tool writes to a shared state dict;
later steps can reference earlier ones via {"ref": "step_N"}.

On step failure, asks the planner to re-plan (max 3 re-plans).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from rag.pipeline import rag_query
from models.llm import generate_response
from tools.sandbox import run_python
from tools.docx_writer import write_docx
from tools.planner import Plan, make_plan, replan
from tools.xlsx_writer import write_xlsx
from tools.pptx_writer import write_pptx

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
MAX_REPLANS = 3

# Allowed directories for file read/write tools.
# Get user's Downloads folder (cross-platform)
_DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

_ALLOWED_FILE_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
    _DOWNLOADS_DIR,  # User's Downloads folder
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_path_arg(args: dict) -> str | None:
    """Accept several common key names for a file path from the LLM."""
    return (
        args.get("path")
        or args.get("file_path")
        or args.get("filepath")
        or args.get("filename")
    )


def _is_path_allowed(path: str) -> bool:
    if not path:
        return False

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidates = [path]
    if not os.path.isabs(path):
        # Try various path combinations
        candidates.append(os.path.join(OUTPUT_DIR, path))  # outputs/filename
        candidates.append(os.path.join(project_root, path))
        candidates.append(os.path.join(backend_root, path))
        candidates.append(os.path.join(_DOWNLOADS_DIR, path))  # Downloads/filename

        if path.startswith("backend/") or path.startswith("backend\\"):
            stripped = path.split("/", 1)[-1] if "/" in path else path.split("\\", 1)[-1]
            candidates.append(os.path.join(backend_root, stripped))

    for candidate in candidates:
        try:
            resolved = os.path.abspath(candidate)
            # Normalize path separators for comparison
            resolved = os.path.normpath(resolved)
        except Exception:
            continue
        for allowed in _ALLOWED_FILE_DIRS:
            try:
                allowed_norm = os.path.normpath(allowed)
                # Check if resolved path is within allowed directory
                try:
                    common = os.path.commonpath([resolved, allowed_norm])
                    if common == allowed_norm:
                        return True
                except ValueError:
                    # On Windows, commonpath raises ValueError if paths are on different drives
                    continue
            except Exception:
                continue
    return False


def _resolve_allowed_path(path: str) -> str | None:
    if not path:
        return None

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidates = [path]
    if not os.path.isabs(path):
        # Try various path combinations
        candidates.append(os.path.join(OUTPUT_DIR, path))  # outputs/filename (PRIORITY)
        candidates.append(os.path.join(project_root, path))
        candidates.append(os.path.join(backend_root, path))
        candidates.append(os.path.join(_DOWNLOADS_DIR, path))  # Downloads/filename

        if path.startswith("backend/") or path.startswith("backend\\"):
            stripped = path.split("/", 1)[-1] if "/" in path else path.split("\\", 1)[-1]
            candidates.append(os.path.join(backend_root, stripped))

    for candidate in candidates:
        try:
            resolved = os.path.abspath(candidate)
            resolved = os.path.normpath(resolved)
        except Exception:
            continue
        for allowed in _ALLOWED_FILE_DIRS:
            try:
                allowed_norm = os.path.normpath(allowed)
                try:
                    common = os.path.commonpath([resolved, allowed_norm])
                    if common == allowed_norm:
                        return resolved
                except ValueError:
                    continue
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Argument / reference resolution
# ---------------------------------------------------------------------------

def _resolve_args(args: dict, state: dict) -> dict:
    """Replace any {"ref": "step_N"} or "step_N.field" strings with actual output."""
    resolved = {}
    for k, v in (args or {}).items():
        # Handle dict references: {"ref": "step_1"}
        if isinstance(v, dict) and "ref" in v:
            ref = v["ref"]
            resolved[k] = state.get(ref, {}).get("output")
        # Handle string references: "step_1" or "step_1.content"
        elif isinstance(v, str) and v.startswith("step_"):
            resolved[k] = _resolve_step_ref(v, state)
        # Handle lists (like sections array)
        elif isinstance(v, list):
            resolved[k] = [_resolve_args_recursive(item, state) for item in v]
        else:
            resolved[k] = v
    return resolved


def _resolve_args_recursive(value, state: dict):
    """Recursively resolve step references in nested structures."""
    if isinstance(value, dict):
        return {k: _resolve_args_recursive(v, state) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_args_recursive(item, state) for item in value]
    elif isinstance(value, str) and value.startswith("step_"):
        return _resolve_step_ref(value, state)
    else:
        return value


def _extract_text_from_output(output) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("summary", "text", "content", "body", "answer", "context_block"):
            if key in output and isinstance(output[key], str):
                return output[key]
        return json.dumps(output, indent=2)
    return str(output)


def _resolve_step_ref(ref: str, state: dict) -> str:
    """Resolve 'step_2' or 'step_2.summary' against state."""
    if not ref or not ref.startswith("step_"):
        return ref

    parts = ref.split(".", 1)
    step_id = parts[0]
    field = parts[1] if len(parts) > 1 else None

    entry = state.get(step_id)
    if not entry:
        return ref

    output = entry.get("output")

    if field:
        if isinstance(output, dict) and field in output:
            return _extract_text_from_output(output[field])
        return ref

    return _extract_text_from_output(output)


# ---------------------------------------------------------------------------
# Tool: write_docx
# ---------------------------------------------------------------------------

def tool_write_docx(args: dict, state: dict) -> dict:
    title = args.get("title", "Document")
    sections = args.get("sections") or []
    output_filename = args.get("output_filename") or f"deliverable_{int(time.time())}.docx"

    if not sections and not title:
        return {"ok": False, "error": "write_docx requires title and sections"}

    if isinstance(sections, str):
        sections = [{"heading": "Content", "body": sections}]

    if not isinstance(sections, list) or not sections:
        return {"ok": False, "error": "write_docx sections must be a non-empty list"}

    norm = []
    for s in sections:
        if isinstance(s, dict):
            norm.append({"heading": s.get("heading", ""), "body": s.get("body", "")})
        else:
            norm.append({"heading": "", "body": str(s)})

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        written = write_docx(
            title=title,
            sections=norm,
            output_path=out_path,
            footer_note="Generated by Loki – Planner Agent.",
        )
        return {"ok": True, "output": {"output_path": written, "filename": output_filename}}
    except Exception as e:
        return {"ok": False, "error": f"write_docx failed: {e}"}


# ---------------------------------------------------------------------------
# Tool: write_xlsx
# ---------------------------------------------------------------------------

def tool_write_xlsx(args: dict, state: dict) -> dict:
    title = args.get("title", "Workbook")
    sheets = args.get("sheets") or []
    output_filename = args.get("output_filename") or f"workbook_{int(time.time())}.xlsx"

    if isinstance(sheets, str):
        return {"ok": False, "error": "write_xlsx: 'sheets' must be a list, not a string"}

    if not isinstance(sheets, list) or not sheets:
        sheets = [{"sheet_name": "Sheet1", "headers": [], "rows": []}]

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        written = write_xlsx(
            title=title,
            output_path=out_path,
            sheets=sheets,
            footer_note="Generated by Loki – Planner Agent.",
        )
        return {"ok": True, "output": {"output_path": written, "filename": output_filename}}
    except Exception as e:
        return {"ok": False, "error": f"write_xlsx failed: {e}"}


# ---------------------------------------------------------------------------
# Tool: write_pptx
# ---------------------------------------------------------------------------

def tool_write_pptx(args: dict, state: dict) -> dict:
    title = args.get("title", "Presentation")
    slides = args.get("slides") or []
    subtitle = args.get("subtitle")
    output_filename = args.get("output_filename") or f"slides_{int(time.time())}.pptx"

    if isinstance(slides, str):
        return {"ok": False, "error": "write_pptx: 'slides' must be a list, not a string"}

    if not isinstance(slides, list):
        slides = []

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        written = write_pptx(
            title=title,
            output_path=out_path,
            slides=slides,
            subtitle=subtitle,
        )
        return {"ok": True, "output": {"output_path": written, "filename": output_filename}}
    except Exception as e:
        return {"ok": False, "error": f"write_pptx failed: {e}"}


# ---------------------------------------------------------------------------
# Tool: read_xlsx
# ---------------------------------------------------------------------------

def tool_read_xlsx(args: dict, state: dict) -> dict:
    from openpyxl import load_workbook

    path_arg = _get_path_arg(args)
    if not path_arg:
        return {"ok": False, "error": "read_xlsx requires 'path'"}

    path = _resolve_allowed_path(path_arg)
    if path is None:
        return {"ok": False, "error": f"read_xlsx: path outside allowed dirs: {path_arg}"}

    if not os.path.isfile(path):
        return {"ok": False, "error": f"read_xlsx: file not found: {path}"}

    try:
        wb = load_workbook(path, data_only=True)
    except Exception as exc:
        return {"ok": False, "error": f"read_xlsx: could not open workbook: {exc}"}

    result = {"path": path, "sheets": []}
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        all_rows = list(ws.iter_rows(values_only=True))

        header_idx = 0
        for i, row in enumerate(all_rows):
            non_empty = sum(1 for c in row if c not in (None, ""))
            if non_empty >= 2:
                header_idx = i
                break

        headers = (
            [str(c) if c is not None else "" for c in all_rows[header_idx]]
            if header_idx < len(all_rows) else []
        )
        data_rows = [
            list(r) for r in all_rows[header_idx + 1:]
            if any(c not in (None, "") for c in r)
        ]

        result["sheets"].append({
            "name": sheet_name,
            "headers": headers,
            "rows": data_rows,
            "row_count": len(data_rows),
            "col_count": len(headers),
            "header_row_index": header_idx,
        })

    return {"ok": True, "output": result}


# ---------------------------------------------------------------------------
# Tool: edit_xlsx
# ---------------------------------------------------------------------------

def tool_edit_xlsx(args: dict, state: dict) -> dict:
    from openpyxl import load_workbook, Workbook

    path_arg = _get_path_arg(args)
    if not path_arg:
        return {"ok": False, "error": "edit_xlsx requires 'path'"}

    path = _resolve_allowed_path(path_arg)
    if path is None:
        return {"ok": False, "error": f"edit_xlsx: path outside allowed dirs: {path_arg}"}

    set_cells = args.get("set_cells") or []
    append_rows = args.get("append_rows") or []
    add_sheet = args.get("add_sheet") or []

    try:
        if os.path.isfile(path):
            wb = load_workbook(path)
        else:
            wb = Workbook()
            default = wb.active
            wb.remove(default)
    except Exception as exc:
        return {"ok": False, "error": f"edit_xlsx: could not open workbook: {exc}"}

    changes = {"set_cells": 0, "appended_rows": 0, "added_sheets": []}

    if isinstance(add_sheet, str):
        add_sheet = [add_sheet]
    for name in add_sheet:
        if not isinstance(name, str):
            continue
        if name not in wb.sheetnames:
            wb.create_sheet(title=name[:31])
            changes["added_sheets"].append(name)

    if not wb.sheetnames:
        wb.create_sheet(title="Sheet1")

    if isinstance(set_cells, dict):
        set_cells = [set_cells]
    for spec in set_cells:
        if not isinstance(spec, dict):
            continue
        sheet_name = spec.get("sheet") or wb.sheetnames[0]
        cell = spec.get("cell")
        value = spec.get("value")
        if not cell:
            continue
        if sheet_name not in wb.sheetnames:
            wb.create_sheet(title=str(sheet_name)[:31])
        ws = wb[sheet_name]
        ws[cell] = value
        changes["set_cells"] += 1

    if isinstance(append_rows, dict):
        append_rows = [append_rows]
    for spec in append_rows:
        if isinstance(spec, dict):
            sheet_name = spec.get("sheet") or wb.sheetnames[0]
            values = spec.get("values")
            if values is None:
                continue
        elif isinstance(spec, (list, tuple)):
            sheet_name = wb.sheetnames[0]
            values = list(spec)
        else:
            sheet_name = wb.sheetnames[0]
            values = [spec]

        if sheet_name not in wb.sheetnames:
            wb.create_sheet(title=str(sheet_name)[:31])
        ws = wb[sheet_name]
        ws.append(values)
        changes["appended_rows"] += 1

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        wb.save(path)
    except Exception as exc:
        return {"ok": False, "error": f"edit_xlsx: could not save: {exc}"}

    return {
        "ok": True,
        "output": {
            "path": path,
            "sheets": wb.sheetnames,
            "changes": changes,
        },
    }


# ---------------------------------------------------------------------------
# Tool: read_file
# ---------------------------------------------------------------------------

def tool_read_file(args: dict, state: dict) -> dict:
    path_arg = _get_path_arg(args)
    if not path_arg:
        return {"ok": False, "error": "read_file requires 'path'"}

    path = _resolve_allowed_path(path_arg)
    if path is None:
        return {
            "ok": False,
            "error": (
                f"read_file: path '{path_arg}' is outside the allowed directories. "
                f"Allowed roots: {[os.path.basename(d) for d in _ALLOWED_FILE_DIRS]}"
            ),
        }

    if not os.path.isfile(path):
        return {"ok": False, "error": f"read_file: file not found: {path}"}

    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "error": f"read_file failed: {exc}"}

    return {
        "ok": True,
        "output": {
            "path": path,
            "bytes": len(raw),
            "chars": len(text),
            "text": text,
        },
    }


# ---------------------------------------------------------------------------
# Tool: write_file
# ---------------------------------------------------------------------------

def tool_write_file(args: dict, state: dict) -> dict:
    path_arg = _get_path_arg(args)
    content = args.get("content", "") or args.get("text", "") or args.get("body", "")

    if not path_arg:
        # A planner may know it needs to save a text deliverable without
        # inventing a filename.  Save it in the managed outputs directory
        # instead of surfacing that harmless omission as a failed workflow.
        requested_name = str(args.get("output_filename") or args.get("name") or "").strip()
        filename = os.path.basename(requested_name) if requested_name else ""
        if not filename:
            filename = f"deliverable_{int(time.time())}.txt"
        elif not os.path.splitext(filename)[1]:
            filename += ".txt"
        path_arg = os.path.join(OUTPUT_DIR, filename)
    else:
        # If path_arg is just a filename (no directory), save to OUTPUT_DIR
        # This handles cases like "p1.py" or "calc.py"
        if os.path.basename(path_arg) == path_arg:
            # Just a filename, no directory component
            path_arg = os.path.join(OUTPUT_DIR, path_arg)

    # Clean up any escaped backslashes or malformed paths
    path_arg = path_arg.replace('\\\\', '\\').strip('"\'')

    path = _resolve_allowed_path(path_arg)
    if path is None:
        return {
            "ok": False,
            "error": (
                f"write_file failed: [Errno 22] Invalid argument: '{path_arg}'. "
                f"Use a simple filename (e.g., 'p1.py') or an absolute path inside "
                f"backend/outputs, backend/docs, data/, or Downloads folder."
            ),
        }

    # Resolve "step_N" / "step_N.field" refs and {"ref": "..."} dicts
    if isinstance(content, str) and content.startswith("step_"):
        content = _resolve_step_ref(content, state)
    elif isinstance(content, dict):
        ref = content.get("ref")
        if ref:
            content = _resolve_step_ref(str(ref), state)
        else:
            content = _extract_text_from_output(content)

    if not isinstance(content, str):
        content = str(content)

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:
        return {"ok": False, "error": f"write_file failed: {exc}"}

    output = {
        "path": path,
        "bytes": len(content.encode("utf-8")),
        "chars": len(content),
    }
    # Outputs are exposed by the deliverables API, so provide the filename for
    # the streaming UI to offer an immediate download link.
    if os.path.dirname(path) == OUTPUT_DIR:
        output["filename"] = os.path.basename(path)

    return {
        "ok": True,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Tool: scan_to_approval
# ---------------------------------------------------------------------------

def tool_scan_to_approval(args: dict, state: dict) -> dict:
    image_path = args.get("image_path")
    case_id = args.get("case_id") or "CASE-SCAN-001"
    equipment_tag = args.get("equipment_tag") or "P-101"

    if not image_path:
        return {"ok": False, "error": "scan_to_approval requires 'image_path'"}

    if not os.path.isfile(image_path):
        return {"ok": False, "error": f"scanned image not found: {image_path}"}

    try:
        from tools.agent import scan_to_approval
        result = scan_to_approval(
            image_path=image_path,
            case_id=case_id,
            equipment_tag=equipment_tag,
        )
    except Exception as exc:
        return {"ok": False, "error": f"scan_to_approval raised: {exc}"}

    if result.error:
        return {"ok": False, "error": result.error}

    return {
        "ok": True,
        "output": {
            "output_path": result.output_path,
            "filename": os.path.basename(result.output_path),
            "ocr_preview": result.ocr_text[:300],
            "findings_preview": result.extracted_findings[:300],
            "sources": result.sources,
            "steps": result.steps,
        },
    }


# ---------------------------------------------------------------------------
# Tool: rag_search
# ---------------------------------------------------------------------------

def tool_rag_search(args: dict, state: dict) -> dict:
    query = args.get("query", "")
    try:
        top_k = int(args.get("top_k", 5))
    except (TypeError, ValueError):
        top_k = 5
    if not query:
        return {"ok": False, "error": "rag_search requires 'query'"}
    rag = rag_query(query, top_k=top_k)
    return {
        "ok": True,
        "output": {
            "chunks": [
                {"filename": s.filename, "score": s.score, "text": s.text_preview}
                for s in rag.sources
            ],
            "context_block": "\n\n".join(
                f"[{s.filename}]\n{s.text_preview}" for s in rag.sources
            ),
            "count": rag.chunks_retrieved,
        },
    }


# ---------------------------------------------------------------------------
# Tool: extract_params
# ---------------------------------------------------------------------------

def tool_extract_params(args: dict, state: dict) -> dict:
    source = args.get("source")
    block = ""
    if isinstance(source, dict):
        block = source.get("context_block", "") or ""
    elif isinstance(source, str) and source in state:
        block = state.get(source, {}).get("output", {}).get("context_block", "") or ""
    if not block:
        for k, v in state.items():
            out = v.get("output") if isinstance(v, dict) else None
            if isinstance(out, dict) and out.get("context_block"):
                block = out["context_block"]
                break
    if not block:
        return {"ok": False, "error": "extract_params: no context available"}

    prompt = f"""Extract operating parameters. Return ONLY a JSON object with keys:
rho (kg/m3), g (m/s2), Q (m3/s), H (m), eta (0-1).
Use null for any value not present. Do not invent.

CONTEXT:
{block[:4000]}"""

    raw = generate_response(prompt, think=False)
    from tools.planner import _extract_json_obj
    params = _extract_json_obj(raw)
    if params is None:
        return {"ok": False, "error": f"extract_params: LLM returned non-JSON: {raw[:200]}"}

    return {"ok": True, "output": params}


# ---------------------------------------------------------------------------
# Tool: run_python
# ---------------------------------------------------------------------------

def tool_run_python(args: dict, state: dict) -> dict:
    code = args.get("code", "")
    try:
        timeout = int(args.get("timeout", 10))
    except (TypeError, ValueError):
        timeout = 10
    if not code:
        return {"ok": False, "error": "run_python requires 'code'"}
    r = run_python(code, timeout=timeout)
    return {
        "ok": r.ok,
        "output": {
            "ok": r.ok,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "exit_code": r.exit_code,
            "elapsed_ms": r.elapsed_ms,
        },
        "error": None if r.ok else (r.stderr or "sandbox execution failed"),
    }


# ---------------------------------------------------------------------------
# Tool: ocr_image
# ---------------------------------------------------------------------------

def tool_ocr_image(args: dict, state: dict) -> dict:
    image_path = args.get("image_path")
    if not image_path:
        return {"ok": False, "error": "ocr_image requires 'image_path'"}
    try:
        from models.ocr import extract_text
        text = extract_text(image_path)
        return {"ok": True, "output": text}
    except Exception as e:
        return {"ok": False, "error": f"ocr_image failed: {e}"}


# ---------------------------------------------------------------------------
# Tool: analyze_image
# ---------------------------------------------------------------------------

def tool_analyze_image(args: dict, state: dict) -> dict:
    image_path = args.get("image_path")
    prompt = args.get("prompt", "Describe this image.")
    if not image_path:
        return {"ok": False, "error": "analyze_image requires 'image_path'"}
    try:
        from models.vision import analyze_image
        text = analyze_image(image_path=image_path, prompt=prompt)
        return {"ok": True, "output": text}
    except Exception as e:
        return {"ok": False, "error": f"analyze_image failed: {e}"}


# ---------------------------------------------------------------------------
# Tool: summarize
# ---------------------------------------------------------------------------

def tool_summarize(args: dict, state: dict) -> dict:
    text = args.get("text", "") or args.get("content", "")
    max_words = args.get("max_words", 120)

    # Resolve "step_N" / "step_N.field"
    if isinstance(text, str) and text.startswith("step_"):
        text = _resolve_step_ref(text, state)

    # Resolve {"ref": "..."}
    if isinstance(text, dict):
        ref = text.get("ref")
        if ref:
            text = _resolve_step_ref(str(ref), state)

    if isinstance(text, dict):
        for key in ("context_block", "text", "summary", "content", "answer"):
            if key in text and isinstance(text[key], str):
                text = text[key]
                break
        else:
            text = json.dumps(text, indent=2)

    if not isinstance(text, str) or not text.strip():
        return {
            "ok": False,
            "error": "summarize: no text content resolved (received empty input)",
        }

    try:
        max_words = int(max_words)
    except (TypeError, ValueError):
        max_words = 120

    from models.registry import get_model
    cfg = get_model("summarization")

    prompt = (
        f"Summarize the following text in at most {max_words} words. "
        f"Be concise. Do not add commentary or headings.\n\n"
        f"TEXT:\n{text[:6000]}"
    )

    summary = generate_response(prompt, think=False)
    return {
        "ok": True,
        "output": {
            "model": cfg.model_name,
            "capability": cfg.capability,
            "max_words": max_words,
            "summary": summary,
        },
    }


def tool_generate_content(args: dict, state: dict) -> dict:
    """Generate original text content using the LLM.

    Args:
        prompt: What to generate (e.g. "Write an essay about Elon Musk")
        max_words: Approximate word count (default 500)

    Returns:
        {ok: True, output: {content: str}}
    """
    prompt = args.get("prompt", "")
    max_words = args.get("max_words", 500)

    if not prompt or not prompt.strip():
        return {
            "ok": False,
            "error": "generate_content: prompt is required",
        }

    try:
        max_words = int(max_words)
    except (TypeError, ValueError):
        max_words = 500

    from models.registry import get_model

    # Detect if this is a code generation request
    is_code_request = any(keyword in prompt.lower() for keyword in [
        'code', 'python', 'javascript', 'java', 'function', 'class',
        'script', 'program', 'implementation', 'algorithm'
    ])

    if is_code_request:
        # For code generation: request clean, executable code only
        cfg = get_model("coding")
        generation_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Generate ONLY clean, executable code. "
            f"Do NOT include:\n"
            f"- Markdown formatting or code fences (```)\n"
            f"- Explanatory text before or after the code\n"
            f"- Tutorial-style documentation\n"
            f"- Section headers or descriptions\n\n"
            f"Output pure, ready-to-run code that can be saved directly to a .py file."
        )
    else:
        # For prose/essay generation: request well-written content
        cfg = get_model("general_reasoning")
        generation_prompt = (
            f"{prompt}\n\n"
            f"Generate approximately {max_words} words of well-written, informative content. "
            f"Be thorough, accurate, and engaging. Structure the content with clear paragraphs."
        )

    try:
        content = generate_response(generation_prompt, think=False)

        # For code: strip any markdown fences or explanatory text that leaked through
        if is_code_request:
            # Remove markdown code fences
            content = content.replace('```python', '').replace('```', '')

            # If content has explanatory sections before code, try to extract just the code
            lines = content.split('\n')
            code_lines = []
            in_code = False

            for line in lines:
                # Skip markdown headers and description paragraphs
                if line.startswith('#') and not line.startswith('# ') and '=' in line:
                    continue
                if line.strip().startswith('###') or line.strip().startswith('**'):
                    continue

                # Detect start of actual code (imports, class, def, or comments)
                if not in_code and (
                    line.strip().startswith('import ') or
                    line.strip().startswith('from ') or
                    line.strip().startswith('class ') or
                    line.strip().startswith('def ') or
                    line.strip().startswith('#')
                ):
                    in_code = True

                if in_code:
                    code_lines.append(line)

            # If we extracted code lines, use those; otherwise use original
            if code_lines and len(code_lines) > 5:
                content = '\n'.join(code_lines).strip()

        return {
            "ok": True,
            "output": {
                "content": content,
                "model": cfg.model_name,
                "word_count": len(content.split()),
            },
        }
    except Exception as e:
        logger.exception("generate_content failed")
        return {
            "ok": False,
            "error": f"generate_content failed: {e}",
        }


def tool_edit_code_file(args: dict, state: dict) -> dict:
    """Edit an existing code file with requested modifications.

    Args:
        path: Filename in outputs directory (e.g., 'p2.py')
        instructions: What changes to make (e.g., 'add error handling')

    Returns:
        {ok: True, output: {path, original_lines, modified_lines, changes_made}}
    """
    path_arg = _get_path_arg(args)
    instructions = args.get("instructions", "") or args.get("changes", "") or args.get("modifications", "")

    if not path_arg:
        return {
            "ok": False,
            "error": "edit_code_file: path is required (e.g., 'p2.py')",
        }

    if not instructions or not instructions.strip():
        return {
            "ok": False,
            "error": "edit_code_file: instructions are required (what changes to make)",
        }

    # Handle simple filenames
    if os.path.basename(path_arg) == path_arg:
        path_arg = os.path.join(OUTPUT_DIR, path_arg)

    path = _resolve_allowed_path(path_arg)
    if path is None or not os.path.exists(path):
        return {
            "ok": False,
            "error": f"edit_code_file: file '{path_arg}' not found in allowed directories. Make sure the file exists in backend/outputs.",
        }

    # Read existing file
    try:
        with open(path, "r", encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        return {
            "ok": False,
            "error": f"edit_code_file: failed to read file: {e}",
        }

    original_lines = len(original_content.split('\n'))

    # Use coding model to generate modified version
    from models.registry import get_model
    cfg = get_model("coding")

    modification_prompt = (
        f"You are editing an existing code file. Here is the current code:\n\n"
        f"```\n{original_content}\n```\n\n"
        f"INSTRUCTIONS: {instructions}\n\n"
        f"Generate the COMPLETE modified version of the code with the requested changes. "
        f"Output ONLY the modified code, no explanations, no markdown fences, no comments about what changed. "
        f"Just the full, working code with the modifications applied."
    )

    try:
        modified_content = generate_response(modification_prompt, think=False)

        # Clean up any markdown that leaked through
        modified_content = modified_content.replace('```python', '').replace('```', '').strip()

        # Write modified content back to file
        with open(path, "w", encoding="utf-8") as f:
            f.write(modified_content)

        modified_lines = len(modified_content.split('\n'))

        return {
            "ok": True,
            "output": {
                "path": path,
                "original_lines": original_lines,
                "modified_lines": modified_lines,
                "changes_made": instructions,
            },
        }

    except Exception as e:
        logger.exception("edit_code_file failed")
        return {
            "ok": False,
            "error": f"edit_code_file failed: {e}",
        }


# ---------------------------------------------------------------------------
# Tool: git_push
# ---------------------------------------------------------------------------

def tool_git_push(args: dict, state: dict) -> dict:
    """Copy a generated file to the separate Git workspace, commit and push.

    Args:
        file: filename (e.g. 'p1.py') or full path inside outputs/
        message: optional commit message (auto-generated if omitted)
    """
    from tools.git_pusher import push_file_to_git

    file_arg = (
        args.get("file") or args.get("path") or
        args.get("filename") or args.get("file_name") or ""
    ).strip()

    if not file_arg:
        return {"ok": False, "error": "git_push: 'file' argument is required (e.g. 'p1.py')"}

    commit_msg = args.get("message") or args.get("commit_message")
    ws = args.get("workspace") or args.get("git_workspace")

    result = push_file_to_git(file_arg, commit_message=commit_msg, git_workspace=ws)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Git push failed")}

    return {
        "ok": True,
        "output": {
            "file": result.get("file"),
            "source": result.get("source_path"),
            "workspace": result.get("workspace"),
            "commit": result.get("commit_summary") or result.get("commit_message"),
            "push": result.get("push_status"),
        },
    }



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "read_file":         tool_read_file,
    "write_file":        tool_write_file,
    "read_xlsx":         tool_read_xlsx,
    "edit_xlsx":         tool_edit_xlsx,
    "write_docx":        tool_write_docx,
    "write_xlsx":        tool_write_xlsx,
    "write_pptx":        tool_write_pptx,
    "scan_to_approval":  tool_scan_to_approval,
    "rag_search":        tool_rag_search,
    "extract_params":    tool_extract_params,
    "run_python":        tool_run_python,
    "ocr_image":         tool_ocr_image,
    "analyze_image":     tool_analyze_image,
    "summarize":         tool_summarize,
    "generate_content":  tool_generate_content,
    "edit_code_file":    tool_edit_code_file,
    "git_push":          tool_git_push,
}


# ---------------------------------------------------------------------------
# Execution loop
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    goal: str
    plan: dict
    replans: int
    step_results: list[dict] = field(default_factory=list)
    final_output: dict = field(default_factory=dict)
    error: str | None = None


def execute_plan(goal: str, plan: Plan) -> ExecutionResult:
    state: dict[str, dict] = {}
    result = ExecutionResult(goal=goal, plan=plan.to_dict(), replans=0)
    current_plan = plan

    for _attempt in range(MAX_REPLANS + 1):
        failed_step = None
        failed_reason = ""

        for step in current_plan.steps:
            sid = step["id"]
            tool_name = step["tool"]
            raw_args = step.get("args", {})
            args = _resolve_args(raw_args, state)
            fn = TOOL_REGISTRY.get(tool_name)

            step_record = {
                "id": sid,
                "tool": tool_name,
                "args": args,
                "started": time.strftime("%H:%M:%S"),
            }

            if not fn:
                step_record["ok"] = False
                step_record["error"] = f"unknown tool {tool_name}"
                result.step_results.append(step_record)
                failed_step = sid
                failed_reason = f"unknown tool {tool_name}"
                break

            try:
                res = fn(args, state)
            except Exception as e:
                res = {"ok": False, "error": f"tool raised: {e}"}

            step_record["ok"] = res.get("ok", False)
            step_record["output"] = res.get("output")
            if not res.get("ok"):
                step_record["error"] = res.get("error", "step failed")

            result.step_results.append(step_record)
            state[sid] = res

            if not res.get("ok"):
                failed_step = sid
                failed_reason = res.get("error", "step failed")
                break

        if failed_step is None:
            result.final_output = state.get(
                current_plan.steps[-1]["id"], {}
            ).get("output", {}) if current_plan.steps else {}
            return result

        if result.replans >= MAX_REPLANS:
            result.error = f"Max re-plans reached. Last failure: {failed_reason}"
            return result

        logger.warning("Re-planning (attempt %d). Failure: %s",
                       result.replans + 1, failed_reason)
        result.replans += 1

        replan_hint = failed_reason
        if "NameError" in failed_reason:
            replan_hint = (
                "The Python code referenced an undefined function. "
                "Re-plan with COMPLETE, self-contained code that only uses built-ins "
                "and imports, OR drop the run_python step if it isn't essential. "
                "Original error: " + failed_reason
            )
        elif "FileNotFoundError" in failed_reason or "not found" in failed_reason.lower():
            replan_hint = (
                "The step referenced a file that does not exist. "
                "Drop any step that requires a user-supplied file, "
                "OR use only file paths that were explicitly given in the GOAL. "
                "Original error: " + failed_reason
            )
        elif "no context available" in failed_reason:
            replan_hint = (
                "extract_params ran before any rag_search produced context. "
                "Either add a rag_search step BEFORE extract_params, or drop "
                "extract_params entirely if it isn't essential. "
                "Original error: " + failed_reason
            )
        elif "has no attribute 'get'" in failed_reason:
            replan_hint = (
                "A tool received arguments in the wrong shape. Re-plan and pass "
                "the expected structure: append_rows is a list of "
                "{'sheet': '...', 'values': [...]} dicts. "
                "Original error: " + failed_reason
            )
        elif "matplotlib" in failed_reason or "requests" in failed_reason:
            replan_hint = (
                "The sandbox does not have matplotlib or requests installed. "
                "Do NOT try to use them again. Instead, produce the deliverable "
                "using write_xlsx / write_docx / write_pptx / write_file. "
                "Original error: " + failed_reason
            )
        elif "SyntaxError" in failed_reason:
            replan_hint = (
                "The Python code you generated has a SYNTAX ERROR. "
                "Common cause: using `try:` or `if:` on the same line as "
                "other statements with semicolons — this is INVALID Python. "
                "RULES FOR THE REVISED CODE:\n"
                "  - Every `try:`, `if:`, `for:`, `while:`, `def:` must start "
                "its OWN LINE.\n"
                "  - The body of each block must be indented 4 spaces.\n"
                "  - Do NOT use semicolons to join a control-flow statement "
                "with other code.\n"
                "  - Do NOT use matplotlib, requests, or any library other than "
                "stdlib, numpy, pandas, openpyxl, sympy.\n"
                "If you cannot write valid code, DROP the run_python step and "
                "use write_xlsx / write_docx / write_pptx instead.\n"
                "Original error: " + failed_reason
            )
        elif "outside the allowed" in failed_reason:
            replan_hint = (
                "The path you passed is outside the sandboxed directories. "
                "Use ABSOLUTE paths inside D:/SIH26/backend/outputs/ or "
                "D:/SIH26/data/. Do not use relative paths like "
                "'backend/outputs/report.txt'. Original error: " + failed_reason
            )
        elif "requires 'path'" in failed_reason or "requires 'file_path'" in failed_reason:
            replan_hint = (
                "A file tool was called without its path argument. "
                "Use the key 'path' (not 'file_path', 'filepath', or 'filename') "
                "when calling read_file, write_file, read_xlsx, or edit_xlsx. "
                "Original error: " + failed_reason
            )

        new_plan = replan(goal, current_plan, replan_hint)
        if not new_plan.steps:
            result.error = f"Planner could not recover. Failure: {failed_reason}"
            return result
        current_plan = new_plan
        result.plan = new_plan.to_dict()

    return result


def plan_and_execute(goal: str) -> ExecutionResult:
    plan = make_plan(goal)
    if not plan.steps:
        return ExecutionResult(
            goal=goal, plan=plan.to_dict(), replans=0,
            error="Planner produced no usable steps",
        )
    return execute_plan(goal, plan)
