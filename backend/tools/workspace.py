"""
tools/workspace.py

Custom tools for the offline agentic CODING environment.

Implements the tool layer requested for the coding agent:

    list_files    — tree listing with sizes (noise dirs pruned)
    read_file     — read a file (optionally a line range), 1-based lines
    search_code   — literal or regex search across text source files
    create_file   — create a NEW file (refuses to overwrite)
    apply_patch   — apply a unified diff, matched by context not line numbers
    run_test      — run pytest targets, parsed into structured results
    run_command   — run an ALLOWLISTED local command (no shell features)
    get_diff      — working diff vs an internal snapshot (project has no git)

Design constraints
------------------
- Every path argument is resolved inside PROJECT_ROOT and rejected if it
  points into an excluded directory (caches, venv, qdrant storage, logs,
  model weights, generated outputs, snapshots).
- Tools return dicts in the executor.py convention:
      {"ok": True,  "output": {...}}
      {"ok": False, "error": "reason"}
- Patch application is context-based: hunks are located by matching their
  context+removed lines against the file, so stale line numbers still apply.
  On mismatch, the error carries the REAL surrounding file content so the
  agent's debug loop can re-plan against ground truth.
- run_command uses subprocess with shell=False and a strict allowlist.
- The working diff is snapshot-based because this project is not a git
  repository. `take_snapshot()` mirrors the text-source tree into
  .agent_state/snapshot/ ; `get_diff()` compares it to the live tree.
"""

from __future__ import annotations

import difflib
import fnmatch
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Roots & exclusion policy
# ---------------------------------------------------------------------------

# Overridable in tests (monkeypatch workspace.PROJECT_ROOT).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directory NAMES never listed/searched/snapshotted/written, at any depth.
_EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".agent_state",
    ".next",
    "dist",
    "build",
}

# Absolute dirs (relative to PROJECT_ROOT) that are not agent-writable
# source: vector-store data, logs, model weights, generated deliverables,
# user uploads, audio samples.
_EXCLUDED_ROOT_DIRS = [
    "qdrant",
    "logs",
    "models",
    "backend/outputs",
    "data/uploads",
    "data/audio",
]

# Only these extensions are treated as text source (snapshot, search, diff).
_TEXT_EXTENSIONS = {
    ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".scss",
    ".md", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".csv", ".sql", ".sh", ".bat", ".ps1",
    ".gitignore", ".dockerignore", ".env",
}

MAX_FILE_BYTES = 512 * 1024          # refuse to read/search larger text files
MAX_READ_LINES = 2000                # default cap for full-file reads
MAX_DIFF_CHARS = 50_000              # cap for get_diff output
MAX_SEARCH_RESULTS = 50
MAX_LIST_ENTRIES = 200


def _snapshot_dir() -> str:
    return os.path.join(PROJECT_ROOT, ".agent_state", "snapshot")


def _excluded_abs_dirs() -> list[str]:
    return [os.path.join(PROJECT_ROOT, *p.split("/")) for p in _EXCLUDED_ROOT_DIRS]


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _is_within(child: str, parent: str) -> bool:
    """True if `child` is `parent` or inside it (case-insensitive on Windows)."""
    try:
        rel = os.path.relpath(child, parent)
    except ValueError:  # different drives
        return False
    rel = os.path.normcase(rel)
    return not rel.startswith("..") and not os.path.isabs(rel)


def _in_excluded_dir(path: str) -> bool:
    p = _norm(path)
    for part in p.split(os.sep):
        if part in _EXCLUDED_DIR_NAMES:
            return True
    for excl in _excluded_abs_dirs():
        if _is_within(p, _norm(excl)):
            return True
    return False


def _resolve_path(path_arg: str, *, must_exist: bool) -> str | None:
    """Resolve a tool path argument to an absolute path inside PROJECT_ROOT.

    Relative paths resolve against PROJECT_ROOT (and BACKEND_ROOT as a
    convenience) — never against the process CWD, so behaviour is
    deterministic regardless of where the server was started.
    """
    if not path_arg:
        return None

    if os.path.isabs(path_arg):
        candidates = [path_arg]
    else:
        # Reject traversal segments outright — deterministic and obvious.
        if any(part == ".." for part in path_arg.replace("\\", "/").split("/")):
            return None
        candidates = [os.path.join(PROJECT_ROOT, path_arg)]
        candidates.append(os.path.join(BACKEND_ROOT, path_arg))
        if path_arg.startswith("backend/") or path_arg.startswith("backend\\"):
            stripped = path_arg.replace("\\", "/").split("/", 1)[1]
            candidates.append(os.path.join(BACKEND_ROOT, stripped))

    for cand in candidates:
        resolved = os.path.abspath(cand)
        if not _is_within(resolved, PROJECT_ROOT):
            continue
        if _in_excluded_dir(resolved):
            continue
        if must_exist and not os.path.exists(resolved):
            continue
        return resolved
    return None


def _path_error(path_arg: str, why: str) -> str:
    return (
        f"{why}: '{path_arg}'. Paths must stay inside the project root "
        f"'{PROJECT_ROOT}' and outside cache/venv/qdrant/logs/models/outputs "
        f"directories. Use paths relative to the project root "
        f"(e.g. 'backend/tools/foo.py' or 'frontend/index.html')."
    )


def _get_path_arg(args: dict) -> str | None:
    return (
        args.get("path")
        or args.get("file_path")
        or args.get("filepath")
        or args.get("filename")
    )


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        return _decode(f.read())


def _is_text_file(path: str) -> bool:
    if os.path.splitext(path)[1].lower() in _TEXT_EXTENSIONS:
        return True
    # extensionless dotfiles like .gitignore
    base = os.path.basename(path)
    if base.startswith(".") and "." not in base[1:]:
        return True
    return False


def _iter_text_files(base: str) -> list[str]:
    """All text source files under `base` (already validated), sorted."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _EXCLUDED_DIR_NAMES
            and not _in_excluded_dir(os.path.join(dirpath, d))
        )
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            if not _is_text_file(fp):
                continue
            try:
                if os.path.getsize(fp) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            out.append(fp)
    return out


# ---------------------------------------------------------------------------
# Tool: list_files
# ---------------------------------------------------------------------------

def tool_list_files(args: dict, state: dict) -> dict:
    path_arg = args.get("path") or "."
    depth = args.get("depth", 2)
    pattern = args.get("pattern")  # fnmatch filter on entry names
    try:
        depth = max(0, int(depth))
    except (TypeError, ValueError):
        depth = 2

    base = _resolve_path(path_arg, must_exist=True)
    if base is None:
        return {"ok": False, "error": _path_error(path_arg, "list_files: cannot resolve path")}
    if not os.path.isdir(base):
        return {"ok": False, "error": f"list_files: not a directory: {path_arg}"}

    entries: list[dict] = []
    truncated = False

    def walk_dir(abs_dir: str, rel: str, level: int) -> None:
        nonlocal truncated
        if truncated:
            return
        try:
            items = sorted(os.listdir(abs_dir))
        except OSError:
            return
        for name in items:
            if truncated:
                return
            if len(entries) >= MAX_LIST_ENTRIES:
                truncated = True
                return
            if name in _EXCLUDED_DIR_NAMES:
                continue
            abs_item = os.path.join(abs_dir, name)
            if _in_excluded_dir(abs_item):
                continue
            rel_item = f"{rel}/{name}".lstrip("/")
            if pattern and not fnmatch.fnmatch(name, pattern):
                # directories may still contain matches -> descend anyway
                if os.path.isdir(abs_item) and level < depth:
                    walk_dir(abs_item, rel_item, level + 1)
                continue
            if os.path.isdir(abs_item):
                entries.append({"path": rel_item, "type": "dir", "size": None})
                if level < depth:
                    walk_dir(abs_item, rel_item, level + 1)
            else:
                try:
                    size = os.path.getsize(abs_item)
                except OSError:
                    size = 0
                entries.append({"path": rel_item, "type": "file", "size": size})

    walk_dir(base, "" if _norm(base) == _norm(PROJECT_ROOT) else os.path.basename(base), 0)

    return {
        "ok": True,
        "output": {
            "base": path_arg,
            "depth": depth,
            "count": len(entries),
            "truncated": truncated,
            "entries": entries,
        },
    }


# ---------------------------------------------------------------------------
# Tool: read_file
# ---------------------------------------------------------------------------

def tool_read_file(args: dict, state: dict) -> dict:
    path_arg = _get_path_arg(args)
    if not path_arg:
        return {"ok": False, "error": "read_file requires 'path'"}

    path = _resolve_path(path_arg, must_exist=True)
    if path is None:
        if _resolve_path(path_arg, must_exist=False) is not None:
            return {"ok": False, "error": f"read_file: file not found: {path_arg}"}
        return {"ok": False, "error": _path_error(path_arg, "read_file: path not allowed")}

    if not os.path.isfile(path):
        return {"ok": False, "error": f"read_file: not a file: {path_arg}"}
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return {"ok": False, "error": f"read_file: file too large (> {MAX_FILE_BYTES} bytes): {path_arg}"}
    except OSError as exc:
        return {"ok": False, "error": f"read_file failed: {exc}"}

    try:
        text = _read_text(path)
    except Exception as exc:
        return {"ok": False, "error": f"read_file failed: {exc}"}

    lines = text.splitlines()
    total = len(lines)

    try:
        start = max(1, int(args.get("start_line", 1) or 1))
    except (TypeError, ValueError):
        start = 1
    end_arg = args.get("end_line")
    if end_arg is None:
        end = total if start == 1 else min(start + MAX_READ_LINES - 1, total)
    else:
        try:
            end = min(int(end_arg), total)
        except (TypeError, ValueError):
            end = total
    if start == 1 and end_arg is None and total > MAX_READ_LINES:
        end = MAX_READ_LINES

    if start > total:
        return {
            "ok": False,
            "error": f"read_file: start_line {start} beyond end of file ({total} lines)",
        }

    selected = lines[start - 1:end]
    numbered = "\n".join(f"{i}\t{line}" for i, line in enumerate(selected, start))

    return {
        "ok": True,
        "output": {
            "path": path_arg,
            "rel_path": os.path.relpath(path, PROJECT_ROOT).replace("\\", "/"),
            "total_lines": total,
            "start_line": start,
            "end_line": end,
            "truncated": end < total,
            "text": numbered,
        },
    }


# ---------------------------------------------------------------------------
# Tool: search_code
# ---------------------------------------------------------------------------

def tool_search_code(args: dict, state: dict) -> dict:
    query = args.get("query") or args.get("pattern") or ""
    if not query:
        return {"ok": False, "error": "search_code requires 'query'"}

    use_regex = bool(args.get("regex", False))
    case_sensitive = bool(args.get("case_sensitive", False))
    glob_filter = args.get("glob")
    try:
        context = max(0, int(args.get("context", 0) or 0))
    except (TypeError, ValueError):
        context = 0
    try:
        max_results = max(1, int(args.get("max_results", MAX_SEARCH_RESULTS)))
    except (TypeError, ValueError):
        max_results = MAX_SEARCH_RESULTS

    scope = args.get("path") or "."
    base = _resolve_path(scope, must_exist=True)
    if base is None:
        return {"ok": False, "error": _path_error(scope, "search_code: cannot resolve path")}
    if not os.path.isdir(base):
        return {"ok": False, "error": f"search_code: not a directory: {scope}"}

    pattern = query
    if use_regex:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            rx = re.compile(query, flags)
        except re.error as exc:
            return {"ok": False, "error": f"search_code: invalid regex: {exc}"}
    else:
        needle = query if case_sensitive else query.lower()

    matches: list[dict] = []
    files_matched = 0
    truncated = False

    for fp in _iter_text_files(base):
        if glob_filter and not fnmatch.fnmatch(os.path.basename(fp), glob_filter):
            continue
        try:
            text = _read_text(fp)
        except Exception:
            continue
        file_lines = text.splitlines()
        hit_lines: list[int] = []
        for idx, line in enumerate(file_lines, 1):
            if use_regex:
                found = rx.search(line) is not None
            else:
                hay = line if case_sensitive else line.lower()
                found = needle in hay
            if found:
                hit_lines.append(idx)
        if not hit_lines:
            continue
        files_matched += 1
        rel = os.path.relpath(fp, PROJECT_ROOT).replace("\\", "/")
        for ln in hit_lines:
            if len(matches) >= max_results:
                truncated = True
                break
            entry = {
                "path": rel,
                "line": ln,
                "text": file_lines[ln - 1][:240],
            }
            if context:
                lo, hi = max(1, ln - context), min(len(file_lines), ln + context)
                entry["context"] = "\n".join(
                    f"{i}\t{file_lines[i - 1][:240]}" for i in range(lo, hi + 1)
                )
            matches.append(entry)
        if truncated:
            break

    return {
        "ok": True,
        "output": {
            "query": query,
            "regex": use_regex,
            "scope": scope,
            "matches": matches,
            "count": len(matches),
            "files_matched": files_matched,
            "truncated": truncated,
        },
    }


# ---------------------------------------------------------------------------
# Tool: create_file
# ---------------------------------------------------------------------------

def tool_create_file(args: dict, state: dict) -> dict:
    path_arg = _get_path_arg(args)
    if not path_arg:
        return {"ok": False, "error": "create_file requires 'path'"}

    content = args.get("content")
    if content is None:
        content = args.get("text")
    if content is None:
        return {"ok": False, "error": "create_file requires 'content'"}
    if not isinstance(content, str):
        content = str(content)

    path = _resolve_path(path_arg, must_exist=False)
    if path is None:
        return {"ok": False, "error": _path_error(path_arg, "create_file: path not allowed")}
    if os.path.exists(path):
        return {
            "ok": False,
            "error": (
                f"create_file: file already exists: {path_arg}. "
                f"Use apply_patch to modify an existing file."
            ),
        }

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
    except Exception as exc:
        return {"ok": False, "error": f"create_file failed: {exc}"}

    return {
        "ok": True,
        "output": {
            "path": path_arg,
            "rel_path": os.path.relpath(path, PROJECT_ROOT).replace("\\", "/"),
            "bytes": len(content.encode("utf-8")),
            "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
        },
    }


# ---------------------------------------------------------------------------
# Tool: apply_patch (unified diff, context-matched)
# ---------------------------------------------------------------------------

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_unified_diff(patch_text: str) -> list[dict]:
    """Parse a unified diff into file sections + hunks.

    A hunk stays active while it still owes lines: old_count lines of
    context/removed and new_count lines of context/added (a context line
    pays down both). Headers, blank separators and trailing newlines are
    therefore never mistaken for hunk body.

    Returns [{"path": str|None, "old_path": str|None, "hunks": [...]}] where
    each hunk is {"header": str, "old_start": int, "lines": [(tag, text)]}
    and tag is ' ', '+', '-'.
    """
    sections: list[dict] = []
    current: dict | None = None
    hunk: dict | None = None
    old_need = 0
    new_need = 0

    raw_lines = patch_text.replace("\r\n", "\n").split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    for line in raw_lines:
        m = _HUNK_HEADER_RE.match(line)
        if m:
            old_start = int(m.group(1))
            old_need = int(m.group(2)) if m.group(2) is not None else 1
            new_need = int(m.group(4)) if m.group(4) is not None else 1
            if current is None:
                current = {"path": None, "old_path": None, "hunks": []}
                sections.append(current)
            hunk = {"header": line.strip(), "old_start": old_start, "lines": []}
            current["hunks"].append(hunk)
            continue

        if line.startswith("--- "):
            current = {"path": None, "old_path": _strip_diff_prefix(line[4:].strip()),
                       "hunks": []}
            sections.append(current)
            hunk = None
            old_need = new_need = 0
            continue

        if line.startswith("+++ ") and current is not None and not current["hunks"]:
            current["path"] = _strip_diff_prefix(line[4:].strip())
            continue

        if hunk is None:
            continue

        active = old_need > 0 or new_need > 0
        if not active:
            # Declared counts are satisfied. LLM-generated diffs routinely
            # undercount, so keep consuming lines that carry an explicit
            # +/-/space body prefix; blank lines and stray text end the hunk.
            if not (line.startswith("+") or line.startswith("-") or line.startswith(" ")):
                continue

        if line.startswith("\\"):  # "\ No newline at end of file"
            continue  # marker, not a counted line
        if line.startswith("+"):
            hunk["lines"].append(("+", line[1:]))
            if new_need > 0:
                new_need -= 1
        elif line.startswith("-"):
            hunk["lines"].append(("-", line[1:]))
            if old_need > 0:
                old_need -= 1
        elif line == "" or line.startswith(" "):
            hunk["lines"].append((" ", line[1:] if line else ""))
            if old_need > 0:
                old_need -= 1
            if new_need > 0:
                new_need -= 1
        else:
            # stray text inside a hunk — treat as context to stay robust
            hunk["lines"].append((" ", line))
            if old_need > 0:
                old_need -= 1
            if new_need > 0:
                new_need -= 1

    return [s for s in sections if s["hunks"]]


def _strip_diff_prefix(p: str | None) -> str | None:
    if not p:
        return None
    if p in ("/dev/null", "nul"):
        return p
    for pref in ("a/", "b/"):
        if p.startswith(pref):
            return p[2:]
    return p


def _detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    return "\n"


def _find_in_hay(hay: list[str], block: list[str], start: int, rstrip: bool):
    """Index of `block` inside `hay` at/after `start` (then from 0), or None."""
    if not block or len(block) > len(hay):
        return None
    if rstrip:
        block = [b.rstrip() for b in block]
        hay_cmp = [h.rstrip() for h in hay]
    else:
        hay_cmp = hay
    ln = len(block)
    for p in range(start, len(hay_cmp) - ln + 1):
        if hay_cmp[p:p + ln] == block:
            return p
    for p in range(0, min(start, len(hay_cmp) - ln + 1)):
        if hay_cmp[p:p + ln] == block:
            return p
    return None


def _match_hunk(old_lines: list[str], hunk: dict, cursor: int):
    """Locate a hunk in the file. Returns (pos, consumed, replacement) or None.

    A "window" [a, b) over the hunk's tagged lines selects what gets applied:
      - ' ' and '-' lines inside the window form the search block (matched
        against the file; `consumed` of them are replaced starting at `pos`)
      - '+' lines inside the window are inserted
      - lines outside the window are left untouched (not consumed, not
        inserted), which is what makes context-trimming safe.

    Windows are tried at increasing levels of tolerance:
      L0 the whole hunk, exact match
      L1 the whole hunk, trailing-whitespace-insensitive
      L2 trailing context trimmed
      L3 leading context trimmed
      L4 removed lines only
    """
    tagged = hunk["lines"]
    n = len(tagged)
    has_solid = any(tag != "+" for tag, _ in tagged)
    if not has_solid:
        # pure addition hunk: no context, no removals — use the declared
        # old_start position from the @@ header, else the cursor.
        pos = hunk.get("old_start", 0) - 1
        if not (0 <= pos <= len(old_lines)):
            pos = min(cursor, len(old_lines))
        return (pos, 0, [text for _, text in tagged])

    windows: list[tuple[int, int]] = [(0, n)]

    b = n  # trim trailing context
    while b > 0 and tagged[b - 1][0] == " ":
        b -= 1
    if b > 0:
        windows.append((0, b))

    a = 0  # trim leading context
    while a < n and tagged[a][0] == " ":
        a += 1
    if a < n:
        windows.append((a, n))

    rem = [i for i, (t, _) in enumerate(tagged) if t == "-"]
    if rem:
        windows.append((rem[0], rem[-1] + 1))

    seen: set[tuple[int, int]] = set()
    for a, b in windows:
        if (a, b) in seen or b <= a:
            continue
        seen.add((a, b))
        search = [tagged[i][1] for i in range(a, b) if tagged[i][0] != "+"]
        replace = [tagged[i][1] for i in range(a, b) if tagged[i][0] != "-"]
        for rstrip in (False, True):
            p = _find_in_hay(old_lines, search, cursor, rstrip)
            if p is None:
                p = _find_in_hay(old_lines, search, 0, rstrip)
            if p is not None:
                return (p, len(search), replace)
    return None


def _apply_hunks(old_lines: list[str], hunks: list[dict]) -> tuple[bool, list[str], dict | None]:
    """Apply hunks to `old_lines`. Returns (ok, new_lines, error_info)."""
    result: list[str] = []
    cursor = 0

    for hi, hunk in enumerate(hunks):
        m = _match_hunk(old_lines, hunk, cursor)
        if m is None:
            # Build diagnostic: show the real file around the best guess.
            search = [text for tag, text in hunk["lines"] if tag != "+"]
            guess = None
            if search:
                probe = search[0].strip()
                if probe:
                    for idx, line in enumerate(old_lines):
                        if probe in line:
                            guess = idx
                            break
            snippet_lo = max(0, (guess if guess is not None else 0) - 10)
            snippet_hi = min(len(old_lines), snippet_lo + 25)
            snippet = "\n".join(
                f"{n}\t{old_lines[n - 1]}" for n in range(snippet_lo + 1, snippet_hi + 1)
            )
            return (
                False,
                [],
                {
                    "hunk_index": hi,
                    "header": hunk["header"],
                    "expected": "\n".join(search[:15]),
                    "actual_snippet": snippet or "(file is empty)",
                    "hint": (
                        "The hunk's context does not match the file. Re-read the "
                        "file with read_file and produce a new patch against the "
                        "ACTUAL content shown above."
                    ),
                },
            )

        pos, consumed, replacement = m
        result.extend(old_lines[cursor:pos])
        result.extend(replacement)
        cursor = pos + consumed

    result.extend(old_lines[cursor:])
    return True, result, None


def tool_apply_patch(args: dict, state: dict) -> dict:
    patch = args.get("patch") or args.get("diff")
    if not patch or not patch.strip():
        return {"ok": False, "error": "apply_patch requires 'patch' (unified diff text)"}

    sections = _parse_unified_diff(patch)
    if not sections:
        return {
            "ok": False,
            "error": (
                "apply_patch: no hunks found. Provide a unified diff with "
                "'--- a/path', '+++ b/path' and '@@' hunk headers."
            ),
        }

    explicit_path = _get_path_arg(args)
    writes: list[tuple[str, str]] = []
    applied_files = []

    for section in sections:
        path_arg = section["path"] or explicit_path
        if not path_arg:
            return {"ok": False, "error": "apply_patch: patch section has no file path (missing '+++ b/path' line)"}
        if section["old_path"] in ("/dev/null", "nul"):
            # new-file patch
            res = tool_create_file(
                {"path": path_arg,
                 "content": "".join(
                     text + "\n" for tag, text in
                     [l for h in section["hunks"] for l in h["lines"] if l[0] == "+"]
                 )},
                state,
            )
            if not res.get("ok"):
                return res
            applied_files.append(path_arg)
            continue

        path = _resolve_path(path_arg, must_exist=True)
        if path is None:
            if _resolve_path(path_arg, must_exist=False) is not None:
                return {"ok": False, "error": f"apply_patch: file not found: {path_arg}"}
            return {"ok": False, "error": _path_error(path_arg, "apply_patch: path not allowed")}
        if not os.path.isfile(path):
            return {"ok": False, "error": f"apply_patch: not a file: {path_arg}"}

        try:
            text = _read_text(path)
        except Exception as exc:
            return {"ok": False, "error": f"apply_patch failed reading {path_arg}: {exc}"}

        nl = _detect_newline(text)
        had_trailing_newline = text.endswith("\n")
        old_lines = text.splitlines()

        ok, new_lines, err = _apply_hunks(old_lines, section["hunks"])
        if not ok:
            err["file"] = path_arg
            return {
                "ok": False,
                "error": (
                    f"apply_patch: hunk #{err['hunk_index'] + 1} of {path_arg} did not match.\n"
                    f"EXPECTED CONTEXT:\n{err['expected']}\n\n"
                    f"ACTUAL FILE CONTENT around the best match:\n{err['actual_snippet']}\n\n"
                    f"{err['hint']}"
                ),
            }

        new_text = nl.join(new_lines)
        if had_trailing_newline and new_lines:
            new_text += nl
        writes.append((path, new_text))
        applied_files.append(path_arg)

    # All hunks matched — write everything (atomic in the success path).
    try:
        for path, new_text in writes:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new_text)
    except Exception as exc:
        return {"ok": False, "error": f"apply_patch failed while writing: {exc}"}

    return {
        "ok": True,
        "output": {
            "files": applied_files,
            "hunks_applied": sum(len(s["hunks"]) for s in sections),
        },
    }


# ---------------------------------------------------------------------------
# Tool: run_test
# ---------------------------------------------------------------------------

def _normalize_pytest_cmd(target: str) -> list[str]:
    return [sys.executable, "-m", "pytest", target, "-q", "--no-header",
            "--tb=short", "-p", "no:cacheprovider"]


def _parse_pytest_output(stdout: str, returncode: int) -> dict:
    failures = [
        line[len("FAILED "):].strip()
        for line in stdout.splitlines()
        if line.startswith("FAILED ")
    ]
    summary = ""
    for line in reversed(stdout.splitlines()):
        if re.search(r"\d+ (passed|failed|error|skipped)", line):
            summary = line.strip()
            break

    def count(word: str) -> int:
        m = re.search(rf"(\d+) {word}", summary)
        return int(m.group(1)) if m else 0

    return {
        "ok": returncode == 0,
        "passed": count("passed"),
        "failed": count("failed") + count("error"),
        "skipped": count("skipped"),
        "failures": failures,
        "summary": summary,
        "stdout_tail": stdout[-3000:],
    }


def tool_run_test(args: dict, state: dict) -> dict:
    target = args.get("target") or _get_path_arg(args)
    if not target:
        return {"ok": False, "error": "run_test requires 'target' (pytest path or node id)"}

    try:
        timeout = int(args.get("timeout", 120) or 120)
    except (TypeError, ValueError):
        timeout = 120
    timeout = max(5, min(timeout, 300))

    cmd = _normalize_pytest_cmd(target)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"run_test: TIMEOUT after {timeout}s running {target}"}
    except Exception as exc:
        return {"ok": False, "error": f"run_test failed to launch: {exc}"}

    elapsed = round(time.perf_counter() - t0, 1)
    parsed = _parse_pytest_output(proc.stdout or "", proc.returncode)
    parsed["target"] = target
    parsed["elapsed_s"] = elapsed
    if proc.returncode != 0 and not parsed["failed"] and not parsed["failures"]:
        # pytest itself failed (bad target, collection error)
        parsed["failed"] = 1
        if not parsed["summary"]:
            parsed["summary"] = (proc.stderr or proc.stdout or "").strip()[-300:]
    error = None
    if not parsed["ok"]:
        names = ", ".join(parsed["failures"][:5])
        error = f"{target}: {parsed['failed']} test(s) failed" + (f": {names}" if names else "")
    return {"ok": parsed["ok"], "output": parsed, "error": error}


# ---------------------------------------------------------------------------
# Tool: run_command (strict allowlist)
# ---------------------------------------------------------------------------

_DANGEROUS_CHARS = set("&|;>`$^\n")
_ALLOWED_PYTHON_MODULES = {"pytest", "py_compile", "compileall"}
_ALLOWED_PROGRAMS = {"pytest", "python", "py", "ls", "dir", "cat", "type"}


def tool_run_command(args: dict, state: dict) -> dict:
    command = args.get("command") or ""
    if not command or not command.strip():
        return {"ok": False, "error": "run_command requires 'command'"}

    if any(ch in _DANGEROUS_CHARS for ch in command):
        return {
            "ok": False,
            "error": (
                f"run_command: rejected — command contains a disallowed shell "
                f"character ({sorted(c for c in _DANGEROUS_CHARS if c in command)}). "
                f"Plain single commands only; no pipes/redirects/chaining."
            ),
        }

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return {"ok": False, "error": f"run_command: could not parse command: {exc}"}
    if not parts:
        return {"ok": False, "error": "run_command: empty command"}

    prog = parts[0].lower()
    if prog not in _ALLOWED_PROGRAMS:
        return {
            "ok": False,
            "error": (
                f"run_command: '{parts[0]}' is not on the allowlist. Allowed: "
                f"pytest, python -m pytest|py_compile|compileall, ls/dir/cat/type. "
                f"Use run_test for tests."
            ),
        }

    if prog in ("python", "py"):
        if len(parts) < 3 or parts[1] != "-m" or parts[2] not in _ALLOWED_PYTHON_MODULES:
            return {
                "ok": False,
                "error": (
                    "run_command: python is only allowed as "
                    "'python -m pytest|py_compile|compileall <args>'. "
                    "For arbitrary code use the plan's run_test step instead."
                ),
            }

    exe = shutil.which(parts[0]) or parts[0]

    try:
        timeout = int(args.get("timeout", 60) or 60)
    except (TypeError, ValueError):
        timeout = 60
    timeout = max(5, min(timeout, 300))

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [exe] + parts[1:], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"run_command: TIMEOUT after {timeout}s: {command}"}
    except Exception as exc:
        return {"ok": False, "error": f"run_command failed to launch: {exc}"}

    elapsed = round(time.perf_counter() - t0, 1)
    out = (proc.stdout or "")[-3000:]
    err = (proc.stderr or "")[-1500:]
    return {
        "ok": proc.returncode == 0,
        "output": {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": out,
            "stderr": err,
            "elapsed_s": elapsed,
        },
        "error": None if proc.returncode == 0 else (err or out or f"exit code {proc.returncode}"),
    }


# ---------------------------------------------------------------------------
# Snapshot & working diff
# ---------------------------------------------------------------------------

def take_snapshot() -> dict:
    """Mirror the current text-source tree into .agent_state/snapshot/."""
    snap_dir = _snapshot_dir()
    if os.path.isdir(snap_dir):
        shutil.rmtree(snap_dir, ignore_errors=True)
    os.makedirs(snap_dir, exist_ok=True)

    files = _iter_text_files(PROJECT_ROOT)
    copied = 0
    total_bytes = 0
    for fp in files:
        rel = os.path.relpath(fp, PROJECT_ROOT)
        dest = os.path.join(snap_dir, rel)
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(fp, dest)
            copied += 1
            total_bytes += os.path.getsize(fp)
        except OSError as exc:
            logger.warning("snapshot: failed to copy %s: %s", fp, exc)

    return {
        "snapshot_dir": snap_dir,
        "files": copied,
        "bytes": total_bytes,
    }


def _diff_files(rel: str, old_path: str, new_path: str) -> tuple[str, int, int]:
    """Unified diff for one file. Returns (diff_text, insertions, deletions)."""
    try:
        old_text = _read_text(old_path)
    except Exception:
        old_text = ""
    try:
        new_text = _read_text(new_path)
    except Exception:
        new_text = ""

    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
    ins = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    return "\n".join(diff_lines), ins, dels


def tool_get_diff(args: dict, state: dict) -> dict:
    snap_dir = _snapshot_dir()
    if not os.path.isdir(snap_dir):
        return {
            "ok": False,
            "error": "get_diff: no snapshot found. A snapshot is taken automatically when the coding task starts.",
        }

    current = {
        os.path.relpath(fp, PROJECT_ROOT).replace("\\", "/")
        for fp in _iter_text_files(PROJECT_ROOT)
    }
    snap_files: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(snap_dir):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIR_NAMES]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), snap_dir).replace("\\", "/")
            snap_files.add(rel)

    added = sorted(current - snap_files)
    deleted = sorted(snap_files - current)
    changed = sorted(f for f in current & snap_files
                     if open(os.path.join(snap_dir, f), "rb").read()
                     != open(os.path.join(PROJECT_ROOT, *f.split("/")), "rb").read())

    diffs: list[str] = []
    insertions = deletions = 0
    for rel in added + changed + deleted:
        snap_path = os.path.join(snap_dir, rel)
        live_path = os.path.join(PROJECT_ROOT, *rel.split("/"))
        d, ins, dels = _diff_files(rel, snap_path, live_path)
        insertions += ins
        deletions += dels
        diffs.append(d)

    full = "\n".join(diffs)
    truncated = len(full) > MAX_DIFF_CHARS
    if truncated:
        full = full[:MAX_DIFF_CHARS] + "\n... (diff truncated)"

    return {
        "ok": True,
        "output": {
            "files_changed": changed,
            "files_added": added,
            "files_deleted": deleted,
            "insertions": insertions,
            "deletions": deletions,
            "diff": full if full.strip() else "",
        },
    }


# ---------------------------------------------------------------------------
# Registries (coding-agent facing)
# ---------------------------------------------------------------------------

CODING_TOOL_REGISTRY = {
    "list_files":   tool_list_files,
    "read_file":    tool_read_file,
    "search_code":  tool_search_code,
    "create_file":  tool_create_file,
    "apply_patch":  tool_apply_patch,
    "run_test":     tool_run_test,
    "run_command":  tool_run_command,
    "get_diff":     tool_get_diff,
}


CODING_TOOL_SPECS = {
    "list_files": {
        "description": (
            "List files and directories in the project (caches/venv/build "
            "dirs are pruned). Use FIRST to explore the workspace."
        ),
        "args": {
            "path": "str (optional, default '.') — directory relative to project root",
            "depth": "int (optional, default 2)",
            "pattern": "str (optional) — fnmatch filter on names, e.g. '*.py'",
        },
        "returns": "{entries: [{path, type, size}], count, truncated}",
    },
    "read_file": {
        "description": (
            "Read a text file with 1-based line numbers. ALWAYS read a file "
            "before patching it."
        ),
        "args": {
            "path": "str — file path relative to project root",
            "start_line": "int (optional, 1-based)",
            "end_line": "int (optional, inclusive)",
        },
        "returns": "{total_lines, start_line, end_line, text}",
    },
    "search_code": {
        "description": "Search for a literal string or regex across project text files.",
        "args": {
            "query": "str — text or regex to find",
            "path": "str (optional) — directory scope, default project root",
            "regex": "bool (optional, default false)",
            "case_sensitive": "bool (optional, default false)",
            "glob": "str (optional) — filename filter e.g. '*.py'",
            "context": "int (optional) — context lines around each match",
        },
        "returns": "{matches: [{path, line, text}], count, files_matched}",
    },
    "create_file": {
        "description": "Create a NEW file with full content. Fails if the file exists.",
        "args": {
            "path": "str — new file path relative to project root",
            "content": "str — full file content",
        },
        "returns": "{path, bytes, lines}",
    },
    "apply_patch": {
        "description": (
            "Apply a unified diff to an EXISTING file. Hunks are matched by "
            "context, not line numbers, but keep context lines accurate. "
            "Supports multiple files in one patch and new-file patches "
            "(--- /dev/null)."
        ),
        "args": {
            "patch": "str — unified diff text",
        },
        "returns": "{files, hunks_applied} — on mismatch the error contains the actual file content",
    },
    "run_test": {
        "description": (
            "Run pytest on a target and return structured results. Use after "
            "every code change."
        ),
        "args": {
            "target": "str — e.g. 'backend/tests/test_foo.py' or a node id",
            "timeout": "int (optional, default 120, max 300)",
        },
        "returns": "{ok, passed, failed, failures, summary, stdout_tail}",
    },
    "run_command": {
        "description": (
            "Run an allowlisted local command (pytest, python -m py_compile, "
            "python -m compileall, ls/dir/cat/type). No pipes or redirects."
        ),
        "args": {
            "command": "str — e.g. \"python -m py_compile backend/tools/foo.py\"",
            "timeout": "int (optional, default 60)",
        },
        "returns": "{exit_code, stdout, stderr, elapsed_s}",
    },
    "get_diff": {
        "description": (
            "Show the working diff of all changes since the task started. "
            "Use at the END of the plan to review your own changes."
        ),
        "args": {},
        "returns": "{files_changed, files_added, files_deleted, insertions, deletions, diff}",
    },
}
