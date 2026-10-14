"""
memory/store.py

SQLite-backed session memory (stdlib only — sqlite3 ships with Python).

Schema
------
    messages(id, session_id, role, content, capability, model, created_at)
        role is "user" | "assistant". capability/model record which
        capability answered (general / coding / document / vision).

    actions(id, session_id, kind, tool, summary, ok, created_at)
        kind is "tool" (a single planner/coding step) or "final"
        (outcome of a whole plan / coding task).

Design notes
------------
- One short-lived connection per operation: FastAPI serves sync endpoints
  in a threadpool, and SQLite serialises writers itself, so per-call
  connections with a busy timeout are the simplest thread-safe option.
- Session ids come from the browser (localStorage). They are normalised
  to [A-Za-z0-9_-] and capped in length; all SQL is parametrised.
- build_llm_messages() converts memory into an Ollama message list:
  recent actions become a system message ("things you recently did"),
  followed by the last N conversation turns, followed by the new message.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MEMORY_DB_PATH

logger = __import__("logging").getLogger(__name__)

_SESSION_RE = re.compile(r"[^A-Za-z0-9_-]")
MAX_SESSION_ID_LEN = 64
DEFAULT_SESSION = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    capability  TEXT,
    model       TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    tool        TEXT,
    summary     TEXT,
    ok          INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_session ON actions(session_id, id);
"""


def normalize_session_id(session_id: str | None) -> str:
    """Sanitise a client-supplied session id; empty -> 'default'."""
    if not session_id:
        return DEFAULT_SESSION
    sid = _SESSION_RE.sub("_", str(session_id).strip())[:MAX_SESSION_ID_LEN]
    return sid or DEFAULT_SESSION


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Messages (conversation memory)
# ---------------------------------------------------------------------------

def add_message(
    session_id: str | None,
    role: str,
    content: str,
    capability: str | None = None,
    model: str | None = None,
) -> None:
    """Persist one conversation turn. Role must be 'user' or 'assistant'."""
    if role not in ("user", "assistant"):
        raise ValueError(f"add_message: role must be user/assistant, got {role!r}")
    if not content or not content.strip():
        return
    sid = normalize_session_id(session_id)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, capability, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, role, content, capability, model, _now()),
        )


def get_history(session_id: str | None, limit: int = 20) -> list[dict]:
    """Last `limit` messages, oldest first. Empty list for unknown sessions."""
    sid = normalize_session_id(session_id)
    with closing(_connect()) as conn, conn:
        rows = conn.execute(
            "SELECT role, content, capability, model, created_at "
            "FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (sid, max(1, limit)),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Actions (what the agent did)
# ---------------------------------------------------------------------------

def log_action(
    session_id: str | None,
    kind: str,
    tool: str | None,
    summary: str,
    ok: bool = True,
) -> None:
    """Record one agent action (a tool step or a final plan outcome)."""
    sid = normalize_session_id(session_id)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO actions (session_id, kind, tool, summary, ok, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, kind, tool, (summary or "")[:500], 1 if ok else 0, _now()),
        )


def get_recent_actions(session_id: str | None, limit: int = 10) -> list[dict]:
    """Last `limit` actions, oldest first."""
    sid = normalize_session_id(session_id)
    with closing(_connect()) as conn, conn:
        rows = conn.execute(
            "SELECT kind, tool, summary, ok, created_at "
            "FROM actions WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (sid, max(1, limit)),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def list_sessions() -> list[dict]:
    """All sessions with message counts and last activity."""
    with closing(_connect()) as conn, conn:
        rows = conn.execute(
            "SELECT session_id, "
            "       COUNT(*) AS message_count, "
            "       MAX(created_at) AS last_activity "
            "FROM messages GROUP BY session_id ORDER BY last_activity DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def clear_session(session_id: str | None) -> None:
    """Delete all memory (messages + actions) for one session."""
    sid = normalize_session_id(session_id)
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM actions WHERE session_id = ?", (sid,))


# ---------------------------------------------------------------------------
# LLM context builders
# ---------------------------------------------------------------------------

def actions_context_block(session_id: str | None, limit: int = 10) -> str | None:
    """
    Recent agent actions as a compact text block, or None when the
    session has no recorded actions. Used to give the model awareness
    of what it previously did (files generated, plans executed, ...).
    """
    actions = get_recent_actions(session_id, limit=limit)
    if not actions:
        return None
    lines = []
    for a in actions:
        label = a.get("tool") or a.get("kind") or "action"
        status = "" if a.get("ok") else " (FAILED)"
        summary = (a.get("summary") or "").strip().replace("\n", " ")
        lines.append(f"- {label}{status}: {summary}" if summary else f"- {label}{status}")
    return "Recent actions you performed in this session:\n" + "\n".join(lines)


def build_history_prefix(
    session_id: str | None,
    max_turns: int = 8,
    include_actions: bool = True,
) -> list[dict] | None:
    """
    Memory as an Ollama message list WITHOUT the new user turn:

        [system: recent actions]   <- only if the agent did something
        [user/assistant turns ...] <- last max_turns messages

    Returns None for a fresh session (no memory yet) so callers can pass
    history=None straight into the model layer.
    """
    messages: list[dict] = []

    if include_actions:
        block = actions_context_block(session_id)
        if block:
            messages.append({
                "role": "system",
                "content": (
                    "You are a local industrial AI assistant. "
                    "Use this memory of your own recent actions to answer "
                    "follow-up questions consistently.\n\n" + block
                ),
            })

    for turn in get_history(session_id, limit=max_turns):
        messages.append({"role": turn["role"], "content": turn["content"]})

    return messages or None


def conversation_transcript(session_id: str | None, max_turns: int = 4) -> str | None:
    """
    Compact "user: ... / assistant: ..." transcript of the recent turns,
    or None when the session has no history. Used to prefix prompts for
    capabilities that take a single string prompt (RAG, vision).
    """
    history = get_history(session_id, limit=max_turns)
    if not history:
        return None
    lines = [f"{t['role']}: {t['content']}" for t in history]
    return "RECENT CONVERSATION:\n" + "\n".join(lines)


def build_llm_messages(
    session_id: str | None,
    user_message: str,
    max_turns: int = 8,
    include_actions: bool = True,
) -> list[dict]:
    """
    Build an Ollama message list from memory:

        [system: recent actions]      <- only if the agent did something
        [user/assistant history ...]  <- last max_turns messages
        [user: user_message]          <- the new turn

    Falls back to a plain single-user-message list for fresh sessions,
    so callers never need to branch on whether memory exists.
    """
    prefix = build_history_prefix(
        session_id, max_turns=max_turns, include_actions=include_actions
    )
    messages = list(prefix) if prefix else []
    messages.append({"role": "user", "content": user_message})
    return messages
