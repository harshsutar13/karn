"""
memory package — persistent session memory for the SIH26 workbench.

Two kinds of memory are stored:

    messages : the conversation (user / assistant turns) per session
    actions  : what the agent DID (tools run, files generated, plans,
               coding tasks) per session

Both live in a single local SQLite database (data/memory.db) so the
agent can ground follow-up questions like "open the file you just made"
or "continue what we were discussing". No external services, no cloud.
"""

from memory.store import (
    add_message,
    get_history,
    log_action,
    get_recent_actions,
    list_sessions,
    clear_session,
    build_llm_messages,
    build_history_prefix,
    conversation_transcript,
    actions_context_block,
)

__all__ = [
    "add_message",
    "get_history",
    "log_action",
    "get_recent_actions",
    "list_sessions",
    "clear_session",
    "build_llm_messages",
    "build_history_prefix",
    "conversation_transcript",
    "actions_context_block",
]
