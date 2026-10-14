"""
orchestration/router.py

Lightweight deterministic capability classifier for SIH26.

Classifies an incoming request into one of four capabilities:
    general   — answered by the general LLM (Gemma)
    coding    — answered by the coding LLM (Qwen3-Coder)
    document  — answered by the RAG pipeline
    vision    — answered by the vision model (Qwen3-VL)

Rules (in priority order)
-------------------------
1. If image_path is supplied → vision  (always)
2. If message matches coding keywords → coding
3. If message matches document/RAG keywords → document
4. Otherwise → general

No LLM call, no ML model, no new dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_CODING_KEYWORDS = {
    # task types
    "code", "write", "implement", "program", "script", "function", "class",
    "algorithm", "coding", "develop",
    # review / debug
    "debug", "fix", "error", "bug", "exception", "traceback", "review",
    "refactor", "optimise", "optimize",
    # languages / tools
    "python", "javascript", "typescript", "sql", "bash", "powershell",
    "java", "c++", "rust", "go", "html", "css", "api", "endpoint",
    # artifacts
    "snippet", "syntax", "import", "def ", "class ", "return",
}

_DOCUMENT_KEYWORDS = {
    # explicit document references
    "document", "according", "based on", "report", "maintenance",
    "manual", "specification", "datasheet", "procedure",
    # industrial / domain
    "pump", "valve", "sensor", "threshold", "vibration", "pressure",
    "temperature", "rpm", "flow", "alarm", "alert", "rms",
    "equipment", "p-101", "refinery", "plant", "inspection",
    # RAG-style phrasing
    "summarise", "summarize", "extract from", "find in", "look up",
}

_LLM_CLASSIFY_PROMPT = """\
Classify the user's request into exactly ONE of these capabilities:

- general : open conversation, chit-chat, math, general knowledge, no document/code/image
- coding  : writing, reviewing, explaining, debugging code; SQL; scripts
- document: questions about industrial documents (SOPs, manuals, inspection reports),
            equipment (pumps, valves, P-tags), RAG-style lookups ("what does the SOP say",
            "summarize the report", "what is the vibration reading on P-101")
- vision  : analyzing an image or scanned document (only if an image path is provided)

If a recent conversation is provided, use it to interpret ambiguous follow-ups
(e.g. "now do the same for P-102" or "make that a function" inherit the topic
of the previous turn).

Answer with ONE word only: general, coding, document, or vision. No punctuation.

{conversation_block}USER MESSAGE:
{message}

CAPABILITY:"""


def classify_request_llm(message: str, conversation: str | None = None) -> str | None:
    """
    Ask the LLM to classify intent. Returns the capability string,
    or None if the LLM failed / returned something unrecognized.

    Runs locally via the shared model manager — no external calls.

    conversation:
        Optional transcript of prior turns, so follow-up requests are
        classified in context instead of on their literal words alone.
    """
    import logging
    logger = logging.getLogger(__name__)

    if not message or not message.strip():
        return None

    try:
        import ollama
        from config import LLM_MODEL, OLLAMA_HOST
        from models.manager import use_model

        client = ollama.Client(host=OLLAMA_HOST)
        conv_block = f"RECENT CONVERSATION:\n{conversation}\n\n" if conversation else ""
        prompt = (_LLM_CLASSIFY_PROMPT
                  .replace("{conversation_block}", conv_block)
                  .replace("{message}", message[:500]))

        with use_model(LLM_MODEL):
            resp = client.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                keep_alive="2m",
                options={"num_predict": 12, "temperature": 0.0},
            )

        raw = (resp.message.content or "").strip().lower()
        # Strip punctuation and any extra tokens
        token = raw.split()[0].strip(".,;:!?\"'`") if raw else ""

        valid = {"general", "coding", "document", "vision"}
        if token in valid:
            return token

        # Sometimes the model writes "capability: coding" — try a substring scan
        for v in valid:
            if v in raw:
                return v

        return None

    except Exception as exc:
        logger.warning("LLM classification failed (%s); falling back to keywords", exc)
        return None
# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    capability: str   # "general" | "coding" | "document" | "vision"
    reason: str


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_request(
    message: str,
    image_path: str | None = None,
    use_llm: bool = True,
    conversation: str | None = None,
) -> ClassificationResult:
    """
    Classify a user request into a capability.

    Strategy:
      1. If image_path is supplied → vision (short-circuit, no LLM needed).
      2. Try the LLM classifier (smarter, context-aware).
      3. Fall back to keyword matching if the LLM is unavailable or fails.

    Set use_llm=False to force the fast keyword path (used by tests).

    conversation:
        Optional transcript of prior turns — lets follow-ups ("make that a
        function") inherit the topic of the previous turn during LLM
        classification.
    """
    import re

    # Rule 1: image supplied → vision (always wins, no LLM)
    if image_path and image_path.strip():
        return ClassificationResult(
            capability="vision",
            reason="image_path supplied — routing to vision model",
        )

    # Rule 2: LLM classifier (with keyword fallback below)
    if use_llm:
        llm_cap = classify_request_llm(message, conversation=conversation)
        if llm_cap:
            return ClassificationResult(
                capability=llm_cap,
                reason=f"LLM classifier chose: {llm_cap}",
            )

    # Rule 3: keyword fallback
    msg_lower = message.lower()

    def _matches(keywords: set) -> str | None:
        for kw in keywords:
            if " " in kw:
                if kw in msg_lower:
                    return kw
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', msg_lower):
                    return kw
        return None

    kw = _matches(_CODING_KEYWORDS)
    if kw:
        return ClassificationResult(
            capability="coding",
            reason=f"keyword fallback: {kw!r}",
        )

    kw = _matches(_DOCUMENT_KEYWORDS)
    if kw:
        return ClassificationResult(
            capability="document",
            reason=f"keyword fallback: {kw!r}",
        )

    return ClassificationResult(
        capability="general",
        reason="keyword fallback: no match — general",
    )