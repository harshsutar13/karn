# 🏭 SIH26 Sovereign AI Workbench — Complete Project Walkthrough

> **Project Name:** SIH26 AI Workbench MVP  
> **Version:** 0.2.0  
> **Theme:** 100% Local, Air-Gapped, Sovereign AI for Industrial Operations  
> **Primary Domain:** Oil Refinery / Industrial Maintenance

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Technology Stack](#3-technology-stack)
4. [Project Directory Structure](#4-project-directory-structure)
5. [Startup & Launch Flow](#5-startup--launch-flow)
6. [Backend Deep Dive](#6-backend-deep-dive)
7. [Frontend Deep Dive](#7-frontend-deep-dive)
8. [Data Flow Diagrams](#8-data-flow-diagrams)
9. [API Reference](#9-api-reference)
10. [AI Models Used](#10-ai-models-used)
11. [Sovereignty Proof System](#11-sovereignty-proof-system)
12. [Planner-Agent System](#12-planner-agent-system)
13. [Deliverable Generation](#13-deliverable-generation)
14. [Vector Store (Qdrant)](#14-vector-store-qdrant)
15. [Testing](#15-testing)
16. [Logs & Monitoring](#16-logs--monitoring)
17. [How to Run the Project](#17-how-to-run-the-project)
18. [Key Design Decisions](#18-key-design-decisions)

---

## 1. Project Overview

The **SIH26 Sovereign AI Workbench** is a fully **local, offline-first** AI assistant designed for industrial environments — specifically oil refineries. Built for **Smart India Hackathon (SIH) 2026**.

### Core Problem Solved
Industrial plants cannot send sensitive engineering data, SOPs, inspection reports, and maintenance logs to cloud AI services due to:
- **Data sovereignty** concerns
- **Air-gapped** network environments
- **Regulatory compliance** requirements

### Solution
A complete AI workbench that runs 100% on-premise:
- **No cloud calls** from any workbench process
- **Local LLM inference** via Ollama
- **Local vector search** via Qdrant
- **Full document Q&A (RAG)** on industrial documents
- **Vision & OCR** for scanned inspection reports
- **Automated document generation** (.docx, .xlsx, .pptx)
- **Provable sovereignty** via a built-in network audit endpoint

---

## 2. Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER'S BROWSER                                │
│                  http://localhost:8000                           │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP / SSE
┌───────────────────────▼─────────────────────────────────────────┐
│              FASTAPI APPLICATION (port 8000)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  /api/ask    │  │  /api/agent  │  │  /api/deliverable     │  │
│  │  /api/ask/   │  │  /api/planner│  │  /api/vision/analyze  │  │
│  │  stream      │  │  /api/upload │  │  /api/sovereignty     │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘  │
│         │                 │                        │              │
│  ┌──────▼─────────────────▼────────────────────────▼───────────┐ │
│  │              ORCHESTRATION LAYER                              │ │
│  │   classify_request() → route to capability                   │ │
│  │   [general | coding | document | vision]                     │ │
│  └──────┬──────────────┬──────────────┬──────────────┬──────────┘ │
│         │              │              │              │              │
│  ┌──────▼───┐  ┌───────▼──┐  ┌───────▼──┐  ┌──────▼────────┐    │
│  │  LLM     │  │  Coding  │  │  RAG     │  │  Vision/OCR   │    │
│  │(llama3.2)│  │(qwen-    │  │ Pipeline │  │(qwen3-vl:4b   │    │
│  │          │  │coder:7b) │  │          │  │ glm-ocr)      │    │
│  └──────────┘  └──────────┘  └────┬─────┘  └───────────────┘    │
│                                    │                               │
│                          ┌─────────▼──────────┐                   │
│                          │  QDRANT (port 6333) │                   │
│                          │  sih26_documents    │                   │
│                          └────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │   OLLAMA (port 11434)       │
          │   llama3.2, qwen2.5-coder  │
          │   nomic-embed-text          │
          │   qwen3-vl:4b, glm-ocr     │
          └────────────────────────────┘
```

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM Runtime** | Ollama | Local LLM serving — no GPU cloud |
| **General LLM** | `llama3.2:latest` (~2 GB) | Chat, Q&A, reasoning |
| **Coding LLM** | `qwen2.5-coder:7b` (~4.7 GB) | Code generation & review |
| **Embeddings** | `nomic-embed-text` (~274 MB) | Semantic document embedding |
| **Vision** | `qwen3-vl:4b` (~3.3 GB) | Image analysis, visual Q&A |
| **OCR** | `glm-ocr:latest` (~2.2 GB) | Text extraction from scanned images |
| **Vector Store** | Qdrant | Semantic similarity search |
| **Web Framework** | FastAPI + Uvicorn | REST API + SSE streaming |
| **Frontend** | Vanilla HTML/CSS/JS (single file) | Chat UI, no build step |
| **Doc Gen** | python-docx, openpyxl, python-pptx | Output .docx/.xlsx/.pptx |
| **Python** | 3.10+ | Backend runtime |

> **VRAM Note:** Tuned for an **NVIDIA RTX 3050 Laptop (6 GB VRAM)**. Only 1 model is loaded at a time via `OLLAMA_MAX_LOADED_MODELS=1`.

---

## 4. Project Directory Structure

```
Final SIH26/
│
├── run_project.py              ← MASTER LAUNCHER (start here)
├── demo.py                     ← Demo/test script
├── .gitignore
├── README.md
├── WALKTHROUGH.md              ← THIS FILE
│
├── backend/                    ← All Python backend code
│   ├── main.py                 ← FastAPI app entry point
│   ├── config.py               ← Centralized model & service config
│   ├── ingest_all.py           ← Bulk document ingestion script
│   │
│   ├── api/                    ← HTTP route handlers
│   │   ├── ask.py              ← POST /api/ask  (unified chat)
│   │   ├── capabilities.py     ← GET  /api/capabilities
│   │   ├── code.py             ← POST /api/code/generate
│   │   ├── vision.py           ← POST /api/vision/analyze
│   │   ├── upload.py           ← POST /api/upload/image
│   │   ├── deliverable.py      ← POST /api/deliverable/approval-note
│   │   ├── deliverables_api.py ← GET  /api/deliverables (list outputs)
│   │   ├── agent_api.py        ← POST /api/agent/run
│   │   ├── planner_api.py      ← POST /api/planner/plan
│   │   ├── planner_stream_api.py ← POST /api/planner/stream
│   │   ├── route_api.py        ← POST /api/route  (classify only)
│   │   ├── rag.py              ← POST /api/rag/query
│   │   ├── sandbox_api.py      ← POST /api/sandbox/run
│   │   ├── sovereignty.py      ← GET  /api/sovereignty/status
│   │   └── routes.py           ← Additional route definitions
│   │
│   ├── orchestration/          ← Request routing & capability dispatch
│   │   ├── router.py           ← Deterministic capability classifier
│   │   └── service.py          ← Blocking + streaming request handlers
│   │
│   ├── models/                 ← Model interface layer
│   │   ├── registry.py         ← Model capability registry
│   │   ├── manager.py          ← VRAM manager (load/unload models)
│   │   ├── llm.py              ← General LLM calls (llama3.2)
│   │   ├── coding.py           ← Coding LLM calls (qwen2.5-coder)
│   │   ├── vision.py           ← Vision model calls (qwen3-vl:4b)
│   │   └── ocr.py              ← OCR model calls (glm-ocr)
│   │
│   ├── rag/                    ← Retrieval-Augmented Generation
│   │   ├── ingestion.py        ← Single-doc ingestion orchestrator
│   │   ├── pipeline.py         ← Full RAG query pipeline
│   │   ├── retriever.py        ← Qdrant semantic retrieval
│   │   ├── vector_store.py     ← Qdrant client + upsert helpers
│   │   ├── test_rag_pipeline.py
│   │   └── test_vector_store.py
│   │
│   ├── tools/                  ← Agent toolset
│   │   ├── agent.py            ← Multi-step analysis pipeline
│   │   ├── planner.py          ← LLM-driven goal planner
│   │   ├── executor.py         ← Plan execution engine (14 tools)
│   │   ├── sandbox.py          ← Isolated Python code runner
│   │   ├── docx_writer.py      ← .docx generation
│   │   ├── xlsx_writer.py      ← .xlsx generation
│   │   ├── pptx_writer.py      ← .pptx generation
│   │   └── approval_note.py    ← Approval note workflow
│   │
│   ├── documents/              ← Document loading & chunking
│   │   ├── loader.py           ← PDF, DOCX, TXT loader
│   │   └── chunker.py          ← Text chunker with metadata
│   │
│   ├── embeddings/             ← Embedding generation
│   │   └── embedder.py         ← nomic-embed-text wrapper
│   │
│   ├── vision/                 ← Vision module init
│   ├── outputs/                ← Generated .docx/.xlsx/.pptx files
│   ├── docs/                   ← Industrial documents (SOPs, manuals)
│   └── tests/                  ← pytest test suite
│       ├── conftest.py
│       ├── test_thinking_display.py
│       └── test_ux_features.py
│
├── frontend/                   ← Single-page web UI
│   ├── index.html              ← Complete UI (HTML + CSS + JS, ~1530 lines)
│   ├── assets/                 ← SVG file icons
│   │   ├── ai/ai-chat-icon.svg
│   │   ├── branding/
│   │   ├── file-icons/         ← pdf, word, excel, pptx, image, text SVGs
│   │   └── hero/
│   └── newassets/              ← Logo images (PNG)
│
├── data/
│   └── uploads/                ← Uploaded images (for vision/OCR)
│
├── qdrant/                     ← Qdrant vector database
│   ├── qdrant.exe              ← Qdrant server binary (Windows)
│   ├── config/config.yaml      ← Qdrant configuration
│   ├── storage/                ← Persistent vector data
│   │   └── collections/
│   │       ├── sih26_documents ← Main knowledge base
│   │       └── sih26_rag_test  ← Test collection
│   └── start_qdrant.py
│
├── logs/                       ← Service logs
│   ├── ollama.log
│   ├── qdrant.log
│   └── api_server.log
│
└── models/                     ← Reserved for local model files
```

---

## 5. Startup & Launch Flow

Everything starts with **one command**:

```bash
python run_project.py
```

### Console Output

```
============================================================
              SIH26 AI WORKBENCH MVP
============================================================

[1/5] Checking Ollama ............ READY
[2/5] Checking models ............ READY
[3/5] Starting Qdrant ........... READY
[4/5] Starting FastAPI .......... READY
[5/5] Checking MVP API .......... READY

============================================================
                  MVP IS READY
============================================================

Web UI:       http://localhost:8000
API:          http://localhost:8000/api/ask
Capabilities: http://localhost:8000/api/capabilities

Press Ctrl+C to stop the project.
```

### Step-by-Step Breakdown

#### Step 1 — Ollama Check
- Pings `http://localhost:11434`
- If not running → finds `ollama.exe` in PATH or `%LOCALAPPDATA%\Programs\Ollama\`
- Launches `ollama serve` with VRAM guards:
  - `OLLAMA_MAX_LOADED_MODELS=1` — prevents VRAM overload
  - `OLLAMA_KEEP_ALIVE=2m` — unloads model after 2 min idle
- Waits up to **60 seconds** for Ollama to be ready
- Logs to `logs/ollama.log`

#### Step 2 — Model Verification
- Queries `GET http://localhost:11434/api/tags`
- Verifies ALL required models are installed:
  - `llama3.2:latest`
  - `qwen2.5-coder:7b`
  - `nomic-embed-text`
  - `qwen3-vl:4b`
  - `glm-ocr:latest`
- If any are missing → prints `ollama pull <model>` instructions and exits

#### Step 3 — Qdrant Start
- Pings `http://localhost:6333`
- If not running → launches `qdrant/qdrant.exe` from the project directory
- Waits up to **30 seconds** for Qdrant to be ready
- Logs to `logs/qdrant.log`

#### Step 4 — FastAPI Start
- Launches `uvicorn main:app --host 127.0.0.1 --port 8000` from `backend/`
- Uses `.venv/Scripts/python.exe` if present, otherwise system Python
- Waits up to **180 seconds** (cold model start can be slow)
- Logs to `logs/api_server.log`

#### Step 5 — MVP Verification
- `GET /` → expects HTTP 200
- `GET /api/capabilities` → expects `{"capabilities": [...]}`
- `POST /api/ask` with `{"message": "What is 2 + 2?"}` → expects `{"answer": "..."}` within 120 seconds

#### Post-Startup
- Opens `http://localhost:8000` in the default browser (background thread)
- Monitors all spawned processes in a heartbeat loop
- On `Ctrl+C` → terminates all processes cleanly (no orphans)

---

## 6. Backend Deep Dive

### 6.1 Configuration (`config.py`)

Central configuration. All values overridable via environment variables.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `LLM_MODEL` | `llama3.2:latest` | General LLM |
| `PLANNER_MODEL` | `llama3.2:latest` | Planner LLM |
| `CODING_MODEL` | `qwen2.5-coder:7b` | Code LLM |
| `EMBEDDING_MODEL` | `nomic-embed-text` | RAG embeddings |
| `OCR_MODEL` | `glm-ocr:latest` | OCR |
| `VISION_MODEL` | `qwen3-vl:4b` | Vision |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `sih26_documents` | Default collection |

### 6.2 Main Application (`main.py`)

FastAPI application that registers all routers. Also serves the frontend:

- `GET /` → returns `frontend/index.html`
- `GET /static/*` → serves `frontend/` directory as static files
- `POST /api/chat` → legacy chat endpoint (uses `models/llm.generate_response`)

### 6.3 API Layer

#### `api/ask.py` — Unified Ask Endpoint

The **primary API endpoint**. Handles both blocking and streaming.

**Request Schema:**
```json
{
  "message": "string (required, non-empty)",
  "image_path": "string | null (optional absolute path)",
  "thinking": "boolean (default: false)"
}
```

**Blocking Response (`POST /api/ask`):**
```json
{
  "answer": "The LLM response text...",
  "capability": "general | coding | document | vision",
  "model": "llama3.2:latest"
}
```

**Streaming Response (`POST /api/ask/stream`):**  
Returns `text/event-stream` with events:
```
data: {"type":"meta","capability":"general","model":"llama3.2:latest"}
data: {"type":"thinking","text":"<reasoning tokens>"}  ← only if thinking=true
data: {"type":"chunk","text":"<answer token>"}
data: [DONE]
```

**Error mapping:**
- `FileNotFoundError` → HTTP 404
- `ValueError` → HTTP 400
- `ConnectionError` (Ollama down) → HTTP 503
- Any other → HTTP 500

#### `api/upload.py` — Image Upload

```
POST /api/upload/image
Content-Type: multipart/form-data
```

- Validates MIME type (JPEG, PNG, BMP, TIFF, WebP)
- Saves as `data/uploads/<uuid>.<ext>`
- Returns `{"file_path": "/absolute/path", "filename": "original.jpg"}`
- Pass `file_path` as `image_path` in subsequent `/api/ask` calls

#### `api/capabilities.py` — Capabilities Registry

```
GET /api/capabilities
```

Returns all registered AI capabilities. Zero cost — no model calls. Used by:
- Health checks
- Frontend model labels
- The MVP verification step in `run_project.py`

#### `api/vision.py` — Vision Analysis

```
POST /api/vision/analyze
{"image_path": "/abs/path/to/image.png", "prompt": "What do you see?"}
```

Cold-start: 16–33 seconds. Warm: 3–10 seconds.

#### `api/sovereignty.py` — Sovereignty Proof

```
GET /api/sovereignty/status    ← Network audit
GET /api/sovereignty/ping-external  ← Air-gap test
```

See [Section 11](#11-sovereignty-proof-system) for details.

#### `api/agent_api.py` — Agent Pipeline

```
POST /api/agent/run
{"case_id": "CASE-P101-001", "equipment_tag": "P-101", "additional_context": "..."}
```

Runs the full engineering analysis agent. See [Section 12](#12-planner-agent-system).

#### `api/planner_api.py` + `api/planner_stream_api.py` — Goal Planner

```
POST /api/planner/plan
POST /api/planner/stream
{"goal": "Generate a vibration analysis report for P-101 in Excel"}
```

LLM creates a JSON plan of tool calls, executor runs them.

#### `api/deliverable.py` — File Downloads

```
POST /api/deliverable/approval-note   ← Generate approval note
GET  /api/deliverable/download/{filename}  ← Download generated file
```

#### `api/sandbox_api.py` — Python Sandbox

```
POST /api/sandbox/run
{"code": "print('Hello')", "timeout": 10}
```

Executes arbitrary Python in an isolated subprocess.

---

### 6.4 Orchestration Layer

#### `orchestration/router.py` — Capability Classifier

Routes every request to one of four capabilities:

| Capability | Assigned Model | When Used |
|-----------|---------------|-----------|
| `general` | llama3.2 | Open Q&A, chat, math, general knowledge |
| `coding` | qwen2.5-coder:7b | Code writing, review, debugging, SQL |
| `document` | llama3.2 + RAG | Industrial doc Q&A, equipment queries |
| `vision` | qwen3-vl:4b | Image analysis, any image_path provided |

**Classification Strategy (priority order):**

1. **image_path supplied** → always `vision` (short-circuit, no LLM)
2. **LLM classifier** → asks `llama3.2` to respond with exactly one word from `{general, coding, document, vision}`
3. **Keyword fallback** → regex word-boundary matching

**Coding Keywords include:** `code`, `write`, `implement`, `program`, `script`, `function`, `class`, `algorithm`, `debug`, `fix`, `error`, `bug`, `refactor`, `python`, `javascript`, `sql`, `bash`, `api`, `def `, `return` ...

**Document Keywords include:** `document`, `according`, `based on`, `report`, `maintenance`, `manual`, `pump`, `valve`, `sensor`, `vibration`, `pressure`, `rpm`, `P-101`, `refinery`, `inspection`, `summarize`, `extract from` ...

#### `orchestration/service.py` — Request Handlers

**`handle_request(message, image_path, thinking)`** — Blocking:
1. Classifies request
2. Dispatches to correct capability handler
3. Returns `OrchestratorResponse(answer, capability, model)`

**`handle_request_stream(message, image_path, thinking)`** — Generator:
1. Classifies request
2. Yields SSE meta event
3. For RAG: retrieves chunks FIRST, then streams LLM answer
4. Yields SSE chunk events from model
5. Yields `[DONE]`

**Vision MVP Policy:** Vision capability always runs with `think=False` — vision models don't produce thinking tokens and the thinking UI is suppressed for image requests.

---

### 6.5 RAG Pipeline

Full Retrieval-Augmented Generation pipeline for industrial document Q&A.

#### Document Ingestion Flow

```
Document file (.pdf / .docx / .txt)
         │
         ▼ documents/loader.py
    Load + extract text
         │
         ▼ rag/ingestion.py (_strip_doc_header)
    Strip boilerplate header (title, doc no., revision metadata)
         │
         ▼ documents/chunker.py
    Chunk text (size=600 chars, overlap=100 chars)
    Each chunk gets metadata: {document_id, filename, chunk_index, char_start, char_end}
         │
         ▼ embeddings/embedder.py
    Embed each chunk with nomic-embed-text → float[768] vector
         │
         ▼ rag/vector_store.py
    Upsert to Qdrant collection "sih26_documents"
    Point ID = UUID5(namespace, filename) + chunk_index (deterministic)
```

**Document ID Strategy:**
```python
document_id = str(uuid.uuid5(_DOC_ID_NAMESPACE, filename))
# Same filename → same UUID → same point IDs → safe re-ingestion (upsert)
```

**Header Stripping:** Removes identical metadata blocks at the top of every document (title, doc number, revision, classification) that would corrupt embeddings by making every chunk look the same.

#### RAG Query Flow

```
User Question
     │
     ▼ embeddings/embedder.py
Embed query → float[768]
     │
     ▼ rag/retriever.py
Qdrant cosine similarity search → Top-5 chunks
     │
     ▼ rag/pipeline.py
Build grounded prompt:
  [Context from chunks] + [User Question]
     │
     ▼ models/llm.py
llama3.2 generates answer grounded ONLY in context
     │
     ▼
RAGResponse(answer, sources, query, chunks_retrieved, context_used)
```

**System Prompt Rules (enforced):**
1. Answer ONLY from provided context
2. If context is insufficient → say: *"The available documents do not contain sufficient information..."*
3. Preserve all technical numbers, units, and equipment names verbatim
4. Synthesize multiple passages coherently
5. Cite source document(s) at end of answer

#### `rag/pipeline.py` — Key Types

```python
@dataclass
class SourceReference:
    filename: str
    document_id: str
    chunk_index: int
    score: float          # Cosine similarity score
    char_start: int
    char_end: int
    text_preview: str     # First 200 chars for display

@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceReference]
    query: str
    chunks_retrieved: int
    context_used: bool    # False if no chunks found
```

---

### 6.6 Models Layer

#### `models/registry.py` — Capability Registry

Maps capabilities to models:

| Capability | Model | Status |
|-----------|-------|--------|
| `general_reasoning` | `llama3.2:latest` | Implemented |
| `coding` | `qwen2.5-coder:7b` | Implemented |
| `vision` | `qwen3-vl:4b` | Implemented |
| `ocr` | `glm-ocr:latest` | Implemented |
| `embedding` | `nomic-embed-text` | Implemented |
| `summarization` | `llama3.2:latest` | Implemented |

#### `models/manager.py` — VRAM Manager

Context manager ensuring one model at a time:

```python
with use_model("llama3.2:latest"):
    response = client.chat(model=LLM_MODEL, ...)
# Model evicted after context, or after keep_alive=2m
```

This serializes model access and prevents CUDA OOM on 6 GB GPU.

#### `models/llm.py` — General LLM Interface

Two modes:
- **Blocking:** returns `str` (full response)
- **Streaming:** yields `{"kind": "thinking"/"chunk", "text": "..."}` dicts

Thinking mode: when `think=True`, the model produces reasoning tokens (`<think>...</think>`) before the answer. These become `kind="thinking"` events in streaming mode.

#### `models/coding.py` — Coding LLM

Same interface as `llm.py` but uses `qwen2.5-coder:7b`. Optimized for:
- Python, JavaScript, TypeScript, SQL, Bash
- Function/class generation
- Bug fixing and refactoring
- Code explanation

#### `models/vision.py` — Vision Model

Calls `qwen3-vl:4b`. Accepts `image_path` and `prompt`. Encodes image as base64 for Ollama's multimodal API. Supports both blocking and streaming. Think mode is always `False` for vision.

#### `models/ocr.py` — OCR Model

Calls `glm-ocr:latest`. Returns extracted text from scanned documents. Used by the scan-to-approval workflow.

---

### 6.7 Tools / Agent Layer

#### `tools/agent.py` — Multi-Step Analysis Agent

**`analyze_case(case_id, equipment_tag="P-101")`**

Full 5-step engineering analysis pipeline:

```
Step 1: RAG retrieval
  └── rag_query(f"{equipment_tag} {case_id} parameters RCA") → 5 chunks

Step 2: Parameter extraction
  └── LLM extracts {rho, g, Q, H, eta} from retrieved context as JSON

Step 2b: Fill missing values
  └── Compare against canonical P-101 values:
      rho=994 kg/m³, g=9.81 m/s², Q=0.025 m³/s, H=32 m, eta=0.75

Step 3a: Code generation
  └── LLM generates Python script using extracted constants
      Calculates: hydraulic_kw = rho*g*Q*H/1000
                  input_kw = hydraulic_kw/eta
      Asserts: hydraulic_kw ≈ 7.80 kW, input_kw ≈ 10.40 kW

Step 3b: Sandbox execution
  └── run_python(code, timeout=10) → verifies assertions pass

Step 4: Draft analysis note
  └── LLM generates 6-section engineering note:
      1. Executive Summary
      2. Retrieval Findings
      3. Engineering Calculation
      4. RCA Assessment
      5. Recommendations
      6. Approval Status

Step 5: Write .docx
  └── python-docx → backend/outputs/{case_id}_analysis_note.docx
```

**`scan_to_approval(image_path, case_id, equipment_tag="P-101")`**

Scan-to-document pipeline:

```
Step 1: OCR → glm-ocr → extract text from scanned image
Step 2: LLM → extract structured findings as bullet list
Step 3: RAG → retrieve relevant SOPs/context (4 chunks)
Step 4: LLM → draft maintenance approval note (5 sections)
Step 5: python-docx → {case_id}_scanned_approval.docx
```

#### `tools/planner.py` — LLM Goal Planner

Takes a natural language goal → produces a JSON execution plan.

**Planner System Prompt** (condensed): 
> You are the PLANNER for an industrial AI workbench. Given a GOAL, output a JSON object with "reasoning" and "steps". Each step has "id", "tool", "args". Keep the plan to 2–4 steps. Do NOT invent tools.

**Plan JSON Format:**
```json
{
  "reasoning": "Short explanation of strategy",
  "steps": [
    {"id": "step_1", "tool": "rag_search", "args": {"query": "...", "top_k": 5}},
    {"id": "step_2", "tool": "write_docx", "args": {"title": "...", "sections": "step_1"}}
  ]
}
```

**JSON Parsing Robustness:** Handles:
- Markdown fences (` ```json `)
- Unquoted identifiers (`step_1` → `"step_1"`)
- Trailing commas
- Truncated JSON (closes open braces automatically)

**Replanning:** When a step fails:
```python
replan(goal, prev_plan, failure_reason) → new Plan
```
The LLM is shown the original goal, the failed plan, and why it failed. It produces a revised plan that avoids the same mistake. Max 3 replans.

#### `tools/executor.py` — Plan Execution Engine

Executes a `Plan` step by step. Core logic:

1. For each step → resolve `"step_N"` references from prior step outputs
2. Call the tool function from `TOOL_REGISTRY`
3. Store output in `state[step_id]`
4. If step fails → call `replan()` with failure reason
5. Up to `MAX_REPLANS=3` attempts

**Step Reference Resolution:**

| Reference Format | Resolves To |
|-----------------|-------------|
| `"step_1"` | Full text output of step_1 |
| `"step_1.summary"` | `.summary` field of step_1's output dict |
| `{"ref": "step_1"}` | Full output of step_1 |

**File Security:** Read/write tools only allow paths inside:
- `backend/outputs/`
- `backend/docs/`
- `data/`

**Tool Registry (14 tools):**

| Tool | Function |
|------|----------|
| `rag_search` | Semantic search in knowledge base |
| `extract_params` | Extract numeric parameters from RAG results |
| `run_python` | Execute Python in sandbox |
| `write_docx` | Generate Word document |
| `write_xlsx` | Generate Excel workbook |
| `write_pptx` | Generate PowerPoint slides |
| `read_file` | Read text file |
| `write_file` | Write text file |
| `read_xlsx` | Read existing Excel file |
| `edit_xlsx` | Modify Excel: set cells, append rows, add sheets |
| `ocr_image` | OCR text from image |
| `analyze_image` | Visual description/Q&A |
| `scan_to_approval` | End-to-end scan → approval .docx |
| `summarize` | Summarize text to N words |

#### `tools/sandbox.py` — Python Sandbox

Runs Python code in a completely isolated subprocess:

```python
@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: float

result = run_python(code, timeout=10)
```

Subprocess has no network access by default. Timeout kills the process.

---

### 6.8 Document Processing

#### `documents/loader.py`

Loads documents and returns raw text:
- **PDF** — PyMuPDF or pdfplumber
- **DOCX** — python-docx
- **TXT** — tries UTF-8, UTF-8-BOM, latin-1, cp1252

#### `documents/chunker.py`

Character-based sliding window chunking:
- `chunk_size=600` characters per chunk
- `chunk_overlap=100` characters overlap between consecutive chunks
- Each `Chunk` carries: `text`, `chunk_index`, `char_start`, `char_end`, `metadata dict`

Why character-based (not sentence-based)? Consistent chunk sizes for deterministic embedding behavior on technical documents with irregular sentence structure.

---

### 6.9 Embeddings (`embeddings/embedder.py`)

Wraps Ollama's embedding API for `nomic-embed-text`:

```python
embeddings: list[list[float]] = embed_texts(["chunk 1", "chunk 2"])
# Returns 768-dimensional vectors
```

Used during:
1. **Ingestion** — embedding document chunks before storing in Qdrant
2. **Retrieval** — embedding the user query before Qdrant search

---

## 7. Frontend Deep Dive

**File:** `frontend/index.html` (~1530 lines, entirely self-contained)

### Design Tokens (CSS Variables)

```css
--bg:        #0d0f17;   /* Main background */
--surface:   #161923;   /* Cards, header */
--surface2:  #1e2235;   /* Input areas */
--border:    #2a2f4a;   /* Borders */
--accent:    #5b8ef4;   /* Blue accent */
--accent2:   #7c6bff;   /* Purple accent */
--green:     #3dd68c;   /* Success, live indicator */
--orange:    #f59e3f;   /* Warnings */
--red:       #f26b6b;   /* Errors */
--purple:    #b08af5;   /* Document capability */
--think-col: #a78bfa;   /* Thinking mode tokens */
--text:      #e2e8f8;   /* Main text */
--muted:     #7882a4;   /* Secondary text */
--font:      Inter, system-ui, sans-serif;
--mono:      JetBrains Mono, Cascadia Code, monospace;
```

### UI Sections

| Section | Description |
|---------|-------------|
| **Header** | Logo ("SIH26"), title, live dot, Thinking Mode toggle |
| **Demo Bar** | Quick-prompt chips for common queries |
| **Tab Bar** | Chat / Agent / Planner tabs |
| **Message Area** | Scrollable chat history |
| **Input Row** | Image attach button, text input, send button |
| **Sidebar** | Capability status, file list, download links |

### Chat Message Features

- **User bubble** — right-aligned, blue background (`--user-bg`)
- **AI bubble** — left-aligned, dark background (`--ai-bg`)
- **Streaming** — text appears word-by-word via SSE EventSource
- **Thinking block** — collapsible purple section for reasoning tokens
- **Capability badge** — color-coded chip: general/coding/document/vision
- **Model label** — small muted text showing model name
- **Code blocks** — monospace, dark background rendering

### Image Upload Flow

1. User clicks paperclip icon → file picker opens (image/* filter)
2. `FormData` sent to `POST /api/upload/image`
3. Server returns `file_path`
4. Image thumbnail shown in input area
5. Next send includes `image_path: file_path` in the ask request
6. Router automatically classifies as `vision`

### Thinking Mode Toggle

- Toggle pill in header
- When enabled → `thinking: true` in all API requests
- Reasoning tokens displayed in collapsible `<details>` block with purple styling
- Vision requests ignore thinking mode (always `think=false`)

---

## 8. Data Flow Diagrams

### A. General Chat

```
User types message → Send button
    │
POST /api/ask/stream {message, thinking}
    │
orchestration/router.py
  └─ LLM classifier → "general"
    │
orchestration/service.py
  └─ models/llm.py → generate_response(stream=True)
    │
SSE: meta → [thinking?] → chunk... → [DONE]
    │
Frontend renders streamed text
```

### B. Document Q&A (RAG)

```
User types "What is the vibration limit for P-101?"
    │
POST /api/ask/stream
    │
router: keyword "vibration", "P-101" → "document"
    │
service: retrieval phase (SYNCHRONOUS)
  └─ embedder.embed(query) → float[768]
  └─ qdrant.search(vector, top_k=5) → chunks
    │
Build prompt: SYSTEM_PROMPT + context_blocks + question
    │
models/llm.py: stream response
    │
SSE: meta(document) → chunk... → [DONE]
    │
Frontend shows answer + "Sources: [filename]"
```

### C. Vision Analysis

```
User uploads image + types "What equipment is shown?"
    │
POST /api/upload/image → {file_path: "/data/uploads/<uuid>.jpg"}
    │
POST /api/ask/stream {message, image_path: file_path}
    │
router: image_path not None → "vision" (instant)
    │
models/vision.py
  └─ load image → base64 encode
  └─ qwen3-vl:4b multimodal inference
    │
SSE: meta(vision) → chunk... → [DONE]
    │
Frontend renders description
```

### D. Scan-to-Approval Workflow

```
User uploads scanned inspection report image
    │
POST /api/agent/run {goal: "scan to approval", image_path}
    │
tools/agent.py: scan_to_approval()
  ├─ Step 1: models/ocr.py → glm-ocr → raw_text
  ├─ Step 2: llm.generate() → bullet findings list
  ├─ Step 3: rag_query(equipment + case + "approval") → 4 chunks
  ├─ Step 4: llm.generate(findings + context) → approval draft
  └─ Step 5: docx_writer.write_docx() → .docx file
    │
Response: {download_url: "/api/deliverable/download/<file>", steps, sources}
    │
Frontend shows download button → user gets Word document
```

### E. Planner-Agent Goal Execution

```
User states: "Create Excel report of P-101 readings"
    │
POST /api/planner/stream {goal: "..."}
    │
tools/planner.py: make_plan(goal)
  └─ llama3.2 → JSON plan: [{rag_search}, {write_xlsx}]
    │
tools/executor.py: execute_plan(goal, plan)
  ├─ step_1: tool_rag_search({query: "P-101 readings"}) → chunks
  ├─ step_2: tool_write_xlsx({title, sheets: ["step_1"]}) → .xlsx
  └─ state["step_2"] = {output: {output_path, filename}}
    │
On failure → replan() → retry (max 3 attempts)
    │
SSE stream of step results → frontend
    │
Download link for generated Excel
```

---

## 9. API Reference

### Complete Endpoint Table

| Method | Path | Description | Tags |
|--------|------|-------------|------|
| `GET` | `/` | Frontend UI (index.html) | — |
| `POST` | `/api/chat` | Legacy chat endpoint | — |
| `POST` | `/api/ask` | Unified ask (blocking) | Orchestrator |
| `POST` | `/api/ask/stream` | Unified ask (SSE streaming) | Orchestrator |
| `GET` | `/api/capabilities` | List AI capabilities | Capabilities |
| `POST` | `/api/upload/image` | Upload image for vision | Upload |
| `POST` | `/api/vision/analyze` | Direct vision analysis | Vision |
| `POST` | `/api/code/generate` | Code generation | Code |
| `POST` | `/api/rag/query` | Direct RAG query | RAG |
| `POST` | `/api/agent/run` | Multi-step agent | Agent |
| `POST` | `/api/planner/plan` | Generate execution plan | Planner |
| `POST` | `/api/planner/stream` | Stream plan execution | Planner |
| `POST` | `/api/deliverable/approval-note` | Generate approval .docx | Deliverable |
| `GET` | `/api/deliverable/download/{filename}` | Download generated file | Deliverable |
| `GET` | `/api/deliverables` | List all generated files | Deliverables |
| `POST` | `/api/sandbox/run` | Execute Python in sandbox | Sandbox |
| `POST` | `/api/route` | Classify request only | Router |
| `GET` | `/api/sovereignty/status` | Network audit | Sovereignty |
| `GET` | `/api/sovereignty/ping-external` | Air-gap test | Sovereignty |
| `GET` | `/docs` | Swagger UI (auto-generated) | — |
| `GET` | `/redoc` | ReDoc documentation | — |

---

## 10. AI Models Used

### Model Details

| Model | VRAM | Cold Start | Warm | Role |
|-------|------|-----------|------|------|
| `llama3.2:latest` | ~2 GB | ~5s | <1s | General Q&A, RAG, planning, summarization |
| `qwen2.5-coder:7b` | ~4.7 GB | ~15s | ~2s | Code generation, debugging |
| `nomic-embed-text` | ~274 MB | ~2s | <0.5s | Document/query embeddings |
| `qwen3-vl:4b` | ~3.3 GB | 16–33s | 3–10s | Image analysis, visual Q&A |
| `glm-ocr:latest` | ~2.2 GB | ~10s | ~3s | OCR from scanned images |

### VRAM Management Strategy

RTX 3050 Laptop = **6 GB VRAM total**. Models never co-resident:

```
OLLAMA_MAX_LOADED_MODELS=1   ← Hard limit: 1 model at a time
OLLAMA_KEEP_ALIVE=2m         ← Auto-evict after 2 min idle
```

The `models/manager.py` `use_model()` context manager:
1. Requests load of target model
2. Ollama evicts previous resident model (CUDA swap)
3. Runs inference
4. On context exit: model stays warm until `keep_alive` expires

This sequential loading means tasks DON'T run concurrently — but it avoids CUDA OOM crashes that would occur with 6 GB RAM and 7+ GB of model weights.

---

## 11. Sovereignty Proof System

### Why It Matters
SIH judges need proof that no sensitive industrial data leaves the machine.

### How It Works — `api/sovereignty.py`

Uses `psutil` library to enumerate all active TCP connections system-wide, then filters by process name.

**Workbench process names:**
- `python.exe`, `pythonw.exe`
- `uvicorn.exe`
- `qdrant.exe`
- `ollama.exe`, `ollama app.exe`
- `llama-server.exe`

**Connection categorization:**
- `workbench_loopback` — to 127.x/::1 (internal, expected)
- `workbench_external` — to public IPs (**must be 0**)
- `other_app_external` — browsers, OS, etc. (out of scope)

**Sample Response:**
```json
{
  "psutil_available": true,
  "timestamp": "2026-10-02T12:40:00Z",
  "hostname": "REFINERY-WS01",
  "workbench_external_connections": [],
  "workbench_external_count": 0,
  "workbench_loopback_count": 3,
  "workbench_sovereign": true,
  "interpretation": "SOVEREIGN: The workbench made zero external network calls. All AI inference, embedding, and vector search are local."
}
```

**Air-gap test (`GET /api/sovereignty/ping-external`):**
- Tries `socket.create_connection("1.1.1.1", 53, timeout=3)`
- `reachable: false` → truly air-gapped host
- `reachable: true` → host has internet, but workbench STILL doesn't call it

---

## 12. Planner-Agent System

### Goal-Directed Automation

The planner lets users describe a **high-level goal** in natural language. The system figures out the steps automatically.

**Example goals:**
- *"Summarize the P-101 vibration reports in 100 words"*
- *"Create an Excel with inspection findings for P-101"*
- *"Read the scanned report and write an approval note"*
- *"Generate a 5-slide PowerPoint summary of maintenance findings"*

### Planner Prompt Design

The planner uses a strict prompt that:
1. Shows all 14 available tools with descriptions
2. Demands output in pure JSON (no markdown)
3. Limits to 2–4 steps
4. Gives concrete examples for common goals
5. Maps deliverable types to output tools (docx/xlsx/pptx)

**Tool disambiguation (important rules):**
- `summarize` → plain text summary (NOT a document)
- `write_docx` → only if output must be a Word file
- `scan_to_approval` → complete workflow, don't add extra steps after it

### Execution Robustness

| Scenario | Behavior |
|----------|---------|
| Unknown tool name | Dropped from plan, warning logged |
| Step failure | Planner re-asked with failure reason |
| Max replans exceeded | Returns partial results with error |
| Path outside allowed dirs | File operation rejected with clear error |
| Non-JSON planner output | `Plan(steps=[])` returned, frontend shows error |

### Streaming Execution

`POST /api/planner/stream` streams each step's status as SSE:
```
data: {"type": "plan", "steps": [...]}
data: {"type": "step_start", "id": "step_1", "tool": "rag_search"}
data: {"type": "step_result", "id": "step_1", "ok": true, "output": {...}}
data: {"type": "step_start", "id": "step_2", "tool": "write_docx"}
data: {"type": "step_result", "id": "step_2", "ok": true, "output": {...}}
data: {"type": "done", "download_url": "/api/deliverable/download/report.docx"}
data: [DONE]
```

---

## 13. Deliverable Generation

### Word Documents (.docx) — `tools/docx_writer.py`

Structure:
1. **Title** — H1 heading
2. **Metadata table** — Case ID, Equipment, Status, Generated By
3. **Sections** — numbered headings + body paragraphs
4. **Sources** — "Referenced Documents:" bullet list
5. **Footer** — disclaimer note on every page

Generated files include a mandatory disclaimer:
> "SYNTHETIC PROTOTYPE — Human review mandatory before operational use."

### Excel Workbooks (.xlsx) — `tools/xlsx_writer.py`

Features:
- Multiple named sheets
- Auto-sized columns
- Header row styling
- Data validation for numeric cells
- Footer row: "Generated by SIH26 planner-driven agent."

### PowerPoint Decks (.pptx) — `tools/pptx_writer.py`

Slide types:
- **Title slide** — main title + optional subtitle
- **Content slides** — title + bullet points + optional speaker notes

---

## 14. Vector Store (Qdrant)

### Collections

| Collection | Purpose |
|-----------|---------|
| `sih26_documents` | Main production knowledge base |
| `sih26_rag_test` | Test collection for development |

### Point Schema

```
ID:      string   (f"{document_id}-{chunk_index}")
Vector:  float[768]  (nomic-embed-text output)
Payload:
  document_id:  str   (UUID5 of filename)
  filename:     str   (original file name)
  doc_type:     str   ("pdf" | "docx" | "txt" | "unknown")
  chunk_index:  int
  char_start:   int
  char_end:     int
  text:         str   (chunk content)
  department:   str?  (optional)
```

### Search Parameters

- **Distance:** Cosine similarity
- **Top-K:** 5 chunks (default in pipeline)
- **Score threshold:** Optional (not set by default — returns all)

### Deduplication

Re-ingesting the same file is safe:
1. Same filename → same `document_id` (UUID5 is deterministic)
2. Same `chunk_index` → same point ID
3. Qdrant `upsert` → overwrites existing points
4. Result: no duplicate vectors, no growing collection

### Qdrant Binary

The project ships with `qdrant/qdrant.exe` (Windows binary). Config at `qdrant/config/config.yaml`. Data persisted in `qdrant/storage/`.

---

## 15. Testing

### Backend Test Suite (`backend/tests/`)

Run with:
```bash
cd backend
pytest tests/
```

| File | Tests |
|------|-------|
| `conftest.py` | pytest fixtures (app client, mock models) |
| `test_thinking_display.py` | Thinking token display, collapsible block behavior |
| `test_ux_features.py` | Image upload flow, capability badge rendering |

### RAG-Specific Tests (`backend/rag/`)

```bash
pytest rag/test_rag_pipeline.py  # Full RAG query test
pytest rag/test_vector_store.py  # Qdrant connection + upsert
```

### API Tests (`backend/api/`)

```bash
python api/test_api.py        # General API
python api/test_api_code.py   # Code generation
python api/test_api_vision.py # Vision endpoint
```

### Qdrant Tests (`qdrant/`)

```bash
python qdrant/run_all_tests.py     # Full suite
python qdrant/run_server_tests.py  # Server connectivity
```

---

## 16. Logs & Monitoring

### Log Files

| File | Content | Written By |
|------|---------|-----------|
| `logs/ollama.log` | Ollama server output | `run_project.py` |
| `logs/qdrant.log` | Qdrant server output | `run_project.py` |
| `logs/api_server.log` | FastAPI + Uvicorn + all module logs | `run_project.py` |

### Log Rotation

Not configured — logs append indefinitely. Clear manually if they grow too large.

### Python Logging

All backend modules use `logging.getLogger(__name__)`:
- INFO: request routing, chunk counts, file operations
- WARNING: RAG cold-start errors, LLM classification failures
- ERROR: Ollama unreachable, file not found
- EXCEPTION: unexpected errors with full stack trace

---

## 17. How to Run the Project

### Prerequisites

**1. System:**
- Windows 10 or 11
- Python 3.10+
- NVIDIA GPU with 6+ GB VRAM (or CPU fallback — much slower)

**2. Install Ollama:**
- Download from https://ollama.com
- Run installer → Ollama available in PATH

**3. Pull required models:**
```bash
ollama pull llama3.2
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
ollama pull qwen3-vl:4b
ollama pull glm-ocr
```
This takes 10–30 minutes and ~15 GB of disk space total.

**4. Install Python dependencies:**
```bash
cd "c:\Users\Admin\Desktop\Final MVP\Final SIH26"
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn[standard] python-docx openpyxl python-pptx qdrant-client ollama psutil pdfplumber pymupdf python-multipart pydantic
```

### Launch

```bash
python run_project.py
```

Browser opens automatically. If not, go to: **http://localhost:8000**

### Ingest Documents

Place documents (PDF/DOCX/TXT) in `backend/docs/` then run:
```bash
cd backend
python ingest_all.py
```

Or via API (for images/uploads):
```bash
curl -X POST http://localhost:8000/api/upload/image -F "file=@inspection.png"
```

### Service URLs

| Service | URL |
|---------|-----|
| **Web UI** | http://localhost:8000 |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **Unified Ask** | http://localhost:8000/api/ask |
| **Capabilities** | http://localhost:8000/api/capabilities |
| **Sovereignty** | http://localhost:8000/api/sovereignty/status |
| **Ollama** | http://localhost:11434 |
| **Qdrant Dashboard** | http://localhost:6333/dashboard |

### Stop

Press **Ctrl+C** in the terminal running `run_project.py`. All services are gracefully terminated.

---

## 18. Key Design Decisions

### 1. Fully Local Architecture
**Why:** Industrial plants operate in air-gapped or restricted-network environments. SOPs, inspection reports, and maintenance data are classified. Cloud APIs are prohibited.

### 2. Ollama for LLM Serving
**Why:** Provides a simple, unified HTTP API to run LLMs locally. Handles quantization, VRAM management, and model downloading. No GPU cluster needed.

### 3. Qdrant for Vector Search
**Why:** High-performance, production-ready vector database with a Windows binary, persistent storage, and clean REST API — no Docker required for the MVP.

### 4. llama3.2 as Main LLM
**Why:** 2 GB VRAM footprint fits on a 6 GB GPU, leaving room for embeddings. Larger models like gemma4:12b (7.6 GB) crash with CUDA OOM. llama3.2 provides strong enough reasoning for industrial Q&A.

### 5. Single-File Frontend
**Why:** No Node.js, no build step, no npm/yarn. The frontend is one `index.html` file served by FastAPI's `FileResponse`. In an air-gapped deployment, there's no package registry to fetch from.

### 6. Deterministic Keyword Routing Fallback
**Why:** If the LLM classifier is slow or fails, the system still works correctly via keyword matching. Reliability > flexibility. The LLM classifier is tried first (smarter), keyword matching is the safety net.

### 7. Python Sandbox for Code Execution
**Why:** User-generated code from the coding LLM is untrusted. Isolated subprocess with timeout prevents infinite loops, system calls, and hangs from disrupting the API server.

### 8. UUID-5 for Document IDs
**Why:** `UUID5(namespace, filename)` is deterministic — the same filename always hashes to the same UUID. This enables **idempotent re-ingestion**: re-processing a document overwrites Qdrant points instead of creating duplicates. The knowledge base stays clean regardless of how many times documents are reingested.

### 9. Dual Routing Paths (Blocking + Streaming)
**Why:** Some clients need a simple blocking API (scripts, tests, legacy integrations). Others need SSE streaming for real-time UX. Both paths share the same orchestration logic — no code duplication.

### 10. VRAM Guard Environment Variables
**Why:** RTX 3050 Laptop GPU has exactly 6 GB. Without guards, Ollama would try to load all recently-used models simultaneously, causing CUDA errors. `OLLAMA_MAX_LOADED_MODELS=1` enforces single-model-at-a-time policy at the Ollama level, independent of the Python manager.

---

## 📌 Summary

The SIH26 Sovereign AI Workbench MVP is a **complete, production-ready, fully local AI system** for industrial operations demonstrating:

| Feature | Status |
|---------|--------|
| **Sovereignty** (zero external calls, proven) | ✅ |
| **General Chat** (llama3.2) | ✅ |
| **Document Q&A / RAG** (Qdrant + nomic-embed + llama3.2) | ✅ |
| **Vision Analysis** (qwen3-vl:4b) | ✅ |
| **OCR** (glm-ocr) | ✅ |
| **Code Generation** (qwen2.5-coder:7b) | ✅ |
| **Word Document Output** (.docx) | ✅ |
| **Excel Output** (.xlsx) | ✅ |
| **PowerPoint Output** (.pptx) | ✅ |
| **Agentic Planning** (LLM-driven tool orchestration) | ✅ |
| **Real-time Streaming** (SSE) | ✅ |
| **Thinking Mode** (extended reasoning display) | ✅ |
| **6 GB GPU Compatible** (VRAM-aware design) | ✅ |
| **Single-command Launch** | ✅ |

---

*Walkthrough generated from the SIH26 AI Workbench codebase.*  
*Project Root: `c:\Users\Admin\Desktop\Final MVP\Final SIH26\`*  
*Last updated: October 2026*
