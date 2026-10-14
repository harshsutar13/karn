from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from models.llm import generate_response
from api.deliverable import router as deliverable_router
from api.sandbox_api import router as sandbox_router
from api.agent_api import router as agent_router
from api.planner_api import router as planner_router
from api.capabilities import router as capabilities_router
from api.code import router as code_router
from api.vision import router as vision_router
from api.ask import router as ask_router
from api.upload import router as upload_router
from api.coding_agent_api import router as coding_agent_router
from api.memory_api import router as memory_router
from api.rag import router as rag_router


app = FastAPI(
    title="SIH26 AI Workbench",
    version="0.2.0",
    description="Local AI Workbench — RAG, document ingestion, and grounded Q&A.",
)

app.include_router(deliverable_router)
app.include_router(sandbox_router)
app.include_router(agent_router)
app.include_router(planner_router)
app.include_router(capabilities_router)
app.include_router(code_router)
app.include_router(vision_router)
app.include_router(ask_router)
app.include_router(upload_router)
app.include_router(coding_agent_router)
app.include_router(memory_router)
app.include_router(rag_router)
from api.sovereignty import router as sovereignty_router
app.include_router(sovereignty_router)

from api.route_api import router as route_router
app.include_router(route_router)
from api.deliverables_api import router as deliverables_router
from api.planner_stream_api import router as planner_stream_router
from api.ide_api import router as ide_router
app.include_router(deliverables_router)
app.include_router(planner_stream_router)
app.include_router(ide_router)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


import os as _os

# Serve the frontend UI at GET /
_FRONTEND_DIR = _os.path.join(_os.path.dirname(__file__), "..", "frontend")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(_os.path.join(_FRONTEND_DIR, "index.html"))

@app.get("/ide", include_in_schema=False)
def ide():
    return FileResponse(_os.path.join(_FRONTEND_DIR, "ide.html"))

# Serve static assets (CSS, JS, images) at /static if needed in future
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    response = generate_response(request.message)

    return ChatResponse(response=response)