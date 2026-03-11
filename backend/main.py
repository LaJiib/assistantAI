"""
AssistantIA — Backend FastAPI
GET /health  |  POST /chat  |  POST /chat/stream

Démarrage : python main.py
"""

import json
import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.llm import MinistralEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "")
HOST       = os.getenv("HOST", "127.0.0.1")
PORT       = int(os.getenv("PORT", "8000"))

app = FastAPI(title="AssistantIA Backend", version="0.1.0")


# ── Schémas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt:      str   = Field(..., min_length=1)
    max_tokens:  int   = Field(512, gt=0, le=4096)
    temperature: float = Field(0.3, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    response: str

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    logger.info("🚀 Démarrage AssistantIA backend…")
    if not MODEL_PATH:
        logger.error("❌ MODEL_PATH non défini (vérifiez .env)")
        sys.exit(1)
    engine = MinistralEngine(model_path=MODEL_PATH)
    await engine.load()
    app.state.engine = engine
    logger.info(f"📡 Écoute sur http://{HOST}:{PORT}")


@app.on_event("shutdown")
async def shutdown() -> None:
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: MinistralEngine | None = app.state.engine
    loaded = engine is not None and engine.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=engine.model_name if engine else "none",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Génère une réponse complète (non-streaming)."""
    engine: MinistralEngine | None = app.state.engine
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    try:
        return ChatResponse(response=engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Génère une réponse en streaming (Server-Sent Events).

    Format SSE :
        data: {"text": "chunk"}\n\n   (un chunk par token ou groupe)
        data: [DONE]\n\n              (fin de génération)

    Erreur mid-stream :
        data: {"error": "message"}\n\n
        data: [DONE]\n\n
    """
    engine: MinistralEngine | None = app.state.engine
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    def sse():
        try:
            for chunk in engine.stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")
