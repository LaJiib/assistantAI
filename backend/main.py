"""
AssistantIA — Backend FastAPI
GET /health  |  POST /chat  |  POST /chat/stream

Démarrage : python main.py
"""

import json
import logging
import os
import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager, suppress

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.llm import MinistralEngine
from api.conversations import router as conversations_router
from api.messages import router as messages_router
from storage.json_manager import JSONManager

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
DATA_FOLDER = os.getenv("DATA_FOLDER", "").strip()
MAX_GENERATION_TOKENS = int(os.getenv("MAX_GENERATION_TOKENS", "512"))
if MAX_GENERATION_TOKENS <= 0:
    logger.warning("MAX_GENERATION_TOKENS invalide (%s), fallback à 512", MAX_GENERATION_TOKENS)
    MAX_GENERATION_TOKENS = 512
MAX_GENERATION_TEMPERATURE = float(os.getenv("MAX_GENERATION_TEMPERATURE", "0.3"))
if MAX_GENERATION_TEMPERATURE < 0.0 or MAX_GENERATION_TEMPERATURE > 2.0:
    logger.warning(
        "MAX_GENERATION_TEMPERATURE invalide (%s), fallback à 0.3",
        MAX_GENERATION_TEMPERATURE,
    )
    MAX_GENERATION_TEMPERATURE = 0.3



# ── Schémas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt:      str   = Field(..., min_length=1)
    max_tokens:  int   = Field(MAX_GENERATION_TOKENS, gt=0, le=32768)
    temperature: float = Field(MAX_GENERATION_TEMPERATURE, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    response: str

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str
    pid:          int


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("🚀 Démarrage AssistantIA backend…")

    if not DATA_FOLDER:
        logger.error("❌ DATA_FOLDER non défini (vérifiez .env)")
        raise RuntimeError("DATA_FOLDER non défini")

    jm = JSONManager(Path(DATA_FOLDER))
    app.state.json_manager = jm
    jm.verify_integrity()

    app.state.engine = None
    app.state.model_loading = False
    app.state.model_error = None
    app.state.model_task = None
    app.state.max_generation_tokens = MAX_GENERATION_TOKENS
    app.state.max_generation_temperature = MAX_GENERATION_TEMPERATURE

    if not MODEL_PATH:
        app.state.model_error = "MODEL_PATH non défini"
        logger.error("❌ MODEL_PATH non défini (vérifiez .env)")
    else:
        app.state.model_loading = True

        async def _load_model():
            try:
                engine = MinistralEngine(model_path=MODEL_PATH)
                await engine.load()
                app.state.engine = engine
                logger.info("✅ Modèle chargé : %s", engine.model_name)
            except Exception as exc:
                app.state.model_error = str(exc)
                logger.exception("❌ Échec chargement modèle")
            finally:
                app.state.model_loading = False

        app.state.model_task = asyncio.create_task(_load_model())

    logger.info(f"📡 Écoute sur http://{HOST}:{PORT}")
    logger.info("⚙️ max_generation_tokens=%d", MAX_GENERATION_TOKENS)
    logger.info("⚙️ max_generation_temperature=%.2f", MAX_GENERATION_TEMPERATURE)

    try:
        yield
    finally:
        # --- Shutdown ---
        task = getattr(app.state, "model_task", None)
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass



app = FastAPI(
    title="AssistantIA Backend",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(conversations_router)
app.include_router(messages_router)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: MinistralEngine | None = getattr(app.state, "engine", None)
    loaded = engine is not None and engine.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=engine.model_name if engine else "none",
        pid=os.getpid(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Génère une réponse complète (non-streaming)."""
    engine: MinistralEngine | None = getattr(app.state, "engine", None)
    max_generation_tokens: int = int(getattr(app.state, "max_generation_tokens", MAX_GENERATION_TOKENS))
    max_generation_temperature: float = float(
        getattr(app.state, "max_generation_temperature", MAX_GENERATION_TEMPERATURE)
    )
    effective_max_tokens = min(request.max_tokens, max_generation_tokens)
    effective_temperature = min(request.temperature, max_generation_temperature)
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    try:
        return ChatResponse(response=engine.generate(
            prompt=request.prompt,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
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
    engine: MinistralEngine | None = getattr(app.state, "engine", None)
    max_generation_tokens: int = int(getattr(app.state, "max_generation_tokens", MAX_GENERATION_TOKENS))
    max_generation_temperature: float = float(
        getattr(app.state, "max_generation_temperature", MAX_GENERATION_TEMPERATURE)
    )
    effective_max_tokens = min(request.max_tokens, max_generation_tokens)
    effective_temperature = min(request.temperature, max_generation_temperature)
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    def sse():
        try:
            for chunk in engine.stream(
                prompt=request.prompt,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
