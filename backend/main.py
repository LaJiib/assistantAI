"""
Iris — Backend FastAPI
GET /health  |  POST /chat  |  POST /chat/stream  |  POST /agent/chat

Démarrage : python main.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager, suppress

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.llm import IrisEngine
from core.agent import IrisDeps, create_iris_agent, IRIS_SYSTEM_PROMPT
from core.tools import ToolRegistry
from api.conversations import router as conversations_router
from api.messages import router as messages_router
from storage.json_manager import JSONManager
from tools import register_builtin_tools

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
TOOLS_FOLDER = os.getenv("TOOLS_FOLDER", "").strip()
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


def _resolve_tools_folder() -> Path:
    """
    Dossier de persistance des outils.

    Priorité :
      1) TOOLS_FOLDER explicite
      2) sibling de DATA_FOLDER (si DATA_FOLDER se termine par /conversations)
      3) DATA_FOLDER/tools_registry
    """
    if TOOLS_FOLDER:
        return Path(TOOLS_FOLDER)

    data_path = Path(DATA_FOLDER)
    if data_path.name.lower() == "conversations":
        return data_path.parent / "tools_registry"
    return data_path / "tools_registry"


# ── Schémas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt:      str   = Field(..., min_length=1)
    max_tokens:  int   = Field(MAX_GENERATION_TOKENS, gt=0, le=32768)
    temperature: float = Field(MAX_GENERATION_TEMPERATURE, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    response: str

class AgentChatRequest(BaseModel):
    """Requête pour l'endpoint agent — utilise IrisAgent (Pydantic AI)."""
    message:     str   = Field(..., min_length=1, description="Message utilisateur")
    max_tokens:  int   = Field(MAX_GENERATION_TOKENS, gt=0, le=32768)
    temperature: float = Field(MAX_GENERATION_TEMPERATURE, ge=0.0, le=2.0)

class AgentChatResponse(BaseModel):
    response: str
    model:    str

class TitleRequest(BaseModel):
    """Requête pour la génération de titre — le frontend envoie le texte brut."""
    message: str = Field(..., min_length=1)

class TitleResponse(BaseModel):
    title: str

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str
    pid:          int


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("🌸 Démarrage Iris backend…")

    if not DATA_FOLDER:
        logger.error("❌ DATA_FOLDER non défini (vérifiez .env)")
        raise RuntimeError("DATA_FOLDER non défini")

    jm = JSONManager(Path(DATA_FOLDER))
    app.state.json_manager = jm
    jm.verify_integrity()

    tools_folder = _resolve_tools_folder()
    registry = ToolRegistry.get_instance(storage_path=tools_folder)
    await register_builtin_tools(registry)
    app.state.tool_registry = registry

    app.state.engine = None
    app.state.iris_agent = None
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
                engine = IrisEngine(model_path=MODEL_PATH)
                await engine.load()
                app.state.engine = engine
                # Créer l'agent Pydantic AI une fois le moteur chargé
                app.state.iris_agent = create_iris_agent(engine)
                logger.info("✅ Iris prête : %s", engine.model_name)
            except Exception as exc:
                app.state.model_error = str(exc)
                logger.exception("❌ Échec chargement modèle")
            finally:
                app.state.model_loading = False

        app.state.model_task = asyncio.create_task(_load_model())

    logger.info("📡 Écoute sur http://%s:%d", HOST, PORT)
    logger.info("🧰 tools_registry=%s", tools_folder)
    logger.info("⚙️  max_tokens=%d  temperature=%.2f", MAX_GENERATION_TOKENS, MAX_GENERATION_TEMPERATURE)

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
    title="Iris — Assistant Consulting Backend",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(conversations_router)
app.include_router(messages_router)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: IrisEngine | None = getattr(app.state, "engine", None)
    loaded = engine is not None and engine.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=engine.model_name if engine else "none",
        pid=os.getpid(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Génère une réponse complète (non-streaming) depuis un prompt brut."""
    engine: IrisEngine | None = getattr(app.state, "engine", None)
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
    Génère une réponse en streaming (Server-Sent Events) depuis un prompt brut.

    Format SSE :
        data: {"text": "chunk"}\\n\\n   (un chunk par token ou groupe)
        data: [DONE]\\n\\n              (fin de génération)

    Erreur mid-stream :
        data: {"error": "message"}\\n\\n
        data: [DONE]\\n\\n
    """
    engine: IrisEngine | None = getattr(app.state, "engine", None)
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


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """
    [DEBUG] Génère une réponse via IrisAgent sans conversation persistée.

    Endpoint de test/debug uniquement. Le frontend utilise
    POST /api/conversations/{id}/messages/ pour la génération persistée.
    """
    iris_agent = getattr(app.state, "iris_agent", None)
    engine: IrisEngine | None = getattr(app.state, "engine", None)

    if not iris_agent or not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Agent Iris non initialisé.")

    try:
        from pydantic_ai.settings import ModelSettings
        settings = ModelSettings(
            max_tokens=min(request.max_tokens, int(getattr(app.state, "max_generation_tokens", MAX_GENERATION_TOKENS))),
            temperature=min(request.temperature, float(getattr(app.state, "max_generation_temperature", MAX_GENERATION_TEMPERATURE))),
        )
        result = await iris_agent.run(
            request.message,
            deps=IrisDeps(),   # Étape 2 : injecter VectorStoreManager ici
            model_settings=settings,
        )
        return AgentChatResponse(
            response=result.data,
            model=engine.model_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/chat/stream")
def agent_chat_stream(request: AgentChatRequest) -> StreamingResponse:
    """
    [DEBUG] Streaming SSE stateless via engine.stream_messages() + IRIS_SYSTEM_PROMPT.

    Endpoint de test/debug uniquement. Le frontend utilise
    POST /api/conversations/{id}/messages/stream pour la génération avec
    persistance et IrisAgent (Pydantic AI + tool calling).
    """
    engine: IrisEngine | None = getattr(app.state, "engine", None)
    max_generation_tokens: int = int(getattr(app.state, "max_generation_tokens", MAX_GENERATION_TOKENS))
    max_generation_temperature: float = float(
        getattr(app.state, "max_generation_temperature", MAX_GENERATION_TEMPERATURE)
    )
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    effective_max_tokens  = min(request.max_tokens,  max_generation_tokens)
    effective_temperature = min(request.temperature, max_generation_temperature)

    messages = [
        {"role": "system", "content": IRIS_SYSTEM_PROMPT},
        {"role": "user",   "content": request.message},
    ]

    def sse():
        try:
            for chunk in engine.stream_messages(
                messages,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/agent/title", response_model=TitleResponse)
def agent_title(request: TitleRequest) -> TitleResponse:
    """
    Génère un titre court (≤ 10 mots) pour une conversation.

    Le frontend envoie uniquement le texte brut du premier message utilisateur.
    Tout le prompt engineering (instruction de titrage) est centralisé ici.
    """
    engine: IrisEngine | None = getattr(app.state, "engine", None)
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    _TITLE_SYSTEM = (
        "Your only task is to generate a very short title (max 10 words) "
        "that summarizes the user's request. "
        "Respond with only the title, no punctuation around it, nothing else."
    )
    messages = [
        {"role": "system", "content": _TITLE_SYSTEM},
        {"role": "user",   "content": request.message},
    ]
    try:
        raw = engine._generate_raw(messages, None, 30, 0.1)
        title = raw.strip().strip("\"'")
        return TitleResponse(title=title or "Nouvelle conversation")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
