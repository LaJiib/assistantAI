"""
Iris — Backend FastAPI
GET  /health              |  Debug : état du serveur
POST /agent/chat          |  Debug : génération stateless via IrisAgent
POST /agent/chat/stream   |  Debug : streaming SSE stateless
POST /agent/title         |  Génération de titre court

Démarrage : python main.py

Architecture A2 :
  - mlx-vlm.server (subprocess, port 8001) ← Gemma 4 via mlx-vlm
  - FastAPI BFF (port 8000) ← Swift client
  - pydantic-ai Agent (ReasoningAwareOpenAIModel) ← orchestre tool loop
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.agent import IrisDeps, create_iris_agent
from core.tools import ToolRegistry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from api.conversations import router as conversations_router
from api.messages import router as messages_router
from storage.json_manager import JSONManager
from tools import register_builtin_tools
from tools.builtin.web import fetch_webpage, web_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Construit le chemin absolu vers le .env (situé à côté de main.py)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

MODEL_PATH   = os.getenv("MODEL_PATH", "")
HOST         = os.getenv("HOST", "127.0.0.1")
PORT         = int(os.getenv("PORT", "8000"))
DATA_FOLDER  = os.getenv("DATA_FOLDER", "").strip()
TOOLS_FOLDER = os.getenv("TOOLS_FOLDER", "").strip()
MLX_SERVER_PORT = int(os.getenv("MLX_SERVER_PORT", "8001"))

MAX_GENERATION_TOKENS = int(os.getenv("MAX_GENERATION_TOKENS", "4096"))
if MAX_GENERATION_TOKENS <= 0:
    logger.warning("MAX_GENERATION_TOKENS invalide (%s), fallback à 4096", MAX_GENERATION_TOKENS)
    MAX_GENERATION_TOKENS = 4096

GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "1.0"))
if not (0.0 <= GENERATION_TEMPERATURE <= 2.0):
    logger.warning("GENERATION_TEMPERATURE invalide (%s), fallback à 1.0", GENERATION_TEMPERATURE)
    GENERATION_TEMPERATURE = 1.0

GENERATION_TOP_P = float(os.getenv("GENERATION_TOP_P", "0.95"))
if not (0.0 <= GENERATION_TOP_P <= 1.0):
    logger.warning("GENERATION_TOP_P invalide (%s), fallback à 0.95", GENERATION_TOP_P)
    GENERATION_TOP_P = 0.95

GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", "64"))
if GENERATION_TOP_K < 0:
    logger.warning("GENERATION_TOP_K invalide (%s), fallback à 0", GENERATION_TOP_K)
    GENERATION_TOP_K = 0

# ── Commande de démarrage mlx_vlm.server ───────────────────────────────────

def _build_mlx_cmd(model_path: str, port: int) -> list[str]:
    return [
        "mlx_vlm.server",
        "--model",        model_path,
        "--host",         "127.0.0.1",
        "--port",         str(port),
    ]


def _free_port(port: int) -> None:
    """Tue tout process qui écoute sur le port donné (évite EADDRINUSE au redémarrage)."""
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections(kind="tcp"):
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    logger.warning("Port %d occupé par pid %d (%s) — terminé", port, proc.pid, proc.name())
                    proc.terminate()
                    proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass


def _forward_stderr(proc: subprocess.Popen, prefix: str) -> None:
    """Thread daemon : lit stderr du subprocess et le relaye dans notre logger."""
    for raw in iter(proc.stderr.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.info("[%s] %s", prefix, line)


async def _wait_until_ready(url: str, timeout: float = 180.0) -> None:
    """Poll /health jusqu'à ce que mlx-vlm-server ait chargé le modèle (max timeout s)."""
    deadline = asyncio.get_running_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200 and r.json().get("loaded_model"):
                    logger.info("✅ mlx-vlm-server prêt sur %s", url)
                    return
            except (httpx.ConnectError, httpx.ReadTimeout, Exception):
                pass
            await asyncio.sleep(1.5)
    raise TimeoutError(f"mlx-vlm-server non prêt après {timeout:.0f}s ({url})")


def _resolve_tools_folder() -> Path:
    """
    Dossier de persistance des outils.
    Priorité : TOOLS_FOLDER explicite > sibling de DATA_FOLDER > DATA_FOLDER/tools_registry
    """
    if TOOLS_FOLDER:
        return Path(TOOLS_FOLDER)
    data_path = Path(DATA_FOLDER)
    if data_path.name.lower() == "conversations":
        return data_path.parent / "tools_registry"
    return data_path / "tools_registry"


# ── Schémas ───────────────────────────────────────────────────────────────────

class AgentChatRequest(BaseModel):
    """Requête pour les endpoints debug agent."""
    message:     str   = Field(..., min_length=1, description="Message utilisateur")
    max_tokens:  int   = Field(MAX_GENERATION_TOKENS, gt=0, le=32768)
    temperature: float = Field(GENERATION_TEMPERATURE, ge=0.0, le=2.0)

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
    # ── Startup ──────────────────────────────────────────────────────────────
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

    app.state.iris_agent  = None
    app.state.mlx_proc    = None
    app.state.local_client = None
    app.state.max_generation_tokens  = MAX_GENERATION_TOKENS
    app.state.generation_temperature = GENERATION_TEMPERATURE
    app.state.generation_top_p       = GENERATION_TOP_P
    app.state.generation_top_k       = GENERATION_TOP_K

    if not MODEL_PATH:
        logger.error("❌ MODEL_PATH non défini (vérifiez .env)")
        raise RuntimeError("MODEL_PATH non défini")

    # ── Lancer mlx-vlm-server en subprocess ────────────────────────────
    _free_port(MLX_SERVER_PORT)
    mlx_cmd = _build_mlx_cmd(MODEL_PATH, MLX_SERVER_PORT)
    logger.info("🚀 Lancement mlx-vlm-server (port %d)…", MLX_SERVER_PORT)
    logger.info("   %s", " ".join(mlx_cmd))

    proc = subprocess.Popen(mlx_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    app.state.mlx_proc = proc
    threading.Thread(target=_forward_stderr, args=(proc, "mlx-vlm"), daemon=True).start()

    # ── Attendre que le serveur soit prêt (bloquant, max 180s) ───────────
    try:
        await _wait_until_ready(f"http://127.0.0.1:{MLX_SERVER_PORT}/health")
    except TimeoutError as exc:
        proc.terminate()
        raise RuntimeError(str(exc)) from exc

    # ── Créer client OpenAI + modèle + agent ─────────────────────────────
    local_client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{MLX_SERVER_PORT}/v1",
        api_key="not-needed",
    )
    app.state.local_client = local_client

    provider = OpenAIProvider(openai_client=local_client)
    model    = OpenAIChatModel(MODEL_PATH, provider=provider)

    app.state.iris_agent = create_iris_agent(
        model,
        tools=[web_search, fetch_webpage],
    )

    logger.info("📡 Écoute sur http://%s:%d", HOST, PORT)
    logger.info("🧰 tools_registry=%s", tools_folder)
    logger.info(
        "⚙️  max_tokens=%d  temperature=%.2f  top_p=%.2f  top_k=%d",
        MAX_GENERATION_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P, GENERATION_TOP_K,
    )

    try:
        yield
    finally:
        # ── Shutdown ──────────────────────────────────────────────────────
        mlx_proc: subprocess.Popen | None = getattr(app.state, "mlx_proc", None)
        if mlx_proc is not None:
            logger.info("⏹  Arrêt mlx-vlm-server (pid %d)…", mlx_proc.pid)
            mlx_proc.terminate()
            try:
                mlx_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                mlx_proc.kill()


app = FastAPI(
    title="Iris — Assistant Consulting Backend",
    version="2.1.0",
    lifespan=lifespan,
)
app.include_router(conversations_router)
app.include_router(messages_router)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    iris_agent = getattr(app.state, "iris_agent", None)
    mlx_proc: subprocess.Popen | None = getattr(app.state, "mlx_proc", None)
    loaded = iris_agent is not None and mlx_proc is not None and mlx_proc.poll() is None
    model_name = Path(MODEL_PATH).name if MODEL_PATH else "none"
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=model_name,
        pid=os.getpid(),
    )


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """
    [DEBUG] Génère une réponse via IrisAgent sans conversation persistée.

    Endpoint de test/debug uniquement. Le frontend utilise
    POST /api/conversations/{id}/messages/stream pour la génération persistée.
    """
    iris_agent = getattr(app.state, "iris_agent", None)
    if iris_agent is None:
        raise HTTPException(status_code=503, detail="Agent Iris non initialisé.")

    try:
        from pydantic_ai.settings import ModelSettings
        settings = ModelSettings(
            max_tokens=min(request.max_tokens, int(getattr(app.state, "max_generation_tokens", MAX_GENERATION_TOKENS))),
            temperature=request.temperature,
        )
        result = await iris_agent.run(
            request.message,
            deps=IrisDeps(),
            model_settings=settings,
        )
        return AgentChatResponse(
            response=result.output,
            model="gemma4",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/chat/stream")
async def agent_chat_stream(request: AgentChatRequest) -> StreamingResponse:
    """
    [DEBUG] Streaming SSE stateless via IrisAgent.

    Endpoint de test/debug uniquement. Le frontend utilise
    POST /api/conversations/{id}/messages/stream pour la génération avec
    persistance et IrisAgent (Pydantic AI + tool calling).
    """
    iris_agent = getattr(app.state, "iris_agent", None)
    if iris_agent is None:
        raise HTTPException(status_code=503, detail="Agent Iris non initialisé.")

    from pydantic_ai.settings import ModelSettings
    settings = ModelSettings(
        max_tokens=min(request.max_tokens, int(getattr(app.state, "max_generation_tokens", MAX_GENERATION_TOKENS))),
        temperature=request.temperature,
    )

    async def sse():
        try:
            async with iris_agent.run_stream(
                request.message,
                deps=IrisDeps(),
                model_settings=settings,
            ) as stream:
                async for chunk in stream.stream_text(delta=True):
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


_TITLE_SYSTEM = (
    "Your only task is to generate a very short title (max 10 words) "
    "that summarizes the user's request. "
    "Respond with only the title, no punctuation around it, nothing else."
)


@app.post("/agent/title", response_model=TitleResponse)
async def agent_title(request: TitleRequest) -> TitleResponse:
    """
    Génère un titre court (≤ 10 mots) pour une conversation.

    Le frontend envoie uniquement le texte brut du premier message utilisateur.
    Utilise le client OpenAI directement (bypass pydantic-ai) pour réduire la latence.
    """
    local_client: AsyncOpenAI | None = getattr(app.state, "local_client", None)
    if local_client is None:
        raise HTTPException(status_code=503, detail="Service non initialisé.")

    try:
        resp = await local_client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {"role": "system", "content": _TITLE_SYSTEM},
                {"role": "user",   "content": request.message},
            ],
            max_tokens=30,
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip().strip("\"'")
        return TitleResponse(title=raw or "Nouvelle conversation")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
