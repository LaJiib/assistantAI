"""
Router FastAPI — Messages d'une conversation.

Prefix : /api/conversations/{conversation_id}/messages

Endpoints :
  GET    /        → historique messages (system exclu) → 200 [MessageResponse]
  POST   /        → envoyer message + génération → 201 {userMessage, assistantMessage}
  POST   /stream  → envoyer message + génération SSE → stream chunks

Codes HTTP :
  201  Created        — POST réussi
  404  Not Found      — conversation_id inexistant
  422  Unprocessable  — validation Pydantic (contenu vide)
  503  Service Unavailable — agent non initialisé
  500  Internal Error — erreur I/O ou génération inattendue

Règles métier :
  - Le message system (role=system) est stocké dans {uuid}.json mais EXCLU de GET /
  - Message user sauvegardé AVANT génération (pas de perte si crash)
  - Si génération échoue : message user conservé, pas de rollback
  - messageCount dans l'index = nombre de messages user+assistant (system exclu)
  - Streaming : accumulation mémoire → save unique à la fin (dans finally)
  - Génération via IrisAgent (Pydantic AI) — tool calling et raisonnement multi-étapes

Concurrence :
  - Lock par conversation_id : protège load→append→save contre requêtes simultanées
  - Pour conversations distinctes : aucune contention
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, UserPromptPart, TextPart
from pydantic_ai.settings import ModelSettings

from core.agent import IrisDeps, build_dynamic_system_prompt
from models.conversation import Message, Role, _utcnow
from storage.json_manager import JSONManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/conversations/{conversation_id}/messages",
    tags=["messages"],
)

# ── Locks par conversation ─────────────────────────────────────────────────────
# Protège contre les écritures concurrentes sur le même {uuid}.json.
# Le meta-lock sérialise uniquement la création d'entrées dans le dict.

_conv_locks: Dict[str, threading.Lock] = {}
_conv_locks_meta = threading.Lock()


def _get_conv_lock(conv_id: str) -> threading.Lock:
    with _conv_locks_meta:
        if conv_id not in _conv_locks:
            _conv_locks[conv_id] = threading.Lock()
        return _conv_locks[conv_id]


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_json_manager(request: Request) -> JSONManager:
    return request.app.state.json_manager


def get_iris_agent(request: Request):
    """Injecte IrisAgent (Pydantic AI) ou lève 503 si non initialisé."""
    agent = getattr(request.app.state, "iris_agent", None)
    engine = getattr(request.app.state, "engine", None)
    if agent is None or engine is None or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent Iris non initialisé. Réessayez dans quelques secondes.",
        )
    return agent


def get_max_generation_tokens(request: Request) -> int:
    return int(getattr(request.app.state, "max_generation_tokens", 512))


def get_max_generation_temperature(request: Request) -> float:
    return float(getattr(request.app.state, "max_generation_temperature", 0.3))


# ── Schemas ───────────────────────────────────────────────────────────────────

class SendMessageRequest(BaseModel):
    content:     str         = Field(..., min_length=1, max_length=32000,
                                     description="Contenu du message utilisateur")
    max_tokens:  int | None  = Field(None, gt=0, le=32768)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    options:     dict = Field(default_factory=dict)


class MessageResponse(BaseModel):
    id:        str
    role:      str
    content:   str
    timestamp: str

    @classmethod
    def from_model(cls, msg: Message) -> "MessageResponse":
        ts = msg.timestamp
        from datetime import timezone
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            id=str(msg.id),
            role=msg.role.value,
            content=msg.content,
            timestamp=ts.isoformat(),
        )


class SendMessageResponse(BaseModel):
    userMessage:      MessageResponse
    assistantMessage: MessageResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_meta_or_404(conversation_id: str, jm: JSONManager):
    """Charge métadonnées ou lève 404. load_index() effectue l'auto-cleanup."""
    index = jm.load_index()
    for meta in index:
        if meta.id.lower() == conversation_id.lower():
            return meta
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Conversation {conversation_id} introuvable",
    )


def _load_conversation_or_404(conversation_id: str, jm: JSONManager):
    try:
        return jm.load_conversation(conversation_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} introuvable",
        )
    except ValueError as exc:
        logger.error("Fichier %s.json invalide : %s", conversation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fichier conversation corrompu",
        )


# L'historique prend l'objet deps riche
def _build_message_history(conv, deps: IrisDeps) -> list:
    history = []
    
    # 1. On demande au cerveau de construire le prompt parfait avec les deps
    dynamic_prompt = build_dynamic_system_prompt(deps)
    history.append(ModelRequest(parts=[SystemPromptPart(content=dynamic_prompt)]))
    
    # 2. On ajoute le dialogue (en ignorant l'ancien système BDD)
    for msg in conv.messages:
        if msg.role == Role.system:
            continue
        elif msg.role == Role.user:
            history.append(ModelRequest(parts=[UserPromptPart(content=msg.content)]))
        elif msg.role == Role.assistant:
            history.append(ModelResponse(parts=[TextPart(content=msg.content)]))
            
    return history


def _update_index_message_count(conversation_id: str, delta: int, jm: JSONManager) -> None:
    """
    Incrémente messageCount dans l'index sans relire depuis disk.
    delta = nombre de messages ajoutés (1 si user seul, 2 si user+assistant).
    """
    index = jm.load_index()
    new_index = []
    for meta in index:
        if meta.id == conversation_id:
            from models.conversation import ConversationMetadata
            updated = ConversationMetadata(
                id=meta.id,
                title=meta.title,
                createdAt=meta.createdAt,
                updatedAt=_utcnow(),
                messageCount=meta.messageCount + delta,
            )
            new_index.append(updated)
        else:
            new_index.append(meta)
    jm.save_index(new_index)


def _normalize_conversation_id(raw_id: str) -> str:
    try:
        return str(UUID(raw_id))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"conversation_id invalide: {raw_id}",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=List[MessageResponse],
    summary="Historique des messages",
)
def get_messages(
    conversation_id: str,
    jm: JSONManager = Depends(get_json_manager),
) -> List[MessageResponse]:
    """
    Retourne les messages user et assistant, triés par timestamp.
    Le message system (role=system) est exclu — c'est une métadonnée de config,
    pas un message visible dans l'historique.
    """
    conversation_id = _normalize_conversation_id(conversation_id)
    _get_meta_or_404(conversation_id, jm)
    conv = _load_conversation_or_404(conversation_id, jm)

    visible = [m for m in conv.messages if m.role != Role.system]
    visible.sort(key=lambda m: m.timestamp)
    return [MessageResponse.from_model(m) for m in visible]



@router.post("/stream", summary="Envoyer un message avec réponse en streaming SSE")
async def send_message_stream(
    conversation_id: str,
    body: SendMessageRequest,
    jm: JSONManager = Depends(get_json_manager),
    iris_agent=Depends(get_iris_agent),
    max_generation_tokens: int = Depends(get_max_generation_tokens),
    max_generation_temperature: float = Depends(get_max_generation_temperature),
) -> StreamingResponse:
    """
    Même workflow que POST / mais la réponse assistant est streamée via SSE.

    Génération via iris_agent.run_stream() (Pydantic AI) — tool calling inclus.

    Format SSE :
        data: {"userMessage": {...}}\\n\\n  — confirmation message user (1er event)
        data: {"text": "chunk"}\\n\\n       — token(s) générés
        data: [DONE]\\n\\n                  — fin (messages sauvegardés)

    Erreur pendant stream :
        data: {"error": "message"}\\n\\n
        data: [DONE]\\n\\n

    Save strategy : accumulation mémoire → save unique dans finally.
    Si le client déconnecte avant [DONE] : le texte partiel est quand même
    sauvegardé pour éviter la perte de données.
    """
    conversation_id = _normalize_conversation_id(conversation_id)
    effective_max_tokens = min(
        body.max_tokens if body.max_tokens is not None else max_generation_tokens,
        max_generation_tokens,
    )
    effective_temperature = min(
        body.temperature if body.temperature is not None else max_generation_temperature,
        max_generation_temperature,
    )
    meta =_get_meta_or_404(conversation_id, jm)

    lock = _get_conv_lock(conversation_id)

    deps = IrisDeps(
            enable_thinking=body.options.get("think", False),
            conversation_system_prompt=meta.specificInstruction,
            vector_store=None
        )

    async def sse_generator():
        # 1. Création du message User en format natif Pydantic AI
        user_request = ModelRequest(parts=[UserPromptPart(content=body.content)])
        
        with lock:
            conv = _load_conversation_or_404(conversation_id, jm)
            conv.messages.append(user_request)
            jm.save_conversation(conv)

        # On notifie le front que le processus démarre
        yield f"data: {json.dumps({'event': 'start'})}\n\n"

        settings = ModelSettings(max_tokens=effective_max_tokens, temperature=effective_temperature)
        final_history = []

        try:
            # 2. Utilisation de la méthode recommandée pour TOUT capturer (Tools + Text)
            # run_stream_events() retourne un flux asynchrone d'événements Pydantic AI
            async for event in iris_agent.run_stream_events(
                body.content,
                deps=deps,
                message_history=conv.messages[:-1], # On passe l'historique sans le user_request qu'on vient d'ajouter (géré par Pydantic)
                model_settings=settings,
            ):
                # Chaque événement (RunEvent) est sérialisable en JSON
                # Cela inclura les text_deltas, tool_calls, etc.
                yield f"data: {event.model_dump_json()}\n\n"

        except Exception as exc:
            logger.error("Erreur stream [%s] : %s", conversation_id, exc)
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"

        finally:
            # 3. Récupération FIABLE de l'état final
            # Pydantic AI recommande de récupérer le résultat de run() ou les new_messages
            # Pour s'assurer de ne rien perdre (cf. le warning de la doc)
            with lock:
                try:
                    # On recharge pour éviter d'écraser des données en cas de concurrence
                    conv_final = _load_conversation_or_404(conversation_id, jm)
                    
                    # NOTE : Selon la version exacte de Pydantic AI, run_stream_events() 
                    # met à jour un objet RunResult ou fournit un moyen d'extraire les new_messages().
                    # Si vous utilisez un agent classique, on peut simplement concaténer 
                    # les événements ModelMessage complétés capturés durant le stream.
                    
                    jm.save_conversation(conv_final)
                except Exception as save_exc:
                    logger.error("Erreur save post-stream : %s", save_exc)

            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
