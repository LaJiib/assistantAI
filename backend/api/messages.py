"""
Router FastAPI — Messages d'une conversation.

Prefix : /api/conversations/{conversation_id}/messages

Endpoints :
  GET    /        → historique messages (system exclu) → 200 [MessageResponse]
  POST   /stream  → envoyer message + génération SSE → stream d'événements typés

Codes HTTP :
  200  OK                — GET réussi
  404  Not Found         — conversation_id inexistant
  422  Unprocessable     — validation Pydantic (contenu vide)
  503  Service Unavailable — agent non initialisé
  500  Internal Error    — erreur I/O ou génération inattendue

Règles métier :
  - Le message system (ModelRequest avec SystemPromptPart seul) est EXCLU de GET /
  - Message user sauvegardé AVANT génération (pas de perte si crash)
  - Si génération échoue : message user conservé, pas de rollback
  - messageCount dans l'index = nombre de messages user+assistant (system exclu)
  - Streaming : accumulation mémoire → save unique dans finally
  - Génération via iris_agent.run_stream_events() — tool calling et reasoning inclus

Protocole SSE BFF :
  data: {"type": "start"}
  data: {"type": "textDelta",      "content": "..."}
  data: {"type": "reasoningDelta", "content": "..."}
  data: {"type": "toolCallStart",  "toolCallId": "...", "toolName": "..."}
  data: {"type": "toolCallResult", "toolCallId": "...", "content": "..."}
  data: {"type": "error",          "content": "..."}
  data: {"type": "done"}
  data: [DONE]

Synchronisation des listes :
  Conversation.messages et Conversation.message_meta sont des listes parallèles
  (même longueur). Tout ajout à l'une doit être accompagné d'un ajout à l'autre
  dans le même bloc lock pour garantir leur cohérence.

Concurrence :
  - Lock par conversation_id : protège load→append→save contre requêtes simultanées
"""

from __future__ import annotations

import json
import logging
import threading
from itertools import zip_longest
from typing import Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

from core.agent import IrisDeps, build_dynamic_system_prompt
from core.sse_transformer import GemmaThinkingParser, parse_thinking_tags, transform_agent_event
from models.conversation import MessageMeta, _utcnow
from storage.json_manager import JSONManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/conversations/{conversation_id}/messages",
    tags=["messages"],
)

# ── Locks par conversation ─────────────────────────────────────────────────────

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


def get_generation_temperature(request: Request) -> float:
    return float(getattr(request.app.state, "generation_temperature", 0.3))


def get_generation_top_p(request: Request) -> float:
    return float(getattr(request.app.state, "generation_top_p", 0.0))


def get_generation_top_k(request: Request) -> int:
    return int(getattr(request.app.state, "generation_top_k", 0))


# ── Schemas ───────────────────────────────────────────────────────────────────

class SendMessageRequest(BaseModel):
    content: str  = Field(..., min_length=1, max_length=32000)
    options: dict = Field(default_factory=dict)


class MessagePartResponse(BaseModel):
    type:       str        # "text", "reasoning", "toolCall", "toolResult"
    content:    str | None = None
    toolCallId: str | None = None
    toolName:   str | None = None


class MessageResponse(BaseModel):
    id:        str
    role:      str   # "user" ou "assistant"
    parts:     List[MessagePartResponse]
    timestamp: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_meta_or_404(conversation_id: str, jm: JSONManager):
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


def _model_message_to_response(
    msg: ModelMessage,
    meta: MessageMeta | None,
) -> MessageResponse | None:
    """
    Convertit un ModelMessage Pydantic AI en MessageResponse lisible par le frontend.

    - ModelRequest avec UserPromptPart → role=user, parts=[text]
    - ModelRequest avec seulement SystemPromptPart/ToolReturnPart → None (filtré)
    - ModelResponse → role=assistant, parts=[text?, toolCall*]
    - Retourne None si aucune part visible.
    """
    ts = meta.timestamp.isoformat() if meta else _utcnow().isoformat()
    msg_id = str(meta.id) if meta else str(UUID(int=0))

    if isinstance(msg, ModelRequest):
        parts: list[MessagePartResponse] = []
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                content = part.content if isinstance(part.content, str) else str(part.content)
                parts.append(MessagePartResponse(type="text", content=content))
            # SystemPromptPart et ToolReturnPart : internes, non exposés
        if not parts:
            return None
        return MessageResponse(id=msg_id, role="user", parts=parts, timestamp=ts)

    elif isinstance(msg, ModelResponse):
        parts = []
        for part in msg.parts:
            if isinstance(part, TextPart) and part.content:
                # Parser les balises de réflexion Gemma stockées dans le texte brut
                for part_type, content in parse_thinking_tags(part.content):
                    parts.append(MessagePartResponse(type=part_type, content=content))
            elif isinstance(part, ToolCallPart):
                parts.append(MessagePartResponse(
                    type="toolCall",
                    toolCallId=part.tool_call_id,
                    toolName=part.tool_name,
                ))
        if not parts:
            return None
        return MessageResponse(id=msg_id, role="assistant", parts=parts, timestamp=ts)

    return None


def _update_index_message_count(conversation_id: str, delta: int, jm: JSONManager) -> None:
    """Incrémente messageCount dans l'index sans relire depuis disk."""
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
    summary="Historique des messages (format multi-parts)",
)
def get_messages(
    conversation_id: str,
    jm: JSONManager = Depends(get_json_manager),
) -> List[MessageResponse]:
    """
    Retourne les messages user et assistant, triés par ordre d'insertion.
    Les messages system (ModelRequest avec SystemPromptPart seul) sont exclus.
    Chaque message contient un tableau de parts (text, toolCall, etc.).
    """
    conversation_id = _normalize_conversation_id(conversation_id)
    _get_meta_or_404(conversation_id, jm)
    conv = _load_conversation_or_404(conversation_id, jm)

    responses: list[MessageResponse] = []
    for msg, meta in zip_longest(conv.messages, conv.message_meta):
        r = _model_message_to_response(msg, meta)
        if r is not None:
            responses.append(r)
    return responses


@router.post("/stream", summary="Envoyer un message avec réponse en streaming SSE typé")
async def send_message_stream(
    conversation_id: str,
    body: SendMessageRequest,
    jm: JSONManager = Depends(get_json_manager),
    iris_agent=Depends(get_iris_agent),
    max_generation_tokens: int = Depends(get_max_generation_tokens),
    generation_temperature: float = Depends(get_generation_temperature),
    generation_top_p: float = Depends(get_generation_top_p),
    generation_top_k: int = Depends(get_generation_top_k),
) -> StreamingResponse:
    """
    Envoie un message et streame la réponse de l'agent via SSE.

    Le backend agit comme BFF :
    - Intercepte les balises de réflexion Gemma (<|channel>thought...<channel|>)
    - Transforme les événements Pydantic AI en protocol SSE JSON propre
    - Sauvegarde l'historique natif (ModelMessage) après le stream

    Le client iOS est "dumb" : il ne renvoie pas d'historique, il consomme
    uniquement les événements SSE typés.
    """
    conversation_id = _normalize_conversation_id(conversation_id)
    effective_max_tokens = max_generation_tokens
    meta = _get_meta_or_404(conversation_id, jm)
    lock = _get_conv_lock(conversation_id)

    deps = IrisDeps(
        enable_thinking=body.options.get("think", False),
        conversation_system_prompt=meta.specificInstruction,
        vector_store=None,
    )

    async def sse_generator():
        # ── 1. Sauvegarder le message user AVANT de lancer la génération ──────
        user_request = ModelRequest(parts=[UserPromptPart(content=body.content)])
        with lock:
            conv = _load_conversation_or_404(conversation_id, jm)
            # Synchronisation atomique : on ajoute le message ET son méta ensemble
            conv.messages.append(user_request)
            conv.message_meta.append(MessageMeta())
            jm.save_conversation(conv)

        yield f"data: {json.dumps({'type': 'start'})}\n\n"

        # ── 2. Construire l'historique pour Pydantic AI ────────────────────────
        # Le system prompt est dynamique (dépend de IrisDeps) → on le recalcule
        # à chaque requête et on le prepend à l'historique stocké.
        # On passe conv.messages[:-1] car le user_request courant est passé
        # séparément en tant que user_prompt de run_stream_events().
        system_prompt = build_dynamic_system_prompt(deps)
        history_for_agent: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=system_prompt)])
        ] + list(conv.messages[:-1])

        settings = ModelSettings(
            max_tokens=effective_max_tokens,
            temperature=generation_temperature,
            top_p=generation_top_p,
            top_k=generation_top_k,  # type: ignore[typeddict-unknown-key]
        )

        run_result = None
        parser = GemmaThinkingParser()

        try:
            # ── 3. Itérer sur les événements Pydantic AI ──────────────────────
            async for event in iris_agent.run_stream_events(
                body.content,
                message_history=history_for_agent,
                deps=deps,
                model_settings=settings,
            ):
                # AgentRunResultEvent : n'est pas un AgentStreamEvent,
                # on le capture pour récupérer new_messages() dans le finally
                if isinstance(event, AgentRunResultEvent):
                    run_result = event.result
                    continue

                # Transformer l'événement natif en 0-N dicts BFF
                for bff_event in transform_agent_event(event, parser):
                    yield f"data: {json.dumps(bff_event)}\n\n"

            # Vider le buffer résiduel du parser Gemma
            for event_type, content in parser.flush():
                if content:
                    yield f"data: {json.dumps({'type': event_type, 'content': content})}\n\n"

        except Exception as exc:
            logger.error("Erreur stream [%s] : %s", conversation_id, exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

        finally:
            # ── 4. Sauvegarder les nouveaux messages de l'agent ───────────────
            # run_result.new_messages() retourne [user_request, *agent_responses]
            # new_messages()[0] = le user_request déjà sauvegardé → on skip
            if run_result is not None:
                try:
                    new_msgs = run_result.new_messages()
                    agent_msgs = new_msgs[1:]  # On exclut le user_request initial

                    if agent_msgs:
                        with lock:
                            conv_final = _load_conversation_or_404(conversation_id, jm)
                            for msg in agent_msgs:
                                # Synchronisation atomique : message ET méta ensemble
                                conv_final.messages.append(msg)
                                conv_final.message_meta.append(MessageMeta())
                            jm.save_conversation(conv_final)
                            # +1 user (déjà compté) + len(agent_msgs) nouvelles entrées
                            _update_index_message_count(conversation_id, len(agent_msgs), jm)
                except Exception as save_exc:
                    logger.error("Erreur save post-stream [%s]: %s", conversation_id, save_exc)
            else:
                logger.warning(
                    "Stream [%s] : run_result est None — réponse assistant non sauvegardée",
                    conversation_id,
                )

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
