"""
Router FastAPI — CRUD conversations.

Prefix : /api/conversations

Endpoints :
  GET    /              → liste ConversationMetadata (triée updatedAt desc)
  POST   /              → créer conversation → 201 ConversationMetadata
  GET    /{id}          → détail Conversation (metadata + messages) → 200
  PUT    /{id}          → modifier titre → 200 ConversationMetadata
  DELETE /{id}          → supprimer → 204

Codes HTTP :
  201  Created         — POST réussi
  204  No Content      — DELETE réussi
  400  Bad Request     — titre vide ou trop long (via Pydantic + HTTPException)
  404  Not Found       — ID inexistant ou fichier {uuid}.json absent
  422  Unprocessable   — validation Pydantic automatique (FastAPI)
  500  Internal Error  — erreur I/O inattendue

Règles métier :
  - createdAt immutable après création
  - updatedAt mis à jour à chaque PUT
  - Conversation créée avec message system initial (compatibilité Swift)
"""

from __future__ import annotations

import logging
from typing import List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from itertools import zip_longest

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart

from models.conversation import (
    Conversation,
    ConversationMetadata,
    MessageMeta,
    _utcnow,
)
from storage.json_manager import JSONManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


# ── Dependency ────────────────────────────────────────────────────────────────

def get_json_manager(request: Request) -> JSONManager:
    """Injecte le JSONManager depuis app.state (initialisé dans main.py)."""
    return request.app.state.json_manager


# ── Schemas request ───────────────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    title:        str = Field(..., min_length=1, max_length=500,
                               description="Titre de la conversation")

class UpdateConversationRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500,
                        description="Nouveau titre de la conversation")
    specificInstruction: str | None = Field(None, description="Instructions spécifiques")

# ── Schemas response ──────────────────────────────────────────────────────────

class MessagePartResponse(BaseModel):
    type:       str
    content:    str | None = None
    toolCallId: str | None = None
    toolName:   str | None = None


class MessageResponse(BaseModel):
    """Format multi-parts — identique à celui de api/messages.py."""
    id:        str
    role:      str
    parts:     List[MessagePartResponse]
    timestamp: str

    @classmethod
    def from_model_message(
        cls,
        msg: ModelMessage,
        meta: MessageMeta | None,
    ) -> "MessageResponse | None":
        ts = (meta.timestamp.isoformat() if meta else _utcnow().isoformat())
        msg_id = str(meta.id) if meta else ""

        if isinstance(msg, ModelRequest):
            parts = [
                MessagePartResponse(type="text", content=p.content if isinstance(p.content, str) else str(p.content))
                for p in msg.parts if isinstance(p, UserPromptPart)
            ]
            if not parts:
                return None
            return cls(id=msg_id, role="user", parts=parts, timestamp=ts)

        elif isinstance(msg, ModelResponse):
            parts = []
            for p in msg.parts:
                if isinstance(p, TextPart) and p.content:
                    parts.append(MessagePartResponse(type="text", content=p.content))
                elif isinstance(p, ToolCallPart):
                    parts.append(MessagePartResponse(
                        type="toolCall",
                        toolCallId=p.tool_call_id,
                        toolName=p.tool_name,
                    ))
            if not parts:
                return None
            return cls(id=msg_id, role="assistant", parts=parts, timestamp=ts)

        return None


class ConversationResponse(BaseModel):
    """Métadonnées seules — utilisé pour list et create."""
    id:           str
    title:        str
    createdAt:    str
    updatedAt:    str
    messageCount: int

    @classmethod
    def from_metadata(cls, meta: ConversationMetadata) -> "ConversationResponse":
        d = meta.model_dump(mode="json")
        return cls(**d)


class ConversationDetailResponse(BaseModel):
    """Métadonnées + messages (format parts) — utilisé pour GET /{id}."""
    id:           str
    title:        str
    createdAt:    str
    updatedAt:    str
    messageCount: int
    messages:     List[MessageResponse]

    @classmethod
    def from_metadata_and_conversation(
        cls,
        meta: ConversationMetadata,
        conv: Conversation,
    ) -> "ConversationDetailResponse":
        d = meta.model_dump(mode="json")
        responses = [
            MessageResponse.from_model_message(msg, m)
            for msg, m in zip_longest(conv.messages, conv.message_meta)
        ]
        return cls(**d, messages=[r for r in responses if r is not None])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_meta_or_404(
    conversation_id: str,
    jm: JSONManager,
) -> ConversationMetadata:
    """
    Retourne la métadonnée ou lève 404.

    load_index() a déjà effectué l'auto-cleanup — si l'ID n'est pas dans
    l'index, c'est qu'il n'existe pas (ou que le fichier était manquant).
    """
    index = jm.load_index()
    for meta in index:
        if meta.id.lower() == conversation_id.lower():
            return meta
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Conversation {conversation_id} introuvable")


def _load_conversation_or_404(
    conversation_id: str,
    jm: JSONManager,
) -> Conversation:
    """Charge {uuid}.json ou lève 404 si absent/invalide."""
    try:
        return jm.load_conversation(conversation_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Conversation {conversation_id} introuvable")
    except ValueError as exc:
        logger.error("Fichier %s.json invalide : %s", conversation_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Fichier conversation corrompu")

def _normalize_conversation_id(raw_id: str) -> str:
    """
    Normalise un UUID reçu en path (majuscules/minuscules) en forme canonique lower-case.
    Ex: 6284B51A-... -> 6284b51a-...
    """
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
    response_model=List[ConversationResponse],
    summary="Lister toutes les conversations",
)
def list_conversations(
    jm: JSONManager = Depends(get_json_manager),
) -> List[ConversationResponse]:
    """
    Retourne la liste des métadonnées, triée par updatedAt décroissant
    (conversation la plus récente en premier).
    """
    index = jm.load_index()
    sorted_index = sorted(index, key=lambda m: m.updatedAt, reverse=True)
    return [ConversationResponse.from_metadata(m) for m in sorted_index]


@router.post(
    "/",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Créer une conversation",
)
def create_conversation(
    body: CreateConversationRequest,
    jm: JSONManager = Depends(get_json_manager),
) -> ConversationResponse:
    """
    Génère un UUID pour l'ID de la conversation. createdAt et updatedAt sont
    """
    conv_id = str(uuid4())
    now = _utcnow()

    # Créer la conversation avec message system initial (compat Swift)
    conv = Conversation(
        id=conv_id,
        messages=[],
    )

    meta = ConversationMetadata(
        id=conv_id,
        title=body.title,
        createdAt=now,
        updatedAt=now,
        messageCount=0,  # le message system ne compte pas
    )

    # Sauvegarder fichier puis mettre à jour l'index
    jm.save_conversation(conv)

    index = jm.load_index()
    index.append(meta)
    jm.save_index(index)

    logger.info("Conversation créée : %s (%r)", conv_id, body.title)
    return ConversationResponse.from_metadata(meta)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Détail d'une conversation",
)
def get_conversation(
    conversation_id: str,
    jm: JSONManager = Depends(get_json_manager),
) -> ConversationDetailResponse:
    """
    Retourne la conversation complète (métadonnées + messages).

    Si {uuid}.json est absent alors que l'entrée index existe,
    load_index() aura déjà nettoyé l'index → 404 retourné.
    """
    conversation_id = _normalize_conversation_id(conversation_id)
    meta = _get_meta_or_404(conversation_id, jm)
    conv = _load_conversation_or_404(conversation_id, jm)
    return ConversationDetailResponse.from_metadata_and_conversation(meta, conv)


@router.put(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Modifier le titre ou les instructions spécifiques d'une conversation",
)
def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    jm: JSONManager = Depends(get_json_manager),
) -> ConversationResponse:
    """
    Modifie le titre. updatedAt est mis à jour, createdAt reste immutable.
    """
    conversation_id = _normalize_conversation_id(conversation_id)
    meta = _get_meta_or_404(conversation_id, jm)

    updated_meta = ConversationMetadata(
        id=meta.id,
        title=body.title if body.title is not None else meta.title,
        createdAt=meta.createdAt,
        updatedAt=_utcnow(),
        messageCount=meta.messageCount,
        specificInstruction=body.specificInstruction if body.specificInstruction is not None else meta.specificInstruction,
    )

    index = jm.load_index()
    new_index = [updated_meta if m.id == conversation_id else m for m in index]
    jm.save_index(new_index)

    logger.info("Conversation %s renommée : %r", conversation_id, body.title)
    return ConversationResponse.from_metadata(updated_meta)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer une conversation",
)
def delete_conversation(
    conversation_id: str,
    jm: JSONManager = Depends(get_json_manager),
) -> None:
    """
    Supprime le fichier {uuid}.json et retire l'entrée de l'index.

    Vérifie l'existence avant suppression pour retourner 404
    plutôt que de réussir silencieusement sur un ID inexistant.
    """
    conversation_id = _normalize_conversation_id(conversation_id)
    _get_meta_or_404(conversation_id, jm)

    try:
        jm.delete_conversation(conversation_id)
    except OSError as exc:
        logger.error("Erreur suppression %s : %s", conversation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur suppression conversation : {exc}",
        )

    logger.info("Conversation supprimée : %s", conversation_id)
