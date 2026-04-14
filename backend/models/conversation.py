"""
Modèles Pydantic pour les conversations.

Structure JSON compatible avec les modèles Swift :
  - ConversationMetadata ↔ ConversationMetadata.swift
  - Conversation         ↔ Conversation.swift

Historique stocké nativement en List[ModelMessage] (Pydantic AI) pour préserver
le contexte multi-tours incluant les tool calls et retours d'outils.

MessageMeta : métadonnées par message (id, timestamp) stockées en liste parallèle
à Conversation.messages. Les deux listes DOIVENT rester synchronisées : ajouter
un élément à l'une implique obligatoirement d'en ajouter un à l'autre dans le
même bloc avec verrou.

Datetime sérialisé en ISO 8601 avec timezone UTC pour compatibilité Swift Codable.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_serializer
from pydantic_ai.messages import ModelMessage

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageMeta(BaseModel):
    """Métadonnées par message (index-alignées avec Conversation.messages)."""
    id:        UUID     = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utcnow)

    @field_serializer("timestamp")
    def serialize_ts(self, v: datetime) -> str:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()


class ConversationMetadata(BaseModel):
    id:           str
    title:        str
    createdAt:    datetime = Field(default_factory=_utcnow)
    updatedAt:    datetime = Field(default_factory=_utcnow)
    messageCount: int      = 0
    specificInstruction: str | None = None

    @field_serializer("createdAt", "updatedAt")
    def serialize_datetime(self, v: datetime) -> str:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()


class Conversation(BaseModel):
    id:           str
    messages:     List[ModelMessage] = Field(default_factory=list)
    message_meta: List[MessageMeta]  = Field(default_factory=list)