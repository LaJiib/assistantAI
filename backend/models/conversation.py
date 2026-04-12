"""
Modèles Pydantic pour les conversations.

Structure JSON 100% identique aux modèles Swift :
  - ConversationMetadata ↔ ConversationMetadata.swift
  - Conversation         ↔ Conversation.swift
  - Message              ↔ Message.swift

Datetime sérialisé en ISO 8601 avec timezone UTC pour compatibilité Swift Codable.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field, field_serializer
from pydantic_ai.messages import ModelMessage

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

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