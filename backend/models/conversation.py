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
from enum import Enum
from typing import List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    user      = "user"
    assistant = "assistant"
    system    = "system"


class Message(BaseModel):
    id:        UUID     = Field(default_factory=uuid4)
    role:      Role
    content:   str
    timestamp: datetime = Field(default_factory=_utcnow)

    @field_serializer("id")
    def serialize_id(self, v: UUID) -> str:
        return str(v)

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        # ISO 8601 avec timezone explicite → Swift Codable décode nativement
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()


class ConversationMetadata(BaseModel):
    id:           str
    title:        str
    systemPrompt: str
    createdAt:    datetime = Field(default_factory=_utcnow)
    updatedAt:    datetime = Field(default_factory=_utcnow)
    messageCount: int      = 0

    @field_serializer("createdAt", "updatedAt")
    def serialize_datetime(self, v: datetime) -> str:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()


class Conversation(BaseModel):
    id:           str
    systemPrompt: str
    messages:     List[Message] = Field(default_factory=list)
