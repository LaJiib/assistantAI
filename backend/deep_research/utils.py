"""Utility functions for the Deep Research subagent."""

import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional
import os
from langchain_core.messages import AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig






# ---------------------------------------------------------------------------
# Token limit utils
# ---------------------------------------------------------------------------

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Détecte si une exception indique un dépassement de contexte."""
    error_str = str(exception).lower()
    token_keywords = [
        "token", "context", "too long", "maximum", "length",
        "prompt is too long", "reduce"
    ]
    return any(keyword in error_str for keyword in token_keywords)


def get_model_token_limit(model_string: str) -> Optional[int]:
    """Limite de contexte pour Gemma4 via Omlx."""
    return 128000  # Gemma4 26B : 128k tokens


def remove_up_to_last_ai_message(
    messages: list[MessageLikeRepresentation],
) -> list[MessageLikeRepresentation]:
    """Tronque l'historique en supprimant jusqu'au dernier message AI."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages


# ---------------------------------------------------------------------------
# Date utils — identique au repo
# ---------------------------------------------------------------------------

def get_today_str() -> str:
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"