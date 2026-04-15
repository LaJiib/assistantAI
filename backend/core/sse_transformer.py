"""
sse_transformer.py — Transformateur d'événements Pydantic AI → SSE BFF.

Rôle : agir comme couche de traduction entre les événements natifs de Pydantic AI
et le protocole SSE JSON exposé au client iOS. Le client ne reçoit jamais
d'événements Pydantic AI bruts.

Protocole SSE BFF émis :
  {"type": "textDelta",      "content": "..."}
  {"type": "reasoningDelta", "content": "..."}
  {"type": "toolCallStart",  "toolCallId": "...", "toolName": "..."}
  {"type": "toolCallResult", "toolCallId": "...", "preview": {...}}  # preview optionnel

Avec mlx-openai-server (--reasoning-parser gemma4) :
  - Les balises <|channel>thought…<channel|> sont strippées côté serveur.
  - Le raisonnement est exposé comme delta.reasoning_content dans le flux SSE.
  - pydantic-ai 1.79.0 le capte nativement via _map_thinking_delta()
    → ThinkingPartDelta(content_delta=...) → reasoningDelta ici.
  - TextPartDelta ne contient jamais de balises de réflexion brutes.
  - Le raisonnement n'est pas persisté en base — pas de parsing historique nécessaire.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import AgentStreamEvent

logger = logging.getLogger(__name__)


# ── Transformateur principal ─────────────────────────────────────────────────

def transform_agent_event(event: "AgentStreamEvent") -> list[dict]:
    """
    Convertit un événement Pydantic AI natif en 0-N dicts BFF prêts à sérialiser.

    Avec mlx-openai-server, le flux pydantic-ai produit :
      - TextPartDelta       → textDelta (texte propre, sans balises de réflexion)
      - ThinkingPartDelta   → reasoningDelta (depuis delta.reasoning_content natif)
      - FunctionToolCallEvent  → toolCallStart
      - FunctionToolResultEvent → toolCallResult

    Les événements non pertinents pour le frontend (FinalResultEvent,
    PartEndEvent, BuiltinTool*) sont silencieusement ignorés.
    PartStartEvent est traité pour récupérer le 1er token de chaque part.
    """
    results: list[dict] = []

    # ── PartStartEvent — premier token de chaque part ───────────────────────
    # CRITIQUE : pydantic-ai place le 1er token dans PartStartEvent.part.content,
    # PAS dans un PartDeltaEvent. Ignorer cet event fait perdre le 1er chunk.
    if isinstance(event, PartStartEvent):
        part = event.part
        if isinstance(part, TextPart) and part.content:
            results.append({"type": "textDelta", "content": part.content})
        elif isinstance(part, ThinkingPart) and part.content:
            results.append({"type": "reasoningDelta", "content": part.content})
        # ToolCallPart via PartStartEvent → géré séparément par FunctionToolCallEvent

    # ── TextPartDelta / ThinkingPartDelta ────────────────────────────────────
    elif isinstance(event, PartDeltaEvent):
        delta = event.delta

        if isinstance(delta, TextPartDelta):
            if delta.content_delta:
                results.append({"type": "textDelta", "content": delta.content_delta})

        elif isinstance(delta, ThinkingPartDelta):
            # ThinkingPartDelta.content_delta : tokens de raisonnement
            # (delta.reasoning_content intercepté nativement par pydantic-ai)
            if delta.content_delta:
                results.append({"type": "reasoningDelta", "content": delta.content_delta})

        # ToolCallPartDelta : déltas des arguments d'outil — ignorés côté frontend
        # (on n'émet toolCallStart qu'une fois via FunctionToolCallEvent)

    # ── Appel d'outil confirmé ───────────────────────────────────────────────
    elif isinstance(event, FunctionToolCallEvent):
        part = event.part
        results.append({
            "type": "toolCallStart",
            "toolCallId": part.tool_call_id,
            "toolName": part.tool_name,
        })

    # ── Résultat d'outil ─────────────────────────────────────────────────────
    elif isinstance(event, FunctionToolResultEvent):
        result_part = event.result
        evt: dict = {
            "type": "toolCallResult",
            "toolCallId": event.tool_call_id,
        }
        # metadata : données structurées pour l'aperçu frontend.
        # Définies via ToolReturn(return_value=..., metadata=...) — jamais envoyées au LLM.
        if isinstance(result_part, ToolReturnPart) and result_part.metadata is not None:
            evt["preview"] = result_part.metadata
        results.append(evt)

    # Tous les autres (PartEndEvent, FinalResultEvent,
    # BuiltinToolCallEvent, BuiltinToolResultEvent) → ignorés

    return results
