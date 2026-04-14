"""
sse_transformer.py — Transformateur d'événements Pydantic AI → SSE BFF.

Rôle : agir comme couche de traduction entre les événements natifs de Pydantic AI
et le protocole SSE JSON exposé au client iOS. Le client ne reçoit jamais
d'événements Pydantic AI bruts, ni de balises de réflexion Gemma.

Protocole SSE BFF émis :
  {"type": "textDelta",      "content": "..."}
  {"type": "reasoningDelta", "content": "..."}
  {"type": "toolCallStart",  "toolCallId": "...", "toolName": "..."}
  {"type": "toolCallResult", "toolCallId": "...", "content": "..."}

Gemma 4 signale son raisonnement avec les balises :
  OPEN  : <|channel>thought
  CLOSE : <channel|>

Ces balises peuvent être découpées sur plusieurs TextPartDelta consécutifs.
GemmaThinkingParser bufférise les données pour les détecter de façon fiable
même si elles arrivent fractionnées.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolReturnPart,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import AgentStreamEvent

logger = logging.getLogger(__name__)

# ── Balises de réflexion Gemma 4 ────────────────────────────────────────────
_OPEN_TAG  = "<|channel>thought"
_CLOSE_TAG = "<channel|>"
# Taille du buffer de sécurité pour détecter les balises découpées entre chunks
_GUARD = max(len(_OPEN_TAG), len(_CLOSE_TAG))


class GemmaThinkingParser:
    """
    Buffer-state machine pour intercepter les balises de réflexion de Gemma.

    Utilisation :
        parser = GemmaThinkingParser()
        for chunk in text_stream:
            for event_type, content in parser.feed(chunk):
                # event_type: "textDelta" ou "reasoningDelta"
                emit(event_type, content)
        # En fin de stream, vider le buffer résiduel
        for event_type, content in parser.flush():
            emit(event_type, content)
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_thinking: bool = False

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """
        Ingère un chunk de texte et retourne une liste de (type, contenu).

        On garde toujours les _GUARD derniers caractères en buffer pour ne pas
        émettre prématurément du texte qui serait la première moitié d'une balise.
        """
        self._buffer += chunk
        events: list[tuple[str, str]] = []

        while True:
            if not self._in_thinking:
                idx = self._buffer.find(_OPEN_TAG)
                if idx == -1:
                    # Pas de balise d'ouverture — émettre tout sauf le garde
                    safe_len = max(0, len(self._buffer) - _GUARD)
                    if safe_len > 0:
                        events.append(("textDelta", self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                    break
                else:
                    # Texte avant la balise d'ouverture
                    if idx > 0:
                        events.append(("textDelta", self._buffer[:idx]))
                    self._buffer = self._buffer[idx + len(_OPEN_TAG):]
                    self._in_thinking = True
            else:
                idx = self._buffer.find(_CLOSE_TAG)
                if idx == -1:
                    # Toujours dans le thinking — émettre tout sauf le garde
                    safe_len = max(0, len(self._buffer) - _GUARD)
                    if safe_len > 0:
                        events.append(("reasoningDelta", self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                    break
                else:
                    # Raisonnement jusqu'à la balise de fermeture
                    if idx > 0:
                        events.append(("reasoningDelta", self._buffer[:idx]))
                    self._buffer = self._buffer[idx + len(_CLOSE_TAG):]
                    self._in_thinking = False

        return events

    def flush(self) -> list[tuple[str, str]]:
        """Vide le buffer résiduel à la fin du stream."""
        if not self._buffer:
            return []
        event_type = "reasoningDelta" if self._in_thinking else "textDelta"
        result = [(event_type, self._buffer)]
        self._buffer = ""
        return result


# ── Parsing statique (messages historiques) ──────────────────────────────────

def parse_thinking_tags(text: str) -> list[tuple[str, str]]:
    """
    Parse un texte complet contenant des balises Gemma et retourne une liste
    de (type, contenu) prête à être convertie en MessagePartResponse.

    Utilisé pour les messages chargés depuis la base de données, où le texte
    stocké dans TextPart peut encore contenir les balises brutes.

    Exemple :
        "Intro <|channel>thought raisonnement <channel|> suite"
        → [("text", "Intro "), ("reasoning", " raisonnement "), ("text", " suite")]
    """
    results: list[tuple[str, str]] = []
    remaining = text
    in_thinking = False

    while remaining:
        if not in_thinking:
            idx = remaining.find(_OPEN_TAG)
            if idx == -1:
                if remaining:
                    results.append(("text", remaining))
                break
            if idx > 0:
                results.append(("text", remaining[:idx]))
            remaining = remaining[idx + len(_OPEN_TAG):]
            in_thinking = True
        else:
            idx = remaining.find(_CLOSE_TAG)
            if idx == -1:
                # Balise de fermeture manquante — tout le reste est du reasoning
                if remaining:
                    results.append(("reasoning", remaining))
                break
            if idx > 0:
                results.append(("reasoning", remaining[:idx]))
            remaining = remaining[idx + len(_CLOSE_TAG):]
            in_thinking = False

    return [(t, c) for t, c in results if c]


# ── Transformateur principal ─────────────────────────────────────────────────

def transform_agent_event(
    event: "AgentStreamEvent",
    parser: GemmaThinkingParser,
) -> list[dict]:
    """
    Convertit un événement Pydantic AI natif en 0-N dicts BFF prêts à sérialiser.

    Les événements non pertinents pour le frontend (FinalResultEvent,
    PartStartEvent, PartEndEvent, BuiltinTool*) sont silencieusement ignorés.
    L'appelant est responsable de capturer AgentRunResultEvent séparément
    (il n'est pas de type AgentStreamEvent).
    """
    results: list[dict] = []

    # ── TextPartDelta : texte ou raisonnement (selon balises Gemma) ──────────
    if isinstance(event, PartDeltaEvent):
        delta = event.delta

        if isinstance(delta, TextPartDelta):
            for event_type, content in parser.feed(delta.content_delta):
                if content:
                    results.append({"type": event_type, "content": content})

        elif isinstance(delta, ThinkingPartDelta):
            # Le modèle supporte le thinking natif (ex: Claude) — émettre directement
            if delta.thinking_delta:
                results.append({"type": "reasoningDelta", "content": delta.thinking_delta})

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
        content = ""
        if isinstance(result_part, ToolReturnPart):
            content = str(result_part.content)
        results.append({
            "type": "toolCallResult",
            "toolCallId": event.tool_call_id,
            "content": content,
        })

    # Tous les autres (PartStartEvent, PartEndEvent, FinalResultEvent,
    # BuiltinToolCallEvent, BuiltinToolResultEvent) → ignorés

    return results
