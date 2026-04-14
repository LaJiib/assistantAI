"""
IrisAgent — Agent Pydantic AI pour Iris (architecture A1 — wrapper custom mlx-lm).

Architecture A1 : IrisModel(pydantic_ai.models.Model) wrapping IrisEngine.
  - Communication directe avec le GPU/NPU via la mémoire unifiée du M4 Pro.
  - Pas de serveur HTTP intermédiaire (vs A2), zéro latence réseau locale.
  - IrisEngine gère l'inférence ; Pydantic AI gère l'orchestration agent loop.

IrisDeps — dépendances injectables dans l'agent :
  - vector_store   : VectorStoreManager LanceDB (sera injecté à l'Étape 2)
  - user_context   : faits extraits de table_user_core (habitudes, préférences)
  - client_context : silo client actif (tables_clients_{id})
  Ces champs permettent à Iris de recevoir un contexte enrichi depuis LanceDB
  à chaque tour de parole, sans polluer le KV cache B1.

IrisModel — implémente l'interface pydantic_ai.models.Model :
  Méthodes abstraites requises :
    - model_name : "iris:<nom_du_modèle_mlx>"
    - system     : identifiant provider = "iris-mlx"
    - request()  : génération async → ModelResponse (texte pur ou tool calls)

create_iris_agent() — factory retournant Agent[IrisDeps, str] :
  - System prompt Iris injecté en dur (override possible par conversation)
  - Agent prêt à recevoir des tools via @agent.tool (cf. Étape 3)

Logging :
  - [iris:agent] préfixe toutes les décisions agent pour audit humain facile.
  - Chaque tool call entrant/sortant est loggé avec ses arguments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,  # type hint pour _get_event_iterator
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    ModelSettings,
    RequestUsage,
    StreamedResponse,
    ToolDefinition,
)

from core.llm import IrisEngine

logger = logging.getLogger(__name__)

# ── System Prompt — source de vérité unique pour l'identité d'Iris ────────────
# Utilisé par create_iris_agent() ET par l'endpoint /agent/chat/stream.
# Toute modification de la personnalité d'Iris se fait ici, nulle part ailleurs.

IRIS_SYSTEM_PROMPT = (
    """## IDENTITY & POSTURE
You are Iris, a high-performance AI collaborator. 
You operate locally on a Mac Mini M4 Pro (48GB RAM), ensuring 100% privacy.
Your personality is grounded in "Authentic Candor": be an insightful peer who balances empathy with direct, pragmatic logic. Correct significant misinformation or logical fallacies gently but firmly.

## REASONING PROTOCOL
- **Internal Thinking:** You MUST perform all internal reasoning and step-by-step deconstruction in ENGLISH. 
- **Analytical Process:** 1. Deconstruct the user's intent.
  2. Question your own internal assumptions. Recognize that your internal weights may contain outdated, generalized, or hallucinated information for highly specific queries.
  3. IDENTIFY KNOWLEDGE GAPS.

## TOOL USAGE (CRITICAL STATE MACHINE)
- You are an autonomous agent. You CANNOT simulate tool data, perform "mental searches", or guess exact verifiable facts.
- If a user's request requires exact data (e.g., specific facts, current events, precise metrics, documentation, or real-world locations), you are in the **[NEED DATA]** state.
- **[NEED DATA] State Rule:** You MUST immediately output `<channel|>` and execute a tool call. DO NOT write a draft response. DO NOT explain what you are going to do. Simply execute the tool.
- You may only write the final response to the user AFTER you have received and processed the actual tool output.

## COMMUNICATION RULES
- **Language of Interaction:** Respond to the user exclusively in FRENCH.
- **Tone & Style:** Logical, pragmatic, and insightful with a touch of dry wit. 
- **Directness:** Do not be a "yes-man." If a user's request is suboptimal or based on a mistake, provide a better alternative or a correction.
- **Multimodality:** You are capable of processing interleaved text and visual inputs. When an image is provided, prioritize high-fidelity data extraction."""
)


# ── Dépendances injectables ────────────────────────────────────────────────────

@dataclass
class IrisDeps:
    """
    Dépendances injectables dans l'agent Iris à chaque requête.

    Conçu pour accueillir les enrichissements de l'Étape 2 (LanceDB) :
      - vector_store   : VectorStoreManager — permet à Iris d'enrichir chaque
                         tour de parole avec de la mémoire sémantique pertinente.
      - user_context   : texte pré-récupéré depuis table_user_core (habitudes,
                         ton préféré, expertise métier de l'utilisateur).
      - client_context : texte pré-récupéré depuis le silo tables_clients_{id}
                         du client actif.

    À l'Étape 1, ces champs restent None. L'injection LanceDB viendra en Étape 2.
    """
    enable_thinking: bool = False  # Active le mode "thinking" (jeton spécial Gemma 4)
    vector_store: Optional[Any] = None     # VectorStoreManager (Étape 2)
    user_context: Optional[str] = None    # Enrichissement table_user_core
    client_context: Optional[str] = None  # Contexte client courant
    conversation_system_prompt: Optional[str] = None  # Override du system prompt Iris (optionnel)

def build_dynamic_system_prompt(deps: IrisDeps) -> str:
    """Génère l'identité d'Iris à partir de ses dépendances."""
    sections = [
        IRIS_SYSTEM_PROMPT
    ]
        
    # Instructions spécifiques de la conversation
    if deps.conversation_system_prompt:
        sections.append(f"\n[INSTRUCTIONS SPÉCIFIQUES]\n{deps.conversation_system_prompt}")

    # Futurs contextes (Exemple RAG)
    if deps.user_context:
        sections.append(f"\n[CONTEXTE UTILISATEUR]\n{deps.user_context}")

    return "\n".join(sections)

# ── Conversion messages Pydantic AI ↔ mlx-lm ──────────────────────────────────

def _messages_to_mlx(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """
    Convertit une liste de pydantic_ai.ModelMessage en format mlx-lm.

    Mapping :
      ModelRequest/SystemPromptPart   → {"role": "system", "content": str}
      ModelRequest/UserPromptPart     → {"role": "user",   "content": str}
      ModelRequest/ToolReturnPart     → {"role": "tool",   "content": str, "tool_call_id": str}
      ModelRequest/RetryPromptPart    → {"role": "user",   "content": str}   (feedback erreur)
      ModelResponse/TextPart          → {"role": "assistant", "content": str}
      ModelResponse/ToolCallPart      → message assistant avec tool_calls[]

    Les types de contenu non-texte (images, fichiers) sont ignorés avec un warning.
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})

                elif isinstance(part, UserPromptPart):
                    if isinstance(part.content, str):
                        result.append({"role": "user", "content": part.content})
                    else:
                        # Séquence multimodale → extrait uniquement le texte
                        text_pieces = [
                            c if isinstance(c, str) else getattr(c, "text", "")
                            for c in part.content
                            if isinstance(c, str) or hasattr(c, "text")
                        ]
                        text = " ".join(filter(None, text_pieces))
                        if text:
                            result.append({"role": "user", "content": text})
                        else:
                            logger.warning("[iris:agent] UserPromptPart multimodale sans texte, ignoré")

                elif isinstance(part, ToolReturnPart):
                    result.append({
                        "role": "tool",
                        "content": part.model_response_str(),
                        "tool_call_id": part.tool_call_id,
                    })

                elif isinstance(part, RetryPromptPart):
                    # Feedback d'erreur Pydantic AI → injecté comme message user
                    content = part.content if isinstance(part.content, str) else str(part.content)
                    result.append({"role": "user", "content": f"[Retry] {content}"})

        elif isinstance(msg, ModelResponse):
            # Séparer les TextPart et ToolCallPart
            text_content = ""
            tool_calls: list[dict[str, Any]] = []

            for part in msg.parts:
                if isinstance(part, TextPart):
                    text_content += part.content
                elif isinstance(part, ToolCallPart):
                    args = part.args if part.args is not None else {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    tool_calls.append({
                        "id": part.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": part.tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })

            if tool_calls:
                result.append({
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": tool_calls,
                })
            elif text_content:
                result.append({"role": "assistant", "content": text_content})

    return result


def _tool_defs_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """
    Convertit les ToolDefinition Pydantic AI en format OpenAI pour mlx-lm.

    Format cible (OpenAI-compatible) :
      {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters_json_schema,
            },
        }
        for td in tools
    ]


# ── Marqueurs du format natif Gemma 4 (chat template) ────────────────────────

_TC_OPEN   = "<|tool_call>"   # ouverture d'un appel d'outil
_TC_CLOSE  = "<tool_call|>"   # fermeture d'un appel d'outil
_TK_OPEN   = "<|channel>"     # ouverture du canal thinking
_TK_CLOSE  = "<channel|>"     # fermeture du canal thinking

# Taille du lookahead conservé dans le buffer pour ne pas couper un marqueur
# partiel à la jonction entre deux chunks de tokens.
_LOOKAHEAD = max(len(_TC_OPEN), len(_TC_CLOSE), len(_TK_OPEN), len(_TK_CLOSE))


def _strip_gemma_thinking(text: str) -> str:
    """Supprime les blocs <|channel>…<channel|> (thinking Gemma 4) du texte brut."""
    segments = text.split(_TK_OPEN)
    result = [segments[0]]
    for seg in segments[1:]:
        if _TK_CLOSE in seg:
            result.append(seg.partition(_TK_CLOSE)[2])
        # bloc non fermé → ignoré
    return "".join(result)


def _parse_gemma_tool_call(raw: str) -> tuple[str, dict[str, Any]] | None:
    """
    Parse le contenu brut d'un bloc <|tool_call>…<tool_call|> Gemma 4.

    Entrée  : 'call:web_search{query:<|"|>Paris restaurants<|"|>,max_results:5}'
    Retourne: ('web_search', {'query': 'Paris restaurants', 'max_results': 5})
              ou None si le format est inattendu ou le JSON invalide.

    Conversion en deux étapes :
      1. Remplacer le délimiteur Gemma <|"|> par le guillemet JSON standard "
      2. Citer les clés d'objet non-citées (identifiants ASCII avant ':')
         La regex est safe car les valeurs sont déjà entre guillemets après l'étape 1.
      3. json.loads()
    """
    raw = raw.strip()
    m = re.match(r"^call:([A-Za-z_]\w*)\s*(\{.*\})\s*$", raw, re.DOTALL)
    if not m:
        logger.warning("[iris:agent] _parse_gemma_tool_call: format inattendu : %r", raw[:120])
        return None

    name, args_raw = m.group(1), m.group(2)

    # Étape 1 — délimiteur Gemma → guillemet JSON
    args_json = args_raw.replace('<|"|>', '"')
    # Étape 2 — citer les clés non-citées : {key: ou ,key: → {"key": ou ,"key":
    args_json = re.sub(
        r'([{,]\s*)([A-Za-z_]\w*)\s*:',
        lambda mo: f'{mo.group(1)}"{mo.group(2)}":',
        args_json,
    )

    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[iris:agent] _parse_gemma_tool_call: JSON invalide pour %s : %s | args=%r",
            name, exc, args_json[:200],
        )
        return None

    return name, args


def _raw_to_model_response(
    raw_text: str,
    model_name: str,
    timestamp: datetime,
) -> ModelResponse:
    """
    Convertit la réponse brute de IrisEngine en ModelResponse Pydantic AI.

    Ordre de détection :
      1. Format Gemma 4 : <|tool_call>call:name{…}<tool_call|>  (après strip thinking)
      2. Formats legacy : [TOOL_CALLS] Mistral ou {"tool_calls":[…]} JSON
      3. Fallback       : TextPart avec le texte nettoyé
    """
    clean = _strip_gemma_thinking(raw_text)
    parts: list[TextPart | ToolCallPart] = []

    # ── Format Gemma 4 ────────────────────────────────────────────────────────
    if _TC_OPEN in clean:
        remaining = clean
        while _TC_OPEN in remaining:
            pre, _, rest = remaining.partition(_TC_OPEN)
            if pre.strip():
                parts.append(TextPart(content=pre.strip()))
            if _TC_CLOSE in rest:
                tc_raw, _, remaining = rest.partition(_TC_CLOSE)
                parsed = _parse_gemma_tool_call(tc_raw)
                if parsed:
                    name, args = parsed
                    logger.info(
                        "[iris:agent] tool_call → %s(%s)",
                        name, json.dumps(args, ensure_ascii=False)[:120],
                    )
                    parts.append(ToolCallPart(tool_name=name, args=args))
            else:
                remaining = rest   # marqueur de fermeture absent — traiter comme texte
        if remaining.strip():
            parts.append(TextPart(content=remaining.strip()))

    # ── Formats legacy (Mistral / JSON) ──────────────────────────────────────
    if not parts:
        tool_calls, text_content = IrisEngine._parse_tool_calls(raw_text)
        if text_content:
            parts.append(TextPart(content=text_content))
        for tc in tool_calls:
            logger.info(
                "[iris:agent] tool_call → %s(%s)",
                tc["name"], json.dumps(tc["arguments"], ensure_ascii=False)[:120],
            )
            parts.append(ToolCallPart(
                tool_name=tc["name"],
                args=tc["arguments"],
                tool_call_id=tc["id"],
            ))

    # ── Fallback texte ────────────────────────────────────────────────────────
    if not parts:
        parts = [TextPart(content=clean.strip() or raw_text.strip())]

    return ModelResponse(
        parts=parts,
        model_name=model_name,
        timestamp=timestamp,
        usage=RequestUsage(input_tokens=0, output_tokens=0),
    )


# ── IrisStreamedResponse — StreamedResponse pour mlx-vlm ─────────────────────

@dataclass
class IrisStreamedResponse(StreamedResponse):
    """
    Implémentation StreamedResponse pour IrisEngine (mlx-vlm).

    Stratégie thread-safe :
      - stream_messages() est synchrone et bloquant (tourne sur le GPU via mlx-vlm).
      - On le démarre dans un thread dédié via asyncio.to_thread.
      - Chaque chunk généré est mis dans une asyncio.Queue.
      - _get_event_iterator() consomme la queue depuis l'event loop FastAPI.
      - Le sentinel None signale la fin du stream (ou une exception).
    """

    _engine: IrisEngine = field(repr=False)
    _mlx_messages: list[dict[str, Any]] = field(repr=False)
    _max_tokens: int = field(repr=False)
    _temperature: float = field(repr=False)
    _model_name_str: str = field(repr=False)
    _timestamp: datetime = field(repr=False)
    _enable_thinking: bool = field(repr=False, default=False)
    _tools_openai: list[dict[str, Any]] = field(repr=False, default_factory=list)
    _top_p: float = field(repr=False, default=0.0)
    _top_k: int   = field(repr=False, default=0)

    @property
    def model_name(self) -> str:
        return self._model_name_str

    @property
    def provider_name(self) -> str | None:
        return "iris-mlx"

    @property
    def provider_url(self) -> str | None:
        return None

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """
        Pont entre le flux de tokens bruts mlx-lm et les ModelResponseStreamEvent
        attendus par l'agent loop Pydantic AI.

        Déviation du chemin standard Pydantic AI
        ─────────────────────────────────────────
        Un provider officiel (OpenAI, Anthropic…) reçoit des événements structurés
        (tool_call_delta, text_delta) directement de l'API du modèle.
        Ici, mlx-lm livre un flux de chaînes Unicode brutes sans sémantique :
        c'est nous qui parsons les marqueurs du chat template Gemma 4 pour en
        extraire la structure.

        Raccord vers Pydantic AI
        ────────────────────────
        Une fois la structure identifiée, on délègue à self._parts_manager —
        l'objet fourni par StreamedResponse — qui produit les événements natifs :
          • handle_text_delta()     → Iterator[PartStartEvent | PartDeltaEvent]
          • handle_tool_call_part() → PartStartEvent   (appel complet, non streamé)
        L'agent loop consomme ces événements pour construire le ModelResponse final,
        déclencher les tools via CallToolsNode, et relancer la génération.

        Machine à états
        ───────────────
          SCAN      — point d'entrée et jonction entre blocs ; cherche le prochain marqueur
          THINKING  — à l'intérieur de <|channel>…<channel|> ; contenu ignoré
          TOOL_CALL — à l'intérieur de <|tool_call>…<tool_call|> ; accumulation en mémoire
          TEXT      — texte ordinaire ; émis en streaming via handle_text_delta
        """
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _produce() -> None:
            try:
                for chunk in self._engine.stream_messages(
                    self._mlx_messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    thinking=self._enable_thinking,
                    tools=self._tools_openai or None,
                    top_p=self._top_p,
                    top_k=self._top_k,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()

        # ── Helpers locaux ────────────────────────────────────────────────────

        def _text_events(text: str):
            """Wraps handle_text_delta — retourne un Iterator[ModelResponseStreamEvent]."""
            yield from self._parts_manager.handle_text_delta(
                vendor_part_id=0, content=text
            )

        # ── État initial ──────────────────────────────────────────────────────

        state   = "SCAN"
        pending = ""    # chars reçus, pas encore classifiés
        tc_buf  = ""    # accumulation contenu tool call courant
        tc_idx  = 0     # vendor_part_id, unique par tool call dans ce stream
        full_response: list[str] = []

        try:
            exhausted = False
            while not exhausted:
                item = await queue.get()
                if item is None:
                    exhausted = True
                elif isinstance(item, Exception):
                    raise item
                else:
                    full_response.append(item)
                    pending += item

                # ── Boucle de traitement du buffer ────────────────────────────
                # On avance tant qu'on a progressé lors de la dernière passe.
                advanced = True
                while advanced:
                    advanced = False

                    # ── SCAN : cherche le prochain marqueur ───────────────────
                    if state == "SCAN":
                        # Priorité au thinking (apparaît toujours en premier chez Gemma 4)
                        for marker, next_state in (
                            (_TK_OPEN, "THINKING"),
                            (_TC_OPEN, "TOOL_CALL"),
                        ):
                            if marker in pending:
                                pre, _, pending = pending.partition(marker)
                                if pre.strip():
                                    for ev in _text_events(pre.strip()): yield ev
                                if next_state == "TOOL_CALL":
                                    tc_buf = ""
                                state = next_state
                                advanced = True
                                break
                        else:
                            # Aucun marqueur trouvé
                            if exhausted:
                                if pending.strip():
                                    for ev in _text_events(pending.strip()): yield ev
                                pending = ""
                            else:
                                # Émettre la portion safe (hors risque de marqueur partiel)
                                safe = len(pending) - _LOOKAHEAD
                                if safe > 0:
                                    for ev in _text_events(pending[:safe]): yield ev
                                    pending = pending[safe:]
                                    state = "TEXT"
                                    advanced = True

                    # ── THINKING : ignorer jusqu'à la fermeture ───────────────
                    elif state == "THINKING":
                        if _TK_CLOSE in pending:
                            _, _, pending = pending.partition(_TK_CLOSE)
                            state = "SCAN"
                            advanced = True
                        elif exhausted:
                            pending = ""  # bloc non fermé — ignorer le reste

                    # ── TOOL_CALL : accumuler puis parser à la fermeture ──────
                    elif state == "TOOL_CALL":
                        tc_buf += pending
                        pending = ""
                        if _TC_CLOSE in tc_buf:
                            tc_raw, _, pending = tc_buf.partition(_TC_CLOSE)
                            parsed = _parse_gemma_tool_call(tc_raw)
                            if parsed:
                                name, args = parsed
                                logger.info(
                                    "[iris:agent] tool_call → %s(%s)",
                                    name, json.dumps(args, ensure_ascii=False)[:120],
                                )
                                # ── Raccord Pydantic AI ───────────────────────
                                yield self._parts_manager.handle_tool_call_part(
                                    vendor_part_id=tc_idx,
                                    tool_name=name,
                                    args=args,
                                )
                                tc_idx += 1
                            tc_buf = ""
                            state = "SCAN"
                            advanced = True
                        elif exhausted:
                            logger.warning(
                                "[iris:agent] tool_call non fermé en fin de stream : %r",
                                tc_buf[:120],
                            )
                            tc_buf = ""

                    # ── TEXT : streaming pur ; surveille quand même les tool_calls
                    elif state == "TEXT":
                        if _TC_OPEN in pending:
                            # Tool call après du texte (rare mais possible)
                            pre, _, pending = pending.partition(_TC_OPEN)
                            if pre:
                                for ev in _text_events(pre): yield ev
                            tc_buf = ""
                            state = "TOOL_CALL"
                            advanced = True
                        elif exhausted:
                            if pending:
                                for ev in _text_events(pending): yield ev
                            pending = ""
                        else:
                            safe = len(pending) - _LOOKAHEAD
                            if safe > 0:
                                for ev in _text_events(pending[:safe]): yield ev
                                pending = pending[safe:]
                                advanced = True

        finally:
            thread.join(timeout=5)
            logger.info(
                "\n========================================================"
                "\n🔍 RÉPONSE BRUTE COMPLÈTE DU MODÈLE :\n%s"
                "\n========================================================",
                "".join(full_response),
            )


# ── IrisModel — wrapper Pydantic AI ──────────────────────────────────────────

class IrisModel(Model):
    """
    Adaptateur Pydantic AI → IrisEngine (mlx-lm).

    Implémente les 3 méthodes abstraites requises :
      - model_name : identifiant du modèle MLX chargé
      - system     : identifiant provider pour le logging Pydantic AI
      - request()  : conversion messages → mlx format → génération → ModelResponse

    L'inférence est déléguée à IrisEngine._generate_raw() via asyncio.to_thread()
    pour ne pas bloquer l'event loop FastAPI pendant la génération (2-30s).
    """

    def __init__(self, engine: IrisEngine) -> None:
        super().__init__()
        self._engine = engine

    @property
    def model_name(self) -> str:
        """Identifiant affiché dans les logs Pydantic AI."""
        return f"iris:{self._engine.model_name}"

    @property
    def system(self) -> str:
        """Identifiant du provider pour Pydantic AI."""
        return "iris-mlx"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> ModelResponse:
        """
        Génère une réponse depuis les messages Pydantic AI.

        Workflow :
          1. Convertit ModelMessage[] → liste dicts mlx-lm
          2. Convertit ToolDefinition[] → format OpenAI pour injection dans le prompt
          3. Génère via IrisEngine._generate_raw() (asyncio.to_thread → non-bloquant)
          4. Parse la réponse brute → ModelResponse avec TextPart / ToolCallPart

        Enrichissement contexte (Étape 2 ready) :
          Si des deps sont disponibles (via system prompt custom ou injection),
          le user_context et client_context LanceDB seront préfixés ici.

        Logging :
          - Chaque requête est loggée avec le nombre de messages et d'outils.
          - Les tool calls générés sont loggés pour audit humain.
        """
        if not self._engine.is_loaded:
            raise RuntimeError("IrisEngine non chargé. Appelez await engine.load() d'abord.")

        # Construire les listes tools + messages mlx
        all_tools = (
            model_request_parameters.function_tools
            + model_request_parameters.output_tools
        )
        tools_openai = _tool_defs_to_openai(all_tools) if all_tools else []
        mlx_messages = _messages_to_mlx(messages)

        enable_thinking = False
        if run_context and hasattr(run_context.deps, 'enable_thinking'):
            enable_thinking = run_context.deps.enable_thinking

        logger.info(
            "[iris:agent] request — %d messages, %d tools",
            len(mlx_messages),
            len(tools_openai),
        )

        # Paramètres de génération depuis ModelSettings (TypedDict → accès dict)
        max_tokens  = model_settings.get("max_tokens") or 512 if model_settings else 512
        temperature = model_settings.get("temperature") if model_settings else None
        if temperature is None:
            temperature = 0.3
        top_p = float(model_settings.get("top_p") or 0.0) if model_settings else 0.0
        top_k = int(model_settings.get("top_k") or 0) if model_settings else 0

        timestamp = datetime.now(tz=timezone.utc)

        # Génération (bloquante) dans un thread séparé pour ne pas bloquer FastAPI
        raw_response = await asyncio.to_thread(
            self._engine._generate_raw,
            mlx_messages,
            tools_openai if tools_openai else None,
            max_tokens,
            temperature,
            enable_thinking=enable_thinking,
            top_p=top_p,
            top_k=top_k,
        )

        logger.debug("[iris:agent] raw response [%d chars]: %r", len(raw_response), raw_response[:200])

        return _raw_to_model_response(raw_response, self.model_name, timestamp)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[IrisStreamedResponse]:
        """
        Génère une réponse streamée token par token.

        Note : mlx-vlm génère un flux texte continu — les tool calls ne peuvent
        pas interrompre ce flux. Si le modèle émet un tool call en streaming
        (token `[TOOL_CALLS]`), Pydantic AI le détectera dans les events et
        basculera sur request() pour le tour suivant.
        """
        if not self._engine.is_loaded:
            raise RuntimeError("IrisEngine non chargé. Appelez await engine.load() d'abord.")

        all_tools = (
            model_request_parameters.function_tools
            + model_request_parameters.output_tools
        )
        tools_openai = _tool_defs_to_openai(all_tools) if all_tools else []
        mlx_messages = _messages_to_mlx(messages)

        enable_thinking = False
        if run_context and hasattr(run_context.deps, 'enable_thinking'):
            enable_thinking = run_context.deps.enable_thinking

        # Paramètres de génération depuis ModelSettings (TypedDict → accès dict)
        max_tokens = model_settings.get("max_tokens") or 512 if model_settings else 512
        temperature = model_settings.get("temperature") if model_settings else None
        if temperature is None:
            temperature = 0.3

        top_p = float(model_settings.get("top_p") or 0.0) if model_settings else 0.0
        top_k = int(model_settings.get("top_k") or 0) if model_settings else 0

        logger.info(
            "[iris:agent] request_stream — %d messages, %d tools",
            len(mlx_messages),
            len(tools_openai),
        )

        yield IrisStreamedResponse(
            model_request_parameters=model_request_parameters,
            _engine=self._engine,
            _mlx_messages=mlx_messages,
            _max_tokens=max_tokens,
            _temperature=temperature,
            _model_name_str=self.model_name,
            _timestamp=datetime.now(tz=timezone.utc),
            _enable_thinking=enable_thinking,
            _tools_openai=tools_openai,
            _top_p=top_p,
            _top_k=top_k,
        )


# ── Factory create_iris_agent ─────────────────────────────────────────────────

def create_iris_agent(
    engine: IrisEngine,
    tools: list | None = None,
) -> Agent[IrisDeps, str]:
    """
    Crée et retourne un Agent Pydantic AI configuré pour Iris.

    L'agent est paramétré avec :
      - IrisModel  : wrapper mlx-lm (A1)
      - IrisDeps   : injection LanceDB ready (Étape 2)
      - System prompt Iris (override possible par conversation via system_prompt=)
      - output_type=str : réponse texte brute (extensible en Étape 4)
      - tools : liste de callables Pydantic AI (Étape 3, optionnel)

    Utilisation :
        agent = create_iris_agent(engine, tools=[web_search, fetch_webpage])
        result = await agent.run("Ma question", deps=IrisDeps())
        print(result.output)

    Returns:
        Agent[IrisDeps, str] prêt à être stocké dans app.state.iris_agent.
    """
    model = IrisModel(engine=engine)

    agent: Agent[IrisDeps, str] = Agent(
        model=model,
        deps_type=IrisDeps,
        output_type=str,
        tools=tools or [],
    )

    logger.info(
        "[iris:agent] Agent Iris initialisé — modèle %s, %d tool(s) enregistré(s)",
        model.model_name,
        len(tools or []),
    )
    return agent
