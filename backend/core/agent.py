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
    "Tu es Iris, une assistante de consulting stratégique haute performance.\n"
    "Tu opères en local sur un Mac Mini M4 Pro (48 Go RAM, 100% privé — aucune "
    "donnée ne quitte le système).\n"
    "Tu es analytique, précise et directe.\n"
    "Pour chaque requête complexe, tu raisonnes étape par étape avant de répondre.\n\n"
    "Si un contexte utilisateur ou client est fourni dans les instructions, "
    "intègre-le naturellement dans ta réponse sans le mentionner explicitement."
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
    enable_thinkng: bool = False  # Active le mode "thinking" (jeton spécial Gemma 4)
    vector_store: Optional[Any] = None     # VectorStoreManager (Étape 2)
    user_context: Optional[str] = None    # Enrichissement table_user_core
    client_context: Optional[str] = None  # Contexte client courant
    conversation_system_prompt: Optional[str] = None  # Override du system prompt Iris (optionnel)

def build_dynamic_system_prompt(deps: IrisDeps) -> str:
    """Génère l'identité d'Iris à partir de ses dépendances."""
    sections = [
        "Tu es Iris, une assistante de consulting stratégique haute performance.",
        "Tu opères en local sur un Mac Mini M4 Pro (48 Go RAM, 100% privé).",
        "Tu es analytique, précise et directe."
    ]

    # Mode Thinking (Jeton spécial Gemma 4)
    if deps.enable_thinking:
        sections.insert(0, "<|think|>")
        
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


def _raw_to_model_response(
    raw_text: str,
    model_name: str,
    timestamp: datetime,
) -> ModelResponse:
    """
    Convertit la réponse brute de IrisEngine en ModelResponse Pydantic AI.

    Parsing :
      - Si [TOOL_CALLS] ou {"tool_calls": ...} détecté → ToolCallPart(s)
      - Sinon → TextPart unique

    Le parsing délègue à IrisEngine._parse_tool_calls() pour la cohérence
    avec le tool calling loop existant.
    """
    tool_calls, text_content = IrisEngine._parse_tool_calls(raw_text)

    parts: list[TextPart | ToolCallPart] = []

    if text_content:
        parts.append(TextPart(content=text_content))

    for tc in tool_calls:
        logger.info(
            "[iris:agent] tool_call → %s(%s)",
            tc["name"],
            json.dumps(tc["arguments"], ensure_ascii=False)[:120],
        )
        parts.append(
            ToolCallPart(
                tool_name=tc["name"],
                args=tc["arguments"],
                tool_call_id=tc["id"],
            )
        )

    if not parts:
        parts = [TextPart(content="")]

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
        Démarre engine.stream_messages() dans un thread et yield les events Pydantic AI.

        Chaque token est émis via ModelResponsePartsManager.handle_text_delta()
        qui produit un PartStartEvent (premier token) puis des PartDeltaEvent (suivants).
        """
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream_in_thread() -> None:
            try:
                for chunk in self._engine.stream_messages(
                    self._mlx_messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_stream_in_thread, daemon=True)
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                # handle_text_delta gère PartStartEvent (1er chunk) et PartDeltaEvent
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=None,
                    content=item,
                ):
                    yield event
        finally:
            thread.join(timeout=5)


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

        logger.info(
            "[iris:agent] request — %d messages, %d tools",
            len(mlx_messages),
            len(tools_openai),
        )

        # Paramètres de génération depuis ModelSettings
        max_tokens = 512
        temperature = 0.3
        if model_settings:
            if hasattr(model_settings, "max_tokens") and model_settings.max_tokens:
                max_tokens = model_settings.max_tokens
            if hasattr(model_settings, "temperature") and model_settings.temperature is not None:
                temperature = model_settings.temperature

        timestamp = datetime.now(tz=timezone.utc)

        # Génération (bloquante) dans un thread séparé pour ne pas bloquer FastAPI
        raw_response = await asyncio.to_thread(
            self._engine._generate_raw,
            mlx_messages,
            tools_openai if tools_openai else None,
            max_tokens,
            temperature,
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

        max_tokens = 512
        temperature = 0.3
        if model_settings:
            if hasattr(model_settings, "max_tokens") and model_settings.max_tokens:
                max_tokens = model_settings.max_tokens
            if hasattr(model_settings, "temperature") and model_settings.temperature is not None:
                temperature = model_settings.temperature

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
        )


# ── Factory create_iris_agent ─────────────────────────────────────────────────

def create_iris_agent(engine: IrisEngine) -> Agent[IrisDeps, str]:
    """
    Crée et retourne un Agent Pydantic AI configuré pour Iris.

    L'agent est paramétré avec :
      - IrisModel  : wrapper mlx-lm (A1)
      - IrisDeps   : injection LanceDB ready (Étape 2)
      - System prompt Iris (override possible par conversation via system_prompt=)
      - output_type=str : réponse texte brute (extensible en Étape 4)

    Utilisation :
        agent = create_iris_agent(engine)
        result = await agent.run("Ma question", deps=IrisDeps())
        print(result.data)

    Tools enregistrables (Étape 3) :
        @agent.tool
        async def mon_outil(ctx: RunContext[IrisDeps], param: str) -> str: ...

    Returns:
        Agent[IrisDeps, str] prêt à être stocké dans app.state.iris_agent.
    """
    model = IrisModel(engine=engine)

    agent: Agent[IrisDeps, str] = Agent(
        model=model,
        deps_type=IrisDeps,
        output_type=str,
    )

    logger.info("[iris:agent] Agent Iris initialisé sur modèle %s", model.model_name)
    return agent
