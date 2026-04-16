"""
IrisAgent — Agent Pydantic AI pour Iris (architecture A2 — mlx-openai-server).

Architecture A2 : ReasoningAwareOpenAIModel(OpenAIChatModel) → mlx-openai-server (port 8001).
  - mlx-openai-server expose une API OpenAI-compatible avec parsers Gemma 4 natifs.
  - pydantic-ai orchestre le tool loop sans aucun parsing custom.
  - delta.reasoning_content est géré nativement par pydantic-ai 1.79.0 :
    OpenAIStreamedResponse._map_thinking_delta() → ThinkingPartDelta.

IrisDeps — dépendances injectables dans l'agent :
  - vector_store   : VectorStoreManager LanceDB (sera injecté à l'Étape 2)
  - user_context   : faits extraits de table_user_core
  - client_context : silo client actif
  - enable_thinking: vestigial — mlx-openai-server active toujours le reasoning

create_iris_agent(model, tools) — factory retournant Agent[IrisDeps, str] :
  - System prompt Iris injecté en dur
  - Agent prêt à recevoir des tools via @agent.tool
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

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
      - vector_store   : VectorStoreManager — mémoire sémantique pertinente.
      - user_context   : texte pré-récupéré depuis table_user_core.
      - client_context : texte pré-récupéré depuis le silo tables_clients_{id}.

    enable_thinking : vestigial — mlx-openai-server active le reasoning via
      --reasoning-parser gemma4, indépendamment de ce champ.
      Conservé pour la compatibilité de l'API Swift (options.think).
    """
    enable_thinking: bool = False          # vestigial avec mlx-openai-server
    vector_store: Optional[Any] = None     # VectorStoreManager (Étape 2)
    user_context: Optional[str] = None    # Enrichissement table_user_core
    client_context: Optional[str] = None  # Contexte client courant
    conversation_system_prompt: Optional[str] = None  # Override system prompt Iris


def build_dynamic_system_prompt(deps: IrisDeps) -> str:
    """Génère l'identité d'Iris à partir de ses dépendances."""
    sections = [IRIS_SYSTEM_PROMPT]

    if deps.conversation_system_prompt:
        sections.append(f"\n[INSTRUCTIONS SPÉCIFIQUES]\n{deps.conversation_system_prompt}")

    if deps.user_context:
        sections.append(f"\n[CONTEXTE UTILISATEUR]\n{deps.user_context}")

    return "\n".join(sections)





# ── Factory create_iris_agent ─────────────────────────────────────────────────

def create_iris_agent(
    model: OpenAIChatModel,
    tools: list | None = None,
) -> Agent[IrisDeps, str]:
    """
    Crée et retourne un Agent Pydantic AI configuré pour Iris.

    L'agent est paramétré avec :
      - OpenAIChatModel pointant vers mlx-vlm-server (port 8001)

      - IrisDeps   : injection LanceDB ready (Étape 2)
      - output_type=str : réponse texte brute
      - tools : liste de callables Pydantic AI (web_search, fetch_webpage)

    Returns:
        Agent[IrisDeps, str] prêt à être stocké dans app.state.iris_agent.
    """
    agent: Agent[IrisDeps, str] = Agent(
        model=model,
        deps_type=IrisDeps,
        output_type=str,
        tools=tools or [],
    )

    logger.info(
        "[iris:agent] Agent Iris initialisé — %d tool(s) enregistré(s)",
        len(tools or []),
    )
    return agent
