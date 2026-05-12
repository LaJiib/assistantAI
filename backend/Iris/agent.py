from dotenv import load_dotenv
import os
import pytz
from datetime import datetime

load_dotenv()


from deepagents import AsyncSubAgent, AsyncSubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, AgentMiddleware
from langchain_openai import ChatOpenAI
from utils.tools import web_search, fetch_webpage
from langchain.agents.middleware import dynamic_prompt
from langchain_core.messages import AIMessage

async_subagents = [
    AsyncSubAgent(
        name="deep_research",
        description=(
            "Conducts in-depth web research on any topic. "
            "Searches multiple sources, reads full pages, and synthesizes "
            "a comprehensive report. Use when the user asks for research, "
            "analysis, or detailed information requiring multiple sources. "
            "Pass a clear and detailed research brief as input."
        ),
        graph_id="deep_research",
        # Pas d'url → ASGI transport, co-déployé
    ),
]

# ---------------------------------------------------------------------------
# Modèle — pointe sur omlx via les variables d'environnement
# ---------------------------------------------------------------------------
# thinking activé : sans ça, omlx injecte un bloc <|channel>thought\n<channel|>
# (réflexion vide) après le tool_response, ce qui pousse Gemma 4 à rappeler
# le tool au lieu de répondre. Avec thinking activé, ce bloc n'est pas injecté
# et le modèle raisonne librement sur le résultat avant de formuler sa réponse.

additional_kwargs = {"chat_template_kwargs": {"enable_thinking": True}, "thinking_tokens": 16384, "max_tokens": 32768}

model = ChatOpenAI(
    model=os.environ["OMLX_MODEL"],
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "local"),
    model_kwargs= {
        "extra_body":additional_kwargs
    },
)

class GemmaContentCleaner(AgentMiddleware):
    def after_model(self, state):
        """
        Intercepte la réponse après le modèle pour nettoyer les résidus de Gemma 4.
        """
        # On récupère le dernier message
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_message = messages[-1]
        
        # On ne traite que les messages de l'assistant (AIMessage)
        if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
            content = last_message.content
            
            # Règle 1 : Si le premier caractère est un saut de ligne (\n)
            if content.startswith("\n"):
                # On supprime le caractère de saut de ligne.
                # Note : En Python, \n est un seul caractère. 
                # Si tu veux supprimer strictement "deux caractères", utilise content[2:]
                content = content[1:]
            
            # Règle 2 : Si après ça le message est vide (ou ne contient que des espaces)
            if not content.strip():
                # On supprime carrément le contenu (chaîne vide)
                content = ""
            
            # On renvoie le dictionnaire de mise à jour de l'état
            # Le reducer "messages" de LangGraph s'occupera d'appliquer la modification
            return {"messages": [last_message.copy(update={"content": content})]}
            
        return {}

@dynamic_prompt
def core_dynamic_prompt(request):
    """
    Generates the dynamic situational context for the agent.
    Injected at the start of every model call.
    """
    # 1. Gestion du fuseau horaire (CEST)
    tz = pytz.timezone('Europe/Paris')
    now = datetime.now(tz)
    
    # 2. Construction du bloc de contexte temporel
    temporal_context = (
        f"CURRENT SITUATIONAL CONTEXT:\n"
        f"- Date: {now.strftime('%A, %B %d, %Y')}\n"
        f"- Time: {now.strftime('%H:%M:%S')}\n"
        f"- Timezone: CEST (UTC+2)\n"
    )

    system_instructions = ("You are Iris a helpful assistant. Always answer in French.")
    return f"{temporal_context}\n\n{system_instructions}"

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
tools = [web_search, fetch_webpage]
graph = create_agent(model,
                    tools,
                    middleware=[TodoListMiddleware(),
                                AsyncSubAgentMiddleware(async_subagents=async_subagents),
                                core_dynamic_prompt,
                                GemmaContentCleaner(),
                                ],)