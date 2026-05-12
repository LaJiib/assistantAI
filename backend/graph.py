"""
graph.py — Étape 0 : graph LangGraph minimal connecté à omlx.

Utilise l'API Anthropic locale exposée par omlx (http://127.0.0.1:8000).
ANTHROPIC_BASE_URL et ANTHROPIC_API_KEY sont lus depuis backend/.env et
transmis automatiquement au SDK anthropic sous-jacent.
"""
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# Modèle — pointe sur omlx via les variables d'environnement
# ---------------------------------------------------------------------------
# thinking activé : sans ça, omlx injecte un bloc <|channel>thought\n<channel|>
# (réflexion vide) après le tool_response, ce qui pousse Gemma 4 à rappeler
# le tool au lieu de répondre. Avec thinking activé, ce bloc n'est pas injecté
# et le modèle raisonne librement sur le résultat avant de formuler sa réponse.
model = ChatAnthropic(
    model=os.environ["OMLX_MODEL"],
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 1024},
)



# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
tools = [get_weather]
graph = create_agent(model, tools)
