# 🤖 AGENTS.md — Iris v2 : Assistant Consulting Autonome
> Source de vérité architecturale pour Claude Code et tout contributeur.
> Mettre à jour ce fichier après chaque changement structurel significatif.

---

## 📌 Vision & Contraintes non négociables

**Iris** est un assistant cognitif personnel pour le consulting, 100 % local.

- **Zéro cloud** : aucune donnée ne quitte le Mac Mini M4 Pro (48 Go RAM).
- **Zéro télémétrie** : pas d'appels externes en dehors des tools explicites.
- **App fonctionnelle à chaque commit** : ne jamais laisser le projet dans un état cassé.
- **Minimal disruption** : corrections et ajouts chirurgicaux, sans refactoring opportuniste.

---

## 🛠 Stack technique — état actuel

| Couche | Technologie | Notes |
|--------|-------------|-------|
| Serveur inférence | `mlx-openai-server` (port 8001) | API OpenAI-compatible localhost, parsers Gemma 4 natifs |
| Modèle | Gemma 4 26B A4B (local, 4-bit) | `--model-type multimodal` — chargé via mlx-vlm |
| Orchestration agent | `pydantic-ai` + `OpenAIModel` | Provider standard + sous-classe légère pour reasoning |
| API BFF | FastAPI + uvicorn (port 8000) | SSE streaming vers Swift |
| Persistance conversations | JSON + JSONManager | `DATA_FOLDER` env ou `backend/storage/data/` |
| Persistance tools | ToolRegistry (JSON) | `TOOLS_FOLDER` env ou `backend/storage/tools/` |
| Frontend | Swift / SwiftUI (macOS + iOS) | Consomme SSE BFF port 8000, gère lifecycle backend |
| Mémoire vectorielle | LanceDB + nomic-embed-text-v1.5 | **Étape 2 — non implémenté** |

---

## 📁 Structure des fichiers backend

```
backend/
├── main.py                     # Lifespan : subprocess mlx-openai-server,
│                               #   crée l'agent pydantic-ai, routes debug
├── requirements.txt
├── .claude/
│   └── skills/
│       ├── mlx-gemma4-assistant/SKILL.md       # ← référence principale
│       ├── mlx-vlm-inference/SKILL.md
│       └── building-pydantic-ai-agents/SKILL.md
├── api/
│   ├── conversations.py        # CRUD conversations (inchangé)
│   └── messages.py             # /messages/stream SSE BFF
├── core/
│   ├── agent.py                # create_iris_agent(), ReasoningAwareOpenAIModel,
│   │                           #   IrisDeps, build_dynamic_system_prompt, IRIS_SYSTEM_PROMPT
│   ├── sse_transformer.py      # transform_agent_event(), parse_thinking_tags()
│   └── tools/
│       ├── registry.py
│       └── schema.py
├── tools/
│   ├── __init__.py
│   └── builtin/                # web_search, fetch_webpage
├── models/
│   └── conversation.py
├── storage/
│   ├── json_manager.py
│   └── tools/
└── tests/
    └── test_messages_api.py    # ⚠️  À réécrire
```

**Fichiers supprimés par la migration :**
- `core/llm.py` — IrisEngine (mlx-vlm direct)
- `core/agent.py` ancienne version — IrisModel custom, IrisStreamedResponse,
  state machine, `_messages_to_mlx`, `_parse_gemma_tool_call`, `_strip_gemma_thinking`

---

## 🔄 Architecture du pipeline de génération

### Flux complet d'une requête utilisateur

```
Swift client
    │  POST /api/conversations/{id}/messages/stream
    ▼
FastAPI BFF (port 8000) — api/messages.py : send_message_stream()
    │  Charge historique JSON → ModelMessage[]
    │  Construit system prompt via build_dynamic_system_prompt(IrisDeps)
    │  iris_agent.run_stream_events(user_msg, message_history=..., deps=deps)
    ▼
Pydantic AI agent loop (ReasoningAwareOpenAIModel)
    │  Sous-classe de OpenAIModel qui intercepte delta.reasoning_content
    │  et l'envoie dans une asyncio.Queue par requête
    │  POST http://127.0.0.1:8001/v1/chat/completions (stream=True)
    ▼
mlx-openai-server (port 8001)
    │  --model-type multimodal      → Gemma 4 via mlx-vlm (texte + vision)
    │  --reasoning-parser gemma4    → strip balises → delta.reasoning_content
    │  --tool-call-parser gemma4    → parse tool calls → delta.tool_calls
    │  --enable-auto-tool-choice
    ▼
Gemma 4 26B A4B — Metal, Apple Silicon
    ▼
SSE OpenAI-compatible :
    │  delta.reasoning_content → intercepté par ReasoningAwareOpenAIModel
    │                            → asyncio.Queue → reasoningDelta BFF
    │  delta.content           → TextPartDelta pydantic-ai → textDelta BFF
    │  delta.tool_calls        → FunctionToolCallEvent pydantic-ai
    ▼
api/messages.py consomme EN PARALLÈLE :
    │  run_stream_events()          → textDelta, toolCallStart, toolCallResult
    │  reasoning_queue              → reasoningDelta
    ▼
Swift client reçoit :
    {"type": "start"}
    {"type": "reasoningDelta", "content": "..."}   ← arrive avant textDelta
    {"type": "textDelta",      "content": "..."}
    {"type": "toolCallStart",  "toolCallId": "...", "toolName": "..."}
    {"type": "toolCallResult", "toolCallId": "...", "content": "..."}
    {"type": "error",          "content": "..."}
    {"type": "done"}
    data: [DONE]
```

### Propriétés du flux reasoning

- `delta.reasoning_content` arrive **en premier** dans le flux SSE, avant `delta.content`
- Reasoning et tool calls sont **mutuellement exclusifs** par tour : le modèle émet
  l'un ou l'autre, jamais les deux simultanément
- Pydantic-ai ignore `delta.reasoning_content` nativement — la sous-classe
  `ReasoningAwareOpenAIModel` intercepte ces tokens avant que le parent les écarte

### Vision multimodale

`--model-type multimodal` active le support image via mlx-vlm.
Le serveur accepte des images au format OpenAI standard (`image_url` base64 ou URL).
Pydantic-ai peut passer des images via `ImageUrl`. Non exposé dans l'API Swift pour l'instant — Étape future.

---

## ⚙️ Configuration mlx-openai-server

### Flags de démarrage

```bash
mlx-openai-server launch \
  --model-path $MODEL_PATH \
  --model-type multimodal \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --served-model-name gemma4 \
  --host 127.0.0.1 \
  --port 8001 \
  --context-length 32768 \
  --temperature 1.0 \
  --queue-timeout 300
```

**Points critiques :**
- `--model-type multimodal` : obligatoire (sinon chat template incorrect)
- `--port 8001` : FastAPI occupe le 8000
- `--context-length 32768` : évite Metal paging (~16 Go GPU à 4-bit)
- `--temperature 1.0` : valeur d'entraînement Gemma 4, ne pas descendre sous 0.7
- `--served-model-name gemma4` doit correspondre à `OpenAIModel("gemma4", …)`

### Subprocess dans le lifespan FastAPI

Voir `.claude/skills/mlx-gemma4-assistant/SKILL.md` §2 pour le code exact.
- Poll `/health` sur `127.0.0.1:8001` — timeout 180s
- Ne **pas** importer `MLXVLMHandler` dans le process FastAPI → GPU semaphore leak
- `proc.terminate()` + `proc.wait(timeout=10)` dans le `finally`

---

## 🤖 ReasoningAwareOpenAIModel

Sous-classe de `OpenAIModel` (pydantic-ai) qui intercepte `delta.reasoning_content`
tout en laissant le tool loop et la gestion de contexte intacts.

```python
# Schéma conceptuel — implémentation dans core/agent.py
class ReasoningAwareOpenAIModel(OpenAIModel):
    """
    Sous-classe minimale d'OpenAIModel qui intercepte delta.reasoning_content
    et l'envoie dans une asyncio.Queue par requête, sans affecter le tool loop.
    """
    # La Queue est injectée par requête (thread-safe par conversation)
    # Override du point d'extension approprié dans pydantic-ai
    # (à déterminer en lisant le source pydantic-ai)
```

L'implémentation exacte dépend des internals de pydantic-ai — Claude Code doit
lire le source pour identifier le bon point d'extension (`request_stream()` ou
`_get_event_iterator()` de `OpenAIStreamedResponse`).

### Wiring dans create_iris_agent()

```python
from openai import AsyncOpenAI
from pydantic_ai import Agent
from core.agent import ReasoningAwareOpenAIModel

_local_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="not-needed",
)

local_model = ReasoningAwareOpenAIModel("gemma4", openai_client=_local_client)

agent = Agent(
    model=local_model,
    system_prompt=IRIS_SYSTEM_PROMPT,
)
```

---

## 🔧 ToolRegistry

- Outils `created_by="system"` : **immuables**.
- `PermissionLevel` : `READ_ONLY < WRITE_SAFE < WRITE_MODIFY < SYSTEM_EXEC < NETWORK < AUTONOMOUS`
- Outils actifs : `web_search`, `fetch_webpage`.

---

## 🧪 État des tests

`tests/test_messages_api.py` : **obsolète, à réécrire après la migration.**

Les nouveaux tests devront mocker `ReasoningAwareOpenAIModel` et le subprocess,
et couvrir : streaming SSE valide (textDelta + reasoningDelta), persistence après
`[DONE]`, 404/422/503.

---

## 🗺 Roadmap — état réel

### ✅ Étape 0 — Stack mlx-vlm direct (ARCHIVÉ)

### 🔧 Étape 1.5 — Migration mlx-openai-server (EN COURS)
- Supprimer `core/llm.py`
- Réécrire `core/agent.py` : `ReasoningAwareOpenAIModel`, `create_iris_agent`
- Mettre à jour `main.py` : lifespan subprocess port 8001
- Simplifier `core/sse_transformer.py`
- Mettre à jour `requirements.txt`
- Réécrire `tests/test_messages_api.py`

### ⬜ Étape 1.7 — Vision
Exposer upload d'images dans l'API Swift → `ImageUrl` vers pydantic-ai.

### ⬜ Étape 2 — Mémoire vectorielle
LanceDB + nomic-embed-text-v1.5, context enrichment dans IrisDeps.

### ⬜ Étape 3 — Sandbox et skills dynamiques
Wasmtime / WASI, `code_interpreter`, chargement dynamique.

---

## 📝 Instructions pour Claude Code

### Avant toute modification
1. Lire `AGENTS.md` en entier.
2. Lire `.claude/skills/mlx-gemma4-assistant/SKILL.md` en entier.
3. Lire `.claude/skills/building-pydantic-ai-agents/SKILL.md` en entier.
4. Lire les fichiers cibles avant de les modifier.

### Skills de référence

| Skill | Lire avant de modifier |
|-------|------------------------|
| `mlx-gemma4-assistant/SKILL.md` | `main.py`, `core/agent.py`, tout ce qui touche au serveur |
| `building-pydantic-ai-agents/SKILL.md` | `core/agent.py`, `core/sse_transformer.py`, tools |

### Règles de travail
- Une étape à la fois — valider avant de passer à la suivante.
- Pas de refactoring opportuniste.
- Conserver tous les commentaires explicatifs.
- Si un choix non trivial se présente : s'arrêter et demander.

### Variables d'environnement (voir `.env.example`)
```
MODEL_PATH       # chemin absolu vers le dossier modèle Gemma 4 local
DATA_FOLDER      # dossier conversations JSON
TOOLS_FOLDER     # dossier registry tools
MLX_SERVER_PORT  # port mlx-openai-server (défaut : 8001)
API_PORT         # port FastAPI BFF (défaut : 8000)
```