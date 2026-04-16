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
| Serveur inférence | `mlx-vlm.server` (port 8001) | API OpenAI-compatible localhost, parsers Gemma 4 natifs |
| Modèle | Gemma 4 26B A4B (local, 8-bit) | chargé via mlx-vlm |
| Orchestration agent | `pydantic-ai` + `OpenAIModel` | Provider standard|
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
Pydantic AI agent loop 
    │  POST http://127.0.0.1:8001/v1/chat/completions (stream=True)
    ▼
mlx-vlm.server (port 8001)
    │  --model <MODEL_PATH>         → Gemma 4 via mlx-vlm (texte + vision)
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


### Vision multimodale

Le serveur accepte des images au format OpenAI standard (`image_url` base64 ou URL).
Pydantic-ai peut passer des images via `ImageUrl`. Non exposé dans l'API Swift pour l'instant — Étape future.

---


**Points critiques :**

- `--port 8001` : FastAPI occupe le 8000
- `--temperature 1.0` : valeur d'entraînement Gemma 4, ne pas descendre sous 0.7



-
---

## 🔧 ToolRegistry

- Outils `created_by="system"` : **immuables**.
- `PermissionLevel` : `READ_ONLY < WRITE_SAFE < WRITE_MODIFY < SYSTEM_EXEC < NETWORK < AUTONOMOUS`
- Outils actifs : `web_search`, `fetch_webpage`.


---

## 🗺 Roadmap — état réel

### ✅ Étape 0 — Stack mlx-vlm direct (ARCHIVÉ)

### 🔧 Étape 1.5 — Migration mlx-vlm.server (EN COURS)
- Supprimer `core/llm.py`
- Réécrire `core/agent.py` : `create_iris_agent`
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
2. Lire `.claude/skills/building-pydantic-ai-agents/SKILL.md` en entier.
3. Lire les fichiers cibles avant de les modifier.



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