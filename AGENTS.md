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
| Inférence | `mlx-vlm >= 0.4.3` | `stream_generate()` + `apply_chat_template()` via processor |
| Modèle | Gemma 4 26B A4B (Q8_0, ~28 Go) | MoE, 3.8B params actifs, 256K contexte |
| Orchestration agent | `pydantic-ai >= 0.0.14` | IrisModel custom wrappant IrisEngine |
| API | FastAPI + uvicorn | SSE streaming, préfixe `/api/conversations/` |
| Persistance conversations | JSON + JSONManager | `DATA_FOLDER` env ou `backend/storage/data/` |
| Persistance tools | ToolRegistry (JSON) | `TOOLS_FOLDER` env ou `backend/storage/tools/` |
| Frontend | Swift / SwiftUI (macOS + iOS) | Consomme SSE, gère lifecycle backend via subprocess |
| Mémoire vectorielle | LanceDB + nomic-embed-text-v1.5 | **Étape 2 — non implémenté** |

---

## 📁 Structure des fichiers backend

```
backend/
├── main.py                     # Lifespan FastAPI, startup modèle, routes debug
├── requirements.txt
├── .claude/
│   └── skills/
│       ├── mlx-vlm-inference.md            # API mlx-vlm, stream_generate, apply_chat_template
│       └── building-pydantic-ai-agents.md  # IrisModel custom, StreamedResponse, event types
├── api/
│   ├── conversations.py        # CRUD conversations
│   └── messages.py             # /messages/stream (SSE BFF) ← point d'entrée principal
│                               # ⚠️  POST /messages/ (non-stream) supprimé
├── core/
│   ├── agent.py                # IrisModel, IrisStreamedResponse, _messages_to_mlx,
│   │                           #   _tool_defs_to_openai, create_iris_agent, IrisDeps
│   ├── llm.py                  # IrisEngine : load, stream_messages, _generate_raw, _stream_raw
│   ├── sse_transformer.py      # GemmaThinkingParser, transform_agent_event, parse_thinking_tags
│   └── tools/
│       ├── registry.py         # ToolRegistry thread-safe, persistance JSON
│       └── schema.py           # ToolSchema, PermissionLevel
├── tools/
│   ├── __init__.py             # register_builtin_tools()
│   └── builtin/                # web_search, fetch_webpage (outils système, immuables)
├── models/
│   └── conversation.py         # ConversationMetadata, MessageMeta
├── storage/
│   ├── json_manager.py         # Lecture/écriture conversations JSON
│   └── tools/                  # Fallback storage registry
└── tests/
    └── test_messages_api.py    # ⚠️  OBSOLÈTE — à réécrire (voir section Tests)
```

---

## 🔄 Architecture du pipeline de génération

### Flux complet d'une requête utilisateur

```
Swift client
    │  POST /api/conversations/{id}/messages/stream
    ▼
api/messages.py : send_message_stream()
    │  Charge historique JSON → ModelMessage[]
    │  Construit system prompt via build_dynamic_system_prompt(IrisDeps)
    │  iris_agent.run_stream_events(user_msg, message_history=..., deps=deps)
    ▼
Pydantic AI agent loop
    │  Appelle IrisModel.request_stream()
    ▼
IrisModel.request_stream()                    [core/agent.py]
    │  _messages_to_mlx()     → list[dict] pour apply_chat_template
    │  _tool_defs_to_openai() → format OpenAI injecté dans le template
    │  yield IrisStreamedResponse(...)
    ▼
IrisStreamedResponse._get_event_iterator()   [core/agent.py]
    │  Lance IrisEngine.stream_messages() dans un thread → asyncio.Queue
    │  State machine sur chunks bruts :
    │    SCAN      → cherche le prochain marqueur spécial
    │    THINKING  → <|channel>thought…<channel|> : ignoré silencieusement
    │    TOOL_CALL → <|tool_call>…<tool_call|> : bufférisé complet
    │                → handle_tool_call_part() une fois fermé
    │    TEXT      → handle_text_delta() chunk par chunk
    ▼
Pydantic AI reçoit les events natifs
    │  FunctionToolCallEvent  → exécute le tool via ToolRegistry
    │  FunctionToolResultEvent → reconstruit contexte via _messages_to_mlx()
    │  PartDeltaEvent(TextPartDelta) → transmis à transform_agent_event()
    ▼
core/sse_transformer.py : transform_agent_event()
    │  GemmaThinkingParser.feed() — filet de sécurité si balise non filtrée en amont
    │  → SSE BFF events JSON
    ▼
Swift client reçoit :
    {"type": "start"}
    {"type": "reasoningDelta", "content": "..."}
    {"type": "textDelta",      "content": "..."}
    {"type": "toolCallStart",  "toolCallId": "...", "toolName": "..."}
    {"type": "toolCallResult", "toolCallId": "...", "content": "..."}
    {"type": "error",          "content": "..."}
    {"type": "done"}
    data: [DONE]
```

### Notes importantes

- **Tool calls non streamés** : accumulés en mémoire jusqu'à `<tool_call|>`, émis en un seul event. L'affichage reprend à la réponse finale.
- **Stop sequence** `<|tool_response>` : native Gemma 4, arrête `stream_generate()` proprement après chaque tool call, permettant à Pydantic AI de reprendre la main.
- **GemmaThinkingParser** dans `sse_transformer.py` : redondant dans le flux normal (la state machine a déjà strippé les thoughts), mais conservé comme filet de sécurité et pour `parse_thinking_tags()` sur les messages historiques.

---

## 🧠 Règles critiques Gemma 4

Ces règles sont non négociables. Toute déviation produit des comportements erratiques.

### 1. Thinking mode obligatoire avec tools

`enable_thinking=True` est **requis** dès que des tools sont présents.
Sans `<|think|>` en début de system prompt, Gemma 4 génère du JSON ReAct
(`{"action": ..., "action_input": ...}`) au lieu du format natif `<|tool_call>`.

Géré automatiquement par `apply_chat_template(..., enable_thinking=True)`.

### 2. Arguments de tool call : dict, jamais string sérialisée

```python
# ❌ Incorrect — produit {{...}} dans le prompt reconstruit
"arguments": json.dumps(args)

# ✅ Correct — le template Jinja formate lui-même
"arguments": args   # dict Python
```

Le template Jinja encadre les arguments de ses propres `{ }`.
Une string JSON produit `{{"key": "value"}}` — corruption du contexte garantie.

### 3. Tool responses dans le même message assistant

```python
# ✅ Format canonique Gemma 4
{
    "role": "assistant",
    "tool_calls": [
        {"function": {"name": "web_search", "arguments": {"query": "..."}}}
    ],
    "tool_responses": [
        {"name": "web_search", "response": "...résultat..."}
    ],
    "content": ""   # vide jusqu'à la réponse finale
}

# ❌ Incorrect — produit <|turn>tool (rôle non documenté)
{"role": "tool", "content": "...", "tool_call_id": "..."}   # message séparé
```

### 4. Gestion des thoughts en multi-turn

- **Entre deux tours utilisateur** : thoughts strippés automatiquement par la macro `strip_thinking` du chat template.
- **Au sein d'une séquence de tool calls** (même tour modèle) : thoughts **conservés** entre les itérations — ne pas stripper entre deux tool calls consécutifs.

### 5. Paramètres de sampling

```python
temperature = 1.0    # valeur d'entraînement — ne pas baisser
top_p       = 0.95
top_k       = 64
```

`temperature=0.3` (valeur actuelle dans le code) dégrade les capacités de raisonnement.
**À corriger dans `api/messages.py` et partout où `ModelSettings` est construit.**

---

## 🐛 Bugs actifs — priorité d'intervention

### 🔴 Bug 1 — arguments sérialisés en string
**Fichiers** : `core/agent.py` (`_messages_to_mlx`), `core/llm.py` (`_build_assistant_message`)
**Symptôme** : `{{query: "..."}}` dans le prompt reconstruit → contexte corrompu
**Correction** : passer `args` directement (dict) sans `json.dumps()`.
`ToolCallPart.args` peut être `ArgsDict` ou `str` selon la version Pydantic AI — parser si string.

### 🔴 Bug 2 — tool response dans un tour séparé
**Fichier** : `core/agent.py` (`_messages_to_mlx`)
**Symptôme** : `<|turn>tool` dans le prompt — rôle non documenté Gemma 4
**Correction** : détecter quand un `ModelRequest` suivant un `ModelResponse` avec tool calls
contient uniquement des `ToolReturnPart`, et fusionner dans `tool_responses` du message assistant.

### 🔴 Bug 3 — enable_thinking False avec tools
**Fichier** : `core/agent.py` (`request()`, `request_stream()`)
**Symptôme** : format ReAct JSON au lieu de `<|tool_call>` natif
**Correction** : activer `enable_thinking=True` automatiquement si `tools_openai` non vide.

### 🟡 Dette — paramètres sampling incorrects
**Fichier** : `api/messages.py`, `main.py`
**Correction** : `temperature=1.0, top_p=0.95, top_k=64`

---

## 🧪 État des tests

### Situation actuelle

`tests/test_messages_api.py` : **17/20 en échec — obsolète, à réécrire après les bugs 1-3.**

Causes des échecs :
- **405** sur `POST /messages/` : endpoint non-stream supprimé, seul `/messages/stream` existe.
- **503** sur les tests stream : mock `make_app()` n'injecte pas `iris_agent` dans `app.state`.
- Schémas de réponse (`userMessage`, `assistantMessage`) ne correspondent plus au protocole SSE BFF.

3 tests passent encore : ceux qui testent les conversations ou le 503 engine.

### Ce que les nouveaux tests devront couvrir

- `GET /messages/` : liste vide, system exclu
- `POST /messages/stream` : SSE valide (`start` → `textDelta` → `done` → `[DONE]`)
- `POST /messages/stream` : persistence après `[DONE]`, messageCount +2
- `POST /messages/stream` conv inexistante → 404
- `POST /messages/stream` content vide → 422
- `POST /messages/stream` agent non initialisé → 503
- Unitaire `_messages_to_mlx` : tool_calls + tool_responses fusionnés dans le même message
- Unitaire `_messages_to_mlx` : arguments restent des dicts

**Le mock devra injecter `engine` ET `iris_agent` dans `app.state`.**

---

## 🔧 ToolRegistry

- Outils `created_by="system"` : **immuables**, ni écrasement ni suppression.
- `PermissionLevel` : `READ_ONLY < WRITE_SAFE < WRITE_MODIFY < SYSTEM_EXEC < NETWORK < AUTONOMOUS`
- Tout tool call loggé avec ses arguments pour audit humain.
- **Note** : registry implémenté mais seuls `web_search` et `fetch_webpage` sont enregistrés.

---

## 🗺 Roadmap — état réel

### ✅ Étape 1 — Core inférence (TERMINÉ)
- `core/llm.py` : IrisEngine, mlx-vlm, streaming
- `core/agent.py` : IrisModel (Pydantic AI), IrisStreamedResponse, state machine
- `core/sse_transformer.py` : GemmaThinkingParser, transform_agent_event
- `api/messages.py` : endpoint SSE complet avec persistance
- `tools/builtin/` : web_search, fetch_webpage
- `core/tools/registry.py` : ToolRegistry thread-safe

### 🔧 Étape 1.5 — Corrections bugs tool calling (EN COURS)
Bugs 1, 2, 3 dans `core/agent.py` et `core/llm.py`.
Suivi : réécrire `tests/test_messages_api.py` après correction.

### ⬜ Étape 2 — Mémoire vectorielle
- `storage/vector_store.py` : LanceDB, nomic-embed-text-v1.5 sur MPS
- Tables : `table_user_core`, `table_pro_knowledge`, `tables_clients_{id}`
- Context enrichment dans `IrisDeps` avant chaque tour
- Worker de consolidation post-session

### ⬜ Étape 3 — Sandbox et skills dynamiques
- `core/sandbox.py` : Wasmtime / WASI, workspace `~/Iris_Workspace/`
- Outil `code_interpreter` (Excel, CSV)
- Chargement dynamique de skills générés par l'IA

---

## 📝 Instructions pour Claude Code

### Avant toute modification
1. Lire ce fichier (`AGENTS.md`) en entier.
2. Lire les skills pertinents (tableau ci-dessous).
3. Lire le fichier cible dans son intégralité.
4. Vérifier l'état des tests : `cd backend && python -m pytest tests/ -v`

### Skills de référence

Disponibles dans `backend/.claude/skills/` :

| Skill | Lire avant de modifier |
|-------|------------------------|
| `mlx-vlm-inference.md` | `core/llm.py`, tout ce qui touche `stream_generate`, `apply_chat_template`, `processor` |
| `building-pydantic-ai-agents.md` | `core/agent.py`, `IrisModel`, `IrisStreamedResponse`, `_parts_manager`, event types Pydantic AI |

### Règles de travail
- **Une correction à la fois** — noter l'état des tests avant/après.
- **Pas de refactoring opportuniste** — corriger uniquement ce qui est demandé.
- **Pas de nouveaux fichiers** sans demande explicite.
- **Conserver tous les commentaires** explicatifs existants.
- Si une correction implique un choix d'architecture non trivial : **s'arrêter et demander**.

### Commandes utiles
```bash
cd backend
python -m pytest tests/ -v
python -m pytest tests/test_messages_api.py -v -k "stream"
uvicorn main:app --reload --port 8000
```

### Variables d'environnement (voir `.env.example`)
```
DATA_FOLDER    # dossier conversations JSON
TOOLS_FOLDER   # dossier registry tools
MODEL_PATH     # chemin modèle mlx-vlm local
```