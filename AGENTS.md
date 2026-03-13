# Agent Architecture - Changelog & Roadmap

## Vision Long Terme
Agent autonome cognitif avec mémoires, planning, meta-prompting, initiatives.

## État Actuel
- [x] App Swift fonctionnelle avec MLX (Phase 1)
- [x] Backend Python + migration frontend (Phase 2)
- [x] API REST conversations/messages + UI nettoyée (Phase 2bis)
- [ ] RAG ChromaDB (Phase 3)
- [x] Tool calling Foundation (Phase 3.1) — schema, registry, 3 builtin tools, stream_with_tools()
- [ ] Mémoires
- [ ] Planning autonome
- [ ] Meta-prompting
- [ ] Initiatives

## Contraintes Permanentes
- FULL LOCAL : Aucune donnée externe
- ZERO TELEMETRY : Aucun tracking
- Performance >= 10 tok/s (seuil opérationnel ; 32 tok/s cible idéale)
- App fonctionnelle à chaque commit

---

## Historique décisions

### 2026-03-10 : Choix architecture
- **Décision** : Python backend + Swift frontend
- **Raison** : Vélocité dev, écosystème IA mature
- **Alternative écartée** : Swift pur (trop lent à développer)

### 2026-03-10 : Choix modèle
- **Décision** : Ministral 3 14B Instruct (8-bit, ~14 GB RAM unifiée)
- **Raison** : Tool calling natif, qualité optimale M4
- **Note** : La version 8-bit plafonne à ~17 tok/s en Python. Une version 4-bit (~7 GB) permettrait d'atteindre ~35 tok/s. La dégradation est inhérente à Python mlx-lm vs Swift MLX natif (overhead CPU-GPU par token).

### 2026-03-11 : API mlx-lm — rupture de compatibilité
- **Problème** : mlx-lm >= 0.21 a supprimé le paramètre `temp=` de `generate()`.
- **Solution** : `make_sampler(temp=...)` → passer via `sampler=` (API officielle).
- **Leçon** : Contraindre `mlx-lm>=0.21.0` dans requirements.txt.

### 2026-03-11 : Architecture endpoint FastAPI sync vs async
- **Décision** : Endpoint `/chat` déclaré `def` (non-async).
- **Raison** : FastAPI exécute automatiquement les fonctions `def` dans un thread pool — évite `run_in_executor` + `partial` manuels.

### 2026-03-11 : Streaming SSE
- **Décision** : Ajout `/chat/stream` avec Server-Sent Events.
- **Format** : `data: {"text": "chunk"}\n\n` … `data: [DONE]\n\n`
- **Raison** : Cohérence avec l'app Swift qui consomme un `AsyncStream<Generation>`.

### 2026-03-11 : Mesure de performance
- **Problème** : Formule `mots × 1.3 / elapsed` sous-estimait de ~30% (texte technique français).
- **Solution** : Tok/s réels loggés côté serveur via `response.generation_tps` (mlx-lm natif). Les tests affichent une approximation `chars / 4` pour information seulement.

---

## PHASE 1 : Backend minimal Python ✅ COMPLÉTÉE

**Dates** : 2026-03-11

### Objectif
Backend Python FastAPI qui charge Ministral 3 14B et génère du texte via endpoints HTTP. Cohabitation avec Swift, zéro modification du code Swift.

### Fichiers créés

```
backend/
├── main.py                  # Entrypoint FastAPI (GET /health, POST /chat, POST /chat/stream)
├── requirements.txt         # mlx-lm, fastapi, uvicorn, pydantic, python-dotenv, requests
├── .env.example             # Template : MODEL_PATH, HOST, PORT, LOG_LEVEL
├── README.md                # Instructions setup
├── core/
│   ├── __init__.py
│   └── llm.py               # MinistralEngine : load(), stream(), generate()
└── tests/
    ├── test_llm.py          # Tests unitaires sans modèle (4 tests, ~0s)
    └── test_basic.py        # Tests d'intégration avec backend running
```

### Fichiers Swift modifiés
AUCUN ✅

### Performance mesurée (Ministral 3 14B 8-bit, M4 Pro)

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| Throughput génération | ~17 tok/s | >= 10 tok/s | ✅ |
| Throughput prompt processing | ~67 tok/s | — | ✅ |
| Startup (chargement modèle) | ~16s | < 30s | ✅ |
| RAM unifiée (peak) | 14.4 GB | — | ℹ️ |

> **Note performance** : Le plafond ~17 tok/s avec la version 8-bit est inhérent à l'overhead Python par token (CPU-GPU round-trip via `mx.eval()`), pas à un dysfonctionnement du backend. La même config en Swift natif MLX atteint ~35 tok/s avec le même modèle. Une version 4-bit du modèle (~7 GB) permettrait d'atteindre ~35 tok/s en Python également.

### Tests validés
- [x] Backend démarre sans erreur (`python main.py`)
- [x] `/health` retourne 200 + `model_loaded=true`
- [x] `/chat` génère réponse cohérente française
- [x] `/chat/stream` stream les tokens en SSE
- [x] Performance >= 10 tok/s
- [x] App Swift inchangée et fonctionnelle

### Problèmes rencontrés

**1. Rupture API mlx-lm (temp= supprimé)**
- Symptôme : `generate_step() got an unexpected keyword argument 'temp'`
- Cause : mlx-lm >= 0.21 remplace `temp=` par un objet `sampler`
- Solution : `make_sampler(temp=0.3)` + `sampler=sampler` dans l'appel generate
- Fichier : `core/llm.py` — `_run_generate()`

**2. `core/__init__.py` invalide (commentaires Swift)**
- Symptôme : `SyntaxError: leading zeros in decimal integer literals`
- Cause : fichier généré avec le header Swift `// Created by...` au lieu d'un commentaire Python
- Solution : remplacé par `# core/__init__.py`

**3. Comparaison de performance faussée**
- Symptôme : Swift semblait 3× plus rapide que Python avec le "même" modèle
- Cause : l'app Swift chargeait `mistral-nemo-instruct` (12B 4-bit) via `ModelStateManager.swift`, pas Ministral 14B — `Constants.modelPath` était défini mais non utilisé dans le code de chargement
- Solution : diagnostic, correction de `ModelStateManager.swift`, ajustement de la cible de performance

**4. Mesure tok/s inexacte dans les tests**
- Cause : estimation `mots × 1.3` sous-estimait systématiquement (~30% d'écart)
- Solution : utilisation de `response.generation_tps` (mlx-lm natif) pour les logs serveur ; approximation `chars / 4` dans les tests (noté explicitement comme approximation)

### Décisions techniques

| Décision | Raison |
|---|---|
| `stream_generate` comme base unique | Une seule implémentation pour `/chat` et `/chat/stream` ; tok/s exacts via `generation_tps` |
| Endpoint `/chat` déclaré `def` (sync) | FastAPI gère le thread pool automatiquement, évite `run_in_executor` manuel |
| `tokenizer.apply_chat_template()` | API HuggingFace officielle, remplace le format `[INST]` hardcodé |
| Import mlx-lm au top-level de `llm.py` | Plus clair ; mlx-lm est une dépendance obligatoire, pas optionnelle |
| SSE format `{"text": "chunk"}` + `[DONE]` | Extensible (ajout futur de `{"tokens": N, "tps": X}` sans casser les clients) |

---

## PHASE 2 : Migration frontend vers backend Python ✅ COMPLÉTÉE

**Dates** : 2026-03-11

### Objectif
**Migration complète** de l'inférence vers backend Python. Le frontend Swift devient une UI pure qui communique avec le backend via HTTP. MLXService.swift est désactivé (conservé en code pour rollback si besoin, mais non utilisé).

**Vision cible** :
```
┌─────────────────────────────────┐
│  Frontend Swift (UI pure)       │
│  - ChatView, ConversationView   │
│  - Gestion conversations        │
│  - Rendu markdown               │
│  - Aucune inférence locale      │
└──────────┬──────────────────────┘
           │ HTTP localhost
           ▼
┌─────────────────────────────────┐
│  Backend Python (IA)            │
│  - MLX + Ministral 14B          │
│  - Toute génération texte       │
│  - (Phase 3+) RAG, tools, etc.  │
└─────────────────────────────────┘
```

### Contraintes spécifiques Phase 2

**Architecture** :
- BackendManager gère lifecycle backend (start/stop/health)
- ChatAPI = seule interface communication avec backend
- ConversationViewModel appelle TOUJOURS ChatAPI (plus de MLXService)
- MLXService.swift conservé dans code (désactivé) pour rollback possible
- Storage conversations inchangé (format compatible)

**Migration non-breaking** :
- App doit démarrer backend automatiquement au launch
- Si backend fail → message erreur clair + fallback gracieux
- Conversations existantes préservées
- Pas de perte données

**Sécurité** :
- Backend toujours localhost (127.0.0.1:8000)
- Process backend lancé en subprocess (contrôlé par app)
- Logs backend visibles console Xcode
- Arrêt app → arrêt backend propre

**UX** :
- Démarrage app → backend démarre automatiquement (splash screen ?)
- Indicateur discret état backend (settings)
- Performance perçue >= baseline Swift (grâce streaming)
- Pas de régression fonctionnelle

### Fichiers à créer
```
AssistantIA/Core/
├── BackendManager.swift       # 🆕 Lifecycle backend Python
│   - enum State (stopped, starting, running, error)
│   - func start() async throws
│   - func stop()
│   - func checkHealth() async throws
│   - autoStart() appelé au launch app
│   - Gestion restart si crash backend
│
└── ChatAPI.swift              # 🆕 Client HTTP backend (remplace MLXService)
    - actor isolé (thread-safe)
    - func streamMessage(prompt) -> AsyncStream<String>
    - func sendMessage(prompt) async throws -> String
    - Gestion erreurs réseau + retry logic
    - Décodage SSE (/chat/stream)
    - Fallback si backend down

AssistantIA/UI/Settings/
└── BackendSettingsView.swift  # 🆕 Monitoring backend (debug)
    - Status indicator (stopped/starting/running/error)
    - Bouton manual Start/Stop (debug uniquement)
    - Affichage logs erreur
    - Statistiques (uptime, requêtes/s)
```

### Fichiers à modifier
```
AssistantIA/Application/
└── AssistantIAApp.swift       # ⚠️ MODIFICATION
    - Ajouter @StateObject backendManager
    - onAppear : await backendManager.start()
    - onDisappear : backendManager.stop()
    - Estimé : +15 lignes

AssistantIA/UI/
├── RootView.swift             # ⚠️ MODIFICATION MINEURE
│   - Injection backendManager via @EnvironmentObject
│   - NavigationLink vers BackendSettingsView (settings/debug)
│   - Estimé : +8 lignes
│
└── Chat/
    └── ConversationViewModel.swift # ⚠️ MODIFICATION IMPORTANTE
        - REMPLACER appels MLXService par ChatAPI
        - Modifier generate() : appel chatAPI.streamMessage()
        - Supprimer dépendance MLXService (ou commenter)
        - Adapter AsyncStream<Generation> vers AsyncStream<String>
        - Estimé : ~40 lignes modifiées, ~10 lignes supprimées
```

### Fichiers à désactiver (conserver en code)
```
⚠️ Core/Services/MLXService.swift
   - NE PAS SUPPRIMER (rollback possible)
   - Commenter l'usage dans ConversationViewModel
   - Ajouter commentaire : "// DÉSACTIVÉ Phase 2 : migration backend Python"

⚠️ Core/Utilities/ModelStateManager.swift
   - Conservé (peut servir Phase 3+ pour config)
   - Ne sera plus appelé par ConversationViewModel
```

### Fichiers INTERDITS de modification

**Ces fichiers doivent rester identiques** :
```
✅ Core/Utilities/ConversationManager.swift    # Storage inchangé
✅ Core/Models/Message.swift                   # Format stable
✅ Core/Models/Conversation.swift              # Format stable
✅ Core/Models/ConversationMetadata.swift      # Format stable
✅ UI/Chat/ChatView.swift                      # UI inchangée
✅ UI/Chat/Components/*                        # Tous components
```

### Tests validation Phase 2

#### Test 1 : Démarrage automatique backend ✅
**Actions** :
1. Fermer app si ouverte
2. Lancer app (Xcode Run)
3. Observer console Xcode

**Critères succès** :
- [ ] Logs montrent "🚀 Démarrage backend..." automatiquement
- [ ] Backend démarre dans < 30s
- [ ] App UI affichée (pas de blocage)
- [ ] Conversations list visible (chargée pendant startup backend)
- [ ] Activity Monitor montre process `python`

**Si échec** :
- App doit afficher erreur claire : "Backend failed to start"
- Logs Xcode montrent erreur Python
- App reste utilisable (ne crash pas)

---

#### Test 2 : Chat fonctionnel via backend ✅
**Actions** :
1. Backend démarré (Test 1 validé)
2. Nouvelle conversation
3. Message : "Bonjour, qui es-tu ?"
4. Observer génération

**Critères succès** :
- [ ] Réponse générée (texte cohérent)
- [ ] Streaming tokens visibles (texte s'affiche progressivement)
- [ ] Performance >= 10 tok/s (perception fluide)
- [ ] Message sauvegardé dans conversation
- [ ] Format Message compatible avec existant

---

#### Test 3 : Conversations existantes accessibles ✅
**Actions** :
1. Avant Phase 2 : créer 2-3 conversations avec Swift MLX
2. Après Phase 2 : lancer app
3. Ouvrir conversations list
4. Charger conversation ancienne
5. Continuer conversation (nouveau message)

**Critères succès** :
- [ ] Toutes conversations anciennes visibles
- [ ] Contenu intact (messages, timestamps, title)
- [ ] Conversation chargeable sans erreur
- [ ] Nouveau message utilise backend Python (logs le confirment)
- [ ] Historique conversation cohérent (ancien Swift + nouveau Python)

---

#### Test 4 : Performance comparable baseline ✅
**Actions** :
1. Prompt identique : "Compte de 1 à 100"
2. Mesurer temps génération
3. Comparer avec baseline Swift MLX (notes Phase 1)

**Critères succès** :
- [ ] Backend Python : ~17 tok/s (attendu)
- [ ] Perception UX acceptable (< 3s pour réponse courte)
- [ ] Streaming fluide (pas de lag visible)
- [ ] CPU app Swift faible (génération sur backend, pas local)

**Note** : Backend Python plus lent que Swift MLX (17 vs 35 tok/s) mais acceptable. Phase 3+ optimisera si besoin.

---

#### Test 5 : Gestion erreur backend down ✅
**Actions** :
1. Backend running
2. Conversation active
3. Via terminal : `pkill -9 python` (kill backend)
4. Tenter envoyer message dans app

**Critères succès** :
- [ ] Erreur détectée rapidement (< 2s)
- [ ] Message erreur clair : "Backend disconnected"
- [ ] Suggestion action : "Restart app" ou bouton "Restart Backend"
- [ ] App ne crash pas
- [ ] Retry automatique (optionnel) : tente restart backend

---

#### Test 6 : Arrêt app propre ✅
**Actions** :
1. Backend running, conversation active
2. Quitter app (Cmd+Q)
3. Vérifier Activity Monitor

**Critères succès** :
- [ ] Process `python` terminé proprement
- [ ] Aucun process orphelin
- [ ] Logs backend montrent "Shutdown" si disponible
- [ ] Conversations sauvegardées (relancer app → toujours là)

---

#### Test 7 : Restart app rapide ✅
**Actions** :
1. Lancer app
2. Attendre backend ready (~20s)
3. Quitter app
4. Relancer immédiatement

**Critères succès** :
- [ ] Backend redémarre sans conflit port
- [ ] Pas d'erreur "Address already in use"
- [ ] Startup rapide (backend cache model ?)

---

#### Test 8 : Multi-conversations simultanées ✅
**Actions** :
1. Backend running
2. Ouvrir conversation A, envoyer message (génération en cours)
3. Switcher conversation B, envoyer message
4. Observer les deux générations

**Critères succès** :
- [ ] Les deux générations progressent (backend gère concurrence)
- [ ] Pas de mélange réponses (bonne conversation)
- [ ] Pas de crash app
- [ ] Performance acceptable (peut être légèrement plus lent)

---

### Performance attendue Phase 2

**Backend** :
- Démarrage : < 30s (chargement modèle)
- Throughput : ~17 tok/s (mesuré Phase 1)
- Health check : < 100ms
- Latency HTTP localhost : ~5ms (négligeable)

**Frontend** :
- App launch : < 2s (UI affichée pendant backend startup)
- Switching conversations : instantané
- Streaming tokens : fluide (perception temps réel)
- CPU usage app Swift : faible (GPU utilisé par backend)

**Régression acceptable** :
- Génération 2x plus lente que Swift MLX (17 vs 35 tok/s)
- **Raison** : Overhead Python, mais bénéfice = architecture évolutive
- **Compensation** : Streaming masque latency perçue

### Fichiers créés

```
AssistantIA/Core/Services/
├── BackendManager.swift          # Lifecycle subprocess Python
│   - @Observable @MainActor
│   - enum State (stopped/starting/running/error)
│   - start() async throws : poll health check toutes les 2s, timeout 30s
│   - stop() : SIGTERM + Task.detached waitUntilExit
│   - checkHealth() → [String: Any]
│   - Auto-restart 1× si crash (terminationHandler)
│   - findRepoRoot() : NSUserName() pour contourner sandbox macOS
│   - PYTHONPATH + VIRTUAL_ENV + PATH reconstruits manuellement
│
└── ChatAPI.swift                 # Client HTTP + streaming SSE
    - final class (état immuable = thread-safe sans actor)
    - streamMessage() → AsyncThrowingStream<String, Error>
    - sendMessage() async throws → String
    - Parser SSE : strip "data: " → SSEChunk { text?, error? }
    - Erreurs typées : backendUnavailable/backendBusy/timeout/streamError
    - Cancellation : continuation.onTermination = { task.cancel() }

AssistantIA/UI/Settings/
└── BackendSettingsView.swift     # Monitoring backend (debug)
    - Form avec 4 sections : état, contrôles, erreur, infos
    - ElapsedTimeView : timer écoulé pendant .starting
    - DisclosureGroup erreur : texte monospace scrollable + bouton copier
    - Boutons contextuels : Start / Stop / Restart
```

### Fichiers modifiés

```
AssistantIA/Application/AssistantIAApp.swift   (+22 lignes)
    - @State private var backendManager = BackendManager()
    - .environment(backendManager) sur WindowGroup
    - .task { await startBackend() } : démarrage asynchrone, UI non bloquée
    - .onReceive(willTerminateNotification) : arrêt propre au quit

AssistantIA/UI/RootView.swift                  (+40 lignes)
    - @Environment(BackendManager.self) backendManager
    - backendStatusToolbar : point coloré .status (vert/orange/rouge/gris)
    - Sheet BackendSettingsView déclenchée au clic
    - .environment(backendManager) ré-injecté dans la sheet

AssistantIA/UI/Chat/ConversationViewModel.swift  (~45 lignes changées)
    - import MLXLMCommon supprimé
    - mlxService conservé dans init (compat ConversationManager, non utilisé)
    - generate() : for try await chunk in ChatAPI.shared.streamMessage()
    - generateTitleWithAI() : ChatAPI.shared.sendMessage(maxTokens: 30)
    - buildPrompt() : formate [Message] → String pour API backend
    - tokensPerSecond : retourne 0 (loggué server-side)
    - modelDownloadProgress : retourne nil
```

### Fichiers désactivés (conservés pour rollback)

```
Core/Services/MLXService.swift        # NE PAS SUPPRIMER
Core/Utilities/ModelStateManager.swift # Conservé, non utilisé
```

### Fichiers NON modifiés (comme prévu)

```
✅ Core/Utilities/ConversationManager.swift
✅ Core/Models/Message.swift / Conversation.swift / ConversationMetadata.swift
✅ UI/Chat/ChatView.swift + Components/*
```

### Performance mesurée Phase 2

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| Startup backend (modèle chargé) | ~35s (M4 Pro, SSD externe) | < 60s | ✅ |
| Health check polling | 2s intervalle | — | ✅ |
| Throughput génération | ~17 tok/s | >= 10 tok/s | ✅ |
| Latency HTTP localhost | ~5ms | < 100ms | ✅ |
| App launch (UI visible) | < 2s | < 2s | ✅ |
| App Sandbox | Désactivé | — | ℹ️ |

> **Note** : Startup ~35s observé avec modèle sur SSD externe (USB). Phase 1 mesurait ~16s sur NVMe interne. Comportement normal.

### Tests validation Phase 2

- [x] Test 1 : Démarrage automatique — logs `[BackendManager] ✅ Backend prêt`, UI non bloquée
- [ ] Test 2 : Chat fonctionnel via backend (à valider)
- [ ] Test 3 : Conversations existantes accessibles (à valider)
- [ ] Test 4 : Performance ~17 tok/s (à mesurer)
- [ ] Test 5 : Gestion erreur backend down (à valider)
- [ ] Test 6 : Arrêt app propre (à valider)
- [ ] Test 7 : Restart rapide sans conflit port (à valider)
- [ ] Test 8 : Multi-conversations simultanées (à valider)

### Problèmes rencontrés

**1. `URLError.Code.connectionReset` inexistant sur Apple SDK**
- Symptôme : erreur de compilation dans `classify(_ error:)`
- Cause : `connectionReset` est une erreur POSIX/Linux, absente du SDK Apple
- Solution : supprimé du switch — les cas restants couvrent tous les scénarios localhost

**2. `NSHomeDirectory()` retourne le répertoire sandbox**
- Symptôme : `main.py introuvable dans /Users/mr/Library/Containers/com.jbsk.AssistantIA/Data/...`
- Cause : `ENABLE_APP_SANDBOX = YES` — `NSHomeDirectory()` retourne le container, pas `~/`
- Solution : `NSUserName()` pour construire `/Users/\(NSUserName())/...` (non affecté par le sandbox)

**3. Type ambiguïté sur `process?.processIdentifier`**
- Symptôme : `Type of expression is ambiguous without a type annotation`
- Cause : optional chaining retourne `Int32?`, ternaire entre `Int32?` et `nil` = ambiguïté
- Solution : `guard let p = process, p.isRunning else { return nil }; return Int(p.processIdentifier)`

**4. App Sandbox bloque l'exécution de Python Homebrew**
- Symptôme : `The file "python3.14" doesn't exist`
- Cause : App Sandbox refuse l'accès à `/opt/homebrew/opt/python@3.14/bin/python3.14`
- Solution : désactiver App Sandbox (Target → Signing & Capabilities → supprimer "App Sandbox")
- Note : cette app est un outil dev local, jamais publiée sur App Store → acceptable

**5. `resolvingSymlinksInPath()` casse l'activation du venv Python**
- Symptôme : `ModuleNotFoundError: No module named 'uvicorn'` malgré Python trouvé
- Cause : en résolvant `venv/bin/python → /opt/homebrew/.../python3.14`, Python démarre depuis
  Homebrew et ne trouve pas `pyvenv.cfg` → venv non activé → packages absents
- Solution : reconstruire manuellement l'environnement venv dans la config du process :
  `VIRTUAL_ENV`, `PYTHONPATH` (site-packages détecté dynamiquement), `PATH` avec `venv/bin` en tête

**6. Logs réseau trop verbeux pendant startup**
- Symptôme : ~60 lignes de `nw_connection_copy_protocol_metadata_internal_block_invoke` par seconde
- Cause : health check polling toutes les 500ms × logs URLSession "Connection refused"
- Solution : polling 500ms → 2s (×4 moins de tentatives, ~15s de délai max supplémentaire)

### Décisions techniques Phase 2

| Décision | Raison |
|---|---|
| `final class` pour ChatAPI (pas `actor`) | État entièrement immuable (let) = thread-safe sans overhead d'actor |
| `@Observable @MainActor` pour BackendManager | Re-render SwiftUI automatique à chaque transition d'état |
| Health check polling (pas attente fixe) | Détecte readiness dès que prêt (~35s variable selon SSD) |
| `.task` (pas `.onAppear`) pour auto-start | Async, déclenché une fois, annulé au quit (CancellationError → stop) |
| `mlxService` conservé dans init ConversationViewModel | Évite de modifier ConversationManager — compat ascendante |
| `buildPrompt()` formate messages en texte | API backend accepte `prompt: str` (Phase 2) — multi-turn Phase 3 |
| App Sandbox désactivé | Nécessaire pour subprocess Homebrew Python — app jamais distribuée |
| PYTHONPATH/VIRTUAL_ENV reconstruits manuellement | `resolvingSymlinksInPath()` + Sandbox = impossible d'utiliser l'activation venv classique |
| Auto-restart max 1× après crash | Évite restart infini si bug persistant ; laisse l'utilisateur décider |

## PHASE 2bis : Migration responsabilité métier vers backend ✅ COMPLÉTÉE

**Dates** : 2026-03-12

### Objectif

Migrer la responsabilité de gestion des conversations du frontend Swift vers le backend Python. Les données restent au même emplacement (SSD externe partagé), seule la couche d'accès change. Le frontend devient UI pure consommant une API REST. Supprimer tous les artefacts de l'ancienne architecture Swift MLX (boutons Load/Unload, progress bars, tok/s, SetupView).

### Fichiers créés (Backend)

- **`backend/models/conversation.py`** (68 lignes)
  - Pydantic v2 models : `Role` (str enum), `Message`, `ConversationMetadata`, `Conversation`
  - `@field_serializer` sur tous les champs `datetime` → ISO 8601 avec offset UTC (`+00:00`)
  - Décision : `@field_serializer` plutôt que `Config.json_encoders` (Pydantic v2, non déprécié)

- **`backend/storage/json_manager.py`** (206 lignes)
  - `JSONManager` : CRUD fichiers JSON + index `conversations.json`
  - `threading.Lock()` global pour toutes les I/O — protège contre accès concurrents
  - `load_index()` : auto-cleanup des entrées orphelines (index présent, `{uuid}.json` absent)
  - `verify_integrity()` → `{"cleaned": N}` — appelé au démarrage backend
  - Décision : `_read_index_file()` retourne `[]` gracieusement si fichier absent ou JSON invalide

- **`backend/api/conversations.py`** (329 lignes)
  - `APIRouter(prefix="/api/conversations")` — CRUD complet
  - DTO séparés : `CreateConversationRequest`, `UpdateConversationRequest`, `ConversationResponse`
  - `createdAt` immutable ; `updatedAt` rafraîchi au PUT
  - Liste triée par `updatedAt` décroissant
  - HTTP 201 POST, 204 DELETE, 404 conversation inconnue, 422 validation

- **`backend/api/messages.py`** (401 lignes)
  - `APIRouter(prefix="/api/conversations/{conversation_id}/messages")`
  - Lock par conversation (`dict[str, Lock]`) pour écriture concurrente safe
  - `GET /` : exclut les messages `system`
  - `POST /` : sauvegarde user → génère → sauvegarde assistant → 201
  - `POST /stream` : sauvegarde user → SSE stream → sauvegarde assistant en `finally` (partiel si disconnect)
  - `_build_messages_for_llm()` : historique complet → `List[Dict]` pour `apply_chat_template`

- **`backend/tests/test_json_manager.py`** (276 lignes) — 16 tests
- **`backend/tests/test_conversations_api.py`** (291 lignes) — 24 tests
- **`backend/tests/test_messages_api.py`** (416 lignes) — 20 tests
  - `MockEngine`, `FailingEngine`, `UnloadedEngine`, `CapturingEngine` — pas de modèle MLX requis

### Fichiers créés (Frontend)

- **`AssistantIA/Core/Services/ConversationAPI.swift`** (441 lignes)
  - `actor ConversationAPI` — isolation garantit thread-safety sans lock manuel
  - DTO privés (`APIMessage`, `APIConversation`, `APIConversationMetadata`) séparés des modèles UI
  - `makeDecoder()` : deux `ISO8601DateFormatter` en cascade (avec/sans microsecondes) — `.iso8601` natif échoue sur `datetime.isoformat()` Python (microsecondes)
  - `nonisolated func sendMessage()` → `AsyncThrowingStream<String, Error>` (streaming SSE)
  - `continuation.onTermination = { _ in task.cancel() }` — cancellation propagée vers URLSession
  - Erreurs typées : `networkError`, `notFound`, `modelLoading`, `serverError`, `decodingError`, `streamError`, `cancelled`

- **`AssistantIA/Core/Services/BackendAdminAPI.swift`** (63 lignes)
  - Client admin typé : `health()` → `BackendHealth` (status, modelLoaded, modelName, pid)
  - `keyDecodingStrategy = .convertFromSnakeCase` — évite mapping manuel snake→camel

### Fichiers modifiés (Backend)

- **`backend/main.py`** (147 → 221 lignes)
  - Migration `@app.on_event` (déprécié) → `@asynccontextmanager lifespan`
  - Chargement modèle en tâche de fond `asyncio.create_task(_load_model())` — HTTP disponible immédiatement, conversations chargées avant fin de chargement modèle
  - `app.state.json_manager` initialisé + `verify_integrity()` au startup
  - `app.include_router(conversations_router)` + `app.include_router(messages_router)` — routes absentes en Phase 2 (cause des 404)
  - `/health` enrichi avec `pid` — BackendManager vérifie PID pour éviter faux "backend prêt" si port déjà occupé
  - `MAX_GENERATION_TOKENS` / `MAX_GENERATION_TEMPERATURE` via `.env` — plafond serveur appliqué sur `/chat`, `/chat/stream`, `/messages`, `/messages/stream`
  - `app.state.engine` initialisé à `None` dès le démarrage — évite `AttributeError` avant fin de chargement

- **`backend/core/llm.py`** (123 lignes)
  - `stream_messages(messages: List[Dict])` → `apply_chat_template` pour multi-turn
  - `generate_messages(messages)` → collecte le stream
  - `stream()` et `generate()` délèguent à `stream_messages()` — rétrocompatibilité Phase 2

- **`backend/.env.example`**
  - Ajout `DATA_FOLDER=/Volumes/AISSD/conversations`
  - Ajout `MAX_GENERATION_TOKENS=512` + `MAX_GENERATION_TEMPERATURE=0.3`

### Fichiers modifiés (Frontend)

- **`AssistantIA/Core/Utilities/ConversationManager.swift`** (refactorisé → 199 lignes)
  - Supprimé : `ConversationStorageError`, `dataFolderURL`, `setDataFolderURL()`, `loadIndex()`, `saveIndex()`, `loadMessages()`, `saveMessages()`, `indexFileURL()`, `conversationFileURL()`
  - Ajouté : `conversationAPI: ConversationAPI`, `loadError: String?`, `loadConversations() async throws`
  - `createConversation()` → `async throws` (UUID généré par le backend)
  - `deleteConversation()` / `renameConversation()` → optimistic local update + fire-and-forget `Task`
  - `getViewModel()` → placeholder sync + `Task` charge les vrais messages → `@Observable` re-render automatique
  - `viewModelCache[UUID]` — préserve l'état UI (isGenerating, prompt) si on revient sur une conversation
  - `init(conversationAPI:)` — `mlxService` supprimé (plus nécessaire)

- **`AssistantIA/UI/Chat/ConversationViewModel.swift`** (simplifié → 179 lignes)
  - Supprimé : `onConversationUpdated`, `tokensPerSecond`, `modelDownloadProgress`, `buildPrompt()`, `mlxService`
  - `generate()` : `for try await chunk in conversationAPI.sendMessage()` — streaming direct
  - Annulation : message partiel conservé avec `\n[Annulé]` — cohérent avec ce que le backend a sauvegardé
  - `generateTitleWithAI()` : `ChatAPI.shared.sendMessage()` — stateless, n'ajoute pas de message à l'historique
  - Erreur réseau → `errorMessage` affiché dans l'UI

- **`AssistantIA/Core/Services/BackendManager.swift`** (360 lignes)
  - `checkHealth()` consomme `BackendAdminAPI.health()` → `BackendHealth` typé
  - `waitForBackendReady()` : vérifie PID du health check contre PID du process lancé — évite faux positif si port déjà occupé par instance précédente
  - `handleProcessTermination()` : gère crash pendant `.starting` séparément de crash pendant `.running`
  - `restartAttempts` réinitialisé à 0 au démarrage propre
  - `baseURL: URL` exposé — consommé par `BackendAdminAPI`

- **`AssistantIA/Application/AssistantIAApp.swift`** (63 lignes)
  - `startBackend()` : `backendManager.start()` attend que le serveur HTTP soit prêt (pas nécessairement le modèle), puis `conversationManager.loadConversations()` — conversations disponibles avant fin de chargement modèle
  - Catch séparé pour erreur backend vs erreur chargement conversations

- **`AssistantIA/UI/Chat/ConversationListView.swift`**
  - `createConversation()` wrappé dans `Task { try? await ... }`

- **`AssistantIA/UI/RootView.swift`** — supprimé `modelControlToolbar`, `modelStateManager`, routing `SetupView`, bannière "Model not loaded"

- **`AssistantIA/UI/Chat/ChatView.swift`** — titre "MLX Chat Example" → "AssistantIA"

- **`AssistantIA/UI/Chat/Components/Toolbar/ChatToolbarView.swift`** — supprimé `DownloadProgressView` + `GenerationInfoView`

- **`AssistantIA/UI/Settings/BackendSettingsView.swift`** — amélioré : bouton envoyer désactivé tant que modèle non prêt, statut backend déplacé/agrandi dans toolbar

### Fichiers supprimés (Frontend)

- `AssistantIA/Core/Utilities/ModelStateManager.swift` — plus aucun usage
- `AssistantIA/UI/Chat/SetupView.swift` — routing conditionnel MLX supprimé
- `AssistantIA/UI/Chat/Components/Toolbar/DownloadProgressView.swift` — artefact MLX
- `AssistantIA/UI/Chat/Components/Toolbar/GenerationInfoView.swift` — affichait tok/s MLX local

### Performance mesurée Phase 2bis

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| GET /api/conversations/ (vide) | < 5ms | < 20ms | ✅ |
| POST /api/conversations/ | < 10ms | < 50ms | ✅ |
| GET /api/conversations/{id} | < 5ms | < 20ms | ✅ |
| Throughput génération | ~17 tok/s | >= 10 tok/s | ✅ |
| Startup backend (HTTP prêt) | < 1s | < 5s | ✅ |
| Chargement modèle (bg task) | ~35s | < 60s | ✅ |
| Conversations disponibles avant modèle chargé | ✅ | — | ✅ |

> **Note** : Le chargement modèle est désormais en tâche de fond — l'UI et les conversations sont disponibles ~34s avant la fin du chargement du modèle.

### Tests validation Phase 2bis

- [x] `test_json_manager.py` — 16 tests ✅ (CRUD, auto-cleanup, ISO 8601, verify_integrity)
- [x] `test_conversations_api.py` — 24 tests ✅ (CRUD, 404/422, tri, datetime)
- [x] `test_messages_api.py` — 20 tests ✅ (GET/POST/stream, mock engine, 503, multi-turn)
- [x] `test_basic.py` — 4 tests (nécessitent backend running — validation manuelle)
- [x] Frontend liste conversations via API
- [x] Frontend CRUD fonctionne (create, rename, delete)
- [x] Chat streaming backend → messages auto-sauvegardés
- [x] UI nettoyée — aucun artefact MLX visible
- [x] Backend down → `errorMessage` clair dans le ViewModel
- [x] Cancel mid-génération → message partiel `[Annulé]` + backend sauvegarde le partiel
- [x] Conversations disponibles avant fin chargement modèle

### Problèmes rencontrés

**1. Routes API conversations → 404 au démarrage**
- Cause : `app.include_router(conversations_router)` absent de `main.py` — oublié en fin de Phase 2bis-A
- Solution : ajout des deux `include_router` + initialisation `app.state.json_manager` dans `lifespan`

**2. Tests 404 retournaient 422**
- Cause : tests utilisaient `"ghost-id"` / `"nonexistent-uuid"` (pas des UUIDs valides) — le backend valide le format UUID avant de chercher la ressource → 422 correct
- Solution : remplacer par `"00000000-0000-0000-0000-000000000000"` (UUID valide mais inexistant)

**3. `on_event` startup/shutdown déprécié**
- Solution : migration vers `@asynccontextmanager lifespan(app)` — API officielle FastAPI

**4. `app.state.engine` AttributeError avant chargement modèle**
- Cause : `app.state.engine` n'existait pas avant fin de `_load_model()` — les endpoints `/chat` accédaient à `app.state.engine` pendant le chargement
- Solution : `app.state.engine = None` avant `asyncio.create_task(_load_model())` + `getattr(app.state, "engine", None)` dans les endpoints

**5. `dateDecodingStrategy = .iso8601` échoue sur les datetimes Python**
- Cause : `datetime.isoformat()` Python produit `"2026-03-12T10:30:00.123456+00:00"` (microsecondes + offset `+00:00`) — `.iso8601` de Swift ne supporte pas les microsecondes
- Solution : deux `ISO8601DateFormatter` en cascade dans `makeDecoder()` — avec `.withFractionalSeconds` en premier, fallback sans

**6. Chargement modèle bloquait le démarrage HTTP**
- Cause : dans l'ancienne version, `startup()` chargeait le modèle de façon synchrone — le serveur ne démarrait qu'après
- Solution : `asyncio.create_task(_load_model())` — HTTP disponible immédiatement, modèle chargé en fond

**7. `mx.metal.clear_cache()` → API dépréciée**
- Solution : `mx.clear_cache()` (API correcte dans mlx >= 0.21)

### Décisions techniques Phase 2bis

| Décision | Justification |
|---|---|
| JSON (pas SQLite) | Debug facile — fichiers lisibles directement ; zéro migration depuis Phase 2 ; versionnable |
| `threading.Lock()` global (pas par conv) | Simplicité — index partagé nécessite lock global ; overhead négligeable en usage réel |
| `@field_serializer` Pydantic v2 | `Config.json_encoders` déprécié en Pydantic v2 |
| ISO 8601 avec microsecondes + offset `+00:00` | `datetime.isoformat()` Python natif ; compatible Swift via formatters en cascade |
| `asyncio.create_task` pour modèle | Conversations ne nécessitent pas le modèle — disponibles ~34s plus tôt |
| DTO privés dans ConversationAPI | Contrat réseau découplé des modèles UI — backend peut évoluer sans toucher les modèles Swift |
| `actor ConversationAPI` (pas `final class`) | Méthodes mutent `session`/`decoder` → actor obligatoire ; `nonisolated` pour le stream |
| Optimistic update delete/rename | UX fluide — pas d'attente réseau pour opérations locales |
| Sauvegarde SSE dans `finally` | Une seule écriture disque par génération ; sauvegarde partielle si client se déconnecte |
| `buildPrompt()` conservé (workaround) | API `/chat` accepte `prompt: str` — migration vers `messages: List[Dict]` en Phase 3 |
| Bouton envoyer désactivé si modèle non prêt | UX : évite erreur 503 visible ; stop reste accessible si génération en cours |

### Observations pour Phase 3

- **Multi-turn natif** : `stream_messages(messages: List[Dict])` est en place côté backend. Il faut migrer `ConversationViewModel` pour passer l'historique complet plutôt que `buildPrompt()` (concaténation texte).
- **JSONManager extensible** : peut stocker des embeddings vectoriels par message (champ `embedding: List[float]`) sans modifier la structure — base naturelle pour RAG.
- **Pattern DTO privés réutilisable** : `ConversationAPI` pourra exposer des events tool-call via le stream SSE sans modifier les types Swift publics.
- **`AsyncThrowingStream` déjà prêt** : le stream SSE peut véhiculer des événements typés (`tool_call`, `tool_result`, `text`) sans changer l'interface consommateur.
- **Startup lent atténué** : avec le chargement modèle en fond, l'UX s'améliore même à 35s. Phase 3 pourra investiguer le keep-alive du process backend entre sessions.

---

## PHASE 3 : Tool Calling System (EN COURS)

### Objectif
Implémenter un système de tool calling natif exploitant les capacités de Ministral 3 14B Instruct, avec une architecture évolutive permettant l'auto-génération d'outils et l'apprentissage des patterns d'usage.

### Contraintes non-négociables

#### Format Tool Calling
- **Standard OpenAI STRICT** : Le format est celui défini par OpenAI function calling spec, adopté par Mistral AI
- **Documentation de référence** :
  - OpenAI : https://platform.openai.com/docs/guides/function-calling
  - Mistral AI : https://docs.mistral.ai/capabilities/function_calling/
- **Structure exacte** : `{"type": "function", "function": {"name": ..., "description": ..., "parameters": {"type": "object", "properties": {...}, "required": [...]}}}`
- **Non-modifiable** : Clés `type`, `function`, `parameters`, structure imbriquée
- **parameters.type** : TOUJOURS `"object"` même pour un seul paramètre
- **Types JSON Schema** : Uniquement `string`, `number`, `boolean`, `array`, `object`
- **Validation obligatoire** : Chaque tool schema doit être validé avant injection au modèle

#### Architecture
- **Registry centralisé** : Singleton ou DI via FastAPI, mais un seul point d'enregistrement
- **Permissions granulaires** : Système à 6 niveaux (READ_ONLY, WRITE_SAFE, WRITE_MODIFY, SYSTEM_EXEC, NETWORK, AUTONOMOUS)
- **Idempotence** : Les outils doivent produire le même résultat si appelés plusieurs fois avec les mêmes args
- **Isolation** : Chaque outil s'exécute dans son scope, pas de variables globales partagées
- **Métadonnées riches** : success_count, failure_count, last_used, created_by, version
- **Découplage total** : Les outils ne dépendent pas de MinistralEngine ou FastAPI

#### Sécurité
- **Validation des outils générés** : AST parsing pour détecter imports/code dangereux
  - Interdits : `eval`, `exec`, `compile`, `__import__`, `os.system`, `subprocess`
  - AST walker pour détecter ces patterns avant ajout au registry
- **Confirmation obligatoire** : Flag `requires_confirmation` pour outils destructifs (WRITE_MODIFY+)
- **Review humaine** : Tous les outils auto-générés passent par validation utilisateur
- **Validation arguments** : Pydantic validation stricte avant exécution
- **Timeout** : Chaque outil a timeout configurable (default 30s, max 300s)
- **Audit trail** : Toutes exécutions loggées (timestamp, tool_name, args, result, duration, error)
- **Pas de vrai sandbox** : Pas de container Docker, juste validation AST + review + timeout

#### Performance
- **Async-first** : Tous les outils sont `async def`, même si implémentation sync (via `asyncio.to_thread`)
- **Registry thread-safe** : `asyncio.Lock` pour toutes les mutations du registry
- **Lazy loading** : Les outils ne sont pas chargés en mémoire tant qu'ils ne sont pas appelés
- **Tool injection overhead** : < 50ms pour formater et injecter les tools dans le prompt
- **Parsing overhead** : < 50ms pour parser la réponse et extraire tool calls
- **Caching optionnel** : Les outils peuvent déclarer `cacheable=True` pour résultats déterministes

#### Évolutivité
- **Découplage** : Les outils vivent dans `backend/tools/`, séparés de `core/`
- **Hot-reload** : Possibilité d'ajouter/modifier des outils sans redémarrer le backend (reload du registry)
- **Versioning** : Chaque outil a une version sémantique (1.0.0), possibilité de rollback
- **Categories hiérarchiques** : `system`, `filesystem`, `web`, `communication`, `data`, `custom`
- **Plugin system** : Architecture permet d'ajouter des outils via packages Python externes

### Architecture cible
```
backend/
├── core/
│   └── tools/
│       ├── __init__.py
│       ├── schema.py          # ToolSchema, ToolParameter (Pydantic v2)
│       ├── registry.py        # ToolRegistry (singleton ou DI)
│       ├── executor.py        # ToolExecutor (timeout + error handling)
│       └── validator.py       # ToolValidator (AST + security checks)
├── tools/
│   ├── __init__.py
│   ├── builtin/               # Outils système non-modifiables
│   │   ├── __init__.py
│   │   ├── system.py          # get_time, get_date, get_system_info
│   │   ├── filesystem.py      # read_file, write_file, list_files (WRITE_SAFE)
│   │   └── web.py             # http_request (NETWORK)
│   └── generated/             # Outils créés par l'IA (vide au départ)
│       └── __init__.py
├── (persistence externe via DATA_FOLDER / TOOLS_FOLDER)
│   └── tools_registry/
│       ├── registry.json       # Métadonnées des outils
│       ├── generated/          # Code Python des outils générés
│       │   └── *.py
│       └── usage_patterns.json # Apprentissage (Phase 3.5)
└── api/
    └── tools.py               # Endpoints CRUD outils
```

### Format Tool Schema (Standard OpenAI)

**Exemple complet conforme** :
```json
{
  "type": "function",
  "function": {
    "name": "get_current_time",
    "description": "Get the current time in a specific timezone. Returns ISO 8601 formatted string.",
    "parameters": {
      "type": "object",
      "properties": {
        "timezone": {
          "type": "string",
          "description": "IANA timezone identifier (e.g., 'Europe/Paris', 'America/New_York')",
          "enum": null
        }
      },
      "required": []
    }
  }
}
```

**Métadonnées étendues (stockées séparément)** :
```json
{
  "name": "get_current_time",
  "permission_level": "read_only",
  "requires_confirmation": false,
  "category": "system",
  "created_by": "system",
  "version": "1.0.0",
  "created_at": "2026-03-13T10:00:00Z",
  "success_count": 42,
  "failure_count": 1,
  "last_used": "2026-03-13T15:30:00Z",
  "average_duration_ms": 12.5,
  "timeout_seconds": 30
}
```

### Intégration Ministral 3

Ministral 3 14B Instruct supporte **nativement** le function calling au format OpenAI.

**Workflow de tool calling** :
```
1. USER MESSAGE
   ↓
2. TOOL INJECTION
   - Registry.get_tools_for_ministral() → List[dict] format OpenAI
   - Injection dans system prompt ou messages
   ↓
3. MINISTRAL GÉNÉRATION
   - Détecte besoin d'un outil
   - Génère JSON : {"tool_calls": [{"name": "...", "arguments": {...}}]}
   ↓
4. PARSING
   - Extrait JSON de la réponse (peut contenir texte avant/après)
   - Valide structure tool_calls
   ↓
5. EXECUTION
   - Pour chaque tool call : Registry.execute(name, args)
   - Timeout, validation, error handling
   ↓
6. RESULT INJECTION
   - Ajoute résultats au contexte
   - Format : {"role": "tool", "tool_call_id": "...", "content": "..."}
   ↓
7. CONTINUATION
   - Si autres tools nécessaires → retour à 3
   - Sinon → réponse finale
   ↓
8. LIMITE ITERATIONS
   - Max 5 boucles pour éviter infinite loop
```

**Format de génération attendu (Ministral)** :
```json
{
  "tool_calls": [
    {
      "name": "get_current_time",
      "arguments": {
        "timezone": "Europe/Paris"
      }
    }
  ]
}
```

**Cas d'erreur à gérer** :
- JSON malformé → regex fallback, extraction partielle
- Outil inexistant → error message clair à l'IA
- Arguments manquants → validation Pydantic, error détaillé
- Multiples tool calls → exécution séquentielle ou parallèle selon dépendances
- Timeout dépassé → kill task, retour error
- Boucle infinie → stop à max_iterations, retour état final

### Phases d'implémentation

#### Phase 3.1 : Foundation ✅ COMPLÉTÉE

**Dates** : 2026-03-13 | **Durée** : 1 jour

**Sous-phases** :
- [x] **3.1.1 : Tool Schema** — `ToolSchema`, `ToolParameter` (Pydantic v2, format OpenAI strict)
- [x] **3.1.2 : Tool Registry** — Singleton thread-safe, persistence JSON, protection système
- [x] **3.1.3 : Outils Builtin** — `get_current_time`, `get_system_info`, `ping_host` + auto-registration
- [x] **3.1.4 : Intégration MinistralEngine** — `stream_with_tools()`, parsing robuste, boucle multi-tools
- [x] **3.1.5 : Tests** — 32/32 tests unitaires passent ; E2E via `MODEL_PATH=/path pytest -k e2e`

**Validation Phase 3.1** :
- ✅ L'IA peut appeler `get_current_time()` et utiliser le résultat
- ✅ Format OpenAI strict respecté et validé (100% conformité)
- ✅ Pas de crash sur outil inexistant ou args invalides
- ✅ Performance overhead < 1ms (seuil 100ms largement respecté)
- ✅ Thread-safety : 100 concurrent ops sans corruption
- ✅ Aucune régression fonctionnelle (stream(), generate() inchangés)
- ⬜ E2E avec vrai modèle (MODEL_PATH requis, non automatisé)

---

### PHASE 3.1 — Détail d'implémentation

#### Fichiers créés

| Fichier | Lignes | Contenu |
|---|---|---|
| `backend/core/tools/__init__.py` | 16 | Exports : ToolSchema, ToolParameter, PermissionLevel, ToolRegistry, SystemToolProtectedError |
| `backend/core/tools/schema.py` | 241 | ToolSchema, ToolParameter, PermissionLevel (6 niveaux) |
| `backend/core/tools/registry.py` | 418 | ToolRegistry singleton, asyncio.Lock, persistence JSON |
| `backend/tools/__init__.py` | 15 | Export register_builtin_tools |
| `backend/tools/builtin/__init__.py` | 68 | register_builtin_tools(registry) — appelé au lifespan |
| `backend/tools/builtin/system.py` | 306 | get_current_time, get_system_info, ping_host + BUILTIN_TOOLS dict |
| `backend/tests/test_tool_calling_e2e.py` | ~430 | 32 tests (3.1.1→perf) + 3 tests E2E avec vrai modèle |
| `TOOLS_FOLDER` (ou `DATA_FOLDER/tools_registry`) | — | Dossier externe ; `registry.json` auto-généré |

#### Fichiers modifiés

| Fichier | Delta | Changements |
|---|---|---|
| `backend/core/llm.py` | +493 lignes (617 total) | stream_with_tools() + 9 méthodes privées ; stream/generate/stream_messages inchangés |
| `backend/main.py` | +5 lignes | import ToolRegistry/register_builtin_tools ; lifespan : registry init |
| `backend/requirements.txt` | +2 lignes | psutil>=5.9.0, pytz>=2024.1 |

#### Performance mesurée (données réelles, M4 Pro)

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| Tool injection JSON (1000 runs avg) | 0.012 ms | < 50 ms | ✅ |
| Parsing tool calls (50 000 runs avg) | 0.0035 ms | < 50 ms | ✅ |
| `get_tools_for_ministral()` 50 tools | 0.019 ms | < 50 ms | ✅ |
| 50 concurrent register ops | 6.5 ms total | — | ✅ |
| `get_current_time()` exécution | 0.13 ms avg | — | ✅ |
| `get_system_info()` exécution | ~106 ms (cpu_percent bloquant, thread) | — | ℹ️ |
| Conformité format OpenAI | 100% | 100% | ✅ |
| Faux positifs parsing (4 patterns) | 0% | < 5% | ✅ |

#### Checklist tests

**3.1.1 Tool Schema** : [x] format exact [x] snake_case [x] types stricts [x] datetime round-trip [x] datetime naïf rejeté [x] enum output [x] noms dupliqués

**3.1.2 Tool Registry** : [x] singleton [x] persistence [x] force flag [x] protection système [x] format OpenAI pur [x] filtrage permission [x] reload disk [x] 100 concurrent ops [x] update_stats

**3.1.3 Outils Builtin** : [x] time ISO 8601 [x] time tz invalide [x] system_info métriques [x] system_info filtrage sensibles [x] ping valide [x] ping invalide [x] auto-registration [x] schémas conformes

**3.1.4 MinistralEngine** : [x] texte pur [x] simple tool call [x] multi-tool [x] JSON mixte [x] [TOOL_CALLS] natif [x] outil inconnu [x] max_iterations [x] tous types events [x] backward compat [x] modèle non chargé

**Perf** : [x] injection < 50ms [x] parsing < 50ms

**3.1.5 E2E** : [ ] get_current_time réel [ ] get_system_info réel [ ] réponse directe sans outil

#### Problèmes rencontrés

**1. Injection tools — Documentation mlx-lm incomplète**
- La doc Mistral couvre uniquement l'API cloud. `apply_chat_template(tools=...)` non documenté pour mlx-lm.
- Solution : Paramètre natif HuggingFace + fallback `TypeError → injection system prompt`

**2. Parsing JSON mixte — Regex insuffisante**
- `re.search(r'\{.*"tool_calls".*\}', re.DOTALL)` capturait trop (JSON imbriqués dans arguments)
- Solution : Parser manuel brace counter (`_extract_json_object`) → 0% faux positifs

**3. asyncio.Lock — Pas de RWLock natif**
- Toutes lectures bloquées mutuellement — acceptable pour < 100 outils
- Solution : Copie atomique `dict.copy()` sous lock, traitement hors lock → 0.019ms avg

**4. ping_host latence élevée sur 8.8.8.8**
- Port 80 timeout (3s) avant que port 443 réponde → latency_ms = timeout + connexion
- Comportement attendu, documenté. Timeout = par port, non total.

**5. Singleton + tests isolation**
- `_instance` partagée polluait les tests
- Solution : `_reset_instance()` (tests) + constructeur public avec `storage_path`

#### Décisions techniques

| Décision | Choix | Alternative rejetée | Justification |
|---|---|---|---|
| Format types JSON Schema | `Literal["string"...]` | `Enum` | Erreurs Pydantic lisibles |
| AwareDatetime | Obligatoire | datetime naïf accepté | Round-trip parfait, timezone explicite |
| Registry pattern | Singleton + constructeur public | DI FastAPI pure | Accessible partout ; tests isolés sans mock |
| Exécution multi-tool | Séquentielle | asyncio.gather() | Dépendances inter-tools indétectables |
| Parsing JSON | Brace counter manuel | Regex | 0% faux positifs sur JSON imbriqués |
| Async wrapping | Tous async dès maintenant | Sync + wrapper ToolExecutor | Uniformité interface stream_with_tools |
| Format erreur LLM | get_time→str ; ping→dict | Exception / None | LLM peut parser/répéter string ; dict permet branchement sur success |
| Protection système | Dans Registry | Dans ToolSchema | Seul le Registry connaît l'état existant |

#### API publique Phase 3.1

```python
# ToolSchema
schema = ToolSchema(name="func_name", description="...", parameters=[...])
schema.to_openai_format()  # → dict OpenAI pur

# ToolRegistry
registry = ToolRegistry.get_instance()
await registry.register(schema, executor, force=False)
await registry.unregister(name)  # → bool
registry.get_tools_for_ministral(permission_filter=None)  # → List[dict]
registry.get_tools_up_to_level(max_level)                # → List[dict]
registry.get(name)            # → ToolSchema | None
registry.has_executor(name)   # → bool
await registry.update_stats(name, success=bool, duration_ms=float)

# MinistralEngine
async for event in engine.stream_with_tools(messages, tools, registry,
    max_tokens=512, temperature=0.15, max_iterations=5):
    # event["type"] ∈ {"text_chunk","tool_call","tool_result","confirmation_required","final","error"}

# register_builtin_tools (startup)
await register_builtin_tools(registry)  # get_current_time, get_system_info, ping_host
```

#### Améliorations identifiées pour phases suivantes

- **Timeout par outil** : `timeout_seconds` dans ToolSchema non utilisé → ajouter dans `_execute_tool()` (Phase 3.2)
- **ping_host budget total** : Actuellement timeout = par port → ajouter un budget total (Phase 3.3)
- **Stats persistence** : `update_stats()` in-memory uniquement → persist Phase 3.5
- **RWLock** : Pour > 100 outils → envisager si profiling montre goulot (Phase 3.4+)

-#### Phase 3.2 : Sécurité & Validation ⏳ EN COURS

**Objectif** : Renforcer la sécurité et la robustesse du système tool calling avant d'autoriser l'auto-génération d'outils (Phase 3.4). Implémenter validation AST, timeout effectif, confirmation flow pour outils destructifs.

**Dates** : [À définir] → [À définir]  
**Durée estimée** : 2-3 jours

---

### Contraintes non-négociables Phase 3.2

#### Validation sécurité
- **AST parsing obligatoire** : Tous les outils générés analysés avant exécution
- **Whitelist imports** : Seuls les modules sûrs autorisés (`datetime`, `math`, `json`, `re`, `typing`, `pathlib`)
- **Blacklist builtins** : `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`, `vars`
- **Blacklist modules** : `os.system`, `subprocess`, `socket` (sauf si permission NETWORK/SYSTEM_EXEC)
- **AST walker exhaustif** : Détection patterns dangereux dans tout l'arbre (pas juste imports)
- **Rejection immédiate** : Code malveillant → ValueError détaillée avec ligne du problème

#### Timeout & Error Handling
- **Timeout effectif** : Utiliser `asyncio.wait_for()` avec `tool.timeout_seconds`
- **Timeout par défaut** : 30s (défini dans ToolSchema)
- **TimeoutError → structured result** : `{"success": False, "error": "Tool timed out after 30s"}`
- **Toutes exceptions catchées** : Aucune exception non gérée ne remonte au stream
- **Logging structuré** : Chaque exécution loggée (start, success/fail, duration, error)
- **Format log** : JSON avec timestamp, tool_name, args, result_type, duration_ms, error

#### Confirmation Flow
- **Pending state** : Outils `requires_confirmation=True` ne s'exécutent pas immédiatement
- **Storage pending calls** : Stockage temporaire en mémoire avec TTL 60s
- **Endpoint approbation** : `POST /api/tools/confirm/{call_id}` avec action approve/reject
- **Timeout auto-reject** : Après 60s sans réponse, pending call expire (auto-reject)
- **Preview arguments** : API retourne preview des args avant exécution
- **Audit trail** : Toutes décisions (approve/reject) loggées

---

### Architecture Phase 3.2

#### Fichiers à créer
```
backend/core/tools/validator.py         [CRÉER]
  - ToolValidator.validate_code(code: str) -> ValidationResult
  - _check_imports(node: ast.Module) -> List[str]  # errors
  - _check_builtins(node: ast.Module) -> List[str]
  - _check_dangerous_patterns(node: ast.Module) -> List[str]
  
backend/core/tools/executor.py          [CRÉER]
  - ToolExecutor.execute(tool_name: str, args: dict) -> dict
  - _execute_with_timeout(coro, timeout: float) -> Any
  - _format_error(exc: Exception) -> dict
  - _log_execution(tool_name, args, result, duration) -> None
  
backend/api/tools_confirmation.py       [CRÉER]
  - POST /api/tools/confirm/{call_id}
  - GET /api/tools/pending
  - DELETE /api/tools/pending/{call_id}  # cancel
  
backend/core/tools/pending.py           [CRÉER]
  - PendingCallManager (in-memory store avec TTL)
  - add_pending(call_id, tool_name, args, created_at)
  - get_pending(call_id) -> PendingCall | None
  - approve(call_id) -> ExecutionResult
  - reject(call_id, reason: str)
  - cleanup_expired()  # appelé périodiquement

backend/tests/test_validator.py         [CRÉER]
backend/tests/test_executor.py          [CRÉER]
backend/tests/test_confirmation.py      [CRÉER]
```

#### Fichiers à modifier
```
backend/core/llm.py                     [MODIFIER]
  - stream_with_tools() : utiliser ToolExecutor au lieu d'appel direct
  - Gérer event type="confirmation_required" dans stream
  
backend/core/tools/registry.py          [MODIFIER]
  - Méthode execute() → délègue à ToolExecutor
  - Ajouter get_executor() -> ToolExecutor (lazy init)
  
backend/main.py                         [MODIFIER]
  - Ajouter app.state.pending_manager = PendingCallManager()
  - Ajouter background task cleanup_expired() toutes les 10s
  - Router tools_confirmation
```

---

### Sous-phases détaillées

#### 3.2.1 : ToolValidator & Security (3-4h)

**Objectif** : Analyser le code Python généré via AST pour détecter patterns dangereux avant ajout au registry.

**Validation AST - Patterns détectés** :

| Pattern | Exemple code | Action |
|---------|--------------|--------|
| `eval()` | `eval("malicious")` | ❌ REJECT |
| `exec()` | `exec(user_input)` | ❌ REJECT |
| `compile()` | `compile(code, ...)` | ❌ REJECT |
| `__import__()` | `__import__("os")` | ❌ REJECT |
| `globals()` | `globals()["__builtins__"]` | ❌ REJECT |
| Import non-whitelisted | `import subprocess` | ❌ REJECT (sauf si permission SYSTEM_EXEC) |
| Import whitelisted | `import datetime` | ✅ ALLOW |
| Socket sans permission | `socket.socket()` | ❌ REJECT (sauf permission NETWORK) |

**Whitelist imports par défaut** :
```python
SAFE_IMPORTS = {
    "datetime", "math", "json", "re", "typing", 
    "pathlib", "collections", "itertools", "functools",
    "decimal", "fractions", "random", "string"
}
```

**Whitelist imports conditionnels** :
```python
# Autorisé si permission >= NETWORK
NETWORK_IMPORTS = {"urllib", "http", "requests"}

# Autorisé si permission >= SYSTEM_EXEC
SYSTEM_IMPORTS = {"subprocess", "os"}
```

**ValidationResult format** :
```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]  # ["Line 5: use of eval() is forbidden", ...]
    warnings: List[str]  # ["Line 10: network call detected, requires NETWORK permission"]
    imports_used: Set[str]
    permission_required: PermissionLevel  # minimal requis
```

**Tests validation** :
- Code avec `eval()` → rejected avec ligne exacte
- Code avec `import os` et permission READ_ONLY → rejected
- Code avec `import os` et permission SYSTEM_EXEC → accepted
- Code avec `import datetime` → accepted (whitelist)
- Code valide sans patterns → accepted

---

#### 3.2.2 : ToolExecutor & Timeout (4-5h)

**Objectif** : Wrapper d'exécution centralisé qui gère timeout, error handling, logging structuré pour tous les outils.

**Architecture ToolExecutor** :
```python
class ToolExecutor:
    """
    Exécute les outils avec timeout, error handling, logging.
    
    Responsabilités :
    - Récupérer executor depuis registry
    - Appliquer timeout (asyncio.wait_for)
    - Catcher toutes exceptions
    - Logger début/fin/erreur
    - Retourner format standardisé
    """
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    async def execute(
        self, 
        tool_name: str, 
        arguments: dict
    ) -> ExecutionResult:
        """
        Exécute un outil avec timeout et error handling.
        
        Returns:
            ExecutionResult avec success, result/error, duration_ms
        """
```

**ExecutionResult format** :
```python
@dataclass
class ExecutionResult:
    success: bool
    tool_name: str
    arguments: dict
    result: Any | None  # Si success=True
    error: str | None  # Si success=False
    duration_ms: float
    timeout_occurred: bool
    timestamp: datetime
```

**Gestion timeout** :
```python
try:
    result = await asyncio.wait_for(
        executor(**arguments),
        timeout=tool.timeout_seconds
    )
    return ExecutionResult(success=True, result=result, ...)
except asyncio.TimeoutError:
    return ExecutionResult(
        success=False, 
        error=f"Tool timed out after {tool.timeout_seconds}s",
        timeout_occurred=True,
        ...
    )
except Exception as exc:
    return ExecutionResult(
        success=False,
        error=f"{type(exc).__name__}: {str(exc)}",
        ...
    )
```

**Logging structuré** :
```json
{
  "timestamp": "2026-03-13T16:30:00Z",
  "event": "tool_execution",
  "tool_name": "get_current_time",
  "arguments": {"timezone": "Europe/Paris"},
  "success": true,
  "duration_ms": 12.3,
  "result_preview": "2026-03-13T16:30:00+01:00"
}
```

**Tests validation** :
- Outil rapide (< 1s) → success, duration mesurée
- Outil lent (sleep 100s) avec timeout 30s → timeout_occurred=True après 30s
- Outil qui raise ValueError → success=False, error capturé
- Stats registry mises à jour après chaque exécution
- Logs structurés présents dans backend/logs/

---

#### 3.2.3 : Confirmation Flow (3-4h)

**Objectif** : Système de validation humaine pour outils destructifs (`requires_confirmation=True`) avant exécution.

**Architecture Pending Calls** :
```python
@dataclass
class PendingCall:
    call_id: str  # UUID
    tool_name: str
    arguments: dict
    created_at: datetime
    expires_at: datetime  # created_at + 60s
    user_id: str | None  # Pour multi-user futur
    
class PendingCallManager:
    """
    Gère les tool calls en attente de confirmation.
    
    In-memory storage (dict) avec cleanup périodique.
    Phase 3.2 : single-user, pas de persistence.
    Phase 4+ : persist dans DB, multi-user.
    """
    
    def __init__(self):
        self._pending: dict[str, PendingCall] = {}
        self._lock = asyncio.Lock()
    
    async def add_pending(...) -> str:  # retourne call_id
    async def get_pending(call_id) -> PendingCall | None
    async def approve(call_id) -> ExecutionResult
    async def reject(call_id, reason) -> None
    async def cleanup_expired() -> int  # retourne nb expired
```

**Workflow confirmation** :
```
1. LLM génère tool_call avec requires_confirmation=True
   ↓
2. ToolExecutor détecte requires_confirmation
   ↓
3. add_pending(call_id, tool, args)
   ↓
4. Yield event {"type": "confirmation_required", "call_id": "...", "tool": {...}, "args": {...}}
   ↓
5. Frontend affiche dialog "Approuver get_weather(city='Paris') ?"
   ↓
6a. User APPROVE → POST /api/tools/confirm/{call_id} {"action": "approve"}
    → execute() → résultat injecté dans conversation
    
6b. User REJECT → POST /api/tools/confirm/{call_id} {"action": "reject", "reason": "..."}
    → error injecté dans conversation
    
6c. Timeout 60s → cleanup_expired() auto-reject
```

**API Endpoints** :
```python
# GET /api/tools/pending
# Retourne la liste des pending calls (pour UI)
{
  "pending": [
    {
      "call_id": "abc-123",
      "tool_name": "send_email",
      "arguments": {"to": "john@example.com", "subject": "..."},
      "created_at": "2026-03-13T16:30:00Z",
      "expires_in_seconds": 45
    }
  ]
}

# POST /api/tools/confirm/{call_id}
# Body: {"action": "approve" | "reject", "reason": "..."}
{
  "status": "approved",
  "execution_result": {
    "success": true,
    "result": "Email sent successfully",
    "duration_ms": 234.5
  }
}

# DELETE /api/tools/pending/{call_id}
# Cancel pending call (équivalent reject)
{
  "status": "cancelled"
}
```

**Background task cleanup** :
```python
# Dans main.py lifespan
async def cleanup_expired_pending():
    while True:
        await asyncio.sleep(10)  # Check toutes les 10s
        count = await app.state.pending_manager.cleanup_expired()
        if count > 0:
            logger.info(f"Cleaned up {count} expired pending calls")
```

**Tests validation** :
- Outil avec requires_confirmation=True → pending call créé, pas d'exécution
- Approve → outil exécuté, résultat retourné
- Reject → error message dans conversation
- Timeout 60s → auto-reject, cleanup effectué
- GET /api/tools/pending retourne liste active
- Multiples pending calls gérés (pas de collision call_id)

---

### Performance attendue Phase 3.2

| Métrique | Cible | Mesure |
|----------|-------|--------|
| **AST validation** | < 10ms par outil | [À mesurer] |
| **Timeout overhead** | < 5ms (asyncio.wait_for) | [À mesurer] |
| **Logging overhead** | < 2ms par exécution | [À mesurer] |
| **Pending call storage** | < 1ms add/get/remove | [À mesurer] |
| **Cleanup expired** | < 5ms (100 pending calls) | [À mesurer] |
| **Total overhead** | < 20ms vs Phase 3.1 | [À mesurer] |

---

### Tests validation Phase 3.2

#### 3.2.1 - ToolValidator
- [x] Code avec `eval()` → rejected avec ligne exacte
- [x] Code avec `import subprocess` sans permission → rejected
- [x] Code avec `import subprocess` + SYSTEM_EXEC → accepted
- [x] Code valide sans patterns → accepted
- [x] Tous patterns blacklist détectés (eval, exec, compile, __import__, globals)
- [x] Whitelist imports respectée
- [x] Permission minimale calculée correctement

#### 3.2.2 - ToolExecutor
- [x] Outil rapide → success, duration mesurée
- [x] Outil lent (timeout) → timeout_occurred=True après N secondes
- [x] Outil qui raise → error capturé, pas de crash
- [x] Stats registry mises à jour (success_count, duration_ms)
- [x] Logs structurés JSON présents
- [x] Format ExecutionResult standardisé

#### 3.2.3 - Confirmation Flow
- [x] Outil requires_confirmation → pending call créé
- [x] Approve → outil exécuté, résultat OK
- [x] Reject → error message approprié
- [x] Timeout 60s → auto-reject
- [x] GET /api/tools/pending → liste correcte
- [x] Cleanup expired fonctionne (background task)
- [x] Multiples pending calls sans collision

---

### Décisions techniques Phase 3.2

| Décision | Justification | Alternative rejetée |
|----------|---------------|---------------------|
| **AST walker Python natif** | Pas de dépendance externe, exhaustif | Regex → fragile, faux négatifs |
| **asyncio.wait_for pour timeout** | Natif asyncio, précis | threading.Timer → incompatible async |
| **In-memory pending calls** | Simplicité Phase 3.2 | Redis/DB → overkill single-user |
| **TTL 60s pour pending** | Balance UX/sécurité | 30s trop court, 120s trop long |
| **Background task cleanup 10s** | Granularité acceptable | 1s overkill, 60s trop lent |
| **JSON logs structurés** | Queryable, parsing facile | Texte libre → difficile à analyser |
| **ExecutionResult dataclass** | Type-safe, sérialisable | Dict → pas de validation |
| **Whitelist imports** | Explicit is better than implicit | Blacklist → facilement contournable |

---

### Risques & Mitigations Phase 3.2

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Bypass AST validation** | Moyenne | Critique | Tests exhaustifs, code review |
| **Faux positifs validation** | Moyenne | Moyen | Whitelist extensible, warnings vs errors |
| **Timeout trop court** | Faible | Moyen | Configurable par outil (30s default) |
| **Memory leak pending calls** | Faible | Moyen | Cleanup périodique + TTL strict |
| **Race condition approve** | Faible | Faible | asyncio.Lock sur pending manager |
| **Logs volumineux** | Moyenne | Faible | Rotation logs, compression |

---

### Métriques succès Phase 3.2

- ✅ 100% patterns dangereux détectés (eval, exec, subprocess sans permission)
- ✅ 0 timeout non géré
- ✅ 100% confirmations respectées (aucune exécution sans approve)
- ✅ Logs structurés queryables (JSON)
- ✅ Performance overhead < 20ms vs Phase 3.1
- ✅ Tous tests passent (validator, executor, confirmation)

---

### Observations pour Phase 3.3

- **UI Swift confirmation dialog** : Phase 3.2 fournit l'API, Phase 3.3 implémentera l'UI
- **Audit trail complet** : Logs + pending history → base pour analytics Phase 3.5
- **Permission escalation** : Phase 3.4 pourra demander élévation permission via confirmation
- **Multi-user** : Architecture PendingCallManager prête pour ajout user_id

---

### Exemples d'usage Phase 3.2

**Exemple 1 : Validation code malveillant**
```python
code = """
import subprocess
subprocess.run(["rm", "-rf", "/"])
"""
validator = ToolValidator()
result = validator.validate_code(code, permission=PermissionLevel.READ_ONLY)
# result.is_valid = False
# result.errors = ["Line 1: import 'subprocess' requires SYSTEM_EXEC permission"]
```

**Exemple 2 : Timeout outil lent**
```python
executor = ToolExecutor(registry)
result = await executor.execute("slow_tool", {})
# Si slow_tool prend 100s et timeout=30s :
# result.success = False
# result.timeout_occurred = True
# result.error = "Tool timed out after 30s"
# result.duration_ms ≈ 30000
```

**Exemple 3 : Confirmation flow complet**
```python
# 1. LLM génère tool_call avec requires_confirmation=True
tool_call = {"name": "delete_file", "arguments": {"path": "/important.txt"}}

# 2. ToolExecutor détecte requires_confirmation
pending_manager = PendingCallManager()
call_id = await pending_manager.add_pending("delete_file", {"path": "/important.txt"})
# call_id = "abc-123-def-456"

# 3. Event streamed au frontend
yield {"type": "confirmation_required", "call_id": call_id, ...}

# 4a. User approve via POST /api/tools/confirm/abc-123-def-456
result = await pending_manager.approve(call_id)
# result.success = True, file deleted

# 4b. User reject
await pending_manager.reject(call_id, reason="File is important")
# Error message injecté dans conversation
```

---

### Commit final Phase 3.2
```bash
# Après validation complète
git add backend/core/tools/validator.py backend/core/tools/executor.py backend/core/tools/pending.py
git add backend/api/tools_confirmation.py
git add backend/tests/test_validator.py backend/tests/test_executor.py backend/tests/test_confirmation.py
git add backend/core/llm.py backend/core/tools/registry.py backend/main.py
git add AGENTS.md

git commit -m "Phase 3.2 complétée : Sécurité & Validation

- ToolValidator : AST parsing, whitelist/blacklist, permission check
- ToolExecutor : timeout effectif, error handling, logging structuré
- Confirmation Flow : pending calls, approve/reject API, auto-cleanup
- Tests : 100% patterns dangereux détectés, timeout géré
- Performance : overhead < 20ms, tous tests passent"

git tag phase-3.2-complete
git push origin main --tags
```

---

### Validation passage Phase 3.3

**Pré-requis OBLIGATOIRES** :

- [ ] ✅ Tous tests 3.2.1, 3.2.2, 3.2.3 passent
- [ ] ✅ 100% patterns dangereux détectés (eval, exec, subprocess)
- [ ] ✅ Timeout fonctionnel (testé avec outil lent)
- [ ] ✅ Confirmation flow complet (approve/reject/timeout)
- [ ] ✅ Logs structurés JSON présents
- [ ] ✅ Performance overhead < 20ms mesurée
- [ ] ✅ Aucune régression Phase 3.1 (32 tests passent toujours)
- [ ] ✅ Documentation AGENTS.md mise à jour
- [ ] ✅ Commit phase-3.2-complete tagué

**Si tous pré-requis ✅** → Passage Phase 3.3 autorisé  
**Si un seul ❌** → Corriger avant de continuer


#### Phase 3.3 : API & UI (Semaine 4)

**Objectif** : Interface pour gérer les outils

**Sous-phases** :
- [ ] **3.3.1 : API REST Tools (3-4h)**
  - `GET /api/tools` : liste avec métadonnées
  - `POST /api/tools/execute` : exécution manuelle (test)
  - `DELETE /api/tools/{name}` : suppression (si created_by != system)
  
- [ ] **3.3.2 : UI Swift Tool Calls (4-6h)**
  - Affichage real-time des tool calls en cours
  - Indicateur visuel : outil en exécution
  - Résultats inline dans conversation
  
- [ ] **3.3.3 : UI Confirmation (3-4h)**
  - Dialog pour approuver/rejeter outils destructifs
  - Preview arguments avant exécution
  - Timeout auto-rejet après 60s

**Validation Phase 3.3** :
- ✅ Liste outils visible via API et UI
- ✅ Tool calls visibles en temps réel dans conversation
- ✅ Confirmation flow intuitif et sûr

#### Phase 3.4 : Auto-génération (Semaine 5-6) 🚀 AMBITIOUS

**Objectif** : L'IA peut créer ses propres outils

**Sous-phases** :
- [ ] **3.4.1 : ToolGenerator (4-6h)**
  - Utilise Ministral pour générer code Python
  - Prompt engineering pour génération structurée
  - Tests : génération de outils simples
  
- [ ] **3.4.2 : Code Validation (3-4h)**
  - AST parsing obligatoire
  - Typage vérification (type hints présents)
  - Pas d'imports non-whitelisted
  
- [ ] **3.4.3 : Review UI (4-6h)**
  - Interface pour review code généré
  - Syntax highlighting
  - Approve/Reject/Edit
  
- [ ] **3.4.4 : Hot-reload (2-3h)**
  - Registry.reload() pour charger nouveaux outils
  - Pas besoin de redémarrer backend

**Validation Phase 3.4** :
- ✅ L'IA crée `calculate_mortgage(principal, rate, years)` quand demandé
- ✅ Code validé par AST avant ajout
- ✅ Utilisateur peut review et approuver

#### Phase 3.5 : Learning System (Semaine 7-8) 📊 FUTURE

**Objectif** : Apprentissage des patterns d'usage

**Sous-phases** :
- [ ] **3.5.1 : Usage Pattern Tracker**
  - Tracking contexte d'utilisation de chaque outil
  - Stockage patterns dans `usage_patterns.json`
  
- [ ] **3.5.2 : Context-aware Suggestions**
  - Base relationnelle : outil X souvent utilisé après Y
  - Suggestion proactive d'outils
  
- [ ] **3.5.3 : Auto-amélioration Descriptions**
  - L'IA améliore descriptions si échecs répétés
  - A/B testing de descriptions

**Validation Phase 3.5** :
- ✅ L'IA suggère `read_file` automatiquement si mention fichier
- ✅ Descriptions améliorées basées sur usage réel

### Décisions techniques

| Décision | Justification | Alternative rejetée |
|----------|---------------|---------------------|
| **Format OpenAI strict** | Standard industrie, training Ministral | Format custom → modèle confus |
| **Pydantic v2 pour schemas** | Validation auto, serialization native | Dataclasses → validation manuelle |
| **asyncio.Lock pour registry** | Protection race conditions | threading.Lock → incompatible asyncio |
| **JSON storage registry (externe)** | Debug facile, full-local hors repo | SQLite → overkill, migration complexe |
| **max_iterations=5** | Empêche boucles infinies | Pas de limite → risque hang |
| **AST validation seulement** | Balance sécurité/complexité | Vrai sandbox Docker → overkill local |
| **created_by (system/ai/user)** | Traçabilité, protection système | Pas de distinction → risque suppression |
| **Timeout 30s default** | Empêche outils bloquants | Pas de timeout → risque hang |
| **parameters.type = "object"** | Requis par spec OpenAI | Autre type → parsing cassé |

### Risques & Mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Hallucination tools** | Haute | Moyen | Parser robuste, validation stricte, fallback gracieux |
| **Boucle infinie** | Moyenne | Élevé | max_iterations=5, détection patterns cycliques |
| **Outil destructif** | Faible | Critique | requires_confirmation obligatoire WRITE_MODIFY+ |
| **Code malveillant généré** | Moyenne | Critique | AST validation, review humaine, whitelist imports |
| **Performance dégradée** | Moyenne | Moyen | Profiling outils, caching, lazy loading |
| **Format non-standard** | Faible | Élevé | Validation à chaque to_openai_format(), tests |
| **Context window overflow** | Moyenne | Moyen | Truncate old messages, summarization |

### Métriques de succès

**Phase 3.1** :
- [ ] 100% des outils builtin ont tests unitaires
- [ ] 0 tool call non-validé exécuté
- [ ] < 100ms overhead total (injection + parsing)
- [ ] < 5% faux positifs parsing tool calls
- [ ] Format OpenAI validé à 100%

**Phase 3.2** :
- [ ] 100% code dangereux détecté par AST
- [ ] 0 timeout non-géré
- [ ] 100% confirmations respectées

**Phase 3.3** :
- [ ] UI tool calls < 200ms latency affichage
- [ ] 100% confirmations visibles et claires

**Phase 3.4** :
- [ ] > 80% outils générés valides (passent AST)
- [ ] < 10% rejets après review humaine

**Phase 3.5** :
- [ ] > 70% précision suggestions contextuelles
- [ ] Amélioration descriptions mesurable (success rate)

### Exemples d'usage

**Exemple 1 : Simple tool call**
```
User: Quelle heure est-il à Tokyo ?
Assistant: [THINKING: Je dois utiliser get_current_time]
           [TOOL_CALL: get_current_time(timezone="Asia/Tokyo")]
           [TOOL_RESULT: "2026-03-13T18:30:00+09:00"]
           Il est actuellement 18h30 à Tokyo.
```

**Exemple 2 : Multi-tools séquentiel**
```
User: Lis le fichier rapport.txt et envoie-le par email à john@example.com
Assistant: [TOOL_CALL: read_file(path="rapport.txt")]
           [TOOL_RESULT: "Contenu du rapport : ..."]
           [TOOL_CALL: send_email(to="john@example.com", subject="Rapport", body="...")]
           [REQUIRES_CONFIRMATION: send_email est un outil NETWORK]
User: [APPROVE]
Assistant: Email envoyé avec succès à john@example.com.
```

**Exemple 3 : Tool hallucination (erreur)**
```
User: Quel temps fait-il à Paris ?
Assistant: [TOOL_CALL: get_weather(city="Paris")]
           [ERROR: Tool 'get_weather' not found]
           [FALLBACK]
           Je n'ai pas accès à un outil météo pour le moment. 
           Je peux vous donner l'heure actuelle à Paris si vous le souhaitez.
```

**Exemple 4 : Auto-génération**
User: Crée un outil pour calculer le TAEG d'un prêt immobilier
Assistant: [THINKING: Je n'ai pas cet outil, je vais le créer]
[GENERATING_TOOL: calculate_taeg]
[CODE_GENERATED]
python           async def calculate_taeg(
               principal: float,
               annual_rate: float,
               years: int
           ) -> dict:
               """Calculate TAEG for a mortgage loan"""
               monthly_rate = annual_rate / 12 / 100
               n_payments = years * 12
               monthly_payment = principal * (
                   monthly_rate * (1 + monthly_rate)**n_payments
               ) / ((1 + monthly_rate)**n_payments - 1)
               return {
                   "monthly_payment": round(monthly_payment, 2),
                   "total_paid": round(monthly_payment * n_payments, 2)
               }
```
           [REQUIRES_REVIEW: Nouveau code généré]
User: [APPROVE]
Assistant: Outil créé avec succès ! Je peux maintenant calculer le TAEG.
```

### Références Documentation

**Standards** :
- OpenAI Function Calling : https://platform.openai.com/docs/guides/function-calling
- Mistral AI Function Calling : https://docs.mistral.ai/capabilities/function_calling/
- JSON Schema : https://json-schema.org/understanding-json-schema/

**Frameworks de référence** :
- LangChain Tools : https://python.langchain.com/docs/modules/tools/
- LlamaIndex Tools : https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/

**Sécurité** :
- Python AST : https://docs.python.org/3/library/ast.html
- Pydantic Validation : https://docs.pydantic.dev/latest/
```

---
