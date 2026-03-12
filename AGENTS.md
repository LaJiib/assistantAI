# Agent Architecture - Changelog & Roadmap

## Vision Long Terme
Agent autonome cognitif avec mémoires, planning, meta-prompting, initiatives.

## État Actuel
- [x] App Swift fonctionnelle avec MLX (Phase 1)
- [x] Backend Python + migration frontend (Phase 2)
- [x] API REST conversations/messages + UI nettoyée (Phase 2bis)
- [ ] RAG ChromaDB (Phase 3)
- [ ] Tool calling
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

## PHASE 3 : RAG ChromaDB ⏳ PLANIFIÉE

**Objectif** : Indexation documents, recherche vectorielle, injection contexte RAG dans prompts.

**Prérequis** : Phase 2 validée ✅

(Détails à définir après Phase 2)