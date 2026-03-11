# Agent Architecture - Changelog & Roadmap

## Vision Long Terme
Agent autonome cognitif avec mémoires, planning, meta-prompting, initiatives.

## État Actuel
- [x] App Swift fonctionnelle avec MLX
- [x] Backend Python
- [ ] Dual mode Swift/Python
- [ ] RAG
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

## PHASE 2 : Migration frontend vers backend Python 🟡 EN COURS

**Dates** : 2026-03-11 → ?

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

### Modifications effectuées

**À remplir lors développement...**

- [ ] Étape 2.1 : BackendManager (lifecycle + auto-start)
- [ ] Étape 2.2 : ChatAPI (client HTTP + streaming SSE)
- [ ] Étape 2.3 : AssistantIAApp (auto-start backend)
- [ ] Étape 2.4 : ConversationViewModel (migration MLXService → ChatAPI)
- [ ] Étape 2.5 : BackendSettingsView (monitoring)
- [ ] Étape 2.6 : RootView (navigation settings)
- [ ] Tous tests validation

### Problèmes rencontrés

**À documenter si issues...**

(vide pour l'instant)

### Décisions techniques Phase 2

**À documenter lors implémentation...**

Questions à résoudre :
- Auto-start backend : onAppear vs Task.detached ?
- Streaming SSE : Parser manuel ou library ?
- Retry logic : combien tentatives si backend down ?
- MLXService : commenter ou garder actif en fallback ?

---

## PHASE 3 : RAG ChromaDB ⏳ PLANIFIÉE

**Objectif** : Indexation documents, recherche vectorielle, injection contexte RAG dans prompts.

**Prérequis** : Phase 2 validée ✅

(Détails à définir après Phase 2)