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

## PHASE 2 : À définir

**Candidats** :
- Dual mode Swift/Python (Swift UI → Python backend)
- RAG local (ChromaDB, zero telemetry)
- Mémoires persistantes

**Prérequis Phase 2** :
- Phase 1 validée ✅
