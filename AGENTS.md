# Agent Architecture - Changelog & Roadmap

## Vision Long Terme
Agent autonome cognitif avec mémoires, planning, meta-prompting, initiatives.

## État Actuel
- [x] App Swift fonctionnelle avec MLX
- [ ] Backend Python
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
- Performance >= 32 tok/s
- App fonctionnelle à chaque commit

## Modifications Phase en cours

### Phase 1 : Backend minimal
**Objectif** : Backend FastAPI + MLX opérationnel
**Status** : 🟡 En cours

**À NE PAS MODIFIER** :
- [ ] Tout le code Swift existant
- [ ] MLXService.swift
- [ ] ConversationManager.swift
- [ ] Structure UI

**Modifications prévues** :
- [ ] Créer backend/ folder
- [ ] FastAPI entrypoint
- [ ] Wrapper MLX-LM
- [ ] Endpoint /health
- [ ] Endpoint /chat basique

**Tests requis avant validation** :
- [ ] Backend démarre sans erreur
- [ ] /health retourne 200 OK
- [ ] /chat génère réponse cohérente Ministral
- [ ] Performance >= 32 tok/s
- [ ] App Swift fonctionne toujours

**Modifications effectuées** :
(vide pour l'instant)

---

## Historique décisions

### 2026-03-10 : Choix architecture
- **Décision** : Python backend + Swift frontend
- **Raison** : Vélocité dev, écosystème IA mature
- **Alternative écartée** : Swift pur (trop lent à développer)

### 2026-03-10 : Choix modèle
- **Décision** : Ministral 3 14B Instruct Q5
- **Raison** : Tool calling natif, perf/qualité optimal M4
- **Alternative écartée** : Mistral Large (trop lourd, gain marginal)
```

---

## PHASE 1 : Backend minimal Python

### **Objectif**
Créer backend Python qui charge Ministral 3 14B et peut générer texte via endpoint HTTP. **Cohabitation** avec Swift actuel, aucune modification Swift.

### **Contraintes spécifiques**
- Backend écoute **uniquement localhost** (127.0.0.1)
- Modèle sur chemin configurable (pas hard-codé)
- Log startup clair (permet debug)
- Pas de CORS (inutile pour local)
- Pas de auth (phase ultérieure)
- Pas de rate limiting (single user local)

### **Structure backend à créer**
```
backend/
├── main.py                 # Entrypoint FastAPI
├── requirements.txt        # Dépendances STRICT minimum
├── .env.example           # Template config
├── core/
│   ├── __init__.py
│   └── llm.py             # Wrapper MLX-LM minimal
├── tests/
│   └── test_basic.py      # Tests manuels curl
└── README.md              # Setup instructions
```

### **Instructions pour Claude Code**
```
OBJECTIF : Développer backend Python minimal avec FastAPI et MLX-LM

CONTRAINTES :
- Full local (bind 127.0.0.1 uniquement)
- Zero telemetry (chromadb anonymized_telemetry=False si utilisé)
- Performance >= 32 tok/s sur M4 Pro
- Code minimal qui marche (pas de sur-ingénierie)
- Logs explicites (startup, shutdown, erreurs)

SPECS FONCTIONNELLES :

1. Structure projet :
   - Créer backend/ dans ~/Developer/AssistantIA/
   - requirements.txt avec UNIQUEMENT : mlx-lm, fastapi, uvicorn, python-dotenv, pydantic
   - .env.example avec MODEL_PATH, HOST, PORT, LOG_LEVEL

2. FastAPI app (main.py) :
   - Endpoint GET /health qui retourne status, model_loaded, model_name
   - Endpoint POST /chat qui prend {prompt, max_tokens} et retourne {response}
   - Startup : charge modèle MLX depuis MODEL_PATH
   - Shutdown : libère GPU proprement (MLX.GPU.clearCache())

3. Wrapper MLX (core/llm.py) :
   - Classe MinistralEngine avec load() et generate()
   - Format prompt Ministral natif : [INST] {prompt} [/INST]
   - Params par défaut : temp=0.3, max_tokens=512
   - Pas de streaming pour l'instant (phase ultérieure)

4. Tests (tests/test_basic.py) :
   - Script Python qui fait requests.get(/health) et .post(/chat)
   - Valide status codes, structure réponse
   - Mesure tokens/sec approximatif

FICHIERS À NE PAS CRÉER/MODIFIER :
- Aucun fichier Swift
- Aucun fichier dans AssistantIA/ folder

OUTPUT ATTENDU :
- Backend qui démarre avec "python backend/main.py"
- Répond sur http://127.0.0.1:8000
- Génère texte cohérent avec Ministral 3 14B
- Performance mesurable ~32+ tok/s
