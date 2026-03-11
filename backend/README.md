# # Assistant IA - Backend Python

Backend Python pour agent autonome cognitif basé sur Ministral 3 14B avec MLX.

## Architecture
```
backend/
├── main.py              # FastAPI entrypoint
├── requirements.txt     # Dépendances Python
├── .env.example        # Template configuration
├── core/
│   ├── __init__.py
│   └── llm.py          # Wrapper MLX-LM
├── data/               # Données locales (gitignored)
│   └── chroma/         # ChromaDB vectorstore
├── logs/               # Logs application (gitignored)
└── tests/
    └── test_basic.py   # Tests manuels
```

## Prérequis

- Python 3.11+
- macOS avec Apple Silicon (M1/M2/M3/M4)
- Ministral 3 14B modèle téléchargé localement
- 48GB RAM recommandé

## Installation

### 1. Setup environnement
```bash
cd ~/Developer/AssistantIA/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
```bash
cp .env.example .env
# Éditer .env avec votre configuration
```

Paramètres `.env` :
```
MODEL_PATH=/path/to/ministral-3-14b-instruct
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=info
```

⚠️ **SÉCURITÉ** : 
- `HOST=127.0.0.1` uniquement (jamais 0.0.0.0 en local)
- Aucune donnée ne sort de la machine
- Logs en local uniquement

### 3. Lancer backend
```bash
python main.py
```

Output attendu :
```
🚀 Starting backend...
📦 Loading model from /path/to/model
✅ Model loaded
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## API Endpoints

### GET /health
Vérifier état backend et modèle.

**Response :**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "ministral-3-14b-instruct"
}
```

### POST /chat
Générer réponse avec Ministral.

**Request :**
```json
{
  "prompt": "Explique FastAPI en une phrase",
  "max_tokens": 100
}
```

**Response :**
```json
{
  "response": "FastAPI est un framework web Python moderne..."
}
```

## Tests

### Test manuel curl
```bash
# Health check
curl http://127.0.0.1:8000/health

# Chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Bonjour", "max_tokens": 50}'
```

### Tests automatisés
```bash
python tests/test_basic.py
```

## Performance

**Attendu sur M4 Pro 48GB :**
- Latency première requête : ~200ms (model déjà chargé)
- Throughput : 32-35 tokens/sec
- VRAM usage : ~9GB (Q5 quantization)

## Troubleshooting

### Backend ne démarre pas

**Symptôme :** `ModuleNotFoundError: mlx_lm`
**Solution :** 
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Symptôme :** `MODEL_PATH not set`
**Solution :** Vérifier `.env` existe avec `MODEL_PATH` correct

### Performance lente

**Symptôme :** < 20 tok/s
**Vérifications :**
1. Utiliser Q5 quantization (pas FP16)
2. Vérifier GPU utilisé : `Activity Monitor → GPU`
3. Fermer autres apps GPU-intensive

### Erreur VRAM

**Symptôme :** `Out of memory`
**Solution :** 
- Utiliser Q4 au lieu de Q5
- Fermer autres apps
- Vérifier `max_tokens` pas trop élevé

## Architecture technique

### Flux requête
```
Client HTTP
    ↓
FastAPI (/chat endpoint)
    ↓
MinistralEngine.generate()
    ↓
MLX-LM (C++ backend)
    ↓
Metal GPU
    ↓
Tokens stream back
```

### Choix techniques

**Pourquoi FastAPI ?**
- Async natif (performance)
- Auto-documentation (Swagger)
- Validation Pydantic (type safety)

**Pourquoi MLX-LM ?**
- Optimisé Apple Silicon
- Performance native Metal
- Compatible Ministral 3 14B

**Pourquoi Ministral 3 14B ?**
- Tool calling natif
- Reasoning step-by-step
- Sweet spot perf/qualité (14B)

## Sécurité

### Contraintes respectées

✅ **Full local** : Bind localhost uniquement  
✅ **Zero telemetry** : Pas d'appels externes  
✅ **Logs locaux** : backend/logs/ uniquement  
✅ **Pas de CORS** : Inutile en localhost  

### Données sensibles

- `.env` : Jamais commit (gitignored)
- `data/` : Données utilisateur (gitignored)
- `logs/` : Logs application (gitignored)

## Roadmap

- [x] Phase 1 : Backend minimal + MLX
- [ ] Phase 2 : Frontend contrôle
- [ ] Phase 3 : RAG ChromaDB
- [ ] Phase 4 : Tool calling
- [ ] Phase 5 : Mémoires
- [ ] Phase 6 : Planning autonome
- [ ] Phase 7 : Meta-prompting
- [ ] Phase 8 : Initiatives

Voir `AGENTS.md` pour détails.

## Support

Problème ? Consulter :
1. Ce README (troubleshooting)
2. `AGENTS.md` (architecture globale)
3. Logs : `backend/logs/app.log`
