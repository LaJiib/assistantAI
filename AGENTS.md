# 🤖 AGENTS.md : Projet IRIS v2 - Assistant Consulting Autonome

## 📌 Vision & Objectifs
Transformer l'actuel "AssistantAI" en un agent de consulting haute performance nommé **Iris**. 
- **100% Local** : Aucune donnée ne quitte le Mac Mini M4 Pro (48 Go RAM).
- **Auto-évolutif** : Capacité à générer, tester et intégrer ses propres outils (Skills).
- **Mémoire Stratifiée** : Gestion intelligente du contexte (Habitudes, Métier, Clients).

---

## 🛠 Stack Technique Cible

### 1. Inférence & Intelligence
- **Moteur LLM** : `mlx-vlm` (Optimisation native Apple Silicon — vision + language).
- **Modèle** : `google/gemma-4-26b-a4b` (Quantisation **Q8_0**).
- **Orchestration** : `Pydantic AI` (Pour la rigueur du Tool Calling et de la logique agent).
- **Streaming** : Support complet des Server-Sent Events (SSE) via FastAPI.

### 2. Stockage & Mémoire (Graph-Lite)
- **Base de Données** : `LanceDB` (Vectorielle, embedded, haute performance sur SSD).
- **Embeddings** : Modèle `nomic-embed-text-v1.5` tournant sur **GPU/MPS**.
- **Segmentation** :
    - `table_user_core` : Profil utilisateur et habitudes.
    - `table_pro_knowledge` : Expertise consulting et méthodologies.
    - `tables_clients_{id}` : Silos étanches par client/projet.



### 3. Sandbox & Auto-Évolution
- **Technologie** : `WebAssembly (Wasm)` via le moteur **Wasmtime**.
- **Isolation** : Utilisation de **WASI** pour restreindre l'accès au système de fichiers.
- **Workspace** : Accès exclusif au dossier `~/Iris_Workspace/`.
- **Logic de Skills** :
    - Iris écrit du code Python -> Compilation/Exécution en Sandbox Wasm.
    - Validation du résultat -> Enregistrement dans `/backend/skills/`.
    - Chargement dynamique dans le registre de l'agent.



---

## 🔄 Roadmap d'Implémentation (Directives Claude Code)

### Étape 1 : Refonte du Core Inférence
- Réécrire `backend/core/llm.py` pour intégrer `mlx-lm`.
- Mettre en place la classe `IrisAgent` avec Pydantic AI.
- Configurer la persistance du **KV Cache** pour minimiser la latence.

### Étape 2 : Système de Mémoire Vectorielle
- Créer `backend/storage/vector_store.py` (Initialisation de LanceDB).
- Développer la logique de "Context Enrichment" : recherche sémantique multi-niveaux avant chaque réponse.
- Créer le worker de **Consolidation** : extraction de faits en post-traitement de session.

### Étape 3 : Sandbox et Skills Dynamiques
- Implémenter le module `backend/core/sandbox.py` (Wasmtime interface).
- Créer l'outil `code_interpreter` pour l'analyse de fichiers (Excel, CSV).
- Mettre en place le système de sauvegarde et de rechargement des nouveaux outils créés par l'IA.



### Étape 4 : Refactorisation de l'API & UI
- Adapter `api/messages.py` pour les flux de streaming.
- Dans l'app Swift (Mac), mettre à jour `ChatAPI.swift` pour traiter le flux asynchrone et afficher les balises de raisonnement `<thought>`.

---

## 🛡️ Sécurité & Audit
- **Souveraineté** : Interdiction d'utiliser des bibliothèques dépendantes du Cloud (OpenAI, LangChain Cloud, etc.).
- **Audit des Skills** : Tout nouveau skill généré doit être loggé en clair pour revue humaine.
- **Isolation** : La sandbox Wasm ne doit avoir **aucun accès réseau**.

---

## 📝 Note pour Claude Code
> "Analyse en priorité le fichier `backend/main.py` et la structure des modèles actuels. Propose une structure de fichiers modulaire qui sépare l'inférence (MLX), la logique agent (Pydantic AI), et la gestion mémoire (LanceDB). Ne supprime pas l'historique JSON existant, il servira d'archive. Réfléchis si tu dois fzire des choix étape par étape et/ou décris les choix possible pour que l'utilisateur fasse un choix"