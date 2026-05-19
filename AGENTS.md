# 🤖 AGENTS.md — Iris Lab
> Source de vérité pour Claude Code et tout contributeur.
> Mis à jour après chaque étape structurelle validée.

---

## 📌 Contraintes produit — non négociables

- **100 % local** : zéro donnée hors de la machine. Zéro appel réseau
  externe sauf tools explicitement activés.
- **Zéro télémétrie** : aucun tracking, aucun reporting tiers.
- **Évolutivité agent** : chaque agent est un microservice A2A indépendant.
  Ajouter ou remplacer un agent ne touche pas aux autres.
- **Human-in-the-loop** : certaines actions nécessitent une validation
  explicite avant exécution. Liste non exhaustive — s'enrichira au fil
  des étapes. Actions identifiées à ce jour :
  - Envoi d'un e-mail sortant (John)
  - Suppression d'e-mails (John)
  - Merge d'un patch sur l'infrastructure réelle (Antoine)

---

## 🛠 Stack technique

| Couche                 | Technologie                        | Rôle                                                   |
|------------------------|------------------------------------|--------------------------------------------------------|
| Hardware               | Mac Mini M4 Pro (48 Go)            | Hôte unique                                            |
| Inférence              | oMLX                               | Serveur LLM local, API OpenAI-compatible               |
| Modèle actuel          | Qwen3.6 (MoE)                      | Modèle général                                         |
| Logique agent          | LangGraph (Python)                 | Moteur interne de chaque agent superviseur             |
| Infrastructure         | Agent Stack (BeeAI / Linux Found.) | Expose les graphes LangGraph comme services A2A        |
| Protocole inter-agents | A2A                                | Communication standardisée entre agents                |
| Protocole UI           | AG-UI                              | Communication standardisée backend → frontend          |
| Frontend               | À définir (CopilotKit ou custom)   | Implémentation cliente du protocole AG-UI              |
| Mémoire long terme     | PostgreSQL (OrbStack)              | Checkpointing LangGraph + mémoire sémantique par agent |
| Accès distant          | Tailscale                          | Tunnel chiffré iPhone → Mac Mini                       |

> **Note AG-UI / Frontend** : AG-UI est un protocole, pas un framework UI.
> CopilotKit est une implémentation cliente possible parmi d'autres.
> Le choix du client frontend est indépendant et peut évoluer sans
> impact sur le backend.

> **Note A2A + AG-UI** : ces deux protocoles sont des décorateurs
> d'infrastructure appliqués par Agent Stack. La logique interne des
> agents (LangGraph) n'en a pas connaissance. Tous les superviseurs
> sont exposés avec A2A + AG-UI par standard Iris Lab.

---

## 🧠 Mémoire & Persistence

Chaque agent dispose d'une mémoire **strictement cloisonnée** dans PostgreSQL.

### Principe d'isolation
Un schéma PostgreSQL dédié par agent (ex: `schema iris`, `schema gatien`).
Aucun agent n'accède au schéma d'un autre.

### Couches de mémoire (par agent)
- **Checkpointing LangGraph** (`langgraph-checkpoint-postgres`) : état des
  graphes, reprise après interruption, historique des runs.
- **Mémoire sémantique long terme** : pgvector dans le schéma de l'agent —
  skills, connaissances accumulées, contexte de travail persistant.
- **Skills** : capacités apprises ou configurées, stockées et versionnées
  par agent.

> Les sous-agents (exécutants) n'ont pas de mémoire long terme propre.
> Ils opèrent dans le contexte du run courant uniquement.

---

## 🏢 Organigramme Iris Lab

### Principe d'architecture commun à tous les superviseurs

- **Exposition** : service A2A + AG-UI via Agent Stack (standard universel)
- **Mémoire** : schéma PostgreSQL dédié, inaccessible aux autres agents
- **Délégation** : sous-agents asynchrones pour les tâches d'exécution
- **Sous-agents** : détails d'implémentation internes, non adressables
  depuis le mesh A2A
- **Déclenchement** : requête A2A entrante, et/ou webhook selon l'agent

---

### Iris — CEO (Front Office)
- **Rôle** : point d'entrée unique. Reçoit les demandes, planifie,
  route vers les agents spécialisés, restitue le résultat.
- **Moteur** : LangGraph (planning + routing)
- **Mémoire** : schéma `iris`
- **Sous-agents async** : délègue aux autres superviseurs via A2A
- **Outils** : —
- **Graph ID** : `iris`

### Antoine — CTO (Ingénierie)
- **Rôle** : maintenance et évolution de l'infrastructure.
  Reçoit des tickets, écrit du code, teste en sandbox, soumet des patchs.
- **Moteur** : LangGraph (Plan → Execute → Verify)
- **Mémoire** : schéma `antoine`
- **Sous-agents async** : génération, test, validation
- **Outils** : lecture/écriture projet, Bash dans Docker (sandbox),
  oMLX avec modèle spécialisé code
- **Contraintes HITL** : merge sur infrastructure réelle
- **Graph ID** : `antoine`

### John — Secrétaire Général (Admin & Opérationnel)
- **Rôle** : gestion agenda et flux de communication. Tri e-mails,
  alertes urgences, planification RDV. Travaille aussi en autonomie
  sur déclenchement webhook.
- **Moteur** : LangGraph
- **Mémoire** : schéma `john`
- **Sous-agents async** : lecture/écriture calendrier, parsing e-mails,
  classement
- **Outils** : IMAP / OAuth (e-mail), CalDAV / Google Calendar
- **Déclenchement** : requête A2A entrante ou webhook mail entrant
- **Contraintes HITL** : envoi e-mail sortant, suppression d'e-mails
  — liste à compléter
- **Graph ID** : `john`

### Sophie — CIO (Mémoire & Connaissances)
- **Rôle** : écoute passive des échanges validés, enrichissement base de
  connaissances, retrieval sur demande.
- **Moteur** : LangGraph (RAG supervisé)
- **Mémoire** : schéma `sophie` (pgvector)
- **Sous-agents async** : indexation, retrieval, résumé
- **Outils** : pgvector (schéma sophie), modèle d'embedding local
- **Graph ID** : `sophie`

### Gatien — Deep Researcher
- **Rôle** : recherche web approfondie. Service A2A autonome,
  accessible par tous les agents du mesh.
- **Moteur** : LangGraph (superviseur + workers chercheurs) — déjà implémenté
- **Mémoire** : schéma `gatien`
- **Sous-agents async** : workers de recherche parallèles (déjà en place)
- **Outils** : SearXNG (recherche web)
- **Note** : `fetch_webpage` désactivé temporairement
- **Graph ID** : `deep_research`

---

## 🗺 Feuille de route

### ✅ Étape 0 — LangGraph ↔ oMLX
Validée. Qwen3.6 (MoE) répond correctement depuis LangGraph.

### 🔄 Étape 1 — Gatien comme service A2A + AG-UI
Envelopper le graphe `deep_research` dans Agent Stack.
**Critère de validation** : appel A2A depuis un script Python retourne
une synthèse cohérente.

### ⏳ Étape 2 — Iris comme service A2A + AG-UI + appel Gatien
Iris exposée via Agent Stack. Iris appelle Gatien via A2A.
**Critère de validation** : flux Iris → A2A → Gatien → réponse à Iris.

### ⏳ Étape 3 — Frontend AG-UI
Connecter un client AG-UI à Iris.
Choix du client (CopilotKit ou autre) à confirmer à cette étape.

### ⏳ Étape 4 — Mémoire PostgreSQL
Intégration `langgraph-checkpoint-postgres` + mémoire sémantique,
schéma par agent.

### ⏳ Étapes suivantes (non scopées)
Antoine, John, Sophie — dans cet ordre probable.

---

## 📝 Méthode de travail

- **Une étape à la fois.** On valide avant de passer à la suivante.
- **Progressivité.** On n'implémente pas ce dont on n'a pas encore besoin.
- **Documentation d'abord.** Ce fichier est mis à jour avant ou
  immédiatement après toute décision structurelle.
- **Pas de refactoring opportuniste.** Rien hors périmètre de l'étape.
- **Arrêt sur ambiguïté.** On s'arrête et on demande plutôt qu'inventer.
- **Standards en priorité.** Patterns Agent Stack / LangGraph officiels
  avant tout code custom.

### Références
- Agent Stack : https://agentstack.beeai.dev
- oMLX : submodule `omlx/`

### Accès documentation LangGraph
```
claude mcp add --transport http docs-langchain --scope user \
  https://docs.langchain.com/mcp
```