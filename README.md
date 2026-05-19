# Iris Lab

Assistant personnel 100 % local tournant sur Mac Mini M4 Pro.
Architecture multi-agents découplée, zéro télémétrie, zéro donnée hors machine.

**Repo** : https://github.com/LaJiib/assistantAI

## Vue d'ensemble

Iris Lab est un ensemble d'agents spécialisés qui collaborent via le protocole A2A
et s'exposent à une interface mobile via le protocole AG-UI.

| Agent   | Rôle                   | Statut      |
|---------|------------------------|-------------|
| Iris    | CEO — Front Office     | En cours    |
| Antoine | CTO — Ingénierie       | Non démarré |
| John    | Secrétaire Général     | Non démarré |
| Sophie  | CIO — Mémoire          | Non démarré |
| Gatien  | Deep Researcher        | En cours    |

> Pour l'architecture détaillée, les décisions techniques et la feuille
> de route : voir [AGENTS.md](./AGENTS.md).

## Stack

- **Inférence** : oMLX (Qwen3.6 MoE) — API OpenAI-compatible, local
- **Logique agent** : LangGraph (Python)
- **Infrastructure** : Agent Stack (BeeAI / Linux Foundation) — exposition A2A + AG-UI
- **Mémoire** : PostgreSQL (OrbStack) — schéma dédié par agent
- **Accès distant** : Tailscale

## Structure du repo

```
/
├── AGENTS.md              # Source de vérité — architecture & feuille de route
├── omlx/                  # Submodule — serveur LLM local
└── backend/
    ├── Iris/              # Agent Iris (superviseur front office)
    ├── deep_research/     # Agent Gatien (deep researcher)
    ├── coding/            # Agent Antoine (CTO) — non démarré
    ├── utils/             # Outils partagés (web_search, etc.)
    ├── langgraph.json     # Config de déploiement LangGraph
    └── requirements.txt
```

## Prérequis

- Mac Mini M4 Pro (ou Apple Silicon équivalent)
- oMLX lancé sur `http://127.0.0.1:8000`
- SearXNG lancé sur `http://localhost:8080`
- PostgreSQL accessible via OrbStack
- Python 3.12+, `uv` recommandé

## Lancement

```bash
cd backend
cp .env.example .env  # remplir les variables
uv run langgraph dev
```

## Accès distant

L'interface AG-UI est accessible depuis un iPhone via Tailscale.
Aucun port n'est ouvert sur l'internet public.