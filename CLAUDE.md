# 🤖 AGENTS.md — Iris v3
> Source de vérité pour Claude Code et tout contributeur.
> Ce fichier est mis à jour après chaque étape structurelle validée.

---

## 📌 Contraintes produit — non négociables

Ces exigences définissent ce qu'est Iris. Les choix techniques en découlent.

- **100 % local** : aucune donnée ne quitte la machine. Zéro appel réseau externe sauf tools explicitement activés par l'utilisateur.
- **Zéro télémétrie** : aucun tracking, aucun reporting vers un service tiers.
- **Évolutivité agent** : l'architecture doit permettre d'ajouter des capacités (tools, graphs, sous-graphs, boucles) sans remettre en cause les fondations.
- **Reproductibilité** : l'état du projet doit être cohérent et fonctionnel à chaque étape validée.

---

## 🛠 Stack technique — état actuel

| Couche | Technologie |
|--------|-------------|
| Inférence | **omlx** (submodule `omlx/`) — API Anthropic-compatible, localhost |
| Modèle | **Gemma 4 26B A4B 8-bit** — local, Apple Silicon |
| Orchestration agent | **LangGraph** |
| Frontend | Swift / SwiftUI — **non prioritaire**, sera reconnecté après stabilisation backend |

> Pour le détail d'omlx (modes de lancement, endpoints, paramètres) :
> lire directement le submodule `omlx/` — Claude Code y a accès.

---

## 🗺 Étape en cours

**Étape 0 — Connexion LangGraph ↔ omlx**

Objectif minimal : un graph LangGraph fonctionnel qui appelle Gemma 4 via omlx
(API Anthropic locale), avec tool calling opérationnel.

Rien d'autre n'est en scope pour l'instant.

---

## 📝 Méthode de travail

- **Une étape à la fois.** On valide avant de passer à la suivante.
- **Progressivité.** On n'implémente pas ce dont on n'a pas encore besoin.
- **Documentation d'abord.** Toute décision structurelle est reflétée dans ce fichier avant ou immédiatement après implémentation.
- **Pas de refactoring opportuniste.** On ne touche à rien hors du périmètre de l'étape.
- **Arrêt sur ambiguïté.** Si un choix non trivial se présente, on s'arrête et on demande plutôt que d'inventer une solution.
- **Standards en priorité.** On privilégie les patterns LangGraph/LangChain officiels plutôt que du code custom. La doc est accessible via le MCP `docs-langchain`.

### Accès documentation LangGraph
```
claude mcp add --transport http docs-langchain --scope user \
  https://docs.langchain.com/mcp
```
### Accès oMLX

Le submodule omlx/ à la racine du repo contient le code source
d'omlx : le lire pour comprendre l'API Anthropic exposée (endpoints,
format des requêtes, authentification, tool calling).