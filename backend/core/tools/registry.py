# core/tools/registry.py
"""
ToolRegistry — registre central thread-safe des outils.

Séparation fondamentale :
    _tools     : dict[str, ToolSchema]    → sérialisable JSON, persisté sur disk
    _executors : dict[str, Callable]      → runtime uniquement, jamais persisté

Thread-safety :
    asyncio.Lock couvre toutes les mutations (_tools, _executors, disk).
    Lectures (get_tools_for_ministral) copient le dict sous lock (atomique,
    sans await), puis traitent hors lock — aucune dégradation de concurrence.

Persistence :
    register / unregister → persist immédiat dans registry.json
    update_stats           → in-memory uniquement (perte au restart acceptable
                             en Phase 3.1 ; persist différé Phase 3.5)

Reconstruction des executors :
    _load_from_disk() restaure les métadonnées (stats, created_at…) mais
    _executors reste vide. Les outils builtin doivent être re-registered au
    startup avec register(…, force=True) pour réinjecter leurs callables.

Path :
    Paramètre constructeur avec injection explicite possible.
    Sans paramètre, le path est résolu ainsi :
      1) TOOLS_FOLDER (si défini)
      2) DATA_FOLDER/tools_registry (ou sibling si DATA_FOLDER finit par
         ".../conversations")
      3) fallback legacy : backend/storage/tools/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .schema import PermissionLevel, ToolSchema

logger = logging.getLogger(__name__)

# Fallback legacy dans le repo (utilisé seulement si aucune config externe).
_LEGACY_STORAGE = Path(__file__).parents[2] / "storage" / "tools"


def _resolve_default_storage() -> Path:
    """
    Résout le dossier de persistance des outils.

    Priorité :
      1) TOOLS_FOLDER explicite
      2) dérivé de DATA_FOLDER (même volume SSD externe)
      3) fallback legacy dans le repo
    """
    tools_folder = os.getenv("TOOLS_FOLDER", "").strip()
    if tools_folder:
        return Path(tools_folder)

    data_folder = os.getenv("DATA_FOLDER", "").strip()
    if data_folder:
        data_path = Path(data_folder)
        # Cas courant : DATA_FOLDER=/Volumes/AISSD/conversations
        if data_path.name.lower() == "conversations":
            return data_path.parent / "tools_registry"
        return data_path / "tools_registry"

    return _LEGACY_STORAGE

# Ordre strict des niveaux (READ_ONLY = moins permissif, AUTONOMOUS = plus permissif)
_PERMISSION_ORDER: list[PermissionLevel] = [
    PermissionLevel.READ_ONLY,
    PermissionLevel.WRITE_SAFE,
    PermissionLevel.WRITE_MODIFY,
    PermissionLevel.SYSTEM_EXEC,
    PermissionLevel.NETWORK,
    PermissionLevel.AUTONOMOUS,
]


# ---------------------------------------------------------------------------
# Exceptions dédiées
# ---------------------------------------------------------------------------


class SystemToolProtectedError(Exception):
    """
    Levée quand on tente d'écraser ou de supprimer un outil created_by="system".
    Bloqué même avec force=True — les outils système sont immuables par design.
    """


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Registre central des outils. Singleton accessible via get_instance().

    Usage production :
        registry = ToolRegistry.get_instance()
        await registry.register(schema, executor)
        tools = registry.get_tools_for_ministral()

    Usage tests (isolation complète, sans affecter le singleton) :
        registry = ToolRegistry(storage_path=tmp_path / "tools")

    Protection système :
        Les outils created_by="system" ne peuvent être ni écrasés (register)
        ni supprimés (unregister) — SystemToolProtectedError est levée.
        Cette règle vit ici (Registry) plutôt que dans ToolSchema car le
        Registry est le seul à connaître l'état existant global.
    """

    _instance: "ToolRegistry | None" = None

    def __init__(self, storage_path: Path | None = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else _resolve_default_storage()
        self._registry_file = self._storage_path / "registry.json"

        # Schémas (persistés sur disk)
        self._tools: dict[str, ToolSchema] = {}
        # Callables runtime (jamais sérialisés)
        self._executors: dict[str, Callable[..., Any]] = {}

        self._lock = asyncio.Lock()
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, storage_path: Path | None = None) -> "ToolRegistry":
        """
        Retourne le singleton global.
        storage_path n'est pris en compte qu'à la première création.
        """
        if cls._instance is None:
            cls._instance = cls(storage_path)
        return cls._instance

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset du singleton — usage tests uniquement."""
        cls._instance = None

    # ── Register / Unregister ─────────────────────────────────────────────────

    async def register(
        self,
        schema: ToolSchema,
        executor: Callable[..., Any],
        *,
        force: bool = False,
    ) -> None:
        """
        Enregistre un outil dans le registry.

        Args:
            schema:   ToolSchema validé (nom snake_case, types JSON Schema).
            executor: Callable Python (async ou sync) qui implémente l'outil.
            force:    Si True, autorise l'écrasement d'un outil existant
                      (hors outils système protégés).

        Raises:
            ValueError:              L'outil existe déjà et force=False.
            SystemToolProtectedError: L'outil existant est created_by="system"
                                      (même avec force=True, refusé).
        """
        async with self._lock:
            existing = self._tools.get(schema.name)

            if existing is not None:
                # Protection inconditionnelle des outils système
                if existing.created_by == "system":
                    raise SystemToolProtectedError(
                        f"L'outil '{schema.name}' est un outil système "
                        f"(created_by='system') et ne peut pas être écrasé, "
                        f"même avec force=True."
                    )
                if not force:
                    raise ValueError(
                        f"L'outil '{schema.name}' est déjà enregistré. "
                        f"Utilisez force=True pour l'écraser."
                    )
                logger.info(
                    "Registry : écrasement de '%s' (force=True, was created_by=%s)",
                    schema.name, existing.created_by,
                )

            self._tools[schema.name] = schema
            self._executors[schema.name] = executor
            self._persist()
            logger.debug(
                "Registry : '%s' enregistré (created_by=%s, permission=%s)",
                schema.name, schema.created_by, schema.permission_level.value,
            )

    async def unregister(self, name: str) -> bool:
        """
        Supprime un outil du registry.

        Returns:
            True  si l'outil a été supprimé.
            False si l'outil est protégé (created_by="system") ou inexistant.

        Note : Les outils created_by="system" ne peuvent jamais être supprimés.
               Cette protection est silencieuse (False) plutôt qu'une exception
               car unregister peut être appelé dans des cleanup génériques.
        """
        async with self._lock:
            schema = self._tools.get(name)

            if schema is None:
                logger.warning(
                    "Registry : unregister('%s') — outil introuvable", name
                )
                return False

            if schema.created_by == "system":
                logger.warning(
                    "Registry : unregister('%s') refusé — outil système protégé", name
                )
                return False

            del self._tools[name]
            self._executors.pop(name, None)
            self._persist()
            logger.debug("Registry : '%s' supprimé", name)
            return True

    # ── Requêtes (lecture) ────────────────────────────────────────────────────

    def inject_executor(self, name: str, executor: Callable[..., Any]) -> None:
        """
        Injecte un callable runtime pour un outil déjà enregistré (schema sur disk).

        Utilisé au redémarrage pour re-brancher les exécuteurs builtin sans
        avoir à supprimer et ré-enregistrer le schema (qui est protégé system).
        """
        self._executors[name] = executor
        logger.debug("Registry : exécuteur injecté pour '%s'", name)

    def get(self, name: str) -> ToolSchema | None:
        """Retourne le schema d'un outil ou None si inconnu."""
        return self._tools.get(name)

    def has_executor(self, name: str) -> bool:
        """True si l'outil a un callable runtime disponible."""
        return name in self._executors

    def get_executor(self, name: str) -> Callable[..., Any] | None:
        """Retourne le callable d'un outil ou None s'il est absent."""
        return self._executors.get(name)

    def get_tools_for_ministral(
        self,
        permission_filter: PermissionLevel | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retourne la liste des outils au format OpenAI pur (injecté au modèle).

        Seul le format OpenAI est retourné — les métadonnées étendues
        (permission_level, success_count, created_by…) ne sont JAMAIS exposées
        ici car elles ne font pas partie du standard OpenAI function calling.

        Thread-safety :
            La copie dict est atomique (pas de await) → safe en asyncio.
            Le traitement to_openai_format() se fait hors lock.

        Args:
            permission_filter: Si précisé, filtre sur ce niveau exact.
                               None → tous les outils enregistrés.
        """
        # Copie snapshot atomique sous lock (sans await → sans yield)
        # En asyncio single-thread, dict(self._tools) ne peut pas être
        # interrompu par une autre coroutine — aucun point de suspension.
        with_lock = self._tools.copy()  # atomique, pas d'await needed here

        result: list[dict[str, Any]] = []
        for schema in with_lock.values():
            if (
                permission_filter is not None
                and schema.permission_level != permission_filter
            ):
                continue
            result.append(schema.to_openai_format())

        return result

    def get_tools_up_to_level(
        self,
        max_level: PermissionLevel,
    ) -> list[dict[str, Any]]:
        """
        Retourne tous les outils dont le niveau de permission <= max_level.

        Exemple : max_level=WRITE_SAFE → retourne READ_ONLY + WRITE_SAFE.
        Utile pour configurer ce qu'un agent de niveau donné peut utiliser.
        """
        try:
            max_index = _PERMISSION_ORDER.index(max_level)
        except ValueError:
            logger.error("Niveau de permission inconnu : %s", max_level)
            return []

        allowed = set(_PERMISSION_ORDER[: max_index + 1])
        tools_snapshot = self._tools.copy()

        return [
            schema.to_openai_format()
            for schema in tools_snapshot.values()
            if schema.permission_level in allowed
        ]

    @property
    def tool_count(self) -> int:
        """Nombre d'outils enregistrés."""
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        """Liste des noms d'outils enregistrés."""
        return list(self._tools.keys())

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def update_stats(
        self,
        name: str,
        *,
        success: bool,
        duration_ms: float,
    ) -> None:
        """
        Met à jour les statistiques d'exécution d'un outil.

        Phase 3.1 : mise à jour in-memory uniquement (perte au restart).
        Phase 3.5 : persist différé ajouté ici (debounce ou shutdown hook).
        """
        async with self._lock:
            schema = self._tools.get(name)
            if schema is None:
                return

            if success:
                schema.success_count += 1
            else:
                schema.failure_count += 1

            # Moyenne cumulative glissante
            total = schema.success_count + schema.failure_count
            if total > 0:
                schema.average_duration_ms = (
                    schema.average_duration_ms * (total - 1) + duration_ms
                ) / total

            schema.last_used = datetime.now(timezone.utc)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """
        Écrit registry.json avec l'état courant de _tools.

        Doit être appelé SOUS _lock (mutations uniquement).
        model_dump(mode="json") produit un dict JSON-compatible directement
        (pas de round-trip via json.loads(model_dump_json())).

        Format :
        {
          "version": 1,
          "tools": {
            "<name>": { ...ToolSchema fields... },
            ...
          }
        }
        """
        data: dict[str, Any] = {
            "version": 1,
            "tools": {
                name: schema.model_dump(mode="json")
                for name, schema in self._tools.items()
            },
        }
        try:
            self._registry_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Registry : erreur persistence disk (%s)", exc)

    def _load_from_disk(self) -> None:
        """
        Charge les schemas depuis registry.json au démarrage.

        IMPORTANT : _executors reste vide après ce chargement.
        Les outils builtin doivent être re-registered au startup via
        register(schema, executor, force=True) pour réinjecter leurs callables.
        force=True préserve les stats chargées car l'outil existant (loaded)
        n'est pas created_by="system" — la protection système est appliquée
        sur l'outil *existant*, pas sur le nouveau.

        Cas gérés :
            Fichier absent    → premier démarrage, registry vide (OK)
            JSON invalide     → log error, registry vide
            Schema invalide   → log warning, schema ignoré (partiel gracieux)
        """
        if not self._registry_file.exists():
            logger.debug("Registry : pas de registry.json — démarrage vide")
            return

        try:
            raw = self._registry_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "Registry : registry.json illisible (%s) — démarrage vide", exc
            )
            return

        tools_data: dict[str, Any] = data.get("tools", {})
        loaded = 0
        skipped = 0

        for name, schema_dict in tools_data.items():
            try:
                schema = ToolSchema.model_validate(schema_dict)
                self._tools[schema.name] = schema
                loaded += 1
            except Exception as exc:
                logger.warning(
                    "Registry : schema '%s' invalide au chargement (%s) — ignoré",
                    name, exc,
                )
                skipped += 1

        if loaded or skipped:
            logger.info(
                "Registry : %d outil(s) chargé(s) depuis disk, %d ignoré(s)",
                loaded, skipped,
            )
        if loaded > 0:
            logger.info(
                "Registry : %d outil(s) sans executor — "
                "re-register les builtin tools au startup.",
                loaded,
            )
