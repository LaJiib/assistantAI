# tools/builtin/__init__.py
"""
Outils builtin — auto-registration au startup.

Usage dans lifespan (main.py) :
    from tools.builtin import register_builtin_tools
    await register_builtin_tools(registry)

Chaque module contribue un dict BUILTIN_TOOLS :
    name → (ToolSchema, async_callable)

register_builtin_tools() itère sur tous les modules et appelle
registry.register(schema, executor, force=True) pour chacun.
force=True car les schemas peuvent déjà exister en mémoire (rechargement
des métadonnées depuis disk au startup) — on veut réinjecter les callables
sans changer les stats existantes. Les outils système sont protégés par
created_by="system" : register() avec force=True sur un outil système
lèverait SystemToolProtectedError… sauf que c'est le MÊME outil qui est
re-registered — la protection bloque UNIQUEMENT si l'existant est "system"
et qu'on tente de le changer. On contourne en re-registrant APRÈS un clear
des schemas chargés depuis disk, ou en gérant l'exception.

Pattern retenu : essai register(force=True), si SystemToolProtectedError
l'executor est injecté directement (l'outil était déjà là, schema inchangé).
"""

from __future__ import annotations

import logging

from core.tools import SystemToolProtectedError, ToolRegistry
from .system import BUILTIN_TOOLS

logger = logging.getLogger(__name__)


async def register_builtin_tools(registry: ToolRegistry) -> None:
    """
    Enregistre tous les outils builtin dans le registry.

    Appelé une fois au startup dans le lifespan FastAPI.
    Idempotent : peut être appelé plusieurs fois sans effet de bord.

    Si un outil système est déjà présent (chargé depuis disk au démarrage),
    l'executor est réinjecté sans modifier le schema (stats préservées).

    Args:
        registry: Instance ToolRegistry cible (singleton ou test).
    """
    registered = 0
    reinjected = 0

    for name, (schema, executor) in BUILTIN_TOOLS.items():
        try:
            await registry.register(schema, executor, force=True)
            registered += 1
        except SystemToolProtectedError:
            # L'outil est déjà présent avec created_by="system" depuis disk.
            # On réinjecte uniquement l'executor (schema inchangé → stats OK).
            registry.inject_executor(name, executor)
            reinjected += 1
            logger.debug("Builtin '%s' : executor réinjecté (schema disk conservé)", name)

    total = registered + reinjected
    logger.info(
        "🔧 Builtin tools : %d enregistré(s), %d executor(s) réinjecté(s) [total: %d]",
        registered, reinjected, total,
    )
