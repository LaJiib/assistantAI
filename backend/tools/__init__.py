# tools/__init__.py
"""
Outils disponibles pour l'agent IA.

Structure :
    tools/builtin/     — Outils système, non-modifiables, créés par 'system'
    tools/generated/   — Outils créés dynamiquement par l'IA (Phase 3.5+)

Point d'entrée :
    register_builtin_tools(registry) — appelé au startup dans lifespan
"""

from .builtin import register_builtin_tools

__all__ = ["register_builtin_tools"]
