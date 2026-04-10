# core/tools/validator.py
"""
ToolValidator — analyse statique AST du code Python généré par l'IA.

Objectif : détecter les patterns dangereux AVANT ajout au registry et exécution.
           Aucun code n'est exécuté — analyse purement structurelle.

Détections :
    - Builtins dangereux : eval, exec, compile, __import__, globals, locals, vars
    - Imports non autorisés (hors whitelist et hors permission-map)
    - Imports nécessitant une permission supérieure au contexte courant

Limites connues :
    - L'aliasing indirect (x = eval; x("code")) n'est pas détectable via AST
      sans analyse de flux de données — hors scope Phase 3.2.1.
    - Les appels via getattr(obj, "eval") sont détectés (attribut eval visible
      dans l'AST), mais getattr(obj, variable_string) ne l'est pas.

Décisions d'architecture :
    - ast.walk() plutôt que NodeVisitor : exhaustif par défaut, code sur les
      snippets courts (< 200 lignes) rend la performance négligeable.
    - frozen=True sur ValidationResult : immutabilité post-validation,
      thread-safe, prévient toute mutation accidentelle.
    - extra_allowed_modules en constructeur : extensibilité sans fichier
      config externe (contrainte full-local).
    - Erreurs uniquement (pas de warnings) : Phase 3.2.1 vise la sécurité
      bloquante ; les warnings sémantiques (random non-déterministe) sont
      hors scope.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Sequence

from .schema import PermissionLevel

# ---------------------------------------------------------------------------
# Constantes de sécurité
# ---------------------------------------------------------------------------

# Builtins dont l'appel est systématiquement rejeté (injection de code)
_BLACKLISTED_BUILTINS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
})

# Modules autorisés sans permission spéciale (lecture seule, pas d'effets de bord)
_DEFAULT_WHITELIST: frozenset[str] = frozenset({
    "datetime",
    "math",
    "json",
    "re",
    "typing",
    "pathlib",
    "collections",
    "itertools",
    "functools",
    "decimal",
    "fractions",
    "random",
    "string",
})

# Modules soumis à permission : module → PermissionLevel minimum requis
_PERMISSION_REQUIRED: dict[str, PermissionLevel] = {
    "subprocess": PermissionLevel.SYSTEM_EXEC,
    "os": PermissionLevel.SYSTEM_EXEC,
    "urllib": PermissionLevel.NETWORK,
    "http": PermissionLevel.NETWORK,
    "requests": PermissionLevel.NETWORK,
}

# Ordre strict des niveaux (index = rang croissant de permissivité)
_PERMISSION_ORDER: list[PermissionLevel] = [
    PermissionLevel.READ_ONLY,
    PermissionLevel.WRITE_SAFE,
    PermissionLevel.WRITE_MODIFY,
    PermissionLevel.SYSTEM_EXEC,
    PermissionLevel.NETWORK,
    PermissionLevel.AUTONOMOUS,
]


def _permission_gte(level: PermissionLevel, minimum: PermissionLevel) -> bool:
    """True si level >= minimum dans la hiérarchie de permissions."""
    return _PERMISSION_ORDER.index(level) >= _PERMISSION_ORDER.index(minimum)


def _permission_max(a: PermissionLevel, b: PermissionLevel) -> PermissionLevel:
    """Retourne le niveau le plus élevé entre a et b."""
    return a if _PERMISSION_ORDER.index(a) >= _PERMISSION_ORDER.index(b) else b


# ---------------------------------------------------------------------------
# ValidationResult — résultat immutable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """
    Résultat immutable de la validation d'un code Python.

    Attributes:
        is_valid:            True si le code peut être exécuté dans le contexte donné.
        errors:              Erreurs bloquantes (format "Line N: <description>").
        imports_used:        Modules importés détectés dans le code.
        permission_required: Permission minimum requise pour exécuter ce code
                             (indépendamment de la permission courante).
    """

    is_valid: bool
    errors: tuple[str, ...]
    imports_used: frozenset[str]
    permission_required: PermissionLevel


# ---------------------------------------------------------------------------
# ToolValidator
# ---------------------------------------------------------------------------


class ToolValidator:
    """
    Analyse statique du code Python généré par l'IA via AST.

    Usage standard :
        validator = ToolValidator()
        result = validator.validate_code(code, PermissionLevel.READ_ONLY)
        if not result.is_valid:
            for error in result.errors:
                print(error)

    Usage avec modules additionnels (ex: outils ML) :
        validator = ToolValidator(extra_allowed_modules={"numpy", "pandas"})
    """

    def __init__(
        self,
        extra_allowed_modules: set[str] | None = None,
    ) -> None:
        """
        Args:
            extra_allowed_modules: Modules additionnels autorisés sans permission
                                   spéciale. Étend la whitelist par défaut sans
                                   modifier les permission-based modules.
        """
        self._whitelist: frozenset[str] = (
            _DEFAULT_WHITELIST | frozenset(extra_allowed_modules)
            if extra_allowed_modules
            else _DEFAULT_WHITELIST
        )

    def validate_code(
        self,
        code: str,
        permission: PermissionLevel,
    ) -> ValidationResult:
        """
        Valide un code Python via analyse AST exhaustive.

        Parcourt l'intégralité de l'arbre AST (ast.walk) pour détecter :
        - Les appels à des builtins dangereux (eval, exec, etc.)
        - Les imports non autorisés ou insuffisamment permissifs

        Args:
            code:       Code Python à valider (non exécuté).
            permission: Niveau de permission du contexte d'exécution cible.

        Returns:
            ValidationResult avec is_valid, errors, imports_used, permission_required.

        Note sur lineno : node.lineno correspond à la ligne de DÉBUT du node
        (convention Python AST). Pour un appel multi-ligne, c'est la ligne
        où l'appel commence — le point le plus utile pour la correction.
        """
        # --- Parse syntaxique ---
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult(
                is_valid=False,
                errors=(f"Line {exc.lineno}: SyntaxError — {exc.msg}",),
                imports_used=frozenset(),
                permission_required=PermissionLevel.READ_ONLY,
            )

        errors: list[str] = []
        imports_used: set[str] = set()
        permission_required = PermissionLevel.READ_ONLY

        # --- Parcours exhaustif de l'AST ---
        for node in ast.walk(tree):

            # ── Appels dangereux ────────────────────────────────────────────
            if isinstance(node, ast.Call):
                func = node.func

                # Appel direct : eval(...), exec(...), __import__(...)
                if isinstance(func, ast.Name) and func.id in _BLACKLISTED_BUILTINS:
                    errors.append(
                        f"Line {node.lineno}: appel dangereux interdit '{func.id}()'"
                    )

                # Appel via attribut : builtins.eval(...), obj.exec(...)
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr in _BLACKLISTED_BUILTINS
                ):
                    errors.append(
                        f"Line {node.lineno}: appel dangereux interdit '*.{func.attr}()'"
                    )

            # ── import module [as alias] ────────────────────────────────────
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    # "import os.path" → top-level est "os"
                    top_module = alias.name.split(".")[0]
                    imports_used.add(top_module)
                    perm = self._check_import(top_module, node.lineno, permission, errors)
                    if perm is not None:
                        permission_required = _permission_max(permission_required, perm)

            # ── from module import name ─────────────────────────────────────
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # "from os.path import join" → top-level est "os"
                    top_module = node.module.split(".")[0]
                    imports_used.add(top_module)
                    perm = self._check_import(top_module, node.lineno, permission, errors)
                    if perm is not None:
                        permission_required = _permission_max(permission_required, perm)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
            imports_used=frozenset(imports_used),
            permission_required=permission_required,
        )

    def _check_import(
        self,
        module: str,
        lineno: int,
        permission: PermissionLevel,
        errors: list[str],
    ) -> PermissionLevel | None:
        """
        Vérifie si un module est autorisé dans le contexte de permission donné.

        Args:
            module:     Nom du module top-level à vérifier.
            lineno:     Ligne de l'import (pour le message d'erreur).
            permission: Permission courante du contexte.
            errors:     Liste mutable où ajouter les erreurs détectées.

        Returns:
            PermissionLevel requis par ce module (pour calcul permission_required),
            ou None si le module est en whitelist (READ_ONLY implicite).
        """
        # Permission-map prime toujours sur la whitelist (y compris extra_allowed_modules).
        # Raison : un module à risque comme subprocess ne doit jamais être débloqué
        # sans permission explicite, même si l'appelant l'ajoute à extra_allowed.
        if module in _PERMISSION_REQUIRED:
            required = _PERMISSION_REQUIRED[module]
            if not _permission_gte(permission, required):
                errors.append(
                    f"Line {lineno}: import '{module}' interdit — "
                    f"nécessite {required.name} "
                    f"(permission actuelle : {permission.name})"
                )
            return required

        # Ni whitelist ni permission-map → bloqué
        errors.append(
            f"Line {lineno}: import '{module}' non autorisé — "
            f"module absent de la whitelist et de la permission-map"
        )
        return None
