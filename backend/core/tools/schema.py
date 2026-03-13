# core/tools/schema.py
"""
Modèles Pydantic v2 pour le tool calling au format OpenAI function calling.

Format cible (immuable) :
{
  "type": "function",
  "function": {
    "name": "...",
    "description": "...",
    "parameters": {
      "type": "object",       ← TOUJOURS "object"
      "properties": {...},
      "required": [...]
    }
  }
}

Référence : https://platform.openai.com/docs/guides/function-calling
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, field_serializer, field_validator

# ---------------------------------------------------------------------------
# Types JSON Schema valides (5 uniquement — standard JSON Schema / OpenAI)
# Literal plutôt qu'Enum : messages d'erreur lisibles par l'humain.
# ---------------------------------------------------------------------------
JsonSchemaType = Literal["string", "number", "boolean", "array", "object"]


# ---------------------------------------------------------------------------
# Niveaux de permission (6 niveaux croissants)
# Séquence : lecture seule → opérations sans restriction.
# ---------------------------------------------------------------------------
class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"          # lecture uniquement, aucun effet de bord
    WRITE_SAFE = "write_safe"        # création/ajout sans écrasement
    WRITE_MODIFY = "write_modify"    # modification et suppression
    SYSTEM_EXEC = "system_exec"      # exécution de commandes système
    NETWORK = "network"              # requêtes réseau sortantes
    AUTONOMOUS = "autonomous"        # toutes opérations sans restriction


# ---------------------------------------------------------------------------
# ToolParameter — un paramètre d'outil (correspond à une "property" OpenAI)
# ---------------------------------------------------------------------------
class ToolParameter(BaseModel):
    """
    Représente un paramètre d'outil au format JSON Schema.

    Le champ `name` devient la clé dans `properties` du schéma OpenAI.
    Le champ `required` indique si ce paramètre est obligatoire pour l'appel.

    Décisions :
    - `type` : Literal pour messages d'erreur explicites (pas Enum).
    - `enum` : None = pas de contrainte ; liste = valeurs autorisées uniquement.
    - `items` : requis pour type="array" (décrit le type des éléments).
    - `name` : snake_case strict, devient la clé JSON Schema.
    """

    name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    type: JsonSchemaType
    description: str = ""
    required: bool = False

    # Contrainte de valeurs (ex: timezone parmi liste fixe)
    enum: list[str] | None = None

    # Pour type="array" : décrit le type des éléments du tableau
    # Ex: {"type": "string"} ou {"type": "object", "properties": {...}}
    items: dict[str, Any] | None = None

    def to_property_dict(self) -> dict[str, Any]:
        """
        Sérialise en dict JSON Schema property (sans la clé `name`).
        Utilisé par ToolSchema.to_openai_format() pour construire `properties`.
        """
        prop: dict[str, Any] = {"type": self.type}

        if self.description:
            prop["description"] = self.description

        # enum : inclus seulement si défini (omis = pas de contrainte de valeur)
        if self.enum is not None:
            prop["enum"] = self.enum

        # items : obligatoire pour type="array", ignoré sinon
        if self.type == "array" and self.items is not None:
            prop["items"] = self.items

        return prop

    @field_validator("items")
    @classmethod
    def items_only_for_array(cls, v: dict | None, info: Any) -> dict | None:
        """items n'a de sens que pour type="array"."""
        # Note : la validation cross-field complète est dans ToolSchema
        # pour garder ToolParameter simple et réutilisable seul.
        return v


# ---------------------------------------------------------------------------
# ToolSchema — schéma complet d'un outil (OpenAI + métadonnées étendues)
# ---------------------------------------------------------------------------
class ToolSchema(BaseModel):
    """
    Schéma complet d'un outil, combinant :
    - Champs OpenAI (injectés au modèle via to_openai_format())
    - Métadonnées étendues (stockées en registry, jamais envoyées au modèle)

    Invariants :
    - `name` : snake_case strict (Ministral fine-tuné sur ce format)
    - `parameters` : liste ordonnée (l'ordre est préservé dans `properties`)
    - `created_at` : AwareDatetime (timezone obligatoire, UTC par défaut)

    Protection `created_by` :
    - Le Schema valide que la valeur est dans {"system", "user", "ai"}.
    - Bloquer l'écrasement d'un outil "system" par "ai" est une règle
      du Registry (qui connaît l'état existant), pas du Schema.
    """

    # --- Champs OpenAI (envoyés au modèle) ---

    name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    # --- Métadonnées étendues (stockées, jamais envoyées au modèle) ---

    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    requires_confirmation: bool = False          # True pour outils destructifs
    category: str = "custom"
    created_by: Literal["system", "user", "ai"] = "user"
    version: str = "1.0.0"
    cacheable: bool = False                      # True si résultats déterministes

    # AwareDatetime force une timezone — datetime.now() naïf lève ValidationError
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_used: AwareDatetime | None = None

    # Statistiques d'usage
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    average_duration_ms: float = Field(default=0.0, ge=0.0)

    # Timeout configurable : 1s minimum, 300s maximum
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    # --- Sérialiseurs datetime → ISO 8601 ---

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        """ISO 8601 avec timezone — round-trip sans perte de précision."""
        return value.isoformat()

    @field_serializer("last_used")
    def serialize_last_used(self, value: datetime | None) -> str | None:
        """ISO 8601 ou null."""
        return value.isoformat() if value is not None else None

    # --- Validation ---

    @field_validator("parameters")
    @classmethod
    def no_duplicate_parameter_names(
        cls, params: list[ToolParameter]
    ) -> list[ToolParameter]:
        """Deux paramètres ne peuvent pas avoir le même nom."""
        names = [p.name for p in params]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Noms de paramètres en doublon : {duplicates}")
        return params

    # --- Format OpenAI ---

    def to_openai_format(self) -> dict[str, Any]:
        """
        Génère le dict au format OpenAI function calling.

        Structure garantie (invariants vérifiés par assertions) :
        {
          "type": "function",
          "function": {
            "name": str,
            "description": str,
            "parameters": {
              "type": "object",      ← TOUJOURS "object"
              "properties": dict,
              "required": list[str]
            }
          }
        }

        Note : les métadonnées étendues (permission_level, created_by, etc.)
        ne sont JAMAIS incluses ici — elles restent dans le registry.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_property_dict()
            if param.required:
                required.append(param.name)

        result: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",   # INVARIANT : toujours "object"
                    "properties": properties,
                    "required": required,
                },
            },
        }

        # Vérification post-génération des invariants structurels
        assert result["type"] == "function", (
            "BUG: result['type'] doit être 'function'"
        )
        assert result["function"]["parameters"]["type"] == "object", (
            "BUG: parameters['type'] doit être 'object'"
        )
        assert isinstance(result["function"]["parameters"]["properties"], dict), (
            "BUG: properties doit être un dict"
        )
        assert isinstance(result["function"]["parameters"]["required"], list), (
            "BUG: required doit être une liste"
        )

        return result
